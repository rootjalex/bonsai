#include "SSA/SplitAggregates.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <map>
#include <optional>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// The components a value was split into, one per lane-component.
using Components = vector<shared_ptr<Value>>;

// The state of the split, threaded through the walk over the region.
struct Splitter {
    Function &func;
    const Divergence &divergence;
    // Gang-wide vectors, which are not a lane's own and are never split: the
    // lane indices, built as a vector to begin with.
    const set<const Instruction *> &already_wide;

    // Values that were split, keyed by what named them. Instructions are
    // keyed by identity; entry-block arguments by name, since an argument is
    // copied by value wherever it is referenced.
    map<const Instruction *, Components> instrs;
    map<string, Components> args;

    // Instructions whose result is a value that already exists -- reading a
    // component of a split vector is just that component -- and the value to
    // use instead.
    map<const Instruction *, shared_ptr<Value>> folded;

    // Component reads of whole vectors, cached so that a vector used by
    // several split operations is only taken apart once. Per block, since a
    // read emitted in one block is not in scope in another that it does not
    // dominate.
    map<std::tuple<string, const void *, uint32_t>, shared_ptr<Value>>
        extracted;

    shared_ptr<Block> block;
    vector<shared_ptr<Instruction>> emitted;

    Splitter(Function &func, const Divergence &divergence,
             const set<const Instruction *> &already_wide)
        : func(func), divergence(divergence), already_wide(already_wide) {}

    // Is this a vector a lane holds one of -- as opposed to a scalar, or to a
    // vector that is already the gang's?
    bool per_lane_vector(const Value &value) const {
        if (!value.get_type().is_vector()) {
            return false;
        }
        const auto *instr = std::get_if<shared_ptr<Instruction>>(&value.data);
        return instr == nullptr || already_wide.count(instr->get()) == 0;
    }

    shared_ptr<Instruction> emit_instr(Type type, Instruction::Op op,
                                       vector<shared_ptr<Value>> operands) {
        auto instr = std::make_shared<Instruction>(func.get_unique_name(),
                                                   std::move(type), op,
                                                   std::move(operands), block);
        emitted.push_back(instr);
        return instr;
    }

    shared_ptr<Value> emit(Type type, Instruction::Op op,
                           vector<shared_ptr<Value>> operands) {
        return std::make_shared<Value>(
            emit_instr(std::move(type), op, std::move(operands)));
    }

    // The same operation as `like`, on other operands and producing `type`:
    // what one component of a split operation is.
    shared_ptr<Value> emit_like(const Instruction &like, Type type,
                                vector<shared_ptr<Value>> operands) {
        auto instr = emit_instr(std::move(type), like.op, std::move(operands));
        instr->intrinsic = like.intrinsic;
        instr->reduce = like.reduce;
        instr->shuffle = like.shuffle;
        instr->queried_type = like.queried_type;
        return std::make_shared<Value>(std::move(instr));
    }

    // Was this value split?
    const Components *components_of(const Value &value) const {
        if (const auto *instr =
                std::get_if<shared_ptr<Instruction>>(&value.data)) {
            const auto it = instrs.find(instr->get());
            return it == instrs.end() ? nullptr : &it->second;
        }
        if (const auto *arg = std::get_if<Argument>(&value.data)) {
            const auto it = args.find(arg->name);
            return it == args.end() ? nullptr : &it->second;
        }
        return nullptr;
    }

    // Is this value one the lanes disagree about, as seen from the block
    // being walked?
    bool is_varying(const Value &value) const {
        return divergence.is_varying(block->name, value);
    }

    // The k-th component of `value`, however it is represented: a component
    // of a split value, a component read out of a whole vector, or a scalar
    // that every component shares.
    shared_ptr<Value> component(const shared_ptr<Value> &value, uint32_t k) {
        if (const Components *split = components_of(*value)) {
            return (*split)[k];
        }
        const Type &type = value->get_type();
        if (!type.is_vector()) {
            return value; // a scalar: the same for every component
        }

        // A whole vector -- uniform, or a per-lane one that is carried whole
        // (see whole()) -- read apart once per block and reused.
        const void *key = value.get();
        if (const auto *instr =
                std::get_if<shared_ptr<Instruction>>(&value->data)) {
            key = instr->get();
        }
        const auto cached = extracted.find({block->name, key, k});
        if (cached != extracted.end()) {
            return cached->second;
        }
        auto index =
            std::make_shared<Value>(Constant{UInt_t::make(32), uint64_t(k)});
        auto value_k = emit(type.element_of(), Instruction::Op::ExtractIdx,
                            {value, std::move(index)});
        extracted[{block->name, key, k}] = value_k;
        return value_k;
    }

    // A split value as one vector again, for a use that takes the vector
    // whole: a field of a struct being built, a value being stored, returned
    // or passed along a jump. Built the way the SSA builder spells a vector
    // literal, a MakeStruct of the vector type (see the VecImm visitor in
    // SSA/Convert.cpp), so that nothing downstream sees a second spelling.
    // Widening then turns it into the struct of gang vectors a per-lane vector
    // is carried as; see widen() in SSA/Vectorize.cpp.
    shared_ptr<Value> whole(const shared_ptr<Value> &value) {
        const Components *split = components_of(*value);
        if (split == nullptr) {
            return value;
        }
        return emit(value->get_type(), Instruction::Op::MakeStruct, *split);
    }

    shared_ptr<Value> resolve(const shared_ptr<Value> &value) {
        const auto it = folded.find(
            std::holds_alternative<shared_ptr<Instruction>>(value->data)
                ? std::get<shared_ptr<Instruction>>(value->data).get()
                : nullptr);
        return it == folded.end() ? value : it->second;
    }
};

// Operations whose result is one value per component of their operands, and
// which therefore split into one operation per component.
bool is_elementwise(const Instruction &instr) {
    switch (instr.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eq:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Lt:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        return true;
    // The intrinsics are all pointwise -- `sqrt`, `fma`, `atanh` and the rest
    // apply to each component -- except `rand`, whose argument is a count, and
    // except the vector-valued ones, `dot`, `cross` and `norm`, which the SSA
    // builder expands into the pieces before anything here sees them (see the
    // Intrinsic visitor in SSA/Convert.cpp).
    case Instruction::Op::Intrinsic:
        return instr.intrinsic != ir::Intrinsic::rand;
    // A bitwise reinterpretation of one vector as another with the same
    // number of components is a reinterpretation of each component.
    case Instruction::Op::Reinterpret:
        return instr.operands.size() == 1 &&
               instr.operands[0]->get_type().is_vector() &&
               instr.operands[0]->get_type().lanes() == instr.type.lanes();
    default:
        return false;
    }
}

// Is this a constant index, and if so which?
optional<uint64_t> constant_index(const Value &value) {
    const auto *constant = std::get_if<Constant>(&value.data);
    if (constant == nullptr) {
        return std::nullopt;
    }
    return std::visit(
        overloads{
            [](uint64_t u) -> optional<uint64_t> { return u; },
            [](int64_t i) -> optional<uint64_t> {
                return i < 0 ? std::nullopt : optional<uint64_t>(uint64_t(i));
            },
            [](auto) -> optional<uint64_t> { return std::nullopt; },
        },
        constant->data);
}

// A constant of `type` holding `k`, spelled the way that type's constants are.
shared_ptr<Value> constant_of(const Type &type, uint64_t k) {
    if (type.is_int()) {
        return std::make_shared<Value>(Constant{type, int64_t(k)});
    }
    return std::make_shared<Value>(Constant{type, k});
}

// The binary operation that folds two components of a reduction.
Instruction::Op reduction_step(ir::VectorReduce::OpType op, const Type &element,
                               const string &name) {
    switch (op) {
    case ir::VectorReduce::Add:
        return Instruction::Op::Add;
    case ir::VectorReduce::Mul:
        return Instruction::Op::Mul;
    case ir::VectorReduce::Min:
        return Instruction::Op::Min;
    case ir::VectorReduce::Max:
        return Instruction::Op::Max;
    case ir::VectorReduce::And:
        return element.is_bool() ? Instruction::Op::LAnd
                                 : Instruction::Op::BwAnd;
    case ir::VectorReduce::Or:
        return element.is_bool() ? Instruction::Op::LOr : Instruction::Op::BwOr;
    default:
        internal_error << "TODO: reduce a per-lane vector with "
                       << to_string(op) << " in " << name
                       << ", which needs the component's index kept beside "
                          "its value";
        return Instruction::Op::Add;
    }
}

} // namespace

SplitResult split_aggregates(Function &func, const string &entry,
                             const Divergence &divergence,
                             const set<const Instruction *> &already_wide) {
    const Cfg region(func, entry);

    Splitter splitter(func, divergence, already_wide);
    SplitResult result;

    // A varying vector parameter becomes one parameter per component, which
    // is what lets a caller hand over the components it already has.
    {
        const shared_ptr<Block> &entry_block = region.block(region.entry);
        vector<Argument> rebuilt;
        for (const Argument &arg : entry_block->args) {
            if (!arg.type.is_vector() ||
                !divergence.args.count({entry, arg.name})) {
                rebuilt.push_back(arg);
                continue;
            }

            Components components;
            for (uint32_t k = 0; k < arg.type.lanes(); k++) {
                const Argument part{arg.type.element_of(),
                                    arg.name + "!" + std::to_string(k)};
                rebuilt.push_back(part);
                components.push_back(std::make_shared<Value>(part));
                result.parameters.insert(part.name);
            }
            entry_block->lookups.erase(arg.name);
            splitter.args[arg.name] = std::move(components);
        }
        entry_block->args = std::move(rebuilt);
    }

    for (BlockId b : region.rpo) {
        const shared_ptr<Block> &block = region.block(b);
        splitter.block = block;
        splitter.emitted.clear();

        for (const auto &instr : block->instrs) {
            for (auto &operand : instr->operands) {
                operand = splitter.resolve(operand);
            }

            const bool varying = divergence.instrs.count(instr.get()) > 0;
            const bool produces_vector = instr->type.defined() &&
                                         instr->type.is_vector() &&
                                         already_wide.count(instr.get()) == 0;

            // Reading a component of a vector.
            if (instr->op == Instruction::Op::ExtractIdx &&
                splitter.per_lane_vector(*instr->operands[0])) {
                const auto &vec = instr->operands[0];
                const auto k = constant_index(*instr->operands[1]);
                if (k.has_value()) {
                    // Of a split vector: that component, and no instruction
                    // at all. Of a whole one: the read stays as it is, and
                    // widening makes it a field read if the vector turns out
                    // to be per lane (see widen() in SSA/Vectorize.cpp).
                    if (const Components *split = splitter.components_of(*vec)) {
                        splitter.folded[instr.get()] = (*split)[*k];
                        continue;
                    }
                } else if (varying) {
                    // At a computed index: which component a lane reads is
                    // its own business, so no single component can be named.
                    // A shuffle across the gang is what it would take, and
                    // what is done instead is what every SIMD compiler does
                    // for a dynamically indexed short vector -- select among
                    // the components by comparing the index against each.
                    const Type &vec_type = vec->get_type();
                    const uint32_t n = vec_type.lanes();
                    const auto &index = instr->operands[1];
                    const Type index_type = index->get_type();
                    shared_ptr<Value> picked = splitter.component(vec, n - 1);
                    for (uint32_t c = n - 1; c-- > 0;) {
                        auto is_c = splitter.emit(
                            Bool_t::make(), Instruction::Op::Eq,
                            {index, constant_of(index_type, c)});
                        picked = splitter.emit(
                            vec_type.element_of(), Instruction::Op::Select,
                            {is_c, splitter.component(vec, c), picked});
                    }
                    splitter.folded[instr.get()] = picked;
                    continue;
                }
            }

            // Folding the components of a per-lane vector into one value.
            if (instr->op == Instruction::Op::Reduce && varying &&
                splitter.per_lane_vector(*instr->operands[0])) {
                const auto &vec = instr->operands[0];
                const Type element = vec->get_type().element_of();
                const uint32_t n = vec->get_type().lanes();
                // Which component is the extremum is the extremum first, then
                // the first component equal to it -- the way lower::argmax
                // spells the scalar case, so that a tie picks the same
                // component whether or not the loop was vectorized.
                const bool by_index =
                    instr->reduce == ir::VectorReduce::Idxmax ||
                    instr->reduce == ir::VectorReduce::Idxmin;
                const ir::VectorReduce::OpType fold =
                    instr->reduce == ir::VectorReduce::Idxmax
                        ? ir::VectorReduce::Max
                    : instr->reduce == ir::VectorReduce::Idxmin
                        ? ir::VectorReduce::Min
                        : instr->reduce;
                const Instruction::Op step =
                    reduction_step(fold, element, instr->name);
                shared_ptr<Value> acc = splitter.component(vec, 0);
                for (uint32_t c = 1; c < n; c++) {
                    acc = splitter.emit(element, step,
                                        {acc, splitter.component(vec, c)});
                }
                if (by_index) {
                    const Type index_type = instr->type;
                    shared_ptr<Value> picked = constant_of(index_type, n - 1);
                    for (uint32_t c = n - 1; c-- > 0;) {
                        auto is_c = splitter.emit(
                            Bool_t::make(), Instruction::Op::Eq,
                            {acc, splitter.component(vec, c)});
                        picked = splitter.emit(
                            index_type, Instruction::Op::Select,
                            {is_c, constant_of(index_type, c), picked});
                    }
                    acc = picked;
                }
                splitter.folded[instr.get()] = acc;
                continue;
            }

            if (!varying || !produces_vector) {
                // Nothing about this instruction changes, but it may still
                // read a value that was split -- a struct built from a
                // per-lane vector, a store of one -- and it takes that value
                // whole.
                for (auto &operand : instr->operands) {
                    operand = splitter.whole(operand);
                }
                splitter.emitted.push_back(instr);
                continue;
            }

            const uint32_t lanes = instr->type.lanes();
            const Type element = instr->type.element_of();

            // A copy of a vector is a copy of its components, and needs no
            // instruction: the components are the operand's. (A `set` is also
            // the one instruction the relooper insists carry the program's own
            // name for the value, which a component made here would not.)
            if (instr->op == Instruction::Op::Set) {
                Components components;
                for (uint32_t k = 0; k < lanes; k++) {
                    components.push_back(
                        splitter.component(instr->operands[0], k));
                }
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            if (is_elementwise(*instr)) {
                Components components;
                for (uint32_t k = 0; k < lanes; k++) {
                    vector<shared_ptr<Value>> operands;
                    for (const auto &operand : instr->operands) {
                        operands.push_back(splitter.component(operand, k));
                    }
                    components.push_back(
                        splitter.emit_like(*instr, element, std::move(operands)));
                }
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            // Broadcasting a scalar gives every component that scalar.
            if (instr->op == Instruction::Op::Bc) {
                Components components(lanes, instr->operands[0]);
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            // A vector literal is already its components.
            if (instr->op == Instruction::Op::MakeStruct) {
                internal_assert(instr->operands.size() == lanes)
                    << "Vector literal " << instr->name << " has "
                    << instr->operands.size() << " components for a "
                    << instr->type;
                Components components;
                for (const auto &operand : instr->operands) {
                    components.push_back(splitter.whole(operand));
                }
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            // Reading an array of scalars at a lane's own vector of indices:
            // `table[o]`, with `o` a lane's four wavelengths rounded to the
            // table's entries (apps/pbrt's spectrum_at_dense), gives the lane
            // a vector with one element per index. Its components are the
            // reads at each component of the index -- reads of a scalar at a
            // per-lane index, which widening makes gathers of -- so the vector
            // is split into those, the shape every use of a per-lane vector
            // expects. The read's execution mask, if it has one, covers each
            // of them: a lane that is off may hold any index at all.
            if (instr->op == Instruction::Op::ExtractIdx &&
                !instr->operands[0]->get_type().element_of().is_vector() &&
                splitter.per_lane_vector(*instr->operands[1])) {
                const auto &array = instr->operands[0];
                internal_assert(array->get_type().is_reference())
                    << "Reading at a vector of indices out of something that "
                    << "is not an array: " << instr->name;
                const auto &index = instr->operands[1];
                Components components;
                for (uint32_t k = 0; k < lanes; k++) {
                    vector<shared_ptr<Value>> operands{
                        array, splitter.component(index, k)};
                    if (instr->operands.size() == 3) {
                        operands.push_back(instr->operands[2]);
                    }
                    components.push_back(splitter.emit(
                        element, Instruction::Op::ExtractIdx, std::move(operands)));
                }
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            // Reading an array of per-lane vectors: each component lives
            // every `lanes` elements apart in memory, so it becomes one
            // strided read per component of the array viewed as its element
            // type. This is the array-of-structures load ispc's section 5.2
            // is about, and the reason the reads are strided.
            if (instr->op == Instruction::Op::ExtractIdx) {
                const auto &array = instr->operands[0];
                const Type &array_type = array->get_type();
                internal_assert(array_type.is_reference())
                    << "Reading a per-lane vector out of something that is "
                    << "not an array: " << instr->name;

                const Type flat = Array_t::make(element, /*size=*/Expr());
                auto flat_array =
                    splitter.emit(flat, Instruction::Op::Reinterpret, {array});

                // The index may already be one per lane (the lane indices),
                // so the arithmetic keeps its type while the constants it is
                // combined with stay scalar.
                const Type index_type = instr->operands[1]->get_type();
                const Type scalar_index = index_type.is_vector()
                                              ? index_type.element_of()
                                              : index_type;

                // How far apart consecutive elements sit, in components.
                //
                // Not the number of components: a vector occupies whatever
                // its target says, and a three-float vector is twelve bytes
                // of data in sixteen bytes of storage. Asking the backend
                // (ir::SizeOf) rather than assuming a padding rule here is
                // what keeps this correct for a target that lays vectors out
                // differently -- and the two sizes fold to a constant during
                // code generation, so nothing is paid for asking.
                auto element_size =
                    splitter.emit(scalar_index, Instruction::Op::SizeOf, {});
                std::get<shared_ptr<Instruction>>(element_size->data)
                    ->queried_type = instr->type;
                auto component_size =
                    splitter.emit(scalar_index, Instruction::Op::SizeOf, {});
                std::get<shared_ptr<Instruction>>(component_size->data)
                    ->queried_type = element;

                auto width = splitter.emit(scalar_index, Instruction::Op::Div,
                                           {element_size, component_size});
                auto base = splitter.emit(index_type, Instruction::Op::Mul,
                                          {instr->operands[1], width});

                Components components;
                for (uint32_t k = 0; k < lanes; k++) {
                    auto offset = constant_of(scalar_index, k);
                    auto index = splitter.emit(index_type, Instruction::Op::Add,
                                               {base, offset});
                    // The read's execution mask, if it has one, covers every
                    // component read in its stead.
                    vector<shared_ptr<Value>> operands{flat_array,
                                                       std::move(index)};
                    if (instr->operands.size() == 3) {
                        operands.push_back(instr->operands[2]);
                    }
                    components.push_back(splitter.emit(
                        element, Instruction::Op::ExtractIdx, std::move(operands)));
                }
                splitter.instrs[instr.get()] = std::move(components);
                continue;
            }

            // Anything else that produces a per-lane vector -- a field read out
            // of a per-lane struct, a load from per-lane memory, a copy -- is
            // carried whole, and taken apart where a component is wanted (see
            // component()). Whatever it reads that was split, it reads whole.
            for (auto &operand : instr->operands) {
                operand = splitter.whole(operand);
            }
            splitter.emitted.push_back(instr);
        }

        block->instrs = splitter.emitted;
    }

    // Anything that folded away has to stop being referenced, including from
    // terminators, and a terminator that takes a vector whole -- a return, a
    // jump -- gets a split one rebuilt; a call gets a per-lane one as its
    // components, since the callee's parameter is split to match.
    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
        const string &name = block->name;
        splitter.block = block;
        splitter.emitted = block->instrs;
        auto fix = [&](shared_ptr<Value> &value) {
            value = splitter.resolve(value);
        };
        auto fix_whole = [&](shared_ptr<Value> &value) {
            fix(value);
            value = splitter.whole(value);
        };
        // A vector the lanes disagree about is handed to a callee as its
        // components, whether it was split here or is carried whole: the
        // callee's own parameter is split (see the entry-block rewrite above),
        // so the two have to line up. How many values each argument became
        // is recorded so that the callee can be expanded the same way.
        auto flatten = [&](vector<shared_ptr<Value>> &args,
                           vector<uint32_t> &shape) {
            vector<shared_ptr<Value>> flattened;
            for (auto &a : args) {
                fix(a);
                const Type &type = a->get_type();
                if (splitter.per_lane_vector(*a) &&
                    (splitter.components_of(*a) != nullptr ||
                     splitter.is_varying(*a))) {
                    for (uint32_t k = 0; k < type.lanes(); k++) {
                        flattened.push_back(splitter.component(a, k));
                    }
                    shape.push_back(type.lanes());
                } else {
                    flattened.push_back(a);
                    shape.push_back(1);
                }
            }
            args = std::move(flattened);
        };
        for (const auto &instr : block->instrs) {
            for (auto &operand : instr->operands) {
                fix(operand);
            }
        }
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &t) {
                           for (auto &a : t.args) {
                               fix_whole(a);
                           }
                       },
                       [&](Terminator::Dispatch &t) {
                           fix(t.cond);
                           for (auto &target : t.targets) {
                               for (auto &a : target.args) {
                                   fix_whole(a);
                               }
                           }
                       },
                       [&](Terminator::Return &t) {
                           if (t.value) {
                               fix_whole(t.value);
                           }
                       },
                       [&](Terminator::ParFor &) {},
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &t) {
                           vector<uint32_t> shape;
                           flatten(t.call.args, shape);
                           result.call_shapes[name] = std::move(shape);
                           for (auto &a : t.cont.args) {
                               fix_whole(a);
                           }
                       },
                       [&](Terminator::MultiCall &t) {
                           // As for Call, plus one thing to keep straight:
                           // splitting an argument into components moves every
                           // argument after it, so the varying positions have
                           // to be remapped, and each varying value has to
                           // split into as many components as the placeholder
                           // it stands in for -- otherwise one call in the run
                           // would hand the callee a different number of
                           // arguments than another.
                           vector<size_t> new_at(t.call.args.size());
                           vector<size_t> width(t.call.args.size());
                           {
                               vector<uint32_t> shape;
                               vector<shared_ptr<Value>> args = t.call.args;
                               flatten(args, shape);
                               size_t at = 0;
                               for (size_t i = 0; i < shape.size(); i++) {
                                   new_at[i] = at;
                                   width[i] = shape[i];
                                   at += shape[i];
                               }
                               t.call.args = std::move(args);
                               result.call_shapes[name] = std::move(shape);
                           }

                           vector<size_t> varying_at;
                           for (const size_t old : t.varying_at) {
                               for (size_t j = 0; j < width[old]; j++) {
                                   varying_at.push_back(new_at[old] + j);
                               }
                           }
                           for (auto &vs : t.varying) {
                               vector<shared_ptr<Value>> flat;
                               for (size_t k = 0; k < vs.size(); k++) {
                                   fix(vs[k]);
                                   const size_t old = t.varying_at[k];
                                   const Type &type = vs[k]->get_type();
                                   const size_t parts =
                                       splitter.per_lane_vector(*vs[k]) &&
                                               (splitter.components_of(*vs[k]) !=
                                                    nullptr ||
                                                splitter.is_varying(*vs[k]))
                                           ? type.lanes()
                                           : 1;
                                   internal_assert(parts == width[old])
                                       << "A varying value splits into "
                                       << parts << " components but the "
                                       << "argument it replaces splits into "
                                       << width[old];
                                   if (parts > 1) {
                                       for (uint32_t c = 0; c < parts; c++) {
                                           flat.push_back(
                                               splitter.component(vs[k], c));
                                       }
                                   } else {
                                       flat.push_back(vs[k]);
                                   }
                               }
                               vs = std::move(flat);
                           }
                           t.varying_at = std::move(varying_at);

                           for (auto &k : t.keys) {
                               fix(k);
                               internal_assert(splitter.components_of(*k) ==
                                               nullptr)
                                   << "TODO: a sort key that splits into "
                                   << "components -- a key is compared, so it "
                                   << "has to stay one value";
                           }

                           for (auto &a : t.cont.args) {
                               fix_whole(a);
                           }
                       },
                   },
                   block->terminator.data);
        block->instrs = splitter.emitted;
    }

    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
