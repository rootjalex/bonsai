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

    // Values that were split, keyed by what named them. Instructions are
    // keyed by identity; entry-block arguments by name, since an argument is
    // copied by value wherever it is referenced.
    map<const Instruction *, Components> instrs;
    map<string, Components> args;

    // Instructions whose result is a value that already exists -- reading a
    // component of a split vector is just that component -- and the value to
    // use instead.
    map<const Instruction *, shared_ptr<Value>> folded;

    // Component reads of uniform vectors, cached so that a vector used by
    // several split operations is only taken apart once.
    map<std::pair<const void *, uint32_t>, shared_ptr<Value>> extracted;

    // Instructions to drop once the walk is done: those that were split and
    // those that folded away.
    set<const Instruction *> dead;

    shared_ptr<Block> block;
    vector<shared_ptr<Instruction>> emitted;

    Splitter(Function &func, const Divergence &divergence)
        : func(func), divergence(divergence) {}

    shared_ptr<Value> emit(Type type, Instruction::Op op,
                           vector<shared_ptr<Value>> operands) {
        auto instr = std::make_shared<Instruction>(func.get_unique_name(),
                                                   std::move(type), op,
                                                   std::move(operands), block);
        emitted.push_back(instr);
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

    // The k-th component of `value`, however it is represented: a component
    // of a split value, a component read out of a uniform vector, or a scalar
    // that every component shares.
    shared_ptr<Value> component(const shared_ptr<Value> &value, uint32_t k) {
        if (const Components *split = components_of(*value)) {
            return (*split)[k];
        }
        const Type &type = value->get_type();
        if (!type.is_vector()) {
            return value; // a scalar: the same for every component
        }

        // A uniform vector, read apart once and reused.
        const void *key = value.get();
        if (const auto *instr =
                std::get_if<shared_ptr<Instruction>>(&value->data)) {
            key = instr->get();
        }
        const auto cached = extracted.find({key, k});
        if (cached != extracted.end()) {
            return cached->second;
        }
        auto index =
            std::make_shared<Value>(Constant{UInt_t::make(32), uint64_t(k)});
        auto value_k = emit(type.element_of(), Instruction::Op::ExtractIdx,
                            {value, std::move(index)});
        extracted[{key, k}] = value_k;
        return value_k;
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
bool is_elementwise(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
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
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        return true;
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

} // namespace

SplitResult split_aggregates(Function &func, const string &entry,
                             const Divergence &divergence,
                             const set<const Instruction *> &already_wide) {
    const BlockMap blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);
    const set<string> region = reachable_from(entry, all_succs);

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }

    Splitter splitter(func, divergence);
    SplitResult result;

    // A varying vector parameter becomes one parameter per component, which
    // is what lets a caller hand over the components it already has.
    {
        auto entry_block = blocks.at(entry);
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

    for (const string &name : reverse_postorder(entry, succs)) {
        auto block = blocks.at(name);
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

            // Reading a component of a split vector is that component: no
            // instruction is needed for it at all.
            if (instr->op == Instruction::Op::ExtractIdx &&
                splitter.components_of(*instr->operands[0]) != nullptr) {
                const auto k = constant_index(*instr->operands[1]);
                internal_assert(k.has_value())
                    << "Reading a per-lane vector at a computed index is a "
                    << "shuffle across lanes, which vectorization does not "
                    << "support: " << instr->name;
                splitter.folded[instr.get()] =
                    (*splitter.components_of(*instr->operands[0]))[*k];
                splitter.dead.insert(instr.get());
                continue;
            }

            if (!varying || !produces_vector) {
                // Nothing about this instruction changes, but it may still
                // read a value that was split -- which only makes sense for
                // the cases handled above.
                for (const auto &operand : instr->operands) {
                    internal_assert(splitter.components_of(*operand) == nullptr)
                        << "Instruction " << instr->name
                        << " uses a value that was split per component, which "
                        << "vectorization only knows how to do for "
                        << "elementwise operations and component reads";
                }
                splitter.emitted.push_back(instr);
                continue;
            }

            const uint32_t lanes = instr->type.lanes();
            const Type element = instr->type.element_of();

            if (is_elementwise(instr->op)) {
                Components components;
                for (uint32_t k = 0; k < lanes; k++) {
                    vector<shared_ptr<Value>> operands;
                    for (const auto &operand : instr->operands) {
                        operands.push_back(splitter.component(operand, k));
                    }
                    components.push_back(
                        splitter.emit(element, instr->op, std::move(operands)));
                }
                splitter.instrs[instr.get()] = std::move(components);
                splitter.dead.insert(instr.get());
                continue;
            }

            // Broadcasting a scalar gives every component that scalar.
            if (instr->op == Instruction::Op::Bc) {
                Components components(lanes, instr->operands[0]);
                splitter.instrs[instr.get()] = std::move(components);
                splitter.dead.insert(instr.get());
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
                    auto offset = std::make_shared<Value>(
                        Constant{scalar_index, int64_t(k)});
                    auto index = splitter.emit(index_type, Instruction::Op::Add,
                                               {base, offset});
                    components.push_back(
                        splitter.emit(element, Instruction::Op::ExtractIdx,
                                      {flat_array, std::move(index)}));
                }
                splitter.instrs[instr.get()] = std::move(components);
                splitter.dead.insert(instr.get());
                continue;
            }

            instr->dump(std::cerr);
            internal_error << "TODO: split the per-lane vector produced by ^";
        }

        block->instrs = splitter.emitted;
    }

    // Anything that folded away has to stop being referenced, including from
    // terminators: a call may pass a component of a split vector.
    for (const string &name : region) {
        auto block = blocks.at(name);
        auto fix = [&](shared_ptr<Value> &value) {
            value = splitter.resolve(value);
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
                               fix(a);
                           }
                       },
                       [&](Terminator::Dispatch &t) {
                           fix(t.cond);
                           for (auto &target : t.targets) {
                               for (auto &a : target.args) {
                                   fix(a);
                               }
                           }
                       },
                       [&](Terminator::Return &t) {
                           if (t.value) {
                               internal_assert(
                                   splitter.components_of(*t.value) == nullptr)
                                   << "TODO: return a per-lane vector, which "
                                   << "needs one returned value per component";
                               fix(t.value);
                           }
                       },
                       [&](Terminator::ParFor &) {},
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &t) {
                           // A split argument is handed over as its
                           // components, which is why a callee's parameters
                           // are split to match. How many each became is
                           // recorded so that the callee can be expanded the
                           // same way.
                           vector<shared_ptr<Value>> flattened;
                           vector<uint32_t> shape;
                           for (auto &a : t.call.args) {
                               fix(a);
                               if (const Components *split =
                                       splitter.components_of(*a)) {
                                   for (const auto &part : *split) {
                                       flattened.push_back(part);
                                   }
                                   shape.push_back(uint32_t(split->size()));
                               } else {
                                   flattened.push_back(a);
                                   shape.push_back(1);
                               }
                           }
                           t.call.args = std::move(flattened);
                           result.call_shapes[name] = std::move(shape);
                           for (auto &a : t.cont.args) {
                               fix(a);
                           }
                       },
                   },
                   block->terminator.data);
    }

    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
