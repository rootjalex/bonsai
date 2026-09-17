#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/CloneFunction.h"
#include "SSA/Linearize.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"
#include "SSA/SkipInactiveBlocks.h"
#include "SSA/SplitAggregates.h"
#include "SSA/UniformizeLoops.h"

#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// The lanes of the gang, i.e. the extent of the loop being vectorized. The
// schedule is expected to have split the loop down to the gang width already
// (see the comment on vectorize() below), so this must come out exact.
uint32_t gang_width(const Terminator::ParFor &parfor) {
    auto constant = [](const shared_ptr<Value> &v) -> int64_t {
        const auto *c = std::get_if<Constant>(&v->data);
        internal_assert(c) << "vectorize() needs constant loop bounds";
        return std::visit(overloads{
                              [](int64_t i) { return i; },
                              [](uint64_t u) { return int64_t(u); },
                              [](bool b) -> int64_t {
                                  internal_error << "boolean loop bound";
                                  return 0;
                              },
                              [](double d) -> int64_t {
                                  internal_error << "floating point loop bound";
                                  return 0;
                              },
                              [](const string &s) -> int64_t {
                                  internal_error << "symbolic loop bound: "
                                                 << s;
                                  return 0;
                              },
                          },
                          c->data);
    };

    const int64_t start = constant(parfor.start);
    const int64_t end = constant(parfor.end);
    const int64_t stride = constant(parfor.stride);
    internal_assert(stride > 0 && end > start)
        << "vectorize() needs a non-empty, forward loop";
    internal_assert((end - start) % stride == 0)
        << "vectorize() needs the loop extent to divide by its stride; "
        << "split() the loop to the gang width first";
    return uint32_t((end - start) / stride);
}

// The vector form of a value that a lane holds one of is ir::widen (see
// IR/Type.h), shared with the backends, which have to recognize the shapes it
// makes: a struct of gang-wide fields, and a struct of one gang-wide vector
// per component of a short vector.

// Is `type` what widen() makes of a short vector: a struct of gang-wide
// vectors, one per component? Reading a component of a per-lane vector is
// reading one of these fields.
bool is_widened_vector(const Type &type) {
    const Struct_t *s = type.as<Struct_t>();
    if (s == nullptr || s->fields.empty()) {
        return false;
    }
    for (uint32_t k = 0; k < uint32_t(s->fields.size()); k++) {
        if (s->fields[k].name != component_field(k) ||
            !s->fields[k].type.is_vector()) {
            return false;
        }
    }
    return true;
}

// The type of a shared pointer to per-lane memory -- ispc's `varying T *
// uniform`. The pointer is one address, so it is not widened; what it points
// at is, so that a load through it is a vector and a store through it writes
// every lane's slot. An array handle is the address of its elements and is
// treated the same way: a local array the lanes each have their own copy of
// becomes an array of gang-wide elements.
Type widen_pointee(const Type &type, uint32_t lanes) {
    if (const Ptr_t *p = type.as<Ptr_t>()) {
        return Ptr_t::make(widen(p->etype, lanes));
    }
    if (const Array_t *a = type.as<Array_t>()) {
        return Array_t::make(widen(a->etype, lanes), a->size);
    }
    internal_error << "widen_pointee of a type that addresses nothing: "
                   << type;
    return type;
}

// A value that already holds one per lane -- the shape widen() produces: a
// vector as wide as the gang, or a struct widen() built, which its name says
// (widen() suffixes the lane count). Such an operand is not broadcast when a
// divergent instruction is widened, and a value of this shape is not widened
// again, where a uniform scalar, a scalar struct or a lane's own short vector
// is.
//
// Told apart by shape rather than by remembering what was widened, because a
// block argument is a copy wherever it is referenced and the copies are only
// ever found again by name (see widen_argument), so what is known about one
// has to be readable off any of them. The one shape this cannot tell apart is
// a lane's own short vector that happens to be exactly as wide as the gang;
// the short vectors of a geometric program are two to four wide. A packed
// vector is never gang-wide, whatever its width: widen() makes a plain
// vector, and packed storage -- the words of an ADT's payload, which are as
// often eight as anything -- is a lane's own value.
bool is_gang_wide(const Type &type, uint32_t lanes) {
    if (const Vector_t *v = type.as<Vector_t>()) {
        return v->lanes == lanes && !v->packed;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        const string suffix = "$v" + std::to_string(lanes);
        return s->name.size() > suffix.size() &&
               s->name.compare(s->name.size() - suffix.size(), suffix.size(),
                               suffix) == 0;
    }
    return false;
}

// A pointer or array handle whose memory has already been laid out per lane:
// the shape widen_pointee() produces. Where is_gang_wide() says a value is
// gang-wide, this says the memory a pointer addresses is.
bool points_to_widened(const Type &type, uint32_t lanes) {
    if (const Ptr_t *p = type.as<Ptr_t>()) {
        return is_gang_wide(p->etype, lanes);
    }
    if (const Array_t *a = type.as<Array_t>()) {
        return is_gang_wide(a->etype, lanes);
    }
    return false;
}

// Does a store or accumulate through `place` write one slot per lane -- either
// because the address itself is per lane (a scatter) or because the memory it
// addresses is laid out per lane? Only then is the value it writes widened;
// a write through a shared address into shared memory stays scalar, and is
// then either a cross-lane reduction (an accumulate) or a store the gang makes
// once if any lane is on.
bool writes_per_lane(const Type &place, uint32_t lanes) {
    return is_gang_wide(place, lanes) || points_to_widened(place, lanes);
}

// The operands of `op` that hold a value each lane has its own copy of, as
// opposed to structural operands: the aggregate an index is applied to, the
// address a store writes through, a lane count, a field number.
//
// Widening only ever touches the first kind. Whether the second kind can be
// varying at all is a per-op question -- a varying address is a
// gather/scatter, a varying field number is not expressible -- and is checked
// where the operand is used.
vector<size_t> value_operands(const Instruction &instr) {
    const size_t n = instr.operands.size();
    switch (instr.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Xor:
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
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Sub: {
        vector<size_t> all(n);
        for (size_t i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    // An index into an aggregate, or through a pointer: the aggregate stays
    // as it is and the index is what goes per-lane.
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::GEP:
        return {1};

    // A struct built from field values, each of which the gang has its own of;
    // the result is a struct of widened fields (see widen()).
    case Instruction::Op::MakeStruct: {
        vector<size_t> all(n);
        for (size_t i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    // Reads one field of a struct: the struct is the per-lane value, and the
    // field named beside it is structural. The result -- that field, one per
    // lane -- is widened like any other.
    case Instruction::Op::LoadField:
        return {0};

    // Every vector shuffled is a value; the indices ride on the instruction.
    case Instruction::Op::Shuffle: {
        vector<size_t> all(n);
        for (size_t i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    // Addresses, and the memory they name: nothing here is a value a lane
    // holds a copy of. A load's result is per lane when the address is, or
    // when the memory is (see widen_region); the address of a field is a
    // structural step from its base; an allocation and an address-of make
    // memory rather than compute on values.
    case Instruction::Op::Load:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::AddressOf:
    case Instruction::Op::Alloca:
    case Instruction::Op::Alloc:
        return {};

    // The address is structural, and whether the value written -- and the
    // execution mask a predicated store carries as a third operand -- goes per
    // lane depends on where the address points; widen_region decides, so
    // nothing is listed here.
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::Store:
        return {};

    case Instruction::Op::Eps:
    case Instruction::Op::Inf:
        return {};

    // Reduces the lanes of one value, which is that value per gang lane.
    case Instruction::Op::Reduce:
        return {0};

    // An intrinsic computes on values, so every operand goes per-lane -- with
    // one exception: `rand`'s argument is how many numbers to draw, which is
    // a count and not one of them.
    case Instruction::Op::Intrinsic: {
        if (instr.intrinsic == ir::Intrinsic::rand) {
            return {};
        }
        vector<size_t> all(n);
        for (size_t i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    // Already one value per lane, built that way: its base and stride are
    // uniform scalars and must stay that way.
    case Instruction::Op::Ramp:
        return {};

    default:
        instr.dump(std::cerr);
        internal_error << "TODO: vectorize the operation above.";
        return {};
    }
}

// Rewrites every reference to the value named `name` in `block` -- operands,
// jump arguments, and the values a terminator uses directly -- to `value`.
void substitute(Block &block, const string &name,
                const shared_ptr<Value> &value) {
    auto replace = [&](shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data) &&
            std::get<Argument>(v->data).name == name) {
            v = value;
        }
    };

    for (const auto &instr : block.instrs) {
        for (auto &operand : instr->operands) {
            replace(operand);
        }
    }

    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) {
                       for (auto &a : j.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::Dispatch &d) {
                       replace(d.cond);
                       for (auto &t : d.targets) {
                           for (auto &a : t.args) {
                               replace(a);
                           }
                       }
                   },
                   [&](Terminator::Return &r) { replace(r.value); },
                   [&](Terminator::ParFor &p) {
                       replace(p.start);
                       replace(p.end);
                       replace(p.stride);
                       for (auto &a : p.body.args) {
                           replace(a);
                       }
                       for (auto &a : p.cont.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       for (auto &a : c.call.args) {
                           replace(a);
                       }
                       for (auto &a : c.cont.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::MultiCall &c) {
                       for (auto &a : c.call.args) {
                           replace(a);
                       }
                       for (auto &vs : c.varying) {
                           for (auto &a : vs) {
                               replace(a);
                           }
                       }
                       for (auto &k : c.keys) {
                           replace(k);
                       }
                       for (auto &a : c.cont.args) {
                           replace(a);
                       }
                   },
               },
               block.terminator.data);

    const auto it = block.lookups.find(name);
    if (it != block.lookups.end()) {
        it->second = value;
    }
}

// Widens a block argument in place. A reference to it is a copy of the same
// Argument rather than a pointer to one, so every copy has to be retyped --
// and after linearization those copies are spread across the region, since
// folding the branches let values be used directly instead of being threaded
// through arguments.
//
// `pointee` says the argument is a shared pointer to per-lane memory rather
// than a per-lane value, and so is retyped with widen_pointee.
void widen_argument(const BlockMap &blocks, const set<string> &region,
                    const string &owner, const string &name, uint32_t lanes,
                    bool pointee = false) {
    // A copy already widened -- because two blocks call an argument by the
    // same name and both are per lane -- is left alone rather than widened a
    // second time; likewise a pointer whose memory is already laid out per
    // lane.
    auto widen_type = [&](Type &type) {
        if (pointee) {
            if (!points_to_widened(type, lanes)) {
                type = widen_pointee(type, lanes);
            }
        } else if (!is_gang_wide(type, lanes)) {
            type = widen(type, lanes);
        }
    };

    for (auto &arg : blocks.at(owner)->args) {
        if (arg.name == name) {
            widen_type(arg.type);
        }
    }

    auto retype = [&](const shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data)) {
            Argument &a = std::get<Argument>(v->data);
            if (a.name == name) {
                widen_type(a.type);
            }
        }
    };

    for (const string &block_name : region) {
        Block &block = *blocks.at(block_name);
        for (const auto &instr : block.instrs) {
            for (const auto &operand : instr->operands) {
                retype(operand);
            }
        }
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &t) {
                           for (auto &a : t.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::Dispatch &t) {
                           retype(t.cond);
                           for (auto &target : t.targets) {
                               for (auto &a : target.args) {
                                   retype(a);
                               }
                           }
                       },
                       [&](Terminator::Return &t) { retype(t.value); },
                       [&](Terminator::ParFor &t) {
                           for (auto &a : t.body.args) {
                               retype(a);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &t) {
                           for (auto &a : t.call.args) {
                               retype(a);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::MultiCall &t) {
                           for (auto &a : t.call.args) {
                               retype(a);
                           }
                           for (auto &vs : t.varying) {
                               for (auto &a : vs) {
                                   retype(a);
                               }
                           }
                           for (auto &k : t.keys) {
                               retype(k);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                   },
                   block.terminator.data);
        const auto it = block.lookups.find(name);
        if (it != block.lookups.end()) {
            retype(it->second);
        }
    }
}

// Broadcasts a uniform value up to the gang width, appending the instructions
// it takes to `sink`. A scalar is a single Bc; a struct is broadcast the way
// ISPC turns a uniform struct varying -- each field read out, broadcast in
// turn, and the widened struct rebuilt from the results -- since Bc is a SIMD
// splat of a scalar and cannot take a struct.
shared_ptr<Value> broadcast(Function &func, const shared_ptr<Block> &block,
                            const shared_ptr<Value> &value, uint32_t lanes,
                            vector<shared_ptr<Instruction>> &sink) {
    const Type type = value->get_type();
    if (const Struct_t *s = type.as<Struct_t>()) {
        vector<shared_ptr<Value>> fields;
        fields.reserve(s->fields.size());
        for (uint32_t i = 0; i < uint32_t(s->fields.size()); i++) {
            auto idx = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(i)});
            auto load = std::make_shared<Instruction>(
                func.get_unique_name(), s->fields[i].type,
                Instruction::Op::LoadField,
                vector<shared_ptr<Value>>{value, idx}, block);
            sink.push_back(load);
            fields.push_back(broadcast(func, block,
                                       std::make_shared<Value>(load), lanes,
                                       sink));
        }
        auto make = std::make_shared<Instruction>(
            func.get_unique_name(), widen(type, lanes),
            Instruction::Op::MakeStruct, fields, block);
        sink.push_back(make);
        return std::make_shared<Value>(make);
    }
    // A uniform short vector meets the gang component by component: each
    // component is read out, splatted, and the widened vector -- a struct of
    // those splats, see widen() -- rebuilt from them.
    if (const Vector_t *v = type.as<Vector_t>()) {
        vector<shared_ptr<Value>> components;
        components.reserve(v->lanes);
        for (uint32_t k = 0; k < v->lanes; k++) {
            auto idx = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(k)});
            auto read = std::make_shared<Instruction>(
                func.get_unique_name(), v->etype, Instruction::Op::ExtractIdx,
                vector<shared_ptr<Value>>{value, idx}, block);
            sink.push_back(read);
            components.push_back(broadcast(
                func, block, std::make_shared<Value>(read), lanes, sink));
        }
        auto make = std::make_shared<Instruction>(
            func.get_unique_name(), widen(type, lanes),
            Instruction::Op::MakeStruct, components, block);
        sink.push_back(make);
        return std::make_shared<Value>(make);
    }
    auto count =
        std::make_shared<Value>(Constant{UInt_t::make(32), uint64_t(lanes)});
    auto bc = std::make_shared<Instruction>(
        func.get_unique_name(), widen(type, lanes), Instruction::Op::Bc,
        vector<shared_ptr<Value>>{value, count}, block);
    sink.push_back(bc);
    return std::make_shared<Value>(bc);
}

// What an accumulate may fold in for a lane that is off: the identity of its
// operation, as a scalar of `type`, appending any instruction it takes to
// `sink`. Adding nothing, multiplying by one, taking the minimum against
// plus infinity or the largest value of the type.
shared_ptr<Value> accumulate_identity(Instruction::Op op, const Type &type,
                                      Function &func,
                                      const shared_ptr<Block> &block,
                                      vector<shared_ptr<Instruction>> &sink) {
    auto constant = [&](double d, int64_t i) -> shared_ptr<Value> {
        if (type.is_float()) {
            return std::make_shared<Value>(Constant{type, d});
        }
        if (type.is_uint()) {
            return std::make_shared<Value>(Constant{type, uint64_t(i)});
        }
        return std::make_shared<Value>(Constant{type, i});
    };
    auto infinity = [&]() {
        auto inf = std::make_shared<Instruction>(
            func.get_unique_name(), type, Instruction::Op::Inf,
            vector<shared_ptr<Value>>{}, block);
        sink.push_back(inf);
        return std::make_shared<Value>(inf);
    };
    const uint32_t bits = type.is_float() ? 0 : type.bits();
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccSub:
        return constant(0.0, 0);
    case Instruction::Op::AccMul:
        return constant(1.0, 1);
    case Instruction::Op::AccMin:
        if (type.is_float()) {
            return infinity();
        }
        if (type.is_uint()) {
            return std::make_shared<Value>(Constant{
                type, bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1});
        }
        return constant(0.0, bits >= 64 ? INT64_MAX
                                        : (int64_t(1) << (bits - 1)) - 1);
    case Instruction::Op::AccMax:
        if (type.is_float()) {
            auto neg = std::make_shared<Instruction>(
                func.get_unique_name(), type, Instruction::Op::Sub,
                vector<shared_ptr<Value>>{constant(0.0, 0), infinity()}, block);
            sink.push_back(neg);
            return std::make_shared<Value>(neg);
        }
        if (type.is_uint()) {
            return constant(0.0, 0);
        }
        return constant(0.0, bits >= 64 ? INT64_MIN
                                        : -(int64_t(1) << (bits - 1)));
    default:
        internal_error << "No identity for " << op_name(op);
        return nullptr;
    }
}

// Turns every varying value in the region into one value per lane: its type
// becomes a vector, and the uniform values it is combined with are broadcast
// to match. Blocks are visited in execution order, so an instruction's
// operands have already been widened by the time it is reached.
void widen_region(Function &func, const string &entry, const Divergence &div,
                  uint32_t lanes) {
    // Everything the earlier stages did is visible here, and what widening
    // chokes on is usually a value one of them left in the wrong shape. The
    // pair of dumps also says which of the two is at fault when the result is
    // merely wrong rather than rejected.
    const bool dump = std::getenv("BONSAI_DUMP_WIDENING") != nullptr;
    if (dump) {
        std::cerr << "--- before widening " << entry << ":\n";
        func.dump(std::cerr);
    }

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

    const auto is_write = [](const Instruction &instr) {
        return instr.op == Instruction::Op::Store ||
               instr.op == Instruction::Op::AccAdd ||
               instr.op == Instruction::Op::AccMul ||
               instr.op == Instruction::Op::AccSub ||
               instr.op == Instruction::Op::AccMin ||
               instr.op == Instruction::Op::AccMax;
    };

    for (const string &name : reverse_postorder(entry, succs)) {
        Block &block = *blocks.at(name);

        for (const Argument &arg : block.args) {
            const bool pointee = div.pointee_args.count({name, arg.name}) > 0;
            if (div.args.count({name, arg.name})) {
                internal_assert(!pointee)
                    << "Argument " << arg.name << " of " << name
                    << " is a per-lane pointer into per-lane memory, which "
                    << "vectorization does not lay out yet";
                widen_argument(blocks, region, name, arg.name, lanes);
            } else if (pointee) {
                widen_argument(blocks, region, name, arg.name, lanes,
                               /*pointee=*/true);
            }
        }

        vector<shared_ptr<Instruction>> widened;
        for (const auto &instr : block.instrs) {
            // An element of a per-lane array at an index the lanes disagree
            // about -- a traversal's stack at each lane's own depth. The array
            // is laid out one gang vector per element (see widen_pointee), so
            // lane l's element i is component l of vector i: element i * lanes
            // + l of the array seen as a flat array of the scalar. The access
            // becomes a gather or scatter over that view, at that index; a
            // per-lane pointer into it is a vector of element addresses. This
            // is what ispc does for a varying index into a varying array.
            const bool per_lane_element =
                (instr->op == Instruction::Op::GEP ||
                 instr->op == Instruction::Op::ExtractIdx) &&
                instr->operands.size() >= 2 &&
                points_to_widened(instr->operands[0]->get_type(), lanes) &&
                instr->operands[0]->get_type().is<Array_t>() &&
                is_gang_wide(instr->operands[1]->get_type(), lanes);
            if (per_lane_element) {
                const shared_ptr<Value> &array = instr->operands[0];
                const Type element = array->get_type().element_of();
                internal_assert(element.is_vector() &&
                                !element.element_of().is<Struct_t>() &&
                                !element.element_of().is_vector())
                    << "[unimplemented] " << instr->name
                    << " reads or addresses a per-lane element of a per-lane "
                    << "array of aggregates, " << array->get_type()
                    << ": each field would have to be gathered on its own";
                const Type scalar = element.element_of();
                const Type index_type = instr->operands[1]->get_type();
                const Type scalar_index = index_type.element_of();
                auto flat = std::make_shared<Instruction>(
                    func.get_unique_name(), Array_t::make(scalar, Expr()),
                    Instruction::Op::Reinterpret,
                    vector<shared_ptr<Value>>{array}, block.shared_from_this());
                widened.push_back(flat);
                const auto index_constant = [&](int64_t k) {
                    return scalar_index.is_uint()
                               ? std::make_shared<Value>(
                                     Constant{scalar_index, uint64_t(k)})
                               : std::make_shared<Value>(
                                     Constant{scalar_index, k});
                };
                auto ramp = std::make_shared<Instruction>(
                    func.get_unique_name(), index_type, Instruction::Op::Ramp,
                    vector<shared_ptr<Value>>{index_constant(0),
                                              index_constant(1)},
                    block.shared_from_this());
                widened.push_back(ramp);
                auto scaled = std::make_shared<Instruction>(
                    func.get_unique_name(), index_type, Instruction::Op::Mul,
                    vector<shared_ptr<Value>>{
                        instr->operands[1],
                        broadcast(func, block.shared_from_this(),
                                  index_constant(int64_t(lanes)), lanes,
                                  widened)},
                    block.shared_from_this());
                widened.push_back(scaled);
                auto at = std::make_shared<Instruction>(
                    func.get_unique_name(), index_type, Instruction::Op::Add,
                    vector<shared_ptr<Value>>{std::make_shared<Value>(scaled),
                                              std::make_shared<Value>(ramp)},
                    block.shared_from_this());
                widened.push_back(at);
                instr->operands[0] = std::make_shared<Value>(flat);
                instr->operands[1] = std::make_shared<Value>(at);
                instr->type = instr->op == Instruction::Op::GEP
                                  ? widen(Ptr_t::make(scalar), lanes)
                                  : widen(scalar, lanes);
                widened.push_back(instr);
                continue;
            }

            // The address of one component of a per-lane short vector in
            // memory -- `d[2]`, where every lane has its own `d` -- is the
            // address of a field: the vector widened to a struct of one gang
            // vector per component (see widen()), so the component the
            // constant index names is that field of it. The mirror of the
            // ExtractIdx below, for a write rather than a read.
            if (instr->op == Instruction::Op::GEP &&
                instr->operands.size() == 2 &&
                points_to_widened(instr->operands[0]->get_type(), lanes) &&
                instr->operands[0]->get_type().is<Ptr_t>() &&
                is_widened_vector(
                    instr->operands[0]->get_type().as<Ptr_t>()->etype)) {
                const auto *c =
                    std::get_if<Constant>(&instr->operands[1]->data);
                internal_assert(c != nullptr)
                    << "Addressing a per-lane vector at a computed index, "
                    << instr->name << ", survived to widening";
                const uint64_t k = std::visit(
                    overloads{
                        [](int64_t i) { return uint64_t(i); },
                        [](uint64_t u) { return u; },
                        [&](const auto &) -> uint64_t {
                            internal_error << "Component index of "
                                           << instr->name
                                           << " is not an integer";
                            return 0;
                        },
                    },
                    c->data);
                const Struct_t *components =
                    instr->operands[0]->get_type().as<Ptr_t>()->etype.as<Struct_t>();
                internal_assert(k < components->fields.size())
                    << instr->name << " addresses component " << k << " of "
                    << instr->operands[0]->get_type();
                instr->op = Instruction::Op::FieldPtr;
                instr->operands[1] =
                    std::make_shared<Value>(Constant{UInt_t::make(32), k});
                instr->type = Ptr_t::make(components->fields[k].type);
                widened.push_back(instr);
                continue;
            }

            // A shared pointer into per-lane memory: the pointer stays one
            // address, and only what it points at changes shape. Its operands
            // are addresses too, or structural, so there is nothing to
            // broadcast.
            if (div.pointee_instrs.count(instr.get())) {
                internal_assert(!div.instrs.count(instr.get()))
                    << instr->name << " is a per-lane pointer into per-lane "
                    << "memory, which vectorization does not lay out yet";
                if (!points_to_widened(instr->type, lanes)) {
                    instr->type = widen_pointee(instr->type, lanes);
                }
                widened.push_back(instr);
                continue;
            }

            // A write goes per lane when its place does -- a per-lane address,
            // or per-lane memory behind a shared one -- whatever the value
            // written is: a uniform value stored into a per-lane slot has to be
            // broadcast to fill every lane's slot. The place's type already
            // says which, since the pointer it comes from is defined before it
            // and has been retyped by now.
            const bool per_lane_write =
                is_write(*instr) &&
                writes_per_lane(instr->operands[0]->get_type(), lanes);

            if (!div.instrs.count(instr.get()) && !per_lane_write) {
                widened.push_back(instr);
                continue;
            }

            // A component of a per-lane short vector is a field of the struct
            // the vector widened to (see widen()), so reading one is a field
            // read, of the field the constant index names. The index is a
            // constant by construction: a computed one was turned into a
            // chain of selects over the components before widening (see
            // SSA/SplitAggregates.h), since it would otherwise be a shuffle.
            if (instr->op == Instruction::Op::ExtractIdx &&
                is_widened_vector(instr->operands[0]->get_type())) {
                const auto *c =
                    std::get_if<Constant>(&instr->operands[1]->data);
                internal_assert(c != nullptr)
                    << "Reading a per-lane vector at a computed index, "
                    << instr->name << ", survived to widening";
                // A field index is an unsigned constant, whatever the program
                // spelled the component's index as.
                const uint64_t k = std::visit(
                    overloads{
                        [](int64_t i) { return uint64_t(i); },
                        [](uint64_t u) { return u; },
                        [&](const auto &) -> uint64_t {
                            internal_error << "Component index of "
                                           << instr->name
                                           << " is not an integer";
                            return 0;
                        },
                    },
                    c->data);
                instr->op = Instruction::Op::LoadField;
                instr->operands[1] =
                    std::make_shared<Value>(Constant{UInt_t::make(32), k});
                instr->type = widen(instr->type, lanes);
                widened.push_back(instr);
                continue;
            }

            // Every lane's copy of a value has to be there for the ones that
            // vary to be combined with it, so uniform operands are broadcast
            // up to the gang width.
            vector<size_t> to_broadcast = value_operands(*instr);
            if (per_lane_write) {
                // The value, and the mask if there is one.
                for (size_t k = 1; k < instr->operands.size(); k++) {
                    to_broadcast.push_back(k);
                }
            }
            for (const size_t k : to_broadcast) {
                shared_ptr<Value> &operand = instr->operands[k];
                if (is_gang_wide(operand->get_type(), lanes)) {
                    continue;
                }
                operand = broadcast(func, block.shared_from_this(), operand,
                                    lanes, widened);
            }

            const bool folds = instr->op == Instruction::Op::AccAdd ||
                               instr->op == Instruction::Op::AccMul ||
                               instr->op == Instruction::Op::AccSub ||
                               instr->op == Instruction::Op::AccMin ||
                               instr->op == Instruction::Op::AccMax;

            // An accumulate whose place stays uniform but whose value now
            // varies is a cross-lane reduction: every lane folds into the one
            // place, so the gang's values are reduced to a scalar before the
            // accumulate, the way ISPC reduces a varying into a uniform. A
            // scatter -- a place that is itself per-lane, a vector of addresses
            // -- is element-wise instead and keeps its vector value, and so
            // is an accumulate into per-lane memory.
            //
            // One made by some lanes only -- it carries a mask -- folds in,
            // for a lane that is off, the identity of its operation instead
            // of that lane's value. The mask stays on the instruction: the
            // place is written once for the gang, and only if some lane is on.
            //
            // A per-lane short vector -- a `vec3f` per lane, carried as a
            // struct of one gang vector per component (see widen()) -- is
            // reduced component by component, and the components rebuilt into
            // the vector the place holds.
            if (folds && !per_lane_write && instr->operands.size() >= 2 &&
                (div.instrs.count(instr.get()) ||
                 is_gang_wide(instr->operands[1]->get_type(), lanes))) {
                ir::VectorReduce::OpType rop = ir::VectorReduce::Add;
                switch (instr->op) {
                case Instruction::Op::AccMul:
                    rop = ir::VectorReduce::Mul;
                    break;
                case Instruction::Op::AccMin:
                    rop = ir::VectorReduce::Min;
                    break;
                case Instruction::Op::AccMax:
                    rop = ir::VectorReduce::Max;
                    break;
                default: // AccAdd, AccSub: both fold the lanes with +
                    rop = ir::VectorReduce::Add;
                    break;
                }
                const shared_ptr<Value> mask =
                    instr->operands.size() == 3 ? instr->operands[2] : nullptr;
                const auto here = block.shared_from_this();

                // One gang vector of a component (or of the whole scalar value),
                // masked to the identity where a lane is off, reduced to one.
                const auto fold = [&](shared_ptr<Value> lanes_of) {
                    if (!is_gang_wide(lanes_of->get_type(), lanes)) {
                        lanes_of = broadcast(func, here, lanes_of, lanes, widened);
                    }
                    const Type element = lanes_of->get_type().element_of();
                    if (mask != nullptr) {
                        shared_ptr<Value> identity = broadcast(
                            func, here,
                            accumulate_identity(instr->op, element, func, here,
                                                widened),
                            lanes, widened);
                        auto select = std::make_shared<Instruction>(
                            func.get_unique_name(), lanes_of->get_type(),
                            Instruction::Op::Select,
                            vector<shared_ptr<Value>>{mask, lanes_of, identity},
                            here);
                        widened.push_back(select);
                        lanes_of = std::make_shared<Value>(select);
                    }
                    auto reduce = std::make_shared<Instruction>(
                        func.get_unique_name(), element, Instruction::Op::Reduce,
                        vector<shared_ptr<Value>>{lanes_of}, here);
                    reduce->reduce = rop;
                    widened.push_back(reduce);
                    return std::make_shared<Value>(reduce);
                };

                shared_ptr<Value> &value = instr->operands[1];
                const Type value_type = value->get_type();
                if (is_widened_vector(value_type)) {
                    const Struct_t *s = value_type.as<Struct_t>();
                    vector<shared_ptr<Value>> components;
                    for (uint32_t k = 0; k < uint32_t(s->fields.size()); k++) {
                        auto read = std::make_shared<Instruction>(
                            func.get_unique_name(), s->fields[k].type,
                            Instruction::Op::LoadField,
                            vector<shared_ptr<Value>>{
                                value, std::make_shared<Value>(Constant{
                                           UInt_t::make(32), uint64_t(k)})},
                            here);
                        widened.push_back(read);
                        components.push_back(fold(std::make_shared<Value>(read)));
                    }
                    const Type vector_type = Vector_t::make(
                        s->fields[0].type.element_of(), uint32_t(s->fields.size()));
                    auto rebuilt = std::make_shared<Instruction>(
                        func.get_unique_name(), vector_type,
                        Instruction::Op::MakeStruct, components, here);
                    widened.push_back(rebuilt);
                    value = std::make_shared<Value>(rebuilt);
                } else if (value_type.is<Struct_t>()) {
                    internal_error
                        << "[unimplemented] " << instr->name
                        << " accumulates a per-lane struct into shared memory";
                } else {
                    value = fold(value);
                }

                // Folded, the accumulate is made once for the gang, if any
                // lane is on, and its mask becomes that one bool. This is
                // what tells the backends the two masked accumulates apart:
                // a mask still a vector is one per lane, into per-lane
                // memory, and a value that is a vector is then a lane per
                // component. A short vector reduced into a shared slot -- a
                // film's normal, `vec3f` -- would otherwise read as three
                // lanes and be gated component by component on the first
                // three lanes' bits.
                if (mask != nullptr) {
                    auto any = std::make_shared<Instruction>(
                        func.get_unique_name(), Bool_t::make(),
                        Instruction::Op::Any, vector<shared_ptr<Value>>{mask},
                        here);
                    widened.push_back(any);
                    instr->operands[2] = std::make_shared<Value>(any);
                }
            }

            // A store has no result, and an instruction built per-lane in the
            // first place (the ramp) is already the right type; everything
            // else now produces one value per lane.
            if (instr->type.defined() && !instr->name.empty() &&
                !is_gang_wide(instr->type, lanes)) {
                instr->type = widen(instr->type, lanes);
            }
            widened.push_back(instr);
        }
        block.instrs = std::move(widened);
    }

    // A block argument that has become per-lane may still be handed a uniform
    // value: a loop-carried value that varies by the second iteration is
    // entered with whatever single value the loop started from. Those are
    // broadcast where they are passed, the same way a uniform operand of a
    // widened instruction is.
    for (const string &name : region) {
        Block &block = *blocks.at(name);
        for (Terminator::Jump *jump : jumps_of(block)) {
            const auto target = blocks.find(jump->name);
            if (target == blocks.end() || !region.count(jump->name)) {
                continue;
            }
            // A call continuation is handed the result as a leading argument
            // that no jump passes, so the values line up with the last ones.
            const size_t offset =
                target->second->args.size() - jump->args.size();
            for (size_t j = 0; j < jump->args.size(); j++) {
                const Type &wanted = target->second->args[j + offset].type;
                if (!is_gang_wide(wanted, lanes) ||
                    is_gang_wide(jump->args[j]->get_type(), lanes)) {
                    continue;
                }
                jump->args[j] = broadcast(func, block.shared_from_this(),
                                          jump->args[j], lanes, block.instrs);
            }
        }
    }

    if (dump) {
        std::cerr << "--- after widening " << entry << ":\n";
        func.dump(std::cerr);
    }
}

// Broadcasts the arguments of every call in the region that a variant callee
// takes per lane but the caller passes as one value.
//
// Which parameters a variant takes per lane was decided per *parameter* (see
// specialize_calls): a lane's own short vector, passed as its components, is
// varying as a whole when any component is, so every component of that
// parameter is gang-wide in the variant -- while a component the caller
// passes may be a constant or a uniform scalar (the zero in `vec3f{u, v, 0}`)
// that nothing in the caller had reason to widen. The types now say exactly
// which arguments fall short, and a splat closes the gap, the same way a
// uniform value handed to a per-lane block argument is splatted where it is
// passed (see the end of widen_region).
void broadcast_call_arguments(const FuncMap &funcs, Function &func,
                              const set<string> &region, uint32_t lanes) {
    const BlockMap blocks = make_block_map(func);
    for (const string &name : region) {
        Block &block = *blocks.at(name);
        auto *call = std::get_if<Terminator::Call>(&block.terminator.data);
        if (call == nullptr) {
            continue;
        }
        const auto callee = funcs.find(call->call.name);
        if (callee == funcs.end() || callee->second->blocks.empty()) {
            continue;
        }
        const vector<Argument> &params = callee->second->blocks.front()->args;
        internal_assert(params.size() == call->call.args.size())
            << "Call in " << name << " passes " << call->call.args.size()
            << " arguments to " << call->call.name << ", which takes "
            << params.size();
        for (size_t i = 0; i < params.size(); i++) {
            shared_ptr<Value> &arg = call->call.args[i];
            if (!is_gang_wide(params[i].type, lanes) ||
                is_gang_wide(arg->get_type(), lanes)) {
                continue;
            }
            arg = broadcast(func, block.shared_from_this(), arg, lanes,
                            block.instrs);
        }
    }
}

// How an argument reaches a callee from a gang: one value the lanes share, one
// value per lane (a vector), or one shared pointer into memory laid out per
// lane -- the caller's own `mut` local, say, that every lane has its own copy
// of (see Divergence::pointee_instrs).
enum class Shape { Uniform, Varying, Pointee };

// Why `v`, as referenced from `block`, is varying: the value, and beneath it
// the varying operands it is computed from, down to the arguments and seeds
// the divergence started at -- and for an argument, the blocks that declare
// it varying and what each of their predecessors passes for it. For reading a
// diagnostic; see the assertion that linearization left nothing divergent.
void explain_varying(const Function &func, const Divergence &div,
                     const string &block, const Value &v, int depth) {
    if (depth > 12) {
        std::cerr << string(2 * depth, ' ') << "...\n";
        return;
    }
    const string pad(2 * depth, ' ');
    std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) {
                std::cerr << pad;
                i->dump(std::cerr);
                std::cerr << "\n";
                for (const auto &operand : i->operands) {
                    if (div.is_varying(block, *operand)) {
                        explain_varying(func, div, block, *operand, depth + 1);
                    }
                }
            },
            [&](const Constant &c) {
                std::cerr << pad;
                c.dump(std::cerr);
                std::cerr << "\n";
            },
            [&](const Argument &a) {
                std::cerr << pad << "argument " << a.name << " : " << a.type
                          << ", declared varying by:\n";
                for (const auto &owner : func.blocks) {
                    if (!div.args.count({owner->name, a.name})) {
                        continue;
                    }
                    std::cerr << pad << "  " << owner->name
                              << (div.masked.count(owner->name) ? " (masked)"
                                                                : "")
                              << ", passed:\n";
                    size_t index = 0;
                    for (; index < owner->args.size(); index++) {
                        if (owner->args[index].name == a.name) {
                            break;
                        }
                    }
                    for (const auto &pred : func.blocks) {
                        for (Terminator::Jump *jump : jumps_of(*pred)) {
                            if (jump->name != owner->name) {
                                continue;
                            }
                            const size_t offset =
                                owner->args.size() - jump->args.size();
                            if (index < offset) {
                                continue;
                            }
                            const Value &passed = *jump->args[index - offset];
                            std::cerr << pad << "    from " << pred->name
                                      << (div.masked.count(pred->name)
                                              ? " (masked)"
                                              : "")
                                      << ": ";
                            passed.dump(std::cerr);
                            std::cerr << (div.is_varying(pred->name, passed)
                                              ? " (varying)"
                                              : " (uniform)")
                                      << "\n";
                        }
                    }
                }
            },
        },
        v.data);
}

// A callee, specialized for how a gang calls it: the shape each of its
// parameters arrives in, and whether the call is made under a mask.
struct VariantKey {
    string callee;
    vector<Shape> shapes;
    bool masked = false;
    uint32_t lanes = 0;

    bool operator<(const VariantKey &o) const {
        return std::tie(callee, shapes, masked, lanes) <
               std::tie(o.callee, o.shapes, o.masked, o.lanes);
    }
};

// The name a variant is generated under. A function called both ways ends up
// with two of these, and a scalar caller keeps calling the original.
string variant_name(const VariantKey &key) {
    string name = key.callee + "$gang" + std::to_string(key.lanes);
    for (size_t i = 0; i < key.shapes.size(); i++) {
        if (key.shapes[i] == Shape::Varying) {
            name += "_v" + std::to_string(i);
        } else if (key.shapes[i] == Shape::Pointee) {
            name += "_p" + std::to_string(i);
        }
    }
    return key.masked ? name + "$masked" : name;
}

// Every variant generated so far, so that a callee reached twice in the same
// shape is specialized once. Scoped to one vectorize() call.
using Variants = map<VariantKey, string>;

shared_ptr<Function> specialize(FuncMap &funcs, const VariantKey &key,
                                const string &name, Variants &variants);

// Points each call in `region` at a variant of its callee taking the
// arguments in the shape this gang has them in.
//
// A call that the lanes make unconditionally needs no mask: every lane is
// executing it, so the callee can store freely. A call inside folded control
// flow does need one, and gets the mask of the block it sits in as an extra
// argument. Both variants can exist at once, for a function called both ways,
// and a callee whose arguments are all uniform needs neither.
void specialize_calls(FuncMap &funcs, Function &func, const set<string> &region,
                      const Divergence &div, const BlockMasks &masks,
                      const set<string> &conditional_calls, uint32_t lanes,
                      const map<string, vector<uint32_t>> &call_shapes,
                      Variants &variants) {
    const BlockMap blocks = make_block_map(func);

    for (const string &name : region) {
        auto block = blocks.at(name);
        auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call == nullptr) {
            continue;
        }

        VariantKey key;
        key.callee = call->call.name;
        key.lanes = lanes;
        key.masked = conditional_calls.count(name) > 0;

        // The key describes the callee's own parameters, not the values being
        // passed: a per-lane vector argument was split into components, and
        // the callee's parameter has to be split the same way for the two to
        // line up. `call_shapes` says how many values each parameter took.
        const auto shape = call_shapes.find(name);
        const vector<uint32_t> components =
            shape != call_shapes.end()
                ? shape->second
                : vector<uint32_t>(call->call.args.size(), 1);

        size_t arg = 0;
        for (const uint32_t count : components) {
            internal_assert(arg < call->call.args.size())
                << "Call in " << name << " passes fewer arguments than the "
                << "split recorded";
            // A split argument is varying by construction; an unsplit one is
            // whatever the analysis says. Which arguments vary cannot come
            // from their types here, since the region is not widened yet.
            const Value &v = *call->call.args[arg];
            key.shapes.push_back(count > 1 || div.is_varying(name, v)
                                     ? Shape::Varying
                                 : div.points_to_varying(name, v)
                                     ? Shape::Pointee
                                     : Shape::Uniform);
            arg += count;
        }

        // Nothing to specialize: every argument arrives as it would from a
        // scalar caller and no mask is needed, so the original function is
        // already the right one to call.
        const bool needs_variant =
            key.masked ||
            std::any_of(key.shapes.begin(), key.shapes.end(),
                        [](Shape s) { return s != Shape::Uniform; });
        if (!needs_variant) {
            continue;
        }

        const auto cached = variants.find(key);
        string name_of_variant;
        if (cached != variants.end()) {
            name_of_variant = cached->second;
        } else {
            name_of_variant = variant_name(key);
            // Recorded before specializing, so that a callee that reaches
            // itself is caught rather than specialized forever.
            variants[key] = name_of_variant;
            specialize(funcs, key, name_of_variant, variants);
        }
        call->call.name = name_of_variant;

        if (key.masked) {
            // The mask the call site runs under, which linearization computed
            // when it folded the branch that made the call conditional.
            const auto mask = masks.find(name);
            internal_assert(mask != masks.end())
                << "Call in " << name << " is conditional but its block has "
                << "no mask";
            call->call.args.push_back(mask->second);
        }
    }
}

// Builds the variant: a copy of the callee whose varying parameters are
// vectors, vectorized the same way a ParFor body is.
//
// A masked variant takes the caller's execution mask as a final parameter and
// runs entirely under it, so its stores write only the lanes the caller had
// enabled. An unmasked variant has no such parameter and no such masking,
// which is why it is worth having both: a call that is always executed should
// not pay for predication (ispc section 5.7 passes the mask the same way, and
// only for functions that need it).
shared_ptr<Function> specialize(FuncMap &funcs, const VariantKey &key,
                                const string &name, Variants &variants) {
    const auto original = funcs.find(key.callee);
    internal_assert(original != funcs.end())
        << "Cannot vectorize a call to unknown function: " << key.callee;

    auto variant = clone_function(*original->second);
    // Only what the entry reaches: a block an earlier rewrite disconnected --
    // the continuation of a call loopify() turned into a jump -- is not part
    // of the region the passes below work on, and yet still passes arguments
    // to blocks that are.
    remove_unreachable_blocks(*variant);
    Block &entry_block = *variant->blocks.front();

    // The entry block carries the function's name (see FunctionBuilder in
    // SSA/Convert.cpp), and that is the name the generated code is emitted
    // under -- so without renaming it the variant would collide with the
    // function it was specialized from.
    const string old_entry = entry_block.name;
    entry_block.name = name;
    for (const auto &block : variant->blocks) {
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &j) {
                           if (j.name == old_entry) {
                               j.name = name;
                           }
                       },
                       [&](Terminator::Dispatch &d) {
                           for (auto &target : d.targets) {
                               if (target.name == old_entry) {
                                   target.name = name;
                               }
                           }
                       },
                       [&](Terminator::Return &) {},
                       [&](Terminator::ParFor &) {},
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &c) {
                           if (c.cont.name == old_entry) {
                               c.cont.name = name;
                           }
                       },
                       [&](Terminator::MultiCall &c) {
                           if (c.cont.name == old_entry) {
                               c.cont.name = name;
                           }
                       },
                   },
                   block->terminator.data);
    }
    internal_assert(entry_block.args.size() == key.shapes.size())
        << "Call to " << key.callee << " passes " << key.shapes.size()
        << " arguments to a function taking " << entry_block.args.size();

    // The varying parameters become one value per lane. They are seeded into
    // the divergence analysis by name, the same way a loop index is. A
    // parameter pointing into the caller's per-lane memory is seeded as such:
    // it stays one address, and what it addresses is per lane.
    set<string> varying_names;
    set<string> pointee_names;
    for (size_t i = 0; i < key.shapes.size(); i++) {
        if (key.shapes[i] == Shape::Varying) {
            varying_names.insert(entry_block.args[i].name);
        } else if (key.shapes[i] == Shape::Pointee) {
            pointee_names.insert(entry_block.args[i].name);
        }
    }

    shared_ptr<Value> mask;
    if (key.masked) {
        // Declared scalar and widened below with everything else varying, so
        // that the masks computed inside the function have the same shape.
        const Argument mask_arg{Bool_t::make(), "!mask"};
        entry_block.args.push_back(mask_arg);
        mask = std::make_shared<Value>(mask_arg);
        entry_block.lookups[mask_arg.name] = mask;
        varying_names.insert(mask_arg.name);
    }

    const string entry = entry_block.name;
    promote_allocas(*variant, entry);

    // Linearization folds a region down to a single path, so the function
    // needs a single exit for that path to end at.
    unify_returns(*variant);

    // The analysis of this region, as its seeds accumulate: the varying
    // parameters throughout, and what uniformizing its loops adds.
    const auto analyze = [&](const set<std::pair<string, string>> &varying_args,
                             const set<string> &masked_blocks) {
        return analyze_divergence(*variant, entry, varying_names, {},
                                  varying_args, pointee_names, mask,
                                  masked_blocks);
    };

    // A loop the lanes leave at different iterations -- which is what a
    // traversal turned into a loop by loopify() is -- becomes one they leave
    // together, with the lanes that are done masked off (see
    // SSA/UniformizeLoops.h).
    const LoopUniformization uniform =
        uniformize_loops(*variant, entry, analyze);
    const set<std::pair<string, string>> varying_args = uniform.varying_args;
    const set<string> masked_blocks = uniform.masked_blocks();

    const Divergence linearizable = analyze(varying_args, masked_blocks);

    // Which of the calls inside are conditional, decided before the branches
    // that make them so are folded away -- and after the loops are folded,
    // since a call in a loop some lanes have left is conditional on their
    // still being in it. A masked variant runs entirely under its caller's
    // mask, so every call it makes is conditional too.
    set<string> conditional_calls;
    {
        const set<string> region =
            reachable_from(entry, compute_successors(*variant));
        const BlockMap blocks = make_block_map(*variant);
        for (const string &block_name : region) {
            if (std::holds_alternative<Terminator::Call>(
                    blocks.at(block_name)->terminator.data) &&
                (key.masked || linearizable.masked.count(block_name))) {
                conditional_calls.insert(block_name);
            }
        }
    }

    // The graph the linearizer is handed, for when what it leaves behind is
    // wrong: which branches it was told were divergent, and what each join
    // was passed, are only readable off this.
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- before linearizing " << name << ":\n";
        variant->dump(std::cerr);
    }
    BlockMasks masks =
        linearize(*variant, entry, linearizable, mask, uniform.loops);
    // Then a uniform branch around each block no lane may be on, so that a
    // gang skips the arms none of its lanes take (see
    // SSA/SkipInactiveBlocks.h).
    skip_inactive_blocks(*variant, entry, masks, mask);

    // Uniformizing a loop adds blocks, so the region is only settled now.
    const set<string> region =
        reachable_from(entry, compute_successors(*variant));

    // Per-lane vectors become one value per component here too, which is what
    // turns a `vec3f` parameter into three `f32` ones -- matching the
    // components the caller hands over.
    const SplitResult split = split_aggregates(
        *variant, entry, analyze(varying_args, masked_blocks));
    varying_names.insert(split.parameters.begin(), split.parameters.end());

    const Divergence div = analyze(varying_args, masked_blocks);
    if (!div.branches.empty()) {
        // The graph as it stands, since which branch is still divergent, and
        // why, is only readable off it -- and for each, the chain of values
        // its condition varies through, down to what seeded it.
        std::cerr << "--- after linearizing " << name << ", still divergent:";
        for (const string &b : div.branches) {
            std::cerr << " " << b;
        }
        std::cerr << "\n";
        const BlockMap blocks = make_block_map(*variant);
        for (const string &b : div.branches) {
            const auto *d = std::get_if<Terminator::Dispatch>(
                &blocks.at(b)->terminator.data);
            if (d != nullptr) {
                std::cerr << "--- why " << b << " diverges:\n";
                explain_varying(*variant, div, b, *d->cond, 0);
            }
        }
        variant->dump(std::cerr);
    }
    internal_assert(div.branches.empty())
        << "Linearization left a divergent branch in " << name;

    // A variant's own calls are specialized the same way, so that a chain of
    // calls from inside a gang is vectorized all the way down.
    specialize_calls(funcs, *variant, region, div, masks, conditional_calls,
                     key.lanes, split.call_shapes, variants);

    widen_region(*variant, entry, div, key.lanes);
    broadcast_call_arguments(funcs, *variant, region, key.lanes);

    // The return type follows whatever the exit ends up carrying, which is a
    // vector if the returned value turned out to be varying.
    for (const auto &block : variant->blocks) {
        if (const auto *ret =
                std::get_if<Terminator::Return>(&block->terminator.data);
            ret != nullptr && ret->value) {
            variant->ret_type = ret->value->get_type();
        }
    }

    // A variant is a gang's code as much as the body it was specialized for,
    // and is generated the same way: straight from this form (see the end of
    // vectorize()).
    auto &attrs = variant->attributes;
    if (std::find(attrs.begin(), attrs.end(),
                  ir::Function::Attribute::vectorized) == attrs.end()) {
        attrs.push_back(ir::Function::Attribute::vectorized);
    }

    funcs[name] = variant;
    return variant;
}

} // namespace

void vectorize(FuncMap &funcs, std::string func, std::string idx) {
    internal_assert(funcs.contains(func))
        << "vectorize applied to unknown func:" << func;
    auto f = funcs[func];
    remove_unreachable_blocks(*f); // see specialize()

    // The loop to vectorize, and the region its body spans.
    shared_ptr<Block> loop;
    for (const auto &block : f->blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == idx) {
            internal_assert(!loop)
                << "Two loops named " << idx << " in " << func;
            loop = block;
        }
    }
    internal_assert(loop) << "Did not find loop: " << idx
                          << " in function: " << func;

    const Terminator::ParFor parfor =
        std::get<Terminator::ParFor>(loop->terminator.data);
    const uint32_t lanes = gang_width(parfor);

    // A mutable local written by one lane is a value that differs between
    // lanes, not memory traffic: promote before anything else looks at the
    // body (see SSA/PromoteAllocas.h).
    promote_allocas(*f, f->blocks[0]->name);

    const string entry = parfor.body.name;

    // Which calls are made under a mask has to be settled before the branches
    // are folded away, since folding them is what makes a conditional call
    // unconditional in the control flow.
    // A loop the lanes leave at different iterations becomes one they leave
    // together, with the lanes that are done masked off (see
    // SSA/UniformizeLoops.h). This has to happen before linearization, which
    // requires every loop it sees to be uniform.
    const LoopUniformization uniform = uniformize_loops(
        *f, entry,
        [&](const set<std::pair<string, string>> &so_far,
            const set<string> &masked_so_far) {
            return analyze_divergence(*f, entry, {idx}, {}, so_far, {},
                                      nullptr, masked_so_far);
        });
    const set<std::pair<string, string>> varying_args = uniform.varying_args;
    const set<string> masked_blocks = uniform.masked_blocks();

    const Divergence before = analyze_divergence(*f, entry, {idx}, {},
                                                 varying_args, {}, nullptr,
                                                 masked_blocks);

    // Which calls are made under a mask, settled before the branches that make
    // them so are folded away -- and after the loops are, since a call in a
    // loop some lanes have left is conditional on their still being in it.
    set<string> conditional_calls;
    {
        const BlockMap blocks = make_block_map(f);
        for (const string &name :
             reachable_from(entry, compute_successors(*f))) {
            if (std::holds_alternative<Terminator::Call>(
                    blocks.at(name)->terminator.data) &&
                before.masked.count(name)) {
                conditional_calls.insert(name);
            }
        }
    }

    // Fold away the branches the lanes disagree about, so that what is left
    // is control flow every lane follows together, with masks standing in for
    // the branches that were folded (see SSA/Linearize.h).
    BlockMasks masks = linearize(*f, entry, before, nullptr, uniform.loops);
    skip_inactive_blocks(*f, entry, masks);

    const BlockMap blocks = make_block_map(f);
    const AdjacencyMap all_succs = compute_successors(*f);
    const set<string> region = reachable_from(entry, all_succs);

    internal_assert(parfor.cont.args.empty())
        << "TODO: thread the continuation arguments of " << idx
        << " through its body";

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }

    // The lane indices, which is what the loop index becomes. An index of
    // this shape is what later makes an access to a[i] a dense vector load
    // rather than a gather.
    Block &body = *blocks.at(entry);
    auto ramp = std::make_shared<Instruction>(
        f->get_unique_name(), widen(parfor.start->get_type(), lanes),
        Instruction::Op::Ramp,
        vector<shared_ptr<Value>>{parfor.start, parfor.stride},
        body.shared_from_this());
    body.instrs.insert(body.instrs.begin(), ramp);
    auto ramp_value = std::make_shared<Value>(ramp);

    internal_assert(!body.args.empty() && body.args[0].name == idx)
        << "ParFor body " << entry << " does not take " << idx
        << " as its first argument";
    body.args.erase(body.args.begin());

    // Everywhere, not just in the body block: linearization drops the block
    // arguments that used to thread the index onwards, so blocks further
    // along the region refer to the index directly.
    for (const string &name : region) {
        substitute(*blocks.at(name), idx, ramp_value);
    }

    // A lane's own vector -- a vec3f per lane, say -- cannot be widened as it
    // is, since a gang of them would be a vector of vectors. Split those into
    // one value per component first (see SSA/SplitAggregates.h).
    const SplitResult split = split_aggregates(
        *f, entry,
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks),
        {ramp.get()});

    // Re-run the analysis now that the region is linearized and the index is
    // the ramp: the masks and blends linearization introduced have to be
    // classified too, and the index is no longer a block argument to seed on.
    const Divergence div =
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks);
    internal_assert(div.branches.empty())
        << "Linearization left a divergent branch in " << *div.branches.begin();

    Variants variants;
    specialize_calls(funcs, *f, region, div, masks, conditional_calls, lanes,
                     split.call_shapes, variants);

    widen_region(*f, entry, div, lanes);
    broadcast_call_arguments(funcs, *f, region, lanes);

    // With the body vectorized, the loop is gone: it runs exactly once, so
    // its header falls straight into the body and the body's Yield falls
    // through to what followed the loop.
    loop->terminator.data =
        Terminator::Jump{parfor.body.name, parfor.body.args};

    for (const string &name : region) {
        auto block = blocks.at(name);
        if (!std::holds_alternative<Terminator::Yield>(
                block->terminator.data)) {
            continue;
        }
        block->terminator.data = Terminator::Jump{parfor.cont.name};

        auto cont = blocks.at(parfor.cont.name);
        std::erase_if(cont->preds, [&](const std::weak_ptr<Block> &p) {
            const auto ptr = p.lock();
            return ptr && ptr->name == loop->name;
        });
        cont->preds.push_back(block);
    }

    // Say that this function has been through here, so that code generation
    // can take its SSA form directly rather than the statements the relooper
    // rebuilds from it. A gang's control flow is what partial linearization
    // left behind, which is the shape structured statements fit worst.
    auto &attrs = f->attributes;
    if (std::find(attrs.begin(), attrs.end(),
                  ir::Function::Attribute::vectorized) == attrs.end()) {
        attrs.push_back(ir::Function::Attribute::vectorized);
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
