#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/CloneFunction.h"
#include "SSA/InvariantDivision.h"
#include "SSA/Linearize.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"
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
                              [](const Undefined &) -> int64_t {
                                  internal_error << "undefined loop bound";
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

    // Ask about the gang as a whole, so they are uniform and never widened
    // (see analyze_divergence); their one operand is the per-lane value
    // asked about. Listed so that a widening that did reach one would say
    // which operand is the value rather than fall through to the error.
    case Instruction::Op::Any:
    case Instruction::Op::Popcount:
    case Instruction::Op::Vote:
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

    // A gang's push: the entry is what each lane pushes its own of, and the
    // queue is the one queue. The mask, when the push carries one, is a gang
    // mask already. How the lanes go into the queue -- a compaction, the
    // lanes that push taking consecutive slots and the count advancing once
    // by their number -- is lower_pushes's (SSA/Defer.cpp), after the
    // schedule; here the push stays one instruction, its entry per lane and
    // its value, the slot each lane took, per lane too (see
    // analyze_divergence).
    case Instruction::Op::Push:
        return {1};

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
void widen_argument(const Cfg &region, BlockId owner, const string &name,
                    uint32_t lanes, bool pointee = false) {
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

    for (auto &arg : region[owner].args) {
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

    for (BlockId b = 0; b < region.size(); b++) {
        Block &block = region[b];
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
    // An undefined value is undefined at the gang's width too: nothing to
    // splat, just the wider type.
    if (const auto *c = std::get_if<Constant>(&value->data);
        c != nullptr && std::holds_alternative<Undefined>(c->data)) {
        return undef_value(widen(type, lanes));
    }
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

    const Cfg region(func, entry);

    const auto is_write = [](const Instruction &instr) {
        return instr.op == Instruction::Op::Store ||
               instr.op == Instruction::Op::AccAdd ||
               instr.op == Instruction::Op::AccMul ||
               instr.op == Instruction::Op::AccSub ||
               instr.op == Instruction::Op::AccMin ||
               instr.op == Instruction::Op::AccMax;
    };

    for (BlockId b : region.rpo) {
        Block &block = region[b];
        const string &name = block.name;

        for (const Argument &arg : block.args) {
            const bool pointee = div.pointee_args.count({name, arg.name}) > 0;
            if (div.args.count({name, arg.name})) {
                internal_assert(!pointee)
                    << "Argument " << arg.name << " of " << name
                    << " is a per-lane pointer into per-lane memory, which "
                    << "vectorization does not lay out yet";
                widen_argument(region, b, arg.name, lanes);
            } else if (pointee) {
                widen_argument(region, b, arg.name, lanes, /*pointee=*/true);
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
    for (BlockId b = 0; b < region.size(); b++) {
        Block &block = region[b];
        for (Terminator::Jump *jump : jumps_of(block)) {
            const BlockId to = region.find(jump->name);
            if (to == NO_BLOCK) {
                continue;
            }
            const Block &target = region[to];
            // A call continuation is handed the result as a leading argument
            // that no jump passes, so the values line up with the last ones.
            const size_t offset = target.args.size() - jump->args.size();
            for (size_t j = 0; j < jump->args.size(); j++) {
                const Type &wanted = target.args[j + offset].type;
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
                              const Cfg &region, uint32_t lanes) {
    for (BlockId b = 0; b < region.size(); b++) {
        Block &block = region[b];
        const string &name = block.name;
        Terminator::Jump *call = block.terminator.callee();
        if (call == nullptr) {
            continue;
        }
        const auto callee = funcs.find(call->name);
        if (callee == funcs.end() || callee->second->blocks.empty()) {
            continue;
        }
        const vector<Argument> &params = callee->second->blocks.front()->args;
        internal_assert(params.size() == call->args.size())
            << "Call in " << name << " passes " << call->args.size()
            << " arguments to " << call->name << ", which takes "
            << params.size();
        const auto widen_to = [&](size_t i, shared_ptr<Value> &arg) {
            if (!is_gang_wide(params[i].type, lanes) ||
                is_gang_wide(arg->get_type(), lanes)) {
                return;
            }
            arg = broadcast(func, block.shared_from_this(), arg, lanes,
                            block.instrs);
        };
        for (size_t i = 0; i < params.size(); i++) {
            widen_to(i, call->args[i]);
        }
        // A run's per-call values stand in for the parameters at
        // `varying_at`, and fall short the same way.
        if (auto *run =
                std::get_if<Terminator::MultiCall>(&block.terminator.data)) {
            for (auto &vs : run->varying) {
                for (size_t k = 0; k < vs.size(); k++) {
                    widen_to(run->varying_at[k], vs[k]);
                }
            }
        }
    }
}

// Settles every vote in the region (see Instruction::Op::Vote) now that the
// region is a gang's and its masks are known.
//
// A vote on a value the lanes agree about is that value. So is a vote that
// decides between values that are already the lanes' own: a run whose
// children are one node per lane -- a traversal entered at a different root
// by every lane, an instance's tree -- is descended one node per lane
// whichever way each is ordered, so there each lane's comparison stands. The
// vote is for the run whose children the lanes share, where deciding per lane
// would make them per lane and the gang could no longer descend as one. There
// it is the majority of the lanes that are on: `2 * popcount(x & m) >
// popcount(m)`, with `m` the block's execution mask -- the region's own when
// the block has no narrower one, and every lane when it has none -- so that a
// lane that is off has no say, and a tie leaves the order as written. The
// count over the mask is what a packet tracer does by hand at every node
// (Wald et al. 2001 take the direction sign the packet's rays share; Embree's
// packet traversal reduces the rays' entry distances to one per child before
// ordering them -- Wald, Woop, Benthin, Johnson & Ernst, "Embree: A Kernel
// Framework for Efficient CPU Ray Tracing", SIGGRAPH 2014), and here it falls
// out of treating the run's order as one decision.
void lower_votes(Function &func, const string &entry, const Divergence &div,
                 const BlockMasks &masks, const shared_ptr<Value> &entry_mask,
                 uint32_t lanes) {
    const Cfg region(func, entry);
    const Type count_type = UInt_t::make(32);
    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
        const string &name = block->name;
        vector<shared_ptr<Instruction>> votes;
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Vote) {
                votes.push_back(instr);
            }
        }
        for (const auto &vote : votes) {
            internal_assert(vote->operands.size() == 1)
                << "A vote on " << vote->operands.size() << " values in "
                << name;
            const shared_ptr<Value> asked = vote->operands[0];

            // Does the vote decide between values the lanes share? The
            // network's selects are in this block, and the ones over the
            // run's children have those children as their arms.
            bool decides_shared = false;
            for (const auto &instr : block->instrs) {
                if (instr->op != Instruction::Op::Select ||
                    instr->operands.size() != 3) {
                    continue;
                }
                const auto *cond = std::get_if<shared_ptr<Instruction>>(
                    &instr->operands[0]->data);
                if (cond == nullptr || cond->get() != vote.get()) {
                    continue;
                }
                if (!div.is_varying(name, *instr->operands[1]) &&
                    !div.is_varying(name, *instr->operands[2])) {
                    decides_shared = true;
                    break;
                }
            }
            if (!div.is_varying(name, *asked) || !decides_shared) {
                replace_uses(func, vote.get(), asked);
                continue;
            }

            // Before the vote, where its operand is already defined.
            auto at = std::find(block->instrs.begin(), block->instrs.end(),
                                vote);
            internal_assert(at != block->instrs.end());
            const auto emit = [&](Type type, Instruction::Op op,
                                  vector<shared_ptr<Value>> operands) {
                auto instr = std::make_shared<Instruction>(
                    func.get_unique_name(), std::move(type), op,
                    std::move(operands), block);
                at = block->instrs.insert(at, instr) + 1;
                return std::make_shared<Value>(instr);
            };

            const auto found = masks.find(name);
            const shared_ptr<Value> mask =
                found != masks.end() ? found->second : entry_mask;
            shared_ptr<Value> on = asked;
            shared_ptr<Value> total = std::make_shared<Value>(
                Constant{count_type, uint64_t(lanes)});
            if (mask != nullptr) {
                on = emit(Bool_t::make(), Instruction::Op::LAnd, {asked, mask});
                total = emit(count_type, Instruction::Op::Popcount, {mask});
            }
            auto yes = emit(count_type, Instruction::Op::Popcount, {on});
            auto twice = emit(count_type, Instruction::Op::Add, {yes, yes});
            auto majority =
                emit(Bool_t::make(), Instruction::Op::Lt, {total, twice});
            replace_uses(func, vote.get(), majority);
        }
    }
}

// How an argument reaches a callee from a gang: one value the lanes share, one
// value per lane (a vector), or one shared pointer into memory laid out per
// lane -- the caller's own `mut` local, say, that every lane has its own copy
// of (see Divergence::pointee_instrs).
enum class Shape { Uniform, Varying, Pointee };

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
                                const string &name,
                                const ir::BranchPolicyMap &policies,
                                Variants &variants);

// Puts the masked call that ends `b` behind a test of its mask: the call
// moves to a block of its own, entered when any lane is on and bypassed
// straight to the continuation otherwise, which is then handed an undefined
// value for the value the call would have returned: no lane is active to read
// it, and the backend need build nothing for it (see ir::Undef). The mask is
// the call's last argument (see specialize_calls). The block joins the
// region.
void guard_call(Function &func, Cfg &region, BlockId b) {
    const shared_ptr<Block> block = region.block(b);
    Terminator::Jump *call = block->terminator.callee();
    internal_assert(call != nullptr && !call->args.empty())
        << "guard_call on " << block->name << ", which makes no call";
    const shared_ptr<Value> mask = call->args.back();

    auto guarded = std::make_shared<Block>();
    guarded->name = block->name + "!call";
    for (size_t i = 0; std::any_of(func.blocks.begin(), func.blocks.end(),
                                   [&](const shared_ptr<Block> &other) {
                                       return other->name == guarded->name;
                                   });
         i++) {
        guarded->name = block->name + "!call" + std::to_string(i);
    }
    guarded->owner = block->owner;
    guarded->terminator = std::move(block->terminator);

    // What the bypass hands the continuation: what the call's edge does, and
    // first, where the call's value would have gone, an undefined value -- no
    // lane is active to read it, since none was active to make the call.
    const Terminator::Jump *cont = guarded->terminator.continuation();
    internal_assert(cont != nullptr);
    Terminator::Jump bypass = *cont;
    const bool drop = std::visit(
        overloads{[](const Terminator::Call &c) { return c.drop; },
                  [](const Terminator::MultiCall &c) { return c.drop; },
                  [](const auto &) { return true; }},
        guarded->terminator.data);
    if (!drop) {
        const Block &target = region[region.id(cont->name)];
        internal_assert(!target.args.empty())
            << "Call continuation " << target.name
            << " takes no result argument";
        bypass.args.insert(bypass.args.begin(),
                           undef_value(target.args[0].type));
    }

    auto any = std::make_shared<Instruction>(
        func.get_unique_name(), Bool_t::make(), Instruction::Op::Any,
        vector<shared_ptr<Value>>{mask}, block);
    block->instrs.push_back(any);
    // targets[0] is where a false condition goes: past the call.
    block->terminator.data = Terminator::Dispatch{
        std::make_shared<Value>(any),
        {std::move(bypass), Terminator::Jump{guarded->name}}};

    func.blocks.insert(
        std::find(func.blocks.begin(), func.blocks.end(), block) + 1, guarded);
    region.add_block(guarded);
}

// Points each call in `region` at a variant of its callee taking the
// arguments in the shape this gang has them in.
//
// A call that the lanes make unconditionally needs no mask: every lane is
// executing it, so the callee can store freely. A call inside folded control
// flow does need one, and gets the mask of the block it sits in as an extra
// argument. Both variants can exist at once, for a function called both ways,
// and a callee whose arguments are all uniform needs neither.
void specialize_calls(FuncMap &funcs, Function &func, Cfg &region,
                      const Divergence &div, const BlockMasks &masks,
                      const shared_ptr<Value> &entry_mask,
                      const set<string> &conditional_calls, uint32_t lanes,
                      const map<string, vector<uint32_t>> &call_shapes,
                      const ir::BranchPolicyMap &policies,
                      Variants &variants) {
    // For the tests below: the region as linearization left it, its guards
    // in place.
    const DomTree dom = compute_dominator_tree(region);
    // Whether some lane of `mask` is on in block `b`: a test on the way
    // there says so, or the mask is the region's own -- a masked variant is
    // called only behind a test of the mask it is handed (see below), so its
    // parameter has a lane on throughout.
    const auto nonempty = [&](BlockId b, const Value &mask) {
        return (entry_mask != nullptr && same_value(mask, *entry_mask)) ||
               known_nonempty(region, dom, b, mask);
    };
    // The conditional calls whose mask no test on the way has shown to have
    // a lane on. Guarded once the walk is over, since a guard is a block.
    vector<BlockId> unguarded;

    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
        const string &name = block->name;
        Terminator::Jump *call = block->terminator.callee();
        if (call == nullptr) {
            continue;
        }
        // A run of calls (see Terminator::MultiCall) is specialized once for
        // all of them: they share every argument but the ones at
        // `varying_at`, and a parameter there arrives in whatever shape the
        // widest of the run's values for it has.
        const auto *run =
            std::get_if<Terminator::MultiCall>(&block->terminator.data);
        const auto values_at = [&](size_t position) {
            vector<const Value *> values{call->args[position].get()};
            if (run != nullptr) {
                for (size_t k = 0; k < run->varying_at.size(); k++) {
                    if (run->varying_at[k] != position) {
                        continue;
                    }
                    values.clear();
                    for (const auto &vs : run->varying) {
                        values.push_back(vs[k].get());
                    }
                }
            }
            return values;
        };

        VariantKey key;
        key.callee = call->name;
        key.lanes = lanes;
        // A recursive callee takes the masked form whether or not this call
        // is conditional. Its own recursive calls are conditional -- a
        // traversal descends into a child for the lanes whose ray meets the
        // node, and no others -- so the masked variant exists anyway, and an
        // unmasked one besides it would be the same recursion twice over:
        // the root visited by one copy, every other node by the other, and
        // loopify() finding a stack to make of only the second. A call every
        // lane makes passes a full mask instead, the way a packet tracer
        // starts at the root with every ray in the packet.
        const bool conditional = conditional_calls.count(name) > 0;
        const auto callee = funcs.find(call->name);
        const bool recurses =
            callee != funcs.end() && is_recursive(*callee->second);
        key.masked = conditional || recurses;

        // The key describes the callee's own parameters, not the values being
        // passed: a per-lane vector argument was split into components, and
        // the callee's parameter has to be split the same way for the two to
        // line up. `call_shapes` says how many values each parameter took.
        const auto shape = call_shapes.find(name);
        const vector<uint32_t> components =
            shape != call_shapes.end() ? shape->second
                                       : vector<uint32_t>(call->args.size(), 1);

        size_t arg = 0;
        for (const uint32_t count : components) {
            internal_assert(arg < call->args.size())
                << "Call in " << name << " passes fewer arguments than the "
                << "split recorded";
            // A split argument is varying by construction; an unsplit one is
            // whatever the analysis says. Which arguments vary cannot come
            // from their types here, since the region is not widened yet.
            Shape shape_of = count > 1 ? Shape::Varying : Shape::Uniform;
            for (const Value *v : values_at(arg)) {
                if (div.is_varying(name, *v)) {
                    shape_of = Shape::Varying;
                } else if (shape_of == Shape::Uniform &&
                           div.points_to_varying(name, *v)) {
                    shape_of = Shape::Pointee;
                }
            }
            key.shapes.push_back(shape_of);
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
            specialize(funcs, key, name_of_variant, policies, variants);
        }
        call->name = name_of_variant;

        if (conditional) {
            // The mask the call site runs under, which linearization computed
            // when it folded the branch that made the call conditional. For a
            // run it is shared by every call, as the arguments after
            // `varying_at` are.
            const auto mask = masks.find(name);
            internal_assert(mask != masks.end())
                << "Call in " << name << " is conditional but its block has "
                << "no mask";
            call->args.push_back(mask->second);
            // A call made under a mask is made only if some lane is on.
            // Linearization runs the block it sits in whether or not any
            // lane reached it, and while everything per lane in the callee
            // is masked, its uniform work is not: a table it indexes with a
            // value no lane computed, a pointer a fully-off join never
            // filled in. In the original program no thread was there to
            // call it, so nothing is lost by not calling it either -- the
            // same rule as a store into shared memory under a mask, and
            // ispc's (a call is never SafeToRunWithMaskAllOff). The arm the
            // call is in is normally behind the linearizer's own test of the
            // mask, which settles it; a call nothing has tested for gets a
            // test of its own (see guard_call).
            if (!nonempty(b, *mask->second)) {
                unguarded.push_back(b);
            }
        } else if (key.masked) {
            // Every lane is on. One bool, as every mask is before widening,
            // which broadcasts it where the variant's parameter is per lane
            // (see broadcast_call_arguments).
            call->args.push_back(
                std::make_shared<Value>(Constant{Bool_t::make(), true}));
        }
    }

    for (BlockId b : unguarded) {
        guard_call(func, region, b);
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
                                const string &name,
                                const ir::BranchPolicyMap &policies,
                                Variants &variants) {
    const auto original = funcs.find(key.callee);
    internal_assert(original != funcs.end())
        << "Cannot vectorize a call to unknown function: " << key.callee;

    auto variant = clone_function(*original->second);
    variant->specialized_from = key.callee;
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
        const Cfg before_folding(*variant, entry);
        for (const auto &block : before_folding.blocks()) {
            if (block->terminator.callee() != nullptr &&
                (key.masked || linearizable.masked.count(block->name))) {
                conditional_calls.insert(block->name);
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
    // Which also puts a uniform branch around each arm no lane may be on, so
    // that a gang skips the arms none of its lanes take -- where it must, and
    // where the schedule says under the callee's own name.
    BlockMasks masks = linearize(*variant, entry, linearizable, mask,
                                 uniform.loops, policies, key.callee);
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- after linearizing " << name << ":\n";
        variant->dump(std::cerr);
    }

    // With the masks known, a run's votes on its order become counts of the
    // lanes that are on (see lower_votes). Before the analysis below, which
    // has to see the counts and not the votes.
    lower_votes(*variant, entry, analyze(varying_args, masked_blocks), masks,
                mask, key.lanes);

    // Uniformizing a loop adds blocks, so the region is only settled now.
    Cfg region(*variant, entry);

    // Per-lane vectors become one value per component here too, which is what
    // turns a `vec3f` parameter into three `f32` ones -- matching the
    // components the caller hands over.
    const SplitResult split = split_aggregates(
        *variant, entry, analyze(varying_args, masked_blocks));
    varying_names.insert(split.parameters.begin(), split.parameters.end());

    // A division the gang makes, whose operands are known small enough, is
    // done in floating point (see SSA/InvariantDivision.h); before the
    // analysis below, which then sees the conversions and the float division
    // as the varying values they are.
    divide_bounded_by_floats(*variant, analyze(varying_args, masked_blocks));
    // And a division by a divisor the lanes agree on gets one multiplier,
    // computed on the scalar (see SSA/InvariantDivision.h).
    divide_by_uniform_divisors(*variant, analyze(varying_args, masked_blocks),
                               entry);
    // And a multiplier the lanes each have their own of, for a divisor known
    // small, is two double divisions rather than a 128-bit one per lane.
    expand_bounded_multipliers(*variant, analyze(varying_args, masked_blocks));

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
    specialize_calls(funcs, *variant, region, div, masks, mask,
                     conditional_calls, key.lanes, split.call_shapes, policies,
                     variants);

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

void vectorize(FuncMap &funcs, std::string func, std::string idx,
               const ir::BranchPolicyMap &policies) {
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

    // Linearization folds the body down to a single path, so the body needs
    // a single end for that path to reach -- as a specialized callee gets one
    // return (see specialize). A body with several yields -- a `continue` in
    // each arm of a branch, a split's tail beside its body, a drain's
    // finished entries beside its saved ones -- gets one block they all
    // jump to.
    unify_yields(*f, entry);

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
        const Cfg before_folding(*f, entry);
        for (const auto &block : before_folding.blocks()) {
            if (block->terminator.callee() != nullptr &&
                before.masked.count(block->name)) {
                conditional_calls.insert(block->name);
            }
        }
    }

    // Fold away the branches the lanes disagree about, so that what is left
    // is control flow every lane follows together, with masks standing in for
    // the branches that were folded (see SSA/Linearize.h).
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- before linearizing " << func << ":\n";
        f->dump(std::cerr);
    }
    BlockMasks masks = linearize(*f, entry, before, nullptr, uniform.loops,
                                 policies, func);
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- after linearizing " << func << ":\n";
        f->dump(std::cerr);
    }
    lower_votes(*f, entry,
                analyze_divergence(*f, entry, {idx}, {}, varying_args, {},
                                   nullptr, masked_blocks),
                masks, nullptr, lanes);

    Cfg region(*f, entry);

    internal_assert(parfor.cont.args.empty())
        << "TODO: thread the continuation arguments of " << idx
        << " through its body";

    // The lane indices, which is what the loop index becomes. An index of
    // this shape is what later makes an access to a[i] a dense vector load
    // rather than a gather.
    Block &body = region[region.entry];
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
    for (BlockId b = 0; b < region.size(); b++) {
        substitute(region[b], idx, ramp_value);
    }

    // A lane's own vector -- a vec3f per lane, say -- cannot be widened as it
    // is, since a gang of them would be a vector of vectors. Split those into
    // one value per component first (see SSA/SplitAggregates.h).
    const SplitResult split = split_aggregates(
        *f, entry,
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks),
        {ramp.get()});

    // A division the gang makes with operands known small enough goes to
    // floating point (see SSA/InvariantDivision.h), before the analysis
    // below classifies what it leaves.
    divide_bounded_by_floats(
        *f, analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                               nullptr, masked_blocks));
    divide_by_uniform_divisors(
        *f,
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks),
        entry);
    expand_bounded_multipliers(
        *f, analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                               nullptr, masked_blocks));

    // Re-run the analysis now that the region is linearized and the index is
    // the ramp: the masks and blends linearization introduced have to be
    // classified too, and the index is no longer a block argument to seed on.
    const Divergence div =
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks);
    internal_assert(div.branches.empty())
        << "Linearization left a divergent branch in " << *div.branches.begin();

    Variants variants;
    specialize_calls(funcs, *f, region, div, masks, /*entry_mask=*/nullptr,
                     conditional_calls, lanes, split.call_shapes, policies,
                     variants);

    widen_region(*f, entry, div, lanes);
    broadcast_call_arguments(funcs, *f, region, lanes);

    // With the body vectorized, the loop is gone: it runs exactly once, so
    // its header falls straight into the body and the body's Yield falls
    // through to what followed the loop.
    loop->terminator.data =
        Terminator::Jump{parfor.body.name, parfor.body.args};

    // The continuation is past the region: what the loop falls through to.
    const BlockMap blocks = make_block_map(f);
    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
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
