#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/CloneFunction.h"
#include "SSA/Defer.h"
#include "SSA/InvariantDivision.h"
#include "SSA/Linearize.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"
#include "SSA/Simplify.h"
#include "SSA/SplitAggregates.h"
#include "SSA/UniformizeLoops.h"

#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
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

// Every parfor nested in the region of `func` entered at `entry` -- the body
// of the loop being vectorized, or a callee specialized for the gang -- made
// the sequential loop it also is (sequentialize, SSA/Rewrite.cpp). A parfor
// promises only that its iterations may run in any order, and one after
// another is an order; in a gang each lane runs the nested loop for itself,
// and the analyses of the region (divergence, the uniformizing of loops the
// lanes leave at different iterations, linearization) want it as the blocks
// a `for` converts to, which they already know. One at a time, since a loop
// nested in a nested loop is in the region too and is met once its outer
// one has been rewritten. A bound loop inside the gang is a schedule error,
// which sequentialize reports.
void sequentialize_nested(Function &func, const string &entry) {
    for (;;) {
        shared_ptr<Block> nested;
        const Cfg region(func, entry);
        for (const auto &block : region.blocks()) {
            if (std::holds_alternative<Terminator::ParFor>(block->terminator.data)) {
                nested = block;
                break;
            }
        }
        if (!nested) {
            return;
        }
        sequentialize(func, nested);
    }
}

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
        return Ptr_t::make(widen(p->etype, lanes), p->readonly);
    }
    if (const Array_t *a = type.as<Array_t>()) {
        return Array_t::make(widen(a->etype, lanes), a->size, a->readonly);
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

// If `index` is the gang's lane indices -- a ramp of unit stride from a
// constant base, `lanes` wide, which is what the loop index became (see
// vectorize()) -- that base; nothing otherwise.
std::optional<uint64_t> lane_ramp_base(const Value &index, uint32_t lanes) {
    const auto *instr = std::get_if<shared_ptr<Instruction>>(&index.data);
    if (instr == nullptr || (*instr)->op != Instruction::Op::Ramp ||
        (*instr)->operands.size() != 2 || index.get_type().lanes() != lanes) {
        return std::nullopt;
    }
    const auto constant = [](const Value &v) -> std::optional<uint64_t> {
        const auto *c = std::get_if<Constant>(&v.data);
        if (c == nullptr) {
            return std::nullopt;
        }
        if (const auto *u = std::get_if<uint64_t>(&c->data)) {
            return *u;
        }
        if (const auto *i = std::get_if<int64_t>(&c->data); i && *i >= 0) {
            return uint64_t(*i);
        }
        return std::nullopt;
    };
    const std::optional<uint64_t> base = constant(*(*instr)->operands[0]);
    const std::optional<uint64_t> stride = constant(*(*instr)->operands[1]);
    if (!base.has_value() || !stride.has_value() || *stride != 1) {
        return std::nullopt;
    }
    return base;
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
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
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
    // two exceptions: `rand`'s argument is how many numbers to draw, which is
    // a count and not one of them; and a `prefetch`'s byte count is a
    // constant of the layout, the same for every lane, so only its address
    // goes per lane (the mask it carries once predicated is a gang's
    // already; see SSA/Linearize.cpp).
    case Instruction::Op::Intrinsic: {
        if (instr.intrinsic == ir::Intrinsic::rand) {
            return {};
        }
        if (instr.intrinsic == ir::Intrinsic::prefetch) {
            return {0};
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
                   [&](Terminator::Spawn &s) {
                       for (auto &a : s.call.args) {
                           replace(a);
                       }
                       for (auto &a : s.cont.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::Defer &d) {
                       replace(d.queue);
                       replace(d.entry);
                       if (d.answer) {
                           replace(d.answer);
                       }
                   },
                   [&](Terminator::Join &j) {
                       for (auto &a : j.next.args) {
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
                       for (auto &k : c.conds) {
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
                       [&](Terminator::Spawn &t) {
                           for (auto &a : t.call.args) {
                               retype(a);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::Defer &t) {
                           retype(t.queue);
                           retype(t.entry);
                           if (t.answer) {
                               retype(t.answer);
                           }
                       },
                       [&](Terminator::Join &t) {
                           for (auto &a : t.next.args) {
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
                           for (auto &k : t.conds) {
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
    // A boolean's extremes: `any` accumulates with the maximum, `all` with
    // the minimum, and a lane that is off contributes false to the one and
    // true to the other.
    if (type.is_bool()) {
        internal_assert(op == Instruction::Op::AccMin ||
                        op == Instruction::Op::AccMax)
            << "No identity for " << op_name(op) << " of a boolean";
        return std::make_shared<Value>(
            Constant{type, op == Instruction::Op::AccMin});
    }
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
               instr.op == Instruction::Op::AccMax ||
               instr.op == Instruction::Op::AccArgmin ||
               instr.op == Instruction::Op::AccArgmax;
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
            // A lane's element of a vector the gang shares, at the lane's own
            // index -- a node's eight children's boxes, each lane testing one
            // (`lo[children]` in a vectorized loop over the children) -- is
            // the vector itself, or a slice of it, in the gang's shape. Of a
            // vector of vectors, held as one vector per component (see
            // CodeGen_LLVM's Vector_t), it is the struct of those components
            // that a per-lane short vector is carried as (ir::widen): the
            // same members, reinterpreted. Of a vector of scalars it is the
            // lanes the ramp names, a shuffle LLVM folds when they are all
            // of them. Not a chain of selects over the elements, which is
            // what a computed index into a short vector becomes (see
            // SSA/SplitAggregates.cpp) and what a read at the gang's own
            // indices never needs.
            if (instr->op == Instruction::Op::ExtractIdx &&
                instr->operands.size() == 2 &&
                !div.is_varying(name, *instr->operands[0]) &&
                instr->operands[0]->get_type().is_vector() &&
                !instr->operands[0]->get_type().as<Vector_t>()->packed) {
                const Vector_t *v =
                    instr->operands[0]->get_type().as<Vector_t>();
                const std::optional<uint64_t> base =
                    lane_ramp_base(*instr->operands[1], lanes);
                if (base.has_value() && *base + lanes <= v->lanes) {
                    if (v->etype.is_vector()) {
                        internal_assert(*base == 0 && v->lanes == lanes)
                            << "[unimplemented] " << instr->name
                            << " reads a slice of a vector of vectors, "
                            << Type(v) << ", from lane " << *base
                            << ": each component would have to be sliced";
                        instr->op = Instruction::Op::Reinterpret;
                        instr->type = widen(v->etype, lanes);
                    } else {
                        instr->op = Instruction::Op::Shuffle;
                        instr->shuffle.clear();
                        for (uint32_t k = 0; k < lanes; k++) {
                            instr->shuffle.push_back(int(*base + k));
                        }
                        instr->type = Vector_t::make(v->etype, lanes);
                    }
                    instr->operands.pop_back();
                    widened.push_back(instr);
                    continue;
                }
            }

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
                const bool readonly = array->get_type().is_readonly();
                auto flat = std::make_shared<Instruction>(
                    func.get_unique_name(),
                    Array_t::make(scalar, Expr(), readonly),
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
                                  ? widen(Ptr_t::make(scalar, readonly), lanes)
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
                const Ptr_t *base_ptr = instr->operands[0]->get_type().as<Ptr_t>();
                const Struct_t *components = base_ptr->etype.as<Struct_t>();
                internal_assert(k < components->fields.size())
                    << instr->name << " addresses component " << k << " of "
                    << instr->operands[0]->get_type();
                instr->op = Instruction::Op::FieldPtr;
                instr->operands[1] =
                    std::make_shared<Value>(Constant{UInt_t::make(32), k});
                instr->type = Ptr_t::make(components->fields[k].type,
                                          base_ptr->readonly);
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
                // The value, and the mask if there is one -- not an address:
                // the record an argmin writes beside its place (`with`) is
                // per-lane memory behind one pointer, as the place is, and a
                // broadcast of it would be a vector of one address.
                for (size_t k = 1; k < instr->operands.size(); k++) {
                    if (instr->operands[k]->get_type().is<Ptr_t>()) {
                        continue;
                    }
                    to_broadcast.push_back(k);
                }
            }
            // A read of a per-lane array at one index -- a node's children's
            // tests and keys, `bool[4]` and `f32[4]` per lane, held as four
            // gang vectors each (see widen()) -- is that element, a gang
            // vector already: the index names the element, the same one for
            // every lane, and is no lane's value to broadcast.
            if (instr->op == Instruction::Op::ExtractIdx &&
                instr->operands.size() == 2 &&
                instr->operands[0]->get_type().is<Array_t>() &&
                is_gang_wide(
                    instr->operands[0]->get_type().as<Array_t>()->etype,
                    lanes) &&
                !is_gang_wide(instr->operands[1]->get_type(), lanes)) {
                to_broadcast.clear();
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
                               instr->op == Instruction::Op::AccMax ||
                               instr->op == Instruction::Op::AccArgmin ||
                               instr->op == Instruction::Op::AccArgmax;
            const bool by_key = instr->op == Instruction::Op::AccArgmin ||
                                instr->op == Instruction::Op::AccArgmax;

            // An accumulate into per-lane memory that holds a short vector --
            // a lane's `vec3f` slot, a struct of one gang vector per component
            // once widened (see widen()) -- is made one accumulate per
            // component: into the component's field of the place, of the
            // component of the value -- or of the value itself, when the
            // program accumulated a scalar into the vector: `tMax *= 1 + 2 *
            // gamma(3)` in stdlib/intersects.bonsai broadcasts over the
            // components implicitly, as the IR's elementwise operations do,
            // and after widening the scalar is one gang vector every
            // component takes. A backend then meets masked accumulates of
            // gang vectors into per-lane memory, which it lowers already,
            // and never one of a struct (the radiance a path's lanes add into
            // their own `vec4f` slots, apps/pbrt). An argmin or argmax
            // carries a pair, not a vector, and is left as it is.
            if (folds && !by_key && per_lane_write) {
                const Ptr_t *place_ptr =
                    instr->operands[0]->get_type().as<Ptr_t>();
                if (place_ptr != nullptr && is_widened_vector(place_ptr->etype)) {
                    const Struct_t *s = place_ptr->etype.as<Struct_t>();
                    const auto &place = instr->operands[0];
                    const auto &value = instr->operands[1];
                    const Type value_type = value->get_type();
                    const bool per_component = is_widened_vector(value_type);
                    internal_assert(
                        per_component
                            ? value_type.as<Struct_t>()->fields.size() ==
                                  s->fields.size()
                            : value_type.is_vector())
                        << "An accumulate into a per-lane " << place_ptr->etype
                        << " of a " << value_type;
                    const auto here = block.shared_from_this();
                    for (uint32_t k = 0; k < uint32_t(s->fields.size()); k++) {
                        auto index = std::make_shared<Value>(
                            Constant{UInt_t::make(32), uint64_t(k)});
                        auto field = std::make_shared<Instruction>(
                            func.get_unique_name(),
                            Ptr_t::make(s->fields[k].type, place_ptr->readonly),
                            Instruction::Op::FieldPtr,
                            vector<shared_ptr<Value>>{place, index}, here);
                        widened.push_back(field);
                        shared_ptr<Value> component = value;
                        if (per_component) {
                            auto read = std::make_shared<Instruction>(
                                func.get_unique_name(), s->fields[k].type,
                                Instruction::Op::LoadField,
                                vector<shared_ptr<Value>>{value, index}, here);
                            widened.push_back(read);
                            component = std::make_shared<Value>(read);
                        }
                        vector<shared_ptr<Value>> operands{
                            std::make_shared<Value>(field), component};
                        if (instr->has_mask()) {
                            operands.push_back(instr->operands[2]);
                        }
                        auto acc = std::make_shared<Instruction>(
                            instr->op, std::move(operands), here);
                        acc->atomic = instr->atomic;
                        widened.push_back(acc);
                    }
                    continue;
                }
            }

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
                // What the lanes' keys fold with, for an argmin or argmax.
                Instruction::Op key_op = instr->op;
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
                case Instruction::Op::AccArgmin:
                    rop = ir::VectorReduce::Min;
                    key_op = Instruction::Op::AccMin;
                    break;
                case Instruction::Op::AccArgmax:
                    rop = ir::VectorReduce::Max;
                    key_op = Instruction::Op::AccMax;
                    break;
                default: // AccAdd, AccSub: both fold the lanes with +
                    rop = ir::VectorReduce::Add;
                    break;
                }
                const shared_ptr<Value> mask =
                    instr->has_mask() ? instr->operands[2] : nullptr;
                const auto here = block.shared_from_this();

                // One gang vector of a component (or of the whole scalar value),
                // masked to the identity where a lane is off, reduced to one.
                // `masked` is the vector as reduced, after the mask.
                shared_ptr<Value> masked;
                const auto fold = [&](shared_ptr<Value> lanes_of) {
                    if (!is_gang_wide(lanes_of->get_type(), lanes)) {
                        lanes_of = broadcast(func, here, lanes_of, lanes, widened);
                    }
                    const Type element = lanes_of->get_type().element_of();
                    if (mask != nullptr) {
                        shared_ptr<Value> identity = broadcast(
                            func, here,
                            accumulate_identity(key_op, element, func, here,
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
                    masked = lanes_of;
                    auto reduce = std::make_shared<Instruction>(
                        func.get_unique_name(), element, Instruction::Op::Reduce,
                        vector<shared_ptr<Value>>{lanes_of}, here);
                    // Booleans fold with or and and: the maximum of bools is
                    // whether any is set, the minimum whether all are.
                    reduce->reduce = !element.is_bool()           ? rop
                                     : rop == ir::VectorReduce::Max ? ir::VectorReduce::Or
                                     : rop == ir::VectorReduce::Min ? ir::VectorReduce::And
                                                                    : rop;
                    widened.push_back(reduce);
                    return std::make_shared<Value>(reduce);
                };

                shared_ptr<Value> &value = instr->operands[1];
                const Type value_type = value->get_type();
                if (by_key) {
                    // An argmin or argmax: the value is a pair of the key and
                    // what travels with it, one of each per lane, widened to
                    // a struct of a gang vector of keys and the widened held
                    // value. The keys fold to their extremum; what travels
                    // with it is taken from the first lane holding that
                    // extremum -- the lane a loop over the elements one by
                    // one would have kept, since the accumulate keeps the
                    // first of equal keys (Embree's select_min likewise) --
                    // field by field, each a gang vector read at that lane.
                    const Struct_t *pair = value_type.as<Struct_t>();
                    internal_assert(pair != nullptr && pair->fields.size() >= 1 &&
                                    is_gang_wide(pair->fields[0].type, lanes))
                        << "The value of " << instr->name << " is a "
                        << value_type << ", not a per-lane pair of a key and "
                        << "what travels with it";
                    const auto field_of = [&](const shared_ptr<Value> &of,
                                              const Struct_t *s, uint32_t k) {
                        auto read = std::make_shared<Instruction>(
                            func.get_unique_name(), s->fields[k].type,
                            Instruction::Op::LoadField,
                            vector<shared_ptr<Value>>{
                                of, std::make_shared<Value>(Constant{
                                        UInt_t::make(32), uint64_t(k)})},
                            here);
                        widened.push_back(read);
                        return std::make_shared<Value>(read);
                    };
                    const shared_ptr<Value> key = fold(field_of(value, pair, 0));
                    // The first lane whose key is the extremum.
                    const Type lane_t = UInt_t::make(32);
                    const Type lanes_t = Vector_t::make(lane_t, lanes);
                    const auto u32 = [&](uint64_t v) {
                        return std::make_shared<Value>(Constant{lane_t, v});
                    };
                    auto ramp = std::make_shared<Instruction>(
                        func.get_unique_name(), lanes_t, Instruction::Op::Ramp,
                        vector<shared_ptr<Value>>{u32(0), u32(1)}, here);
                    widened.push_back(ramp);
                    auto holds = std::make_shared<Instruction>(
                        func.get_unique_name(),
                        Vector_t::make(Bool_t::make(), lanes), Instruction::Op::Eq,
                        vector<shared_ptr<Value>>{
                            masked, broadcast(func, here, key, lanes, widened)},
                        here);
                    widened.push_back(holds);
                    auto candidates = std::make_shared<Instruction>(
                        func.get_unique_name(), lanes_t, Instruction::Op::Select,
                        vector<shared_ptr<Value>>{
                            std::make_shared<Value>(holds),
                            std::make_shared<Value>(ramp),
                            broadcast(func, here, u32(lanes), lanes, widened)},
                        here);
                    widened.push_back(candidates);
                    auto first = std::make_shared<Instruction>(
                        func.get_unique_name(), lane_t, Instruction::Op::Reduce,
                        vector<shared_ptr<Value>>{
                            std::make_shared<Value>(candidates)},
                        here);
                    first->reduce = ir::VectorReduce::Min;
                    widened.push_back(first);
                    const shared_ptr<Value> lane = std::make_shared<Value>(first);
                    // What travels with the key, at that lane: a gang vector's
                    // lane, a widened struct's fields each at that lane.
                    const std::function<shared_ptr<Value>(const shared_ptr<Value> &)>
                        at_lane = [&](const shared_ptr<Value> &of)
                        -> shared_ptr<Value> {
                        const Type &t = of->get_type();
                        if (is_gang_wide(t, lanes) && t.is<Vector_t>()) {
                            auto read = std::make_shared<Instruction>(
                                func.get_unique_name(), t.element_of(),
                                Instruction::Op::ExtractIdx,
                                vector<shared_ptr<Value>>{of, lane}, here);
                            widened.push_back(read);
                            return std::make_shared<Value>(read);
                        }
                        if (const Struct_t *s = t.as<Struct_t>();
                            s != nullptr && is_gang_wide(t, lanes)) {
                            vector<shared_ptr<Value>> parts;
                            for (uint32_t k = 0; k < uint32_t(s->fields.size());
                                 k++) {
                                parts.push_back(at_lane(field_of(of, s, k)));
                            }
                            auto made = std::make_shared<Instruction>(
                                func.get_unique_name(), narrow(t, lanes),
                                Instruction::Op::MakeStruct, std::move(parts),
                                here);
                            widened.push_back(made);
                            return std::make_shared<Value>(made);
                        }
                        return of; // uniform: every lane's is the same
                    };
                    vector<shared_ptr<Value>> parts = {key};
                    for (uint32_t k = 1; k < uint32_t(pair->fields.size()); k++) {
                        parts.push_back(at_lane(field_of(value, pair, k)));
                    }
                    auto rebuilt = std::make_shared<Instruction>(
                        func.get_unique_name(), narrow(value_type, lanes),
                        Instruction::Op::MakeStruct, std::move(parts), here);
                    widened.push_back(rebuilt);
                    value = std::make_shared<Value>(rebuilt);
                    // The values written alongside the pair: one per lane
                    // like the pair's own held fields, each taken from the
                    // same winning lane; their places are shared memory.
                    for (size_t i = instr->alongside_begin();
                         i + 1 < instr->operands.size(); i += 2) {
                        internal_assert(
                            !is_gang_wide(instr->operands[i]->get_type(), lanes))
                            << "[unimplemented] " << instr->name
                            << " writes a per-lane place alongside its pair";
                        instr->operands[i + 1] = at_lane(instr->operands[i + 1]);
                    }
                } else if (is_widened_vector(value_type)) {
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
// What `linearize` asks of a callee by name: whether it may run on behalf of
// no lane (safe_with_all_lanes_off, SSA/Linearize.h), looked up in `funcs`,
// where the masked variants this pass makes live beside the functions they
// came from.
std::function<bool(const string &)> safe_callee_in(const FuncMap &funcs) {
    return [&funcs](const string &name) {
        const auto found = funcs.find(name);
        if (found == funcs.end()) {
            return false;
        }
        return safe_with_all_lanes_off(
            *found->second, [&funcs](const string &callee) -> const Function * {
                const auto f = funcs.find(callee);
                return f == funcs.end() ? nullptr : f->second.get();
            });
    };
}

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
    // A parameter the call hands a constant to, where the callee's control
    // depends on it (parameter_steers_control): the constant's token
    // (constant_token), "" for every other parameter. Part of the key, so
    // that a callee called with a flag true and with it false gets a clone
    // for each, the constant folded into the clone before it is vectorized
    // (specialize): what `render[VolPath].specialize(have_media)` on the
    // caller means for the functions the loop calls, which take the flag
    // as a parameter and branch on it. The values ride along for the clone
    // to substitute; they are not compared, the tokens are.
    vector<string> constants;
    vector<shared_ptr<Value>> constant_values;
    // For a callee with no body here (a foreign function, `extern func`):
    // the call's own types -- the arguments' as the scalars they are before
    // widening, the result's from the continuation, undefined when the
    // call drops it -- which are all there is to build its per-lane wrapper
    // from (specialize_foreign). Not compared.
    vector<Type> foreign_params;
    Type foreign_result;

    bool operator<(const VariantKey &o) const {
        return std::tie(callee, shapes, masked, lanes, constants) <
               std::tie(o.callee, o.shapes, o.masked, o.lanes, o.constants);
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
        if (i < key.constants.size() && !key.constants[i].empty()) {
            name += "_c" + std::to_string(i) + key.constants[i];
        }
    }
    return key.masked ? name + "$masked" : name;
}

// A constant a call hands a callee, as the clone's key and name carry it: a
// bool or an integer (a float steers no branch a clone would fold), its
// value and nothing a symbol table objects to. The parameter's type is the
// callee's, so the value alone tells two constants apart.
std::optional<string> constant_token(const Value &v) {
    const auto *c = std::get_if<Constant>(&v.data);
    if (c == nullptr ||
        !(c->type.is<Bool_t>() || c->type.is<Int_t>() || c->type.is<UInt_t>())) {
        return std::nullopt;
    }
    return std::visit(
        overloads{[](bool b) -> std::optional<string> {
                      return string(b ? "t" : "f");
                  },
                  [](int64_t i) -> std::optional<string> {
                      return i < 0 ? "m" + std::to_string(-i) : std::to_string(i);
                  },
                  [](uint64_t u) -> std::optional<string> {
                      return std::to_string(u);
                  },
                  [](const auto &) -> std::optional<string> {
                      return std::nullopt;
                  }},
        c->data);
}

// The one constant every value of `values` is, or none: a run of calls
// (Terminator::MultiCall) hands a parameter several values, and the clone
// folds the constant only when they agree.
std::optional<string> common_constant_token(const vector<const Value *> &values) {
    std::optional<string> token;
    for (const Value *v : values) {
        const auto t = v != nullptr ? constant_token(*v) : std::nullopt;
        if (!t.has_value() || (token.has_value() && *token != *t)) {
            return std::nullopt;
        }
        token = t;
    }
    return token;
}

// Whether the callee's control depends on its `i`th parameter: the
// parameter, under its name in whichever block it was threaded to, is a
// dispatch's condition or an operand of what computes one, within a few
// steps -- or is handed on, as it is, to a call whose callee's control
// depends on it, by this same rule: pbrt's `haveMedia` flag reaches the
// path step's branch on it through `li_vol_path`, which only passes it
// along, and a clone that stopped there would hand the step a parameter
// where the caller had a constant, and the step's media arms would stay.
// What makes a constant argument worth a clone of its own
// (specialize_calls): a flag or a tag the callee branches on folds its
// branches, and the arms no caller of this clone can reach go with them,
// before the linearizer would have masked them and the gang widened them;
// a literal the callee only computes with would multiply the clones for
// nothing a branch reads. LLVM's function specialization weighs a constant
// argument the same way, by the branches and switches it decides. `seen`
// holds the (function, parameter) pairs under consideration, so that a
// recursion -- the step handing its flag to its own next bounce -- is
// decided by the branches it makes and not asked about itself.
bool parameter_steers_control(const FuncMap &funcs, const Function &callee,
                              size_t i,
                              set<std::pair<const Function *, size_t>> &seen) {
    if (callee.blocks.empty() || i >= callee.blocks.front()->args.size()) {
        return false;
    }
    if (!seen.insert({&callee, i}).second) {
        return false;
    }
    const string &param = callee.blocks.front()->args[i].name;
    const std::function<bool(const Value &, int)> reaches =
        [&](const Value &v, int depth) -> bool {
        if (const auto *a = std::get_if<Argument>(&v.data)) {
            return a->name == param;
        }
        const auto *instr = std::get_if<shared_ptr<Instruction>>(&v.data);
        if (instr == nullptr || depth == 0) {
            return false;
        }
        for (const auto &operand : (*instr)->operands) {
            if (operand && reaches(*operand, depth - 1)) {
                return true;
            }
        }
        return false;
    };
    for (const auto &block : callee.blocks) {
        const auto *d = std::get_if<Terminator::Dispatch>(&block->terminator.data);
        if (d != nullptr && d->cond && reaches(*d->cond, 4)) {
            return true;
        }
    }
    // Forwarded: the parameter itself (depth 0: not a value computed from
    // it, which is no longer the constant) as an argument of a call, to a
    // callee whose control depends on that argument.
    for (const auto &block : callee.blocks) {
        const Terminator::Jump *call = block->terminator.callee();
        if (call == nullptr) {
            continue;
        }
        const auto target = funcs.find(call->name);
        if (target == funcs.end() || !target->second) {
            continue;
        }
        for (size_t j = 0; j < call->args.size(); j++) {
            if (call->args[j] && reaches(*call->args[j], 0) &&
                parameter_steers_control(funcs, *target->second, j, seen)) {
                return true;
            }
        }
    }
    return false;
}

bool parameter_steers_control(const FuncMap &funcs, const Function &callee,
                              size_t i) {
    set<std::pair<const Function *, size_t>> seen;
    return parameter_steers_control(funcs, callee, i, seen);
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
    // The continuation is reached two ways now: from the guard directly when
    // no lane is on, and from the call. Its own list of predecessors says so,
    // as the graph does; a list that still named only the block the call
    // used to be in made a rebuilt one look like a second edge had appeared.
    region[region.id(cont->name)].preds.push_back(guarded);
}

// A run whose calls each have a condition (Terminator::MultiCall::conds: a
// node that holds its children's boxes, each child's call under the
// parent's test of its box) is made by a gang the way Embree's chunk
// traversal makes it (bvh_intersector_hybrid.cpp, the packet traversal of
// a wide node): one node for the gang, every child whose box any lane
// meets visited, by the lanes that meet it. So each call's condition
// becomes the lanes that make it -- the block's mask and the child's test,
// per lane -- which specialize_calls hands the callee as that call's
// `!mask` argument and tests for a lane on. Made here, before the
// divergence analysis the widening reads, so that the masks are seen per
// lane; a run no lane's mask narrows (the region's own, with no mask of
// its own) keeps the tests as they are.
void mask_conditioned_runs(Function &func, const string &entry,
                           const BlockMasks &masks,
                           const shared_ptr<Value> &entry_mask) {
    const Cfg region(func, entry);
    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
        auto *run =
            std::get_if<Terminator::MultiCall>(&block->terminator.data);
        if (run == nullptr || run->conds.empty()) {
            continue;
        }
        const auto found = masks.find(block->name);
        const shared_ptr<Value> on =
            found != masks.end() ? found->second : entry_mask;
        if (on == nullptr) {
            continue;
        }
        for (shared_ptr<Value> &cond : run->conds) {
            auto both = std::make_shared<Instruction>(
                func.get_unique_name(), Bool_t::make(), Instruction::Op::LAnd,
                vector<shared_ptr<Value>>{on, cond}, block);
            block->instrs.push_back(both);
            cond = std::make_shared<Value>(both);
        }
    }
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
        auto *run = std::get_if<Terminator::MultiCall>(&block->terminator.data);
        // A run whose calls each have a condition (Terminator::MultiCall::
        // conds, a node whose children's boxes it tests) makes each call
        // under its own mask -- the lanes on in the block for which that
        // child's test holds (mask_conditioned_runs) -- handed to the
        // callee as the call's `!mask` argument below; the condition the
        // run keeps is whether that mask has a lane on.
        const bool conditioned = run != nullptr && !run->conds.empty();
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
        key.masked = conditional || recurses || conditioned;

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

        key.constants.assign(key.shapes.size(), "");
        key.constant_values.assign(key.shapes.size(), nullptr);
        if (callee != funcs.end()) {
            // A parameter the call hands a constant to, where the callee's
            // control depends on it, is folded into the clone (specialize):
            // the constant a `specialize` on the caller made of its flag
            // reaches the branches the callee makes on it. A per-lane short
            // vector split into components is never a constant.
            size_t position = 0;
            for (size_t i = 0; i < components.size(); i++) {
                if (components[i] == 1 && key.shapes[i] == Shape::Uniform) {
                    const auto token = common_constant_token(values_at(position));
                    if (token.has_value() &&
                        parameter_steers_control(funcs, *callee->second, i)) {
                        key.constants[i] = *token;
                        key.constant_values[i] =
                            std::make_shared<Value>(*values_at(position).front());
                    }
                }
                position += components[i];
            }
        } else {
            // A callee with no body here is a foreign function: its variant
            // is built from the call's own types (specialize_foreign), the
            // arguments' as they are before the region is widened and the
            // result's from the continuation that takes it.
            const auto *c = std::get_if<Terminator::Call>(&block->terminator.data);
            internal_assert(c != nullptr)
                << "A run of calls to the foreign function " << call->name
                << " in " << name << ": a foreign function is called one "
                << "call at a time (docs/foreign-functions.md)";
            // A value the gang already holds one per lane of -- the loop's
            // index, a ramp -- has the gang's type here; the wrapper wants
            // the scalar the function takes, and widens it itself.
            const auto scalar_of = [&](const Type &t) {
                return is_gang_wide(t, lanes) ? narrow(t, lanes) : t;
            };
            for (const auto &a : call->args) {
                key.foreign_params.push_back(scalar_of(a->get_type()));
            }
            if (!c->drop) {
                const Block &target = region[region.id(c->cont.name)];
                internal_assert(!target.args.empty())
                    << "Call continuation " << target.name
                    << " takes no result argument";
                key.foreign_result = scalar_of(target.args[0].type);
            }
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

        if (conditioned) {
            // Each call of the run under the lanes that make it: the mask
            // is one more value the calls differ in, at the `!mask`
            // parameter the variant takes last, and the call's condition
            // is whether a lane is on in it -- a child no lane's box test
            // let through is not visited, by anyone. Behind that test no
            // guard is needed (compare guard_call).
            const auto mask = masks.find(name);
            call->args.push_back(
                mask != masks.end()  ? mask->second
                : entry_mask != nullptr
                    ? entry_mask
                    : std::make_shared<Value>(Constant{Bool_t::make(), true}));
            run->varying_at.push_back(call->args.size() - 1);
            for (size_t k = 0; k < run->conds.size(); k++) {
                const shared_ptr<Value> lanes = run->conds[k];
                run->varying[k].push_back(lanes);
                auto any = std::make_shared<Instruction>(
                    func.get_unique_name(), Bool_t::make(),
                    Instruction::Op::Any, vector<shared_ptr<Value>>{lanes},
                    block);
                block->instrs.push_back(any);
                run->conds[k] = std::make_shared<Value>(any);
            }
        } else if (conditional) {
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
            // ispc's for a call it cannot see into (a call is never
            // SafeToRunWithMaskAllOff there). The arm the call is in is
            // normally behind the linearizer's own test of the mask, which
            // settles it; a call nothing has tested for gets a test of its
            // own (see guard_call) -- unless the variant is safe with every
            // lane off (safe_with_all_lanes_off, SSA/Linearize.h: no
            // memory, no effect, nothing uniform that a lane's value
            // indexes), when running it for no lane computes nothing
            // anyone reads and the test would cost more than it spares:
            // the right side of `a && b` where `b` is a pure function.
            if (!nonempty(b, *mask->second) &&
                !safe_callee_in(funcs)(name_of_variant)) {
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
// The variant of a callee with no body here: a foreign function (`extern
// func`, docs/foreign-functions.md), one scalar call with the C calling
// convention, which a gang cannot hand a vector of arguments or a mask.
// The variant is a wrapper that makes the call once per lane that is on,
// in lane order -- ispc's `foreach_active` written out by the compiler --
// each lane's arguments picked out of the gang's vectors, each result put
// into its lane of the gang's, the lanes that are off skipped: a lane no
// thread reached holds arguments nothing computed, and the library would
// be handed them. Built in the gang's shape already -- every parameter a
// vector where the call's argument varies, the mask a vector of bools,
// the result a vector -- so it goes through none of the widening a bonsai
// callee's clone does, and the call site treats it as any variant: the
// mask appended, the call guarded by a test of the mask where nothing has
// tested it (specialize_calls). The values are threaded through the chain
// as block arguments, as every value is in this form.
shared_ptr<Function> specialize_foreign(FuncMap &funcs, const VariantKey &key,
                                        const string &name) {
    internal_assert(key.foreign_params.size() == key.shapes.size())
        << "The foreign call to " << key.callee << " passes "
        << key.shapes.size() << " arguments and has " << key.foreign_params.size()
        << " types for them";
    for (size_t i = 0; i < key.shapes.size(); i++) {
        internal_assert(key.shapes[i] != Shape::Pointee)
            << "Cannot vectorize the call to the foreign function " << key.callee
            << ": its argument " << i
            << " points into the gang's per-lane memory, and a foreign "
               "function takes an array as one address for every lane "
               "(docs/foreign-functions.md). Hand each lane its value instead.";
    }
    const uint32_t lanes = key.lanes;
    const bool returns = key.foreign_result.defined();
    // A foreign value is one pointer-sized handle the program only passes on
    // (ir::Foreign_t), never one per lane: the wrapper would have to build a
    // gang of them for the result, and no backend has a vector of handles.
    internal_assert(!returns || !key.foreign_result.is<Foreign_t>())
        << "Cannot vectorize the call to the foreign function " << key.callee
        << ": it returns a foreign type, " << key.foreign_result
        << ", and is called by the gang with a lane's arguments or under a "
           "mask, so each lane would hold a foreign value of its own, which "
           "a gang cannot (docs/foreign-functions.md). Call it on uniform "
           "arguments outside any divergent branch, or keep the loop that "
           "reaches it scalar.";
    const Type acc_type = returns ? widen(key.foreign_result, lanes) : Type();

    auto wrapper = std::make_shared<Function>();
    wrapper->ret_type = returns ? acc_type : Void_t::make();
    wrapper->attributes.push_back(ir::Function::Attribute::vectorized);
    wrapper->specialized_from = key.callee;

    // The values every block of the chain carries: the parameters, the mask
    // and the result so far, by these names.
    struct Carried {
        Type type;
        string name;
    };
    vector<Carried> carried;
    for (size_t i = 0; i < key.shapes.size(); i++) {
        carried.push_back({key.shapes[i] == Shape::Varying
                               ? widen(key.foreign_params[i], lanes)
                               : key.foreign_params[i],
                           "a" + std::to_string(i)});
    }
    const size_t mask_at = carried.size();
    if (key.masked) {
        carried.push_back({widen(Bool_t::make(), lanes), "!mask"});
    }
    const size_t acc_at = carried.size();
    if (returns) {
        carried.push_back({acc_type, "acc"});
    }

    const auto new_block = [&](const string &suffix) {
        auto b = std::make_shared<Block>();
        b->name = suffix.empty() ? name : name + "!" + suffix;
        b->owner = wrapper;
        wrapper->blocks.push_back(b);
        return b;
    };
    // Declares the carried values as `b`'s arguments (the entry declares
    // the parameters and the mask, and starts the result undefined) and
    // returns them in order.
    const auto declare = [&](const shared_ptr<Block> &b, bool entry) {
        vector<shared_ptr<Value>> values;
        for (size_t i = 0; i < carried.size(); i++) {
            if (entry && i == acc_at) {
                values.push_back(undef_value(carried[i].type));
            } else {
                values.push_back(
                    b->add_argument(Argument{carried[i].type, carried[i].name}));
            }
        }
        return values;
    };
    const auto u32 = [](uint64_t k) {
        return std::make_shared<Value>(Constant{UInt_t::make(32), k});
    };

    const auto entry = new_block("");
    vector<shared_ptr<Block>> tests(lanes), calls(lanes), inserts(lanes);
    for (uint32_t lane = 0; lane < lanes; lane++) {
        tests[lane] = new_block("lane" + std::to_string(lane));
        calls[lane] = new_block("call" + std::to_string(lane));
        if (returns) {
            inserts[lane] = new_block("ins" + std::to_string(lane));
        }
    }
    const auto done = new_block("done");

    entry->terminator.data =
        Terminator::Jump{tests[0]->name, declare(entry, /*entry=*/true)};
    for (uint32_t lane = 0; lane < lanes; lane++) {
        const string next = lane + 1 < lanes ? tests[lane + 1]->name : done->name;

        // Is this lane on? If not, past its call with everything as it was.
        {
            const auto &test = tests[lane];
            vector<shared_ptr<Value>> values = declare(test, false);
            if (key.masked) {
                auto on = std::make_shared<Instruction>(
                    wrapper->get_unique_name(), Bool_t::make(),
                    Instruction::Op::ExtractIdx,
                    vector<shared_ptr<Value>>{values[mask_at], u32(lane)}, test);
                test->instrs.push_back(on);
                // targets[0] is where a false condition goes: past the call.
                test->terminator.data = Terminator::Dispatch{
                    std::make_shared<Value>(on),
                    {Terminator::Jump{next, values},
                     Terminator::Jump{calls[lane]->name, values}}};
            } else {
                test->terminator.data =
                    Terminator::Jump{calls[lane]->name, values};
            }
        }
        // The call, on this lane's arguments.
        {
            const auto &call = calls[lane];
            vector<shared_ptr<Value>> values = declare(call, false);
            vector<shared_ptr<Value>> args;
            for (size_t i = 0; i < key.shapes.size(); i++) {
                if (key.shapes[i] != Shape::Varying) {
                    args.push_back(values[i]);
                    continue;
                }
                auto x = std::make_shared<Instruction>(
                    wrapper->get_unique_name(), key.foreign_params[i],
                    Instruction::Op::ExtractIdx,
                    vector<shared_ptr<Value>>{values[i], u32(lane)}, call);
                call->instrs.push_back(x);
                args.push_back(std::make_shared<Value>(x));
            }
            call->terminator.data = Terminator::Call{
                Terminator::Jump{key.callee, std::move(args)},
                Terminator::Jump{returns ? inserts[lane]->name : next, values},
                /*drop=*/!returns};
        }
        // The result into its lane of the gang's: the continuation takes it
        // first, then everything carried.
        if (returns) {
            const auto &ins = inserts[lane];
            const shared_ptr<Value> r =
                ins->add_argument(Argument{key.foreign_result, "r"});
            vector<shared_ptr<Value>> values = declare(ins, false);
            auto bc = std::make_shared<Instruction>(
                wrapper->get_unique_name(), acc_type, Instruction::Op::Bc,
                vector<shared_ptr<Value>>{r, u32(lanes)}, ins);
            ins->instrs.push_back(bc);
            auto put = std::make_shared<Instruction>(
                wrapper->get_unique_name(), acc_type, Instruction::Op::Shuffle,
                vector<shared_ptr<Value>>{values[acc_at],
                                          std::make_shared<Value>(bc)},
                ins);
            for (uint32_t j = 0; j < lanes; j++) {
                put->shuffle.push_back(j == lane ? int(lanes) : int(j));
            }
            ins->instrs.push_back(put);
            values[acc_at] = std::make_shared<Value>(put);
            ins->terminator.data = Terminator::Jump{next, values};
        }
    }
    {
        vector<shared_ptr<Value>> values = declare(done, false);
        done->terminator.data =
            Terminator::Return{returns ? values[acc_at] : nullptr};
    }

    funcs[name] = wrapper;
    return wrapper;
}

shared_ptr<Function> specialize(FuncMap &funcs, const VariantKey &key,
                                const string &name,
                                const ir::BranchPolicyMap &policies,
                                Variants &variants) {
    const auto original = funcs.find(key.callee);
    if (original == funcs.end()) {
        return specialize_foreign(funcs, key, name);
    }

    auto variant = clone_function(*original->second);
    variant->specialized_from = key.callee;
    // A deferral in gang code is the predicated push and a yield: the lanes
    // that push leave one by one, as the linearizer has every lane leave
    // (Terminator::Defer; SSA/Defer.cpp split_defer_terminators).
    split_defer_terminators(*variant);
    // Only what the entry reaches: a block an earlier rewrite disconnected --
    // the continuation of a call loopify() turned into a jump -- is not part
    // of the region the passes below work on, and yet still passes arguments
    // to blocks that are.
    remove_unreachable_blocks(*variant);
    Block &entry_block = *variant->blocks.front();

    // The constants the call hands this clone (VariantKey::constant_values):
    // every use of such a parameter, in every block it was threaded to under
    // its name, becomes the constant, and the simplifier folds what the
    // constant decides -- the branch on a specialized flag, with the arm no
    // caller of this clone can take unreachable and gone -- before the
    // linearizer would have masked that arm and the gang widened it. The
    // parameter itself stays, the caller still passes it, and reads nothing.
    bool folded = false;
    for (size_t i = 0;
         i < key.constant_values.size() && i < entry_block.args.size(); i++) {
        const shared_ptr<Value> &constant = key.constant_values[i];
        if (constant == nullptr) {
            continue;
        }
        const string param = entry_block.args[i].name;
        for_each_value(*variant, [&](shared_ptr<Value> &v) {
            const auto *a = v ? std::get_if<Argument>(&v->data) : nullptr;
            if (a != nullptr && a->name == param) {
                v = constant;
            }
        });
        folded = true;
    }
    if (folded) {
        simplify(*variant);
        remove_unreachable_blocks(*variant);
    }

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
                       [&](Terminator::Spawn &s) {
                           if (s.cont.name == old_entry) {
                               s.cont.name = name;
                           }
                       },
                       [&](Terminator::Defer &) {},
                       [&](Terminator::Join &j) {
                           if (j.next.name == old_entry) {
                               j.next.name = name;
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
    // A parfor in the callee -- a tree query's loop over a leaf's elements
    // or a node's children that the schedule left as a loop -- is, in a
    // gang, a loop each lane runs for itself (sequentialize_nested).
    sequentialize_nested(*variant, entry);

    // Linearization folds a region down to a single path, so the function
    // needs a single exit for that path to end at.
    unify_returns(*variant);

    // The analysis of this region, as its seeds accumulate: the varying
    // parameters throughout, what uniformizing its loops adds, and, once
    // the branches are folded, the arms that run under a mask the graph no
    // longer shows (`folded_arms`, set after linearization).
    set<string> folded_arms;
    const auto analyze = [&](const set<std::pair<string, string>> &varying_args,
                             const set<string> &masked_blocks) {
        return analyze_divergence(*variant, entry, varying_names, {},
                                  varying_args, pointee_names, mask,
                                  masked_blocks, folded_arms);
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
    // The blocks the linearizer folds -- the arms of the divergent branches
    // -- run under a mask afterwards as they did before, only now the mask is
    // a value rather than a branch, and the analyses below cannot see it in
    // the graph. They are told: seeded as masked, so that a store or a call
    // in an arm still makes the memory it writes per lane. (Found through a
    // local passed by address to a call inside an arm: with the arm no
    // longer masked in the analysis's eyes, the call was a store into shared
    // memory that happens if any lane is on, and every lane got the arm's
    // writes.) Not through `masked_blocks`, which says "a folded loop" too,
    // and would make every latch inside an arm a divergent one.
    folded_arms = linearizable.masked;
    BlockMasks masks = linearize(*variant, entry, linearizable, mask,
                                 uniform.loops, policies, key.callee,
                                 safe_callee_in(funcs));
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- after linearizing " << name << ":\n";
        variant->dump(std::cerr);
    }

    // With the masks known, a run whose calls each have a condition makes
    // each under the lanes that meet it (mask_conditioned_runs), and a
    // run's votes on its order become counts of the lanes that are on (see
    // lower_votes). Both before the analysis below, which has to see the
    // masks and the counts.
    mask_conditioned_runs(*variant, entry, masks, mask);
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
    // vector if the returned value turned out to be varying -- and is made
    // one where the caller takes it so. A call on a lane's arguments may
    // return anything, the caller's analysis says (call_effects in
    // SSA/AnalyzeDivergence.cpp), and its continuation takes the result
    // widened; the key records those arguments as the shapes that are not
    // uniform. A variant made for such a call whose return came out uniform
    // all the same -- material_bxdf's copy for the mixed material returns a
    // constant whatever it is handed -- hands back the widened value, the
    // uniform one broadcast before the return, so that what the callee
    // returns and what the continuation takes agree by construction (the
    // LLVM backend met an i8 where the continuation's phi wanted <16 x i8>).
    const bool varying_result =
        std::any_of(key.shapes.begin(), key.shapes.end(),
                    [](Shape s) { return s != Shape::Uniform; });
    for (const auto &block : variant->blocks) {
        auto *ret = std::get_if<Terminator::Return>(&block->terminator.data);
        if (ret == nullptr || !ret->value) {
            continue;
        }
        if (varying_result && !is_gang_wide(ret->value->get_type(), key.lanes)) {
            vector<shared_ptr<Instruction>> sink;
            ret->value = broadcast(*variant, block, ret->value, key.lanes, sink);
            block->instrs.insert(block->instrs.end(), sink.begin(), sink.end());
        }
        variant->ret_type = ret->value->get_type();
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

// A block that ends in a jump to a block nothing else reaches is one block
// with it: the target's arguments are what the jump passes, its
// instructions follow the block's, and its terminator becomes the block's.
// Repeated down the chain through the block named `last`. What an inlined
// straight-line callee leaves -- the site jumping into the copy, the
// copy's blocks jumping on to one another and then to the continuation
// with the callee's value -- becomes the site block itself, with that
// value an instruction of it rather than an argument of the continuation,
// which is what a rule that matches values can see (see
// inline_pure_variants). A block of the copy may name a value of an
// earlier block of it without taking it as an argument (a gang's blocks
// read their function's values by name), so what each absorbed block's
// arguments stood for is kept and applied down the whole chain.
void absorb_successors(Function &func, const shared_ptr<Block> &block,
                       const string &last) {
    map<string, shared_ptr<Value>> renames;
    for (;;) {
        const auto *jump = std::get_if<Terminator::Jump>(&block->terminator.data);
        if (jump == nullptr) {
            return;
        }
        shared_ptr<Block> next;
        for (const shared_ptr<Block> &b : func.blocks) {
            if (b->name == jump->name) {
                next = b;
                break;
            }
        }
        if (!next || next == func.blocks.front() || next == block ||
            jump->args.size() != next->args.size()) {
            return;
        }
        size_t ways_in = 0;
        for (const shared_ptr<Block> &b : func.blocks) {
            for (const Terminator::Jump *j : jumps_of(*b)) {
                if (j->name == next->name) {
                    ways_in++;
                }
            }
        }
        if (ways_in != 1) {
            return;
        }
        for (size_t i = 0; i < next->args.size(); i++) {
            renames[next->args[i].name] = jump->args[i];
        }
        for_each_value(*next, [&](shared_ptr<Value> &v) {
            if (const auto *a = std::get_if<Argument>(&v->data)) {
                if (const auto it = renames.find(a->name); it != renames.end()) {
                    v = it->second;
                }
            }
        });
        for (const shared_ptr<Instruction> &instr : next->instrs) {
            instr->owner = block;
            block->instrs.push_back(instr);
        }
        // A name both tables hold denotes one value: a function's names are
        // one per value, and what an inlined copy brings is its own (see
        // inline_call). Two instructions under one name would be the
        // callee's table leaking into the caller's, which the simplifier
        // would read as the one for the other.
        for (const auto &[name, v] : next->lookups) {
            if (renames.count(name)) {
                continue;
            }
            if (const auto it = block->lookups.find(name); it != block->lookups.end()) {
                const auto *mine = std::get_if<shared_ptr<Instruction>>(&it->second->data);
                const auto *theirs = std::get_if<shared_ptr<Instruction>>(&v->data);
                internal_assert(mine == nullptr || theirs == nullptr ||
                                mine->get() == theirs->get())
                    << "absorbing " << next->name << " into " << block->name
                    << ": `" << name << "` names " << (*mine)->name << " in one and "
                    << (*theirs)->name << " in the other";
            }
            block->lookups[name] = v;
        }
        // The merged block stands for both: a provenance the target carried
        // is kept where the block had none.
        if (!block->provenance.defined()) {
            block->provenance = next->provenance;
        }
        block->terminator = std::move(next->terminator);
        const bool done = next->name == last;
        std::erase(func.blocks, next);
        if (done) {
            return;
        }
    }
}

// The variants this gang made that are pure and straight-line -- no memory
// of anyone's, no effect, one block ending in a return, as a masked
// `distmin` of a box is once its own branches are selects -- are copied in
// at each call of them in `f` (inline_call), and each copy folded into the
// block that called it (absorb_successors). What the callee computed is
// then a value of the caller's, which the simplifier's rules can see: the
// slab test the masked `distmin` repeats is the one the node's test
// computed, so the two become one, and the prune against the best reads
// the test's own distance. Left as a call, the callee stays opaque until
// LLVM inlines it, after our rules have run. Such a variant is never
// guarded by a test of its mask (specialize_calls: safe with every lane
// off), so running its instructions where the call was runs no more than
// the call did; and a variant nothing calls afterwards is removed.
void inline_pure_variants(FuncMap &funcs, const shared_ptr<Function> &f,
                          const Variants &variants) {
    // Past this a straight-line callee is copied at each call anyway, by
    // LLVM, but copying it here at every site would make the SSA the
    // rewrites after this walk that much larger for no rule's gain.
    constexpr size_t kMostInstructions = 512;
    const auto safe = safe_callee_in(funcs);
    set<string> inlined;
    // Straight-line: every block ends in a plain jump or the return, so the
    // blocks are one chain (the linearizer's merges, before the simplifier
    // has folded them into one block), with nothing to branch on.
    const auto straight_line = [&](const Function &g) {
        size_t instructions = 0;
        for (const shared_ptr<Block> &block : g.blocks) {
            if (!std::holds_alternative<Terminator::Jump>(block->terminator.data) &&
                !std::holds_alternative<Terminator::Return>(block->terminator.data)) {
                return false;
            }
            instructions += block->instrs.size();
        }
        return instructions <= kMostInstructions;
    };
    for (const auto &[key, name] : variants) {
        const auto g = funcs.find(name);
        // A `[[noinline]]` function stays a call, as the schedule asked (the
        // variant carries the attribute of the function it was made from).
        if (g == funcs.end() ||
            std::find(g->second->attributes.begin(), g->second->attributes.end(),
                      ir::Function::Attribute::noinline) != g->second->attributes.end() ||
            !straight_line(*g->second) || !safe(name)) {
            continue;
        }
        bool any = false;
        // inline_call appends the copy's blocks; the index walk reaches them,
        // and finds no call of this variant in them.
        for (size_t i = 0; i < f->blocks.size(); i++) {
            const shared_ptr<Block> block = f->blocks[i];
            const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
            if (call == nullptr || call->call.name != name) {
                continue;
            }
            const string continuation = call->cont.name;
            inline_call(funcs, f, block);
            absorb_successors(*f, block, continuation);
            any = true;
        }
        if (any) {
            inlined.insert(name);
        }
    }
    if (inlined.empty()) {
        return;
    }
    refresh_preds(*f);
    for (const string &name : inlined) {
        bool called = false;
        for (const auto &[other, fn] : funcs) {
            for (const shared_ptr<Block> &block : fn->blocks) {
                if (const auto *callee = block->terminator.callee();
                    callee != nullptr && callee->name == name) {
                    called = true;
                }
            }
        }
        if (!called) {
            funcs.erase(name);
        }
    }
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

    // A mutable local written by one lane is a value that differs between
    // lanes, not memory traffic: promote before anything else looks at the
    // body (see SSA/PromoteAllocas.h).
    promote_allocas(*f, f->blocks[0]->name);

    // The loop is read after the promotion, not before it. A `mut` local of
    // the enclosing loop that the body reads reaches it as an address
    // threaded through the body's arguments, and promoting it deletes that
    // argument from the body and from the loop's jump to it alike
    // (PromoteAllocas.cpp, erase_threading). This copy is what the loop's
    // entry is rebuilt from once the body is a gang, below; taken before the
    // promotion, it handed the body one argument more than the body took.
    const Terminator::ParFor parfor =
        std::get<Terminator::ParFor>(loop->terminator.data);
    const uint32_t lanes = gang_width(parfor);

    const string entry = parfor.body.name;

    // A parfor nested in the body is, in a gang, a loop each lane runs for
    // itself; made the sequential loop it also is before the analyses look
    // at the region (sequentialize_nested).
    sequentialize_nested(*f, entry);

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
    // The folded arms stay masked for the analyses after the fold; see the
    // same step in the variant path above.
    const set<string> folded_arms = before.masked;
    BlockMasks masks = linearize(*f, entry, before, nullptr, uniform.loops,
                                 policies, func, safe_callee_in(funcs));
    if (std::getenv("BONSAI_DUMP_LINEARIZE") != nullptr) {
        std::cerr << "--- after linearizing " << func << ":\n";
        f->dump(std::cerr);
    }
    mask_conditioned_runs(*f, entry, masks, nullptr);
    lower_votes(*f, entry,
                analyze_divergence(*f, entry, {idx}, {}, varying_args, {},
                                   nullptr, masked_blocks, folded_arms),
                masks, nullptr, lanes);

    Cfg region(*f, entry);

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
                           nullptr, masked_blocks, folded_arms),
        {ramp.get()});

    // A division the gang makes with operands known small enough goes to
    // floating point (see SSA/InvariantDivision.h), before the analysis
    // below classifies what it leaves.
    divide_bounded_by_floats(
        *f, analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                               nullptr, masked_blocks, folded_arms));
    divide_by_uniform_divisors(
        *f,
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks, folded_arms),
        entry);
    expand_bounded_multipliers(
        *f, analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                               nullptr, masked_blocks, folded_arms));

    // Re-run the analysis now that the region is linearized and the index is
    // the ramp: the masks and blends linearization introduced have to be
    // classified too, and the index is no longer a block argument to seed on.
    const Divergence div =
        analyze_divergence(*f, entry, {}, {ramp.get()}, varying_args, {},
                           nullptr, masked_blocks, folded_arms);
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

    // The continuation is past the region: what the loop falls through to,
    // handed what the loop's header handed it. Those are values live across
    // the loop -- the query's accumulator, the tree, the ray -- defined
    // before the header and so in scope at the body's end as at the header,
    // and uniform across the gang, since nothing in the body made them.
    const BlockMap blocks = make_block_map(f);
    for (BlockId b = 0; b < region.size(); b++) {
        const shared_ptr<Block> &block = region.block(b);
        if (!std::holds_alternative<Terminator::Yield>(
                block->terminator.data)) {
            continue;
        }
        block->terminator.data =
            Terminator::Jump{parfor.cont.name, parfor.cont.args};

        auto cont = blocks.at(parfor.cont.name);
        std::erase_if(cont->preds, [&](const std::weak_ptr<Block> &p) {
            const auto ptr = p.lock();
            return ptr && ptr->name == loop->name;
        });
        cont->preds.push_back(block);
    }

    // The gang's pure straight-line callees, copied in where they are
    // called (see inline_pure_variants).
    inline_pure_variants(funcs, f, variants);

    // Every copy made for this gang remembers the loop that made it, which
    // is how a directive over the gang's state finds its copies
    // (Function::gang_loop: `occluded.specialize(count(i_lane) > 14)`).
    for (const auto &[key, name] : variants) {
        if (const auto made = funcs.find(name); made != funcs.end()) {
            made->second->gang_loop = idx;
        }
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
