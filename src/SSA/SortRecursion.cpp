#include "SSA/SortRecursion.h"

#include "Error.h"
#include "IR/Equality.h"
#include "Utils.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Appends an instruction to `block`. The same helper QueueRecursion uses, and
// for the same reason: everything a sort key is built from is already live
// where the run is, so nothing needs threading through block arguments.
shared_ptr<Value> append(Function &func, const shared_ptr<Block> &block,
                         Type type, Instruction::Op op,
                         vector<shared_ptr<Value>> operands,
                         const string &name = "") {
    auto instr = std::make_shared<Instruction>(
        name.empty() ? func.get_unique_name() : name, std::move(type), op,
        std::move(operands), block);
    block->instrs.push_back(instr);
    return std::make_shared<Value>(std::move(instr));
}

// The name an instruction's value goes by.
const string &name_of(const shared_ptr<Value> &v) {
    return std::get<shared_ptr<Instruction>>(v->data)->name;
}

// One compare-and-swap of a sorting network, applied to the keys and to
// everything travelling with them.
//
// `keep` is true where element i is the smaller of the pair, and both elements
// are rewritten to selects on it. Note that both arms of every select are
// values that already exist -- the point of doing this here is that they are
// node indices by now, so a select is a select and not a materialisation.
//
// The comparison is put to a vote (Instruction::Op::Vote). The run is made
// once by whoever executes it, in one order; when that is a gang whose lanes
// hold different keys -- rays of both signs along the split axis -- the order
// cannot be each lane's own, and the vote is how the gang settles on one. For
// a gang of one, which is every scalar traversal, the vote is the comparison
// itself; and a run that loopify() puts on a stack is visited lane by lane
// again, so queue_recursion() drops the votes it finds there.
void compare_and_swap(Function &func, const shared_ptr<Block> &block,
                      vector<shared_ptr<Value>> &keys,
                      vector<vector<shared_ptr<Value>>> &varying,
                      vector<shared_ptr<Value>> &conds, size_t i, size_t l,
                      bool ascending) {
    const Type &key_type = keys[i]->get_type();
    internal_assert(equals(key_type, keys[l]->get_type()))
        << "A sort's keys have different types, " << key_type << " and "
        << keys[l]->get_type();

    shared_ptr<Value> compared =
        append(func, block, Bool_t::make(), Instruction::Op::Lt,
               ascending ? vector<shared_ptr<Value>>{keys[i], keys[l]}
                         : vector<shared_ptr<Value>>{keys[l], keys[i]});
    // Named after the comparison it decides, so that the vote reads as what
    // it is in a dump, and so that a scalar traversal -- where loopify drops
    // it again -- numbers everything after it as before.
    shared_ptr<Value> keep =
        append(func, block, Bool_t::make(), Instruction::Op::Vote, {compared},
               name_of(compared) + "!vote");

    auto pick = [&](const shared_ptr<Value> &a, const shared_ptr<Value> &b) {
        return append(func, block, a->get_type(), Instruction::Op::Select,
                      {keep, a, b});
    };

    // Read both sides out before writing either, or the second select would
    // pick from a value the first one just replaced.
    const shared_ptr<Value> key_i = keys[i], key_l = keys[l];
    keys[i] = pick(key_i, key_l);
    keys[l] = pick(key_l, key_i);

    internal_assert(varying[i].size() == varying[l].size());
    for (size_t k = 0; k < varying[i].size(); k++) {
        const shared_ptr<Value> vi = varying[i][k], vl = varying[l][k];
        varying[i][k] = pick(vi, vl);
        varying[l][k] = pick(vl, vi);
    }
    // A branch's condition travels with the branch: the child the parent's
    // test let through is the one that is made, wherever the order put it.
    if (!conds.empty()) {
        const shared_ptr<Value> ci = conds[i], cl = conds[l];
        conds[i] = pick(ci, cl);
        conds[l] = pick(cl, ci);
    }
}

// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

// Sorts `keys`, and `varying` along with them, into ascending order.
//
// A bitonic network, so the comparisons are a fixed sequence that does not
// depend on the values -- which is the only kind of sort available here, since
// the run has to come out as a fixed number of calls whatever the keys turn out
// to be at run time.
void sort_network(Function &func, const shared_ptr<Block> &block,
                  vector<shared_ptr<Value>> &keys,
                  vector<vector<shared_ptr<Value>>> &varying,
                  vector<shared_ptr<Value>> &conds) {
    const size_t count = keys.size();
    const size_t n = next_pow2(count);
    for (size_t k = 2; k <= n; k *= 2) {
        for (size_t j = k / 2; j > 0; j /= 2) {
            for (size_t i = 0; i < n; i++) {
                const size_t l = i ^ j;
                if (l > i && l < count) {
                    compare_and_swap(func, block, keys, varying, conds, i, l,
                                     (i & k) == 0);
                }
            }
        }
    }
}

//===--------------------------------------------------------------------===//
// A run that is the lanes of vectors, sorted over the calls that are made
//===--------------------------------------------------------------------===//
//
// A node that holds its children's boxes tests them in a loop over the
// children (Lower/Trees.cpp, from_children), which leaves the run's
// conditions as the lanes of one mask, its keys as the lanes of one array of
// keys, and its children as the lanes of the node's vector of references:
// call k is made under `mask[k]`, ordered by `keys[k]`, on `children[k]`.
// The network above would order all of them -- the children the ray missed
// too, by their infinite keys, in scalar compare-and-swaps that carry the
// key, the reference and the condition of every pair -- when only the hits
// are ever visited. Embree's traversal of such a node (bvh_traverser1.h,
// traverseClosestHit) orders the hits alone.
//
// So a run of this shape is sorted as the vectors it is. The misses' keys
// are set infinite, the keys are made integers that order as the floats do
// with each lane's own index in the low bits -- Embree's `distance_i`, which
// makes every key distinct so that the lane travels with its key -- and one
// bitonic network of vector minimums and maximums puts them in descending
// order: the misses first, then the hits from the farthest to the nearest in
// the last lane. Each varying value follows its key through the network.
// What the run is made of afterwards says it in the shape of a run: call k
// is lane n - 1 - k of the sorted vectors (the nearest first) and is made
// when `k < hits`, the number of lanes the mask has on. A run still made as
// calls is made in that order; one put on a stack (SSA/QueueRecursion.h)
// reads the vectors behind those lanes and writes the hits that wait with
// one compacting store.

// A value's identity: the instruction it is, or the parameter it names.
bool same_source(const Value &a, const Value &b) {
    const auto *ia = std::get_if<shared_ptr<Instruction>>(&a.data);
    const auto *ib = std::get_if<shared_ptr<Instruction>>(&b.data);
    if (ia != nullptr && ib != nullptr) {
        return ia->get() == ib->get();
    }
    const auto *aa = std::get_if<Argument>(&a.data);
    const auto *ab = std::get_if<Argument>(&b.data);
    return aa != nullptr && ab != nullptr && aa->name == ab->name;
}

std::optional<uint64_t> constant_index(const Value &value) {
    const auto *constant = std::get_if<Constant>(&value.data);
    if (constant == nullptr) {
        return std::nullopt;
    }
    return std::visit(
        overloads{
            [](uint64_t u) -> std::optional<uint64_t> { return u; },
            [](int64_t i) -> std::optional<uint64_t> {
                return i < 0 ? std::nullopt
                             : std::optional<uint64_t>(uint64_t(i));
            },
            [](auto) -> std::optional<uint64_t> { return std::nullopt; },
        },
        constant->data);
}

// What `v` is lane `k` of -- `extract_idx(source, k)` -- or nothing.
shared_ptr<Value> lane_of(const shared_ptr<Value> &v, uint64_t k) {
    const auto *i = std::get_if<shared_ptr<Instruction>>(&v->data);
    if (i == nullptr || (*i)->op != Instruction::Op::ExtractIdx ||
        (*i)->operands.size() != 2) {
        return nullptr;
    }
    const std::optional<uint64_t> idx = constant_index(*(*i)->operands[1]);
    if (!idx.has_value() || *idx != k) {
        return nullptr;
    }
    return (*i)->operands[0];
}

// The element type of a source of `n` lanes: a vector of `n`, or an array of
// `n`; nothing for anything else.
std::optional<Type> element_of_n(const Type &type, uint64_t n) {
    if (const Vector_t *v = type.as<Vector_t>()) {
        return v->lanes == n && !v->packed ? std::optional<Type>(v->etype)
                                            : std::nullopt;
    }
    if (const Array_t *a = type.as<Array_t>()) {
        if (!a->size.defined()) {
            return std::nullopt;
        }
        const std::optional<uint64_t> size =
            get_constant_value<uint64_t>(a->size);
        return size.has_value() && *size == n ? std::optional<Type>(a->etype)
                                              : std::nullopt;
    }
    return std::nullopt;
}

// The run's conditions, keys and varying values as the lanes of one source
// each, when that is what they are: the mask of booleans, the keys (32-bit
// floats, which is what the integer ordering below is written for), and a
// scalar per varying position.
struct Lanes {
    shared_ptr<Value> mask;
    shared_ptr<Value> keys;
    vector<shared_ptr<Value>> varying; // one per position in varying_at
    vector<Type> varying_type;         // the element type of each
};

std::optional<Lanes> lanes_of(const Terminator::MultiCall &call) {
    const size_t n = call.varying.size();
    if (n < 2 || (n & (n - 1)) != 0 || call.conds.size() != n ||
        call.keys.size() != n) {
        return std::nullopt;
    }
    Lanes out;
    out.varying.resize(call.varying_at.size());
    const auto common = [&](shared_ptr<Value> &slot, const shared_ptr<Value> &v,
                            size_t k) {
        shared_ptr<Value> source = lane_of(v, k);
        if (!source) {
            return false;
        }
        if (!slot) {
            slot = std::move(source);
        } else if (!same_source(*slot, *source)) {
            return false;
        }
        return true;
    };
    for (size_t k = 0; k < n; k++) {
        if (!common(out.mask, call.conds[k], k) ||
            !common(out.keys, call.keys[k], k)) {
            return std::nullopt;
        }
        for (size_t j = 0; j < call.varying_at.size(); j++) {
            if (!common(out.varying[j], call.varying[k][j], k)) {
                return std::nullopt;
            }
        }
    }
    const std::optional<Type> mask_t = element_of_n(out.mask->get_type(), n);
    const std::optional<Type> key_t = element_of_n(out.keys->get_type(), n);
    if (!mask_t.has_value() || !mask_t->is_bool() || !key_t.has_value() ||
        !key_t->is_float() || key_t->bits() != 32) {
        return std::nullopt;
    }
    for (const shared_ptr<Value> &v : out.varying) {
        const std::optional<Type> t = element_of_n(v->get_type(), n);
        if (!t.has_value() || !t->is_scalar() || !t->is_int_or_uint()) {
            return std::nullopt;
        }
        out.varying_type.push_back(*t);
    }
    return out;
}

// Sorts the run `call` in `block`, which is the lanes `lanes` describe, as
// described above; see there.
void sort_lanes(Function &func, const shared_ptr<Block> &block,
                Terminator::MultiCall &call, const Lanes &lanes) {
    const uint32_t n = uint32_t(call.varying.size());
    const Type u32 = UInt_t::make(32), i32 = Int_t::make(32);
    const Type f32 = Float_t::make_f32(), b = Bool_t::make();
    const Type u32xn = Vector_t::make(u32, n), i32xn = Vector_t::make(i32, n);
    const Type f32xn = Vector_t::make(f32, n), bxn = Vector_t::make(b, n);
    const auto cu32 = [&](uint64_t v) {
        return std::make_shared<Value>(Constant{u32, v});
    };
    const auto ci32 = [&](int64_t v) {
        return std::make_shared<Value>(Constant{i32, v});
    };
    const auto emit = [&](Type type, Instruction::Op op,
                          vector<shared_ptr<Value>> operands) {
        return append(func, block, std::move(type), op, std::move(operands));
    };
    const auto bc = [&](const shared_ptr<Value> &v, const Type &vt) {
        return emit(vt, Instruction::Op::Bc, {v, cu32(n)});
    };
    // A source that is an array is read whole, as the vector it holds.
    const auto as_vector = [&](const shared_ptr<Value> &source,
                               const Type &element) {
        if (source->get_type().is<Vector_t>()) {
            return source;
        }
        auto ramp = emit(u32xn, Instruction::Op::Ramp, {cu32(0), cu32(1)});
        return emit(Vector_t::make(element, n), Instruction::Op::ExtractIdx,
                    {source, std::move(ramp)});
    };
    // One lane per bit of a pattern over the lanes.
    const auto lane_pattern = [&](const std::function<bool(uint32_t)> &on) {
        vector<shared_ptr<Value>> bits;
        for (uint32_t i = 0; i < n; i++) {
            bits.push_back(std::make_shared<Value>(Constant{b, on(i)}));
        }
        return emit(bxn, Instruction::Op::MakeStruct, std::move(bits));
    };
    const auto shuffle = [&](const shared_ptr<Value> &v,
                             const vector<int> &perm) {
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), v->get_type(), Instruction::Op::Shuffle,
            vector<shared_ptr<Value>>{v}, block);
        instr->shuffle = perm;
        block->instrs.push_back(instr);
        return std::make_shared<Value>(std::move(instr));
    };

    const shared_ptr<Value> mask = as_vector(lanes.mask, b);
    const shared_ptr<Value> keys = as_vector(lanes.keys, f32);
    vector<shared_ptr<Value>> values;
    for (size_t j = 0; j < lanes.varying.size(); j++) {
        values.push_back(as_vector(lanes.varying[j], lanes.varying_type[j]));
    }

    // The misses infinitely far, whatever their keys say: they sort first in
    // descending order, out of the hits' way.
    auto inf = emit(f32, Instruction::Op::Inf, {});
    auto far = emit(f32xn, Instruction::Op::Select, {mask, keys, bc(inf, f32xn)});
    // As integers that order as the floats do -- a negative float's magnitude
    // bits flipped, so that the more negative is the smaller -- with the
    // lane's own index in the low bits: Embree's `distance_i`. No two lanes
    // are then equal, so a minimum or maximum is one lane's key exactly, and
    // whether a lane kept its own says whether it kept its own child.
    auto bits = emit(i32xn, Instruction::Op::Reinterpret, {far});
    auto sign = emit(i32xn, Instruction::Op::Shr, {bits, bc(ci32(31), i32xn)});
    auto flip = emit(i32xn, Instruction::Op::BwAnd,
                     {sign, bc(ci32(0x7fffffff), i32xn)});
    auto ordered = emit(i32xn, Instruction::Op::Xor, {bits, flip});
    auto lane = emit(i32xn, Instruction::Op::Ramp, {ci32(0), ci32(1)});
    auto high = emit(i32xn, Instruction::Op::BwAnd,
                     {ordered, bc(ci32(-int64_t(n)), i32xn)});
    shared_ptr<Value> packed = emit(i32xn, Instruction::Op::BwOr, {high, lane});

    // The bitonic network over the lanes, descending: at each step every lane
    // meets its partner, and the one of the pair that wants the larger takes
    // the maximum, the other the minimum.
    for (uint32_t k = 2; k <= n; k *= 2) {
        for (uint32_t j = k / 2; j > 0; j /= 2) {
            vector<int> perm(n);
            for (uint32_t i = 0; i < n; i++) {
                perm[i] = int(i ^ j);
            }
            auto partner = shuffle(packed, perm);
            auto lo = emit(i32xn, Instruction::Op::Min, {packed, partner});
            auto hi = emit(i32xn, Instruction::Op::Max, {packed, partner});
            auto wants_max = lane_pattern([&](uint32_t i) {
                return ((i & k) == 0) == ((i & j) == 0);
            });
            auto next = emit(i32xn, Instruction::Op::Select, {wants_max, hi, lo});
            auto moved = emit(bxn, Instruction::Op::Ne, {next, packed});
            for (shared_ptr<Value> &v : values) {
                auto theirs = shuffle(v, perm);
                v = emit(v->get_type(), Instruction::Op::Select,
                         {moved, std::move(theirs), v});
            }
            packed = std::move(next);
        }
    }

    // How many calls are made: the lanes the mask has on, which are the last
    // `hits` lanes of the sorted vectors.
    auto hits = emit(u32, Instruction::Op::Popcount, {mask});

    // The run in the sorted order: the nearest -- the last lane -- first.
    for (uint32_t k = 0; k < n; k++) {
        for (size_t j = 0; j < values.size(); j++) {
            call.varying[k][j] =
                emit(lanes.varying_type[j], Instruction::Op::ExtractIdx,
                     {values[j], cu32(n - 1 - k)});
        }
        call.conds[k] = emit(b, Instruction::Op::Lt, {cu32(k), hits});
    }
}

} // namespace

size_t sort_recursion(Function &func) {
    size_t sorted = 0;
    for (const auto &block : func.blocks) {
        auto *call =
            std::get_if<Terminator::MultiCall>(&block->terminator.data);
        if (call == nullptr || call->keys.empty()) {
            continue;
        }
        internal_assert(call->keys.size() == call->varying.size())
            << "The run in " << block->name << " has " << call->keys.size()
            << " sort keys for " << call->varying.size() << " calls";

        // A run of one is already sorted, and a network over it would emit
        // nothing anyway -- but the keys still have to go, since they are only
        // meaningful to this pass.
        internal_assert(call->conds.empty() ||
                        call->conds.size() == call->varying.size())
            << "The run in " << block->name << " has " << call->conds.size()
            << " conditions for " << call->varying.size() << " calls";
        if (call->varying.size() > 1) {
            if (const std::optional<Lanes> lanes = lanes_of(*call)) {
                sort_lanes(func, block, *call, *lanes);
            } else {
                sort_network(func, block, call->keys, call->varying,
                             call->conds);
            }
            sorted++;
        }
        call->keys.clear();
    }
    return sorted;
}

std::optional<SortedRun> sorted_run(const Terminator::MultiCall &call) {
    const size_t n = call.varying.size();
    if (n < 2 || call.conds.size() != n || !call.keys.empty()) {
        return std::nullopt;
    }
    SortedRun run;
    run.lanes = uint32_t(n);
    vector<shared_ptr<Value>> values(call.varying_at.size());
    for (size_t k = 0; k < n; k++) {
        // The condition `k < hits`.
        const auto *lt =
            std::get_if<shared_ptr<Instruction>>(&call.conds[k]->data);
        if (lt == nullptr || (*lt)->op != Instruction::Op::Lt ||
            (*lt)->operands.size() != 2 ||
            constant_index(*(*lt)->operands[0]) != k) {
            return std::nullopt;
        }
        const shared_ptr<Value> &hits = (*lt)->operands[1];
        if (!run.hits) {
            run.hits = hits;
        } else if (!same_source(*run.hits, *hits)) {
            return std::nullopt;
        }
        // The values, lane n - 1 - k of one vector each.
        for (size_t j = 0; j < call.varying_at.size(); j++) {
            shared_ptr<Value> source = lane_of(call.varying[k][j], n - 1 - k);
            if (!source || !source->get_type().is<Vector_t>() ||
                source->get_type().lanes() != n) {
                return std::nullopt;
            }
            if (!values[j]) {
                values[j] = std::move(source);
            } else if (!same_source(*values[j], *source)) {
                return std::nullopt;
            }
        }
    }
    for (size_t j = 0; j < call.varying_at.size(); j++) {
        run.values[call.varying_at[j]] = values[j];
    }
    return run;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
