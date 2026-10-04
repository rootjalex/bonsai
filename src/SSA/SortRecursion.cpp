#include "SSA/SortRecursion.h"

#include "Error.h"
#include "IR/Equality.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
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
// So a run of this shape is sorted as the vectors it is, and over the hits
// alone, the way Embree's AVX-512 traversal does it
// (traverseClosestHitAVX512VL8) -- the strategy a machine with a register
// compress takes (lane_sort_strategy; ir::Target). The keys are made
// integers that order as the floats do with each lane's own index in the
// low bits -- Embree's `distance_i`, which makes every key distinct and
// lets a sorted key say which lane it came from. The hits' keys are then
// compressed to the front (ir::Intrinsic::compress, `vpcompressd`; the
// lanes past them are never read), and a switch on their count picks the
// arm, each arm finishing by itself with the count a constant:
//
// - no hit: straight on to what follows the run, as a run whose every
//   condition is false goes;
// - one hit: a run of one call is a call, to the hit lane of each vector
//   packed to the front and read from lane 0 (one `vpcompressq` and a
//   `vmovq`), which loopify descends into with nothing written to its
//   stack -- three node tests in five on incoherent rays, and Embree's own
//   one-hit case;
// - two, three and four hits: the packed keys read out as scalars and
//   sorted by insertion, a minimum and a maximum per key already placed
//   (Embree's dist_A0/B0, A1/B1/C1, A2/B2/C2/D2 chains), and a run of
//   exactly that many calls, nearest first and none conditional, each
//   call's values taken from the vectors at the index in its key's low
//   bits (Embree's permuteExtract). loopify writes such a run as h - 1
//   scalar pushes at constant offsets from the stack top and descends into
//   the first (SSA/QueueRecursion.cpp): Embree's `stackPtr[0].ptr = ...;
//   stackPtr[1].ptr = ...; stackPtr += 2; cur = ...`. Nothing is permuted,
//   no waiting mask is formed from the count, and no compacting store is
//   made, since the count is a constant in the arm.
// - more hits than that: all the lanes at once, the misses' keys made
//   infinite so that they sort out of the hits' way, by Batcher's bitonic
//   network (Batcher, "Sorting Networks and their Applications", AFIPS
//   1968), descending, the misses first and the hits from the farthest to
//   the nearest in the last lane; the children and whatever travels with
//   them are then taken out by the lane index in each key's low bits with
//   one variable permute each (ir::Intrinsic::permute, `vpermq`,
//   `vpermps`), and the run says it in the shape of a run: call k is lane
//   n - 1 - k of the permuted vectors (the nearest first) and is made when
//   `k < hits`. loopify reads that shape (SortedRun, SSA/SortRecursion.h)
//   and writes the hits that wait with one compacting store. Embree's
//   fallback for five or more hits is the same shape: the keys sorted
//   descending and written to the stack in a loop.
//
// The arms for the common counts are what the switch is for: a node a ray
// passes through usually hits one or two of its children, and those pay
// neither the network nor the join the general case needs.

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
// floats, which is what the integer ordering below is written for) where
// the run has keys, and a scalar per varying position.
struct Lanes {
    shared_ptr<Value> mask;
    shared_ptr<Value> keys;            // null for a run without keys
    vector<shared_ptr<Value>> varying; // one per position in varying_at
    vector<Type> varying_type;         // the element type of each
};

std::optional<Lanes> lanes_of(const Terminator::MultiCall &call) {
    const size_t n = call.varying.size();
    if (n < 2 || (n & (n - 1)) != 0 || call.conds.size() != n ||
        (!call.keys.empty() && call.keys.size() != n)) {
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
        if (!common(out.mask, call.conds[k], k)) {
            return std::nullopt;
        }
        if (!call.keys.empty() && !common(out.keys, call.keys[k], k)) {
            return std::nullopt;
        }
        for (size_t j = 0; j < call.varying_at.size(); j++) {
            if (!common(out.varying[j], call.varying[k][j], k)) {
                return std::nullopt;
            }
        }
    }
    const std::optional<Type> mask_t = element_of_n(out.mask->get_type(), n);
    if (!mask_t.has_value() || !mask_t->is_bool()) {
        return std::nullopt;
    }
    if (out.keys) {
        const std::optional<Type> key_t =
            element_of_n(out.keys->get_type(), n);
        if (!key_t.has_value() || !key_t->is_float() || key_t->bits() != 32) {
            return std::nullopt;
        }
    }
    for (const shared_ptr<Value> &v : out.varying) {
        const std::optional<Type> t = element_of_n(v->get_type(), n);
        if (!t.has_value() || !t->is_scalar() ||
            !(t->is_int_or_uint() || t->is_float())) {
            return std::nullopt;
        }
        out.varying_type.push_back(*t);
    }
    return out;
}

// A new, empty block of `func`, named `stem` or the first `stem_k` not yet
// taken (the helper SSA/QueueRecursion.cpp has, for the same purpose).
shared_ptr<Block> fresh_block(Function &func, const string &stem) {
    std::set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
    }
    string name = stem;
    for (size_t i = 0; taken.count(name); i++) {
        name = stem + "_" + std::to_string(i);
    }
    auto block = std::make_shared<Block>();
    block->name = std::move(name);
    block->owner = func.blocks.front()->owner;
    return block;
}

// One layer of a sorting network: the comparators that run at once, each a
// pair (i, j) that leaves the smaller of the two in lane i and the larger in
// lane j.
using Layer = vector<std::pair<uint32_t, uint32_t>>;

// Batcher's bitonic network over `n` lanes, n a power of two, descending:
// at each layer every lane meets the partner at the bit it flips, and the
// one of the pair the merge direction says takes the larger.
vector<Layer> descending_bitonic(uint32_t n) {
    vector<Layer> layers;
    for (uint32_t k = 2; k <= n; k *= 2) {
        for (uint32_t j = k / 2; j > 0; j /= 2) {
            Layer layer;
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    const bool i_takes_max = ((i & k) == 0) == ((i & j) == 0);
                    layer.push_back(i_takes_max ? std::make_pair(l, i)
                                                : std::make_pair(i, l));
                }
            }
            layers.push_back(std::move(layer));
        }
    }
    return layers;
}

// How a run that is the lanes of vectors is sorted and handed to loopify,
// by the machine the code is for. The choice is made here, in the rewrite,
// rather than in each backend, so that the traversal a schedule produces
// reads in the SSA dump and is pinned by a golden per machine, and so that
// the stack and the loop it builds are written once (see ir::Target).
enum class LaneSortStrategy {
    // Embree's compact-and-sort (bvh_traverser1.h,
    // traverseClosestHitAVX512VL8), described above: the hits' keys packed
    // to the front of a register in one instruction, a switch on their
    // count, the common counts sorted as scalars and pushed one by one, the
    // rest by a network over all the lanes. What a machine with a register
    // compress -- AVX-512, SVE -- takes.
    Compact,
};

LaneSortStrategy lane_sort_strategy(const Target &target) {
    // One strategy today, whatever the machine. A machine without a register
    // compress -- AVX2, NEON -- gets the same code with LLVM's expansion of
    // the compress, a store and a load per lane; Embree's own traversal on
    // such a machine is its generic traverseClosestHit (bvh_traverser1.h): a
    // bit-scan loop over the mask (`bscf`) with a scalar load of the child at
    // each bit, two hits a compare and a store, three and four pushed and
    // sorted on the stack by scalar compare-exchanges. That is the strategy
    // to write here when a machine without a compress is measured.
    (void)target;
    return LaneSortStrategy::Compact;
}

// Sorts the run `call` in `block`, which is the lanes `lanes` describe, as
// described above; see there. Clears the run's keys, and moves the run into
// the arms of a switch on the count of hits, which `block` ends in
// afterwards; the reference `call` is spent.
void sort_lanes(Function &func, const shared_ptr<Block> &block,
                Terminator::MultiCall &call, const Lanes &lanes,
                const ConstantIntervals &intervals) {
    const uint32_t n = uint32_t(call.varying.size());
    const Type u32 = UInt_t::make(32), i32 = Int_t::make(32);
    const Type f32 = Float_t::make_f32(), b = Bool_t::make();
    const Type i32xn = Vector_t::make(i32, n);
    const Type f32xn = Vector_t::make(f32, n), bxn = Vector_t::make(b, n);
    const auto cu32 = [&](uint64_t v) {
        return std::make_shared<Value>(Constant{u32, v});
    };
    const auto ci32 = [&](int64_t v) {
        return std::make_shared<Value>(Constant{i32, v});
    };
    const auto emit = [&](const shared_ptr<Block> &into, Type type,
                          Instruction::Op op,
                          vector<shared_ptr<Value>> operands) {
        return append(func, into, std::move(type), op, std::move(operands));
    };
    const auto intrinsic = [&](const shared_ptr<Block> &into, Type type,
                               ir::Intrinsic::OpType which,
                               vector<shared_ptr<Value>> operands) {
        auto v = append(func, into, std::move(type), Instruction::Op::Intrinsic,
                        std::move(operands));
        std::get<shared_ptr<Instruction>>(v->data)->intrinsic = which;
        return v;
    };
    const auto bc = [&](const shared_ptr<Block> &into,
                        const shared_ptr<Value> &v, const Type &vt) {
        return emit(into, vt, Instruction::Op::Bc, {v, cu32(n)});
    };
    // One lane per bit of a pattern over the lanes.
    const auto lane_pattern = [&](const shared_ptr<Block> &into,
                                  const std::function<bool(uint32_t)> &on) {
        vector<shared_ptr<Value>> bits;
        for (uint32_t i = 0; i < n; i++) {
            bits.push_back(std::make_shared<Value>(Constant{b, on(i)}));
        }
        return emit(into, bxn, Instruction::Op::MakeStruct, std::move(bits));
    };
    const auto shuffle = [&](const shared_ptr<Block> &into,
                             const shared_ptr<Value> &v,
                             const vector<int> &perm) {
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), v->get_type(), Instruction::Op::Shuffle,
            vector<shared_ptr<Value>>{v}, into);
        instr->shuffle = perm;
        into->instrs.push_back(instr);
        return std::make_shared<Value>(std::move(instr));
    };
    // One layer of a network over the keys `x`: every lane meets its
    // partner, and of each pair the lane the comparator puts the larger in
    // takes the maximum, the other the minimum; a lane in no comparator
    // meets itself and keeps its key.
    const auto layer = [&](const shared_ptr<Block> &into, shared_ptr<Value> x,
                           const Layer &comparators) {
        vector<int> perm(n);
        vector<bool> takes_max(n, false);
        for (uint32_t i = 0; i < n; i++) {
            perm[i] = int(i);
        }
        for (const auto &[i, j] : comparators) {
            perm[i] = int(j);
            perm[j] = int(i);
            takes_max[j] = true;
        }
        auto partner = shuffle(into, x, perm);
        auto lo = emit(into, i32xn, Instruction::Op::Min, {x, partner});
        auto hi = emit(into, i32xn, Instruction::Op::Max, {x, partner});
        auto wants_max =
            lane_pattern(into, [&](uint32_t i) { return takes_max[i]; });
        return emit(into, i32xn, Instruction::Op::Select, {wants_max, hi, lo});
    };
    const shared_ptr<Value> mask = lanes_as_vector(func, block, lanes.mask, n);
    const shared_ptr<Value> keys = lanes_as_vector(func, block, lanes.keys, n);
    vector<shared_ptr<Value>> values;
    for (size_t j = 0; j < lanes.varying.size(); j++) {
        values.push_back(lanes_as_vector(func, block, lanes.varying[j], n));
    }

    // As integers that order as the floats do -- a negative float's magnitude
    // bits flipped, so that the more negative is the smaller -- with the
    // lane's own index in the low bits: Embree's `distance_i`. No two lanes
    // are then equal, so a minimum or maximum is one lane's key exactly, and
    // a sorted key says which lane it came from. The misses' keys are left
    // as they are: the compress below drops them, and only the network over
    // all the lanes (the last arm) needs them out of the hits' way, where
    // they are made infinite. The index is a bit field of the key: with the
    // keys scalars, `key & (n - 1)` is the lane to read a child from.
    //
    // The flip goes when the keys are known not to be negative
    // (SSA/ConstantIntervals.h): a non-negative float's bits order as the
    // float does already. Embree's `distance_i` is `asInt(tNear)` with no
    // flip at all, because its traversal clamps the ray's `tnear` to zero
    // at entry (TravRay) and a box's entry distance is `max(.., tnear)`;
    // a program that does the same is read the same way here, the fact
    // carried from the clamp to the key through the calls between.
    auto bits = emit(block, i32xn, Instruction::Op::Reinterpret, {keys});
    const ConstantInterval key_range = intervals.of(func, *block, lanes.keys);
    shared_ptr<Value> ordered = bits;
    if (!(key_range.min_defined && key_range.min >= 0)) {
        auto sign = emit(block, i32xn, Instruction::Op::Shr,
                         {bits, bc(block, ci32(31), i32xn)});
        auto flip = emit(block, i32xn, Instruction::Op::BwAnd,
                         {sign, bc(block, ci32(0x7fffffff), i32xn)});
        ordered = emit(block, i32xn, Instruction::Op::Xor, {bits, flip});
    }
    auto lane = emit(block, i32xn, Instruction::Op::Ramp, {ci32(0), ci32(1)});
    auto high = emit(block, i32xn, Instruction::Op::BwAnd,
                     {ordered, bc(block, ci32(-int64_t(n)), i32xn)});
    auto packed = emit(block, i32xn, Instruction::Op::BwOr, {high, lane});

    // How many calls are made: the lanes the mask has on. Their keys, packed
    // to the front in lane order, are what the arms for two to four hits
    // read; nothing reads the lanes past them, so the compress fills them
    // with nothing (Embree's `compact`).
    auto hits = emit(block, u32, Instruction::Op::Popcount, {mask});
    auto compressed =
        intrinsic(block, i32xn, ir::Intrinsic::compress, {packed, mask});

    // The value of each varying parameter at the lane a key names: the
    // key's low bits are the lane (see `packed`), and the value is read
    // from the vector there.
    const auto at_key = [&](const shared_ptr<Block> &into,
                            const shared_ptr<Value> &key) {
        auto masked = emit(into, i32, Instruction::Op::BwAnd,
                           {key, ci32(int64_t(n) - 1)});
        auto idx = emit(into, u32, Instruction::Op::Reinterpret, {masked});
        vector<shared_ptr<Value>> row;
        for (size_t j = 0; j < values.size(); j++) {
            row.push_back(emit(into, lanes.varying_type[j],
                               Instruction::Op::ExtractIdx, {values[j], idx}));
        }
        return row;
    };

    // The switch on the count: target k for k hits, the last target for
    // every count past the ones specialized.
    vector<Terminator::Jump> targets;
    {
        // No hit: no call is made and the run is over, so straight to what
        // follows it, as a run whose every condition is false goes -- not
        // through the join, whose permutes and extracts would run for
        // nothing (one node test in five on incoherent rays).
        auto none = fresh_block(func, block->name + "!sort0");
        none->terminator.data = call.cont;
        func.blocks.push_back(none);
        targets.push_back(Terminator::Jump{none->name, {}});
    }
    {
        // One hit: nothing to order and nothing to wait -- a run of one
        // call is a call. Its arguments are the hit lane of each vector,
        // packed to the front by the mask and read from lane 0 (one
        // `vpcompressq` and a `vmovq`: Embree's one-hit case, its first
        // permuteExtract), with no keys, no network and no run for loopify
        // to write a stack from: a single call goes straight to the child
        // (SSA/QueueRecursion.cpp). Three node tests in five on incoherent
        // rays hit one child, and this arm is what they cost.
        auto one = fresh_block(func, block->name + "!sort1");
        Terminator::Call single;
        single.call = call.call;
        single.cont = call.cont;
        single.drop = call.drop;
        for (size_t j = 0; j < values.size(); j++) {
            auto packed_j = intrinsic(one, values[j]->get_type(),
                                      ir::Intrinsic::compress,
                                      {values[j], mask});
            single.call.args[call.varying_at[j]] =
                emit(one, lanes.varying_type[j], Instruction::Op::ExtractIdx,
                     {packed_j, cu32(0)});
        }
        one->terminator.data = std::move(single);
        func.blocks.push_back(one);
        targets.push_back(Terminator::Jump{one->name, {}});
    }
    const uint32_t specialized = std::min<uint32_t>(n, 4);
    for (uint32_t h = 2; h <= specialized; h++) {
        // h hits, and h a constant here: the h packed keys read out as
        // scalars and sorted by insertion -- each key placed with a minimum
        // and a maximum against every key before it, Embree's chains of
        // dist_A0 and dist_B0, then A1, B1, C1, then A2 to D2 -- and a run of
        // exactly h calls, the nearest first and none conditional, each
        // call's values read from the vectors at the lane its key names.
        // loopify writes it as h - 1 scalar pushes at constant offsets from
        // the stack top and a descent into the first (SSA/QueueRecursion
        // .cpp), Embree's `stackPtr[0].ptr = ...; stackPtr++; cur = ...`.
        auto block_h =
            fresh_block(func, block->name + "!sort" + std::to_string(h));
        vector<shared_ptr<Value>> ascending;
        for (uint32_t k = 0; k < h; k++) {
            shared_ptr<Value> key = emit(block_h, i32, Instruction::Op::ExtractIdx,
                                         {compressed, cu32(k)});
            for (shared_ptr<Value> &placed : ascending) {
                auto lo = emit(block_h, i32, Instruction::Op::Min, {placed, key});
                auto hi = emit(block_h, i32, Instruction::Op::Max, {placed, key});
                placed = lo;
                key = hi;
            }
            ascending.push_back(key);
        }
        Terminator::MultiCall run;
        run.call = call.call;
        run.cont = call.cont;
        run.drop = call.drop;
        run.varying_at = call.varying_at;
        for (const shared_ptr<Value> &key : ascending) {
            run.varying.push_back(at_key(block_h, key));
        }
        block_h->terminator.data = std::move(run);
        func.blocks.push_back(block_h);
        targets.push_back(Terminator::Jump{block_h->name, {}});
    }
    if (specialized < n) {
        // Every count past them: all the lanes at once, the misses' infinite
        // keys and the hits' together, by the bitonic network descending,
        // the misses first and the hits from the farthest to the nearest in
        // the last lane; the children and what travels with them taken out
        // by the sorted order -- the lane index in each key's low bits --
        // with one variable permute each; and the run in that order, the
        // nearest first, call k made when `k < hits`: the shape loopify
        // reads as a SortedRun.
        auto all = fresh_block(func, block->name + "!sortall");
        // The misses infinitely far, whatever their keys say, so that they
        // sort first in descending order, out of the hits' way: the key of
        // +inf (its bits are already in order, and its low bits clear) with
        // the lane's index.
        auto inf_key =
            emit(all, i32xn, Instruction::Op::BwOr,
                 {bc(all, ci32(0x7f800000), i32xn), lane});
        shared_ptr<Value> sorted = emit(all, i32xn, Instruction::Op::Select,
                                        {mask, packed, inf_key});
        for (const Layer &comparators : descending_bitonic(n)) {
            sorted = layer(all, sorted, comparators);
        }
        auto idx = emit(all, i32xn, Instruction::Op::BwAnd,
                        {sorted, bc(all, ci32(int64_t(n) - 1), i32xn)});
        vector<shared_ptr<Value>> permuted;
        for (const shared_ptr<Value> &v : values) {
            permuted.push_back(
                intrinsic(all, v->get_type(), ir::Intrinsic::permute, {v, idx}));
        }
        for (uint32_t k = 0; k < n; k++) {
            for (size_t j = 0; j < values.size(); j++) {
                call.varying[k][j] =
                    emit(all, lanes.varying_type[j], Instruction::Op::ExtractIdx,
                         {permuted[j], cu32(n - 1 - k)});
            }
            call.conds[k] = emit(all, b, Instruction::Op::Lt, {cu32(k), hits});
        }
        call.keys.clear();
        all->terminator.data = std::move(block->terminator.data);
        func.blocks.push_back(all);
        targets.push_back(Terminator::Jump{all->name, {}});
    }
    // Not one switch on the count but Embree's traverseClosestHit in its own
    // order: no hit first (`if (unlikely(mask == 0)) goto pop`, a kortest),
    // then one hit (`r = bscf(mask); if (likely(mask == 0))`: `bits & (bits
    // - 1) == 0`, one instruction on the mask as a word), then the counts
    // from two up as the switch, on `hits - 2`. A switch on the count with
    // arms for none, one, two, three and four hits and a default is five
    // cases, and LLVM makes a jump table of four or more -- an indirect
    // branch behind a load, and the dispatch done twice where the prefetch
    // of each hit lane tests the count before it. None and one are the
    // common cases on incoherent rays (three node tests in five hit one
    // child); taken first, the switch left has three cases, below the table
    // threshold, and the two tests are the ones Embree runs. The same shape
    // as the any hit's (SSA/QueueRecursion.cpp).
    auto some = emit(block, b, Instruction::Op::Any, {mask});
    auto hit = fresh_block(func, block->name + "!sorthit");
    func.blocks.push_back(hit);
    shared_ptr<Value> one;
    if (n == 8 || n == 16 || n == 32 || n == 64) {
        const Type bits_t = UInt_t::make(n);
        const Type word_t = n < 32 ? u32 : bits_t;
        shared_ptr<Value> word =
            emit(hit, bits_t, Instruction::Op::Reinterpret, {mask});
        if (n < 32) {
            word = emit(hit, word_t, Instruction::Op::Cast, {word});
        }
        auto one_of = std::make_shared<Value>(Constant{word_t, uint64_t(1)});
        auto none_of = std::make_shared<Value>(Constant{word_t, uint64_t(0)});
        auto below = emit(hit, word_t, Instruction::Op::Sub, {word, one_of});
        auto cleared = emit(hit, word_t, Instruction::Op::BwAnd, {word, below});
        one = emit(hit, b, Instruction::Op::Eq, {cleared, none_of});
    } else {
        one = emit(hit, b, Instruction::Op::Eq, {hits, cu32(1)});
    }
    auto from_two = fresh_block(func, block->name + "!sort2up");
    func.blocks.push_back(from_two);
    auto count_from_two = emit(from_two, u32, Instruction::Op::Sub, {hits, cu32(2)});
    vector<Terminator::Jump> arms_from_two(targets.begin() + 2, targets.end());
    from_two->terminator.data =
        Terminator::Dispatch{count_from_two, std::move(arms_from_two)};
    hit->terminator.data = Terminator::Dispatch{
        one, {Terminator::Jump{from_two->name, {}}, targets[1]}};
    block->terminator.data =
        Terminator::Dispatch{some, {targets[0], Terminator::Jump{hit->name, {}}}};
}

} // namespace

shared_ptr<Value> lanes_as_vector(Function &func,
                                  const shared_ptr<Block> &block,
                                  const shared_ptr<Value> &source,
                                  uint32_t lanes) {
    if (source->get_type().is<Vector_t>()) {
        return source;
    }
    const std::optional<Type> element =
        element_of_n(source->get_type(), lanes);
    internal_assert(element.has_value())
        << "The lanes of " << source->get_type() << " as a vector of "
        << lanes;
    const Type u32 = UInt_t::make(32);
    auto ramp = append(func, block, Vector_t::make(u32, lanes),
                       Instruction::Op::Ramp,
                       {std::make_shared<Value>(Constant{u32, uint64_t(0)}),
                        std::make_shared<Value>(Constant{u32, uint64_t(1)})});
    return append(func, block, Vector_t::make(*element, lanes),
                  Instruction::Op::ExtractIdx, {source, std::move(ramp)});
}

std::optional<LaneRun> lane_run(const Terminator::MultiCall &call) {
    if (!call.keys.empty()) {
        return std::nullopt;
    }
    const std::optional<Lanes> lanes = lanes_of(call);
    if (!lanes.has_value()) {
        return std::nullopt;
    }
    LaneRun run;
    run.mask = lanes->mask;
    run.lanes = uint32_t(call.varying.size());
    for (size_t j = 0; j < call.varying_at.size(); j++) {
        run.values[call.varying_at[j]] = lanes->varying[j];
    }
    return run;
}

size_t sort_recursion(Function &func, const Target &target,
                      const ConstantIntervals &intervals) {
    const LaneSortStrategy strategy = lane_sort_strategy(target);
    internal_assert(strategy == LaneSortStrategy::Compact);
    size_t sorted = 0;
    // Over a copy of the list: a run sorted as lanes adds the blocks of its
    // switch to the function.
    const vector<shared_ptr<Block>> blocks = func.blocks;
    for (const auto &block : blocks) {
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
            if (const std::optional<Lanes> lanes = lanes_of(*call);
                lanes.has_value() && lanes->keys) {
                // Clears the keys itself and moves the run out of `block`,
                // which ends in a switch afterwards: `call` is spent.
                sort_lanes(func, block, *call, *lanes, intervals);
                sorted++;
                continue;
            }
            sort_network(func, block, call->keys, call->varying,
                         call->conds);
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
