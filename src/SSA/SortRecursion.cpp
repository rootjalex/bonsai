#include "SSA/SortRecursion.h"

#include "Error.h"
#include "IR/Equality.h"
#include "SSA/MaskLanes.h"
#include "SSA/Storage.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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
// So a run of this shape is sorted over the hits alone, in the one shape
// both of Embree's kernels share -- traverseClosestHit for N = 4 and
// traverseClosestHitAVX512VL8 for N = 8 are one procedure in two
// representations -- a chain that peels the hit lanes one at a time:
//
// - no hit: straight on to what follows the run, as a run whose every
//   condition is false goes (`if (unlikely(mask == 0)) goto pop`, a
//   kortest);
// - peel the lowest lane the mask has left (SSA/MaskLanes.h: `tzcnt` on
//   the mask's word), and read the call at it -- its child, whatever
//   travels with the child, and its key -- one extract at the lane each
//   (Embree's `node->child(r)` and `((unsigned int*)&tNear)[r]`: a scalar
//   load off the row, and an indexed load off the key vector);
// - insert it among the calls peeled before it, nearest first, by a
//   compare against each in turn and an exchange of the pair where the new
//   one is nearer (Embree's dist_A0/B0, then A1/B1/C1, then A2 to D2), the
//   exchange a select per component -- the key, the child, what travels
//   with it;
// - clear the lane (`blsr`) and test whether any is left, the flag the
//   clear leaves (`if (likely(mask == 0))`). None left is the arm for
//   exactly that many hits: one hit is a call to the child peeled, with
//   nothing written to a stack (three node tests in five on incoherent
//   rays, and Embree's one-hit case); h hits are a run of exactly h calls,
//   the nearest first and none conditional, which loopify writes as h - 1
//   scalar pushes at constant offsets from the stack top and a descent
//   into the first (SSA/QueueRecursion.cpp): Embree's `stackPtr->ptr = c1;
//   stackPtr->dist = d1; stackPtr++; cur = c0`. Some left peels the next.
// - past four peels (a node wider than four): all the lanes at once, each
//   key with its lane's index in its low bits so that a sorted key says
//   which lane it came from (Embree's `distance_i`) and the misses' keys
//   made infinite so that they sort out of the hits' way, by Batcher's
//   bitonic network (Batcher, "Sorting Networks and their Applications",
//   AFIPS 1968), descending, the misses first and the hits from the
//   farthest to the nearest in the last lane; the children and whatever
//   travels with them are then taken out by the lane index in each key's
//   low bits with one variable permute each (ir::Intrinsic::permute,
//   `vpermq`, `vpermps`), and the run says it in the shape of a run: call
//   k is lane n - 1 - k of the permuted vectors (the nearest first) and is
//   made when `k < hits`. loopify reads that shape (SortedRun,
//   SSA/SortRecursion.h) and writes the hits that wait with one compacting
//   store. Embree's fallback for five or more hits is the same shape: the
//   keys sorted descending and written to the stack in a loop.
//
// The chain shares its work downward: the arm for three hits does one more
// peel and two more exchanges than the arm for two, not three from
// nothing, and its dispatch costs nothing beyond the clear -- no count is
// taken, and no keys are packed or compressed, for the counts the arms
// cover. The items are scalars throughout: the child reference as the
// layout stores it, the key's bits, and whatever else travels. That is what
// Embree's four-wide kernel holds (a StackItem's ptr and dist, in general
// registers) and what its eight-wide kernel holds broadcast in vector
// registers, with a compress for the peel and a permute per child taken
// out; the scalar items are the one representation here, for every lane
// count -- the count decides the mask's word and how many peels come
// before the general arm, and nothing else (the user's direction,
// 2026-10-05: one parametric form, the item what the layout stores).

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

// Does `v` refer to the value named `name`? A definition and the block
// arguments that thread it onwards share a name (the helper
// SSA/PromoteAllocas.cpp has, for the same reason).
bool refers_to(const Value &v, const string &name) {
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) { return i->name == name; },
            [](const Constant &) { return false; },
            [&](const Argument &a) { return a.name == name; },
        },
        v.data);
}

// What is stored into a lane source that is an array of the function's own
// -- the keys the children's loop computes, a bound it carries to the child
// -- as every store into it: the block, the index and the value. Nothing
// for a source that is not such an array, or one that anything but those
// stores and the reads of its lanes touches.
struct Fill {
    vector<std::tuple<const Block *, shared_ptr<Value>, shared_ptr<Value>>>
        stores;
};

std::optional<Fill> fill_of(const Function &func,
                            const shared_ptr<Value> &source) {
    if (!source->get_type().is<Array_t>()) {
        return std::nullopt;
    }
    string name;
    if (const std::optional<Argument> a = source->get_argument()) {
        name = a->name;
    } else if (const auto *i =
                   std::get_if<shared_ptr<Instruction>>(&source->data)) {
        name = (*i)->name;
    } else {
        return std::nullopt;
    }
    bool allocated = false;
    std::set<const Instruction *> addresses; // the GEPs into it
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Alloca && instr->name == name) {
                allocated = true;
                continue;
            }
            bool touches = false;
            for (const auto &operand : instr->operands) {
                touches = touches || refers_to(*operand, name);
            }
            if (!touches) {
                continue;
            }
            if (instr->op == Instruction::Op::GEP && instr->operands.size() == 2 &&
                refers_to(*instr->operands[0], name) &&
                instr->operands[1]->get_type().is_scalar()) {
                addresses.insert(instr.get());
                continue;
            }
            if (instr->op == Instruction::Op::ExtractIdx &&
                instr->operands.size() == 2 &&
                refers_to(*instr->operands[0], name)) {
                continue;
            }
            return std::nullopt;
        }
    }
    if (!allocated) {
        return std::nullopt;
    }
    // Every address into it is stored through or loaded from, and nothing
    // else; the stores are the fill.
    Fill fill;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            for (size_t k = 0; k < instr->operands.size(); k++) {
                const auto *held =
                    std::get_if<shared_ptr<Instruction>>(&instr->operands[k]->data);
                if (held == nullptr || !addresses.count(held->get())) {
                    continue;
                }
                if (instr->op == Instruction::Op::Store && k == 0 &&
                    instr->operands.size() == 2) {
                    fill.stores.emplace_back(block.get(), (*held)->operands[1],
                                             instr->operands[1]);
                } else if (instr->op != Instruction::Op::Load || k != 0) {
                    return std::nullopt;
                }
            }
        }
    }
    if (fill.stores.empty()) {
        return std::nullopt;
    }
    return fill;
}

// Whether two arrays are filled alike: the same stores, in the same blocks,
// at the same indices, of the same values.
bool same_fill(const Fill &a, const Fill &b) {
    if (a.stores.size() != b.stores.size()) {
        return false;
    }
    for (size_t i = 0; i < a.stores.size(); i++) {
        const auto &[block_a, index_a, value_a] = a.stores[i];
        const auto &[block_b, index_b, value_b] = b.stores[i];
        if (block_a != block_b || !same_source(*index_a, *index_b) ||
            !same_source(*value_a, *value_b)) {
            return false;
        }
    }
    return true;
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
    // Embree's traversal of a node whose children's boxes it holds, in the
    // one shape both its kernels share (bvh_traverser1.h, traverseClosestHit
    // for N = 4 and traverseClosestHitAVX512VL8 for N = 8), described above:
    // the hit lanes peeled one at a time, lowest first, each one's call read
    // at its lane and inserted among the ones peeled before it, the lanes
    // left tested after every peel, and a network over all the lanes past
    // four peels. The items scalars -- the child reference as the layout
    // stores it, the key's bits -- for every lane count.
    Peel,
};

LaneSortStrategy lane_sort_strategy(const Target &target) {
    // One strategy today, whatever the machine. The alternative a target
    // might take, were it measured better there, is Embree's eight-wide
    // representation of the same chain: the items kept in vector registers,
    // the peel one compress and a shuffle per lane, each child taken out by
    // a permute (permuteExtract) -- a permute per item where the scalar
    // items cost an indexed load each. On this machine the scalar items are
    // never behind for the counts the arms cover, at four lanes or eight.
    (void)target;
    return LaneSortStrategy::Peel;
}

// Sorts the run `call` in `block`, which is the lanes `lanes` describe, as
// described above; see there. Clears the run's keys, and moves the run into
// the arms of the chain, which `block` ends by entering afterwards; the
// reference `call` is spent.
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
    // bits flipped, so that the more negative is the smaller: Embree's
    // `distance_i`. The chain reads a key at a lane and compares it as this
    // integer; only the general arm packs the lane's index into it.
    //
    // The flip goes when the keys' sign bits are known clear in every value
    // (SSA/ConstantIntervals.h, `sign_clear`): such bits order as the
    // floats do already. Embree's `distance_i` is `asInt(tNear)` with no
    // flip at all, because its traversal clamps the ray's `tnear` to zero
    // at entry (TravRay) and a box's entry distance is `max(.., tnear)`; a
    // program that does the same -- with a clamp that leaves a non-negative
    // number, `select(tnear > 0, tnear, 0)`, since std::max passes a NaN or
    // a negative zero through -- is read the same way here, the fact
    // carried from the clamp through the integer max to the key.
    auto bits = emit(block, i32xn, Instruction::Op::Reinterpret, {keys});
    const ConstantInterval key_range = intervals.of(func, *block, lanes.keys);
    shared_ptr<Value> ordered = bits;
    if (!key_range.sign_clear) {
        auto sign = emit(block, i32xn, Instruction::Op::Shr,
                         {bits, bc(block, ci32(31), i32xn)});
        auto flip = emit(block, i32xn, Instruction::Op::BwAnd,
                         {sign, bc(block, ci32(0x7fffffff), i32xn)});
        ordered = emit(block, i32xn, Instruction::Op::Xor, {bits, flip});
    }

    // Where a peeled call's values are read from. A source the children's
    // loop filled -- the keys, a bound carried to the child -- is an array
    // of the function's own, held in a register once the loop is vectorized,
    // and a lane of a register at a run-time index goes through memory: the
    // chain stores the vector into a cell of its own once, at its first
    // comparison, and reads each lane with one indexed load (Embree's
    // `((unsigned int*)&tNear)[r]` off the tNear its node test spills --
    // spilled here on the nodes that hit more than one child, not on every
    // visit). Two sources the loop filled alike -- the key and the carried
    // bound both `distmin` of the child's box, as apps/rtq's are -- share
    // one cell and one load per lane, so the distance pushed is the key
    // itself, Embree's `dist`, and ordering the one orders the other. Any
    // other source -- the children, read off the node's row; a register
    // vector of some other making -- is read at the lane as an extract,
    // which the simplifier makes a scalar load off the row where the row is
    // read-only storage (SSA/Simplify.cpp, ExtractIdx).
    //
    // A cell holds its values as the integers of their width -- a float's
    // bits -- and the chain carries and exchanges them as such, handing a
    // float back to the call as a float: a 32-bit integer in a general
    // register is what the compare orders, what one `cmov` exchanges and
    // what the push stores, where a float would go through a vector
    // register and a mask for the same exchange (Embree's `unsigned int
    // d0`, the dist of its StackItem).
    struct Source {
        shared_ptr<Value> vec; // the lanes as a vector, read in `block`
        Type elem;             // what a lane is
        Type held;             // what a cell holds a lane as: elem's bits
        shared_ptr<Value> cell; // the array the vector is stored into, or null
    };
    vector<Source> sources; // the keys, then one per varying position
    {
        vector<std::optional<Fill>> fills;
        const auto add = [&](const shared_ptr<Value> &source,
                             const shared_ptr<Value> &vec, const Type &elem) {
            Source s{vec, elem, elem, nullptr};
            if (elem.is_float()) {
                s.held = Int_t::make(elem.bits());
            }
            std::optional<Fill> fill = fill_of(func, source);
            if (fill.has_value()) {
                for (size_t i = 0; i < sources.size(); i++) {
                    if (fills[i].has_value() && same_fill(*fills[i], *fill)) {
                        s.cell = sources[i].cell;
                        break;
                    }
                }
                if (!s.cell) {
                    s.cell = make_alloca(
                        func, func.blocks.front(),
                        Array_t::make(s.held, UIntImm::make(u32, uint64_t(n))));
                    std::get<shared_ptr<Instruction>>(s.cell->data)->scratch =
                        true;
                }
            }
            sources.push_back(std::move(s));
            fills.push_back(std::move(fill));
        };
        add(lanes.keys, keys, f32);
        for (size_t j = 0; j < values.size(); j++) {
            add(lanes.varying[j], values[j], lanes.varying_type[j]);
        }
    }
    // The cells stored, once, where the chain first compares.
    const auto store_cells = [&](const shared_ptr<Block> &into) {
        std::set<const Value *> stored;
        shared_ptr<Value> ramp;
        for (const Source &s : sources) {
            if (!s.cell || !stored.insert(s.cell.get()).second) {
                continue;
            }
            if (!ramp) {
                ramp = emit(into, Vector_t::make(u32, n), Instruction::Op::Ramp,
                            {cu32(0), cu32(1)});
            }
            shared_ptr<Value> held = s.vec;
            if (!equals(s.held, s.elem)) {
                held = emit(into, Vector_t::make(s.held, n),
                            Instruction::Op::Reinterpret, {s.vec});
            }
            // The lanes the mask has off hold zero: nothing reads them (the
            // chain reads the cell at the hits' lanes alone), and saying so
            // here is what lets the simplifier read the hits' values knowing
            // the mask holds -- a sort key's own test of the child's box,
            // `tNear <= min(slabs, tfar)`, is decided by the mask's `tNear <=
            // min(slabs, tfar, best)` (SSA/Simplify.cpp, implied_comparison),
            // and the infinity `distmin` blends in for a miss is never made.
            // One zero-masked move in a register and a whole store, which
            // the lane loads below forward from; a masked store would say the
            // same and forward to nothing (the loads behind it would wait for
            // it to retire), as measured.
            const Type held_t = Vector_t::make(s.held, n);
            held = emit(into, held_t, Instruction::Op::Select,
                        {mask, held,
                         bc(into, std::make_shared<Value>(Constant{s.held, uint64_t(0)}),
                            held_t)});
            auto at = emit(into, Vector_t::make(Ptr_t::make(s.held), n),
                           Instruction::Op::GEP, {s.cell, ramp});
            into->instrs.push_back(std::make_shared<Instruction>(
                Instruction::Op::Store, vector<shared_ptr<Value>>{at, held},
                into));
        }
    };
    // Source `i` at `lane`, as the cell holds it where it has one and the
    // cells are stored (one load per cell and lane), else out of the vector
    // as the lane is.
    std::map<std::pair<const Value *, const Value *>, shared_ptr<Value>> loaded;
    const auto read = [&](const shared_ptr<Block> &into, size_t i,
                          const shared_ptr<Value> &lane, bool cells) {
        const Source &s = sources[i];
        if (!cells || !s.cell) {
            return emit(into, s.elem, Instruction::Op::ExtractIdx, {s.vec, lane});
        }
        shared_ptr<Value> &v = loaded[{s.cell.get(), lane.get()}];
        if (!v) {
            auto at = emit(into, Ptr_t::make(s.held), Instruction::Op::GEP,
                           {s.cell, lane});
            v = emit(into, s.held, Instruction::Op::Load, {at});
        }
        return v;
    };
    // Whether `read` hands source `i` back as the cell holds it.
    const auto as_held = [&](size_t i, bool cells) {
        return cells && sources[i].cell && !equals(sources[i].held, sources[i].elem);
    };
    // A key read at a lane, as the integer that orders as the float does:
    // the vector form above, lane by lane.
    const auto key_of = [&](const shared_ptr<Block> &into,
                            const shared_ptr<Value> &key, bool cells) {
        shared_ptr<Value> bits = key;
        if (!as_held(0, cells)) {
            bits = emit(into, i32, Instruction::Op::Reinterpret, {key});
        }
        if (key_range.sign_clear) {
            return bits;
        }
        auto sign = emit(into, i32, Instruction::Op::Shr, {bits, ci32(31)});
        auto flip = emit(into, i32, Instruction::Op::BwAnd,
                         {sign, ci32(0x7fffffff)});
        return emit(into, i32, Instruction::Op::Xor, {bits, flip});
    };
    // Varying value `j` of a call as the call takes it: a float again where
    // the chain carried its bits.
    const auto handed = [&](const shared_ptr<Block> &into, size_t j,
                            const shared_ptr<Value> &v, bool cells) {
        if (!as_held(1 + j, cells)) {
            return v;
        }
        return emit(into, lanes.varying_type[j], Instruction::Op::Reinterpret,
                    {v});
    };

    // The chain (described above). `rest` is the mask with the lanes peeled
    // so far taken off, read as SSA/MaskLanes.h reads it, and `sorted` the
    // calls peeled so far, nearest first. An item is a call: the lane it
    // was peeled from, its key as the integer above, and its value at each
    // varying position.
    struct Item {
        shared_ptr<Value> lane;
        shared_ptr<Value> key;
        vector<shared_ptr<Value>> values;
    };
    MaskLanes walk(func, n, u32);
    // No hit: no call is made and the run is over, so straight to what
    // follows it, as a run whose every condition is false goes -- through
    // no arm, whose reads would run for nothing (one node test in five on
    // incoherent rays): Embree's `if (unlikely(mask == 0)) goto pop`, a
    // kortest.
    auto none = fresh_block(func, block->name + "!sort0");
    none->terminator.data = call.cont;
    func.blocks.push_back(none);
    auto some = emit(block, b, Instruction::Op::Any, {mask});
    auto first = fresh_block(func, block->name + "!peel1");
    func.blocks.push_back(first);
    // How many lanes are peeled before the general arm over all of them:
    // Embree's four, at either of its widths. A node of no more lanes than
    // that has an arm for every count and no general arm.
    constexpr uint32_t kPeeled = 4;
    const uint32_t peeled = std::min(n, kPeeled);
    shared_ptr<Block> peel = first;
    shared_ptr<Value> rest = walk.whole(peel, mask);
    vector<Item> sorted;
    for (uint32_t h = 1; h <= peeled; h++) {
        // The lowest lane left, and the call at it, read lane by lane
        // (Embree's `r = bscf(mask); node->child(r); tNear[r]`). The first
        // call is read as extracts, which is all its one-hit arm needs (its
        // child, off the row) before any cell is stored; where a second is
        // peeled the cells are stored and the first call's key and cell-held
        // values read again, off them, for the comparisons -- the first
        // read, dead by then where no arm but the one-hit one wanted it,
        // costs nothing.
        auto lane = walk.lowest(peel, rest);
        Item item;
        item.lane = lane;
        const bool cells = h >= 2;
        if (h == 2) {
            store_cells(peel);
            Item &first_call = sorted[0];
            first_call.key =
                key_of(peel, read(peel, 0, first_call.lane, true), true);
            for (size_t j = 0; j < values.size(); j++) {
                if (sources[1 + j].cell) {
                    first_call.values[j] =
                        read(peel, 1 + j, first_call.lane, true);
                }
            }
        }
        if (cells) {
            item.key = key_of(peel, read(peel, 0, lane, true), true);
        }
        for (size_t j = 0; j < values.size(); j++) {
            item.values.push_back(read(peel, 1 + j, lane, cells));
        }
        // Inserted among the calls peeled before it, nearest first: a
        // compare against each in turn, and the pair exchanged where the new
        // one is nearer, a select per component (Embree's dist_A0/B0, then
        // A1/B1/C1, then A2 to D2).
        for (Item &placed : sorted) {
            auto nearer = emit(peel, b, Instruction::Op::Lt,
                               {placed.key, item.key});
            const auto pick = [&](const shared_ptr<Value> &x,
                                  const shared_ptr<Value> &y) {
                return emit(peel, x->get_type(), Instruction::Op::Select,
                            {nearer, x, y});
            };
            // (An exchanged call's lane is whichever won, which nothing
            // after the first comparison asks; the lane stays unset.)
            Item lo, hi;
            lo.key = pick(placed.key, item.key);
            hi.key = pick(item.key, placed.key);
            for (size_t j = 0; j < values.size(); j++) {
                lo.values.push_back(pick(placed.values[j], item.values[j]));
                hi.values.push_back(pick(item.values[j], placed.values[j]));
            }
            placed = std::move(lo);
            item = std::move(hi);
        }
        sorted.push_back(std::move(item));

        // The arm for exactly h hits. One hit is a call, to the child
        // peeled, with no keys, no run and nothing for loopify to write a
        // stack from: a single call goes straight to the child (Embree's
        // one-hit case). More are a run of exactly h calls, the nearest
        // first and none conditional: h - 1 scalar pushes and a descent,
        // once loopify has written it.
        auto arm =
            fresh_block(func, block->name + "!sort" + std::to_string(h));
        func.blocks.push_back(arm);
        if (h == 1) {
            Terminator::Call single;
            single.call = call.call;
            single.cont = call.cont;
            single.drop = call.drop;
            for (size_t j = 0; j < values.size(); j++) {
                single.call.args[call.varying_at[j]] =
                    handed(arm, j, sorted[0].values[j], cells);
            }
            arm->terminator.data = std::move(single);
        } else {
            Terminator::MultiCall run;
            run.call = call.call;
            run.cont = call.cont;
            run.drop = call.drop;
            run.varying_at = call.varying_at;
            for (const Item &placed : sorted) {
                vector<shared_ptr<Value>> args;
                for (size_t j = 0; j < values.size(); j++) {
                    args.push_back(handed(arm, j, placed.values[j], cells));
                }
                run.varying.push_back(std::move(args));
            }
            arm->terminator.data = std::move(run);
        }
        if (h == n) {
            // Every lane peeled: none can be left, and no test is made.
            peel->terminator.data = Terminator::Jump{arm->name, {}};
            break;
        }
        // The lane cleared, and whether any is left: none is this arm, some
        // is the next peel -- or, past the last one, the general arm
        // (Embree's `if (likely(mask == 0))` after each `bscf`).
        rest = walk.without_lowest(peel, rest, lane);
        auto more = walk.any(peel, rest);
        auto next = fresh_block(
            func, block->name + (h < peeled ? "!peel" + std::to_string(h + 1)
                                            : string("!sortall")));
        func.blocks.push_back(next);
        peel->terminator.data = Terminator::Dispatch{
            more,
            {Terminator::Jump{arm->name, {}}, Terminator::Jump{next->name, {}}}};
        peel = next;
    }

    if (peeled < n) {
        // Every count past the peels: all the lanes at once (described
        // above). Each key with its lane's index in its low bits, so that
        // no two are equal and a sorted key says which lane it came from
        // (Embree's `distance_i`), and the misses infinitely far, whatever
        // their keys say, so that they sort first in descending order, out
        // of the hits' way: the key of +inf (its bits are already in order,
        // and its low bits clear) with the lane's index. Then the bitonic
        // network descending, the children and what travels with them taken
        // out by the sorted order with one variable permute each, and the
        // run in that order, the nearest first, call k made when `k <
        // hits`: the shape loopify reads as a SortedRun.
        const shared_ptr<Block> all = peel;
        auto hits = emit(all, u32, Instruction::Op::Popcount, {mask});
        auto lane = emit(all, i32xn, Instruction::Op::Ramp, {ci32(0), ci32(1)});
        auto high = emit(all, i32xn, Instruction::Op::BwAnd,
                         {ordered, bc(all, ci32(-int64_t(n)), i32xn)});
        auto packed = emit(all, i32xn, Instruction::Op::BwOr, {high, lane});
        auto inf_key =
            emit(all, i32xn, Instruction::Op::BwOr,
                 {bc(all, ci32(0x7f800000), i32xn), lane});
        shared_ptr<Value> network = emit(all, i32xn, Instruction::Op::Select,
                                         {mask, packed, inf_key});
        for (const Layer &comparators : descending_bitonic(n)) {
            network = layer(all, network, comparators);
        }
        auto idx = emit(all, i32xn, Instruction::Op::BwAnd,
                        {network, bc(all, ci32(int64_t(n) - 1), i32xn)});
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
    }
    // `block` enters the chain; `call` is spent either way.
    block->terminator.data = Terminator::Dispatch{
        some, {Terminator::Jump{none->name, {}}, Terminator::Jump{first->name, {}}}};
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
    internal_assert(strategy == LaneSortStrategy::Peel);
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
