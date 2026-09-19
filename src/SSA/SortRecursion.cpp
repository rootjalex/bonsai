#include "SSA/SortRecursion.h"

#include "Error.h"
#include "IR/Equality.h"
#include "Utils.h"

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
                      vector<vector<shared_ptr<Value>>> &varying, size_t i,
                      size_t l, bool ascending) {
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
                  vector<vector<shared_ptr<Value>>> &varying) {
    const size_t count = keys.size();
    const size_t n = next_pow2(count);
    for (size_t k = 2; k <= n; k *= 2) {
        for (size_t j = k / 2; j > 0; j /= 2) {
            for (size_t i = 0; i < n; i++) {
                const size_t l = i ^ j;
                if (l > i && l < count) {
                    compare_and_swap(func, block, keys, varying, i, l,
                                     (i & k) == 0);
                }
            }
        }
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
        if (call->varying.size() > 1) {
            sort_network(func, block, call->keys, call->varying);
            sorted++;
        }
        call->keys.clear();
    }
    return sorted;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
