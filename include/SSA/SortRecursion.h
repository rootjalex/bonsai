#ifndef BONSAI_SSA_SORT_RECURSION_H
#define BONSAI_SSA_SORT_RECURSION_H

#include "IR/Target.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <optional>

namespace bonsai {
namespace ir {
namespace ssa {

// Puts the calls of every sorted MultiCall in `func` into ascending order of
// their keys, and clears the keys.
//
// This is what `sort()` finally does. The schedule supplies a key per branch of
// a `from`; Lower/Sorts.cpp works out what those keys are and hangs them on the
// recursion; and this rewrites the run so that the branch with the smallest key
// is called first.
//
// It runs here rather than at the Stmt level for one reason: by this point a
// branch is whatever the layout made of it -- for a BVH, a node index -- so
// reordering is a couple of selects between integers. At the Stmt level a
// branch is still a subtree value, and selecting between two of those means
// materialising both, which is a load of the node the traversal was about to
// skip.
//
// It must run before loopify puts the recursion on a stack (see
// SSA/QueueRecursion.h), because that replaces the run with pushes and there is
// then nothing left to permute.
//
// Each comparison of the network is wrapped in a Vote (see Instruction::Op),
// since the run is made in one order by whoever makes it: for a scalar
// traversal that is the comparison, and for a gang it is the lanes' majority,
// which is what keeps the children a gang descends into uniform when its rays
// would each have ordered them differently.
//
// A run whose conditions, keys and children are the lanes of vectors -- the
// run a node that holds its children's boxes makes, from the loop over them
// -- is sorted as those vectors, over the children that are hit alone (see
// sort_lanes in SortRecursion.cpp): a switch on the count of hits, whose
// arms for one to four hits are plain runs of that many calls, nearest
// first, and whose arm for more is left in the shape `sorted_run` below
// reads back.
//
// Returns the number of runs it reordered, which is zero for a function whose
// recursion no schedule sorted. `target` is the machine the code is for,
// which decides how a run that is the lanes of vectors is ordered and
// handed to loopify (see lane_sort_strategy in SortRecursion.cpp).
size_t sort_recursion(Function &func, const Target &target);

// A run sort_recursion() ordered as vectors, read off the shape it left: call
// k's varying values are lane `lanes - 1 - k` of one vector per varying
// parameter -- the nearest child in the last lane, the ones behind it in the
// lanes below, the children not hit in the lowest lanes -- and its condition
// is `k < hits`. The shape of the arm for five hits and up of a node wider
// than four, where the count is not a constant; what loopify reads to write
// the waiting children to its stack with one compacting store rather than
// a conditional push each (see SSA/QueueRecursion.h). Nothing for a run of
// any other shape.
struct SortedRun {
    // The sorted vector of each varying parameter, by the parameter's index.
    std::map<size_t, std::shared_ptr<Value>> values;
    std::shared_ptr<Value> hits; // how many calls are made, a u32
    uint32_t lanes = 0;
};
std::optional<SortedRun> sorted_run(const Terminator::MultiCall &call);

// A run whose conditions and varying values are the lanes of one source
// each -- the run a node that holds its children's boxes makes, before any
// sort or without one: call k is made under lane k of `mask` on lane k of
// each of `values`. A source is a vector of `lanes` lanes or an array of as
// many, read whole by lanes_as_vector(). What loopify reads to write the
// children hit with one compacting store (SSA/QueueRecursion.h); nothing for
// a run of any other shape, or for one a sort left as a SortedRun.
struct LaneRun {
    std::shared_ptr<Value> mask;
    // The source of each varying parameter's lanes, by the parameter's index.
    std::map<size_t, std::shared_ptr<Value>> values;
    uint32_t lanes = 0;
};
std::optional<LaneRun> lane_run(const Terminator::MultiCall &call);

// `source`, a vector of `lanes` lanes or an array of as many, as the vector
// of its lanes: itself, or one read of the array, appended to `block`.
std::shared_ptr<Value> lanes_as_vector(Function &func,
                                       const std::shared_ptr<Block> &block,
                                       const std::shared_ptr<Value> &source,
                                       uint32_t lanes);

} // namespace ssa
} // namespace ir
} // namespace bonsai

#endif // BONSAI_SSA_SORT_RECURSION_H
