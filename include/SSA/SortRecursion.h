#ifndef BONSAI_SSA_SORT_RECURSION_H
#define BONSAI_SSA_SORT_RECURSION_H

#include "SSA/SSA.h"

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
// Returns the number of runs it reordered, which is zero for a function whose
// recursion no schedule sorted.
size_t sort_recursion(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai

#endif // BONSAI_SSA_SORT_RECURSION_H
