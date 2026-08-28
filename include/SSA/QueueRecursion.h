#pragma once

#include "SSA/SSA.h"

#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Turns a branching recursion into a loop over an explicit stack, so that a
// traversal costs a bounded amount of stack instead of one frame per node.
//
// This is what `loopify(N)` asks for. Tail recursion becomes a loop by itself
// (see loopify in SSA/Rewrite.h): the call is the last thing the function
// does, so going round again *is* making it. A traversal that recurses twice
// -- once per child -- cannot be turned into a loop that way, because the
// second call still has to happen after the first one comes back. What can be
// done instead is to stop making the calls at all and write down what they
// would have been:
//
//     visit(node):                     count = 1; stack[0] = node
//       body(node)                     while count != 0:
//       visit(left)          ==>         count -= 1; n = stack[count]
//       visit(right)                     body(n)
//                                        stack[count] = left;  count += 1
//                                        stack[count] = right; count += 1
//
// which visits the same nodes, in a different order -- the stack is LIFO, so
// the last child pushed is the first one visited.
//
// The recursion has to be *tail-modulo-recursion* for this to be sound: after
// a recursive call returns, the only thing left to do is make more recursive
// calls and return. Anything else -- work that depends on a call having
// finished -- has no place to happen once the call is only a note on a stack,
// and is rejected rather than reordered. A traversal that accumulates into a
// pointer its callers passed in, which is what tree queries lower to, meets
// this: the accumulation happens before the children are visited.
//
// `size` is the stack's depth, which the schedule gives and which nothing
// here checks against the tree: overflowing it is the program's problem, the
// same as it is for the recursion this replaces.
void queue_recursion(Function &func, size_t size);

} // namespace ssa
} // namespace ir
} // namespace bonsai
