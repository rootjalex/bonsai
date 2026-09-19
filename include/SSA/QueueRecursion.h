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
//     visit(node):                     n = node; count = 0; live = true
//       body(node)                     while live:
//       visit(left)          ==>         body(n)
//       visit(right)                     if the calls are the last thing:
//                                          stack[count] = right; count += 1
//                                          n = left
//                                        else if count == 0: live = false
//                                        else: count -= 1; n = stack[count]
//
// which visits the same nodes in the same order: the first call is made at
// once, and the rest wait on the stack, which is LIFO, so the last one pushed
// is the first one taken. The first child never touches the stack, which is
// pbrt's loop exactly -- the near child is the next node and the far one is
// pushed -- and saves a push and a pop at every node with children.
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
//
// The function may already be a gang's -- vectorize() applied first, so that
// the recursion was specialized for one node the lanes share and a mask of
// the lanes that reached it. Then what varies from call to call is that node
// and that mask, and those are what go on the stack: one stack of node
// references and one of masks, the packet traversal of Wald et al. 2001. The
// pushes are made only if some lane is on, as the calls they replace were
// only made then: a node no ray in the packet reached is neither descended
// into nor pushed. A run's votes on its order (Instruction::Op::Vote) are
// dropped here either way, since a run that is pushes on a stack is visited
// entry by entry; and a gang's accumulator is one bool per lane, so `any` is
// settled only when every lane's is.
void queue_recursion(Function &func, size_t size);

} // namespace ssa
} // namespace ir
} // namespace bonsai
