#pragma once

#include "IR/Schedule.h"
#include "SSA/SSA.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

using FuncMap = std::map<std::string, std::shared_ptr<ssa::Function>>;

// Splits the parfor `idx` of `func` into an outer parfor `outer` that steps
// through the range by `factor` and an inner parfor `inner` over a chunk, the
// body seeing `outer + inner` as its index. With `exact` the range is taken
// to be a whole number of chunks (checked where it is constant); without it
// the body is guarded by `outer + inner < end` -- the GuardWithIf tail of
// Halide's split -- so a range of any length runs exactly its own iterations,
// and when the inner loop is then vectorized the guard is the gang's mask,
// the last gang running partly full. Implemented in SSA/Rewrite.cpp.
void split(FuncMap &funcs, std::string func, std::string idx, int factor,
           std::string outer, std::string inner, bool exact);

// The same with a chunk the program sizes: `factor` is an expression over
// values of `func` -- parameters, or instructions in blocks that dominate the
// loop, named as the program names them -- and integer constants, one value
// for the whole loop, and the chunk is that many iterations. pbrt tiles a
// pass of its wavefront as a band of scanlines sized from the resolution
// (`maxQueueSize`, wavefront/integrator.cpp), so the split that makes our
// pass a band names the value the program computes the way pbrt does. The
// expression is emitted in the loop's block, its values threaded there as
// any value is; the stride has to be one, since whether a run-time chunk is
// a whole number of strides cannot be checked; and without a tail the range
// is the caller's assertion, as a run-time range already is. A constant
// expression is the overload above.
void split(FuncMap &funcs, std::string func, std::string idx, const Expr &factor,
           std::string outer, std::string inner, bool exact);

// Makes the parfor `idx` of `func` a parfor `worker` over [0, count) of
// persistent workers, each claiming iterations of `idx` from a counter set to
// the loop's start before it -- `i = atomicAdd(counter, stride); while (i <
// end) { body(i); i = atomicAdd(counter, stride) }` -- the persistent threads
// of Aila and Laine (HPG 2009) and the self-scheduling of Tang and Yew (ICPP
// 1986); see ir::Persistent. `count` is an expression over constants and
// values of `func`, as split's run-time chunk is. The loop must not be bound
// and must not be a queue's drain. Implemented in SSA/Rewrite.cpp.
void persistent(FuncMap &funcs, std::string func, std::string idx,
                const Expr &count, std::string worker);

// Records that the parfor `index` runs on `resource`, and checks that this
// agrees with whatever the loops around it are already bound to. Nothing about
// the graph changes: a bind is a tag, and code generation is where it becomes
// a launch. Implemented in SSA/Bind.cpp.
void bind(FuncMap &funcs, std::string func, std::string index,
          Resource resource);

// May a loop bound to `inner` sit inside one bound to `outer`?
//
// The orderings a schedule has to respect. A GPUThread runs inside a GPUBlock,
// never the other way round, and the two kinds of machine do not nest inside
// each other at all. Anything not named here is allowed: the point is to catch
// a schedule that asks for something no hardware does, not to enumerate every
// pairing that happens to be sensible. What bind() checks when a loop is
// bound, and reorder() again when two bound loops change places.
bool may_nest(Resource outer, Resource inner);

// reorder(), the interchange of two nested parfors, is declared in
// SSA/ReorderLoops.h.

// Fuses two nested parfor loops into one that walks the rectangle they cover,
// recovering each index from the step number. `outer` must run `inner` and
// nothing else. Where the ranges do not divide by their strides the rectangle
// is larger than they are, and the steps past either end are skipped.
// Implemented in SSA/CollapseLoops.cpp.
void collapse(FuncMap &funcs, std::string func, std::string outer,
              std::string inner, std::string collapsed);

// Turns `func`'s recursion into a loop: every call of itself in tail
// position becomes a jump back to its entry. A recursion that runs through
// other functions first -- `f` calls `g`, which calls `f` back, a path step
// that hands its next bounce to the routing of its hit, which hands it back
// -- is made direct first: each callee on a path back to `func` that the
// schedule has not held as a function of its own (`held`, `[[noinline]]`)
// is copied into `func` at its call (inline_call), until every call back
// is a call of `func` itself. What a loop needs on hardware that allows no
// recursion (the RT cores' programs), and what pbrt's megakernel is.
// `size` puts a branching recursion on a stack instead (SSA/QueueRecursion.h).
void loopify(FuncMap &funcs, std::string func, int size = 0);

// Copies the callee of the call that ends `site`, a block of `caller`, into
// `caller` in place of the call: the site jumps to a copy of the callee's
// entry with the call's arguments, and each return of the copy jumps to the
// call's continuation with the value the call kept, the continuation's own
// arguments threaded through the copy under fresh names. The copy's blocks
// are the callee's with a suffix; the callee itself is left as it was for
// its other callers. Implemented in SSA/Rewrite.cpp.
void inline_call(FuncMap &funcs, const std::shared_ptr<Function> &caller,
                 const std::shared_ptr<Block> &site);

// defer(), the other way of running a recursion's pending calls, is declared
// in SSA/Defer.h.

// Vectorizes the parfor loop `idx` into a single SIMD gang, in the manner of
// Pharr & Mark, "ispc: A SPMD Compiler for High-Performance CPU Programming"
// (2012), with control flow handled by the strictly stronger partial
// linearization of Moll & Hack (PLDI 2018) rather than ispc's full
// if-conversion. Implemented in SSA/Vectorize.cpp.
//
// The gang width is the loop's extent, which must be constant: to vectorize a
// wider loop, split() it to the gang width first and vectorize the inner
// loop. The loop then runs exactly once and disappears.
//
// `policies` is what the schedule said about the tests the gang's control
// flow gets (ir::BranchPolicy), for `func` and for every function the gang
// is specialized into, each looked up by the name the schedule knows it by.
void vectorize(FuncMap &funcs, std::string func, std::string idx,
               const ir::BranchPolicyMap &policies = {});

} // namespace ssa
} // namespace ir
} // namespace bonsai
