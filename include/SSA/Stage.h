#pragma once

// stage(): a stage boundary after a call, as a queue.
//
// `f.stage(g, q)`. Where `f.defer(g, q)` queues a *call* -- the arguments go
// on the queue and the drain makes the call -- this queues a *return*: the
// call to `g` is made where it is in `f`, its value and everything `f` still
// needs after it go on the queue, and `q`'s drain runs the rest of `f` from
// there. It is pbrt's boundary after `IntersectClosest`: the trace has run,
// and what the material kernel is handed is the hit with the path's state
// (`MaterialEvalWorkItem`). `vol_path_step.stage(trace, hits)` beside
// `vol_path_step.defer(vol_path_step, rays)` is pbrt's bounce -- the `rays`
// drain traces and pushes hits, the `hits` drain shades and pushes the next
// rays -- with each stage a loop of its own for a schedule to bind,
// vectorize or, on the GPU, make a kernel of.
//
// Built from what defer() already has. The compiler factors `f` at the call,
// as LLVM's coroutine splitting factors a function at a suspend point: the
// blocks from the call's continuation on are copied into a function
// `f!after` whose parameters are the call's value and the names those
// blocks use but do not define -- the values of `f` live after the call --
// and the call site becomes `r = g(...); return f!after(r, live...)`, a
// tail call. That tail call is then deferred onto `q` by defer(), which
// knows how to size the queue, store the arguments and build the drain.
//
// Only one call to `g` in `f`, not in tail position (a tail call is
// defer()'s case). Applied after the deferral of `f` itself where there is
// one: a self-call of `f` that defer() has already made a push is not a
// call, so the chain from the owner through `f` to `f!after` has no other
// recursion, which defer() requires.

#include "SSA/Defer.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// Returns the struct types made for the program to declare, as defer() does.
std::vector<Type> stage(FuncMap &funcs, const std::string &func,
                        const std::string &callee, const QueueSpec &queue);

// Whether every call to `callee` in `func` lies after `func`'s one call to
// `staged` -- in the blocks that call's continuation reaches -- so that
// `func.stage(staged, q)` moves them all into `func!after`. What a deferral
// of `callee`'s recursion asks before the stage is applied: with the
// recursive calls in the rest, the drain of the recursion's queue never
// pushes onto it, and the queue needs one buffer (QueueSpec::
// drain_pushes_self). False when `func` calls `staged` never or more than
// once, or `callee` never.
bool calls_after(const Function &func, const std::string &staged,
                 const std::string &callee);

} // namespace ssa
} // namespace ir
} // namespace bonsai
