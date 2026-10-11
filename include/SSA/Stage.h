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
// Built from what defer() already has. The compiler factors `f` at the call:
// the blocks from the call's continuation on are copied into a function
// `f!after` whose parameters are the call's value and the names those
// blocks use but do not define -- the values of `f` live after the call --
// and the call site becomes `r = g(...); return f!after(r, live...)`, a
// tail call. That tail call is then deferred onto `q` by defer(), which
// knows how to size the queue, store the arguments and build the drain.
//
// In the literature the split is the CPS conversion of that one call site --
// a selective CPS transformation (Nielsen, "A Selective CPS Transformation",
// MFPS 2001; the general form in Appel, "Compiling with Continuations", 1992,
// and Kennedy, "Compiling with Continuations, Continued", ICFP 2007), which
// is natural in this SSA because a block with arguments already is a
// continuation (Kelsey, "A Correspondence between Continuation Passing Style
// and Static Single Assignment Form", 1995). Naming the continuation as a
// function of its free variables is lambda lifting (Johnsson, "Lambda
// Lifting: Transforming Programs to Recursive Equations", FPCA 1985), and
// the queue entry it becomes is that closure defunctionalized (Reynolds
// 1972; Danvy and Nielsen 2001; see SSA/Defer.h). LLVM's coroutine splitting
// factors a function at a suspend point the same way.
//
// Only one call to `g` in `f`, not in tail position (a tail call is
// defer()'s case). Applied after the deferral of `f` itself where there is
// one: a self-call of `f` that defer() has already made a push is not a
// call, so the chain from the owner through `f` to `f!after` has no other
// recursion, which defer() requires.
//
// A deferral written after the stage -- `f.defer(h, q2)` where `f`'s call
// to `h` lies in the rest -- finds its call in `f!after`: the schedule's
// name for a function stands for it and for the continuations split off
// from it (call_holder below, which the conversion asks before every
// deferral). The order is the schedule's to choose because the drains run
// in the order their queues are deferred: pbrt's sample stage comes before
// its trace, and the trace's function is the one that defers the medium
// sampling (`vol_path_step.stage(vol_ray_samples, traced)` ahead of
// `vol_path_step.defer(vol_medium_sample, medium_samples)`,
// apps/pbrt/schedules/gpu-optix.bonsai).

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

// The cut on the same queue (docs/schedule-semantics.md, 4.5):
// `f.stage(g, q, l)` and `f.defer(g, q, l)`, where `q` is a queue whose
// drain already reaches `f` by calls each in tail position, and `l` a
// fresh name. pbrt's sample kernel: a pass over the ray queue that draws
// and stores, ahead of the trace kernel's pass over the same queue, with
// nothing of the ray copied anywhere.
//
// `f` is cut at its call to `g` as stage() or defer_continuation() cuts it
// -- after the call, so that the first half ends with `g`'s value in hand,
// or before it, so that the second half begins with the call -- and then,
// in place of a deferral of the tail call onto a new queue:
//
// - The drain of `q` is copied, and the copy, named `l`, runs before it in
//   the same round over the same count: the first pass. Its body is the
//   drain's as it was, so it runs the chain down to `f` and `f` up to the
//   cut, where the tail call has become stores: each value the rest takes
//   that the drain does not already hold -- the call's value, and a live
//   value computed in `f` before the cut -- is stored in an array of the
//   record, one slot per producer iteration like a reducer's
//   (`<q>_<name>_slots`), at the entry's slot, which every function of the
//   chain already carries (`_slot_<q>`, SSA/Defer.cpp). The chain carries
//   a pointer to each such slot down to `f` as it carries the queue
//   (`_<name>_<q>`). pbrt indexes its samples array by the pixel, which is
//   the same index.
// - The drain itself keeps its name, bounds and launch facts, and becomes
//   the second pass: its body calls the rest (`f!after`, or `f!g!k`) with
//   the values it handed the chain -- the entry's fields, the uniforms, the
//   slot addresses, traced through the chain's tail calls back to the
//   drain's own call -- and with each stored value loaded back from its
//   array at the slot.
//
// What is stored is decided as 4.2 decides an entry's fields, less the
// recomputation of pure functions, which is not done here yet. Refused with
// an error: a queue with no drain yet (the directive comes after the
// deferrals that made it), one split by a specialize or whose drain is
// already bound or split (it comes before the directives on the loop), a
// chain whose calls are not in tail position or whose functions are called
// from outside it (the owner's own call of the chain has to be a push, or
// it would run the first half alone), entries that carry no record slot, a
// cut that not every path through the chain reaches (the skip bit of 4.5 is
// not built), and an address of `f`'s own frame live after the cut.
std::vector<Type> stage_same_queue(FuncMap &funcs, const std::string &func,
                                   const std::string &callee,
                                   const QueueSpec &queue,
                                   const std::string &loop);
std::vector<Type> defer_same_queue(FuncMap &funcs, const std::string &func,
                                   const std::string &callee,
                                   const QueueSpec &queue,
                                   const std::string &loop);

// `f.defer(g, q)` at a call that is not in tail position and not spawned:
// the call *and its continuation* are deferred. Where stage() runs the call
// where it is and queues its value with the rest, this queues the call's
// arguments with the rest's live values, and the drain makes the call and
// then runs the rest -- pbrt's medium-sample queue, whose kernel walks the
// medium the queued ray is in and then goes on to the surface. Built as the
// tail call it is: the call and the rest become one function,
// `f!g!k(args..., live...) { r = g(args); return f!after(r, live) }`, the
// site tail-calls it, and defer() queues that. The complete continuation is
// captured, as defer() requires -- `f`'s own rest here, and the callers'
// by the same rule as any deferral: a caller between the site and the
// queue's owner may not go on after its call into the chain.
//
// Both forms are wanted. This one queues work whose routing is known before
// the call (the medium of the ray); stage() queues work routed by the call's
// result (the material of the hit), which is pbrt's boundary after the
// trace. One call to `g` in `f`; the key of a `q.specialize(x)` is a
// parameter of `f!g!k`.
std::vector<Type> defer_continuation(FuncMap &funcs, const std::string &func,
                                     const std::string &callee,
                                     const QueueSpec &queue);

// The function among `func` and the continuations split off from it
// (`func!after`, `func!after!after`, ...: a stage's rest, or a deferral's
// with its continuation) that makes a call to `callee`; `func` itself when
// it does, or when none of them does, so that the caller's own refusal
// names what the schedule wrote.
std::string call_holder(const FuncMap &funcs, const std::string &func,
                        const std::string &callee);

// Every function standing for `func` that makes a call to `callee` and is
// reached from `owner` by calls: `func` itself, the continuations split off
// from it (`func!after`) and the copies a queue's specialize made of it
// (`func!Diffuse`, `func!Basic!Diffuse`: SSA/Defer.cpp names a sub-queue's
// callee by its variant path). The schedule's name for a function stands
// for all of them -- `vol_surface.defer(vol_subsurface, bssrdfs)` written
// after `hits.specialize(material)` has to turn the call into a push in
// each material's copy, the original being unreached once its copies took
// its place. Empty when none of them makes the call: the caller then names
// `func` and refuses in its own words.
std::vector<std::string> call_holders(const FuncMap &funcs, const std::string &func,
                                      const std::string &callee,
                                      const std::string &owner);

// The copies of `callee` (`callee!Basic`, `callee!Universal`: the ones a
// queue's specialize made of it because `func` hands it the key) that the
// functions standing for `func` call where `owner` reaches them, each once,
// in name order. What a deferral names when call_holders finds nothing: the
// call it was written on is there, but on copies of its callee, and a
// deferral's callee is one function.
std::vector<std::string> copies_called(const FuncMap &funcs, const std::string &func,
                                       const std::string &callee,
                                       const std::string &owner);

// Whether `func`'s one call to `callee` is neither in tail position nor
// spawned, and `callee` is not a recursion back into `func`: what makes
// `func.defer(callee, q)` a defer_continuation rather than a defer. A
// recursion that goes on after its call, or several calls, is defer()'s
// to refuse, in its own words.
bool has_nontail_call(const FuncMap &funcs, const std::string &func,
                      const std::string &callee);

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
