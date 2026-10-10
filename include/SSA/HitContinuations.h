// A raygen's continuation after its nearest-hit trace becomes the
// closest-hit program of that launch.
//
// pbrt's trace kernel has one shape (gpu/optix/optix.cu):
// `__raygen__findClosest` reads the ray, traces, and on a miss pushes the
// escaped ray; `__closesthit__triangle` re-reads the ray's work item at
// `optixGetLaunchIndex`, builds the hit and pushes it onto the material
// queue. The hit's consumer is the hit program, and the raygen holds the
// miss path alone. Our RT-core lowering (Lower/Trees.cpp) made the trace
// a function call, `_rt_traceN(ray) -> option[hit]`, whose caller goes on
// with the answer -- the one shape in which the hit's consumer is
// necessarily the raygen -- and measured against pbrt's that is the whole
// deficit of the surface scenes: a raygen's code after `optixTrace` runs
// once the warp's traversals are all back, every memory latency in it
// exposed (53% of head's camera launch stalled on the first use of memory
// loaded after the trace), where a hit program's run is scheduled as each
// ray completes and its loads overlap the other rays' traversal
// (apps/pbrt/PLAN.md, 2026-10-08).
//
// This pass gives the trace pbrt's shape, with the machinery stage() has:
// `f.stage(g, q)` cuts `f` at its call to `g` and runs the rest as the
// queue's drain (SSA/Stage.h; Nielsen's selective CPS transformation, the
// continuation named as a function of its free variables), and the
// hardware binding of a query is the same cut with the hit program as the
// "drain". For a loop bound to OptixThread -- a raygen over a drained
// queue -- whose body reaches, through a chain of calls each in tail
// position, one call to a nearest-hit query's trace function (held by
// Lower/Trees.cpp so that the frontend inliner left it a call):
//
// - The closest-hit program is the loop body again, specialized on a hit:
//   a copy of the body's region and of the call chain down to the trace,
//   with the call of the trace function redirected to the function's
//   hit-side variant, `_rt_traceN!hit` (and `!hit$boxes` over the
//   custom-primitive inputs), which Lower/Trees.cpp made beside it: the
//   same function once the hardware reports a hit -- the context's `hit`
//   word set for the raygen, the elements bound from the build input's
//   record and the primitive index, the witness from the hardware. The
//   copy takes the body's own parameters -- its index and its captures --
//   which CodeGen_OptiX supplies from the launch parameters (the index as
//   `_begin + optixGetLaunchIndex * _stride`, the raygen's own), so the
//   program re-reads the entry the raygen read and recomputes what came
//   before the trace, as pbrt's does; the continuation's loads and pushes
//   then run inside the program.
// - The raygen keeps the miss path: in its own chain the trace call's
//   continuation becomes `if hit { return } else { the continuation }`,
//   which the simplify after specializes on the miss.
//
// Sound when what the program recomputes could not have changed: the
// region from the body's entry to the trace call, through the chain, has
// no effect but loads and writes to its own locals (a store elsewhere, an
// atomic, a push, a call with an effect refuse the split), so every load
// in it reads memory as it stood when the launch began -- the drained
// queue, which its drain never writes, the scene, the lights -- and the
// program reads the same; and the continuation reaches no further trace
// and no loop back to one (the probe walk and the transmittance walk
// trace in a loop and keep the raygen continuation, as pbrt's
// `__raygen__randomHit` and `__raygen__shadow_Tr` do). A loop that does
// not qualify keeps the old shape; BONSAI_EXPLAIN_HIT_CONTINUATIONS=1
// prints each refusal.
//
// Runs in SSA/Convert.cpp after gate_drains (the drains are final) and
// before the pushes are lowered and the functions simplified. The copies
// of the chain carry Attribute::device_only, since they read the hit from
// the hardware and no host compiles them; the program carries
// OptixProgram::for_loop and ::symbol, by which CodeGen_OptiX puts it in
// the loop's module in the generic program's place.
#pragma once

#include "SSA/Rewrite.h"

namespace bonsai {
namespace ir {
namespace ssa {

void split_hit_continuations(FuncMap &funcs);

} // namespace ssa
} // namespace ir
} // namespace bonsai
