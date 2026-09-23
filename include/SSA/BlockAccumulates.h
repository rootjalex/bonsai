#pragma once

// One accumulate per block where every thread of the block made one.
//
// The film under the GPU schedule: `atomic radiance_out[p] += weight * L`
// with `render.bind(p, GPUBlock)` and `render.bind(s, GPUThread)`, where the
// address names the block's pixel and not the thread's sample. Every thread
// of the block adds to the same word, which is N atomics serialized at L2 --
// N the block, sixty-four and more -- and, before the hardware atomic was
// selected, a compare-and-swap loop each (PLAN.md, "What the megakernel's
// profile said"). This pass makes it one: each thread accumulates into a
// slot of its own for the length of its body, and where the body ends the
// block reduces the slots -- a warp shuffle tree, one shared-memory slot per
// warp, the warps' sums added by every thread after a barrier, which is the
// `block_reduce_*` intrinsic CodeGen_PTX lowers -- and the thread with the
// loop's first index makes the one accumulate to memory. That accumulate is
// atomic if another block may write the address too, and plain when the
// contention analysis (SSA/Contention.h) shows the address is the block's
// own, as a pixel indexed by the block loop's index is.
//
// What qualifies is an atomic accumulate whose address is *uniform over the
// thread loop's body*: computed from constants, values from before the body,
// and the body's uniform arguments -- everything but the thread index and
// what merges inside the body -- by pure arithmetic and field addressing, so
// that every thread of the block names the same place, and the leader can
// compute it again where the body ends. The accumulate may run under a
// condition or several times: its slot starts at the operation's identity,
// so what the slot holds at the end is exactly what the thread would have
// added. The reassociation -- the block's values summed in a tree rather
// than in arrival order -- is the one the scheduling language's guarantees
// allow (Halide's: a schedule may sum in any order).
//
// The block reduction is the standard one: Harris, "Optimizing Parallel
// Reduction in CUDA" (NVIDIA, 2007), with warp shuffles in place of the
// shared-memory tree's last five levels (Luitjens, "Faster Parallel
// Reductions on Kepler", NVIDIA Developer Blog, 2014). Privatizing a
// reduction variable per thread and combining at the end is the reduction
// recognition of parallelizing compilers (Allen & Kennedy, "Optimizing
// Compilers for Modern Architectures", ch. 6).
//
// Runs on the SSA after the schedule's rewrites and the demotion of atomics
// (SSA/DemoteAtomics.h), before the allocas are promoted, so the slots it
// makes become ordinary values.

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

struct ReduceBlockAccumulates {
    static void run(Function &f);
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
