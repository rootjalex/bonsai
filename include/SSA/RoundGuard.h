#pragma once

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
struct Program;
} // namespace ir
} // namespace bonsai

namespace bonsai {
namespace ir {
namespace ssa {

// The round loop's guard, pipelined for GPU drains.
//
// A deferral whose callee is on its own chain drains in rounds: a loop whose
// header reads the count of the queue drained this round and stops when it is
// zero (SSA/Defer.cpp). When the drains are on the GPU that read is the one
// thing the host waits for -- every launch is asynchronous, the count's
// zeroing is a memset on the stream, and the launch covers the capacity with
// the count read on the device (Bind.cpp) -- so the stream drains once a
// round for a four-byte copy, and on a scene whose waves run hundreds of
// rounds the waits and the reissue cost as much as a third of the render
// (apps/pbrt/PLAN.md, the launch gates section: transparent-machines, 1.86 s
// of a 4.6 s render between the kernels).
//
// pbrt never reads a count back: its wavefront runs a fixed maxDepth+1
// rounds and takes the empty launches (one wait per band, Rendering the
// Moana island scene on a GPU: wavefront rendering's shape). This pass keeps
// the dynamic termination and moves the read off the critical path instead:
// the header's test becomes a call to the runtime's bonsai_cuda_round_guard,
// which each round enqueues an asynchronous snapshot of the count into a
// ring of pinned slots and tests the snapshot taken `depth` rounds ago,
// whose copy is long done. The host runs at most depth+1 rounds ahead and
// never drains the stream. Testing late is exact: a round with no entries
// pushes nothing, so a count once zero stays zero, and the rounds run past
// the true end launch kernels whose device-side guard sees zero and returns
// -- at most `depth` empty rounds per entry of the loop, where pbrt runs
// maxDepth+1 whatever the depth.
//
// The rewrite fires only where it is sound and buys something: the header's
// test must be `count == 0` of the queue of a GPU-bound drain in the loop,
// and every queue drain in that loop must be GPU-bound -- a host drain reads
// its own trip count back anyway, so the loop would wait regardless.
// `bonsai_cuda_round_guard(round, count_address) -> u32` is declared into
// the program's foreign functions (runtime/bonsai_cuda.h implements it; the
// JIT pins it by address).
void pipeline_round_guards(FuncMap &funcs, ir::Program &program);

} // namespace ssa
} // namespace ir
} // namespace bonsai
