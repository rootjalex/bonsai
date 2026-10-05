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

// The round loop's guard and the inner queues' header fills, for GPU drains.
//
// A deferral whose callee is on its own chain drains in rounds: a loop whose
// header reads the count of the queue drained this round and stops when it
// is zero (SSA/Defer.cpp). On the GPU that read was lowered by the generic
// host-touches-device path; this pass makes the loop what it is --
// `do { launch the round } while (bonsai_cuda_round_guard(round, &count))`,
// one four-byte synchronous read per round, the honest lowering: the loop
// stops exactly where the recursion did (the user's ruling, 2026-10-05; a
// pipelined ring that tested a snapshot `depth` rounds stale and overran
// the end by that depth was built, measured, and taken back out). The read
// waits for the round's kernels, which is the price of asking; if that
// boundary ever costs, the answer is a trip count the compiler can prove --
// pbrt runs a fixed maxDepth+1 rounds and never asks -- not a stale answer.
//
// The pass also hoists the inner queues' header fills: a queue made inside
// another queue's rounds -- the material and shadow queues, inside the ray
// queue's rounds -- had its whole header (count zero and the arrays, the
// arrays never changing) restaged through the staging ring every round,
// where Defer empties a queue before its producers; on the renderer that
// was a dozen staged copies and their ring events a round, a third of the
// launch gap (apps/pbrt/PLAN.md). The fill moves to the round loop's
// preheader, and nothing of it stays in the loop: the count is already
// reset once a round after the drain. Defer cannot place it there itself (a
// queue is built while the graph is not yet walkable past its own
// unterminated blocks); this pass sees the finished loop.
//
// The rewrite fires only where it is sound: the header's test must be
// `count == 0` of the queue of a GPU-bound drain in the loop, and every
// queue drain in that loop must be GPU-bound -- a host drain reads its own
// trip count back anyway. `bonsai_cuda_round_guard(round, count_address) ->
// u32` is declared into the program's foreign functions
// (runtime/bonsai_cuda.h implements it; the JIT pins it by address).
void rewrite_round_guards(FuncMap &funcs, ir::Program &program);

} // namespace ssa
} // namespace ir
} // namespace bonsai
