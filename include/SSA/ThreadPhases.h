#pragma once

#include "SSA/SSA.h"

#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// The thread loops of a block loop's body, and whether each is the whole
// block.
//
// A loop bound to GPUBlock runs one block per iteration, and the loops
// bound to GPUThread inside its body are the block's threads. A launch has
// one block size, so the block is the largest of their extents, and each
// thread loop is a phase of the kernel: every thread of the block runs the
// phase's body under `tid < extent`, with a barrier between phases, and the
// block-level code between them runs on every thread with its effects the
// leader's (CodeGen_PTX). This is Halide's FuseGPUThreadLoops -- the thread
// loops of a block fused into one loop of the largest extent, each body
// predicated by its own, barriers injected between -- as the device
// generator emits it block by block.
//
// A phase whose extent is every sibling's is the whole block. It runs with
// no predicate, and the block's collectives -- a block reduction
// (SSA/BlockAccumulates.h), which every thread of the block has to reach
// -- are made only in such a phase. Whether two extents are one is decided
// by definition (SSA/Definitions.h), not by syntax: the same constant, or
// bounds that resolve to the same values, as the loop whose trip count sized
// a queue and the drain over that queue's capacity do. The block reduction
// pass and the PTX backend both read it here, so the two agree about which
// phase may hold a collective.
struct ThreadPhase {
    const Terminator::ParFor *loop = nullptr;
    // The block whose terminator the loop is.
    std::string block;
    bool whole_block = true;
};

// The thread loops in the body of `block_loop`, a parfor of `func` bound to
// GPUBlock, in block-name order.
std::vector<ThreadPhase> thread_phases(const Function &func,
                                       const Terminator::ParFor &block_loop);

// Whether `loop`, a parfor of `func` bound to GPUThread and the terminator of
// `block`, is the whole of its block: the only thread loop of the GPUBlock
// loop around it, or one of every sibling's extent; and a thread loop with
// no block loop around it is a launch of one block of its own threads.
bool whole_block(const Function &func, const std::string &block,
                 const Terminator::ParFor &loop);

} // namespace ssa
} // namespace ir
} // namespace bonsai
