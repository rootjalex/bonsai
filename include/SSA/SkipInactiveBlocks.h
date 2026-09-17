#pragma once

#include "SSA/Linearize.h"
#include "SSA/SSA.h"

#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Skips a linearized block when no lane of the gang is on in it.
//
// Linearization (SSA/Linearize.h) turns a divergent branch into straight-line
// code that computes both arms and blends, so every gang pays for every arm
// whether or not any of its lanes takes it. In a path tracer most lanes are
// dead after a few bounces and most arms are taken by no lane, so most of that
// work is wasted. This is Shin's branch-on-superword-condition-code (Shin,
// Hall & Chame, "Introducing control flow into vectorized code", PACT 2007),
// as ISPC emits it around every block it predicates: a uniform branch on
// `any(mask)` around the block, taken together by the whole gang, which
// leaves the block's work out entirely when the mask is empty.
//
// For each block with an execution mask, when the block does real work and
// its mask is not the mask the whole region runs under, a guard block is put
// in front of it holding `any(mask)` and a dispatch: the block when some lane
// is on, its successor when none is. The block's arguments move to the guard,
// so the edges into it need no change. A value the block defines and a later
// block uses is not defined along the bypass, so it becomes an argument of the
// successor -- the value from the block, zero from the guard -- and the later
// uses take that argument; the zero is never read, since every such use is
// under a mask that is empty when the block was skipped, or a blend that
// selects the other side. A block whose own values travel to its successor as
// arguments already is left alone: those are the slots of a join with several
// live paths (see the `remaining > 1` case in Linearize.cpp), whose fallback
// is the previous value and not zero.
//
// `masks` are the masks linearization returned, updated here where a mask
// became an argument of a later block. `entry_mask` is the mask the region
// runs under, if any: a block under only that mask is not guarded, since the
// caller has already tested it (see the `$masked` call guard in
// CodeGen_LLVM_SSA.cpp). Returns how many blocks were guarded.
size_t skip_inactive_blocks(Function &func, const std::string &entry,
                            BlockMasks &masks,
                            const std::shared_ptr<Value> &entry_mask = nullptr);

} // namespace ssa
} // namespace ir
} // namespace bonsai
