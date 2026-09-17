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
// For each block with an execution mask, when its mask is not the mask the
// whole region runs under, a guard block is put in front of it holding
// `any(mask)` and a dispatch: the block when some lane is on, and when none
// is, past the *run* of blocks after it whose masks are narrowings of its own
// -- the arms nested inside this one, which no lane can be on either -- to
// the first block that is not. One test skips the whole arm, and the blocks
// inside it need no test of their own for the outer condition; an inner
// block with a narrower mask still gets a guard of its own, whose bypass lands
// inside or at the end of the outer run. The first block's arguments move to
// the guard, so the edges into it need no change. A value the run defines and
// a later block uses is not defined along the bypass, so it becomes an
// argument of the block after the run and the later uses take that argument.
// What the guard hands it is what the run would have: for a blend the run
// made -- `select(mask, value, before)`, which is how linearization merges an
// arm's value into a join, at the end of the arm (see SSA/Linearize.h) -- the
// value from before the arm, since no lane is in the mask; that is what a
// lane outside the arm reads from the blend when the arm does run, so nothing
// downstream can tell the arm was skipped. Anything else only the run
// computes gets a zero, which nothing reads: every use of such a value is
// under a mask that is empty when the run was skipped. Values used only inside
// the run are left alone, which is most of what an arm computes. A run is
// skipped only when it does real work, and only when the block after it is
// reached from the run alone, or from the run and other guards' bypasses,
// which hand it the same: a join with several live paths (see the
// `remaining > 1` case in Linearize.cpp) has no one path to take a value
// from, and is not touched.
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
