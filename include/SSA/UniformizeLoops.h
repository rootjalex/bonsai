#pragma once

#include "SSA/AnalyzeDivergence.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// Turns divergent loops into uniform ones, so that partial linearization has
// only uniform loops left to deal with (Moll & Hack, PLDI 2018, section 5).
//
// A loop is *divergent* when the lanes of a gang disagree about when to leave
// it: its trip count is per-lane. Such a loop cannot be vectorized as it
// stands, because a single scalar branch has to decide for the whole gang
// whether to go round again. The transform folds the divergent exits into
// data flow and leaves one uniform exit behind:
//
//   * A *pure latch* is inserted: a block with the loop's only back edge and
//     its only exit. It branches on `any(live)` -- go round again as long as
//     at least one lane still wants to -- which is a question about the gang
//     as a whole and so uniform. Lanes that have left simply stop having any
//     effect, since their bit of the live mask is clear.
//
//   * The header gains a *live mask*, an argument holding one bool per lane,
//     seeded on entry and cleared for a lane at the iteration it leaves. Mask
//     generation uses it as the header's execution mask, which is what stops a
//     lane that has already left from storing anything on later iterations.
//
//   * Every exiting edge is *rebound* to the pure latch, through a break block
//     whose predicate is the edge's. The break block records which lanes left
//     that way in an *exit mask* per destination, and captures the values that
//     edge carried in a *tracker* per destination argument. A lane leaving on
//     iteration three and one leaving on iteration nine each keep their own
//     value -- this is what makes a loop-carried value that is used after the
//     loop come out right, and it is why such a value is varying even when
//     everything it is computed from is uniform.
//
// The loops it leaves behind are still scalar loops; nothing here widens
// anything. The blends that turn the header and pure latch arguments into
// per-lane selects are the ordinary work of linearization, which is why the
// masks below are handed to it rather than computed here.
struct UniformLoop {
    // The loop header, whose execution mask is the live mask below.
    std::string header;
    // The header argument holding the live mask.
    std::shared_ptr<Value> live;
    // Where the live mask is seeded, as the block that jumps into the loop
    // and the index of the seed among that jump's arguments. The seed is the
    // mask the loop is entered under, which is not known until mask
    // generation has reached the preheader -- linearization fills it in.
    std::string preheader;
    size_t seed_arg = 0;
    // Everything inside the loop. All of it runs under the live mask, in the
    // same way that everything in a region runs under the mask the region was
    // entered with: a block in here that no divergent branch decides is still
    // not executing for a lane that has already left.
    std::set<std::string> blocks;
};

struct LoopUniformization {
    std::vector<UniformLoop> loops;
    // The block arguments introduced above, as (block, argument). Every one of
    // them holds one value per lane by construction -- a mask, or a value
    // captured at the iteration a lane left -- and none of it is visible to an
    // analysis that only follows operands, so they are seeded as varying.
    std::set<std::pair<std::string, std::string>> varying_args;

    bool empty() const { return loops.empty(); }
};

// Uniformizes every divergent loop in the region of `func` rooted at `entry`.
// `divergence` is the analysis of that region as it stands, which is what says
// which loop exits the lanes disagree about.
LoopUniformization uniformize_loops(Function &func, const std::string &entry,
                                    const Divergence &divergence);

} // namespace ssa
} // namespace ir
} // namespace bonsai
