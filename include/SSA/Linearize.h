#pragma once

#include "SSA/AnalyzeDivergence.h"
#include "SSA/SSA.h"

#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Removes the divergent branches from the region rooted at `entry`, so that
// every lane of a gang can follow the same path through it.
//
// This is the partial linearization of Moll & Hack, "Partial Control-Flow
// Linearization" (PLDI 2018), figure 5: blocks are visited in block index
// order, a divergent branch is replaced by a single edge to the successor of
// least index, and the successors it no longer branches to are recorded in a
// deferral relation so that a later block picks them up. Uniform branches are
// left alone, which is the point of the algorithm -- if-converting them would
// make every lane execute both sides for no reason.
//
// The paper is only concerned with the shape of the control flow and assumes
// the code is predicated separately. That predication is done here too, since
// the two need to agree:
//
//   * every block gains an execution mask, a boolean per lane saying which
//     lanes would have reached it in the original control flow;
//   * a block argument (this IR's phi) becomes a chain of selects over those
//     masks, since after linearization the values from every path are
//     computed and only one of them is right for a given lane;
//   * a store in a block that is not always executed takes its block's mask,
//     so the disabled lanes do not write.
//
// The masks are ordinary boolean values here; the widening in vectorize()
// turns them into vectors along with everything else derived from the loop
// index.
//
// Requires reducible control flow and, for now, a region without loops: a
// divergent loop has to be turned into a uniform one first (section 3.3 of
// the paper), which is not implemented yet.
void linearize(Function &func, const std::string &entry,
               const Divergence &divergence);

} // namespace ssa
} // namespace ir
} // namespace bonsai
