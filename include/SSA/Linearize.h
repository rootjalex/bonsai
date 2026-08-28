#pragma once

#include "SSA/AnalyzeDivergence.h"
#include "SSA/SSA.h"
#include "SSA/UniformizeLoops.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

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
// `entry_mask` is the mask the region starts under. A ParFor body runs with
// every lane enabled and passes nothing here; a function specialized for a
// conditional call site runs under the mask of that call, and then every
// block inside it is predicated by that mask -- including the ones whose own
// predicate is uniform, which would otherwise need no mask at all.
//
// Requires reducible control flow, and that every loop in the region be
// uniform -- a divergent one has to be folded into data flow first, by
// SSA/UniformizeLoops.h, whose result is passed back in as `loops`. Uniform
// loops are handled as section 3.3 of the paper describes: figure 5 is run
// with the back edges deleted, and each back edge is re-inserted at its latch
// afterwards. That is sound because a uniform loop has no divergent exit that
// could defer a block past the latch, so reaching the latch means no lane left
// the loop this iteration.
//
// A uniformized loop is uniform in that sense, but its header still needs a
// mask: the live mask says which lanes have not left yet, and it is what stops
// a lane that finished on an earlier iteration from storing anything now.
// `loops` supplies it, and gets back the mask its loop is entered under, which
// is only known once mask generation reaches the preheader.
//
// Returns the execution mask of each block that has one. A block that is
// absent runs with every lane of the gang enabled and needs no predication;
// callers use this to predicate anything linearization does not handle
// itself, such as a call made under a mask.
using BlockMasks = std::map<std::string, std::shared_ptr<Value>>;

BlockMasks linearize(Function &func, const std::string &entry,
                     const Divergence &divergence,
                     const std::shared_ptr<Value> &entry_mask = nullptr,
                     const std::vector<UniformLoop> &loops = {});

} // namespace ssa
} // namespace ir
} // namespace bonsai
