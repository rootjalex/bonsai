#pragma once

#include "IR/Schedule.h"
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
//     computed and only one of them is right for a given lane. Each select
//     sits at the end of the path whose value it lets in, as ispc places its
//     masked assignments, rather than at the join as the Region Vectorizer
//     places its blends: a path's mask and raw value then die with the path,
//     and a path that is skipped (below) hands on the value from before it;
//   * a store in a block that is not always executed takes its block's mask,
//     so the disabled lanes do not write.
//
// Linearized code computes every arm whether or not any lane takes it, and in
// a path tracer most arms are taken by no lane of a gang once a few bounces
// are in. So an arm that does real work is put behind a uniform branch on
// `any(mask)` -- Shin's branch-on-superword-condition-code (Shin, Hall &
// Chame, "Superword-Level Parallelism in the Presence of Control Flow",
// CGO 2005), as ispc emits it around each arm of a varying `if`
// (IfStmt::emitMaskMixed) and as Moll & Hack install it with their "BOSCC
// gadget" (section 6). The gadget is a block on the edge into the arm holding
// the test and a uniform branch: to the arm when some lane is on, and past
// the arm's whole dominance region otherwise, to the block that region leaves
// for in the linearized graph. It is installed here, once the fold has said
// what that block is, so that every arm of a branch gets one -- the second
// arm's bypass lands where the first arm's does, after both -- where a gadget
// installed before the fold, as Moll & Hack do, can skip only the arm the
// index puts first.
//
// What makes the bypass sound is that nothing computed inside a skipped region
// is read outside it except through a join's argument, by SSA dominance in
// the original graph, and a join's argument is blended in running order along
// the linearized path. Where that path passes a bypass's landing, the running
// value becomes an argument of the landing block: the blend so far from
// inside the region, and the value from before the region along the bypass,
// which is exactly what a lane outside the arm would have read from the blend.
// The masks of edges inside a region that decide a later block's mask are
// threaded the same way, with `false` along the bypass, since no lane took
// them. No value is ever invented for the bypass.
//
// Which arms get a gadget. One is installed wherever it must be: before a
// region that touches memory or makes a call, work that must not be done on
// behalf of no lane (reading a payload field at a constant index is not
// that; see touches_memory). For an arm of pure arithmetic the test is a
// wager -- a `vptest` and a branch, paid on every pass, against the arm's
// arithmetic, saved only on the passes no lane takes it -- and whether it
// pays depends on how the program's data falls across the arms, which
// nothing here can know. ispc wagers by size (PREDICATE_SAFE_IF_STATEMENT_COST:
// an arm of six of its cost units or more gets the test); here the wager is
// the schedule's. `f.skip(Shape.Sphere)` puts a test in front of the arm of a
// match on a Shape that takes a Sphere, `f.skip(Shape)` in front of every arm
// of such a match, `f.skip()` in front of every arm f has (see
// ir::BranchPolicy). An arm is found by the provenance lowering left on its
// first block (ir::Provenance), so the directive holds through inlining and
// specialization, and a match written in an `[[inline]]` helper is named by
// the helper. apps/pbrt/schedules/packet.bonsai says where the render's go,
// and PLAN.md item 7 has the measurement behind it.
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
// `policies` is what the schedule said about which arms get a gadget beyond
// the ones that must (see above and ir::BranchPolicy), and `policy_name` the
// name it knows this function by: its own, or for a specialized variant the
// function it was specialized from.
//
// Returns the execution mask of each block that has one. A block that is
// absent runs with every lane of the gang enabled and needs no predication;
// callers use this to predicate anything linearization does not handle
// itself, such as a call made under a mask.
using BlockMasks = std::map<std::string, std::shared_ptr<Value>>;

BlockMasks linearize(Function &func, const std::string &entry,
                     const Divergence &divergence,
                     const std::shared_ptr<Value> &entry_mask = nullptr,
                     const std::vector<UniformLoop> &loops = {},
                     const ir::BranchPolicyMap &policies = {},
                     const std::string &policy_name = "");

} // namespace ssa
} // namespace ir
} // namespace bonsai
