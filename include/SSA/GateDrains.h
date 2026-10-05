#pragma once

// Launch gates for queue drains: a drain that cannot have an entry is not
// run. pbrt's wavefront integrator skips whole kernel launches by per-scene
// facts set at load time -- `haveBasicEvalMaterial[]` for the material
// kernels, `haveSubsurface` for the BSSRDF kernels, `haveMedia` for the
// medium kernels (wavefront/integrator.cpp, integrator.h) -- and without
// them every depth of a GPU schedule launches every drain, 6-20 us an empty
// launch, 12-41% of the kernel time of the swept scenes that bounce deep.
// This pass is those facts as a rewrite of the queues' owner: each drain's
// parfor is entered through a dispatch on a condition under which the queue
// can be fed at all, and the false arm jumps straight to the parfor's
// continuation. Zero iterations write nothing, so skipping is the launch's
// cost and nothing else.
//
// A gate says "no entry can be pushed", and comes from three places:
//
// 1. A split queue's variant (QueueSpec::split, `hits.specialize(material)`):
//    whether any entry can carry the variant's tag is a fact about the
//    scene's data -- whether any material in the pool IS that arm -- which
//    only the driver can know. Each ADT a queue is split on adds one `u64`
//    parameter to the owner, `<adt>_arms`, bit `tag` meaning the arm occurs;
//    the leaf queue `hits!...!Diffuse` is gated on its bit. The driver owns
//    the promise: all-on (the generated header's `..._ARMS_ALL`) is always
//    sound and is what a driver that computes nothing passes. An `option`
//    key gets no mask; its two queues launch as before.
//
// 2. A push dominated by a condition the owner can read: a push to the
//    medium-sample queue sits under `if have_media && medium >= 0`
//    (render.bonsai, as pbrt's trace kernel pushes only when the medium
//    queue exists at all, which is `haveMedia`), and `have_media` reaches
//    the drained function as an invariant parameter whose value at the
//    drain's call is the owner's own (SSA/Defer.cpp's invariant-parameter
//    hoist). The owner-readable conjuncts of such a condition gate the
//    queue; conjuncts it cannot read (`medium >= 0`, an entry's own state)
//    are dropped, which only weakens the gate toward launching.
//
// 3. Everything else by reachability: a queue fed only from inside gated
//    drains inherits their gates -- pbrt's `haveSubsurface` is exactly the
//    subsurface arm's bit reaching the BSSRDF queues through the push graph
//    (the probe queue's only pusher is the subsurface material's kernel, the
//    exit queue's the probe drain, ..). gate(Q) = OR over Q's pushes of
//    (the gate of the drain the push is in, AND the push's own owner-readable
//    conjuncts), to a fixpoint from "unfeedable", with a push outside any
//    drain (the producer's own, the camera rays) gated true.
//
// A gate that folds to true leaves the drain exactly as it was; everything
// here only ever removes launches the queue's emptiness already made
// no-ops. The pass runs after every directive and before lower_pushes()
// (SSA/Convert.cpp), while a push is still one Push instruction naming its
// queue, and before the signature read-back so the added mask parameters
// are in the owner's exported type.

#include "IR/Program.h"
#include "IR/Schedule.h"
#include "SSA/SSA.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// Gates every queue drain in `fmap` whose queue provably cannot be fed
// under a scene fact: the split masks (added as owner parameters, returned
// for the generated header -- ir::Program::ArmMask), the owner-readable
// push conditions, and the closure of both over the push graph.
// `queue_splits` is the schedule's, as written; `storages` resolves the
// split keys' variants and tags.
std::vector<ir::Program::ArmMask>
gate_drains(FuncMap &fmap, const std::vector<ir::QueueSpecialize> &queue_splits,
            const std::map<std::string, ir::Program::AdtStorage> &storages);

} // namespace ssa
} // namespace ir
} // namespace bonsai
