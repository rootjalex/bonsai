#pragma once

// Whether every entry of one pass of a queue's drain is a different
// sample's: the fact that lets the drained function's accumulates into a
// sample's record slot come off their atomics (SSA/Contention.h,
// Function::injective_slots).
//
// A queue's entry carries the slot of the producer iteration it came from
// -- pbrt's pixelSampleState index -- and two entries of one pass write the
// same slot only when one iteration's work pushed onto the queue twice
// before the pass. That work is an activity that begins at an iteration of
// one of the owner's innermost parallel loops -- the sample loop, or a
// drain's loop, whose iteration is one entry -- and goes on through the
// calls it makes and through the queues it pushes onto whose drains run
// later in the same round: an entry pushed onto `hits` is run by the hits
// drain before the shadow drain, and what it pushes onto `shadow` is the
// same sample's. So the question is a count over paths: the greatest
// number of pushes onto the queue that one activity can make, where a push
// onto another queue of the round counts what that queue's drained
// function makes, a push onto a queue drained in rounds counts nothing
// (its entries run in the next round), a loop around a push counts more
// than one, and a recursion counts more than one. At most one along every
// path of every root is what makes the slots distinct; two drains that
// each push for a sample are fine when no path reaches both, which is how
// the renderer's material kernels and its medium kernel both push shadow
// rays without a sample ever sending two in a round.
//
// The deferral asks this when it makes a queue's drain (SSA/Defer.cpp,
// slots_distinct) and records the answer on the drain's slot reads. The
// program is read as it stands at that point: the calls the deferral is
// about to turn into pushes are its pushes, and the queues made before it
// are pushed onto already.

#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Whether no activity of `owner` pushes more than one entry onto `queue`
// per pass of its drain. `callee` is the function whose calls the deferral
// is turning into pushes onto `queue` (empty when the queue is pushed by
// Push instructions already, as a queue made earlier is). `why`, when
// given, says which activity pushes more, or what could not be counted.
bool pushes_at_most_once(
    const std::map<std::string, std::shared_ptr<Function>> &funcs,
    const Function &owner, const std::string &queue,
    const std::string &callee, std::string *why = nullptr);

} // namespace ssa
} // namespace ir
} // namespace bonsai
