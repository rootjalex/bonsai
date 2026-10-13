#pragma once

// Take the lock off an atomic accumulate nothing can contend for.
//
// A program says `atomic` because a `parfor` around the accumulate *may* be
// run in parallel, not because it is: whether it is at all is the schedule's
// decision, made later. So an atomic under a loop that nothing was bound to is
// an ordinary accumulate paying for a guarantee no one needs, and removing it
// is the compiler's job rather than something the program should have to
// predict.
//
// That is what lets an algorithm expose its parallelism honestly. The
// alternative -- writing a sequential loop to avoid the cost -- buys the same
// speed by giving up the ability to schedule the loop at all, which is the
// wrong trade in a language whose whole point is that the schedule decides.
//
// Runs after the schedule has been applied, because that is what decides which
// loops carry a binding. See SSA/Contention.h for what "can contend" means and
// how far it can be shown.
//
// The loops that run an accumulate at the same time are not all in its
// function. A function a bound loop's body calls runs once per concurrent
// iteration -- the drained function of a GPU queue, the `step` of a `parfor`
// over threads -- and an accumulate in it into a reducer the loop's owner
// handed down is the same word written by every thread, with no loop of the
// callee's to show it. So which functions are called in parallel is settled
// over the whole program first (functions_called_in_parallel), and in one of
// those an atomic comes off only for storage the call made for itself.

#include "Lower/Pass.h"

namespace bonsai {
namespace ir {
namespace ssa {

class DemoteAtomics {
  public:
    const std::string name() const { return "demote-atomics"; }

    // Rewrites in place: only the `atomic` flag on accumulate instructions
    // changes, never the graph. `called_in_parallel` says whether some bound
    // loop of the program calls `f`, directly or through other calls: then
    // its callers' loops enclose every block of it, and an accumulate to an
    // address that is not the call's own allocation keeps its atomic.
    static void run(Function &f, bool called_in_parallel);
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
