#pragma once

// Whether two iterations that run at the same time can write the same place.
//
// This is the question an atomic accumulate exists to answer. A `parfor`
// promises its iterations are independent, and an accumulate to a location two
// of them share is not -- so the program says `atomic`, and what that costs
// depends on whether the schedule actually parallelised the loop around it. An
// atomic under a loop nothing was bound to is a plain accumulate wearing a
// lock, and taking the lock off is this analysis's job.
//
// The general question is undecidable: an address can be any computed value,
// so deciding whether two of them ever coincide is deciding equality of
// arbitrary functions. Even restricted to affine subscripts it is integer
// linear feasibility, which is NP-complete in general. So this does not try to
// be complete -- it answers cheaply where it can and says so when it cannot,
// and a caller that cannot tell keeps the atomic. Under scheduling guarantees
// that permit reassociation, being unsure costs speed rather than correctness.
//
// What bonsai's design buys is the expensive half for free. In C one must
// *infer* which loops are parallel, which is what dependence analysis is for;
// here `parfor` plus a `bind` says so outright, and SSA makes what an index is
// derived from a walk back along def chains rather than a fixed point.

#include "SSA/SSA.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// What is known about the addresses two concurrent iterations write.
//
// Three-valued on purpose. "Cannot tell" is a different fact from "they
// collide", and keeping them apart is what lets a stronger test be added later
// without changing any caller: an affine solver would move cases out of
// Unknown and leave the other two alone. Collapsing to a bool now would bake
// in the conservatism.
enum class Contention {
    // No two concurrently running iterations write the same address, so an
    // atomic is unnecessary.
    Disjoint,
    // They provably do. The atomic is doing real work -- and had the program
    // not asked for one, this is a race worth reporting.
    Shared,
    // Neither could be shown. Keep the atomic.
    Unknown,
};

// One enclosing loop that the schedule made parallel.
//
// The bounds are carried even though the check below does not read them: an
// affine test needs the iteration space, and collecting it here means adding
// that test does not mean revisiting how the nest is walked. Same for the
// index's Value, which is what a def-chain walk compares against.
struct ParallelLoop {
    std::string index;
    std::shared_ptr<Value> index_value;
    std::shared_ptr<Value> start, end, stride;
};

// The parallel loops enclosing each block of a function, outermost first.
//
// Only loops the schedule bound to hardware are here. An unbound parfor still
// says its iterations *may* run in any order, but nothing is going to run two
// of them at once, so nothing can collide.
std::map<std::string, std::vector<ParallelLoop>>
parallel_loops_by_block(const Function &f);

// Whether the address `ptr` is written by at most one concurrent iteration.
//
// `ptr` is the pointer operand of an accumulate: a chain of GEPs from some
// base. `enclosing` is what the map above holds for the block it lives in.
Contention contention_of(const std::shared_ptr<Value> &ptr,
                         const std::vector<ParallelLoop> &enclosing);

// The common case as a predicate, for callers that only want to know whether
// they may drop the atomic. Unknown counts as unsafe.
inline bool is_parallel_safe(const std::shared_ptr<Value> &ptr,
                             const std::vector<ParallelLoop> &enclosing) {
    return contention_of(ptr, enclosing) == Contention::Disjoint;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
