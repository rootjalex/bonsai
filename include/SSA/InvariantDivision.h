#pragma once

#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// Division by invariant integers using multiplication.
//
// Granlund & Montgomery, "Division by Invariant Integers using
// Multiplication", PLDI 1994; the unsigned round-up method is their figure
// 4.2 and the signed method their figure 5.2, both as Warren gives them in
// Hacker's Delight, chapter 10. A C compiler applies the method only to a
// divisor it can see -- a constant -- because the multiplier costs a wide
// division to compute and the compiler has nowhere to put that cost but where
// the division already is. Here the divisor is often a run-time value that
// does not change while a loop runs: a Halton radical inverse divides an index
// by the same prime for every digit, and a gang of lanes divides by a prime
// each. So the multiplier is computed once, where the divisor is defined, and
// every division by it inside a loop becomes a multiply-high and a few shifts,
// which a vector unit has and a divider does not: an x86 core divides one
// 64-bit lane at a time through a scalar `div`, taking the lanes apart and
// putting them back around each one, and multiplies eight at once.
//
// For every integer division or remainder whose divisor is defined outside a
// loop the division is in, the divisor's multiplier (Intrinsic::div_multiplier)
// and shift amounts are placed right after the divisor's definition, threaded
// to the division's block as any other value is, and the division is rewritten
// in place. Every other division by the same divisor, in a loop or not, uses
// the multiplier too, since it is there. A division by a constant is left to
// the backend, which does this itself.
//
// Run before a schedule vectorizes anything: a division the vectorizer meets
// under a mask is guarded against dividing by a lane's stale zero, and the
// guard is defined inside the loop; the arithmetic this leaves behind needs no
// guard, since nothing in it traps. Returns how many divisions were rewritten.
size_t divide_by_invariants(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
