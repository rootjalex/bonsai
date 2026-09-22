#pragma once

#include "SSA/AnalyzeDivergence.h"
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
// the multiplier too, since it is there.
//
// A division by a constant is rewritten too, everywhere, with its multiplier
// computed here at compile time (Hacker's Delight, chapter 10; the sequences
// Halide's lower_int_uint_div emits, see the source): a shift for a power of
// two, a multiply-high and a shift otherwise, and nothing at all for one. LLVM
// does the same for a scalar, and for a vector of lanes only where the target
// has a wide enough multiply-high: a vector of 64-bit lanes it takes apart
// into one division per lane, where the multiply-high this leaves is built
// from the 32-bit products every machine has (see CodeGen/
// ExpandVectorMulHigh.h). Doing it here also keeps the arithmetic in view of
// the simplifier, and reaches every backend alike.
//
// Run before a schedule vectorizes anything: a division the vectorizer meets
// under a mask is guarded against dividing by a lane's stale zero, and the
// guard is defined inside the loop; the arithmetic this leaves behind needs no
// guard, since nothing in it traps. Returns how many divisions were rewritten.
size_t divide_by_invariants(Function &func);

// Division by floating-point division, where it is exact.
//
// No vector unit divides integers, and every one divides floats in a single
// instruction. For 0 <= a, b with a + b < 2^m, m the mantissa's bits, the
// correctly rounded float quotient a/b truncates to the integer quotient:
// the rounding error is below a/b 2^-m, and a/b is at least 1/b away from
// the next integer on either side unless it is one, in which case it is
// exact. So a division whose operands are known to be small enough -- both
// below 2^23 for a float, both below 2^52 for a double -- is done as one:
// the operands converted, divided, the quotient truncated back, and the
// remainder a - qb. Only a division the lanes of a gang make (`divergence`
// says which) is worth it; a scalar has its divider. Bounds come from
// upper_bound in the source, the same analysis that lets a small dividend
// divide a constant by a comparison. A division the bounds do not reach is
// left, and the x86 code generator does what it can with doubles
// (CodeGen_LLVM::vector_int_division). Returns how many were rewritten.
size_t divide_bounded_by_floats(Function &func, const Divergence &divergence);

// Division by a divisor the lanes of a gang agree on.
//
// The second kind of invariance. `divide_by_invariants` finds a divisor a
// loop does not change; inside a gang the divergence analysis knows a
// divisor the lanes do not change -- a uniform value dividing a varying one,
// apps/pbrt's Halton index over the sampler's base scale -- and no machine
// divides a vector of integers, so the lanes were divided one at a time.
// The same rewrite, with that criterion: the multiplier is computed once,
// on the scalar divisor where it is defined (a 128-bit division for a
// 64-bit divisor, one, rather than one per lane), and every lane's division
// is a multiply-high by its broadcast. Run by the vectorizer before it
// widens, after divide_bounded_by_floats, which is cheaper where it
// applies. `region_entry` names the block the vectorized region begins at:
// a divisor defined beyond it reaches the region as that block's argument,
// and its multiplier is computed there, at the region's top. Returns how
// many divisions were rewritten.
size_t divide_by_uniform_divisors(Function &func, const Divergence &divergence,
                                  const std::string &region_entry);

} // namespace ssa
} // namespace ir
} // namespace bonsai
