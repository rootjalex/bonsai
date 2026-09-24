#pragma once

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Loop interchange of two nested parfors, `f.reorder(inner, outer)`: the loop
// named `inner` is inside `outer` afterwards. Written for a nest that has
// them the other way round.
//
// WHAT IT MEANS
//
// The nest need not be perfect. Take the outer loop `p` and the inner `s`:
//
//     parfor p in Rp {
//         Pro(p);                      // the prologue: the body before s
//         parfor s in Rs { Body(p, s); }
//         Epi(p);                      // the epilogue: the body after s
//     }
//
// `f.reorder(p, s)` makes of it
//
//     parfor p!prologue in Rp { Eff(p); }        // if anything stays in Pro
//     parfor s in Rs { parfor p in Rp { Pure(p); Body(p, s); } }
//     parfor p!epilogue in Rp { Epi(p); }        // if Epi is not empty
//
// -- loop distribution (fission) of the outer loop at its inner loop, then
// interchange of the perfect nest that leaves, in the terms of Allen and
// Kennedy. Each instruction of the prologue goes one of three ways, by what
// it is rather than by whether it happens to depend on `p`:
//
//   * Pure arithmetic -- anything that computes a value from its operands
//     alone, reading no memory: recomputed wherever it is used, in the new
//     inner body (`Pure(p)`: the pixel's `i` and `j` from `p`), in the
//     epilogue loop, and in the prologue loop when something kept there
//     reads it. Index arithmetic is cheaper to redo than to store and
//     reload, and an instruction that does not depend on `p` is recomputed
//     all the same: LLVM hoists it out of the loops it ends up in, under its
//     own rules about what may run before it is known to be needed.
//   * A value made some other way -- a read of memory (a load, a field read,
//     an element of an array), a random draw, a fetch-and-add: computed once
//     per `p` in the prologue loop, where the program computed it, and
//     *scalar-expanded* into an array `<name>!expanded` of one entry per
//     iteration of `p`, which the inner body and the epilogue read back at
//     the iteration's number, `(p - start) / stride`. A read may not be
//     moved into the inner body because the body's own iterations may write
//     what it reads (`w = weight_out[p]` before `atomic weight_out[p] += ...`
//     reads the value before any sample, and would read a partial sum after
//     the move), and may not be moved before the prologue loop because the
//     prologue's own effects may write it.
//   * An effect -- a store, an accumulate, a print, a push -- stays in the
//     prologue loop, once per `p`, before every `Body(p, .)`, as it was. So
//     does a `mut` local only the prologue uses, with its stores and loads.
//   * What is moved before the loop, *hoisted*: a read of memory that
//     nothing in the whole region of `p` writes, at an address that is the
//     same on every iteration -- a load of a slot of the function (the
//     inliner's result slot for a value the program computed before the
//     loop, the sample count a match on the sampler settled), an element of
//     a parameter the function does not write through (the scene) at an
//     invariant index -- is the same on every iteration and reads memory
//     that is there whether or not the loop runs, so it is read once, in the
//     block that held the `p` loop. Arithmetic that cannot trap on such
//     reads, on the header's values and on constants is hoisted with them
//     (a division that may be by zero is not: nothing that could fault runs
//     when the original would not have run it). So is a `mut` local of the
//     prologue whose every write is a whole store of a hoisted value in the
//     prologue and whose address escapes to nothing that writes it (a
//     callee's non-`mut` parameter is a read): specialize()'s copy of the
//     integrator, stored once and handed to every kernel, is one slot
//     before the loop rather than one per iteration. What writes what is
//     found over the region -- stores and accumulates by their address's
//     root, pushes, stored addresses, `mut` parameters of callees -- and the
//     classification runs to a fixed point, since a hoisted slot makes its
//     loads invariant and those may make more so. BONSAI_EXPLAIN_REORDER=1
//     prints each instruction's fate.
//
// A `mut` local of the outer iteration that the inner loop or the epilogue
// uses -- storage `Pro` allocates and `Body` or `Epi` writes through -- and
// that was not hoisted is privatized: one slot per iteration of `p` in an
// array, the local becoming the address of its slot, which is index
// arithmetic and recomputed where used. An array-typed local is refused; it
// would need a two-dimensional expansion the SSA form has no handle type
// for.
//
// The epilogue is the region from the inner loop's continuation to the outer
// body's yields, moved whole into the epilogue loop's body; what it reads of
// the prologue follows the rules above. Empty -- a bare yield -- means no
// epilogue loop. The prologue is required to be the one block that ends at
// the inner parfor: a prologue with control flow of its own -- a branch, or
// a call to a function that is not inlined, which ends a block in this form
// -- would have to be replicated as guards in the inner body (a conditional
// inner loop) and is refused rather than guessed at. The bounds of `s` have
// to be there before `p` starts, since the interchanged nest evaluates them
// before either loop runs: constants, values `p`'s header hands its body,
// or what the prologue works out from those and from unwritten slots by
// arithmetic that cannot trap, which is hoisted. A bound worked out from
// the index (a triangular nest, which has no rectangle to interchange
// without new bounds), from a read the loop may write, or by a division
// that may be by zero is refused. Nothing that could fault is run when the
// original would not have run it: an outer range that is empty runs nothing
// that can fail after the interchange either. Where both loops are bound,
// the pair is checked to
// nest the way the hardware does once swapped (Bind.cpp's rule); each piece
// keeps the binding its loop had. The distributed loops are variants of `p`
// in the sense of `loop_matches` (SSA/Convert.cpp), so `f.bind(p, R)`
// written after the reorder binds all three.
//
// WHY IT IS LEGAL
//
// A parfor states that its iterations are independent: no iteration reads
// what another writes, other than through a reducer's accumulate, whose
// order the program has given up (see ir::Bind and the schedule's Halide-
// style guarantees). In dependence terms the `p` loop carries no dependence,
// so every direction vector of the nest is `(=, *)`: interchange is legal by
// the direction-vector test (Allen and Kennedy, Theorem 5.1 and Section
// 5.2; Wolf and Lam's unimodular criterion with an empty dependence set),
// and distribution is legal at any cut since no dependence cycle crosses it
// (Kuck et al. 1981; Allen and Kennedy, Section 6.2 and Section 5.7, where
// scalar expansion -- Section 5.4 -- is what carries a scalar across the
// cut). Within one iteration the order `Pro(p)` before every `Body(p, .)`
// before `Epi(p)` is what the program wrote, and the distributed form keeps
// it: the prologue loop finishes before the nest starts, and the nest before
// the epilogue loop. Values cross the cut in one of the two classical ways,
// recomputation for pure arithmetic and scalar expansion for the rest.
//
// What the interchange does change is the order a reducer's adds arrive in.
// A location accumulated only inside `Body(p, .)` keeps its order under
// `reorder(p, s)` while `s` is unbound -- each pixel's samples are still
// added in sample order, which is what keeps apps/pbrt bit for bit against
// pbrt -- and a later `bind(s, ...)` reassociates it, a trade the schedule
// is allowed to make. A location the prologue or the epilogue accumulates
// into as well as the body is reassociated by the distribution itself,
// bound or not, since the prologue's adds now all precede the body's. Halide
// has the same rule from the other side: `Stage::reorder` refuses to move a
// reduction variable unless the update is provably associative and
// commutative (src/Func.cpp).
//
// HALIDE
//
// Halide's `reorder` (Ragan-Kelley et al. 2013, Section 3.2, "the domain
// order") is a permutation of a Func's dimension list before any loop
// exists, and a Func's nest is perfect by construction: the body evaluates
// one point. What sits at an intermediate loop level in Halide is another
// Func's realization, `g.compute_at(f, y)`, anchored to the loop by the
// variable's name and so moving with it -- a separate stage with storage of
// its own between the two loops. That is exactly the distribute-and-expand
// form this transform reaches from a nest a program wrote: Halide never had
// the prologue question because its language keeps the prologue in a Func,
// and bonsai's answer is Halide's, with the recomputation of index
// arithmetic as the one economy.
//
// LITERATURE
//
//   R. Allen and K. Kennedy, "Optimizing Compilers for Modern Architectures:
//     A Dependence-based Approach", Morgan Kaufmann, 2001. Chapter 5 (loop
//     interchange, Sections 5.2-5.3; scalar expansion, 5.4; loop
//     distribution, 5.7) and Section 6.2.
//   M. E. Wolf and M. S. Lam, "A Loop Transformation Theory and an Algorithm
//     to Maximize Parallelism", IEEE TPDS 2(4), 1991. Interchange as a
//     unimodular transformation, legal when it keeps every distance vector
//     lexicographically positive.
//   D. J. Kuck, R. H. Kuhn, D. A. Padua, B. Leasure and M. Wolfe,
//     "Dependence Graphs and Compiler Optimizations", POPL 1981. Loop
//     distribution (fission) over the strongly connected components of the
//     dependence graph; scalar expansion.
//   K. Kennedy and K. S. McKinley, "Maximizing Loop Parallelism and Improving
//     Data Locality via Loop Fusion and Distribution", LCPC 1993. Distribution
//     and interchange used together on an imperfect nest (their Gaussian
//     elimination example).
//   M. Wolfe, "High Performance Compilers for Parallel Computing", Addison-
//     Wesley, 1996. Section 9.3 (loop fission) and 9.5 (loop interchange).
//   J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand and S.
//     Amarasinghe, "Halide: A Language and Compiler for Optimizing
//     Parallelism, Locality, and Recomputation in Image Processing
//     Pipelines", PLDI 2013. Section 3.2, the domain order and the call
//     schedule; `Stage::reorder` in src/Func.cpp.
//
// Implemented in SSA/ReorderLoops.cpp.
void reorder(FuncMap &funcs, std::string func, std::string inner,
             std::string outer);

} // namespace ssa
} // namespace ir
} // namespace bonsai
