#pragma once

#include "SSA/SSA.h"

#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Replaces the parfor named `index` in `func` by its body written out once
// per iteration, the copies one after another in the loop's order and the
// index a constant in each: Halide's `unroll`. This is what `unroll(l)`
// does (ir::Unroll).
//
// The loop's start, end and stride have to be constants, since the number
// of copies is decided here; a loop whose extent only a run knows is
// refused -- split it by a constant chunk and unroll the inner loop. A loop
// a bind() put on hardware is refused too: its iterations are what the
// hardware runs, not a body to write out.
//
// What it is for. A node that holds its children's boxes tests them in a
// loop over the children whatever their number (Lower/Trees.cpp,
// from_children), so that a schedule can vectorize that loop into one test
// of every child at once, Embree's and FCPW's wide nodes. A binary node's
// two tests are written out in a row by FCPW (bvh.inl,
// processSubtreeForClosestPoint) and by pbrt: each child's box read off
// its own row and tested, then the two ordered. Unrolled, the loop is
// that: the index a constant, each copy's reads of the children's boxes
// fold to its own child's once the simplifier may read a built vector's
// lanes back (SSA/Simplify.h, fold_built_lanes), and the mask and the
// carried bounds the loop filled per child are each written once.
//
// The body is copied with clone_region (SSA/CloneFunction.h), every copy
// renamed, and in each copy the index is replaced by its iteration's
// constant and dropped from every block that took it (substitute_argument,
// SSA/InsertPreheader.h): a copy is the body with the index a constant,
// not a body handed one. Its yields are first made one (unify_yields),
// which is the one place the copy hands on to the next, and the last copy
// to the loop's continuation. A parfor nested in the body is copied with
// it, under its own name in every copy. Returns the number of copies made.
size_t unroll_loop(Function &func, const std::string &index);

} // namespace ssa
} // namespace ir
} // namespace bonsai
