#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Moves a motion off the extent it was applied to and onto the query, undone.
//
//     rel(q, transform(m, x))   ==>   rel(untransform(m, q), x)
//
// and then binds each distinct `untransform(m, q)` once: hoisted out of every
// recursion whose variables it does not mention, and at the top of every arm
// of a match that uses it -- the dispatch on a variant element inside a leaf's
// element loop, whose arm without a tree has no recursion to hoist ahead of
// and would otherwise test its shape once per copy of the term.
//
// A query over a tree held in an element writes `intersects(r, transform(m,
// tri))` -- the triangle where its instance puts it -- because that is the form
// with one uniform ray and one varying geometric object, which is the only form
// the ordinary bounding rule applies to. Predicate analysis then emits the same
// motion in every pruning condition, `intersects(r, transform(m, node_box))`,
// and the box changes at every node. Nothing about that is loop-invariant. What
// makes it invariant is the identity above: once the motion is on the ray it
// depends on the instance alone, and *then* it is ordinary code motion to
// evaluate it once per instance. That is pbrt's `Transform::ApplyInverse(r,
// &tMax)` at the top of `TransformedPrimitive::Intersect`, and it turns the
// cost of an instanced node from an eight-corner box transform into a slab
// test.
//
// The identity holds for the topological relations under any bijection, and
// for the metrics by the contract an `untransform` for the query type takes on
// -- see GeomOp::untransform. It does not hold for the ordering predicates,
// which ask about the world's axes, and those are left alone. The rewrite
// fires only where the program supplies the `untransform` LowerGeometrics
// would look up; where it does not, the query stays as written, correct and
// slow.
//
// A schedule's sort keys follow the query too. pbrt's front-to-back rule reads
// the sign of the ray's direction along the node's split axis, and inside an
// instance that axis is an axis of the instance's frame: pbrt applies the rule
// to the ray after `ApplyInverse`, and so must we, or a rotated instance is
// walked in the wrong order. Not wrongly answered -- an argmin does not depend
// on the order -- which is exactly why it would never have shown.
//
// Runs after LowerTrees, so the pruning conditions exist to be rewritten;
// after LowerSorts, so the keys are on the recursion to be carried; and before
// LowerGeometrics, so a motion is still a GeomOp rather than a call.
class PullQueries : public lower::Pass {
  public:
    const std::string name() const override { return "pull-queries"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
