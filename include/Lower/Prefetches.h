#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Applies the `prefetch` scheduling directive (ir::Prefetch):
//
//     trace.prefetch(tris.Interior.children);
//
// names the field of an arm that holds the tree's references -- the children
// a node tests -- and asks that the storage each hit child refers to be
// brought into cache in the loop that tests them. The tree lowering tests
// an arm's array of children in a parfor named for the field, with the test's
// outcome stored into a mask array indexed by the loop's index
// (Lower/Trees.cpp, from_children); this pass appends to that loop's body
//
//     if _mask[i] { prefetch(children[i]) }
//
// with `prefetch` of a tree reference (ir::Intrinsic::prefetch), which the
// layout lowering turns into the address and size of the row the reference
// names (Lower/Layouts.cpp). After LowerSorts, so that the keys' stores are
// in the loop before this is appended, and before LowerLayouts, which needs
// the reference still typed as one.
//
// Halide's `prefetch(g, at, from, offset)` places a prefetch in loop `at`
// for the region of `g` iteration `from + offset` touches, found by bounds
// inference (src/Prefetch.cpp); here the region is the row behind an
// indirection, which the layout knows and bounds inference cannot see, and
// `at` is the loop named by the field. Embree prefetches each hit child as
// it extracts it (kernels/bvh/bvh_traverser1.h, `BVH::prefetch`).
//
// Refused: a cursor that is not `<tree>.<arm>.<field>`, a field that does
// not hold references to the tree, and a function with no traversal of the
// tree named.
class LowerPrefetches : public Pass {
  public:
    const std::string name() const override { return "lower-prefetches"; }

    // Requires the full program: the schedule and the functions.
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
