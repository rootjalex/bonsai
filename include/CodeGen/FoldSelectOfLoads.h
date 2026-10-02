#pragma once

#include <llvm/IR/PassManager.h>

namespace bonsai {

// A choice between two loads from one base at two constant offsets, made
// the same way in every lane,
//
//     select (splat c), (load (gep p, 64)), (load (gep p, 96))
//
// becomes one load through a chosen offset,
//
//     load (gep p, (select c, 64, 96))
//
// with the select of the two constants placed where `c` is defined -- for a
// condition a loop never changes, outside the loop. A BVH traversal's node
// test picks the near and the far bound of each slab by the sign of the
// ray's direction: Embree reads them through byte offsets it computes once
// per ray (node_intersector1.h, TravRay::nearX, `(char*)&node->lower_x +
// ray.nearX`), where the code here was blending the two loaded vectors at
// every node, six blends for a BVH8 node, a tenth of its test.
//
// Run last in the optimizer, after LLVM's own passes, for two reasons. By
// then the inliner and GVN have merged the several copies of the test a
// traversal makes -- the mask's, the sort key's, the carried bound's, each
// a call of the same function on the same node -- into one select of two
// loads, where before inlining they are three. And LLVM's instcombine has
// run: it turns a load from a *selected pointer* back into two loads and a
// select, and a select of vectors under one bool then comes out as a
// branch; a select of two integer offsets under a gep is not a form it
// rewrites, and nothing after this point does. Both loads were already
// made unconditionally, so reading one of them through a chosen offset
// reads what they read; the two left behind go when nothing else uses
// them. Only when nothing between the earlier load and the select may
// write what they read -- a store into the function's own stack memory
// through another pointer does not count.
struct FoldSelectOfLoads : llvm::PassInfoMixin<FoldSelectOfLoads> {
    llvm::PreservedAnalyses run(llvm::Function &function,
                                llvm::FunctionAnalysisManager &);
};

} // namespace bonsai
