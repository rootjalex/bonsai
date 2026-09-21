#pragma once

#include <llvm/IR/PassManager.h>

namespace bonsai {

// Expands the high half of a 64-by-64-bit multiply on a vector of lanes --
// `trunc (lshr (mul (zext a), (zext b)), 64)` over a vector of i128, or its
// sext form for the signed product -- into the 32-by-32-to-64 products the
// machine has, the way long multiplication is done by hand: with
// a = ah*2^32 + al and b = bh*2^32 + bl, the top 64 bits of a*b are ah*bh
// plus the carries out of the two cross terms and the low product.
//
// Run last in the optimizer, after LLVM's own passes. The shape it expands
// is what the code generator emits for `mulhi` (CodeGen_LLVM::multiply_high)
// and what LLVM's AggressiveInstCombine (foldMulHigh, since LLVM 23) turns
// any hand-written ladder back into. A scalar i128 product is one `mul`
// instruction on x86-64 and the i128 form is right for it; a *vector* of
// i128 products is legal nowhere, and the x86 backend scalarizes it -- an
// extract, a `mulx` and an insert per lane -- where the ladder is eight
// `vpmuludq` for a whole gang. The Halton sampler divides by its primes
// through a multiplier and this high product, and that scalarization cost a
// killeroo render two percent. Only vectors are touched; the doubled width
// has to be 128 or more, which no register holds.
struct ExpandVectorMulHigh : llvm::PassInfoMixin<ExpandVectorMulHigh> {
    llvm::PreservedAnalyses run(llvm::Function &function,
                                llvm::FunctionAnalysisManager &);
};

} // namespace bonsai
