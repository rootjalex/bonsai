#include "CodeGen/CodeGen_X86.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IntrinsicsX86.h>

namespace bonsai {

llvm::Value *CodeGen_X86::gather_indexed(llvm::Type *elem_llvm,
                                         llvm::Value *base,
                                         llvm::Value *indices, uint64_t scale,
                                         uint64_t disp, uint64_t align,
                                         uint32_t lanes, llvm::Value *mask,
                                         const std::string &name) {
    // Eight 32-bit lanes with 32-bit indices at a scale the addressing has,
    // on a machine with the instruction: the shape a gang's gathers have.
    // Anything else takes the generic form.
    auto *indices_t = llvm::dyn_cast<llvm::FixedVectorType>(indices->getType());
    const bool word = elem_llvm->isIntegerTy(32) || elem_llvm->isFloatTy();
    const bool scaled = scale == 1 || scale == 2 || scale == 4 || scale == 8;
    const bool avx512 = has_feature("avx512vl");
    const bool avx2 = has_feature("avx2");
    if (!word || !scaled || lanes != 8 || indices_t == nullptr ||
        indices_t->getNumElements() != 8 ||
        !indices_t->getElementType()->isIntegerTy(32) || !(avx512 || avx2)) {
        return CodeGen_LLVM::gather_indexed(elem_llvm, base, indices, scale, disp,
                                            align, lanes, mask, name);
    }

    // The displacement goes into the base: the intrinsic has base, index
    // and scale, and a scalar add is what a displacement costs.
    llvm::Value *at = base;
    if (disp != 0) {
        at = builder->CreateInBoundsGEP(
            i8_t, base, llvm::ConstantInt::get(i64_t, disp), name + "_base");
    }
    llvm::Type *vtype = llvm::FixedVectorType::get(elem_llvm, 8);
    llvm::Value *on = mask != nullptr
                          ? mask
                          : llvm::Constant::getAllOnesValue(
                                llvm::FixedVectorType::get(i1_t, 8));
    llvm::Value *off_lanes = llvm::Constant::getNullValue(vtype);
    const bool is_float = elem_llvm->isFloatTy();

    if (avx512) {
        // vgatherdps/vpgatherdd ymm{k}, [base + ymm_index * scale]: the mask
        // is a k register, and the lanes it leaves off keep the first
        // operand, zero here as LLVM's passthrough is.
        const llvm::Intrinsic::ID id =
            is_float ? llvm::Intrinsic::x86_avx512_mask_gather3siv8_sf
                     : llvm::Intrinsic::x86_avx512_mask_gather3siv8_si;
        return builder->CreateIntrinsic(
            id, {},
            {off_lanes, at, indices, on, llvm::ConstantInt::get(i32_t, scale)},
            nullptr, name);
    }
    // AVX2's gather takes its mask as a vector of the result's type, on in
    // the sign bit: the lanes to load are all ones.
    const llvm::Intrinsic::ID id = is_float
                                       ? llvm::Intrinsic::x86_avx2_gather_d_ps_256
                                       : llvm::Intrinsic::x86_avx2_gather_d_d_256;
    llvm::Value *sign =
        builder->CreateSExt(on, llvm::FixedVectorType::get(i32_t, 8));
    if (is_float) {
        sign = builder->CreateBitCast(sign, vtype);
    }
    return builder->CreateIntrinsic(
        id, {}, {off_lanes, at, indices, sign, llvm::ConstantInt::get(i8_t, scale)},
        nullptr, name);
}

} // namespace bonsai
