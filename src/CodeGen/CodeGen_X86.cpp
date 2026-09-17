#include "CodeGen/CodeGen_X86.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IntrinsicsX86.h>

namespace bonsai {

int CodeGen_X86::vector_register_bits() const {
    // The widest register the machine has: zmm with AVX-512, ymm with AVX,
    // xmm otherwise. A gang of sixteen 32-bit lanes is one zmm register, and
    // telling LLVM to prefer less (see optimize_module) would split every one
    // of its operations in two.
    if (has_feature("avx512f")) {
        return 512;
    }
    if (has_feature("avx")) {
        return 256;
    }
    return 128;
}

llvm::Value *CodeGen_X86::gather_indexed(llvm::Type *elem_llvm,
                                         llvm::Value *base,
                                         llvm::Value *indices, uint64_t scale,
                                         uint64_t disp, uint64_t align,
                                         uint32_t lanes, llvm::Value *mask,
                                         const std::string &name) {
    // Eight or sixteen 32-bit lanes with 32-bit indices at a scale the
    // addressing has, on a machine with the instruction: the shapes a gang's
    // gathers have. Eight lanes is a 256-bit gather, AVX-512VL's with its mask
    // in a k register or AVX2's with the mask in a vector; sixteen is the
    // 512-bit gather AVX-512F has. Anything else takes the generic form.
    auto *indices_t = llvm::dyn_cast<llvm::FixedVectorType>(indices->getType());
    const bool word = elem_llvm->isIntegerTy(32) || elem_llvm->isFloatTy();
    const bool scaled = scale == 1 || scale == 2 || scale == 4 || scale == 8;
    const bool avx512vl = has_feature("avx512vl");
    const bool eight = lanes == 8 && (avx512vl || has_feature("avx2"));
    const bool sixteen = lanes == 16 && has_feature("avx512f");
    if (!word || !scaled || !(eight || sixteen) || indices_t == nullptr ||
        indices_t->getNumElements() != lanes ||
        !indices_t->getElementType()->isIntegerTy(32)) {
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
    llvm::Type *vtype = llvm::FixedVectorType::get(elem_llvm, lanes);
    llvm::Value *on = mask != nullptr
                          ? mask
                          : llvm::Constant::getAllOnesValue(
                                llvm::FixedVectorType::get(i1_t, lanes));
    llvm::Value *off_lanes = llvm::Constant::getNullValue(vtype);
    const bool is_float = elem_llvm->isFloatTy();

    if (sixteen || avx512vl) {
        // vgatherdps/vpgatherdd {zmm,ymm}{k}, [base + index * scale]: the
        // mask is a k register, and the lanes it leaves off keep the first
        // operand, zero here as LLVM's passthrough is.
        const llvm::Intrinsic::ID id =
            sixteen ? (is_float ? llvm::Intrinsic::x86_avx512_mask_gather_dps_512
                                : llvm::Intrinsic::x86_avx512_mask_gather_dpi_512)
                    : (is_float ? llvm::Intrinsic::x86_avx512_mask_gather3siv8_sf
                                : llvm::Intrinsic::x86_avx512_mask_gather3siv8_si);
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
        builder->CreateSExt(on, llvm::FixedVectorType::get(i32_t, lanes));
    if (is_float) {
        sign = builder->CreateBitCast(sign, vtype);
    }
    return builder->CreateIntrinsic(
        id, {}, {off_lanes, at, indices, sign, llvm::ConstantInt::get(i8_t, scale)},
        nullptr, name);
}

} // namespace bonsai
