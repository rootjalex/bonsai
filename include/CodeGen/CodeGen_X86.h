#pragma once

#include "CodeGen/CodeGen_LLVM.h"

#include <string>

namespace bonsai {

// The LLVM code generator for x86-64: CodeGen_LLVM, plus the places where
// naming the machine's own instruction beats LLVM's generic form. Chosen by
// make_llvm_codegen for an x86-64 triple; on a machine without the
// instructions it names -- a generic x86-64, which is what the goldens pin
// -- every method falls back to the target-agnostic one.
struct CodeGen_X86 : public CodeGen_LLVM {
  protected:
    // 512 with AVX-512, 256 with AVX, 128 otherwise: the machine's widest
    // register, which a gang of sixteen or eight 32-bit lanes fills.
    int vector_register_bits() const override;

    // A gather of eight or sixteen 32-bit elements as `vpgatherdd`/
    // `vgatherdps` with a base register, an index register of 32-bit
    // offsets and a scale, through the x86 intrinsic that takes exactly
    // those. LLVM's masked gather takes a vector of 64-bit addresses
    // instead, and recovers the base-plus-index form only when it can still
    // see the addresses being formed from a base and 32-bit indices; once
    // its own loop-invariant code motion has hoisted the address vector out
    // of a loop, it cannot, and the gather goes through two registers of
    // pointers (`vpgatherqd`) with the pointers kept live across the loop.
    // Naming the instruction keeps one register of indices live instead of
    // two of pointers, and the form is the same whatever LLVM moves.
    llvm::Value *gather_indexed(llvm::Type *elem_llvm, llvm::Value *base,
                                llvm::Value *indices, uint64_t scale,
                                uint64_t disp, uint64_t align, uint32_t lanes,
                                llvm::Value *mask,
                                const std::string &name) override;
};

} // namespace bonsai
