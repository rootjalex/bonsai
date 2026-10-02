#pragma once

#include "CodeGen/CodeGen_LLVM.h"

#include <optional>
#include <set>
#include <string>

namespace bonsai {

// The LLVM code generator for x86-64: CodeGen_LLVM, plus every decision
// that is a fact about this machine -- what its return registers hold, how
// wide its vector registers are, which of its own instructions beat LLVM's
// generic form, and what its C library provides. Chosen by make_llvm_codegen
// for an x86-64 triple; on a machine without the instructions it names -- a
// generic x86-64, which is what the goldens pin -- the instruction choices
// fall back to the target-agnostic ones, and the ABI facts still hold.
struct CodeGen_X86 : public CodeGen_LLVM {
  protected:
    // The aggregates the SysV return convention cannot hold in registers.
    //
    // A first-class aggregate return is legalised by handing each leaf its
    // own register: XMM0 and XMM1 for floating-point and vector leaves and
    // EAX, EDX and ECX for integer ones, and the third floating-point leaf
    // goes on the *x87 stack* -- an `fstps` on the way out and an `flds` on
    // the way in, four billion of them in a render of the pavilion, and the
    // FP scheduler stalling around them. C++ never sees this because its ABI
    // returns anything over two eightbytes through a pointer the caller
    // provides. So does this, for exactly the aggregates the return registers
    // cannot hold; the ones they can -- `option[i32]`, a pair of floats --
    // keep the register return, which is cheaper than a store and a load.
    llvm::Type *indirect_return_type(const ir::Type &ret_type) override;

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

    // A vector of integer lanes divided through doubles (or floats, for
    // narrow lanes) and truncated back, which is exact for lanes of 32 bits
    // or fewer. x86 has no vector integer division: LLVM takes one apart
    // into a `div` per lane -- an extract, a twenty-cycle divide and an
    // insert, sixteen times over for a gang -- where a vector of doubles is
    // divided in one instruction. Nothing for scalars, constant divisors
    // and lanes wider than 32 bits, which stay an integer division.
    llvm::Value *vector_int_division(llvm::Value *a, llvm::Value *b,
                                     bool is_signed, bool remainder) override;

    // A single-precision reciprocal as Embree takes it on this machine: the
    // estimate instruction -- `vrcp14ps` with AVX-512VL, fourteen bits, or
    // `rcpps`, twelve -- refined by one Newton step, `r + r * (1 - x * r)`,
    // as two fused multiply-adds (common/simd/vfloat4_sse2.h, vfloat8_avx.h,
    // vfloat16_avx512.h, math/vec3fa.h). A scalar or a short vector is
    // padded to the four lanes of an xmm register, as Embree's Vec3fa is;
    // a double, or a width the machine has no estimate for, keeps the
    // division.
    llvm::Value *reciprocal(llvm::Value *x, const std::string &name) override;

    // A vector permuted by a vector of indices as the one instruction the
    // machine has for the shape: `vpermd` for eight 32-bit lanes (AVX2) and
    // sixteen (AVX-512F), `vpermilps` for four (AVX); `vpermq` for eight
    // 64-bit lanes (AVX-512F) and four (AVX-512VL), `vpermilpd` for two
    // (AVX), and on AVX2 alone four 64-bit lanes as pairs of 32-bit halves
    // through `vpermd`; `vpermw` for 16-bit lanes (AVX-512BW). Embree's
    // `permute` and `permutex2var` (common/simd/vint8_avx2.h,
    // vllong4_avx2.h) are these. Any other shape is the base's extract per
    // lane.
    llvm::Value *dynamic_shuffle(llvm::Value *vec, llvm::Value *indices,
                                 const std::string &name) override;

    // Whether dynamic_shuffle above has an instruction for a vector of
    // `lanes` lanes of `bits` bits on this machine.
    bool has_permute(unsigned bits, unsigned lanes) const;

    // One lane of a vector at an index computed at run time: the index
    // broadcast, the vector permuted by it and lane zero taken -- a permute
    // and a move -- where LLVM's variable extract stores the vector to the
    // stack and loads the lane back, and the load waits on the store.
    // A leaf's argmin picks the nearest lane's triangle this way. A constant
    // index and a shape without a permute keep the base's extract.
    llvm::Value *extract_lane(llvm::Value *vec, llvm::Value *idx) override;

    // glibc's libmvec: the vector entry points of libm, asked for rather
    // than assumed (see the definition). Only on Linux, which is where glibc
    // is; a host without it has no vector maths library and says nothing.
    void probe_host_libraries() override;
    // The libmvec entry point for `name` on `lanes` lanes of `bits`-bit
    // floats with `arity` vector arguments, if the host has one: the Vector
    // Function ABI name, `_ZGVdN8v_sinf` for sinf on eight lanes for AVX2.
    std::optional<std::string> vector_math_symbol(const std::string &name,
                                                  uint32_t lanes, unsigned bits,
                                                  unsigned arity) const override;

  private:
    // What the host's libmvec provides, found by asking it. Every symbol
    // here exists in this machine's libmvec and is of an ISA level this
    // machine runs.
    std::set<std::string> host_vector_math;
};

} // namespace bonsai
