#pragma once

#include <ostream>
#include <set>
#include <string>

namespace bonsai {
namespace ir {

// The machine code is generated for, after Halide's Target: its
// architecture, its CPU and the instruction-set features it has, resolved
// once from `--triple` and `--mcpu` -- or from this machine when neither is
// named (see codegen::resolve_target) -- before anything is lowered.
//
// Two kinds of reader. The backends select instructions by it, as they
// always did through LLVM's own target machine (CodeGen_LLVM::has_feature,
// CodeGen_X86::dynamic_shuffle), and they are built from the same triple,
// CPU and feature string so that the two never disagree. The rewrites choose
// a strategy by it where the best code differs by machine -- how a
// traversal's sorted run of children is ordered and pushed, which on a
// machine with a register compress is Embree's compact-and-sort and on one
// without is its bit-scan loop (SSA/SortRecursion.cpp) -- the way Halide's
// lowering passes ask their Target what it has (natural_vector_size,
// has_feature) while its backends only select instructions. The choice lives
// in the rewrite rather than in each backend so that the traversal a
// schedule produces can be read in the SSA dump and pinned by a golden per
// machine, and so that the stack and loop it builds are written once.
struct Target {
    enum class Arch {
        Unknown,
        X86,
        ARM,
        NVPTX,
    };
    Arch arch = Arch::Unknown;
    // The pointer width in bits.
    int bits = 64;

    // What LLVM is told, spelled as LLVM spells it: the triple, the CPU name,
    // and the feature string -- the host's whole list, `+avx2,-sse4a,...`,
    // when the target is this machine, and empty for a named CPU, whose
    // features LLVM knows from the name.
    std::string triple;
    std::string cpu;
    std::string llvm_features;
    // Every feature the machine has by LLVM's name (`avx512f`, `neon`,
    // `sve`), the ones the CPU implies included: what has_feature answers.
    std::set<std::string> features;
    // Whether the target is this machine, features and all, rather than one
    // `--triple` or `--mcpu` named. Following the host is what lets the
    // backend prefer 256-bit vectors where the machine does and probe the
    // host's libraries (CodeGen_LLVM::make_target_machine); naming the
    // machine is what makes generated code the same on another one, which
    // is why the tests that diff it name one.
    bool follows_host = false;

    bool has_feature(const std::string &feature) const;

    // The questions the rewrites decide by.
    //
    // Whether one instruction packs the lanes a mask has on to the front of
    // a register (ir::Intrinsic::compress): AVX-512's `vpcompressd` and
    // `vpcompressq`, which need avx512vl for a vector narrower than 512
    // bits, and SVE's `compact`. AVX2 and NEON have nothing, and LLVM
    // expands the operation into a store and a load per lane.
    bool has_compress() const;
    // Whether one instruction permutes a vector by a vector of run-time
    // indices (ir::Intrinsic::permute): `vpermd`/`vpermps` from AVX2 and
    // the wider ones from AVX-512 (CodeGen_X86::dynamic_shuffle); NEON's
    // `tbl` over bytes; SVE's `tbl`.
    bool has_variable_permute() const;
    // The widest vector register in bits: 512 with AVX-512, 256 with AVX,
    // 128 with SSE2 or NEON, and 0 for a machine with no vector registers.
    unsigned vector_bits() const;
};

std::ostream &operator<<(std::ostream &os, Target::Arch arch);
// The triple and the CPU, and whether they are the host's.
std::ostream &operator<<(std::ostream &os, const Target &target);

} // namespace ir
} // namespace bonsai
