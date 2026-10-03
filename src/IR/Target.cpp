#include "IR/Target.h"

namespace bonsai {
namespace ir {

bool Target::has_feature(const std::string &feature) const {
    return features.contains(feature);
}

bool Target::has_compress() const {
    switch (arch) {
    case Arch::X86:
        // The 512-bit forms come with avx512f; the 256- and 128-bit forms a
        // traversal of four or eight children uses need avx512vl as well,
        // which every AVX-512 CPU but Knights Landing has.
        return has_feature("avx512f") && has_feature("avx512vl");
    case Arch::ARM:
        return has_feature("sve");
    case Arch::NVPTX:
    case Arch::Unknown:
        return false;
    }
    return false;
}

bool Target::has_variable_permute() const {
    switch (arch) {
    case Arch::X86:
        return has_feature("avx2");
    case Arch::ARM:
        return has_feature("neon") || has_feature("sve");
    case Arch::NVPTX:
    case Arch::Unknown:
        return false;
    }
    return false;
}

unsigned Target::vector_bits() const {
    switch (arch) {
    case Arch::X86:
        if (has_feature("avx512f")) {
            return 512;
        }
        if (has_feature("avx")) {
            return 256;
        }
        return 128; // SSE2 is the x86-64 baseline
    case Arch::ARM:
        // SVE's registers are as wide as the implementation makes them, 128
        // bits at least; NEON's are 128.
        return (has_feature("neon") || has_feature("sve")) ? 128 : 0;
    case Arch::NVPTX:
    case Arch::Unknown:
        return 0;
    }
    return 0;
}

std::ostream &operator<<(std::ostream &os, Target::Arch arch) {
    switch (arch) {
    case Target::Arch::Unknown:
        return os << "unknown";
    case Target::Arch::X86:
        return os << "x86";
    case Target::Arch::ARM:
        return os << "arm";
    case Target::Arch::NVPTX:
        return os << "nvptx";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Target &target) {
    os << target.triple;
    if (!target.cpu.empty()) {
        os << " " << target.cpu;
    }
    if (target.follows_host) {
        os << " (host)";
    }
    return os;
}

} // namespace ir
} // namespace bonsai
