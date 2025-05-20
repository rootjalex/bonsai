#pragma once

#include <cstdint>

extern "C" {
struct Statistics {
    float solMean;
    float solMean2;
    float totalFirstSourceContribution;
    uint32_t nSolEstimates;
    uint32_t totalWalkLength;
    uint32_t totalSplits;
    float firstSphereRadius;
};
struct PDE {
    float absCoeff;
    float freq;
    bool hasReflBoundCond = false;
};
typedef float vec3_float __attribute__((vector_size(12)));
struct Box {
    vec3_float low;
    vec3_float high;
};
struct WalkSettings {
    Box box;
    float epsShellAbs;
    float epsShellRefl;
    float silPrecision;
    float russianRouletteThreshold;
    uint32_t maxWalkLength;
    int32_t stepsBeforeApplyingTikhonov;
    uint8_t flags;
};
struct SamplePoint {
    vec3_float pt;
    vec3_float normal;
    float pdf;
    float distToAbs;
    float distToRefl;
    uint8_t type_and_quantity;
};
struct Triangle {
    vec3_float p0;
    vec3_float p1;
    vec3_float p2;
};
typedef uint8_t vec2_uint8_t __attribute__((vector_size(2)));
struct _tree_layout1 {
    vec3_float low;
    uint32_t pad0;
    vec3_float high;
    uint8_t nPrims;
    uint8_t axis;
    vec2_uint8_t split0on_nPrims;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    Triangle * prims;
    uint32_t count;
    _tree_layout1 * group0_index;
} __attribute__((packed));

Statistics * solve(const PDE& pde, const WalkSettings& s, const uint32_t n, const SamplePoint * pts, const uint32_t nWalks, const _tree_layout0& tris);
}

inline std::ostream& operator<<(std::ostream& os, const Statistics& stats) {
    os << "Statistics {\n";
    os << "  solMean: " << stats.solMean << "\n";
    os << "  solVariance: " << stats.solMean2 << "\n";
    os << "  totalFirstSourceContribution: " << stats.totalFirstSourceContribution << "\n";
    os << "  nSolEstimates: " << stats.nSolEstimates << "\n";
    os << "  totalWalkLength: " << stats.totalWalkLength << "\n";
    os << "  totalSplits: " << stats.totalSplits << "\n";
    os << "  firstSphereRadius: " << stats.firstSphereRadius << "\n";
    os << "}";
    return os;
}

enum class SampleType : uint8_t {
    InDomain = 0,
    OnAbsorbingBoundary = 1,
    OnReflectingBoundary = 2
};

enum class EstimationQuantity : uint8_t {
    Solution = 0,
    SolutionAndGradient = 1,
    None = 2
};

inline const char* to_string(SampleType type) {
    switch (type) {
        case SampleType::InDomain: return "InDomain";
        case SampleType::OnAbsorbingBoundary: return "OnAbsorbingBoundary";
        case SampleType::OnReflectingBoundary: return "OnReflectingBoundary";
        default: return "Unknown";
    }
}

inline const char* to_string(EstimationQuantity eq) {
    switch (eq) {
        case EstimationQuantity::Solution: return "Solution";
        case EstimationQuantity::SolutionAndGradient: return "SolutionAndGradient";
        case EstimationQuantity::None: return "None";
        default: return "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const SamplePoint& sp) {
    SampleType type = static_cast<SampleType>(sp.type_and_quantity & 0b00000011);              // bits 0–1
    EstimationQuantity quantity = static_cast<EstimationQuantity>((sp.type_and_quantity >> 2) & 0b11); // bits 2–3
    bool aligned = (sp.type_and_quantity >> 4) & 0x1;                                           // bit 4

    os << "SamplePoint {\n";
    os << "  pt: (" << sp.pt[0] << ", " << sp.pt[1] << ", " << sp.pt[2] << ")\n";
    os << "  normal: (" << sp.normal[0] << ", " << sp.normal[1] << ", " << sp.normal[2] << ")\n";
    os << "  pdf: " << sp.pdf << "\n";
    os << "  distToAbsorbingBoundary: " << sp.distToAbs << "\n";
    os << "  distToReflectingBoundary: " << sp.distToRefl << "\n";
    os << "  type: " << to_string(type) << "\n";
    os << "  estimationQuantity: " << to_string(quantity) << "\n";
    os << "  estimateBoundaryNormalAligned: " << (aligned ? "true" : "false") << "\n";
    os << "}";
    return os;
}

extern "C" void print_u64x4(uint64_t *vec) {
    std::cout << "Seed: [";
    for (int i = 0; i < 4; ++i) {
        std::cout << vec[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]\n";
}
