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
