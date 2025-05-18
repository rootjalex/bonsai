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
    float firstSphereRadius;
    uint8_t type_and_quantity;
};

Statistics * solve(const PDE& pde, const WalkSettings& s, const uint32_t n, const SamplePoint * pts, const uint32_t nWalks);
}
