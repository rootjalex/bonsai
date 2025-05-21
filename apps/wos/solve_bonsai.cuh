#pragma once

#include <cuda_runtime.h>

struct AABB {
    float3 low;
    float3 high;
};

struct Box {
    float3 low;
    float3 high;
};

struct HarmonicGreensBall {
    float3 center;
    float radius;
    float rClamp;
};

struct PDE {
    float absCoeff;
    float freq;
    bool hasReflBoundCond = false;
};

struct Point {
    float3 vec;
};

struct SamplePoint {
    float3 pt;
    float3 normal;
    float pdf;
    float distToAbs;
    float distToRefl;
    uint8_t type_and_quantity;
};

struct Statistics {
    float solMean;
    float solMean2;
    float totalFirstSourceContribution;
    uint32_t nSolEstimates;
    uint32_t totalWalkLength;
    uint32_t totalSplits;
    float firstSphereRadius;
};

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};

struct WalkResults {
    float3 pt;
    float totalSourceContribution;
    float throughput;
    uint32_t walkLength;
    float distToAbs;
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

struct __tuple_0 {
    Point _field0;
    Point _field1;
};

struct __tuple_1 {
    float3 _field0;
    float _field1;
    float _field2;
};

struct __tuple_2 {
    bool _field0;
    float _field1;
};

struct _option0 {
    WalkResults value;
    bool set = false;
};

struct _tree_layout1 {
    float3 low;
    uint32_t nPrims;
    float3 high;
    uint16_t axis;
    uchar2 split0on_nPrims;
} __attribute__((packed));

struct _tree_layout0 {
    uint32_t pCount;
    Triangle *prims; // of size pCount
    uint32_t count;
    _tree_layout1 *group0_index; // of size count
} __attribute__((packed));

struct _tree_layout2 {
    uint16_t offset;
} __attribute__((packed));

struct _tree_layout3 {
    uint16_t pOffset;
} __attribute__((packed));

struct _ctx0 {
    uint32_t n;
    Statistics *_alloc0; // of size n
    PDE *pde;
    WalkSettings *s;
    SamplePoint *pts; // of size n
    uint32_t nWalks;
    _tree_layout0 *tris;
};

Statistics *solve(PDE *pde, WalkSettings *s, uint32_t n, SamplePoint *pts,
                  uint32_t nWalks, _tree_layout0 *tris);
