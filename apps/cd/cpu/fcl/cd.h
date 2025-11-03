#pragma once

#include <cstdint>
#include "runtime/bonsai_cpp.h"

using float3 = vector<float, 3>;
struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};
struct _tree_layout1 {
    float3 low;
    uint32_t nPrims;
    float3 high;
    uint32_t offset;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    Triangle* prims;
    uint32_t nCount;
    _tree_layout1* group0_index;
} __attribute__((packed));
struct _tree_layout2 {
} __attribute__((packed));
struct _tree_layout3 {
} __attribute__((packed));
struct _tree_layout5 {
    float3 low;
    uint32_t nPrims;
    float3 high;
    uint32_t offset;
} __attribute__((packed));
struct _tree_layout4 {
    uint32_t pCount;
    Triangle* prims;
    uint32_t nCount;
    _tree_layout5* group0_index;
} __attribute__((packed));
struct _tree_layout6 {
} __attribute__((packed));

set<std::tuple<Triangle, Triangle>> collisions(const _tree_layout0& triangles1, const _tree_layout4& triangles2);
