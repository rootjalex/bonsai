#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <optional>
#include <iostream>
#include <string>
#include <tuple>
#include <variant>
#include "bonsai_cpp.h"

struct Interior;
struct Leaf;
using BVH = std::variant<Interior, Leaf>;
using float3 = vector<float, 3>;
struct Interior {
    float3 low;
    float3 high;
    BVH* left;
    BVH* right;
};
struct AABB {
    float3 low;
    float3 high;
};
struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};
struct Leaf {
    float3 low;
    float3 high;
    uint16_t nprims;
    Triangle* data;
};
struct Arm_Interior {
    uint32_t offset;
} __attribute__((packed));
struct Arm_Leaf {
    uint32_t poffset;
} __attribute__((packed));
using uint8_t4 = vector<uint8_t, 4>;
struct alignas(32) Nodes1 {
    float3 low;
    float3 high;
    uint16_t nprims;
    uint8_t axis;
    uint8_t pad0;
    uint8_t4 split0on_nprims;
} __attribute__((packed));
struct alignas(32) Nodes2 {
    float3 low;
    float3 high;
    uint16_t nprims;
    uint8_t axis;
    uint8_t pad0;
    uint8_t4 split0on_nprims;
} __attribute__((packed));
struct Triangles1 {
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t node_count;
    Nodes1* nodes1;
} __attribute__((packed));
struct Triangles2 {
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t node_count;
    Nodes2* nodes2;
} __attribute__((packed));

Triangles1 build_triangles1(const BVH* __restrict__ CT);
Triangles2 build_triangles2(const BVH* __restrict__ CT);
std::vector<std::tuple<Triangle, Triangle>> collisions(const Triangles1* __restrict__ triangles1, const Triangles2* __restrict__ triangles2);
