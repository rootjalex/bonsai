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
using float3x8 = vector<float3, 8>;
struct Interior {
    std::array<BVH*, 8> children;
    float3x8 lo;
    float3x8 hi;
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
    uint8_t nprims;
    Triangle* data;
};
struct Point {
    float3 vec;
};
using bool3 = vector<bool, 3>;
using float2 = vector<float, 2>;
using float3x2 = vector<float3, 2>;
using float3x4 = vector<float3, 4>;
using float4 = vector<float, 4>;
using float4x3 = vector<float4, 3>;
using float5 = vector<float, 5>;
using float6 = vector<float, 6>;
using float7 = vector<float, 7>;
using float8 = vector<float, 8>;
using float9 = vector<float, 9>;
using int8_t3 = vector<int8_t, 3>;
using int8_t4 = vector<int8_t, 4>;
using int8_t4x3 = vector<int8_t4, 3>;
using int8_t9 = vector<int8_t, 9>;
using uint8_t3 = vector<uint8_t, 3>;
struct Qbox3 {
    uint8_t3 lo;
    uint8_t3 hi;
};
using Qbox38 = vector<Qbox3, 8>;
using uint32_t8 = vector<uint32_t, 8>;
struct Interiors {
    float3 mlo;
    float3 mex;
    Qbox38 child_bounds;
    uint32_t8 children;
} __attribute__((packed));
struct Triangles {
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t interior_count;
    Interiors* interiors;
} __attribute__((packed));
using uint16_t3 = vector<uint16_t, 3>;
using uint64_t4 = vector<uint64_t, 4>;
using uint8_t3x4 = vector<uint8_t3, 4>;
using uint16_t3 = vector<uint16_t, 3>;

Triangles build_triangles(const BVH* __restrict__ CT);
Triangle closest_point(const Point* __restrict__ p, const Triangles* __restrict__ triangles);
