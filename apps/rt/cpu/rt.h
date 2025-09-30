#pragma once

#include <array>
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
using vec3_float = vector<float, 3>;
struct Interior {
    vec3_float low;
    vec3_float high;
    BVH* left;
    BVH* right;
};
struct AABB {
    vec3_float low;
    vec3_float high;
};
struct Triangle {
    vec3_float p0;
    vec3_float p1;
    vec3_float p2;
};
struct Leaf {
    vec3_float low;
    vec3_float high;
    uint16_t nprims;
    Triangle* data;
};
struct FInterval {
    float low;
    float high;
};
struct Point {
    vec3_float vec;
};
struct Ray {
    vec3_float o;
    vec3_float d;
    float tmax = std::numeric_limits<float>::infinity();
};
struct Sphere {
    vec3_float center;
    float radius;
};
struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};
struct Arm_Interior {
    uint32_t offset;
} __attribute__((packed));
struct Arm_Leaf {
    uint32_t poffset;
} __attribute__((packed));
using vec3_bool = vector<bool, 3>;
using vec2_float = vector<float, 2>;
using vec2_vec3_float = vector<vec3_float, 2>;
using vec4_vec3_float = vector<vec3_float, 4>;
using vec8_vec3_float = vector<vec3_float, 8>;
using vec4_float = vector<float, 4>;
using vec3_vec4_float = vector<vec4_float, 3>;
using vec5_float = vector<float, 5>;
using vec6_float = vector<float, 6>;
using vec7_float = vector<float, 7>;
using vec8_float = vector<float, 8>;
using vec9_float = vector<float, 9>;
using vec3_int8_t = vector<int8_t, 3>;
using vec4_int8_t = vector<int8_t, 4>;
using vec3_vec4_int8_t = vector<vec4_int8_t, 3>;
using vec9_int8_t = vector<int8_t, 9>;
using vec4_uint8_t = vector<uint8_t, 4>;
struct alignas(32) Nodes {
    uint32_t q_min : 30;
    uint32_t q_max : 30;
    uint8_t nprims;
    vec4_uint8_t split0on_nprims;
} __attribute__((packed));
struct Triangles {
    vec3_float wlow;
    vec3_float whigh;
    vec3_float bins;
    vec3_float bins_inv;
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t node_count;
    Nodes* nodes;
} __attribute__((packed));
using vec4_uint64_t = vector<uint64_t, 4>;
using vec3_uint8_t = vector<uint8_t, 3>;
using vec4_vec3_uint8_t = vector<vec3_uint8_t, 4>;
using vec3_uint16_t = vector<uint16_t, 3>;

Triangles build_triangles(const BVH* __restrict__ CT);
std::optional<Triangle> trace(const Ray* __restrict__ ray, const Triangles* __restrict__ triangles);
