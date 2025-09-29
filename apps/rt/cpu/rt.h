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
using vec8_vec3_float = vector<vec3_float, 8>;
struct Interior {
    std::array<BVH*, 8> children;
    vec8_vec3_float lo;
    vec8_vec3_float hi;
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
    uint8_t nprims;
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
using vec3_bool = vector<bool, 3>;
using vec2_float = vector<float, 2>;
using vec2_vec3_float = vector<vec3_float, 2>;
using vec4_vec3_float = vector<vec3_float, 4>;
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
using vec8_uint64_t = vector<uint64_t, 8>;
struct alignas(32) Interiors {
    vec8_vec3_float lo;
    vec8_vec3_float hi;
    vec8_uint64_t children;
} __attribute__((packed));
struct Triangles {
    uint64_t primitive_count;
    Triangle* primitives;
    Interiors* interiors;
} __attribute__((packed));
using vec4_uint64_t = vector<uint64_t, 4>;
using vec3_uint8_t = vector<uint8_t, 3>;
using vec4_vec3_uint8_t = vector<vec3_uint8_t, 4>;
using vec3_uint16_t = vector<uint16_t, 3>;

Triangles build_triangles(const BVH* __restrict__ CT);
std::optional<Triangle>* chrt(const int64_t n, const Ray* rays, const Triangles* __restrict__ triangles);
std::optional<Triangle> trace(const Ray* __restrict__ ray, const Triangles* __restrict__ triangles);
