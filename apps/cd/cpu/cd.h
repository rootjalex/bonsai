#pragma once

#include <cstdint>
#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
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
struct Arm_Interior {
    uint32_t offset;
} __attribute__((packed));
struct Arm_Leaf {
    uint32_t poffset;
} __attribute__((packed));
using vec4_uint8_t = vector<uint8_t, 4>;
struct Nodes {
    vec3_float low;
    vec3_float high;
    uint16_t nprims;
    uint8_t axis;
    uint8_t pad0;
    vec4_uint8_t split0on_nprims;
} __attribute__((packed));
struct Triangles1 {
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t node_count;
    Nodes* nodes;
} __attribute__((packed));
struct Triangles2 {
    uint32_t primitive_count;
    Triangle* primitives;
    uint32_t node_count;
    Nodes* nodes;
} __attribute__((packed));

Triangles1 build_triangles1(const BVH* CT);
Triangles2 build_triangles2(const BVH* CT);
std::vector<std::tuple<Triangle, Triangle>> collisions(const Triangles1* triangles1, const Triangles2* triangles2);
