#pragma once

#include <cstdint>
#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include "bonsai_cpp.h"

using vec3_float = vector<float, 3>;
struct AABB {
    vec3_float low;
    vec3_float high;
};
struct Camera {
    float aspect_ratio;
    int32_t width;
    uint32_t samples_per_pixel = 100;
    int32_t max_depth = 10;
    float vfov = 90;
    vec3_float lookfrom = vec3_float{0.0f, 0.0f, 0.0f};
    vec3_float lookat = vec3_float{0.0f, 0.0f, -1};
    vec3_float vup = vec3_float{0.0f, 1.0f, 0.0f};
    float defocus_angle = 0.0f;
    float focus_dist = 10;
};
struct FInterval {
    float low;
    float high;
};
struct Sphere {
    vec3_float center;
    float radius;
};
struct MaterialSphere {
    Sphere s;
    uint32_t material;
    vec3_float albedo;
    float fuzz;
};
struct Point {
    vec3_float vec;
};
struct Ray {
    vec3_float o;
    vec3_float d;
    float tmax = std::numeric_limits<float>::infinity();
};
struct Triangle {
    vec3_float p0;
    vec3_float p1;
    vec3_float p2;
};
struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};
struct Arm_Interior {
    uint16_t offset;
} __attribute__((packed));
struct Arm_Leaf {
    uint16_t poffset;
} __attribute__((packed));
using vec3_bool = vector<bool, 3>;
using vec2_float = vector<float, 2>;
using vec4_vec3_float = vector<vec3_float, 4>;
using vec4_float = vector<float, 4>;
using vec3_vec4_float = vector<vec4_float, 3>;
using vec5_float = vector<float, 5>;
using vec6_float = vector<float, 6>;
using vec7_float = vector<float, 7>;
using vec8_float = vector<float, 8>;
using vec9_float = vector<float, 9>;
struct Hit_record {
    vec3_float p;
    vec3_float normal;
    float t;
    bool front_face;
};
using vec3_int8_t = vector<int8_t, 3>;
using vec4_int8_t = vector<int8_t, 4>;
using vec3_vec4_int8_t = vector<vec4_int8_t, 3>;
using vec9_int8_t = vector<int8_t, 9>;
using vec2_uint8_t = vector<uint8_t, 2>;
struct Nodes {
    vec3_float center;
    float radius;
    uint8_t nprims;
    uint8_t pad0;
    vec2_uint8_t split0on_nprims;
} __attribute__((packed));
struct Scatter_record {
    vec3_float attenuation;
    Ray ray;
    bool hit;
};
struct Spheres {
    uint32_t primitive_count;
    MaterialSphere* primitives;
    uint32_t node_count;
    Nodes* nodes;
} __attribute__((packed));
using vec4_uint64_t = vector<uint64_t, 4>;
using vec3_uint8_t = vector<uint8_t, 3>;
using vec4_vec3_uint8_t = vector<vec3_uint8_t, 4>;
using vec3__Float16 = vector<_Float16, 3>;
using vec3_int32_t = vector<int32_t, 3>;

Sphere bounding_sphere(const Sphere* a, const Sphere* b);
vec3_int32_t** image(const Camera* c, const Spheres* spheres);
