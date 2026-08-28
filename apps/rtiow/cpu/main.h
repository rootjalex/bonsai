#pragma once

#include <cstdint>
#include "runtime/bonsai_cpp.h"


extern "C" {
typedef float float3 __attribute__((ext_vector_type(3)));
struct Sphere {
    float3 center;
    float radius;
};
struct MaterialSphere {
    Sphere s;
    uint32_t material;
    float3 albedo;
    float fuzz;
};
typedef uint8_t uint8_t2 __attribute__((ext_vector_type(2)));
struct _tree_layout1 {
    float3 center;
    float radius;
    uint8_t nPrims;
    uint8_t axis;
    uint8_t2 split0on_nPrims;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    MaterialSphere* prims;
    uint32_t nCount;
    _tree_layout1* group0_index;
} __attribute__((packed));
struct _tree_layout2 {
    uint16_t offset;
} __attribute__((packed));
struct _tree_layout3 {
    uint16_t pOffset;
} __attribute__((packed));
typedef int32_t int32_t3 __attribute__((ext_vector_type(3)));
struct Camera {
    float aspect_ratio;
    int32_t width;
    uint32_t samples_per_pixel = 100;
    int32_t max_depth = 10;
    float vfov = 90.0f;
    float3 lookfrom = float3{0.0f, 0.0f, 0.0f};
    float3 lookat = float3{0.0f, 0.0f, -1.0f};
    float3 vup = float3{0.0f, 1.0f, 0.0f};
    float defocus_angle = 0.0f;
    float focus_dist = 10.0f;
};

void bounding_sphere(Sphere& _ret0, const Sphere& a, const Sphere& b);
int32_t3** image(const Camera& c, const _tree_layout0& spheres);
}
