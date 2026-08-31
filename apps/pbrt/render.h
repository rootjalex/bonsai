#pragma once

#include "runtime/bonsai_cpp.h"
#include <cstdint>

extern "C" {
typedef float float3 __attribute__((ext_vector_type(3)));
struct Sphere {
    float3 center;
    float radius;
};
struct Sph {
    Sphere s;
};
struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};
struct Tri {
    Triangle t;
};
union Shape_payload {
    Sph Sph;
    Tri Tri;
};
struct Shape {
    uint8_t tag;
    Shape_payload payload;
};
struct _tree_layout1 {
    float3 low;
    float3 high;
    uint16_t nPrims;
    uint8_t axis;
    uint8_t pad0;
    std::array<uint8_t, 4> split0on_nPrims;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    Shape *prims;
    uint32_t nCount;
    _tree_layout1 *group0_index;
} __attribute__((packed));
struct _tree_layout2 {
    uint32_t offset;
} __attribute__((packed));
struct _tree_layout3 {
    uint32_t pOffset;
} __attribute__((packed));
typedef float float4 __attribute__((ext_vector_type(4)));
struct Transform {
    float4 r0;
    float4 r1;
    float4 r2;
    float4 r3;
};
struct PerspectiveCamera {
    Transform camera_from_raster;
    Transform render_from_camera;
};

void Shape_Sph(Shape &_ret0, const Sphere &s);
void Shape_Tri(Shape &_ret1, const Triangle &t);
void render(const PerspectiveCamera &camera, const uint32_t width,
            const uint32_t height, float3 *out, const _tree_layout0 &shapes);
}
