#pragma once

#include <cstdint>
#include "runtime/bonsai_cpp.h"


extern "C" {
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
typedef float float3 __attribute__((ext_vector_type(3)));
struct Sphere {
    float3 center;
    float radius;
};

void render(const PerspectiveCamera& camera, const Sphere& s, const uint32_t width, const uint32_t height, float3* out);
}
