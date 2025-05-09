#pragma once

#include <cstdint>

extern "C" {
typedef float vec3_float __attribute__((vector_size(12)));
struct Sphere {
    vec3_float center;
    float radius;
};
typedef int32_t vec3_int32_t __attribute__((vector_size(12)));
struct Camera {
    float aspect_ratio;
    int32_t width;
    uint32_t samples_per_pixel = 100;
    int32_t max_depth = 10;
    float vfov = 90;
    vec3_float lookfrom = vec3_float{0, 0, 0};
    vec3_float lookat = vec3_float{0, 0, -1};
    vec3_float vup = vec3_float{0, 1, 0};
    float defocus_angle = 0;
    float focus_dist = 10;
};
struct MaterialSphere {
    Sphere s;
    uint32_t material;
    vec3_float albedo;
    float fuzz;
};
typedef uint8_t vec2_uint8_t __attribute__((vector_size(2)));
struct _spheres_layout1 {
    vec3_float center;
    float radius;
    uint16_t nPrims;
    vec2_uint8_t spheres_spliton_nPrims;
} __attribute__((packed));
struct _spheres_layout0 {
    uint32_t pCount;
    MaterialSphere * prims;
    uint32_t count;
    _spheres_layout1 * spheres_index;
} __attribute__((packed));

void bounding_sphere(Sphere& _ret0, const Sphere& a, const Sphere& b);
vec3_int32_t * * image(const Camera& c, const _spheres_layout0& spheres);
}
