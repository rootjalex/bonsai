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
    uint32_t samples_per_pixel;
    int32_t max_depth;
    float vfov;
    vec3_float lookfrom;
    vec3_float lookat;
    vec3_float vup;
    float defocus_angle;
    float focus_dist;
};
struct MaterialSphere {
    Sphere s;
    uint32_t material;
    vec3_float albedo;
    float fuzz;
};
typedef uint8_t vec2_uint8_t __attribute__((vector_size(2)));
struct _spheres_layout0 {
    vec3_float center;
    float radius;
    uint16_t nPrims;
    vec2_uint8_t spheres_spliton_nPrims;
} __attribute__((packed));
struct _spheres_layout1 {
    uint32_t pCount;
    MaterialSphere * prims;
    uint32_t count;
    _spheres_layout0 * spheres_index;
} __attribute__((packed));

void bounding_sphere(Sphere& _ret0, const Sphere& a, const Sphere& b);
vec3_int32_t * * image(const Camera& c, const _spheres_layout1& spheres);
}
