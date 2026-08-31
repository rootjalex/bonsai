#pragma once

#include "runtime/bonsai_cpp.h"
#include <cstdint>

#ifdef AJR_PROFILE
inline uint32_t distmin_aabb_counter = 0;
inline uint32_t distmin_triangle_counter = 0;

inline uint32_t fcpw_aabb_counter = 0;
inline uint32_t fcpw_triangle_counter = 0;

inline uint32_t cgal_aabb_counter = 0;
inline uint32_t cgal_triangle_counter = 0;

inline void ajr_profiler_reset() {
    distmin_aabb_counter = 0;
    distmin_triangle_counter = 0;

    fcpw_aabb_counter = 0;
    fcpw_triangle_counter = 0;

    cgal_aabb_counter = 0;
    cgal_triangle_counter = 0;
}
#endif

using float3 = vector<float, 3>;
struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};
struct _tree_layout1 {
    float3 low;
    uint32_t nPrims;
    float3 high;
    uint32_t offset;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    Triangle *prims;
    uint32_t nCount;
    _tree_layout1 *group0_index;
} __attribute__((packed));
struct _tree_layout2 {
} __attribute__((packed));
struct _tree_layout3 {
} __attribute__((packed));
struct Ray {
    float3 o;
    float3 d;
    float tmax = std::numeric_limits<float>::infinity();
};

std::optional<Triangle> trace(const Ray ray, const _tree_layout0 &triangles);
