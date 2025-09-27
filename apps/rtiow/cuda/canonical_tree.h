#pragma once

#include "cuda/rtiow.h"

using vec3_float = float3;
using f32 = float;

__host__ Sphere get_bounding_sphere(const BVH *node);
#ifndef __CUDA_ARCH__
__host__ Sphere get_bounding_sphere(const BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        const Interior &interior = std::get<Interior>(*node);
        return {interior.center, interior.radius};
    }
    if (std::holds_alternative<Leaf>(*node)) {
        const Leaf &leaf = std::get<Leaf>(*node);
        return {leaf.center, leaf.radius};
    }
    assert(false && "unexpected");
}
#endif // __CUDA_ARCH__

__host__ void free_canonical_tree(BVH *node);
#ifndef __CUDA_ARCH__
__host__ void free_canonical_tree(BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree(interior.left);
        free_canonical_tree(interior.right);
        free(&interior);
        return;
    }

    if (std::holds_alternative<Leaf>(*node)) {
        Leaf &leaf = std::get<Leaf>(*node);
        free(leaf.data);
        free(&leaf);
        return;
    }

    assert(false && "unexpected");
}
#endif // __CUDA_ARCH__

// Builds the canonical tree using a median split.
__host__ BVH *build_canonical_tree(std::vector<MaterialSphere> &spheres);
#ifndef __CUDA_ARCH__
__host__ BVH *build_canonical_tree(std::vector<MaterialSphere> &spheres) {
    constexpr uint32_t MAX_TREE_DEPTH = 64;
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth = 0) -> BVH * {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        if (count <= 2) {
            vec3_float center = spheres[low].s.center;
            f32 radius = spheres[low].s.radius;
            if (count == 2) {
                Sphere merged =
                    bounding_sphere(&spheres[low].s, &spheres[low + 1].s);
                center = merged.center;
                radius = merged.radius;
            }
            auto *data =
                (MaterialSphere *)(malloc(sizeof(MaterialSphere) * count));
            for (int i = 0; i < count; ++i) {
                data[i] = spheres[low + i];
            }
            return new BVH(Leaf{
                .center = center,
                .radius = radius,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Internal node
        vec3_float min_bound = spheres[low].s.center;
        vec3_float max_bound = spheres[low].s.center;

        for (uint32_t i = low + 1; i < high; ++i) {
            min_bound = min(min_bound, spheres[i].s.center);
            max_bound = max(max_bound, spheres[i].s.center);
        }

        // Choose axis with greatest extent
        vec3_float extent = max_bound - min_bound;
        int axis = 0;
        float ex = extent.x;
        float ey = extent.y;
        float ez = extent.z;

        if (ey > ex)
            axis = 1;
        if (ez > ((axis == 0) ? ex : ey))
            axis = 2;

        // Partition at midpoint along chosen axis
        auto mid_it = spheres.begin() + low + count / 2;
        std::nth_element(spheres.begin() + low, mid_it, spheres.begin() + high,
                         [&](const MaterialSphere &a, const MaterialSphere &b) {
                             if (axis == 0)
                                 return a.s.center.x < b.s.center.x;
                             if (axis == 1)
                                 return a.s.center.y < b.s.center.y;
                             return a.s.center.z < b.s.center.z;
                         });

        const uint32_t mid = low + count / 2;
        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        // Compute bounding volume
        Sphere a = get_bounding_sphere(left), b = get_bounding_sphere(right);
        Sphere merged = bounding_sphere(&a, &b);

        return new BVH(Interior{
            .center = merged.center,
            .radius = merged.radius,
            .left = left,
            .right = right,
        });
    };

    return partition(/*low=*/0, /*high=*/spheres.size(), /*depth=*/0);
}
#endif // __CUDA_ARCH__
