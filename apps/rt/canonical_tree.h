#pragma once

#include "cpu/rt.h"

#include <cstdint>
#include <limits>
#include <vector>

std::pair<vec3_float, vec3_float>
compute_aabb(uint32_t low, uint32_t high,
             const std::vector<Triangle> &triangles) {
    Triangle tri = triangles[low];
    vec3_float aabb_min = tri.p0;
    vec3_float aabb_max = tri.p0;
    for (uint32_t i = low; i < high; ++i) {
        Triangle t = triangles[i];
        for (vec3_float v : {t.p0, t.p1, t.p2}) {
            aabb_min = min(aabb_min, v);
            aabb_max = max(aabb_max, v);
        }
    }
    return {aabb_min, aabb_max};
}

float surface_area(const vec3_float &min, const vec3_float &max) {
    vec3_float extent = max - min;
    return 2.0f * (extent[0] * extent[1] + extent[0] * extent[2] +
                   extent[1] * extent[2]);
}

vec3_float triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

std::pair<vec3_float, vec3_float> triangle_bounds(const Triangle &tri) {
    vec3_float min_ = min(min(tri.p0, tri.p1), tri.p2);
    vec3_float max_ = max(max(tri.p0, tri.p1), tri.p2);
    return {min_, max_};
}

BVH *build_canonical_tree_sah(std::vector<Triangle> &triangles,
                              int max_prims_per_leaf = 15,
                              int max_tree_depth = 64, int num_bins = 32,
                              float traversal_cost = 1.0f,
                              float intersection_cost = 1.5f) {
    struct Split {
        int axis;
        float position;
        float cost;
        uint32_t left_count;
    };
    constexpr auto MAX = std::numeric_limits<float>::max();
    struct Bin {
        vec3_float min = vec3_float{MAX, MAX, MAX};
        vec3_float max = vec3_float{-MAX, -MAX, -MAX};
        uint32_t count = 0;
    };

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);
        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            return new BVH(Leaf{
                .lo = aabb_min,
                .hi = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Compute centroid bounds for splitting.
        vec3_float centroid_min = triangle_centroid(triangles[low]);
        vec3_float centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            vec3_float c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        // Find best split using SAH.
        Split best_split = {-1, 0.0f, MAX, 0};
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        // Try splitting along each axis.
        for (int axis = 0; axis < 3; ++axis) {
            float extent = centroid_max[axis] - centroid_min[axis];
            if (extent < 1e-6f)
                continue; // Skip degenerate axis.

            std::vector<Bin> bins(num_bins);
            float bin_width = extent / num_bins;
            float inv_bin_width = 1.0f / bin_width;

            // Assign triangles to bins.
            for (uint32_t i = low; i < high; ++i) {
                vec3_float c = triangle_centroid(triangles[i]);
                int bin_idx =
                    std::min(num_bins - 1,
                             static_cast<int>((c[axis] - centroid_min[axis]) *
                                              inv_bin_width));

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                bins[bin_idx].min = min(bins[bin_idx].min, tri_min);
                bins[bin_idx].max = max(bins[bin_idx].max, tri_max);
                bins[bin_idx].count++;
            }

            // Compute prefix sums for efficient SAH evaluation.
            std::vector<vec3_float> left_min(num_bins), left_max(num_bins);
            std::vector<uint32_t> left_count(num_bins);

            left_min[0] = bins[0].min;
            left_max[0] = bins[0].max;
            left_count[0] = bins[0].count;

            for (int i = 1; i < num_bins; ++i) {
                left_min[i] = min(left_min[i - 1], bins[i].min);
                left_max[i] = max(left_max[i - 1], bins[i].max);
                left_count[i] = left_count[i - 1] + bins[i].count;
            }

            for (int split = 0; split < num_bins - 1; ++split) {
                if (left_count[split] == 0 || left_count[split] == count)
                    continue;
                vec3_float right_min = bins[split + 1].min;
                vec3_float right_max = bins[split + 1].max;
                uint32_t right_count = bins[split + 1].count;

                for (int i = split + 2; i < num_bins; ++i) {
                    if (bins[i].count > 0) {
                        right_min = min(right_min, bins[i].min);
                        right_max = max(right_max, bins[i].max);
                        right_count += bins[i].count;
                    }
                }

                float left_area =
                    surface_area(left_min[split], left_max[split]);
                float right_area = surface_area(right_min, right_max);

                float cost = traversal_cost +
                             (left_area / parent_area) * intersection_cost *
                                 left_count[split] +
                             (right_area / parent_area) * intersection_cost *
                                 right_count;

                if (cost < best_split.cost) {
                    best_split.axis = axis;
                    best_split.position =
                        centroid_min[axis] + (split + 1) * bin_width;
                    best_split.cost = cost;
                    best_split.left_count = left_count[split];
                }
            }
        }

        if (best_split.axis == -1 || best_split.cost >= leaf_cost) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Partition triangles based on best split.
        auto mid_it =
            std::partition(triangles.begin() + low, triangles.begin() + high,
                           [&](const Triangle &tri) {
                               vec3_float c = triangle_centroid(tri);
                               return c[best_split.axis] < best_split.position;
                           });

        uint32_t mid = std::distance(triangles.begin(), mid_it);

        // Handle edge case where all triangles end up on one side.
        if (mid == low || mid == high) {
            // Fall back to median split.
            mid = low + count / 2;
            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 vec3_float ca = triangle_centroid(a);
                                 vec3_float cb = triangle_centroid(b);
                                 return ca[best_split.axis] <
                                        cb[best_split.axis];
                             });
        }

        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), /*depth=*/0);
}

BVH *build_canonical_tree_median_split(std::vector<Triangle> &triangles,
                                       int max_prims_per_leaf = 15,
                                       int max_tree_depth = 64) {
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        if (count <= max_prims_per_leaf) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (int i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        vec3_float extent = aabb_max - aabb_min;
        int axis = 0;
        if (extent[1] > extent[0])
            axis = 1;
        if (extent[2] > extent[axis])
            axis = 2;

        // Partition around midpoint along axis.
        auto mid_it = triangles.begin() + low + count / 2;
        std::nth_element(triangles.begin() + low, mid_it,
                         triangles.begin() + high,
                         [&](const Triangle &a, const Triangle &b) {
                             float ca = (a.p0[axis] + a.p1[axis] + a.p2[axis]);
                             float cb = (b.p0[axis] + b.p1[axis] + b.p2[axis]);
                             return ca < cb;
                         });

        const uint32_t mid = low + count / 2;
        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), /*depth=*/0);
}

void free_canonical_tree(BVH *node) {
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
