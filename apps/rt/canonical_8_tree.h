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

BVH *build_canonical_tree_8_sah(std::vector<Triangle> &triangles,
                                int max_prims_per_leaf = 8,
                                int max_tree_depth = 64, int num_bins = 32,
                                float traversal_cost = 1.0f,
                                float intersection_cost = 1.5f) {

    struct Split {
        int axis;
        float positions[7]; // 7 split positions for 8-way
        float cost;
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

        // Create leaf if below threshold
        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Compute centroid bounds
        vec3_float centroid_min = triangle_centroid(triangles[low]);
        vec3_float centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            vec3_float c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        // Find best 8-way split
        Split best_split;
        best_split.cost = MAX;
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        // Try splitting along each axis
        for (int axis = 0; axis < 3; ++axis) {
            float extent = centroid_max[axis] - centroid_min[axis];
            if (extent < 1e-6f)
                continue;

            // Simple approach: divide into 8 equal parts
            float split_positions[7];
            for (int i = 0; i < 7; ++i) {
                split_positions[i] =
                    centroid_min[axis] + (i + 1) * extent / 8.0f;
            }

            // Evaluate this 8-way split
            std::vector<vec3_float> group_mins(8, vec3_float{MAX, MAX, MAX});
            std::vector<vec3_float> group_maxs(8, vec3_float{-MAX, -MAX, -MAX});
            std::vector<uint32_t> group_counts(8, 0);

            // Assign triangles to groups and compute bounds
            for (uint32_t i = low; i < high; ++i) {
                vec3_float c = triangle_centroid(triangles[i]);
                int group = 7; // defaults to last group
                for (int j = 0; j < 7; ++j) {
                    if (c[axis] < split_positions[j]) {
                        group = j;
                        break;
                    }
                }

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                group_mins[group] = min(group_mins[group], tri_min);
                group_maxs[group] = max(group_maxs[group], tri_max);
                group_counts[group]++;
            }

            // Calculate SAH cost for this split
            float split_cost = traversal_cost * 7; // 7 internal traversal steps
            for (int i = 0; i < 8; ++i) {
                if (group_counts[i] > 0) {
                    float area = surface_area(group_mins[i], group_maxs[i]);
                    split_cost += (area / parent_area) * intersection_cost *
                                  group_counts[i];
                }
            }

            if (split_cost < best_split.cost) {
                best_split.axis = axis;
                for (int i = 0; i < 7; ++i)
                    best_split.positions[i] = split_positions[i];
                best_split.cost = split_cost;
            }
        }

        // Check if splitting is worth it
        if (best_split.cost >= leaf_cost) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Partition triangles into 8 groups
        std::vector<std::vector<uint32_t>> groups(8);
        for (uint32_t i = low; i < high; ++i) {
            vec3_float c = triangle_centroid(triangles[i]);
            int group = 0;
            for (int j = 0; j < 7; ++j) {
                if (c[best_split.axis] >= best_split.positions[j]) {
                    group = j + 1;
                } else {
                    break;
                }
            }
            groups[group].push_back(i);
        }

        // Reorder triangles based on groups
        std::vector<Triangle> temp_triangles;
        temp_triangles.reserve(count);
        std::vector<uint32_t> group_starts(9);
        group_starts[0] = low;

        for (int g = 0; g < 8; ++g) {
            for (uint32_t idx : groups[g]) {
                temp_triangles.push_back(triangles[idx]);
            }
            group_starts[g + 1] = group_starts[g] + groups[g].size();
        }

        std::copy(temp_triangles.begin(), temp_triangles.end(),
                  triangles.begin() + low);
        Interior *node = new Interior();
        for (int i = 0; i < 8; ++i) {
            if (group_starts[i] < group_starts[i + 1]) {
                node->children[i] =
                    partition(group_starts[i], group_starts[i + 1], depth + 1);
                auto [child_min, child_max] = compute_aabb(
                    group_starts[i], group_starts[i + 1], triangles);
                node->lo[i] = child_min;
                node->hi[i] = child_max;
            } else {
                // empty child
                node->children[i] = nullptr;
                node->lo[i] = vec3_float{MAX, MAX, MAX};
                node->hi[i] = vec3_float{-MAX, -MAX, -MAX};
            }
        }

        return new BVH(*node);
    };

    return partition(0, triangles.size(), 0);
}

void free_canonical_tree_8(BVH *node) {
    if (node == nullptr) {
        return;
    }
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree_8(interior.children[0]);
        free_canonical_tree_8(interior.children[1]);
        free_canonical_tree_8(interior.children[2]);
        free_canonical_tree_8(interior.children[3]);
        free_canonical_tree_8(interior.children[4]);
        free_canonical_tree_8(interior.children[5]);
        free_canonical_tree_8(interior.children[6]);
        free_canonical_tree_8(interior.children[7]);
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