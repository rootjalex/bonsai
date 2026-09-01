#include "cd.h"
// BVH construction following FCL's approach
// Reference: https://github.com/flexible-collision-library/fcl
// Specifically: include/fcl/geometry/bvh/BVH_model-inl.h (recursiveBuildTree)
// and include/fcl/geometry/bvh/detail/BV_splitter-inl.h (computeRule)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

// Compute AABB for a range of triangles
std::pair<float3, float3> compute_aabb(uint32_t low, uint32_t high,
                                       const std::vector<Triangle> &triangles) {
    float3 aabb_min = {INFINITY, INFINITY, INFINITY};
    float3 aabb_max = {-INFINITY, -INFINITY, -INFINITY};

    for (uint32_t i = low; i < high; ++i) {
        const Triangle &tri = triangles[i];
        for (const float3 *p : {&tri.p0, &tri.p1, &tri.p2}) {
            aabb_min[0] = std::min(aabb_min[0], (*p)[0]);
            aabb_min[1] = std::min(aabb_min[1], (*p)[1]);
            aabb_min[2] = std::min(aabb_min[2], (*p)[2]);
            aabb_max[0] = std::max(aabb_max[0], (*p)[0]);
            aabb_max[1] = std::max(aabb_max[1], (*p)[1]);
            aabb_max[2] = std::max(aabb_max[2], (*p)[2]);
        }
    }

    return {aabb_min, aabb_max};
}

// FCL-style BVH construction with median split. They use 1 primitive per leaf,
// and have unlimited tree depth.
BVH *build_fcl_tree_median_split(
    std::vector<Triangle> &triangles, int max_prims_per_leaf = 1,
    int max_tree_depth = std::numeric_limits<int>::max()) {
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        uint32_t count = high - low;

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        // Leaf node creation
        if (count <= max_prims_per_leaf || depth >= max_tree_depth) {
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

        // FCL's splitting approach:
        // 1. Choose split axis (longest extent)
        // https://github.com/flexible-collision-library/fcl/blob/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/include/fcl/geometry/bvh/detail/BV_splitter-inl.h#L210
        float3 extent = aabb_max - aabb_min;
        int axis = 0;
        float max_extent = extent[0];
        if (extent[1] > max_extent) {
            axis = 1;
            max_extent = extent[1];
        }
        if (extent[2] > max_extent) {
            axis = 2;
        }

        // 2. Compute split value (median of centroids).
        // https://github.com/flexible-collision-library/fcl/blob/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/include/fcl/geometry/bvh/detail/BV_splitter-inl.h#L619
        std::vector<float> centroid_coords;
        centroid_coords.reserve(count);

        for (uint32_t i = low; i < high; ++i) {
            const Triangle &tri = triangles[i];
            // Triangle centroid: average of three vertices
            float3 centroid = (tri.p0 + tri.p1 + tri.p2) / 3.0f;
            float coord = (axis == 0)   ? centroid[0]
                          : (axis == 1) ? centroid[1]
                                        : centroid[2];
            centroid_coords.push_back(coord);
        }

        // ...find median value.
        std::vector<float> sorted_coords = centroid_coords;
        std::nth_element(sorted_coords.begin(),
                         sorted_coords.begin() + count / 2,
                         sorted_coords.end());
        float split_value = sorted_coords[count / 2];

        // 3. Partition triangles.
        uint32_t c1 = 0; // Boundary between left and right partitions
        for (uint32_t i = 0; i < count; ++i) {
            const Triangle &tri = triangles[low + i];
            float3 centroid = (tri.p0 + tri.p1 + tri.p2) / 3.0f;
            float coord = (axis == 0)   ? centroid[0]
                          : (axis == 1) ? centroid[1]
                                        : centroid[2];

            // FCL's apply() function tests if point is on "right" side of split
            // A point is on the right if its coordinate > split_value
            // https://github.com/flexible-collision-library/fcl/blob/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/include/fcl/geometry/bvh/BVH_model-inl.h#L917
            if (bool on_right = (coord > split_value); !on_right) {
                // Place in left partition
                std::swap(triangles[low + i], triangles[low + c1]);
                c1++;
            }
        }

        // Handle degenerate case where all primitives end up on one side.
        // https://github.com/flexible-collision-library/fcl/blob/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/include/fcl/geometry/bvh/BVH_model-inl.h#L929
        if (c1 == 0 || c1 == count) {
            c1 = count / 2;
        }

        // Recursively build left and right subtrees
        const uint32_t mid = low + c1;
        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), 0);
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
