#include "cpu/rt.h"

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

BVH *build_canonical_tree(std::vector<Triangle> &triangles,
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

        // Partition around midpoint along axis
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
