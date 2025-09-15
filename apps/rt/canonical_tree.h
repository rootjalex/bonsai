#include "cpu/rt.h"

class IndexTriangle {
  public:
    IndexTriangle() {
        sides[0] = 0;
        sides[1] = 0;
        sides[2] = 0;
    }
    IndexTriangle(size_t a, size_t b, size_t c) {
        sides[0] = a;
        sides[1] = b;
        sides[2] = c;
    }

    size_t &operator[](size_t index) {
        assert(index < 3);
        return sides[index];
    }

    const size_t &operator[](size_t index) const {
        assert(index < 3);
        return sides[index];
    }

  private:
    size_t sides[3];
};

template <typename S>
Triangle build_triangle(const uint64_t i,
                        const std::vector<vector<S, 3>> &vertices,
                        const std::vector<IndexTriangle> &triangles) {
    assert(i < triangles.size());
    const IndexTriangle &before = triangles[i];
    return build_triangle(before, vertices);
};

template <typename S>
Triangle build_triangle(const IndexTriangle &before,
                        const std::vector<vector<S, 3>> &vertices) {
    size_t i0 = before[0];
    assert(i0 < vertices.size());
    size_t i1 = before[1];
    assert(i1 < vertices.size());
    size_t i2 = before[2];
    assert(i2 < vertices.size());
    const auto &p0 = vertices[i0];
    const auto &p1 = vertices[i1];
    const auto &p2 = vertices[i2];

    Triangle after;
    after.p0 = {p0[0], p0[1], p0[2]};
    after.p1 = {p1[0], p1[1], p1[2]};
    after.p2 = {p2[0], p2[1], p2[2]};
    return after;
}

template <typename S>
std::pair<vec3_float, vec3_float>
compute_aabb(uint32_t low, uint32_t high,
             const std::vector<vector<S, 3>> &vertices,
             const std::vector<IndexTriangle> &triangles) {
    Triangle t = build_triangle(low, vertices, triangles);
    vec3_float aabb_min = t.p0;
    vec3_float aabb_max = t.p0;
    for (uint32_t i = low; i < high; ++i) {
        Triangle t = build_triangle(i, vertices, triangles);
        for (vec3_float v : {t.p0, t.p1, t.p2}) {
            aabb_min = min(aabb_min, v);
            aabb_max = max(aabb_max, v);
        }
    }
    return {aabb_min, aabb_max};
}

template <typename S>
BVH *build_canonical_tree(const std::vector<vector<S, 3>> &vertices,
                          std::vector<IndexTriangle> &triangles,
                          int max_prims_per_leaf = 15,
                          int max_tree_depth = 64) {
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;

        auto [aabb_min, aabb_max] =
            compute_aabb(low, high, vertices, triangles);

        if (count <= max_prims_per_leaf) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (int i = 0; i < count; ++i) {
                data[i] = build_triangle(low + i, vertices, triangles);
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
                         [&](const IndexTriangle &aa, const IndexTriangle &bb) {
                             Triangle a = build_triangle(aa, vertices);
                             Triangle b = build_triangle(bb, vertices);
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
