#include "solve_bonsai.h"


template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;

template <size_t DIM>
using Vectori = Eigen::Matrix<int, DIM, 1>;

inline vec3_float min(const vec3_float &a, const vec3_float &b) {
    vec3_float result;
    result[0] = std::fmin(a[0], b[0]);
    result[1] = std::fmin(a[1], b[1]);
    result[2] = std::fmin(a[2], b[2]);
    return result;
}

inline vec3_float max(const vec3_float &a, const vec3_float &b) {
    vec3_float result;
    result[0] = std::fmax(a[0], b[0]);
    result[1] = std::fmax(a[1], b[1]);
    result[2] = std::fmax(a[2], b[2]);
    return result;
}

_tree_layout0 build_tree(const std::vector<Vector<3>> &boundaryPositions,
                         const std::vector<Vectori<3>> &boundaryIndices,
                         const size_t max_prims_per_leaf = 2) {

    _tree_layout0 tree;
    tree.pCount = boundaryIndices.size();
    assert(tree.pCount < std::numeric_limits<uint16_t>::max());

    auto build_triangle = [&](const uint64_t i) {
        const auto &idxs = boundaryIndices[i];
        const auto &p0 = boundaryPositions[idxs(0)];
        const auto &p1 = boundaryPositions[idxs(1)];
        const auto &p2 = boundaryPositions[idxs(2)];
        Triangle tri;
        tri.p0 = {p0(0), p0(1), p0(2)};
        tri.p1 = {p1(0), p1(1), p1(2)};
        tri.p2 = {p2(0), p2(1), p2(2)};
        return tri;
    };

    // Build triangle list
    Triangle *triangles = (Triangle *)malloc(sizeof(Triangle) * tree.pCount);
    for (size_t i = 0; i < boundaryIndices.size(); ++i) {
        triangles[i] = build_triangle(i);
    }
    tree.prims = triangles;


    // Leaf and internal node count
    const size_t leaf_count = (tree.pCount + (max_prims_per_leaf - 1)) / max_prims_per_leaf;
    const size_t internal_count = leaf_count - 1;

    tree.count = leaf_count + internal_count;
    assert(tree.count & 1 == 0); // needs even number.
    tree.group0_index = (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.count);

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        uint32_t this_index = next_node++;

        // Compute AABB of all triangles in range
        vec3_float aabb_min = triangles[low].p0;
        vec3_float aabb_max = triangles[low].p0;
        for (uint32_t i = low; i < high; ++i) {
            for (vec3_float v : {triangles[i].p0, triangles[i].p1, triangles[i].p2}) {
                aabb_min = min(aabb_min, v);
                aabb_max = max(aabb_max, v);
            }
        }
        tree.group0_index[this_index].low = aabb_min;
        tree.group0_index[this_index].high = aabb_max;

        if (count <= max_prims_per_leaf) {
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            *reinterpret_cast<uint16_t *>(&tree.group0_index[this_index].split0on_nPrims) = low;
        } else {
            // Internal node
            tree.group0_index[this_index].nPrims = 0;

            vec3_float extent = aabb_max - aabb_min;
            int axis = 0;
            if (extent[1] > extent[0]) axis = 1;
            if (extent[2] > extent[axis]) axis = 2;
            tree.group0_index[this_index].axis = axis;

            // Partition around midpoint along axis
            auto mid = low + count / 2;
            std::nth_element(
                triangles + low, triangles + mid, triangles + high,
                [axis](const Triangle &a, const Triangle &b) {
                    float ca = (a.p0[axis] + a.p1[axis] + a.p2[axis]) / 3.f;
                    float cb = (b.p0[axis] + b.p1[axis] + b.p2[axis]) / 3.f;
                    return ca < cb;
                });

            uint32_t left = handle_range(low, mid, depth + 1);
            uint32_t right = handle_range(mid, high, depth + 1);

            uint32_t offset = right - this_index;
            *reinterpret_cast<uint16_t *>(&tree.group0_index[this_index].split0on_nPrims) = offset;
        }

        return this_index;
    };

    handle_range(0, tree.pCount, 0);
    return tree;
}
