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
                         const size_t max_prims_per_leaf = 4) {
    constexpr uint64_t MAX_TREE_DEPTH = 64;

    _tree_layout0 tree;
    tree.pCount = boundaryIndices.size();
    if (tree.pCount >= std::numeric_limits<uint16_t>::max()) {
        std::cerr << "Use larger index type for primitive offsets!\n";
        exit(-1);
    }
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


    // // Leaf and internal node count
    // const size_t leaf_count = (tree.pCount + (max_prims_per_leaf - 1)) / max_prims_per_leaf;
    // // Upper bound for unbalanced binary tree
    // const size_t internal_count = 2 * leaf_count - 1;

    // Upper bound for unbalanced binary tree
    tree.count = 2 * tree.pCount;
    if (tree.count >= std::numeric_limits<uint16_t>::max()) {
        std::cerr << "Use larger index type for references!\n";
        exit(-1);
    }

    tree.group0_index = (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.count);

    uint32_t next_node = 0;

    uint32_t max_depth = 0;

    uint32_t leaf_nodes = 0;
    uint32_t interior_nodes = 0;

    uint32_t *leaf_numbers = (uint32_t*)malloc(sizeof(uint32_t) * max_prims_per_leaf);

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        max_depth = std::max(max_depth, depth);
        if (depth >= MAX_TREE_DEPTH) {
            std::cerr << "tree build surpassed max tree depth: " << depth << "\n";
            exit(-1);
        }
        if (low >= tree.pCount) {
            std::cerr << "tree build out of range: " << low << " with " << tree.pCount << "primitives\n";
            exit(-1);
        }
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
        // tree.group0_index[this_index].pad0 = 0;

        if (count <= max_prims_per_leaf) {
            leaf_numbers[count]++;
            leaf_nodes++;
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            *reinterpret_cast<uint16_t *>(&tree.group0_index[this_index].split0on_nPrims) = low;
        } else {
            interior_nodes++;
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

    std::cout << "Bonsai max depth: " << max_depth << std::endl;
    std::cout << "       leaf nodes: " << leaf_nodes << std::endl;
    std::cout << "       interior nodes: " << interior_nodes << std::endl;
    for (uint32_t i = 0; i < max_prims_per_leaf; i++) {
        std::cout << "       leaf count = " <<  i << " has " << leaf_numbers[i] << std::endl;
    }
    free(leaf_numbers);

    if (next_node != tree.count) {
        if (next_node >= tree.count) {
            std::cerr << "Debug tree build: " << tree.count << " versus " << next_node << std::endl;
            exit(-1);
        }
        for (uint64_t i = next_node; i < tree.count; i++) {
            tree.group0_index[i].low = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            tree.group0_index[i].high = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
            tree.group0_index[i].nPrims = 0;
            tree.group0_index[i].axis = 0;
            // tree.group0_index[i].pad0 = 0;
            *reinterpret_cast<uint16_t *>(&tree.group0_index[i].split0on_nPrims) = 0;
        }
    }
    return tree;
}
