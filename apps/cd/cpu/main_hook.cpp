#include "main.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline vec3_float random_vec3_float() {
    return {random_float(), random_float(), random_float()};
}

std::ostream &operator<<(std::ostream &os, const vec3_float &v) {
    os << '[' << v[0] << ", " << v[1] << ", " << v[2] << ']';
    return os;
}

static inline vec3_float min(const vec3_float &a, const vec3_float &b) {
    vec3_float result;
    result[0] = std::fmin(a[0], b[0]);
    result[1] = std::fmin(a[1], b[1]);
    result[2] = std::fmin(a[2], b[2]);
    return result;
}

static inline vec3_float max(const vec3_float &a, const vec3_float &b) {
    vec3_float result;
    result[0] = std::fmax(a[0], b[0]);
    result[1] = std::fmax(a[1], b[1]);
    result[2] = std::fmax(a[2], b[2]);
    return result;
}

static inline float dot(const vec3_float &a, const vec3_float &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline vec3_float cross(const vec3_float &a, const vec3_float &b) {
    return (vec3_float){a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
}

// Compute the 1D intersection interval of triangle projected onto one axis
static void compute_interval(float v0, float v1, float v2, float d0, float d1,
                             float d2, float isect[2]) {
    if (d0 * d1 > 0.0f) {
        isect[0] = v2 + (v0 - v2) * d2 / (d2 - d0);
        isect[1] = v2 + (v1 - v2) * d2 / (d2 - d1);
    } else if (d0 * d2 > 0.0f) {
        isect[0] = v1 + (v0 - v1) * d1 / (d1 - d0);
        isect[1] = v1 + (v2 - v1) * d1 / (d1 - d2);
    } else if (d1 * d2 > 0.0f) {
        isect[0] = v0 + (v1 - v0) * d0 / (d0 - d1);
        isect[1] = v0 + (v2 - v0) * d0 / (d0 - d2);
    } else {
        isect[0] = isect[1] = v0;
    }
}

bool intersects(const Triangle &t1, const Triangle &t2) {
    const float eps = 1e-6f;

    vec3_float e1 = t1.p1 - t1.p0;
    vec3_float e2 = t1.p2 - t1.p0;
    vec3_float n1 = cross(e1, e2);
    float d1 = -dot(n1, t1.p0);

    float du0 = dot(n1, t2.p0) + d1;
    float du1 = dot(n1, t2.p1) + d1;
    float du2 = dot(n1, t2.p2) + d1;

    if ((du0 > eps && du1 > eps && du2 > eps) ||
        (du0 < -eps && du1 < -eps && du2 < -eps))
        return false;

    vec3_float f1 = t2.p1 - t2.p0;
    vec3_float f2 = t2.p2 - t2.p0;
    vec3_float n2 = cross(f1, f2);
    float d2 = -dot(n2, t2.p0);

    float dv0 = dot(n2, t1.p0) + d2;
    float dv1 = dot(n2, t1.p1) + d2;
    float dv2 = dot(n2, t1.p2) + d2;

    if ((dv0 > eps && dv1 > eps && dv2 > eps) ||
        (dv0 < -eps && dv1 < -eps && dv2 < -eps))
        return false;

    vec3_float D = cross(n1, n2);

    int index = 0;
    float absx = std::fabs(D[0]), absy = std::fabs(D[1]),
          absz = std::fabs(D[2]);
    if (absy > absx)
        index = 1, absx = absy;
    if (absz > absx)
        index = 2;

    float v1_0 = t1.p0[index], v1_1 = t1.p1[index], v1_2 = t1.p2[index];
    float v2_0 = t2.p0[index], v2_1 = t2.p1[index], v2_2 = t2.p2[index];

    float isect1[2], isect2[2];
    compute_interval(v1_0, v1_1, v1_2, dv0, dv1, dv2, isect1);
    compute_interval(v2_0, v2_1, v2_2, du0, du1, du2, isect2);

    if (isect1[0] > isect1[1])
        std::swap(isect1[0], isect1[1]);
    if (isect2[0] > isect2[1])
        std::swap(isect2[0], isect2[1]);

    return !(isect1[1] < isect2[0] || isect2[1] < isect1[0]);
}

_tree_layout0 build_tree(int64_t tree_size,
                         const size_t max_prims_per_leaf = 4) {
    constexpr uint64_t MAX_TREE_DEPTH = 64;

    _tree_layout0 tree;
    tree.pCount = tree_size;
    if (tree.pCount >= std::numeric_limits<uint16_t>::max()) {
        std::cerr << "Use larger index type for primitive offsets!\n";
        exit(-1);
    }
    assert(tree.pCount < std::numeric_limits<uint16_t>::max());

    auto build_triangle = [&](const uint64_t i) {
        Triangle tri;
        tri.p0 = random_vec3_float();
        tri.p1 = random_vec3_float();
        tri.p2 = random_vec3_float();
        return tri;
    };

    // Build triangle list
    Triangle *triangles = (Triangle *)malloc(sizeof(Triangle) * tree.pCount);
    for (size_t i = 0; i < tree.pCount; ++i) {
        triangles[i] = build_triangle(i);
    }
    tree.prims = triangles;

    // // Leaf and internal node count
    // const size_t leaf_count = (tree.pCount + (max_prims_per_leaf - 1)) /
    // max_prims_per_leaf;
    // // Upper bound for unbalanced binary tree
    // const size_t internal_count = 2 * leaf_count - 1;

    // Upper bound for unbalanced binary tree
    tree.count = 2 * tree.pCount;
    if (tree.count >= std::numeric_limits<uint16_t>::max()) {
        std::cerr << "Use larger index type for references!\n";
        exit(-1);
    }

    tree.group0_index =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.count);

    uint32_t next_node = 0;

    uint32_t max_depth = 0;

    uint32_t leaf_nodes = 0;
    uint32_t interior_nodes = 0;

    uint32_t *leaf_numbers =
        (uint32_t *)malloc(sizeof(uint32_t) * max_prims_per_leaf);

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        max_depth = std::max(max_depth, depth);
        if (depth >= MAX_TREE_DEPTH) {
            std::cerr << "tree build surpassed max tree depth: " << depth
                      << "\n";
            exit(-1);
        }
        if (low >= tree.pCount) {
            std::cerr << "tree build out of range: " << low << " with "
                      << tree.pCount << "primitives\n";
            exit(-1);
        }
        uint32_t count = high - low;
        uint32_t this_index = next_node++;

        // Compute AABB of all triangles in range
        vec3_float aabb_min = triangles[low].p0;
        vec3_float aabb_max = triangles[low].p0;
        for (uint32_t i = low; i < high; ++i) {
            for (vec3_float v :
                 {triangles[i].p0, triangles[i].p1, triangles[i].p2}) {
                aabb_min = min(aabb_min, v);
                aabb_max = max(aabb_max, v);
            }
        }
        tree.group0_index[this_index].low = aabb_min;
        tree.group0_index[this_index].high = aabb_max;
        tree.group0_index[this_index].pad0 = 0;

        if (count <= max_prims_per_leaf) {
            leaf_numbers[count]++;
            leaf_nodes++;
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            *reinterpret_cast<uint16_t *>(
                &tree.group0_index[this_index].split0on_nPrims) = low;
        } else {
            interior_nodes++;
            // Internal node
            tree.group0_index[this_index].nPrims = 0;

            vec3_float extent = aabb_max - aabb_min;
            int axis = 0;
            if (extent[1] > extent[0])
                axis = 1;
            if (extent[2] > extent[axis])
                axis = 2;
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
            *reinterpret_cast<uint16_t *>(
                &tree.group0_index[this_index].split0on_nPrims) = offset;
        }

        return this_index;
    };

    handle_range(0, tree.pCount, 0);

    std::cout << "Bonsai max depth: " << max_depth << std::endl;
    std::cout << "       leaf nodes: " << leaf_nodes << std::endl;
    std::cout << "       interior nodes: " << interior_nodes << std::endl;
    for (uint32_t i = 0; i < max_prims_per_leaf; i++) {
        std::cout << "       leaf count = " << i << " has " << leaf_numbers[i]
                  << std::endl;
    }
    free(leaf_numbers);

    if (next_node != tree.count) {
        if (next_node >= tree.count) {
            std::cerr << "Debug tree build: " << tree.count << " versus "
                      << next_node << std::endl;
            exit(-1);
        }
        for (uint64_t i = next_node; i < tree.count; i++) {
            tree.group0_index[i].low = {std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max()};
            tree.group0_index[i].high = {std::numeric_limits<float>::min(),
                                         std::numeric_limits<float>::min(),
                                         std::numeric_limits<float>::min()};
            tree.group0_index[i].nPrims = 0;
            tree.group0_index[i].axis = 0;
            tree.group0_index[i].pad0 = 0;
            *reinterpret_cast<uint16_t *>(
                &tree.group0_index[i].split0on_nPrims) = 0;
        }
    }
    return tree;
}
int main() {
    auto t1s = build_tree(128);
    auto t2s = build_tree(128);
    __dyn_array0 out = {
        .buffer = nullptr,
        .size = 0,
        .capacity = 0,
    };
    collisions(out, t1s, t2s);
    std::cout << "buffer capacity: " << out.capacity << '\n';
    std::cout << "collisions detected: " << out.size << '\n';
    assert(out.size < out.capacity);
    auto *collisions = reinterpret_cast<__tuple_0 *>(out.buffer);
    assert(collisions && "buffer is nullptr!");
    for (int i = 0; i < out.size; ++i) {
        auto [t1, t2] = collisions[i];
        assert(intersects(t1, t2) && "found non-colliding pair!");
    }

    return 0;
}
