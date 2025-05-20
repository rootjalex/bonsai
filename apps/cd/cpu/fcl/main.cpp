#include "main.h"

#include <fcl/fcl.h>

#include <chrono>
#include <iostream>

// Functions in this namespace are pulled from the FCL library to mimic their
// collision detection setup. We refer readers to their repository:
// https://github.com/flexible-collision-library/fcl
namespace fcl {
// Loads the object file at filename, and fills the points and trinagles arrays.
template <typename S>
bool load_object_file(const std::string &filename,
                      std::vector<Vector3<S>> &points,
                      std::vector<Triangle> &triangles) {
    // Format is assumed to be Wavefront OBJ.
    // The relative path from root is: `bonsai/apps/cd/cpu/fcl/objects`
    std::string path = "../objects/" + filename;
    FILE *file = fopen(path.data(), "rb");
    if (file == nullptr) {
        std::cerr << "file: " << filename << " does not exist" << std::endl;
        return false;
    }

    bool has_normal = false;
    bool has_texture = false;
    char line_buffer[2000];
    while (fgets(line_buffer, 2000, file)) {
        char *first_token = strtok(line_buffer, "\r\n\t ");
        if (!first_token || first_token[0] == '#' || first_token[0] == 0)
            continue;

        switch (first_token[0]) {
        case 'v': {
            if (first_token[1] == 'n') {
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                has_normal = true;
            } else if (first_token[1] == 't') {
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                has_texture = true;
            } else {
                S x = (S)atof(strtok(nullptr, "\t "));
                S y = (S)atof(strtok(nullptr, "\t "));
                S z = (S)atof(strtok(nullptr, "\t "));
                points.emplace_back(x, y, z);
            }
        } break;
        case 'f': {
            Triangle tri;
            char *data[30];
            int n = 0;
            while ((data[n] = strtok(nullptr, "\t \r\n")) != nullptr) {
                if (strlen(data[n]))
                    n++;
            }

            for (int t = 0; t < (n - 2); ++t) {
                if ((!has_texture) && (!has_normal)) {
                    tri[0] = atoi(data[0]) - 1;
                    tri[1] = atoi(data[1]) - 1;
                    tri[2] = atoi(data[2]) - 1;
                } else {
                    const char *v1;
                    for (int i = 0; i < 3; i++) {
                        // vertex ID
                        if (i == 0)
                            v1 = data[0];
                        else
                            v1 = data[t + i];

                        tri[i] = atoi(v1) - 1;
                    }
                }
                triangles.push_back(tri);
            }
        }
        }
    }
    return true;
}

template <typename S>
S rand_interval(S rmin, S rmax) {
    S t = rand() / ((S)RAND_MAX + 1);
    return (t * (rmax - rmin) + rmin);
}

template <typename S>
void euler_to_matrix(S a, S b, S c, Matrix3<S> &R) {
    auto c1 = std::cos(a);
    auto c2 = std::cos(b);
    auto c3 = std::cos(c);
    auto s1 = std::sin(a);
    auto s2 = std::sin(b);
    auto s3 = std::sin(c);
    R << c1 * c2, -c2 * s1, s2, c3 * s1 + c1 * s2 * s3, c1 * c3 - s1 * s2 * s3,
        -c2 * s3, s1 * s3 - c1 * c3 * s2, c3 * s1 * s2 + c1 * s3, c2 * c3;
}

template <typename BV>
BVHModel<BV> build_tree(const std::vector<Vector3<typename BV::S>> &vertices,
                        const std::vector<Triangle> &triangles) {
    detail::SplitMethodType method = detail::SPLIT_METHOD_BV_CENTER;
    BVHModel<BV> m;
    m.bv_splitter.reset(new detail::BVSplitter<BV>(method));
    m.beginModel();
    m.addSubModel(vertices, triangles);
    m.endModel();
    return m;
}

template <typename BV>
std::vector<fcl::Contact<typename BV::S>> collide_test(BVHModel<BV> &m1,
                                                       BVHModel<BV> &m2) {
    using S = typename BV::S;
    Transform3<S> pose1 = Transform3<S>::Identity();
    Transform3<S> pose2 = Transform3<S>::Identity();

    CollisionResult<S> result;
    detail::MeshCollisionTraversalNode<BV> node;

    assert(detail::initialize<BV>(
               node, m1, pose1, m2, pose2,
               CollisionRequest<S>(std::numeric_limits<int>::max(), false),
               result) &&
           "initialization error");

    collide(&node);
    std::vector<Contact<S>> contacts;
    result.getContacts(contacts);
    return contacts;
}

} // namespace fcl

// Functions required for Bonsai tree construction.
namespace bonsai {
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

template <typename S>
_tree_layout0 build_tree(const std::vector<fcl::Vector3<S>> &input_vertices,
                         const std::vector<fcl::Triangle> &input_triangles,
                         int max_prims_per_leaf = 32, int max_tree_depth = 64) {
    _tree_layout0 tree;
    tree.pCount = input_triangles.size();
    if (tree.pCount >= std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Use larger index type for primitive offsets, "
                  << tree.pCount
                  << " >= " << std::numeric_limits<uint32_t>::max();
        exit(-1);
    }
    assert(tree.pCount < std::numeric_limits<uint32_t>::max());

    auto build_triangle = [&](const uint64_t i) {
        assert(i < input_triangles.size());
        const fcl::Triangle &t = input_triangles[i];
        size_t i0 = t[0];
        assert(i0 < input_vertices.size());
        size_t i1 = t[1];
        assert(i1 < input_vertices.size());
        size_t i2 = t[2];
        assert(i2 < input_vertices.size());
        const auto &p0 = input_vertices[i0];
        const auto &p1 = input_vertices[i1];
        const auto &p2 = input_vertices[i2];
        Triangle tri;
        tri.p0 = {p0[0], p0[1], p0[2]};
        tri.p1 = {p1[0], p1[1], p1[2]};
        tri.p2 = {p2[0], p2[1], p2[2]};
        return tri;
    };

    // Build triangle list
    Triangle *triangles = (Triangle *)malloc(sizeof(Triangle) * tree.pCount);
    for (size_t i = 0; i < input_triangles.size(); ++i) {
        triangles[i] = build_triangle(i);
    }
    tree.prims = triangles;

    // Upper bound for unbalanced binary tree
    tree.count = 2 * tree.pCount;
    if (tree.count >= std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Use larger index type for references, " << tree.count
                  << " >= " << std::numeric_limits<uint32_t>::max();
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
        if (depth >= max_tree_depth) {
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
            *reinterpret_cast<uint32_t *>(
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
            *reinterpret_cast<uint32_t *>(
                &tree.group0_index[this_index].split0on_nPrims) = offset;
        }

        return this_index;
    };

    handle_range(0, tree.pCount, 0);
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
            *reinterpret_cast<uint32_t *>(
                &tree.group0_index[i].split0on_nPrims) = 0;
        }
    }
    return tree;
}

} // namespace bonsai

// Functions for verifying the correctness of the collision detection
// algorithms and running the benchmarks.
namespace {
static inline float dot(const vec3_float &a, const vec3_float &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline vec3_float cross(const vec3_float &a, const vec3_float &b) {
    return (vec3_float){a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
}

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

template <typename S>
Triangle construct_triangle(const fcl::Triangle &t,
                            const std::vector<fcl::Vector3<S>> &v) {
    fcl::Vector3<S> x = v[t[0]];
    fcl::Vector3<S> y = v[t[1]];
    fcl::Vector3<S> z = v[t[2]];

    auto p0 = vec3_float{x[0], x[1], x[2]};
    auto p1 = vec3_float{y[0], y[1], y[2]};
    auto p2 = vec3_float{z[0], z[1], z[2]};
    return Triangle{p0, p1, p2};
}

// Runs an collision detection test on the two OBJ files for Bonsai and FCL.
template <typename S>
void run_test(const std::string &obj1_filename,
              const std::string &obj2_filename) {
    if constexpr (!(std::is_floating_point_v<S> && sizeof(S) == 4)) {
        std::cerr << "the bonsai kernel currently assumes f32 input";
        exit(-1);
    }

    using clock = std::chrono::high_resolution_clock;
    std::vector<fcl::Vector3<S>> v1, v2;
    std::vector<fcl::Triangle> T1, T2;
    if (!fcl::load_object_file(obj1_filename, v1, T1)) {
        exit(-1);
    }
    assert(!v1.empty() && "no vertices found!");
    assert(!T1.empty() && "no triangles found!");
    std::cout << obj1_filename << ": " << T1.size() << " triangles\n";
    if (!fcl::load_object_file(obj2_filename, v2, T2)) {
        exit(-1);
    }
    assert(!v2.empty() && "no vertices found!");
    assert(!T2.empty() && "no triangles found!");
    std::cout << obj2_filename << ": " << T2.size() << " triangles\n";

    // ---- FCL tree construction ----
    auto t0 = clock::now();
    fcl::BVHModel<fcl::AABB<S>> m1 = fcl::build_tree<fcl::AABB<S>>(v1, T1);
    fcl::BVHModel<fcl::AABB<S>> m2 = fcl::build_tree<fcl::AABB<S>>(v2, T2);
    auto t1 = clock::now();
    auto fcl_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcl]    tree construction time: " << fcl_time << " ms\n";

    // ---- FCL collision detection ----
    t0 = clock::now();
    const std::vector<fcl::Contact<S>> fcl_collisions =
        fcl::collide_test<fcl::AABB<S>>(m1, m2);
    t1 = clock::now();
    fcl_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcl]    collision detection time: " << fcl_time << " ms\n";

    // ---- Bonsai tree construction ----
    t0 = clock::now();
    _tree_layout0 b1 = bonsai::build_tree<S>(v1, T1);
    _tree_layout0 b2 = bonsai::build_tree<S>(v2, T2);
    t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction time: " << bonsai_time << " ms\n";

    // ---- Bonsai collision detection ----
    t0 = clock::now();
    __dyn_array0 out = {
        .buffer = nullptr,
        .size = 0,
        .capacity = 0,
    };
    collisions(out, b1, b2);
    t1 = clock::now();
    auto *bonsai_collisons = reinterpret_cast<__tuple_0 *>(out.buffer);
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] collision detection time: " << bonsai_time
              << " ms\n";

    // Verify outputs match and are valid intersections.
    const int64_t bonsai_count = out.size;
    const int64_t fcl_count = fcl_collisions.size();
    if (bonsai_count != fcl_count) {
        std::cerr << "different collision detection counts, bonsai: "
                  << bonsai_count << " vs fcl: " << fcl_count << '\n';
        exit(-1);
    }
    std::cout << "collision count: " << bonsai_count << '\n';
    for (int i = 0; i < fcl_collisions.size(); ++i) {
        auto [bt1, bt2] = bonsai_collisons[i];
        assert(intersects(bt1, bt2) &&
               "found non-intersecting triangles in bonsai!");

        Triangle ft1 = construct_triangle(T1[fcl_collisions[i].b1], v1);
        Triangle ft2 = construct_triangle(T2[fcl_collisions[i].b2], v2);
        assert(intersects(ft1, ft2) &&
               "found non-intersecting triangles in fcl!");
    }
    std::cout << "---\n";
}

} // namespace

int main() {
    // Wavefront OBJ files are taken from FCL [1] from `prims` [2].
    // [1] https://github.com/flexible-collision-library/fcl
    // [2] https://github.com/nickdesaulniers/prims/tree/master/meshes
    run_test<float>("env.obj", "rob.obj");       // [1]
    run_test<float>("dragon.obj", "bunny.obj");  // [2]
    run_test<float>("teapot.obj", "bunny.obj");  // [2]
    run_test<float>("teapot.obj", "dragon.obj"); // [2]

    return 0;
}
