#ifdef AJR_PROFILING
#include "cd.h"
#else
#include "cd_gen.h"
#endif

#include <fcl/fcl.h>

#include <chrono>
#include <iostream>

#ifdef AJR_PROFILING
#include "profile.h"
#endif

#include "runtime/bonsai_benchmark.h"

constexpr int k = 7;
constexpr int m = 1;

// Compute AABB for a range of triangles
std::pair<float3, float3> compute_aabb(uint32_t low, uint32_t high,
                                       const std::vector<Triangle> &triangles) {
    float3 aabb_min = {INFINITY, INFINITY, INFINITY};
    float3 aabb_max = {-INFINITY, -INFINITY, -INFINITY};

    for (uint32_t i = low; i < high; ++i) {
        const Triangle &tri = triangles[i];
        for (const float3 *p : {&tri.p0, &tri.p1, &tri.p2}) {
            aabb_min.x = std::min(aabb_min.x, p->x);
            aabb_min.y = std::min(aabb_min.y, p->y);
            aabb_min.z = std::min(aabb_min.z, p->z);
            aabb_max.x = std::max(aabb_max.x, p->x);
            aabb_max.y = std::max(aabb_max.y, p->y);
            aabb_max.z = std::max(aabb_max.z, p->z);
        }
    }

    return {aabb_min, aabb_max};
}

// FCL-style BVH construction with median split. They use 1 primitive per leaf,
// and have unlimited tree depth.
template <typename tree_layout, typename node_layout>
tree_layout build_fcl_tree_median_split(
    std::vector<Triangle> &triangles, int max_prims_per_leaf = 1,
    int max_tree_depth = std::numeric_limits<int>::max()) {

    std::vector<node_layout> nodes;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        const uint32_t index = nodes.size();
        uint32_t count = high - low;

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        // Leaf node creation
        if (count <= max_prims_per_leaf || depth >= max_tree_depth) {
            nodes.emplace_back(node_layout{.low = aabb_min,
                                           .high = aabb_max,
                                           .nPrims = count,
                                           .offset = low});
            return index;
        }

        // FCL's splitting approach:
        // 1. Choose split axis (longest extent)
        // https://github.com/flexible-collision-library/fcl/blob/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/include/fcl/geometry/bvh/detail/BV_splitter-inl.h#L210
        float3 extent = aabb_max - aabb_min;
        int axis = 0;
        float max_extent = extent.x;
        if (extent.y > max_extent) {
            axis = 1;
            max_extent = extent.y;
        }
        if (extent.z > max_extent) {
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
            float coord = (axis == 0)   ? centroid.x
                          : (axis == 1) ? centroid.y
                                        : centroid.z;
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
            float coord = (axis == 0)   ? centroid.x
                          : (axis == 1) ? centroid.y
                                        : centroid.z;

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

        // Reserve node space
        nodes.emplace_back(node_layout{
            .low = aabb_min, .high = aabb_max, .nPrims = 0, .offset = 0});

        // Recursively build left and right subtrees
        const uint32_t mid = low + c1;
        (void)partition(low, mid, depth + 1);
        const auto right = partition(mid, high, depth + 1);
        nodes[index].offset = right - index;

        return index;
    };

    (void)partition(0, triangles.size(), 0);

    tree_layout tree;
    tree.pCount = triangles.size();
    tree.prims = triangles.data();
    tree.nCount = nodes.size();
    tree.group0_index = new node_layout[nodes.size()];
    std::memcpy(tree.group0_index, nodes.data(),
                nodes.size() * sizeof(node_layout));

    return tree;
}

// Functions in this namespace are pulled from the FCL library to mimic their
// collision detection setup. We refer readers to their repository:
// https://github.com/flexible-collision-library/fcl
namespace fcl {
// Loads the object file at filename, and fills the points and trinagles arrays.
template <typename S>
bool load_object_file(const std::string &directory, const std::string &object,
                      std::vector<Vector3<S>> &points,
                      std::vector<Triangle> &triangles) {
    // Format is assumed to be Wavefront OBJ.
    std::string path = directory + "/" + object + ".obj";
    FILE *file = fopen(path.data(), "rb");
    if (file == nullptr) {
        std::cerr << "file: " << path << " does not exist" << std::endl;
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
    detail::SplitMethodType method = detail::SPLIT_METHOD_MEDIAN;
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
    // similar to bonsai, only return whether a contact occurs.
    constexpr bool enable_contact = false;
    CollisionRequest<S> request(std::numeric_limits<int>::max(),
                                enable_contact);
    // request.enableStatistics(true);
    collide(&m1, pose1, &m2, pose2, request, result);

    std::vector<Contact<S>> contacts;
    result.getContacts(contacts);
    return contacts;
}

} // namespace fcl

// Functions required for Bonsai tree construction.
namespace bonsai {
inline float3 min(const float3 &a, const float3 &b) {
    float3 result;
    result[0] = std::fmin(a[0], b[0]);
    result[1] = std::fmin(a[1], b[1]);
    result[2] = std::fmin(a[2], b[2]);
    return result;
}

inline float3 max(const float3 &a, const float3 &b) {
    float3 result;
    result[0] = std::fmax(a[0], b[0]);
    result[1] = std::fmax(a[1], b[1]);
    result[2] = std::fmax(a[2], b[2]);
    return result;
}

} // namespace bonsai

// Functions for verifying the correctness of the collision detection
// algorithms and running the benchmarks.
namespace {
static inline float dot(const float3 &a, const float3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline float3 cross(const float3 &a, const float3 &b) {
    return (float3){a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]};
}

template <typename S>
Triangle construct_triangle(const fcl::Triangle &t,
                            const std::vector<fcl::Vector3<S>> &v) {
    assert(t[0] < v.size());
    fcl::Vector3<S> x = v[t[0]];
    assert(t[1] < v.size());
    fcl::Vector3<S> y = v[t[1]];
    assert(t[2] < v.size());
    fcl::Vector3<S> z = v[t[2]];

    auto p0 = float3{x[0], x[1], x[2]};
    auto p1 = float3{y[0], y[1], y[2]};
    auto p2 = float3{z[0], z[1], z[2]};
    return Triangle{p0, p1, p2};
}

template <typename S>
std::vector<Triangle>
construct_triangles(const std::vector<fcl::Triangle> &fcl_triangles,
                    const std::vector<fcl::Vector3<S>> &fcl_vertices) {
    std::vector<Triangle> triangles;
    for (const fcl::Triangle &triangle : fcl_triangles) {
        triangles.push_back(construct_triangle(triangle, fcl_vertices));
    }
    return triangles;
}

// Runs a collision detection test on the two OBJ files for Bonsai and FCL.
template <typename S>
void run_test(const std::string &directory, const std::string &obj1,
              const std::string &obj2) {
    if constexpr (!(std::is_floating_point_v<S> && sizeof(S) == 4)) {
        std::cerr << "the bonsai kernel currently assumes f32 input";
        exit(-1);
    }

    using clock = std::chrono::high_resolution_clock;
    std::vector<fcl::Vector3<S>> v1, v2;
    std::vector<fcl::Triangle> T1, T2;
    // std::cout << "Loading " + obj1 << " file." << std::endl;
    if (!fcl::load_object_file(directory, obj1, v1, T1)) {
        exit(-1);
    }
    assert(!v1.empty() && "no vertices found!");
    assert(!T1.empty() && "no triangles found!");
    // std::cout << "Loading " + obj2 << " file." << std::endl;
    if (!fcl::load_object_file(directory, obj2, v2, T2)) {
        exit(-1);
    }
    assert(!v2.empty() && "no vertices found!");
    assert(!T2.empty() && "no triangles found!");

    std::cout << "    (\"" << obj1 << "\", \"" << obj2 << "\", ";

    auto t0 = clock::now();
    std::vector<Triangle> T1s = construct_triangles(T1, v1);
    std::vector<Triangle> T2s = construct_triangles(T2, v2);
    // std::cout << "Building bonsai tree on " + obj1 << "." << std::endl;
    auto tree1 = build_fcl_tree_median_split<_tree_layout0, _tree_layout1>(T1s);
    // std::cout << "Building bonsai tree on " + obj2 << "." << std::endl;
    auto tree2 = build_fcl_tree_median_split<_tree_layout4, _tree_layout5>(T2s);
    // BVH *canonical_tree1 = build_fcl_tree_median_split(T1s);
    // BVH *canonical_tree2 = build_fcl_tree_median_split(T2s);
    // Triangles1 tree1 = build_triangles1(canonical_tree1);
    // Triangles2 tree2 = build_triangles2(canonical_tree2);
    auto t1 = clock::now();
    auto bonsai_build_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << bonsai_build_time << ", ";
    // std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
    //           << std::endl;
#ifdef AJR_PROFILING
    ajr_profiler_reset();
#endif
    // ---- Bonsai collision detection ----
    // t0 = clock::now();
    // Warm-up
    auto bonsai_collisions = collisions(tree1, tree2);

    auto bonsai_cd_time =
        benchmark_function(
            [&]() { bonsai_collisions = collisions(tree1, tree2); }, k, m) /
        (double)1e6;
    std::cout << bonsai_cd_time << ", ";

    // t1 = clock::now();
    // bonsai_time =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
    //     t0).count();
    // std::cout << "[bonsai] collision detection : " << bonsai_time << " ms"
    //           << std::endl;
#ifdef AJR_PROFILING
    std::cout << " [bonsai]  aabb hits  tested = " << bonsai_aabb_counter
              << "\n";
    std::cout << " [bonsai]  tri hits   tested = " << bonsai_tri_counter
              << "\n";
    std::cout << " [bonsai]  recursions tested = " << bonsai_rec_counter
              << "\n";
    ajr_profiler_reset();
#endif
    // ---- FCL tree construction ----
    // std::cout << "Building FCL trees." << std::endl;
    t0 = clock::now();
    fcl::BVHModel<fcl::AABB<S>> m1 = fcl::build_tree<fcl::AABB<S>>(v1, T1);
    fcl::BVHModel<fcl::AABB<S>> m2 = fcl::build_tree<fcl::AABB<S>>(v2, T2);
    t1 = clock::now();
    auto fcl_build_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << fcl_build_time << ", ";
    // std::cout << "[fcl]    tree construction   : " << fcl_time << " ms"
    //           << std::endl;

    // ---- FCL collision detection ----
    // t0 = clock::now();
    // warm-up
    std::vector<fcl::Contact<S>> fcl_collisions =
        fcl::collide_test<fcl::AABB<S>>(m1, m2);
    // t1 = clock::now();
    // fcl_time =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
    //     t0).count();
    // std::cout << "[fcl]    collision detection : " << fcl_time << " ms"
    //           << std::endl;
    auto fcl_cd_time =
        benchmark_function(
            [&]() { fcl_collisions = fcl::collide_test<fcl::AABB<S>>(m1, m2); },
            k, m) /
        (double)1e6;
    std::cout << fcl_cd_time << ")\n";
#ifdef AJR_PROFILING
    std::cout << " [fcl]  aabb hits  tested = " << fcl_aabb_counter << "\n";
    std::cout << " [fcl]  tri hits   tested = " << fcl_tri_counter << "\n";
    std::cout << " [fcl]  recursions tested = " << fcl_rec_counter << "\n";
    ajr_profiler_reset();
#endif
    // ---- Bonsai tree construction ----

    // Verify outputs match and are valid intersections.
    const int64_t bonsai_count = bonsai_collisions.size();
    const int64_t fcl_count = fcl_collisions.size();
    if (bonsai_count != fcl_count) {
        std::cerr << "different collision detection counts, bonsai: "
                  << bonsai_count << " vs fcl: " << fcl_count << '\n';
        exit(-1);
    }

    // std::cout << "collision count: " << bonsai_count << std::endl;
    // std::cout << "\n";
}

} // namespace

int main(int argc, char *argv[]) {
    // Wavefront OBJ files are taken from FCL [1] from `prims` [2].
    // [1] https://github.com/flexible-collision-library/fcl
    // [2] https://github.com/nickdesaulniers/prims/tree/master/meshes
    assert(argc == 4);
    assert(argv[1]);
    assert(argv[2]);
    assert(argv[3]);
    std::string obj_dir = argv[1];
    std::string obj1 = argv[2];
    std::string obj2 = argv[3];
    run_test<float>(obj_dir, obj1, obj2);
    return 0;
}