#include "canonical_tree.h"
#include "cd.h"

#include <fcl/fcl.h>

#include <chrono>
#include <iostream>

// Functions in this namespace are pulled from the FCL library to mimic their
// collision detection setup. We refer readers to their repository:
// https://github.com/flexible-collision-library/fcl
namespace fcl {
// Loads the object file at filename, and fills the points and trinagles arrays.
template <typename S>
bool load_object_file(const std::string &object,
                      std::vector<Vector3<S>> &points,
                      std::vector<Triangle> &triangles) {
    // Format is assumed to be Wavefront OBJ.
    std::string path = "apps/cd/data/" + object + ".obj";
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
void run_test(const std::string &obj1, const std::string &obj2) {
    if constexpr (!(std::is_floating_point_v<S> && sizeof(S) == 4)) {
        std::cerr << "the bonsai kernel currently assumes f32 input";
        exit(-1);
    }

    using clock = std::chrono::high_resolution_clock;
    std::vector<fcl::Vector3<S>> v1, v2;
    std::vector<fcl::Triangle> T1, T2;
    if (!fcl::load_object_file(obj1, v1, T1)) {
        exit(-1);
    }
    assert(!v1.empty() && "no vertices found!");
    assert(!T1.empty() && "no triangles found!");
    if (!fcl::load_object_file(obj2, v2, T2)) {
        exit(-1);
    }
    assert(!v2.empty() && "no vertices found!");
    assert(!T2.empty() && "no triangles found!");

    auto t0 = clock::now();
    std::vector<Triangle> T1s = construct_triangles(T1, v1);
    std::vector<Triangle> T2s = construct_triangles(T2, v2);
    BVH *canonical_tree1 = build_fcl_tree_median_split(T1s);
    BVH *canonical_tree2 = build_fcl_tree_median_split(T2s);
    Triangles1 tree1 = build_triangles1(canonical_tree1);
    Triangles2 tree2 = build_triangles2(canonical_tree2);
    auto t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
              << std::endl;

    free_canonical_tree(canonical_tree1);
    free_canonical_tree(canonical_tree2);

    // ---- Bonsai collision detection ----
    t0 = clock::now();
    std::vector<std::tuple<Triangle, Triangle>> bonsai_collisions =
        collisions(&tree1, &tree2);
    t1 = clock::now();
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] collision detection : " << bonsai_time << " ms"
              << std::endl;

    // ---- FCL tree construction ----
    t0 = clock::now();
    fcl::BVHModel<fcl::AABB<S>> m1 = fcl::build_tree<fcl::AABB<S>>(v1, T1);
    fcl::BVHModel<fcl::AABB<S>> m2 = fcl::build_tree<fcl::AABB<S>>(v2, T2);
    t1 = clock::now();
    auto fcl_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcl]    tree construction   : " << fcl_time << " ms"
              << std::endl;

    // ---- FCL collision detection ----
    t0 = clock::now();
    const std::vector<fcl::Contact<S>> fcl_collisions =
        fcl::collide_test<fcl::AABB<S>>(m1, m2);
    t1 = clock::now();
    fcl_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcl]    collision detection : " << fcl_time << " ms"
              << std::endl;

    // ---- Bonsai tree construction ----

    // Verify outputs match and are valid intersections.
    const int64_t bonsai_count = bonsai_collisions.size();
    const int64_t fcl_count = fcl_collisions.size();
    if (bonsai_count != fcl_count) {
        std::cerr << "different collision detection counts, bonsai: "
                  << bonsai_count << " vs fcl: " << fcl_count << '\n';
        exit(-1);
    }
    std::cout << "collision count: " << bonsai_count << std::endl;
    std::cout << "\n";
}

} // namespace

int main(int argc, char *argv[]) {
    // Wavefront OBJ files are taken from FCL [1] from `prims` [2].
    // [1] https://github.com/flexible-collision-library/fcl
    // [2] https://github.com/nickdesaulniers/prims/tree/master/meshes
    assert(argc == 3);
    assert(argv[1]);
    assert(argv[2]);
    std::string obj1 = argv[1];
    std::string obj2 = argv[2];
    run_test<float>(obj1, obj2);
    return 0;
}
