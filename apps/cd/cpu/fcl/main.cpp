#include "main.h"
#include <chrono>
#include <fcl/fcl.h>
#include <iostream>

std::ostream &operator<<(std::ostream &os, const vec3_float &v) {
    os << '[' << v[0] << ", " << v[1] << ", " << v[2] << ']';
    return os;
}

std::ostream &operator<<(std::ostream &os, const Triangle &t) {
    os << '{' << t.p0 << ',' << t.p1 << ',' << t.p2 << '}';
    return os;
}

// https://stackoverflow.com/questions/17404513/floating-point-equality-and-tolerances
bool nearly_equal(float a, float b, int factor = 8) {
    float min_a =
        a -
        (a - std::nextafter(a, std::numeric_limits<float>::lowest())) * factor;
    float max_a =
        a + (std::nextafter(a, std::numeric_limits<float>::max()) - a) * factor;
    return min_a <= b && max_a >= b;
}

bool nearly_equal(vec3_float a, vec3_float b) {
    return nearly_equal(a[0], b[0]) && nearly_equal(a[1], b[1]) &&
           nearly_equal(a[2], b[2]);
}
bool nearly_equal(Triangle a, Triangle b) {
    return nearly_equal(a.p0, b.p0) && nearly_equal(a.p1, b.p1) &&
           nearly_equal(a.p2, b.p2);
}

namespace fcl {
template <typename S>
void load_object_file(const std::string &filename,
                      std::vector<Vector3<S>> &points,
                      std::vector<Triangle> &triangles) {
    // Format is assumed to be Wavefront OBJ.
    std::string path = "../objects/" + filename;
    FILE *file = fopen(path.data(), "rb");
    if (file == nullptr) {
        std::cerr << "file: " << filename << " does not exist" << std::endl;
        return;
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

template <typename S>
void generate_random_transforms(S extents[6],
                                aligned_vector<Transform3<S>> &transforms,
                                std::size_t n) {
    transforms.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto x = rand_interval(extents[0], extents[3]);
        auto y = rand_interval(extents[1], extents[4]);
        auto z = rand_interval(extents[2], extents[5]);

        const auto pi = constants<S>::pi();
        auto a = rand_interval((S)0, 2 * pi);
        auto b = rand_interval((S)0, 2 * pi);
        auto c = rand_interval((S)0, 2 * pi);

        {
            Matrix3<S> R;
            euler_to_matrix(a, b, c, R);
            Vector3<S> T(x, y, z);
            transforms[i].setIdentity();
            transforms[i].linear() = R;
            transforms[i].translation() = T;
        }
    }
}

// TODO(cgyurgyik): Are these trees *actually* the same?
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
std::vector<fcl::Contact<typename BV::S>>
collide_test(BVHModel<BV> &m1, BVHModel<BV> &m2, bool verbose) {
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

    node.enable_statistics = verbose;
    collide(&node);
    std::vector<Contact<S>> contacts;
    result.getContacts(contacts);
    return contacts;
}

} // namespace fcl

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
                         int max_prims_per_leaf = 32) {
    constexpr uint64_t MAX_TREE_DEPTH = 64;

    _tree_layout0 tree;
    tree.pCount = input_triangles.size();
    if (tree.pCount >= std::numeric_limits<uint16_t>::max()) {
        std::cerr << "Use larger index type for primitive offsets!\n";
        exit(-1);
    }
    assert(tree.pCount < std::numeric_limits<uint16_t>::max());

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

} // namespace bonsai

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

template <typename S>
void run_test(const std::string &obj1_filename,
              const std::string &obj2_filename, bool verbose = true) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<fcl::Vector3<S>> p1, p2;
    std::vector<fcl::Triangle> t1, t2;
    fcl::load_object_file(obj1_filename, p1, t1);
    assert(!p1.empty());
    assert(!t1.empty());
    std::cout << obj1_filename << ": " << t1.size() << " triangles\n";
    fcl::load_object_file(obj2_filename, p2, t2);
    assert(!p2.empty());
    assert(!t2.empty());
    std::cout << obj2_filename << ": " << t2.size() << " triangles\n";

    fcl::BVHModel<fcl::AABB<S>> m1 = fcl::build_tree<fcl::AABB<S>>(p1, t1);
    fcl::BVHModel<fcl::AABB<S>> m2 = fcl::build_tree<fcl::AABB<S>>(p2, t2);
    auto fcl_t1 = clock::now();
    const std::vector<fcl::Contact<S>> fcl_collisions =
        fcl::collide_test<fcl::AABB<S>>(m1, m2, verbose);
    auto fcl_t2 = clock::now();

    _tree_layout0 b1 = bonsai::build_tree<S>(p1, t1);
    _tree_layout0 b2 = bonsai::build_tree<S>(p2, t2);

    __dyn_array0 out = {
        .buffer = nullptr,
        .size = 0,
        .capacity = 0,
    };
    auto bonsai_t1 = clock::now();
    collisions(out, b1, b2);
    auto bonsai_t2 = clock::now();
    auto *bonsai_collisons = reinterpret_cast<__tuple_0 *>(out.buffer);

    const int64_t bonsai_count = out.size;
    const int64_t fcl_count = fcl_collisions.size();
    assert(bonsai_count == fcl_count &&
           "different collision detection counts!");
    int64_t unequal_collision_count = 0;
    for (int i = 0; i < fcl_collisions.size(); ++i) {
        auto [bt1, bt2] = bonsai_collisons[i];
        assert(intersects(bt1, bt2) &&
               "found non-intersecting triangles in bonsai!");

        Triangle ft1 = construct_triangle(t1[fcl_collisions[i].b1], p1);
        Triangle ft2 = construct_triangle(t2[fcl_collisions[i].b2], p2);
        assert(intersects(ft1, ft2) &&
               "found non-intersecting triangles in fcl!");
        if (nearly_equal(bt1, ft1) && nearly_equal(bt2, ft2)) {
            continue;
        }
        ++unequal_collision_count;
        if (!verbose) {
            continue;
        }
        std::cout << "[bonsai] " << bt1 << " <-> " << bt2 << "\n";
        std::cout << "[fcl]    " << ft1 << " <-> " << ft2 << "\n";
        std::cout << "\n --- \n";
    }
    std::cout << "collision count: " << bonsai_count
              << ", unequal: " << unequal_collision_count << '\n';

    auto fcl_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(fcl_t2 - fcl_t1)
            .count();
    auto bonsai_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           bonsai_t2 - bonsai_t1)
                           .count();
    std::cout << "[fcl]    collision detection time: " << fcl_time << " ms\n";
    std::cout << "[bonsai] collision detection time: " << bonsai_time
              << " ms\n";
}
} // namespace

int main() {
    // Imported from the FCL library.
    run_test<float>("fcl/env.obj", "fcl/rob.obj", /*verbose=*/false);
}