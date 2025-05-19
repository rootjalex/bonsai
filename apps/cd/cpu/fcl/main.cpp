#include "main.h"
#include <fcl/fcl.h>
#include <iostream>

namespace fcl {
template <typename S>
void load_object_file(const char *filename, std::vector<Vector3<S>> &points,
                      std::vector<Triangle> &triangles) {

    FILE *file = fopen(filename, "rb");
    if (!file) {
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

template <typename BV>
BVHModel<BV> build_tree(const std::vector<Vector3<typename BV::S>> &vertices,
                        const std::vector<Triangle> &triangles) {
    detail::SplitMethodType method = detail::SPLIT_METHOD_MEAN;
    BVHModel<BV> m;
    m.bv_splitter.reset(new detail::BVSplitter<BV>(method));
    m.beginModel();
    m.addSubModel(vertices, triangles);
    m.endModel();
    return m;
}

template <typename BV>
bool collide_test(const Transform3<typename BV::S> &tf, BVHModel<BV> &m1,
                  BVHModel<BV> &m2, bool verbose) {
    using S = typename BV::S;
    Transform3<S> pose1(tf);
    Transform3<S> pose2 = Transform3<S>::Identity();

    CollisionResult<S> local_result;
    detail::MeshCollisionTraversalNode<BV> node;

    assert(detail::initialize<BV>(
               node, m1, pose1, m2, pose2,
               CollisionRequest<S>(std::numeric_limits<int>::max(), false),
               local_result) &&
           "initialization error");

    node.enable_statistics = verbose;
    collide(&node);
    std::cout << "number of contacts: " << local_result.numContacts() << "\n";
    return local_result.numContacts() > 0;
}

} // namespace fcl

namespace bonsai {

template <typename S>
_tree_layout0 build_tree(const std::vector<fcl::Vector3<S>> &vertices,
                         const std::vector<fcl::Triangle> &triangles) {

    return _tree_layout0{};
}

} // namespace bonsai

namespace {
template <typename S>
void run_test(int transform_count = 5, bool verbose = true) {
    std::vector<fcl::Vector3<S>> p1, p2;
    std::vector<fcl::Triangle> t1, t2;
    fcl::load_object_file("../env.obj", p1, t1);
    assert(!p1.empty());
    assert(!t1.empty());

    fcl::load_object_file("../rob.obj", p2, t2);
    assert(!p2.empty());
    assert(!t2.empty());

    fcl::aligned_vector<fcl::Transform3<S>> transforms;
    S extents[] = {-3000, -3000, 0, 3000, 3000, 3000};
    fcl::generate_random_transforms(extents, transforms, transform_count);
    for (int i = 0, e = transforms.size(); i < e; ++i) {
        fcl::BVHModel<fcl::AABB<S>> m1 = fcl::build_tree<fcl::AABB<S>>(p1, t1);
        fcl::BVHModel<fcl::AABB<S>> m2 = fcl::build_tree<fcl::AABB<S>>(p2, t2);
        fcl::collide_test<fcl::AABB<S>>(transforms[i], m1, m2, verbose);

        // _tree_layout0 b1 = bonsai::build_tree<S>(p1, t1);
        // _tree_layout0 b2 = bonsai::build_tree<S>(p2, t2);

        __dyn_array0 out = {
            .buffer = nullptr,
            .size = 0,
            .capacity = 0,
        };
        // collisions(out, b1, b2);
        std::cout << "collisions detected: " << out.size << '\n';
        auto *collisions = reinterpret_cast<__tuple_0 *>(out.buffer);
    }
}
} // namespace

int main() { run_test<float>(); }