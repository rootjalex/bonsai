#include "canonical_tree.h"
#include "cd.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <chrono>
#include <iostream>

// Wavefront OBJ loading, kept so the Bonsai kernel has meshes to run on. The
// vertex and index containers are local rather than a third-party library's,
// so building this app needs nothing beyond the generated header.
namespace mesh {

struct Vec3 {
    float v[3];
    Vec3() : v{0.0f, 0.0f, 0.0f} {}
    Vec3(float x, float y, float z) : v{x, y, z} {}
    float operator[](int i) const { return v[i]; }
};

struct Triangle {
    std::size_t v[3];
    std::size_t &operator[](int i) { return v[i]; }
    std::size_t operator[](int i) const { return v[i]; }
};

// Loads the object file at filename, and fills the points and trinagles arrays.
bool load_object_file(const std::string &object, std::vector<Vec3> &points,
                      std::vector<Triangle> &triangles) {
    // Format is assumed to be Wavefront OBJ.
    std::string path = "artifact/scenes/cd/" + object + ".obj";
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
                float x = (float)atof(strtok(nullptr, "\t "));
                float y = (float)atof(strtok(nullptr, "\t "));
                float z = (float)atof(strtok(nullptr, "\t "));
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

} // namespace mesh

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

Triangle construct_triangle(const mesh::Triangle &t,
                            const std::vector<mesh::Vec3> &v) {
    assert(t[0] < v.size());
    mesh::Vec3 x = v[t[0]];
    assert(t[1] < v.size());
    mesh::Vec3 y = v[t[1]];
    assert(t[2] < v.size());
    mesh::Vec3 z = v[t[2]];

    auto p0 = float3{x[0], x[1], x[2]};
    auto p1 = float3{y[0], y[1], y[2]};
    auto p2 = float3{z[0], z[1], z[2]};
    return Triangle{p0, p1, p2};
}

std::vector<Triangle>
construct_triangles(const std::vector<mesh::Triangle> &mesh_triangles,
                    const std::vector<mesh::Vec3> &mesh_vertices) {
    std::vector<Triangle> triangles;
    for (const mesh::Triangle &triangle : mesh_triangles) {
        triangles.push_back(construct_triangle(triangle, mesh_vertices));
    }
    return triangles;
}

// Runs a collision detection test on the two OBJ files.
void run_test(const std::string &obj1, const std::string &obj2) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<mesh::Vec3> v1, v2;
    std::vector<mesh::Triangle> T1, T2;
    if (!mesh::load_object_file(obj1, v1, T1)) {
        exit(-1);
    }
    assert(!v1.empty() && "no vertices found!");
    assert(!T1.empty() && "no triangles found!");
    if (!mesh::load_object_file(obj2, v2, T2)) {
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
    auto bonsai_collisions = collisions(&tree1, &tree2);
    t1 = clock::now();
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] collision detection : " << bonsai_time << " ms"
              << std::endl;

    std::cout << "collision count: " << bonsai_collisions.size() << std::endl;
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
    run_test(obj1, obj2);
    return 0;
}
