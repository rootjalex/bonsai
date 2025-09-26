#include "bonsai_cpp.h"
#include "canonical_tree.h"
#include "rt.h"
#include "util.h"
#include <omp.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <random>

namespace {

std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai")) {
            break;
        }
        current_path = current_path.parent_path();
    }

    std::string object_path = "apps/rt/data/" + object + ".obj";
    std::string material_path = "apps/rt/data/" + object;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string _, err;
    bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &_, &err,
                                   object_path.c_str(), material_path.c_str());
    if (!err.empty()) {
        std::cerr << "error: " << err << std::endl;
    }
    if (!result) {
        std::cerr << "failed to load " << object_path << std::endl;
        return {};
    }

    std::vector<Triangle> triangles;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;

        // Loop over faces (triangles)
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv =
                shapes[s]
                    .mesh.num_face_vertices[f]; // Should be 3 for triangles

            if (fv == 3) {
                Triangle tri;

                // Get vertices
                for (int v = 0; v < 3; v++) {
                    tinyobj::index_t idx =
                        shapes[s].mesh.indices[index_offset + v];

                    float x = attrib.vertices[3 * idx.vertex_index + 0];
                    float y = attrib.vertices[3 * idx.vertex_index + 1];
                    float z = attrib.vertices[3 * idx.vertex_index + 2];

                    if (v == 0)
                        tri.p0 = {x, y, z};
                    else if (v == 1)
                        tri.p1 = {x, y, z};
                    else
                        tri.p2 = {x, y, z};
                }

                triangles.push_back(tri);
            }

            index_offset += fv;
        }
    }
    return triangles;
}

void run_test(const std::string &object_file, const std::string &ray_file,
              int64_t ray_count) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(object_file);
    assert(!triangles.empty());

    auto ct_begin = clock::now();
    BVH *canonical_tree = build_canonical_tree(triangles);
    auto ct_end = clock::now();

    Triangles tree = build_triangles(canonical_tree);

    auto st_begin = clock::now();
    free_canonical_tree(canonical_tree);
    auto st_end = clock::now();

    std::vector<Ray> rays = load_rays_binary(ray_file, ray_count);
    // PARALLEL
    // std::vector<Triangle> hits;
    //     std::vector<std::optional<Triangle>> results(rays.size());
    //     auto trace_begin = clock::now();

    // #pragma omp parallel for schedule(dynamic)
    //     for (size_t i = 0; i < rays.size(); ++i) {
    //         results[i] = trace(&rays[i], &tree);
    //     }
    //     // Collect hits sequentially after parallel work
    //     for (const std::optional<Triangle> &result : results) {
    //         if (!result.has_value()) {
    //             continue;
    //         }
    //         hits.push_back(*result);
    //     }
    //     auto trace_end = clock::now();

    // SINGLE-THREAD
    std::vector<Triangle> hits;
    hits.reserve(rays.size());
    auto trace_begin = clock::now();
    for (int i = 0; i < rays.size(); ++i) {
        if (const std::optional<Triangle> t = trace(&rays[i], &tree)) {
            hits.push_back(*t);
        }
    }
    auto trace_end = clock::now();

    auto ct_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(ct_end - ct_begin)
            .count();
    auto st_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(st_end - st_begin)
            .count();
    auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          trace_end - trace_begin)
                          .count();
    std::cout << "hits             : " << hits.size() << "\n";
    std::cout << "canonical tree   : " << ct_time << "ms\n";
    std::cout << "specialized tree : " << st_time << "ms\n";
    std::cout << "trace time       : " << trace_time << " ms\n";
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc == 4);
    std::string object_file = argv[1];
    int64_t ray_count = std::atoi(argv[2]);
    std::string ray_file = argv[3];
    // Wavefront OBJ files are taken from FCL [1] from `prims` [2].
    // [1] https://github.com/flexible-collision-library/fcl
    // [2] https://github.com/nickdesaulniers/prims/tree/master/meshes
    run_test(object_file, ray_file, ray_count);
    return 0;
}
