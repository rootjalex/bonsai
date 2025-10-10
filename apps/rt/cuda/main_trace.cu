#include "helpers.h"

#include "rt.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

// (do not touch)
// AUTO-GENERATED canonical_tree

std::vector<Ray> load_rays_binary(const std::string &filename,
                                  int64_t ray_count) {
    std::vector<Ray> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cout << "Error: Could not open file " << filename
                  << " for reading\n";
        return rays;
    }

    // Read number of rays
    size_t count;
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (ray_count > count) {
        std::cout << "the requested ray count: " << ray_count
                  << " is greater than the total ray count: " << count
                  << " You need to re-generate the rays.";
    }
    assert(ray_count <= count);

    rays.reserve(ray_count);

    // Read ray data
    for (size_t i = 0; i < ray_count; ++i) {
        Ray ray;
        file.read(reinterpret_cast<char *>(&ray.o.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o.z), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.z), sizeof(float));
        rays.push_back(ray);
    }

    file.close();
    return rays;
}

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
        std::cout << "error: " << err << std::endl;
    }
    if (!result) {
        std::cout << "failed to load " << object_path << std::endl;
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

void run(const std::string &object, const std::string &partition,
         const std::vector<int64_t> &ray_counts) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());
    Heuristic heuristic;
    if (partition == "sah") {
        heuristic = Heuristic::SurfaceArea;
    } else if (partition == "ms") {
        heuristic = Heuristic::MedianSplit;
    } else {
        std::cout << "unexpected construction partitioning strategy: "
                  << partition << std::endl;
        std::cout << std::flush;
        exit(1);
    }
    BVH *canonical_tree = build_canonical_tree_$N$(triangles, heuristic);
    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_$N$(canonical_tree);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" + "camera" +
                               ".rays";
        Ray *rays = nullptr;
        {
            std::vector<Ray> r = load_rays_binary(ray_file, ray_count);
            assert(!r.empty());
            rays = reinterpret_cast<Ray *>(malloc(sizeof(Ray) * ray_count));
            std::copy(r.begin(), r.end(), rays);
            r.clear();
        }
        if (is_first_run) {
            (void)chrt(ray_count, rays, &tree); // warm-up run
            is_first_run = false;
        }

        auto trace_begin = clock::now();
        cuda::std::optional<Triangle> *hits = chrt(ray_count, rays, &tree);
        auto trace_end = clock::now();

        int64_t count = 0;
        for (int i = 0; i < ray_count; ++i) {
            if (hits[i].has_value()) {
                ++count;
            }
        }
        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits             : " << count << "\n";
        std::cout << "trace time       : " << trace_time << " ms\n";
        std::cout << std::flush;
    }
}

bool is_digit(std::string s) {
    for (char c : s) {
        if (std::isdigit(c)) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 4);
    int i = 1;
    std::string object_file = argv[i++];
    std::string partition = argv[i++];
    std::vector<int64_t> ray_counts;
    assert(is_digit(argv[i]));
    const int64_t size = std::atoi(argv[i++]);

    ray_counts.reserve(size);
    for (; i < 4 + size; ++i) {
        assert(is_digit(argv[i]));
        ray_counts.push_back(std::atoi(argv[i]));
    }
    run(object_file, partition, ray_counts);
    return 0;
}
