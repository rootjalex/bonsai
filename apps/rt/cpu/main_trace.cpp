#include "bonsai_cpp.h"
#include "rt.h"
#include "util.h"

#include <omp.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace {

// (do not touch)
// AUTO-GENERATED canonical_tree

std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai"))
            break;
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
    if (!err.empty())
        std::cerr << "error: " << err << std::endl;
    if (!result) {
        std::cerr << "failed to load " << object_path << std::endl;
        return {};
    }

    std::vector<Triangle> triangles;
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv == 3) {
                Triangle tri;
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

void run(std::string object, std::string partition, bool is_single_threaded,
         std::vector<int64_t> ray_counts, std::string ray_type,
         std::string layout) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    Heuristic heuristic;
    if (partition == "sah")
        heuristic = Heuristic::SurfaceArea;
    else if (partition == "ms")
        heuristic = Heuristic::MedianSplit;
    else {
        std::cerr << "unexpected construction partitioning strategy: "
                  << partition << std::endl;
        exit(1);
    }

    // we have at most 4 bits for the snapped-grid extent quantization and can
    // add an additional value since we know this value will always be non-zero.
    // Otherwise, we use embree's default.
    const int32_t max_prims_per_leaf = layout.starts_with("eq") ? 15 : 32;
    BVH *canonical_tree =
        build_canonical_tree_$N$(triangles, heuristic, max_prims_per_leaf);

    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_$N$(canonical_tree);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" + ray_type +
                               ".rays";
        std::vector<Ray> rays = load_rays_binary(ray_file, ray_count);
        assert(!rays.empty());

        if (is_first_run) {
            for (int i = 0; i < std::max<size_t>(rays.size(), 512u); ++i)
                (void)trace(&rays[i], &tree); // warmup
            is_first_run = false;
        }

        size_t hit_count = 0;
        auto trace_begin = clock::now(), trace_end = clock::now();

        if (is_single_threaded) {
            std::vector<Triangle> hits;
            hits.reserve(rays.size());
            trace_begin = clock::now();
            for (int i = 0; i < rays.size(); ++i) {
                if (std::optional<Triangle> t = trace(&rays[i], &tree)) {
                    hits.push_back(*t);
                }
            }
            trace_end = clock::now();
            hit_count = hits.size();
        } else {
            const size_t max_threads = omp_get_max_threads();
            std::vector<std::vector<Triangle>> hits_per_thread(max_threads);
            for (std::vector<Triangle> &v : hits_per_thread) {
                v.reserve(rays.size() / max_threads + 64);
            }
            trace_begin = clock::now();

#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                auto &hits = hits_per_thread[tid];

#pragma omp for schedule(dynamic, 64) nowait
                for (int i = 0; i < rays.size(); ++i) {
                    if (const std::optional<Triangle> t =
                            trace(&rays[i], &tree)) {
                        hits.push_back(*t);
                    }
                }
            }

            trace_end = clock::now();
            for (const std::vector<Triangle> &v : hits_per_thread)
                hit_count += v.size();
        }

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits       : " << hit_count << "\n";
        std::cout << "trace time : " << trace_time << " ms\n";
    }
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 7);
    int i = 1;
    std::string object_file = argv[i++];
    std::string partition = argv[i++];
    std::string schedule = argv[i++];
    assert(schedule == "single-thread" || schedule == "parallel");
    const bool is_single_threaded = schedule == "single-thread";
    std::string ray_type = argv[i++];
    std::string layout = argv[i++];

    std::vector<int64_t> ray_counts;
    const int64_t size = std::atoi(argv[i++]);
    ray_counts.reserve(size);
    for (; i < 7 + size; ++i)
        ray_counts.push_back(std::atoi(argv[i]));

    run(object_file, partition, is_single_threaded, ray_counts, ray_type,
        layout);
    return 0;
}
