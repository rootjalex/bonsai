#include "helpers.h"

#include "wos.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
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

std::vector<Point>
generate_random_points(const std::vector<Triangle> &triangles,
                       int64_t num_queries) {
    // Compute bounding box
    float3 bbox_min = triangles[0].p0;
    float3 bbox_max = triangles[0].p0;

    for (const Triangle &tri : triangles) {
        for (int v = 0; v < 3; v++) {
            float3 vertex = (v == 0) ? tri.p0 : (v == 1) ? tri.p1 : tri.p2;
            bbox_min.x = std::min(bbox_min.x, vertex.x);
            bbox_min.y = std::min(bbox_min.y, vertex.y);
            bbox_min.z = std::min(bbox_min.z, vertex.z);
            bbox_max.x = std::max(bbox_max.x, vertex.x);
            bbox_max.y = std::max(bbox_max.y, vertex.y);
            bbox_max.z = std::max(bbox_max.z, vertex.z);
        }
    }

    float3 box_extent = {bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y,
                         bbox_max.z - bbox_min.z};

    // Generate random query points
    std::vector<Point> query_points;
    query_points.reserve(num_queries);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int64_t i = 0; i < num_queries; i++) {
        float3 random_vec = {dist(rng), dist(rng), dist(rng)};
        Point p;
        p.vec.x = bbox_min.x + box_extent.x * random_vec.x;
        p.vec.y = bbox_min.y + box_extent.y * random_vec.y;
        p.vec.z = bbox_min.z + box_extent.z * random_vec.z;
        query_points.push_back(p);
    }

    return query_points;
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
            int fv = shapes[s].mesh.num_face_vertices[f];

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
         const std::string &layout, int64_t n_runs, int64_t n_queries) {
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

    std::cout << "Running " << n_runs << " iterations with " << n_queries
              << " queries per run" << std::endl;

    // Generate random query points once (reused across runs)
    std::vector<Point> query_points =
        generate_random_points(triangles, n_queries);
    assert(!query_points.empty());

    Point *points =
        reinterpret_cast<Point *>(malloc(sizeof(Point) * n_queries));
    std::copy(query_points.begin(), query_points.end(), points);

    // Warm-up run
    (void)closest_points(n_queries, points, &tree);

    // Statistics tracking
    std::vector<long long> query_times;
    query_times.reserve(n_runs);

    // Run N times
    for (int64_t run = 0; run < n_runs; run++) {
        auto query_begin = clock::now();
        Triangle *results = closest_points(n_queries, points, &tree);
        auto query_end = clock::now();

        auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              query_end - query_begin)
                              .count();
        query_times.push_back(query_time);

        std::cout << "Run " << (run + 1) << "/" << n_runs
                  << " - query time: " << query_time << " ms" << std::endl;

        free(results);
    }

    // Calculate statistics
    long long total_time = 0;
    long long min_time = query_times[0];
    long long max_time = query_times[0];

    for (auto time : query_times) {
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }

    double avg_time = static_cast<double>(total_time) / n_runs;

    // Calculate median (if N >= 5, we could also calculate with trimmed mean)
    std::vector<long long> sorted_times = query_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    long long median_time = sorted_times[n_runs / 2];

    // Output statistics
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "n_queries        : " << n_queries << std::endl;
    std::cout << "n_runs           : " << n_runs << std::endl;
    std::cout << "min time         : " << min_time << " ms" << std::endl;
    std::cout << "max time         : " << max_time << " ms" << std::endl;
    std::cout << "median time      : " << median_time << " ms" << std::endl;
    std::cout << "avg time         : " << avg_time << " ms" << std::endl;
    std::cout << "total time       : " << total_time << " ms" << std::endl;
    std::cout << std::flush;

    free(points);
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
    // Expected arguments: ./program object partition layout N N_QUERIES
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <object> <partition> <layout> <N_runs> <N_queries>"
                  << std::endl;
        return 1;
    }

    std::string object_file = argv[1];
    std::string partition = argv[2];
    std::string layout = argv[3];

    assert(is_digit(argv[4]));
    int64_t n_runs = std::atoi(argv[4]);

    assert(is_digit(argv[5]));
    int64_t n_queries = std::atoi(argv[5]);

    run(object_file, partition, layout, n_runs, n_queries);

    return 0;
}
