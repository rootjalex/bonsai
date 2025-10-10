#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <vector>

// FCPW includes
#include <fcpw/fcpw.h>

// Polyscope for OBJ loading (same as FCPW demo)
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

// Bonsai includes (adjust as needed)
// #include "bonsai.h"

namespace {

struct Triangle {
    float v0[3];
    float v1[3];
    float v2[3];
};

// Helper to construct Triangle from vertex data
Triangle construct_triangle(const std::vector<fcpw::Vector3> &vertices,
                            std::size_t i0, std::size_t i1, std::size_t i2) {
    Triangle t;
    const auto &v0 = vertices[i0];
    const auto &v1 = vertices[i1];
    const auto &v2 = vertices[i2];

    for (int i = 0; i < 3; i++) {
        t.v0[i] = static_cast<float>(v0[i]);
        t.v1[i] = static_cast<float>(v1[i]);
        t.v2[i] = static_cast<float>(v2[i]);
    }
    return t;
}

std::vector<Triangle>
construct_triangles(const std::vector<fcpw::Vector3> &vertices,
                    const std::vector<std::size_t> &indices) {
    std::vector<Triangle> triangles;
    for (std::size_t i = 0; i < indices.size(); i += 3) {
        triangles.push_back(construct_triangle(vertices, indices[i],
                                               indices[i + 1], indices[i + 2]));
    }
    return triangles;
}

// Helper to compute distance between two points
float distance(const float p1[3], const float p2[3]) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Runs a closest point query test on an OBJ file for Bonsai and FCPW
void run_test(const std::string &obj1, int64_t num_queries) {
    using clock = std::chrono::high_resolution_clock;

    // Load mesh using polyscope (same as FCPW demo)
    std::vector<std::array<double, 3>> verts1;
    std::vector<std::array<std::size_t, 3>> faces1;
    polyscope::loadMesh(obj1, verts1, faces1);

    // Convert to FCPW format
    std::vector<fcpw::Vector3> vertices1;
    for (const auto &v : verts1) {
        vertices1.emplace_back(v[0], v[1], v[2]);
    }
    std::vector<std::size_t> indices1;
    for (const auto &f : faces1) {
        indices1.push_back(f[0]);
        indices1.push_back(f[1]);
        indices1.push_back(f[2]);
    }

    assert(!vertices1.empty() && "no vertices found!");
    assert(!indices1.empty() && "no triangles found!");

    std::cout << "Mesh: " << vertices1.size() << " vertices, "
              << indices1.size() / 3 << " triangles" << std::endl;

    // Compute bounding box for random point generation
    fcpw::Vector3 bbox_min = vertices1[0];
    fcpw::Vector3 bbox_max = vertices1[0];
    for (const auto &v : vertices1) {
        for (int i = 0; i < 3; i++) {
            bbox_min[i] = std::min(bbox_min[i], v[i]);
            bbox_max[i] = std::max(bbox_max[i], v[i]);
        }
    }

    // Expand bounding box slightly
    fcpw::Vector3 bbox_size = bbox_max - bbox_min;
    bbox_min = bbox_min - bbox_size * 0.1f;
    bbox_max = bbox_max + bbox_size * 0.1f;

    // Generate random query points within bounding box
    std::vector<fcpw::Vector3> query_points;
    std::srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < num_queries; i++) {
        fcpw::Vector3 p;
        for (int j = 0; j < 3; j++) {
            float t =
                static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            p[j] = bbox_min[j] + t * (bbox_max[j] - bbox_min[j]);
        }
        query_points.push_back(p);
    }

    std::cout << "Generated " << num_queries << " random query points"
              << std::endl;
    std::cout << std::endl;

    // ---- FCPW Setup ----
    auto t0 = clock::now();

    // Create FCPW scene for mesh
    fcpw::Scene<3> fcpw_scene;
    std::vector<fcpw::PrimitiveType> object_types = {
        fcpw::PrimitiveType::Triangle};
    fcpw_scene.setObjectTypes(object_types);

    std::vector<std::vector<fcpw::Vector3>> fcpw_vertices(1);
    std::vector<std::vector<std::size_t>> fcpw_indices(1);

    fcpw_vertices[0] = vertices1;
    fcpw_indices[0] = indices1;

    fcpw_scene.setObjectVertices(fcpw_vertices);
    fcpw_scene.setObjectTriangles(fcpw_indices);
    fcpw_scene.build(fcpw::AggregateType::Bvh_SurfaceArea, true);

    auto t1 = clock::now();
    auto fcpw_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcpw]   tree construction   : " << fcpw_time << " ms"
              << std::endl;

    // ---- Bonsai tree construction ----
    t0 = clock::now();
    std::vector<Triangle> T1s = construct_triangles(vertices1, indices1);

    // Uncomment and adjust for your Bonsai API:
    // BVH *canonical_tree = build_fcl_tree_median_split(T1s);
    // Triangles tree = build_triangles(canonical_tree);

    t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
              << std::endl;

    // Uncomment for cleanup:
    // free_canonical_tree(canonical_tree);

    // ---- FCPW Closest Point Query ----
    std::vector<float> fcpw_distances;

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        fcpw::Interaction<3> query_interaction;
        query_interaction.p = query_points[i];

        fcpw::Interaction<3> closest_interaction;
        bool found =
            fcpw_scene.findClosestPoint(query_interaction, closest_interaction);

        if (found) {
            float dist = (closest_interaction.p - query_interaction.p).norm();
            fcpw_distances.push_back(dist);
        }
    }
    t1 = clock::now();
    fcpw_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcpw]   closest point query : " << fcpw_time << " ms "
              << "(" << num_queries << " queries)" << std::endl;

    // ---- Bonsai Closest Point Query ----
    std::vector<float> bonsai_distances;

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        float query_point[3] = {static_cast<float>(query_points[i][0]),
                                static_cast<float>(query_points[i][1]),
                                static_cast<float>(query_points[i][2])};

        // Uncomment and adjust for your Bonsai API:
        // ClosestPointResult result = closest_point(&tree, query_point);
        // bonsai_distances.push_back(result.distance);

        // Placeholder for now:
        bonsai_distances.push_back(0.0f);
    }
    t1 = clock::now();
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] closest point query : " << bonsai_time << " ms "
              << "(" << num_queries << " queries)" << std::endl;

    // ---- Verify Results Match ----
    const float tolerance = 1e-5f;
    int mismatches = 0;
    float max_error = 0.0f;

    for (int i = 0; i < num_queries; i++) {
        float error = std::abs(fcpw_distances[i] - bonsai_distances[i]);
        max_error = std::max(max_error, error);

        if (error > tolerance) {
            mismatches++;
            if (mismatches <= 5) {
                std::cerr << "Mismatch at query " << i << ": "
                          << "fcpw=" << fcpw_distances[i] << " vs "
                          << "bonsai=" << bonsai_distances[i] << " "
                          << "(error=" << error << ")" << std::endl;
            }
        }
    }

    if (mismatches > 0) {
        std::cerr << "Total mismatches: " << mismatches << " out of "
                  << num_queries << " queries" << std::endl;
        std::cerr << "Maximum error: " << max_error << std::endl;
        exit(-1);
    }

    std::cout << "All " << num_queries << " queries matched!" << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    std::cout << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <obj_file> [num_queries]"
                  << std::endl;
        return -1;
    }

    std::string obj1 = argv[1];
    int64_t num_queries = std::atoi(argv[2]);

    run_test(obj1, num_queries);
    return 0;
}