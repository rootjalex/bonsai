#include <cassert>
#include <chrono>
#include <cmath>
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

// Runs a closest point query test on two OBJ files for Bonsai and FCPW
void run_test(const std::string &obj1, const std::string &obj2) {
    using clock = std::chrono::high_resolution_clock;

    // Load mesh 1 using polyscope (same as FCPW demo)
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

    // Load mesh 2 using polyscope
    std::vector<std::array<double, 3>> verts2;
    std::vector<std::array<std::size_t, 3>> faces2;
    polyscope::loadMesh(obj2, verts2, faces2);

    // Convert to FCPW format
    std::vector<fcpw::Vector3> vertices2;
    for (const auto &v : verts2) {
        vertices2.emplace_back(v[0], v[1], v[2]);
    }
    std::vector<std::size_t> indices2;
    for (const auto &f : faces2) {
        indices2.push_back(f[0]);
        indices2.push_back(f[1]);
        indices2.push_back(f[2]);
    }

    assert(!vertices2.empty() && "no vertices found!");
    assert(!indices2.empty() && "no triangles found!");

    std::cout << "Mesh 1: " << vertices1.size() << " vertices, "
              << indices1.size() / 3 << " triangles" << std::endl;
    std::cout << "Mesh 2: " << vertices2.size() << " vertices, "
              << indices2.size() / 3 << " triangles" << std::endl;
    std::cout << std::endl;

    // ---- FCPW Setup ----
    auto t0 = clock::now();

    // Create FCPW scene for mesh 1
    fcpw::Scene<3> fcpw_scene1;
    std::vector<fcpw::PrimitiveType> object_types1 = {
        fcpw::PrimitiveType::Triangle};
    fcpw_scene1.setObjectTypes(object_types1);

    std::vector<std::vector<fcpw::Vector3>> fcpw_vertices1(1);
    std::vector<std::vector<std::size_t>> fcpw_indices1(1);

    fcpw_vertices1[0] = vertices1;
    fcpw_indices1[0] = indices1;

    fcpw_scene1.setObjectVertices(fcpw_vertices1);
    fcpw_scene1.setObjectTriangles(fcpw_indices1);
    fcpw_scene1.build(fcpw::AggregateType::Bvh_SurfaceArea, true);

    // Create FCPW scene for mesh 2
    fcpw::Scene<3> fcpw_scene2;
    std::vector<fcpw::PrimitiveType> object_types2 = {
        fcpw::PrimitiveType::Triangle};
    fcpw_scene2.setObjectTypes(object_types2);

    std::vector<std::vector<fcpw::Vector3>> fcpw_vertices2(1);
    std::vector<std::vector<std::size_t>> fcpw_indices2(1);

    fcpw_vertices2[0] = vertices2;
    fcpw_indices2[0] = indices2;

    fcpw_scene2.setObjectVertices(fcpw_vertices2);
    fcpw_scene2.setObjectTriangles(fcpw_indices2);
    fcpw_scene2.build(fcpw::AggregateType::Bvh_SurfaceArea, true);

    auto t1 = clock::now();
    auto fcpw_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcpw]   tree construction   : " << fcpw_time << " ms"
              << std::endl;

    // ---- Bonsai tree construction ----
    t0 = clock::now();
    std::vector<Triangle> T1s = construct_triangles(vertices1, indices1);
    std::vector<Triangle> T2s = construct_triangles(vertices2, indices2);

    // Uncomment and adjust for your Bonsai API:
    // BVH *canonical_tree1 = build_fcl_tree_median_split(T1s);
    // BVH *canonical_tree2 = build_fcl_tree_median_split(T2s);
    // Triangles1 tree1 = build_triangles1(canonical_tree1);
    // Triangles2 tree2 = build_triangles2(canonical_tree2);

    t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
              << std::endl;

    // Uncomment for cleanup:
    // free_canonical_tree(canonical_tree1);
    // free_canonical_tree(canonical_tree2);

    // ---- FCPW Closest Point Query ----
    // Sample query points from mesh 1's vertices
    const int num_queries = std::min(100, static_cast<int>(vertices1.size()));
    std::vector<float> fcpw_distances;

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        fcpw::Interaction<3> query_interaction;
        query_interaction.p = vertices1[i];

        fcpw::Interaction<3> closest_interaction;
        bool found = fcpw_scene2.findClosestPoint(query_interaction,
                                                  closest_interaction);

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
        float query_point[3] = {static_cast<float>(vertices1[i][0]),
                                static_cast<float>(vertices1[i][1]),
                                static_cast<float>(vertices1[i][2])};

        // Uncomment and adjust for your Bonsai API:
        // ClosestPointResult result = closest_point(&tree2, query_point);
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
        std::cerr << "Usage: " << argv[0] << " <obj1> <obj2>" << std::endl;
        return -1;
    }

    std::string obj1 = argv[1];
    std::string obj2 = argv[2];
    run_test(obj1, obj2);
    return 0;
}
