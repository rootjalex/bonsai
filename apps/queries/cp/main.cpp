#include "runtime/bonsai_cpp.h"
#include "cp_gen.h"

#include <fcpw/fcpw.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// #include <CGAL/Simple_cartesian.h>
// #include <CGAL/AABB_tree.h>
// #include <CGAL/AABB_traits_3.h>
// #include <CGAL/AABB_triangle_primitive_3.h>

#include "runtime/bonsai_benchmark.h"

constexpr int k = 7;
constexpr int m = 1;

std::vector<Triangle> load_obj(const std::string &object) {
    // std::string object_path = "apps/queries/rt/" + object + ".obj";
    // std::string material_path = "apps/queries/rt/" + object;

    // std::string object_path = "/Users/ajroot/Downloads/xxx/" + object + ".obj";
    // std::string material_path = "/Users/ajroot/Downloads/xxx/" + object;

    std::string object_path = object + ".obj";
    std::string material_path = object;

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
                    if (v == 0) {
                        tri.p0[0] = x;
                        tri.p0[1] = y;
                        tri.p0[2] = z;
                    } else if (v == 1) {
                        tri.p1[0] = x;
                        tri.p1[1] = y;
                        tri.p1[2] = z;
                    } else {
                        tri.p2[0] = x;
                        tri.p2[1] = y;
                        tri.p2[2] = z;
                    }
                }
                triangles.push_back(tri);
            }
            index_offset += fv;
        }
    }
    return triangles;
}

_tree_layout0 copy_tree(fcpw::Aggregate<3> *aggregate) {
    auto bvh_ptr = dynamic_cast<fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle>*>(aggregate);
    if (!bvh_ptr) {
        std::cerr << "Error: Aggregate is not a BVH<3>!\n";
        std::abort();
    }

    _tree_layout0 tree;
    tree.nCount = static_cast<uint32_t>(bvh_ptr->flatTree.size());
    if (tree.nCount > 0) {
        tree.group0_index =
            static_cast<_tree_layout1*>(malloc(sizeof(_tree_layout1) * tree.nCount));
        if (!tree.group0_index) {
            std::cerr << "Malloc (nodes) failed" << std::endl;
            abort();
        }
        // copy nodes
        // TODO: could this be a memcpy?
        for (uint32_t i = 0; i < tree.nCount; ++i) {
            tree.group0_index[i].low[0] = bvh_ptr->flatTree[i].box.pMin[0];
            tree.group0_index[i].low[1] = bvh_ptr->flatTree[i].box.pMin[1];
            tree.group0_index[i].low[2] = bvh_ptr->flatTree[i].box.pMin[2];
            tree.group0_index[i].high[0] = bvh_ptr->flatTree[i].box.pMax[0];
            tree.group0_index[i].high[1] = bvh_ptr->flatTree[i].box.pMax[1];
            tree.group0_index[i].high[2] = bvh_ptr->flatTree[i].box.pMax[2];
            tree.group0_index[i].nPrims = bvh_ptr->flatTree[i].nReferences;
            tree.group0_index[i].offset = (tree.group0_index[i].nPrims == 0) ? bvh_ptr->flatTree[i].secondChildOffset : bvh_ptr->flatTree[i].referenceOffset;
        }
    } else {
        std::cerr << "Copying empty tree from FCPW " << std::endl;
        abort();
    }

    tree.pCount = static_cast<uint32_t>(bvh_ptr->primitives.size());
    if (tree.pCount > 0) {
        tree.prims = static_cast<Triangle*>(malloc(sizeof(Triangle) * tree.pCount));
        if (!tree.prims) {
            std::cerr << "Malloc (triangles) failed" << std::endl;
            abort();
        }
        for (uint32_t i = 0; i < tree.pCount; ++i) {
            const auto soup = bvh_ptr->primitives[i]->soup;
            tree.prims[i].p0.x = soup->positions[bvh_ptr->primitives[i]->indices[0]][0];
            tree.prims[i].p0.y = soup->positions[bvh_ptr->primitives[i]->indices[0]][1];
            tree.prims[i].p0.z = soup->positions[bvh_ptr->primitives[i]->indices[0]][2];
            tree.prims[i].p1.x = soup->positions[bvh_ptr->primitives[i]->indices[1]][0];
            tree.prims[i].p1.y = soup->positions[bvh_ptr->primitives[i]->indices[1]][1];
            tree.prims[i].p1.z = soup->positions[bvh_ptr->primitives[i]->indices[1]][2];
            tree.prims[i].p2.x = soup->positions[bvh_ptr->primitives[i]->indices[2]][0];
            tree.prims[i].p2.y = soup->positions[bvh_ptr->primitives[i]->indices[2]][1];
            tree.prims[i].p2.z = soup->positions[bvh_ptr->primitives[i]->indices[2]][2];
            /*
            tree.prims[i].p0.x() = soup->positions[bvh_ptr->primitives[i]->indices[0]][0];
            tree.prims[i].p0.y() = soup->positions[bvh_ptr->primitives[i]->indices[0]][1];
            tree.prims[i].p0.z() = soup->positions[bvh_ptr->primitives[i]->indices[0]][2];
            tree.prims[i].p1.x() = soup->positions[bvh_ptr->primitives[i]->indices[1]][0];
            tree.prims[i].p1.y() = soup->positions[bvh_ptr->primitives[i]->indices[1]][1];
            tree.prims[i].p1.z() = soup->positions[bvh_ptr->primitives[i]->indices[1]][2];
            tree.prims[i].p2.x() = soup->positions[bvh_ptr->primitives[i]->indices[2]][0];
            tree.prims[i].p2.y() = soup->positions[bvh_ptr->primitives[i]->indices[2]][1];
            tree.prims[i].p2.z() = soup->positions[bvh_ptr->primitives[i]->indices[2]][2];
            */
        }
    } else {
        std::cerr << "No triangles?" << std::endl;
        abort();
    }
    return tree;
}




void verify_results(const std::vector<float> &fcpw_distances,
                    const std::vector<float> &bonsai_distances,
                    int64_t num_queries, float distance_tolerance) {
    int distance_mismatches = 0;
    float max_difference = 0.0f;
    int max_difference_idx = -1;

    for (int i = 0; i < num_queries; i++) {
        float fcpw_dist = fcpw_distances[i];
        float bonsai_dist = bonsai_distances[i];
        float distance_diff = std::abs(fcpw_dist - bonsai_dist);

        if (distance_diff > max_difference) {
            max_difference = distance_diff;
            max_difference_idx = i;
        }

        if (distance_diff > distance_tolerance) {
            distance_mismatches++;

            if (distance_mismatches <= 16) {
                std::cerr << "\n=== DISTANCE MISMATCH at query " << i
                          << " ===" << std::endl;
                std::cerr << "FCPW distance:   " << fcpw_dist << std::endl;
                std::cerr << "Bonsai distance: " << bonsai_dist << std::endl;
                std::cerr << "Difference:      " << distance_diff << std::endl;
            }
        }
    }
    if (distance_mismatches == 0) {
        return;
    }

    std::cerr << "\n=== VERIFICATION RESULTS ===" << std::endl;
    std::cerr << "Total queries: " << num_queries << std::endl;
    std::cerr << "Distance tolerance: " << distance_tolerance << std::endl;
    std::cerr << "Distance mismatches (exceeds tolerance): "
              << distance_mismatches << std::endl;
    std::cerr << "Maximum distance difference: " << max_difference;
    if (max_difference_idx >= 0) {
        std::cout << " (at query " << max_difference_idx << ")";
    }
    std::cerr << std::endl;
    std::cerr << "\nERROR: " << distance_mismatches
              << " queries have distances exceeding tolerance of "
              << distance_tolerance << std::endl;
    exit(-1);
}

void run_test(const fcpw::Scene<3> &fcpw_scene, const _tree_layout0 &tree,
              const fcpw::Vector3 &bbox_min, const fcpw::Vector3 &box_extent,
              bool &fcpw_timedout, bool &bonsai_timedout, int64_t num_queries) {
    using clock = std::chrono::high_resolution_clock;

    // Generate random query points within bounding box
    std::vector<fcpw::Vector3> fcpw_points;
    fcpw_points.reserve(num_queries);
    std::vector<Point> bonsai_points;
    bonsai_points.reserve(num_queries);
    std::srand(42); // Fixed seed for reproducibility

    for (int i = 0; i < num_queries; i++) {
        fcpw::Vector3 random_vec(
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        fcpw::Vector3 p(bbox_min[0] + box_extent[0] * random_vec[0],
                        bbox_min[1] + box_extent[1] * random_vec[1],
                        bbox_min[2] + box_extent[2] * random_vec[2]);

        fcpw_points.push_back(p);
        bonsai_points.push_back(Point{
            .vec = {p[0], p[1], p[2]},
        });
    }

    // ---- FCPW Closest Point Query ----
    std::vector<float> fcpw_distances(num_queries);

    // Do warm-up
    for (int i = 0; i < std::min(num_queries, (int64_t)512); i++) {
        fcpw::Interaction<3> closest_interaction;
        bool found =
            fcpw_scene.findClosestPoint(fcpw_points[i], closest_interaction);
        assert(found);
        fcpw_distances[i] = closest_interaction.d;
    }

    // Do benchmarking
    auto fcpw_time = benchmark_function(
                         [&]() {
                             for (int i = 0; i < num_queries; ++i) {
                                 fcpw::Interaction<3> closest_interaction;
                                 bool found = fcpw_scene.findClosestPoint(
                                     fcpw_points[i], closest_interaction);
                                 fcpw_distances[i] = closest_interaction.d;
                             }
                         },
                         fcpw_timedout, k, m) /
                     (double)1e6; // ns -> ms

    // ---- Bonsai Closest Point Query ----
    std::vector<float> bonsai_distances(num_queries);

    // Do warm-up
    for (int i = 0; i < std::min(num_queries, (int64_t)512); i++) {
        bonsai_distances[i] = std::sqrtf(closest(bonsai_points[i], tree));
    }

    auto bonsai_time = benchmark_function(
                           [&]() {
                               for (int i = 0; i < num_queries; ++i) {
                                   bonsai_distances[i] = std::sqrtf(
                                       closest(bonsai_points[i], tree));
                               }
                           },
                           bonsai_timedout, k, m) /
                       (double)1e6;

    verify_results(fcpw_distances, bonsai_distances, num_queries,
                   /*distance_tolerance=*/1e-2f);

    std::cout << "\"fcpw\": " << fcpw_time << ", ";
    std::cout << "\"bonsai\": " << bonsai_time << "}, " << std::endl;

    if (fcpw_time >= (timeout_sec * 1e3) &&
        bonsai_time >= (timeout_sec * 1e3)) {
        // timeout
        exit(0);
    }
}

int main(int argc, char *argv[]) {
    assert(argc == 2);

    std::string obj = argv[1];

    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(obj);
    assert(!triangles.empty());

    std::vector<fcpw::Vector3> vertices1;
    std::vector<fcpw::Vector3i> indices1;

    vertices1.reserve(triangles.size() * 3);
    indices1.reserve(triangles.size());

    for (const Triangle &tri : triangles) {
        int idx0 = vertices1.size();
        vertices1.emplace_back(tri.p0[0], tri.p0[1], tri.p0[2]);
        int idx1 = vertices1.size();
        vertices1.emplace_back(tri.p1[0], tri.p1[1], tri.p1[2]);
        int idx2 = vertices1.size();
        vertices1.emplace_back(tri.p2[0], tri.p2[1], tri.p2[2]);

        indices1.emplace_back(idx0, idx1, idx2);
    }

    // Compute bounding box for random point generation
    fcpw::Vector3 bbox_min = vertices1[0];
    fcpw::Vector3 bbox_max = vertices1[0];
    for (const auto &v : vertices1) {
        for (int i = 0; i < 3; i++) {
            bbox_min[i] = std::min(bbox_min[i], v[i]);
            bbox_max[i] = std::max(bbox_max[i], v[i]);
        }
    }
    fcpw::Vector3 box_extent = bbox_max - bbox_min;

    // ---- FCPW Setup ----
    auto t0 = clock::now();

    // Create FCPW scene for mesh
    fcpw::Scene<3> fcpw_scene;

    // Set object count
    fcpw_scene.setObjectCount(1);

    // Set vertices and triangles for object 0
    fcpw_scene.setObjectVertices(vertices1, 0);
    fcpw_scene.setObjectTriangles(indices1, 0);

    // Build the BVH
    bool printStats = false;
    bool reduceMemoryFootprint = true;
    bool vectorize = false;
    fcpw_scene.build(fcpw::AggregateType::Bvh_SurfaceArea, vectorize,
                     printStats, reduceMemoryFootprint);

    auto t1 = clock::now();
    // auto fcpw_time =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // std::cout << "[fcpw]   tree construction   : " << fcpw_time << " ms"
    //           << std::endl;

    // ---- Bonsai tree construction ----
    // t0 = clock::now();

    auto tree = copy_tree(fcpw_scene.getSceneData()->aggregate.get());

    // t1 = clock::now();
    // auto bonsai_time =
    //     std::chrono::duration_cast<std::chrono::mi lliseconds>(t1 - t0).count();
    // std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
    //           << std::endl;

    bool fcpw_timedout = false, bonsai_timedout = false;

    for (size_t i = 8; i < 27; i++) {
        uint64_t count = 1ull << i;
        std::cout << "    {\"size\": " << count << ", ";
        run_test(fcpw_scene, tree, bbox_min, box_extent, fcpw_timedout,
                 bonsai_timedout, count);
    }

    return 0;
}