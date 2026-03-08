#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

#include "bonsai_cpp.h"
#include "wos.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <fcpw/fcpw.h>

namespace {

// (do not touch)
// AUTO-GENERATED canonical_tree
float3 triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

constexpr float FCPW_EPSILON = std::numeric_limits<float>::epsilon();

// https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/core/bounding_volumes.h#L135

inline float surface_area(float3 box_min, float3 box_max) {
    float3 e = max(box_max - box_min, float3{1e-5f, 1e-5f, 1e-5f});
    float prod = e.x * e.y * e.z;
    return 2.0f * (prod / e.x + prod / e.y + prod / e.z);
}

inline std::pair<float3, float3>
compute_triangle_bbox_fcpw(const Triangle &tri) {
    // Start with invalid box
    float3 pMin = float3{std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
    float3 pMax = float3{std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};

    // expandToInclude each vertex (with epsilon like FCPW)
    float3 eps = float3{FCPW_EPSILON, FCPW_EPSILON, FCPW_EPSILON};
    for (int i = 0; i < 3; i++) {
        float3 v;
        if (i == 0)
            v = tri.p0;
        if (i == 1)
            v = tri.p1;
        if (i == 2)
            v = tri.p2;
        pMin = min(pMin, v - eps);
        pMax = max(pMax, v + eps);
    }

    return {pMin, pMax};
}

// Based on FCPW's BVH SAH implementation.
// https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L143
int64_t node_count = 0, leaf_count = 0; // Track counts for comparison.
BVH *build_canonical_tree_2_sah(std::vector<Triangle> &triangles, int leaf_size,
                                int max_tree_depth, int nBuckets = 8) {
    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr auto MIN = std::numeric_limits<float>::lowest();

    struct BoundingBox {
        float3 pMin = float3{MAX, MAX, MAX};
        float3 pMax = float3{MIN, MIN, MIN};

        void expandToInclude(const BoundingBox &b) {
            pMin = min(pMin, b.pMin);
            pMax = max(pMax, b.pMax);
        }

        void expandToInclude(float3 centroid) {
            float3 eps = float3{FCPW_EPSILON, FCPW_EPSILON, FCPW_EPSILON};
            pMin = min(pMin, centroid - eps);
            pMax = max(pMax, centroid + eps);
        }

        float3 extent() const { return pMax - pMin; }
    };

    // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L212
    std::vector<std::pair<float3, float3>> referenceBoxes(triangles.size());
    std::vector<float3> referenceCentroids(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) {
        referenceBoxes[i] = compute_triangle_bbox_fcpw(triangles[i]);
        referenceCentroids[i] = triangle_centroid(triangles[i]);
    }

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t start, uint32_t end, uint32_t depth) -> BVH * {
        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L157
        node_count++;
        uint32_t nReferences = end - start;

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L159
        BoundingBox bb, bc;
        for (uint32_t p = start; p < end; p++) {
            bb.pMin = min(bb.pMin, referenceBoxes[p].first);
            bb.pMax = max(bb.pMax, referenceBoxes[p].second);
            bc.expandToInclude(referenceCentroids[p]);
        }

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L170
        if (nReferences <= leaf_size || depth >= max_tree_depth) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * nReferences));
            for (uint32_t i = 0; i < nReferences; ++i) {
                data[i] = triangles[start + i];
            }
            leaf_count++;
            return new BVH(Leaf{
                .low = bb.pMin,
                .high = bb.pMax,
                .nprims = static_cast<uint8_t>(nReferences),
                .data = data,
            });
        }

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L200
        float3 extent = bb.extent();
        int splitDim = -1;
        float splitCoord = 0.0f;
        float splitCost = MAX;

        for (int dim = 0; dim < 3; dim++) {
            float extent_dim = extent[dim];
            if (extent_dim < 1e-6f)
                continue;

            float bucketWidth = extent_dim / nBuckets;

            // Bin references into buckets.
            std::vector<BoundingBox> buckets(nBuckets);
            std::vector<int> bucketCounts(nBuckets, 0);

            for (uint32_t p = start; p < end; p++) {
                float centroid_dim = referenceCentroids[p][dim];
                int bucketIndex =
                    (int)((centroid_dim - bb.pMin[dim]) / bucketWidth);
                bucketIndex = std::clamp(bucketIndex, 0, nBuckets - 1);

                buckets[bucketIndex].pMin =
                    min(buckets[bucketIndex].pMin, referenceBoxes[p].first);
                buckets[bucketIndex].pMax =
                    max(buckets[bucketIndex].pMax, referenceBoxes[p].second);
                bucketCounts[bucketIndex]++;
            }

            // Sweep right to left to build right bucket bounding boxes.
            std::vector<BoundingBox> rightBuckets(nBuckets);
            std::vector<int> rightBucketCounts(nBuckets, 0);
            BoundingBox boxRefRight;

            for (int b = nBuckets - 1; b > 0; b--) {
                boxRefRight.expandToInclude(buckets[b]);
                rightBuckets[b] = boxRefRight;
                rightBucketCounts[b] = bucketCounts[b];
                if (b != nBuckets - 1) {
                    rightBucketCounts[b] += rightBucketCounts[b + 1];
                }
            }

            // Evaluate bucket split costs.
            BoundingBox boxRefLeft;
            int nReferencesLeft = 0;

            for (int b = 1; b < nBuckets; b++) {
                boxRefLeft.expandToInclude(buckets[b - 1]);
                nReferencesLeft += bucketCounts[b - 1];
                if (nReferencesLeft > 0 && rightBucketCounts[b] > 0) {
                    float leftArea =
                        surface_area(boxRefLeft.pMin, boxRefLeft.pMax);
                    float rightArea = surface_area(rightBuckets[b].pMin,
                                                   rightBuckets[b].pMax);
                    float cost = nReferencesLeft * leftArea +
                                 rightBucketCounts[b] * rightArea;
                    if (cost < splitCost) {
                        splitCost = cost;
                        splitDim = dim;
                        splitCoord = bb.pMin[dim] + b * bucketWidth;
                    }
                }
            }
        }
        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L104
        if (splitDim == -1) {
            float3 centroid_extent = bc.extent();
            splitDim = (centroid_extent.x > centroid_extent.y &&
                        centroid_extent.x > centroid_extent.z)
                           ? 0
                       : (centroid_extent.y > centroid_extent.z) ? 1
                                                                 : 2;
            splitCoord = (bc.pMin[splitDim] + bc.pMax[splitDim]) * 0.5f;
        }

        uint32_t mid = start;
        for (uint32_t i = start; i < end; i++) {
            if (referenceCentroids[i][splitDim] < splitCoord) {
                std::swap(triangles[i], triangles[mid]);
                std::swap(referenceBoxes[i], referenceBoxes[mid]);
                std::swap(referenceCentroids[i], referenceCentroids[mid]);
                mid++;
            }
        }
        if (mid == start || mid == end) {
            mid = start + nReferences / 2;
        }

        // Recursively build left and right children
        BVH *left = partition(start, mid, depth + 1);
        BVH *right = partition(mid, end, depth + 1);

        return new BVH(Interior{
            .low = bb.pMin,
            .high = bb.pMax,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), 0);
}

void free_canonical_tree_2(BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree_2(interior.left);
        free_canonical_tree_2(interior.right);
        delete node;
        return;
    }

    if (std::holds_alternative<Leaf>(*node)) {
        Leaf &leaf = std::get<Leaf>(*node);
        free(leaf.data);
        delete node;
        return;
    }

    assert(false && "unexpected");
}

enum class Heuristic {
    SurfaceArea = 0,
    MedianSplit = 1,
};

BVH *build_canonical_tree_2(std::vector<Triangle> &triangles,
                            Heuristic heuristic = Heuristic::SurfaceArea,
                            int max_prims_per_leaf = 4,
                            int max_tree_depth = 64) {
    switch (heuristic) {
    case Heuristic::SurfaceArea: {
        return build_canonical_tree_2_sah(triangles, max_prims_per_leaf,
                                          max_tree_depth);
    }
    case Heuristic::MedianSplit:
        assert(false && "unimplemented");
        return nullptr;
    }
}

std::tuple<Point, Point> closest_point(const Point *__restrict__ pt,
                                       const Triangle *__restrict__ tri) {
    const float3 p = (*pt).vec;
    const float3 a = (*tri).p0;
    const float3 b = (*tri).p1;
    const float3 c = (*tri).p2;
    const float3 ab = (b - a);
    const float3 ac = (c - a);
    const float3 ap = (p - a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if ((d1 <= 0.0f)) {
        if ((d2 <= 0.0f)) {
            return std::tuple<Point, Point>{
                Point{.vec = a}, Point{.vec = float3{1.0f, 0.0f, 0.0f}}};
        }
    }
    const float3 bp = (p - b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if ((0.0f <= d3)) {
        if ((d4 <= d3)) {
            return std::tuple<Point, Point>{
                Point{.vec = b}, Point{.vec = float3{0.0f, 1.0f, 0.0f}}};
        }
    }
    const float _t2 = ((d1 * d4) - (d3 * d2));
    if ((_t2 <= 0.0f)) {
        if ((0.0f <= d1)) {
            if ((d3 <= 0.0f)) {
                const float _t4 = (d1 / (d1 - d3));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (float3{_t4} * ab))},
                    Point{.vec = float3{(1.0f - _t4), _t4, 0.0f}}};
            }
        }
    }
    const float3 cp = (p - c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if ((0.0f <= d6)) {
        if ((d5 <= d6)) {
            return std::tuple<Point, Point>{
                Point{.vec = c}, Point{.vec = float3{0.0f, 0.0f, 1.0f}}};
        }
    }
    const float _t7 = ((d5 * d2) - (d1 * d6));
    if ((_t7 <= 0.0f)) {
        if ((0.0f <= d2)) {
            if ((d6 <= 0.0f)) {
                const float _t9 = (d2 / (d2 - d6));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (float3{_t9} * ac))},
                    Point{.vec = float3{(1.0f - _t9), 0.0f, _t9}}};
            }
        }
    }
    const float _t12 = ((d3 * d6) - (d5 * d4));
    if ((_t12 <= 0.0f)) {
        const float _t19 = (d4 - d3);
        if ((0.0f <= _t19)) {
            const float _t18 = (d5 - d6);
            if ((0.0f <= _t18)) {
                const float _t17 = (_t19 / (_t19 + _t18));
                return std::tuple<Point, Point>{
                    Point{.vec = (b + (float3{_t17} * (c - b)))},
                    Point{.vec = float3{0.0f, (1.0f - _t17), _t17}}};
            }
        }
    }
    const float _t22 = (1.0f / ((_t12 + _t7) + _t2));
    const float v = (_t7 * _t22);
    const float w = (_t2 * _t22);
    const float u = (_t12 * _t22);
    return std::tuple<Point, Point>{
        Point{.vec = ((a + (ab * float3{v})) + (ac * float3{w}))},
        Point{.vec = float3{u, v, w}}};
}

float distmin(const Point *__restrict__ p, const Triangle *__restrict__ tri) {
    const std::tuple<Point, Point> pts = closest_point(p, tri);
    return norm(((*p).vec - std::get<0>(pts).vec));
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

} // namespace

std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai"))
            break;
        current_path = current_path.parent_path();
    }

    std::string object_path = "artifact/scenes/rt/" + object + ".obj";
    std::string material_path = "artifact/scenes/rt/" + object;

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

void run_test(const std::string &obj1, int64_t num_queries) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(obj1);
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

    // Generate random query points within bounding box
    std::vector<fcpw::Vector3> query_points;
    std::srand(42); // Fixed seed for reproducibility

    for (int i = 0; i < num_queries; i++) {
        fcpw::Vector3 random_vec(
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        fcpw::Vector3 p(bbox_min[0] + box_extent[0] * random_vec[0],
                        bbox_min[1] + box_extent[1] * random_vec[1],
                        bbox_min[2] + box_extent[2] * random_vec[2]);

        query_points.push_back(p);
    }

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
    auto fcpw_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcpw]   tree construction   : " << fcpw_time << " ms"
              << std::endl;

    // ---- Bonsai tree construction ----
    t0 = clock::now();

    BVH *canonical_tree =
        build_canonical_tree_2(triangles, Heuristic::SurfaceArea);
    Triangles tree = build_triangles(canonical_tree);

    t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
              << std::endl;

    free_canonical_tree_2(canonical_tree);

    // ---- FCPW Closest Point Query ----
    std::vector<float> fcpw_distances;
    fcpw_distances.reserve(num_queries);

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        fcpw::Interaction<3> closest_interaction;
        bool found =
            fcpw_scene.findClosestPoint(query_points[i], closest_interaction);
        assert(found);
        fcpw_distances.push_back(closest_interaction.d);
    }
    t1 = clock::now();
    fcpw_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[fcpw]   closest point query : " << fcpw_time << " ms"
              << std::endl;

    // ---- Bonsai Closest Point Query ----
    std::vector<float> bonsai_distances;
    bonsai_distances.reserve(num_queries);

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        Point point{
            .vec = {query_points[i][0], query_points[i][1], query_points[i][2]},
        };
        Triangle result = closest_point(&point, &tree);

        // Compute the actual distance to the returned triangle.
        float dist = distmin(&point, &result);
        bonsai_distances.push_back(dist);
    }
    t1 = clock::now();
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] closest point query : " << bonsai_time << " ms"
              << std::endl;

    verify_results(fcpw_distances, bonsai_distances, num_queries,
                   /*distance_tolerance=*/1);
}

int main(int argc, char *argv[]) {
    assert(argc == 3);

    std::string obj = argv[1];
    int64_t n_queries = std::atoll(argv[2]);

    run_test(obj, n_queries);
    return 0;
}
