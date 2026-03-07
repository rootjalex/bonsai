#include <embree4/rtcore.h>
#include <embree4/rtcore_builder.h>

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

// (do not touch)
// AUTO-GENERATED canonical_tree

struct Vec3 {
    float x, y, z;
};

void error_function(void *ptr, enum RTCError error, const char *str) {
    std::cerr << "Embree error " << error << ": " << str << std::endl;
}

RTCDevice create_device(const std::string &layout) {
    std::string configuration;
    if (layout == "bvh8i") {
        configuration = "tri_accel=bvh8.triangle4i"; // Moeller
    } else if (layout == "bvh8") {
        configuration = "tri_accel=bvh8.triangle4"; // Moeller
    } else if (layout == "bvh8v") {
        configuration = "tri_accel=bvh8.triangle4v"; // Pluecker
    } else if (layout == "qbvh8i") {
        configuration = "tri_accel=qbvh8.triangle4i"; // Pluecker
    } else if (layout == "qbvh8") {
        configuration = "tri_accel=qbvh8.triangle4"; // Moeller
    } else {
        std::cerr << "Unknown BVH layout: " << layout << std::endl;
        exit(1);
    }
    // intersection cost and traversal cost are defaulted to 1.0.
    // 8 * triangle4 = 32 triangles
    configuration += ",object_accel_max_leaf_size=8";
    // binned SAH
    configuration += ",quality=medium";

    // (Gathers information about the BVH structure)
    // configuration += ",verbose=2,benchmark=2";

    RTCDevice device = rtcNewDevice(configuration.c_str());

    if (!device) {
        std::cerr << "Failed to create Embree device\n";
        exit(1);
    }

    rtcSetDeviceErrorFunction(device, error_function, nullptr);
    return device;
}

std::tuple<Point, Point> closest_point_tri(const Point *__restrict__ pt,
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

float distmin_tri(const Point *__restrict__ p,
                  const Triangle *__restrict__ tri) {
    const std::tuple<Point, Point> pts = closest_point_tri(p, tri);
    return norm(((*p).vec - std::get<0>(pts).vec));
}

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

void verify_results(const std::vector<float> &embree_distances,
                    const std::vector<float> &bonsai_distances,
                    int64_t num_queries, float distance_tolerance) {
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < num_queries; i++) {
        float diff = std::abs(embree_distances[i] - bonsai_distances[i]);
        max_diff = std::max(max_diff, diff);

        if (diff > distance_tolerance) {
            mismatches++;
            if (mismatches <= 10) { // Only print first 10 mismatches
                std::cout << "Mismatch at query " << i
                          << ": embree=" << embree_distances[i]
                          << ", bonsai=" << bonsai_distances[i]
                          << ", diff=" << diff << std::endl;
            }
        }
    }
    if (mismatches == 0) {
        return;
    }
    std::cout << "Verification: " << (num_queries - mismatches) << "/"
              << num_queries
              << " queries matched (tolerance=" << distance_tolerance << ")"
              << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;

    if (mismatches > 0) {
        std::cout << "WARNING: " << mismatches << " mismatches found!"
                  << std::endl;
    }
}

void run_test(const std::string &obj1, int64_t num_queries) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(obj1);
    assert(!triangles.empty());

    std::vector<float>
        vertices1; // Store as flat array: x0,y0,z0, x1,y1,z1, ...
    std::vector<unsigned int>
        indices1; // Store as flat array: i0,i1,i2, i3,i4,i5, ...

    vertices1.reserve(triangles.size() * 3 * 3); // 3 vertices * 3 components
    indices1.reserve(triangles.size() * 3);

    for (const Triangle &tri : triangles) {
        int idx0 = vertices1.size() / 3;
        vertices1.push_back(tri.p0[0]);
        vertices1.push_back(tri.p0[1]);
        vertices1.push_back(tri.p0[2]);

        int idx1 = vertices1.size() / 3;
        vertices1.push_back(tri.p1[0]);
        vertices1.push_back(tri.p1[1]);
        vertices1.push_back(tri.p1[2]);

        int idx2 = vertices1.size() / 3;
        vertices1.push_back(tri.p2[0]);
        vertices1.push_back(tri.p2[1]);
        vertices1.push_back(tri.p2[2]);

        indices1.push_back(idx0);
        indices1.push_back(idx1);
        indices1.push_back(idx2);
    }

    // Compute bounding box for random point generation
    float bbox_min[3] = {vertices1[0], vertices1[1], vertices1[2]};
    float bbox_max[3] = {vertices1[0], vertices1[1], vertices1[2]};

    for (size_t i = 0; i < vertices1.size(); i += 3) {
        for (int j = 0; j < 3; j++) {
            bbox_min[j] = std::min(bbox_min[j], vertices1[i + j]);
            bbox_max[j] = std::max(bbox_max[j], vertices1[i + j]);
        }
    }

    float box_extent[3] = {bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1],
                           bbox_max[2] - bbox_min[2]};

    // Generate random query points within bounding box.
    std::vector<Vec3> query_points;
    std::srand(42); // Fixed seed for reproducibility

    for (int i = 0; i < num_queries; i++) {
        float rx =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float ry =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float rz =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        Vec3 p{bbox_min[0] + box_extent[0] * rx,
               bbox_min[1] + box_extent[1] * ry,
               bbox_min[2] + box_extent[2] * rz};

        query_points.push_back(p);
    }

    // ---- Embree Setup ----
    auto t0 = clock::now();

    // Create Embree device and scene
    RTCDevice device = create_device("bvh8i");
    if (!device) {
        std::cerr << "Error: Failed to create Embree device" << std::endl;
        return;
    }

    RTCScene scene = rtcNewScene(device);

    // Create geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Set vertex buffer
    float *vb = (float *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
        vertices1.size() / 3);
    std::memcpy(vb, vertices1.data(), vertices1.size() * sizeof(float));

    // Set index buffer
    unsigned *ib = (unsigned *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
        indices1.size() / 3);
    std::memcpy(ib, indices1.data(), indices1.size() * sizeof(unsigned));

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    auto t1 = clock::now();
    auto embree_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[embree] tree construction   : " << embree_time << " ms"
              << std::endl;

    // ---- Bonsai tree construction ----
    t0 = clock::now();

    BVH *canonical_tree =
        build_canonical_tree_$N$(triangles, Heuristic::SurfaceArea);
    Triangles tree = build_triangles(canonical_tree);

    t1 = clock::now();
    auto bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] tree construction   : " << bonsai_time << " ms"
              << std::endl;

    free_canonical_tree_$N$(canonical_tree);

    // ---- Embree Closest Point Query ----
    struct ClosestPointResult {
        float distance;
        unsigned int primID;
        const float *vertices;
        const unsigned *indices;
    };

    auto pointQueryFunc = [](RTCPointQueryFunctionArguments *args) -> bool {
        assert(args->userPtr);
        ClosestPointResult *result = (ClosestPointResult *)args->userPtr;

        // Get triangle vertices.
        unsigned int primID = args->primID;
        const unsigned *indices = result->indices;
        const float *vertices = result->vertices;

        unsigned int i0 = indices[primID * 3 + 0];
        unsigned int i1 = indices[primID * 3 + 1];
        unsigned int i2 = indices[primID * 3 + 2];

        Triangle tri;
        tri.p0[0] = vertices[i0 * 3 + 0];
        tri.p0[1] = vertices[i0 * 3 + 1];
        tri.p0[2] = vertices[i0 * 3 + 2];

        tri.p1[0] = vertices[i1 * 3 + 0];
        tri.p1[1] = vertices[i1 * 3 + 1];
        tri.p1[2] = vertices[i1 * 3 + 2];

        tri.p2[0] = vertices[i2 * 3 + 0];
        tri.p2[1] = vertices[i2 * 3 + 1];
        tri.p2[2] = vertices[i2 * 3 + 2];

        Point pt;
        pt.vec[0] = args->query->x;
        pt.vec[1] = args->query->y;
        pt.vec[2] = args->query->z;

        // Use the same distance function as Bonsai.
        float dist = distmin_tri(&pt, &tri);

        if (dist < result->distance) {
            result->distance = dist;
            result->primID = primID;
            args->query->radius = dist;
        }

        return true; // Continue search
    };

    std::vector<float> embree_distances;
    embree_distances.reserve(num_queries);

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        RTCPointQuery query;
        query.x = query_points[i].x;
        query.y = query_points[i].y;
        query.z = query_points[i].z;
        query.radius = std::numeric_limits<float>::infinity();
        query.time = 0.f;

        ClosestPointResult result;
        result.distance = std::numeric_limits<float>::infinity();
        result.primID = RTC_INVALID_GEOMETRY_ID;
        result.vertices = vertices1.data();
        result.indices = indices1.data();

        RTCPointQueryContext context;
        rtcInitPointQueryContext(&context);

        rtcPointQuery(scene, &query, &context, pointQueryFunc, (void *)&result);

        embree_distances.push_back(result.distance);
    }
    t1 = clock::now();
    embree_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[embree] closest point query : " << embree_time << " ms"
              << std::endl;

    // ---- Bonsai Closest Point Query ----
    std::vector<float> bonsai_distances;
    bonsai_distances.reserve(num_queries);

    t0 = clock::now();
    for (int i = 0; i < num_queries; i++) {
        Point point{
            .vec = {query_points[i].x, query_points[i].y, query_points[i].z},
        };
        Triangle result = closest_point(&point, &tree);

        // Compute the actual distance to the returned triangle.
        bonsai_distances.push_back(distmin_tri(&point, &result));
    }
    t1 = clock::now();
    bonsai_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[bonsai] closest point query : " << bonsai_time << " ms"
              << std::endl;

    verify_results(embree_distances, bonsai_distances, num_queries,
                   /*distance_tolerance=*/1e-4f);

    // Cleanup Embree resources.
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <obj_file> <num_queries>"
                  << std::endl;
        return 1;
    }

    std::string obj_file = argv[1];
    int64_t num_queries = std::stoll(argv[2]);
    std::cout << obj_file << ", " << num_queries << std::endl;

    run_test(obj_file, num_queries);

    return 0;
}