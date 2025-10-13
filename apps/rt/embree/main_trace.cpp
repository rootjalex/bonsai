#include <embree4/rtcore.h>
#include <embree4/rtcore_builder.h>
#include <omp.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

namespace {

// Simple structures
struct Vec3 {
    float x, y, z;
};

struct Ray {
    float o[3];
    float d[3];
};

struct Triangle {
    Vec3 p0, p1, p2;
};

RTCRayHit to_ray_hit(const Ray &ray) {
    struct RTCRayHit rayhit;
    rayhit.ray.org_x = ray.o[0];
    rayhit.ray.org_y = ray.o[1];
    rayhit.ray.org_z = ray.o[2];
    rayhit.ray.dir_x = ray.d[0];
    rayhit.ray.dir_y = ray.d[1];
    rayhit.ray.dir_z = ray.d[2];
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    return rayhit;
}

std::vector<RTCRayHit> load_rays_binary(const std::string &filename,
                                        int64_t ray_count) {
    std::vector<RTCRayHit> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for reading"
                  << std::endl;
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
        file.read(reinterpret_cast<char *>(&ray.o[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[2]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[2]), sizeof(float));
        rays.push_back(to_ray_hit(ray));
    }

    file.close();
    return rays;
}

Triangle get_triangle(RTCScene scene, unsigned int geom_id,
                      unsigned int prim_id) {
    // Get the geometry from the scene
    RTCGeometry geom = rtcGetGeometry(scene, geom_id);

    // Get vertex buffer
    float *vertices =
        (float *)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_VERTEX, 0);

    // Get index buffer
    unsigned *indices =
        (unsigned *)rtcGetGeometryBufferData(geom, RTC_BUFFER_TYPE_INDEX, 0);

    // Get the three vertex indices for this primitive
    unsigned int i0 = indices[prim_id * 3 + 0];
    unsigned int i1 = indices[prim_id * 3 + 1];
    unsigned int i2 = indices[prim_id * 3 + 2];

    // Reconstruct the triangle
    Triangle tri;
    tri.p0.x = vertices[i0 * 3 + 0];
    tri.p0.y = vertices[i0 * 3 + 1];
    tri.p0.z = vertices[i0 * 3 + 2];

    tri.p1.x = vertices[i1 * 3 + 0];
    tri.p1.y = vertices[i1 * 3 + 1];
    tri.p1.z = vertices[i1 * 3 + 2];

    tri.p2.x = vertices[i2 * 3 + 0];
    tri.p2.y = vertices[i2 * 3 + 1];
    tri.p2.z = vertices[i2 * 3 + 2];

    return tri;
}

// Error callback
void error_function(void *ptr, enum RTCError error, const char *str) {
    std::cerr << "Embree error " << error << ": " << str << std::endl;
}

// Create Embree device with specific BVH layout
RTCDevice create_device(const std::string &layout, int32_t leaf_size = 8) {
    std::string configuration;

    if (layout == "auto" || layout == "default") {
    } else if (layout == "bvh8i") {
        configuration = "tri_accel=bvh8.triangle4i";
    } else if (layout == "bvh8v") {
        configuration = "tri_accel=bvh8.triangle4v";
    } else {
        std::cerr << "Unknown BVH layout: " << layout << std::endl;
        exit(1);
    }
    configuration += " quality=medium,max_triangles_per_leaf=32";
    // Gathers information about the BVH structure
    // configuration += ",verbose=2,benchmark=2";

    RTCDevice device = rtcNewDevice(configuration.c_str());

    if (!device) {
        std::cerr << "Failed to create Embree device\n";
        exit(1);
    }

    rtcSetDeviceErrorFunction(device, error_function, nullptr);
    return device;
}

// Create Embree scene from triangles
RTCScene create_scene(RTCDevice device,
                      const std::vector<Triangle> &triangles) {
    RTCScene scene = rtcNewScene(device);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM); // binned-SAH
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Allocate vertex buffer
    float *vertices = (float *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
        triangles.size() * 3);

    // Allocate index buffer
    unsigned *indices = (unsigned *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
        triangles.size());

    // Fill buffers
    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle &tri = triangles[i];

        // Vertices
        vertices[i * 9 + 0] = tri.p0.x;
        vertices[i * 9 + 1] = tri.p0.y;
        vertices[i * 9 + 2] = tri.p0.z;
        vertices[i * 9 + 3] = tri.p1.x;
        vertices[i * 9 + 4] = tri.p1.y;
        vertices[i * 9 + 5] = tri.p1.z;
        vertices[i * 9 + 6] = tri.p2.x;
        vertices[i * 9 + 7] = tri.p2.y;
        vertices[i * 9 + 8] = tri.p2.z;

        // Indices
        indices[i * 3 + 0] = i * 3 + 0;
        indices[i * 3 + 1] = i * 3 + 1;
        indices[i * 3 + 2] = i * 3 + 2;
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    return scene;
}

// Trace a single ray
std::optional<Triangle> trace(RTCScene scene, RTCRayHit &ray) {
    struct RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    rtcIntersect1(scene, &ray, &args);

    if (ray.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return {};
    }
    return get_triangle(scene, ray.hit.geomID, ray.hit.primID);
}

// Load dummy triangles (replace with your actual loading code)
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

void run(std::string object, std::string layout, bool is_single_threaded,
         std::string ray_type, std::vector<int64_t> ray_counts) {
    using clock = std::chrono::high_resolution_clock;

    // Load triangles (replace with your actual loader)
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    // Create Embree device with specified BVH layout
    RTCDevice device = create_device(layout);

    // Build scene
    auto build_begin = clock::now();
    RTCScene scene = create_scene(device, triangles);
    auto build_end = clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          build_end - build_begin)
                          .count();

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" + ray_type +
                               ".rays";
        // Load rays (replace with your actual loader)
        std::vector<RTCRayHit> rays = load_rays_binary(ray_file, ray_count);
        assert(!rays.empty());

        // Warmup
        if (is_first_run) {
            for (size_t i = 0; i < std::max<size_t>(rays.size(), 512u); ++i) {
                RTCRayHit rh = rays[i % rays.size()];
                (void)trace(scene, rh);
            }
            is_first_run = false;
        }

        size_t hit_count = 0;
        auto trace_begin = clock::now(), trace_end = clock::now();

        if (is_single_threaded) {
            std::vector<Triangle> hits;
            hits.reserve(rays.size());
            trace_begin = clock::now();
            for (size_t i = 0; i < rays.size(); ++i) {
                if (const std::optional<Triangle> t = trace(scene, rays[i])) {
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
                for (size_t i = 0; i < rays.size(); ++i) {
                    if (std::optional<Triangle> t = trace(scene, rays[i])) {
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
        std::cout << "trace time : " << trace_time << " ms\n\n";
    }

    // Cleanup
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 6);

    int i = 1;
    std::string object_file = argv[i++];
    std::string bvh_layout = argv[i++];
    std::string schedule = argv[i++];
    std::string ray_type = argv[i++];
    assert(schedule == "single-thread" || schedule == "parallel");
    const bool is_single_threaded = schedule == "single-thread";

    std::vector<int64_t> ray_counts;
    const int64_t size = std::atoi(argv[i++]);
    ray_counts.reserve(size);
    for (; i < 5 + size; ++i)
        ray_counts.push_back(std::atoi(argv[i]));
    run(object_file, bvh_layout, is_single_threaded, ray_type, ray_counts);
    return 0;
}