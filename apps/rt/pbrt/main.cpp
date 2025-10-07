#include <pbrt/pbrt.h>

// Core PBRT geometry and primitives
#include <pbrt/cpu/aggregates.h>
#include <pbrt/shapes.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pbrt;

// Store triangle input as floats
struct InputTriangle {
    float v0x, v0y, v0z;
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;
};

struct InputRay {
    float ox, oy, oz;
    float dx, dy, dz;
    float tmin, tmax;
};

// Load OBJ file
std::vector<InputTriangle> load_obj(const std::string &filename) {
    std::vector<InputTriangle> tris;
    std::vector<Point3f> vertices;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return tris;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(Point3f(x, y, z));
        } else if (type == "f") {
            std::vector<int> idx;
            std::string token;
            while (iss >> token) {
                int i = std::stoi(token.substr(0, token.find('/'))) - 1;
                idx.push_back(i);
            }
            for (size_t i = 2; i < idx.size(); ++i) {
                InputTriangle t;
                Point3f &v0 = vertices[idx[0]];
                Point3f &v1 = vertices[idx[i - 1]];
                Point3f &v2 = vertices[idx[i]];
                t.v0x = v0.x;
                t.v0y = v0.y;
                t.v0z = v0.z;
                t.v1x = v1.x;
                t.v1y = v1.y;
                t.v1z = v1.z;
                t.v2x = v2.x;
                t.v2y = v2.y;
                t.v2z = v2.z;
                tris.push_back(t);
            }
        }
    }
    return tris;
}

std::vector<InputRay> load_rays_binary(const std::string &fname, int64_t cnt) {
    std::vector<InputRay> r(cnt);
    std::ifstream f(fname, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open ray file: " << fname << std::endl;
        return {};
    }
    f.read(reinterpret_cast<char *>(r.data()), cnt * sizeof(InputRay));
    return r;
}

// Simple BVH node structure
struct BVHNode {
    Bounds3f bounds;
    BVHNode *left = nullptr;
    BVHNode *right = nullptr;
    int triIndex = -1; // -1 for interior nodes

    bool isLeaf() const { return triIndex >= 0; }
};

// Build a simple BVH manually
BVHNode *buildBVH(const std::vector<InputTriangle> &tris,
                  std::vector<int> &triIndices, int start, int end,
                  const std::string &partition) {
    BVHNode *node = new BVHNode();

    // Compute bounds
    for (int i = start; i < end; ++i) {
        const auto &t = tris[triIndices[i]];
        node->bounds = Union(node->bounds, Point3f(t.v0x, t.v0y, t.v0z));
        node->bounds = Union(node->bounds, Point3f(t.v1x, t.v1y, t.v1z));
        node->bounds = Union(node->bounds, Point3f(t.v2x, t.v2y, t.v2z));
    }

    int nTris = end - start;
    if (nTris == 1) {
        node->triIndex = triIndices[start];
        return node;
    }

    // Choose split axis
    Vector3f diagonal = node->bounds.Diagonal();
    int axis = (diagonal.x > diagonal.y && diagonal.x > diagonal.z) ? 0
               : (diagonal.y > diagonal.z)                          ? 1
                                                                    : 2;

    // Partition triangles
    int mid = start + nTris / 2;
    if (partition == "ms") {
        // Middle split - already computed
    } else {
        // Simple SAH approximation - sort by centroid
        std::sort(triIndices.begin() + start, triIndices.begin() + end,
                  [&](int a, int b) {
                      const auto &ta = tris[a];
                      const auto &tb = tris[b];
                      Point3f ca((ta.v0x + ta.v1x + ta.v2x) / 3.0f,
                                 (ta.v0y + ta.v1y + ta.v2y) / 3.0f,
                                 (ta.v0z + ta.v1z + ta.v2z) / 3.0f);
                      Point3f cb((tb.v0x + tb.v1x + tb.v2x) / 3.0f,
                                 (tb.v0y + tb.v1y + tb.v2y) / 3.0f,
                                 (tb.v0z + tb.v1z + tb.v2z) / 3.0f);
                      return ca[axis] < cb[axis];
                  });
        mid = start + nTris / 2;
    }

    node->left = buildBVH(tris, triIndices, start, mid, partition);
    node->right = buildBVH(tris, triIndices, mid, end, partition);

    return node;
}

// Ray-triangle intersection
bool intersectTriangle(const Ray &ray, const InputTriangle &tri, float tMax,
                       float &t, float &u, float &v) {
    Point3f v0(tri.v0x, tri.v0y, tri.v0z);
    Point3f v1(tri.v1x, tri.v1y, tri.v1z);
    Point3f v2(tri.v2x, tri.v2y, tri.v2z);

    Vector3f e1 = v1 - v0;
    Vector3f e2 = v2 - v0;
    Vector3f s1 = Cross(ray.d, e2);
    Float divisor = Dot(s1, e1);

    if (divisor == 0.0f)
        return false;

    Float invDivisor = 1.0f / divisor;
    Vector3f s = ray.o - v0;
    Float b1 = Dot(s, s1) * invDivisor;
    if (b1 < 0.0f || b1 > 1.0f)
        return false;

    Vector3f s2 = Cross(s, e1);
    Float b2 = Dot(ray.d, s2) * invDivisor;
    if (b2 < 0.0f || b1 + b2 > 1.0f)
        return false;

    Float tHit = Dot(e2, s2) * invDivisor;
    if (tHit < 0.0f || tHit > tMax)
        return false;

    t = tHit;
    u = b1;
    v = b2;
    return true;
}

// Traverse BVH
struct HitInfo {
    int triIndex;
    float u, v;
    float t;
};

bool traverseBVH(const BVHNode *node, const Ray &ray, float tMax,
                 const std::vector<InputTriangle> &tris, HitInfo &hit) {
    if (!node)
        return false;

    // Check bounds
    Float t0, t1;
    if (!node->bounds.IntersectP(ray.o, ray.d, tMax, &t0, &t1))
        return false;

    if (node->isLeaf()) {
        float t, u, v;
        if (intersectTriangle(ray, tris[node->triIndex], tMax, t, u, v)) {
            if (t < hit.t) {
                hit.t = t;
                hit.u = u;
                hit.v = v;
                hit.triIndex = node->triIndex;
                return true;
            }
        }
        return false;
    }

    if (traverseBVH(node->left, ray, tMax, tris, hit)) {
        return true;
    }
    return traverseBVH(node->right, ray, tMax, tris, hit);
}

std::optional<HitInfo> trace_one(const InputRay &in, const BVHNode *bvh,
                                 const std::vector<InputTriangle> &tris) {
    Ray ray(Point3f(in.ox, in.oy, in.oz), Vector3f(in.dx, in.dy, in.dz), 0.f,
            nullptr);

    HitInfo hit;
    hit.t = in.tmax;
    hit.triIndex = -1;

    if (traverseBVH(bvh, ray, in.tmax, tris, hit) && hit.triIndex >= 0) {
        return hit;
    }

    return std::nullopt;
}

// pin the current thread to a core (0-based)
inline void pin_thread_to_core(int core_id) {
#ifdef __APPLE__
    thread_affinity_policy_data_t policy = {core_id};
    thread_port_t thread = mach_thread_self();
    thread_policy_set(thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                      THREAD_AFFINITY_POLICY_COUNT);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// raise thread priority (best effort)
inline void set_high_priority() {
#ifdef __APPLE__
    pthread_t t = pthread_self();
    sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(t, SCHED_FIFO, &param);
#else
    // Linux: nice value -20 is highest
    nice(-20);
#endif
}

// Test runner
void run(std::string object, std::string partition, bool is_single_threaded,
         std::vector<int64_t> ray_counts) {
    using clock = std::chrono::high_resolution_clock;

    // Optional: pin main thread and raise priority
    pin_thread_to_core(0);
    set_high_priority();

    std::cout << "Loading OBJ: " << object << std::endl;
    auto triangles = load_obj(object);
    assert(!triangles.empty());
    std::cout << "Loaded " << triangles.size() << " triangles\n";

    std::cout << "Building BVH (" << partition << ")" << std::endl;
    std::vector<int> triIndices(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i)
        triIndices[i] = i;

    BVHNode *bvh =
        buildBVH(triangles, triIndices, 0, triangles.size(), partition);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" +
                               std::to_string(75) + ".rays";
        auto rays = load_rays_binary(ray_file, ray_count);
        assert(!rays.empty());

        if (is_first_run) {
            for (size_t i = 0; i < std::max<size_t>(rays.size(), 512u); ++i)
                (void)trace_one(rays[i % rays.size()], bvh, triangles);
            is_first_run = false;
        }

        size_t hit_count = 0;
        auto trace_begin = clock::now(), trace_end = clock::now();

        if (is_single_threaded) {
            std::vector<HitInfo> hits;
            hits.reserve(rays.size());
            trace_begin = clock::now();
            for (size_t i = 0; i < rays.size(); ++i) {
                if (auto t = trace_one(rays[i], bvh, triangles)) {
                    hits.push_back(*t);
                }
            }
            trace_end = clock::now();
            hit_count = hits.size();
        } else {
#ifdef _OPENMP
            const size_t max_threads = omp_get_max_threads();
            std::vector<std::vector<HitInfo>> hits_per_thread(max_threads);
            for (auto &v : hits_per_thread) {
                v.reserve(rays.size() / max_threads + 64);
            }
            trace_begin = clock::now();

#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                auto &hits = hits_per_thread[tid];
                pin_thread_to_core(tid); // pin each thread to a core

#pragma omp for schedule(dynamic, 64) nowait
                for (size_t i = 0; i < rays.size(); ++i) {
                    if (auto t = trace_one(rays[i], bvh, triangles)) {
                        hits.push_back(*t);
                    }
                }
            }

            trace_end = clock::now();
            for (const auto &v : hits_per_thread)
                hit_count += v.size();
#else
            std::cerr << "OpenMP not available, using single-threaded\n";
            exit(1);
#endif
        }

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits       : " << hit_count << "\n";
        std::cout << "trace time : " << trace_time << " ms\n";
    }
}

// main
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <obj> <partition> <single-thread|parallel> <N> "
                     "<rays...>\n";
        return 1;
    }
    int i = 1;
    std::string object_file = argv[i++];
    std::string partition = argv[i++];
    std::string schedule = argv[i++];
    assert(schedule == "single-thread" || schedule == "parallel");
    const bool is_single_threaded = (schedule == "single-thread");

    const int64_t size = std::atoi(argv[i++]);
    std::vector<int64_t> ray_counts;
    ray_counts.reserve(size);
    for (; i < 5 + size; ++i)
        ray_counts.push_back(std::atoll(argv[i]));

    run(object_file, partition, is_single_threaded, ray_counts);
    return 0;
}
