#include <pbrt/pbrt.h>

// Core PBRT geometry and primitives
#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/primitive.h>
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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pbrt;

// A little helper: store your triangle input as floats
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

// Load your OBJ file
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

// Build a PBRT primitive (BVH) from your InputTriangles
Primitive *build_pbrt_aggregate(const std::vector<InputTriangle> &tris,
                                const std::string &partition) {
    Allocator alloc;

    // Build vertex list + index list
    std::vector<Point3f> vertices;
    std::vector<int> indices;

    vertices.reserve(tris.size() * 3);
    indices.reserve(tris.size() * 3);

    for (size_t i = 0; i < tris.size(); ++i) {
        const auto &t = tris[i];
        int base = vertices.size();
        vertices.push_back(Point3f(t.v0x, t.v0y, t.v0z));
        vertices.push_back(Point3f(t.v1x, t.v1y, t.v1z));
        vertices.push_back(Point3f(t.v2x, t.v2y, t.v2z));
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
    }

    Transform *identity = alloc.new_object<Transform>();
    // Create triangle mesh - empty lists for normals, uvs etc.
    TriangleMesh *mesh = alloc.new_object<TriangleMesh>(
        *identity, false, indices, vertices, std::vector<Normal3f>(),
        std::vector<Vector3f>(), std::vector<int>(), std::vector<Point2f>(),
        nullptr);

    // Now create primitives for each triangle
    std::vector<Primitive *> prims;
    prims.reserve(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        // Create individual Triangle shape
        Triangle *tri = Triangle::Create(mesh, i, alloc);
        // Use SimplePrimitive with no material
        prims.push_back(alloc.new_object<SimplePrimitive>(tri, nullptr));
    }

    // Build BVH
    if (partition == "sah") {
        return BVHAggregate::Create(std::move(prims), 1, SplitMethod::SAH,
                                    alloc);
    } else if (partition == "ms") {
        return BVHAggregate::Create(std::move(prims), 1, SplitMethod::Middle,
                                    alloc);
    } else {
        std::cerr << "Unknown partition method: " << partition << std::endl;
        std::exit(1);
    }
}

// Trace a single ray through the PBRT aggregate
struct HitInfo {
    size_t triIndex;
    float u, v; // barycentric coords
    float t;
};

std::optional<HitInfo> trace_one(const InputRay &in, Primitive *agg) {
    // Convert your InputRay to pbrt Ray
    // PBRT v4 Ray constructor: Ray(Point3f o, Vector3f d, Float time, Medium
    // medium)
    Ray ray(Point3f(in.ox, in.oy, in.oz), Vector3f(in.dx, in.dy, in.dz), 0.f,
            nullptr);

    // For tMax, we need to pass it to Intersect
    ShapeIntersection si;
    // The Intersect method returns whether a hit occurred and fills si
    if (agg->Intersect(ray, in.tmax, &si)) {
        HitInfo h;
        h.u = si.intr.uv[0];
        h.v = si.intr.uv[1];
        h.t = si.tHit;
        // The faceIndex gives you which triangle in the mesh was hit
        h.triIndex = si.intr.faceIndex;
        return h;
    }
    return std::nullopt;
}

// Test runner
void run(const std::string &objfile, const std::string &partition,
         bool singleThread, const std::vector<int64_t> &raysCounts) {
    std::cout << "Loading OBJ: " << objfile << std::endl;
    auto tris = load_obj(objfile);
    assert(!tris.empty());
    std::cout << "Loaded " << tris.size() << " triangles\n";

    std::cout << "Building BVH (" << partition << ")" << std::endl;
    Primitive *agg = build_pbrt_aggregate(tris, partition);

    bool first = true;
    for (int64_t rc : raysCounts) {
        std::cout << "\n=== rays = " << rc << " ===\n";
        std::string rayfile =
            "apps/rt/rays/" + objfile + "_" + std::to_string(rc) + "_75.rays";
        auto rays = load_rays_binary(rayfile, rc);
        assert(!rays.empty());

        // Warm up
        if (first) {
            std::cout << "Warming up..." << std::endl;
            for (size_t i = 0; i < std::max<size_t>(rays.size(), 512u); ++i) {
                (void)trace_one(rays[i % rays.size()], agg);
            }
            first = false;
        }

        size_t hitCount = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (singleThread) {
            for (auto &r : rays) {
                if (auto h = trace_one(r, agg))
                    hitCount++;
            }
        } else {
#ifdef _OPENMP
            size_t localHits = 0;
#pragma omp parallel for reduction(+ : localHits)
            for (size_t i = 0; i < rays.size(); ++i) {
                if (trace_one(rays[i], agg))
                    localHits++;
            }
            hitCount = localHits;
#else
            std::cerr << "OpenMP not available, using single-threaded\n";
            for (auto &r : rays) {
                if (auto h = trace_one(r, agg))
                    hitCount++;
            }
#endif
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "hits     : " << hitCount << "\n";
        std::cout << "time ms  : " << ms << "\n";
        std::cout << "rays/sec : " << (rc * 1000.0 / ms) << "\n";
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
    std::string obj = argv[i++];
    std::string part = argv[i++];
    std::string sched = argv[i++];
    bool single = (sched == "single-thread");
    int N = std::atoi(argv[i++]);
    std::vector<int64_t> counts;
    counts.reserve(N);
    for (int k = 0; k < N; ++k) {
        counts.push_back(std::atoll(argv[i++]));
    }
    run(obj, part, single, counts);
    return 0;
}
