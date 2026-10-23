// The driver for apps/wosx: the same boundary, the same problem and the same
// sample points to WoSX and to the compiled program, on the same tree.
//
//   ./apps/wosx/wosx.out [options] <mesh.ply | mesh.ply.gz>
//
// What it does, in order. It reads a mesh (PLY, as pbrt's scenes ship
// them; apps/rtq's reader) and normalizes it into the unit sphere as WoSX's
// demos normalize theirs (wosx::normalize), so that WoSX's epsilon shell,
// 1e-3 in its demos, means the same here. It gives the triangles to WoSX's
// FcpwDirichletBoundaryHandler, which has FCPW build its vectorized BVH
// over them (Scene::build with the surface-area heuristic, vectorized), and
// copies that tree byte for byte into the layout the schedule declared
// (apps/rtq/schedules/layouts/fcpw4w16.bonsai; apps/rtq/fcpw_tree.h does
// the copy and checks the structs), so that both sides traverse one tree.
// It lays a grid of sample points over a slice through the middle of the
// mesh, keeps the ones inside the boundary (WoSX's own insideDomain, a
// parity count of ray hits along the axes) and outside the epsilon shell,
// and solves the Laplace problem with the harmonic Dirichlet data of
// wosx.bonsai at each of them on both sides: WoSX's WalkOnSpheres::solve
// with its WalkSettings at the demos' defaults (no source term, no
// gradient, the weight window off), single-threaded; and the program's
// solve_all. Each is timed as the least of several runs after a warm-up,
// and the two estimates are compared with each other and with the known
// solution. The streams of random numbers cannot be made the same -- WoSX
// seeds each point's generator from the clock -- so the agreement is
// statistical: the difference between the two estimates at a point against
// the standard error WoSX's own statistics give for it.
//
// Two comparisons, told apart by --threads. Without it, one thread against
// one thread: WoSX's solve with runSingleThreaded, and the program's parfor
// over the points left unbound by the schedule, a plain loop. With it, the
// threads against the threads: WoSX's solve through its tbb::parallel_for
// over the points, and the program's parfor bound to the CPU threads by
// the schedule (schedules/fcpw4w16-threads.bonsai), which the compiled code
// hands to bonsai_parallel_for -- runtime/bonsai_parallel.h's
// tbb::parallel_for, the same TBB. Both pools take the CPUs the process is
// left (compare.sh pins the run), so the two sides have the same threads.
// WoSX's and FCPW's headers before the program's: the program's element
// `Triangle` is a struct in the global namespace, and WoSX's scalar-BVH
// headers (reflectance_boundary_bvh/geometry.h, reached when FCPW is built
// without enoki) say `Triangle` unqualified for FCPW's, which the program's
// would make ambiguous were it declared first.
//
// A third comparison, --gpu: WoSX's GPU solver against a schedule that runs
// the program on the GPU. WoSX's GPU solver is its walk on spheres as a
// Slang compute shader run through slang-rhi (include/wosx/gpu; the
// driver asks for the CUDA backend by name, as WoSX's demos do for the
// kernels below), over FCPW's binary BVH -- the one tree FCPW transfers to
// a GPU -- built here by WoSX's own GPU boundary handler from the same
// positions and indices, and the PDE is this directory's
// gpu/wosx-pde-definition.slang, the same Dirichlet data. Two kernels, as
// WoSX's demos choose between them (demo_apps/basic_2d/gpu_demo.cpp,
// runWalkOnSpheres): --gpu-reference persistent (the default, as it is
// WoSX's on CUDA) has a fixed number of resident threads claim one walk at
// a time from a counter until the walks are gone; threads has one thread
// per copy of a sample point run a walk per dispatch, the points copied
// enough times to fill the card. The program's side is the schedule's: the
// driver loads the program's kernels ahead of the clock
// (bonsai_gpu_prepare), stages every buffer where the generated header says
// the schedule reads it (solve_all_sides), and forbids a copy inside the
// timed call. Built with WOSX_GPU, which compare.sh does for a schedule
// named *-gpu*, against deps/slang-rhi-build (build_slang_rhi.sh); without
// it --gpu stops with a sentence.
#include <wosx/point_estimation/walk_on_spheres.h>
#include <wosx/utils/fcpw_geometric_queries.h>
#if defined(WOSX_GPU)
#include <wosx/wosx_gpu.h>
#endif

#include "wosx.h"

// The tree the schedule declared: FCPW's vectorized Mbvh in apps/rtq's
// layouts (the default); FCPW's scalar binary BVH -- WoSX's default,
// FCPW_USE_ENOKI off -- in apps/wosx/schedules/layouts/fcpwb.bonsai when
// built with WOSX_BINARY_BVH (compare.sh, for the schedules named fcpwb*);
// or the same binary tree with each node holding its children's boxes, as
// FCPW's traversal reads it, in layouts/fcpwb2.bonsai when built with
// WOSX_BINARY_BVH2 (the schedules named fcpwb2*, the GPU comparison's).
#if defined(WOSX_BINARY_BVH)
#include "apps/wosx/fcpw_bvh.h"
#elif defined(WOSX_BINARY_BVH2)
#include "apps/wosx/fcpw_bvh2.h"
#else
#include "apps/rtq/fcpw_tree.h"
#endif
#include "apps/rtq/mesh.h"

#include <oneapi/tbb/info.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using wosx::Vector3;

// wosx.bonsai: dirichlet -- the boundary data, harmonic, so the solution
// inside is the same function. The two must agree.
float dirichlet(const Vector3 &p) { return p.x() * p.x() - p.y() * p.y(); }

#if defined(WOSX_GPU)
// WoSX: the PDE handed to its GPU solver -- the host side of
// gpu/wosx-pde-definition.slang's HarmonicPDE, which has no resources to
// bind: the Dirichlet data is a function of the position alone, and there
// is no source, no absorption and no reflecting boundary. WoSX finds the
// Slang struct by this name and reads the PDE's kind off getType.
class HarmonicGPUPDE : public wosx::GPUPDE {
public:
    void allocate(wosx::GPUContext &) override {}
    void setResources(const wosx::ShaderCursor &, bool) const override {}
    std::string getReflectionType() const override { return "HarmonicPDE"; }
    wosx::GPUPDEType getType() const override { return wosx::GPUPDEType::Poisson; }
};
#endif

// The CPUs this process may run on -- what numactl left it -- for the
// record at the top of the output.
std::string allowed_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return "?";
    }
    std::string out;
    int run = -1;
    for (int cpu = 0; cpu <= CPU_SETSIZE; cpu++) {
        const bool in = cpu < CPU_SETSIZE && CPU_ISSET(cpu, &set);
        if (in && run < 0) {
            run = cpu;
        } else if (!in && run >= 0) {
            out += (out.empty() ? "" : ",") + std::to_string(run);
            if (cpu - 1 > run) {
                out += "-" + std::to_string(cpu - 1);
            }
            run = -1;
        }
    }
    return out;
}

// Seconds for `f`, the least of `repeats` runs after one unmeasured warm-up
// (bonsai-benchmark-repeats: never one run), `setup` run before each off
// the clock.
template <typename S, typename F>
double timed(int repeats, S &&setup, F &&f) {
    using clock = std::chrono::steady_clock;
    setup();
    f();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
        setup();
        const auto start = clock::now();
        f();
        const auto end = clock::now();
        best = std::min(best, std::chrono::duration<double>(end - start).count());
    }
    return best;
}

// The mesh as WoSX's demos take theirs: positions and index triples,
// normalized into the unit sphere about the centroid (wosx::normalize). An
// OBJ file is read by WoSX's own loader (loadBoundaryMesh, FCPW's OBJ
// reader), as its demos read their meshes; a PLY file, as pbrt's scenes
// ship theirs, by apps/rtq's reader.
struct Boundary {
    std::vector<Vector3> positions;
    std::vector<wosx::Vector3i> indices;
    Vector3 lo, hi;        // the box of the normalized positions
    size_t open_edges = 0; // edges of one triangle: none for a closed surface
};

// Whether the surface is closed, read off the mesh: an edge of a closed
// surface belongs to two triangles, an edge of one triangle is a rim. What
// WoSX's domainIsWatertight is set from (its demos set it by hand per
// problem), and what decides whether the sample points are the slice's
// interior (a parity count of ray hits, which a rim makes meaningless) or
// the whole slice.
size_t count_open_edges(const std::vector<wosx::Vector3i> &indices) {
    std::vector<uint64_t> edges;
    edges.reserve(indices.size() * 3);
    for (const wosx::Vector3i &t : indices) {
        for (int k = 0; k < 3; k++) {
            const uint32_t a = uint32_t(t[k]), b = uint32_t(t[(k + 1) % 3]);
            edges.push_back((uint64_t(std::min(a, b)) << 32) | std::max(a, b));
        }
    }
    std::sort(edges.begin(), edges.end());
    size_t open = 0;
    for (size_t i = 0; i < edges.size();) {
        size_t j = i;
        while (j < edges.size() && edges[j] == edges[i]) {
            j++;
        }
        if (j - i == 1) {
            open++;
        }
        i = j;
    }
    return open;
}

bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Boundary load_boundary(const std::string &path) {
    Boundary b;
    if (ends_with(path, ".obj")) {
        wosx::loadBoundaryMesh<3>(path, b.positions, b.indices);
    } else {
        const rtq::Mesh mesh = rtq::load_ply(path);
        b.positions.reserve(mesh.vertices.size());
        for (const rtq::Vec3 &v : mesh.vertices) {
            b.positions.emplace_back(v.x, v.y, v.z);
        }
        b.indices.reserve(mesh.triangles());
        for (size_t i = 0; i < mesh.triangles(); i++) {
            b.indices.emplace_back(int(mesh.indices[3 * i + 0]), int(mesh.indices[3 * i + 1]),
                                   int(mesh.indices[3 * i + 2]));
        }
    }
    if (b.positions.empty() || b.indices.empty()) {
        std::cerr << path << " holds no triangles\n";
        std::exit(1);
    }
    b.open_edges = count_open_edges(b.indices);
    wosx::normalize<3>(b.positions);
    b.lo = Vector3::Constant(std::numeric_limits<float>::infinity());
    b.hi = -b.lo;
    for (const Vector3 &p : b.positions) {
        b.lo = b.lo.cwiseMin(p);
        b.hi = b.hi.cwiseMax(p);
    }
    return b;
}

// The sample points: the centers of a side-by-side grid over the box's x
// and y, on the plane through the middle of its z, the ones inside the
// boundary and outside the epsilon shell (a point in the shell gets one
// walk from WoSX and its boundary value, which is not the walk being
// measured). Each keeps its grid cell, for the image.
struct Sample {
    Vector3 p;
    float dist; // to the boundary, WoSX's SamplePoint::distToAbsorbingBoundary
    uint32_t cell;
};

std::vector<Sample> sample_points(const Boundary &b, const wosx::GeometricQueries<3> &queries,
                                  uint32_t side, float epsilon) {
    std::vector<Sample> samples;
    const float z = 0.5f * (b.lo.z() + b.hi.z());
    for (uint32_t j = 0; j < side; j++) {
        for (uint32_t i = 0; i < side; i++) {
            Vector3 p(b.lo.x() + (b.hi.x() - b.lo.x()) * (float(i) + 0.5f) / float(side),
                      b.lo.y() + (b.hi.y() - b.lo.y()) * (float(j) + 0.5f) / float(side), z);
            if (!queries.insideDomain(p)) {
                continue;
            }
            const float dist = queries.computeDistToAbsorbingBoundary(p, false);
            if (dist <= epsilon) {
                continue;
            }
            samples.push_back(Sample{p, dist, j * side + i});
        }
    }
    return samples;
}

// A slice of estimates as a PPM: the grid's cells, a point's value on a
// blue-white-red scale over [-range, range] (the Dirichlet data's own
// range in the unit sphere), the cells with no point black.
void write_slice(const std::string &path, uint32_t side, const std::vector<Sample> &samples,
                 const std::vector<float> &values, float range) {
    std::vector<uint8_t> rgb(size_t(side) * side * 3, 0);
    for (size_t k = 0; k < samples.size(); k++) {
        const float t = std::clamp(values[k] / range, -1.0f, 1.0f);
        const float r = t >= 0 ? 1.0f : 1.0f + t;
        const float g = 1.0f - std::fabs(t);
        const float bl = t <= 0 ? 1.0f : 1.0f - t;
        const uint32_t cell = samples[k].cell;
        const uint32_t x = cell % side, y = side - 1 - cell / side; // y up
        uint8_t *px = &rgb[(size_t(y) * side + x) * 3];
        px[0] = uint8_t(255 * r);
        px[1] = uint8_t(255 * g);
        px[2] = uint8_t(255 * bl);
    }
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << side << ' ' << side << "\n255\n";
    out.write(reinterpret_cast<const char *>(rgb.data()), std::streamsize(rgb.size()));
    if (!out) {
        std::cerr << "cannot write " << path << '\n';
        std::exit(1);
    }
}

void usage() {
    std::cerr << "usage: wosx.out [options] <mesh.ply | mesh.ply.gz>\n"
                 "  --side N      the sample points are a slice of N x N cells "
                 "(default 128)\n"
                 "  --walks N     walks per point (default 64)\n"
                 "  --epsilon E   the epsilon shell, in the unit sphere's units "
                 "(default 1e-3, WoSX's demos')\n"
                 "  --max-steps N the walk length limit (default 1024, WoSX's "
                 "basic demo's)\n"
                 "  --repeats N   timed runs per side, the least kept (default 3)\n"
                 "  --seed S      the program's seed (default 1; WoSX seeds from "
                 "the clock)\n"
                 "  --image P     write the slices as P-bonsai.ppm, P-wosx.ppm "
                 "and P-exact.ppm\n"
                 "  --open / --closed  say whether the surface is closed, instead "
                 "of reading it off the mesh (an edge of one triangle is a rim); "
                 "open, every point of the slice is a sample point (WoSX's "
                 "domainIsWatertight false)\n"
                 "  --threads     both sides solve the points over the threads "
                 "(WoSX's tbb::parallel_for; the program's parfor bound to the "
                 "CPU threads, which its schedule has to say)\n"
                 "  --shuffle     solve the sample points in a random order "
                 "rather than along the slice, on both sides: a thread's run of "
                 "points is then spread over the mesh instead of a neighbourhood "
                 "of it, which tells cache sharing between the threads from load "
                 "balance when a schedule's gain is in question (the image is "
                 "unaffected)\n"
                 "  --fcpw-stats  have FCPW print its tree's statistics\n"
                 "  --gpu         WoSX's GPU solver against the program on the GPU "
                 "(a driver built with WOSX_GPU and a schedule that binds the GPU; "
                 "compare.sh does both for a schedule named *-gpu*)\n"
                 "  --gpu-reference persistent|threads  which of WoSX's two kernels "
                 "the reference runs: persistent (default, WoSX's default on CUDA) "
                 "has resident threads claim one walk at a time; threads has a thread "
                 "per copy of a point run one walk per dispatch\n"
                 "  --gpu-threads N  WoSX's resident threads for either kernel "
                 "(default 131072, WoSX's)\n"
                 "  --gpu-logs    have WoSX's GPU code print what it loads and binds\n"
                 "  --wosx-dir D  the WoSX checkout whose shaders the reference loads "
                 "(default deps/wosx); --pde-dir D the directory of "
                 "wosx-pde-definition.slang (default apps/wosx/gpu)\n"
                 "The mesh is a PLY (plain or gzipped) or an OBJ. Without --threads "
                 "both sides solve on the calling thread; compare.sh pins the run "
                 "either way.\n";
    std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
    uint32_t side = 128, walks = 64, max_steps = 1024;
    float epsilon = 1e-3f;
    int repeats = 3;
    uint64_t seed = 1;
    bool shuffle = false;
    bool fcpw_stats = false, threads = false;
    bool gpu = false, gpu_logs = false;
    std::string gpu_reference = "persistent";
    uint32_t gpu_threads = 131072;
    std::string wosx_dir = "deps/wosx", pde_dir = "apps/wosx/gpu";
    int closed = -1; // -1: as the mesh says; 0: --open; 1: --closed
    std::string image, path;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--side" && i + 1 < argc) {
            side = uint32_t(std::atoi(argv[++i]));
        } else if (arg == "--walks" && i + 1 < argc) {
            walks = uint32_t(std::atoi(argv[++i]));
        } else if (arg == "--epsilon" && i + 1 < argc) {
            epsilon = float(std::atof(argv[++i]));
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps = uint32_t(std::atoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--image" && i + 1 < argc) {
            image = argv[++i];
        } else if (arg == "--fcpw-stats") {
            fcpw_stats = true;
        } else if (arg == "--open") {
            closed = 0;
        } else if (arg == "--closed") {
            closed = 1;
        } else if (arg == "--threads") {
            threads = true;
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--gpu") {
            gpu = true;
        } else if (arg == "--gpu-reference" && i + 1 < argc) {
            gpu_reference = argv[++i];
        } else if (arg == "--gpu-threads" && i + 1 < argc) {
            gpu_threads = uint32_t(std::atoi(argv[++i]));
        } else if (arg == "--gpu-logs") {
            gpu_logs = true;
        } else if (arg == "--wosx-dir" && i + 1 < argc) {
            wosx_dir = argv[++i];
        } else if (arg == "--pde-dir" && i + 1 < argc) {
            pde_dir = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
        } else {
            path = arg;
        }
    }
    if (path.empty() || side == 0 || walks == 0 || repeats < 1 || !(epsilon > 0) ||
        gpu_threads == 0 || (gpu_reference != "persistent" && gpu_reference != "threads")) {
        usage();
    }
    if (gpu) {
#if !defined(WOSX_GPU)
        std::cerr << "--gpu needs a driver built with WOSX_GPU against slang-rhi "
                     "(apps/wosx/build_slang_rhi.sh; compare.sh builds one for a "
                     "schedule named *-gpu*)\n";
        return 1;
#endif
#if !defined(BONSAI_HAS_GPU)
        std::cerr << "--gpu against a program whose schedule binds nothing to the GPU: "
                     "the program would run on the host. Name a *-gpu* schedule.\n";
        return 1;
#endif
        if (threads) {
            std::cerr << "--threads says nothing under --gpu and is ignored\n";
            threads = false;
        }
        std::cout << "the GPU on both sides: WoSX's " << gpu_reference << " kernel with "
                  << gpu_threads << " resident threads against the schedule's kernels; host cpus allowed: "
                  << allowed_cpus() << "\n";
    } else if (threads) {
        std::cout << "the threads on both sides: " << tbb::info::default_concurrency()
                  << " (TBB's default arena); cpus allowed: " << allowed_cpus() << "\n";
    } else {
        std::cout << "one thread on both sides; cpus allowed: " << allowed_cpus() << "\n";
    }

    const Boundary boundary = load_boundary(path);
    const bool open = closed == 0 || (closed < 0 && boundary.open_edges > 0);
    std::cout << "mesh: " << path << "\n  " << boundary.indices.size() << " triangles, "
              << boundary.positions.size()
              << " vertices, normalized into the unit sphere; box (" << boundary.lo.transpose()
              << ") to (" << boundary.hi.transpose() << "); "
              << (boundary.open_edges == 0 ? std::string("closed")
                                           : std::to_string(boundary.open_edges) + " rim edges")
              << (closed < 0 ? "" : open ? ", taken as open" : ", taken as closed") << "\n";

    // WoSX's side: the boundary handler builds FCPW's tree (vectorized: the
    // Mbvh the layout is the bytes of), and the queries are populated from
    // it as WoSX's demos populate theirs. The domain is the box, as the
    // demos set it: a walk that leaves the box has escaped.
    // Vectorized for apps/rtq's four-wide layouts; scalar -- WoSX's default,
    // FCPW's CMake leaving FCPW_USE_ENOKI off, and the one tree FCPW puts on
    // a GPU -- for the two binary layouts.
#if defined(WOSX_BINARY_BVH) || defined(WOSX_BINARY_BVH2)
    constexpr bool kVectorized = false;
#else
    constexpr bool kVectorized = true;
#endif
    wosx::FcpwDirichletBoundaryHandler<3> handler;
    handler.buildAccelerationStructure(boundary.positions, boundary.indices, /*buildBvh=*/true,
                                       /*enableBvhVectorization=*/kVectorized,
                                       /*printStats=*/fcpw_stats);
    // Not closed (--open): WoSX's insideDomain says yes to every point, as
    // it does for its non-watertight demos, and the walks from the whole
    // slice end on whichever side of the boundary they reach.
    wosx::GeometricQueries<3> queries(/*domainIsWatertight=*/!open, boundary.lo, boundary.hi);
    wosx::populateGeometricQueriesForDirichletBoundary<3>(handler, queries);

    // Our side: the same tree in the layout. (Not const: a schedule on the
    // GPU stages the tree's buffers to the device, which marks them.)
#if defined(WOSX_BINARY_BVH)
    wosx_bvh::Tree tree = wosx_bvh::copy_fcpw_bvh(handler.scene);
    wosx_bvh::describe(tree);
    bonsai_buffer *tree_buffers[2] = {&tree.prims_buffer, &tree.nodes_buffer};
#elif defined(WOSX_BINARY_BVH2)
    wosx_bvh2::Tree tree = wosx_bvh2::copy_fcpw_bvh2(handler.scene);
    wosx_bvh2::describe(tree);
    bonsai_buffer *tree_buffers[2] = {&tree.prims_buffer, &tree.nodes_buffer};
#else
    rtq::FcpwTree tree = rtq::copy_fcpw_tree(handler.scene);
    rtq::describe(tree);
    bonsai_buffer *tree_buffers[2] = {&tree.packets_buffer, &tree.nodes_buffer};
#endif

    std::vector<Sample> samples = sample_points(boundary, queries, side, epsilon);
    if (shuffle) {
        // The points in a random order (seeded, so a run repeats): each
        // keeps its cell, so the image is the same picture.
        std::mt19937_64 order(seed);
        std::shuffle(samples.begin(), samples.end(), order);
    }
    const uint32_t n = uint32_t(samples.size());
    std::cout << "sample points: " << n << " of " << side << " x " << side
              << " on the slice z = " << 0.5f * (boundary.lo.z() + boundary.hi.z())
              << (open ? " outside the shell (the surface is open: the whole slice); "
                       : " inside the boundary and outside the shell; ")
              << walks
              << " walks each, epsilon " << epsilon << ", at most " << max_steps << " steps\n";
    if (n == 0) {
        std::cerr << "no sample point lies inside the boundary: is the mesh closed? "
                     "(--open takes the whole slice)\n";
        return 1;
    }

    // WoSX: the Laplace problem with the Dirichlet data, and the walk's
    // settings as its basic demo makes them (demo_apps/basic_2d/demo.cpp,
    // runWalkOnSpheres, at the config's defaults) with the source term
    // ignored, since there is none.
    wosx::PDE<float, 3> pde;
    pde.dirichlet = [](const Vector3 &x, bool) { return dirichlet(x); };
    pde.source = [](const Vector3 &) { return 0.0f; };
    const wosx::WalkSettings settings(
        /*epsilonShellForAbsorbingBoundary=*/epsilon,
        /*epsilonShellForReflectingBoundary=*/0.0f,
        /*silhouettePrecision=*/0.0f,
        /*russianRouletteThreshold=*/0.0f,
        /*splittingThreshold=*/std::numeric_limits<float>::max(),
        /*maxWalkLength=*/int(max_steps),
        /*stepsBeforeApplyingTikhonov=*/0,
        /*stepsBeforeUsingMaximalSpheres=*/0,
        /*solveDoubleSided=*/false,
        /*useGradientControlVariates=*/true,
        /*useGradientAntitheticVariates=*/true,
        /*useCosineSamplingForDerivatives=*/false,
        /*ignoreAbsorbingBoundaryContribution=*/false,
        /*ignoreReflectingBoundaryContribution=*/true,
        /*ignoreSourceContribution=*/true,
        /*printLogs=*/false);
    // WoSX's answer, whichever of its solvers ran: each point's estimate,
    // how many of its walks ended and their mean length, and -- from the CPU
    // solver's statistics alone, the GPU's keep none -- the variance of its
    // estimate, for `z` below.
    std::vector<float> reference(n), ref_length(n);
    std::vector<int> ref_count(n);
    std::vector<double> ref_variance(n, 0.0);
    double t_wosx = 0.0;
    if (!gpu) {
        std::vector<wosx::SamplePoint<float, 3>> sample_pts;
        sample_pts.reserve(n);
        for (const Sample &s : samples) {
            sample_pts.emplace_back(s.p, Vector3::Zero(), wosx::SampleType::InDomain,
                                    wosx::EstimationQuantity::Solution, /*pdf=*/1.0f, s.dist,
                                    /*distToReflectingBoundary=*/0.0f);
        }
        std::vector<wosx::SampleStatistics<float, 3>> stats(n);
        const std::vector<int> walks_per_point(n, int(walks));
        const wosx::WalkOnSpheres<float, 3> wos(queries);
        // A solve adds to a point's running estimate, so each timed run
        // starts the statistics and the generators afresh (SamplePoint::reset
        // reseeds from the clock, as WoSX made them).
        t_wosx = timed(
            repeats,
            [&] {
                for (auto &pt : sample_pts) {
                    pt.reset();
                }
                stats.assign(n, wosx::SampleStatistics<float, 3>());
            },
            [&] {
                wos.solve(pde, settings, walks_per_point, sample_pts, stats,
                          /*runSingleThreaded=*/!threads);
            });
        for (uint32_t i = 0; i < n; i++) {
            reference[i] = stats[i].getEstimatedSolution();
            ref_count[i] = stats[i].getSolutionEstimateCount();
            ref_length[i] = stats[i].getMeanWalkLength();
            ref_variance[i] = stats[i].getEstimatedSolutionVariance();
        }
    } else {
#if defined(WOSX_GPU)
        // WoSX's GPU solver, set up as its demo sets it up
        // (demo_apps/basic_2d/gpu_demo.cpp, runWalkOnSpheres): a task handle
        // on the CUDA backend that loads FCPW's and WoSX's Slang modules and
        // this directory's PDE; WoSX's GPU boundary handler, which has FCPW
        // build the binary BVH over the same positions and indices (the same
        // deterministic build the CPU handler above ran, so the same tree
        // the layout holds) and transfers it; the queries over the same box;
        // the walk settings as above. The solver's sample points carry the
        // first distances the CPU computed, as the demo's do
        // (computeBoundaryDistance false).
        std::shared_ptr<wosx::GPUTaskHandle<float, 3>> handle =
            std::make_shared<wosx::GPUTaskHandle<float, 3>>(
                std::filesystem::absolute(wosx_dir).string(),
                std::filesystem::absolute(pde_dir).string(), "cuda");
        std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbing =
            std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
                boundary.positions, boundary.indices, fcpw_stats);
        std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflecting =
            std::make_shared<wosx::GPUEmptyReflectingBoundaryHandler>();
        std::shared_ptr<wosx::GPUGeometricQueries<3>> gpu_queries =
            std::make_shared<wosx::GPUGeometricQueries<3>>(absorbing, reflecting, boundary.lo,
                                                           boundary.hi,
                                                           /*domainIsWatertight=*/!open);
        std::shared_ptr<wosx::GPUPDE> gpu_pde = std::make_shared<HarmonicGPUPDE>();
        handle->setGeometricQueries(gpu_queries);
        handle->setPDE(gpu_pde);
        handle->init();
        std::shared_ptr<wosx::GPUWalkSettings> gpu_settings =
            std::make_shared<wosx::GPUWalkSettings>();
        gpu_settings->epsilonShellForAbsorbingBoundary = epsilon;
        gpu_settings->epsilonShellForReflectingBoundary = 0.0f;
        gpu_settings->silhouettePrecision = 0.0f;
        gpu_settings->russianRouletteThreshold = 0.0f;
        gpu_settings->maxWalkLength = max_steps;
        gpu_settings->stepsBeforeUsingMaximalSpheres = 0;
        gpu_settings->solveDoubleSided = 0;
        gpu_settings->useGradientControlVariates = 1;
        gpu_settings->useGradientAntitheticVariates = 1;
        gpu_settings->ignoreAbsorbingBoundaryContribution = 0;
        gpu_settings->ignoreReflectingBoundaryContribution = 1;
        gpu_settings->ignoreSourceContribution = 1;
        const bool persistent = gpu_reference == "persistent";
        wosx::GPUWalkOnSpheresSolver<float, 3> solver(handle, gpu_settings, gpu_threads,
                                                      persistent, gpu_logs);
        std::vector<wosx::GPUSamplePoint<3>> gpu_pts(n);
        for (uint32_t i = 0; i < n; i++) {
            gpu_pts[i].pt = wosx::float3{samples[i].p.x(), samples[i].p.y(), samples[i].p.z()};
            gpu_pts[i].normal = wosx::float3{0.0f, 0.0f, 0.0f};
            gpu_pts[i].type = wosx::SampleType::InDomain;
            gpu_pts[i].estimationQuantity = wosx::EstimationQuantity::Solution;
            gpu_pts[i].pdf = 1.0f;
            gpu_pts[i].distToAbsorbingBoundary = samples[i].dist;
            gpu_pts[i].distToReflectingBoundary = 0.0f;
            gpu_pts[i].estimateBoundaryNormalAligned = 0;
        }
        // The threads kernel copies every point enough times to give the
        // card its resident threads and runs one walk per copy per dispatch
        // (computeWorkloadParameters), which can only round the walks per
        // point up; both sides then run the rounded count, so that the work
        // stays the same.
        uint32_t copies = 1, dispatches = 1;
        if (!persistent) {
            const std::pair<uint32_t, uint32_t> load = solver.computeWorkloadParameters(n, walks);
            copies = load.first;
            dispatches = load.second;
            if (copies * dispatches != walks) {
                std::cout << "WoSX's threads kernel runs " << copies << " copies of each point "
                          << dispatches << " walks each: " << copies * dispatches
                          << " walks a point on both sides, not " << walks << "\n";
                walks = copies * dispatches;
            }
        }
        // Each timed run starts from fresh statistics and generators
        // (populateSamplePoints reseeds and resets, off the clock); the clock
        // is around the solve alone -- the dispatch and the wait for it --
        // as a user of WoSX's solver pays it.
        t_wosx = timed(
            repeats, [&] { solver.populateSamplePoints(gpu_pts, copies, false); },
            [&] {
                if (persistent) {
                    solver.solve(walks);
                } else {
                    solver.solve(1, dispatches);
                }
            });
        std::vector<wosx::GPUSampleStatistics<float, 3>> gpu_stats;
        solver.getSampleStatistics(gpu_stats);
        if (gpu_stats.size() != n) {
            std::cerr << "WoSX's GPU solver returned " << gpu_stats.size()
                      << " statistics for " << n << " points\n";
            return 1;
        }
        for (uint32_t i = 0; i < n; i++) {
            reference[i] = gpu_stats[i].getEstimatedSolution();
            ref_count[i] = gpu_stats[i].getSolutionEstimateCount();
            ref_length[i] = gpu_stats[i].getMeanWalkLength();
        }
#endif
    }

    // The program: the same points with the same first distances (WoSX's
    // SamplePoint::distToAbsorbingBoundary, computed above off the clock,
    // as its demo computes it), the problem as one record. Its arrays as
    // buffers, so that a schedule on the GPU finds each where the generated
    // header says the schedule reads it (solve_all_sides: the points, the
    // distances and the tree on the device; what the host's loops write on
    // the host), staged before the clock starts, with a copy inside the
    // timed call forbidden rather than timed (runtime/bonsai_buffer.h); a
    // CPU schedule's buffers are on the host already and nothing moves.
    std::vector<Point> points(n);
    std::vector<float> dists(n);
    for (uint32_t i = 0; i < n; i++) {
        points[i].p = float3{samples[i].p.x(), samples[i].p.y(), samples[i].p.z()};
        dists[i] = samples[i].dist;
    }
    Problem problem;
    problem.epsilon = epsilon;
    problem.max_steps = max_steps;
    problem.lo = float3{boundary.lo.x(), boundary.lo.y(), boundary.lo.z()};
    problem.hi = float3{boundary.hi.x(), boundary.hi.y(), boundary.hi.z()};
    std::vector<float> solution(n), mean_steps(n);
    std::vector<uint32_t> walks_ended(n);
    bonsai_buffer b_points = bonsai_buffer_wrap(points.data(), uint64_t(n) * sizeof(Point));
    bonsai_buffer b_dists = bonsai_buffer_wrap(dists.data(), uint64_t(n) * sizeof(float));
    bonsai_buffer b_solution = bonsai_buffer_wrap(solution.data(), uint64_t(n) * sizeof(float));
    bonsai_buffer b_mean_steps = bonsai_buffer_wrap(mean_steps.data(), uint64_t(n) * sizeof(float));
    bonsai_buffer b_walks_ended =
        bonsai_buffer_wrap(walks_ended.data(), uint64_t(n) * sizeof(uint32_t));
    bonsai_buffer *buffers[] = {&b_points,     &b_dists,         &b_solution,     &b_mean_steps,
                                &b_walks_ended, tree_buffers[0], tree_buffers[1]};
    static_assert(sizeof(solve_all_sides) == sizeof(buffers) / sizeof(buffers[0]),
                  "solve_all takes these seven buffers, in this order");
#if defined(BONSAI_HAS_GPU)
    // The program's kernels loaded ahead of the clock, as WoSX's shaders are.
    bonsai_gpu_prepare();
#endif
    const double t_bonsai = timed(
        repeats,
        [&] {
            bonsai_buffer_stage_all(buffers, solve_all_sides,
                                    sizeof(buffers) / sizeof(buffers[0]));
            bonsai_buffer_implicit_copies(0);
        },
        [&] {
            solve_all(n, &b_points, &b_dists, problem, walks, seed, &b_solution, &b_mean_steps,
                      &b_walks_ended, tree.layout);
        });
    bonsai_buffer_implicit_copies(1);
    for (bonsai_buffer *out : {&b_solution, &b_mean_steps, &b_walks_ended}) {
        bonsai_buffer_stage(out, BONSAI_HOST);
    }

    // The work: steps (one distance query each, the one thing a walk costs)
    // per second on each side, from each side's own count of steps over the
    // walks that ended; the agreement: how many walks ended and how long
    // they were on each side, each side's root-mean-square error against the
    // known solution, and `z`, a test of a bias between the two sides, of
    // order one when they estimate the same thing and large when one is
    // biased against the other. Against WoSX's CPU solver `z` is the mean
    // signed difference between the two estimates over all the points
    // divided by its standard error (the two estimates independent, each
    // with WoSX's variance over its walks, summed over the points -- summed
    // rather than taken per point, since a point's own variance from a few
    // dozen walks is too rough to divide by). WoSX's GPU statistics keep no
    // variance, so against them `z` is the paired statistic: the mean of
    // the per-point differences over the standard error estimated from the
    // differences themselves, which under no bias have mean zero at every
    // point. The table's heading says which.
    double steps_wosx = 0, steps_bonsai = 0, err_wosx = 0, err_bonsai = 0;
    double difference = 0, difference_variance = 0, difference_squares = 0;
    size_t ended_wosx = 0, ended_bonsai = 0, paired = 0;
    std::vector<float> exact(n);
    for (uint32_t i = 0; i < n; i++) {
        exact[i] = dirichlet(samples[i].p);
        ended_wosx += size_t(ref_count[i]);
        ended_bonsai += walks_ended[i];
        steps_wosx += double(ref_length[i]) * ref_count[i];
        steps_bonsai += double(mean_steps[i]) * walks_ended[i];
        err_wosx += std::pow(double(reference[i]) - exact[i], 2);
        err_bonsai += std::pow(double(solution[i]) - exact[i], 2);
        if (ref_count[i] > 0 && walks_ended[i] > 0) {
            const double d = double(solution[i]) - reference[i];
            difference += d;
            difference_squares += d * d;
            difference_variance += ref_variance[i] / ref_count[i] + ref_variance[i] / walks_ended[i];
            paired++;
        }
    }
    double z = 0.0;
    if (!gpu) {
        z = difference_variance > 0 ? difference / std::sqrt(difference_variance) : 0.0;
    } else if (paired > 1) {
        const double mean = difference / double(paired);
        const double variance = (difference_squares - double(paired) * mean * mean) / double(paired - 1);
        z = variance > 0 ? mean / std::sqrt(variance / double(paired)) : 0.0;
    }
    const double total_walks = double(n) * walks;
    std::printf("\n%8s %10s %12s %12s %10s  %s\n", "points", "walks", "WoSX Msteps/s",
                "bonsai Msteps/s", "speedup",
                gpu ? "walks ended (WoSX, bonsai); their mean length (WoSX, bonsai); rms error "
                      "against the solution (WoSX, bonsai); z (paired)"
                    : "walks ended (WoSX, bonsai); their mean length (WoSX, bonsai); rms error "
                      "against the solution (WoSX, bonsai); z");
    std::printf("%8u %10.0f %12.2f %12.2f %9.2fx  %zu, %zu; %.2f, %.2f; %.2e, %.2e; %.2f\n", n,
                total_walks, steps_wosx / t_wosx * 1e-6, steps_bonsai / t_bonsai * 1e-6,
                t_wosx / t_bonsai, ended_wosx, ended_bonsai,
                steps_wosx / double(std::max<size_t>(ended_wosx, 1)),
                steps_bonsai / double(std::max<size_t>(ended_bonsai, 1)),
                std::sqrt(err_wosx / n), std::sqrt(err_bonsai / n), z);
    std::printf("times: WoSX %.3f s, bonsai %.3f s (least of %d)\n", t_wosx, t_bonsai, repeats);

    if (!image.empty()) {
        const float range = 1.0f; // |x^2 - y^2| <= 1 in the unit sphere
        write_slice(image + "-bonsai.ppm", side, samples, solution, range);
        write_slice(image + "-wosx.ppm", side, samples, reference, range);
        write_slice(image + "-exact.ppm", side, samples, exact, range);
        std::cout << "slices written as " << image << "-{bonsai,wosx,exact}.ppm\n";
    }
    return 0;
}
