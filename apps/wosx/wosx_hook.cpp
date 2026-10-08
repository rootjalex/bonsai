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
#include "wosx.h"

#include <wosx/point_estimation/walk_on_spheres.h>
#include <wosx/utils/fcpw_geometric_queries.h>

#include "apps/rtq/fcpw_tree.h"
#include "apps/rtq/mesh.h"

#include <oneapi/tbb/info.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using wosx::Vector3;

// wosx.bonsai: dirichlet -- the boundary data, harmonic, so the solution
// inside is the same function. The two must agree.
float dirichlet(const Vector3 &p) { return p.x() * p.x() - p.y() * p.y(); }

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
                 "  --fcpw-stats  have FCPW print its tree's statistics\n"
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
    bool fcpw_stats = false, threads = false;
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
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
        } else {
            path = arg;
        }
    }
    if (path.empty() || side == 0 || walks == 0 || repeats < 1 || !(epsilon > 0)) {
        usage();
    }
    if (threads) {
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
    wosx::FcpwDirichletBoundaryHandler<3> handler;
    handler.buildAccelerationStructure(boundary.positions, boundary.indices, /*buildBvh=*/true,
                                       /*enableBvhVectorization=*/true, /*printStats=*/fcpw_stats);
    // Not closed (--open): WoSX's insideDomain says yes to every point, as
    // it does for its non-watertight demos, and the walks from the whole
    // slice end on whichever side of the boundary they reach.
    wosx::GeometricQueries<3> queries(/*domainIsWatertight=*/!open, boundary.lo, boundary.hi);
    wosx::populateGeometricQueriesForDirichletBoundary<3>(handler, queries);

    // Our side: the same tree in the layout.
    const rtq::FcpwTree tree = rtq::copy_fcpw_tree(handler.scene);
    rtq::describe(tree);

    const std::vector<Sample> samples = sample_points(boundary, queries, side, epsilon);
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
    // A solve adds to a point's running estimate, so each timed run starts
    // the statistics and the generators afresh (SamplePoint::reset reseeds
    // from the clock, as WoSX made them).
    const double t_wosx = timed(
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

    // The program: the same points with the same first distances (WoSX's
    // SamplePoint::distToAbsorbingBoundary, computed above off the clock,
    // as its demo computes it), the problem as one record.
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
    const double t_bonsai = timed(repeats, [] {}, [&] {
        solve_all(n, points.data(), dists.data(), problem, walks, seed, solution.data(),
                  mean_steps.data(), walks_ended.data(), tree.layout);
    });

    // The work: steps (one distance query each, the one thing a walk costs)
    // per second on each side, from each side's own count of steps over the
    // walks that ended; the agreement: how many walks ended and how long
    // they were on each side, each side's root-mean-square error against the
    // known solution, and `z`, the mean signed difference between the two
    // estimates over all the points divided by its standard error (the two
    // estimates independent, each with WoSX's variance over its walks,
    // summed over the points): a number of order one when the two sides
    // estimate the same thing, and large when one is biased against the
    // other. Summed over the points rather than taken per point, since a
    // point's own variance from a few dozen walks is too rough to divide by.
    double steps_wosx = 0, steps_bonsai = 0, err_wosx = 0, err_bonsai = 0;
    double difference = 0, difference_variance = 0;
    size_t ended_wosx = 0, ended_bonsai = 0;
    std::vector<float> exact(n), reference(n);
    for (uint32_t i = 0; i < n; i++) {
        exact[i] = dirichlet(samples[i].p);
        reference[i] = stats[i].getEstimatedSolution();
        ended_wosx += size_t(stats[i].getSolutionEstimateCount());
        ended_bonsai += walks_ended[i];
        steps_wosx += double(stats[i].getMeanWalkLength()) * stats[i].getSolutionEstimateCount();
        steps_bonsai += double(mean_steps[i]) * walks_ended[i];
        err_wosx += std::pow(double(reference[i]) - exact[i], 2);
        err_bonsai += std::pow(double(solution[i]) - exact[i], 2);
        const int count = stats[i].getSolutionEstimateCount();
        if (count > 0 && walks_ended[i] > 0) {
            const double variance = stats[i].getEstimatedSolutionVariance();
            difference += double(solution[i]) - reference[i];
            difference_variance += variance / count + variance / walks_ended[i];
        }
    }
    const double z = difference_variance > 0 ? difference / std::sqrt(difference_variance) : 0.0;
    const double total_walks = double(n) * walks;
    std::printf("\n%8s %10s %12s %12s %10s  %s\n", "points", "walks", "WoSX Msteps/s",
                "bonsai Msteps/s", "speedup",
                "walks ended (WoSX, bonsai); their mean length (WoSX, bonsai); rms error "
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
