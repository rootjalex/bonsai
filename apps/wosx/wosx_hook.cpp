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
// The parfor over the points, left unbound by the schedule, is a plain
// loop; were a schedule to bind it to the threads, the compiled code would
// hand it to bonsai_parallel_for, defined here to run the body in place
// (BONSAI_PARALLEL_EXTERNAL), so that the comparison stays one thread
// against one thread, as WoSX's runSingleThreaded is.
#define BONSAI_PARALLEL_EXTERNAL
#include "wosx.h"

#include <wosx/point_estimation/walk_on_spheres.h>
#include <wosx/utils/fcpw_geometric_queries.h>

#include "apps/rtq/fcpw_tree.h"
#include "apps/rtq/mesh.h"

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

extern "C" void bonsai_parallel_for(int64_t n, void *context,
                                    void (*body)(void *, int64_t, int64_t)) {
    if (n > 0) {
        body(context, 0, n);
    }
}

namespace {

using rtq::Mesh;
using rtq::Vec3;
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
// normalized into the unit sphere about the centroid (wosx::normalize).
struct Boundary {
    std::vector<Vector3> positions;
    std::vector<wosx::Vector3i> indices;
    Vector3 lo, hi; // the box of the normalized positions
};

Boundary make_boundary(const Mesh &mesh) {
    Boundary b;
    b.positions.reserve(mesh.vertices.size());
    for (const Vec3 &v : mesh.vertices) {
        b.positions.emplace_back(v.x, v.y, v.z);
    }
    b.indices.reserve(mesh.triangles());
    for (size_t i = 0; i < mesh.triangles(); i++) {
        b.indices.emplace_back(int(mesh.indices[3 * i + 0]), int(mesh.indices[3 * i + 1]),
                               int(mesh.indices[3 * i + 2]));
    }
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
                 "  --fcpw-stats  have FCPW print its tree's statistics\n"
                 "Both sides solve on the calling thread; compare.sh pins it.\n";
    std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
    uint32_t side = 128, walks = 64, max_steps = 1024;
    float epsilon = 1e-3f;
    int repeats = 3;
    uint64_t seed = 1;
    bool fcpw_stats = false;
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
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
        } else {
            path = arg;
        }
    }
    if (path.empty() || side == 0 || walks == 0 || repeats < 1 || !(epsilon > 0)) {
        usage();
    }
    std::cout << "one thread on both sides; cpus allowed: " << allowed_cpus() << "\n";

    const Mesh mesh = rtq::load_ply(path);
    const Boundary boundary = make_boundary(mesh);
    std::cout << "mesh: " << path << "\n  " << mesh.triangles() << " triangles, "
              << mesh.vertices.size() << " vertices, normalized into the unit sphere; box ("
              << boundary.lo.transpose() << ") to (" << boundary.hi.transpose() << ")\n";

    // WoSX's side: the boundary handler builds FCPW's tree (vectorized: the
    // Mbvh the layout is the bytes of), and the queries are populated from
    // it as WoSX's demos populate theirs. The domain is the box, as the
    // demos set it: a walk that leaves the box has escaped.
    wosx::FcpwDirichletBoundaryHandler<3> handler;
    handler.buildAccelerationStructure(boundary.positions, boundary.indices, /*buildBvh=*/true,
                                       /*enableBvhVectorization=*/true, /*printStats=*/fcpw_stats);
    wosx::GeometricQueries<3> queries(/*domainIsWatertight=*/true, boundary.lo, boundary.hi);
    wosx::populateGeometricQueriesForDirichletBoundary<3>(handler, queries);

    // Our side: the same tree in the layout.
    const rtq::FcpwTree tree = rtq::copy_fcpw_tree(handler.scene);
    rtq::describe(tree);

    const std::vector<Sample> samples = sample_points(boundary, queries, side, epsilon);
    const uint32_t n = uint32_t(samples.size());
    std::cout << "sample points: " << n << " of " << side << " x " << side
              << " on the slice z = " << 0.5f * (boundary.lo.z() + boundary.hi.z())
              << " inside the boundary and outside the shell; " << walks
              << " walks each, epsilon " << epsilon << ", at most " << max_steps << " steps\n";
    if (n == 0) {
        std::cerr << "no sample point lies inside the boundary: is the mesh closed?\n";
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
                      /*runSingleThreaded=*/true);
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
    const double t_bonsai = timed(repeats, [] {}, [&] {
        solve_all(n, points.data(), dists.data(), problem, walks, seed, solution.data(),
                  mean_steps.data(), tree.layout);
    });

    // The work: walks and steps (one distance query each, the one thing a
    // walk costs) per second on each side, from each side's own count of
    // steps; the agreement: each side's root-mean-square error against the
    // known solution, and the difference between the sides at each point
    // over the standard error WoSX's statistics give for that difference
    // (the two estimates independent, each with WoSX's variance over its
    // walks), whose root mean square is one when the two agree to their
    // noise.
    double steps_wosx = 0, steps_bonsai = 0, err_wosx = 0, err_bonsai = 0, z2 = 0;
    size_t z_count = 0;
    std::vector<float> exact(n), reference(n);
    for (uint32_t i = 0; i < n; i++) {
        exact[i] = dirichlet(samples[i].p);
        reference[i] = stats[i].getEstimatedSolution();
        steps_wosx += double(stats[i].getMeanWalkLength()) * stats[i].getSolutionEstimateCount();
        steps_bonsai += double(mean_steps[i]) * walks;
        err_wosx += std::pow(double(reference[i]) - exact[i], 2);
        err_bonsai += std::pow(double(solution[i]) - exact[i], 2);
        const double variance = stats[i].getEstimatedSolutionVariance();
        const int count = stats[i].getSolutionEstimateCount();
        if (variance > 0 && count > 1) {
            const double se = std::sqrt(variance / count + variance / walks);
            z2 += std::pow((double(solution[i]) - reference[i]) / se, 2);
            z_count++;
        }
    }
    const double total_walks = double(n) * walks;
    std::printf("\n%8s %10s %12s %12s %10s  %s\n", "points", "walks", "WoSX Msteps/s",
                "bonsai Msteps/s", "speedup",
                "walk length (WoSX, bonsai); rms error against the solution; z");
    std::printf("%8u %10.0f %12.2f %12.2f %9.2fx  %.2f, %.2f; %.2e, %.2e; %.2f over %zu points\n",
                n, total_walks, steps_wosx / t_wosx * 1e-6, steps_bonsai / t_bonsai * 1e-6,
                t_wosx / t_bonsai, steps_wosx / total_walks, steps_bonsai / total_walks,
                std::sqrt(err_wosx / n), std::sqrt(err_bonsai / n),
                z_count ? std::sqrt(z2 / double(z_count)) : 0.0, z_count);
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
