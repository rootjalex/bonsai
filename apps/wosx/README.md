# apps/wosx

The walk on spheres -- Muller's random walk that solves a Laplace problem
by asking the boundary, at every step, only how far away it is -- written
in bonsai and compared against WoSX on WoSX's own tree.
`wosx.bonsai` is the program, `rng.bonsai` its random numbers,
`schedules/*.bonsai` the ways of running it, and `wosx_hook.cpp` the C++
driver that reads a mesh, has WoSX build the tree, lays out the sample
points, and runs and checks both sides. `PLAN.md` is the working record of
what was built, measured and decided, and why. The trees, their bytes and
the distance query's arithmetic are apps/rtq's, imported: this app adds the
walk.

The reference is WoSX (deps/wosx, "Walk on Spheres Extensions", Sawhney;
NVIDIA's fork of Zombie), a header-only C++ library whose geometric queries
are FCPW's. Its CPU solver, `WalkOnSpheres<T, DIM>`, loops while the
distance to the absorbing boundary is above an epsilon shell, stepping by
that distance in a uniform direction, and takes the Dirichlet value where
the walk projects onto the boundary; the distance is FCPW's
`findClosestPoint` with an infinite radius. The program computes what WoSX
computes, with the distance query apps/rtq already measured against FCPW
(`closest`: FCPW's closest point on a triangle and its point-to-box
distances, transcribed in apps/rtq/metrics/fcpw.bonsai), and the schedule
runs it the way WoSX and FCPW run it: the walk a loop, the children of a
node tested at once and descended into nearest first over FCPW's stack,
the leaf's sixteen triangles tested at once. The tree is built by WoSX's own
boundary handler (FCPW's builder, vectorized), so both sides traverse the
same splits, the same leaves and the same boxes.

Only the plain walk is here: no source term (the Laplace problem), no
gradient estimate, no Russian roulette or splitting, no reflecting boundary
(walk on stars), no caching. Each is an extension to add once the basic
walk measures right.

## WoSX, an optional dependency

WoSX is a submodule at `deps/wosx`, and nothing else in the repository needs
it. It carries FCPW -- its own pin of FCPW, which is the one its numbers are
measured on, not the `deps/fcpw` apps/rtq compares against -- with enoki
and Eigen under it, and pcg32, as submodules of its own; its other
submodules (TBB, nanoflann, nanobind, polyscope) are not needed here, TBB
coming from the conda environment. Header-only, nothing to build:

```bash
git submodule update --init deps/wosx
git -C deps/wosx submodule update --init deps/fcpw deps/pcg32
git -C deps/wosx/deps/fcpw submodule update --init deps/enoki deps/eigen
```

`compare.sh` stops with a message if any of these is missing.

## Running

The whole table, from the repository root, inside the `bonsai` conda
environment, with the compiler built:

```bash
apps/wosx/benchmark.sh                 # every mesh, both schedules, CSV, table, geomeans, plot
apps/wosx/benchmark.sh --table-only    # fold, print and plot the logs already there
python3 apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots   # the plot alone
```

It runs `compare.sh` over WoSX's own demo meshes (the comb electrodes, the
whale, the car, the octopus, the beast, the rover) and the thirteen pbrt
meshes apps/rtq measures (`RTQ_SCENES` names the checkout), under the two
matching schedules, waiting for an idle machine before each mesh and
resuming a stopped run (a mesh's log exists only when complete; `--fresh`
measures again; `PAUSE` holds at a boundary); folds the logs into
`apps/wosx/results/wosx-results.csv`, each row stamped with the compiler's
commit; prints, per schedule, every mesh's row and the geomean of the
speedups with the least and the greatest named; and draws the plot into
`apps/wosx/plots` (`plot.py`: the meshes along the x axis by triangle
count, two bars each -- one thread against one thread, solid, and the
threads against the threads, hatched -- each bonsai's rate over WoSX's,
the line at 1 WoSX; the landscape is left out, since nearly every walk on
it escapes on its first step, and `--exclude` names the meshes left out).
There is no parameter the speedup is swept over, as apps/pbrt sweeps the
samples per pixel: a step is one distance query, and the epsilon shell
and the walk count set how many steps a point takes, not what a step
costs, so they stay at WoSX's demo defaults. `--meshes`, `--schedules`,
`--side`, `--walks` and `--repeats` narrow or widen it; `--help` lists them.

One mesh, one run (`BONSAI_BUILD_DIR` names the compiler's build
directory; unset, the script takes the first of `build`, `build-*` whose
CMake cache found an LLVM, and says which):

```bash
apps/wosx/compare.sh [--schedule fcpw4w16,fcpw4w16-threads] [--side 128] [--walks 64] [--repeats 3] <mesh>
apps/wosx/compare.sh --image apps/wosx/dragon ~/projects/pbrt-v4-scenes/sssdragon/geometry/dragon.ply.gz
apps/wosx/compare.sh deps/wosx/demo_apps/potential_flow/data/whale.obj
```

Two schedules are built and run by default, each its own table under a
`=== schedule` line. `fcpw4w16` is one thread against one thread: WoSX's
`solve` with `runSingleThreaded`, the program's parfor over the points
left unbound by the schedule (a plain loop), both under numactl on the
physical core the kernel ranks best, as apps/rtq measures.
`fcpw4w16-threads` is the threads against the threads: WoSX's `solve`
through its `tbb::parallel_for` over the points, the program's parfor
bound to the CPU threads (which the compiled code hands to the runtime's
`tbb::parallel_for`, the same TBB), both pinned to the performance cores
-- the cores sharing the best core's last-level cache, the frequency
chiplet on this machine, every hardware thread of it -- so both pools have
the same threads and none of the slower cores. `RTQ_CPUS=<cpus>` pins
either to another set.

The mesh is a binary PLY of either byte order, plain or gzipped, as pbrt's
scenes ship them, or an OBJ, read by WoSX's own loader as its demos read
theirs. It is normalized into the unit sphere, as WoSX's demos normalize
theirs, so that the epsilon shell -- `--epsilon`, 1e-3 by default as in
WoSX's demos -- means the same for every mesh. Whether the surface is
closed the driver reads off the mesh (an edge of one triangle is a rim;
`--open` and `--closed` override): closed, the sample points are the
slice's interior by WoSX's own test (its parity count of ray hits along the
axes), as for a watertight WoSX problem; open, the whole slice, as for a
non-watertight one (WoSX's `domainIsWatertight` false), the walks ending on
whichever side of the surface they reach.

What the driver does: it loads and normalizes the mesh; has WoSX's
`FcpwDirichletBoundaryHandler` build FCPW's vectorized BVH over it and
copies that tree into the layout; lays a `side` by `side` grid of points on
the plane through the middle of the mesh, keeps the ones above, outside
the shell, and computes each one's distance to the boundary once, as
WoSX's demos do before solving (both sides start their walks from it); and
solves the Laplace problem with the Dirichlet data `x^2 - y^2` at each of
them on both sides, `walks` walks per point, each side the least of
`--repeats` runs after a warm-up. The data is harmonic, so the solution
inside a closed surface is the same function and both estimates are
checked against it, not only against each other: the error is the walk's
own (the shell's bias, the Monte Carlo variance) and nothing else.

The table it prints: the number of points and walks, million steps per
second on each side (a step is one distance query, the one thing a walk
costs; each side's rate from its own count of steps over the walks that
ended), the speedup (WoSX's time over ours), how many walks ended on each
side (the rest escaped the box) and their mean length (the numbers that
say both did the same work), each side's root-mean-square error against
the known solution, and `z`: the mean signed difference between the two
estimates over all the points divided by its standard error from WoSX's
own per-point variances, of order one when the two sides estimate the same
thing and large when one is biased against the other -- the streams cannot
be the same, since WoSX seeds each point's generator from the clock.
`--image P` also writes the slice three ways, `P-bonsai.ppm`, `P-wosx.ppm`
and `P-exact.ppm`, blue to red over [-1, 1], the cells with no point
black.

## Layout of the files

- `wosx.bonsai`: the walk (`walk`, WoSX's `WalkOnSpheres::walk` as a tail
  recursion over the walk's position and its distance, in WoSX's order: a
  step, the escape test, the length limit, a query; `terminal`, its
  projection query at the end), the problem's record (`Problem`: the
  shell, the length limit, the domain's box), the Dirichlet data, the
  sphere sampler, and the exported `solve_all` over a batch of points with
  their first distances. Imports apps/rtq's elements, FCPW's arithmetic
  and the `closest` query.
- `rng.bonsai`: WoSX's pcg32 -- the generator apps/pbrt/rng.bonsai already
  transcribed, since pbrt's RNG is the same header, with pcg32's seeding
  order and its float.
- `schedules/fcpw4w16.bonsai`: the walk as a loop (`walk.loopify()`) and
  the distance query directive for directive as apps/rtq's schedule of
  the same query against FCPW, over FCPW's four-wide tree with
  sixteen-lane leaves in FCPW's bytes
  (apps/rtq/schedules/layouts/fcpw4w16.bonsai, imported).
  `schedules/fcpw4w16-threads.bonsai`: the same with the points bound to
  the CPU threads (`solve_all.bind(i, CPUThread)`), WoSX's parallel solve.
- `wosx_hook.cpp`: the driver; `compare.sh`, `benchmark.sh`, `plot.py`:
  the scripts.
  The mesh reader and the copy of FCPW's tree into the layout are
  apps/rtq's (`apps/rtq/mesh.h`, `apps/rtq/fcpw_tree.h`), shared.
- `results/`: the measured data, generated and gitignored -- the logs, one
  CSV with one row per measured (mesh, schedule), the rover unzipped.
- `scratch/`: an agent's working files for this app, gitignored.

The generated `wosx.h`, `wosx.o`, `wosx.bir` and `wosx.ll` are left in this
directory by `compare.sh` only while it runs, and are not committed.
