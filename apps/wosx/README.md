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

The GPU comparison runs WoSX's GPU solver against the program on the GPU.
WoSX's GPU solver is the same walk as a Slang compute shader run through
slang-rhi (deps/wosx/include/wosx/gpu), over FCPW's binary BVH -- the one
tree FCPW puts on a GPU -- and by default with persistent threads: 131072
resident threads, in groups of 256, each claiming one walk at a time from
a global counter until the walks are gone, consecutive claims consecutive
points, each walk merged into its point's statistics under a per-point
spinlock. The schedule matching it (`schedules/fcpwb-gpu.bonsai`, over
the same tree in FCPW's bytes) derives that kernel from the unchanged
program with four directives:
`reorder` puts the walks outside the points and distributes what a point
does around its walks into loops of their own, `collapse` makes one loop
of every (walk, point) pair with the point fastest, `persistent` hands it
to 131072 workers claiming from a counter, and a `split` and two `bind`s
put the workers on the GPU's blocks and threads. Its variants depart from
it one thing at a time (a plain launch, the walks fastest, a point a
claim, a layout that stores the children's boxes in the parent, FCPW's
four-wide tree instead of its binary one), and the GPU table measures
each. Two meshes, the crown and the beast, have fewer walks (29440 and
86016) than the kernel has threads, so on them a thread claims one walk
and the card runs nearly empty: both sides' time is the longest walks'
latency, the same on both, and their bars sit at 1 on every schedule
(PLAN.md, "The crown and the beast"; with 1024 walks a point they gain
like the others).

Two more schedules defer the walk step by step onto a queue, so that a
round of the drain runs the next step of every walk still going and a
warp's lanes hold live steps rather than walks of which the longest keeps
the rest waiting: `schedules/fcpwb-gpu-defer.bonsai`, one queue across
the card, its rounds launches; and `schedules/fcpwb-gpu-defer-block.bonsai`,
a queue per point in the block's shared memory (`queue(i, DeviceShared)`,
the word a queue that lives there has to carry), its rounds a loop of the
kernel's. Over the table the card-wide one is at 0.48x of WoSX by geomean
and the per-block one at 1.03x, the matching schedule's equal on the
largest meshes and behind it on the rest (PLAN.md, "Both deferrals built,
both measured on the head"); both compute the same estimate. The chunk
forms of the per-block one (`fcpwb-gpu-defer-chunk2` to `-chunk16`, and
`wost-fcpwb-gpu-defer-chunk2`, `-chunk4` on the stars; 2026-10-30) put two
to sixteen points' walks on one block's queue, the compacting block
wavefront: it loses to the persistent claim on both walks, and more the
more walks it holds, since a block's rounds end with its longest walk and
the tail holds the multiprocessor's slot (PLAN.md, "The compacting block
wavefront"); what would mend it, a block refilling its queue with new
points as its lanes free, is new scheduling, on PLAN.md's Next list.

The walk on stars is the second solver here (`wost.bonsai`, over
apps/rtq's silhouette query and the cone and edge arithmetic of its FCPW
scalar metric, with what the two walks share in `walk_common.bonsai`),
compared against WoSX's `WalkOnStars` the same way
(Sawhney, Miller, Gkioulekas and Crane, "Walk on Stars", 2023; WoSX's
`include/wosx/point_estimation/walk_on_stars.h` on the CPU and
`gpu/point-estimation/walk-on-stars.slang` on the GPU, the same walk). A
problem with a reflecting (Neumann) part of the boundary beside the
absorbing (Dirichlet) one: a step is not a sphere to the nearest boundary
but a star, the largest ball around the position inside which the
reflecting boundary is seen as one star-shaped sheet, found by the
closest silhouette point of the reflecting boundary on FCPW's normal cone
hierarchy (its tree whose nodes carry a cone of normals beside the box)
and bounded by the distance to the absorbing boundary; and the step is a
ray cast from the position in a uniform direction, which where it hits
the reflecting boundary leaves the walk on the surface, to continue into
the domain's hemisphere, and where it does not goes to the star's surface
as the walk on spheres would. Three queries a step, over three sets --
the absorbing triangles on FCPW's scalar BVH, the reflecting triangles
and their silhouette edges on the one normal cone hierarchy, read as two
trees (`schedules/layouts/fcpw_stars.bonsai` on the CPU, FCPW's own arrays
wrapped; `fcpw_stars_gpu.bonsai` on the GPU) -- where the walk on spheres
has one query over one set. The driver splits each mesh at the slice the
sample points lie on, the lower half absorbing with the Dirichlet data,
the upper half reflecting with zero flux, has WoSX's two boundary
handlers build FCPW's two trees and copies them into the layouts
(`fcpw_stars_trees.h`); with a reflecting boundary the solution has no
closed form, so the table's error column is the difference between the
two sides. The schedules are the walk on spheres' with the three
traversals in FCPW's order (`wost-fcpwb`, `-threads`, `-persistent`,
`wost-fcpwb-gpu`), and `benchmark.sh --stars` runs its tables.

Not here, on either walk: no source term (the Laplace problem), no
gradient estimate, no Russian roulette or splitting, no non-zero flux or
Robin coefficient on the reflecting boundary, no caching. Each is an
extension to add once the basic walks measure right.

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

The GPU comparison needs one thing built: slang-rhi, the GPU runtime
WoSX's and FCPW's GPU code run on, a submodule of WoSX's FCPW that fetches
the Slang compiler it needs as a prebuilt release at configure time (the
version FCPW's CMake pins). Once, from the repository root, inside the
conda environment:

```bash
git -C deps/wosx submodule update --init deps/nanoflann      # WoSX's GPU headers reach it
git -C deps/wosx/deps/fcpw submodule update --init deps/slang-rhi
bash apps/wosx/build_slang_rhi.sh      # into deps/slang-rhi-build, CUDA backend only
```

`compare.sh` stops with a message if a `*-gpu*` schedule is asked for and
it is not there. The program's own GPU code needs nothing but the CUDA
driver (runtime/bonsai_cuda.h finds it at run time).

## Running

The whole table, from the repository root, inside the `bonsai` conda
environment, with the compiler built:

```bash
apps/wosx/benchmark.sh                 # every mesh, both schedules, CSV, table, geomeans, plot
apps/wosx/benchmark.sh --table-only    # fold, print and plot the logs already there
python3 apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots   # the two CPU figures alone
apps/wosx/benchmark.sh --gpu           # the GPU table: WoSX's GPU solver against the GPU schedules
apps/wosx/benchmark.sh --stars         # the walk on stars' table, wost.bonsai against WoSX's WalkOnStars
apps/wosx/benchmark.sh --stars --gpu   # the walk on stars on the GPU
```

`--stars` runs the walk on stars' comparison over the same meshes: the
schedules default to `wost-fcpwb`, `wost-fcpwb-threads` and
`wost-fcpwb-persistent` (with `--gpu`, to `wost-fcpwb-gpu`), the logs go
to `apps/wosx/results/stars-logs` (`stars-gpu-logs`), the CSV to
`apps/wosx/results/wosx-stars-results.csv` (`wosx-stars-gpu-results.csv`),
the figures to `apps/wosx/plots/wosx-stars-cpu-fcpwb-speedup.*` and
`wosx-stars-gpu-speedup.*` (`plot.py --stars`) and the mesh table to
`wosx-stars-meshes.*`, whose walk lengths are the walk on stars' own.

`--gpu` runs the GPU comparison over the same meshes: the schedules
default to `fcpwb-gpu` (the matching schedule) and the six variants
listed under "Layout of the files", the logs go to
`apps/wosx/results/gpu-logs`, the CSV to
`apps/wosx/results/wosx-gpu-results.csv` and the figure to
`apps/wosx/plots/wosx-gpu-speedup.*` (`plot.py --gpu`), and the idle check
waits for the GPU to be empty of every other process as well as the CPU.
Either run writes `/tmp/benchmark.lock` while it measures -- the sessions'
convention for claiming the machine, one line naming the holder -- and
waits while another's claim is there.

It runs `compare.sh` over WoSX's own demo meshes (the comb electrodes, the
whale, the car, the octopus, the beast, the rover) and the thirteen pbrt
meshes apps/rtq measures (`RTQ_SCENES` names the checkout), under the two
matching schedules, waiting for an idle machine before each mesh and
resuming a stopped run (a mesh's log exists only when complete; `--fresh`
measures again; `PAUSE` holds at a boundary); folds the logs into
`apps/wosx/results/wosx-results.csv`, each row stamped with the compiler's
commit; prints, per schedule, every mesh's row and the geomean of the
speedups with the least and the greatest named; and draws the two CPU
figures into `apps/wosx/plots`, one per tree:
`wosx-cpu-fcpw4w16-speedup.*` (FCPW's four-wide tree, both sides
vectorized) and `wosx-cpu-fcpwb-speedup.*` (WoSX's default binary tree,
both sides as WoSX ships). In each (`plot.py`) the meshes run along the
x axis by triangle count with two bars a mesh -- the threads against the
threads, sixteen persistent workers against WoSX's threads -- each
bonsai's rate over WoSX's (an application is measured in parallel under
the schedule the reference runs; the one-thread cell below is a check by
name, and no bar), the y axis
Speedup and the dashed line at 1 WoSX, named in the legend, the colours
apps/pbrt's figures' (the user, 10-29): the paper's YlGnBu ramp at eight
equal steps laid over each figure's bars in the legend's order, palest
first, `parallel` and `persistent` the same shades on the CPU and GPU
figures (plot.py's SCHEDULE_COLOURS), drawn at the paper's
page width in apps/pbrt/figure_style.py's style. Beside each figure a
`.txt` of the statistics the paper quotes (per schedule one line that
begins with the range, least-greatest, and the geomean, then the count and
the least, quartile, median and greatest cells with their meshes -- the
user, 10-30; what each change of schedule bought, the same way; notes
reading the figure), and the mesh facts in one table rather than on
every plot, `wosx-meshes.txt` and `wosx-meshes.tex` (a tabular to
`\input`), so the figures carry the mesh names alone. The landscape is
left out, since nearly every walk on it escapes on its first step, and
`--exclude` names the meshes left out.
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
apps/wosx/compare.sh --schedule wost-fcpwb,wost-fcpwb-gpu deps/wosx/demo_apps/electrostatics/data/mems-comb-electrodes.obj
```

A schedule named `wost-*` runs the walk on stars (`wost.bonsai`, the driver
built with `WOSX_STARS`); the rest of its
name says the tree and how it is run as for the walk on spheres, on the
binary trees alone. Its driver options: `--eps-reflecting` (the
reflecting shell, the least star radius; 1e-3, WoSX's default) and
`--precision` (the silhouette precision; 1e-3, WoSX's). Its `--query-bound`
default follows WoSX's GPU walk on stars, 1.001 times the old distance
plus the step.

Three schedules on each of two trees are the table's defaults
(benchmark.sh), each its own table under a `=== schedule` line; compare.sh
alone builds and runs `fcpw4w16,fcpw4w16-threads` unless `--schedule` says
otherwise. The `fcpw4w16*` schedules run FCPW's four-wide tree, both sides
vectorized; the `fcpwb*` schedules run WoSX's default tree, FCPW's scalar
binary BVH, both sides as WoSX ships, the same three ways (`fcpwb`,
`fcpwb-threads`, `fcpwb-persistent`). `fcpw4w16` is one thread against one thread: WoSX's
`solve` with `runSingleThreaded`, the program's parfor over the points
left unbound by the schedule (a plain loop), both under numactl on the
physical core the kernel ranks best, as apps/rtq measures -- a check of
the compiled traversal against the hand-written one, run by name
(benchmark.sh --schedules) and not by the tables, whose defaults are the
two parallel schedules of each tree (the user, 2026-10-28: an application
runs in parallel under the reference's schedule; on the large meshes the
one-thread cell ran WoSX's single thread for an hour).
`fcpw4w16-threads` is the threads against the threads: WoSX's `solve`
through its `tbb::parallel_for` over the points, the program's parfor
bound to the CPU threads (which the compiled code hands to the runtime's
`tbb::parallel_for`, the same TBB), both pinned to the performance cores
-- the cores sharing the best core's last-level cache, the frequency
chiplet on this machine, every hardware thread of it -- so both pools have
the same threads and none of the slower cores. `fcpw4w16-persistent` is
the threaded schedule with the points handed to sixteen persistent workers
(`solve_all.persistent(i, t, 16)`, one worker per hardware thread of those
cores) that claim a point at a time from a counter, against the same WoSX
run: beside `fcpw4w16-threads` it measures what a dynamic claim is worth
over TBB's partition of the points when points differ in cost.
`RTQ_CPUS=<cpus>` pins any of them to another set. `RTQ_SKIP_CMAKE=1` runs
the compiler in `BONSAI_BUILD_DIR` as built, without rebuilding it first:
an A/B of two build directories, one at an older commit (PLAN.md,
2026-10-27). `WOSX_KEEP=1` leaves the built program and driver
(`wosx.out`) in place after the run, for a profile under nsys or ncu.
`WOSX_FLAGS="..."` adds compiler flags to every schedule of the run
(`--gpu-max-registers 64`, say, which caps a kernel's registers -- on the
OptiX path through the module's maxRegisterCount since 2026-10-28, OptiX
ignoring the PTX's own directive), for an experiment on one build.

A schedule named `*-gpu*` is the GPU comparison (`compare.sh --schedule
fcpwb-gpu <mesh>`): the driver is built with `WOSX_GPU` against slang-rhi
and run with `--gpu`, pinned like a single-thread run (the host thread
only; both sides run on the card). WoSX's side is its GPU solver as its
demos set it up (demo_apps/basic_2d/gpu_demo.cpp): a task handle on the
CUDA backend that loads FCPW's and WoSX's Slang modules and this
directory's PDE (`gpu/wosx-pde-definition.slang`, the same Dirichlet
data), WoSX's GPU boundary handler building FCPW's binary tree over the
same positions and indices, and `GPUWalkOnSpheresSolver` with the walk
settings above. `--gpu-reference persistent` (the default, WoSX's on CUDA)
runs its persistent-thread kernel, `threads` its other one -- a thread per
copy of a point, the points copied enough times to fill the card, one walk
per dispatch, which rounds the walks per point up and both sides then run
the rounded count; `--gpu-threads N` sets the resident threads (default
131072, WoSX's). The program's side stages every buffer where the
generated header says its schedule reads it (the points, the distances
and the tree on the device; what the host's loops write on the host) and
forbids a copy inside the timed call, and the clock on both sides is
around the solve alone, the dispatches and the wait for them.

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

For the walk on stars (a `wost-*` schedule) the driver splits the mesh
at the slice's plane -- a triangle whose centroid is above it goes to
the reflecting (Neumann) part, the rest to the absorbing (Dirichlet)
part -- and has WoSX's `FcpwDirichletBoundaryHandler` build FCPW's scalar
BVH over the absorbing part and its `FcpwNeumannBoundaryHandler` the
normal cone hierarchy over the reflecting part with its silhouette edges
(convex edges left out of the silhouette tests for an interior
single-sided problem, as WoSX's demos leave them), copies both into the
program's three layouts (`fcpw_stars_trees.h`, which writes the rows as
FCPW's own GPU transfer writes them), and solves the mixed problem with
the Dirichlet data below, zero flux above and no source on both sides,
WoSX's walk settings for a walk on stars (every step a star, the
reflecting shell and the silhouette precision at 1e-3). The sample points
are the slice's interior by the parity count over both parts. With a
reflecting boundary the solution is not the Dirichlet data, so the error
column is the root-mean-square difference between the two sides'
estimates (zero for WoSX by construction) and `z` is read as before.

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
be the same, since WoSX seeds each point's generator from the clock. Against
WoSX's GPU solver, whose statistics keep no variance, `z` is the paired
statistic instead -- the mean of the per-point differences over the
standard error estimated from the differences themselves -- and the
table's heading says `z (paired)`.
`--image P` also writes the slice three ways, `P-bonsai.ppm`, `P-wosx.ppm`
and `P-exact.ppm`, blue to red over [-1, 1], the cells with no point
black. `--shuffle` solves the points in a random order on both sides
rather than along the slice: a thread's run of points is then a scatter
over the mesh instead of a neighbourhood of it, which tells cache sharing
between the threads from load balance when a parallel schedule's gain is
in question (PLAN.md, the persistent schedule).

## Layout of the files

- `wosx.bonsai`: the walk (`walk`, WoSX's `WalkOnSpheres::walk` as a tail
  recursion over the walk's position and its distance, in WoSX's order: a
  step, the escape test, the length limit, a query; `terminal`, its
  projection query at the end), the query (`closest_within`, FCPW's
  findClosestPoint with a BoundingSphere: the argmin over the triangles
  within the sphere's radius, which shrinks to each closer one found), the
  problem's record (`Problem`: the shell, the length limit, the domain's
  box, and `query_bound`, the factor of the step just taken that bounds
  the next query's radius -- WoSX's GPU walk's 2.002, its CPU walk's
  infinity; the driver sets it after the reference it runs, `--query-bound`
  overrides), the Dirichlet data, the sphere sampler, and the exported
  `solve_all` over a batch of points with their first distances. Compiled
  over apps/rtq's elements and one of its FCPW metrics, which compare.sh
  passes before it, as apps/rtq's compare.sh does for rtq.bonsai.
- `walk_common.bonsai`: what the two walks share -- the Dirichlet data,
  the uniform direction on the sphere, the box test a walk escapes by,
  and the walk's outcome record.
- The silhouette query and its elements are apps/rtq's, since apps/rtq
  benchmarks it on its own against FCPW (`apps/rtq/silhouette.bonsai`:
  `closest_silhouette` over `extern silhouettes : set[Edge]`;
  `apps/rtq/elements.bonsai`: the silhouette edge `Edge`, the node's box
  with its cone `ConeBox`, the query `SilhouetteQuery`; the silhouette
  section of `apps/rtq/metrics/fcpw_scalar.bonsai`: FCPW's cone-overlap
  test as the query's `intersects` over the cone box, the box's distance
  as its `distmin` with no far-corner bound, the silhouette test on an
  edge within the sphere as `intersects` over the edge and the edge's
  squared distance as `distmin`, and the ray's and the point's metrics
  over a cone box, which are the box's). The walk imports the query.
- `wost.bonsai`: the walk on stars (`walk_stars`, WoSX's
  `WalkOnStars::walk` as a tail recursion over the position, the normal
  where the walk stands on the reflecting boundary, the distance and the
  last step, in WoSX's order: the star's radius, a uniform direction
  turned into the domain's hemisphere on the boundary, the ray cast
  against the reflecting boundary from the position offset off the
  boundary (WoSX's offsetPointAlongDirection, Ray Tracing Gems ch. 6), a
  hit or the star's surface, the escape test, the length limit, the
  distance query bounded by 1.001 times the old distance plus the step
  as WoSX's GPU walk bounds it; `star_radius`, WoSX's
  computeStarRadiusForReflectingBoundary and its shrink by a hundredth),
  its two queries over the triangles (`hit_reflecting`,
  `closest_absorbing`; the silhouette query is apps/rtq's), the problem's
  record (`StarProblem`) and the exported `solve_stars`. Compiled over
  apps/rtq's elements and the scalar FCPW metric.
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
  `schedules/fcpw4w16-persistent.bonsai`: the points claimed by sixteen
  persistent workers bound to the threads.
  `schedules/fcpw4w16-packet.bonsai`: the walks as the lanes -- a point's
  walks split by 32 and vectorized, then the traversal sorted and put on
  a stack, a packet traversal of the same four-wide tree (single-threaded;
  correct and slow, PLAN.md says why). Not a default of the comparison;
  run it with `--schedule fcpw4w16,fcpw4w16-packet`.
  `schedules/fcpw4w16-packet-points.bonsai`: the same packet over the
  sample points instead, each lane's walks a loop inside.
- `apps/rtq/schedules/layouts/fcpwb.bonsai` and `fcpwb_gpu.bonsai`
  (apps/rtq's; the first imported here by the CPU `fcpwb*` schedules, the
  second by the `fcpwb-gpu*` ones): WoSX's default tree, FCPW's scalar
  binary BVH (FCPW_USE_ENOKI off; the one tree FCPW puts on a GPU). The
  CPU layout is FCPW's own arrays wrapped where they lie -- its 32-byte
  node rows, its primitive pointers, its Triangle objects and its polygon
  soup, the corners gathered through the object's indices at the leaf as
  WoSX's own leaf gathers them (`apps/rtq/fcpw_bvh.h`'s `wrap_fcpw_bvh`,
  under RTQ_FCPW_CPU_BYTES). The GPU layout is FCPW's GPU bytes: the node
  rows and the triangles as FCPW's GPU rows (GPUTriangle: three corners
  and an index, 40 bytes, a tile of one element each with the program's
  edges derived), written by the driver (`copy_fcpw_bvh`) as FCPW's GPU
  transfer writes them. Until 2026-10-25 the CPU comparison read the GPU
  rows too, a gather and an indirection fewer a triangle than WoSX's leaf
  pays (apps/rtq/PLAN.md, "Genuine apples-to-apples"). The node is
  declared bounding its children, as FCPW traverses it, and the layout
  reads the children's boxes out of the children's own rows
  (`this[left].low.x`...), so the parent holds them logically while its
  bytes stay FCPW's -- the device the PLDI 2026 paper's closest-point
  benchmark used for this tree. `schedules/fcpwb.bonsai`: the matching
  schedule on it, nothing vectorized, the two children tested at their
  parent one after the other (`unroll(triangles.Interior.children)`: the
  children's tests written out, each child's box read off its own row, as
  FCPW writes them) and the nearer entered first in FCPW's CPU order (by
  the distance to the box; where both children contain the point, by the
  far corner's; the left child on a tie), the two ordered by one float
  compare and a swap (a run of two children is sorted as scalars,
  SSA/SortRecursion.cpp; before 2026-10-25 it went through Embree's lane
  peel, sixty instructions over FCPW's visit); `fcpwb-threads` and
  `fcpwb-persistent`: the threads and the persistent workers on it;
  `schedules/fcpwb-packet.bonsai` and `schedules/fcpwb-packet-points.bonsai`:
  the two packets over it, one thread; `schedules/fcpwb-packet-threads.bonsai`,
  `fcpwb-packet-16-threads.bonsai`, `fcpwb-packet-8-threads.bonsai` and
  `fcpwb-packet-4-threads.bonsai`: the packet over a point's walks with the
  points over the threads, in gangs of thirty-two lanes, of sixteen (the
  machine's vector width), of eight and of four, the CPU figure's exploration
  bars for vectorization on this tree (the user, 2026-10-28; the packet's
  cost is PLAN.md's, "The packet over the points": a point's walks spread
  over the mesh a few steps in); `schedules/fcpwb-packet-defer-threads.bonsai`
  and `fcpwb-packet-defer-sort-threads.bonsai`: the walks deferred step by
  step onto a queue and the queue drained in gangs of sixteen lanes with
  the packet traversal -- a point's queue, so a gang holds live steps and
  no lane idles behind a longer walk; and a chunk of sixty-four points'
  queue sorted by the walks' positions (`walk.sort(walk, key)` on the
  deferred call, the same directive as the GPU's reordering schedule,
  walk_keys.bonsai's Morton code) so a gang's lanes stand in one cell of a
  32x32x32 grid and ask the tree about one neighbourhood (the user,
  2026-10-29: "maybe a defer + vectorize would work"; PLAN.md, "Why the
  packets lose on the CPU": the compaction alone changes nothing, the sort
  is worth 1.8x). Beside them, as files run by name: `-defer-chunk-threads`
  (the chunk's queue unsorted), `-defer-sort-fine-threads` and
  `-defer-sort-fine-c256-threads` (a 1024x1024x1024 grid's thirty-bit key,
  chunks of 64 and of 256 points), `-defer-sort-c16-threads` (chunks of 16:
  a mesh with few sample points has few chunks for the threads), and
  `-defer-sort-8-threads` (the sorted drain in gangs of eight: 1.03x on the
  head against sixteen's 1.19x), and `-defer-sort-fine-perlane-threads`,
  the per-lane traversal (each lane its own stack and order, the node's row
  gathered from the lanes' own addresses: 0.54x on the head; it found the
  compiler's bug of 2026-10-29, fixed that day). The comparison
  builds FCPW scalar for these
  (`compare.sh --schedule fcpwb,fcpwb-packet`), so WoSX's side runs as it
  ships, and compiles the program over `apps/rtq/metrics/fcpw_scalar.bonsai`,
  the scalar build's arithmetic (true divisions, FCPW's GPU kernels'
  too; Eigen's two reduction orders, `dot` summing `(p0 + p1) + p2` with
  the newer product fused and `squaredNorm` summing `p0 + (p1 + p2)` with
  the left one fused, written as `fma` chains so that every key and every
  test against the radius is FCPW's to the bit), where the `fcpw4w16*`
  schedules take `metrics/fcpw.bonsai`, the vectorized build's.
  `schedules/fcpwb-gpu.bonsai`: the schedule matching WoSX's GPU solver
  (above), and its variants `fcpwb-gpu-parallel` (every walk a thread of one
  launch, no persistent workers), `fcpwb-gpu-reorder` (the walks fastest: a
  warp's claims are one point's walks), `fcpwb-gpu-256k` (262144 workers
  where WoSX has 131072: more than the card holds, so the hardware fills
  its residency), `fcpwb-gpu-reorder-256k` (the walks fastest with those
  workers) and `fcpwb-gpu-parallel-reorder-blk64` (the flat launch with the
  walks fastest, in blocks of 64); and the launch made through OptiX:
  `fcpwb-gpu-optix` (the flat launch as a ray-generation program, the
  traversal the kernel's software one, no query on the ray tracing
  hardware -- the control), `fcpwb-gpu-optix-ser` (the same with the
  threads reordered before every step by the cell of a 32x32x32 grid over
  the box their walk stands in -- shader execution reordering, OptiX's
  `optixReorder`, written as a sort on the walk's recursive call,
  `walk.sort(walk, |x, problem| walk_sort_key(...))`, the key in
  `walk_keys.bonsai`) and `fcpwb-gpu-optix-ser-persistent` (the persistent
  claim of a walk at a time under that reordering); beside them, runnable
  by name and not a bar, `fcpwb-gpu-optix-persistent` (the persistent
  claim under OptiX with no reordering: against `fcpwb-gpu` it measures
  OptiX's compilation of the same kernel with the same warps -- 1.04x
  against 1.44x on the head, PLAN.md "The OptiX launch under Nsight
  Compute"). The figure draws eight of them (the user, 2026-10-31: at most
  eight schedules a figure -- the matching schedule first, then its close
  variants that perform well, then the exploration's, and every schedule
  that is the best on some mesh kept): persistent, parallel,
  persistent-reorder, persistent-256k, persistent-reorder-256k and
  parallel-reorder-blk64 (the last four each the best of the table on at
  least one mesh; the 256k and blk64 ones came out of the exploration
  PLAN.md records, 2026-10-24), then optix, the best of the three OptiX
  forms (the user's ask of 2026-10-28 for the reordering at each step of
  the walk), and defer-block, the best of the deferrals; optix-ser,
  optix-ser-persistent and defer-chunk2 keep their rows and are drawn when
  `plot.py --gpu-schedules` names them. `fcpwb-gpu-point` (a point a claim,
  its walks in sequence; "persistent-point") was a bar until 2026-10-28:
  a tenth of WoSX's rate on every mesh, since a point a claim leaves most
  of the card's threads with no point to claim, and the user had it
  removed from the table's defaults and the figure; its rows stay in the
  CSV and its file stays runnable by name. All of them take their
  traversal from
  `schedules/kernel/fcpwb.bonsai`, which orders the children as WoSX's GPU
  kernel does (bvh.slang: by the distance to the box alone, the left
  child on a tie) where the CPU schedules order as FCPW's CPU traversal
  does -- each platform's comparison against that platform's reference.
  `schedules/fcpwb-gpu-defer.bonsai`: the walks deferred step by step
  onto one card-wide queue (`queue(root)`, `defer`), the pushes, the
  rounds' drains and the pass over the record each in blocks of 256 on
  the device; `schedules/fcpwb-gpu-defer-block.bonsai`: a point per block,
  its walks the threads, the steps deferred onto a queue per point in the
  block's shared memory (`queue(i, DeviceShared)`), the drain's rounds a
  loop inside the kernel with a thread phase per round;
  `schedules/fcpwb-gpu-defer-chunk2.bonsai`, `-chunk4`, `-chunk8`,
  `-chunk16`: the same queue owned by a chunk of two, four, eight or
  sixteen points -- the compacting block wavefront (the user, 2026-10-30;
  PLAN.md, "The compacting block wavefront") -- the chunk split off the
  point loop, the nest reordered twice and collapsed as the card-wide
  deferral has it, the chunk's walks the block's threads (128 to 1024)
  and the queue sized to them (the chunks claimed by persistent blocks,
  `scratch/fcpwb-gpu-defer-chunk2-persistent.bonsai`, the compiler refuses
  today: the block-level claim is not built, PLAN.md's Next list);
  `schedules/fcpwb-gpu-defer-sort.bonsai`: the card-wide queue sorted each
  round by the walks' positions (`walk.sort(walk, key)`, walk_keys.bonsai's
  Morton code, the device radix sort), measured by name (0.75x on the head
  and the dragon, PLAN.md, "Seven more without the bounded queue").
- `schedules/layouts/fcpwb2.bonsai`: the same binary tree in another
  layout, each interior node's row holding its two children's boxes and
  rows (56 bytes, built from FCPW's rows by the driver, `fcpw_bvh2.h`), so
  that a visit reads one row where FCPW's layout reads three; the
  triangles as for fcpwb. `schedules/fcpwb2.bonsai`: the matching
  schedule's traversal on it on one thread, in FCPW's CPU order;
  `schedules/fcpwb2-threads.bonsai` and `schedules/fcpwb2-persistent.bonsai`:
  the same with the points over the threads and with the persistent
  workers, run as the CPU table's cells are and drawn on the fcpwb CPU
  figure as its exploration bars (parallel-fcpwb2, persistent-fcpwb2:
  what the layout alone is worth in parallel, WoSX's side unchanged on
  FCPW's own rows; the user, 2026-10-28);
  `schedules/fcpwb2-gpu.bonsai`: WoSX's GPU kernel over it, in the
  kernel's order (`schedules/kernel/fcpwb2.bonsai`), a candidate for the
  optimized schedule.
  `schedules/fcpw4w16-gpu.bonsai`: the same kernel over FCPW's four-wide
  tree, which FCPW cannot put on a GPU.
- `schedules/layouts/fcpw_stars.bonsai`: the walk on stars' three sets in
  the bytes FCPW's CPU reads -- the absorbing triangles on FCPW's scalar
  BVH as apps/rtq's `fcpwb.bonsai` reads it (FCPW's node array, its
  primitive pointers, its Triangle objects by address, its polygon soup,
  the corners gathered through the object's indices at the leaf), the
  reflecting triangles and the silhouette edges on FCPW's one normal cone
  hierarchy as apps/rtq's `fcpws.bonsai` reads it (SnchNode rows in FCPW's
  CPU order handed to two layouts that each read the leaf's range that is
  theirs; the edges as FCPW's SilhouetteEdge objects by address, their two
  face normals derived from the positions for the two-face case as FCPW
  recomputes them); the parent holds its children's boxes and cones
  logically, read from the children's rows, as for fcpwb.
  `fcpw_stars_trees.h` wraps FCPW's own arrays for it, nothing copied.
  `schedules/layouts/fcpw_stars_gpu.bonsai`: the same three sets in the
  rows FCPW's GPU reads -- 60-byte GPUSnchNode rows, GPUTriangle rows with
  the corners stored, 56-byte GPUEdge rows with the normals stored -- for
  the GPU schedules (`kernel/wost_fcpwb.bonsai`); `fcpw_stars_trees.h`
  copies FCPW's trees into them as FCPW's own GPU transfer writes them
  (extractSnchNodes, extractTriangles, extractSilhouetteEdges), without
  its GPU header.
  `schedules/cpu/wost_fcpwb.bonsai`: the three traversals in FCPW's CPU
  order (the closest point by the box's distance and, for two children
  holding the point, the far corner's; the silhouette query by the box's
  distance with the cone and box tests at the parent; the ray by the slab
  entry), under a name an import can say (a schedule's name carries
  dashes, which an import path cannot); `schedules/wost-fcpwb.bonsai`
  imports it and nothing more, one thread against one thread (a check by
  name, as `fcpwb` above); `wost-fcpwb-threads` and `wost-fcpwb-persistent`
  add the threads and the persistent workers, the table's two schedules.
  `schedules/kernel/wost_fcpwb.bonsai`:
  the same traversals in WoSX's GPU kernel's order (the closest point by
  the box's distance alone); `schedules/wost-fcpwb-gpu.bonsai`: fcpwb-gpu's
  four directives over it, the schedule matching WoSX's GPU walk on stars;
  and the walk on spheres' GPU variants on this walk, each the same
  directives on `solve_stars` under the same legend label --
  `wost-fcpwb-gpu-parallel`, `-reorder`, `-256k`, `-reorder-256k`,
  `-parallel-reorder-blk64`, `-optix` and `-optix-ser` (the reordering
  before each step of a walk whose step is three traversals) -- with
  `wost-fcpwb-gpu-defer-block`, the per-point deferral of
  fcpwb-gpu-defer-block on this walk (the three traversals written out,
  the walk deferred rather than loopified), the exploration the user asked
  for on 2026-10-28, `wost-fcpwb-gpu-defer-chunk2`, `-chunk4`, the
  compacting block wavefront on this walk (a chunk of two or four points
  a block, their walks its threads; the user, 2026-10-30), and
  `wost-fcpwb-gpu-defer`, the card-wide queue on this walk (fcpwb-gpu-
  defer's directives on solve_stars; the stars' GPU figure's `defer` bar:
  0.82x by geomean, the best schedule on the rover at 1.97x and ahead of
  the claim on the ivy and the dambreak, a quarter of WoSX's rate on the
  small meshes whose run is its hundreds of launches -- the long walks'
  divergence worth the compaction where a run is long enough to pay for
  the rounds; PLAN.md, "Three more without the bounded queue") with
  `wost-fcpwb-gpu-defer-sort`, the queue sorted each round by the walks'
  positions, measured by name. The stars' GPU figure draws eight by the
  same rule as the walk on spheres' (the user, 2026-10-31): persistent,
  parallel, persistent-reorder, persistent-256k, persistent-reorder-256k,
  parallel-reorder-blk64 (the last four each the best on some mesh), then
  optix-ser, the best OptiX form here, and defer, the card-wide queue,
  which is the best schedule on the rover; optix, optix-ser-persistent,
  defer-block and defer-chunk2 keep their rows and are drawn when named.
- `gpu/wosx-pde-definition.slang`: the PDE WoSX's GPU solver is given,
  the Laplace problem with the program's Dirichlet data, in the shape
  WoSX asks for (a module named `wosx_pde_definition` with a `PDEType`).
  `build_slang_rhi.sh`: builds slang-rhi once, into deps/slang-rhi-build.
- `wosx_hook.cpp`: the driver, for both walks (built with `WOSX_STARS` for
  the walk on stars); `compare.sh`, `benchmark.sh`, `plot.py`: the scripts.
  The mesh reader and the copy of FCPW's tree into the layout are
  apps/rtq's (`apps/rtq/mesh.h`, `apps/rtq/fcpw_tree.h`), shared.
- `results/`: the measured data, generated and gitignored -- the logs, one
  CSV with one row per measured (mesh, schedule), the rover unzipped;
  `results/nsys/` and `results/ncu/` the Nsight Systems traces and Nsight
  Compute reports of the GPU kernels with their CSV exports, read with
  apps/rtq's `nsys_kernels.py` and `ncu_kernels.py` (PLAN.md, 2026-10-27).
- `scratch/`: an agent's working files for this app, gitignored.

The generated `wosx.h`, `wosx.o`, `wosx.bir` and `wosx.ll` are left in this
directory by `compare.sh` only while it runs, and are not committed.
