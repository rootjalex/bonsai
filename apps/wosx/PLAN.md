# apps/wosx: the working record

What was built, measured and decided for the walk on spheres against WoSX,
and why. README.md says what the app is and how to run it; this file is
the history and the open questions.

## 2026-10-08: the basic walk

The goal: a walk on spheres benchmark beside apps/pbrt's and apps/rtq's,
compared against WoSX (deps/wosx) as those compare against pbrt and Embree
-- the same tree, the same arithmetic, the same question, the schedule
following the reference's loop step for step. The stale apps/wos was not
built on; this app starts fresh and imports from apps/rtq what already
matches FCPW: the elements, FCPW's arithmetic (metrics/fcpw.bonsai), the
`closest` query, the four-wide tree in FCPW's bytes (layouts/fcpw4w16) and
the directives of its schedule for that query.

What WoSX does, read from its source (include/wosx/point_estimation/
walk_on_spheres.h, core/sampling.h, core/distributions.h, utils/
fcpw_geometric_queries.h, demo_apps/basic_2d):

- `WalkOnSpheres::walk`: `while (dist > epsilonShell) { ball(x, dist);
  source contribution; dir = uniform sphere; x += dist * dir; throughput
  *= directionSampledPoissonKernel (1 for the harmonic kernel); if outside
  the box -> EscapedDomain; weight window (nothing at the defaults:
  Russian roulette threshold 0, splitting threshold max); walkLength++; if
  >= maxWalkLength -> ExceededMaxWalkLength; dist = FCPW distance }`, then
  ReachedAbsorbingBoundary. `getTerminalContribution` projects the point
  onto the boundary -- a second FCPW query at the same point -- and
  returns the Dirichlet value there for both the reached and the
  exceeded case; an escaped walk adds no estimate. The first distance of
  every walk is the sample point's, computed once by the demo
  (`SamplePoint::distToAbsorbingBoundary`), so WoSX asks the boundary
  `steps + 1` times a walk: once after each step, once for the projection.
  The recursion follows that order exactly -- the first distance handed in
  per point (the driver computes it with WoSX's own query, off the clock,
  for both sides), a query after each step, and `terminal`'s query at the
  end -- after a first version that asked at the start of each step and
  read the projection off the last answer (the same count, a different
  placement); the user asked for WoSX's algorithm matched.
- The distance: `aggregate->findClosestPoint(BoundingSphere(x, maxFloat),
  interaction, false)` and `interaction.d` -- apps/rtq's `closest` and its
  record's `d`; the projection is the same call's `p`.
- The direction: `z = 1 - 2 u0; r = sqrt(max(0, 1 - z^2)); phi = 2 pi u1`.
  WoSX's `2.0f*M_PI*u[1]` is evaluated in double and stored to a float;
  the program's is a float product. A last bit of phi, on streams that
  cannot be the same anyway.
- The generator: pcg32 (Wenzel Jakob's), one per sample point, seeded
  from the clock and a counter (`seedRng`). It is the generator
  apps/pbrt/rng.bonsai transcribed from pbrt (pbrt's RNG is the same
  header), so `RNG`, `next_uint` and `set_sequence` are imported from
  there; `rng.bonsai` adds pcg32's float (`(u >> 9) | 0x3f800000` as a
  float, minus one) and its seeding order. The driver seeds the program
  with `--seed` and the point's index, so a run repeats.
- The settings: the basic demo's defaults (demo_apps/basic_2d/demo.cpp,
  runWalkOnSpheres): epsilon 1e-3 on a mesh normalized into the unit
  sphere, maxWalkLength 1024, the weight window off, gradient off (the
  `Solution` quantity), the source term ignored for the Laplace problem.

The problem: Laplace with the Dirichlet data `g = x^2 - y^2`, harmonic, so
that the solution is `g` itself and either side's estimate is checked
against the truth as well as against the other side. The sample points are
a slice through the middle of the mesh, the cells inside the boundary by
WoSX's own `insideDomain` (a parity count of ray hits along the three
axes), minus the ones in the shell.

The agreement test: the two sides' streams differ, so the estimates at a
point differ by their noise. WoSX's `SampleStatistics` keeps the variance
of its estimates per point; the driver forms `(ours - theirs) /
sqrt(var / N_theirs + var / N_ours)` per point and prints the root mean
square, which is one when both sides draw from the same distribution.

### First measurements (2026-10-08, cpu 11, least of 3)

`compare.sh --side 128 --walks 64 --repeats 3`, epsilon 1e-3, at most
1024 steps, the `fcpw4w16` schedule:

| mesh | points | walks | WoSX Msteps/s | bonsai Msteps/s | speedup | walk length (WoSX, bonsai) | rms error (WoSX, bonsai) | z |
|---|---|---|---|---|---|---|---|---|
| dragon (7.2M tris) | 1989 | 127296 | 0.42 | 0.51 | 1.20x | 10.81, 10.84 | 3.27e-3, 3.20e-3 | 1.15 |
| ganesha (4.3M tris) | 6960 | 445440 | 0.55 | 0.67 | 1.23x | 16.61, 16.61 | 1.66e-2, 1.67e-2 | 1.18 |

Both sides walk the same length and miss the exact solution by the same
amount; the slices (`--image`) are the same picture. `z` a little above
one: the per-point variance is estimated from 64 walks of a heavy-tailed
estimator, so the standard error is understated where a point's walks
happened to be quiet. The speedup is the distance query's, consistent with
apps/rtq's FCPW closest-point rows at 1.1-1.3x -- the walk's own cost is
small beside the traversal of a tree this size (a step is 2-2.5 us).

What the generated code is (`compiler -b llvm`, read before the timing):
everything inlines into `solve_all`; the walk is one loop (loopify);
`closest` is inlined at its three call sites -- the step's query and the
two `terminal` calls (shell, length limit) -- each with its own 96-deep
stacks, where WoSX calls one `findClosestPoint` through a virtual call
from two places. LLVM's choice; the stack slots are allocas of the
function and the three copies are the same traversal.

### The second day's additions (2026-10-08, afternoon)

The user's three asks: that the full solver runs (it does: WoSX's
`WalkOnSpheres::solve` with its PDE, settings, points and statistics, the
call its demos make; the Laplace problem leaves the source branch off on
both sides), more meshes, and a parallel comparison beside the
single-threaded one.

- **Meshes.** WoSX ships six 3D demo meshes as OBJ (the comb electrodes,
  600 triangles; the whale, 100k; the car, 325k; the octopus, 497k; the
  beast, 606k; the rover, zipped, 200 MB of OBJ), read here by WoSX's own
  loader; plus apps/rtq's thirteen pbrt meshes. `benchmark.sh` runs them
  all, in order of triangle count in the table.
- **Open surfaces.** Most pbrt meshes are not closed (head has 766 rim
  edges). The driver counts rim edges (an edge of one triangle) and sets
  WoSX's `domainIsWatertight` from that: closed, the slice's interior by
  WoSX's parity test; open, the whole slice, WoSX's own handling of a
  non-watertight boundary. Both sides see the same points either way.
- **Threads.** `fcpw4w16-threads` binds the points' parfor to the CPU
  threads (`solve_all.bind(i, CPUThread)` -> runtime/bonsai_parallel.h's
  tbb::parallel_for) against WoSX's `solve` through tbb::parallel_for, the
  same TBB; the driver's `--threads` switches WoSX's side. compare.sh pins
  the parallel run to the performance cores -- the cores sharing the best
  core's last-level cache (cpus 8-15,24-31 here, the frequency chiplet,
  both hardware threads of each core) -- and TBB sizes both pools by that
  mask (16). The hook no longer defines its own serial bonsai_parallel_for:
  the single-thread schedule leaves the parfor unbound, so the compiled
  code has no call to hand out.
- **The accounting bug the head showed.** The first open-mesh run printed
  WoSX's walks at 9.8 steps and ours at 14.0: the driver divided WoSX's
  steps by its ended walks and ours by all walks. Counted alike -- the
  program now returns each point's count of ended walks -- both sides end
  46.1-46.2k of the 65k walks at 13.7-13.8 steps.
- **The agreement statistic.** The per-point z (the difference over the
  point's own standard error, root-mean-squared) read 16 on the head: a
  point's few walks on an open surface give a sample variance near zero,
  and dividing by it is meaningless. Replaced by one global z: the mean
  signed difference between the sides over all points divided by its
  standard error from the summed per-point variances -- a bias test, of
  order one when the sides estimate the same thing (0.5-1.9 on the head,
  both schedules).

Whale, both schedules, 64 x 64, 16 walks, one run: 1.29x one thread
against one thread, 1.30x the threads against the threads (16 on each
side).

### The table (2026-10-08, `benchmark.sh`, side 128, 64 walks, least of 3)

Nineteen meshes in order of triangle count, WoSX's six and apps/rtq's
thirteen; the rates in million steps per second; the walk length and the
rms error against the exact solution given as (WoSX, bonsai).
`fcpw4w16`: one thread against one thread on cpu 11.

| mesh | triangles | closed | points | WoSX | bonsai | speedup | walk length | rms error | z |
|---|---|---|---|---|---|---|---|---|---|
| comb | 600 | yes | 2240 | 5.37 | 6.57 | 1.22x | 6.32, 6.33 | 1.24e-3, 1.24e-3 | -0.05 |
| pavilion | 11366 | no | 16253 | 2.77 | 3.31 | 1.19x | 12.20, 12.22 | 4.41e-2, 4.42e-2 | -0.08 |
| head | 17674 | no | 16284 | 2.43 | 2.93 | 1.21x | 13.71, 13.69 | 1.39e-1, 1.39e-1 | -0.96 |
| zero-day | 48960 | no | 16303 | 0.89 | 1.17 | 1.31x | 8.08, 8.11 | 7.74e-2, 7.73e-2 | 0.62 |
| whale | 99648 | yes | 4829 | 1.53 | 1.83 | 1.20x | 11.48, 11.45 | 5.57e-3, 5.76e-3 | -1.61 |
| bmw | 110592 | no | 16328 | 1.73 | 2.15 | 1.24x | 14.93, 14.92 | 1.18e-1, 1.18e-1 | -1.02 |
| crown | 155520 | yes | 460 | 1.76 | 2.15 | 1.21x | 6.58, 6.61 | 2.85e-3, 3.03e-3 | -2.01 |
| ivy | 179603 | no | 16348 | 1.31 | 1.70 | 1.29x | 22.53, 22.59 | 7.81e-2, 7.79e-2 | 0.34 |
| car | 324640 | yes | 11259 | 1.93 | 2.31 | 1.20x | 14.67, 14.68 | 1.43e-2, 1.40e-2 | 1.03 |
| villa | 390784 | no | 16318 | 1.27 | 1.52 | 1.20x | 14.21, 14.20 | 8.10e-2, 8.12e-2 | -0.17 |
| octopus | 497236 | yes | 2342 | 0.82 | 0.96 | 1.17x | 11.66, 11.64 | 3.06e-3, 3.04e-3 | -0.51 |
| beast | 606136 | yes | 1344 | 1.11 | 1.34 | 1.21x | 11.62, 11.61 | 7.14e-3, 6.95e-3 | 1.66 |
| dambreak | 1015024 | no | 16188 | 1.69 | 2.12 | 1.25x | 18.57, 18.58 | 1.68e-2, 1.70e-2 | 0.21 |
| sportscar | 1091232 | no | 16124 | 1.37 | 1.71 | 1.25x | 13.92, 13.92 | 2.92e-2, 2.93e-2 | -0.12 |
| rover | 1616316 | no | 16294 | 0.98 | 1.24 | 1.27x | 17.50, 17.50 | 8.08e-2, 8.09e-2 | -0.21 |
| landscape | 1916928 | no | 6144 | 0.08 | 0.16 | 1.81x | 1.13, 1.16 | 2.89e-1, 2.89e-1 | -0.99 |
| lte-orb | 3423232 | no | 16233 | 0.20 | 0.26 | 1.24x | 15.20, 15.23 | 7.50e-2, 7.34e-2 | 1.62 |
| ganesha | 4323658 | no | 16215 | 0.54 | 0.68 | 1.26x | 15.69, 15.70 | 9.97e-2, 9.88e-2 | -1.54 |
| dragon | 7219045 | no | 16229 | 0.25 | 0.31 | 1.24x | 15.02, 15.03 | 1.25e-1, 1.25e-1 | -0.66 |

Geomean 1.256 over the 19 (least 1.17 octopus, greatest 1.81 landscape).

`fcpw4w16-threads`: the threads against the threads, both pinned to cpus
8-15,24-31 (16 threads each side).

| mesh | triangles | closed | points | WoSX | bonsai | speedup | walk length | rms error | z |
|---|---|---|---|---|---|---|---|---|---|
| comb | 600 | yes | 2240 | 57.16 | 79.15 | 1.39x | 6.34, 6.33 | 1.28e-3, 1.24e-3 | 1.01 |
| pavilion | 11366 | no | 16253 | 31.93 | 40.60 | 1.27x | 12.22, 12.22 | 4.43e-2, 4.42e-2 | -1.36 |
| head | 17674 | no | 16284 | 28.72 | 36.43 | 1.27x | 13.68, 13.69 | 1.39e-1, 1.39e-1 | 0.09 |
| zero-day | 48960 | no | 16303 | 10.60 | 14.43 | 1.35x | 8.06, 8.11 | 7.74e-2, 7.73e-2 | 0.51 |
| whale | 99648 | yes | 4829 | 17.91 | 22.88 | 1.28x | 11.45, 11.45 | 5.72e-3, 5.76e-3 | 1.33 |
| bmw | 110592 | no | 16328 | 20.52 | 26.35 | 1.28x | 14.92, 14.92 | 1.18e-1, 1.18e-1 | -0.50 |
| crown | 155520 | yes | 460 | 21.07 | 26.44 | 1.26x | 6.64, 6.61 | 2.96e-3, 3.03e-3 | -1.20 |
| ivy | 179603 | no | 16348 | 15.61 | 20.66 | 1.32x | 22.53, 22.59 | 7.76e-2, 7.79e-2 | -0.44 |
| car | 324640 | yes | 11259 | 23.19 | 28.95 | 1.25x | 14.67, 14.68 | 1.40e-2, 1.40e-2 | 1.06 |
| villa | 390784 | no | 16318 | 15.00 | 18.76 | 1.25x | 14.21, 14.20 | 8.11e-2, 8.12e-2 | 0.38 |
| octopus | 497236 | yes | 2342 | 9.38 | 11.87 | 1.27x | 11.65, 11.64 | 3.06e-3, 3.04e-3 | 0.49 |
| beast | 606136 | yes | 1344 | 13.19 | 16.69 | 1.26x | 11.61, 11.61 | 6.89e-3, 6.95e-3 | 0.79 |
| dambreak | 1015024 | no | 16188 | 20.49 | 25.85 | 1.26x | 18.55, 18.58 | 1.69e-2, 1.70e-2 | 0.21 |
| sportscar | 1091232 | no | 16124 | 16.53 | 21.01 | 1.27x | 13.92, 13.92 | 2.92e-2, 2.93e-2 | 0.68 |
| rover | 1616316 | no | 16294 | 11.58 | 14.94 | 1.29x | 17.48, 17.50 | 8.11e-2, 8.09e-2 | -0.62 |
| landscape | 1916928 | no | 6144 | 1.03 | 2.11 | 2.06x | 1.16, 1.16 | 2.89e-1, 2.89e-1 | 0.53 |
| lte-orb | 3423232 | no | 16233 | 1.50 | 1.87 | 1.24x | 15.20, 15.23 | 7.39e-2, 7.34e-2 | 0.19 |
| ganesha | 4323658 | no | 16215 | 4.88 | 5.89 | 1.20x | 15.69, 15.70 | 9.88e-2, 9.88e-2 | -1.13 |
| dragon | 7219045 | no | 16229 | 1.93 | 2.14 | 1.11x | 15.01, 15.03 | 1.24e-1, 1.25e-1 | -0.67 |

Geomean 1.299 over the 19 (least 1.11 dragon, greatest 2.06 landscape).

What the table says:

- Both sides solve the same problem on every mesh: the ended counts
  agree (every walk ends on the six closed meshes; on the open ones the
  two sides' counts differ by the streams' noise), the mean walk lengths
  agree to the second decimal, the rms errors against the exact solution
  are the same on both sides, and z is of order one everywhere (the
  crown's -2.01 on 460 points is one draw in nineteen; the threaded run
  of the same mesh reads -1.20).
- Single-threaded, the speedup is 1.17-1.31x on every real mesh, the
  distance query's margin apps/rtq measured against FCPW's closest point,
  since a step is one query.
- Threaded, 1.24-1.39x on the meshes up to 2M triangles -- a little above
  the single-threaded cell on each -- and lower on the two largest:
  ganesha 1.20x, dragon 1.11x. There sixteen threads give both sides
  only 7.7-9x over one thread (dragon: WoSX 0.25 -> 1.93, bonsai 0.31 ->
  2.14) against 11-14x on the rest: a 7M-triangle tree does not fit the
  chiplet's cache and the solve is bound by memory, where the traversal's
  arithmetic margin counts for less. The same on both sides; nothing in
  our code is per-thread except TBB's partition.
- The landscape is not a measurement of the traversal. It is a terrain,
  a height field whose normalized box is 0.001 thick in y, so a walk's
  first step leaves the box almost always: 1.1-1.2k of 393k walks end,
  at 1.1 steps, on both sides, and the rate is one query per surviving
  walk over the time of all of them. Its 1.81x and 2.06x are the cost of
  a walk that escapes (a sphere sample, a box test, no query), which is
  cheaper in the compiled loop than in WoSX's; it is kept in the table
  because it is one of apps/rtq's thirteen, and read as that.
- The rover (WoSX's largest demo, 1.6M triangles after its zip) runs
  like the other meshes of its size.

### Issues raised to the user

1. **Which FCPW tree WoSX runs on.** In this comparison both sides run
   on the same four-wide Mbvh: the driver asks WoSX's handler to
   vectorize, WoSX's queries go through the one aggregate its scene
   built, and `copy_fcpw_tree` casts that aggregate to the Mbvh and
   refuses anything else, so the table is one tree against itself. The
   question is about WoSX's default. As shipped, WoSX's CMake forces
   `FCPW_USE_ENOKI OFF` and its boundary handler's default is
   `enableBvhVectorization=false`: WoSX's CPU solver runs on FCPW's scalar
   binary BVH (`Bvh<3, Triangle>`, each node holding its own box), not the
   vectorized Mbvh apps/rtq matched. The comparison here builds WoSX's
   handler with vectorization on -- a public argument of its API, FCPW
   compiled with enoki as FCPW's own CMake would on this machine -- and
   runs the schedule over the Mbvh apps/rtq already has the bytes and the
   directives for. That is the configuration a user of WoSX who turned
   vectorization on would get, and the apples-to-apples point for the
   traversal apps/rtq measured; it is not WoSX's default. Matching the
   default means a second layout (FCPW's scalar `BvhNode`: a box, a
   reference offset, a count, the second child's offset; the leaves runs
   of triangles in FCPW's primitive array) over a binary tree whose node
   bounds itself, and FCPW's scalar closest-point arithmetic
   (`findClosestPointTriangle` in Eigen, true divisions, a root per
   triangle), neither of which apps/rtq has. Listed as the next step; the
   user decides whether the default is the comparison that matters.
2. **The reference's extra query.** WoSX's API separates the distance
   query from the projection (`GeometricQueries` returns a distance alone,
   so that a distance grid can serve it too), so every walk ends with a
   second `findClosestPoint` at the point the last one answered. Not the
   method's requirement, the interface's; the program asks it too
   (`terminal`), since the comparison is of the algorithm as WoSX runs
   it. The driver reports the mean walk length on both sides so the work
   stays visible.
3. **Where pcg32 should live.** apps/wosx imports apps/pbrt/rng.bonsai
   for the generator. A `stdlib/pcg32.bonsai` that both apps import would
   be the clean home; moving pbrt's `RNG`, `next_uint`, `set_sequence`
   and `advance` there touches apps/pbrt, which another session is
   working in, so it was not done here.
4. **Sharing with apps/rtq.** `closest` was moved out of apps/rtq/
   rtq.bonsai into apps/rtq/queries.bonsai with `trace` and `occluded`
   (rtq.bonsai keeps the exports and imports it), the mesh reader into
   apps/rtq/mesh.h and the copy of FCPW's tree into apps/rtq/fcpw_tree.h,
   so that this app's driver includes them rather than carrying copies.
   The schedule's seven lines for `closest` are repeated from apps/rtq's
   schedule, since that file also schedules the two ray queries this
   program does not have. compare.sh's core-pinning block is a copy of
   apps/rtq's.
5. **apps/wos** (the stale starter, one file): deleted 2026-10-08 on the
   user's word, this app standing.

The user's rulings of 2026-10-08 afternoon: the scalar-BVH comparison
(issue 1) waits; pcg32 stays in apps/pbrt (issue 3) for now.

### The plot (2026-10-08)

`plot.py`, apps/rtq's bar chart for this app: the meshes by triangle
count, two bars each -- one thread against one thread, the threads against
the threads -- bonsai's rate over WoSX's, the landscape left out (its bar
is the cost of an escaping walk). Over the 18 plotted meshes the geomeans
are 1.231 single-threaded (1.17-1.31) and 1.267 threaded (1.11-1.39). No
parameter is swept, as apps/pbrt sweeps the samples per pixel: the shell
and the walk count set how many steps a point takes, not what a step
costs, so they stay at WoSX's demo defaults (a sweep over the shell would
move the queries nearer the surface, and could be a second figure if the
leaf's share of a query is ever the question).

### Next

- The scalar-BVH layout and FCPW's scalar arithmetic, for WoSX's default
  configuration (issue 1).
- The extensions, one at a time against WoSX: a source term (Poisson),
  the gradient estimate, walk on stars for Neumann boundaries.
- Why the two largest meshes scale worse across the threads on both sides
  (the table's note): a profile of the threaded dragon solve, cache misses
  against the single-threaded one.
