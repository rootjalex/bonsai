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
  2.14) against 11-14x on the rest. Read at the time as the solve being
  bound by memory on a tree too big for the chiplet's cache; the
  persistent schedule below shows what the loss actually is -- the
  threads working far apart in that tree -- and recovers most of it.
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

### The walks as a parfor (2026-10-08, evening)

The user asked why the loop over a point's walks was a `for`. It followed
WoSX's solve literally -- one generator per point, the walks drawn from it
in turn -- but the walks are independent, and a `for` cannot be scheduled.
It is now a parfor: each walk seeds its own pcg32 from the point and the
walk (`seed, i * walks + w`; the streams could never match WoSX's anyway,
which seeds from the clock), and a walk adds to the point's sums by atomic
accumulates, the one thing the walks share. The comparison's schedules
leave the walks a plain loop, so nothing in the measurement changes: the
single-thread build has no atomics (SSA/DemoteAtomics.h takes them off a
loop nothing runs in parallel), and the whale measures as before, 1.19x
one thread and 1.27x the threads, the same ended counts, lengths and errors.

The threaded build did keep three atomics at first: the contention analysis
(SSA/Contention.cpp) read the sums' address -- an alloca inside the bound
point loop's body -- as one that does not depend on the point index, and
so as shared by every point. An allocation inside the loop's body is each
iteration's own, which no other iteration can reach; the analysis now
knows that (tier zero, `allocated_inside`), and the kernel has no atomics.
backends/llvm/demote-atomics has the case (`own_bound`).

### The persistent schedule (2026-10-08, evening)

The user asked for the new directive tried on the threads schedule, with
as many workers as the pinned cores have hardware threads: eight cores,
two threads each, sixteen (compare.sh's pin set 8-15,24-31; TBB's arena
is sized by the same mask, so WoSX's side has sixteen too).
`schedules/fcpw4w16-persistent.bonsai` is the threads schedule with
`solve_all.persistent(i, t, 16)` and `bind(t, CPUThread)`: sixteen
workers claim points one at a time from a counter (`w` is the program's
walks loop, so the workers are `t`). The counter claim is the only atomic
left in the kernel; the walks' sums are the point's own storage and the
contention analysis takes their atomics off. compare.sh runs it as a
threaded schedule, and benchmark.sh has it as a third default.

A fresh sweep of the nineteen meshes under all three schedules (side 128,
64 walks, least of 3; the single-thread and threads columns are this
run's, within noise of the morning's). The threads column is bonsai's rate
under `fcpw4w16-threads`, the persistent column under
`fcpw4w16-persistent`, both in million steps per second on the same
sixteen threads; the gain is the ratio; the speedup is over WoSX's
parallel solve in the persistent run.

| mesh | triangles | threads | persistent | gain | speedup over WoSX |
|---|---|---|---|---|---|
| comb | 600 | 78.75 | 78.48 | 1.00 | 1.37x |
| pavilion | 11366 | 40.48 | 40.21 | 0.99 | 1.26x |
| head | 17674 | 36.50 | 36.07 | 0.99 | 1.26x |
| zero-day | 48960 | 14.44 | 14.29 | 0.99 | 1.34x |
| whale | 99648 | 22.85 | 22.91 | 1.00 | 1.27x |
| bmw | 110592 | 26.33 | 26.27 | 1.00 | 1.28x |
| crown | 155520 | 25.49 | 27.00 | 1.06 | 1.27x |
| ivy | 179603 | 20.67 | 20.66 | 1.00 | 1.32x |
| car | 324640 | 28.99 | 28.91 | 1.00 | 1.25x |
| villa | 390784 | 18.84 | 18.82 | 1.00 | 1.26x |
| octopus | 497236 | 11.82 | 11.87 | 1.00 | 1.28x |
| beast | 606136 | 16.55 | 16.66 | 1.01 | 1.27x |
| dambreak | 1015024 | 25.84 | 25.85 | 1.00 | 1.27x |
| sportscar | 1091232 | 21.08 | 20.98 | 1.00 | 1.27x |
| rover | 1616316 | 14.96 | 15.18 | 1.01 | 1.31x |
| landscape | 1916928 | 1.94 | 1.94 | 1.00 | 1.93x |
| lte-orb | 3423232 | 1.75 | 2.83 | 1.62 | 1.91x |
| ganesha | 4323658 | 5.76 | 8.08 | 1.40 | 1.70x |
| dragon | 7219045 | 2.19 | 3.42 | 1.56 | 1.89x |

Geomeans over the 19: single-thread 1.248, threads 1.294, persistent
1.388 (over the 18 plotted, without the landscape: 1.223, 1.266, 1.363).
Ended counts, walk lengths and errors are the same under both threaded
schedules on every row: the work is the same, only its distribution over
the threads changed.

What the gain is. It is not load balance: the open meshes, where forty
percent of the walks escape on their first steps and the points' costs
differ most (head, bmw, rover, dambreak), gain nothing; the three meshes
that gain are exactly the three trees past 3M triangles, whatever their
openness. The test that settles it is the driver's new `--shuffle`, which
solves the points in a random order on both sides -- every schedule's
imbalance is what it was, but a thread's run of points is spread over
the mesh instead of being a neighbourhood of it. Dragon, shuffled, both
threaded schedules: bonsai 1.94 and 1.94 (unshuffled 2.19 and 3.42), WoSX
1.70 (unshuffled 1.85). So the gain is cache sharing between the threads.
A point's walks traverse the part of the tree around the point; sixteen
workers claiming consecutive points work on neighbouring points at the
same moment and share the tree's nodes and triangles in the chiplet's
cache, where TBB's partition hands each thread its own contiguous range
of the slice -- sixteen threads in sixteen distant regions of a tree that
does not fit the cache -- and the shuffle, which gives every thread a
scatter, is worst of all. On a tree that fits the cache the order does
not matter, which is the first sixteen rows. This is also what the
morning's "memory-bound" scaling of the two largest meshes was: the
threads schedule's 7x on the dragon against the persistent's 11x over one
thread is the same tree walked by the threads apart or together.

Two things follow. The persistent schedule is the better parallel
schedule on this app for big trees and costs nothing on small ones, which
is why it is the third default rather than a variant; and WoSX's own
parallel solve has the same loss (its tbb::parallel_for over the points,
the same partition), which is why the speedup over it widens to 1.7-1.9x
on those meshes rather than our rate merely rising. A fairer comparison
to WoSX on those three would give WoSX a dynamic schedule too; that is a
change to WoSX, not to this benchmark, and the table says which schedule
each row is.

### The packet schedule (2026-10-09, morning; compiler commit 29d34c71)

`schedules/fcpw4w16-packet.bonsai`: the walks as the lanes, not the tree.
The same four-wide Mbvh in FCPW's bytes, the walk a loop, and a point's
walks split by 32 and vectorized (`solve_all.split(w, w_gang, w_lane, 32,
true).vectorize(w_lane)`), then `closest.sort(...).loopify(96)` and the
leaf's `skip`: a packet traversal of a wide node -- one node for the gang,
a child visited when any lane's query reaches its box, by the lanes that
reach it, the children's order decided once for the gang by the majority
of the lanes that are on (the user's decision over Embree's `any`), one
stack of nodes, per-lane bounds and masks. Until this morning no wide node
could be packet-traced: the driver applied every `sort` before any other
directive whatever the schedule wrote, so the sort's chain was built on the
scalar function and vectorized lane by lane, and the vectorizer refused a
run with a condition per call outright. Both are fixed at the root (the
compiler commit says how), and the directives are applied in the order
written.

Correct: on the whale (side 128, 64 walks, cpu 11) every walk ends on
both sides with the same mean length as the matching schedule's (11.46)
and the same rms error (5.63e-3), with and without the sort, and the new
tests check a gang's lanes against the scalar trace on the walls scene
(correctness/cpp/bvh4_child_volumes_packet).

Slow: least of 3, pinned, against the matching schedule's 1.82 Msteps/s
(1.20x WoSX's 1.52):

| gang of walks | bonsai Msteps/s | over WoSX |
|---|---|---|
| 32 (the schedule) | 0.21 | 0.14x |
| 16 | 0.25 | 0.17x |
| 8 | 0.26 | 0.17x |
| unsorted, 32 | 0.06 | 0.04x |

Why (perf record, cpu 11, the whale): 87% of the cycles are in the gang's
traversal, and the hot instructions are the leaf's triangle test -- the
closest point to a triangle computed 32 lanes wide (`vsqrtps`, `vdivps`,
the Voronoi-region compares on zmm registers), once per triangle of every
leaf any lane reaches. The matching schedule tests a leaf's sixteen
triangles at once for one query and never wastes a lane; the packet tests
each triangle for 32 queries at once and wastes every lane whose walk is
elsewhere. A point's 32 walks start at the same point, so the first step's
query is one query for the gang, but a walk's steps are a random path and
after a few steps the 32 are spread over the mesh: the leaves a gang
reaches are the union of 32 walks' leaves, each tested 32 wide for a lane
or two. Taking the slowdown at face value, about two lanes are useful at a
leaf on average. That is the thinning Embree's hybrid kernel switches away
from (bvh_intersector_hybrid.cpp: below 3 of 4, 5 of 8, 14 of 16 rays
active it traverses the rest one ray at a time), and the walks are less
coherent than a camera's rays from the second step on. The narrower gangs
help little because the leaf's cost is per triangle per lane either way.
The unsorted packet is three times slower again: without the nearest-first
order the bound rarely culls a popped node, and the gang visits most of
the tree.

What would make a packet pay here, in order of how much it is:
1. a leaf test that is per triangle *and* per lane only for the lanes on
   -- the scalar-per-lane leaf (the matching schedule's sixteen-wide test
   over the triangles) entered per active lane when few are on, which is
   Embree's hybrid switch: a schedule point, not a directive today;
2. coherence from the problem: a gang of 32 *points'* first steps rather
   than one point's 32 walks (the first query of neighbouring sample
   points is coherent, and WoSX's own `--shuffle` experiment showed how
   much neighbouring points share), i.e. `solve_all.split(i, ...)
   .vectorize(i_lane)` over the points with the walks inside -- a
   different schedule of the same program, worth measuring next;
3. Embree's per-ray-distance stack entry in place of the mask, so a lane
   whose best improved after a push does not enter the popped node.
None of these is started; the compiler change stands on its own and the
schedule is the first packet traversal of a wide node to run.

### The packet over the points, and WoSX's default binary tree (2026-10-09, late morning)

The user's next two asks, through the scheduling language alone. Whale,
side 128, 64 walks, cpu 11, least of 3; every row agrees with WoSX (all
walks end, the same mean length, the same rms error).

| tree | schedule | WoSX Msteps/s | bonsai Msteps/s | over WoSX |
|---|---|---|---|---|
| four-wide Mbvh (vectorized FCPW) | fcpw4w16, matching | 1.52 | 1.82 | 1.20x |
| | fcpw4w16-packet, a point's 32 walks | 1.52 | 0.21 | 0.14x |
| | fcpw4w16-packet-points, 32 points | 1.53 | 0.16 | 0.10x |
| binary BVH (WoSX's default, FCPW scalar) | fcpwb, matching | 0.65 | 0.08 | 0.12x |
| | fcpwb-packet, a point's 32 walks | 0.65 | 0.07 | 0.10x |
| | fcpwb-packet-points, 32 points | 0.65 | 0.06 | 0.09x |

**The packet over the points** (`solve_all.split(i, i_gang, i_lane, 32,
true).vectorize(i_lane)`, a lane's 64 walks a loop inside) is slower than
the packet over a point's walks, not faster: the lanes' first queries are
neighbours, but a lane runs its walks one after another, so from the second
step on the gang's 32 queries are 32 unrelated walks at unrelated stages,
and a lane whose walk ended early starts its next walk while the others
are mid-walk. The coherence the first step has is one step in eleven.

**WoSX's default tree.** WoSX as shipped runs FCPW with FCPW_USE_ENOKI off:
`Bvh<3, BvhNode<3>, Triangle>`, a binary tree whose node is its own box
(pMin, pMax, 32 bytes with the child offset and the reference count), its
leaves at most four triangles read through indices into the polygon soup,
and a traversal that tests both children's boxes at the parent and
descends into the nearer first (bvh.inl, processSubtreeForClosestPoint;
the box's far distance also tightens the search radius). WoSX itself runs
2.3x slower on it than on the vectorized tree (0.65 against 1.52
Msteps/s). `schedules/layouts/fcpwb.bonsai` declares it in FCPW's node
bytes (checked by static asserts in `fcpw_bvh.h`, which copies the rows
and materializes the triangles out of the soup in the build's order),
and compare.sh builds the driver with FCPW scalar for the `fcpwb*`
schedules, so WoSX's side runs exactly as it ships.

What the language cannot say about it: a node bounding its children
(`with AABB(lo, hi) on children`, the four-wide tree's declaration) needs
the children's boxes as the node's fields, and FCPW's binary node holds
only its own box; a layout field cannot read another row (`Nodes[index +
1u].pMin` is "unknown var: Nodes"). So the tree is declared bounding
itself, each child's box is tested as the child is entered, and the
children are visited left then right with no key to order them. On a
closest-point query that costs dearly: with no nearest-first order the
bound stays loose until a leaf happens to be near, and the traversal
visits far more of the tree than FCPW's, which is the 0.12x of the
matching schedule -- the first matching schedule in this app that is far
behind its reference, and the reason is a layout-language gap, not the
code the compiler makes of the traversal. The packets over it inherit
that and add the leaf-lane waste of the four-wide case; the binary leaf
of at most four triangles makes the per-lane leaf test cheaper, which is
why the packets lose less against the matching schedule here (0.07 and
0.06 against 0.08) than on the wide tree (0.21 against 1.82).

**To put to the user.** The one extension that would let the binary tree
be declared as FCPW traverses it is a layout field that reads a row of a
group by index -- `Nodes[index + 1u].pMin`, a load through a reference
the lowering already makes for `Interior from Nodes[...]` -- so that an
interior node's `lo`/`hi` can be its two children's boxes and the tree
declared `with AABB(lo, hi) on children` with the sort key `distmin(q,
AABB{lo[i], hi[i]})`, FCPW's order exactly. Not started: it is a new form
in the layout language, and the user decides those.

## 2026-10-21: the GPU comparison

The user's ask: "extend the wosx comparison to also perform a GPU
comparison. We need a) a matching schedule to their schedule and b) an
optimized schedule, which I suspect involves applying persistent threads
at least, so long walks don't take an entire warp with them."

### What WoSX's GPU solver is

Read from deps/wosx/include/wosx/gpu and deps/wosx/deps/fcpw/include/
fcpw/gpu (Slang, run through slang-rhi; the demos ask for the CUDA
backend, demo_apps/basic_2d/gpu_demo.cpp):

- **The tree.** FCPW's *binary* BVH: `GPUScene::transferToGPU` takes the
  non-vectorized `Bvh` and nothing else, so WoSX's GPU boundary handler
  builds with vectorization off and copies the rows (GPUBvhNode: the
  node's own box, nPrimitives, offset; 32 bytes) and the triangles
  (GPUTriangle: the three corners and the index, 40 bytes, out of the
  polygon soup). The traversal (bvh.slang, `Bvh::findClosestPoint`) is the
  CPU scalar one's order (bvh.inl, processSubtreeForClosestPoint): a stack
  of 64; pop a node and skip it if its box is now farther than the best;
  at a leaf test the triangles one by one; at an interior node test BOTH
  children's boxes, lowering the radius to each box's far corner, and push
  the farther then the nearer. Not the four-wide Mbvh the CPU comparison
  runs on both sides.
- **The kernel, by default.** `runPersistentPointEstimator`
  (entrypoints/common-shaders/point-estimation.cs.slang), chosen whenever
  the backend is CUDA and `disablePersistentThreads` is not set: 131072
  resident threads (`nResidentThreads`, the demos' default) in groups of
  256, each claiming ONE WALK at a time from a 64-bit global counter until
  the claims pass `points * walks`, the work item naming the point as
  `work % points` -- consecutive claims are consecutive points -- and the
  walk as the rest; one walk's statistics merged into the point's under a
  per-point spinlock (compare-exchange, two device barriers, exchange).
  So the user's suspicion is WoSX's own design: their default GPU
  schedule *is* persistent threads at the grain of a walk, and a warp
  whose lane finishes a short walk refills that lane with the next claim
  while a long walk runs beside it.
- **The other kernel.** `runPointEstimator`: one thread per copy of a
  sample point, the points copied `ceil(131072 / points)` times (capped at
  the walk count) so the card is full, each copy running one walk per
  dispatch over `walks / copies` dispatches, the copies merged afterwards
  (`computeWorkloadParameters`; it can only round the walks per point up).
- **Where it differs from WoSX's CPU walk.** The GPU walk bounds each
  step's query by `1.001 * 2 * dist` -- the walk stepped `dist`, so the
  next distance is at most twice that -- where the CPU asks with an
  infinite radius (walk_on_spheres.h:304 calls the two-argument
  `computeDistToAbsorbingBoundary`, whose FCPW implementation makes a
  `BoundingSphere(x, maxFloat)`); pcg32 one per thread, seeded from the
  clock with the thread as the stream, where the CPU has one per point.
  The arithmetic is the CPU scalar build's: the triangle test divides
  (`d1 / (d1 - d3)`, geometry.slang:297 and triangles.inl:420 alike) where
  FCPW's *vectorized* build, the CPU comparison's other tree, uses enoki's
  reciprocal with one Newton step. slang-rhi compiles with Slang's default
  floating-point mode (no fast math), and so is our GPU build compiled
  exact. The section "The baseline made exact" below says how the program
  follows each of these.

### The matching schedule, by directives alone

The program is unchanged. `schedules/fcpwb-gpu.bonsai`, over FCPW's tree
in FCPW's bytes (the next section):

    solve_all.reorder(i, w);
    solve_all.collapse(w, i, k);
    solve_all.persistent(k, t, 131072);
    solve_all.split(t, t_blk, t_thr, 256, true);
    solve_all.bind(t_blk, GPUBlock);
    solve_all.bind(t_thr, GPUThread);

`reorder(i, w)` puts the walks outside the points and, the nest not being
perfect, distributes what a point's body does around its walks -- the
three sums it starts, the mean it takes -- into `i!prologue` and
`i!epilogue` loops, privatizing the sums into a slot per point
(SSA/ReorderLoops.h, as written for pbrt's pixel loop). `collapse(w, i,
k)` makes the perfect nest one loop over every (walk, point) pair, the
point fastest, which is WoSX's `work % points`. `persistent(k, t,
131072)` is Aila and Laine's scheme at WoSX's count and grain, and the
split and binds are its groups of 256. The walk's adds to its point's
sums become three device-scope atomic adds into the privatized slots (the
PTX: `atom.global.add.f32` and two `.u32`, beside the counter's two claim
sites); WoSX's spinlock-and-merge is the same accumulation by other means.
The two distributed loops over the points run on the host around the
launch, where a `parfor` left unbound runs; the generated header's sides
table puts the points, the distances and the tree on the device and the
three outputs on the host, and the driver stages them so.

**One compiler change** (src/SSA/Convert.cpp, `nested_loop`): after a
reorder the schedule's `i` is three loops -- the nest's, the prologue's
and the epilogue's, variants in the sense of `loop_matches` -- and
`collapse(w, i, k)` resolved `i` function-wide and refused the ambiguity;
`reorder(w, i)` written a second time (fcpwb2-gpu-walks, to put the loops
back in the program's order once distributed) hit the same. A directive on
a nested pair now resolves its second loop by position: of the loops the
name matches in the anchor's function, with the anchor's variant suffix
first and bare second, the one nested with the anchor (inside it for
collapse; either way round for reorder). A variant of the first loop with
no such partner -- a distributed loop beside the nest -- is passed over,
and a directive that finds no pair at all is refused. Tests:
ssa/reorder-collapse (both orders), correctness/llvm/reorder-collapse,
correctness/gpu/reorder-collapse-persistent (the whole chain, on the GPU).

### The tree: FCPW's bytes, FCPW's order

WoSX's GPU tree is the binary one, whose byte-for-byte layout
(layouts/fcpwb.bonsai) the 2026-10-09 section could only traverse left
then right -- a layout field, it said, cannot read a child row's box, so a
node cannot bound its children and no key can order them, 0.12x of WoSX's
CPU. That was wrong about the language: the user pointed at the PLDI 2026
paper's closest-point benchmark (apps/queries/cp/cp.bonsai on the branch
ajr/pldi-benchmarks), whose layout has the parent *logically* hold its
children's boxes, read out of the children's rows with `this[left].low`
-- `this` being the group, `this[row]` one of its rows -- and the device
is in this branch's parser and layout lowering still (Parser.cpp adds
`this` to the layout's scope; Lower/Layouts.cpp binds it to the group
inside a Materialize). Two forms were tried. The paper's -- the node
bounding itself, the children's boxes as four extra Interior fields and
the sort key choosing between them -- fails this branch's layout validator
(ValidateLayout.cpp: no path for the Interior arm). The `on children` form
works: the tree declared as the four-wide one is, `Interior(children :
array[BVH2, 2], lo hi : vector[vec3f, 2]) with AABB(lo, hi) on children`,
and the layout's Interior arm building `children` from `index + 1u` and
`index + offset` and `lo`, `hi` from `this[left].low[0]`,
`this[right].low[0]`, ... component by component (a `vector[vec3f, 2]` is
built from three two-lane vectors, and a field's `.x` does not lower where
its `[0]` does). So FCPW's rows are read as FCPW reads them -- the node's
row, then its two children's rows -- the children tested at the parent,
the radius lowered to each box's far corner, the nearer entered first,
and the bytes are FCPW's GPUBvhNode to the byte. The whale, one thread,
WoSX's scalar solver against `fcpwb` (side 128, 64 walks, least of 3):
0.54 against 0.53 Msteps/s, 0.99x -- from 0.12x on the left-then-right
layout. The same change goes to `fcpwb-packet` and `fcpwb-packet-points`
(a sort, voted by the gang's majority) and adds `fcpwb-threads` and
`fcpwb-persistent`, so the CPU table runs the three schedules on both
trees; the user asked that every schedule on this tree carry the fix and
that both CPU tables (this app's and apps/rtq's) be measured fresh.

`layouts/fcpwb2.bonsai` is the same tree in another layout, kept as a
variant: each interior row holds its two children's boxes component by
component and their two rows, 56 bytes, built by the driver
(`fcpw_bvh2.h`) from FCPW's rows at copy time, so a visit reads one row
where FCPW's layout reads three. `schedules/fcpwb2.bonsai` is its CPU
schedule, `fcpwb2-gpu` the GPU kernel over it.

### The baseline made exact (2026-10-22, evening)

The user, on the first whale numbers: "we must be comparing equivalent
algorithms as well! We need exactly-matching algorithm/layout/schedule for
the baseline", and three things to fix in the language. All four are in.

1. **`this[left].low.x` lowers** (Lower/Layouts.cpp, fill): a layout's
   materializations never pass through Lower/Canonicalize.cpp, which turns
   a function body's `p.x` into an extract of the lane, so the fill makes
   the same rewrite when the read has its vector type, and converts the
   packed stored vector to the unpacked kind as the row's own fields are
   converted. The two-lane boxes of layouts/fcpwb.bonsai are written with
   `.x`, `.y`, `.z` now.
2. **The paper's form validates** (Parser.cpp, parse_fields_expr): a
   field read through `this[row]` is typed as the field the group declared
   under that name, which is in scope where the arm reads it, so the arm
   has a path IR/ValidateLayout.cpp matches -- the node bounding itself
   with the children's boxes as extra Interior fields, as
   apps/queries/cp/cp.bonsai wrote it, compiles (tests/bonsai/ssa/
   layout-row-read.bonsai). The `on children` form stays the one the
   schedules use, since it is what gives the children's test at the parent
   and the far-corner tightening.
3. **FCPW's 40-byte triangle rows** (IR/Layout.cpp, IR/Expr.cpp): a tile
   may hold one element -- the structure of arrays degenerating to an
   array of structures -- so `indirect group prims[pCount] { group[1] { pa
   pb pc primitiveIndex; v0 = pa; e1 = pa - pb; e2 = pc - pa; ... } }` is
   GPUTriangle row for row (fcpw/gpu/bvh_interop_structures.h,
   extractTriangles' bytes), the program's edges derived at each read as
   FCPW's own kernel computes `pb - pa` and `pc - pa` at each test (`-(pa -
   pb)` is `pb - pa` to the bit). Both binary layouts store the triangles
   so; the drivers (fcpw_bvh.h, fcpw_bvh2.h) write the rows as FCPW's GPU
   transfer does and check them by size and offset. What the two triangle
   types were: apps/rtq's `Triangle` element is Embree's slot -- a vertex,
   two precomputed edges, two ids, 44 bytes -- and a plain `array[Triangle]`
   stores exactly that; FCPW's GPU row is three corners and an index, 40
   bytes; the layout language could derive one from the other only inside a
   tile of two or more, hence the restriction lifted here. Tests:
   tests/bonsai/ssa/tiled-elements-one.bonsai; an LLVM golden of the same
   was dropped because the test runner's output differed between a ctest
   run and a manual one (two value numbers, `%156` against `%158`) -- a
   nondeterminism of the LLVM path to look into, not of this change.
4. **The algorithm.** Two differences from the reference's own code were
   in the program and the metric, and both are gone.
   - *The query's radius.* WoSX's GPU walk bounds each step's query by
     `RADIUS_EXPAND_PERCENTAGE * 2 * dist` (walk-on-spheres.slang) and
     FCPW's sphere begins at that radius squared; its CPU walk asks with
     no bound (walk_on_spheres.h:304; fcpw_geometric_queries.h:1068,
     `BoundingSphere(queryPt, maxFloat)`). The program asked with no bound
     on both. Now the query is FCPW's: a `Sphere` element (apps/rtq/
     elements.bonsai, FCPW's BoundingSphere, a center and a squared
     radius), `closest_within(s : Sphere)` in wosx.bonsai -- the argmin
     over `distmin(s, t)` -- and the metric's Sphere forms: a box whose
     `d2Min` is past `s.r2` has an infinite key, which never beats the
     running best, so the traversal does not enter it (FCPW's `overlap`
     returning false); a triangle past it is no answer (FCPW's `d * d <=
     s.r2`); the far corner tightens the best as before. The walk sets the
     radius from `Problem.query_bound`, the factor of the step just taken,
     which the driver sets after the reference it runs against: 2.002
     under --gpu, infinite otherwise (--query-bound overrides), and the
     terminal projection's sphere is infinite on both solvers, as WoSX's
     is. Both references compare `d2Min <= r2` at every box and `d * d <=
     r2` at every triangle, so the compares are theirs too; with the radius
     infinite they are what FCPW's CPU does with maxFloat.
   - *The arithmetic.* apps/rtq/metrics/fcpw.bonsai is FCPW's vectorized
     build (enoki's `rcp` estimate and Newton step for every division, one
     root per packet), the build the four-wide tree is compared against.
     The scalar build WoSX ships, and FCPW's GPU kernels, divide:
     metrics/fcpw_scalar.bonsai is that arithmetic -- `d1 / (d1 - d3)`,
     `1.0 / det`, `1.0 / (va + vb + vc)`, a root from the region returned
     -- line for line otherwise the same, and compare.sh compiles it
     before the program for every `fcpwb*` schedule, metrics/fcpw.bonsai
     for the `fcpw4w16*` ones, as apps/rtq compiles a metric before
     rtq.bonsai. The program no longer imports a metric.
   Every schedule names `closest_within` now (the walk's one query; the
   old `closest` is apps/rtq's and unused), with `s : Sphere` in the sort
   key. What remains different from WoSX and cannot be matched: the
   generators (WoSX's are seeded from the clock, ours from --seed), the
   partition of the points over threads (TBB's against the schedule's),
   and WoSX's GPU merge of a walk into its point's statistics under a
   spinlock where the program adds atomically.

   The whale, side 128, 64 walks, least of 3, the baseline exact: the
   four-wide tree one thread against one thread 1.15x (1.77 against 1.54
   Msteps/s; the compares against the radius are what FCPW does, and they
   cost the 1.20x of 2026-10-08 a few percent); WoSX's default tree one
   thread against one thread 0.83x (0.54 against 0.65); and the GPU
   matching schedule 1.14x (210 against 185 Msteps/s). Every count agrees
   on every cell.

   **The binary tree's 0.83x, and what it is not.** The 0.99x measured
   earlier the same evening on this cell (scratch/gpu/whale-fcpwb-cpu.log)
   was not lost to the exactness changes: the program took 6.65 s there and
   6.59 s here, while WoSX's own run took 6.62 s there and 5.45 s here --
   the reference varied by a fifth between two runs an hour apart, the
   program by one percent, so the earlier cell was a slow run of the
   reference's (its other runs that day, 0.65 Msteps/s, agree with this
   one). The gap that remains is the program's, and its machine code
   says where (scratch/gpu/fcpwb-exact.{bir,ll,dis} beside FCPW's scalar
   query compiled with the driver's flags, fcpw_scalar_ref.{cpp,rdis}):
   the leaf is cheaper than FCPW's -- its triangle test is inlined with
   the same seven branches and four divisions, where FCPW calls
   findClosestPointTriangle out of line per triangle and constructs an
   Interaction -- but the interior visit is dearer. The sort over the two
   children (SSA/SortRecursion.cpp's peel chain at width two) writes each
   child's test as a byte on the stack (`seta 0x6(%rsp)`, `seta
   0x7(%rsp)`) and reads both back as one word (`movzwl 0x6(%rsp)`), which
   the store buffer cannot forward -- a stall of ten cycles and more on
   every interior node -- then stores the keys and carried bounds as two
   small arrays and loads them back by lane index, so the next node's
   address waits on a chain of selects and memory round trips, where
   FCPW's loop compares in registers and branches, letting the core
   speculate the next node's loads before the compares resolve. To be
   confirmed with the store-forwarding counter (`perf stat -e
   ld_blocks.store_forward`) once the machine is free; the fix is in the
   compiler -- a sort of two is one comparator, and a lane's test belongs
   in a register mask, not a byte -- and is the user's to call.

### The user's three rulings on it (2026-10-22 night, 2026-10-23 morning)

1. **"A mask should be bits in a scalar register."** A new SSA pass,
   SSA/PackBoolArrays.cpp, holds every small local array of booleans
   (`mut bool[N]`, N at most 64) as the bits of one word: an element
   written is its bit set or cleared, an element read is its bit tested,
   and the array read whole -- the mask the sort and the loopified descent
   peel -- is the word's low N bits reinterpreted, which the peel turns
   straight back into the word (MaskLanes) and the backend folds to
   nothing. It runs after the directives (a vectorized children's loop has
   made its array a vector by then) and before the allocations are
   promoted; it leaves alone an array whose address escapes, one written
   in a bound parallel loop from outside it, and one whose elements are a
   gang's vectors (the packet traversal's per-lane tests, a different
   shape). A two-lane mask is two bits now like a four-lane one is four
   (MaskLanes and Simplify accept n = 2; runtime/u2.h for the C++
   backend). On fcpwb the stall is gone from the machine code -- no byte
   `seta` to the stack, no `movzwl` of two of them, six `tzcnt` where the
   two-lane mask used to be a select over the lane ramp and a vector
   minimum -- 1131 instructions to 1106, 114 branches to 108. Tests:
   ssa/child-volumes-sorted-two, backends/llvm/child-volumes-sorted-two,
   correctness/llvm/bool-array-word; thirty-two goldens of scalar
   traversals changed by the word's shifts and ors and nothing else,
   checked with the value numbers normalized. One bug on the way: the
   type test was `is_bool()`, which is true of a vector of booleans too,
   and packed a packet schedule's `boolx32[2]` masks; it is `is<Bool_t>()`.
2. **"Tightening unconditionally."** Lower/Trees.cpp lowers the
   argmin's accumulator by every child's far corner, tested in or out
   (FCPW's `s.r2 = min(s.r2, d2Max)` on both children). It is sound
   because a child tested out reaches no nearer than its bound, which is
   no nearer than the best, or infinitely far where the volume holds
   nothing the metric reaches -- every metric in apps/rtq answers `inf`
   for a ray's `distmax` against a box it misses. The branch per child is
   gone; the filtered extremum, where the bound would not hold, was and
   is left alone (Carry::reach_of unset). lower/argmin-tighten-children
   and lower/argmin-over-map show the `if` gone.
3. **"Match the sort key per platform."** FCPW has three rules. Its CPU
   traversal (bvh.inl) orders the two children by d2Min, and where both
   contain the point -- both d2Min zero -- by d2Max, the nearer far corner
   first; its GPU kernel (bvh.slang) by d2Min alone; and its four-wide
   Mbvh (mbvh.inl, sortOrder4) by d2Min through a five-comparator network
   whose order among tied children is the network's, not any key's. The
   CPU binary schedules (fcpwb, fcpwb2, fcpwb-packet, fcpwb-packet-points,
   and the threads and persistent ones that import fcpwb) carry the CPU
   rule as one key -- the key sees both children's boxes, so it is exact
   and division-free:
   `select(distmin(s, b0) == 0 && distmin(s, b1) == 0, distmax(s, b_i),
   distmin(s, b_i))`, a tie keeping the left child as FCPW's strict
   compares do. The GPU schedules take their traversal from
   schedules/kernel/fcpwb.bonsai and kernel/fcpwb2.bonsai, the d2Min key
   (an import path cannot carry a hyphen, hence the directory). The
   four-wide schedules keep d2Min: among children that both contain the
   point FCPW's network visits, for two tied lanes 0 and 1, lane 1 first;
   for three tied, 2, 0, 1; no per-child key reproduces that, and a
   lowering of the network itself would be a new scheduling construct --
   left for the user. Still unmatched on the GPU: FCPW's kernel pushes the
   node index alone and re-tests the popped node's box against the radius,
   where our kernels carry the bound with the index and test that.

   **Measured** (2026-10-23/24, the three tables rerun under the compiler
   with both fixes, 688a8802; the before-tables under the old compiler
   with the exact baseline, kept in scratch/before-1023/). WoSX's and
   Embree's own rates agree run to run within a percent on every column,
   so every movement below is ours. Geomeans over the eighteen plotted
   meshes:

   | CPU schedule | before | after |
   |---|---|---|
   | `fcpwb` (matching, sequential) | 0.798 | 0.963 |
   | `fcpwb-threads` | 0.914 | 1.097 |
   | `fcpwb-persistent` | 0.946 | 1.139 |
   | `fcpw4w16` | 1.187 | 1.174 |
   | `fcpw4w16-threads` | 1.229 | 1.219 |
   | `fcpw4w16-persistent` | 1.314 | 1.299 |

   The binary tree's three schedules gained twenty percent each, every
   one of the nineteen cells (the comb 0.55 -> 0.73, the whale 0.83 ->
   0.99, the pavilion 0.63 -> 0.84): the 0.83x is 0.963, behind WoSX
   still on seven meshes (comb 0.73, pavilion 0.84, crown 0.93, head
   0.94, car and sportscar 0.95, bmw 0.96), level or ahead on the eleven
   others, and the threaded and persistent forms are ahead on every mesh
   but the comb. What remains of the gap is the two differences the
   section above leaves unmatched (the pop's re-test, the four-wide
   network's tie order do not apply here; the binary CPU walk's remaining
   difference is read from the machine below, "The binary CPU walk's gap",
   2026-10-24 afternoon: the generated two-child visit). The four-wide
   tree's three schedules are each one percent down, every cell by one
   to two percent, bonsai's rate alone. The rtq table (apps/rtq) moved
   the same way: its closest-point rows on the four-wide tree, `fcpw4w16`
   near 1.189 -> 1.176 and volume 1.207 -> 1.193, where its ray rows
   gained (primary 1.506 -> 1.531, diffuse 1.634 -> 1.642, ao 1.869 ->
   1.885) and Embree's five matching rows stayed within noise (nearest
   primary 1.005, diffuse 1.047, any ao 1.010 -> 1.023, closest near
   2.362 -> 2.356, volume 2.994 -> 2.976; tuned rows carried over). The
   unconditional tightening is the one change that touches a
   closest-point traversal and not a ray's, so it is the likely cost on
   the wide tree -- four far corners folded in a visit where the entry
   test used to fold one or two -- to be confirmed in the generated code
   before it is acted on. The GPU table is in its own section below
   (the matching kernel 1.099 -> 1.161).

### The variants, for (b)

Each departs from the matching schedule in one thing:

- `fcpwb-gpu-parallel`: no persistent workers; every (walk, point) pair a
  thread of one launch in blocks of 256, the hardware's block scheduler
  doing the balancing. What the persistent claim is worth here.
- `fcpwb-gpu-reorder`: the walk fastest -- a warp's claims are one point's
  thirty-two walks, which start at the same place -- against WoSX's
  consecutive points. Coherence.
- `fcpwb-gpu-point`: a point a claim, its walks in sequence on one
  thread (the CPU persistent schedule's grain). What the grain costs.
- `fcpwb-gpu-256k`: the matching kernel with 262144 workers where WoSX's
  has 131072 -- more than the card holds, so the hardware fills its
  residency and the late blocks find the counter spent and leave. From
  the exploration below.
- `fcpwb-gpu-reorder-256k`: the walk fastest with those workers. From
  the exploration below.
- `fcpwb-gpu-parallel-reorder-blk64`: the flat launch with the walk
  fastest, in blocks of 64, so that a mesh of few points still reaches
  every multiprocessor. From the exploration below.

The figure's legend names them persistent, parallel, persistent-reorder,
persistent-point, persistent-256k, persistent-reorder-256k and
parallel-reorder-blk64 (the user, 2026-10-24: names as short as the pbrt
figures', and schedules only). Two more schedules are measured on request
but are a sweep of their own, not the figure's, since they change the
layout or the tree rather than the schedule: `fcpwb2-gpu`, the matching
kernel over the layout that stores the children's boxes in the parent
(one row read a visit where FCPW's layout reads three), and
`fcpw4w16-gpu`, the matching kernel over FCPW's four-wide tree in FCPW's
bytes, which FCPW cannot put on a GPU (four boxes a node, sixteen
triangles a leaf, half the depth).

### The reference's side, and the harness

wosx_hook.cpp under `WOSX_GPU` (compare.sh builds it so for a `*-gpu*`
schedule, against slang-rhi built once by build_slang_rhi.sh into
deps/slang-rhi-build, which fetches FCPW's pinned Slang release): WoSX's
GPUTaskHandle on the CUDA backend, its GPU boundary handler over the same
positions and indices (the same deterministic build the CPU handler ran,
so the same tree the layout holds), GPUGeometricQueries over the same box,
the PDE in gpu/wosx-pde-definition.slang (the Laplace problem with
`x^2 - y^2`, in the module shape WoSX asks for), GPUWalkOnSpheresSolver
persistent (`--gpu-reference persistent`, the default) or threads. The
clock on both sides is around the solve alone -- WoSX's dispatch and the
wait for it; our launch, its captures and the wait -- after the
statistics and generators are reset off the clock; least of three after a
warm-up, as the CPU table. WoSX's GPU statistics keep no variance, so the
bias test `z` against them is the paired statistic over the points. The
GPU runs are pinned like single-thread runs (the host thread), both sides
on the card; benchmark.sh --gpu waits for an empty GPU and honours
/tmp/benchmark.lock, the sessions' claim file.

### Measured

The table over the nineteen meshes (side 128, 64 walks a point, least of
three), twice: 2026-10-22 22:16 under the old compiler (875c374a with the
exact baseline uncommitted) and 2026-10-24 00:04-00:25 under the compiler
after the two fixes above (688a8802; the bit-word pass b48a0dde and the
unconditional tightening, plus bonsai-f9's device sort, which a program
without a sorted GPU queue never reaches). WoSX's own rates agree run to
run within a percent on every schedule's column, so the movement is
ours. Geomeans over the eighteen plotted meshes (the landscape left out,
as in the plot), before -> after, with the whale's cell:

| schedule | before | after | whale | after's range |
|---|---|---|---|---|
| `fcpwb-gpu` (matching) | 1.099 | 1.161 | 1.12 -> 1.21 | 0.94 (beast) .. 1.27 (dragon) |
| `fcpwb-gpu-parallel` | 1.042 | 1.094 | 1.09 -> 1.17 | 0.90 (zero-day) .. 1.17 |
| `fcpwb-gpu-reorder` | 1.180 | 1.257 | 1.19 -> 1.27 | 0.98 (beast) .. 1.49 (ganesha) |
| `fcpwb-gpu-point` | 0.094 | 0.102 | 0.07 -> 0.07 | 0.03 .. 0.17 |
| `fcpwb2-gpu` | 1.051 | 1.067 | 1.09 -> 1.16 | 0.84 (beast) .. 1.19 (ivy) |
| `fcpw4w16-gpu` | 0.946 | 0.989 | 0.91 -> 0.94 | 0.72 (comb) .. 1.30 (lte-orb) |

What the table says. The matching kernel is faster than WoSX's
persistent kernel on seventeen of eighteen meshes, level on the crown
(1.00) and behind on the beast (0.94); those two are where every one of
our kernels does worst, and the table does not say why (their walks are
of middling length, 6.6 and 11.7 steps against the ivy's 22.6). The
walk-fastest grain is worth another eight percent over the matching
claim on every mesh but the crown (coherence of a warp's thirty-two
walks from one point against WoSX's consecutive points). The flat launch
gives up six percent to the persistent claim, so the claim is worth that
here. The layout that stores the children's boxes in the parent is
slower than FCPW's layout on this kernel (1.067 against 1.161) though
it reads one row a visit where FCPW's reads three; why, the PTX has to
say. A point a claim is a tenth of WoSX: the sixty-four walks of a point
in sequence on one thread is the CPU's grain, and the GPU has no use for
it. The four-wide tree, which FCPW cannot put on a GPU, is level with
WoSX's binary one (0.989): half the depth, but four boxes tested a visit
against the two the binary tree's sort tests, and sixteen triangles a
leaf.

The compiler's two fixes (the bit word for the sort's hit flags, and the
radius tightened by the far corner at every child) moved the three
schedules on FCPW's layout by five to seven percent, the four-wide tree
by five, and the wide-row layout by one and a half. Two of that layout's
cells went the other way: the beast, 0.90 -> 0.84 (bonsai 75.2 -> 72.6
Msteps/s), and the dragon, 1.10 -> 1.03; the other four schedules gained
on both meshes, so this is the wide-row kernel alone, to be read from
the PTX before anything is concluded.

### The schedule exploration (2026-10-24)

The user asked for at least two more GPU schedules that are the best
schedule on at least one mesh, found by a real exploration, profile-
guided where it helps. Two rounds on a few meshes, then the table.

**What the kernel's PTX says.** A kept build of `fcpwb-gpu` (compare.sh
with WOSX_KEEP, the PTX read out of wosx.out, `ptxas -v` for sm_120):
64 registers a thread, a 528-byte stack frame (the traversal's stack of
64), no spills. At 64 registers a multiprocessor holds 1024 threads, so
the RTX 5090's 170 multiprocessors hold 174080 of this kernel, and WoSX's
131072 resident workers fill three quarters of the card. That is the one
number the exploration turned on.

**Round 1** (crown, beast, whale, head, dragon; scratch/gpu-explore/
round1.csv; bonsai's rate in Msteps/s, the matching schedule's first):

| schedule | crown | beast | whale | head | dragon |
|---|---|---|---|---|---|
| persistent (131072 workers) | 69 | 82 | 222 | 528 | 40.8 |
| 65536 workers | 70 | 70 | 151 | 353 | 25.4 |
| 262144 workers | 68 | 84 | 242 | 577 | 46.1 |
| blocks of 128 | 68 | 83 | 219 | 524 | 40.4 |
| blocks of 1024 | 67 | 76 | 201 | 461 | 38.3 |
| four walks a claim (split, then persistent) | 24 | 31 | 149 | 426 | 34.6 |
| walk fastest, two walks a claim | 41 | 48 | 216 | 536 | 44.4 |
| walk fastest, four a claim | 25 | 27 | 138 | 467 | 37.7 |
| walk fastest, eight a claim | 14 | 15 | 80 | 344 | 29.4 |
| flat launch, walk fastest | 60 | 74 | 226 | 564 | 44.2 |

Twice the workers is worth nine to thirteen percent on every mesh with
enough walks to fill them, and costs nothing on the small ones; half the
workers costs a fifth. Blocks of 1024 lose because 131072 workers are
then 128 blocks, fewer than the 170 multiprocessors. Claiming several
walks at once is a disaster on a mesh of few points -- the crown's 29440
walks in claims of four are 7360 busy threads on a card of 174080 -- and
no better than one at a time on the large ones; `fcpwb-gpu-point`, a
claim of sixty-four, is the end of that curve.

**Round 2** (crown, beast, comb, octopus, whale, head, dragon;
round2.csv), combining the worker count with the walk-fastest order and
trying small blocks for the flat launch on the meshes of few points:

| schedule | crown | beast | comb | octopus | whale | head | dragon |
|---|---|---|---|---|---|---|---|
| persistent | 68 | 85 | 805 | 65 | 219 | 525 | 40.7 |
| 196608 workers | 70 | 84 | 811 | 72 | 239 | 581 | 45.9 |
| 262144 workers | 69 | 82 | 781 | 74 | 244 | 579 | 46.2 |
| walk fastest, 196608 | 69 | 77 | 861 | 73 | 249 | 618 | 52.5 |
| walk fastest, 262144 | 66 | 79 | 871 | 71 | 245 | 627 | 52.3 |
| flat, walk fastest, blocks of 256 | 60 | 74 | 1022 | 65 | 221 | 572 | 44.3 |
| flat, blocks of 64 | 70 | 78 | 880 | 71 | 220 | 545 | 42.5 |
| flat, blocks of 128 | 68 | 79 | 896 | 70 | 220 | 543 | 41.0 |
| flat, walk fastest, blocks of 64 | 65 | 78 | 1074 | 71 | 236 | 602 | 48.5 |

196608 and 262144 workers are the same within noise everywhere (both
fill the card); 262144 is kept, since it needs no knowledge of the card.
The walk-fastest order with the card full is the best schedule on the
whale, the head and the dragon by fifteen to twenty-eight percent over
the matching one. The flat launch with the walk fastest in blocks of 64
is the best on the comb by a third (its 143360 walks are 560 blocks of
256 for a card of 170 multiprocessors with six block slots each; of 64,
2240) and close behind on the large meshes. On the crown and the beast,
the two meshes with the fewest points (460 and 1344: fewer walks than
the card has threads), every schedule is within noise of WoSX's, and no
schedule finds anything there -- both sides are at the floor of a launch
whose threads mostly claim nothing. The losers' files were removed; the
three winners are `fcpwb-gpu-256k`, `fcpwb-gpu-reorder-256k` and
`fcpwb-gpu-parallel-reorder-blk64`.

**The table** (2026-10-24 10:26-10:42, every mesh, the seven schedules;
the compiler at 0751d3c2, which differs from 688a8802's only in the
sorted queue's key, nothing this program uses; WoSX's rates as before).
Geomeans over the eighteen plotted meshes, and where each is the best:

| schedule | geomean | least | greatest | best on |
|---|---|---|---|---|
| persistent (matching) | 1.172 | 0.99 (beast) | 1.28 (dragon) | -- |
| parallel | 1.094 | 0.89 (zero-day) | 1.18 (ivy) | -- |
| persistent-reorder | 1.258 | 0.94 (beast) | 1.49 (ganesha) | octopus (by our rate) |
| persistent-point | 0.100 | 0.03 | 0.17 | -- |
| persistent-256k | 1.270 | 0.96 (beast) | 1.44 (dragon) | beast |
| persistent-reorder-256k | 1.361 | 0.98 (beast) | 1.61 (ganesha) | the other fifteen |
| parallel-reorder-blk64 | 1.308 | 0.96 (beast) | 1.53 (ganesha, dragon) | comb |

The best schedule of the seven, `persistent-reorder-256k`, is WoSX's
kernel with two changes of the schedule and none of the program: a
warp's claims one point's walks, and the card full. The beast stays
the one mesh under WoSX on every schedule (0.94 to 0.99).

### The crown and the beast (2026-10-24, afternoon)

The user asked why the two gain nothing on the GPU. Profiled
(scratch/gpu-gap/: nsys timelines of both, an ncu report of the crown's
two kernels, a sweep of the walk count; the machine claimed from
bonsai-f9 for it):

- They are the two meshes with fewer walks than the kernel has threads:
  460 and 1344 sample points on the slice, so 29440 and 86016 walks
  against 131072 persistent threads (174080 resident slots). Every thread
  claims at most one walk, the persistent loop never claims a second, and
  the card runs nearly empty: on the crown ncu has WoSX's kernel at 15%
  achieved occupancy (7.4 warps a multiprocessor) with 2.9 of 32 lanes
  active per warp on average, ours at 25% with 2.6 lanes (the occupancy
  is the small meshes' symptom -- 32% and 49% on a full mesh -- but the
  lanes are not: both kernels run at about three of 32 everywhere, see
  'The lanes of a warp' below). The kernel's
  time is the longest walks' chain of dependent steps, not throughput:
  460 walks (one a point) take 2 ms, 29440 take 3 ms, 117760 take 4 ms,
  and only at 471040 (1024 a point) does the time grow with the work,
  10 ms (walks-sweep.txt).
- In that regime both kernels take the same time, since a step's latency
  is the same tree's dependent node loads on both sides (nsys: WoSX
  3.00 ms, ours 2.92 ms a launch on the crown; 12.21 and 12.09 ms on the
  beast), and nothing a schedule changes -- which walks a warp holds, how
  many threads, the block size -- matters while the multiprocessors wait
  on three lanes. Our kernel executes 21% more warp-instructions than
  WoSX's (706M against 582M on the crown) in 64 registers to its 82, and
  waits twice as long between issued instructions (14.5 cycles against
  7.2); none of it shows in the time.
- The bars are then noise about 1.0: the least of three of a 3 ms launch
  moves 5 to 10% run to run (the crown 1.04, 1.06 and 1.10 in three runs
  today, the beast 0.99 and 1.06).
- With the card full the two are like the others: at 1024 walks a point
  the matching schedule is 1.16x on the crown and 1.24x on the beast, and
  WoSX's own rate rises from 67 to 273 Msteps/s on the crown.

Whether the table should give every mesh enough walks to fill the card
(a walk count per mesh, or more sample points) is the user's call; its
64 walks a point are the demo's.

### The binary CPU walk's gap, read from the machine (2026-10-24, afternoon)

(Superseded 2026-10-25: the two-lane visit this section reads -- the
transposed boxes, the lanes picked through the stack, the integer keys --
was replaced by FCPW's own visit, 'The binary node's visit made FCPW's'
below. The 2026-10-26 table, the first clean one on the CPU-bytes layout,
has `fcpwb` one thread at 1.02 geomean over the nineteen meshes, comb 0.81,
pavilion 0.88, head 0.97, the rest 0.99 to 1.07; what is left is in the
closest-point query's generated code, apps/rtq/PLAN.md, 'What is left
between the sides, as code'.)

The user asked why `fcpwb`, one thread against one thread, is under WoSX
on the small meshes (comb 0.73, pavilion 0.84, crown 0.93, head 0.94;
level or ahead from a million triangles up). Read on the comb
(scratch/cpu-gap/: the kept wosx.ll and wosx.out, perf records of cycles
and of instructions, gdb hit counts of the traversal loops):

- It is instruction count, not stalls: the two sides retire at the same
  rate (our share of the process's cycles 57.2%, of its instructions
  57.3%, the process at 2.6 instructions a cycle), and ours executes 1.45x
  FCPW's instructions for the same queries (16.4G against 11.3G over the
  run), which is the 0.72x.
- Not more work: counting the traversal loops over one run (2240 walks,
  29k queries) has ours visiting 528702 nodes and testing 322898
  triangles where FCPW visits 639987 and tests 384941 -- 17% fewer. FCPW
  enters a child at `d2Min <= r2` and our argmin only on a strict
  improvement, and on the comb's flat plates a box's near distance equals
  the best found often (the closest point on a plate in the slice's plane,
  the plate's boxes bounded by that plane). The leaf costs the same per
  triangle on both sides (about 130 instructions). The difference is the
  interior node: FCPW's two-child visit is about 70 instructions
  straight-line (bvh.inl: two overlaps, two compares, two pushes and the
  nearer's pop), ours about 130 (65d30..65f52 of the kept binary and the
  pop), and half our visits are two-child visits (272238 pushes).
- Where the sixty go, read off the assembly: about fifteen shuffles
  assembling the two children's boxes as per-axis two-lane vectors from
  the children's rows (the layout's `lo = vector[vec3f, 2]{vector[f32, 2]
  {this[left].low.x, this[right].low.x}, ...}` lowered as gathers and
  transposes -- vpermt2ps, vinsertps, vunpckhps, vmovshdup -- where
  FCPW's distances come straight from the loaded scalars); about
  twenty-five choosing the entered children's keys and indices by a
  data-dependent lane through the stack (the sort's lowering stores the
  two-lane keys and carries to memory and reloads them at the lane tzcnt
  found, with k-mask moves and blends around it); about ten ordering the
  two keys as integers after a float-to-sortable-int conversion of each
  (sar, and, xor, then cmovs) where FCPW does one compare and a swap;
  about eight for the key's both-contain select, which FCPW evaluates
  only in its both-hit branch.
- Why the big meshes do not show it: their traversal waits on node loads
  from L2 and memory, which hides the arithmetic; the comb's 13 KB tree
  and 24 KB of triangles sit in L1, where instructions are the time. The
  ratio follows the tree size (pavilion 0.84 at 11k triangles, head 0.94
  at 17k, crown 0.93 at 155k, the cars 0.95, the million-triangle meshes
  1.02 to 1.04).

What would close it is codegen, not the schedule: the two-child sort as
a compare-and-swap in registers (the two-lane case of the sort's
lowering), the children's boxes computed per child from their rows
rather than transposed into two lanes (the lowering of a `vector[vec3f,
2]` built by component), and the keys compared as floats. Left for the
user to decide; the comb's 0.73 is the whole of it, and the threaded and
persistent forms are ahead on every mesh but the comb regardless.

### The lanes of a warp, and two deferral trials (2026-10-24, evening)

The user asked whether the schedules had moved since the table, and
whether a deferral -- the steps queued, the long walks grouped and the
live ones compacted -- would help. No schedule has been added since the
seven (the afternoon went to the two profiles above and the figures).
Measured first, ncu with metrics alone on both sides' persistent kernels
at the table's settings (scratch/gpu-gap/ncu-lanes.sh, {head,dragon}-ncu-
lanes.csv; 131072 threads, 64 walks a point, the lock held 17:50-17:53):

| | head, WoSX | head, ours | dragon, WoSX | dragon, ours |
|---|---|---|---|---|
| kernel time | 22.8 ms | 19.4 ms | 305 ms | 243 ms |
| registers a thread | 82 | 64 | 82 | 64 |
| achieved occupancy | 31.6% | 49.2% | 31.5% | 49.1% |
| lanes active per warp instruction, of 32 | 3.10 | 2.78 | 2.38 | 2.14 |
| issue slots busy | 46.5% | 64.8% | 31.0% | 48.7% |
| cycles between a warp's issues | 8.4 | 9.1 | 12.5 | 12.1 |
| warp instructions | 19.4 G | 22.9 G | 171 G | 210 G |
| L2 hit rate, DRAM of peak | 100%, 0.0% | 99.9%, 0.2% | 79.9%, 12.7% | 82.8%, 19.2% |

**Reading.** Fewer than one lane in ten does anything, on either side,
on a full card as on the crown: the crown note's "three lanes" is the
kernels' normal state, not the small meshes' symptom (corrected above).
Two divergences multiply. A warp claims 32 walks together and holds them
until the longest has ended -- the lanes whose walks are done wait at
the reconvergence after the walk loop, and WoSX's `while (true) {
InterlockedAdd; walk }` (point-estimation.cs.slang) has the same shape --
and within a step the lanes' traversals visit different numbers of
nodes. The first can only be estimated here: walk lengths with a mean
of 14 and a long tail (the cap is 1024 steps) have a greatest-of-32 of
about four times the mean, so about a quarter of the lanes hold a live
walk at a time; the traversal's divergence is then the remaining 2.5x,
the usual figure for incoherent traversal. Our lead over WoSX (1.18x on
the head, 1.25x on the dragon in these runs) is in the registers -- 64
against 82, so 49% occupancy against 32% -- and the issue rate, not in
the lanes, where we are a little worse (more warp instructions a step:
4.5 k thread-instructions a step against WoSX's 4.2 k on the head).

**What a deferral would do.** `walk.defer(walk, q)` makes the step the
drain's unit: each round runs one step of every walk still alive,
compacted, so the walk-length divergence goes -- every lane of every
warp holds a live walk until the last rounds -- and the traversal's
stays. The ceiling is about the 4x estimated above in warp instructions
where the issue slots set the pace (the head: 65% busy), less on the
dragon, where L2 misses do. The costs: an entry of about 40 bytes
written and read per step (position, distance, step count, the
generator's 16 bytes, the point) -- a few percent of a step's node
traffic -- and a launch per round, as many rounds as the longest walk
(up to the cap of 1024), the late rounds nearly empty: pbrt's guarded
launch over the capacity, about 10 us each, so up to 10 ms, which is
the comb's whole kernel (1.1 ms) and half the head's and nothing on the
dragon's. A `sort` of the queue by a Morton key of the position each
round would also keep a warp's walks on one part of the tree
(persistent-reorder's 7% is that coherence on the first step alone).

**Two trials, both refused by the compiler** (scratch/defer/):

- `fcpwb-gpu-defer.bonsai`, card-wide: the nest interchanged and
  collapsed as the reorder schedules have it, `steps =
  solve_all.queue(root)`, `solve_all.defer(walk, steps)`,
  `walk.defer(walk, steps)`, the pairs and the drain in blocks of 256.
  Refused by SSA/Defer.cpp:2504: after the call to walk "solve_all goes
  on to use total, an address (f32*) computed inside the producer's
  iteration that is not one of its mutable locals" -- `total` is the
  point's slot in the array `reorder(i, w)` privatized the sums into,
  and the pass keeps a continuation's value only if it is in scope at
  the pass, a local's slot, a loop index, or arithmetic on indices
  (`rematerializable`: Add, Sub, Mul, Shl, Cast). The fix at the root:
  the element address of storage that outlives the iteration (the slot
  array is the owner's, hoisted) is the index's arithmetic too, to be
  recomputed at the pass as the indices are. A small extension of what
  the pass accepts, for the user to say yes to.
- `fcpwb-gpu-defer-block.bonsai`, per block: a point's walks to one
  block, `steps = solve_all.queue(i)`, the walks and the drain bound to
  the block's threads, so that the sums stay the owner iteration's
  locals. Refused by CodeGen_PTX.cpp:335: "the loop over i, bound to
  GPUBlock, holds two loops bound to GPUThread (w and steps). A launch
  has one block size, so two thread loops would have to be fused into
  one, which is not built yet." By estimate it is the weaker form
  anyway: with 64 walks a point the tail of the longest runs one lane,
  about 37% of the lanes live against the card-wide form's near 100%.

A third form needs no queue: the persistent claim taken per lane
inside the step loop -- a lane whose walk ended claims the next walk
while the warp's other lanes step on -- compaction in registers, no
rounds, no entry traffic (Aila and Laine's persistent threads fetched
per warp; this is per lane). It is not a composition of today's
directives (`persistent` claims a whole walk) and would be a new mode
or primitive, to ask the user about before building. WoSX does neither,
so either form is a lead of the schedule alone, which is the table's
story.

### Which solver next (2026-10-24, the user's question)

The user asked whether the plain walk is too simple a solver to show
richer schedules on, and which others are worth beating -- walk on stars?
What deps/wosx has: WoS and WoSt (walk on stars, Neumann and Robin
boundaries) on both the CPU and the GPU through the same persistent
kernel; boundary value caching on both; the reverse walk (splatting
walks from the boundary onto evaluation points) on the CPU only. The
2D demo runs all four; the 3D demos (electrostatics, thermal conduction,
potential flow, geometric deformation) are WoS and WoSt.

- **WoSt first.** Its step is heterogeneous where WoS's is one query:
  the star radius is a closest-silhouette-point query on the Neumann
  boundary's tree (FCPW's SNCH, nodes carrying normal cones), the step
  is a ray cast against that boundary (the ray stops where it hits, the
  walk continues on the surface), the Neumann contribution samples a
  boundary point inside the star through the tree, and the Dirichlet
  distance is the closest-point query we have; Russian roulette and
  splitting on the throughput besides. That is the shape `stage` and
  per-kernel queues were built for -- pbrt's trace and shade queues, a
  queue per query kind with its own kernel -- and the ray cast can go
  to the RT cores as pbrt's does. Program work: the SNCH layout with
  cones, the silhouette query over edges, the boundary sampling
  descent, the Green's function terms; the ray query apps/rtq has.
- **The source term** (Poisson) is the cheap completion of WoS: one
  Green's-function sample in the ball per step, no new tree. More
  arithmetic a step; the schedule story does not change.
- **The reverse walk** is the workload for a branching deferral: its
  splitting pushes a variable number of walk states onto a std::queue
  and drains it per point (reverse_walk_on_stars.h), and its
  contributions scatter to many evaluation points through a tree over
  them. Linear deferral was the user's choice for now; this is what
  would motivate lifting it. CPU only in WoSX.
- **Boundary value caching** is two phases: walks from boundary
  samples (WoS or WoSt again), then an all-pairs boundary integral at
  the evaluation points -- a dense kernel over two sets, a layout and
  tiling story more than a traversal one.

### Both deferrals built, both measured on the head (2026-10-25, night)

The user's ruling on the two refusals (2026-10-25, 00:3x): fix the root
cause in the deferral, not a bandaid; fix codegen for adjacent thread
loops, since the drain's loop is a masked loop of the pusher's size; a
queue in shared memory must say so in the schedule (the queue
constructor's storage word); tests first, clean implementations, then
run; and a walk on stars benchmark beside this one. Both fixes are in.

**The deferral (commit 55c7be9b).** The pass classified a continuation's
free variable by what it is -- an address by its type, a recomputable
value by a whitelist of five opcodes -- where the questions are where its
storage lives and how it is computed. Now (src/SSA/Defer.cpp): an address
is kept across the drain when the storage it points into outlives the
pass -- a parameter's, a record slot's, an allocation's made before the
drain, which is the reorder's `total!expanded` (Tofte and Talpin's rule
that a pointer is captured only where its region encloses the capture);
and any pure instruction over the loops' indices, the values in scope and
the iteration's locals is recomputed at the pass (Briggs, Cooper and
Torczon: rematerialize what is cheaper to compute again than to spill),
an address whatever it costs -- kept, it would be a subscript the
contention analysis cannot see through -- and a value up to a budget of
sixteen instructions. The same rule for an address handed down the chain,
so a walk handed the point's privatized sum keeps the address in its
entry. Tests: ssa/defer-remat-address, ssa/defer-address-parameter, the
same under correctness/llvm, and correctness/gpu/defer-remat-address.

**The block body (this commit).** The PTX backend's model of a block loop's
body gained the one concept it lacked, block-level storage, and with it
the rest fell into place as Halide's FuseGPUThreadLoops has it: an
allocation in the block body outside the thread loops is one per block,
the block's shared memory (Allocate::Memory::Shared; a `.shared` global
for a size the compiler knows, as the block reduction's slots already
were, a run of the kernel's `extern .shared` array for a size the launch
sets); the thread loops of a body are phases, the block the largest of
their extents, each phase under `tid < extent` unless its extent is every
sibling's by definition (SSA/ThreadPhases.h, which the block reduction
pass reads too, so that a collective is made only in a phase that is the
whole block); a barrier after every effect the leader makes, so that
every thread reads what it wrote, and the barriers the non-aligned
`barrier.sync`, since the threads reach them from divergent paths. A
size or a bound the launch has to know is a launch value
(SSA/LaunchValues.h): a pure function of what the body captures, computed
again on the host -- the cast of a queue's capacity to the drain's index
type, which the old host code refused as "computed inside the body". A
queue placed at a GPUBlock loop has its arrays, header and record in the
block's shared memory and must say `queue(i, DeviceShared)`
(SSA/QueueStorage.h refuses one that does not, and one that says so
anywhere but at a block loop); the round loop of such a drain runs inside
the kernel, its count a plain load after the drain's barrier
(SSA/RoundGuard.h leaves it alone). The runtime's launch takes the
block's shared bytes and opts a kernel into more than 48 KB. Tests:
correctness/gpu/{block-shared, block-phases, queue-shared},
backends/ptx/{block-shared, queue-shared}, ssa/queue-shared, the two
error tests queue-storage-shared-{untagged, not-per-block}; the two old
refusals' error tests are gone with the refusals.

**Measured on the head** (`scratch/defer/compare-defer.sh`, 16284 points,
64 walks, least of 3; WoSX's persistent kernel 443-447 Msteps/s):

| schedule | Msteps/s | over WoSX | walks ended, mean length |
|---|---|---|---|
| fcpwb-gpu (matching) | 527-530 | 1.19x | 740408, 13.70 |
| fcpwb-gpu-defer (card-wide step queue) | 312 | 0.70x | 740408, 13.70 |
| fcpwb-gpu-defer-block (a queue per point, shared memory) | 457 | 1.03x | 740408, 13.70 |

Both deferrals compute the same estimate (the same walks end, the same
mean length, the same error against the solution; `z` -0.46 and -0.03).
Neither beats the matching schedule here. The card-wide one pays an
entry of some forty bytes per step out through L2 and a launch plus a
synchronous round guard per round, and the rounds run as long as the
longest walk (up to 1024), thinning out as walks end; what it buys, the
lanes a round keeps full, is worth less on the head than it costs, the
head's walks being 13.7 steps on average with a long tail. The per-block
one keeps the entries on the SM and its rounds inside the kernel, but a
block is one point's 64 walks -- two warps -- and once a point's walks
thin out the block idles on its barriers while the card's other resident
blocks fill in; it comes within 14% of the matching schedule. What to try
next: the per-block form with 1024 walks a point (a block of 1024, more
live work per barrier), the card-wide form with a `sort` of the queue by
position each round, and a profile of each (nsys: rounds, time a round,
the guard's share) on the head and the dragon.

**The table** (the same night; `benchmark.sh --gpu --schedules
fcpwb-gpu-defer,fcpwb-gpu-defer-block`, now a log per (mesh, schedule)
cell so that a schedule is added to a table without the rest being run
again): the card-wide deferral 0.48x of WoSX by geomean over the 19
meshes (0.19 on the octopus to 0.75 on the ivy), the per-block one 1.03x
(0.72 on the crown and the beast to 1.76 on the landscape), against the
matching schedule's 1.17x and the best variant's (persistent-reorder-256k)
1.36x. The per-block deferral is the matching schedule's equal on the
largest meshes -- ganesha 1.24x against 1.23x, dragon 1.24x against 1.28x
-- and ahead on the landscape, where nearly every walk escapes on its
first step (1.76x against 1.27x: a round there is one step for all, and
the block's rounds end at once), and behind on the small and medium
meshes, where the round structure costs more than the lanes it keeps
full are worth; the card-wide one is behind everywhere, the entry traffic
and the launch per round never paid for. Both compute the same estimate
on every mesh (the same walks end, the same mean length, `z` within
noise). The two are in the GPU figure as `defer` and `defer-block`.

### The walk on stars (2026-10-25)

The second solver, as the user asked ("add a WoSt benchmark and do the
same as we've done here for WoS"). README.md says what it is and where
each piece lives; this is what was built, what went wrong on the way,
and what the comb measured.

**The program.** `wost.bonsai` is WoSX's `WalkOnStars::walk` as a tail
recursion over the position, the normal where the walk stands on the
reflecting boundary, the distance to the absorbing boundary and the last
step, in WoSX's order, for the homogeneous Neumann problem -- zero flux,
no source, no absorption -- which is the walk's geometry with none of its
integrands (the throughput stays one; the reflecting boundary adds
nothing; no roulette or splitting at the defaults). WoSX's CPU and GPU
walks are the same walk here but for the distance bound, which the driver
sets after the reference (1.001 times the old distance plus the step on
the GPU, none on the CPU); WoSX's CPU solve precomputes the first star
radius per point and reuses it for every walk, which is the same radius
every walk would compute. `stars.bonsai` holds the three sets (the
absorbing triangles, the reflecting triangles, the reflecting boundary's
silhouette edges), FCPW's cone-overlap test and silhouette test
transcribed, and the query's metrics; `walk_common.bonsai` what the two
walks share, and wosx.bonsai imports it (no change to the walk on
spheres' code; its schedules compile as before).

**The trees.** FCPW keeps the reflecting boundary's triangles and its
silhouette edges in one normal cone hierarchy, each leaf referencing its
range of both. The layout (`schedules/layouts/fcpw_stars.bonsai`) reads
that one node array as two trees over two sets, each leaf reading the
range that is its own; the rows are FCPW's GPU rows (GPUSnchNode, 60
bytes; GPUEdge, 56), and the absorbing tree is fcpwb's. Group names are
program-wide in the layout language, so the three layouts' groups carry
distinct names (aprims/anode, rprims/rnode, edges/snode). The driver's
`fcpw_stars_trees.h` copies FCPW's trees by peeking at the Bvh's node
vector and its silhouette references, as fcpw_bvh.h peeks for the walk on
spheres, and writes the rows field by field as FCPW's extractSnchNodes,
extractTriangles and extractSilhouetteEdges write them: FCPW's own
interop header (bvh_interop_structures.h) reaches for slang-rhi, which
only a GPU build has, so the CPU comparison cannot include it.

**What went wrong, and the language question it raised.** The first
build ran, and WoSX ended every walk where ours ended 41% of them, ours
twice as long. Three experiments localized it: with one step a walk the
sides agreed to six walks in nine thousand; with the reflecting shell
wider than the plate (the star the distance to the absorbing boundary,
the silhouette query skipped) they agreed; so the star radius was wrong.
A probe (`scratch/wost/probe.bonsai`, `probe.cpp`, `probe.sh`: exported
wrappers of the three queries called from a C++ harness that asks FCPW
the same questions at the driver's sample points and at the points one,
two and three steps on, on and off the boundary) showed the ray cast
exact at every point and the silhouette query answering a fixed edge at
distance zero nearly everywhere. The SSA said why: `closest_silhouette`
was `argmin(key, map(record, silhouettes))` with the key `inf` for an
edge that is no silhouette or beyond the sphere, and the lowering types
an argmin over an unfiltered set as total -- an element of a non-empty set
-- and builds its option as `(best, true)`, while the accumulator's
running best starts at the identity `inf` and an element whose key is
`inf` never improves on it (`<`), so when every edge's key was `inf`,
which is most stars, the result was the identity's reference to nowhere,
read as a row. The walk on spheres' `closest_within` has the same shape
and the same dead `if !c` after it, which never bit because its sphere
always holds a triangle (the triangle inequality on the bound). The fix
in the program is the language's own idiom for candidacy, apps/rtq's
`trace`: the test an edge has to pass is the filter's `intersects(q,
e)` (within the sphere and a silhouette from its center), which the tree
bounds with `intersects(q, ConeBox)` (the cone valid, the box within the
radius, the cones overlapping -- FCPW's child test), and the key
`distmin(q, e)` is the squared distance, finite; the argmin is then over
a filter and answers none where no edge passes. After it the probe shows
the star radius equal to FCPW's at every point of every step, and the
silhouette query differing only on exact ties at the sphere's radius
(an edge at distance equal to the bound, the last bit of the two
implementations' arithmetic deciding), which the radius does not see.

Raised to the user, not changed: the language types `argmin(f, S)` as
total (an element) when no filter stands between, and `inf` keys as
"not a candidate" in the lowering's pruning; the two together make a
total argmin whose keys are all `inf` return an invalid reference rather
than one of the tied elements or none. The honest fix is in the
compiler: either every argmin over a set is optional (any key may be
`inf`; existing programs that unwrap the plain form would change --
apps/rtq's `closest`, wosx.bonsai's `closest_within`, apps/pbrt's
intersection) or a total argmin seeds its best from the first element it
visits (which the pruned traversal cannot do cheaply) or asserts that one
improved. A compiler test for the case (an argmin over an unfiltered map
whose keys are all `inf`) should exist and fail today. The program-side
form above is right under either reading and is what FCPW's own code
does (the silhouette check is a filter), so the benchmark does not wait
on the decision.

**The comb (side 128, 64 walks, least of 3).** Both sides end every walk
(152064), mean lengths 9.78 and 9.79, rms difference 3e-3, z within 2 on
every run. One thread 0.67-0.72x of WoSX, the threads 0.87-0.88x, the
persistent workers 0.87x, the GPU's matching schedule 1.07x (3 ms a side,
too short to read). The gap on the CPU is the binary tree's visit, the
same as the walk on spheres' on the comb (above, "The binary CPU walk's
gap"), now paid on three traversals a step; the whale onward will say
whether it closes on the big meshes as the walk on spheres' does.

### The binary node's visit made FCPW's (2026-10-25)

The user, reading the CPU figure's note: the baseline must match FCPW's
scalar child visit exactly, and the binary case is the simple base case.
What the two-lane visit was, from the SSA and the machine code: no
directive vectorized it, but two things made it a different program from
FCPW's. The children's boxes are a two-lane pair in the layout
(`lo = vector[vec3f, 2]{ {left.low.x, right.low.x}, ... }`), and the
children's loop read both rows and picked its lane every iteration; and
the sort's lowering took any run whose children are the lanes of a vector
through Embree's peel chain -- the hits as a mask word, a tzcnt per peel,
the keys through a memory cell as integers with the sign flipped -- a
shape for four and eight lanes, applied to two, where FCPW tests the two
boxes, compares once and swaps.

Three changes, in the compiler, each with tests at the three levels:

- **A run of two lanes is sorted as scalars** (src/SSA/SortRecursion.cpp,
  sort_recursion): the compare-and-swap network the pass already had for
  runs that are not lanes -- one float comparison of the two keys and a
  select each for the keys, the children, the carried bounds and the
  conditions; then loopify pushes the far child under its condition and
  enters the near one under its, FCPW's `processSubtreeForClosestPoint`
  and bvh.slang's visit to the branch. The peel stays for four lanes and
  up. Goldens child-volumes-sorted-two and tiled-elements-one re-blessed
  after reading them.
- **`unroll(l)`, a scheduling directive** (the user's proposal; Halide's):
  a parfor of constant extent replaced by its body once per iteration, the
  index that iteration's constant throughout the copy and dropped from
  every block that took it (SSA/Unroll.cpp; substitute_argument in
  SSA/InsertPreheader.cpp does the replacing and the dropping), the copies
  chained at the body's one yield (unify_yields). A first version kept the
  index as the copy's entry argument fed by the constant, and the two
  copies' arguments shared a name, which this SSA reads as one value: the
  second copy's tests were the first child's again, and the comb ended 1623
  walks of 143360 where WoSX ended all; the execution test did not catch
  it, since its copies happened to stay apart. The user's reading -- the
  index should not exist in a copy -- is the fix. Refused on a loop whose
  extent a run decides and on a bound loop (error test unroll-nonconstant).
- **The simplifier reads a built vector's lanes back once the directives
  have run** (SSA/Simplify.cpp, `fold_built_lanes`): `extract_idx(
  make_struct<f32x2>(a, b), 1)` is `b` -- a rule the simplifier had refused
  deliberately, since the lane sort recognizes a node's children by that
  shape (the four-wide FCPW layouts build their children vectors by
  component); after the directives nothing needs the shape, so the fold is
  on from "simplify after the directives". For a vector of scalars only: a
  vector of vectors is built from its components, not its lanes. Nineteen
  goldens moved, every one a `build<f32x3>(a, b, c)[k]` or a splat of a
  built constant folding to its operand; read before blessing.
- Found on the way and fixed at its cause: loopify's lane path for an `any`
  over a two-child node dispatched on a count with one arm (`!hits2up`),
  which the relooper refuses as a switch of one; now a jump
  (QueueRecursion.cpp). No program had an occlusion query over a binary
  tree before the execution test bvh2_unrolled.

**The schedules**: `fcpwb`, `kernel/fcpwb`, `fcpwb2`, `kernel/fcpwb2`,
`cpu/wost_fcpwb` and `kernel/wost_fcpwb` gain `.unroll(<set>.Interior.
children)` after their sort. The sort comes first on a function (Lower/
Sorts.cpp insists), then the unroll, then the stack.

**Measured, fcpwb one thread against one thread, least of 3** (side 128,
64 walks; the two steps apart so that each is seen):

| mesh | before | two-lane sort | + unroll |
|---|---|---|---|
| comb | 0.73x | 0.78x | 0.82x |
| pavilion | 0.84x | 0.86x | 0.90x |
| head | 0.94x | 0.98x | 0.99x |

Every walk ends on both sides, the lengths match, z within one. A strict
gain at each step on each mesh; not level on the comb (a second run of
the final compiler gave 0.84x and 0.99x on the comb and the head). The
same visit on the other binary-tree schedules, one cell each, before and
after: the GPU matching schedule `fcpwb-gpu` on the head 1.19x to 1.32x
(bvh.slang's two tests, now ours too); the walk on stars `wost-fcpwb` on
the comb, one thread, 0.67x to 0.84x -- three traversals a step, each
paying the loop's visit before -- and `wost-fcpwb-gpu` 1.07x to 1.30x.
The tables are to be measured again under it. What the machine
code of the visit still has that FCPW's does not (apps/wosx/scratch/
sort2/wosu2.asm against the FCPW traversal disassembled from the same
driver build): the `vec3f` fields loaded as 8 + 4 bytes and permuted into
a three-lane register (`vpermt2ps`, four a visit) where Eigen loads
scalars; the horizontal sums of the three-lane squared distances through
shuffles (`vmovshdup`, `vshufpd`) where Eigen's are scalar fused
multiply-adds; the key's both-contain select and the metric's "infinite
beyond the radius" select as mask compares and blends. Those are codegen
of `vec3f` arithmetic and the program's own tests, no longer the sort's
or the loop's.

### The silhouette query read against FCPW's, and Eigen's two sums (2026-10-25, afternoon)

The user asked how FCPW's silhouette traversal can return early, whether
the two sides perform the same query, and whether our code for it is
better or worse. Read side by side (bvh.inl,
processSubtreeForClosestSilhouettePoint and SilhouetteEdge::
findClosestSilhouettePoint; our `closest_silhouette` as the SSA and the
LLVM of `wost-fcpwb` come out), with `r2` FCPW's running bound (the query
radius squared, shrunk to each silhouette found), `min_r2` WoSX's
reflecting shell squared, `R2` the query radius squared as our program
keeps it and `best` our argmin's running key (infinite to start):

    FCPW                                        ours
    pop (node, d2); skip if d2 > r2             pop (node, d2); skip unless d2 < best
    leaf, per edge:                             leaf, per edge:
      skip if edge is the current best
      return false if min_r2 >= r2        (A)
      d = distance(x, edge)                       d = distance(x, edge)
      skip if d*d > r2                    (B)     skip if d*d > R2
      skip unless one_face || silhouette          skip unless one_face || silhouette
      best = edge; r2 = min(r2, d*d)              if d*d < best: best = edge, d*d
      break if min_r2 >= r2               (A)
    child k: valid && box_d2 <= r2 && cone (C)  child k: valid && box_d2 <= R2 && cone && box_d2 < best
    push the far hit, enter the near            push the far hit, enter the near

The same answer, not the same search. (A) is unsafe as a closest-point
query -- it stops with whatever silhouette lies within the shell, and a
closer one may exist -- and is safe only because WoSX clamps the result to
`max(d, minRadius)` afterwards, so every silhouette inside the shell gives
the same radius; our argmin is exact and `star_radius` applies the same
clamp, so the star radii agree (the probe below: zero star-radius
mismatches at every point). FCPW's GPU kernel has the same two shell
exits (bvh.slang lines 509 and 563). (B) and (C) are the running bound
tested before the orientation test and before the cone test in FCPW,
after them here: the filter's predicate is the program's `intersects` and
the lowering appends its own `&& distmin < best` to it, evaluated left to
right; the running best is not the program's to name, so the author cannot
write FCPW's order, and the lowering has no semantic grounds to reorder
conjuncts (the user's ruling: no such heuristic). Where no silhouette is
found, most queries, `best` stays infinite and the two orders make the
same tests with the same outcomes. The metric floor `max(d*d, min_r2)`
would express (A)'s effect on the node gate and the per-edge test, not
the `break`, and is not to be made until the isolated benchmarks say
better or worse.

The generated code: each child's box distance once (the SSA's second
`distmin` call, from the fused improvement test, is merged by LLVM; the
`.ll` shows one `fcmp` against `best` on the value already computed), the
edge row one contiguous 56-byte read, the two unrolled child tests
straight-line, `acos`/`asin`/`atan2` as the same libm calls. FCPW's per-
edge test reaches the edge through `silhouetteRefs` and a virtual call,
reads both endpoints through the soup's index arrays, and recomputes both
face normals from the soup on every test (`SilhouetteEdge::normal`: three
position loads, two subtractions, a cross product and a normalization
each); ours reads `n0` and `n1` from the row. That is why the pavilion
profile had our silhouette query at less time than WoSX's despite (B) and
(C) -- about 24 s against 29 s of the 44 s run, from perf samples with
libm split evenly -- and the closest point (12 s against 8 s) and the ray
cast (6 s against 4 s) on the same binary trees at more. Those are sample
attributions; the isolated apps/rtq benchmarks on FCPW's scalar binary
trees, CPU and GPU, are what settles it (the plan below).

Eigen's arithmetic, measured rather than read (apps/wosx/scratch/wost/
eigen_order2_main.cpp, each FCPW expression as FCPW writes it, built with
its flags, against the four fused forms of a three-term sum, 300000 random
inputs): `a.dot(b)` is `(p0 + p1) + p2` with the newer product fused at
each step, `fma(a.z, b.z, fma(a.y, b.y, a.x * b.x))`; `squaredNorm()`,
`norm()`, the box distances `d2Min`/`d2Max` are `p0 + (p1 + p2)` with the
left product fused, `fma(v.x, v.x, fma(v.y, v.y, v.z * v.z))`; `cross` and
`sqrt(r1*r1 + r2*r2)` fuse their left products, as our contraction does.
The `dot` intrinsic fused `(p0 + p1) + p2` the other way round -- Redux.h's
halving unroller, read first, said `p0 + (p1 + p2)` for both, and was
wrong for `dot`. metrics/fcpw_scalar.bonsai now has `dot_eigen` and
`sqnorm_eigen`, written as `fma` chains so that they are FCPW's whether or
not the program is compiled with --ffp-contract, used at every dot product
and norm of the triangle tests and the box distances; stars.bonsai's dot
products and `vlen` likewise. The probe (scratch/wost/probe.sh, the three
queries against FCPW's through WoSX's at the sample points and three steps
on): the comb, 569 points, had 5-9 silhouette mismatches a step, every one
an edge at exactly the radius (the cut's edges lie on the absorbing
boundary, so `d*d <= r2` is a tie decided by the last bit) and the star
radii equal; now 0 at every step, and 0 on the whale (99648 triangles,
1153 points, three steps) too. Two artifacts of the probe fixed on the
way: it asked the silhouette query where the shell already reached the
radius, which FCPW answers with nothing (`squaredMinRadius >= s.r2`) and
`star_radius` does not ask; and two answers under the shell count as
agreement, since FCPW stops at the first (A).

Not matched, by design: the hit record's unit normal. Eigen's
`normalized()` inside `Triangle::intersect` is vectorized by clang's SLP
pass (two squares packed, an unfused add, one fused multiply-add), which
no scalar form reproduces; the normal is a record field, reported by
apps/rtq's deviation column, never a key or a test. A compiler bug found
on the way, to fix when the compiler is unfrozen (bonsai-f9's run), with a
test: `if c && cond` with `c` an option fails in `try_match_types`
(src/IR/Expr.cpp:317, "bool versus bool"): the option's truth cast and a
comparison carry bool types that do not compare equal.

The isolated benchmarks the user asked for are apps/rtq's now (its
PLAN.md, "The queries the walks make, isolated"; schedules fcpwb, fcpws
and their -gpu forms; the silhouette query and its arithmetic moved there
from this app, which imports them). Measured 2026-10-25 over the thirteen
meshes against the FCPW WoSX ships: on the CPU the ray queries over FCPW's
binary tree are behind its scalar traversal (0.81-0.88 by geomean, 0.56-
0.59 on pavilion and landscape), the closest point ahead (1.10-1.21) and
the silhouette query ahead (1.14-1.24); on the GPU every query is ahead
of FCPW's shaders (1.08-1.35, the silhouette 1.22-1.25). That is the walk
on stars' profile read right -- the ray cast was the dear query, the
silhouette the cheap one -- and it names the next piece of codegen: the
two-child ray visit on the CPU, whose machine code "The binary CPU walk's
gap" above compared with FCPW's (the `vec3f` loads and horizontal sums).

### The GPU tables remeasured under the barrier change, and the comb mesh's tie (2026-10-27)

The user's ask (10-26 evening): with apps/rtq's plots current, rerun the
wosx comparison, make its plots current, investigate every cell where we
are worse, profile where we are better, GPU first. The before-copies are
the 10-26 tables (commit 46e2b4e5, measured 03:15-04:07):
results/wosx-gpu-results.before-2026-10-26-barrier.csv and
wosx-stars-gpu-results.before-2026-10-26-barrier.csv with their logs in
gpu-logs-before-2026-10-26-barrier/ (171) and
stars-gpu-logs-before-2026-10-26-barrier/ (19). The reruns (`benchmark.sh
--gpu`, `--stars --gpu`, 3 repeats a cell as before; commit f9065872, the
compiler built at a582c9fd) ran 09:18-10:33 after bonsai-f9's night run;
the stars table had to be resumed once (a peer's compiler at 100% of a
core for ten minutes is "machine busy" to the harness, which stops), and
the comb mesh was remeasured warm after its cold first cells (the same
numbers, within 1%). The figures: plots/wosx-gpu-speedup.* (nine
schedules, as before) and wosx-stars-gpu-speedup.*, each with its .txt.

**The table, cell by cell against 10-26** (speedup geomeans over 19 meshes;
"ours" and "WoSX" are each side's rate today over its rate then):

| schedule | 10-26 | 10-27 | ours | WoSX |
|---|---|---|---|---|
| persistent (fcpwb-gpu, the matching one) | 1.294 | 1.258 | 0.975 | 0.999 |
| persistent-256k | 1.390 | 1.373 | 0.983 | 0.996 |
| persistent-reorder | 1.404 | 1.375 | 0.978 | 0.998 |
| persistent-reorder-256k | 1.471 | 1.468 | 0.993 | 0.993 |
| parallel | 1.200 | 1.309 | 1.100 | 1.010 |
| parallel-reorder-blk64 | 1.430 | 1.504 | 1.060 | 1.007 |
| persistent-point | 0.120 | 0.116 | 0.974 | 1.008 |
| defer | 0.531 | 0.524 | 0.996 | 1.008 |
| defer-block | 1.158 | 1.150 | 0.992 | 0.999 |
| walk on stars, persistent | 1.473 | 1.484 | 1.001 | 0.988 |

WoSX's rates reproduce to a percent on every mesh (the stars' to 1-2%,
crown and landscape apart, whose few long walks scatter more). The one-
launch schedules gained 6-10% on our side; the persistent family lost 1-3%
on sportscar, rover, landscape and villa and held elsewhere; and every
comb cell fell 10-26% on our side alone -- fcpwb-gpu 1.19 to 0.88,
persistent-256k 1.09 to 0.93, defer-block 1.04 to 0.91, parallel-reorder-
blk64 1.50 to 1.24 -- the only cells of a sound schedule under parity.
parallel-reorder-blk64 is now the best of the seven on 12 of 18 meshes
(persistent-reorder-256k on 5; it was 13 and 1). The walk on stars holds:
1.15x (beast) to 1.72x (ivy), no cell moved beyond the scatter of three
repeats of long walks (ours 0.93-1.13 per mesh, 1.001 in geomean).

**The barrier change is not the comb drop.** The same program through
build/compiler (built 22:14 at 0d73c5ff, before a582c9fd: two
`barrier.sync` in the kernel, 60 registers) and through build-rtq's (the
change: none, 54 registers) -- compare.sh told to rebuild neither
(RTQ_SKIP_CMAKE=1, added to it today as apps/rtq's has) -- runs alike on
comb (726/738 against 730/744 Msteps/s, two rounds of ten repeats,
alternating), whale (244/245 against 245/242), sportscar (267/270 against
266/267), rover (219/218 against 217/217) and dragon (44.4/44.6 against
44.3/44.3). The PTX differs by the two barrier lines; ptxas reschedules
(1600 to 1608 SASS instructions, 174 opcodes moved) with no effect on the
rate. For the persistent kernel the barriers held finished warps (ncu:
3.26 of a warp's ~9.5 cycles between issues stalled at the barrier) without
delaying the kernel's end, which the longest walks set either way.

**The comb drop is the two-child tie order, read from a compiler built at
46e2b4e5** (apps/wosx/scratch/wt-46e2b4e5, a worktree with its own build
and that commit's sources, apps/rtq's included -- today's
`[[packed]]` fields do not parse there). Its kernel's PTX differs from
0d73c5ff's by 62 lines, every one the operand order of the sort's
`setp`/`selp`/`and.pred` -- 96229693's tie fix (10-26 12:30), by which a
tie keeps the left child first as FCPW's bvh.slang does (`if
(distToChildNodes[2] < distToChildNodes[0]) swap`, and nothing else) where
the 10-26 kernel put the right child first; 38 of 1600 SASS opcode lines,
60 registers both. Rates (that compiler against today's): comb 890 (1.08x)
against 726-744 (0.90x), sportscar 273.9 against 266-270, whale 246.1
against 242-245. Kernel-only under nsys (results/nsys/comb-fcpwb-gpu*.
sqlite, apps/rtq/nsys_kernels.py): 0.935 ms then, 1.184 ms now, the least
of six; WoSX's 1.11-1.31 ms in both runs. Under ncu (results/ncu/comb-
fcpwb-gpu*, apps/rtq/ncu_kernels.py; clocks at base, so slower than
nsys), our kernel then and now: warp instructions 844.7M to 894.6M
(+5.9%), global loads 77.8M to 79.9M (+2.7% -- FCPW's own kernel makes
79.1M: the new count is FCPW's visit), SM-active cycles 2.17M to 2.40M on
the average SM (+10%) and 2.8-2.95M to 3.52-3.58M on the busiest (+22%),
which sets the kernel's end. The reading: comb is a thin plate on which a
step's sphere center lies inside both children's boxes at most nodes --
`d2Min` is zero for both, the tie the rule decides -- and with FCPW's
order our kernel enters the nodes FCPW enters, a few more per step on the
average and many more on the longest walks (interior points, whose steps
are the large spheres), which a persistent kernel waits for at its end.
The 10-26 comb cell, 1.19x, was a tie order FCPW does not use; the
matching one is FCPW's, and under it we stand at 0.88-0.93x on this mesh
while at 1.11x ourselves-against-ourselves on every other. At FCPW's visit
order we are still 5-10% behind FCPW on comb: 894.6M warp instructions to
their 708M at 3.68 against 4.56 lanes active, over the same loads -- six
steps a walk over a 421-node tree, so the per-step and per-walk code
outside the traversal (the sphere step, the RNG, the claim, the per-point
atomics) is the cost here, and there we spend more instructions than
WoSX's shader. That is the one thing left to read on the GPU table; the
tie order is right.

**Why we are ahead where we are ahead** (head, the matching schedule, ncu,
results/ncu/head-fcpwb-gpu.*; the 1.31x of the table is 1.31x kernel-only,
27.03 against 20.56 ms): we execute 13% *more* warp instructions than
WoSX's kernel (21.9G against 19.4G) in 76% of the time. Ours: 54 registers
to their 82, so 46.6% achieved occupancy against 31.6% and 1.72 eligible
warps a cycle against 0.67, issue slots 70.7% used against 47.1%, the
long-scoreboard stall 2.87 against 3.56 cycles an issue -- more latency
hidden; and fewer bytes: 2.25G against 2.58G load instructions fetching
5.36G against 7.49G sectors (a node's two boxes read as the fields the
test needs, not the row), the stack's traffic 0.6-0.7x theirs (108M
against 156M local stores). Lanes active per instruction are about three
on both sides (2.83 against 3.10), as read on 10-24: the lead is registers
and bytes, not instructions or lanes. The walk on stars reads the same
way (comb, the one stars case short enough for the profiler: head's and
dragon's stars kernels run 1.4 and 9.8 s a launch and Nsight Compute's
replay of WoSX's trips the display watchdog, CUDA_ERROR_LAUNCH_TIMEOUT in
slang-rhi's cuEventRecord; results/ncu/comb-wost-fcpwb-gpu.*,
results/nsys/comb-wost-fcpwb-gpu.*): 1.33x on the table, kernel-only 2.06
against 2.90-3.08 ms under nsys (1.4x), ncu 2.48 against 3.41. Ours: 73
registers to their 120 (34.5% occupancy to 23.6%, 1.75 eligible warps a
cycle to 0.73, issue slots 69% to 48%), 10% fewer load instructions
fetching 19% fewer sectors, the stack's traffic a third to a half of theirs
(6.6M against 12.2M local stores, 11.6M against 33.0M sectors), 5% more
warp instructions (1.85G against 1.77G), 4.5 against 5.0 lanes active.

**The comb loss at the instruction level** (Nsight Compute's source page,
`--page source --print-source sass`: every SASS instruction's warp-level
and thread-level executions; results/ncu/*-source-{ours,wosx}.csv). Per
lane we run 1.9% more instructions than WoSX (3.29G against 3.23G
thread-level), the same node visits (29.5M against 29.6M child-offset
loads), fewer loads (313M against 461M) and half the stack traffic; the
27% more warp instructions are lanes, 3.68 active a warp instruction
against 4.58. Two causes. (1) Float min and max were lowered as a compare
and a select -- std::max's rule, kept for x86's maxss -- which NVPTX
cannot fold to max.f32 (its NaN rule is maxnum's), so a box test's twelve
clamps were a setp and a selp each: 240M compares and selects against
WoSX's 122M plus 60M FMNMX. Fixed in aa4dbaaa: device code lowers them to
llvm.maxnum and llvm.minnum, one FMNMX each, as Slang's min() and max()
and CUDA's fminf and fmaxf are. pbrt's device code writes std::max, which
nvcc keeps a compare and a select, so pbrt --gpu and ours differ where a
NaN leads a max or a min and nowhere else; apps/pbrt's GPU compare is
being rerun to see whether any image notices. The GPU tables here are to
be remeasured under it. (2) The traversal body is in the kernel three
times, as the 10-08 reading found it in the CPU code (above: then a
matter of code size and stack slots; on a warp it is lanes). The walk's
recursion has three traversal call sites once loopify
has made it a loop -- the step's closest_within in the loop body, and the
two `return terminal(...)` exits (at eps and at max-steps), each lowered
where it stands, inside the loop's region before its join -- and LLVM
inlines `_recloop_func1` at all three. A lane whose walk ends runs its
terminal query at once while the warp's other lanes idle: 1.22M node
visits at 1.6-1.8 active lanes, against WoSX's terminal query after its
loop for every ended lane together, 340k at 5.2 lanes -- about 180M of
the 188M excess warp instructions. The same shape is in the walk on stars
(terminal_stars at the exits), not in rtq (no recursion), possibly in
pbrt's megakernels (a loopified path step's exits). Proposed, not built:
loopify's exits carry their arguments to one block after the loop -- its
post-dominator, where a warp reconverges -- and the exit's tail call runs
there, once; identical calls on several exits merge, different ones
dispatch on an exit tag. The meaning is the same on every target (after
an exit the loop runs no more code), the work never more, the exit code
in one copy. Awaiting the go-ahead.

The CPU tables, measured the same day (12:19-14:08, `benchmark.sh` on
build-rtq's compiler: six schedules on 19 meshes, 115 cells) against the
10-26 table cell by cell (the before copy,
results/wosx-results.before-2026-10-27-tie.csv). WoSX reproduces to 0.2%
in every schedule's geomean. The packet family did not move (fcpw4w16
1.194 to 1.195, -threads 1.251 to 1.243, -persistent 1.327 to 1.328). The
binary-tree family gained on every mesh, our side alone: fcpwb 1.023 to
1.178 (our rate x1.15 in geomean), fcpwb-threads 1.103 to 1.194 (x1.09),
fcpwb-persistent 1.158 to 1.275 (x1.10); comb, under parity in all three
on 10-26 (0.81, 0.97, 0.99), is 1.32, 1.31 and 1.30 (our rate x1.62 on
fcpwb), and no cell is under 0.97 any more (the least, fcpwb-threads on
dragon, 1.06). The commits between the two tables that touch the binary
tree and not the packet one: the two-child tie order 96229693 (the CPU's
sort is the same sort), `[[packed]] vec3f` 858c6d45 and the
reference-exact records c7fc94d7. Which gives what is not separated here:
comb's x1.62 has the tie's signature (the GPU's comb moved the other way
under the same order, above -- there the longer walks end a persistent
kernel late, a cost a CPU thread does not pay), the 5-15% on every mesh
the bytes'; an A/B at each commit would say. One note on the run: the
compiler in build-rtq was rebuilt at 12:24 while the table ran, against
the rule; the whale and car cells that overlapped it reproduce their
10-26 rates to 0.5% on both sides (each cell keeps the best of its
repeats), and the rebuilt change (aa4dbaaa, the device's min and max) is
device-only, so the table stands. The figures are redrawn:
plots/wosx-cpu-fcpw4w16-speedup.*, plots/wosx-cpu-fcpwb-speedup.* and
their .txt.

**The exit funnel, built and measured** (9846e3e6, the afternoon). loopify
now brings every exit of the loop it makes to one block after it,
`<func>!exit`, which reconverges -- a new intrinsic, `reconverge()`, an
empty convergent inline asm on the device and nothing on the host, the one
thing LLVM's jump threading leaves standing -- and dispatches on which exit
was taken, the two `return terminal(...)` exits folded into one arm
(SSA/Rewrite.cpp, funnel_exits). The matching schedule, A/B against a
compiler at aa4dbaaa (the min/max change in, the funnel out), two rounds of
ten repeats alternating, WoSX's rate measured in every run as the control:
comb 0.97x and 0.95x of WoSX to 1.18x and 1.13x -- our rate 764 and 752 to
947 and 916 Msteps/s, WoSX's 788-809 throughout -- and head 1.36x and 1.36x
to 1.43x and 1.44x (609 and 611 to 641 and 646; WoSX 448-450). The kernel
holds the traversal twice now, the step's and the one terminal, as WoSX's
shader does (computeDistToAbsorbingBoundary in the loop,
projectToAbsorbingBoundary after it); it held three. The GPU tables are to
be remeasured under both changes, 21:30 tonight after f9's pbrt sweep.

**What the funnel cost apps/pbrt, and the fix** (9dbdebc9, the evening).
bonsai-f9's sweep and heatmap under 9846e3e6: the wavefront untouched on
scenes without media (within 0.4%, images identical), but its medium march
kernel 26% slower on explosion (246.5 -> 310.9 ms at 16 spp) and the
megakernels 2-4%, the SER flat one 20% on killeroo-gold (59.4 -> 71.6 ms),
with the OptiX continuation stack of its raygen grown 352 -> 416 bytes per
thread. Read in the SER raygen's PTX, a liveness analysis at its
continuation points (scratchpad funnel/ptx_live.py): 17 more 32-bit
registers live across the reorder and 48 across the shadow trace than
before the funnel, twelve of them the previous hit's ShapeSampleContext --
the loop header's own value, read after the loop. The cause: a slot an exit
did not fill was `undef` on its edge, and LLVM folds a phi of a value and
undef into the value when the value dominates the funnel, which the
header's arguments do; the arm then read the header's value past the loop,
and it stayed live over the whole body, every exit being a path to the use,
where its last use in the body was early. The SSA of the bounce loop's
funnel also showed some twenty of the scene's tables, kernel arguments,
carried as slots for nothing. The fix, in funnel_exits: the payload laid
out as the variant it is -- slots shared by type across the arms, so the
funnel costs the largest arm's registers rather than every arm's -- a
parameter every edge of its arm hands one value defined before the loop
handed to the arm directly, and a slot an exit does not fill handed the
zero of its type, never undef. Under it the SER raygen's live set is the
pre-funnel one again (189 registers across the reorder against 205 and 188,
186 across the shadow trace against 213 and 185) and its continuation stack
352 bytes; f9 times the three-way A/B (9c191640, 9846e3e6, 9dbdebc9)
tomorrow. The medium march is another matter: its spill stores at the
128-register cap are 2524 bytes under the fix, 2560 under the funnel, 2164
before it, and the attribution compiles put the difference on neither the
min/max and specialize changes (a compiler at aa4dbaaa spills the 2164) nor
the reconvergence itself (the funnel with no device asm, threaded back by
LLVM with its merged arms kept, spills the 2560) nor the folding of
identical exit regions into one arm (the fix with no merging at all spills
the same 2524): every form of the funnel has them and only no funnel does
not, which is not understood and goes to f9's timed A/B on explosion
tomorrow (9c191640, 9846e3e6, the fix), the medium kernel the cell to watch.
One more thing the fix's PTX showed, and the second commit: the medium
march has a local of its body, `segment_passed`, whose ADDRESS is one arm's
slot, filled on one edge of five. Under the funnel the other edges handed
undef, LLVM folded the phi to the one address and the local stayed in
registers; under the fix's zero they handed a null, the address became one
of several the phi might hold, and the local went to memory -- a 16-byte
local depot in the medium kernel and the shadow raygens, a store and a
load per lane. So a slot of data is zero (its lifetime is what the fold
stretches) and a slot that holds an address is undefined (the arm must see
the one address; an address's own lifetime is a register or two); the
depot is gone, the wavefront recompiled and read through ptxas (9dbdebc9
and 39c67ab9). For the walk on spheres the fix changes the funnel's slots
alone (the terminal's `x` and `steps`, zero where undef was), and the
three-way A/B on the matching GPU schedule says it keeps the funnel's win:
the same program through a compiler at aa4dbaaa (no funnel), at 9846e3e6
(the funnel) and at 39c67ab9 (the fix), two rounds of ten repeats in turn,
WoSX in every run as the control -- comb 0.89x and 0.94x / 1.15x and 1.13x
/ 1.18x and 1.22x of WoSX (our rate 732 and 761 / 939 and 927 / 926 and 956
Msteps/s, WoSX's 781-822), head 1.36x and 1.37x / 1.44x and 1.44x / 1.46x
and 1.46x (608 and 613 / 645 and 644 / 646 and 649; WoSX 443-449).

**The medium march's remainder, read in the PTX** (the night of 10-27/28,
while the CPU tables ran). bonsai-f9's night run under 39c67ab9 settled what
the two commits bought apps/pbrt: the flat SER megakernel recovered most of
its loss (killeroo-gold 16 spp 71.6 -> 64.3 ms, 59.4 before the funnel; the
four VolPath raygens' continuation stacks back to the pre-funnel 352/512/544/
672 bytes), the other megakernels 1-3%, and the medium march NOTHING
(heatmap11 vs heatmap10: 1.0004x over 180 rows; `_kernel_medium_samples_blk`
on explosion 310.2 ms, 310.9 under 9846e3e6, 246.5 before the funnel). So the
local depot of 39c67ab9 was a side issue there, and the residual spills are
the whole 26%. The cause is in the kernel's PTX, saved from the evening's
compiles (scratchpad funnel/medium-{aa4dbaaa,c,fix2}.ptx; the files differ by
98 lines of 25,000, all of them the funnel): every exit of the march leaves a
body block by a conditional branch -- the latch itself ends `@p bra header;
bra funnel` -- and LLVM places the copies the funnel's phis need in that
block, BEFORE the branch, where they run on every iteration of the march.
Under 39c67ab9 the latch carries eight 64-bit copies of live values -- six
of the loop's state into the union slots (slot0 is beta on one edge and
T_maj on another, so no slot can share the state's own register and every
one is a second copy of a live value) and two for the header's own phis,
which the slots' interference keeps the coalescer from removing -- seven
zero fills (the slots the latch's exit does not fill) and the tag; the two
other exits carry five copies and four fills each; the no-funnel compiler's
latch carries five 32-bit moves for the program's own merge after the arms,
and no 64-bit ones. Under 9846e3e6 the latch carried the tag alone
-- the undefined slots let the phis fold into the header's values, which is
the live-range extension read in the SER raygen on Monday -- so both forms
of the funnel load the latch, one with registers, one with copies, and ptxas
spills either way: +392 bytes of spill stores is five edges times ten 64-bit
slots times eight bytes, the slots stored on every edge and reloaded once
(spill loads +44). The SER raygen's bounce-loop funnel has the same shape at
a larger scale: six exits, each a conditional branch out of the body, each
preceded under 39c67ab9 by thirty copies of one register, the materialized
0f00000000 -- the zero fills -- and by none under 9846e3e6; that is the 8%
of the 20% that remains on killeroo-gold, at no stack cost. Two facts of
LLVM 23 (deps/llvm-project) say the copies cannot be moved off the body on
the GPU by any option: PHIElimination's SplitPHIEdges (PHIElimination.cpp
801-891) has the rule "if we are looking at a loop exiting edge, split it so
we won't insert code in the loop", but an early exit skips it whenever the
incoming value's only uses are phis (isLiveOutPastPHIs; LiveVariables does
not count phi uses), which is every slot of ours -- the hidden option
`-no-phi-elim-live-out-early-exit` lifts that -- and NVPTX sets
RequiresStructuredCFG, under which MachineBasicBlock::canSplitCriticalEdge
refuses every split that does not enter a loop header (its transitional
flag `-disable-nvptx-require-structured-cfg` also turns tail duplication and
tail merging back on, which LLVM disables for the PTX target on purpose).
On the CPU the first option alone would do; on the GPU the funnel must need
no copies in the body, which is a matter of what its phis receive. The
payload to build, not yet built: slots keyed by VALUE, not by type -- an arm
parameter handed one SSA value over all its edges shares slot_V with every
other parameter handed that value, so the phi is the value's own LCSSA phi
and coalesces with the value's register (the latch's eight state copies
vanish: the slot IS the state's register); a parameter handed different
values by different exits keeps a slot of its own (two distinct values, so
the phi can never fold, and undefined is free on every edge that does not
fill it); on an edge that does not fill a slot the phi receives undefined,
except where undefined would let it fold into one value that dominates the
funnel: a slot whose one value is the loop's parameter i receives, on the
exit out of the back-edge block, the back edge's own argument i -- the
parameter's next value, already in the parameter's register there, so the
phi has two values and no copy -- and a slot whose one value is some other
value dominating the funnel receives the zero of its type (data) or
undefined (address), or is left to fold where the value is live to the
latch anyway: that last choice to be measured, not guessed, on the medium
kernel (ptxas spill stores, 2164 the mark) and the SER raygen (the live sets
at the reorder and the shadow trace). With the tests: ssa/loopify-exits's
slots per value, and a loop whose early exit hands the header's parameter
while the latch's exit does not, the latch's edge showing the back edge's
argument in the slot. A day of compiler work and every table again (CPU
too: the funnel is in the CPU builds); when is the user's call.

**The GPU tables under 39c67ab9** (21:44-22:26; before-copies
wosx-gpu-results.before-2026-10-27-funnel.csv and
wosx-stars-gpu-results.before-2026-10-27-funnel.csv, the morning's logs in
gpu-logs-before-2026-10-27-funnel/ and stars-gpu-logs-before-2026-10-27-
funnel/). Against the morning's table (42901acc+, before the device's
min/max and the funnel), WoSX's own rates within 0.6% throughout, geomeans
over the nineteen meshes: persistent 1.258 -> 1.358 (our rate x1.08),
persistent-256k 1.373 -> 1.487 (x1.08), parallel 1.309 -> 1.407 (x1.07),
persistent-reorder 1.375 -> 1.482 (x1.08), persistent-reorder-256k 1.468 ->
1.572 (x1.08), parallel-reorder-blk64 1.504 -> 1.584 (x1.05),
persistent-point 0.116 -> 0.124 (x1.07); no cell below 0.97 of before. The
walk on stars on the GPU 1.484 -> 1.543 (x1.03), dragon 1.59 -> 1.62. The
table holds the harness's seven schedules; the two deferral schedules of
10-25 (fcpwb-gpu-defer, fcpwb-gpu-defer-block) were rows left from their
own runs and are in the before-copy, not remeasured. The figures
(plots/wosx-gpu-speedup.*, plots/wosx-stars-gpu-speedup.*) are drawn from
these tables. rtq's two GPU schedules were remeasured the same hour
(apps/rtq/PLAN.md, 2026-10-27 night: the ray and point queries +1-4%, the
silhouette queries -1.5% and -2.5%, min/max's doing, rtq having no funnel).
The CPU tables follow tonight after bonsai-f9's pbrt window: the solver's
and rtq's by morning, the walk on stars' (5 h 50) in the day.

**The solver's CPU table under 39c67ab9** (01:38-03:25 on 10-28, the machine
otherwise idle after bonsai-f9's window ended at 01:34; before-copy
wosx-results.before-2026-10-27-funnel.csv, the 10-26 logs in
logs-before-2026-10-27-funnel/). The CPU builds carry the funnel as the GPU
ones do, so this is the funnel's first CPU measurement, and it is a tie:
over the 114 cells our rate is 0.9997 of before (geomean) and WoSX's own
0.998, and no cell's speedup is below 0.97 of its before. Geomeans over the
nineteen meshes, before -> after: fcpw4w16 1.195 -> 1.200, fcpw4w16-persistent
1.328 -> 1.341, fcpw4w16-threads 1.243 -> 1.247, fcpwb 1.178 -> 1.177,
fcpwb-persistent 1.275 -> 1.271, fcpwb-threads 1.194 -> 1.194. The cells
that moved most moved for both sides at once -- landscape under
fcpwb-persistent, our rate 0.93 of before and WoSX's 0.90, the speedup 1.99
both times; comb under fcpwb-threads 1.31 -> 1.27, the lowest of the table
at the 0.97 line, our rate 0.935 and WoSX's 0.964 -- the machine's hour, not
the compiler's. This is the CPU half of the reading above: the copies the
funnel puts in the loop body are in the CPU code too, and the CPU's
register allocator, which splits live ranges and can rematerialize a zero,
absorbs them; the GPU path has no allocator of its own between LLVM and
ptxas and cannot move them off the body. The figures
plots/wosx-cpu-fcpw4w16-speedup.* and plots/wosx-cpu-fcpwb-speedup.* are
drawn from this table; the walk on stars' CPU table follows in the day.

**The walk on stars' CPU table under 39c67ab9, and the one-thread cell
retired** (10-28: a head start 03:55-06:44, resumed 08:07-13:46 around
bonsai-f9's morning window, the last cells 13:47-14:25; before-copy
wosx-stars-results.before-2026-10-27-funnel.csv, which is the 10-25 table,
edad5f6c, the last time this table was measured). Three meshes are new to
it, lte-orb, ganesha and dragon, and their one-thread cell -- our thread
against WoSX's `runSingleThreaded` -- ran WoSX's single thread for an hour
each (ivy's 59 minutes, lte-orb's 69), which is where the day went and
what brought the user's ruling: an application is measured in parallel
under the schedule the reference runs, and one thread against one thread
is a microbenchmark's question, apps/rtq's. So the one-thread schedules
(fcpw4w16, fcpwb, wost-fcpwb) leave benchmark.sh's defaults and plot.py's
bars -- the CPU figures have two bars a mesh now, the threads against
WoSX's `tbb::parallel_for` and the persistent workers against the same --
and stay runnable by name; the rows already measured stay in the CSVs, and
ganesha and dragon have the two parallel cells alone. The figure
plots/wosx-stars-cpu-fcpwb-speedup.*, over eighteen meshes (landscape left
out as before): parallel 1.21x of WoSX (geomean; 1.08x dragon to 1.44x
comb), persistent 1.23x (1.01x crown to 1.46x), no cell under 0.97. The
10-25 figure had 0.94x and 0.95x over fifteen. The move is ours alone:
cell by cell against the 10-25 table WoSX's rate is 0.99 of before and ours
1.28 on the parallel cells (comb 1.64, dambreak 1.57, rover 1.56, ivy 1.41,
head 1.14 the least) and 1.49 on the one-thread cells (comb 2.14, dambreak
2.04, rover 1.96). It is not the funnel's: the solver's table above, whose
before-copy is of 10-27, tied. It is the CPU work of 10-25 and 10-26 that
this table had not been measured under -- the binary node's visit made
FCPW's (3e45970c) and the two children's tests written out (71b720c2), the
sorted run of two as a branch (96229693, 30059901), the CPU comparison
reading the bytes FCPW's and WoSX's CPU read (86b5160d, 9132c286), an
element's fields read where they are used (525f8b38), `[[packed]] vec3f`
(858c6d45) -- which moved the solver's table at the time and which the
walk on stars, three queries a step over the same trees, gains from three
times over. The solver's two CPU figures were redrawn from their table
with the two bars (no cell remeasured); the geomeans of the parallel bars
are as in the paragraph above.

**The funnel replaced by the merge alone, and the walk keeps its gain**
(10-28, 15:16-15:18). The user's reading of the funnel: its motivation was
equal code at several ways out of a loop, not a place every exit must
pass; a hub is not an optimization in general -- it is the precondition of
AMDGPU's software-managed lane mask (LLVM's UnifyLoopExits before
StructurizeCFG), and NVPTX runs no such pass and leaves where a warp
reconverges to ptxas -- and a loopify argument switching it on would be a
hack. So loopify now merges the exits whose code is the same code and
brings nothing else together (merge_exits in src/SSA/Rewrite.cpp): the
walk's two `return terminal(...)` are one block entered from both exits
with that exit's values, its two `return {}` another, and no block after
the loop carries a tag or waits. The A/B, this compiler against 39c67ab9's
on comb and head, fcpwb-gpu, two rounds of ten repeats alternating with
WoSX's rate as the control: our rate 941/931 -> 932/926 Msteps/s on comb
(0.99x), 648/648 -> 653/648 on head (1.00x), WoSX's own 817/807 -> 788/805
and 448/449 -> 451/444; the speedups 1.15x/1.16x -> 1.18x/1.15x on comb and
1.44x/1.45x -> 1.45x/1.46x on head, against the pre-funnel 0.88x and 1.31x.
So the gain of 9846e3e6 was the merged code -- the terminal traversal in
the kernel once, entered from both exits -- and not the wait at one block,
which the reading of 10-27 had credited; the march's 26% and the SER
megakernel's 8% bought nothing. What the merge costs, read in
tests/bonsai/backends/ptx/loopify-exits: one register move per differing
value per exit per iteration (three moves where the funnel's body had
twelve and the empty asm). Every wosx table is remeasured under this
compiler next; pbrt's megakernel and medium march are bonsai-f9's to
measure against a53c8f84.

**The four tables remeasured under the merge (6f46500c)** (10-28,
16:18-19:27: the walk on spheres' GPU table 16:18-16:52, the walk on
stars' 16:52-17:00, the solver's CPU table 17:29 and the stars' CPU table
19:27, each table's figure redrawn as it folded). Before each the 39c67ab9
table was copied beside it (`results/<table>.before-2026-10-28-merge.csv`,
its logs in `<logs>-before-2026-10-28-merge/`) and compared cell by cell
with WoSX's rate as the control (scratch: wosx_compare_before.py). The
matching schedules tie their previous measurement everywhere: fcpwb-gpu
1.358x -> 1.365x by geomean over 19 meshes (our rate 1.006 of before,
WoSX's 1.006), wost-fcpwb-gpu 1.543x -> 1.535x (0.998, 1.003), the
solver's four CPU schedules 0.994-0.998 of before with WoSX 0.993-1.003,
the stars' two 0.992-0.994 with WoSX 0.986-0.988; no CPU cell under 0.97
of WoSX but the landscape's known 0.22x, and of the stars GPU cells only
the comb moved more than two percent (our rate 0.92 of before; its 3 ms
solve is the one the table cannot time) and the whale on WoSX's side
(1.16). One move against our own measurement, not against WoSX: the
walk on spheres' parallel schedule (one launch, a thread per walk) came
in at 0.968 of its before by geomean and at 0.94-0.96 on the eleven
meshes from the pavilion up, WoSX's rate flat on every one of them,
while the two 256k schedules moved the other way (persistent-256k 1.025,
persistent-reorder-256k 1.021, their large meshes 1.04-1.06) and the rest
within a percent. Our rate alone, in opposite directions by schedule:
that reads as the merge's code -- the exits sitting where the program
wrote them change the registers the kernel needs and so the occupancy the
plain launch and the over-full launch get -- rather than the machine. The
parallel cells stand at 1.08x-1.53x over WoSX; an A/B of that schedule
alone against the saved 39c67ab9 compiler would pin the cause, and is not
run. So after the merge: spheres GPU persistent 1.38x over 18 meshes
(persistent-reorder-256k 1.64x, the best), stars GPU 1.54x, solver CPU
1.21x/1.31x on the wide tree and 1.17x/1.23x on WoSX's default, stars CPU
1.21x/1.24x -- the figures of 16:52, 17:00, 17:29 and 19:27.

### The schedule exploration of 10-28: the point schedule out, OptiX with reordering, the stars' variants, the CPU's layout and packets

The user, 10-28 evening: the persistent-point schedule's bar gone from
the GPU figure; whether persistent-reorder-256k is a loop reorder or SER;
a schedule using OptiX with shader execution reordering at each step of
the walk; more schedules on the walk on stars' GPU figure and on the CPU
figures, vectorization on fcpwb among them -- "be creative, try some
things".

**The answers before measuring.** persistent-reorder-256k is a loop
interchange: `reorder(i, w)` then `reorder(w, i)` swaps the two loops of
the nest before `collapse`, so the work items are numbered walk-fastest
and a warp's 32 consecutive claims are one point's walks; nothing in any
wosx schedule reordered threads at run time. The compiler has the
construct, written for pbrt (docs/schedule-semantics.md, the sort on a
call; Lower/Sorts.cpp apply_reorder): `f.sort(g, |args| key)` inside a
loop bound to OptixThread is `optixReorder(key, bits)` before the call,
the hint the key's low bits, at most sixteen. The stars' variants had
been planned in this file's Next list on 10-25 and never run; the CPU's
packet forms were measured one thread against one thread on 10-09 at a
tenth of WoSX's rate and left out of the defaults since.

**What was built** (apps/wosx/schedules, README.md's list). The walk on
spheres through OptiX: `fcpwb-gpu-optix` (the flat launch as a
ray-generation program, `bind(k, OptixThread)`, the traversal the
kernel's software one -- the control), `fcpwb-gpu-optix-ser` (the same
with `walk.sort(walk, |x, problem| walk_sort_key(x, problem.lo,
problem.hi))`: before every step the launch's threads are regrouped by
the cell of a 32x32x32 grid over the box the walk stands in, a fifteen-bit
Morton code -- walk_keys.bonsai, a file only the reordering schedules
import) and `fcpwb-gpu-optix-ser-persistent` (the 131072-worker claim
under that reordering, `bind(t, OptixThread)` on the workers' loop, a
combination the suite had no test of; it compiles and computes the right
estimate). The walk on stars' GPU variants: parallel, reorder, 256k,
reorder-256k, parallel-reorder-blk64, the per-point deferral
(defer-block), optix, optix-ser and optix-ser-persistent, each the walk
on spheres' directives on solve_stars. The CPU: `fcpwb2-threads` and
`fcpwb2-persistent` (the layout where a node's row holds its children's
boxes, one row read a visit where FCPW's take three, in parallel; WoSX's
side unchanged on FCPW's own rows), `fcpwb-packet-threads` and
`fcpwb-packet-16-threads` (a point's walks vectorized in gangs of 32 and
of 16 lanes, the points over the threads). One rule met on the way: a
function's `sort` directives come before its other directives
(Lower/Sorts.cpp:738), so the reordering schedules cannot import
kernel/fcpwb.bonsai, whose `walk.loopify()` would come first; they import
the layout and spell the traversal out, as the deferral schedules do.
compare.sh gives every GPU build the OptiX SDK's and CUDA's include
paths, which the generated header of an OptiX program needs
(runtime/bonsai_optix.h); plot.py draws the new bars (GPU_BARS,
STARS_GPU_BARS, CPU_EXTRA_BARS) and no longer draws persistent-point by
default (its rows stay in the CSV, its file runnable by name).

**Measured on the comb and the head** (scratch/explore: the comb one
repeat as a check that each schedule builds and computes WoSX's estimate
-- every one did, the same walks ended on both sides, the mean lengths
and errors equal, z within 2 -- then the head with three repeats beside
each schedule's control; speedups over WoSX):

| walk on spheres, GPU | head |
|---|---|
| persistent (matching, CUDA) | 1.45x |
| parallel (one launch, CUDA) | 1.47x |
| optix | 1.07x |
| optix-ser | 1.02x |
| optix-ser-persistent | 0.93x |

| walk on stars, GPU | head |
|---|---|
| persistent (matching) | 1.65x |
| parallel | 1.00x |
| persistent-reorder | 1.74x |
| persistent-256k | 1.62x |
| persistent-reorder-256k | 1.81x |
| parallel-reorder-blk64 | 1.44x |
| defer-block | 0.90x |
| optix | 0.94x |
| optix-ser | 1.21x |
| optix-ser-persistent | 1.19x |

| CPU, WoSX's default tree | head | comb |
|---|---|---|
| fcpwb-threads | 1.16x | 1.06x |
| fcpwb2-threads | 1.18x | 1.24x |
| fcpwb2-persistent (fcpwb-persistent 1.17x) | 1.18x | 1.25x |
| fcpwb-packet-threads (32 lanes) | 0.49x | 1.95x |
| fcpwb-packet-16-threads | 0.65x | 2.01x |

Three readings. (1) The reordering is real and the OptiX launch eats it:
on the stars walk, three traversals a step, regrouping the threads by
position lifts the same launch from 0.94x to 1.21x; on the spheres walk,
one short traversal a step (13.7 steps a walk on the head), the reorder's
call per step costs more than the coherence gives (1.07x -> 1.02x). Both
ways the plain ray-generation launch of this software traversal runs far
under the CUDA launch of the same code (spheres 1.07x against 1.47x,
stars 0.94x against 1.00x), and the persistent claim under OptiX adds
nothing (spheres 0.93x, stars 1.19x), so no OptiX schedule reaches the
CUDA persistent ones. The runtime compiles the module at OptiX's default
optimization with no debug or exception flags (bonsai_optix.h; the
validating mode is off), so the cost is the raygen's own -- what OptiX
makes of a 64-entry traversal stack and a long loop in a program meant
to trace -- and a profile would say where, which for an OptiX launch
needs the user's yes. (2) The walk on stars rewards the walk on spheres'
best schedules more than that walk did: reorder-256k 1.81x on the head
against the matching 1.65x (the spheres' 1.75x against 1.45x, the 10-28
table), and the plain one-launch form collapses to 1.00x on this walk,
whose 78-step walks have a long tail that a warp of the plain launch
waits out; the per-point deferral, which the Next list expected a
three-query step to reward, is at 0.90x -- a block of one point's 64
walks idles on its barriers once they thin, as on the spheres. (3)
Vectorizing over a point's walks is the mesh's call: on the comb, whose
walks are six steps and stay together, the packet doubles the rate
(1.95x and 2.01x against 1.06x); on the head it halves it, and sixteen
lanes beat thirty-two by a third (0.65x against 0.49x: a gang of sixteen
is one vector instruction where thirty-two is two, and the leaves it
reaches are the union of sixteen walks' rather than thirty-two). The
fcpwb2 layout buys 2% on the head and 18% on the comb in parallel.

**The tables** (10-28, 19:44-22:07: benchmark.sh --gpu, --stars --gpu,
then the CPU table, each resuming from its logs so that only the new
schedules' cells ran -- 57, 171 and 76 cells -- and each figure redrawn
as its table folded; every new cell computed WoSX's estimate, the same
walks ended, lengths and errors equal). Speedups over WoSX by geomean
over 18 meshes:

| walk on spheres, GPU | geomean | min | max |
|---|---|---|---|
| persistent (matching) | 1.38x | 1.13 beast | 1.53 ivy |
| parallel | 1.37x | 1.08 comb | 1.53 ivy |
| persistent-reorder | 1.51x | 1.10 beast | 1.72 ganesha |
| persistent-256k | 1.54x | 1.17 beast | 1.71 ivy |
| persistent-reorder-256k | 1.64x | 1.10 beast | 1.98 ganesha |
| parallel-reorder-blk64 | 1.59x | 1.08 beast | 1.91 ganesha |
| optix | 1.03x | 0.77 comb | 1.12 ganesha |
| optix-ser | 0.97x | 0.65 comb | 1.12 ivy |
| optix-ser-persistent | 0.90x | 0.62 comb | 1.05 ivy |

| walk on stars, GPU | geomean | min | max |
|---|---|---|---|
| persistent (matching) | 1.54x | 1.27 whale | 1.80 dambreak |
| parallel | 1.07x | 0.82 zero-day | 1.46 comb |
| persistent-reorder | 1.64x | 1.21 beast | 1.98 dambreak |
| persistent-256k | 1.52x | 0.96 crown | 1.82 dambreak |
| persistent-reorder-256k | 1.66x | 1.28 octopus | 1.97 ivy |
| parallel-reorder-blk64 | 1.45x | 1.17 zero-day | 1.82 comb |
| optix | 0.97x | 0.82 zero-day | 1.17 dambreak |
| optix-ser | 1.04x | 0.35 whale | 1.88 ivy |
| optix-ser-persistent | 0.99x | 0.36 whale | 1.80 ivy |
| defer-block | 0.99x | 0.72 zero-day | 1.35 dambreak |

| CPU, WoSX's default tree | geomean | min | max |
|---|---|---|---|
| fcpwb-threads (matching) | 1.17x | 1.06 dragon | 1.30 comb |
| fcpwb-persistent | 1.23x | 1.16 whale | 1.46 dragon |
| fcpwb2-threads | 1.22x | 1.15 crown | 1.35 ganesha |
| fcpwb2-persistent | 1.27x | 1.18 crown | 1.64 ganesha |
| fcpwb-packet-threads (32 lanes) | 0.39x | 0.26 rover | 1.86 comb |
| fcpwb-packet-16-threads | 0.55x | 0.39 rover | 2.03 comb |
| fcpwb-packet-8-threads (the user's ask, 23:12) | 0.56x | 0.44 rover | 1.41 comb |
| fcpwb-packet-4-threads (the same) | 0.57x | 0.47 ganesha | 1.00 comb |

What the tables add to the head's readings. (1) The OptiX launch of the
software traversal runs at 0.75x of the CUDA launch of the same code on
the spheres (optix over parallel, 0.70 ivy to 0.82 beast, every mesh)
and at 0.91x on the stars. The reordering over it is 0.93x on the
spheres (one short traversal a step) and 1.07x by geomean on the stars,
but that geomean hides two populations: on the five million-walk meshes
with the longest walks it is at or near the best of the ten -- ivy 1.88x
over WoSX (656 steps a walk), rover 1.72x (405 steps; the best of the
ten there), dambreak 1.64x (351), ganesha 1.41x (235), car 1.39x (109)
-- and on the three launches of 150 to 305 thousand walks it runs at a
third to a half of the plain OptiX launch (whale 0.35x over WoSX against
optix's 0.96x, octopus 0.43x against 0.91x, comb 0.87x against 1.08x),
the same three meshes where the spheres' reordering loses most (0.65x,
0.69x, 0.83x). The correlate in the table is the launch's walk count,
not the walk's length (whale's 30 steps, octopus's 75, car's 109) nor
the tree's size (whale's 100k triangles against bmw's 111k, which gains);
what the hardware does with a reorder among fewer threads in flight is a
profile's question, and so the user's. The persistent claim under OptiX
stays under the plain OptiX launch on both walks (0.93x and 0.96x of
optix-ser). (2) On the stars the loop interchange alone is the move:
persistent-reorder 1.64x and persistent-reorder-256k 1.66x (the head's
1.74x and 1.81x were the head's), the best of the ten on 7 and 5 meshes,
and twice the threads buys nothing without it (persistent-256k 0.99x of
persistent, where the spheres' was 1.12x); the plain launch is 0.69x of
the persistent claim on the stars against 0.99x on the spheres -- the
long tail of a 78-to-656-step walk waited out by its warp -- and
parallel-reorder-blk64 recovers 1.36x of that, the best of the ten on
comb, crown and beast, the launches too small to fill the card.
defer-block is 0.99x over WoSX, 0.64x of the persistent claim, as the
head said. (3) On the CPU the fcpwb2 layout is worth 1.04x over the
threads (0.96 comb to 1.18 dragon), and persistent fcpwb2 at 1.27x is
the best CPU bar on WoSX's own tree; the packets lose on seventeen of
eighteen meshes, 0.34x and 0.47x of the threads by geomean, the comb
alone gaining (1.43x and 1.56x of the threads), sixteen lanes over
thirty-two by a third everywhere. The user then asked for gangs of eight
and of four (23:12, 38 cells): 0.56x and 0.57x -- the geomean flat from
sixteen down, the comb's gain gone (1.41x, then 1.00x) and the worst mesh
a little higher (0.44, 0.47). So below the register's width the gang's
width does not matter: the vectorized walk runs at about half the scalar
threaded walk at sixteen, eight and four lanes alike, and the cost is the
packet's -- a leaf any lane reaches tested gang-wide for the lane or two
that are there, and the gang held to its slowest walk -- not the lane
count; the one gain, thirty-two to sixteen, was the two vector
instructions a gang of thirty-two issues where sixteen issue one. The
packets stay in the defaults and the figure as the measured answer to the
question, not as a candidate. The figures: wosx-gpu-speedup (nine bars a
mesh), wosx-stars-gpu-speedup (ten), wosx-cpu-fcpwb-speedup (eight), each
with its .txt.

### The OptiX launch under Nsight Compute (2026-10-28, night)

The user, on the exploration's reading that the OptiX launch of the
software traversal runs at 0.75x of the CUDA launch of the same code: use
Nsight Compute on it -- "I suspect it is a register allocation issue".
The programs kept from compare.sh (WOSX_KEEP) on the head, 1042176 walks:
fcpwb-gpu-parallel (the CUDA launch, `_kernel_k_blk_0`) and fcpwb-gpu-optix
(the same code as the ray-generation program `__raygen__k`), the full
section set under kernel replay on the second, timed launch of each run;
then the reordering launch (fcpwb-gpu-optix-ser) on the head and on the
whale, and the plain OptiX launch on the whale. The reports, raw metrics
and SASS listings are results/ncu/optix/ (not in git); the scripts that
read them, scratch/ncu-optix/.

| head, the timed launch | CUDA | OptiX |
|---|---|---|
| duration | 15.7 ms | 21.3 ms (1.36x) |
| block x grid | 256 x 4071 | 128 x 8142 |
| registers a thread | 50 | 67 |
| achieved occupancy | 51.8% | 45.5% |
| warp instructions | 18.6 G | 23.6 G (1.27x) |
| lane instructions | 55.1 G | 57.0 G (1.03x) |
| lanes active per warp instruction | 2.96 | 2.43 |
| issue slots busy | 68% | 62% |
| L1 hit rate | 89% | 87% |
| spills | none | none |
| SASS instructions | 1208 | 1080 |

The time is accounted for twice over: 1.27x the warp instructions times
1.10x the issue rate (68% against 62%) is 1.39x, against the measured
1.36x. The per-lane work is the same (1.03x); the warps execute it with
fewer lanes together.

**Where the instructions come from** (both SASS listings by opcode with
their executed counts, scratch/ncu-optix/sass-agg.py). The traversal's
node-visit loop is the same on both sides: 103.8 million warp iterations
at 3.00 lanes (the stack pushes). The leaf's triangle code is not: 66.4
million warp passes at 3.21 lanes under CUDA, 105.1 million at 2.03 under
OptiX -- the same 213 million lane passes. In the CUDA kernel's SASS the
lanes that reach a leaf leave the node loop (BREAK) and wait at a
reconvergence barrier (BSYNC) until the warp's other lanes have reached
theirs, then test their triangles together, as pseudocode:

    loop {                        // one traversal step
        loop { visit node; if leaf: break }   // lanes leaving wait here
        test the leaf's triangles           // with every lane that left
    }

In OptiX's SASS the triangle code is inside the one loop's divergent
body:

    loop {
        visit node
        if leaf: test the leaf's triangles  // this iteration's leaf lanes only
    }

so a lane at a leaf tests its triangles while the lanes still descending
are masked, every iteration; the arithmetic of the triangle test is where
the warp instructions grow (FADD +1.5 G, FFMA +0.87 G, FMUL +0.43 G, the
same loads at 2.67 lanes instead of 3.10: LDG +0.34 G). Three smaller
costs of OptiX's compilation beside it: 300 million BMOV (barrier
registers saved around OptiX's calls -- the division slow paths are
CALL.ABS into OptiX's own copies where the CUDA kernel's are CALL.REL
within the function; 22 lines against none), 133 million LDC (the launch
parameters re-read from constant bank 3 through a register index on every
use, where the CUDA kernel's parameters ride the uniform datapath -- LDCU,
USHF, ULOP3 -- and are read once), and no read-only-path loads (51
LDG.E.STRONG.SM, where the CUDA kernel's 55 are LDG.E.CONSTANT).

**Whose code it is.** The two PTX bodies differ -- the CUDA kernel's
parameters are kernel arguments and the raygen's a constant-memory struct
(`params`), so the launch_readonly analysis (CodeGen_LLVM_SSA.cpp) marks
every load of the CUDA kernel `.nc` and 24 of the raygen's 50: the triangle
vertices' loads reach their array through a pointer read from `params`,
whose root it does not see, an issue of ours listed below; and the CUDA
path's paired-float arithmetic (f32x2) is switched off for OptiX on
purpose, which sm_120's SASS unpacks anyway (no packed instruction on
either side). But the raygen's PTX through plain ptxas, its one OptiX call
replaced by `%tid.x`, is a 48-register kernel with the CUDA kernel's
structure: 30 reconvergence barriers against 29, 8 BREAK against 6, no
BMOV, 28 read-only-path loads, 86 branches against 87. Through OptiX's
compiler the same PTX is 67 registers, 22 BMOV lines, no read-only loads
and the one loop. The overhead is OptiX's compilation of the program, not
our PTX. The thread mapping is ruled out too: fcpwb-gpu-optix-persistent
(the 131072-worker claim under OptiX, no reordering, so a warp's 32 claims
are the same walks as the CUDA persistent kernel's) is 1.04x over WoSX
against the CUDA persistent 1.44x and the plain OptiX launch 1.07x (head,
3 repeats).

**The registers.** The directive the compiler writes for
--gpu-max-registers, `.maxnreg`, OptiX ignores: a raygen carrying
`.maxnreg 48` compiled to 67 registers, the same SASS to the instruction.
What OptiX honors is the module's maxRegisterCount, which the runtime had
left at OptiX's choice; it now reads the directive out of the PTX and
passes it (66b6590d: max_registers_of in runtime/bonsai_optix.h; the
tests backends/ptx/raygen-max-registers and correctness/gpu/
raygen-max-registers; compare.sh's WOSX_FLAGS passes compiler flags for a
run). The cap swept on the head, 3 repeats each:

| --gpu-max-registers | registers OptiX used | occupancy | issue | duration | over WoSX |
|---|---|---|---|---|---|
| none | 67 | 45.5% | 62% | 21.3 ms | 1.07x |
| 32, 40, 48, 56, 64 | 64 each | 52% | 66% | 20.0-20.4 ms | 1.12-1.15x |
| the CUDA launch | 50 | 51.8% | 68% | 15.7 ms | 1.44-1.51x |

OptiX compiles to 64 under any cap below it -- its floor -- and the
instruction count does not move (23.6 G), so the register allocation is
part of the answer, five to seven percent of the launch, and the loop
structure the rest, which no option reaches. The same commit makes
pbrt's --gpu-max-registers 128 real on its OptiX path for the first time
(bonsai-f9: its raygens ran at OptiX's own 128, the SER megakernel's
pipeline at 255); whether pbrt's figures run capped or at OptiX's choice,
as pbrt itself does, is bonsai-f9's question to the user.

**The reordering launches.** A pipeline with shader execution reordering
reports to Nsight Compute as one block of 512 threads at 255 registers,
OptiX's execution mode for it (pbrt's profile saw the same shape); the
counters are the hardware's. On the head against the plain OptiX launch:
4.79 lanes active per warp instruction against 2.43 and 16.2 G warp
instructions against 23.6 G (fewer than the CUDA kernel's 18.6 G) -- the
coherence the reordering promises -- but 32.7% warps active against
45.5%, 0.72 eligible warps a cycle against 1.35, 47.7% issue against 62%,
L1 hit 76% against 87% (a point's walks no longer share a warp's cache
lines), and the SMs active for 78% of the launch's elapsed cycles against
94%: 22.5 ms against 21.3. On the whale (the walk on spheres', 309056
walks, the table's 0.83x): lanes 3.71 against 2.20, instructions 0.76x,
warps active 31.9% against 43.6%, the SMs active 60% of the elapsed cycles
against 90%, 24.0 ms against 17.1. The reordering buys its coherence and
loses it to the SER mode's thinner residency and to SMs with nothing to
run, the more the smaller the launch -- the table's correlate: the three
launches of 150 to 305 thousand walks lose, the million-walk launches
gain where a step is three traversals (the stars). The stars' whale
(0.35x) was not profiled; the same mechanism over a longer tail is the
reading, not a measurement.

### Why the packets lose on the CPU, and the deferred drains (2026-10-29)

The user, on the CPU table's packets at half the scalar walk's rate:
"investigate why packet tracing does not [give] speedups over WoSX on
CPU. Perhaps it is due to path lengths, or maybe we're generating bad
codegen. Maybe a defer + vectorize would work to compact paths of similar
length and still benefit from vectorization?" Three hypotheses, each
measured on the head (17k triangles, 16284 points, 64 walks, 13.7 steps a
walk) and the rover (1.6M triangles, 17.5 steps), sixteen threads under
WoSX's schedule, the table's settings, least of three; then the
deferral the user proposed, built as schedules and measured.

**How the lanes were counted.** The compiler's LLVM module of each
program, instrumented after the fact (scratch: `instrument.py`, a
counter pair per basic block -- passes, and the popcount of a chosen
sixteen-lane mask live at its top -- as atomic adds on globals printed by
a destructor; assembled by the compiler's own llc, linked with the
driver as compare.sh links it, run on the mesh with one repeat so the
counts are two runs' worth, the warm-up's and the timed one's): the
walk loop's header with the gang's live mask; the traversal loop's
header with the popped node's mask (the per-lane bound against the
lane's best is re-tested right after it, so this counts a pop before its
cull, as the scalar's header count does); the leaf's entry; and the
triangle loop's body with the mask the test is made under. Beside the
counts, `perf stat` for the instructions of each whole driver run (WoSX's
side the same work in every run, about 153 G instructions, so the
difference between runs is ours), and `perf annotate` of the gang's
traversal for its instruction mix.

**The head, per lane query (23.8 M queries in the two runs):**

| | scalar fcpwb-threads | packet, gangs of 16 | sorted drain (below) |
|---|---|---|---|
| lanes live at the walk loop's header, of 16 | -- | 5.6 | 15.75 (the drain's gangs) |
| node passes (the gang's or the scalar's visits) | 398.9 M | 639.7 M | 311.1 M |
| lanes at a node pass | 1 | 2.32 | 3.98 |
| a lane's node passes, against the scalar's | 1 | 3.7x | 3.1x |
| leaf passes / lanes at a leaf | 141.1 M / 1 | 151.1 M / 1.66 | 82.1 M / 2.46 |
| a lane's leaves, against the scalar's | 1 | 1.78x | 1.43x |
| triangle passes / lanes at a test | 451.6 M / 1 | 480.2 M / 1.67 | 259.7 M / 2.48 |
| our instructions, the driver's run | 97.7 G | 182 G | 93 G |
| Msteps/s (WoSX 14.9-15.0) | 17.4 | 9.9 | 17.7-18.0 |
| over WoSX | 1.17x | 0.66x | 1.19-1.21x |

**The rover, the same (26.3 M queries):** scalar 843.7 M node passes
(32.5 a query), 999.4 M triangle tests; packet 2889 M node passes at
1.62 lanes (a lane's passes 5.5x the scalar's), 1799 M triangle passes
at 1.32 lanes (2.4x); sorted drain 1632 M at 2.36 lanes (4.6x), 1177 M
at 1.70 (2.0x). Rates: WoSX 6.1-6.2, threads 7.14 (1.17x), packet 2.38
(0.39x), sorted drain 3.57-3.76 (0.58-0.61x).

**(1) The walks' lengths.** Real, and not the cost. A gang of sixteen
of a point's walks runs as many passes of the walk loop as its longest
walk has steps, and at the loop's header 5.6 of the sixteen lanes are
live by mean (the terminal pass of an ended walk counted): a third of
the gang's passes are for all sixteen, the rest for the tail. The
deferral compacts exactly this -- `solve_all.defer(walk, steps)` and
`walk.defer(walk, steps)` put every step on a queue, the drain's round r
runs the r-th step of every walk still going, and its gangs are full
(15.75 of sixteen) -- and changes nothing: a point's queue
(`fcpwb-packet-defer-threads`) runs the head at 10.2 Msteps/s against the
packet's 9.9, 0.68x over WoSX against 0.66x, and a chunk of sixty-four
points' queue (`fcpwb-packet-defer-chunk-threads`, the gangs drawn from
4096 walks a round) 0.66x. The walk loop's empty lanes cost nothing
because the pass's work is the traversal, and the traversal's lanes are
set by where the sixteen walks stand, not by how many are live.

**(2) The lanes' spread, which is the cost.** At a node of the packet's
traversal 2.32 of sixteen lanes are on, at a triangle test 1.67 -- the
reading of 10-09 ("a lane or two"), now counted. The gang visits 640 M
nodes for the scalar walk's 399 M, so a sixteen-wide instruction stream
does 1.6x the passes of the scalar one, each pass for two lanes. The
fix the user proposed, in its sorted form: `walk.sort(walk, |x, problem|
walk_sort_key(x, problem.lo, problem.hi))` on the deferred call -- the
sort on a call (docs/schedule-semantics.md), the directive the GPU's
reordering schedule writes for OptiX's shader execution reordering, here
an exact sort of the chunk's queue by the walk's position quantized to a
32x32x32 grid over the problem's box (walk_keys.bonsai) before each
round's drain, so a gang's sixteen consecutive entries stand in one cell
and ask the tree about one neighbourhood. `fcpwb-packet-defer-sort-
threads`: lanes at a node 2.32 -> 3.98, node passes 640 M -> 311 M,
triangle passes 480 M -> 260 M, the head 9.9 -> 17.7 Msteps/s, 0.66x ->
1.19x over WoSX, at the scalar threads' 1.17x; the comb 2.05x (the
packet's 2.03x: its six-step walks stay together without a sort); the
rover 0.39x -> 0.59x, lanes at a node 1.62 -> 2.36. A finer key (ten
bits an axis, a 1024x1024x1024 grid: `walk_sort_key_fine`, the
`-sort-fine-` schedules) and chunks of 256 points (16384 walks a round)
move neither mesh (head 1.21x, rover 0.58x and 0.61x): the lanes in a
cell are as near as the key can put them, and on a tree of a million
triangles the walks of one cell part in its first levels. The sort's
cost is under a percent of the run (`bonsai_sort_indices` is below
perf's 0.7% floor; the drain kernel with the walk step in it too).

**(3) The code.** Not bad, and not free. What the compiler makes of the
packet is what the schedule says: the traversal one node for the gang
(no gathers in it -- the node's row is read once and broadcast, 107
`vbroadcastss` sites at 10.6% of the gang function's cycles: twelve
coordinates of two children's boxes a visit, nine of a triangle's
corners a test, each broadcast to serve the two or four lanes that are
on), per-lane bounds and masks on the stack as masked stores, the
children ordered by a vote of the lanes' keys; the walk step vectorized
whole -- the generator's 64-bit multiplies, the direction's sine and
cosine as sixteen-wide polynomials with a sixteen-call scalar path for
arguments outside the range, never taken for an angle in [0, 2 pi) --
and the leaf's skip a uniform test. The instruction mix of the sorted
drain's traversal: compares 12.7%, mask arithmetic (`kortestw`, `korw`,
`kandw`, `kmovw`) 17%, broadcasts 10.6%, multiplies and fused
multiply-adds 14.6%, register moves 6.7%. Counted whole: the sorted
drain runs the head in 93 G of our instructions against the scalar
walk's 97.7 G and the packet's 182 G, for 311 M vector node passes
against 399 M scalar visits -- a gang's pass costs about 1.4x a scalar
visit's instructions and carries four lanes, and the two cancel to the
parity the table shows. The one thing in the generated traversal that
costs a lane work the scalar does not do is the order: a lane in the
minority of the vote enters the farther child first, its bound stays
loose, and it reaches 1.43x the leaves the scalar walk reaches for the
same query (1.78x in the packet, where sixteen unrelated walks vote),
and pops 3.1x the nodes, most of them culled at the pop by the per-lane
bound (the stack holds a lane's distance to the node, Embree's
per-ray entry, re-tested against the lane's best; the leaf and triangle
excess, 1.4x, is the order's real cost, the node excess mostly the
pops). That is the packet traversal's rule and not a bug; a per-lane
order needs a per-lane stack, which is the per-lane drain (Next).

**The deferral's schedules.** `fcpwb-packet-defer-threads` (a point's
queue), `fcpwb-packet-defer-chunk-threads` (a chunk of sixty-four
points'), `fcpwb-packet-defer-sort-threads` (the chunk's, sorted by the
fifteen-bit key), `fcpwb-packet-defer-sort-fine-threads` and
`-sort-fine-c256-threads` (the thirty-bit key, chunks of 64 and 256),
`fcpwb-packet-defer-sort-c16-threads` (the fifteen-bit key, chunks of
16); all compute WoSX's estimate (the walks ended, the lengths and the errors
equal in every run). A queue above the point loop wants the nest
perfect: `walk.defer(walk, steps)` with `queue(i_chunk)` is refused
(Defer.cpp:1817, "the loop i_in of solve_all goes on after its inner
loop w ... Write it as a loop of its own after the nest") until
`solve_all.reorder(i_in, w); solve_all.reorder(w, i_in)` distributes the
point's prologue and epilogue -- the sums' zeroing, the division by the
walks that ended, the stores -- around the (point, walk) nest, the GPU
deferral's idiom (fcpwb-gpu-defer.bonsai). The per-lane form,
`fcpwb-packet-defer-sort-fine-perlane-threads` (the traversal loopified
before the drain is vectorized), stops the compiler (Next). The table's
defaults carry the point's queue and the sorted chunk's; the CPU figure
draws them in place of the gangs of thirty-two and of four, which the
table keeps and the palette's nine colours do not hold (plot.py says
which; one list edit draws them again).

**The table** (benchmark.sh, 09:30-09:54, the two schedules' 36 cells
resumed into the CPU table, every cell WoSX's estimate; the figure
redrawn). Over WoSX by geomean on 18 meshes:

| CPU, WoSX's default tree | geomean | min | max |
|---|---|---|---|
| fcpwb-threads (matching) | 1.17x | 1.06 dragon | 1.30 comb |
| fcpwb-packet-16-threads | 0.55x | 0.39 rover | 2.03 comb |
| fcpwb-packet-defer-threads (a point's queue) | 0.55x | 0.38 rover, dragon | 2.11 comb |
| fcpwb-packet-defer-sort-threads (a chunk's, sorted) | 0.71x | 0.41 crown | 1.90 comb |

The point's queue is the packet to the cell (0.38-0.90 against
0.39-0.83 on the seventeen losing meshes, the comb 2.11x against 2.03x):
the compaction is worth nothing anywhere. The sorted chunk gains on
every mesh but two: it beats the scalar threads on the pavilion (1.34x
against 1.18x, 11k triangles) and the head (1.19x against 1.17x), stands
at 0.73-0.82x on the mid-sized meshes (ivy 0.82, zero-day 0.78, whale,
villa 0.75, car 0.73) against the packet's 0.47-0.66x, and at 0.45-0.49x
on the three largest (lte-orb, ganesha, dragon) against 0.39-0.41x --
the larger the tree, the less a cell of the key gathers. The two it
loses on: the comb (1.90x against the packet's 2.03x: its six-step
walks stay together on their own and the sort is a cost), and the crown
(0.41x against 0.83x), which is the chunk, not the sort: the crown's
slice has 460 sample points, eight chunks of sixty-four for sixteen
threads, and half the threads idle. Chunks of sixteen points
(`fcpwb-packet-defer-sort-c16-threads`, 29 chunks) put the crown at
0.81x and the head at 1.10x (the sort's reach a quarter of the walks);
the chunk is a per-mesh trade between the threads' balance and the
gang's coherence, and the table's schedule keeps sixty-four.

**What the numbers say.** The vectorized walk on WoSX's default tree
loses to the scalar one for the gang's emptiness at the tree's nodes,
two to four lanes of sixteen, which the walks' lengths do not cause and
the deferral's compaction does not mend; sorting the drain by position
fills the gang to four lanes and brings the vectorized walk to the
scalar threads' rate on a small tree (the head, the comb's six-step
walks beyond it) and to 0.6x on a large one; the rest of the gap is the
vote's order (1.4x the leaves) and the vector pass's own cost (1.4x a
scalar visit's instructions). The CPU's best bars on this tree stay the
scalar ones, fcpwb2-persistent at 1.27x. The two ways left to a
vectorized walk that pays are the per-lane order and the hybrid switch
(Next).

### The per-lane drain compiled, and the vote's order measured (2026-10-29, afternoon)

The user: "Please fix that bug, and see what schedules can be made to
alleviate these issues. Be explorative!"

**The bug.** The per-lane form of the sorted drain -- the closest-point
traversal sorted and put on its stack *before* the drain is vectorized,
so that each lane walks the tree over a stack of its own -- stopped the
compiler in the LLVM backend's lane extract. The cause, read off the final
SSA: a lane that walks on its own reads the leaf's Triangle object through
its own pointer (FCPW's leaf holds pointers to its objects; the layout's
`ptr group objects`), and the vectorizer widened that reference to one
array handle per lane, `const _tree_layout2[]x16`, a type nothing after it
knew. The element read of it was typed as the vector of handles itself,
the backend took the handles for a value in registers and asked for a lane
of them, the C++ backend's path refused the address of an element of it,
and the linearizer -- whose rule masks a gather by its *index* varying --
left a read at per-lane addresses unmasked whatever the lanes held: once
the first two were mended the program compiled and faulted, a `vpgatherqd`
under an all-ones mask through the off lanes' handles. Fixed at the root
in the compiler (the commit names each piece): the type named
(Type::is_reference_per_lane), the element read typed as the gang-wide
form of the array's element and gathered from the lanes' bases a field at
a time, the x86 gather's one-base form bypassed for a base per lane, the
address of an element of a per-lane handle one pointer per lane also when
the element is an aggregate, and the linearizer masking a read through a
handle that varies and a load through a pointer that varies, whatever the
index. The same handle turned out to be what a packet traversal's best
hit leaves each lane with, dereferenced after the walk, which is why
tests/bonsai/correctness/cpp/bvh4_ptr_arena_packet, written on 10-09,
never passed: it passes now, and with ssa/ptr-rows-packet and
backends/llvm/ptr-rows-packet pins the form (the tile's address a `gep` on
the vector of handles, every load through it a masked gather). The
per-lane arena test itself hits an older limitation of the divergence
analysis and sits in tests/bonsai/pending (Next).

**What the per-lane drain measures** (`fcpwb-packet-defer-sort-fine-
perlane-threads`, the sorted chunk's drain with each lane's own order; the
head, 16 threads, 3 repeats): 8.0 Msteps/s, 0.54x over WoSX, against the
sorted gang's 17.7-18.0 (1.19-1.21x) and the plain packet's 9.9 (0.66x),
WoSX's estimate to the walk. So the vote's order is not what to fix: a
lane keeping its own nearer-first order saves the 1.4x of leaves the
majority's order cost it, and pays for it with a gather at every node --
the node's row read from sixteen addresses instead of one broadcast, 237 G
of our instructions for the run against the sorted gang's 93 G (2.5x) --
and the gang shares nothing but its instruction stream. On a CPU the
shared node is the packet's whole gain, and a traversal whose lanes
disagree about the order is better served by lanes that agree, which is
the sort, than by lanes that each go their own way.

**A gang of eight for the sorted drain** (`fcpwb-packet-defer-sort-8-
threads`: the same sorted chunk, the drain in gangs of eight), the Next
list's question -- four lanes of sixteen live at a node pass would be four
of eight: the head 1.03x over WoSX against the gangs of sixteen's 1.19x,
the rover 0.58x against 0.59x, the comb 1.37x against 1.90x. Sixteen wins
everywhere: a gang of eight runs twice the passes for the same lanes, and
a 256-bit pass costs what a 512-bit one does, so the fuller gang buys
nothing and the halved width is paid in full. The vector width is the
right gang, and the lanes it holds are the sort's to fill.

**Where the exploration stands** (over WoSX, the head / the rover): the
scalar threads 1.17x / 1.17x; the packet of sixteen 0.66x / 0.39x; the
walks deferred and compacted 0.68x / 0.38x; the sorted drain 1.19x /
0.59x (gangs of eight 1.03x / 0.58x; the thirty-bit key 1.21x / 0.58x;
chunks of 256 1.21x / 0.61x; chunks of 16 1.10x / --); the per-lane
sorted drain 0.54x / --. The sort is the one directive that moved the
vectorized walk, by 1.8x; nothing after it moved it further, and the two
forms that attack the vote's order (per-lane) and the gang's emptiness
(eight lanes) each cost more than they return. A vectorized walk on WoSX's
default tree that beats the scalar one wants a shared node *and* lanes
that agree on the order *and* a leaf test that is not sixteen wide for
two lanes -- the hybrid switch, which remains the one schedule word not
yet in the language (Next).

### The unsigned lane offsets measured on the CPU (2026-10-30, evening)

The compiler's fix of the morning (caa9eb9b, found by the pbrt packet
schedules on bistro and landscape) changed how a gang's gather forms its
addresses. A gang reads one element per lane at `base + offset[lane]`,
the offsets 32-bit numbers formed by the program's unsigned arithmetic,
and the machine reads a 32-bit gather index as signed:

    before:  addr[lane] = base + signed(offset[lane])      -- an offset of 2^31 or more lands below base
    after:   addr[lane] = (base + 2^31) + signed(offset[lane] xor 2^31)
                                                           -- the same address for every offset below 2^32

Every gather over 32-bit offsets now carries one xor and a differently
formed base, and its address computation is no longer marked in bounds,
so the question for the CPU figures was which bars' code moved. Compiled
under the new compiler and counted: `fcpw4w16-threads` and
`-persistent` have no gathers at all (the four-wide tree's lanes read a
node's children as one row), nor do `fcpwb-packet-16-threads` and the
scalar `fcpwb-threads`; `fcpwb-packet-defer-threads` has five gathers,
all over 64-bit offsets, and its code is yesterday's line for line;
`fcpwb-packet-defer-sort-threads` has fourteen, one of them over 32-bit
offsets -- the one place in the drawn CPU schedules the fix touches. The
GPU backends do not form addresses this way, so the GPU tables stand.

**Measured** (18:17-18:53, the four schedules over the nineteen meshes,
three repeats each, the machine otherwise idle): nothing moved. Every
cell came back within 2.6% of its 10-28/29 value and the geomeans to two
decimals -- `fcpwb-packet-defer-sort-threads` 0.73x -> 0.73x,
`fcpwb-packet-defer-threads` 0.58x -> 0.58x, `fcpw4w16-threads` 1.24x ->
1.24x, `fcpw4w16-persistent` 1.34x -> 1.33x -- so the xor and the
unmarked address cost nothing the clock sees, and the three schedules
whose code did not change repeated their numbers, which is the table's
noise floor for a day. The table keeps the 10-28/29 cells for those
three (their code being the same, and their fresh logs lost to a cleanup
of mine after the run: the run had finished before the cleanup assumed
it was still going) and today's cells for the sorted drain; the previous
table and the sorted drain's previous logs stand beside it as
`wosx-results.before-2026-10-30-unsigned-offsets.csv` and
`logs-before-2026-10-30-unsigned-offsets`. The two CPU figures are
redrawn from it; all five wosx figures stand on the current compiler.

### The compacting block wavefront, on both walks (2026-10-30, night)

The user asked for a compacting block wavefront for the walk on spheres
and the walk on stars, and whether shared memory is large enough for it.
The shape: a chunk of points per block, the chunk's walks the block's
threads, their steps deferred onto one queue in the block's shared memory
and drained in rounds inside the kernel, each round's push compacting the
live walks to the front -- fcpwb-gpu-defer-block's per-point queue made a
chunk's, the CPU's fcpwb-packet-defer-chunk-threads brought to the card.
It is a composition of today's directives and compiled first time:
`split(i, i_chunk, i_in, C)`, the two reorders (the point's prologue and
epilogue distributed, the nest perfect, the sums a slot per point of the
chunk), `collapse(i_in, w, k)`, `queue(i_chunk, DeviceShared)`, the two
`defer`s, the chunk to the block and the pushes, the drain and the pass
to the threads (schedules/fcpwb-gpu-defer-chunk{2,4,8,16}.bonsai,
wost-fcpwb-gpu-defer-chunk{2,4}.bonsai). The block's size is the chunk's
walks, 128 to 1024.

**Shared memory.** The RTX 5090 gives a block 48 KB without asking, 99 KB
(101,376 bytes) when the kernel opts in, 100 KB to a multiprocessor. The
launch's dynamic shared memory, read off the host code: the walk on
spheres takes 74 bytes a walk (a 40-byte entry -- position, distance,
step count, generator, the pair -- and a 32-byte one for the two buffers,
plus the record's byte), the walk on stars 100 (two 28-byte halves beside
the 40 and the byte). So a block holds about 650 walks of the one and 480
of the other at 48 KB, 1,300 and 1,000 at 99 KB: ten to twenty points'
walks, which is the answer -- large enough for a chunk, not for a card.
The chunk of sixteen (1024 walks, 76 KB) found a runtime defect on the
way: the opt-in asked CUDA for the device's whole limit, which it refuses
for any kernel that also has static `.shared` words (a block queue's
header, the block's sums), so every launch past 48 KB aborted. Fixed at
the root in runtime/bonsai_cuda.h (the launch's own size, raised when a
later launch asks for more; the limit checked against the static part),
with tests/bonsai/correctness/gpu/queue-shared-large.bonsai; the chunk of
sixteen then ran, 0.69x on the head, the trend's next step down.

**Measured** (head and dragon, 64 walks a point, three repeats, Msteps/s
over WoSX's persistent kernel; the same-day references in the same
column):

| schedule | head | dragon |
|---|---|---|
| persistent (fcpwb-gpu, the matching schedule) | 1.45x | 1.44x |
| persistent-reorder / reorder-256k (the table) | 1.60x / 1.75x | 1.69x / 1.98x |
| defer-block (a point a block, 64 walks) | 1.21x | 1.44x |
| defer-chunk2 (128 walks) | 1.18x | 1.34x |
| defer-chunk4 (256) | 0.95x | |
| defer-chunk8 (512; one kernel, 28.4 ms) | 0.79x | |
| defer-chunk16 (1024, 76 KB of shared memory) | 0.69x | |
| defer (card-wide queue, a launch a round) | 0.79x | 0.77x |
| stars: persistent / defer-block / chunk2 / chunk4 | 1.64x / 0.92x / 0.84x / 0.43x | |

Over the tables' eighteen drawn meshes (the GPU figures, redrawn 22:37
and 22:47 with the two bars): the one-point block 1.17x by geomean on the
walk on spheres (0.78x on the crown to 1.27x on the dambreak), the chunk
of two 1.11x (0.68x on the beast); on the stars 0.99x and 0.87x (0.51x on
the crown). Against the persistent claim, bar for bar: 0.85x and 0.80x on
the walk on spheres, 0.64x and 0.57x on the stars. The landscape, left out
of the figures, is the one mesh where the block forms win (1.88x and
1.79x; 3.47x on the stars): its walks escape the box in a step, mean
length 1.15, so there is no tail to hold the block and every lane's one
step is the block's whole work.

**Reading.** The compacting block loses to the per-lane persistent claim
on both walks, and loses more the more walks it holds. A block's rounds
end with its longest walk: in round r the lanes live are the walks longer
than r, so the block's lane use over its life is about the mean walk
length over the longest in the block -- 13.7 over a few hundred for the
walk on spheres, 78 over a thousand (the cap) for the stars -- and the
longest grows with the chunk while the mean does not. The block holds its
multiprocessor's slot through that tail, where a persistent lane whose
walk ended claims the next walk at once and the hardware's scheduler
replaces a finished small block at once (which is why the one-point block
and the chunk of two come nearest, and the stars, with the longer tail,
fall furthest). The compaction itself works -- the statistics match on
every cell -- it compacts within too small a population. The card-wide
queue compacts across every walk and loses the same 0.8x to its launch a
round and its entries' trip through L2. What would make the block form
win is the refill: a block that claims new points into its shared queue
whenever its live count falls under its lanes, so the tail of one chunk
runs under the first steps of the next and the lanes stay full to the end
of the work -- the per-lane persistent claim with the block's compaction
and no entry traffic past the SM. That is not a composition of today's
directives (`persistent` claims a whole iteration; the claim inside the
drain's round is new), and a first half of it, `persistent` on the chunk
loop with the block as the worker (scratch/fcpwb-gpu-defer-chunk2-
persistent.bonsai, kept out of schedules/ since it does not compile), the
compiler refuses today: the claim's fetch-and-add stands in
the block loop's body outside its thread phases, where every thread would
run it (CodeGen_LLVM_SSA.cpp:608, "not built yet" -- one lane fetches,
the block reads it from shared memory after a barrier). Both are the
user's to open; the measurements say the refill is the one worth having.

**The refill, corrected (the user, the same night).** The refill is a
composition of today's directives after all: `persistent` on the
collapsed (walk, point) loop with the workers split into blocks -- each
thread claims the next pair from the counter, as the matching schedule
does -- and the pairs' first steps deferred onto a queue placed at the
block loop in shared memory with a *given* size, `queue(t_blk, 256,
DeviceShared)`, since a claim loop has no trip count to size the queue
by. What is unbuilt is one lowering: a bounded queue's flush. Today a
written size may only enlarge the queue (the deferral refuses a size
below the pushes it can count: "a capacity below it needs a flush
policy, which is not built"), so a queue always holds every push and the
drain runs after the last of them. The refill needs the pushes to stop
when the queue is full, a round to run, and the pushes to resume -- the
policy the defer design of 09-19 foresaw ("triggers a drain before a
write that would overflow"). Proposed to the user, who wants a plan
approved first: in a block, rounds of {drain the count into the other
buffer; each lane claims one pair and pushes its first step while the
counter has pairs and there is room for every lane; swap} until the
counter is spent and the queue is empty; the compiler refuses a size
below the block's lanes (the one way to spin: nothing ever claimed, the
counter never spent); a round never fills the queue further (a step
pushes at most one next step, enforced); a walk ends within the program's
step cap, so room appears. The proposal's schedule is
apps/wosx/scratch/fcpwb-gpu-defer-refill.bonsai (gitignored, since the
compiler refuses it). Not to be built without the user's approval.

**Three more without the bounded queue** (the user: "what else can you
try without this?"; head / dragon over WoSX, three repeats, the same
night; execution schedules over FCPW's tree as WoSX runs it -- the user
ruled the same night that this work explores execution schedules only
and the tree is matched exactly, so two layout variants tried on the GPU
that night were withdrawn and are no part of the comparison):

| schedule | head | dragon | what it is |
|---|---|---|---|
| wost-fcpwb-gpu-defer | 1.33x | 1.57x | the card-wide queue on the stars (its launches a few percent of a second's run) |
| wost-fcpwb-gpu-defer-sort | 1.13x | 1.42x | the same, the queue sorted each round by the walks' positions (the device radix sort) |
| fcpwb-gpu-defer-sort | 0.75x | 0.75x | the walk on spheres' card-wide queue sorted each round (unsorted 0.79x / 0.77x) |
| (the references) persistent / reorder-256k | 1.45x / 1.75x | 1.44x / 1.98x | the stars 1.64x / 1.74x and 1.63x / 1.88x |

The card-wide queue on the stars is the one of the three worth a bar,
and the table shows two faces of it. On the two meshes above it stands
between the block forms and the claim (1.33x against defer-block's 0.92x
and the claim's 1.64x on the head; 1.57x against 1.05x and 1.63x on the
dragon): the compaction across every walk worth what the long walks'
divergence cost, the launches cheap at a second a run. Over the
nineteen meshes (measured 22:55-23:05, the stars' GPU figure redrawn
with the `defer` bar) it is 0.82x by geomean over the eighteen drawn, the
best of the twelve schedules on the rover (1.97x) and ahead of the claim
on the ivy (1.89x) and the dambreak (1.51x), and a quarter of WoSX's
rate on the small meshes -- the whale 0.23x, the crown 0.28x, the
octopus 0.34x, the beast 0.35x, the comb 0.41x -- where a run is a few
tens of milliseconds and the hundreds of rounds, each a launch over a few
thousand live steps, are the run; the head and the dragon of the table
(1.31x, 1.57x) repeat the single runs. Sorting the queue each round costs
more than the coherence returns on both walks (the stars 1.33x to 1.13x
on the head, 1.57x to 1.42x on the dragon; the spheres 0.79x to 0.75x) --
a radix sort of up to a million entries per round, a few hundred rounds.
So no schedule of the night passes the persistent reorder with 262144
workers by geomean; the card-wide queue on the stars is the first of the
deferrals to beat the claim on any mesh of size, and what it wants is the
small meshes' rounds without their launches -- the block form's rounds
inside the kernel with the card-wide form's population, which is the
refill (above). The sorted drains are measured by name.

### The sorted drain with persistent claims, and the order of two directives (2026-10-31, night)

The user's question of the evening, "is there a good persistent-defer
schedule that could be tried?": `fcpwb-packet-defer-sort-persistent`, the
sorted drain (fcpwb-packet-defer-sort-threads) with its chunks of
sixty-four points claimed by sixteen persistent workers from a counter
instead of handed out by TBB's partition -- `persistent(i_chunk, t, 16);
bind(t, CPUThread)` for `bind(i_chunk, CPUThread)`, nothing else changed.
A worker's queue is still one chunk's; the refill that would keep its gangs
full across chunks, new points claimed into the queue as walks end, is the
bounded queue (above), not built.

Written first with the traversal's sort and loopify ahead of the
vectorize, it ran at 02:26 (f9's 'clear' for the hour before their GPU
run; two cells, seventy seconds) at 0.54x on the head and 0.50x on the
rover, where the TBB drain runs 1.20x and 0.58x. The final SSA
(`--dump-ssa-final`) said why: the gang's box test took an `AABB$v16`, a
node per lane; the recursion had no multicall; the LLVM held 147 gathers
against the threads file's 14 -- the per-lane traversal of
fcpwb-packet-defer-sort-fine-perlane-threads (10-29: 0.54x on the head,
the same number), which is what vectorizing an already loopified traversal
makes: every lane its own stack and its own node, each node load a gather.
Directives apply in the order written (the ruling of 10-08); the threads
file vectorizes first, so its sort votes by majority and its gang visits
one node at a time. The file was reordered to the threads file's order with
the claim alone changed, and checked under build/'s compiler before the
second run: 14 gathers, the box test on one `AABB`, the recursion by
multicall. The other eighteen CPU schedules were audited for the order:
the two written sort-first are the four-wide tree's matching schedule
(the vectorize over a node's four children, FCPW's own form) and the
perlane file, both on purpose. The 02:26 cells are kept under
results/logs-before-2026-10-31-sort-order.

Measured 03:25 to 03:36 under the stall rule (f9's lock off since 03:03
with no 'done', nothing running, nothing of theirs changed in twenty
minutes, their session waiting; f9 told before and after), least of three,
the nineteen meshes; speedup over WoSX:

| mesh | persistent sorted drain | TBB sorted drain | persistent (scalar) | threads (scalar) |
|---|---|---|---|---|
| geomean, 19 | 0.767x | 0.733x | 1.264x | 1.195x |
| comb | 2.31x | 1.99x | 1.30x | 1.30x |
| pavilion | 1.35x | 1.34x | 1.18x | 1.18x |
| landscape | 1.23x | 1.29x | 1.97x | 1.89x |
| head | 1.22x | 1.20x | 1.18x | 1.17x |
| ivy | 0.85x | 0.84x | 1.23x | 1.21x |
| whale | 0.80x | 0.76x | 1.16x | 1.15x |
| zero-day | 0.79x | 0.79x | 1.23x | 1.20x |
| villa | 0.77x | 0.75x | 1.19x | 1.17x |
| car | 0.76x | 0.74x | 1.17x | 1.15x |
| octopus | 0.72x | 0.65x | 1.18x | 1.15x |
| sportscar | 0.72x | 0.70x | 1.19x | 1.17x |
| bmw | 0.69x | 0.67x | 1.18x | 1.16x |
| dambreak | 0.65x | 0.60x | 1.20x | 1.17x |
| rover | 0.61x | 0.58x | 1.22x | 1.17x |
| ganesha | 0.59x | 0.45x | 1.39x | 1.16x |
| beast | 0.53x | 0.58x | 1.19x | 1.18x |
| lte-orb | 0.49x | 0.47x | 1.42x | 1.10x |
| dragon | 0.48x | 0.44x | 1.46x | 1.06x |
| crown | 0.42x | 0.41x | 1.19x | 1.14x |

The claim is worth four to five percent over TBB's partition on the drain
(0.767x against 0.733x), as it is worth six on the scalar walk (1.264x
against 1.195x), on every mesh but the landscape and the beast. It makes
the schedule the best of the tree on three meshes -- the comb, where the
sorted drain already led the scalar claim (1.99x against 1.30x), the
pavilion and the head -- and leaves it a loser by geomean: the drain's
deficit on the large meshes (the crown 0.42x, the dragon 0.48x, the lte-orb
0.49x) is the drain's own, the vote's order against a lane's own (the lane
counts of 10-29), and where the chunks come from does not touch it. By the
figure's rule (every schedule that is the best on some mesh is kept) it is
the fcpwb CPU figure's seventh bar, `persistent-defer-sort16`, after the
persistent bar.

The compiler commit of 02:19 the same night (b72c215b: the hoist of
loop-invariant allocations moving pure size arithmetic with the
allocation; the hit continuations keeping the raygen shape around a
reordering hint) changes no wosx code: fourteen defer schedules -- the two
CPU drains and every GPU deferral on both walks -- print identical final
SSA under the 10-29 compiler (build-rtq) and the new one (the hoist runs
before the dump, Convert.cpp), the CPU drains' queue arrays being at the
worker's entry already, sized from the walks count. The tables stand.

The compiler commit of 00:24 on 11-01 (e08e0195: the generated header
states each extern queue array's capacity as an expression over the
exported function's parameters, for pbrt's driver) changes no wosx code
either, checked at the level the change reaches. All sixty-three schedule
files compiled under the 10-29 compiler and under e08e0195's build
(compile only, under a lock of this session's, 00:25-00:28), three outputs
each -- the final SSA, the LLVM module and the generated header. Twenty
are byte-identical in all three. The other forty-three differ in ways that
predate this commit: the thirty-four GPU schedules' headers differ in one
line, the path of the sort library the compiler was built beside
(build-rtq against build); nine CPU packet schedules' LLVM differs by the
unsigned lane offsets of caa9eb9b (the index flipped by 2^31 against a
base moved the same way), the 10-30 change the 10-30 tables were measured
under; and the persistent sorted drain's final SSA differs by b72c215b's
hoist alone -- its twenty-four queue arrays, sized sixty-four times the
walks count, moved from the claim loop's body to the worker's entry --
the compiler its table of 03:25 was measured with (its new SSA is
byte-identical to that night's check dump). The fourteen defer schedules
dumped under the 10-30 23:45 compiler for the hoist check print identical
SSA and LLVM under e08e0195's build. The wosx driver reads no capacity
from the header; no wosx schedule puts a queue in extern storage, the one
place the new line is printed. The tables stand.

The compiler commit of 18:2x on 11-01 (6488f426: the vectorizer clones a
callee for each constant argument its control depends on, and makes a
foreign call with varying arguments once per active lane) does reach the
wosx packet schedules, and the same check (63 files, three outputs, compile
only, 18:32-18:34) shows how far. Forty-nine are byte-identical: every GPU
schedule, the four-wide tree's, the scalar CPU walks and the stars. The
fourteen `fcpwb-packet-*` files differ, five of them the fcpwb CPU figure's
bars (packet16, packet8, defer16, defer-sort16, persistent-defer-sort16).
The change is the entry into the packet traversal. Before, one visit
function served every node, the entry call passing the root's index as its
argument:

    visit(node = 0, sphere, best):        -- node is a parameter
        box   = boxes[node]
        left  = node + 1
        right = node + offsets[node]
        ... recurse into left and right ...

After, the entry call gets its own clone with the constant folded in, and
the inner recursion, where the node varies, is the function it was:

    visit_root(sphere, best):             -- the clone for the entry call
        box   = boxes[0]
        left  = 1
        right = offsets[0]
        ... recurse as before ...

So the final SSA differs by thirty to forty-six lines (the clone's name,
`extract_idx(_, 0)` for `extract_idx(_, triangles_index)`, two adds gone),
and the LLVM modules by 1200-1826 lines -- but after normalizing value names
and numbering, by four to six: one temporary renamed and one select moved,
because LLVM had already folded the root's constant at the call. Lowered to
object code with the compiler's own LLVM (llc -O3 for the host CPU, as the
compiler emits it) and disassembled, the eight plain packet schedules are
byte-identical in their text, among them packet16, packet8 and defer16; the
six sorted drains, among them defer-sort16 and persistent-defer-sort16,
have the same text size and the same instructions, with one block of some
fifteen instructions in the root visit allocated to other registers (r8 for
rdi, k1 for k2, a zeroing moved). That is one visit per query, in register
names; nothing a least-of-three measurement resolves. The tables stand.

The compiler commit of 20:52 on 11-01 (4e3f074c: the OptiX pipeline is
linked for the depth of the traces its programs make, and its continuation
stack sized for that depth -- two where a hit or miss program traces in
turn, as pbrt's fused schedules' shadow ray does, one otherwise) reaches
the wosx GPU drivers through runtime/bonsai_optix.h, which they compile,
so the check ran again (63 files, compile only, 21:47-21:49, the lock taken
over from a peer's stalled hold under the twenty-minute rule). Fifty-six
schedules byte-identical in all three outputs; the seven OptiX schedules
differ in four lines of host LLVM each: the load call passes the module's
trace depth as a third argument, and it is 1 for every one of them (wosx
traces from the raygen program alone; its hit programs trace nothing). In
the runtime the link's depth was the constant 1 and is now that argument,
and the stack expression for depth one is, term for term, the one it was.
The OptiX bars of both GPU figures (10-28) were measured on this pipeline.
The tables stand.

The compiler commit of 06:59 on 11-02 (1ab9281e: a gang's short vectors in
memory, and a scalar accumulated into a vector), checked the same way at
07:01-07:08 -- compile only, under a peer's hold, the compiler binary's
checksum the same before and after. Forty-three of the sixty-three schedule
files are byte-identical in final SSA, LLVM module and header: every GPU
schedule and every scalar CPU one. The commit's correctness fix -- a vector
scaled by a scalar through an accumulate (`tMax *= 1 + 2 * gamma(3)` in
stdlib/intersects) had scaled one component of the three -- does not reach
wosx: the walk's tree is fcpw's own (apps/rtq/metrics/fcpw_scalar.bonsai,
with its own box and triangle tests), and no file wosx compiles accumulates
a scalar into a vector or stores a per-lane vector into an array of vectors
(wosx, wost, walk_common, rng, apps/rtq's elements, queries and both
metrics, stdlib/numerics, apps/pbrt/rng). The twenty CPU gang schedules --
the fifteen fcpwb-packet-* and the five fcpw4w16 CPU ones -- differ in
final SSA by one pattern alone, a vector constant read apart component by
component before each component is spread over the gang's lanes:

    before:  z = (0, 0, 0)                     after:  z16 = broadcast(0)
             x = z.x;  x16 = broadcast(x)
             y = z.y;  y16 = broadcast(y)
             w = z.z;  w16 = broadcast(w)

at three sites a schedule (the zero vector the gang's box test and
closest-point test clamp against) and a fourth in the sorted drains (a
constant 31.0 in the walk's gang code). Thirteen of the twenty have
byte-identical LLVM: LLVM had folded the read of a broadcast's component
already. The seven sorted drains' LLVM differs in value names alone (the
numbering shifts by three), and their object code, lowered by the
compiler's own LLVM 23 and disassembled, is identical instruction for
instruction (equal text sizes; 3,478 instructions either way in
fcpwb-packet-defer-sort-threads). The tables stand; no remeasure.

The compiler commit of 10:20 on 11-02 (1f5c68b1: a bypass that lands on a
join keeping its arguments, and a per-lane vector accumulated into an array
of vectors), checked the same way at 10:21-10:24 under the peer's hold, the
binary's checksum the same before and after: all sixty-three schedule files
byte-identical in final SSA, LLVM module and header against 1ab9281e's
outputs. Both changes give a lowering to shapes that stopped the compiler
before, and every wosx schedule compiled before. The tables stand.

The compiler commit of 15:57 on 11-02 (a4d805c7: what a divergent loop's
code after it reads of the loop, and the join that threads a blended name
on), checked the same way at 15:59-16:01 under the peer's hold, the binary's
checksum the same before and after: sixty-two of the sixty-three schedule
files byte-identical in final SSA, LLVM module and header against
1f5c68b1's outputs. The one that moved is
fcpwb-packet-defer-sort-fine-perlane-threads, the sorted drain whose
traversal runs per lane, and it moved in the way the commit describes: a
divergent loop is one whose lanes leave on different iterations, and a
value a lane carries out of it and reads afterwards -- here the per-lane
tree walk's node offset and its step count -- is now handed on under a
name of its own, so the loop's exit tracks one more value and selects it
once more, and the read after the loop names that copy where it named the
tracked value before:

    before:  after_loop: use(track_1)
    after:   after_loop: use(offset_out)      offset_out = select(left_now, offset, track_7)

twenty-five structural lines of SSA, the same values. Its LLVM module
(3,734 lines either way) differs in value numbering alone, and its object
code, lowered by the compiler's own LLVM 23 and disassembled, is identical
instruction for instruction (text 25,997 bytes either way). The schedule is
drawn in no figure and has no rows in the current table. The tables stand.

### Next

- The refill for the compacting block, as the user's plan (above): a
  bounded block queue drained when full, fed by per-thread persistent
  claims. Not to be built before the user approves the plan.
- The block-level persistent claim -- one lane's fetch-and-add broadcast
  through shared memory -- so that `persistent(i_chunk, b)` with the
  block as the worker compiles (scratch/fcpwb-gpu-defer-chunk2-persistent,
  refused today). Compiler work, the user's to open.
- A true wavefront for the stars: the three traversals of a step staged
  onto three queues (`stage`), each kernel one traversal type over one
  tree; the last of the night's list, not yet tried.
- OptiX's one loop: a form of the traversal that survives OptiX's
  compiler with the leaf visit as the exit of an inner loop over the
  interior nodes -- the lanes gathered before the triangle tests, as ptxas
  structures the CUDA kernel on its own -- written so in the PTX, tried on
  fcpwb-gpu-optix and measured against the 1.27x of warp instructions
  above; a codegen question, the user's to open.
- The read-only root through `params`: 26 of the raygen's 50 loads are
  plain where the CUDA kernel's are `.nc`, the triangle vertices' among
  them, because launch_readonly's roots_of does not see through a pointer
  read from the launch parameters; OptiX's SASS takes no read-only path
  in either case, so the gain would be the PTX's equality, not the
  launch's time.
- Whether wosx's OptiX schedules pass --gpu-max-registers 64 by default
  (five to seven percent of the launch) -- the user's call, as for pbrt's
  128.
- The user's question of whether the rest of the comb's CPU gap -- the
  `vec3f` loads and reductions -- is worth a codegen change; the
  binary-tree schedules, fcpwb2 and the walk on stars' tables are
  measured under the unrolled visit (cd2300a2, bb4759e9, the exploration
  of 10-28 above).
- The walk on stars' CPU gap read from the machine as for the walk on
  spheres; its GPU variants are measured (the exploration of 10-28).
- A profile of the OptiX ray-generation launch of the software traversal
  (fcpwb-gpu-optix at 0.75x of fcpwb-gpu-parallel, the same code), and
  of the reordering's two populations on the stars (a third to a half of
  the plain launch on whale, octopus and comb; at or near the best of
  the ten on the million-walk meshes): Nsight Compute on an OptiX
  launch, which needs the user's yes.
- The argmin typing question above, for the user; `closest_within`'s and
  `closest_absorbing`'s dead `if !c`, to be rewritten as the filter form
  or dropped once it is decided.
- The two deferral schedules' variations above (1024 walks, a per-round
  sort on the card-wide queue).
- DONE 2026-10-25 (night): the CPU comparison moved onto the bytes FCPW's
  CPU reads. The layout extension landed that day (compiler 75282086: a
  tile's derived field reads another named group, by index or by address)
  and apps/rtq's layouts/fcpwb.bonsai and fcpws.bonsai became FCPW's own
  arrays wrapped (apps/rtq/PLAN.md, "Genuine apples-to-apples"); here the
  CPU schedules follow: fcpwb, fcpwb-packet and fcpwb-packet-points import
  apps/rtq's fcpwb (fcpwb-threads and fcpwb-persistent through fcpwb), and
  cpu/wost_fcpwb imports layouts/fcpw_stars.bonsai, now the CPU bytes of
  the walk on stars' three trees -- the absorbing boundary's BvhNode array,
  Triangle pointers and objects and positions; the reflecting boundary's
  SnchNode array (CPU field order), its pointers, objects and positions;
  the silhouette edges' SnchNode array, SilhouetteEdge pointers and
  objects, their ends gathered through the soup and the two face normals
  derived as SilhouetteEdge::normal computes them -- with the GPU rows
  moved to layouts/fcpw_stars_gpu.bonsai, which kernel/wost_fcpwb imports;
  the -gpu schedules keep fcpwb_gpu as before. fcpw_stars_trees.h wraps
  under `!WOSX_GPU` (wrap_absorbing checks every reference and index and
  wraps through rtq_snch::wrap_const, wrap_reflecting through
  rtq_snch::wrap_snch; Trees::buffers() hands the driver twelve buffers)
  and copies the GPU rows under WOSX_GPU as before; wosx_hook.cpp defines
  RTQ_FCPW_CPU_BYTES when not WOSX_GPU, wraps FCPW's binary tree through
  rtq_bvh::wrap_fcpw_bvh under WOSX_BINARY_BVH, builds its buffer list as
  a vector and refuses to run when the generated `_sides` count and the
  list disagree. Checked on head (repeats 1, the machine not quiet):
  fcpwb 0.64x with every statistic agreeing (walks ended 741276 against
  740408, mean length 13.72/13.70, rms 1.39e-01 both, z -0.31), down from
  0.94 on the GPU rows as apps/rtq's leaf-heavy queries fell by what the
  stored corners gave them; wost-fcpwb 0.98x (ended 736866/736829, length
  77.82/77.66, z 1.07) against 0.91 on the GPU rows (a single repeat each,
  so the direction is not yet established); fcpwb-gpu-defer and
  wost-fcpwb-gpu build and agree as before (the GPU section untouched).
  Still to do: the CPU tables remeasured quiet (`benchmark.sh` and
  `--stars`) so that the CPU columns of the figures read against the bytes
  WoSX reads, and the fcpwb2 schedules, which are WoSX's own tree and
  untouched.
- The walk on stars' silhouette query ignores its `min_r2` where FCPW's
  traversal stops once a silhouette within the shell's inner radius is
  found (`squaredMinRadius >= s.r2`, at the leaf and per edge): with a
  positive minimum radius FCPW does less work and may return a nearer-than-
  needed edge rather than the nearest, ours always the nearest. apps/rtq
  passes a minimum radius of zero, so its comparison is unaffected; the
  walk on stars' is not, and the two sides have to ask the same question
  before its table is read as a layout or schedule comparison.
- The hybrid switch as a schedule point (the packet section), now with
  the lane counts of 10-29 behind it: a gang of the sorted drain holds
  four lanes of sixteen at a node and two and a half at a triangle on the
  head, 2.4 and 1.7 on the rover, so a traversal that goes one lane at a
  time below a lane count -- Embree's switch -- would run most of the
  passes scalar; and the per-lane drain (the same two directives in the
  per-lane order, fcpwb-packet-defer-sort-fine-perlane-threads), which
  keeps each lane's own nearer-first order and so the scalar's visit count
  where the gang's majority vote costs a lane 1.4x the leaves, once it
  compiles (the next item).
- The per-lane traversal over Embree's pointer arena
  (tests/bonsai/pending/bvh4_ptr_arena_perlane): the argmin's one record
  pointer, carried round the loopified loop as a block argument, is named
  `_best0!loop` at the uniformized header and `_best0` on the entry edge,
  and the divergence analysis, telling values apart by name, reads the
  join as two values the lanes disagree about and refuses the function
  ("a per-lane pointer into per-lane memory"). The wosx per-lane drain
  does not hit it (its record pointer stays the function's parameter). A
  skip of the latch's self-pass was tried on 10-29 and made other carried
  values uniform; the fix belongs where the two names are made one.
- FCPW's scalar arithmetic for WoSX's default configuration (issue 1) is
  now metrics/fcpw_scalar.bonsai in apps/rtq.
- The extensions, one at a time against WoSX: a source term (Poisson),
  the gradient estimate, a non-zero flux or Robin coefficient on the
  reflecting boundary (the Neumann contribution's boundary sampling
  through the tree).
- Why the two largest meshes scale worse across the threads on both sides
  (the table's note): a profile of the threaded dragon solve, cache misses
  against the single-threaded one.

## Notes for WoSX's authors (2026-10-27)

Two things the GPU comparison turned up in WoSX's own walk. Both are kept
as they are on our side, since the comparison is of the same algorithm
(the user, 10-27: "we want to compare fairly to what WoSX does"); they are
noted here for whoever maintains WoSX.

- **The terminal query is a second traversal the walk's last query already
  answered.** `getTerminalContribution` calls `projectToAbsorbingBoundary`
  -- FCPW's findClosestPoint with an infinite radius -- at the walk's final
  position, for the boundary point whose Dirichlet value the walk returns.
  The loop's last `computeDistToAbsorbingBoundary` ran the same query at
  the same position and kept the distance alone, since the distance
  interface returns a distance (it serves distance grids as well as
  meshes). Returning the closest point from the step's query, or keeping
  it in the walk state, saves one full traversal per walk: on comb a walk
  is six steps, so the terminal query is about a seventh of the traversal
  work, and on a warp it runs with only the lanes that ended at the
  boundary.

- **The step's query is an argmin where a min would do.**
  `computeDistToAbsorbingBoundary` needs the distance; FCPW's
  findClosestPoint finds the closest point and fills an Interaction (point,
  normal, uv, primitive index, distance) at every improvement. A traversal
  carrying the best distance alone -- the same node order and pruning, the
  leaf's point-triangle distance without the record's bookkeeping -- does
  less per leaf and holds fewer registers. The saving is small beside the
  traversal (the distance to a triangle locates the closest point's region
  on the way; what goes is the record's writes), and our side does the same
  argmin with FCPW's own records (apps/rtq), so the comparison stands; WoSX
  need not.
