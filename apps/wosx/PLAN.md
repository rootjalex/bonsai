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
before the table: the per-block form with 1024 walks a point (a block of
1024, more live work per barrier), the card-wide form with a `sort` of
the queue by position each round, and a profile of each (nsys: rounds,
time a round, the guard's share) on the head and the dragon.

### Next

- The two deferral schedules over the GPU table
  (`benchmark.sh --gpu --schedules fcpwb-gpu-defer,fcpwb-gpu-defer-block`),
  their bars added to the GPU figure; the variations above.
- Walk on stars as the second solver (the user's ask, 2026-10-25): the
  program (`wost.bonsai`: the star radius by a closest-silhouette query
  over the reflecting boundary's tree, the step a ray cast against it, the
  Dirichlet distance the query we have; homogeneous Neumann, no source),
  the SNCH layout (FCPW's GPUSnchNode rows, 60 bytes: box, normal cone,
  the leaf's primitives and silhouette edges; GPUEdge rows, 56 bytes, the
  two corners and the two adjacent normals) and the silhouette query in
  the tree language (the cone test at the node, the silhouette test at
  the edge), the driver splitting each mesh into an absorbing and a
  reflecting part and building both of WoSX's handlers, the schedules
  (CPU and GPU) and the table against WoSX's WalkOnStars on both.
- The layout-language extension above, if the user wants WoSX's default
  tree matched; the hybrid switch as a schedule point (the packet section).
- The scalar-BVH layout and FCPW's scalar arithmetic, for WoSX's default
  configuration (issue 1).
- The extensions, one at a time against WoSX: a source term (Poisson),
  the gradient estimate, walk on stars for Neumann boundaries.
- Why the two largest meshes scale worse across the threads on both sides
  (the table's note): a profile of the threaded dragon solve, cache misses
  against the single-threaded one.
