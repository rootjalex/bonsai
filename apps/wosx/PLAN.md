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

### Issues raised to the user

1. **Which FCPW tree WoSX runs on.** As shipped, WoSX's CMake forces
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
5. **apps/wos** (the stale starter, one file) is left in place; it should
   go once this app stands.

### Next

- The scalar-BVH layout and FCPW's scalar arithmetic, for WoSX's default
  configuration (issue 1).
- The extensions, one at a time against WoSX: a source term (Poisson),
  the gradient estimate, walk on stars for Neumann boundaries.
- A benchmark.sh over several closed meshes, as apps/rtq's, once the
  single-mesh comparison is trusted.
