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

One mesh, one run, from the repository root, inside the `bonsai` conda
environment, with the compiler built (`BONSAI_BUILD_DIR` names its build
directory; unset, the script takes the first of `build`, `build-*` whose
CMake cache found an LLVM, and says which):

```bash
apps/wosx/compare.sh [--schedule fcpw4w16] [--side 128] [--walks 64] [--repeats 3] <mesh.ply[.gz]>
apps/wosx/compare.sh --image apps/wosx/dragon ~/projects/pbrt-v4-scenes/sssdragon/geometry/dragon.ply.gz
```

The mesh is a binary PLY of either byte order, plain or gzipped, as pbrt's
scenes ship them, and it has to be closed: the sample points are the ones
inside it. It is normalized into the unit sphere, as WoSX's demos normalize
theirs, so that the epsilon shell -- `--epsilon`, 1e-3 by default as in
WoSX's demos -- means the same for every mesh.

What the driver does: it loads and normalizes the mesh; has WoSX's
`FcpwDirichletBoundaryHandler` build FCPW's vectorized BVH over it and
copies that tree into the layout; lays a `side` by `side` grid of points on
the plane through the middle of the mesh and keeps the ones WoSX says are
inside (its parity count of ray hits along the axes) and outside the
shell; and solves the Laplace problem with the Dirichlet data `x^2 - y^2`
at each of them on both sides, `walks` walks per point, each side the
least of `--repeats` runs after a warm-up. The data is harmonic, so the
solution inside is the same function and both estimates are checked
against it, not only against each other: the error is the walk's own (the
shell's bias, the Monte Carlo variance) and nothing else.

The table it prints: the number of walks, million steps per second on
each side (a step is one distance query, the one thing a walk costs; each
side's rate from its own count of steps), the speedup (WoSX's time over
ours), the mean walk length on each side (the one number that says both
did the same work), each side's root-mean-square error against the known
solution, and `z`: the difference between the two estimates at each point
over the standard error of that difference from WoSX's own per-point
variance, whose root mean square is one when the two sides agree to their
noise -- the streams cannot be the same, since WoSX seeds each point's
generator from the clock. `--image P` also writes the slice three ways,
`P-bonsai.ppm`, `P-wosx.ppm` and `P-exact.ppm`, blue to red over [-1, 1],
the cells with no point black.

The measurement is single-threaded and pinned, as apps/rtq's is: WoSX's
`solve` with `runSingleThreaded`, the program's parfor over the points
left unbound by the schedule (a plain loop), both under numactl on the
core the kernel ranks best. `RTQ_CPUS=<cpu>` pins to another.

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
- `wosx_hook.cpp`: the driver; `compare.sh`: the script. The mesh reader
  and the copy of FCPW's tree into the layout are apps/rtq's
  (`apps/rtq/mesh.h`, `apps/rtq/fcpw_tree.h`), shared.
- `scratch/`: an agent's working files for this app, gitignored.

The generated `wosx.h`, `wosx.o`, `wosx.bir` and `wosx.ll` are left in this
directory by `compare.sh` only while it runs, and are not committed.
