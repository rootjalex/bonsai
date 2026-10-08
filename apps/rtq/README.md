# apps/rtq

A ray query -- the nearest triangle a ray hits, whether it hits any, and
the triangle closest to a point -- written in bonsai and compared against
Embree on Embree's own tree.
`rtq.bonsai` is the program, `metrics/*.bonsai` the reference's arithmetic
it runs over, `schedules/*.bonsai` the ways of running it,
`schedules/trees/*.bonsai` the trees and `schedules/layouts/*.bonsai` their
bytes, and `rtq_hook.cpp` the C++ driver that reads a mesh, has the
reference build the tree into the layout, makes the rays and points, and
runs and checks both sides. `PLAN.md` is the working record of what was
built, measured and decided, and why.

The reference is Embree 4 as shipped: on a machine with AVX it builds a
BVH8 over Triangle4 leaves (eight children per node, the children's boxes
stored in the parent, leaves of up to seven blocks of four triangles, the
leaf flag and block count in the child reference's low bits) and traverses
it with `rtcIntersect1`, `rtcOccluded1` and `rtcPointQuery` (the last with
the callback of Embree's closest-point tutorial, the one form Embree offers
for a closest-point query). The program computes what Embree computes --
its Moeller-Trumbore triangle test, its slab test and its closest point on
a triangle, transcribed -- and the schedule runs it the way Embree runs it: the
children of a node tested at the node, the hits sorted by entry distance and
pushed, the nearest descended into, over a stack of Embree's depth. The tree
is built by Embree's builder (`rtcBuildBVH`, with the settings of Embree's
internal BVH8Triangle4 builder), so both sides traverse the same splits, the
same leaves and the same boxes.

## Embree and FCPW, optional dependencies

Embree is a submodule at `deps/embree`, pinned to v4.4.1, and nothing else
in the repository needs it. To compare:

```bash
git submodule update --init deps/embree
apps/rtq/build_embree.sh      # into deps/embree-build and deps/embree-install
```

`build_embree.sh` builds the triangle kernels for AVX2 and AVX-512 with the
conda environment's clang and TBB, everything else at Embree's defaults.
`compare.sh` stops with a message if the submodule is not checked out or
not built.

FCPW ("Fastest Closest Points in the West", Sawhney) is the second
reference, for the closest-point query above all: a submodule at
`deps/fcpw`, header-only, carrying Eigen and enoki as submodules of its
own, so there is nothing to build:

```bash
git submodule update --init --recursive deps/fcpw
```

The `fcpw*` schedules compare against it (`compare.sh --schedule fcpw4w16`);
`compare.sh` stops with a message if it is not checked out.

## Running

The whole table, from the repository root, inside the `bonsai` conda
environment, with the compiler built:

```bash
apps/rtq/benchmark.sh                 # 13 meshes x both references, CSV, plots, geomeans
apps/rtq/benchmark.sh --plot-only     # fold and plot the logs already there
```

It runs `compare.sh` over the thirteen meshes of `~/projects/pbrt-v4-scenes`
(`RTQ_SCENES` names another checkout) under the two matching schedules,
`embree` and `fcpw4w16`, pinned and least of five; folds the logs into
`apps/rtq/results/rtq-results.csv` with `tocsv.sh`, each row stamped with
the compiler's commit (`<hash>+` when the tree has uncommitted compiler
changes) and the cells not measured this time carried over; draws the three
figures into `apps/rtq/plots` with `plot.py`; and prints, per reference and
ray or point set, the geomean over the plotted meshes with any cell under
0.97 named -- the rule the comparison is judged by (every geomean a win, no
cell more than 3% under its reference). It waits for an idle machine before
each mesh, resumes a stopped run (a cell's log exists only when complete;
`--fresh` measures every cell again), and holds at a mesh boundary while
`apps/rtq/results/logs/PAUSE` exists. `--meshes`, `--schedules`,
`--repeats` and the sides narrow or widen it; `--help` lists them.

One mesh, one run, with the compiler built (`BONSAI_BUILD_DIR` names its
build directory; unset, the script takes the first of `build`, `build-*`
whose CMake cache found an LLVM, and says which):

```bash
apps/rtq/compare.sh [--schedule embree,tuned] [--side 1024] [--repeats 5] <mesh.ply[.gz]>
```

Two schedules are built and run by default, each its own table under a
`=== schedule` line. `embree` follows Embree's traversal step for step, and
its table says whether the compiler makes of that structure what Embree's
hand-written code is; `tuned` departs from it where a step measures worse
on this machine, and its table says what the schedule language can do
beyond it. `--schedule` names one, or several separated by commas.

The mesh is a binary PLY of either byte order, plain or gzipped, as pbrt's
scenes ship them (`~/projects/pbrt-v4-scenes/ganesha/geometry/ganesha.ply.gz`,
`.../sssdragon/geometry/dragon.ply.gz`, `.../head/geometry/head.ply`).

What the driver does: it loads the mesh; builds Embree's own scene over it
and, with Embree's builder, the same tree into the layout; shoots a `side`
by `side` image of primary rays from a camera a scene diagonal and a half
away; from Embree's primary hits makes a batch of short ambient-occlusion
rays (a tenth of the diagonal) and a batch of unbounded diffuse bounces,
cosine-distributed about the normal; times Embree and the program on each
set under the query it is for -- the nearest hit of the primary and diffuse
rays, the any hit of the ambient-occlusion rays, Aila and Laine's three
sets as every traversal paper since has measured them (`--batch` with
`--query` runs another pairing, for a probe) -- each the least of
`--repeats` runs after one warm-up; and does the same for the closest-point
query over two batches of points, as many as there were primary hits --
`near`, the hit points pushed off the surface along the normal by up to a
tenth of the diagonal, and `volume`, points uniform in the mesh's box grown
by a tenth on every side. Every answer is checked against Embree's: a
nearest-hit disagreement where the two triangles are at the same distance
is a tie, which either side may answer either way, as is a closest-point
disagreement where the two triangles are equally close (a point whose
closest point is on an edge or a vertex two triangles share, the common
case); any other disagreement fails the run. Both sides answer with the
reference's whole record, not the triangle's id alone -- for a ray what
Embree writes into its RTCRayHit (t, u, v, the unnormalized Ng) or FCPW
into its Interaction (the distance, the hit point, the unit normal, uv),
for a point the closest point and its distance (and FCPW's uv) -- and each
row reports the largest deviation of those fields over the queries both
sides answered with the same triangle (`records within t ..., n ..., uv
..., p ...`: a ray's t relative, the normal relative to the reference's
length, uv absolute, the point and the closest point's distance as
fractions of the diagonal), which should read as last bits except where
the reference's own arithmetic is ill-conditioned (a sliver triangle's
barycentrics). The closest-point rows also say how many triangles
Embree's traversal handed its callback per query.

The measurement is single-threaded and pinned. Both sides run their rays as
one plain loop on one core: the program's parfor over the rays is left
unbound by the schedule, so it lowers to a sequential loop (binding it to
the CPU threads and running on one thread measured the same, to within the
run-to-run drift), and Embree's rays go through a loop of `rtcIntersect1`.
`compare.sh` picks the core and pins the run to it with numactl: the
physical core the kernel ranks highest for performance (on this machine's
Ryzen 9 9950X3D, a core of the frequency chiplet, cpus 8-15, not the
V-cache one), with the memory of its NUMA node. A multi-threaded number
mixes the kernel's cost with a pool's balancing, and unpinned the two sides
can land on unlike cores; it is not the comparison. `RTQ_CPUS=9` pins to
that CPU instead. The driver prints the CPUs it was left at the top of its
output.

The tree's storage is mapped the way Embree's allocator maps its own
(`os_malloc`: 2 MB huge pages where the system has them set aside, else a
plain mapping advised for transparent huge pages, which this machine backs
with 2 MB pages), so the two traversals pay the same for their page walks;
`RTQ_PAGES=4k` maps it on plain pages instead, for measuring what the page
size is worth. `RTQ_METRICS=<file>` compiles the program with that
metrics file in place of the reference's own, for measuring what an
arithmetic differs by against the same reference (the table's own numbers
are always the reference's arithmetic). Embree's rays are converted to its structs before the clock
starts and reset between runs off the clock: a struct written field by
field right before `rtcIntersect1` stalls Embree's first load of it, which
is the driver's cost, not the traversal's.

The table it prints, per batch and query: rays, million rays per second for
each side, the speedup (Embree's time over ours), and the agreement counts.
`--embree-stats` asks Embree to print its own tree's statistics (node and
leaf counts, SAH), beside the tree the driver built, to check that the two
are the same tree. `--batch primary|ao|diffuse|near|volume` and `--query
intersect|occluded|closest` narrow a run to one kernel over one kind of
ray or point, which is what a profile of it wants (`perf record` over the
driver, the two sides' kernels told apart by symbol).

## Layout of the files

- `elements.bonsai`: the elements -- Embree's ray; the triangle as a vertex
  and two edges with its ids, as `TriangleM` stores a slot; the box; the
  point -- and `extern triangles : set[Triangle]`. Compiled first, so that
  the metric file's answer records are declared over them and the
  program's exports over the records.
- `rtq.bonsai`: the three queries over the set, plus the exported batches,
  each answered with the reference's record. Which stored slots are
  triangles is the layout's to say, not the queries'.
- `metrics/embree.bonsai`, `metrics/fcpw.bonsai`: the reference's
  arithmetic, transcribed, one file compiled beside the program per
  comparison -- `intersects`, `distmin`, `distmax` and `contains` for a
  ray and a box, a ray and a triangle, a point and a box, a point and a
  triangle. Embree's: its slab test on the floats' bits
  (`node_intersector1.h`), its Moeller-Trumbore test
  (`triangle_intersector_moeller.h`), its point-to-box distance
  (`pointQuerySphereDistAndMask`) and its tutorial's closest point on a
  triangle with true divisions and its root per triangle. FCPW's: its
  slab test with a true-division reciprocal and float compares
  (`intersectWideBox`), its Moeller-Trumbore with enoki's reciprocal and
  an absolute epsilon on the determinant (`intersectWideTriangle`), its
  point-to-box distances (`overlapWideBox`) and its closest point on a
  triangle with enoki's reciprocal and a root per lane squared back
  (`findClosestPointWideTriangle`). Each library's reciprocal is written
  out over the machine's estimate instruction, `rcp_approx` (the one
  reciprocal primitive the language has): Embree's `rcp_embree`, `r + r (1
  - a r)` as two fused multiply-adds, and enoki's `rcp_fcpw`, `2r - (r m) r`
  in one fused step, which round differently in a third of all inputs. The
  same computation on both sides is what makes a table measure the
  traversal and not the algorithm.
- `schedules/trees/bvh8.bonsai`, `schedules/trees/bvh4.bonsai`: the
  trees, declared once each -- a node holding its eight (or four)
  children as an array with one box annotation over them, a leaf holding
  a run of triangles. Embree's BVH8 and FCPW's eight-wide Mbvh are the
  one tree in different bytes, and so are Embree's BVH4 and FCPW's
  four-wide Mbvh; every layout below imports the tree it stores.
- `schedules/layouts/embree8.bonsai`: Embree's bytes for the eight-wide
  tree: one arena of bytes holding 256-byte node rows (eight 64-bit child
  references, then the bounds as six vectors of eight floats in Embree's
  order) and 176-byte Triangle4 blocks (every field a vector over four
  triangles), each where its reference says. A reference is Embree's
  `NodeRef` with a byte offset where Embree has an address: `switch
  ref[0:3]` reads the kind from its low bits, and each arm is a lookup
  that brings its own shape, `Interior from arena[ref[4:63] * 16u] { ...
  }`; a leaf is the run of blocks that begins at its offset, as many as
  its kind bits count. The driver writes rows and blocks into the arena
  as Embree's allocator places them, checks the struct sizes and offsets
  against Embree's at compile time, and relocates the largest nodes after
  the build as Embree does. The arena is a `ptr group`, its rows reached
  by address, so a reference is Embree's pointer exactly and nothing is
  added per visit (PLAN.md, "The layout as Embree's"). Which lanes of a
  block are triangles is the layout's to say, not the program's: the tile
  carries `where geomID != 4294967295u`, Embree's `TriangleM::valid`, and
  the lanes that fail it are not elements of the set -- the padding slots
  of a leaf's last block -- so the closest-point query needs no filter and
  the lowering may prune it by what a child surely holds as well as by
  what it might.
- `schedules/embree.bonsai`: the traversal's order (`sort` by each child's
  entry distance), its eight-wide node test and four-wide leaf test
  (`vectorize` of the loop over a node's children and of the loop over a
  leaf's triangles, named `triangles.Interior.children` and
  `triangles.Leaf.data`), its stack (`loopify(564)`, Embree's stack
  depth), the leaf's early exit after a block's edge tests (`skip` of the
  triangle test's early returns, Embree's `early_out`), and the prefetch of
  every hit child's storage as the node test finds it (`prefetch` of the
  children, Embree's `BVH::prefetch`), for both ray queries; and the
  closest-point query sorted by the point's squared distance to each
  child's box, as Embree's `pointQuery` orders its traversal, with the
  same node test, leaf, stack and prefetch.
- `schedules/tuned.bonsai`: the same without either query's early exit,
  which measures 5-7% slower on the any hit's incoherent rays and 5-14%
  slower on the nearest hit's (see PLAN.md); the schedule that departs
  from Embree's where a step is measured worse.
- `schedules/layouts/embree4.bonsai`, `schedules/embree4.bonsai`,
  `schedules/tuned4.bonsai`: the same over Embree's four-wide tree, the
  BVH4 over Triangle4 leaves that Embree builds on a machine without AVX2
  -- 128-byte node rows of four children, four lanes in the directives, a
  stack of 244. The driver reads the width off the layout it is compiled
  against and asks Embree's device for the tree of that width
  (`tri_accel=bvh4.triangle4`), so a table's two sides always traverse
  trees of one width: `--schedule embree4,tuned4`.
- `schedules/layouts/fcpw4w16.bonsai` and its three siblings (`fcpw8w16`,
  `fcpw4w8`, `fcpw8w8`): FCPW's bytes for the same two trees -- an array
  of node rows holding the children's boxes as component vectors and the
  child slots as ints, a leaf being a row whose first slot is negative and
  describes a run of packets, and an array of packets holding sixteen (or
  eight) triangles' three vertices as component vectors with their ids --
  at FCPW's two branching factors (4, the default; 8 with its
  FCPW_USE_EIGHT_WIDE_BRANCHING) and two leaf widths (16 on an AVX-512
  machine, 8 on AVX2; its FCPW_SIMD_WIDTH). FCPW stores vertices where
  the program's triangle has a vertex and two edges, so the packet
  derives the edges (`e1 = pa - pb`): a tile's members may be stored
  fields or derived ones, and the stored ones are its bytes. FCPW's leaf
  holds a count of triangles in whole packets, the last part full, and
  the layout's `range` over the packets by that count leaves the lanes
  past it out, as FCPW does.
- `schedules/fcpw4w16.bonsai` and its siblings (`fcpw8w16`, `fcpw4w8`,
  `fcpw8w8`): the three queries run the way FCPW's Mbvh runs them --
  sorted by the point's (or ray's) distance to each child's box, the any
  hit included (FCPW's `intersectFromNode` is one traversal for both ray
  queries, `checkForOcclusion` only stopping it at the first hit, so its
  any hit sorts where Embree's does not), the children tested at once and
  the packet as its lanes, FCPW's stack depth, no prefetch, no early exit
  on the ray test (FCPW's has none), and the closest point's leaf exits
  once every lane has its Voronoi region. The driver, compiled for FCPW (`RTQ_FCPW`, with the
  width and branching the schedule's name says), builds FCPW's scene,
  copies its two arrays into the layout, and queries it through its own
  calls: `findClosestPoint`, and `intersect` for the rays.
- `queries.bonsai`: the three queries themselves (`trace`, `occluded`,
  `closest`), which rtq.bonsai imports and exports batches of; a file of
  their own so that apps/wosx can ask `closest` without the batches.
- `rtq_hook.cpp`: the driver; `compare.sh`, `build_embree.sh`: the scripts.
  `mesh.h` (the PLY reader) and `fcpw_tree.h` (the copy of FCPW's Mbvh
  into the layout, with the struct checks) are the driver's parts
  apps/wosx's driver shares.
- `plot.py`: the comparison as three figures (first hit, any hit, closest
  point; meshes by triangle count, the ray or point sets by hatching, the
  reference by colour) from the results CSV, `python apps/rtq/plot.py
  apps/rtq/results/rtq-results.csv -o apps/rtq/plots`. Each reference on
  its own default tree: Embree's BVH8 (`embree`) and FCPW's four-wide Mbvh
  with sixteen-lane leaves (`fcpw4w16` -- FCPW's default branching, the
  tree its author calls the better tested, and the leaf width its build
  picks on an AVX-512 machine); `--embree`, `--fcpw` name others.
- `results/`, `plots/`: the measured data -- one CSV, one row per measured
  cell (date, commit, reference, schedule, mesh, query, ray set, both rates,
  the speedup, the agreement) -- and the figures drawn from it. Generated,
  gitignored, kept here rather than in a build directory so that clearing a
  build does not erase them.
- `scratch/`: an agent's working files for this app, gitignored -- probe
  scripts, their logs, pinned worktrees of the compiler for a measurement,
  the script that folds `compare.sh` logs into the CSV. Kept beside the app
  rather than under /tmp so that a reboot does not take them.

The generated `rtq.h`, `rtq.o`, `rtq.bir` and `rtq.ll` are left in this
directory by `compare.sh` only while it runs, and are not committed.
