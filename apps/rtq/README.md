# apps/rtq

A ray query -- the nearest triangle a ray hits, whether it hits any, and
the triangle closest to a point -- written in bonsai and compared against
Embree on Embree's own tree, and against FCPW on each of FCPW's: its
vectorized tree, its scalar binary tree, and its normal cone hierarchy,
where a fourth query, the closest silhouette point, is compared too (the
trees and the queries apps/wosx's walks make, isolated).
`rtq.bonsai` is the program (`snch.bonsai` the one with the silhouette
query), `metrics/*.bonsai` the reference's arithmetic it runs over,
`schedules/*.bonsai` the ways of running it, `schedules/trees/*.bonsai` the
trees and `schedules/layouts/*.bonsai` their bytes, and `rtq_hook.cpp` the
C++ driver that reads a mesh, has the reference build the tree into the
layout, makes the rays and points, and runs and checks both sides.
`PLAN.md` is the working record of what was built, measured and decided,
and why.

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

The `fcpw<branching>w<width>` schedules compare against it (`compare.sh
--schedule fcpw4w16`); `compare.sh` stops with a message if it is not
checked out. The `fcpwb` and `fcpws` schedules, over FCPW's scalar trees,
compare against the FCPW that WoSX ships (`deps/wosx/deps/fcpw`, a later
checkout whose `bvh.inl` differs; `git submodule update --init --recursive
deps/wosx`), since they isolate the queries apps/wosx's walks make against
the build they make them against.

## Running

The whole table, from the repository root, inside the `bonsai` conda
environment, with the compiler built:

```bash
apps/rtq/benchmark.sh                 # 13 meshes x both references, CSV, plots, geomeans
apps/rtq/benchmark.sh --plot-only     # fold and plot the logs already there
```

It runs `compare.sh` over the thirteen meshes of `~/projects/pbrt-v4-scenes`
(`RTQ_SCENES` names another checkout) under the two matching schedules,
`embree` and `fcpw4w16` (`--schedules embree,fcpw4w16,packet16` adds the
packet schedule, run for the two ray queries), pinned and least of five; folds the logs into
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

Two more FCPW schedules run through `compare.sh` alone for now, over the
scalar trees FCPW builds with enoki off, the build WoSX ships: `fcpwb`,
the three queries over FCPW's binary BVH (`schedules/fcpwb.bonsai` over
`schedules/layouts/fcpwb.bonsai`, the driver built with `RTQ_FCPW_BINARY`),
and `fcpws`, the same three over its normal cone hierarchy plus the closest
silhouette point over the tree's silhouette edges (`schedules/fcpws.bonsai`
over `schedules/layouts/fcpws.bonsai`, the program `snch.bonsai`, the
driver built with `RTQ_FCPW_SNCH`: FCPW's silhouettes computed first, with
the convex edges left out as WoSX's reflecting boundary handler leaves
them out, then the build). On the CPU both layouts are the bytes FCPW's
own CPU traverses -- its node arrays, its primitive and silhouette pointer
arrays, its Triangle and SilhouetteEdge objects and its polygon soup's
positions, wrapped where they lie, not copied -- so that a leaf gathers a
triangle's corners through the object's indices as FCPW's leaf does. The silhouette rows (`near silhouette`, `volume
silhouette`) ask, about each point of the two point sets, for the closest
silhouette edge within a tenth of the diagonal, with no minimum radius so
that both sides search to the exact closest, WoSX's precision and its
normal flip (`findClosestSilhouettePoint`); the agreement counts the same
edge or none on both sides, a tie where two edges are equally far, and the
records' distance and point. Both are compiled over
`metrics/fcpw_scalar.bonsai`, the scalar build's arithmetic
(`RTQ_SKIP_CMAKE=1` runs the compiler as built, without rebuilding it):

```bash
RTQ_SKIP_CMAKE=1 apps/rtq/compare.sh --schedule fcpwb,fcpws ~/projects/pbrt-v4-scenes/head/geometry/head.ply
```

`fcpwb-gpu` and `fcpws-gpu` are the same two on the GPU: each batch's loop
split by 256 and bound to the GPU's blocks and threads, one launch and one
query a thread, the traversals bvh.slang's (the closest point ordered by
the box distance alone, as the kernel orders it); the reference FCPW's
GPUScene -- its compute shaders over the tree it transfers to the card,
the very rows the layouts hold -- run by the driver built with `RTQ_GPU`
against slang-rhi and the Slang release `apps/wosx/build_slang_rhi.sh`
fetched. Both sides are timed around the dispatch and the wait alone: the
queries are uploaded and the answers read back off the clock
(`fcpw_gpu_reference.h`, FCPW's own `allocate`, `runShader` and `read`
taken apart), and the program's buffers are staged on the device before
the clock with a copy inside it forbidden (`runtime/bonsai_buffer.h`). The
host thread is pinned as a single-thread run is; the GPU is the GPU.

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
beyond it. `--schedule` names one, or several separated by commas. The
packet schedule, `packet16`, runs the rays in gangs of sixteen -- the
traversal made by the gang as a whole, one node the lanes share, one stack
(a packet traversal) -- and against it `compare.sh` has Embree answer in
packets of sixteen too (`rtcIntersect16`, `rtcOccluded16`, the driver's
`--packet 16`): the same sixteen consecutive rays a packet on both sides,
over the same BVH8. Embree's packet kernel is its hybrid (packet traversal
while more than fourteen of the sixteen rays are active at a node, each
ray alone below that), with its default `RTC_RAY_QUERY_FLAG_INCOHERENT`;
`--coherent` asks for its frustum kernel instead, for a probe. Embree has
no packet form of the point query, so the packet comparison is the two ray
queries.

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
  point; FCPW's sphere; and, for FCPW's normal cone hierarchy, the
  silhouette edge (`Edge`: two ends, the two adjacent faces' unit normals,
  the index, whether it has one face), the node's box with its cone
  (`ConeBox`) and the silhouette query (`SilhouetteQuery`: the sphere, the
  squared minimum radius, the precision, FCPW's normal flip) -- and
  `extern triangles : set[Triangle]`. Compiled first, so that the metric
  file's answer records are declared over them and the program's exports
  over the records.
- `rtq.bonsai`: the three queries over the set, plus the exported batches,
  each answered with the reference's record. Which stored slots are
  triangles is the layout's to say, not the queries'.
- `silhouette.bonsai`: the fourth query, FCPW's closest silhouette point
  (`closest_silhouette`) over `extern silhouettes : set[Edge]`, an argmin
  over a filter -- the edge within the sphere and a silhouette from its
  center, which the tree bounds with FCPW's cone test over the cone box --
  answering none where no edge passes; `snch.bonsai`: rtq.bonsai's
  batches plus the silhouette batch, the program for the `fcpws` schedule.
  A file of its own so that apps/wosx's walk on stars imports the query
  and the programs over a plain tree of triangles need no layout for the
  set.
- `metrics/embree.bonsai`, `metrics/fcpw.bonsai`, `metrics/fcpw_scalar.bonsai`:
  the reference's arithmetic, transcribed, one file compiled beside the
  program per comparison -- `intersects`, `distmin`, `distmax` and
  `contains` for a ray and a box, a ray and a triangle, a point and a box,
  a point and a triangle. Embree's: its slab test on the floats' bits
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
  traversal and not the algorithm. `metrics/fcpw_scalar.bonsai` is FCPW's
  scalar build's (enoki off, the build WoSX ships, and its GPU kernels'):
  true divisions, one root per triangle, and Eigen's two reduction orders
  written as `fma` chains -- `dot` summing `(p0 + p1) + p2` with the newer
  product fused, `squaredNorm` and the box distances `p0 + (p1 + p2)` with
  the left one fused, measured rather than read (apps/wosx/PLAN.md, "The
  silhouette query read against FCPW's") -- so that every key and every
  test against a radius is FCPW's to the bit; and the silhouette section:
  FCPW's cone-overlap test as the silhouette query's `intersects` over the
  cone box and the box's distance as its `distmin` with no far-corner
  bound, the silhouette test on an edge (`isSilhouetteEdge`) within the
  sphere as `intersects` over the edge and its squared distance as
  `distmin`, the record of the closest point on the edge, and the ray's
  and the point's metrics over a cone box, which are the box's.
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
- `schedules/packet16.bonsai`: the rays in gangs of sixteen over the same
  layout, against Embree's `rtcIntersect16` and `rtcOccluded16`
  (`BVHNIntersectorKHybrid`, its packet kernel). The loop over the batch
  is split by sixteen and vectorized; the traversal's loops are left as
  loops, so a node's children are tested one at a time against the
  sixteen rays, as Embree's chunk loop tests them, and `sort` then
  `loopify` make the gang's recursion one packet traversal over one stack
  of Embree's chunk depth (561), the order of the hit children settled
  once for the gang by a vote (the majority of the lanes that are on, the
  user's decision); `skip` puts Embree's `none(valid)` exits
  after each edge test of the sixteen-wide triangle test; no prefetch, as
  Embree's chunk loop has none. The file lists what Embree's kernel does
  that no directive says: the hybrid's switch to single rays at fourteen
  or fewer active, its partition of a packet by direction octant, and its
  any-compare scan in place of a full sort (PLAN.md, "Ray packets against
  rtcIntersect16", has the audit). `schedules/packet16-anyhit.bonsai` is
  the any hit alone in gangs, the other two queries scalar: the interim
  file the any-hit packet cell was first measured with, while the nearest
  hit's packet form was still refused past the vectorizer; only its `any`
  rows are packet cells.
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
- `schedules/layouts/fcpwb.bonsai`, `schedules/fcpwb.bonsai`: FCPW's
  scalar binary BVH -- WoSX's default tree, FCPW_USE_ENOKI off, the one
  tree FCPW puts on a GPU -- in the bytes FCPW's CPU reads, nothing
  copied: FCPW's own node array (32-byte BvhNode rows, each its own box),
  its primitive pointer array, its Triangle objects (32 bytes, the three
  vertex indices) and its polygon soup's positions, each wrapped where it
  lies (`fcpw_bvh.h`, `wrap_fcpw_bvh`), the leaf's tile storing the
  pointer and deriving the corners through the object's indices as FCPW's
  leaf gathers them (`t = objects[tri]`, `pa = positions[t.i0].p`: a tile's
  derived field reading the layout's other groups, by index or at an
  address); the node declared bounding its children with the layout
  reading the children's boxes out of the children's own rows, so that a
  visit reads what FCPW's visit reads; and the three queries run as FCPW's
  scalar traversals run them: both children tested at the parent and
  written out (`unroll`), the nearer entered first by the box distance
  (the closest point's far-corner rule where both children hold the
  point), a stack of sixty-four, the leaf's Voronoi exits.
- `schedules/layouts/fcpwb_gpu.bonsai`: the same tree in FCPW's GPU bytes
  -- the node rows as above, the triangles as FCPW's GPUTriangle rows (the
  corners stored, a tile of one with the program's edges derived), which
  is what FCPW's GPU reads; the tree `fcpwb-gpu` traverses, copied in by
  `fcpw_bvh.h`'s `copy_fcpw_bvh`. Imported by apps/wosx's `fcpwb-gpu*`
  schedules too; its CPU schedules import `fcpwb.bonsai` above (the
  leaf-layout bias PLAN.md's "The machine code behind the table" describes
  is removed for both apps since 2026-10-25).
- `schedules/layouts/fcpws.bonsai`, `schedules/fcpws.bonsai`: FCPW's
  normal cone hierarchy in the bytes FCPW's CPU reads -- its SnchNode rows
  of sixty bytes (the box, the cone, the leaf's triangle and silhouette
  ranges in FCPW's CPU order), its triangle references and objects as
  above, its silhouette edge references and SilhouetteEdge objects (40
  bytes, four vertex indices, the ends and the two opposite corners) and
  its positions, wrapped (`fcpw_snch.h`, `wrap_snch`) -- as two trees over
  the one node array: the triangles the rays and points query, the
  silhouette edges the silhouette query searches, the driver handing the
  same node rows to both, with ConeBox the bounding element; the edge
  tile derives its ends from the positions and its two face normals as
  FCPW's CPU recomputes them at every test (Eigen's cross and guarded
  normalize), read only where the edge has two faces; and the four
  queries run as FCPW's traversals run them, the silhouette query's
  children tested by cone and box at the parent and ordered by the box
  distance alone.
- `schedules/layouts/fcpws_gpu.bonsai`: the same two trees in FCPW's GPU
  bytes (GPUSnchNode rows, GPUTriangle rows, GPUEdge rows of fifty-six
  with the normals stored), what FCPW's GPU reads; the tree `fcpws-gpu`
  traverses, written in by `fcpw_snch.h`'s `copy_snch`.
- `queries.bonsai`: the three queries themselves (`trace`, `occluded`,
  `closest`), which rtq.bonsai imports and exports batches of; a file of
  their own so that apps/wosx can ask `closest` without the batches.
- `rtq_hook.cpp`: the driver; `compare.sh`, `build_embree.sh`: the scripts.
  `mesh.h` (the PLY reader), `fcpw_tree.h` (the copy of FCPW's Mbvh into
  the layout, with the struct checks), `fcpw_bvh.h` (the same for FCPW's
  binary BVH) and `fcpw_snch.h` (the normal cone hierarchy's rows, written
  as FCPW's GPU transfer writes them) are the driver's parts apps/wosx's
  driver shares.
- `plot.py`: the comparison as three figures (first hit, any hit, closest
  point; meshes by triangle count, the ray or point sets by hatching, the
  reference by colour; the y axis Speedup, the legend saying over which
  reference) from the results CSV, `python apps/rtq/plot.py
  apps/rtq/results/rtq-results.csv -o apps/rtq/plots`, drawn at the
  paper's page width in apps/pbrt/figure_style.py's style; and, where the
  CSV has the rows, the two figures of the walks' queries over FCPW's
  scalar trees, `rtq-fcpwb-fcpws-cpu` (over FCPW's scalar build) and
  `rtq-fcpwb-fcpws-gpu` (over its GPUScene, which takes only the binary
  trees), named by the two schedules whose cells they hold, seven bars a
  mesh, the query by colour and the set by hatching, the legend two rows
  of five with the four queries in the first two columns. Beside each
  figure a `.txt` of the statistics the paper quotes (per reference and
  set the geomean, the least and greatest cell and its mesh, notes reading
  the figure), and the mesh facts in one table rather than on every plot,
  `rtq-meshes.txt` and `rtq-meshes.tex` (a tabular to `\input`), so the
  figures carry the mesh names alone. Each reference on
  its own default tree: Embree's BVH8 (`embree`) and FCPW's four-wide Mbvh
  with sixteen-lane leaves (`fcpw4w16` -- FCPW's default branching, the
  tree its author calls the better tested, and the leaf width its build
  picks on an AVX-512 machine); `--embree`, `--fcpw` name others. (The
  packet schedule's rows, `packet16` against Embree's packets, fold into
  the CSV and the geomeans but have no figure of their own since 10-26.)
- `nsys_kernels.py`: the reading of an Nsight Systems trace of one GPU
  comparison, `nsys profile --trace=cuda --gpu-metrics-devices=0
  --gpu-metrics-frequency=100000 -o results/nsys/<mesh>-hf ./rtq.out
  <mesh.ply> --side 1024 --repeats 20` over the driver compare.sh leaves
  behind under RTQ_KEEP=1, exported with `nsys export --type sqlite`:
  both sides' kernels in launch order (FCPW's rayIntersection and
  closestPoint through slang-rhi, ours `_kernel_i_blk_*`), each run's
  least and mean GPU duration against the driver's period per repeat --
  dispatch, kernel, wait -- registers, grid and block, and the mean of
  every sampled GPU metric inside each kernel's spans; beside the printed
  reading two CSVs for plots, `<stem>-launches.csv` (one row per launch)
  and `<stem>-metrics.csv` (kernel, metric, mean, samples). The traces and
  their CSVs are kept under `results/nsys/` (gitignored), `summary.csv`
  there the kernel-only and wall-clock speedups per cell (PLAN.md, "What
  Nsight Systems says of the GPU win", 10-26).
- `ncu_kernels.py`: two kernels of one Nsight Compute report side by side,
  from its raw page as CSV (`ncu --set full --kernel-name
  regex:'rayIntersection|_kernel_i_blk_2' --launch-count 4 -o
  results/ncu/<mesh>-firsthit ./rtq.out <mesh.ply> --side 1024 --repeats 1
  --batch primary --query intersect`, then `ncu --import ... --csv --page
  raw > <stem>.raw.csv`): duration, throughputs, occupancy, issue rate,
  instructions and active threads per instruction, the load counts and
  cache hit rates, the local-memory (stack) traffic and the warp stall
  reasons, with the ratio; and `<stem>-tidy.csv` beside it, every metric
  of every profiled launch as (launch, kernel, metric, value), for plots.
  The reports are under `results/ncu/` (gitignored); what they said is in
  PLAN.md, "What Nsight Compute says of the two ray kernels" (10-26).
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
