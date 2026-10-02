# apps/rtq

A ray query -- the nearest triangle a ray hits, and whether it hits any --
written in bonsai and compared against Embree on Embree's own tree.
`rtq.bonsai` is the program, `schedules/*.bonsai` the ways of running it,
`schedules/trees/bvh8.bonsai` the tree and its bytes, and `rtq_hook.cpp` the
C++ driver that reads a mesh, has Embree build the tree into the layout,
makes the rays, and runs and checks both sides. `PLAN.md` is the working
record of what was built, measured and decided, and why.

The reference is Embree 4 as shipped: on a machine with AVX it builds a
BVH8 over Triangle4 leaves (eight children per node, the children's boxes
stored in the parent, leaves of up to seven blocks of four triangles, the
leaf flag and block count in the child reference's low bits) and traverses
it with `rtcIntersect1` and `rtcOccluded1`. The program computes what Embree
computes -- its Moeller-Trumbore triangle test and its slab test,
transcribed -- and the schedule runs it the way Embree runs it: the
children of a node tested at the node, the hits sorted by entry distance and
pushed, the nearest descended into, over a stack of Embree's depth. The tree
is built by Embree's builder (`rtcBuildBVH`, with the settings of Embree's
internal BVH8Triangle4 builder), so both sides traverse the same splits, the
same leaves and the same boxes.

## Embree, an optional dependency

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

## Running

From the repository root, inside the `bonsai` conda environment, with the
compiler built (`BONSAI_BUILD_DIR` names its build directory, default
`build`):

```bash
apps/rtq/compare.sh [--schedule embree] [--side 1024] [--repeats 5] [--threads N] <mesh.ply[.gz]>
```

The mesh is a binary PLY of either byte order, plain or gzipped, as pbrt's
scenes ship them (`~/projects/pbrt-v4-scenes/ganesha/geometry/ganesha.ply.gz`,
`.../sssdragon/geometry/dragon.ply.gz`, `.../head/geometry/head.ply`).

What the driver does: it loads the mesh; builds Embree's own scene over it
and, with Embree's builder, the same tree into the layout; shoots a `side`
by `side` image of primary rays from a camera a scene diagonal and a half
away; from Embree's primary hits makes a batch of short ambient-occlusion
rays (a tenth of the diagonal) and a batch of unbounded diffuse bounces,
cosine-distributed about the normal; and for each batch and each query
(nearest hit, any hit) times Embree and the program, each the least of
`--repeats` runs after one warm-up. Every answer is checked against
Embree's: a nearest-hit disagreement where the two triangles are at the
same distance is a tie, which either side may answer either way; any other
disagreement fails the run.

The measurement is single-threaded and pinned. By default both sides run
their rays as one plain loop on one core, and `compare.sh` picks that core
and pins the run to it with numactl: the physical core the kernel ranks
highest for performance (on this machine's Ryzen 9 9950X3D, a core of the
frequency chiplet, cpus 8-15, not the V-cache one), with the memory of its
NUMA node. A multi-threaded number mixes the kernel's cost with a pool's
balancing, and unpinned the two sides can land on unlike cores; it is not
the comparison. `--threads N` pins both sides to the N best-ranked physical
cores and runs them over one TBB pool of N; `--threads 0` is the unpinned
all-core run; `RTQ_CPUS=8,9` pins to exactly those CPUs. The driver prints
the CPUs it was left at the top of its output.

The table it prints, per batch and query: rays, million rays per second for
each side, the speedup (Embree's time over ours), and the agreement counts.
`--embree-stats` asks Embree to print its own tree's statistics (node and
leaf counts, SAH), beside the tree the driver built, to check that the two
are the same tree.

## Layout of the files

- `rtq.bonsai`: Embree's ray, triangle (a vertex and two edges, with its
  ids, as `TriangleM` stores a slot), box test (`node_intersector1.h`) and
  triangle test (`triangle_intersector_moeller.h`), and the two queries
  over `extern triangles : set[Triangle]`, plus the exported batches.
- `schedules/trees/bvh8.bonsai`: the tree -- a node holding its eight
  children as an array with one box annotation over them, a leaf holding
  a run of triangles -- and the layout, Embree's bytes: a 256-byte node
  row of eight 64-bit child references and the bounds as six vectors of
  eight floats in Embree's order, the leaves as 176-byte Triangle4 blocks
  (every field a vector over four triangles), and a reference decoded from
  its own bits as Embree's `NodeRef` is, in Scion's spelling: the
  reference is the layout's parameter, `switch ref[0:3]` reads its low
  bits, `Nodes[ref[4:63]]` the rest, `prims[a : a + n]` a leaf's slice.
  The driver checks the sizes and offsets against Embree's at compile
  time, and relocates the largest nodes after the build as Embree does.
- `schedules/embree.bonsai`: the traversal's order (`sort` by each child's
  entry distance), its eight-wide node test (`vectorize` of the loop over a
  node's children, named `triangles.Interior.children`) and stack
  (`loopify(564)`, Embree's stack depth), and the rays across the cores.
- `rtq_hook.cpp`: the driver; `compare.sh`, `build_embree.sh`: the scripts.

The generated `rtq.h`, `rtq.o`, `rtq.bir` and `rtq.ll` are left in this
directory by `compare.sh` only while it runs, and are not committed.
