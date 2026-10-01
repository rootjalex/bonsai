# apps/rtq: where this is and what is next

The goal is a ray query written in bonsai that, run with a schedule
equivalent to Embree's traversal over Embree's own tree, costs what Embree
costs -- the comparison that says whether the abstraction gets in the way
of a hand-written kernel, on the same execution pattern. Run
`apps/rtq/compare.sh <mesh.ply>` to see where things stand; `README.md`
says how.

## What Embree does, read as a program, a tree and a schedule (2026-10-01)

Embree 4.4.1, `deps/embree`, read directly.

**The structure.** For a static triangle scene on a machine with AVX,
`Scene::createTriangleAccel` (kernels/common/scene.cpp) picks
`BVH8Triangle4`: `BVHN<8>` whose `AABBNode` (kernels/bvh/bvh_node_aabb.h)
is eight `NodeRef` children (64 bits each) followed by the children's
bounds as six vectors of eight floats, in the order lower_x, upper_x,
lower_y, upper_y, lower_z, upper_z -- 256 bytes, no box of the node's own.
A leaf is one to seven `Triangle4` blocks (kernels/geometry/triangle.h:
`TriangleM<4>`, SoA: v0, e1 = v0 - v1, e2 = v2 - v0, geomIDs, primIDs, 176
bytes), a slot a block does not fill zeroed with ids of -1. A `NodeRef`
(kernels/bvh/bvh_node_ref.h) is a pointer whose low four bits are the type:
0 an AABB node, `tyLeaf` (8) + n a leaf of n blocks; `emptyNode` is 8, a
leaf of nothing.

**The build.** `BVHNBuilderSAH<8, Triangle4>` (kernels/bvh/bvh_builder_sah.cpp)
with sahBlockSize 4, intersection cost 1, minLeafSize 4, maxLeafSize inf
clamped to 4 * maxLeafBlocks = 28; `BVHNBuilderVirtual<8>::build` sets the
branching factor to 8 and the depth to maxBuildDepthLeaf = 40; medium
quality is the binned SAH over 32 bins (kernels/builders/heuristic_binning.h),
an N-ary node made by repeatedly splitting its largest child
(bvh_builder_sah.h). `rtcBuildBVH` (kernels/common/rtcore_builder.cpp) runs
the same `BVHBuilderBinnedSAH::build` over the same `PrimRef`s with the
settings it is handed, so the driver's call with those settings builds the
same tree -- the same splits, leaves and boxes; the order of primitives
within a leaf's blocks can differ where the build partitions in parallel,
which decides only exact ties. Embree then re-lays out the top 0.5% of its
nodes by area for locality (`layoutLargeNodes`); the driver does not.

**The traversal** (kernels/bvh/bvh_intersector1.cpp, node_intersector1.h,
bvh_traverser1.h). `TravRay` precomputes `rdir = rcp_safe(dir)`,
`org_rdir = org * rdir` and which of lower/upper is near per axis. A node
is tested as a whole: for each axis, `tNear = msub(near, rdir, org_rdir)`
over the eight children at once, `tNear = max(tNearX, tNearY, tNearZ,
tnear)`, `tFar = min(tFarX, tFarY, tFarZ, tfar)`, hit where `tNear <=
tFar` (on AVX-512, `maxi`/`mini` and the comparison on the bit patterns as
integers). `traverseClosestHit` descends into the nearest hit child and
pushes the others with their distances, nearest on top; a popped entry
whose distance is past `ray.tfar` is dropped. The leaf tests a block's four
triangles at once (`MoellerTrumboreIntersector1<4>`: `Ng = cross(e2, e1)`,
`C = v0 - O`, `R = cross(C, D)`, `den = dot(Ng, D)`, `U = dot(R, e2) ^
sgn(den)`, `V = dot(R, e1) ^ sgn(den)`, valid where `den != 0, U >= 0, V >=
0, U + V <= |den|`, `T = dot(Ng, C) ^ sgn(den)`, hit where `|den| * tnear
< T <= |den| * tfar`), picks the nearest valid slot, and sets `ray.tfar` to
it. `occluded` traverses the hit children in stored order
(`traverseAnyHit`) and stops at the first hit. The stack is 1 + 7 * 80 + 3
= 564 entries.

## What the app is

The program (`rtq.bonsai`) is the two tests above transcribed, over the
same stored triangle (a vertex, two edges, two ids), and two queries:
`argmin(distmin, filter(intersects, triangles))` and `any(intersects,
triangles)`. The tree (`schedules/trees/bvh8.bonsai`) is a node holding
`children : array[BVH8, 8]` with one `with AABB(lo, hi) on children`, and a
leaf holding a run of triangles. The layout is Embree's bytes (2026-10-01,
the user's direction: "get the layout matching perfectly, use Scion's
techniques"): a 256-byte node row of `children : vector[u64, 8]` and the
six bound vectors `lower_x, upper_x, lower_y, upper_y, lower_z, upper_z :
vector[f32, 8]` in Embree's order, with `lo` and `hi` *derived* from them
(`lo = vector[vec3f, 8]{lower_x, lower_y, lower_z}`, Scion's `x = e`);
176-byte Triangle4 blocks for the leaves, a group of the elements whose
rows are an anonymous `group[4]` -- Scion's array-of-structs-of-arrays --
so that each field is a vector over the four (`v0.x[4], v0.y[4], v0.z[4],
e1..., e2..., geomID[4], primID[4]`); and a reference decoded from its bits:
`kind = ref & 15`, 0 a node at row `ref >> 4`, else a leaf of `kind - 8`
blocks from block `ref >> 4`. The driver asserts the row and block sizes
and the field offsets against Embree's, and the byte counts it reports
(285,184 for head's 1114 rows, 872,256 for its 4956 blocks) are the ones
Embree's own statistics print for its tree. The schedule
(`schedules/embree.bonsai`) sorts the children by `distmin(r, AABB{lo[i],
hi[i]})` -- the slab test's entry distance, infinite for a miss -- and
`loopify(564)`s both traversals; `occluded` is not sorted, as Embree's is
not.

The driver builds the tree with `rtcBuildBVH` under the settings above,
its callbacks writing rows and slots exactly as `AABBNode::clear/set` and
`TriangleM::fill` do, with the reference encoding above (the root is row
0, since Embree creates the root first). Its ray sets are a 1024 x 1024
camera image and, from Embree's hits, an ambient-occlusion batch and a
diffuse batch (Aila and Laine's sets). Timing is the least of `--repeats`
runs after a warm-up.

The user's rule for the measurement (2026-10-01): both sides
single-threaded, pinned with numactl to a performance core; a
multi-threaded, unpinned number is not a fair comparison. So the driver
supplies the program's parallel loop itself (`BONSAI_PARALLEL_EXTERNAL`)
and runs Embree's rays through the same loop -- a plain serial loop at
`--threads 1`, the default -- and `compare.sh` pins the run to the physical
core with the highest `amd_pstate_prefcore_ranking` (the frequency chiplet
of the 9950X3D, cpus 8-15; the V-cache chiplet is 0-7), with its node's
memory. All-core, unpinned runs made before that rule are not recorded;
the tables below are pinned and single-threaded.

## What the compiler needed (built 2026-10-01)

Three things the tree and layout languages had said but the lowering did
not do, and two the user decided after seeing the first form.

1. **A node that bounds its children prunes at the parent.** `with AABB(l,
   h) on c` was parsed and recorded and then nothing read it: a tree
   written that way was walked with no box test at all. Now the children's
   tests are made at the node and the run of recursive calls carries one
   condition per branch -- `YieldFrom::conds`, `MultiRecurse::conds`,
   `Terminator::MultiCall::conds`, travelling exactly as the sort keys do
   -- so `sort_recursion` permutes each child's test with its key and
   reference, `queue_recursion` pushes a child only where its test held
   (and descends into the first only where its did, otherwise popping), and
   the LLVM and C++ backends make each call of an unlooped run under its
   condition. Tests: lower/child-volumes, ssa/child-volumes-sorted,
   backends/llvm/child-volumes, correctness/cpp/bvh4_child_volumes.
2. **The children as an array, tested in a loop.** The user's design:
   multi-child iteration is a `parfor` over the children that computes a
   mask (so a schedule can vectorize it into Embree's eight-wide node test),
   feeding the masked, sorted recursion -- the sorted run extended with the
   mask, not a new node. Scion's form was taken (`children : array[BVH, 8]`,
   one volume annotation over them, each child's box read out of `lo` and
   `hi` at its index): the tree lowering emits `_mask : bool[8]; parfor
   children in [0:8] { _mask[children] = <child's test> }` and the `from`
   carries `_mask[0..7]`; the sort lowering puts `_keys[children] = <key>`
   into the same loop, so one pass over the children computes what the
   parent tests and what it orders by. The loop is named for the field so
   that `trace.vectorize(triangles.Interior.children)` can find it (the
   user's spelling; not yet wired).
3. **A vector of vectors is a struct of component vectors.** `vector[vec3f,
   8]` had no LLVM type. It is now, in both backends, one vector per
   component -- the shape a varying `vec3f` already has in a gang
   (`ir::widen`) and the shape Embree stores a node's bounds in -- with
   packed storage one packed array per component (`float3x8_packed` in the
   header: `std::array<float, 8> x, y, z`), extraction of a lane per
   component, and conversion between the two per component. The user's
   decision: store the bounds SoA by component as Embree does, not as
   Scion's AoS `f32x3x8`.
4. **A group that stores nothing is a group of references.** `group[n] ref
   : u64 { kind = ref & 15u; switch kind { ... } }` -- Embree's `NodeRef`
   -- gets no array of empty rows, and the C++ header no empty struct.
5. **A parfor's body is a scope to CSE.** Value numbering and copy
   propagation scoped every loop but `parfor`, so a temporary bound inside
   the children's loop for the node's row was taken for the same binding
   after the loop and read where it was not defined.

6. **A promoted load nothing reads.** `promote_allocas` kept its load
   replacements keyed by raw pointer and read the loads' names after the
   walk; a load with no reader lost its last reference during the walk (the
   by-name index held it), and the read was of freed memory. Latent before
   this app -- AddressSanitizer fires on existing tests (ssa/loopify-nested,
   ssa/loopify-queue) -- it surfaced here as `std::bad_alloc` from the
   compiler, deterministically for some output paths and not others, which
   is what a heap-layout bug looks like. The map is keyed by the shared
   pointer now.

7. **Tiled element storage (Scion's AoSoA).** `indirect group prims[pCount]
   { group[4] { v0 : vec3f; ... }; }` -- an indirect group whose rows are an
   anonymous constant group of the element's fields -- is a group of the
   set's elements stored in tiles (`ir::Group::element`): the storage is an
   array of tile structs, each field a packed vector over the tile's lanes,
   and the layout names it as the array of elements it logically is
   (`ir::TiledArray`), so that a leaf's `range(prims, a, n)` and a reference
   to an element index it as any array. `Lower/TiledArrays.cpp`, after the
   for-each and element-reference lowerings, spells every such read as the
   tile's: element k is lane k % 4 of tile k / 4, one lane of each field
   vector, built into the element. Neither this branch nor the Scion
   artifact stored an inner group field-major before; the paper's Fig. 7
   (AoSoA) does, and this is that. Tests: lower/tiled-elements,
   backends/llvm/tiled-elements, correctness/cpp/bvh4_tiled_elements.
8. **A vector of vectors built from its components.** `vector[vec3f,
   8]{xs, ys, zs}` from three eight-wide vectors (Build::make, both
   backends): the shape the type already has in a gang and in storage, so
   the build is nothing. It is how a layout derives `lo` from `lower_x,
   lower_y, lower_z`. The parser's `vector[T, n]{...}` constructor is new;
   the editor grammar needed no change for it.

9. **Scion's reference syntax** (the user's choice, spelled with `switch`):
   `layout triangles(ref : u64 = 0u) { ...; switch ref[0:3] { 0 => Interior
   from Nodes[ref[4:63]]; _ => Leaf { data = prims[a : a + n]; }; }; }`.
   The reference is the layout's parameter with the root's value as its
   default (`ir::Group::start`, where the walk begins); `x[a:b]` of an
   integer is its bits a through b, both included, folded to a shift and a
   mask at parse time (no mask for a range to the top bit); `x[a : b]` of an
   array is `range(x, a, b - a)`; `switch <expr>` makes the expression a
   derived field of its own and switches on it. The parser turns the
   top-level switch and derived fields into the direct group of references
   the hand-written form spells (`group[n] ref : u64 { ... }`), so the
   lowering is unchanged -- lower/reference-bits.bonsai's golden is
   lower/tiled-elements.bonsai's to the letter but for the derived field's
   name. No new keywords, so the editor grammar needed no change.
10. **Embree's large-node relocation**, in the driver
   (`relocate_large_nodes`): Embree's `layoutLargeNodes` takes the half a
   percent of nodes of greatest area off a heap from the root and copies
   them into fresh memory in depth-first order; the driver permutes the rows
   to the same order, root first, the rest in build order, and rewrites the
   references. The one difference is that Embree leaves the copied nodes'
   old slots as holes, which a permutation closes.

Also: `validate_volume` accepts an initializer holding one value per child
of an array of children; `valid_path` accepts an array of references stored
as a vector of integers.

## What the first measurement said (2026-10-01)

The scalar Embree-shaped schedule (`schedules/embree.bonsai`), one thread
on both sides pinned to cpu 11 (the best-ranked core), the least of 5 runs
after a warm-up, `--side 2048`, machine otherwise idle. Every ray agreed
with Embree on every mesh; the only disagreements were exact ties (two
triangles at one distance), which the driver counts separately.
`--embree-stats` on the head mesh shows Embree's own tree and the driver's
are the same tree: 1114 nodes and 4569 leaves on both sides.

Million rays per second, and bonsai's speed as a fraction of Embree's:

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 38.65 | 11.21 | 0.29x | 45.14 | 40.95 | 0.91x |
|                     | ao      | 13.59 |  3.93 | 0.29x | 15.06 |  7.59 | 0.50x |
|                     | diffuse | 12.38 |  3.56 | 0.29x | 14.04 |  6.89 | 0.49x |
| ganesha (4,323,658) | primary | 19.20 |  5.53 | 0.29x | 21.54 | 15.44 | 0.72x |
|                     | ao      |  6.07 |  1.87 | 0.31x |  6.50 |  3.22 | 0.50x |
|                     | diffuse |  5.67 |  1.74 | 0.31x |  5.61 |  2.92 | 0.52x |
| dragon (7,219,045)  | primary | 25.67 |  8.84 | 0.34x | 27.42 | 20.53 | 0.75x |
|                     | ao      |  4.40 |  1.59 | 0.36x |  4.68 |  2.57 | 0.55x |
|                     | diffuse |  3.92 |  1.41 | 0.36x |  4.17 |  2.32 | 0.56x |

Rays: 4,194,304 primary; one ao and one diffuse ray per primary hit
(713,933 on head, 1,119,510 on ganesha, 425,148 on dragon). Ties: 1, 18
and 17 of the primary nearest hits; none elsewhere. Trees: ganesha 287,516
nodes, 1,149,621 leaves, 1,207,801 blocks; dragon 404,753 nodes, 1,746,119
leaves, 1,973,446 blocks.

Reading it. The nearest-hit query runs at a third of Embree everywhere and
the any-hit query at a half to nine tenths; the mesh's size and the rays'
coherence move the ratios little. So the loss is not memory -- a traversal
over the same tree in the same order touches the same lines -- but what is
done per node and per leaf, and the difference between the two queries
says where. The any-hit traversal does Embree's tests one child and one
triangle at a time against Embree's eight and four at once, and that is
most of its 0.5x on incoherent rays (on coherent primary rays over the
small mesh, where the first hit comes early, it is 0.9x). The nearest-hit
traversal is further behind because of the sort: Embree sorts only the
children hit -- one hit, no sort; two, one compare; three, three; four,
five (bvh_traverser1.h) -- while this schedule's network orders all eight
keys, with their references and conditions, misses included (an infinite
key), in scalar compare-and-swaps at every node. Item 1 below, and the sort
over the hits only that follows from it.

## Where the loss is, and what closes it

The scalar schedule does per child what Embree does per node, and per
triangle what Embree does per block. In order of what they are worth:

1. **The eight-wide node test: `trace.vectorize(triangles.Interior.children)`.**
   The children's loop exists for this. What it needs: the cursor
   `<tree>.<arm>.<field>` resolved to that loop (the loop is named for the
   field, and `resolve_loops` already searches the functions `trace`
   reaches); the vectorizer turning `lo[children]` -- a lane of a vector of
   vectors indexed by the lane id -- into the component vectors themselves
   (an identity shuffle LLVM folds), the mask array into a mask vector and
   the keys into a key vector; and the sorting network and pushes reading
   lanes of those. With the bounds already stored by component, the loads
   come out as Embree's six vector loads. The user approved the spelling;
   the vectorizer work is next.

   With the mask a vector, the sort can do what Embree's does: count the
   hits and order only those -- Embree's `traverseClosestHit` switches on
   the count, with no sort for one hit and a three-compare network for
   three -- instead of the full eight-lane network over misses too, which
   the measurement above says is most of the nearest-hit query's extra
   loss over the any-hit query's. That is a property of the sort lowering
   (Lower/Sorts.cpp, SSA/SortRecursion.cpp) given the conditions it
   already carries, not a new directive.
2. **The four-wide triangle test.** Embree tests a `Triangle4` block as
   four lanes and picks the nearest valid one. The storage is Embree's now
   (item 7 above); what remains is the schedule: the leaf's element loop is
   a sequential for-all today, and a four-wide test needs it as a parfor of
   the tile's lanes with the argmin's accumulation a reduction across them
   -- the user's note (2026-10-01): "the loop over leaves might also need
   to be massaged into a parfor so we can vectorize it". The spelling by
   analogy with the children is `trace.vectorize(triangles.Leaf.data)`.
3. **The distance on the stack.** Embree pushes each child with its entry
   distance and drops a popped entry whose distance is past the best hit
   since found. Here a popped child is visited and its children all fail
   against the tightened best: the pruning happens one level later. The
   stack would carry the sort key and the pop re-test `key < best` -- which
   is sound when the key is the pruning metric, as it is here. A loopify
   refinement, for the user to decide.
4. **The bounds' byte order.** Done (item 8 above): the six component
   vectors are stored in Embree's order and `lo`, `hi` derived from them.
5. **Scion's reference syntax.** Done (item 9 above), and Embree's
   large-node relocation with it (item 10). The references are indices
   where Embree's are pointers; the root is the layout's default reference,
   row 0, where Embree stores a pointer to it. Same arithmetic per step
   (an index times the row size against a pointer); the bits differ.

Arithmetic that cannot match: Embree's `rcp` (an approximate reciprocal
and a Newton step) against a division, in the node test and the triangle
test's `1 / |den|`; Embree's `T <= |den| * tfar` against the current best
versus this program's `t < best` from the argmin. Both decide ties only;
the driver counts them.

## Known-open, smaller

- The exported batch answers with the primitive id alone; Embree also
  writes `t`, `u`, `v` and the normal. Recovering `t` from the argmin
  without a second triangle test wants the key beside the element.
- A mesh that fits in one leaf gets a one-child root from the driver, where
  Embree's root would be the leaf.
- `--side 2048` was used for the tables above so that a single-threaded
  Embree run lasts over a tenth of a second; the default 1024 is fine for
  checking agreement.
