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
multi-threaded, unpinned number is not a fair comparison. The schedule
leaves the parfor over the rays unbound, so it lowers to a plain sequential
loop (2026-10-02, at the user's reminder: the schedule had bound it to the
CPU threads, which the driver ran on one thread anyway; measured back to
back on head the two were the same to within the drift, the unbound loop a
hair ahead), Embree's rays go through a plain loop of `rtcIntersect1`, and
`compare.sh` pins the run to the physical core with the highest
`amd_pstate_prefcore_ranking` (the frequency chiplet of the 9950X3D, cpus
8-15; the V-cache chiplet is 0-7), with its node's memory. All-core,
unpinned runs made before that rule are not recorded; the tables below are
pinned and single-threaded.

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

## The schedule, step by step (2026-10-01)

The user's order, after the layout matched: the sort key's index type, the
eight-wide node test, the sort over the hits only, the carried bound with
the pop cull, the any-hit order, the four-wide leaf test, `rcp`. Each step
measured as above (head, cpu 11, least of 5, `--side 2048`), every ray
checked against Embree.

**Step 1, the key's index type.** The key `distmin(r, AABB{lo[i], hi[i]})`
is computed in the children's loop with `i` the loop's index, and the
schedule had declared `i : u8` while the loop's index is a `u32`. The
lowering converted, so the key read the child's box as `lo[cast<u8>(i)]`
beside the test's `lo[i]`, two expressions no later pass can tell are one
(the truncation discards bits only the loop bounds say are zero), and the
slab test ran twice per child. The sort lowering now substitutes the loop's
index as it is and refuses another type with a message naming the loop's
(`tests/bonsai/error/sort-key-index-type.bonsai`); the schedule says
`i : u32`. Head, million rays per second:

| rays    | intersect: Embree | bonsai | was   | now   | occluded: Embree | bonsai | ratio |
|---------|------:|------:|------:|------:|------:|------:|------:|
| primary | 38.49 | 14.34 | 0.29x | 0.37x | 45.09 | 40.75 | 0.90x |
| ao      | 13.60 |  4.79 | 0.29x | 0.35x | 15.09 |  7.59 | 0.50x |
| diffuse | 12.36 |  4.34 | 0.29x | 0.35x | 14.04 |  6.89 | 0.49x |

**Step 2, the eight-wide node test:**
`trace.vectorize(triangles.Interior.children)` (and the same on
`occluded`). The loop over the children, which the tree lowering makes and
the sort fills with the keys, is vectorized: its eight iterations are the
gang's eight lanes, so the node's six bound vectors are loaded once and the
slab test is Embree's `intersectNode` -- the ray broadcast, six fused
multiply-subtracts, the maxes and mins, one compare -- with the mask and
the keys coming out as vectors, stored once each. Every ray agrees.

What the compiler needed (SSA/Vectorize.cpp, SSA/SplitAggregates.cpp,
SSA/Convert.cpp, SSA/Linearize.cpp, SSA/CodeGen_Stmt.cpp,
CodeGen/CodeGen_LLVM.cpp):

- A lane's element of a vector the gang shares, read at the lane's own
  index (`lo[children]`), is the vector itself: of a vector of vectors
  -- held as one vector per component -- the struct of those components
  that a per-lane `vec3f` is carried as, by a reinterpretation the LLVM
  backend does member for member; of a vector of scalars, a shuffle of
  the lanes the ramp names. Not the chain of eight compares and selects a
  computed index into a short vector becomes.
- A vectorized loop whose continuation takes arguments: the body's exit
  hands them on as the header did. (This was the known limitation of two
  parfors in one function; `ssa/vectorize-two-parfors.bonsai` now pins
  the fixed behaviour.)
- The short-circuit `&&` (the slab test, and only where it hit, the
  comparison with the best) defined both its values under the merge
  parameter's own name, which to the analyses -- values are told apart by
  name -- made the merge a pass-through and lost the arm's value in the
  gang; each definition keeps its own name now, and the merge is a proper
  blend. And the linearizer's `any` guard in front of an arm whose mask
  was that merge parameter kept the parameter after it was blended away;
  the guard is rewritten with the arm.
- The printer spells a constant of a vector type as a broadcast.

Head, with the other agent's render running on another core (load 3.8;
Embree's own numbers are down a tenth from the idle runs above):

| rays    | intersect: Embree | bonsai | ratio | occluded: Embree | bonsai | was   | now   |
|---------|------:|------:|------:|------:|------:|------:|------:|
| primary | 35.08 | 11.86 | 0.34x | 39.91 | 46.47 | 0.90x | 1.16x |
| ao      | 12.59 |  4.13 | 0.33x | 13.67 |  8.38 | 0.50x | 0.61x |
| diffuse | 11.43 |  3.73 | 0.33x | 12.70 |  7.61 | 0.49x | 0.60x |

Reading it: the any-hit query, which has nothing after the node test but
the pushes, gains a fifth to a quarter and passes Embree on coherent
primary rays. The nearest-hit query does not move: its node step is
dominated by what follows the test -- the full eight-lane sorting network
over the keys, references and mask bits, misses included, and eight
conditional pushes -- which is step 3. A profile of the run (perf, head,
cpu 11) puts the nearest-hit kernel at twice the any-hit kernel's time for
the same rays.

**Step 3, the sort over the hits only, and the pushes as one store.** The
user: "Sort applied to masked children should only sort hits!" and
"Instead of 8 conditional pushes, we should have a compact+store, no?" No
new directive: the sort lowering sees that the run's conditions, keys and
children are the lanes of one vector each (the mask and key arrays the
children's loop fills, the node's vector of references) and sorts them as
vectors (SSA/SortRecursion.cpp, `sort_lanes`): the misses' keys made
infinite, the keys turned into integers that order as the floats do with
the lane in the low bits (Embree's `distance_i`, so every key is distinct
and the lane travels with it), one bitonic network of vector minimums and
maximums in descending order -- misses first, then the hits from the
farthest to the nearest in the last lane -- with the children following
their keys by a select per step. The run is left in a shape loopify reads
back (`sorted_run`): where any child is hit, the waiting hits are written
to the stack by one compacting store (`masked.compressstore`, lanes
`8 - hits .. 6`), the count advances once by `hits - 1`, and the nearest,
lane 7, is descended into; where none is, the next node is popped --
Embree's `if (mask == 0) goto pop`. Every ray agrees. Head, idle:

| rays    | intersect: Embree | bonsai | was   | now   | occluded: Embree | bonsai | ratio |
|---------|------:|------:|------:|------:|------:|------:|------:|
| primary | 38.44 | 29.58 | 0.34x | 0.77x | 44.88 | 72.94 | 1.63x |
| ao      | 13.52 |  5.97 | 0.33x | 0.44x | 15.01 | 10.05 | 0.67x |
| diffuse | 12.32 |  5.47 | 0.33x | 0.44x | 14.00 |  9.03 | 0.64x |

(The any-hit numbers are higher than step 2's because the machine was
idle for this run; its code did not change.)

What still differs from Embree's `traverseClosestHitAVX512VL8`: Embree
compacts the hits first (`vpcompressd`) and then switches on their
number -- one hit, no sort; two, one min and max; three, three; four,
five; more, an insertion sort -- so a node with one or two hits, the
common case, costs a handful of operations where this network always
costs its six steps (`shuffle`, `min`, `max`, `select` on the keys and
`shuffle`, `select` on the children: about forty vector operations). And
Embree's children come out of the sorted keys' low bits by one permute
(`vpermt2q`), where here they ride through the network. The incoherent
rays, at 0.44x, also spend their time in the leaves (step 6). Next: the
carried bound with the pop cull (step 4), the any-hit order and its
compacting push (step 5), the four-wide leaf (step 6), `rcp` (step 7);
the count-specialized sort is a refinement to measure after those.

**Step 4, the carried bound and the pop cull.** The argmin lowering now
has each child carry its bound -- the metric's lower bound over the
child's box, the `distmin` the mask already computes, stored into
`_carry0` beside the mask in the children's loop -- as a second value of
the traversal's recursion, `rec(tris, _bound0 := -inf)`, and the whole
node body sits under `if (_bound0 < best)`. Embree's `StackItem.dist` and
`if (stackPtr->dist > ray.tfar) continue`, as the lowering of argmin and
not a directive: the user had said this pruning "should definitely be
generated". Where the parent tested a child and let it through, the
child may still be stale when it comes off the stack, because its
siblings were visited in between and tightened the best; the test on
arrival is what skips it. The bound travels as the stack's second
array, written by the same compacting store as the children and read
back by the pop; the sort network carries it beside the children. The
test also runs on the child descended into straight away, where the
parent's test just held -- one compare Embree does not make; a stale
first child is impossible, so it is pure cost, small and noted.

What the compiler needed (Lower/Trees.cpp, Lower/Layouts.cpp): a `from`
whose branches are the child and a carried value (`Carry`, requested by
the extremum through whichever rewrite builds the `from` -- the fused
filter's, here); the recursion declared around the outermost match with
the carried value as its argument, and the trees put in front when the
traversal is wrapped in its recursion; the layout lowering keeping a
recursion's non-tree arguments. The recursion's function then takes the
bound as a parameter, and loopify gives it a stack of its own.

Head (the other agent's render running on another core, load 3.5):

| rays    | intersect: Embree | bonsai | ratio | occluded: Embree | bonsai | ratio |
|---------|------:|------:|------:|------:|------:|------:|
| primary | 35.17 | 26.96 | 0.77x | 40.25 | 46.58 | 1.16x |
| ao      | 12.66 |  5.51 | 0.44x | 13.76 |  8.42 | 0.61x |
| diffuse | 11.47 |  5.09 | 0.44x | 12.79 |  7.67 | 0.60x |

No change in the ratios from step 3: on head the stale pops the cull
skips are few, and the time of the incoherent rays is in the leaves.
Every ray agrees.

**Step 5, the any-hit order and its compacting push.** The any-hit run --
the children under their mask, no keys -- is the lanes of vectors too
(`LaneRun`, SSA/SortRecursion.h), and loopify writes it as Embree's
`traverseAnyHit` does: the hits in their stored order, every one but the
last pushed, the last continued with. The hit in the highest lane is
descended into and the others go to the stack by one compacting store
with the mask less that lane. The highest lane is the mask's bits as an
integer with its leading zeros counted (`kmov`, `lzcnt`; Embree's `bsr`
of its movemask) where the lanes are a byte or more, the maximum of the
lane indices the mask keeps otherwise (`ssa/child-volumes-any.bonsai`,
four wide). A first version took the lane by that reduction for eight
lanes too and cost a tenth on incoherent rays against the conditional
pushes it replaced, the reduction and the lane's extraction sitting on
the path to the next node's address; with the count of leading zeros it
is level with them on incoherent rays and ahead on coherent ones. Head,
back to back under the same load (the other agent's render on another
core):

| rays    | occluded: Embree | conditional pushes | ratio | Embree | compacting push | ratio |
|---------|------:|------:|------:|------:|------:|------:|
| primary | 40.87 | 47.77 | 1.17x | 42.96 | 59.29 | 1.38x |
| ao      | 13.96 |  8.61 | 0.62x | 14.39 |  8.56 | 0.60x |
| diffuse | 12.96 |  7.79 | 0.60x | 13.41 |  8.00 | 0.60x |

Every ray agrees.

**Step 6, the four-wide leaf.** `trace.vectorize(triangles.Leaf.data)`
and the same on `occluded`: Embree's `TriangleMIntersector1Moeller<4>`,
a Triangle4 block's four triangles tested as four lanes and the nearest
of the lanes that hit taken (`Intersect1EpilogM`, `select_min`). The
user's hunch held: the loop over a leaf's triangles had to be a parfor
to be vectorized, and the argmin's update a reduction. What the compiler
needed:

- The loop over a leaf's elements, where the leaf is a slice of a tiled
  group that begins and ends on a tile (spelled `b * 4u`, `k * 4u` in
  the layout, through a cast or a shift), is a loop over the tiles with a
  parfor over the lanes of each inside (Lower/ForEachs.cpp), named for
  the arm's field as the children's loop is (`ForEach::label`,
  Lower/Trees.cpp). The element's index is spelled as the tile's times
  the width plus the lane, and the tiled array's lowering reads the tile
  and the lane off that spelling with no division (Lower/TiledArrays.cpp)
  -- exact, the lane being the index of a loop over `[0, 4)`.
- The argmin's update is `_best0 argmin= (t, ref)`, an accumulate, where
  it was a plain write under the fused filter's test; the quantifiers'
  likewise (`max=` for `any`, `min=` for `all`). One element at a time
  the backend folds the repeated compare; in a gang the vectorizer
  reduces the lanes: the keys of the lanes that hit folded to their
  minimum, the first lane holding it found, its reference taken, and one
  accumulate of the pair into the running best if any lane hit
  (SSA/Vectorize.cpp). Into per-lane memory -- the packet traversal of
  rays, where every lane has a best of its own -- it is a store of the
  lanes whose key beats their slot's (CodeGen_LLVM_SSA.cpp,
  SSA/CodeGen_Stmt.cpp). The SSA builder takes an argmin or argmax
  accumulate (SSA/Convert.cpp), and the fused-multiply-add contraction
  takes a value from a dominating block as it is (SSA/Contract.cpp).

Head (the other agent's render on another core, load 4.9):

| rays    | intersect: Embree | bonsai | was   | now   | occluded: Embree | bonsai | was   | now   |
|---------|------:|------:|------:|------:|------:|------:|------:|------:|
| primary | 37.93 | 35.47 | 0.77x | 0.94x | 44.33 | 77.80 | 1.38x | 1.76x |
| ao      | 13.37 |  8.85 | 0.44x | 0.66x | 14.83 | 16.68 | 0.60x | 1.13x |
| diffuse | 12.17 |  8.26 | 0.44x | 0.68x | 13.82 | 15.53 | 0.60x | 1.12x |

Every ray agrees. The any-hit query is now ahead of Embree on every
batch; the nearest-hit query is at 0.94x on coherent rays and two thirds
on incoherent ones, where what remains is the sort network over all
eight lanes against Embree's count-specialized one (step 3's note), the
compare on arrival Embree does not make on the descended child (step 4),
and `rcp` (step 7).

**Step 7, `rcp`.** The program says `rcp(zero_fix(r.d))` and
`rcp(absDen)` where it divided, and `rcp` is a new intrinsic of the
language (the user's choice): on x86 the estimate instruction --
`vrcp14ps` with AVX-512VL, `rcpps` otherwise -- refined by one Newton
step as two fused multiply-adds, `r + r * (1 - x * r)`, a scalar or a
three-vector padded to the register's four lanes; exactly Embree's `rcp`
and `Vec3fa` reciprocal (common/simd/vfloat4_sse2.h, math/vec3fa.h),
which its `rcp_safe(dir)` and `MoellerTrumboreHitM::finalize` take. A
target with no estimate to name divides (CodeGen_LLVM, CodeGen_X86;
tests at the parse, LLVM and execution levels, the last checking the
refinement against the exact quotient to eight ulps). The estimate is
declared speculatable so that the ray's reciprocal direction hoists out
of the traversal's loop as the division did; without that LLVM kept it,
and the whole ray setup after it, at every node, and the any-hit query
lost a fifth. The leaf's `rcp` is not computed for the any-hit query,
whose test reads only whether the triangle was hit, as Embree's
occluded epilog does not finalize.

Head, back to back, idle (load 1.3):

| rays    | intersect: Embree | divide | ratio | Embree | rcp   | ratio | occluded: Embree | divide | ratio | Embree | rcp   | ratio |
|---------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| primary | 37.26 | 34.99 | 0.94x | 36.43 | 33.16 | 0.91x | 44.03 | 77.41 | 1.76x | 42.66 | 70.74 | 1.66x |
| ao      | 13.07 |  8.68 | 0.66x | 12.86 |  8.46 | 0.66x | 14.49 | 16.44 | 1.13x | 14.29 | 15.86 | 1.11x |
| diffuse | 11.90 |  8.10 | 0.68x | 11.72 |  7.90 | 0.67x | 13.52 | 15.30 | 1.13x | 13.32 | 14.77 | 1.11x |

Every ray agrees either way (the two reciprocals differ in the last
bits, and the driver counts ties). The rcp program is a few percent
behind the dividing one on coherent rays in this pairing, within the
drift Embree's own numbers show between the two runs (3%); it is what
Embree computes, so it stays.

## Where it stands (end of 2026-10-01), and what is left

All seven steps are in: the schedule is Embree's traversal -- the
eight-wide node test, the hits sorted and written to the stack by one
compacting store with their bounds, the nearest descended into, a popped
entry culled against the best, the any-hit order, the four-wide leaf with
the nearest lane taken, and Embree's reciprocal. Every ray agrees on
every mesh. Million rays per second, one thread pinned to cpu 11, least
of 5, `--side 2048`, load about 1.3:

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 35.36 | 31.47 | 0.89x | 40.41 | 59.02 | 1.46x |
|                     | ao      | 12.68 |  8.09 | 0.64x | 13.79 | 14.47 | 1.05x |
|                     | diffuse | 11.50 |  7.52 | 0.65x | 12.81 | 13.51 | 1.05x |
| ganesha (4,323,658) | primary | 18.19 | 13.31 | 0.73x | 20.24 | 21.80 | 1.08x |
|                     | ao      |  5.68 |  3.59 | 0.63x |  6.04 |  5.28 | 0.88x |
|                     | diffuse |  5.31 |  3.37 | 0.63x |  5.21 |  4.45 | 0.85x |
| dragon (7,219,045)  | primary | 23.96 | 21.25 | 0.89x | 25.18 | 28.41 | 1.13x |
|                     | ao      |  4.08 |  2.82 | 0.69x |  4.32 |  3.76 | 0.87x |
|                     | diffuse |  3.64 |  2.50 | 0.69x |  3.86 |  3.22 | 0.83x |

Against the first measurement: nearest-hit from 0.29-0.36x to 0.63-0.89x,
any-hit from 0.49-0.91x to 0.83-1.46x. The numbers drift between runs by
up to a tenth on both sides with the machine's state (the other agent's
renders came and went during the day; Embree's own head primary any-hit
read 40.4 to 44.3 across the day's runs), so a ratio is good to a few
hundredths.

What was left that evening, in the order it was worth (the first two are
done the next day; see the section after this one):

1. **The sort over the hits is a full network.** Embree compacts the hits
   (`vpcompressd`) and switches on their count -- one hit, no sort; two,
   one min and max; three, three; four, five; more, an insertion sort --
   and takes the children out of the sorted keys' low bits by one permute
   (`vpermt2q`). Here one bitonic network of six steps runs whatever the
   count, the children riding through it by a select per step: about
   forty vector operations where Embree's common case is a handful. The
   nearest-hit query's remaining gap on coherent rays is mostly this. A
   lowering of the same sort directive: the count switch over the
   compacted vectors, with a dynamic permute in the IR to take the
   children and the bounds out by the sorted lanes (a `permute(v, idx)`
   lowered to `vpermps`/`vpermt2q`, a shuffle with run-time indices,
   which the IR does not have).
2. **The incoherent rays.** At two thirds of Embree on ao and diffuse
   rays for the nearest hit, and a sixth behind for any hit on the large
   meshes, where coherent rays are at or ahead. Both sides walk the same
   nodes and leaves, so the difference is per step: the sort (above), the
   compare on arrival the descended child gets and Embree's does not
   (step 4), and whatever the gang's leaf costs over Embree's hand-written
   one -- to be read off a profile of the ao batch, node step against leaf
   step.
3. **The references are indices** where Embree's are pointers, and the
   root is the layout's default reference where Embree stores a pointer
   to it; the relocation closes holes Embree leaves. Same work per step,
   different bits.

Arithmetic that cannot match: Embree's `T <= |den| * tfar` against the
current best versus this program's `t < best` from the argmin. It decides
ties only; the driver counts them. (Embree's `rcp` matches since step 7.)

## The differences, read off the machine code, and what was done about them (2026-10-02)

The user asked for the generated code and Embree's to be compared
difference by difference, the most concerning addressed, starting with a
dynamic sorting network for the masked children (with Halide's dynamic
shuffle as the model), and a profile. The objects compared: `-b cpp`'s
`rtq.o` (`objdump -d`) against Embree's `bvh_intersector1_bvh8.cpp.avx512`
object (`BVHNIntersector1<8, 1, false, ArrayIntersector1<
TriangleMIntersector1Moeller<4, true>>>::intersect` and `::occluded`), and
`perf record -e cycles:u` of the driver on head, `--side 2048`, both sides
in one run, pinned to cpu 11.

**The profile (morning, before any change).** Of the samples, our
nearest-hit kernel took 31.3% and Embree's 25.4%; our any-hit kernel 16.7%
and Embree's 18.8%. Embree's nearest-hit kernel by region: ray setup 27%
(12.8% on the instruction after the ray's load -- the driver stores the
RTCRayHit it just built and `intersect` loads it as vectors, a
store-forwarding stall per ray that is Embree's API cost, in our favor),
node test 21%, sort and push 21%, leaf 25% (22% test, 4% epilog), pop 5%.
Ours, by instruction, had the sort as its hottest region: the compare-exchange
network and the children's selects through it.

**The differences, in the order of their cost, and their state:**

1. *The sort.* Ours: one six-layer bitonic network over all eight lanes
   whatever the count, the children and their bounds following the keys
   with a shuffle and a masked move per layer each, about eighty-five
   instructions for a node with any hit. Embree's: the hits' keys
   compressed to the front (`vpcompressd`), a switch on their count --
   nothing for one, one min/max for two, three for three, six for four,
   an insertion sort past that -- and the children taken out of the sorted
   keys' low bits by one `vpermt2q` each, about fourteen instructions for
   one hit and thirty for two. **Done**: two intrinsics, `permute(v, idx)`
   (Halide's dynamic_shuffle; `vpermd`/`vpermq`/`vpermilps`/`vpermw` on x86
   by shape, an extract per lane elsewhere) and `compress(v, mask)` (LLVM's
   vector.compress, `vpcompressd` into a zeroed register), and the lane
   sort rewritten as Embree's: compress, a switch on the count with Knuth's
   shortest networks for two to four hits and the bitonic for more, one
   permute per varying vector at the join, the pushes unchanged
   (SSA/SortRecursion.cpp; commits 14eaafa6, 5833b513). About twenty
   instructions for one hit, thirty for two.
2. *The node test's near and far bounds.* Ours: six `vblendmps` per node
   picking each slab's near and far vector by the direction's sign, plus
   two mask-register reloads per node from register pressure. Embree's:
   byte offsets computed once per ray (`TravRay::nearX`) and the six
   vectors loaded straight through them. **Done**: a pass run last in
   LLVM's optimizer folds `select (splat c), (load (gep p, 64)), (load
   (gep p, 96))` into `load (gep p, (select c, 64, 96))` with the select
   placed where `c` is defined, outside the loop
   (CodeGen/FoldSelectOfLoads.h). Last, because the inliner and GVN first
   merge the three copies of the test (the mask's, the sort key's, the
   carried bound's -- each a call of the same function) into one, and
   because instcombine turns a load from a *selected pointer* back into two
   loads and a select (and then a branch); it leaves a selected integer
   offset under a gep alone. A codegen-time fold was tried first and did
   both of those things wrong. The reloads went with the sort's constants.
3. *The leaf's nearest lane.* Ours: `llvm.vector.reduce.fmin`, libm's fmin
   semantics, which x86 has no instruction for: a compare for NaN and a
   blend around each of three `vminss`, twelve instructions. Embree's:
   `vreduce_min`, four. **Done**: a float min or max over lanes is the
   binary min (`b < a ? b : a`, std::min, as the binary op already was)
   folded over halves: a shuffle and one `vminps` per halving
   (CodeGen_LLVM::visit(VectorReduce); 04f493f2). Also changed the
   stdlib's `max` over a vec3, from fmax to the same fold.
4. *The carried bound's compare on arrival.* Ours: after the slab test's
   `kortest`, a second compare of the eight entry distances against the
   best and a second `kortest`, three instructions per node. Embree:
   clamps `tFar` by the current `tfar` inside the same min, no extra
   compare. **Open**: folding `tNear <= tFar && tNear < best` into
   `tNear <= min(tFar, best)` changes `<` to `<=` at equality, which is
   the tie rule noted below; worth a tenth of the node test.
5. *The leaf's early exit.* Embree stops a Triangle4 block after the
   edge tests when no lane passed (`if (none(valid)) return`), before `T`,
   the range compares and the `rcp`; the vectorized leaf here computes all
   of it and tests the mask once at the end, about ten instructions per
   block that misses, which is most blocks. The `rcp` is among them
   because its estimate is speculatable and LLVM hoists it above the test.
   **Done, and measured a loss** (afternoon of 2026-10-02): `trace.skip(
   intersectsp_ray_tri)` -- the skip directive now names a helper's `if`s
   wherever inlining carried them (commit 85247b46; it used to sit under
   `trace` and do nothing) -- puts `kortestb; je` after the edge tests and
   `kandw; kmovd; test; je` after the depth test, Embree's two exits. Back
   to back with and without, machine quiet: nearest hit level on primary
   rays, 6% slower on head's ao and diffuse rays; any hit (`occluded.skip`,
   one guard, the second return's fall-through being dead for any hit)
   level on primary, 5-7% slower on the incoherent rays of all three
   meshes. Whether any lane of a block survives the edge tests is a coin
   flip on those rays, and the mispredictions cost more than the ten
   instructions skipped. Kept in embree.bonsai, which follows Embree step
   for step (the user: "an apples-to-apples benchmark"); left out of the
   any-hit query in tuned.bonsai (see "Two schedules" below).
6. *The pushed children are not prefetched.* Embree issues `prefetcht0`
   for four lines of every hit child as it extracts it, before the sort,
   so a child popped later is in cache; twenty prefetches in its
   nearest-hit kernel, eight in any-hit. Ours issues none. Both sides walk
   the same nodes, so on the large meshes the incoherent rays (ao,
   diffuse) wait on the same misses, which is the likeliest part of the
   gap left there (0.73x-0.79x). **Done**: the `prefetch` scheduling
   directive, `trace.prefetch(triangles.Interior.children)`, Halide's
   `prefetch(g, at, from, offset)` with the region taken from the
   reference rather than from bounds inference (IR/Schedule.h,
   ir::Prefetch): Lower/Prefetches.cpp appends `if _mask[i] {
   prefetch(children[i]) }` to the children's loop, Lower/Layouts.cpp
   lowers the prefetch of a reference through the arm switch a visit goes
   through to each arm's row address with the widest arm's bytes (256,
   node and leaf alike, as `BVH::prefetch` fetches four lines of either),
   SSA/Linearize.cpp predicates it like a store, and
   CodeGen_LLVM::emit_prefetch issues it per lane with four `prefetcht0`
   per hit child -- first as a `tzcnt`/`blsr` loop over the mask with the
   pointer vector spilled once before the loop (measured below, a loss),
   then as Embree's sequence: the addresses compressed in a register and
   each lane's taken out at a constant index behind a test on the count
   (the evening section below). On the one-arena layout one prefetch
   statement per node, no switch on the kind.
7. *The any-hit kernel's register pressure.* It reloads its six node
   constants (the broadcast reciprocal direction and origin) from the stack
   every node; the leaf's temporaries and the ray's constants fill the
   thirty-two registers. Embree's leaf reloads the ray's origin and
   direction from the ray struct instead, which is free (a broadcast from
   memory). **Done** (commit cc7ec269): the reloads were LLVM's doing, not
   register pressure -- instcombine pushes a negation into a multiply's
   right operand and hoists the negation of a loop-invariant one, so the
   cross product with the ray's direction kept three negated copies of the
   direction live through the leaf. The cross product itself was already
   Embree's bit for bit (the same fused multiply-subtract, the same
   roundings): an instruction selection problem, not a semantic one. The
   contraction pass now puts the sign on the operand computed in the loop,
   it folds into `vfmsub`, and the any-hit kernel went from 267 to 256
   instructions with no reloads; back to back, 2% faster.
8. *The leaf's lane pick.* Picking the nearest lane's triangle: fourteen
   instructions, the four lanes' references written to a stack slot and
   the picked one read back; Embree's eight, reading the primitive id from
   the block. **Tried and reverted**: a lane extracted at a run-time index
   as the index broadcast, a permute by it and lane zero taken (commit
   bb22d1f7) made the any-hit kernel 10-12% *slower*, found by building
   the compiler at each of the day's commits in a scratch worktree and
   measuring head four ways back to back. The same lowering applied to the
   child the any-hit traversal descends into -- the lane the mask's first
   bit names, on the one chain the traversal waits on -- and there the
   broadcast, its widening to 64-bit lanes (`vpmovzxdq`) and `vpermq zmm`
   are each a few cycles, where the stack slot's store does not wait on
   the index and the load is forwarded from it. In the leaf the two were
   level. The stack slot stays (CodeGen_X86.h records why).
9. *The packed sort key.* Four instructions (sign flip for negative
   floats) where Embree's is one `vpternlogd`, because Embree knows its
   distances are non-negative (`tnear` clamped to zero). **Open**, small;
   a fact the program could state.
10. *The `hits != 0` test* after each sort case is a `kmov` and a `test`
    where the popcount is at hand. **Open**, trivial.
11. *Pointers against indices*, and Embree's epilog writing `t`, `u`, `v`
    and the normal where ours writes the id: same work per step, different
    bits; in Embree's favor and ours respectively, both small.

**Where it stands after 1-3** (million rays per second, one thread pinned
to cpu 11, least of 5, `--side 2048`; every ray agrees on every mesh):

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 38.34 | 42.13 | 1.10x | 43.86 | 76.38 | 1.74x |
|                     | ao      | 13.34 | 10.70 | 0.80x | 14.76 | 16.38 | 1.11x |
|                     | diffuse | 12.18 |  9.86 | 0.81x | 13.69 | 15.19 | 1.11x |
| ganesha (4,323,658) | primary | 19.16 | 16.68 | 0.87x | 21.61 | 25.14 | 1.16x |
|                     | ao      |  6.11 |  4.47 | 0.73x |  6.59 |  6.19 | 0.94x |
|                     | diffuse |  5.67 |  4.15 | 0.73x |  5.59 |  5.07 | 0.91x |
| dragon (7,219,045)  | primary | 25.11 | 25.44 | 1.01x | 26.54 | 32.10 | 1.21x |
|                     | ao      |  4.27 |  3.35 | 0.79x |  4.54 |  4.19 | 0.92x |
|                     | diffuse |  3.82 |  2.93 | 0.77x |  4.05 |  3.64 | 0.90x |

Against the previous evening: nearest-hit from 0.63-0.89x to 0.73-1.10x,
any-hit from 0.83-1.46x to 0.90-1.74x. Each step on head, back to back
(nearest-hit primary/ao/diffuse; any-hit the same): the sort alone 1.08x/
0.81x/0.81x and 1.65x/1.11x/1.11x; the reduction 1.04x/0.79x/0.78x (within
the drift); the picked loads 1.10x/0.80x/0.81x and 1.74x/1.11x/1.11x.
The whole kernels grew in instructions while running shorter paths --
nearest-hit 347 to 397, any-hit 230 to 267 -- since the sort is now five
case bodies where it was one network; Embree's are 604 and 359, with its
epilog's filter and mask handling in them.

What was left at that point, in the order it was worth: the incoherent
rays on the large meshes (6, the prefetch), the leaf's early exit (5), and
the small ones (4, 7-10). The afternoon's work on them is the section
below.

## Two schedules, the instructions side by side, and the prefetch (afternoon of 2026-10-02)

**Two comparison points** (the user's direction). `schedules/embree.bonsai`
follows Embree step for step and keeps a step even where it measures worse
-- the leaf's early exit, item 5 -- because its table answers one question
only: does the compiler make of Embree's structure what Embree's
hand-written code is. `schedules/tuned.bonsai` is where a step is dropped
once measured worse, and answers what the schedule language can do beyond
Embree on this machine; for now it is the matching schedule for the
nearest hit and the any-hit query without the skip. It is to be changed
only after the matching schedule is level with Embree or ahead.
`compare.sh` builds and runs both by default (`--schedule embree,tuned`).

**The instructions, side by side** (the user asked for them exactly; both
kernels as `objdump -d` lists them, the nearest-hit query, 2026-10-02
afternoon). The node test, from the node row's address to the branch on
the mask. Embree (`bvh_intersector1_bvh8.cpp.avx512`, 0x1b0-0x232, 22
instructions): `test $0x8,%r14b; jne` (is this a leaf), then per axis a
`vmovaps 0x40(%r14,%rXX,1),%ymmN` of the near or far bound through the
offset chosen per ray and a `vfmadd132ps` with the reciprocal direction and
the origin term, six of each; `vpmaxsd` of the three near distances and
`tnear` (three; integer maxima, which order non-negative floats), `vpminsd`
of the three far and `tfar` (three); `vpcmpled; kortestb; je`. Ours
(gen21's `trace_all`, 23): `kxorb %k0,%k0,%k1` (a zero mask for the bound
compare), `lea` (the row's address), six `vmovups (%rXX,%rbx,1),%ymmN`
through the same chosen offsets (CodeGen/FoldSelectOfLoads), six
`vfmadd132ps`, three `vmaxps` with `tnear` in the third, three `vminps`
with `tfar`, `vcmpleps; kortestb; je`, then `vcmpltps %ymm19,%ymm24,%k1{%k2}`
-- the carried bound against the entry distances, item 4, which Embree
folds into its `tfar` clamp and which here is one more instruction under
the hit mask. The leaf check Embree does at the node (`test $0x8`) is done
here where the popped reference is decoded. The float and integer maxima
cost the same. Otherwise one for one.

The leaf test, from the Triangle4 block's address to the early exit.
Embree (0x5d0-0x6ef, 48 instructions): `imul $0xb0` (the block's offset),
nine `vmovaps` of the block's vectors, three `vbroadcastss` of the ray's
direction from the ray struct and three `vsubps (%rsi){1to4}` of its origin
from memory (`C = v0 - org`), six `vmulps`/`vfmsub231ps` pairs (the two
cross products, `Ng = e2 x e1` and `R = C x dir`), then for `den`, `U` and
`V` a `vmulps` and two `vfmadd231ps` each (three dot products), `vandps`
(`|den|`), `vandpd` (the sign), two `vxorps` (the signs onto `U`, `V`),
`vxorps` (zero), `vcmpnltps`, `vcmpnltps{k}`, `vcmpneqps{k}`, `vaddps`,
`vcmpleps{k}`, `kortestb; jne`. After the exit (0x706-0x73c, 11): `vmulps`
and two `vfmadd213ps` (`T`), `vxorps`, `vmulps 0xc(%rsi){1to4}` and
`vcmpltps`, `vmulps 0x20(%rsi){1to4}` and `vcmpleps{k}` (`|den| tnear < T
<= |den| tfar`, the multiplies from the ray struct), `kandb; kortestb; je`.
Ours (46 to the exit): `imul $0xb0`, nine `vmovups`, three `vsubps` (two
reading the origin from the stack, where the ray's constants were spilled,
one from a register; the direction stays in three registers, so there are
no broadcasts), the same six `vmulps`/`vfmsub231ps` pairs -- the cross
products are bit for bit Embree's since commit cc7ec269 -- the same three
`vmulps` and six `vfmadd231ps`, two `vandps`, two `vxorps`, `vxorps`
(zero), `vcmpleps`, `vcmpleps{k}`, `vaddps`, `vcmpneqps{k}`,
`vcmpleps{k}`, `kortestb; je` (the skip's guard, item 5). After it (12):
two `vmulps` (`|den| tnear`, `|den| tfar`, both operands in registers),
two `vfmadd213ps` and a `vmulps` (`T`), `vxorps`, `vcmpltps`,
`vcmpleps{k}`, `kandw`, `kmovd; test $0xf; je` -- the one real difference
left in the leaf, item 10: the mask tested through a general register
where Embree's `kortestb` tests it in place -- then `vrcp14ps`,
`vfnmadd213ps`, `vfmadd132ps` and the masked `vmulps` of the hit. The leaf
is Embree's within two instructions either way.

**The bisect** that found item 8's loss: the compiler built at 5a52652c
(the "after 1-3" table), cc7ec269 (the fused multiply-subtract sign) and
bb22d1f7 (`popcount != 0` to `any`, the permute extract) in a worktree
under the scratchpad with the repository's `deps` symlinked in, and head
measured with the four binaries back to back, twice. Any hit on head
(primary/ao/diffuse, million rays per second): 78.2/16.7/15.5, then
80.1/17.0/15.8 (the sign fix, +2%), then 72.4/14.9/13.9 (the permute
extract, -10 to -12%), and the same again with the skip (72.5/14.9/14.0,
the skip's cost falling on the nearest hit instead: 43.0/11.1/10.1 to
43.1/10.5/9.5). Nearest hit unchanged by the first two. The revert
restored 81.1/15.6/14.5 with both skips in.

**The prefetch, as generated** (gen21's nearest-hit kernel, the interior
children's loop; the leaves' is the same over the other mask): `vpandq`
(`ref & 15`), `vptestnmq` (kind is zero), `ktestb` (any such hit), `vpsllq
$4; vpandq` (`(ref >> 4) * 256`), `vpbroadcastq; vpaddq` (the row
addresses), `kmovd; movzbl` (the lanes as an integer), `vmovdqa64 %zmm17,
0xc0(%rsp)` (the addresses spilled once), then per hit child `tzcnt;
blsr; mov 0xc0(%rsp,%rcx,8),%rcx; prefetcht0 (%rcx); prefetcht0 0x40;
prefetcht0 0x80; prefetcht0 0xc0; jne`. Embree's per hit child, inside
its count-switched sort: `vpermt2q` (the child out of the sorted keys'
low bits, needed anyway), `vmovq`, four `prefetcht0`. Ours costs the two
loops' setup per node, about twenty instructions, and three per child
beyond Embree's. The kernels grew from 413 to 456 and 258 to 306
instructions.

**What the prefetch measured.** The matching schedule without and with it,
back to back on each mesh, one thread, least of 5, `--side 2048`, every
ray agreeing. Not the usual conditions: another project's single-threaded
job sat on cpu 11's sibling and the pbrt session was compiling, so this
ran pinned to cpu 12 (Embree's own rates read about 4% under the quiet
cpu 11 run of 11:20, both sides alike; the ratios are what to read). The
ratio is bonsai over Embree, without the prefetch then with it:

| mesh (triangles)    | rays    | nearest hit    | any hit        |
|---------------------|---------|----------------|----------------|
| head (17,674)       | primary | 1.13x to 1.10x | 1.78x to 1.70x |
|                     | ao      | 0.78x to 0.73x | 1.02x to 0.91x |
|                     | diffuse | 0.76x to 0.73x | 1.02x to 0.89x |
| ganesha (4,323,658) | primary | 0.91x to 0.85x | 1.16x to 1.07x |
|                     | ao      | 0.73x to 0.73x | 0.89x to 0.82x |
|                     | diffuse | 0.73x to 0.72x | 0.86x to 0.82x |
| dragon (7,219,045)  | primary | 1.04x to 0.99x | 1.19x to 1.14x |
|                     | ao      | 0.77x to 0.76x | 0.87x to 0.84x |
|                     | diffuse | 0.75x to 0.76x | 0.85x to 0.82x |

**Where it stands** (the two schedules as committed, both with the
prefetch; `compare.sh` on the quiet machine at 12:36, cpu 11, least of 5,
`--side 2048`, every ray agreeing; million rays per second):

| mesh (triangles)    | rays    | intersect: Embree | embree.bonsai | ratio | occluded: Embree | embree.bonsai | ratio | tuned.bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 38.49 | 42.29 | 1.10x | 45.21 | 76.02 | 1.68x | 74.02 | 1.64x |
|                     | ao      | 13.62 |  9.94 | 0.73x | 15.10 | 13.71 | 0.91x | 14.52 | 0.96x |
|                     | diffuse | 12.41 |  9.02 | 0.73x | 14.07 | 12.48 | 0.89x | 13.27 | 0.94x |
| ganesha (4,323,658) | primary | 19.20 | 16.33 | 0.85x | 21.63 | 22.89 | 1.06x | 22.92 | 1.06x |
|                     | ao      |  6.12 |  4.42 | 0.72x |  6.58 |  5.40 | 0.82x |  5.66 | 0.86x |
|                     | diffuse |  5.72 |  4.14 | 0.72x |  5.65 |  4.63 | 0.82x |  4.80 | 0.85x |
| dragon (7,219,045)  | primary | 25.78 | 25.38 | 0.98x | 27.46 | 31.18 | 1.14x | 31.10 | 1.13x |
|                     | ao      |  4.40 |  3.38 | 0.77x |  4.69 |  3.92 | 0.84x |  4.13 | 0.88x |
|                     | diffuse |  3.95 |  3.00 | 0.76x |  4.20 |  3.45 | 0.82x |  3.61 | 0.86x |

The tuned schedule's nearest-hit column is the matching one's (the same
directives). Against the 11:20 table without the prefetch (nearest
1.13/0.77/0.77, 0.90/0.73/0.73, 1.04/0.77/0.76; any hit 1.79/1.04/1.03,
1.16/0.89/0.88, 1.20/0.88/0.86), the prefetch's cost reads the same on the
quiet machine as on cpu 12.

The prefetch is a loss everywhere it is not level: 3-6% on primary rays,
3-8% on the any-hit query's incoherent rays, and nothing gained on the
nearest-hit query's incoherent rays on the large meshes, which is where it
was meant to pay.

**The profile of the ao batch** (12:38, quiet machine, cpu 11; `perf
record` of the driver on ganesha's ao rays, nearest hit only, `--batch ao
--query intersect`, the two kernels told apart by symbol -- ours is the one
function `trace_all`, Embree's `BVHNIntersector1<8,...>::intersect`; both
sides run the same rays the same number of times, so the ratio of their
sample shares per event is the ratio of the counts). Ours over Embree's,
first without the prefetch, then with it:

| event                              | no prefetch | with prefetch |
|------------------------------------|------------:|--------------:|
| cycles                             |       1.15x |         1.16x |
| instructions                       |       0.97x |         1.25x |
| branch misses                      |       0.87x |         0.90x |
| L1 data load misses                |       0.92x |         0.88x |
| demand fills from DRAM             |      12.5x  |         1.30x |

(Dragon's ao rays, with the prefetch: 1.03x, 1.05x, 0.85x, 0.80x, 1.39x in
the same order.) So on the incoherent rays our kernel executes no more
instructions than Embree's, mispredicts fewer branches and misses L1 less
often, and takes twelve times the demand fills from DRAM: Embree's
prefetch is what turns its node and leaf misses into fills that arrive
before the demand, and without one of our own every miss is paid in full.
Our prefetch does convert them -- the demand fills drop to 1.3x Embree's --
but costs exactly what it saves: 28% more instructions (two lane loops per
node where Embree's prefetches ride inside its count switch), and in the
loop the `movq 0xc0(%rsp,%rcx,8)` that reads a lane's address back from the
512-byte spill stalls on store forwarding (Zen does not forward a 64-byte
store to an 8-byte load inside it), which lands as 10.6% of the kernel's
cycles on the two `prefetcht0` after it. The hottest instructions
otherwise: the node row's second bound load (5.0%, the node miss), the
leaf tile's load (4.7%, the leaf miss), and the instruction after the
`vpcompressq` push (6.9%) -- the compacting store to memory is 8 uops at a
throughput of 3 cycles on Zen 5 (88 uops and 54 cycles on Zen 4, per
uops.info), two of them per node (children and keys), where Embree pushes
with plain stores. Embree's own hottest: the leaf load (7.1%), its node
fma (5.6%), its prefetches (7.2%), its sort's moves (7.5%).

So item 6's premise was right about *what* the incoherent rays wait on and
wrong about the remedy as built. What follows from it, in order: (a) a
prefetch that costs what Embree's does -- the addresses of both arms
blended into one vector by the kind (one loop, not two), the lanes taken
by `vpcompressq` in a register and constant-index extracts guarded by the
count, no spill, no store-forwarding stall; (b) the push as a register
compress and one unmasked store of the whole vector, with a vector's slack
past the stack's capacity, instead of the compacting store to memory; (c)
then re-measure the incoherent rays, which the profile says should move. On head the tree is 1.2 MB and in cache, so the result there is
the prefetch's own cost: two lane loops' setup, about twenty instructions
per node, and eight per hit child. On ganesha and dragon the lines a
child needs are evidently not what the incoherent rays wait on -- or the
hardware already has them in flight by the time the sort has picked the
child, and the pushed ones are popped after L1 has turned over. Either
way, item 6's premise -- that the 0.73x-0.77x on incoherent rays is the
misses Embree's prefetch hides -- is not borne out; what those rays wait
on has to be read off a profile of the ao batch (`perf stat` cycles,
instructions, branch misses and cache misses, Embree's kernel against
ours), which is the next step. The directive stays in embree.bonsai, which
mirrors Embree; the tuned schedule still carries it only because the user
asked that it stay the matching schedule for now apart from the any-hit
skip -- it is the first thing to drop from it. The tuned schedule's own
rows from this run (the no-skip any-hit builds) were hit by a burst of
the pbrt session's compiles midway and are not reported; its quiet
numbers are the 11:20 table above (any hit 1.78x/1.12x/1.12x head,
1.21x/0.98x/0.94x ganesha, 1.23x/0.95x/0.92x dragon).

## The layout matched: one arena, references as offsets (afternoon of 2026-10-02)

The user asked whether the layout matched Embree's, and it did not in
two respects, both about the child slots rather than the rows: Embree's
`NodeRef` holds the child's address with the kind in its low four bits,
ours held a row number or a tile number into one of two arrays; and
Embree's nodes and leaves share one address space, allocated in build
order from a thread-local block allocator and the largest nodes copied
to fresh space after the build, where ours were two arrays. The order
of work set: the layout first, then the schedule, then the code.

**The layout construct.** A lookup that brings its own shape
(ir::Lookup::shape): `Interior from arena[ref[4:63] * 16u] { children :
vector[u64, 8]; ... }` says the arm's row is at that byte offset of a
group of bytes, `indirect group arena[bytes] { byte : u8; }`, laid out as
the arm says. Inside the shape, a group of the tree's elements with no
size, `indirect group tiles { group[4] { ... }; }`, is the run of tiles
that begins where the row does, as many as the arm's `range(tiles, 0u,
n)` asks for -- Embree's leaf, a pointer and a count. The references are
byte offsets from the arena's start rather than addresses, so the tree
stays plain data that can be copied or handed to a device; the one `add`
of the base per visit is the deliberate difference, recorded in
bvh8.bonsai and the README. No new keyword: `from` and `group` as they
were, a layout after the index where there was none.

**The lowering.** The row is element zero of an array of the shape that
begins at the offset -- `(reinterpret<Row[]>(&arena[off]))[0]` -- the same
spelling a row of a group of rows has, so that its fields are read by the
same code; this mattered. Spelled as a dereference of a pointer to the
row, `(*reinterpret<Row*>(&arena[off])).children`, the SSA form loaded the
whole 256-byte row as a value and took the field out of it, LLVM split
the aggregate load element by element and rebuilt the children vector
with two `vpermt2q` and a broadcast, and the nearest-hit kernel ran at
0.58x on head's ao rays; loading each field through its own address
(`Load(FieldPtr)`) was worse still, 0.46x, with the near/far fold and
the shared loads of the three slab tests lost. As an array element the
code is the two-array layout's exactly, plus the base `add`. Likewise the
leaf's run: a pointer reinterpreted as the array, not dereferenced, which
read a handle out of the bytes and faulted. And `&a[0]` is `a`
(PtrTo::make, as the user suggested), so both arms' prefetch addresses are
the one offset and the prefetch is one statement with no switch on the
kind, as `BVH::prefetch` is.

**The driver** builds as before, by index, then assembles the arena: the
relocated large nodes first in depth-first order from the root (so the
root is at offset 0, the layout's default reference), the remaining
rows in build order, the blocks after them, every reference rewritten to
its offset, the arena 64-byte aligned as Embree's blocks are. The element
reference an argmin keeps is the pair of the leaf's offset and the
element's index in its run.

**Measured** (quiet cpu 11, the two-array layout then the arena, both with
the matching schedule and the prefetch, back to back; bonsai over
Embree, primary/ao/diffuse):

| mesh    | nearest hit, two arrays | arena          | any hit, two arrays | arena          |
|---------|-------------------------|----------------|---------------------|----------------|
| head    | 1.07x/0.71x/0.71x       | 1.10x/0.73x/0.73x | 1.59x/0.90x/0.88x | 1.74x/1.00x/0.99x |
| ganesha | 0.85x/0.73x/0.73x       | 0.88x/0.76x/0.76x | 1.07x/0.83x/0.83x | 1.18x/0.93x/0.93x |
| dragon  | 0.99x/0.76x/0.76x       | 1.03x/0.81x/0.82x | 1.13x/0.83x/0.82x | 1.24x/0.93x/0.92x |

Three to six percent on the nearest hit and ten to twelve on the any hit,
everywhere: the kind decode and the second prefetch loop were worth that
much. The nearest-hit kernel is 432 instructions where it was 456, with
one prefetch loop of four `prefetcht0`; the any-hit kernel 273 where it
was 306.

**Where it stands, both schedules on the arena** (`compare.sh` on the
quiet machine at 17:10, cpu 11, least of 5, `--side 2048`, every ray
agreeing; million rays per second; the tuned schedule's nearest hit is the
matching one's):

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | embree.bonsai | ratio | tuned.bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 38.30 | 43.34 | 1.13x | 45.01 | 79.78 | 1.77x | 80.99 | 1.80x |
|                     | ao      | 13.53 | 10.03 | 0.74x | 15.03 | 15.04 | 1.00x | 16.34 | 1.09x |
|                     | diffuse | 12.34 |  9.09 | 0.74x | 13.99 | 13.79 | 0.99x | 14.93 | 1.07x |
| ganesha (4,323,658) | primary | 19.15 | 16.85 | 0.88x | 21.60 | 25.52 | 1.18x | 25.25 | 1.17x |
|                     | ao      |  6.12 |  4.62 | 0.76x |  6.57 |  6.15 | 0.94x |  6.39 | 0.97x |
|                     | diffuse |  5.71 |  4.34 | 0.76x |  5.66 |  5.25 | 0.93x |  5.39 | 0.95x |
| dragon (7,219,045)  | primary | 25.71 | 26.36 | 1.03x | 27.39 | 34.04 | 1.24x | 33.50 | 1.22x |
|                     | ao      |  4.41 |  3.55 | 0.80x |  4.70 |  4.36 | 0.93x |  4.53 | 0.97x |
|                     | diffuse |  3.95 |  3.18 | 0.81x |  4.20 |  3.85 | 0.92x |  3.98 | 0.95x |

What is left on the incoherent rays is 0.74x-0.81x on the nearest hit
and 0.92x-1.00x on the any hit, with the layout now Embree's and the
schedule Embree's step for step, so it is the code generation's: the push
as a compacting store to memory and the prefetch's per-child cost, next,
in that order.

## The push and the prefetch without memory in the way (evening of 2026-10-02)

The two code-generation differences the profile of 12:38 named, fixed
in the order given, both measured back to back against the arena builds
of 17:10 (gen25/gen26 against gen27/gen28 in the scratchpad), every ray
agreeing.

**The push.** A compacting store with slack past its slots
(ir::Store::slack, `compress_store_whole(` in the Stmt form, `compact
whole` in the SSA dump): the lanes that are on packed to the front in a
register and the whole vector stored, the lanes past the count landing
in slots nothing reads. loopify (SSA/QueueRecursion.cpp) allocates each
stack a vector longer than the size asked for -- the widest run's lanes,
`!stack : mut u32[68]` for a size of 64 and four lanes -- and sets the
flag on both pushes, the sorted run's and the lane run's. On x86 the
push is now `vpcompressq %zmm22,%zmm0{%k1}{z}; vmovdqu64
%zmm0,0x128(%rsp,%r15,8)` where it was `vpcompressq %zmm22,
0x168(%rsp,%r15,8){%k1}`, the 8-uop, 3-cycle (Zen 5) or 88-uop (Zen 4)
compacting store to memory. Embree's push is a scalar store per child
from its count switch; ours stays one store per stack per node.

**The prefetch.** CodeGen_LLVM::emit_prefetch takes the lanes as Embree's
traversal takes the hit children out of a node: the addresses as
integers compressed in a register (`vpcompressq`), the mask counted
(`popcnt`), and a chain of blocks, one per lane, each entered when the
count exceeds the lane's number, taking the lane's address out at a
constant index (`vmovq`, `vpextrq`, a `vextracti32x4` first for the
upper lanes) and fetching its four lines. One misprediction per node at
the lane past the last, as Embree's count switch takes; no spill, no
store-forwarding stall. On the arena there is one such chain per node
(no switch on the kind): the nearest-hit kernel is 508 instructions with
32 `prefetcht0`, the any-hit kernel 328 with 32, the chain's eight lanes
unrolled where the loop was one body (432 and 273 before).

**What it measured** (ratio bonsai over Embree, before then after,
primary/ao/diffuse; cpu 11, least of 5, `--side 2048`):

| mesh    | nearest hit, both schedules | any hit, embree.bonsai | any hit, tuned.bonsai |
|---------|-----------------------------|------------------------|-----------------------|
| head    | 1.13/0.74/0.73 to 1.23/0.78/0.77 | 1.80/1.00/0.99 to 1.84/0.99/0.97 | 1.79/1.08/1.06 to 1.81/1.07/1.05 |
| ganesha | 0.88/0.76/0.78 to 0.92/0.78/0.78 | 1.18/0.94/0.93 to 1.18/0.92/0.92 | 1.17/0.97/0.95 to 1.16/0.97/0.95 |
| dragon  | 1.03/0.80/0.81 to 1.08/0.83/0.83 | 1.24/0.93/0.92 to 1.24/0.93/0.91 | 1.22/0.96/0.95 to 1.14/0.92/0.90 (its block ran last; see the official table) |

The nearest hit gains 4-9% on primary rays and 0-5% on incoherent ones
on every mesh. The profile of ganesha's ao rays, ours over Embree's
kernel per event, closes the loop on the 12:38 table: cycles 1.16x to
1.06x, instructions 1.25x to 1.07x, branch misses 0.90x to 0.89x, L1
misses 0.88x to 0.80x, demand fills from DRAM 1.30x to 1.12x. The
prefetch now costs about what Embree's does and converts the misses as
Embree's does; what is left of the nearest-hit gap on incoherent rays
(0.78x-0.83x) is no longer in these two places.

The any hit is level on primary rays and 1-2% down on incoherent ones
with the matching schedule. Its profile on ganesha's ao rays, before then
after: cycles 1.08x to 1.10x, instructions 1.29x to 1.28x, branch misses
1.08x to 1.11x, L1 misses 0.94x to 0.92x, DRAM fills 1.43x to 1.27x --
fewer instructions and fewer fills, 3% more branch misses, and the
cycles follow the misses. The hottest instructions after: the first
lane's `prefetcht0` pair (8.2% and 5.8%, behind the `vmovq` that waits
on the compress), the leaf tile's load (8.2%), the push's `vmovdqu64`
(4.5%, behind its compress). The any-hit kernel's own 1.28x instructions
and 1.1x branch misses against Embree's are the matching schedule's
(the leaf skip, measured 5-7% on these rays) and the sort-free run's
shape, not these two fixes; the matching schedule keeps both by
direction. Where the any hit's remaining 0.92x on incoherent rays sits
is the next profile.

**The official table** (17:34, `compare.sh --side 2048` on each mesh,
cpu 11 quiet, least of 5, million rays per second; the nearest hit is
the tuned schedule's too, within 0.01x):

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | embree.bonsai | ratio | tuned.bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 38.43 | 47.26 | 1.23x | 45.17 | 83.48 | 1.85x | 82.42 | 1.83x |
|                     | ao      | 13.61 | 10.58 | 0.78x | 15.02 | 14.90 | 0.99x | 16.15 | 1.07x |
|                     | diffuse | 12.37 |  9.59 | 0.78x | 14.02 | 13.60 | 0.97x | 14.76 | 1.05x |
| ganesha (4,323,658) | primary | 19.20 | 17.77 | 0.93x | 21.62 | 25.36 | 1.17x | 25.15 | 1.16x |
|                     | ao      |  6.10 |  4.82 | 0.79x |  6.58 |  6.06 | 0.92x |  6.32 | 0.96x |
|                     | diffuse |  5.73 |  4.52 | 0.79x |  5.64 |  5.17 | 0.92x |  5.35 | 0.95x |
| dragon (7,219,045)  | primary | 25.74 | 28.02 | 1.09x | 27.43 | 34.13 | 1.24x | 33.47 | 1.22x |
|                     | ao      |  4.40 |  3.66 | 0.83x |  4.68 |  4.31 | 0.92x |  4.52 | 0.96x |
|                     | diffuse |  3.95 |  3.28 | 0.83x |  4.20 |  3.81 | 0.91x |  4.19 | 0.95x |

(The tuned dragon row of the back-to-back table above, 1.14/0.92/0.90,
was its last block and is not borne out here: 1.22/0.96/0.95, the same
as before the fixes.) Note that `compare.sh` without `--side` runs
1024-wide batches, a quarter of the rays, and the incoherent ratios
read higher there (ganesha ao 0.85x nearest, 0.94x any hit at 17:32);
the tables in this plan are all `--side 2048`.

## Known-open, smaller

- The exported batch answers with the primitive id alone; Embree also
  writes `t`, `u`, `v` and the normal. Recovering `t` from the argmin
  without a second triangle test wants the key beside the element.
- A mesh that fits in one leaf gets a one-child root from the driver, where
  Embree's root would be the leaf.
- `--side 2048` was used for the tables above so that a single-threaded
  Embree run lasts over a tenth of a second; the default 1024 is fine for
  checking agreement.
