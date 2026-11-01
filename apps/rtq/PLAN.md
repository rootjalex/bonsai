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
misses 0.88x to 0.80x, demand fills from DRAM 1.30x to 1.12x. (A
caution found afterwards: in an `--batch ao --query intersect` run
Embree's intersect symbol also holds the untimed primary pass that
makes the ao rays' origins, about a sixth of its samples, so these
ratios and the 12:38 table's understate ours over Embree's by that
much; the before/after movement is real, the levels are corrected in
the next section, which subtracts the pass. The any-hit ratios are
clean: there the primary pass is a different symbol.) The prefetch now
costs about what Embree's does and converts the misses as Embree's
does; what is left of the nearest-hit gap on incoherent rays
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

## The two kernels read side by side, and counted (night of 2026-10-02)

The user asked for the generated code of both sides to be inspected and
profiled, the differences found, and the general compiler improvements
in them named. Both kernels of the current build (gen27, the arena, the
push and prefetch fixes) against Embree 4.4.1's
`BVHNIntersector1<8,1,false,ArrayIntersector1<TriangleMIntersector1Moeller<4,true>>>::intersect`
and `::occluded` (3232 and 2002 bytes of code; ours 508 and 328
instructions), annotated with three profiles of ganesha's ao rays on
cpu 11: the five-event cycle profile, a precise op-sampled profile (AMD
IBS, one sample per 100003 dispatched-and-retired ops, no skid, so a
block's sample density is its execution count), and the TLB events.
The listings and annotations are in the scratchpad
(`annot-gen27-trace-cycles.txt`, `annot-embree-intersect-cycles.txt`,
the `occ`/`occluded` pair, `ibsops-*.txt`).

**The clean per-event table.** Embree's intersect symbol in an
`--batch ao --query intersect` run holds the untimed primary pass too;
the `--query occluded` run of the same batch holds only that pass in
that symbol, so subtracting the two gives Embree's ao-ray counts. Ours
over Embree's, ganesha ao rays:

| event                   | nearest hit | any hit |
|-------------------------|------------:|--------:|
| cycles                  |       1.27x |   1.10x |
| instructions            |       1.39x |   1.28x |
| branch misses           |       1.02x |   1.11x |
| L1 data load misses     |       0.93x |   0.92x |
| demand fills from DRAM  |       1.29x |   1.27x |

The cycles match the measured rates (0.79x and 0.92x). Instructions per
cycle: ours 0.90 against Embree's 0.83 on the nearest hit. So the
nearest-hit gap is instruction count, nearly alone: we execute 39% more
instructions on the same rays, mispredict the same, miss L1 less.

**The same work.** The op-sampled counts, three runs of the ao batch,
per block: node tests ours 37.35M, Embree 37.28M; pops 11.95M and
12.39M; leaf tiles tested 7.01M and 6.84M; the hit distribution read
off Embree's exclusive paths, of all node tests: 20% hit no child, 61%
hit exactly one, 15% two, 3% three, 1% four or more. Both traversals
visit the same nodes in the same order with the same hits, as the
agreement check says they must. Ops per node visit, everything
amortized: ours 105 against Embree's 64 on the nearest hit, 81 against
55 on the any hit (ops, not instructions: our `zmm` compresses,
permutes, extracts and 64-byte stores are two or more macro-ops each
where Embree's `ymm` forms are one; hence 1.63x in ops where the
counter says 1.39x in instructions).

**Per node, instruction by instruction** (the executed path, from the
listings; Embree in parentheses):

- A node hit by one child, entered from a pop: ours 86 (47). Pop 8
  (7); kind test 3 (2; ours copies the reference first because the
  base add needs it whole); the slab test 28 (21): the same six loads,
  six fmas, three max and three min, then ours has `kxorb`, two
  compares with a `kortestb; je` between them, `kmovd`, `kortestb; je`
  where Embree has one integer compare (its near is clamped to
  `max(tnear, 0)` at entry so floats compare as ints) and one `kortestb;
  je`, its `tfar` folded into the far min; the prefetch 13 (within
  Embree's 17-instruction one-hit tail, which also finds the child):
  `vpandq` the kind off, broadcast the base, `vpaddq`, a zero idiom,
  `vpcompressq`, `vmovq`, four `prefetcht0`, the count test; then
  ours falls into the generic sort-and-push path -- the key build 9
  (blend to +inf under each of the two masks, `vpmovd2m`, `vpandd`,
  `vpxord` under the sign mask, `vpor` the lane, `vpcompressd`, and
  `vpshufd; vpermq` to reverse the lanes so the nearest sits last) and
  its tail 22 (the waiting-lane mask from the count: `movl; subl;
  vpbroadcastd; vpcmpltd`; `vpermd` of the distances, `vpmovzxdq;
  vpermq zmm` of the children, zero idiom, `vpcompressq`,
  `vextracti32x4 $3; vpextrq` for the nearest, zero idiom,
  `vcompressps`, `vextracti128; vpshufd; vucomiss; jbe` testing the
  nearest child against the bound again, two 64/32-byte stack stores
  of nothing, the count update). Embree's one-hit case: `vpternlogd`
  makes the keys in one op, `vpcompressd`, `vpermt2q` picks the child,
  `vmovq`, done.
- A node hit by two: ours 100 (67). Ours adds the second lane's
  prefetch (10), the two-key network (`vpermd; vpminsd; vpmaxsd;
  vpermt2d`, 5) on top of a second copy of the key build (11), then
  the same 22-instruction tail; Embree: a second permute and
  prefetch, `vpminsd/vpmaxsd`, two permutes for the child and the
  pushed one, two scalar stores (20).
- A leaf tile: 51 (49). The same nine loads and the same arithmetic;
  Embree re-broadcasts the ray's origin and direction from the ray
  struct, ours keeps them in registers.
- The any-hit node, one child hit: ours 71 (40). Ours tests the
  found-flag at the loop head (2) and reloads three constants from
  the stack (3); after the prefetch chain it enters a push block of
  19 instructions for every hit node -- `lzcnt` and `xor 7` for the
  highest lane, a broadcast and a compare for the waiting mask, a
  zero idiom, `vpcompressq`, the 64-byte stack store (of nothing, for
  one hit), and the next child taken out of the children vector at a
  run-time lane by a 64-byte spill and an 8-byte reload inside it
  (`vmovdqa64 %zmm22,0xc0(%rsp); movq 0xc0(%rsp,%r11,8)`, which Zen
  does not forward) -- where Embree's is `tzcnt`, one scalar load of
  the child from the node, and for more hits a `blsr` loop of scalar
  stores. Its leaf loop tests the flag per tile (3).

**What is not it.** Branch misses are equal (nearest hit). The TLB:
Embree's tree sits on 2 MB pages (`hugepages = enabled` in its device
report; 74K 2 MB TLB reloads and 6K 4 KB ones in its kernel), ours on
4 KB pages (3.8M reloads, 0.72M page walks over three runs, no 2 MB
reloads) -- thirteen times Embree's reloads, but only 1.3 per ray,
about 1% of cycles; it is the likeliest source of the 1.28x demand
fills from DRAM (page-table lines), and a one-line fix in the driver
(2 MB-aligned arena with `madvise(MADV_HUGEPAGE)`), or in the runtime
for every large buffer. The per-ray entry: 9% of Embree's kernel
cycles sit on the first instructions after its 16-byte load of the
ray's direction, which the driver's `to_rayhit` wrote field by field
just before the call -- a load across fresh scalar stores is not
forwarded -- plus the indirect call and its 9.6 KB frame; our batch
loop reads an array prepared in advance. That is why ours is faster on
primary rays (1.23x head, 1.09x dragon) despite more instructions per
node, and it is a fairness point for the driver: convert the rays
before the timed loop.

**The general compiler improvements, ranked by what they save.**

1. *A run of one call is a call.* In the count switch the sorted run
   lowers to (SSA/SortRecursion.cpp, SSA/QueueRecursion.cpp), case 1
   must not fall into the generic path: no keys, no sort, no
   compacting store of an empty mask, no count update; the next node
   is `compress(children, mask)[0]` -- a `vpcompressq` and a `vmovq`,
   the compress the prefetch already takes (compress the raw
   references, derive the prefetch addresses from the packed vector)
   -- or Embree's scalar `tzcnt` and a load of that child from the
   node. The same for the lane run's push block (any hit). Saves about
   31 instructions and some 30 cycles of dependent chain on 61% of all
   node visits: the nearest hit's 105 ops per visit toward 80, the
   any hit's 81 toward 65. The largest item by far.
2. *A short-circuit `&&`/`||` whose right side is pure and cheap,
   in a vectorized context, is a plain mask `and`.* The slab test is
   `intersects && near < bound`: SSA/Linearize.cpp predicates the right
   side behind a `reduce<any>` of the left, which is `kxorb`,
   `kortestb; je`, a second blend, and six cycles more on the chain to
   the next node, per node. Unobservable to flatten when the right side
   has no effects and no loads (a compare of values in registers); the
   user's rule that `&&` short-circuits stands for right sides that
   cost (alpha tests, lookups), which a cost bound keeps.
3. *The sort key.* `(bits & ~7) | lane` is one `vpternlogd` if the
   sign flip is applied first; the blend to +inf of the lanes not hit
   is dead when the keys are compressed next; and the sign flip itself
   goes when the key is known non-negative -- `near = max(.., tnear)`
   is, once `tnear >= 0` is known, which Embree asserts by
   `max(tnear, 0)` at entry and rtq.bonsai does not (the program could,
   and a sign-range fact through max/min/abs in the simplifier would
   carry it). 5 instructions to 1-2 per multi-hit node.
4. *The cull of the child descended into is redundant*: the node test
   just proved its near distance under the bound, and the bound cannot
   have moved. `vextracti128; vpshufd; vucomiss; jbe` per multi-hit
   node (QueueRecursion emits the pop cull on the direct descent too).
5. *The waiting-lane mask from the count* -- `movl; subl;
   vpbroadcastd; vpcmpltd` under a constant mask -- is a scalar
   `((1 << h) - 1) & ~top` moved into a mask register, 2-3 ops.
6. *Any hit.* (a) The quantifier's early exit as a branch out of both
   loops where the hit is found, not a flag tested at each loop head
   (2 per node, 3 per tile). (b) A lane extract at a run-time index
   from a vector that was loaded from storage the function never
   writes is a scalar load at the lane's address (CodeGen_LLVM, with
   the mutability fact), not a spill and a reload inside it: the push
   block's store-forwarding stall. (c) The three constant reloads per
   node are register pressure the leaf's temporaries cause; worth a
   look once (a) and (b) have changed the shape.
7. *Small, each a line*: the zero idiom LLVM emits before every `{z}`
   compress (three per node; LLVM's X86 lowering of
   `vector.compress` with a zero passthru); the loop-invariant
   `vpbroadcastq` of the arena base, not hoisted; the per-ray offset
   setup (36 scalar ops to Embree's 15 -- the far offset is
   `near ^ 0x20`); the copy before the kind test.
8. *Memory*: 2 MB pages for the arena (driver or runtime), see above.

With 1-5 in, a one-hit node is about 55 instructions against Embree's
47 and a two-hit node about 85 against 67; what remains is the
eight-wide permute-and-compress push against Embree's per-count
constant permutes and scalar stores -- the same work differently
shaped -- and the `zmm` forms' extra ops. The order to do them is the
order above; 1 and 2 are where the incoherent-ray gap mostly is, and
each is measured on head, ganesha and dragon against gen27 as the
earlier fixes were.

## The generic ones built, the driver made fair, and what is not generic (later on 2026-10-02)

The user asked for the generic compiler optimizations in the list to be
implemented, the allocation to match Embree's exactly, and the rest
named with the clean way to do them. Built, with tests at the IR, LLVM
and execution levels, the full suite at 1266 of 1266:

1. **A run of one call is a call** (SSA/SortRecursion.cpp,
   SSA/QueueRecursion.cpp). The sorted run's count switch has two new
   arms: zero hits go straight to what follows the run (the pop), not
   through the join's permutes; one hit is a `Terminator::Call` whose
   varying arguments are lane 0 of each vector packed by the mask
   (`compress` + `extract 0`), which loopify takes straight to the
   child with nothing written to the stack. The lane run (any hit) is
   dispatched the same way, a switch on the count with a one-hit arm
   (`compress(children, mask)[0]`) in place of the `lzcnt`, the
   compacting store of nothing and the spill-and-reload extract. On x86
   the one-hit path of the nearest hit is now `vpcompressq; vmovq` for
   the child and `vcompressps; vucomiss` for its key and the cull.
2. **A call to a callee safe with every lane off needs no test**
   (SSA/Linearize.cpp, SSA/Linearize.h, SSA/Vectorize.cpp):
   `safe_with_all_lanes_off`, ispc's SafeToRunWithMaskAllOff applied
   through a call -- nothing in the callee touches memory but its own
   locals and the result slots its caller passes it, no effect, no
   loop over lanes, callees likewise -- and `must_skip` no longer
   forces a gadget for such a call, nor `specialize_calls` a test of
   its own. Two refinements the slab test needed on the way: a lane
   extract from a value in registers (a node's eight boxes at the
   lane's index) is not memory, since LLVM clamps the index where it
   goes through a stack slot; and a load at an address no lane had a
   hand in (`lane_independent`: uniform parameters, constants,
   arithmetic and loads over them, never a reduction's result) is
   valid whether or not a lane is on -- the accumulator's bound, the
   node's row. With these the slab test's `intersects && distmin <
   bound` is one compare, one `kortest` and one branch per node, as
   Embree's, where it was two of each and a `kxorb`.
3. **The sort's +inf blend** only in the all-lanes arm
   (SSA/SortRecursion.cpp): the compress drops the misses.
4. **No zero idiom before a compress nothing reads past the count**
   (CodeGen_LLVM::compress_lanes, `zero_rest`): the prefetch's and the
   slack store's take a poison passthru.
5. **The arena mapped as Embree maps its blocks** (rtq_hook.cpp,
   OsMemory): Embree's `os_malloc` call for call -- `MAP_HUGETLB`
   where the size wastes under 1.5% of a 2 MB page and the system has
   pages set aside, else a plain mapping with `MADV_HUGEPAGE`, which
   is what Embree gets here (`hugepages = enabled` in its report, THP
   in madvise mode) -- and freed as `os_free` does. `RTQ_PAGES=4k`
   keeps plain pages for measuring the difference.
6. **The driver made fair**: Embree's ray structs are made before the
   clock and reset between runs off it (`timed` takes a setup), where
   each was written field by field right before `rtcIntersect1` and
   Embree's first 16-byte load of it waited on the stores.

**What the fairness fix was worth to Embree** (its own rates, million
rays per second, old driver to new, cpu 11): head primary 38.3 to
63.9 (+67%), any hit 45.0 to 84.9 (+89%); ao 13.6 to 15.7 and 15.0
to 17.8; ganesha primary 19.1 to 24.1 (+26%) and 21.5 to 29.3 (+36%),
ao 6.1 to 6.6 and 6.5 to 7.2; dragon primary 25.7 to 34.4 (+34%) and
27.4 to 38.3 (+40%), ao 4.4 to 4.7 and 4.7 to 5.1. So the earlier
tables' primary-ray ratios above 1.0x were the driver's artifact, not
the traversal's; the incoherent ones were off by a tenth. Every table
before this section carries that; the ones from here on do not.

**The compiler change alone** (our rates, gen27/gen26 to gen29/gen30,
both on 4 KB pages, million rays per second, primary/ao/diffuse):

| mesh    | nearest hit                     | any hit, matching               | any hit, tuned                  |
|---------|----------------------------------|---------------------------------|---------------------------------|
| head    | 44.1/10.0/9.1 to 51.2/12.8/11.4 (+16/+28/+24%) | 82.6/14.8/13.5 to 84.9/15.9/14.5 (+3/+8/+7%) | 80.0/16.2/14.8 to 83.4/17.4/15.8 (+4/+8/+7%) |
| ganesha | 17.6/4.78/4.47 to 19.9/5.62/5.18 (+13/+18/+16%) | 25.2/6.02/5.15 to 26.1/6.38/5.42 (+3/+6/+5%) | 25.2/6.35/5.36 to 26.2/6.73/5.65 (+4/+6/+5%) |
| dragon  | 27.9/3.66/3.27 to 29.6/4.11/3.63 (+6/+12/+11%) | 34.0/4.31/3.81 to 34.5/4.54/3.98 (+2/+5/+4%) | 33.6/4.53/3.97 to 34.2/4.78/4.19 (+2/+6/+6%) |

Ops per node visit on ganesha's ao rays (IBS, as above): nearest hit
105 to 88 against Embree's 65, any hit 81 to 73 against 54. **The
pages** (gen29, 4 KB to Embree's 2 MB): head level; ganesha +1.3/+2.5/
+3.3% nearest hit and +3.4% any hit on ao; dragon +2.3/+3.6/+4.1% and
+4.4% -- more than the 1% the TLB counts suggested, the page walks
being on the critical path of a dependent chain of misses.

**The official table** (18:45, `compare.sh --side 2048`, the new
driver, cpu 11 quiet, least of 5, million rays per second; the nearest
hit is the tuned schedule's too):

| mesh (triangles)    | rays    | intersect: Embree | bonsai | ratio | occluded: Embree | embree.bonsai | ratio | tuned.bonsai | ratio |
|---------------------|---------|------:|------:|------:|------:|------:|------:|------:|------:|
| head (17,674)       | primary | 64.11 | 51.42 | 0.80x | 85.99 | 85.25 | 0.99x | 82.54 | 0.96x |
|                     | ao      | 15.73 | 12.81 | 0.81x | 17.87 | 15.95 | 0.89x | 17.25 | 0.97x |
|                     | diffuse | 14.02 | 11.41 | 0.81x | 16.31 | 14.50 | 0.89x | 15.70 | 0.96x |
| ganesha (4,323,658) | primary | 24.11 | 20.22 | 0.84x | 29.38 | 26.67 | 0.91x | 26.65 | 0.91x |
|                     | ao      |  6.65 |  5.76 | 0.87x |  7.25 |  6.59 | 0.91x |  6.97 | 0.96x |
|                     | diffuse |  6.19 |  5.34 | 0.86x |  6.20 |  5.68 | 0.92x |  5.91 | 0.96x |
| dragon (7,219,045)  | primary | 34.61 | 30.34 | 0.88x | 38.37 | 35.50 | 0.93x | 35.10 | 0.91x |
|                     | ao      |  4.71 |  4.26 | 0.90x |  5.11 |  4.76 | 0.93x |  5.00 | 0.97x |
|                     | diffuse |  4.19 |  3.79 | 0.90x |  4.53 |  4.19 | 0.93x |  4.40 | 0.97x |

**What is not a generic rewrite, and the clean way to each:**

- *The cull of the child descended into* (item 4). The argmin's prune
  test sits at the head of the loop body, which both the pop and the
  direct descent enter, so the child the node test just proved under
  the bound is tested again (`vucomiss; ja` in the one-hit arm). The
  clean fix is in loopify: the body gets two entries, the pop's at the
  test and the descent's past it, which is sound where the run's lane
  condition includes the compare of the same key against the same
  bound -- a fact the sorted run can carry (SortedRun gains the
  condition's compare) rather than one to rediscover by threading.
- *The sort key's sign flip* (item 3, the rest). The keys are the
  masked gang variant of `distmin`'s result, a call's return, so
  "non-negative" has to cross a function boundary: an interprocedural
  range summary of return values (returns non-negative: `max(.., c >=
  0)`, `abs`, `inf`, selects of such), plus the program clamping
  `tnear` and `tfar` to zero at the ray's setup as Embree's TravRay
  does. Two instructions per multi-hit node; and the `(bits & ~7) |
  lane` as one `vpternlogd` is LLVM's to form once both constants sit
  in registers, which its allocation here declines.
- *The any hit's found flag* (item 6a). The quantifier's early exit
  is the monotone accumulator's `alive` argument into the loop head,
  tested there once per node and once per leaf tile. The clean fix is
  an SSA simplification: an edge whose argument is a constant that
  decides the target's first dispatch is threaded past it. General,
  small, and not written yet.
- *A lane extract at a run-time index from a vector loaded from
  storage the function never writes* (item 6b) is a scalar load at
  the lane's address, Embree's `movq (%rax,%rdi,8)`. The one-hit arm
  removed the common case; the push block of two or more hits still
  spills and reloads. The clean rule needs the backend to know the
  storage is never written here -- the layout's groups are read-only
  unless `mut`, a fact the SSA's pointer types do not carry yet.
- *The waiting-lane mask from the count* (item 5): the scalar form is
  as many ops as the vector compare; dropped.
- *The relooper*: with arms that leave the count switch differently,
  CodeGen_Stmt now duplicates the sort's join into each network arm
  (the `ssa/*` goldens grew by about 80 lines); the LLVM path is
  direct and unaffected. A switch whose arms `break` to a shared tail
  is the structure to recognise.
- *Two zero idioms per one-hit node* remain on the SSA-level
  compresses (the children's and the key's), whose rest nothing
  reads; a "don't care" on `ir::Intrinsic::compress` would drop them.

What remains per node against Embree's one-hit case (`vpcompressd;
vpermt2q; vmovq` and its prefetch): our two compresses and two
extracts for the child and its key, the redundant cull, the base add,
the key's +inf select the program writes (`distmin`'s), and the
prefetch's own compress of the addresses. The matching schedule's
leaf skip and the sort-free run's shape are the rest of the any hit's
1.28x ops.

## The rulings on the non-generic list, and the order of work (night of 2026-10-02)

Read against the list above, in the order they are to be done.

1. **The SSA compress gets a "front only" flag.** `ir::Intrinsic::compress`
   carries, as `Store::slack` does, that nothing reads past the packed
   lanes; SortRecursion's one-hit arm and QueueRecursion's `!one` block
   set it, CodeGen_LLVM lowers it with `compress_lanes(...,
   zero_rest=false)` (and CodeGen_X86::dynamic_shuffle must see it), and
   the two zero idioms per one-hit node go. Tests at the ssa, llvm and
   execution levels.
2. **The found flag: thread the constant edges in SSA.** The repository
   has a jump threading pass, but it is `Opt/JumpThreading`, a statement
   pass that replicates code between two tests of one condition
   (Mueller and Whalley), run before the SSA loop exists; QueueRecursion
   wrote the `!live` header argument expecting LLVM's jump threading to
   turn the constant edges into direct branches, and LLVM's refuses to
   thread through a loop header (`ThreadAcrossLoopHeaders` is off: it
   would make the loop irreducible), which `!visit` is. The fix is an
   SSA rule: a jump into an instruction-free block whose terminator
   dispatches on one of its arguments, passing a constant there, goes
   straight to the chosen successor; every edge left passes true, the
   argument folds, the dispatch goes. No copy, no new loop entry. The
   any hit's found flag is the same mechanism.
3. **Loopify never tests the prune at a direct descent.** The ruling: the
   entry test is the argmin's or argmax's pruning condition, which the
   children's mask has already tested against a bounding volume, so the
   descent is always safe without it -- and it is correctness-neutral
   besides, since pruning only avoids subtrees that cannot improve the
   best. Build it as Embree's two nested loops, reducible: `!visit`
   (dispatch on live, then the prune test) and `!body` (the node test
   onward); the one-hit arm and the sorted run's first jump to `!body`,
   pops to `!visit`. The descent then carries no distance at all, which
   removes the key's compress and extract from the one-hit arm.
4. **Read-only in the type system, then the lane as a scalar load.** All
   data but reducers and `mut` arrays is read-only; the SSA `Ptr_t`
   does not say so yet (mutability lives per parameter in
   `ir::Function_t::ArgSig::is_mutable`, Lower/Mutability makes `mut`
   things pointers, CodeGen_LLVM sets NoAlias/ReadOnly on some
   parameters). First verify the compiler rejects aliasing -- two `mut`
   parameters bound to one storage, a `mut` parameter over an extern the
   callee reads -- and add the rejection if it does not; then the
   immutability bit on the storage type, `!invariant.load` and
   `readonly` in LLVM, and two rules: `ExtractIdx(Load(p), i)` is
   `Load(p + i)` on immutable storage, and `ExtractIdx(compress(v, m),
   0)` is `ExtractIdx(v, cttz(m))` where `m != 0` is known. Together they
   are Embree's `child(bsf(mask))`, one scalar load off the row. Values
   in registers keep compress and lane 0 (the permute by index measured
   worse, above).
5. **The sign flip: a constant-interval analysis.** A linear dataflow
   analysis over the SSA after Halide's `ConstantInterval` and
   `constant_integer_bounds` (fetch them; Halide is not on disk),
   extended to floats; weak through diverging control flow (union at
   block arguments, top for loop-carried ones), and the division pass's
   own `upper_bound` becomes a client of it. The sort drops the flip when
   the key's lower bound is at least zero. The program clamps `tnear` and
   `tfar` to zero at ray setup as Embree's TravRay does; since the key is
   formed in the box-test callee from a field of the ray, the callee's
   summary takes the call site's argument intervals (field-sensitive,
   memoized) or the clamp sits in the box test.
6. **The C++ backend is not a concern and is never benchmarked.** Checked:
   `-b cpp` compiles through CodeGen_LLVM to the `.o` and writes only a
   header; a function whose SSA was kept is generated straight from it
   (`-v` says "generated from SSA" per function); the relooper makes only
   what `-p ssa` prints. compare.sh's `rtq.o` is the LLVM path already.

## The arms finish by themselves, and the machine is a thing the rewrite can ask (2026-10-03)

Item 1 of the list above was begun as written -- a "front only" flag on
the compress -- and the user asked two questions over it that changed the
work. First, whether the flag should be a lane count rather than a bit,
so that the arms for two and three hits could be made better too. The
answer: the flag decided one thing in the instruction, whether
`vpcompress` merges into a zeroed register or into whatever it has, so it
is binary by nature, and a count would change no instruction; what made
two to four hits cost more than Embree's was the shape of the arms, which
all converged on a join where the count was a run-time value again -- the
sort network, a reverse of the whole vector, a permute of every value
that travels, extracts of all eight lanes under `k < hits`, a waiting mask
formed from the count, two register compresses and two whole-vector
stores -- where Embree's arms each finish by themselves with the count a
constant. Second, whether there should be an IR node for the sorted run
lowered per backend, since the best code differs on x86, ARM and PTX. The
answer: the node exists (a `MultiCall` with keys and conditions, which
SortRecursion lowers and loopify reads back as a `SortedRun` or a
`LaneRun`), the best code does differ by machine (AVX-512 and SVE have a
compress and a variable permute; AVX2 and NEON have neither, and Embree's
own path there is its bit-scan loop with scalar loads and a scalar stack
sort; a GPU thread has no cross-lane vector and sorts by insertion), and
the choice belongs in the SSA rewrite keyed on the target, as Halide's
lowering passes consult their Target while its backends only select
instructions -- so that the traversal reads in `-p ssa` and is pinned by a
golden per machine, and the stack and the loop are written once. The
decision was: a real target description first, then the per-arm pushes as
the strategy a compress machine takes, the AVX2/NEON strategy when such a
machine is measured. Three things were built.

**1. The compress names what fills its rest, or nothing** (ir::Intrinsic,
Expr.cpp, Parser.cpp, CodeGen_LLVM.cpp, runtime/bonsai_cpp.h). No flag:
`compress(v, mask)` leaves the lanes past the packed ones unspecified,
which is Embree's `compact` (`_mm256_mask_compress_epi32(v, mask, v)`,
merging into itself), and `compress(v, mask, fill)` fills them, LLVM's
passthru as a third operand (a vector of v's type or one element of it;
printed and parsed as written). The traversal's every compress is the
two-argument form; the permute tests ask for zero where they check for
it. Read off the machine code afterwards: LLVM had already lowered the
zero passthru as the zero-masking form, `vpcompressq %zmm22,%zmm22{%k1}{z}`,
and the `vpxor` the earlier reading took for the passthru's zeroing is
LLVM's own dependency-breaking idiom before a masked write, so the
operand changed no instruction in either kernel. A semantic cleanup, not
a speedup; the speedup is item 3.

**2. `ir::Target` is a target** (include/IR/Target.h, src/IR/Target.cpp,
src/CodeGen/ResolveTarget.cpp): the triple, the CPU, the feature string
LLVM is given and the resolved set of features by LLVM's names, with the
questions a rewrite asks -- `has_compress()` (avx512f with avx512vl; sve),
`has_variable_permute()`, `vector_bits()` -- resolved once in the CLI from
`--triple` and `--mcpu`, or from the host when neither is named, through
LLVM's `MCSubtargetInfo` so that the CPU's implied features are in the
set, and kept in `CompilerOptions::target`; CodeGen_LLVM::make_target_machine
now builds its target machine from the same three strings, so the two
never disagree. `sort_recursion(func, target)` reads it through
`lane_sort_strategy`, one strategy today (`Compact`) for every machine,
with the AVX2/NEON strategy named in its comment for when such a machine
is measured. What went: the enum `Target { Host }` that keyed
`Program::schedules`, a map with one entry -- `Program::schedule` is one
`Schedule` now, the printer writes `schedule {`, and 142 goldens under
ssa, lower, opt and parsing moved by that line alone (checked: nothing
else in them changed); `CompilerOptions::target`, the output kind,
became `backend`. Tested by the backends goldens, which name a CPU and
now resolve it the same way the backend does.

**3. The arms of the count switch finish by themselves**
(SSA/SortRecursion.cpp, sort_lanes). For two, three and four hits the
packed keys are read out as scalars and sorted by insertion, a minimum
and a maximum per key already placed -- Embree's dist_A0/B0, A1/B1/C1,
A2..D2 chains -- each sorted key's low bits name the lane the child and
its carried bound are read from, and the arm ends in a run of exactly h
unconditional calls, nearest first, which loopify's ordinary path writes
as h - 1 scalar pushes at constant offsets from the top and a descent
into the first: Embree's `stackPtr[0].ptr = ..; stackPtr[0].dist = ..;
stackPtr++; cur = ..`. No join, no permute, no waiting mask, no compacting
store below five hits; the one-hit arm is h = 1 of the same rule. The arm
for five hits and up, which only a node wider than four has, keeps the
bitonic network over all the lanes, the permutes and the compacting store
(the `SortedRun` shape, now produced by that arm alone), as Embree's own
fallback sorts descending and writes the stack in a loop. A four-wide
node's stacks lose their slack (`u32[64]` where they were `u32[68]`);
an eight-wide node keeps it for the last arm (`u32[72]`). The two-hit arm
on Zen 5, from the keys to the descent:

    vpcompressd %ymm9,%ymm9{%k1}{z}        -- the packed keys
    vpextrd $0x1,%xmm9,%ecx ; vmovd %xmm9,%eax
    cmp %ecx,%eax ; mov %ecx,%ebx ; cmovl %eax,%ebx ; cmovg %eax,%ecx
    and $0x7,%ebx ; and $0x7,%ecx
    mov 0x140(%rsp,%rbx,8),%rdx            -- the near child, off the spilled row
    vmovaps %ymm13,0xa0(%rsp)              -- the distances spilled
    vmovd 0xa0(%rsp,%rbx,4),%xmm13 ; vmovd 0xa0(%rsp,%rcx,4),%xmm9
    mov 0x140(%rsp,%rcx,8),%rax            -- the far child
    movslq %r13d,%rcx ; inc %r13d
    mov %rax,0xae8(%rsp,%rcx,8) ; vmovd %xmm9,0x1f8(%rsp,%rcx,4)
    vucomiss %xmm13,%xmm5 ; jbe <pop>      -- the entry cull (item 3 of the list)

Twenty-two instructions where Embree's two-hit case is about fourteen
(it indexes its permutes by the key's own low bits, so it needs no `and`,
and its keys stay in vector registers as broadcasts, so the sort is a
`vpminsd` and a `vpmaxsd` where ours is a compare and two `cmov`); the
scalar loads off the spilled row and the spilled distances are what item
4 of the list (the lane as a scalar load from read-only storage) turns
into loads off the node itself. The kernels: trace_all 570 instructions
static (501 before: the arms are unrolled per count where the join was
shared), occluded_all unchanged at 327 and byte for byte the same code.

**What it measured**, back to back on the quiet machine (cpu 11, `--side
2048`, least of five, million rays per second), two scratch worktrees --
HEAD and HEAD plus this -- so neither side carries the other session's
in-flight compiler change; every ray agrees with Embree on every run,
which is also the execution test of the eight-wide arms including the
last one:

| mesh    | rays    | nearest hit: Embree | before | after | before | after | any hit (matching, tuned) before | after |
|---------|---------|------:|------:|------:|------:|------:|------:|------:|
| head    | primary | 64.3 | 51.7 | 56.7 | 0.80x | 0.90x | 0.99x, 0.99x | 1.00x, 0.99x |
|         | ao      | 15.8 | 12.9 | 13.7 | 0.82x | 0.86x | 0.90x, 0.98x | 0.98x, 0.98x |
|         | diffuse | 14.0 | 11.5 | 12.2 | 0.82x | 0.86x | 0.89x, 0.99x | 1.00x, 0.98x |
| ganesha | primary | 23.4 | 19.5 | 21.1 | 0.84x | 0.92x | 0.90x, 0.92x | 0.92x, 0.91x |
|         | ao      |  6.4 |  5.6 |  5.8 | 0.87x | 0.92x | 0.91x, 0.96x | 0.92x, 0.96x |
|         | diffuse |  5.9 |  5.1 |  5.4 | 0.86x | 0.92x | 0.91x, 0.95x | 0.92x, 0.90x |
| dragon  | primary | 33.4 | 29.5 | 31.3 | 0.88x | 0.95x | 0.92x, 0.92x | 0.94x, 0.92x |
|         | ao      |  4.6 |  4.1 |  4.2 | 0.90x | 0.93x | 0.92x, 0.96x | 0.91x, 0.98x |
|         | diffuse |  4.1 |  3.7 |  3.8 | 0.90x | 0.93x | 0.92x, 0.96x | 0.93x, 0.96x |

(head's incoherent rays are the second of two back-to-back pairs; the
first pair's "after" run was 5% low on both sides at once, the machine's
state.) So the nearest hit gained 8-10% on primary rays and 2-6% on
incoherent ones -- the three- and four-hit arms are where the join cost
most, and primary rays meet them near the root -- and the any hit, whose
lane run was not touched, is level within the run-to-run noise. The
matching schedule now stands at 0.90-0.95x of Embree on primary rays and
0.86-0.93x on incoherent ones for the nearest hit, 0.92-1.00x for the any
hit; tuned 0.90-0.98x.

**Tests.** ssa/child-volumes-sorted (the arms on a four-wide node, its
description rewritten), ssa/child-volumes-sorted-wide (new: an eight-wide
node, whose golden has the last arm, its stack slack and the compacting
store), the vectorized, tiled, prefetch and arena goldens re-blessed and
read; backends/llvm's child-volumes-vectorized, tiled-leaf-vectorized and
prefetch-children, blessed from the clean worktree; the fill operand at
the parsing (permute-compress), LLVM (permute, permute-avx512) and
execution (correctness/cpp/permute, which checks the fill past the packed
lanes and only the packed lanes without one) levels; the correctness
traversals all pass; the full suite in the clean worktree passes.

**Next**, the list's order as ruled: thread the found flag's constant
edges into the header's dispatch in SSA (item 2); loopify's two nested
loops so a direct descent skips the entry cull (item 3: the `vucomiss;
jbe` above, and the near child's distance with it); read-only in the
pointer types and the lane as a scalar load off the row (item 4: the
`0x140(%rsp,..)` loads above become `(%node,..)`); the interval analysis
for the sign flip (item 5). And, when a machine without a compress is
measured, the bit-scan strategy behind `lane_sort_strategy`.

## The header's test threaded away, and what it was worth (2026-10-03)

Item 2 of the rulings: the loop a traversal becomes has a header
`!visit(node, live, count): dispatch live [exit, body]` into which every
edge but the one that ran out of stack passes `true`, and the plan's
reading was that the header tested `live` once per node and once per leaf
tile. Built as the general SSA rule the ruling asked for
(SSA/Simplify.cpp, thread_argument_dispatches): an edge into a block that
has no instructions and dispatches on one of its own arguments, passing a
constant there, goes straight to the target the constant picks, and the
targets take over the arguments their subtrees read -- Mueller and
Whalley's threading without the replication, since the block replicated
holds nothing but the branch; the one thing LLVM's own jump threading
refuses to do, because the block is a loop header, and the one case where
that is harmless, because the header's targets are entered from it alone,
so the target becomes the header and the loop keeps one entry. Edges that
pass a run-time value keep the block, which then dispatches for them
alone (the arguments move under fresh names); where every edge left
agrees on a constant, the block's dispatch folds to that jump. In both
kernels every edge was a constant -- the one-hit arm and the pushes come
straight from the node test, with nothing between that could have settled
the accumulator -- so `!visit` is gone: `_recloop_func0!loop(top, count)`
is the header, with the entry, the one-hit arm, the push and the pop as
its predecessors, and the stack running out goes to `!visited` and its
return. In the statement form (`-p ssa`) the loop reads `do { .. continue
.. return } while (true)` where it read `while ((*!live))`.

The relooper needed a correction for that shape, and not the `break` the
header's comment in QueueRecursion had assumed the statement form lacked:
a traversal's loop is always left by its return, so no `break` arises.
What broke was pass 2 of its classification: it found one latch for a
header and asked which arm of the header's two-way dispatch reaches it;
with several latches -- each push and the pop all jump back now -- each
arm reached a different one, and it took the body's first branch for a
`while` whose exit was the pop path. It now asks whether an arm reaches
any latch of the loop, and a header whose both arms do is the first
branch of a `while (true)` (pass 3).

**What it measured**: nothing, to within the noise. Back to back from two
scratch worktrees on the quiet machine (cpu 11, `--side 2048`, least of
five, every ray agreeing with Embree): head 57.9 -> 58.7 / 13.8 -> 13.9 /
12.3 -> 12.3 million rays per second on the nearest hit's primary/ao/
diffuse rays, ganesha 22.2 -> 22.1 / 6.10 -> 6.07 / 5.67 -> 5.67, dragon
32.5 -> 32.5 / 4.46 -> 4.46 / 3.95 -> 3.95; the any hit the same to the
second decimal. The machine code says why: the old listing has no test of
`live` anywhere -- LLVM's SimplifyCFG folds a conditional branch on a phi
of constants (FoldCondBranchOnPHI) even at a loop header, which its
JumpThreading pass declines -- so the x86 kernels already ran without it,
and the plan's accounting of a test per node and per leaf tile was wrong.
What the rule changes is the IR that is read and the statement form that
is printed, and the loop structure LLVM is handed: it now sees the pushes'
back edges and the pop's as two loops, one inside the other, which is
Embree's shape and what item 3 builds on. Kept for that; the plan's
remaining list is re-counted against the listings below rather than
against this reading.

**Tests.** A three-level test of the rule on its own, apart from
traversals: ssa/thread-dispatch (a merged `b` and `y`, one edge passing `b
= true` and threaded past the second `if`, the other passing a run-time
`d` and keeping the test, the arms taking `y` under a fresh name),
backends/llvm/thread_dispatch (`c` tested once, `d` once, `b` never) and
correctness/cpp/thread_dispatch (all four combinations against the
function as written). The loopify, quantifier, sort and traversal goldens
under ssa, the LLVM traversal goldens and the PTX rtiow-primer re-blessed
and read; ssa/loopify-queue's description rewritten; the full suite 1272
of 1272.

**Next**, as before: item 3, the entry cull skipped on a direct descent
-- the lowering marks the gate it writes at a node's entry (the argmin's
`improves_on` and the quantifier's `still_undecided`, Lower/Trees.cpp)
with a provenance, and loopify sends the edges that know the gate passes
(every descent: the parent's test of the child's box decided it) to the
block inside it, so the pop alone runs the test, as Embree's pop cull
does; then items 4 and 5.

## The entry gate skipped on a direct descent, and `break` in the statement form (2026-10-03)

Item 3 of the rulings. Built in three parts.

**1. The gate is marked where it is written.** A new provenance kind,
`ir::Provenance::EntryGate` (include/IR/Provenance.h): the `if` a tree
query puts at a node's entry -- the extremum's test of the carried bound
against the best (Lower/Trees.cpp, carry_bound_around: Embree's `if
(stackPtr->dist > ray.tfar) continue`) and the quantifier's `if
(still_undecided)` -- carries it, Convert puts it on the arm's block as it
does an `if`'s, and Unswitch, which hoists the two arms' identical gates
out of a match, now hands the hoisted `if` the provenance the two agreed
on (it rebuilt the `if` bare before; that is where the mark was being
lost). The statement printer writes `// gate` on such an `if` and the
block dump on such a block, since a reader of a traversal looks for it.

**2. loopify sends a direct descent past it** (SSA/QueueRecursion.cpp).
With the body's entry block being the gate and nothing else -- pure
instructions and a two-way dispatch whose one arm's block is marked, its
jump arguments made of the header's arguments and the parameters -- a
descent known to be live (every one-hit, push and sorted arm; the entry
and the pop are not descents) jumps to the arm inside the gate with the
child's values in place of the header's. The gate runs for the pop alone.
In both kernels every descent qualifies: `_recloop_func1!loop(top, bound,
count): load best; lt; dispatch [pop-side, !then_0(..)]` and
`!then_0(top, triangles, r, best, bound, count): // gate` with the sort
arms as its predecessors, the entry and `!visit_next` as the header's.

**3. The statement form can write it.** The traversal is now Embree's two
nested loops: the pop's around the descents', and the inner one is left
from below its header -- by a leaf, by a node with nothing hit -- which
the statement form had no way to say. `ir::Break` (include/IR/Stmt.h) is
new: printed `break`, lowered by CodeGen_LLVM through an `escape_blocks`
stack beside the `latch_blocks` that `continue` uses, `break;` in the C++
backend, the frontend has none (Convert rejects it). The relooper
(SSA/CodeGen_Stmt.cpp) learned the shape: a loop's blocks are those from
which a latch is reached without passing the header; a header's two-way
dispatch is the loop's `while` test only where the loop is left by that
test and by returns, and a loop left from below its header for anything
else is a `while (true)`; where such a loop is left to is the nearest
common post-dominator of its exit edges, over the graph structuring
follows (a parfor's body is a region of its own, so its yield is not an
exit), returning paths aside -- and inside the loop, arriving at that
block is the `break`, whatever lies on the way (a leaf's tests, a found
flag's store) written in front of it. The result for the sorted BVH4
(ssa/child-volumes-sorted):

    do {                                     -- the pop's loop
      if (bound < best) {  // gate            -- Embree's pop cull
        do {                                 -- the descents' loop
          if (kind == 0) {
            <children's test>
            switch (hits) {
            case 0: break;                   -- to the pop
            case 1: top = ..; continue;      -- the descent, no gate
            case 2: push; top = ..; continue;
            ...
            }
          } else { <leaf> }
          break
        } while (true)
      }
      if (count != 0) { pop; continue } else { return }
    } while (true)

Three shapes that broke on the way and what each taught: a loop with
several latches was paired with the first latch found and misread as a
`while` whose exit was the pop path (the arms must be tested against
every latch); the children's parfor body counted as a way out of the inner
loop (the loop analysis must use the flow graph the relooper uses, where a
parfor's successor is its continuation alone); and a loop with a `return`
inside lost its `while` to the `while (true)` form (a path to a return is
not a way out below the header). The classification prints under
`BONSAI_RELOOPER=1`.

**What the kernels lost**: trace_all 568 -> 559 instructions, `vucomiss`
8 -> 4 and `vcompressps` 2 -> 1 -- the cull per descent and the one-hit
arm's key compress and extract -- branches 44 -> 41; occluded_all 331 ->
323, the found flag's load and test per node gone, branches 27 -> 26.
LLVM now sees the two loops as such (`.outer` with the pop's phis, the
inner over the descents).

**What it measured**: level, within noise, on every number (cpu 11,
`--side 2048`, least of five, two worktrees, every ray agreeing): head
58.4 -> 58.3 / 13.9 -> 14.0 / 12.3 -> 12.4 million rays per second
(nearest hit, primary/ao/diffuse), ganesha 22.1 -> 22.0 / 6.05 -> 6.13 /
5.62 -> 5.71, dragon 32.5 -> 32.4 / 4.45 -> 4.47 / 3.95 -> 3.96; the any
hit the same. Items 2 and 3 together took a test and a branch, a load, a
compress and an extract out of every node visit and moved nothing, so the
traversal's time is not in these scalar instructions: the dependent chain
of node fetches and, in the arms, the spill of the node's vectors and the
reload at the key's index (item 4) are the better candidates, and an IBS
re-profile of ops and cycles per node visit should come before item 5.

**The scalar key sort, investigated** (asked: should an integer `min` with
an operand already in a vector be done as broadcast + `vpminsd` +
extract?). The SSA already writes `min`/`max`; the `cmp` and two `cmov`
are x86's lowering of a scalar integer min, which it has no instruction
for, and no simplifier rule is missing (the only `select(a < b, a, b)` in
sort code is the scalar network's, which LLVM already turns into `smin`,
and the compare must stay for the child's select). Measured on a
pinned-CPU listing (scratchpad minmax.bonsai): two keys sorted as scalars
are 11 instructions and as a vector with the second key broadcast 10;
three keys are 23 either way, because LLVM rewrites the three-key vector
chain into a horizontal reduction with an inserted sentinel
(`vpinsrd $3, 0x7fffffff; vpshufd; vpminsd; vpshufd; vpminsd`) and comes
out even. Not a strict improvement; a rule `min(extract(k, a), b)` with
`b` broadcast is worse (a `vmovd` and a `vpbroadcastd` for the broadcast).
If it is ever done it is an x86 instruction-selection rule on `min`/`max`
of lane extracts, never in the SSA (a GPU thread's "vector" min is eight
scalar ones), and the kernel context has to be measured, not the listing.

**Tests.** ssa/child-volumes-sorted (description rewritten for the two
loops and the gate), child-volumes-sorted-wide, child-volumes-any,
quantifier-early-exit (five loops, two breaks), ray-any-early-exit,
blocks/mixed-traversal, the prefetch, tiled and arena goldens, and the
lower goldens that now print `// gate`; backends/llvm's three traversal
goldens; the whole of correctness/cpp (140) passes; full suite 1278 of
1278. The pbrt scalar schedule, which the peer session found broken by an
intermediate state of this (a loop left to two places), compiles.

## Read-only storage in the types, and the lane read off the row (2026-10-03)

Item 4 of the rulings, with the user's direction that externs are always
immutable and that this should reach LLVM. Built in four parts.

**1. The storage says so.** `Array_t::readonly` and `Ptr_t::readonly`
(include/IR/Expr.h, include/IR/Type.h; `Type::is_readonly()`,
`as_readonly()`): a reference or pointer to storage nothing in the
program writes. Printed `const u8[bytes]`, `(const _tree_layout3*)`;
`same_as` and the interning tell the two types apart, `equals()` ignores
the mark as it ignores an array's size and a vector's packedness. An
extern is such storage -- there is no `extern mut`, the host fills it
before the program runs and holds it still while it does -- so the parser
marks every extern's declared type (Parser.cpp, parse_extern) and the
layout lowering marks every group array and the row reinterprets it
writes (Layouts.cpp, layout_to_structs, address_of_row). The mark
travels: Convert's GEP and FieldPtr inherit it from their base,
Vectorize's widening keeps it, and `PtrTo::make` gives a read-only
pointer to a place whose access chain is rooted in read-only storage.
Not a parameter's: a function that does not write through a parameter
says nothing about another parameter aliasing it, so the aliasing
question the rulings raised never arises -- the mark is on the storage,
not on who holds it.

**2. LLVM is told.** `CodeGen_LLVM::mark_invariant` puts
`!invariant.load` on a dereference, an array element read and a dense
vector load whose pointer or array is read-only; the gathers are left
(the metadata is a load's). MarkDeviceMemory strips it from a texture
handle's load (NVPTX's ReplaceImageHandles asserts on the `ld.global.nc`
the tag makes of it; the pass already kept its own tag off those).

**3. The lane read off the row.** SSA/Simplify.cpp, two ExtractIdx
rules. (A) A lane of a vector of scalars loaded from read-only storage --
through `load`, `load_field` of a stored struct, an element of a
read-only array, and the cast that reads a layout's packed vector as the
ordinary one -- at a lane only known when it runs, is a scalar load at
the lane's address: `reinterpret_cast<const u64[8]>((&row.children))[i]`.
A lane known at compile time is left alone (one instruction off the
register the vector's other uses hold anyway), and so is a vector of
vectors (stored one component at a time). (B) Lane 0 of a compress with
no fill that nothing else reads is the lane the mask's lowest set bit
names, `v[ctz(reinterpret<u8>(m))]`, with `ctz` a new intrinsic
(llvm.cttz; `tzcnt`); with no bit on both are unspecified, so no
condition on the mask. The single-use test matters: without it the sorted
arms' packed keys, read at lanes 0, 1 and 2, took the count path for lane
0, the keys went to general registers, LLVM did the insertion sort as
`cmp`/`cmov` and the kernel grew by 68 instructions. A four-lane mask has
no integer to count and keeps the compress (child-volumes-sorted);
eight lanes count (child-volumes-sorted-wide, the rtq kernels).

**4. What the kernel does now.** The two-hit arm is Embree's sequence:
`vpcompressd` the keys, `vpextrd`/`vmovd` the two, `cmp`/`cmovl`/`cmovg`,
`and $7` each, `mov (%node,%idx,8)` for the near and the far child off
the row, the far bound off a spilled `tnear` (`vmovaps %ymm,(%rsp)`;
`vmovd (%rsp,%idx,4)` -- the one spill left, off the chain), the push,
the jump. The one-hit arm: `tzcnt %edx,%eax; mov (%rsi,%rax,8),%rsi`,
Embree's `bsf` and `node->child(r)`. trace_all 502 -> 555 instructions
(the sort in general registers costs movs and cmovs; `vpminsd`/`vpmaxsd`
24 -> 14, `cmov` 0 -> 20, `vmovd` 1 -> 14) and occluded_all 329 -> 326.
More instructions, and faster: the `vmovdqa64 %zmm,(%rsp)` of the
children and the `mov (%rsp,%rbx,8)` that waited on it were on the chain
to the next node's address, and they are gone.

**Measured** (cpu 11, `--side 2048`, least of 5, two worktrees back to
back, every ray agreeing on every run). Nearest hit, matching schedule,
million rays per second and the ratio to Embree:

| mesh | primary | ao | diffuse |
|---|---|---|---|
| head | 58.2 -> 61.8 (0.91 -> 0.96x) | 13.9 -> 14.4 (0.89 -> 0.91x) | 12.4 -> 12.8 (0.89 -> 0.91x) |
| ganesha | 21.9 -> 23.2 (0.90 -> 0.96x) | 6.12 -> 6.36 (0.92 -> 0.96x) | 5.68 -> 5.90 (0.92 -> 0.95x) |
| dragon | 32.3 -> 33.4 (0.94 -> 0.97x) | 4.48 -> 4.60 (0.95 -> 0.98x) | 3.97 -> 4.07 (0.95 -> 0.98x) |

Any hit, matching schedule: head 86.6 -> 90.4 (1.01 -> 1.05x) / 16.0 ->
17.1 (0.90 -> 0.96x) / 14.6 -> 15.6 (0.90 -> 0.95x); ganesha 26.6 -> 28.5
(0.92 -> 0.97x) / 6.58 -> 6.99 (0.91 -> 0.97x) / 5.65 -> 5.95 (0.92 ->
0.96x); dragon 35.7 -> 37.5 (0.93 -> 0.99x) / 4.75 -> 4.95 (0.93 ->
0.97x) / 4.19 -> 4.37 (0.93 -> 0.97x). The tuned schedule's any hit is at
or above Embree on every incoherent set now: head 1.06x / 1.06x, ganesha
1.02x / 1.00x, dragon 1.03x / 1.02x; its nearest hit on dragon's primary
rays 1.00x. The first item since the arms to move the time, and by the
same 3-6% on every mesh and ray set -- what items 2 and 3 pointed at:
the chain to the next node's address, not the scalar instruction count.

**Left out, on purpose.** The constant-lane case of rule A (measure it
if a kernel ever shows an `extract` of a stored vector on its chain); a
lane of a stored vector of vectors (a scalar load per component, built
back into the short vector); the parameter case of the mark (a `const`
view a callee may trust needs the aliasing contract the rulings asked
about, which no exported boundary states yet); CSE of the address
arithmetic the rules make (LLVM merges it).

**Tests.** ssa/stored-lane (the four shapes: a run-time lane, a single-use
compress's lane 0, a compress read twice, a constant lane),
backends/llvm/stored-lane (`!invariant.load`, `llvm.cttz`),
correctness/cpp/stored_lane (run; a main passing a `uint32_t8` must be
compiled with `-march=native` or the vector ABI differs and the mask
arrives in halves); child-volumes-sorted and -wide read the new arms;
every lower/ssa/llvm golden that prints an extern's type gains `const`;
the suite is 1288 of 1288. Also in this stretch: the inliner's name
collision the pbrt session hit (ssa/inline-names,
correctness/llvm/inline-names; commit 301a11e1).

**Next**, as before: an IBS re-profile of ops and cycles per node visit
against Embree, now that the chain has changed; then item 5, the
constant-interval analysis for the key's sign flip.

## The kernels counted again after item 4 (2026-10-03)

The profile of the night of 2026-10-02 redone on the committed compiler
(a3c16034): ganesha's ao rays, `--side 2048`, cpu 11 quiet, the driver
built with the matching schedule and kept (scratchpad build_rtq.sh), one
`perf record` per kernel and event -- AMD IBS at one sample per 100003
dispatched-and-retired ops (`ibs_op/cnt_ctl=1/`, no skid, so a block's
sample count is its execution count times its op count) and `cycles:u`
at one per 200003 -- and `perf annotate -n` read per block (scratchpad
blocks.py; `annot-*`, `*-ibs-*.lst`, `ibs-tables.txt`). Embree's intersect
symbol in the any-hit run holds the untimed primary pass alone, and that
is taken off its counts in the nearest-hit run, instruction by
instruction.

**The same nodes.** The node test's instructions run once per node
visit and are one op each, so their mean sample count is the visits:
nearest hit 712.2 ours against 710.9 Embree, any hit 675.7 against
678.9. Pops 265 against 267, leaves 127 against 138. The two traversals
visit the same nodes, as the agreement check says they must, and every
number below is per node visit on that count.

**Ops and cycles per node visit**, ours against Embree's:

| | nearest hit, ops | nearest hit, cycles | any hit, ops | any hit, cycles |
|---|---:|---:|---:|---:|
| ray setup and epilogue | 8.0 vs 7.3 | 3.4 vs 3.9 | 7.7 vs 7.6 | 4.5 vs 4.3 |
| pop and the kind dispatch | 4.2 vs 2.9 | 5.2 vs 4.5 | 2.1 vs 1.4 | 5.0 vs 3.5 |
| node test | 25.5 vs 23.1 | 26.1 vs 25.9 | 25.4 vs 22.0 | 26.2 vs 29.5 |
| after the test, and the one-hit descent | 15.4 vs 14.8 | 16.7 vs 17.4 | 15.2 vs 11.1 | 17.4 vs 15.2 |
| the sorted arms (nearest) / the pushes (any) | 9.9 vs 5.7 | 13.4 vs 7.3 | 6.6 vs 2.9 | 9.6 vs 5.1 |
| leaf | 11.9 vs 13.8 | 19.9 vs 22.2 | 11.7 vs 11.9 | 19.3 vs 20.9 |
| **total** | **75.2 vs 67.5** | **84.9 vs 81.2** | **68.8 vs 57.0** | **82.0 vs 78.4** |

Nearest hit: 1.11x the ops (1.35x before the arms, 1.64x before that),
1.046x the cycles, 0.89 against 0.83 instructions per cycle. Any hit:
1.21x the ops, 1.046x the cycles. The cycles are the measured rates
within the sampling's own cost.

**Where the cycles are.** In the nearest hit the node test is level
(26.1 against 25.9: our two extra ops there -- the separate `tNear <
best` compare where Embree folds the best into `tFar`'s min as
`ray.tfar`, the `addq` that makes an address of an offset reference,
and one `vmovaps 0xc0(%rsp)` reload of a broadcast the allocator
spilled -- cost no time), the leaf is in our favour by 2.3, the pop and
the after-test path are within one, and the whole gap and more, 6.1
cycles of the 3.7, sits in the sorted arms: 13.4 against 7.3. Those
arms are now the sequence Embree runs, so what is left is what each
executes before it: our keys are five ops -- distmin's `select(hit,
tNear, inf)`, then the sign flip's `vandps`, `vpmovd2m` and masked
`vxorps`, then `vorps` with the lane -- where Embree's `distance_i` is
one `vpternlogd`, and the second child's prefetch waits on the compress
(`vpextrq $1`, 6.1 cycles in its block against 1.8 ops). The chain from
the mask to the near child's address runs blend, and, mask move, xor,
or, compress, extract, cmp/cmov, and, load; Embree's runs ternlog,
compress, permute, move. Item 5 takes the three flip ops out of it, and
the `vpmovd2m` is the costly one (a vector-to-mask move on the chain);
a rule `compress(select(c, x, y), m)` -> `compress(x, m)` where `m`
implies `c` (the live mask is `hit && nearer`) takes the blend out, the
compress's rest being unspecified anyway. In the any hit the node test
is in our favour by 3.3 cycles, and the after-test path and the pushes
cost 6.7 more: the one compacting store of the waiting children
(`lzcnt`, `xorb $7`, `vpbroadcastd`, `vpcmpneqd`, `vpcompressq`, a
64-byte store: 9.6 cycles) against Embree's loop of scalar pushes by
`tzcnt` and `blsr` (5.1), and `vpandq`/`vpaddq` making addresses of the
eight references before the prefetch where Embree's children are
pointers already.

**Candidates this leaves, in order.** Item 5 as ruled (the flip), with
the dead blend's rule beside it. The any hit's pushes as scalar pushes
by the count, the way the nearest hit's arms became (a `tzcnt`/`blsr`
run, or arms by count without a sort). The best folded into the box
test's `tFar` in place of `r.tfar` (the query's carried bound is the
ray's far plane; one compare for two). References as pointers would
take `addq`, `vpandq` and `vpaddq` out of every visit, but a reference
is an index by design (it relocates; see the layout notes), so that
stays as it is.

## The key's sign flip gone: what every value lies between (2026-10-03)

Item 5 of the rulings. Built in three parts.

**1. The analysis.** SSA/ConstantIntervals.h: `ConstantInterval`, Halide's
(src/ConstantInterval.h: two optional bounds, union, intersection,
arithmetic, min, max, abs, cast_to), with the bounds doubles so that floats
are in -- an integer bound is exact to 2^53 and dropped past it, an
integer interval stays within its type's range, an infinity is a bound
like any other, and a NaN is outside the lattice (the bounds say where a
value is when it is a number; the sort's lanes that are read are the
hits', whose box test is false for a NaN). `ConstantIntervals`, Halide's
constant_integer_bounds (src/ConstantBounds.cpp) over this SSA form: one
pass per function in reverse postorder, a constant itself, an instruction
its operation on its operands (add, sub, mul, div, mod, min, max, abs,
select, casts, bc, ramp, extract, reduce, shuffle, load_field,
make_struct, bitwise and/or/xor, shifts, popcount, the comparisons, the
intrinsics abs/min/max/fma/sqrt/sqr/exp/sin/cos/clz/ctz/permute/compress),
a block argument the union of what every edge hands it -- an edge from
inside the loop the block heads making it everything, except a value
handed back as itself, which adds nothing. What one pass cannot see it
reads from memory and across calls: a local allocation's contents are the
union of everything stored into it, field by field and an array's
elements as one (a store through a pointer that resolves, through GEPs,
FieldPtrs and the block arguments the Definitions helper follows, to an
Alloca; everything once the address goes anywhere this does not follow),
read by the loads of the next pass, three passes at most; a parameter is
the union of the arguments every live call site hands it, field by field
(a struct parameter's fields each their own: the ray's `tnear`), a call's
result the union of the callee's returns, and what a callee stores
through a pointer parameter lands in the caller's local, six rounds at
most over the program with every summary starting as everything and
narrowing, so that stopping anywhere is sound. A recursion handing its own
parameter back is left out of that parameter's union, which is what lets
the traversal's ray keep what the query said of it; a function nothing
exported reaches (the traversal the lowering extracted and then inlined
into its callers, kept in the map) feeds no call site, or its unnarrowed
parameters would widen every callee's. `make_struct<T>()` with no operands
is all zeros, the backends' null value -- an option's empty variant, whose
`set` is then [0, 1] after the union and whose payload is the other arm's.
`BONSAI_INTERVALS=1` prints what was found, per function.

**2. The clients.** The sort (SSA/SortRecursion.cpp, sort_lanes) asks for
the keys' interval -- the keys array's contents -- and forms the key as
the bits alone when the lower bound is at least zero: a non-negative
float's bits order as the float does, and Embree's `distance_i` is
`asInt(tNear)` for the same reason. The division pass's `upper_bound`
(SSA/InvariantDivision.cpp), which searched by shape for what a dividend
is below -- constants, selects, sums, products, masks, shifts, casts,
remainders, and the `select(x < c, x, x - c)` a remainder becomes -- is a
client now, each pass analysing its function once and an instruction the
rewrite itself made read off its operands when asked about (the Select
rule for the remainder chain moved into the analysis). Two goldens moved,
both for the better: `i % 4` on a `parfor i in 0:8`'s index, which the
old search could not bound and the analysis reads as [0, 7] from the
loop's bounds, is now the one comparison `select(i < 4, i, i - 4)` where
it was the eight instructions of a signed remainder by a constant
(ssa/defer-reducer, ssa/defer-stage-cycle).

**3. The program.** rtq.bonsai clamps the ray at the queries' entry, as
Embree's TravRay does (`tnear = max(tnear, 0)`, `tfar = max(tfar, 0)`):
`clamped(r)`, called by `trace` and `occluded`. The box test's `tNear =
max(.., r.tnear)` is then at least zero, and the fact travels from the
clamp in `trace` (where the traversal is inlined; the extracted
`_traverse_tree1` is dead) into `_recloop_func1`'s `r`, round its own
recursion, into the children loop's `r`, through the inlined box test's
option slot (`store intersectsp_ray_aabb$r21 make_struct<_option1>(..)`,
`load`, two `load_field`s), `distmin`'s own slot (`store distmin$r0 @151`
or `inf`), the keys array (`store gep(_keys0, children) _t141`), to the
sort's `extract_idx(_keys0, k)`: `*_keys0 : [0, inf]`. The
nearest-hit traversal's keys are formed as `reinterpret<i32x8>`, `& -8`,
`| lane`, `compress`; the `shr 31`, `& 0x7fffffff`, `xor` are gone.

**The kernel.** The two-hit arm's key formation went from `vblendmps`
(distmin's inf), `vandps`, `vpmovd2m`, masked `vxorps`, `vorps`,
`vpcompressd` -- six ops with a vector-to-mask move on the chain -- to
`vblendmps`, `vpternlogd $0xf8` (`(bits & -8) | lane` in one, which LLVM
forms once nothing sits between the and and the or), `vpcompressd`:
Embree's own three. trace_all 555 -> 550 instructions. The blend is the
one op left over Embree's `distance_i`, and it is not the sort's: it is
`distmin`'s `inf` for a missed box, promoted out of the slot into a
select whose misses the compress then drops; removing it needs a
demanded-lanes pass (LLVM's SimplifyDemandedVectorElts does not know
vector.compress's unselected lanes are dead), which is noted and not
done.

**Measured** (cpu 11, `--side 2048`, least of 5, two worktrees back to
back, baseline 591787e4, every ray agreeing). Nearest hit, matching
schedule, million rays per second and the ratio to Embree
(primary / ao / diffuse): head 60.0 -> 62.1 (0.96 -> 0.97x) / 14.3 ->
14.6 (0.91 -> 0.93x) / 12.7 -> 13.0 (0.91 -> 0.93x); ganesha 22.6 -> 23.4
(0.94 -> 0.97x) / 6.25 -> 6.48 (0.95 -> 0.98x) / 5.79 -> 5.99 (0.94 ->
0.97x); dragon 34.0 -> 34.3 (1.00 -> 1.00x) / 4.59 -> 4.67 (0.98 -> 0.99x)
/ 4.06 -> 4.13 (0.97 -> 0.99x). The any hit, which sorts nothing, is level
within noise on every number. Three ops off the sorted arms' chain for 2
to 4% on the meshes whose nodes are hit twice or more the most, 1 to 2% on
dragon; the nearest hit stands at 0.97-1.00x of Embree on primary rays
and 0.93-0.99x on incoherent ones.

**Tests.** ssa/sort-key-nonnegative (a program with its own ray carrying
`tnear` and `tfar`, Embree's slab test, and two sorted queries, `trace`
clamped and `trace_raw` not: the first's traversal has no flip, the
second's has it), backends/llvm/sort-key-nonnegative (the same through
LLVM: `xor <4 x i32>` in one traversal only), correctness/cpp/
sort_key_nonnegative (both queries on the four-wide walls scene; on a ray
with `tnear = -5` from between two walls the raw query reports the wall
two units behind the origin as the nearest, Embree's `tnear < t`, where
the clamped one reports the wall ahead -- the negative keys sort right
with the flip, the non-negative ones right without).

## The any hit's pushes as scalar pushes, by count (2026-10-03)

The first of the candidates the profile after item 4 left ("The kernels
counted again"): the any hit's path for two hits and more cost 9.6 cycles
a node visit against Embree's push loop's 5.1. The compacting store writes
every waiting lane at once, but to do so it forms the highest lane
(`lzcnt`), broadcasts it, compares it against the lane ramp, ands the
mask, compresses the children (`vpcompressq`), stores a whole vector and
takes the picked child out at a run-time index -- a fixed cost paid
whatever the count, and the count is two at most visits that are not one.
Embree's traverseAnyHit (bvh_traverser1.h) takes the hits one at a time
off its movemask: `bscf` -- `bsf` and `btc`, tzcnt and blsr -- gives the
lowest lane, which is pushed unless it is the last, and the last is
continued with; for two hits that is tzcnt, a load, a store, blsr, tzcnt,
a load. Scalar because at a small count a few scalar ops are the whole
job; past four the compacting store wins, writing them all at once.

**What was built** (SSA/QueueRecursion.cpp, the LaneRun branch). The
switch on the count, which had arms for none (the pop), one (compress and
extract lane 0, which Simplify's rule read as tzcnt and a load) and
everything else (the compacting store), has arms for one to four hits:
the mask read as an integer, the lanes counts of its zeros -- `ctz(bits)`
the lowest, `ctz(bits & (bits - 1))` the next, `31 - clz(bits)` the
highest -- the highest descended into and the others stored one at a time
at `count`, `count + 1`, `count + 2`, the count advanced by the arm's
constant; the default arm, five hits and more, keeps the compacting store.
The lanes are counted at the index's width (u32) rather than the mask's
(u8): llvm.cttz and llvm.ctlz on an i8 with a zero defined take a fixup
first (`or 0x100` before the tzcnt, `shl 24` before the lzcnt) that LLVM
drops only where it knows the word is not zero, which it does for the
first lane (the mask was tested for any) and not for the cleared ones,
where on an i32 tzcnt and lzcnt mean exactly what the intrinsics do, and
every lane is one instruction. A mask whose width is no integer's (the
four-wide trees) reads its lanes as reductions, the least or the greatest
lane index the mask keeps, the lowest cleared by a compare against it;
when every lane is hit the lanes are the constants and nothing is counted.
Each child is `extract_idx(children, lane)` at a run-time lane, which
item 4's rule reads as one scalar load off the node's row. The order is
Embree's: pushes lowest lane first, so that the next pop takes the second
highest.

**The kernel.** The two-hit arm is `tzcnt eax,ebp; lzcnt ebp,ebp; mov
ecx,r12d; inc r12d; mov rax,[r13+rax*8]; xor ebp,0x1f; mov
[rsp+rcx*8+0x78],rax` and the descent's `mov r13,[r13+rax*8]` that the
arms share: two counts, two scalar loads, one store, the count's
increment, no vector touched. The three-hit arm adds `blsr`, a `tzcnt`, a
load and a store; the four-hit arm another three. `vpcompressq`,
`vmovdqu64`, `vpbroadcastd` and `vpcmpneqd` are on the default arm alone.
LLVM lowers the switch as a tree: `test r14b,cl` (`bits & (bits - 1)`,
the one-hit test with no popcount) and then `popcnt` and compares. The
prefetches of the hit children, Embree's per-lane sequence, are
unchanged in front of the switch.

**Measured** (cpu 11, `--side 2048`, least of 5, two worktrees back to
back, baseline 6c3b62cc, every ray agreeing; a first pair of runs
overlapped a multi-core render of the other session's and was thrown
away, these are from a quiet machine). The any hit, bonsai's million
rays per second and the ratio to Embree, matching schedule (primary / ao
/ diffuse): head 89.3 -> 93.1 (1.03 -> 1.08x) / 17.2 -> 17.4 (0.97x) /
15.6 -> 15.8 (0.96 -> 0.97x); ganesha 28.6 -> 29.2 (0.98 -> 0.99x) /
7.01 -> 7.06 (0.97x) / 5.99 -> 6.02 (0.97x); dragon 37.7 -> 38.4 (0.98
-> 1.01x) / 5.03 -> 5.04 (0.98 -> 0.99x) / 4.41 -> 4.42 (0.97 -> 0.98x).
Tuned schedule: head 88.0 -> 91.0 (1.02 -> 1.06x) / 18.4 -> 19.1 (1.05
-> 1.07x) / 16.9 -> 17.4 (1.05 -> 1.06x); ganesha 28.6 -> 29.2 (1.00 ->
0.99x) / 7.41 -> 7.49 (1.03 -> 1.04x) / 6.27 -> 6.28 (1.02x); dragon
37.3 -> 37.0 (0.98 -> 0.99x) / 5.26 -> 5.09 (1.04 -> 1.03x) / 4.58 ->
4.46 (1.02 -> 1.05x) -- dragon's second tuned run was slower on every
kernel, Embree's own included (its nearest hit 34.2 -> 33.9), so that
drift is the machine's and the ratios are the reading. The nearest hit's
kernel is byte for byte the one before (trace_all's 550 instructions;
the disassemblies differ in the function's address alone) and its
numbers are level (head 62.4 -> 62.4 / 14.64 -> 14.65 / 13.08 -> 13.04).
Sampled on head (perf, cycles, instructions and branch misses at one in
20011, `occluded_all` against Embree's occluded kernel): on primary rays
20319 -> 19414 cycle samples (-4.5%; Embree 19450, so level),
instructions 20538 -> 19514 (Embree 19292, 1.2% more), branch misses 108
-> 111 (Embree 105-110); on ao rays cycles 18662 -> 17591 (-5.7%; Embree
16878, so 1.04x), instructions 18703 -> 17640 (Embree 16985-17071, 3.3%
more), branch misses 644 -> 648 against Embree's 603. So the switch's
tree costs no mispredictions over the old three-way one, and the 7% more
mispredictions than Embree on incoherent rays were there before: that,
with the 3% more instructions, is where the any hit's remaining gap on
incoherent rays sits. A modest return on the 9.6 against 5.1 cycles the
profile put on this path: the gain lands on the rays whose node visits
have two hits the most, the primary ones, where the incoherent rays'
visits are mostly one hit, whose arm was already tzcnt and a load.

**Tests.** ssa/any-hit-arms (an eight-wide tree with the children's boxes
in the node, `any` with the test vectorized: the switch's arms with their
ctz/clz counts, the scalar loads and the default arm's store),
backends/llvm/any-hit-arms (the same through LLVM: `llvm.cttz.i32` and
`llvm.ctlz.i32` on the i8 zero-extended, the invariant scalar loads, the
stack's stores, `llvm.experimental.vector.compress` in the default arm
only), correctness/cpp/any_hit_arms with its main (one eight-wide node
whose children's boxes are stepped in height so that a ray straight down
crosses one, two, three, four, five or all eight of them, in lane sets
that are not contiguous, with the triangle under the ray in the set's
lowest lane, a middle one or the highest; every arm's pushes and pops
have to deliver every child, a short segment and a ray above every box
come out clear). The four-wide goldens moved: ssa/child-volumes-any and
the sort-key-nonnegative pair's `occluded` have the per-count arms with
`reduce.smin`/`reduce.smax` for the lanes, and the compacting store is
gone from a four-wide any hit altogether, every count being four or
fewer.

## The simplifier's conjunction rules, and the fold that stays out of reach (2026-10-03)

**The question.** The nearest hit's node test compares twice where
Embree compares once: `tNear <= tFar` with `tFar = min(x, y, z, r.tfar)`,
then `tNear < best` as a second masked `vcmpltps` -- Embree shrinks
`tray.tfar` to the best hit's `t` as hits land, so its slab test's min
chain carries the best and one compare prunes. The prune is the same
(Trees.cpp: a child is entered iff its box's entry distance beats the
running best, a pop is dropped iff its carried bound does not; the sets of
nodes visited agree except at an exact tie, which Embree enters and we do
not), so this is one instruction per node of about twenty-five and, at
the leaf, one compare with its kortest and branch, where Embree's `T <=
absDen * tfar` has the best inside tfar. The tree IR (BONSAI_DUMP_AFTER=
lower-trees) is `_mask1[c] = intersects(r, box_c) && (distmin(r, box_c) <
_best0.0)`; after inlining, per lane, `hit && (select(hit, tNear, inf) <
bc(best))`. The way to `tNear <= min(x, y, z, bc(min(r.tfar, best)))` is
one lowering choice and three rule families: the volume's prune made
non-strict (`<=`, Embree's choice; PredicateAnalysis keeps the leaf's `<`
in the may-bound, and a `<=` there is sound), a side of a conjunction
decided inside the other (`hit` true under `hit &&`, so the select is
`tNear`), `(a <= b) && (a <= c)` to `a <= min(b, c)`, and the new min
moved next to `bc(r.tfar)` and the two broadcasts grouped into one scalar
min, hoisted per leaf -- plus a prerequisite: the slab test exists three
times in our SSA (inlined for `intersects`, inlined again for `distmin`'s
carry, and inside the masked `distmin` that is still a call here, which
LLVM inlines and merges after our simplifier has run), and the rules
match values, not shapes.

**What was added** (SSA/Simplify.cpp, Simplify.h), each exact under a
stated condition, each linear in what it walks:

- Decided sides: in `a & b` the value `a` is true wherever `b` matters
  and `b` wherever `a` does (false under `|`); in `select(c, t, f)` the
  value `c` is true inside `t` and false inside `f`. An occurrence of the
  one inside the other, reached through pure value computations that
  read no memory -- lanewise ones, for a mask, so that a lane's fact
  decides that lane alone -- becomes the constant, and what it decided
  folds. Halide's learn_true. Speculation is not in question: the SSA
  form holds both sides as computed values (a short-circuit `&&` with
  work on its right side was made control flow by Convert), and the
  walk's budget bounds it. The rules that call it stand down inside it,
  so a rule under a rule cannot compound.
- `(a <= b) & (a <= c)` is `a <= min(b, c)`, `(b <= a) & (c <= a)` is
  `max(b, c) <= a`, the duals under `|` with min and max swapped, and the
  same with `<`; also through the mask form `select(p, q, false)`. Over
  integers only: min and max are std::min and std::max (CodeGen_LLVM.
  cpp, `b < a ? b : a`), which pass a NaN in the second argument over, so
  `a <= min(b, NaN)` is `a <= b` where `(a <= b) & (a <= NaN)` is false.
  The float rule would be exact given `c` not a NaN, and nothing can say
  it: SSA/ConstantIntervals.h bounds a value where it is a number and has
  no word for NaN-freedom (`max(x, 0)` is bounded below by 0 though
  std::max(NaN, 0) is NaN), and the argmin's best is a division's result
  anyway.
- An operation over broadcasts alone is the broadcast of the scalar
  operation, `min(bc(s), bc(t))` is `bc(min(s, t))`, exact for every
  type since every lane held the same operands. And over integers a
  broadcast joining a chain of mins or maxes that holds one joins it,
  `min(min(x, bc(s)), bc(t))` is `min(x, bc(min(s, t)))`, which is where
  the per-leaf scalar min would come from; not over floats, where
  std::min is neither associative nor commutative under a NaN. With the
  slab chain as rtq.bonsai writes it, `min(min(x, y), min(z, tfar))`, the
  hoisted form `min(min(x, y), min(z, bc(min(best, tfar))))` equals the
  original exactly iff `best` and `tFarZ` are not NaNs: a NaN `tFarZ`
  wipes its whole subtree, `tfar` and the joined `best` with it, where
  the original kept `best` at the top. So even with a fact about `best`
  the fold needs one about the slab distances, which are `inf - inf` for
  a degenerate ray.
- The and/or rules now run on masks (vectors of bools) as well as bools,
  which they did not; a `bc(true) & m` folds to `m`.

**The way that is exact.** Embree's AVX-512 slab test is `maxi`, `mini`
and `asInt(tNear) <= asInt(tFar)`: integer min and max on the bit
patterns, which order as the floats do once `tnear` is clamped to zero
(a negative component loses to `tnear` in the max, and a negative `tFar`
fails the compare against a non-negative `tNear` whichever negative the
min kept), and which are associative and commutative with no NaN case at
all -- a NaN's bits sit above inf's. Written that way in rtq.bonsai, the
integer rules above complete the fold exactly, in Embree's own
instructions. A program-level choice, the user's; the compiler cannot
make it, since it changes what a NaN does.

**Re-evaluated.** Both kernels are byte for byte what the compiler at
HEAD made (trace_all 549 instructions, occluded_all 375, with both
compilers); no benchmark, the code being identical. In rtq's SSA the
decided-sides rule removed two `!mask &&` ands inside the masked
`distmin`'s option merge, which LLVM had already removed. The kernel's
second compare stands, for the reasons above.

**Goldens moved** (17): the vectorized traversals' option merges, `select(
m, select(m && n, a, b), c)` to `select(m, select(n, a, b), c)` with the
`select(m, false, true)` that fed the inner and gone as dead
(any-hit-arms, child-volumes-any, child-volumes-vectorized,
tiled-leaf-vectorized, prefetch-children, arena-rows-vectorized,
vectorize-pure-call, vectorize-arm-local, vectorize-packet-traversal,
vectorize-skip-cursor, vectorize-sorted-traversal, vectorize-union, and
skip-leaf-helper's masked helper). Through LLVM: child-volumes-vectorized
nine instructions fewer (three vector ands, two selects, two xors),
vectorize_packet_traversal one fewer, vectorize_union a shuffle's
operands in the other order, and tiled-leaf-vectorized one select more
-- LLVM's blends around the sort key's `-inf` fill came out differently
under the simpler mask; one instruction, on the generic CPU the goldens
pin. Every execution test is as it was.

**Tests.** ssa/and-of-compares (the decided select under `&&` and `||`,
`i <= n && i <= m` to one compare against `min(n, m)`, `n < i || m < i`
to `min(n, m) < i`, the float pair left alone, and the vectorized form
comparing the lanes against one broadcast scalar min), backends/llvm/
and-of-compares (`llvm.umin.i32` and `llvm.smin.i32` where there were
two compares, the two `fcmp ole` kept, the splat of one scalar umin),
correctness/cpp/and_of_compares with its main (the ties the joined
compare must keep, the arm a decided select must take, a NaN on either
side of the float case, the lanes).

**Found on the way.** `BONSAI_DUMP_AFTER=lower-trees` prints the program
and then aborts in the layout printer (IRHandle::accept on a null Expr,
under a layout's `group[...]`); the dump is complete, the abort is after
it. Not fixed.

## The layout as Embree's: references that are pointers (2026-10-04)

**The ruling.** The user: the layout has to be Embree's, and storing an
offset where Embree stores a pointer was not it -- Scion's branch has
pointer references (`ptr` groups, `Group::Type::Pointer`), and this branch
had dropped them. The "deliberate difference" recorded above (the one
`add` of the arena's base per visit, so that the tree stays relocatable)
is withdrawn.

**What was built.** `ptr group arena[bytes] { byte : u8; }` is a third
kind of group beside direct and indirect (Lexer/Parser/Token, IR/Layout,
Printer, ValidateLayout): its rows are reached by address, a lookup
`arena[a]` is the row at address `a` with nothing of the group's base
added (Lower/Layouts.cpp, field_in_layout), and the group's own storage
is only owned and uploaded. A layout's reference parameter may start at a
field the layout stores, `layout triangles(ref : u64 = root) { root :
u64; ... }` -- the root's address, Embree's `bvh->root`, which the driver
fills -- resolved to that field's read when the walk begins (LowerMatches).
The BVH8 layout is now Embree's NodeRef exactly: `switch ref[3:3]`, bit 3
the kind as `isLeaf()`; `Interior from arena[ref]`, the reference being
the address since an AABB node's kind bits are zero (`getAABBNode`),
nothing masked; `Leaf from arena[ref[4:63] * 16u]` with the count in the
low three bits (`leaf(num)`). An element reference may begin at such an
address rather than at a storage variable: the reference the argmin keeps
is then the leaf's address and the element's place in it
(Lower/ElementReferences.cpp). The prefetch of a hit child is of the
reference as it is, kind bits included, as `BVH::prefetch` does: the
arms' addresses differ only in the low bits one of them clears, which
LowerReferencePrefetches now sees through, and an address handed to a
prefetch as an integer is taken as one by both backends (vectors of
integers rather than of pointers, which the C++ backend cannot name). The
driver writes addresses as it assembles the arena and puts the root's in
`root`. Embree's 64-bit NodeRef is what the children hold on both sides,
so the bytes are now the same bytes with the same values up to the base
address.

**What LLVM then did, and the fix.** With the base add gone the block
shapes changed, and LLVM lowered the switch on the hit count -- arms for
none, one, two, three and four hits and a default, five cases -- as a jump
table in both kernels (its threshold is four): an indirect branch behind a
load, and the one-hit path, three node tests in five on incoherent rays,
went through it, after the prefetch chain had already tested the count.
The any hit lost 6% of its instructions on primary rays to that. Lowering
small dispatches as compare chains in the backend did nothing: LLVM's
SimplifyCFG folds a chain of equalities on one value back into a switch.
So the dispatch is now Embree's traverseAnyHit and traverseClosestHit in
their own order, in the SSA (SSA/QueueRecursion.cpp, SSA/SortRecursion.
cpp): no hit first (`if (unlikely(mask == 0)) goto pop`, a kortest), then
one hit (`r = bscf(mask); if (likely(mask == 0))`: `bits & (bits - 1) ==
0` on the mask as a word), then the counts from two up as the switch on
`hits - 2`, three cases, below the table threshold. No indirect branch in
either kernel; the one-hit path is `tzcnt`, the load, the kind test.

**The kernels.** trace_all 549 -> 531 instructions, occluded_all 375 ->
371. Per node visit: the `add` of the base gone; the kind test `test
r12b, 8; jne` where it was `mov; and 0xf; je`; the eight children's
addresses no longer formed (`vpandq`, `vpaddq` gone: the compress feeds
the prefetch directly); the spilled broadcast's reload gone with the
base's register freed. Still there: the second compare against the best,
and the one-hit test done twice -- once by the prefetch chain the backend
emits per lane (`lea; test; je`) and once by the dispatch (`blsr; jne`),
on the same mask at two widths, which LLVM does not unify -- the next
item.

**Measured** (cpu 11, `--side 2048`, least of 5, the baseline HEAD's
compiler on HEAD's apps/rtq from its worktree, every ray agreeing).
Nearest hit, bonsai's million rays per second and the ratio to Embree
(primary / ao / diffuse): head 62.5 -> 63.7 (0.98 -> 1.00x) / 14.67 ->
14.92 (0.93 -> 0.95x) / 13.04 -> 13.27 (0.93 -> 0.95x); ganesha 23.5 ->
23.8 (0.97 -> 0.99x) / 6.49 -> 6.55 (0.98 -> 0.99x) / 6.02 -> 6.09 (0.97
-> 0.99x); dragon 34.8 -> 35.4 (1.00 -> 1.03x) / 4.69 -> 4.77 (0.99 ->
1.01x) / 4.14 -> 4.21 (0.99 -> 1.01x). Any hit level: head 93.2 -> 93.0
(1.08 -> 1.09x) / 17.40 -> 17.44 (0.97x) / 15.81 -> 15.89 (0.97x);
ganesha 29.3 -> 29.2 (1.00x) / 7.07 -> 7.06 (0.98x) / 6.01 -> 6.02
(0.97x); dragon 38.5 -> 38.6 (1.00 -> 1.01x) / 5.03 -> 5.02 (0.98x) /
4.44 -> 4.39 (0.98 -> 0.97x). The tuned schedule within 1% of before
everywhere. perf on head's any hit: primary rays instructions 15150 ->
15050, cycles level; ao rays instructions 13650 -> 13900 (+1.8%, the
one-hit test twice), cycles 13625 -> 13720, branch misses 648 -> 631.

**Tests.** lower/ptr-arena-rows (the address formed from the reference,
no base), ssa/ptr-arena-rows-vectorized (the gang's reads through the
reference, the prefetch of the reference itself), correctness/cpp/
bvh4_ptr_arena with its main (the driver writes addresses and the root).
The dispatch's new shape moved the traversal goldens (any-hit-arms,
child-volumes-*, tiled-leaf-vectorized, prefetch-children, arena-rows-
vectorized, skip-leaf-helper, sort-key-nonnegative): `if (any) { if (one)
.. else switch (hits - 2) }` where there was `switch (hits)`.

**Next.** The prefetch of each hit child emitted inside the arm that
takes it, where the lanes are known, instead of a chain behind count
tests before the dispatch: Embree's `prefetch(child)` as each child is
taken, and the end of the doubled one-hit test. Then the slab test on
integer bits as Embree's AVX-512 build writes it.

## The slab test on the floats' bits, as Embree's AVX-512 build writes it (2026-10-04)

**The ruling.** The user: where Embree's own algorithm does the
reinterpretation to integers, ours does too -- a faithful translation of
their intersection code for the scalar case, with the compiler responsible
for vectorizing it -- and it is not an instruction-selection problem for
the compiler to solve. Embree's AVX-512 `intersectNode` (node_intersector1.h)
is `maxi(tNearX, tNearY, tNearZ, tnear)`, `mini(tFarX, tFarY, tFarZ,
tfar)`, `asInt(tNear) <= asInt(tFar)`, with `maxi`/`mini` the integer max
and min of the floats' bits (vfloat8_avx.h) paired as `maxi(maxi(a, b),
maxi(c, d))` (emath.h).

**What was built.** rtq.bonsai's `intersectsp_ray_aabb` is that: the slab
distances and the segment's ends `reinterpret[[i32]]`, the max and min
chains on the integers, the compare on them, and the interval handed back
as floats from the bits. The bits of non-negative floats order as the
floats do; the clamp (`clamped`) makes `tnear` non-negative so the max is
at least it whatever the slab distances are, and a negative `tFar` (a box
behind the ray) fails the compare against a non-negative `tNear` whichever
negative the min kept -- the same reasoning Embree's code rests on. Every
ray agrees with Embree. The node test is now Embree's instructions,
`vpmaxsd` x3, `vpminsd` x3, `vpcmpled`, plus our masked `vcmpltps` against
the best.

**What it broke, and the fix.** The sort key's sign flip came back: the
key is the entry distance's bits, and the distance is now a float read
from an integer max, which the interval analysis could not bound, so the
sort could not see that the key is non-negative (+14 instructions on the
nearest hit, three per arm). SSA/ConstantIntervals.cpp bounds the
reinterpretation now, both ways, for non-negative values: a float in [0,
inf] is bits in [0, 0x7f800000], the bits of +inf the greatest, and back.
A NaN is outside this lattice as it is outside every float interval, so
its bits, above inf's, are not what these bounds speak of -- the reading
the flip rule already took, and the one Embree's test rests on. The fact
then travels: `tnear` in [0, inf) from the clamp, its bits in [0,
0x7f800000], the integer max at least that, the float read from the max
in [0, inf) again, the key non-negative, no flip. The counts are back to
the float version's (trace_all 531, occluded_all 371).

**What stands between this and Embree's one compare.** Integer min and
max are associative and have no NaN case, so the rules of d3162556 could
now fold the best into the chain -- `(a <= b) && (a < c)` is `a <= min(b,
c - 1)` over integers whenever `c` is not the type's minimum, which the
interval analysis knows of the best's bits -- if the prune's compare were
on the bits. It is `distmin(r, box) < best` on floats: the float read from
the max against the float best. The missing pieces, in order: the masked
`distmin` is still a call in our SSA when the simplifier runs (the
vectorizer's clone of a call under a mask; nothing inlines it before LLVM
does), so the slab test the compare refers to is not visible to the rules;
a rule turning a float compare of values whose bits are known non-negative
into the compare of the bits, which is exact within the lattice's reading
of NaN and needs the interval analysis in the simplifier's hands; and the
strict-against-non-strict rule above. The first is the prerequisite noted
in the simplifier's section, and the next compiler item after the
prefetch's.

**Tests.** ssa/sort-key-through-bits, backends/llvm/sort-key-through-bits
and correctness/cpp/sort_key_through_bits with its main: sort-key-
nonnegative's two traversals with the slab test on bits -- the clamped
query's keys without the flip and the raw query's with it, `llvm.smax`/
`llvm.smin` and a signed compare for the node test, and the same answers
run; the raw query's answers on a segment that begins behind the origin
are printed and not judged, a negative `tnear` being outside the integer
test's precondition as it is outside Embree's (`assert(ray.valid())`).

**Measured** (cpu 11, `--side 2048`, least of 5, the baseline d1acf33e's
compiler on its apps/rtq in its worktree, every ray agreeing): level. The
nearest hit on head 63.75 -> 63.86 Mrays/s on primary rays (0.99x
Embree), 14.92 -> 14.91 on ao, 13.27 -> 13.27 on diffuse (0.94x); the any
hit 92.9 -> 93.3 (1.08x), 17.44 -> 17.47 (0.97 -> 0.98x), 15.84 -> 16.00
(0.97 -> 0.98x); ganesha and dragon within 1% on every kernel. perf on
head's any hit: instructions 15399 -> 15168 on primary rays (-1.5%) and
13700 -> 13578 on ao (-0.9%), cycles -2% and -0.5%. The test itself was
never the cost; what it buys is the form in which the best can be folded
in exactly, above.

## The children's prefetch in the arm that takes them (2026-10-04)

**What the backend did.** `.prefetch(children)` became one vector
prefetch of the eight children's addresses under the hits' mask, in the
node's block, and CodeGen_LLVM::emit_prefetch fetched each lane behind a
test of the count: compress the addresses, take lane 0, fetch its four
lines, test for a second hit (`lea; test; je`), take lane 1, fetch, test
for a third (`cmp; je`), and so on -- the count dispatched there and then
again in the arms the run became, on the same mask at two widths LLVM did
not unify (perf: the any hit on ao rays running 1.8% more instructions
after the layout commit, the one-hit test done twice).

**What was built** (SSA/QueueRecursion.cpp). The arms know their
children, so each fetches them itself with no test, as Embree's
traverseAnyHit and traverseClosestHit call `prefetch(child)` as they take
a child: a lane run's arm for h hits fetches its h lanes' addresses
(`addresses[ctz(bits)]`, ...); a sorted run's arm for h hits fetches each
child's address at the lane its key names, through the same `v[idx]` the
child itself was taken out at -- or the row load the simplifier had made
of it already -- and the one-hit arm's through the same compress; the arm
over every lane (five hits and more, or the network) keeps the one vector
prefetch, copied in; and the vector prefetch in the node's block is taken
out once every arm has been seen. The prefetch is found from an arm by
walking up its predecessors, nearest first, through the joins of the
linearizer's gadgets, and only one whose block dominates the arm will do:
a gadget's arm holds a copy of the children's test with a prefetch of its
own, and dominates nothing below the join, so there (the four-wide tests)
the prefetches stay where they were. A vector of pointers -- an offset
layout's `&arena[..]` over the lanes, which the backends keep whole -- is
left alone too; a `ptr group`'s references are integers and come apart by
lane. With the address at a lane, `concat(cast<u64x8>(children))[lane]`,
LLVM spilled the vector to read one lane where the child itself was one
load off the row: the simplifier now folds a shuffle of one vector in its
own order to the vector (the concat of one source a widened read makes),
and the read-only lane rule then makes the prefetch's address the child's
own load. The one-hit path of the any hit is now `kortest; je; kmov;
blsr; jne; tzcnt; mov r15, [r15 + rax*8]; prefetcht0 [r15] x4; jmp`:
Embree's.

**The kernels.** More code and less work: trace_all 531 -> 571
instructions, occluded_all 371 -> 412, the arms each carrying their own
prefetch lines where the chain carried one set behind tests; the children
vector's load only in the arm that compresses it.

**Measured** against abf39b0b (its compiler on its apps/rtq, in its
worktree) on cpu 11, `--side 2048`, least of 5, every ray agreeing, a pair
taken only where Embree's own number held between its two sides -- two
runs were lost to the peer's render being killed on the core as the
window opened and one to a compare.sh run by hand on the same core, which
pins to cpu 11 as this one does and shares apps/rtq/rtq.out with it.
Matching schedule (primary / ao / diffuse): the nearest hit on head 63.37
-> 63.97 Mrays/s (0.99 -> 1.00x) / 14.84 -> 15.14 (0.94 -> 0.96x) / 13.23
-> 13.47 (0.95 -> 0.96x); ganesha 23.79 -> 23.82 (0.99 -> 1.00x) / 6.57 ->
6.56 (1.00x) / 6.05 -> 6.07 (1.00 -> 0.99x); dragon 34.89 -> 34.76 (1.03 ->
1.02x) / 4.72 -> 4.71 (1.02 -> 1.01x) / 4.19 -> 4.15 (1.01 -> 1.00x). The
any hit on head 92.62 -> 95.58 (1.08 -> 1.12x) / 17.41 -> 17.73 (0.98 ->
1.00x) / 15.81 -> 16.03 (0.98 -> 0.99x); ganesha 29.17 -> 29.73 (1.00 ->
1.02x) / 6.98 -> 7.17 (0.97 -> 0.99x) / 5.93 -> 6.10 (0.97 -> 0.98x);
dragon 38.20 -> 38.60 (1.01 -> 1.02x) / 4.98 -> 5.07 (0.99 -> 1.00x) / 4.38
-> 4.46 (0.97 -> 1.00x). Tuned schedule: head's nearest hit 62.67 -> 64.34
(0.99 -> 1.01x) / 14.76 -> 15.09 (0.94 -> 0.96x) / 13.15 -> 13.43 (0.95 ->
0.97x) and its any hit 90.47 -> 94.84 (1.06 -> 1.11x) / 18.89 -> 19.36
(1.07 -> 1.09x) / 17.28 -> 17.71 (1.07 -> 1.09x); dragon's any hit 38.00 ->
38.07 (1.01 -> 1.00x) / 5.20 -> 5.40 (1.03 -> 1.06x) / 4.59 -> 4.68 (1.03
-> 1.05x) and its nearest hit level; ganesha's any hit 28.76 -> 29.61 (1.00
-> 1.02x) / 7.23 -> 7.55 (1.02 -> 1.05x) / 6.22 -> 6.27 (1.01 -> 1.02x),
its before from the first run and its after from the third (each pair's
other half lost as above), and its nearest hit level. So the any hit gains
1.5-3% on every mesh and batch and head's nearest hit 1-2%, and nothing
loses; the any hit now stands at 1.02-1.12x of Embree on primary rays and
0.98-1.00x on incoherent ones, the nearest hit at 1.00-1.02x and
0.96-1.01x.

perf on head's any hit (`occluded_all` against Embree's occluded kernel,
cpu 13, one sample in 20011 events of cycles, instructions and branch
misses; Embree's own counts drift up to 2% between the two runs, which is
the resolution): on primary rays instructions 33349 -> 32000 (-4.0%;
Embree 34025-34124, so 6% fewer than Embree's now), cycles 33317 -> 31981
(-4.0%; Embree 33921-34201), branch misses 117 -> 116 (Embree 116); on ao
rays instructions 29811 -> 29770 and cycles 29806 -> 29703 (Embree
29283-29548 and 28897-29436), branch misses 635 -> 670 against Embree's
605-610 -- level within the drift. That the ao batch's instruction count
does not move where the primary batch's falls 4% is the batch: of its
713933 rays 25474 are blocked, so a dispatch with a hit in it is a small
part of an ao ray's instructions, which go mostly to the ray's setup and
to node tests that hit nothing, where nothing changed. The two kernels'
instruction mixes differ by the forty prefetch lines, two compares and two
moves; no spill was added (16 stores to the frame in each).

**Tests.** ssa/any-hit-prefetch, backends/llvm/any-hit-prefetch and
correctness/cpp/any_hit_prefetch with its main: any-hit-arms with
`.prefetch(tris.Interior.children)`, the prefetches in the arms (one in
the one-hit arm, two in the two-hit arm, ..., the vector one in the
default arm, none in the node's block), through LLVM the `prefetch` calls
on the loaded children, and the same answers run. The two four-wide
goldens with a prefetch (prefetch-children, ptr-arena-rows-vectorized)
move only by the concat fold.

## The best proven non-negative: facts from conditions, and what a pointer points at (2026-10-04)

**Why.** The fold of the running best into the slab test's `tFar` (the
section on the slab test on bits, above) turns the node test's float
compare `float(tNear) < best` into one on the bits, which is exact only
where both sides are known non-negative. `tNear`'s bits are, from the
clamp; the best was not: the interval analysis gave the argmin's running
best as everything. The user's question was whether the compiler could
prove it non-negative from the best starting at infinity and only ever
taking a `distmin`'s value, with `distmin` itself non-negative. It can,
and this is what that took.

**What the analysis could not see** (SSA/ConstantIntervals.cpp), in the
order the facts travel:

- The hit's distance is `T * rcp(absDen)`, and `T` is a dot product with
  its sign flipped by the denominator's (`asInt(..) ^ asInt(sgnDen)`),
  unbounded on its own. It is positive only because it is used under the
  test `absDen * tnear < T`, and the analysis took nothing from
  conditions. Now a condition narrows the values it compares where it is
  known to hold: in the block a branch on it leads to -- a block's facts
  are what every way in agrees on, a predecessor's own facts plus its
  branch's reading of the condition for the edge taken, none for a block a
  back edge reaches -- and in the arm of a select, read with the select's
  condition taken as true in the first arm and false in the second, the
  arm's computation read again under the fact through pure, lanewise (for
  a mask's facts) instructions to a bounded depth, never wider than its
  plain reading. A condition is taken apart through `&&`, `||`, `!`, the
  linearizer's `select(c, false, true)`, and a broadcast, down to the
  comparisons: `x < y` puts `x` at most `y`'s greatest and `y` at least
  `x`'s least, one closer over integers. LLVM's LazyValueInfo and Halide's
  learn_true, over this lattice. Facts are keyed by the value's name,
  which a value keeps when it is carried into the blocks below as an
  argument of each.
- `rcp` had no rule: the reciprocal of an interval on one side of zero is
  between the reciprocals of its ends, each moved outward past the
  estimate-and-Newton-step's error, infinite at zero.
- Zero times an interval with no upper bound came out as no bound at
  all (the corner was marked undefined), and zero times a defined infinity
  as everything, where the lattice's reading elsewhere (the square root,
  the division) leaves the not-a-number case out: `[0, inf] * [0, inf]`
  is `[0, inf]` where it is a number. Both corners read that way now.
- A load through a pointer parameter was always everything. The best
  lives behind one: the traversal's caller stores infinity into a local
  and hands its address over. A function's summary now carries what each
  pointer parameter pointed at on entry, the union over its call sites of
  what the caller's memory held there (a local's contents, or the caller's
  own parameter's pointee and stores), and a load through a parameter
  reads that together with the function's own stores through it. A
  function handing its own parameter to itself adds nothing, as with the
  parameters' values; and in `called`, a callee that is this function
  handed this function's parameter contributes no stores either, since
  taking the summary's word would hand each round the round before's,
  which began at everything, and nothing would narrow.
- The argmin's accumulate (`_best0 argmin= ..`) escaped its pointer, so
  the best's contents were everything whatever was stored. The least or
  greatest of what was there and what arrives is one of the two, so the
  min, max, argmin and argmax accumulates are stores for these bounds;
  and a vectorized store or accumulate carries its mask as a third
  operand, which the store case had refused.
- Within a function, the passes until the memory the loads read is the
  memory the stores made were capped at three; the distance reaches the
  best through two locals (the leaf's two option merges) and then the
  pointer, and its loads need a pass per link. The cap is eight, with the
  fixpoint ending it sooner where there is less to carry.
- A rewrite's new value is read off its operands on demand (`of`), its
  operands first where they are new as well, so a chain the simplifier
  builds is bounded like what the pass saw.

**Found and fixed on the way.** The bits-to-float rule (d1acf33e's
companion in abf39b0b) took any non-negative integer interval as giving a
non-negative float, and so read the sign-flipped `T`, `U` and `V` as
`[0, inf]` from the unsigned xor's `[0, 2^32)`: bits at or past the sign
bit are negative floats. The rule now needs the bits below `0x80000000`.
Nothing had relied on the wrong fact.

**What it says now.** In the traversal, `(*_best0)._field0` is `[0, inf]`
at every load -- the entry's, the node test's, the leaf's -- the stored
tuple's key `[0, inf]`, the pointee on entry `{[0, inf], ..}` from the
caller's infinity and the callee's stores. `--dump-intervals` prints the
analysis to stdout for a test to pin (BONSAI_INTERVALS printed it to
stderr already), with a parameter's pointee as `*p (on entry)`. The
kernels are byte for byte the ones before (trace_all 571, occluded_all
412): nothing reads the new facts yet; the fold is next.

**Tests.** ssa/intervals-from-conditions: a fact from a branch's
condition and the reciprocal's sign (`guarded`), the same inside a
select's arm (`chosen`), and a running best behind a pointer across two
calls (`narrow`, `run`), read off the dump. The suite is otherwise
unmoved (1350).

## The best folded into the slab test's `tFar` (2026-10-04)

**Before.** The nearest hit's node test compared twice: the slab test's
`asInt(tNear) <= asInt(tFar)` (one `vpcmpled`), then the argmin's prune
`distmin(r, box) < best` as a masked `vcmpltps` on the float entry
distance against the best broadcast per node -- with the masked `distmin`
a call in our SSA that recomputed the slab test, which only LLVM merged.
Embree compares once, `tNear <= mini(tFarX, tFarY, tFarZ, asInt(tfar))`,
its `tray.tfar` being the best. **After:**

    limit = min(bits(r.tfar), bits(best) - 1)        -- scalar, per node
    tFar' = min(min(bits(tFarX), bits(tFarY)), min(bits(tFarZ), bc(limit)))
    mask  = tNear <= tFar'

the kernel's node test `vpmaxsd` x3, `vpminsd` x3, `vpcmpled`, `kortest`,
Embree's, and the `vcmpltps` gone. The `- 1` keeps our prune strict where
Embree's is `<=`: on the bits of non-negative numbers `x < b` is `x <= b -
1`, and the best's bits are never the integer's least. The limit is formed
where the best changes, once per accepted hit, as Embree rebroadcasts
`tray.tfar` where a hit lands (item 7 below); formed per node it was five
instructions (`vmovd`, `dec`, `cmp`, `cmovl`, `vpbroadcastd`, the best moved
between register files and back) where the two it replaced had been one
broadcast and one compare, and dragon's nearest hit ran 2-3% more
instructions and as many more cycles.

**What it took**, each a general piece, in the order the rewrite needs
them:

1. **The masked variant inlined** (SSA/Vectorize.cpp, inline_pure_variants).
   A gang's callee that is pure and straight-line -- safe with every lane
   off (no memory, no effect), every block a plain jump or the return --
   is copied in at each call (inline_call) and the copy's chain of blocks
   folded into the calling block through the continuation
   (absorb_successors), each absorbed block's arguments standing for what
   the jump passed, the substitutions carried down the whole chain since a
   gang's blocks name values of the blocks above without taking them as
   arguments. Such a variant was never guarded, so this runs no more than
   the call did; a `[[noinline]]` callee stays a call; a variant nothing
   calls afterwards is removed.
2. **Common subexpressions** (SSA/Simplify.cpp, find_equal). The
   simplifier had no general merge; the inlined slab test stayed a copy.
   Now an operation over the same operands computed again where the first
   is in scope is the first: a table per block, consulted along with its
   dominators' -- but not past a block that ends in a call or a parfor,
   since a deferral (SSA/Defer.h) cuts the graph exactly there and hands
   the continuation only what its entry captured (the first attempt merged
   a `gep(out, p)` of a continuation with the enclosing body's, and the
   drain had no such value; correctness/llvm/defer-nested-producer caught
   it). A read of mutable memory merges only within its block and only
   while nothing has been written between (EarlyCSE's generation); a read
   of read-only storage merges anywhere. The key is the operation, type,
   operands by name and what rides on the instruction (an intrinsic's name,
   a reduction's kind, a shuffle's order); what no key describes -- a
   `sizeof`, whose measured type is on the instruction alone -- is not
   merged (the first attempt merged `sizeof(f32)` with `sizeof(f32x3)`;
   correctness/cpp/vectorize_soa and vectorize_visect caught it). A value
   merged into from another block is registered in its block's lookups,
   which is how the backends find a name across blocks. The simplifier now
   runs to a fixed point (eight passes at most): an instruction visited
   before a merge changed its operand -- `!x` of a `!y` that became `y`'s
   double negation -- is not looked at again within a pass.
3. **The rules the chain needs**: a field of a select between structs is
   the select between the fields where a side then folds (the option's
   `set` read through `select(miss, none, some)`); a field of a struct
   built with no operands is its type's zero; a field read through a
   reinterpretation of a struct build with the same parts (the gang's
   `vector[vec3f, 8]` built from three component vectors and read as the
   struct of three) is the part; `select(c, false, true)` is `!c` and
   `select(c, true, false)` is `c`; `!!x` on masks as on bools; bits read
   as one type and back are what they were; a reinterpretation over a
   broadcast is the broadcast of the scalar one.
4. **The exact facts** (SSA/ConstantIntervals.h). The compare-to-bits
   rule is exact only where the right side is a non-negative number and
   the left side's sign bit is clear in every value: a NaN on the right
   compares true on the bits and false as a float. The bounds alone could
   not say that -- they speak of where a value is when it is a number --
   so a float interval carries three flags: `number` (never a NaN),
   `nonnegative` (at least +0 where a number), `sign_clear` (the sign bit
   clear in every value, a NaN's included). A constant has them from its
   value; `abs` clears the sign of anything; bits known non-negative read
   as a float with its sign clear, a number where none lie past
   infinity's; a comparison known to hold was between numbers, and so
   were the values they were computed from through the operations a NaN
   comes out of (a sum, product, quotient, absolute value, reciprocal,
   root), and a number above a non-negative one is positive; min, max,
   select, memory and reductions carry what both sides have; a product or
   sum of numbers is a number unless it may meet zero times infinity or
   infinity less infinity, non-negative where both are. A float's bits as
   an integer are exact now: between the bounds' bits for a non-negative
   number, up to 0x7fffffff where only the sign is known clear,
   everything otherwise -- which took the earlier reading of a NaN as
   outside the lattice out of the reinterpret rules; the sort's flip rule
   (SSA/SortRecursion.cpp) reads `sign_clear` of the key instead of its
   bounds, which is what a flip is about. And a gang's blocks name the
   values of the blocks above them without taking them as arguments, so
   the analysis resolves a name through the dominators to the block that
   declares it, and an argument named as a parameter or a local allocation
   is that parameter or allocation (the follower of definitions read six
   such references to `_best0` at the loop's gate as six different values).
5. **The compare on the bits, and the join** (SSA/Simplify.cpp): a `<` or
   `<=` of two 32-bit floats whose right side is a non-negative number and
   whose left side's sign is clear is the same comparison of their bits;
   and a strict comparison beside a non-strict one over integers joins it
   through one step on the side not shared where the interval analysis
   says the step cannot wrap, `(a <= b) & (a < c)` being `a <= min(b, c -
   1)`. Then the rules of d3162556 join the compares and move the min
   into the chain beside `bc(tfar)`, and group the two broadcasts into one
   scalar min. The simplify after the directives gets the analysis of the
   program as it then stands (SSA/Convert.cpp), where the best is still a
   load through the pointer its caller passed; after the allocas are
   promoted it is a value carried round the loop that one pass cannot
   bound.
6. **The program says what Embree asserts.** `clamped` was `max(r.tnear,
   0.0)`, Embree's `max(tnear, 0)`; std::max hands a NaN or a negative zero
   through, so the clamp did not make `tnear` a non-negative number and
   nothing downstream was provable exactly (the key's flip came back, and
   the fold needed the entry distance's sign clear). It is `select(r.tnear
   > 0.0, r.tnear, 0.0)` now, and `tfar` the same: identical on every ray
   Embree accepts as valid (`ray.valid()`: `tnear >= 0`, not a NaN), and a
   non-negative number for the compiler to see. The sort-key tests' clamps
   moved the same way.
7. **The limit formed where the best changes** (SSA/MirrorLoads.h). A
   pure value a loop computes from a cell in memory -- a `mut` local or a
   pointer parameter whose address is used only by loads, stores and
   accumulates -- whose other inputs are available everywhere (constants,
   parameters, pure computations of them, cloned where needed) is computed
   where the cell is written instead, into a mirror cell of its own, at the
   entry and after every write, and the loop reads the mirror; promotion
   then makes the mirror a value carried round the loop. From the node
   test's load of the best the chain is the field, the bits, the `- 1`, the
   min with the ray's `tfar` bits and the broadcast; a chain costing fewer
   than two instructions (a field read, a reinterpretation, a negation are
   free) is left alone, since a mirror costs its update at every write --
   the any hit's `!(*_holds0)` was mirrored and its kernel grew by 18
   instructions before that rule. In the kernel the limit is now a ymm
   register (spilled to the frame, `vpminsd ymm5, ymm2, [rsp+0xa0]`, a
   memory operand) and the node path has nothing but Embree's
   instructions; the five scalar ones run once per accepted hit, after the
   leaf's `argmin=`. trace_all 571 -> 578 static instructions (the leaf's
   update and the select clamps), occluded_all 412 -> 413.

**Measured** against 7e0558cd (its compiler on its apps/rtq, in its
worktree) on cpu 11, `--side 2048`, least of 5, every ray agreeing, Embree's
own number holding between the two sides on every pair but ganesha's any hit
on primary rays (Embree 28.4 against 29.4 between the sides, ours 29.85 ->
29.87, so that pair's 1.05 -> 1.02x is Embree's drift and not ours).
Matching schedule (primary / ao / diffuse): the nearest hit on head 64.35 ->
64.68 Mrays/s (1.01 -> 1.02x) / 15.19 -> 15.63 (0.96 -> 0.99x) / 13.50 ->
13.87 (0.96 -> 0.99x); ganesha 23.24 -> 24.11 (0.97 -> 1.00x) / 6.56 -> 6.67
(0.99 -> 1.01x) / 6.09 -> 6.16 (0.98 -> 1.00x); dragon 35.13 -> 35.11
(1.02x) / 4.76 -> 4.80 (1.01 -> 1.02x) / 4.20 -> 4.22 (1.00 -> 1.01x). The
any hit, which the fold does not touch, level: head 96.81 -> 96.30 (1.13 ->
1.12x) / 17.87 -> 17.77 (1.00 -> 0.99x) / 16.14 -> 16.08 (0.99x); ganesha
7.16 -> 7.20 (0.99 -> 1.00x) and 6.10 -> 6.12 (0.98 -> 0.99x) on the
incoherent batches; dragon 39.01 -> 38.96 (1.02 -> 1.01x) / 5.12 -> 5.13
(1.00x) / 4.49 -> 4.52 (0.99 -> 1.00x). The tuned schedule the same way:
its nearest hit on head +2.9% on ao and diffuse (0.96 -> 0.99x), ganesha
+1.5-2.3%, dragon +0.2-1.3%, its any hit within 1% of before everywhere.
perf on the nearest hit (samples, one in 20011 events): head's instructions
48300 -> 48000 on primary rays (-0.6%) and 34499 -> 33600 on ao (-2.6%),
cycles the same way; dragon's 88000 -> 88460 on primary (+0.5%) and 65484
-> 64863 on ao (-0.9%), where without the mirror they were +1.8% and +2.7%;
branch misses level on both. The gain is per node visited -- the broadcast
of the best and the masked compare against it, gone from every node, small
beside the node's loads and slab test -- and shows most on the incoherent
batches; dragon's primary rays are level.

**Tests.** ssa/compare-on-bits, backends/llvm/compare-on-bits and
correctness/cpp/compare_on_bits with its main (`unknown` left on the floats
since its right side may be a NaN; `known` on the bits below `0.0 < b`;
`joined` and `gang` the whole shape, `x <= min(y, z - 1)` scalar and over a
gang with the limit broadcast once; run over NaNs of both signs, negative
zero, the least denormal, the greatest float and the infinities against
C++'s float arithmetic, 792 cases agreeing); ssa/common-subexpressions
(across blocks, read-only reads anywhere, mutable loads within a block
until a store, a program's name kept); ssa/intervals-from-conditions
re-blessed with the flags. The sort-key tests (ssa, llvm and cpp) carry
the fold in their traversals and the select clamp. 96 goldens moved,
read by kind: duplicated field reads, broadcasts and loads merged
(`let _t = b.low` once where it was read three times), `!c` for `select(c,
false, true)`, the folded node test in every nearest-hit traversal, and
LLVM's renumbering; every execution test as before. Suite 1360.

## Ten more meshes (2026-10-04)

The user asked for the comparison on more geometry. Ten meshes from
pbrt-v4-scenes, of rising triangle count and unlike shape -- a stone slab,
a machined part, a car body, a crown, a tangle of ivy, a house, a car's
polygon mesh, a splash of fluid, a patch of landscape vegetation, a dense
orb -- each through compare.sh at `--side 2048`, least of 5, both
schedules, pinned to cpu 11 with the machine quiet, every ray agreeing
with Embree. The scene's PLY is read as it is; the driver frames the mesh's
box, so how much of the image the mesh fills differs by mesh, and with it
how many of the 4.2 M primary rays hit (the ao and diffuse batches are one
ray per primary hit).

| mesh | file | triangles | primary hits |
|---|---|---|---|
| pavilion | barcelona-pavilion/geometry/mesh_00099.ply | 11,366 | 691,328 |
| zero-day | zero-day/geometry/9dbbf607...3f12.ply | 48,960 | 38,186 |
| bmw | bmw-m6/geometry/mesh_00037.ply | 110,592 | 918,517 |
| crown | crown/geometry/mesh_00397.ply | 155,520 | 140,151 |
| ivy | sanmiguel/geometry/enredadera_00001.ply | 179,603 | 104,440 |
| villa | villa/geometry/mesh_00072.ply | 390,784 | 1,379,012 |
| sportscar | sportscar/geometry/OSD_CarPoly_0000_m028.ply | 1,091,232 | 586,077 |
| dambreak | dambreak/geometry/dambreak1.ply | 1,015,024 | 1,312,003 |
| landscape | landscape/geometry/mesh_00370.ply | 1,916,928 | 349,979 |
| lte-orb | lte-orb/geometry/mesh-3.ply.gz | 3,423,232 | 159,416 |

bonsai over Embree, the matching schedule, nearest hit then any hit, each
primary / ao / diffuse (the three meshes used throughout, from the fold's
measurement above, first):

| mesh | nearest hit | any hit |
|---|---|---|
| head (17,674) | 1.02 / 0.99 / 0.99 | 1.12 / 0.99 / 0.99 |
| ganesha (4,323,658) | 1.00 / 1.01 / 1.00 | 1.02 / 1.00 / 0.99 |
| dragon (7,219,045) | 1.02 / 1.02 / 1.01 | 1.01 / 1.00 / 1.00 |
| pavilion | 1.01 / 1.00 / 0.99 | 1.09 / 1.02 / 1.02 |
| zero-day | 1.13 / 1.02 / 1.02 | 1.17 / 1.01 / 1.01 |
| bmw | 1.04 / 1.05 / 1.03 | 1.11 / 1.04 / 1.05 |
| crown | 1.07 / 1.03 / 1.03 | 1.10 / 1.01 / 1.01 |
| ivy | 1.04 / 1.00 / 0.99 | 1.06 / 0.98 / 0.97 |
| villa | 1.01 / 1.02 / 1.02 | 1.08 / 1.02 / 1.02 |
| sportscar | 1.02 / 1.04 / 1.03 | 0.98 / 1.02 / 1.00 |
| dambreak | 0.99 / 1.00 / 1.00 | 1.02 / 0.99 / 1.00 |
| landscape | 1.12 / 1.22 / 1.21 | 1.11 / 1.15 / 1.15 |
| lte-orb | 1.05 / 1.01 / 1.01 | 1.05 / 1.00 / 0.99 |

The tuned schedule's nearest hit is within 1% of the matching one's on
every mesh; its any hit is 1.04-1.18x on primary rays and 1.02-1.15x on the
incoherent batches (head 1.10 / 1.10 / 1.09, ivy 1.05 / 1.10 / 1.08,
sportscar 1.04 / 1.11 / 1.09, lte-orb 1.04 / 1.07 / 1.06).

Read across the thirteen: the nearest hit is 0.99-1.13x everywhere, the
any hit 0.97-1.17x, and the cells under 1.00x are all the any hit on
incoherent rays (ivy 0.97-0.98x, head 0.99x, lte-orb and dambreak 0.99x)
plus sportscar's any hit on primary rays (0.98x, level within the run-to-
run spread -- 1.05x in the profile's run). Two notes on method. A quick
256-wide single run is not a measurement: taken that way, unpinned, before
these, sportscar's nearest hit read 0.75x and landscape's 0.86x, both of
which are 1.02x and 1.12x at 2048 x 5 pinned -- a first run over a 50-165
MB tree pays its page faults. And perf's attribution to Embree's
`intersect` symbol folds in the driver's own use of it to make the
secondary batches, so bonsai/Embree instruction ratios on the nearest hit
are not to be read off a profile; the `occluded` symbol is clean (the
driver never calls it for itself), and the nearest hit's before/after
counts of our own kernel are.

The profile of the weak cell (head and ivy, diffuse rays, any hit): our
kernel runs 1.6-2.6% more instructions and 5-10% more branch misses than
Embree's `occluded` kernel, at 0.98-0.99x; the node loop is Embree's
instruction for instruction (23 per node, the same 6 loads, 6 fma, 3 max, 3
min, compare, kortest, branch), the one-hit path 13 against Embree's 15,
and the leaf's block test the same arithmetic. What differs is after the
hit: Embree returns; ours was draining its stack entry by entry, below.

## The any hit stops when it finds one (2026-10-04)

**What the profile showed.** In the any hit's kernel, after the leaf that
blocks the ray, the code ran `test r11d, r11d; je done; dec r11d; mov r15,
[rsp + 8*r11 + 0x38]; test bpl, 1; mov bpl, 1; je node; jmp` -- once per
stack entry, down to an empty stack, and before that `add r12, 0xb0; cmp
r15, r12; je; test bpl, 1; mov bpl, 1; jne` once per block left in the
leaf: a settled `any` taking its stack down one entry at a time to decide
nothing, where Embree's traverseAnyHit returns. The final SSA said the
same: the pop was `if (count != 0) { pop; continue } else return`, with the
found flag tested only at the loop's head, so a found ray went round the
pop for every entry it had pushed: a load and a branch per entry, for
every ray that finds a hit (17% of head's primary rays, 7.5% of its
diffuse ones).

**Why.** `any` meaning "stop when you find one" is SSA/QueueRecursion.cpp's
find_monotone_accumulator (020952f9): the pop goes on only while
`count != 0 && !holds`, and a direct descent past a leaf that may have
settled the answer asks too. It is written as a check -- the traversal's
boolean is found to move one way by reading the function's stores to it --
rather than a fact carried down from the lowering, on the reasoning that a
fact carried through every pass goes missing silently, while a check can at
worst decline. It declined: since 44a66094 the quantifiers' update is an
accumulate instruction, `acc.max holds x` for `any` (max of bools is or)
and `acc.min` for `all`, and the check read `Store` instructions alone,
found no store to `holds`, and found no accumulator. The test written for
exactly this, ssa/ray-any-early-exit, says in its comment that the pop is
`if (count != 0 && !acc)`; its golden had carried `if (count != 0)` since
87d540a1, blessed without the `&& !acc`, and nobody read it against the
comment.

**The fix.** `writes_accumulator` (QueueRecursion.cpp): a store, an
`acc.max` or an `acc.min` of a bool, two operands or three with a gang's
mask; the scan counts them, the max rising and the min falling whatever
the value, the handle is rebuilt from any of them, and may_settle_before's
walk sees them as the stores they are. The pop is `count != 0 && !holds`
again. LLVM then sees the pop's test on the loop's back edge and drops the
header's own `!holds` test with its phi (the header block is gone from the
LLVM goldens, its two phis moved to the body's preheader), so the kernel's
pop is `test r11d, r11d; je done; test bpl, bpl; jne done; dec; mov; jmp
node`: Embree's `cmp r9, r11; je; mov rax, [r9-8]; add r9, -8` plus one
test of the found flag, which Embree does not need since its flag is
`ray.tfar` set to -inf, read by the next node test. occluded_all 413 ->
406 static instructions; trace_all 579, untouched.

**Measured** against 7af0d6b3 (its compiler on its apps/rtq, in its
worktree) on cpu 11, `--side 2048`, least of 5, every ray agreeing,
Embree's own number holding between the sides on every pair. The any hit
on primary rays: head 95.92 -> 96.81 Mrays/s (1.13x Embree), ivy 82.90 ->
84.29 (1.07 -> 1.08x), dragon 39.05 -> 39.65 (1.02 -> 1.03x), sportscar
60.37 -> 60.77 (1.04 -> 1.06x). On the incoherent batches, within the
run-to-run spread but every pair the right way: ivy ao 6.48 -> 6.54
(0.98x) and diffuse 5.93 -> 5.99 (0.97 -> 0.98x), dragon 5.14 -> 5.18 (1.00
-> 1.01x) and 4.50 -> 4.54 (0.99 -> 1.00x), head level (17.71 -> 17.65,
16.02 -> 16.05). The tuned schedule's any hit the same way, +1-2% on
primary rays and level on the rest. The nearest hit's kernel is the one
before instruction for instruction (trace_all's two disassemblies differ
only in rip-relative offsets): within 1% on head, ivy and dragon, and
1-2% under on sportscar in this run, a 1.1 M triangle mesh whose nearest
hit lives in memory and moves that much between runs. perf of the any hit
on head (samples, one in 20011 events): primary rays' instructions 32300
-> 31900 (-1.2%) and cycles 32267 -> 31888; diffuse rays level (32500 ->
32600, branch misses 837 -> 820), the blocked rays being 7.5% of the batch
with shallow stacks, so the drain was the smaller part of the 2% gap there.
What remains is the leaf's loop, below.

**Left as it is.** Inside the leaf the blocks after the one that found
the hit are still stepped over, six instructions each -- the lowering's
`guard_iteration` is `if (undecided) body` per element, a skip and not an
exit, and the source language has no `break` to write (ir::Break is the
relooper's alone). A leaf of this tree holds one or two blocks of four, so
the step-over is a few instructions per found ray; a loop exit on a
monotone accumulator would be an SSA rewrite with the Monotone analysis,
for when a tree with long leaves makes it worth having.

**Tests.** The ten any-hit goldens: ssa/ray-any-early-exit (now what its
comment says), ssa/any-hit-arms, any-hit-prefetch, child-volumes-any,
sort-key-nonnegative and sort-key-through-bits move by the pop's three
instructions (the pop is printed twice by the statement form, so six lines
in some); backends/llvm/any-hit-arms, any-hit-prefetch, sort-key-nonnegative
and sort-key-through-bits by the header's test folding away, 35 lines each
beyond LLVM's renumbering. Every execution test as before; suite 1366.

## An interval entry answers only while its instruction is alive (2026-10-04)

Found on the way to the leaf's exit, below: ssa/vectorize-sorted-traversal's
golden disagreed between two builds of the same source -- the compare-on-bits
rule fired on `0 < max(root0, root1)` under one binary and not the other,
and the two copies of that compare were merged by one and not the other --
while `--dump-intervals` was identical between them and each binary agreed
with itself run after run, ASLR or none. The analysis keeps what it read of
every instruction under the instruction's address
(SSA/ConstantIntervals.cpp, State::instrs), and the simplifier that asks it
afterwards (the simplify after the directives, cc0095cf; a rewrite's new
value read off its operands on demand and kept, b939082c) frees instructions
and makes new ones through up to eight passes. A new instruction could take
a freed one's address and be answered with the freed one's bounds: a stale
answer, unsound in general, and one that differs from build to build, since
which address the allocator hands back depends on everything allocated
before it. Now the table holds a weak reference to the instruction beside
what was read of it (State::Known; recorded(), record()), and a lookup whose
instruction is gone, or whose address now belongs to another, is a value not
yet read, read off its operands as a new one is. One golden moved,
ssa/vectorize-sorted-traversal, re-blessed to what a clean table gives -- a
broadcast and a reinterpretation merged with their earlier copies, nothing
else -- and the suite was otherwise unmoved, so no committed golden carried a
stale fold; two differently built compilers now agree on it to the line
(1d3c31ab). No test can pin an allocator down to reusing an address; the
check is that the goldens no longer depend on which binary blessed them,
which is how this was found.

## The leaf's loop leaves when the answer is settled (2026-10-04)

**What was left.** After the pop stopped (above), the any hit's leaf loop
still carried the lowering's per-element guard: `for block in leaf { if
(!holds) test(block) }` (Lower/Trees.cpp, guard_iteration), a skip and not
an exit -- the source language has no `break` to write and ir::Break is the
relooper's alone. Per block tested that was `test bpl, 1; mov bpl, 1; jne`
in front of the block's loads, three instructions on every block a ray
tests, hit or no hit, and after a hit the remaining blocks were stepped
over; Embree's leaf loop is `for (i < num) if (occluded(prim[i])) return
true`, with nothing in front of a block and the return at the hit.

**The rewrite** (SSA/QueueRecursion.cpp, exit_settled_loops, run by
loopify before it builds the stack, with the Monotone analysis it already
has): a dispatch on the accumulator's "still undecided" (the form
undecided_in writes, read back: its load, negated for a rising
accumulator, `any` over a gang's lanes) whose skipping arm reaches a loop's
latch through blocks that compute values and nothing else goes to the
loop's exit instead -- the header's dispatch's edge out of the loop, or the
latch's own where the loop's test is at its bottom -- with what that exit
edge passes, each argument a constant or a name threaded to the gate,
provided the back edge hands the header that name back unchanged (an index
the latch steps is not such a value: skipping to the end leaves it at the
count, leaving now leaves it where it is; such a loop is left to skip as
before). And where the answer is settled it leaves there: an `acc.max` or
`acc.min` of a scalar bool in a block of the loop whose jump reaches the
latch through such blocks becomes a dispatch on what it accumulated, `acc.max
holds hit` leaving when `hit` is true, so that the iteration's own test of
its result is the exit, as Embree's is, and the gate at the top is passed
only by iterations that moved nothing. Loop unswitching, on a condition
invariant forward in time only. A leaf loop that is still a parfor over its
elements when loopify runs (a tiled layout's leaf not vectorized, as in
any-hit-prefetch) keeps its skip: a parfor's body cannot leave the loop.

**In the kernel.** With the exit at the write, `holds` is false on every
edge into the loop's header, and LLVM folds the gate's test and the flag's
materialization away: the leaf loop is `add r12, 0xb0; cmp r15, r12; je
pop; <loads>; <test>; kortest; je next` and on a hit `kandw; kmovd; test
al, 0xf; je next; jmp done` -- the block's own result is the exit, the
`setne bpl` and the flag's register are gone, and the pop after the loop is
`test r11d; je done; dec; mov; jmp node`, Embree's. occluded_all 406 -> 397
static instructions (413 before the pop stopped); trace_all 579, untouched.

**Measured** against 1d3c31ab (its compiler on its apps/rtq, in a worktree;
this change's compiler likewise in a worktree of its own, so that neither
side carried the other session's uncommitted work in the shared tree) on
cpu 11, `--side 2048`, least of 5, every ray agreeing, Embree's own number
holding between the sides on every pair but ivy's any hit on primary rays
(Embree 78.85 -> 76.94, ours 84.46 -> 83.20, dropped). The any hit on head's
incoherent rays 17.80 -> 17.94 Mrays/s on ao and 16.19 -> 16.29 on diffuse
(0.99 -> 1.00x), the tuned schedule's 19.55 -> 19.75 and 17.82 -> 17.96
(1.09 -> 1.10x); ivy, dragon and sportscar level within 1% on every
incoherent batch (ivy's 0.98x and 0.97x as before, sportscar's tuned ao
11.25 -> 11.42); primary rays within the run-to-run spread (head 97.48 ->
96.83, 1.14 -> 1.12x against an Embree that itself moved +0.8%; sportscar
61.36 -> 60.76, 1.06x; dragon level) -- a primary ray on head runs 26
instructions on average, most of them missing the root, so the leaf has
little to say there. The nearest hit's kernel is unchanged and its numbers
level within 0.6% on every pair but head's primary, whose base-side run was
low (63.03 -> 64.64). perf on head's diffuse rays: instructions 32600 ->
32504 (-0.3%), branch misses 824 -> 804 (-2.4%), cycles level; three
instructions a block came off a loop whose block costs forty, and the
blocks after a hit were one or two. A smaller gain than the pop's, in the
same direction, and the code is Embree's now where it was not.

**Where the any hit's last 2% is**, by the profile on head's diffuse rays
(one sample per 20011 events, our kernel against Embree's `occluded` in the
same run, split by address into the kernel's regions): instructions 32504
against 32103 (+1.2%), cycles 32500 against 31714 (+2.5%), branch misses
804 against 759 (+6%). By region, instruction samples ours / Embree's: the
ray's setup 1969 / 1750, the node loop with the one-hit path 14860 / 15262,
the arms for two hits and more 4412 / 3153, the leaf 9910 / 10717, the pop
1353 / 1101. The arms' excess is accounting: Embree extracts the first hit
child before it knows how many there are, in what this split calls its node
loop, and ours extract every child in the arm; per two-, three- and
four-hit node the two sides run the same number of instructions (26, 37 and
48 against 25, 36 and 47, counted in the disassembly), and the node loop
and arms together are 19272 / 18415. The pop's quarter more is the `jmp`
back to the node test where Embree's falls through. The leaf's 8% less is
ours. What is left as a speed difference is the mispredictions, 45 more
samples -- about 0.2 more per ray -- spread over the node test, the one-hit
test and the arms' count dispatch at a few each; at some fifteen cycles a
miss that is the cycle gap, and they are the same data-dependent branches
Embree has (its `kortest; je`, `and ecx, edi; jne` and `je` after the second
child carry the same weight in its profile). The kernel is Embree's shape
now, region for region; the next percent on incoherent rays is not in the
code's shape.

**Tests.** The ten any-hit goldens again: ssa/ray-any-early-exit (the leaf
loop printed as `do { if (idx < n) { if (!holds) { ..; holds max= hit; if
(!hit) { idx++; continue } } } .. }`, the break the statement form writes
as its continue's absence), ssa/quantifier-early-exit, any-hit-arms,
child-volumes-any, sort-key-nonnegative and sort-key-through-bits the same
way; backends/llvm/any-hit-arms, sort-key-nonnegative and
sort-key-through-bits by the `_holds` phi leaving the loop and the branch to
the function's exit after the block's test. Every execution test as before.

## The tuned schedule: the nearest hit without the leaf's early-out (2026-10-05)

The user's direction: the layout and the data stay Embree's; the work is
the schedule, measured on all thirteen meshes, general gains only. The
first candidate was the one the tuned schedule had deferred "until the
matching schedule is settled": the nearest hit's `trace.skip(intersectsp_ray_tri)`,
Embree's `early_out` after a block's edge tests, which the tuned any hit
dropped on 2026-10-02 for 5-7%. Measured with HEAD's compiler on cpu 11,
`--side 2048`, least of 5, the two schedules back to back on each mesh,
every ray agreeing; the tuned nearest hit without the skip over with it,
ao / diffuse:

| mesh | ao | diffuse | | mesh | ao | diffuse |
|---|---|---|---|---|---|---|
| head | +9.4% | +10.0% | | villa | +7.3% | +6.7% |
| ganesha | +5.1% | +5.3% | | sportscar | +8.8% | +8.6% |
| dragon | +6.2% | +5.5% | | dambreak | +5.7% | +1.6% |
| pavilion | +2.6% | +3.3% | | landscape | +0.8% | +0.6% |
| zero-day | +12.5% | +11.9% | | lte-orb | +5.8% | +5.9% |
| bmw | +9.4% | +7.9% | | crown | +14.2% | +14.4% |
| ivy | +13.6% | +12.1% | | | | |

Primary rays within 2% either way (pavilion -1.6%, bmw +1.9%, the rest
under 1%): a primary ray's leaf work is a small part of its 26-odd
instructions. Against Embree the tuned nearest hit on incoherent rays goes
from 0.99-1.21x to 1.02-1.23x -- head 0.99 -> 1.09x, ivy 0.99 -> 1.13x,
crown 1.03 -> 1.18x, zero-day 1.02 -> 1.14x, bmw 1.03 -> 1.12x, sportscar
1.04 -> 1.13x, villa 1.02 -> 1.11x, dragon 1.02 -> 1.09x, lte-orb 1.02 ->
1.07x, ganesha 1.01 -> 1.07x, dambreak 1.00 -> 1.06x, pavilion 1.00 ->
1.02x, landscape 1.21 -> 1.23x -- and on primary rays stays where it was,
1.00-1.16x. The any hit's cells are the same schedule as before and moved
within 1%. Why the early-out loses: the branch on "any lane survived the
edge tests" is unpredictable on incoherent rays (a block's four triangles
are rarely all missed or all passed on their edges), and the ten
instructions it skips -- the depth test and the reciprocal -- cost less
than its mispredictions; on primary rays the branch is predictable and the
two cancel. tuned.bonsai carries the measurement; embree.bonsai keeps the
step, being Embree's structure step for step.

## The any hit sorted: a loss everywhere (2026-10-05)

The second schedule experiment: the any hit's children sorted by entry
distance as the nearest hit's are (`occluded.sort(.., distmin(..))`), the
thought being that an occluder is found sooner nearer the origin. Measured
as the first (cpu 11, `--side 2048`, least of 5, thirteen meshes, the two
schedules back to back, every ray agreeing), sorted over unsorted on the
any hit, primary / ao / diffuse: head 0.85 / 0.91 / 0.89, ganesha 0.88 /
0.96 / 1.06, dragon 1.00 / 0.98 / 0.97, pavilion 0.92 / 0.92 / 0.92,
zero-day 1.01 / 0.91 / 0.91, bmw 0.93 / 0.91 / 0.86, crown 0.97 / 0.90 /
0.90, ivy 0.90 / 0.93 / 0.89, villa 0.90 / 0.92 / 0.87, sportscar 0.86 /
0.93 / 0.89, dambreak 0.90 / 0.92 / 0.89, landscape level, lte-orb 0.98 /
0.89 / 0.90; the nearest hit, unchanged, within 1%. A loss of 5-15% on
nearly every cell: the sort's work at every multi-hit node (the keys, the
compare network or the counted arms' compares, the carried distances) is
paid on every node, and the nodes it saves -- those behind an occluder
found earlier -- are few, since a shadow ray that is blocked is blocked
soon in traversal order anyway and one that is not visits everything
regardless. Embree's traverseAnyHit pushes in stored order for this reason
("simpler in sequence traversal order"); the tuned any hit keeps that.

## The any hit ordered by the children's area, both ways (2026-10-05)

The user's question: is an occluder likelier in a small box, packed
tightly, or in a large one, holding the most geometry? Two more scratch
schedules sorted the any hit's children by the area of their boxes -- half
the surface, `dx*dy + dy*dz + dz*dx`, the SAH's measure -- the smallest
first and the largest first (keys `area_key` and its negation, functions of
the box alone, written into rtq.bonsai for the run and taken out again).
Measured as above, each over the unsorted tuned any hit, primary / ao /
diffuse:

| mesh | smallest first | largest first |
|---|---|---|
| head | 0.83 / 0.87 / 0.87 | 0.79 / 0.86 / 0.87 |
| ganesha | 0.91 / 0.89 / 0.90 | 0.88 / 0.89 / 0.91 |
| dragon | 0.86 / 0.91 / 0.92 | 0.88 / 0.92 / 0.92 |
| pavilion | 0.77 / 0.89 / 0.88 | 0.74 / 0.87 / 0.87 |
| zero-day | 0.98 / 0.88 / 0.88 | 0.97 / 0.86 / 0.86 |
| bmw | 0.86 / 0.88 / 0.88 | 0.86 / 0.86 / 0.86 |
| crown | 0.91 / 0.87 / 0.87 | 0.89 / 0.85 / 0.85 |
| ivy | 0.88 / 0.85 / 0.86 | 0.89 / 0.85 / 0.85 |
| villa | 0.81 / 0.87 / 0.86 | 0.79 / 0.86 / 0.85 |
| sportscar | 0.85 / 0.88 / 0.86 | 0.80 / 0.86 / 0.86 |
| dambreak | 0.86 / 0.88 / 0.88 | 0.83 / 0.87 / 0.87 |
| landscape | 1.00 / 0.98 / 0.99 | 1.00 / 0.98 / 0.99 |
| lte-orb | 0.85 / 0.87 / 0.87 | 0.89 / 0.88 / 0.90 |

Both lose 8-24% on every mesh but landscape, whose rays mostly miss and
whose nodes mostly have one hit child, where neither order has anything to
sort. Between the two, the smallest box first is the less bad on primary
rays on ten of thirteen meshes, by 1-5% (head 0.83 against 0.79, pavilion
0.77 against 0.74, sportscar 0.85 against 0.80), and the two are within
1-2% of each other on the incoherent batches with no consistent sign. So:
an occluder is found somewhat sooner in the small boxes than in the large
ones on coherent rays, and either order costs far more than it saves --
the sort's keys, compares and carried values at every multi-hit node,
against a few nodes saved behind an occluder that stored order would have
reached nearly as soon. The nearest hit, the same in all three schedules,
measured within 1% throughout. The any hit's order stays Embree's: the
children as stored.

## Comparisons a fact implies, and the blend that stays (2026-10-05)

The plan's next item was the sort key's dead blend: `distmin` is infinite
where the box is missed, so the key the sort packs carries `select(!(tNear
<= tFar), inf, tNear)` -- `vpcmpgtd k2; vbroadcastss ymm{k2}` per
multi-hit node -- although every lane the compress keeps and every lane the
counted arms read is a hit's, where the select is `tNear`. Proving that
needs two things the simplifier did not have, and now has (SSA/Simplify.cpp,
`assume`):

- **Comparisons implied by a fact.** A fact `x <= min(F)` decides `x <=
  min(R)` true for `R` within `F`, the minimum over more values being no
  greater; `max(L) <= y` likewise from `max(F) <= y`; a strict fact gives
  the non-strict comparison too, a non-strict one not the strict. Over
  integers, whose order is total. A fact is taken apart through `&` where
  true and `|` where false, `!` with the truth flipped, and a comparison
  itself (`!(a <= b)` is `b < a`); a chain's leaves are read through a
  broadcast, `bc(min(a, b))` being `min(bc(a), bc(b))`, so a gang's chain
  and the scalars broadcast meet leaf for leaf. The slab test's `tNear <=
  min(x, y, z, tfar)` follows from the pruned `tNear <= min(x, y, z, tfar,
  best - 1)`. (`facts_of`, `chain_leaves`, `implied`.)
- **A cell with one store, read through.** The vectorizer hands a loop's
  mask, keys and children out through local cells threaded by name
  (`_mask1`, `_keys0`, `_carry0`), and promotion makes values of them only
  later; a read of such a cell is the store's value where the store's block
  dominates the read's, comes before it in one block, and no path from the
  allocation to the read misses it (a cell allocated inside a loop needs
  that on top of dominance). (`Cell`, `cell_named`, `stored_into`.)

With these, a compress read under its mask does lose the blend, and so do
the counted arms' reads of the carried distance -- and the kernel grew,
579 -> 592 instructions: the arms had read the distance with one load from
the cell, `vmovss xmm, [rsp + lane*4]`, and read it now from the register
the rewrite handed them, a lane at a run-time index, which is a permute and
a move; and the fifth arm, the network over all eight lanes, permutes the
carried distances by the sorted order and reads positions `k < hits`, a
positional argument no lane rule makes, so the blend stayed alive for it
through the cell anyway. The compress rule alone was level (586, one load
and one dead store traded for the chain on the register). With both rules
withdrawn and the machinery left for the `&&` and select rules, `x <=
min(n, m) && x <= n` became `x <= min(n, m)` (tested at the three levels,
2256 cases), the nearest hit lost one instruction (the network arm's
`select(mask, packed, inf)` reading the packed keys under the mask) -- and
two goldens gained instructions: a comparison decided true deep in a chain
another value shares makes `assume` rebuild the chain above it as new
instructions while the old chain stays for its other users (defer-gang-
bools: `!mask && n <= 0` decided under a select's condition, the arm
rebuilt as `!mask && odd`, four instructions more; vectorize-sorted-
traversal seven more). The same duplication is latent in the rule's
existing `c && select(c, ..)` form, bounded by its budget; broadening it
without a measured gain anywhere is not a trade to make. Everything is
withdrawn: no change to the simplifier, no test, nothing in the kernels.
What the item would need is the sort lowering reading the carry from the
compress rather than the cell, which is the layout of the arms' stack
writes, a different question, and the blend is two instructions on
multi-hit nodes, about 0.3% of the nearest hit. Closed as not worth its
cost; this record is what the attempt leaves.

## Embree's four-wide tree: the layout matched, the schedule matched (2026-10-05)

The user's next direction: other trees from Embree, the layout matched
first, then a matching schedule, evaluated on every mesh and ray type.
The first is Embree's BVH4 over Triangle4 leaves -- what Embree builds on
a machine without AVX2 ("BVH8 reduces performance on AVX only-machines",
scene.cpp) and the tree its compact and robust variants use.

**The layout** (schedules/trees/bvh4.bonsai): bvh8.bonsai with four
children to a node -- Embree's AABBNode for N = 4, 128 bytes, the four
references then six vectors of four floats in Embree's order -- and
otherwise its twin: the same NodeRef with the kind in the low four bits
(byteNodeAlignment = 4 * N = 16 leaves them free), the same Triangle4
blocks, the same `ptr group` arena. The driver (rtq_hook.cpp) reads the
width off the layout it is compiled against, `kWidth =
sizeof(NodeRow::children) / 8`, and everything that depended on eight
follows it: the builder's branching factor, the loops over a row's lanes,
the row's size and offsets (32 N bytes; the bounds at 8 N, 12 N, .., 28 N)
checked at compile time against Embree's, and which tree Embree's own
device is asked for -- `tri_accel=bvh4.triangle4` for a four-wide layout,
so that both sides of a table traverse a tree of the width being compared
(`--embree-stats` shows `accel = bvh4.triangle4`). The leaf's block count
limit is the same seven (NodeRef::maxLeafBlocks = items_mask - tyLeaf, for
either N), so rtcBuildBVH's settings are the eight-wide ones but the
branching factor; the relocation of the largest nodes runs as before. On
head: 1735 nodes of 128 bytes where the eight-wide tree had 779 of 256;
every ray agrees with Embree's four-wide kernel on every mesh.

**The schedules** (embree4.bonsai, tuned4.bonsai): embree.bonsai and
tuned.bonsai with four lanes where they have eight and the stack sized for
N = 4 (stackSizeSingle = 1 + (N - 1) * maxDepth + 3 = 244). What each
directive stands for in Embree's four-wide code is what it stood for in the
eight-wide: traverseClosestHit sorts two hits by one compare and three or
four by sort3 and sort4 -- our counted arms, and with four lanes no network
arm, the counts covering every lane; intersectNode's four-wide test;
traverseAnyHit's pushes in stored order. Kernels: trace_all 446
instructions, occluded_all 329 (579 and 397 eight wide). One difference the
schedule cannot state: the layout prefetches the widest arm's bytes for a
hit child, 176 (a Triangle4 block, three lines) where Embree's
BVH::prefetch fetches two lines for N = 4, node or leaf alike.

**Measured** on the thirteen meshes (cpu 11, `--side 2048`, least of 5,
both schedules back to back per mesh, every ray agreeing with Embree's
four-wide kernel), bonsai over Embree's BVH4, nearest hit then any hit,
each primary / ao / diffuse:

| mesh | embree4 nearest | embree4 any | tuned4 nearest | tuned4 any |
|---|---|---|---|---|
| head | 1.02 / 0.92 / 0.92 | 1.00 / 0.89 / 0.89 | 1.01 / 1.01 / 1.01 | 0.99 / 0.95 / 0.95 |
| ganesha | 0.96 / 0.97 / 0.96 | 0.93 / 0.91 / 0.92 | 0.98 / 1.00 / 1.00 | 0.92 / 0.94 / 0.95 |
| dragon | 1.01 / 0.99 / 0.99 | 0.96 / 0.95 / 0.95 | 1.00 / 1.02 / 1.01 | 0.94 / 0.97 / 0.97 |
| pavilion | 1.04 / 0.92 / 0.92 | 1.00 / 0.90 / 0.89 | 1.01 / 0.95 / 0.95 | 0.99 / 0.90 / 0.90 |
| zero-day | 1.19 / 0.95 / 0.95 | 1.16 / 0.89 / 0.89 | 1.18 / 1.06 / 1.05 | 1.16 / 0.97 / 0.98 |
| bmw | 0.97 / 0.94 / 0.94 | 0.94 / 0.90 / 0.91 | 0.99 / 1.02 / 1.01 | 0.94 / 0.95 / 0.96 |
| crown | 1.06 / 0.96 / 0.96 | 1.04 / 0.91 / 0.90 | 1.07 / 1.08 / 1.07 | 1.03 / 0.98 / 0.98 |
| ivy | 1.01 / 0.93 / 0.93 | 1.01 / 0.90 / 0.90 | 1.02 / 1.03 / 1.02 | 0.98 / 0.98 / 0.97 |
| villa | 0.96 / 0.94 / 0.94 | 0.94 / 0.90 / 0.91 | 0.97 / 1.00 / 1.01 | 0.92 / 0.94 / 0.95 |
| sportscar | 0.99 / 0.98 / 0.97 | 0.96 / 0.92 / 0.92 | 1.01 / 1.04 / 1.03 | 0.95 / 0.97 / 0.96 |
| dambreak | 0.95 / 0.95 / 0.94 | 0.92 / 0.90 / 0.92 | 0.99 / 1.00 / 1.00 | 0.91 / 0.94 / 0.94 |
| landscape | 1.18 / 1.16 / 1.16 | 1.08 / 1.05 / 1.06 | 1.17 / 1.16 / 1.15 | 1.07 / 1.05 / 1.04 |
| lte-orb | 1.04 / 0.98 / 0.98 | 1.01 / 0.94 / 0.93 | 1.04 / 1.03 / 1.02 | 0.99 / 0.99 / 0.98 |

Not the parity the eight-wide matching schedule has: embree4 was 3-8%
behind Embree's four-wide nearest hit on incoherent rays and 5-11% behind
its any hit, with primary rays level to ahead; tuned4 (no leaf early-out
on either query) recovered the nearest hit to 0.95-1.08x and the any hit
to 0.90-0.99x on those batches. Embree's own four-wide tree runs 5-20%
slower than its eight-wide one (head's primary rays 54.2 against 63.5
Mrays/s), so the absolute rates are lower on both sides.

**The cause: a four-wide mask had no integer.** The profile on head's
diffuse any hit: 12% more instructions than Embree's four-wide kernel at
the same branch misses (ivy +15%, primary rays +9%). The passes read a
mask's lanes as the bits of an integer only for 8, 16, 32 or 64 lanes
(`kmov`, then `tzcnt`, `lzcnt`, `blsr`), and a four-lane mask fell back to
the lane-index forms -- the least lane the mask keeps as a masked select of
the lane ramp and a min reduction, the greatest as a max reduction, a lane
cleared by a compare against it -- so where Embree's arm for three hits is
`bsf; load; blsr` three times over, ours was a masked blend, two shuffles
with minimums, a broadcast and a compare each time, and the one-hit descent
went through a compress and a move (`vpcompressq ymm{k}{z}; vmovq`) where
the eight-wide one is a `tzcnt` and a load. The pass said so itself: "a
four-wide mask has no integer to count". Now it has: four lanes are four
bits, `reinterpret<u4>`, widened to the index's word with a cast, and
counted and cleared there as eight are (SSA/QueueRecursion.cpp's arms,
SSA/SortRecursion.cpp's one-hit test, SSA/Simplify.cpp's lowest_lane rule).
LLVM takes an i4 as it is (`kmovd; and 15`); the C++ backend, which the
comparison compiles with, gets `uint4_t` in runtime/u4.h beside the 24- and
56-bit types -- a byte read as its low four bits, so the upper bits a
vector of four bools leaves unspecified never show. The four-wide any hit
is 329 -> 297 instructions and its arms Embree's shape (`tzcnt; lzcnt;
tzcnt; three loads; prefetches; a store`); the nearest hit's one-hit path
is `tzcnt; mov r14, [r14 + rax*8]`. Fifteen goldens moved, all four-wide
traversals, read by kind: `hits == 1` became `bits & (bits - 1) == 0`, the
min and max reductions `ctz` and `31 - clz`, and the one-hit `compress(..)
[0]` a `ctz` and a scalar load off the row (child-volumes-*, arena-rows-*,
prefetch-children, tiled-leaf-vectorized, skip-leaf-helper, the sort-key
pair, at the ssa and llvm levels); every execution test as before.

**Measured again** with the mask as bits, the same way (head, ganesha
and bmw rerun after the other session's last GPU chunk had drained;
Embree's own rate held within 1-4% of the first run everywhere, the
diffuse batches within 2%):

| mesh | embree4 nearest | embree4 any | tuned4 nearest | tuned4 any |
|---|---|---|---|---|
| head | 1.02 / 0.94 / 0.97 | 1.13 / 1.02 / 1.02 | 1.02 / 1.03 / 1.03 | 1.11 / 1.10 / 1.10 |
| ganesha | 1.00 / 0.99 / 0.98 | 1.05 / 1.03 / 1.02 | 1.00 / 1.02 / 1.02 | 1.04 / 1.07 / 1.05 |
| dragon | 1.03 / 1.00 / 1.00 | 1.07 / 1.03 / 1.04 | 1.03 / 1.04 / 1.03 | 1.04 / 1.06 / 1.05 |
| pavilion | 1.04 / 0.95 / 0.94 | 1.13 / 1.03 / 1.03 | 1.03 / 0.97 / 0.97 | 1.11 / 1.04 / 1.04 |
| zero-day | 1.20 / 0.97 / 0.97 | 1.22 / 1.03 / 1.03 | 1.21 / 1.08 / 1.08 | 1.22 / 1.14 / 1.13 |
| bmw | 1.00 / 0.98 / 0.98 | 1.08 / 1.04 / 1.05 | 1.01 / 1.05 / 1.04 | 1.08 / 1.12 / 1.10 |
| crown | 1.10 / 0.99 / 0.99 | 1.15 / 1.04 / 1.04 | 1.10 / 1.10 / 1.09 | 1.14 / 1.15 / 1.15 |
| ivy | 1.07 / 0.97 / 0.97 | 1.12 / 1.02 / 1.02 | 1.06 / 1.07 / 1.06 | 1.10 / 1.13 / 1.12 |
| villa | 0.99 / 0.97 / 0.98 | 1.08 / 1.04 / 1.05 | 0.99 / 1.03 / 1.03 | 1.07 / 1.10 / 1.10 |
| sportscar | 1.02 / 0.99 / 0.99 | 1.09 / 1.05 / 1.04 | 1.04 / 1.06 / 1.05 | 1.09 / 1.11 / 1.10 |
| dambreak | 0.99 / 0.98 / 0.98 | 1.06 / 1.03 / 1.03 | 1.00 / 1.03 / 1.00 | 1.05 / 1.08 / 1.04 |
| landscape | 1.17 / 1.18 / 1.17 | 1.16 / 1.20 / 1.20 | 1.18 / 1.20 / 1.19 | 1.16 / 1.20 / 1.20 |
| lte-orb | 1.07 / 1.00 / 1.00 | 1.10 / 1.04 / 1.05 | 1.06 / 1.04 / 1.04 | 1.08 / 1.09 / 1.08 |

The matching schedule's any hit went from 5-11% behind to 2-6% ahead on
incoherent rays and 5-22% ahead on primary; its nearest hit from 3-8%
behind to level on incoherent rays, 0.94-1.00x (pavilion the low end, as
on the eight-wide tree), and 0.99-1.20x on primary. The tuned schedule
is ahead of Embree's four-wide kernel on every cell but pavilion's
nearest hit on incoherent rays (0.97) and villa's on primary rays
(0.99), by 3-15% on the incoherent batches. The gain against the first
measurement, schedule for schedule: the any hit +5-17% on incoherent rays
(ganesha's ao +17%, head's +13%, crown +11%) and +2-13% on primary; the
nearest hit +4-15% on incoherent rays (ganesha +15%, head +14%, crown
+11%) and level on primary. The three-line prefetch against Embree's two
stays as it is: with these numbers it is not a cost to chase.

What the four-wide tree says about the eight-wide one: nothing changes
there (the eight-wide mask was bits already), but Embree's four-wide
kernel is the weaker of its two on this machine -- 5-20% slower than its
eight-wide at the same work -- and our four-wide schedules, generated from
the same program and the same directives with a 4 where there was an 8,
are relatively further ahead of it than the eight-wide ones are of theirs.

## The four-wide nearest hit profiled against Embree's: the sorted arms are the gap (2026-10-05)

The matching four-wide schedule's nearest hit is 0.94x Embree's BVH4
kernel on pavilion's incoherent rays, 0.96x on head's and level on
dambreak's. Profiled on the diffuse batch of each (perf-nearest.sh in
the scratchpad: `cycles:u`, `instructions:u`, `branch-misses:u`, one
sample per 20011 events, side 2048, five repeats, cpu 11; Embree's
`intersect` symbol carries the driver's own use of it, tracing the
primary rays once to make the secondary batch, which the same run with
the occluded query has and nothing else, so its kernel is the difference
of the two), bonsai over Embree:

| mesh     | instructions | cycles | branch misses |
|----------|-------------:|-------:|--------------:|
| pavilion |        1.079 |  1.087 |         1.022 |
| head     |        1.056 |  1.064 |         1.007 |
| dambreak |        1.025 |  1.020 |         0.917 |

The cycles follow the instructions; the mispredictions are level. It
is more instructions, so the question is where. Both kernels split into
the same five regions by address (ours from trace_all.s; Embree's from
`perf annotate` of its BVH4 intersect, the node loop at 0x2e4a00, the
one-hit descent at 0x2e4a85, the two-to-four-hit arms from 0x2e4ac4,
the pop at 0x2e4ca0, the leaf from 0x2e4cc0), instruction samples by
region, Embree's with its setup use taken out by scaling (the per-region
subtraction wants the occluded run's annotation of the symbol; recorded
when the machine is free):

| region (head, diffuse)    | bonsai | Embree (scaled) |  ratio |
|---------------------------|-------:|----------------:|-------:|
| per-ray setup             |   2358 |            3272 |   0.72 |
| node test + one-hit path  |  26098 |           23594 |   1.11 |
| sorted arms, 2-4 hits     |   6089 |            4307 |   1.41 |
| leaf (Moeller) + epilog   |  11842 |           11928 |   0.99 |
| pop + cull                |   1604 |            2317 |   0.69 |
| total                     |  47991 |           45446 |   1.056 |

Pavilion the same shape but sharper: node test + arms 15101 against
12558 (+20%), the leaf 7528 against 7028, setup and pop 1471 against
2752; dambreak node test + arms +7%, the leaf level. The boundary
between the node loop and the arms is blurred by sampling skid (the
arms' first instruction, `and al, 0xf`, carries 3369 of head's samples,
the skid of the taken `jne` into the arms), so the two are best read
together: the traversal of interior nodes is 7-20% more instructions,
the leaf is level, and the setup and the pop are ours by a margin.

Read instruction by instruction, the node test and the one-hit descent
are the same length -- 36 instructions a visit to Embree's 38; Embree
spills tNear to its stack every visit (`vmovdqa [rsp+0x150], xmm1`), we
spill only in the arms -- so the extra is in the arms, and there the
two kernels are shaped differently. Embree's traverseClosestHit for N =
4 (bvh_traverser1.h) walks the hit bits: `bscf` (tzcnt and blsr) gives
a lane, the child is loaded at it and prefetched, the lane's tNear is
read from the spilled vector with one indexed load
(`((unsigned int*)&tNear)[r]`), and `if (likely(mask == 0))` after each
lane is the dispatch -- the ZF of the blsr. Two hits are then one
compare (`d0 < d1`) and a push of the farther; three and four are
Embree's sort3/sort4 over (child, dist) pairs held in xmm registers.
Its two-hit arm is 20 instructions beyond the one-hit prefix. Ours
(SSA/SortRecursion.cpp, the Compact strategy, Embree's AVX512VL8 shape
for its eight-wide kernel) packs every lane's key with its lane index
(`ramp`, `vpternlogd`), compresses the hits' keys to the front
(`vpcompressd`), counts them (`popcnt`) and dispatches on the count (a
`cmp 2; je; cmp 3; jne` chain), extracts the packed keys to scalars
(`vpextrd`), sorts them by min/max (cmov), recovers each lane with `and
3`, loads the child at it, then spills the key vector and re-reads the
pushed child's distance by lane (`vmovaps [rsp+0x90], xmm27; vmovss
xmm27, [rsp+4*rcx+0x90]`), and finishes in a tail LLVM merged across
the arms (`mov ebx, 1; ...; add ebp, ebx`). Our two-hit arm is 38
instructions beyond the prefix; our three-hit arm about 54 to Embree's
40. Two further differences of a size each: a hit child is prefetched
in three lines (the widest arm's 176 bytes, Lower/Layouts.cpp) where
Embree's BVH4 fetches two (bvh.h, `prefetch`: `N >= 8 || types >
BVH_FLAG_ALIGNED_NODE` for the third and fourth), one instruction and
one line request more per hit child; and the leaf's hit epilog finds
the lane of the minimum by a blend of the lane ramp and two
shuffle-and-min steps (seven instructions) where Embree's is a compare,
a kmov and a tzcnt (the `lowest_lane` rule of SSA/Simplify.cpp does not
see this shape, `reduce<min>(select(eq, ramp, 4))`).

On the eight-wide tree the same arms are level with Embree because
Embree's eight-wide kernel takes the compress shape too
(traverseClosestHitAVX512VL8); the four-wide kernel is where Embree's
arms are the cheap scalar walk and ours are not.

The plan, in order of expected gain:

1. **The sorted arms as a walk of the bits** (SSA/SortRecursion.cpp):
   where the mask is bits, take the hits' lanes with ctz and blsr as the
   any hit's arms already do (SSA/QueueRecursion.cpp), read each hit's
   key from the key vector at its lane -- the one indexed load off the
   spilled vector that the push already makes today -- dispatch on the
   remaining bits after each lane (Embree's `likely(mask == 0)`), order
   two hits by one compare on the two keys (branchless, as now), three
   and four by compare-exchanges over (child, key) pairs, and push the
   key read as the distance. What goes: the ramp, the ternlog, the
   compress, the extracts, the `and 3` per hit, the popcount and its
   compare chain, the second read of the pushed distance. Estimated
   from the counts above at 18 instructions per two-hit node, about
   half of the arms' instructions, which is the size of the gap. The
   `sortall` arm for five or more hits (n > 4) keeps the packed keys
   and the bitonic network, computed inside that arm alone. Measured on
   both widths: the eight-wide's two- to four-hit arms get the same
   change, and it has to hold its 0.99-1.13x there. The key's inf blend
   (the previous item) stays: the walk reads hit lanes only, but the
   proof that a hit lane's key is its tNear is the implied-comparisons
   machinery already declined.
2. **The third prefetch line**, measured in isolation: a scratch build
   with the prefetch's bytes at 128 (not committed), the three meshes
   both ways. If it pays, the width of a child's prefetch is a knob on
   the directive for the user to approve, not a rule in the lowering.
3. **The leaf's argmin lane as bits**: `lowest_lane` taught the shape
   the hit epilog makes (a compare of the minimum broadcast against the
   distances, then the least lane), ctz of the compare's bits. Seven
   instructions to three per hit block; small, a codegen defect.
4. **The profile firmed up**: precise sampling (`:pp`) and the
   occluded run's annotation subtracted per region, so the arms are
   confirmed as the gap before and after item 1.

## The sort as Embree's chain: one form for every width (2026-10-05)

The user's direction on item 1 above: the parametric form, with a single
item representation determined by the layout and not by the sort
algorithm. Both of Embree's kernels are one procedure, a chain that
peels the hit lanes one at a time -- the lowest lane left (tzcnt), the
call read at it, inserted among the calls peeled before it by a compare
and an exchange of the pair, the lane cleared (blsr) and the lanes left
tested, none left being the arm for exactly that many hits -- and they
differ only in how an item is held: Embree's four-wide kernel as scalars
(a StackItem's ptr and dist, in general registers), its eight-wide one
broadcast in vector registers with a compress for the peel and a permute
per child taken out. The lowering (SSA/SortRecursion.cpp) is now that
chain with scalar items for every lane count: the child reference as the
layout stores it, read off the node's row at the lane; the key's bits,
read off a cell. The lane count decides the mask's word (SSA/MaskLanes.h,
the lane-reading helpers the any hit's arms already used, now shared) and
how many peels come before the general arm (four, Embree's at either
width; a node of four lanes has no general arm), and nothing else. Past
four peels the general arm is as before: keys packed with their lane
index, the bitonic network, permutes, the compacting store (SortedRun).
Gone from the counted arms: the ramp, the ternlog, the compress, the
extracts, the popcount and its compare chain, the `and 3` per hit, the
second read of the pushed distance.

The cell. A source the children's loop fills -- the keys, a bound carried
to the child -- is an array of the function's own that the vectorized
loop holds in a register, and a lane of a register at a run-time index
goes through memory. The chain stores the vector into a cell once, at
its first comparison (so only on nodes that hit more than one child,
where Embree's node test spills tNear on every visit), and reads each
lane with one indexed load, `mov r13d, [rsp+rdx*4+0x30]`, Embree's
`((unsigned int*)&tNear)[r]`. Two sources the loop fills alike -- the key
and the carried bound are both `distmin` of the child's box in rtq.bonsai,
stored from one value at one index in one block (fill_of, same_fill) --
share one cell and one load per lane, so the distance pushed is the key
itself and one compare and one cmov order both. A cell holds its values
as the integers of their width: a 32-bit integer in a general register
is what the compare orders, one cmov exchanges and the push stores, where
a float would go through a vector register and a mask (`setl; kmovd;
vmovss {k}`) for the same exchange. The prefetch of a child now issues
where its lane is peeled, right after the child's load, once, shared by
every arm below it (SSA/QueueRecursion.cpp finds the lanes through the
arms' selects).

Measured on the thirteen meshes, base (90310848, a worktree) and new run
back to back per mesh (bench-ab.sh), cpu 11, side 2048, least of 5,
Embree's own rate holding within 1% on every pair but two (zero-day's
primary rays under the tuned four-wide schedule, +2.5%; landscape under
the tuned eight-wide, -4 to -5%, a disturbance during that pair). Every
ray agrees with Embree on every mesh, width and schedule. The nearest hit's
rate, new over base, and the any hit's (which the sort does not touch):

| mesh      | embree4 near pri/ao/dif | embree near pri/ao/dif | tuned4 near pri/ao/dif | tuned near pri/ao/dif | any hit, all four |
|-----------|------------------------|------------------------|------------------------|-----------------------|-------------------|
| head      | 1.045 1.039 1.040      | 1.078 1.052 1.054      | 1.052 1.040 1.043      | 1.069 1.058 1.056     | 0.99-1.02 |
| ganesha   | 1.058 1.044 1.048      | 1.072 1.046 1.048      | 1.056 1.044 1.048      | 1.053 1.035 1.041     | 0.99-1.02 |
| dragon    | 1.046 1.037 1.042      | 1.056 1.025 1.033      | 1.050 1.033 1.033      | 1.062 1.027 1.033     | 0.99-1.02 |
| pavilion  | 1.074 1.046 1.047      | 1.057 1.069 1.069      | 1.057 1.039 1.040      | 1.059 1.078 1.075     | 1.00-1.01 |
| zero-day  | 0.997 1.034 1.037      | 0.997 1.044 1.045      | 0.983 1.032 1.031      | 0.999 1.042 1.042     | 0.99-1.03 |
| bmw       | 1.060 1.033 1.044      | 1.072 1.042 1.053      | 1.042 1.034 1.041      | 1.058 1.043 1.046     | 0.99-1.02 |
| crown     | 1.021 1.025 1.027      | 1.034 1.040 1.040      | 1.022 1.039 1.040      | 1.023 1.045 1.047     | 0.99-1.02 |
| ivy       | 1.055 1.058 1.058      | 1.051 1.061 1.064      | 1.046 1.065 1.067      | 1.038 1.072 1.075     | 1.00-1.02 |
| villa     | 1.052 1.034 1.045      | 1.052 1.024 1.045      | 1.041 1.034 1.047      | 1.067 1.044 1.054     | 0.99-1.02 |
| sportscar | 1.055 1.033 1.043      | 1.077 1.037 1.042      | 1.059 1.039 1.040      | 1.078 1.025 1.057     | 0.98-1.03 |
| dambreak  | 1.059 1.045 1.060      | 1.097 1.049 1.077      | 1.058 1.051 1.057      | 1.085 1.051 1.066     | 0.99-1.02 |
| landscape | 0.993 1.002 1.002      | 1.001 0.993 0.986      | 0.981 1.007 1.006      | 0.991 0.968 0.969     | 1.00-1.01 |
| lte-orb   | 1.036 1.044 1.041      | 1.155 1.040 1.037      | 1.032 1.037 1.037      | 1.037 1.041 1.037     | 0.99-1.02 |

The nearest hit gains 2-8% on incoherent rays and 2-10% on primary rays
on every mesh but landscape and zero-day's primary rays, at both widths
and under both schedules (lte-orb's eight-wide primary +15%, where the
compress form had been 0.94x of Embree). Landscape, where the schedules
are 1.15-1.25x Embree throughout and a node rarely hits more than one
child, is level; zero-day's primary rays are level (its own four-wide
tuned cell -2% against Embree's +2.5% drift). The any hit is within noise
of itself, as the sort does not touch it (the eight-wide object's
occluded kernel is allocated a little differently by the compiler and
measures +1-2%). Against Embree, the nearest hit now stands at:

| schedule | primary       | ao            | diffuse       |
|----------|---------------|---------------|---------------|
| embree4  | 1.04 - 1.20x  | 0.98 - 1.19x  | 0.99 - 1.17x  |
| embree   | 1.07 - 1.13x  | 1.04 - 1.21x  | 1.04 - 1.19x  |
| tuned4   | 1.04 - 1.17x  | 1.00 - 1.20x  | 1.00 - 1.21x  |
| tuned    | 1.06 - 1.15x  | 1.10 - 1.25x  | 1.09 - 1.23x  |

The four-wide matching schedule's incoherent-ray cells, 0.94-1.00x before
(pavilion and head the low end at 0.94 and 0.95), are 0.98-1.04x now
(head 0.98/0.99, pavilion 0.99/0.99, the rest at or above 1.00); the
eight-wide matching schedule's 0.99-1.06x are 1.04-1.08x. The tuned
four-wide is at or above Embree on every cell (pavilion's incoherent
rays 1.00, villa's primary 1.04, the two cells that were below).

What the code says, four-wide, the path a node with two hit children
takes (peel4/trace_all.s in the scratchpad, 437 instructions in all
against 446 before). The chain's own work is Embree's: `tzcnt` and the
clear for the lanes, `mov rcx, [r14+rax*8]` for a child off the row, one
`vmovdqa [rsp+0x30], xmm2` for the cell, `mov r13d, [rsp+rdx*4+0x30]`
per key with no clamping, `cmp; cmovl; cmovg; cmovl` for the order of
both children and the farther's distance together, `je` on the clear's
flag for the dispatch. No popcount, no compress, no ternlog, no `and 3`.
Yet the static path is 44 instructions where the compress form's was 38
and Embree's is 20, because of what clang adds around it: three loop-
carried registers spilled at the top of the multi-hit path and five
reloaded in the arms (the loop holds eleven values in general registers
-- the node, the stack count, the best hit's leaf and index, the ray's
tfar bits and the six slab offsets -- and the chain wants eight more at
its widest; Embree carries the same eleven and gets by with one save and
restore, its compiler rematerializing a constant), the arms' push tails
merged into one block with the count's increment in a register (`mov
ebx, k; ...; add r12d, ebx`, about five instructions), the key's inf blend
(four: the key's own miss test against the ray's tfar, dead on every lane
the chain reads, which only the implied-comparisons machinery declined
in "Comparisons a fact implies" could say), the third prefetch line per
child, and the `movzx; and 0xf; dec` the four-bit mask's widening leaves
where `blsr` would do. The measurement says the shorter dependency chain
to the next node's address and the free dispatch outweigh the extra
moves; the three-hit path is longer still, about 76 against 54, and
rarer. The next things to go after, in order: the argmin's payload kept
in its storage rather than in registers (the best hit's leaf and index
are written on an improvement and read once at the end; Embree writes
its hit into the ray and carries only tfar), which frees two registers;
the push tails made distinct (the count updated first, the stores at
constant offsets below it); the inf blend by a structural fact at the
cell's store (a min over a superset of operands is no larger, so the
mask's bound implies the key's); the prefetch width (item 2 above).

Tests: the sorted traversals' goldens re-blessed at the ssa and llvm
levels, all of one kind -- the switch on the count and its packed keys
replaced by the chain (ssa/child-volumes-sorted at four lanes with every
count an arm, ssa/child-volumes-sorted-wide at eight with the general arm
after four peels and its keys packed inside it, ssa/sort-key-through-bits
with one cell shared by the key and the carried bound and the key read
at a lane as its bits, ssa/sort-key-nonnegative with the flip per lane,
arena-rows-vectorized, ptr-arena-rows-vectorized, child-volumes-
vectorized, tiled-leaf-vectorized, skip-leaf-helper, prefetch-children;
the same five at the llvm level); ssa/sort-peel-prefetch and
backends/llvm/sort-peel-prefetch new, the chain over references that are
pointers with each child's prefetch at its peel; the any hit's goldens
unchanged through the shared MaskLanes; every execution test as before;
suite 1380.

## The key stored for the hits alone: the blend gone, the cell gone (2026-10-05)

The inf blend, from the section above. The sort key is the schedule's
`distmin(r, box)`, the box's entry distance where the ray hits it and
infinity where it misses; its miss test is the slab test against the
ray's own tfar. The node's hit mask is the same slab test against the
running best, folded into tfar as `min(tfar, best - 1)` earlier. The
chain reads the key only at lanes the mask has on, and a lane the mask has
on cannot miss by the key's test -- the mask's bound is a min over the
key's bound and the best, and a min over more things is never larger --
so the infinity was never read, and it cost four vector instructions on
every node that hit more than one child: two mins to redo tFar against
the ray's tfar, a compare and a masked broadcast of infinity. Two things
kept the compiler from seeing it. The key was stored for every child,
whatever its test said, so no store carried the fact that only hits are
read; and by the time the chain's cell copy existed the mirror pass had
put the mask's bound behind a block argument, `(*@mirror)`, where the
superset of min operands is no longer visible.

What changed, each piece its own general fact:

- The chain stores its cell zero-masked by the hit mask
  (SSA/SortRecursion.cpp, store_cells): `select(mask, keys, 0)` and a
  whole store, where before it stored the keys as they were. The lanes the
  mask has off hold zero, which nothing reads; what the select buys is
  that its hits' arm is read knowing the mask holds (the simplifier's
  existing assume on a select), and under the mask the key's own test is
  decided. One zero-masked register move, `vmovdqa32 xmm0{k1}{z}, xmm2`,
  in place of the four instructions of the blend.
- The simplifier learns three things (SSA/Simplify.cpp). A read of an
  array a block has just stored, at the index it stored at, is the value
  stored (`stored_at`), looked for back through the chain of blocks each
  the only way into the next -- the children's loop's mask and keys,
  written in the loop's body and read whole by the run that follows,
  which is what hands the key's select to the cell store in the first
  place; the any hit's arms lost a reload of the mask the same way. A
  comparison a fact implies through a chain of mins or maxes over
  integers is decided (`implied_comparison`): `a <= min(S)` says `a <=
  min(T)` for T within S and denies `min(T) < a`, the strict forms where
  the fact is strict, and the mirror image for a max; the leaves compared
  structurally a few steps deep, since the ray's tfar read as its bits in
  two blocks is two instructions. It fires at the simplify after the
  directives, before the mirror pass, where the mask's chain still reads
  `min(slabs, bc(min(tfar, best - 1)))`. And a conjunction assumed true
  is each conjunct assumed true (Halide's learn_true), which a masked
  store's value is now simplified under as well.

Tried first and withdrawn, measured: the children's loop recording the
key and the bound for the children that pass the test alone, so that the
loop's own store was masked and the chain read the array in place with no
cell. The code was the shortest yet -- one masked store a visit, Embree's
spill of tNear -- and lost 2-14% on primary rays (pavilion 0.86, head
0.92) and 1-4% on pavilion's and dragon's incoherent rays, with the rest
1-4% ahead: a masked store forwards to nothing, so the lane loads behind
it wait for it to retire, where the cell's whole store forwards to them.
The loop-side pieces (one `let` per key or bound before the test, the
records in one `if`, the linearizer's bypass dropped for a lane's own
slot) went with it; the store-to-load forwarding and the implied
comparison stayed, and the fold moved to the cell.

The kernel's path for two hits now, four-wide, after the one-hit prefix:

```
tzcnt    edx, ebx                    ; l1
mov      [rsp+0x18], r12             ; three spills of loop state
vmovdqa32 xmm0{k1}{z}, xmm2          ; the keys, zero where the mask is off
mov      [rsp+0x28], r15 ; mov [rsp+0x20], r13
lea      esi, [rbx+0xe]
mov      r12, [r14+rdx*8]            ; c1
prefetcht0 x3
vmovdqa  [rsp+0x30], xmm0            ; the cell, one whole store
mov      r13d, [rsp+rdx*4+0x30]      ; k1
mov      r15d, [rsp+rax*4+0x30]      ; k0
mov      rax, r12 ; cmp r15d, r13d ; mov edx, r13d
cmovl    rax, rcx ; cmovg edx, r15d ; cmovl rcx, r12
and      esi, ebx
je       two
```

Twenty-four instructions where the cell form's were thirty and Embree's
are twenty (the kernel 441 in all); what remains over Embree is the
spills and their reloads (eight), the merged push tails (five), the third
prefetch line (two) and the four-bit mask's widening (two).

Measured on the thirteen meshes (bench-ab.sh: base 3c7eaa71 and new run
back to back per mesh, cpu 11, side 2048, least of 5; every ray agrees
with Embree on every mesh, width and schedule), the nearest hit's rate new
over base, primary / ao / diffuse:

| mesh      | embree4           | embree            | tuned4            | tuned             |
|-----------|-------------------|-------------------|-------------------|-------------------|
| head      | 1.004 1.009 1.009 | 1.003 1.007 1.006 | 1.008 1.017 1.016 | 1.004 1.011 1.014 |
| ganesha   | 1.000 1.004 1.002 | 1.010 1.007 1.006 | 1.005 1.012 1.011 | 1.015 1.019 1.019 |
| dragon    | 1.005 1.005 1.003 | 1.005 1.008 1.000 | 1.008 1.012 1.013 | 1.010 1.011 1.011 |
| pavilion  | 0.999 1.006 1.007 | 1.007 1.008 1.009 | 1.012 1.011 1.011 | 1.005 1.004 1.005 |
| zero-day  | 0.992 1.008 1.010 | 0.995 1.005 1.006 | 0.997 1.006 1.009 | 0.995 1.017 1.017 |
| bmw       | 1.008 1.010 1.009 | 1.000 1.007 1.002 | 1.009 1.012 1.008 | 1.013 1.019 1.027 |
| crown     | 1.007 1.013 1.013 | 1.003 1.009 1.010 | 1.007 1.009 1.007 | 1.007 1.020 1.020 |
| ivy       | 0.997 1.016 1.014 | 1.018 1.004 1.006 | 1.019 1.016 1.013 | 1.003 1.013 1.012 |
| villa     | 1.011 1.005 1.008 | 1.002 1.007 1.002 | 1.019 1.015 1.015 | 1.010 1.017 1.022 |
| sportscar | 1.011 1.009 1.009 | 1.005 1.005 1.015 | 1.015 1.023 1.011 | 1.010 1.020 1.020 |
| dambreak  | 1.010 1.006 1.004 | 0.999 1.005 0.989 | 1.007 1.011 1.009 | 1.013 1.019 1.000 |
| landscape | 0.995 0.996 0.997 | 1.005 1.001 1.004 | 1.012 0.998 0.998 | 1.004 1.004 0.999 |
| lte-orb   | 1.003 1.002 1.005 | 1.004 0.996 1.002 | 1.008 1.002 1.005 | 1.007 1.015 1.008 |

The size the count predicted: +0.5-2% on incoherent rays on every mesh
but landscape (level, few multi-hit nodes), under both schedules at both
widths, the tuned schedules a little more than the matching ones; primary
rays level within noise. Embree's own rate held within 1-2% on every pair
but two cells where the machine was disturbed (crown's four-wide primary
and pavilion's tuned four-wide, whose new-over-base reads 1.007 and
1.011 all the same); lte-orb's eight-wide pair first read 0.72 on primary
rays with Embree's rate steady, and re-run on a quiet machine reads as
the table has it. The any hit is unchanged. Against Embree
the ranges of the previous section move up by about a point on
incoherent rays: embree4 0.99-1.20x primary / 0.99-1.20x ao / 0.99-1.21x
diffuse, embree 1.07-1.13x / 1.05-1.22x / 1.05-1.21x, tuned4 1.06-1.19x /
1.01-1.20x / 1.02-1.21x, tuned 1.09-1.15x / 1.13-1.26x / 1.10-1.25x.

The argmin's payload in storage, probed and found to make no difference
(2026-10-05, the user's ask). The accumulator carries the best hit's key
and the reference to its element, and LLVM promotes the whole pair into
registers, two of them for the reference, which the chain's two-hit path
then spills loop state around (three spills, five reloads). A probe in a
worktree, not committed: the pair updated by a branch, the key stored as
a value and the reference's fields through volatile stores, so that LLVM
could not promote the reference's slot and the loop carried the key
alone. The code came out as intended -- the two-hit path 19 instructions
in place of 24, no spill in it, the node pointer and the first child in
the freed registers (kernel 437 against 441) -- and measured level: on
head, pavilion, dambreak and ganesha at both widths, new over base 0.98-
1.01 on every cell, Embree's rate steady. The spills are store-forwarded
and off the critical path; what paces a node visit is the chain from the
mask to the next node's address and the key loads to the compare, not the
count of moves beside it, which is also why the chain beat the compress
form while being longer. So a principled form of this (the accumulator's
payload lowered to its storage, written on an improvement) is not worth
its machinery for rtq, and the register-pressure follow-ups (the exchange
sorted inside the arm, the merged tails) are expected to be level too.
What remains against Embree's four-wide kernel on incoherent rays is 1-2%
on head and pavilion under the matching schedule, with the tuned
schedules ahead everywhere; the next items are the prefetch width (item
2 of the profile section, a short A/B) and the other Embree trees.

Tests: the sorted traversals' goldens re-blessed, read by kind -- the
cell stored zero-masked and the key stored as the entry distance with no
infinity (ssa/child-volumes-sorted, -wide, sort-key-nonnegative,
sort-key-through-bits, sort-peel-prefetch, arena-rows-vectorized,
ptr-arena-rows-vectorized, child-volumes-vectorized, tiled-leaf-vectorized,
skip-leaf-helper, vectorize-sorted-traversal and their llvm forms); the
any hit's mask reload gone to the comparison (ssa/any-hit-arms,
any-hit-prefetch, child-volumes-any, llvm/any-hit-prefetch); a masked
store's value simplified under its mask, `select(m && odd, ..)` to
`select(odd, ..)` (ssa/defer-gang-bools, vectorize-loop,
vectorize-loop-mutual, vectorize-packet-traversal and their llvm forms);
one error golden's source line. Every execution test as before; suite
1380.

## The thirteen meshes at 9440bea3: bonsai over Embree on every cell (2026-10-05)

The user's ask: every mesh, both trees (Embree's BVH8 and its BVH4, each
over Triangle4 leaves, the layouts matched byte for byte), both queries,
every ray type, on a quiet machine, the speedup bonsai achieves over
Embree. compare.sh with `--schedule embree4,tuned4` and then
`--schedule embree,tuned`, side 2048 (4,194,304 primary rays; the ao and
diffuse sets are one ray per primary hit), least of five repeats, one
thread on both sides pinned to cpu 11, nothing else on the machine (the
pbrt session held its work for a reserved block, 18:20-18:28, and for a
re-run of two meshes at 18:31). The compiler is the tree's at 9440bea3
(ad9da0f0's code; the one uncommitted file in the tree is the GPU
runtime's header, nothing rtq reads). Every ray agrees with Embree up to
ties on all 52 (mesh, schedule) pairs.

Which cells matter. The driver measures three ray sets against both
queries, six cells a schedule, but a renderer asks each query of
particular rays: the nearest hit of primary and diffuse rays (the camera
and the bounce), the any hit of ao rays (occlusion). Those are the three
columns per schedule below; the other three (the nearest hit of ao rays,
the any hit of primary and diffuse rays) are measured as controls and
follow in parentheses in the second table.

Bonsai's rate over Embree's, the cells that matter (nearest hit: primary,
diffuse; any hit: ao):

| mesh | embree4 pri dif ao | tuned4 pri dif ao | embree pri dif ao | tuned pri dif ao |
|---|---|---|---|---|
| head      | 1.10 1.00 1.02 | 1.09 1.09 1.10 | 1.10 1.05 1.01 | 1.07 1.16 1.12 |
| ganesha   | 1.06 1.04 1.02 | 1.05 1.07 1.05 | 1.08 1.06 1.01 | 1.10 1.13 1.07 |
| dragon    | 1.08 1.03 1.02 | 1.08 1.07 1.04 | 1.08 1.05 1.02 | 1.09 1.11 1.07 |
| pavilion  | 1.12 0.99 1.02 | 1.10 1.01 1.03 | 1.08 1.06 1.03 | 1.06 1.11 1.05 |
| zero-day  | 1.22 1.02 1.02 | 1.22 1.12 1.13 | 1.15 1.08 1.04 | 1.15 1.22 1.16 |
| bmw       | 1.07 1.04 1.05 | 1.06 1.11 1.11 | 1.11 1.08 1.05 | 1.09 1.17 1.14 |
| crown     | 1.14 1.03 1.03 | 1.14 1.15 1.14 | 1.10 1.07 1.04 | 1.12 1.26 1.18 |
| ivy       | 1.12 1.04 0.99 | 1.11 1.14 1.13 | 1.08 1.05 1.00 | 1.09 1.22 1.13 |
| villa     | 1.06 1.02 1.03 | 1.05 1.10 1.09 | 1.10 1.08 1.03 | 1.10 1.17 1.11 |
| sportscar | 1.10 1.04 1.04 | 1.10 1.11 1.10 | 1.10 1.09 1.06 | 1.11 1.17 1.15 |
| dambreak  | 1.08 1.04 1.02 | 1.07 1.07 1.06 | 1.08 1.06 1.01 | 1.09 1.10 1.07 |
| landscape | 1.16 1.18 1.21 | 1.18 1.18 1.19 | 1.13 1.19 1.16 | 1.16 1.19 1.16 |
| lte-orb   | 1.11 1.05 1.03 | 1.10 1.09 1.06 | 1.10 1.05 1.03 | 1.10 1.13 1.08 |

The ranges over the thirteen meshes:

| schedule | nearest, primary | nearest, diffuse | any, ao | cells under 1.00 |
|---|---|---|---|---|
| embree4 (Embree's BVH4 schedule) | 1.06-1.22 | 0.99-1.18 | 0.99-1.21 | pavilion diffuse 0.99, ivy ao 0.99 |
| tuned4 | 1.05-1.22 | 1.01-1.18 | 1.03-1.19 | none |
| embree (Embree's BVH8 schedule) | 1.08-1.15 | 1.05-1.19 | 1.00-1.16 | none (ivy ao 1.00) |
| tuned | 1.06-1.16 | 1.10-1.26 | 1.05-1.18 | none |

Every cell, the controls in parentheses (nearest hit of ao rays, any hit
of primary rays, any hit of diffuse rays):

| mesh | embree4 | tuned4 | embree | tuned |
|---|---|---|---|---|
| head      | 1.10 1.00 1.02 (0.99 1.13 1.02) | 1.09 1.09 1.10 (1.09 1.11 1.09) | 1.10 1.05 1.01 (1.05 1.13 1.01) | 1.07 1.16 1.12 (1.16 1.13 1.11) |
| ganesha   | 1.06 1.04 1.02 (1.03 1.05 1.01) | 1.05 1.07 1.05 (1.08 1.03 1.04) | 1.08 1.06 1.01 (1.06 1.04 1.01) | 1.10 1.13 1.07 (1.13 1.04 1.04) |
| dragon    | 1.08 1.03 1.02 (1.03 1.05 1.03) | 1.08 1.07 1.04 (1.08 1.02 1.04) | 1.08 1.05 1.02 (1.06 1.04 1.01) | 1.09 1.11 1.07 (1.14 1.04 1.05) |
| pavilion  | 1.12 0.99 1.02 (0.99 1.13 1.02) | 1.10 1.01 1.03 (1.01 1.12 1.04) | 1.08 1.06 1.03 (1.07 1.12 1.02) | 1.06 1.11 1.05 (1.10 1.10 1.05) |
| zero-day  | 1.22 1.02 1.02 (1.01 1.22 1.02) | 1.22 1.12 1.13 (1.11 1.23 1.13) | 1.15 1.08 1.04 (1.08 1.21 1.03) | 1.15 1.22 1.16 (1.21 1.17 1.15) |
| bmw       | 1.07 1.04 1.05 (1.02 1.09 1.06) | 1.06 1.11 1.11 (1.10 1.07 1.10) | 1.11 1.08 1.05 (1.09 1.11 1.06) | 1.09 1.17 1.14 (1.18 1.07 1.12) |
| crown     | 1.14 1.03 1.03 (1.02 1.15 1.03) | 1.14 1.15 1.14 (1.14 1.13 1.14) | 1.10 1.07 1.04 (1.09 1.13 1.03) | 1.12 1.26 1.18 (1.25 1.11 1.17) |
| ivy       | 1.12 1.04 0.99 (1.04 1.12 1.02) | 1.11 1.14 1.13 (1.17 1.10 1.16) | 1.08 1.05 1.00 (1.05 1.06 0.99) | 1.09 1.22 1.13 (1.23 1.05 1.11) |
| villa     | 1.06 1.02 1.03 (1.01 1.09 1.04) | 1.05 1.10 1.09 (1.08 1.07 1.09) | 1.10 1.08 1.03 (1.07 1.09 1.04) | 1.10 1.17 1.11 (1.16 1.09 1.10) |
| sportscar | 1.10 1.04 1.04 (1.03 1.08 1.03) | 1.10 1.11 1.10 (1.11 1.06 1.08) | 1.10 1.09 1.06 (1.09 1.07 1.05) | 1.11 1.17 1.15 (1.20 1.07 1.12) |
| dambreak  | 1.08 1.04 1.02 (1.03 1.06 1.04) | 1.07 1.07 1.06 (1.08 1.04 1.03) | 1.08 1.06 1.01 (1.06 1.03 1.04) | 1.09 1.10 1.07 (1.13 1.03 1.04) |
| landscape | 1.16 1.18 1.21 (1.18 1.16 1.20) | 1.18 1.18 1.19 (1.19 1.16 1.19) | 1.13 1.19 1.16 (1.19 1.13 1.15) | 1.16 1.19 1.16 (1.20 1.12 1.17) |
| lte-orb   | 1.11 1.05 1.03 (1.04 1.09 1.04) | 1.10 1.09 1.06 (1.08 1.07 1.05) | 1.10 1.05 1.03 (1.05 1.07 1.02) | 1.10 1.13 1.08 (1.13 1.06 1.08) |

Reading it. Both tuned schedules are at or above Embree on every cell of
every mesh, the eight-wide one by 5-26% on the cells that matter and the
four-wide one by 1-22%. The matching schedules, the same traversal as
Embree's kernels over the same bytes, are at or above Embree on every
cell but two four-wide incoherent ones a point under (pavilion's diffuse
nearest hit and ivy's ao any hit at 0.99; head's diffuse at 1.00), the
residue of the gap the profile section opened at 0.94-1.00. Primary rays
are the strongest column for the matching schedules (1.06-1.22x), where
the chain's descent and the leaf's early-out pay; the tuned schedules'
strongest is the incoherent rays (diffuse 1.10-1.26x eight-wide), where
the nearest hit without the early-out and the any hit's arms pay. The
any hit's ao column is the narrowest everywhere (0.99-1.21x), its code
having been Embree's shape since the arms-by-count work; what separates
the tuned any hit from the matching one there is the schedule's leaf
order. Landscape stands apart at 1.13-1.21x on every schedule and cell:
its nodes rarely hit more than one child, so the whole traversal is the
node test and the one-hit descent, where we have been ahead since the
fold.

Two cells were disturbed in the first pass and re-run, each judged by
Embree's own rate: head's eight-wide pass read our side at 62.4 Mrays/s
against its 69.4-70.4 in every other run today with Embree's steady at
64.0 (0.98x on primary rays); re-run, 70.35 and 1.10x, the table's
value. Pavilion's four-wide pass had Embree's own rate under the tuned4
schedule at 59.5/21.5/20.9 Mrays/s against its usual 66/23.5/22.8, ours
steady (a false 1.21x/1.10x/1.11x); the re-run's tuned4 pass was clean
(66.05, 1.10/1.01/1.01, the table's values) while its embree4 pass took
the same dip on Embree's side (56.7 on primary rays, a false 1.31x), so
pavilion's embree4 cells are the first pass's, whose Embree rate was its
usual. Embree's BVH4 kernel on pavilion has dipped this way three times
today (blend4t's base, this pass, the re-run), always its side and
always the primary rays most, the only mesh that does it; worth watching,
but it is Embree's number moving, not ours, and the table takes each
cell from the pass where Embree's rate was its usual. Against this
morning's runs (ad9da0f0's measurement) Embree's own rate holds within
2% on every other cell but ivy's four-wide diffuse, where both sides
dropped 5% together (3.87 and 4.05 against 4.06 and 4.21; the ratio
1.04 either way), and lte-orb's eight-wide primary, whose morning log
still held the known disturbed 0.72 and reads 1.10x here as its re-run
did then.

The day's arc on the four-wide matching schedule's incoherent rays, the
cells this afternoon's work was about: 0.94-1.00x at 90310848 (the tree
matched), 0.98-1.04x after the chain (3c7eaa71), 0.99-1.05x after the
blend (ad9da0f0), with the tuned4 schedule 1.01-1.18x over the same
cells. The next items stand as the section above left them: the
prefetch width (a short A/B), then the other Embree trees.

## The closest point against Embree's rtcPointQuery (2026-10-05)

The user's ask: what other trees and queries Embree supports (the section
after this one), and how we compare on the closest point, on the two
trees in hand. The query is new in rtq; nothing in the compiler changed.

What Embree offers. rtcPointQuery (rtcore_scene.h:154) takes a point and
a radius and runs BVHNIntersector1::pointQuery
(bvh_intersector1.cpp:211-297): the nearest hit's loop with the point's
squared distance to each child's box where the ray has its entry
distance -- pointQuerySphereDistAndMask (node_intersector1.h:200-212), `v
= min(max(org, lower), upper) - org; dist = v.v; mask = dist <= rad^2`
-- the children ordered by it with the same traverseClosestHit, and a
popped node dropped where its distance is beyond the radius squared. The
leaf is the user's: PrimitivePointQuery1 (geometry/primitive.h:30-45)
calls Geometry::pointQuery (common/geometry.cpp:237-264) for each valid
triangle of each block, which fills an argument struct and calls the
function the query was given. Embree computes no distance to a primitive
itself. Its closest-point tutorial's callback
(tutorials/closest_point/closest_point_device.cpp:267-346) reads the
triangle's vertices from the mesh, runs closestPointTriangle
(tutorials/common/math/closest_point.h: Ericson's Voronoi-region test,
Real-Time Collision Detection 5.1.5), takes the distance -- a square root
-- and where it is below the radius shrinks the radius to it and records
the triangle, returning true to say so. One more thing the code says,
verified (context.h:102-155, geometry.cpp:253-261,
bvh_intersector1.cpp:282): without instances, the radius the node test
uses is never refreshed. `context->query_radius` is set at construction
from the query's radius and rewritten only on the instance path, and
`tquery.rad = context->query_radius` after a leaf shrinks the radius
copies the same value -- so a query from infinity has every valid child
of every visited node pass the mask and be pushed, and the shrinking
radius prunes at the pop alone (`stackPtr->dist > cull_radius`).
Correct, but the node test culls nothing and every node sorts all of its
children.

The query here (rtq.bonsai). `Point`; `distmin(Point, AABB)`, Embree's
clamped squared distance, and `distmax`; `distmin(Point, Triangle)`,
Ericson's test in Embree's order over the triangle as the block stores it
(`a = v0`, `ab = -e1`, `ac = e2`, `p - b` and `p - c` from `p - a`),
returning the squared distance; `valid(t)`, Embree's TriangleM::valid
(`geomID != -1`), which the point query needs where the ray queries do
not, the padding slot being the zero triangle at the origin; `closest(q)
= argmin(|t| distmin(q, t), filter(|t| valid(t), triangles))`; and the
exported batch. Squared distances throughout -- they order as distances
do and are never negative -- so there is no root anywhere, where Embree's
callback takes one per triangle. The schedule is the nearest hit's,
directive for directive: `sort` by `distmin(q, box)`, `vectorize` of the
children and of the leaf, `loopify`, `prefetch`, in all four files (the
tuned ones the same as the matching, no step having measured worse yet).
The lowering makes the traversal with `distmin(q, box) < best` at the
node and again at the pop. The leaf is where Embree's shape cannot be
followed, Embree having no leaf code for this query; the four-wide leaf
over the block's stored edges is the query's own form. The kernel,
four-wide: 552 lines of LLVM, 161 four-wide float operations, four vector
divisions, no square root, no scalar float operation -- the seven
Voronoi regions evaluated as selects over the four lanes.

The driver. Two batches of points, as many as there were primary hits
(side 1024: 9.5k on zero-day to 345k on villa): `near`, the hit points
pushed along the normal facing the camera by a random distance up to a
tenth of the diagonal (what a distance field or a walk on spheres asks);
`volume`, uniform in the mesh's box grown by a tenth on every side.
Embree's side is rtcPointQuery per point with a fresh context, the radius
infinite as the tutorial's queries begin, the tutorial's callback
transcribed (the vertices from the mesh, closestPointTriangle, the
distance, the radius shrunk) without its branches for instances; the
query structs made before the clock and reset between runs, as the rays'
are. The row also counts the triangles Embree handed its callback per
query. Agreement: the two triangles' distances in double; equal to a
hundred-thousandth of the diagonal is a tie -- a point whose closest
point lies on an edge or a vertex two triangles share, 1-18% of the
points, or a point within a hair of the surface, whose distance is mostly
last bits, which is why the test is against the scene's scale and not the
distance's (the first form, relative to the distance, read gaps of 1e-3
on such points). The largest tie gap over the 26 cells is 2.9e-7 of the
diagonal, float rounding; no differences on any mesh at either width.

Measured (compare.sh --query closest, side 1024, least of five, one
thread on both sides on cpu 11, the machine quiet 19:10-19:18; the
matching schedules alone, the tuned ones being the same directives for
this query; a first pass at side 2048 was stopped, Embree's side taking
minutes a mesh). Bonsai's rate over Embree's; Embree's rate in million
queries a second and its callbacks per query from the eight-wide pass:

| mesh | triangles | BVH4 near | BVH4 volume | BVH8 near | BVH8 volume | Embree Mq/s near / volume | callbacks a query near / volume |
|---|---|---|---|---|---|---|---|
| head | 17674 | 3.92x | 4.42x | 4.00x | 4.57x | 0.77 / 0.58 | 49.7 / 57.7 |
| ganesha | 4323658 | 5.74x | 7.27x | 6.06x | 8.45x | 0.11 / 0.03 | 282.3 / 308.3 |
| dragon | 7219045 | 5.90x | 7.07x | 6.44x | 8.06x | 0.05 / 0.03 | 653.4 / 534.0 |
| pavilion | 11366 | 3.31x | 4.47x | 3.16x | 4.51x | 2.24 / 0.68 | 17.8 / 46.9 |
| zero-day | 48960 | 3.78x | 4.07x | 4.01x | 4.24x | 0.39 / 0.33 | 110.2 / 121.8 |
| bmw | 110592 | 4.19x | 4.57x | 4.29x | 4.82x | 0.52 / 0.31 | 70.1 / 108.2 |
| crown | 155520 | 4.39x | 4.50x | 4.60x | 4.82x | 0.36 / 0.25 | 103.2 / 134.2 |
| ivy | 179603 | 5.43x | 4.85x | 5.33x | 4.74x | 0.37 / 0.51 | 83.0 / 68.9 |
| villa | 390784 | 4.38x | 4.88x | 4.56x | 5.16x | 0.33 / 0.22 | 112.6 / 153.0 |
| sportscar | 1091232 | 4.78x | 5.34x | 4.95x | 5.79x | 0.16 / 0.12 | 238.2 / 221.6 |
| dambreak | 1015024 | 4.04x | 5.48x | 3.98x | 5.92x | 0.72 / 0.32 | 36.7 / 40.8 |
| landscape | 1916928 | 3.52x | 3.09x | 3.58x | 3.26x | 0.41 / 0.72 | 68.6 / 35.3 |
| lte-orb | 3423232 | 4.77x | 5.03x | 4.99x | 5.70x | 0.07 / 0.02 | 526.8 / 1320.1 |

Four-wide 3.3-5.9x on near points and 3.1-7.3x on volume points;
eight-wide 3.2-6.4x and 3.3-8.5x. The gap grows with the mesh: head and
pavilion at 3-4.5x with Embree handing its callback 20-60 triangles a
query, ganesha and dragon at 6-8.5x with 300-650. Where the time goes,
by perf on head's near points, eight-wide, cycles: Embree's callback 36%
of the samples, Geometry::pointQuery's per-triangle dispatch 21%, its
traversal 18%; ours, the whole query, 19%. Our entire query costs what
Embree's traversal alone costs, and Embree's leaf -- callback and
dispatch, 58% of the samples -- is three times our whole query. Three
things make the gap, largest first: Embree's leaf is a function call per
triangle (an argument struct written then read, two indirect calls, the
vertices gathered through the mesh's indices, a square root) where ours
reads four triangles where they lie and tests them as four lanes; Embree's
node test culls nothing, the radius it uses being the initial infinity,
so every visited node pushes and sorts all its children where ours tests
them against the best; and both grow with the mesh, Embree's callback
count most (lte-orb's volume points: 1320 triangles a query). A question
of what Embree would do given a tighter radius was asked and withdrawn:
the reference is measured as it ships, a real query from infinity.

What this comparison is. The one query in rtq where the matching schedule
cannot follow Embree's code at the leaf, because Embree has no leaf code
for it; the table compares the closest-point query a user of Embree can
write, the way Embree's own tutorial writes it, with the query here. Next
for it: the range query (the same rtcPointQuery, the callback never
shrinking the radius), item 1 of the plan below; and `skip` on the
Voronoi regions' early returns as a tuned experiment, expected level, the
leaf being four lanes of selects.

## Embree's trees and queries: what is matched, what is left (2026-10-05)

The inventory, from the checkout at deps/embree (4.4.1), for planning what
rtq covers next; file paths are Embree's.

Trees. Two widths, BVHN<4> and BVHN<8> (bvh.h), chosen per scene in
Scene::create*Accel (common/scene.cpp:152-712): eight wide on a machine
with AVX for a scene that is not compact, four wide otherwise and always
for a compact scene. Node kinds, each with its tag in the reference's low
bits (bvh_node_ref.h): the AABB node (bvh_node_aabb.h, the one matched);
its motion-blur forms, bounds plus per-child deltas linear in time
(AABBNodeMB, bvh_node_aabb_mb.h) and with a time interval per child for
time splits (AABBNodeMB4D, bvh_node_aabb_mb4d.h); the oriented-box node
for curves, an affine space per child (OBBNode, bvh_node_obb.h, and its
MB form); and the quantized node, eight-bit bounds with a start and scale
per node, dequantized at the test (QuantizedNode, bvh_node_qaabb.h),
reachable only through the device config (`tri_accel=qbvh4.triangle4i`,
`qbvh8.triangle4i`, `qbvh8.triangle4`, scene.cpp:223-230), not through a
scene flag. Triangle leaves, by `mode = 2 * compact + robust`
(scene.cpp:156-218): Triangle4 (a vertex and two edges, Moeller; the one
matched) by default; Triangle4v (three vertices, Pluecker, with the
robust node test) under RTC_SCENE_FLAG_ROBUST; Triangle4i (three vertex
indices into the mesh, the vertices gathered at the test) under
RTC_SCENE_FLAG_COMPACT, four wide only. Build quality LOW selects the
two-level build (a top-level tree over per-geometry trees, refit or
rebuilt per geometry) with the same leaves. Motion blur puts Triangle4i
leaves under MB4D nodes (Triangle4vMB only by config). Beyond triangles:
quads (Quad4v, Quad4i); curves, lines and points under mixed AABB and OBB
nodes (Curve8v, Curve4i/8i, Line4i, Point4i, chosen per geometry in
bvh_builder_hair.cpp); grids as subgrids with an embedded quantized box
(SubGridQBVH4/8); subdivision patches with eager grids (SubdivPatch1,
GridSOA), four wide only; user geometry (Object); instances
(InstancePrimitive) and instance arrays (InstanceArrayPrimitive). Each
geometry kind gets an accel of its own and AccelN walks them in turn
(acceln.cpp).

Queries (include/embree4/rtcore_scene.h). rtcIntersect1 and rtcOccluded1
(matched); their packet forms rtcIntersect4/8/16 and rtcOccluded4/8/16,
which are separate kernels (BVHNIntersectorKHybrid: a packet traversal
that falls to single rays when few are active); each with an optional
arguments struct (coherent/incoherent flags, a feature mask, a context,
filter callbacks per hit, an intersect callback for user geometry).
rtcForwardIntersect/Occluded, a ray continued from a user-geometry
callback into another scene (for instancing done by hand). rtcPointQuery
(matched today, below) and rtcPointQuery4/8/16, which are scalar loops
over it (rtcore.cpp:514-597). rtcCollide, the overlap of two scenes' trees
with every leaf pair handed to a callback, no narrow phase, user geometry
only (bvh_collider.cpp). Geometry masks and filter functions, which are
predicates per hit. rtcInterpolate is not a tree query. The stream API
(rtcIntersect1M/NM/Np) is gone in 4 (CHANGELOG.md).

The plan, in the order recommended, each item its layout match first,
then the matching schedule, then every mesh and ray type, as the ray
queries were done:

1. The range query: rtcPointQuery whose callback never shrinks the radius
   -- every triangle within a distance -- against `filter(|t| distmin(p,
   t) <= r2, triangles)` collected or counted. The same tree and the same
   callback shape as the closest point, an hour's work, and it measures
   the `filter` lowering over a tree, which nothing in rtq does yet.
2. Triangle4v, Embree's robust mode: the leaf as three vertices (the same
   176 bytes), the Pluecker test (triangle_intersector_pluecker.h), the
   robust node test (intersectNodeRobust: the reciprocal direction scaled
   down by three ulps for the near slabs and up by three for the far,
   node_intersector1.h:106-110), RTC_SCENE_FLAG_ROBUST on Embree's side.
   The first of the other trees by the standing direction.
3. Triangle4i and the quantized node, Embree's compact direction: the leaf
   as indices into a second storage (the mesh's vertices), the vertices
   gathered at the test -- the layout language's indirection -- and the
   quantized node's eight-bit bounds dequantized at the test, comparable
   eight wide against Triangle4 through `tri_accel=qbvh8.triangle4`.
4. Ray packets: rtcIntersect8/16 against the parfor over the rays
   vectorized -- the packet traversal the schedule asks for -- on coherent
   primary rays where Embree's packets pay and on incoherent ones where
   its hybrid falls back to single rays. The item that measures the
   vectorizer against Embree's hand-written packet kernels.
5. Instances: a top-level tree over transforms and per-object trees,
   `flatten` over a match (docs/trees-of-mixed-primitives.md), against
   rtcIntersect1 through InstancePrimitive; needs a multi-object input to
   the driver. pbrt's scenes are where this matters.
6. Lower, as pbrt needs them: motion blur (Triangle4i under MB4D nodes,
   two vertex sets and a time), quads (Quad4v: two triangles sharing an
   edge, tested as one), curves and hair (OBB nodes and the curve
   intersectors, the hardest), rtcCollide (a join of two trees by box
   overlap, apps/queries/joins; Embree's is user geometry only, so the
   comparison would be on boxes), and masks and filter functions (REST
   predicates per hit; Embree's filter is a callback per hit, the point
   query's shape again).

## The closest point against FCPW, on FCPW's trees (2026-10-05)

The user's ask: compare the closest-point query against FCPW ("Fastest
Closest Points in the West", Sawhney; deps/fcpw, 3b69ddb, an optional
submodule in Embree's spirit, header-only with Eigen and enoki as
submodules of its own) on as many of its high-performance trees as
possible, copying its tree layout and its schedule exactly and building
its tree in the driver, so that the comparison is fair; and say which of
its trees or schedules the layout and scheduling languages cannot
represent.

What FCPW is, for triangles (include/fcpw). A Scene over a triangle soup
builds a binary BVH (aggregates/bvh.inl: binned SAH, 8 buckets, leaves of
`leafSize` triangles) and, when asked to vectorize, collapses it into its
"Mbvh" (aggregates/mbvh.inl, collapseBvh): nodes of
FCPW_MBVH_BRANCHING_FACTOR children, 4 by default or 8 with
FCPW_USE_EIGHT_WIDE_BRANCHING, each node holding its children's boxes as
component vectors (`MbvhNode<3>`: boxMin x y z, boxMax x y z, then the
child slots as ints -- 112 bytes at four children, 224 at eight), and
leaves of packets of FCPW_SIMD_WIDTH triangles stored as three vertices
by component with their ids (`MbvhLeafNode<W, 3>`: 640 bytes at sixteen,
320 at eight). The width is FCPW's build's choice by the instruction set
(enoki's detection: AVX-512 gives 16, AVX2 8, SSE 4), and it also sets
the binary build's leaf size, so the two widths are two different trees.
A child slot is an index into the node array; whether it names a leaf,
the row at the index says, by the sign of its first slot, and a leaf row
holds no boxes but its description (`-(first packet + 1)`, the packet
count, the offset into FCPW's primitive array, the triangle count). An
empty slot is `maxInt` with an inverted box. The query
(findClosestPointFromNode): a stack of (node, squared distance) 96 or
154 deep, the root pushed untested; a popped node dropped where its
distance is beyond the radius squared; at a node the children's boxes
tested at once (overlapWideBox: the point clamped into each box, the
squared length of the offset, `d2Min <= r2`), the children pushed in
descending order of that distance by a sorting network over all four
slots so the nearest is on top (enqueueNodes<4>), and -- this is the one
thing our lowering does not do -- the radius lowered to the least of the
pushed children's farthest box distance (`tMaxMin = min(tMaxMin,
tMax[W])`). At a leaf every packet's lanes are tested at once
(findClosestPointWideTriangle: Ericson's regions as masked selects, with
`if (all(active)) return` after each region, enoki's `rcp` for the
divisions, a square root per lane), and the lanes below the triangle
count are scanned in order with `<=`, the later lane winning a tie. At a
branching factor of 8 the generic enqueueNodes pushes the children in
lane order and swaps the strictly nearest to the top. Nothing is
prefetched. The scalar binary Bvh (vectorize off) tests the two children
from the parent, nearer first with a tie broken by the farther distance,
shrinks the radius by each child's farthest distance, and runs Ericson's
scalar test per triangle through the mesh's vertex indices.

What was built, in the order the user's directions came. The first
measurement, with Embree's arithmetic standing in for FCPW's and the
program's own validity filter on, read 1.24-1.73x on near points and
1.02-1.76x on volume points over FCPW's four configurations; the user
ruled it not yet a legitimate comparison -- the two sides must run the
same computation -- and set three things right before any FCPW number is
reported: FCPW's arithmetic transcribed for the FCPW comparisons, the
padding validity owned by the layout and not the program (no filter on the
closest point), and the lowering's tightening of the radius applied
regardless, with the vectorizer making one reduction of it.

- The reference's arithmetic in a file of its own. rtq.bonsai now holds
  the elements, the three queries and the exports; metrics/embree.bonsai
  and metrics/fcpw.bonsai each define `intersects`, `distmin`, `distmax`
  and `contains` for a ray and a box, a ray and a triangle, a point and a
  box and a point and a triangle, as that reference computes them, and
  compare.sh compiles the one the schedule's reference calls for beside
  the program (`RTQ_METRICS` names another, for measuring an arithmetic
  against the same reference). FCPW's (core/wide_query_operations.h): the
  slab test with a true-division reciprocal direction, float compares,
  `max(0, hmax(tNear))` and `min(tMax, hmin(tFar))` (intersectWideBox);
  Moeller-Trumbore in the textbook form with the determinant rejected
  within the float epsilon in absolute terms and `rcp` for its reciprocal
  (intersectWideTriangle); the point-to-box distances `u = bMin - c; v = c
  - bMax`, `|max(max(u, v), 0)|^2` and `|min(u, v)|^2` (overlapWideBox);
  and the closest point on a triangle in FCPW's region order with `rcp`
  for its three divisions and a root per lane, squared back as FCPW's leaf
  squares it (`d2 = d * d`) to compare with the radius squared. The
  reciprocals are each library's own, written out. Both begin with the
  machine's estimate (`vrcp14ps` here), but Embree's step is `r + r (1 - m
  r)` as two fused multiply-adds and enoki's is `2r - (r m) r` as one fused
  multiply-subtract of a rounded product, and a probe over 285,715 inputs
  in [1, 2) found the two differ in the last bit on 109,804 of them (38%;
  Embree's is the correctly rounded quotient on 99.4%, enoki's on 62%) --
  enough to move a grazing ray across `v >= 0`, `v + w <= 1` or `d <=
  tMax`, which showed as a handful of rays per mesh disagreeing on six of
  the thirteen while the FCPW metric still used the compiler's `rcp`,
  which was Embree's step baked into CodeGen_X86. The user's ruling: the
  compiler offers the machine's instruction and nothing more, and each
  library's refinement is an inline function in its metric file. So the
  language has `rcp_approx` -- the estimate instruction (`vrcp14ps`/
  `rcpps`, PTX's `rcp.approx`, the quotient where there is none), Halide's
  `fast_inverse` -- and no `rcp` any more; metrics/embree.bonsai has
  `rcp_embree` (`r = rcp_approx(a); r + r * (1.0 - a * r)`, over a scalar
  and over a vec3f as Vec3fa's is) and metrics/fcpw.bonsai `rcp_fcpw` (`r =
  rcp_approx(m); t0 = r + r; t1 = r * m; t0 - t1 * r`), which compare.sh's
  `--ffp-contract` (SSA/Contract.h: `c - a b` is `fma(-a, b, c)`, `a b + c`
  is `fma(a, b, c)`, a product consumed by a multiply is left alone) makes
  exactly Embree's `fmadd(r, fnmadd(a, r, 1), r)` and enoki's `fnmadd(t1, r,
  t0)` over its own `t1`. enoki's `fixupimm` after the step (the table
  0x0087A622: a zero's reciprocal the signed infinity, an infinity's the
  signed zero, one's exactly one) is not written: the estimate is exact at
  one, and a zero or infinite denominator fails every comparison the same
  way on both sides. What the interval analysis makes of a reciprocal
  (ssa/intervals-from-conditions.bonsai, ssa/intervals-reciprocal-step.bonsai):
  the estimate of a non-negative interval is non-negative, the rule the
  primitive has, and its bounds allow for the instruction's flushing at
  the ends of the normal range, which differs by instruction (`rcpps`
  takes a denormal operand as zero and returns a denormal quotient as
  zero; `vrcp14ps` takes only an operand below 2^-128 as zero and returns
  the denormal quotient; correctness/llvm/rcp-step-edges runs the edges
  on the host). The refined form written out is read as what it is, one
  Newton step over the estimate, by a rule that knows the two libraries'
  spellings, fused and not (ConstantIntervals.cpp, reciprocal_step and
  newton_step): the estimate's bounds and sign, not known to be a number,
  since where the estimate is infinite the step is inf - inf, a NaN.
  Followed as the
  arithmetic it is spelled as the step would be everything, since
  intervals do not carry the correlation between `r` and `x` (`x r` reads
  as [0, inf], not as within 2^-10 of one; `1 - x r` as [-inf, 1]). The
  spelling is part of the fact: the factored `r (2 - x r)` is -inf where
  the estimate is infinite and these four are NaNs, so it is not read. rtq needs the
  sign: the fold of the running best into the slab test (the next
  section) requires the best's key to be a non-negative number, and the
  key is `T * rcp_embree(absDen)`. Tests: backends/llvm/rcp (the bare
  `rcpps` on the generic target, Embree's step as its fma pair),
  backends/llvm/rcp-approx (`rcp14` on Zen 4, enoki's step as `fma(-(r m),
  r, 2r)`, Embree's beside it), correctness/cpp/rcp (the refinement checked
  against the exact quotient to a few ulps), correctness/llvm/rcp-step-edges
  (the step at a tiny, a huge, a zero and an infinite operand on the host:
  a NaN, a non-negative tiny value, a NaN, a NaN -- the facts the interval
  rule stands on, as they hold on both instructions). One thing
  FCPW's node test does that no box test can: `mask &= neq(node.child,
  maxInt)`, since an empty slot's inverted box (boxMin = maxFloat, boxMax
  = minFloat) passes a slab test whose near and far are a min and a max
  of the two slab distances -- Embree's test on the bits rejects such a
  box, FCPW's does not. The box carries the same mark as the slot, so the
  FCPW ray-box test here tests it as FCPW does, a second conjunct of the
  slab test's (`tMin <= tMax && proper`): one compare and an and per node.
  Writing that conjunct exposed a compiler bug, described below. Embree's file is
  the arithmetic the ray sections above transcribed, and its closest
  point now takes the root Embree's callback takes (`distance(q, p)`) and
  squares it back into the node test's domain, where Embree squares its
  radius instead. The committed FCPW numbers above were taken with
  Embree's arithmetic and are void; the committed Embree closest-point
  numbers were taken with a metric that took no root, cheaper than
  Embree's callback by one `sqrtps` per block.
- The validity of a padding lane is the layout's. A tile's body may end
  `where <predicate>` over its members (`group[4] { ... } where geomID !=
  4294967295u`, Embree's `TriangleM::valid` on the stored ids), and the
  lanes that fail it are not elements of the set (ir::Group::valid,
  ir::TiledArray::valid; the keyword `where`, the parser's clause read
  with the body's members in scope; Lower/ForEachs.cpp runs the leaf's
  body under the predicate at the lane, which the vectorized loop turns
  into one lane mask, the lane's members read through the same
  substitution the derived fields use, Lower/TiledArrays.cpp's
  tiled_lane_value). layouts/embree8.bonsai and embree4.bonsai carry it
  on their Triangle4 tiles; FCPW's layouts exclude their padding by the
  count. The program's `closest` is `argmin(|t| distmin(q, t),
  triangles)` with no filter; `valid` is gone from it. Tests at the three
  levels: lower/tiled-elements-where (the guard in the leaf loop),
  backends/llvm/tiled-elements-where (one `icmp ne <4 x i32>` folded into
  the mask), correctness/cpp/bvh4_tiled_elements_where (decoy walls in the
  padding lanes, just before the real ones, which no ray and no point may
  count). The predicate costs the ray queries one vector compare and an
  and per block on Embree's layout, which Embree does not pay -- its
  zero triangle fails the hit test -- measured below.
- The radius tightened by what a child surely holds, regardless. The
  lowering had this bound (Lower/Trees.cpp, reachable_bound) but emitted
  it only outside a children's loop and only without a filter; with the
  filter gone it applies, and from_children now emits it inside the loop
  too (Carry::reach_of, Carry::tighten: under the child's own test,
  `best.key min= distmax(q, box)` for each child that passed, set where
  no filter of the program's stands between the metric and the set --
  `trace` keeps none, a ray's box holding no promise of a hit). The
  vectorizer makes the accumulate in the gang loop one masked
  cross-lane minimum (SSA/Vectorize.cpp): in the LLVM a `select` of the
  children's `distmax` to `+inf` under the mask and a two-step shuffle
  minimum tree into one compare with the best. FCPW's `tMaxMin =
  min(tMaxMin, tMax[W])`, exactly; and now Embree's schedules prune by it
  too, which Embree's callback form cannot (measured below). Tests:
  lower/argmin-tighten-children (the `min=` in the children's loop of the
  closest query and none in the ray's), backends/llvm/argmin-tighten-
  children (the reduction), and the execution test above, whose closest
  points cross the tightening.
- One tree declaration per width, shared. The four-wide tree and the
  eight-wide tree are each declared once (schedules/trees/bvh4.bonsai,
  bvh8.bonsai: the children's boxes in the parent, a leaf of triangles),
  and the six layouts import them (schedules/layouts/embree4, embree8,
  fcpw4w16, fcpw8w16, fcpw4w8, fcpw8w8.bonsai): Embree's BVH4 and FCPW's
  four-wide Mbvh are one tree in two sets of bytes, as the user pointed
  out, and so are the eight-wide pair. The schedules import a layout.
- The layout language learned derived fields in a tile of elements. A
  tile had to store exactly the element's fields; FCPW's packet stores
  three vertices where the program's triangle, Embree's slot, has a
  vertex and two edges, so no FCPW layout could be written. Now a tile's
  members are stored fields, which are its bytes in their own order, and
  derived ones (`e1 = pa - pb`), the same `name = expr` form a node row
  has for `lo`; between them they provide the element's fields by name
  and type (ir::provides_fields_of). The tile's struct holds the stored
  fields alone (Lower/Layouts.cpp, tile_struct); ir::TiledArray carries
  the derived values, over placeholders no program can name; every
  element read substitutes the lane of each stored field into them
  (Lower/TiledArrays.cpp), and the vectorizer makes the per-lane
  expressions whole-vector ones as it does the stored fields. `e1 = pa -
  pb` is exactly the negation of FCPW's own `ab = pb - pa`, and `e2 = pc
  - pa` its `ac`, two differences FCPW computes at every test as well.
  Tests at the three levels: lower/tiled-elements-derived,
  backends/llvm/tiled-elements-derived (the derived vertices as `fadd <4
  x float>` over the tile's vectors, nothing extracted to make them),
  correctness/cpp/bvh4_tiled_elements_derived (a vertex and two edges
  stored, the three vertices derived and checked). The user's note,
  recorded under Known-open: ir::TiledArray is a one-level special case
  (an indirect group whose body is one anonymous constant-size group), and
  this extension inherits that; the composable form would make every
  constant-size group along the way a lane dimension with the element's
  index decomposed mixed-radix through the nesting, and the row machinery
  (fields, derived fields, switches, lookups) the one lowering for both.
- A leaf whose count is not a multiple of the tile width
  (Lower/ForEachs.cpp): FCPW's leaf holds `child[3]` triangles in whole
  packets, the last part full, and masks the lanes past the count with
  `W = min(WIDTH, nReferences - startReference)`. The loop over such a
  range now runs the tiles that hold the count with `tile * width + lane <
  count` in front of the body, which the vectorized lanes' loop runs as
  one test under a lane mask -- FCPW's mechanism, over FCPW's bytes
  untouched. A count of whole tiles lowers as before.
- Four layouts, FCPW's bytes (schedules/trees/mbvh4w16, mbvh8w16, mbvh4w8,
  mbvh8w8.bonsai: branching 4 or 8, leaf width 16 or 8 -- AVX-512's and
  AVX2's), each a group of node rows indexed by the child slot with
  `kind = child0[31:31]` switching between the interior arm (the slots
  cast to children, the boxes derived as `lo`/`hi`) and the leaf arm
  (`data = range(packets, (-child0 - 1) * W, child3)`), and a tiled group
  of packets whose stored fields are FCPW's and whose element fields are
  derived (`v0 = pa; e1 = pa - pb; e2 = pc - pa; geomID = 0; primID =
  primitiveIndex`). The driver (rtq_hook.cpp, RTQ_FCPW) builds FCPW's
  scene, reaches its protected arrays through a derived class, checks
  every size and offset against FCPW's structs at compile time, and
  copies both arrays byte for byte into allocations of FCPW's alignment.
  The root is row 0.
- Four schedules (fcpw4w16, fcpw8w16, fcpw4w8, fcpw8w8.bonsai): `sort` by
  `distmin(q, box)`, `vectorize` of the children and of the leaf,
  `loopify(96)` or `(154)`, `skip(distmin)` for the `all(active)` early
  exits after each region, no prefetch; the ray queries scheduled alike
  (FCPW's intersectFromNode is the same traversal with the ray's entry
  distance), with the prefetch left out. compare.sh compiles the driver
  for FCPW from the schedule's name (`fcpw<branching>w<width>` ->
  FCPW_USE_EIGHT_WIDE_BRANCHING, FCPW_SIMD_WIDTH, with the flags FCPW's
  own CMake sets: -march=native, -ffp-contract=fast, -fno-math-errno,
  NDEBUG), and against Embree otherwise; the driver's reference side is
  one API over the two (make_reference, build_tree, reference_intersect,
  reference_occluded, reference_closest), FCPW's queries through its
  public calls (findClosestPoint from a radius of the largest float, the
  query every user of it makes; intersect, with checkForOcclusion for the
  any hit). `--explain N` prints the first N disagreements of a row with
  both sides' answers and the distances compared, the rays' with the
  barycentrics of each hit in double.

What the languages cannot say of FCPW's schedules, and what is matched
in substance but not in mechanism:

1. (Settled.) The radius shrunk by the children's farthest distance: the
   user ruled it a pruning rule of the lowering's, not a schedule point
   -- an exact bound applies wherever it is sound -- and it is now emitted
   inside the children's loop and reduced across the lanes, as above.
2. The eight-wide order. FCPW's generic enqueueNodes pushes the children
   that passed in lane order and swaps the strictly nearest to the top:
   nearest first, the rest in stored order. `sort` orders all of them; a
   partial order is not a form the language has, and a key that ranks the
   nearest alone would compute the minimum over the lanes inside the
   key. The fcpw8* schedules sort, and say so.
3. The scalar Bvh's tie-break: its two children ordered by d2Min, and by
   d2Max where both d2Min are zero -- a lexicographic key, where `sort`
   takes one float. Not built here; FCPW's scalar tree is its slow path,
   and the user asked for its fast ones.
4. Mechanism, which the scheduling language abstracts over: FCPW's
   four-slot sorting network over scalar indices against our
   peel-and-insert chain over the hit lanes; its scalar scan of a packet's
   lanes with `<=` (the later lane and the later packet winning a tie)
   against our lane reduction; its root pushed untested. The arithmetic
   is FCPW's to the bit (the metrics file above); the order of visits is
   the same; the instructions around them are not FCPW's.

The compiler bug the conjunct exposed, and its fix. `if tMin <= tMax &&
proper` in the FCPW slab test, vectorized over the children, failed in
Build::make ("Build<Vector_t> of f32x3x4 by components takes 3 values of
type f32x4, but received t1 of type f32x3"): the children's box, built in
the traversal from the row's three `lo` casts, had two of its three
operands replaced -- `@12` by the callee's `t1`, `@20` by the callee's
`tMin`. Three facts made it. (1) A `let` of a fresh value renames the
value's instruction to the `let`'s name (`t1 = (b.high - r.o) * rinvD` is
one multiply, named `t1`; SSA/SSA.cpp Block::make_instruction), and the
block's table of names kept the old `@N` beside the new name -- a TODO in
the code said as much -- so `intersectsp_ray_aabb`'s table held `@12 ->
t1` and `@20 -> tMin` forever after. (2) The gang inliner copies a pure
straight-line variant into its caller (SSA/Vectorize.cpp
inline_pure_variants, SSA/Rewrite.cpp inline_call) and merges the copy's
table into the caller's block (absorb_successors), renaming the names the
callee defines and keeping every other key as it was -- `@12` and `@20`
among them, which in `_recloop_func2` named the caller's own casts of
the row's `lo`. (3) The simplifier's common-subexpression table takes
what an instruction's name denotes from the block's table
(SSA/Simplify.cpp remember), so when the children's second box built the
same loads again, the CSE handed out the callee's `t1` for the caller's
`@12` and the callee's `tMin` for its `@20`. Why the early-return form
compiled: the merge landed in a block whose instruction names did not
happen to coincide with the stale keys; the bug was in the numbering, not
the shape. The fixes: a rename drops the instruction's old `@N` from the
tables that held it (a program's own name stays, since `y = ...; x = y`
renames the value `x` and `y` still denotes it); a copied callee brings
only the names it defines, its instructions' and its arguments', and
nothing the callee's table held besides (a program name of its own for a
value, `v`, `d`, `t1`, would shadow the caller's); and absorbing a block
asserts that a name both tables hold denotes one instruction. Every
existing golden is byte-identical under the three. Tests:
ssa/inline-callee-names (the FCPW slab test with its named vectors and
its conjunct inlined into a vectorized traversal; the box built from the
row's casts, none of the callee's names in the function),
backends/llvm/rcp-approx, correctness/cpp/bvh4_fcpw_slabs (the FCPW
arithmetic, conjunct and enoki reciprocal over Embree's bytes, every ray's
answer the one the scene was built to give). What was left as it was: the
aggregate splitter's table entries for the vectors it takes apart, which
the stages after the split read and map through the components -- an
erasure there broke every vectorized traversal, so the names are kept
and the copy boundary is where they are dropped.

Also found, by `--explain`, before the arithmetic was matched: FCPW's ray
intersection misses hits. Its wide Moeller-Trumbore rejects a triangle
whose determinant is at most `epsilon` in absolute terms, with `epsilon`
the float machine epsilon (core/core.h:81), so small triangles in small
units are skipped as parallel: on head at side 512, 2 of 262k primary
rays and 3 of 45k diffuse rays hit a nearer triangle under Embree's
arithmetic that FCPW passed over, each inside the triangle by
double-precision barycentrics; on villa 201 of a million; and on lte-orb,
a mesh in small units, FCPW finds no hit at all -- which also emptied the
point batches there, built from the reference's primary hits (the near
points are to become surface samples, reference-independent; Known-open).
With FCPW's own test in the program, both sides miss alike and every ray
agrees. The closest point disagreed on ivy and dambreak by dozens to
hundreds of points, every one involving a degenerate triangle (zero
area) where the checker's own Ericson test went to NaN and could not say
which side was right -- a defect of the checker, to be made robust
(Known-open), not yet of either side.

What remains between FCPW and us after the arithmetic is matched
operation for operation, filed as precision-limited by the user's ruling
(2026-10-06). On ganesha under fcpw4w16, 11 of 1,048,576 primary rays and
2 of 279,885 ambient-occlusion rays differ, every any-hit answer the same,
and the pattern repeats on dragon, bmw, crown, ivy and villa (a handful of
rays each) and not at all on head, pavilion, zero-day, sportscar,
dambreak, landscape and lte-orb. Every ray `--explain` printed is a hit
within float noise of a triangle edge: barycentrics of `v = 3.97e-07`, `u
= -1.07e-06`, `u + v = 1.000076`, `v = 0.00056` on a mesh whose
coordinates sit near 300 and whose triangles are about 0.05 across, where
a float barycentric is good to about a part in a thousand -- each side
accepting an edge the other rejects, and the two hits being different
triangles, the checker counts them as disagreements rather than ties.
The transcription is FCPW's operation by operation (the slab test, the
Moeller-Trumbore chain, enoki's reciprocal, `cross` as enoki's
`fmsub(v1.yzx, v2.zxy, v1.zxy * v2.yzx)` and `dot` as its `hsum(a * b)`,
both of which our contraction rule fuses as clang's `-ffp-contract=fast`
does on paper), so what is left can only be the two compilers' fusion
choices in corners, and bit identity is not chased.

The measurements -- Embree before and after the layout's lane predicate
and the tightening, and the FCPW table over its four configurations --
are a later section's, taken in one exclusive block of the machine, after
the regression the next section describes was found and fixed.

## The reciprocal's sign, read from the step: the fold restored (2026-10-06)

The morning's Embree table came in 2-8% below the night's on every
nearest-hit cell (ganesha under `embree` 25.77 -> 23.79 Mrays/s, crown
109.0 -> 101.6, pavilion 79.98 -> 74.90; the any-hit and closest-point
cells level; Embree's own rates steady within 1%), with nothing between
the two but a5f00468 -- `rcp_approx` as the one reciprocal primitive and
each library's Newton step written out in its metric file. The modules
said where: the five mirrored node-test limits (SSA/MirrorLoads.h,
Embree's `tray.tfar`) had gone to none, on all four Embree schedules.

The chain. The simplifier compares two floats on their bits when the left
is sign-clear and the right a non-negative number (Simplify.cpp's `<` and
`<=` rule), and when the slab test's `tNear <= tFar` and the argmin's
`tNear < best.key` are both on bits, the pair joins into one compare
against `min(bits(tFar), bits(best) - 1)`, whose limit MirrorLoads keeps
in a register across the traversal. The best's key is `t = T *
rcp_embree(absDen)`. With `rcp` a primitive, its interval rule said "a
non-negative number where the operand is"; with the step written out, `r
+ r (1 - a r)`, the interval analysis followed it as the arithmetic it is
and derived nothing -- not even a sign -- so the key was not known
non-negative, the compare stayed a float compare, the pair did not join,
and the limit was not there to mirror. Everything downstream of the leaf
was unchanged; the 2-8% is the node test's limit re-read from memory at
every node, which is the thing MirrorLoads exists to avoid.

Why the arithmetic cannot give the sign, so that the rule is understood
and not just added: the estimate `r = rcp_approx(x)` is +inf for a tiny
`x` (the instruction takes it as zero), and there `1 - x r` is -inf and `r
(1 - x r) + r` is inf - inf. Intervals read the two infinities as
unrelated and see a possible negative; the step is in fact a NaN there.
The information lost is a correlation (the two infinities are one), which
a non-relational lattice does not carry. Nor is this a matter of
reassociation that a simplifier could undo: the factored `r (2 - x r)` is
-inf at that `x`, a negative number where the written step is a NaN, so
the spelling is part of the fact. The sign survives rounding (round to
nearest keeps the sign of the exact result, an fma rounds once), so it is
not associativity that stands in the way.

Three ways were weighed. Relaxing the fold's precondition to "non-negative
where it is a number" finds nothing to relax: the analysis had no sign
fact for the key at all. A compositional analysis in three sound pieces --
the estimate's specification as a product rule (`x rcp_approx(x)` within
2^-10 of one wherever both are normal), a shared-factor sign rule for
sums of multiples of one value (`fma(r, h, r)` is `r (1 + h)` for its
sign wherever `r` is finite, and an infinite `r` makes an infinity or a
NaN, never a finite negative), and a backward refinement through the
estimate ("the estimate is finite" meaning "the operand is not tiny", as
`numbers_beneath` turns "a number" into "numbers") -- derives the sign
for every spelling of one step without naming any, at about twice the
code; it does not reach a second Newton step either, `x r1` for a refined
`r1` being a relation intervals cannot hold. The user's choice was the
minimal rule: the step recognized as the shape it is.

The rule (SSA/ConstantIntervals.cpp, `reciprocal_step` and `newton_step`):
an Add, Sub or fma that is the last operation of one Newton step over an
estimate -- Embree's `r + r (1 - x r)` and enoki's `2 r - (r x) r`, each
as written and as `--ffp-contract` fuses it, `fma(r, fma(-x, r, 1), r)`
and `fma(-(r x), r, r + r)`, operands either way round, `-v` the
contraction's `sub(-0, v)`, the one `r` and `x` throughout, compared
through Sets -- is read as the estimate's own bounds and sign and not
known to be a number (where the estimate is infinite the step is a NaN).
That one fact restores the chain: `t` is `(+)`, the argmin accepts it
only under `t < best.key`, which marks it a number (a held `<` is between
numbers), the stored best is `[0, inf] n +`, the compare goes to bits,
the pair joins, the limit is mirrored. Five mirrors again on all four
Embree schedules; the FCPW modules unchanged (their node test's limit
includes the ray's `tfar` joined with slab distances that may be NaNs, so
the fold never applied there). Found and fixed on the way: the estimate
rule's bounds were unsound at the ends of the normal range, giving a
finite upper bound for an operand below the smallest normal (where
`rcpps` returns +inf) and a positive lower bound for an operand above
2^126 (where `rcpps` returns zero); `vrcp14ps` differs (only below 2^-128
is taken as zero; a denormal quotient is returned, not flushed), and the
bounds now hold on both. Tests: ssa/intervals-from-conditions (`refined`
and `refined_enoki`, the spellings as written, `(+)` where they were
everything), ssa/intervals-reciprocal-step (both spellings, a three-vector
with broadcast constants, a bounded operand, a negative operand without
the sign, the factored spelling and a step over a different operand both
left as everything), correctness/llvm/rcp-step-edges (the step on the
host at a tiny, a huge, a zero, an infinite and ordinary operands, in
both spellings, asserting what holds on both instructions).

Tried and dropped: a common-subexpression table that keys operands through
the Sets that name them, which deduplicated a function inlined twice with
its parameter bound to the same argument. Sound, and worth six lines of
the four-wide modules, but it cost the goldens their program names (`ps.pdf`
printed as `@7`) and in one sort-key golden made the second inlined copy
match less rather than more (the key's canonical form depends on when
the Set was seen); the fold did not need it. Not committed; the patch is
in the session's scratchpad.

Found on the way and fixed at the user's direction (2026-10-06): the
front-end simplifier's zero rules fired on floats under the exact
arithmetic that is the default. `x * 0` folded to `0` (a NaN for an
infinite or NaN `x`, `-0.0` for a negative one), `0 + x` and `x + 0` to
`x` (`+0.0` for `x = -0.0`), `0 - x` to `-x` (`+0.0` where `-x` is
`-0.0`) and `x / x` to `1` (a NaN at zero, at an infinity and at a NaN);
a negated constant was folded as `0 - c`, which made the literal `-0.0`
into `+0.0`; and a float divided by a constant zero tripped the integer
check instead of folding to IEEE's answer. Now the zero rules fire on a
float only under `--fast-math` (CompilerOptions::fast_math: no NaNs, no
infinities, the sign of a zero not observed -- the assumption each rule
makes, and what clang's -ffast-math promises LLVM); `x - 0.0` stays
folded, exact for every float when the zero is positive; a negated
constant is negated; the division folds (Opt/Simplify.cpp,
`zero_folds`). Tests: opt/float-zero-folds (the rules left standing,
`-0f` kept) and opt/float-zero-folds-fast-math (every one folded),
backends/llvm/float-zero-folds (the operations reach LLVM as written, with
no flags for it to fold them by), correctness/llvm/float-zero-folds (the
NaN and signed-zero answers on the host, sixteen of them). Three goldens
that spell a negation `0.0 - x` now read `0f - x`, and
intervals-from-conditions' `lo = abs(a) * 0.0` stays a multiply, `[0, 0]
(+)`, every fact the test documents unchanged. The rcp-step-edges test's
`[[noinline]]` wrappers, added to keep its operands out of the fold's
reach, stay: they also keep LLVM from folding the constant calls. rtq's
modules are unchanged by it: the Embree and FCPW four-wide modules compile
to the same instructions before and after, attribute lines aside, so the
Embree half of the table below stands for this commit too.

## The table after the fold's return: Embree on its two trees, FCPW on its four, in one CSV (2026-10-06)

The measurement the FCPW section and the one before it point at. Two
exclusive blocks of the machine, one thread on both sides pinned to cpu 11,
least of five repeats, every other session holding: Embree's four schedules
over its two trees from 15:26 to 15:55 (the compiler at 9499ccbb, HEAD at
the time, e672abee's fold restored; side 2048 for the rays, 1024 for the
points), and FCPW's four configurations from 17:22 to 17:38 (the compiler
at e35c7b02, whose only change since is the zero-fold fix above, which
leaves rtq's modules the same instructions; side 1024 throughout, FCPW's
single-thread rate being a third of Embree's so that a million rays already
run well over the tenth of a second the larger side was chosen for). The
block was cut at the Embree/FCPW boundary for another session's benchmark
and the FCPW half taken later, from a clean worktree, the live tree being
mid-edit by then.

Three of the user's rules, from this run on. The numbers live in one
file, rebuilt from the run's logs each time and holding only fresh data:
`build-rtq2/results/rtq-results.csv` (git-ignored under `build-*`), one row
per measured cell -- date, commit, machine, cpu, repeats, reference,
schedule, width, mesh, triangles, nodes, leaves, query (nearest, any,
closest), ray set (primary, ao, diffuse, near, volume), side, count, the
reference's rate, ours, the speedup, same, ties, differ, both sides' hit
counts, Embree's callbacks per query, units -- so that plots are made from
it without rerunning the matrix. Meshes are ordered by triangle count,
pavilion (11k) to dragon (7.2M). And only the matching schedules are
measured from here on, to save benchmarking time; the tuned cells of this
run are in the CSV and not in the tables below.

Embree, bonsai's rate over Embree's, the cells a renderer asks for
(nearest hit of primary and diffuse rays, any hit of ao rays); every ray
agrees with Embree up to ties on every cell:

| mesh | triangles | embree4: nearest pri dif, any ao | embree: nearest pri dif, any ao |
|---|---|---|---|
| pavilion | 11366 | 1.08 0.96, 1.00 | 0.99 0.99, 1.01 |
| head | 17674 | 1.06 0.97, 0.98 | 1.05 1.01, 0.99 |
| zero-day | 48960 | 1.14 0.98, 1.00 | 1.10 1.03, 1.02 |
| bmw | 110592 | 1.03 0.99, 1.02 | 1.04 1.03, 1.02 |
| crown | 155520 | 1.08 0.99, 1.02 | 1.04 1.03, 1.02 |
| ivy | 179603 | 1.08 1.01, 1.00 | 1.03 1.02, 0.99 |
| villa | 390784 | 1.02 1.00, 1.01 | 1.03 1.04, 1.00 |
| dambreak | 1015024 | 1.05 1.01, 1.01 | 1.04 1.04, 1.00 |
| sportscar | 1091232 | 1.05 1.01, 1.02 | 1.05 1.05, 1.03 |
| landscape | 1916928 | 1.12 1.17, 1.19 | 1.10 1.19, 1.11 |
| lte-orb | 3423232 | 1.07 1.02, 1.01 | 1.06 1.02, 0.99 |
| ganesha | 4323658 | 1.04 1.01, 1.01 | 1.06 1.04, 0.99 |
| dragon | 7219045 | 1.07 1.02, 1.01 | 1.06 1.02, 0.99 |

The controls (nearest hit of ao rays; any hit of primary and diffuse
rays), which follow the same pattern:

| mesh | embree4: nearest ao; any pri dif | embree: nearest ao; any pri dif |
|---|---|---|
| pavilion | 0.96; 1.11 1.00 | 1.00; 1.10 1.00 |
| head | 0.97; 1.04 0.98 | 1.01; 1.11 0.99 |
| zero-day | 0.99; 1.20 1.00 | 1.03; 1.17 1.01 |
| bmw | 0.98; 1.06 1.03 | 1.04; 1.06 1.01 |
| crown | 1.00; 1.11 1.01 | 1.04; 1.11 1.01 |
| ivy | 1.01; 1.09 1.00 | 1.01; 1.04 0.98 |
| villa | 0.99; 1.06 1.02 | 1.03; 1.04 1.00 |
| dambreak | 1.01; 1.04 1.01 | 1.03; 1.03 1.01 |
| sportscar | 1.00; 1.05 1.01 | 1.06; 1.05 1.02 |
| landscape | 1.18; 1.13 1.19 | 1.19; 1.11 1.07 |
| lte-orb | 1.01; 1.07 1.01 | 1.03; 1.06 0.99 |
| ganesha | 1.01; 1.03 1.00 | 1.04; 1.03 0.99 |
| dragon | 1.02; 1.03 1.00 | 1.03; 1.02 0.99 |

The closest point against rtcPointQuery with the tutorial's callback,
Embree's arithmetic with the root as the callback takes it (the earlier
committed closest-point numbers, taken with a metric that took no root,
are superseded):

| mesh | embree4: near vol | embree: near vol |
|---|---|---|
| pavilion | 3.03 4.11 | 3.00 4.19 |
| head | 3.67 4.14 | 3.78 4.30 |
| zero-day | 3.50 3.72 | 3.72 4.01 |
| bmw | 3.89 4.31 | 4.02 4.53 |
| crown | 4.09 4.20 | 4.29 4.51 |
| ivy | 5.01 4.52 | 4.91 4.36 |
| villa | 4.07 4.59 | 4.31 4.87 |
| dambreak | 3.83 5.37 | 3.78 5.83 |
| sportscar | 4.45 5.00 | 4.62 5.51 |
| landscape | 3.36 2.93 | 3.43 3.12 |
| lte-orb | 4.46 4.81 | 4.70 5.37 |
| ganesha | 5.45 7.20 | 5.71 8.21 |
| dragon | 5.52 6.83 | 6.16 7.90 |

| schedule | nearest, primary | nearest, diffuse | any, ao | closest, near | closest, volume | cells under 1.00 |
|---|---|---|---|---|---|---|
| embree4 | 1.02-1.14 | 0.96-1.17 | 0.98-1.19 | 3.03-5.52 | 2.93-7.20 | pavilion diffuse 0.96; head diffuse 0.97, ao 0.98; zero-day, bmw, crown diffuse 0.98-0.99 |
| embree | 0.99-1.10 | 0.99-1.19 | 0.99-1.11 | 3.00-6.16 | 3.12-8.21 | pavilion primary and diffuse 0.99; any ao 0.99 on head, ivy, lte-orb, ganesha, dragon |

Reading it against the night's table (the section at 9440bea3). These
cells sit two to five points under it on both widths, with Embree's own
rate steady within about a percent on every cell, and the history of our
own rate on the matching schedules across the four runs since then says
where: the layout's lane predicate (`where` on Embree's Triangle4 tiles,
one vector compare and an and per block that Embree does not pay, its zero
triangle failing the hit test) cost 0.4-3.5% on the four-wide nearest hit
of primary rays, 0.2-5% on the eight-wide, and 1-3% on the any hit of ao
rays on both widths, measured the night the predicate went in with the
fold still in place; the reciprocal written out then cost 1-8% more on the
nearest hit (the fold lost, the section above); and the fold's return
gave 1-6% of that back (ganesha eight-wide 23.79 -> 25.21 Mrays/s against
25.77 before the regression, crown 101.6 -> 103.5 against 109.0). The any
hit's cells did not move through the last two steps, as they should not
have: the fold is the nearest hit's. What did not come back is the
eight-wide nearest hit on pavilion (79.98 before the reciprocal change,
74.90 after it, 74.87 now, Embree steady at 75.5) and most of crown's and
villa's, 3-6% below their pre-change rates with the fold demonstrably in
place (five mirrors in the module). Something besides the fold changed in
that module between a0b863cf and HEAD -- the step's own code against the
intrinsic's lowering, the estimate rule's wider bounds, or the compiler
commits beside it -- and the probe is set up: a worktree at a0b863cf
built beside this one, the two `embree` modules diffed with attribute
noise removed, and the two meshes A/B'd under the table's settings; it
waits for a machine slot and is the next thing measured.

FCPW, bonsai's rate over FCPW's, on FCPW's bytes with FCPW's arithmetic
(the metrics file, operation for operation), four configurations named
`fcpw<branching>w<leaf width>`:

| mesh | fcpw4w16: nearest pri dif, any ao | fcpw8w16 | fcpw4w8 | fcpw8w8 |
|---|---|---|---|---|
| pavilion | 1.85 1.95, 2.06 | 2.11 1.98, 2.19 | 2.01 1.93, 2.07 | 2.11 1.90, 2.09 |
| head | 1.79 1.83, 1.98 | 1.86 1.78, 1.87 | 1.75 1.82, 1.99 | 1.92 1.77, 1.86 |
| zero-day | 2.49 1.95, 2.07 | 3.48 2.06, 2.24 | 2.40 1.83, 2.01 | 3.01 1.95, 2.09 |
| bmw | 1.81 1.80, 2.10 | 1.84 1.78, 2.03 | 1.75 1.78, 2.09 | 1.83 1.78, 2.04 |
| crown | 2.34 1.88, 1.99 | 2.49 1.97, 2.10 | 2.02 1.78, 1.93 | 2.25 1.92, 2.01 |
| ivy | 1.81 1.64, 1.70 | 1.87 1.70, 1.75 | 1.78 1.59, 1.69 | 2.06 1.70, 1.71 |
| villa | 1.71 1.80, 1.98 | 1.73 1.87, 1.95 | 1.68 1.77, 1.94 | 1.70 1.84, 1.90 |
| dambreak | 1.66 1.53, 1.80 | 1.66 1.64, 1.79 | 1.64 1.50, 1.75 | 1.59 1.59, 1.71 |
| sportscar | 1.62 1.64, 1.82 | 1.72 1.69, 1.79 | 1.61 1.62, 1.83 | 1.63 1.68, 1.81 |
| landscape | 1.43 2.25, 2.49 | 1.50 2.31, 2.62 | 1.35 2.24, 2.48 | 1.36 2.18, 2.45 |
| lte-orb | 1.57 -, - | 1.59 -, - | 1.48 -, - | 1.50 -, - |
| ganesha | 1.46 1.54, 1.62 | 1.45 1.63, 1.66 | 1.40 1.46, 1.54 | 1.37 1.53, 1.56 |
| dragon | 1.39 1.41, 1.45 | 1.34 1.50, 1.51 | 1.36 1.37, 1.39 | 1.39 1.45, 1.48 |

| mesh | fcpw4w16: nearest ao; any pri dif | fcpw8w16 | fcpw4w8 | fcpw8w8 |
|---|---|---|---|---|
| pavilion | 1.95; 2.29 2.06 | 2.01; 2.76 2.17 | 1.94; 2.46 2.06 | 1.92; 2.69 2.07 |
| head | 1.86; 2.14 1.99 | 1.77; 2.33 1.89 | 1.86; 2.06 2.00 | 1.76; 2.38 1.87 |
| zero-day | 1.96; 2.84 2.10 | 2.06; 3.97 2.24 | 1.85; 2.72 2.03 | 1.97; 3.29 2.11 |
| bmw | 1.92; 2.04 2.23 | 1.86; 2.13 2.16 | 1.89; 1.99 2.22 | 1.89; 2.08 2.14 |
| crown | 1.89; 2.59 1.98 | 1.99; 2.83 2.10 | 1.80; 2.24 1.93 | 1.92; 2.54 2.01 |
| ivy | 1.66; 1.97 1.67 | 1.71; 2.07 1.71 | 1.62; 1.96 1.65 | 1.71; 2.24 1.67 |
| villa | 1.87; 1.93 1.99 | 1.86; 2.04 1.92 | 1.83; 1.92 1.97 | 1.82; 1.96 1.88 |
| dambreak | 1.65; 1.80 1.74 | 1.77; 1.83 1.72 | 1.61; 1.79 1.69 | 1.67; 1.74 1.63 |
| sportscar | 1.74; 1.75 1.81 | 1.74; 1.89 1.80 | 1.70; 1.73 1.81 | 1.73; 1.75 1.75 |
| landscape | 2.25; 1.63 2.49 | 2.31; 1.90 2.61 | 2.24; 1.52 2.47 | 2.19; 1.81 2.45 |
| lte-orb | -; 1.59 - | -; 1.67 - | -; 1.49 - | -; 1.57 - |
| ganesha | 1.56; 1.57 1.52 | 1.63; 1.59 1.55 | 1.47; 1.50 1.46 | 1.51; 1.47 1.46 |
| dragon | 1.44; 1.46 1.44 | 1.48; 1.47 1.52 | 1.38; 1.40 1.40 | 1.43; 1.43 1.48 |

| mesh | fcpw4w16: near vol | fcpw8w16 | fcpw4w8 | fcpw8w8 |
|---|---|---|---|---|
| pavilion | 1.20 1.48 | 1.22 1.64 | 1.28 1.42 | 1.20 1.59 |
| head | 1.40 1.53 | 1.39 1.72 | 1.35 1.45 | 1.39 1.69 |
| zero-day | 1.46 1.40 | 1.57 1.57 | 1.37 1.28 | 1.50 1.45 |
| bmw | 1.42 1.40 | 1.50 1.63 | 1.38 1.31 | 1.48 1.55 |
| crown | 1.51 1.38 | 1.66 1.59 | 1.43 1.30 | 1.59 1.48 |
| ivy | 1.49 1.44 | 1.68 1.58 | 1.40 1.34 | 1.60 1.50 |
| villa | 1.45 1.40 | 1.54 1.62 | 1.39 1.29 | 1.51 1.48 |
| dambreak | 1.42 1.23 | 1.55 1.38 | 1.36 1.18 | 1.51 1.34 |
| sportscar | 1.41 1.21 | 1.51 1.32 | 1.31 1.12 | 1.42 1.19 |
| landscape | 1.35 1.19 | 1.42 1.30 | 1.25 1.23 | 1.32 1.33 |
| lte-orb | - - | - - | - - | - - |
| ganesha | 1.39 1.01 | 1.53 1.04 | 1.27 0.94 | 1.40 0.98 |
| dragon | 1.29 0.97 | 1.38 1.01 | 1.18 0.93 | 1.24 0.97 |

| schedule | nearest, primary | nearest, diffuse | any, ao | closest, near | closest, volume | cells under 1.00 |
|---|---|---|---|---|---|---|
| fcpw4w16 | 1.39-2.49 | 1.41-2.25 | 1.45-2.49 | 1.20-1.51 | 0.97-1.53 | dragon volume 0.97 |
| fcpw8w16 | 1.34-3.48 | 1.50-2.31 | 1.51-2.62 | 1.22-1.68 | 1.01-1.72 | none |
| fcpw4w8 | 1.35-2.40 | 1.37-2.24 | 1.39-2.48 | 1.18-1.43 | 0.93-1.45 | ganesha volume 0.94, dragon volume 0.93 |
| fcpw8w8 | 1.36-3.01 | 1.45-2.18 | 1.48-2.45 | 1.20-1.60 | 0.97-1.69 | ganesha volume 0.98, dragon volume 0.97 |

Reading it. The ray queries are 1.3-3.5x FCPW on every cell, the any hit
and the nearest hit alike, the small meshes most (zero-day's eight-wide
primary rays 3.48x: FCPW's own rate drops from 85 to 63 Mrays/s between
its four- and eight-wide trees there while ours holds) and the largest
meshes least (dragon 1.34-1.52x). The closest point is 1.2-1.7x on the near batch
everywhere and 0.93-1.7x on the volume batch, under FCPW only on the two
largest meshes, ganesha and dragon, where a volume query's traversal is
longest; there the two known-open items of the closest point's lowering
are the suspects -- the sort key's sign flip (a squared distance the
interval analysis does not know non-negative, four instructions a node)
and FCPW's eight-wide partial order, which the fcpw8* schedules replace
with a full sort -- and FCPW's leaf, a wide Ericson test over sixteen
lanes with `all(active)` exits, is the shape our `skip` reproduces. The
night before, a quick check on ganesha at side 1024 with the early-return
spelling of the FCPW slab test (the conjunct form could not be compiled
until the inliner bug above was fixed) and the compiler's own `rcp` read
1.04-1.20x on the nearest hit and 1.45-1.65x on the any hit; the committed
form reads 1.37-1.66x there with FCPW's rates the same, so the nearest hit
gained 40% between the two spellings and the reciprocal work, the any hit
nothing. lte-orb has no ao, diffuse or point cells against FCPW: FCPW's
triangle test, with its absolute epsilon, finds no hit at all on that
small-unit mesh (the FCPW section above), so the batches built from the
reference's primary hits are empty; its primary cells compare a miss with
a miss on both sides, agreeing on every ray. Disagreements: on the rays at
most 30 of a million per mesh and none on five of the thirteen, the
precision-limited edge hits of the user's ruling; on the points none but
on ivy (109-261) and dambreak (634-755), every one through a degenerate
triangle the checker cannot classify (known-open, the checker's to fix).

## The cells under the reference, profiled: not pruning, not the layout, the leaf's code (2026-10-06, evening)

The user's ask: profile every cell more than 2% under its reference and
say whether it is a pruning problem, a code generation problem, or a
schedule or layout mismatch. The cells, from the table above: the FCPW
closest point's volume batch on ganesha and dragon under three of the four
configurations (0.93-0.98x), and Embree's four-wide nearest hit of diffuse
rays on pavilion and head (0.96-0.97x). Method: perf on one cell at a time
(cycles and instructions per symbol, both sides in one process, pinned to
cpu 11; `prof.sh` and `build-out.sh` in the session's scratchpad), the two
traversals read side by side, the generated kernel read in its SSA and
LLVM forms.

**FCPW closest point, volume batch.** Pruning is matched: both sides push
the child's box distance with its stack entry and drop a popped node whose
distance is past the current best, test the children's boxes at once
against it, and lower the best to the least of the pushed children's
farthest box distance; every point agrees. The layout is FCPW's bytes. The
perf split on ganesha under fcpw4w8, the same queries on both sides:

| side | share of cycles | share of instructions | relative IPC |
|---|---|---|---|
| ours, `closest_all` | 43.2% | 23.8% | 0.55 |
| FCPW, traversal + enqueue | 40.7% | 29.5% | 0.73 |

Fewer instructions, more cycles: a stall-bound kernel, and the leaf was
why, in three ways, all code generation.

1. *Seven roots a packet where FCPW takes one.* findClosestPointWideTriangle
   selects the closest point per Voronoi region under a mask and takes one
   `norm(x - pt)`; the metric as first transcribed returned `sqrt(dot) *
   sqrt(dot)` from each of the seven regions -- FCPW's values per lane, but
   after vectorization seven `vsqrtps` a packet on the divider unit, which
   retires one every several cycles. metrics/fcpw.bonsai now has FCPW's
   shape: offset_fcpw answers `x - pt` by region with the early returns,
   distmin takes the one root and squares it.
2. *No early exits, because `closest.skip(distmin)` did nothing.* The SSA
   showed the leaf as one straight block. expand_branch_policies
   (SSA/Convert.cpp) carried a cursor into the query's extracted recursion
   only when its name was a function of the module, and a geometric op's
   implementations are renamed per operand type by Lower/Geometrics.cpp
   (`distmin_Point_Triangle`) while the arms' provenance keeps `distmin`,
   the name the schedule uses and ir::skips matches by -- so the check
   passed and the policy went nowhere. A cursor now also counts when some
   arm of the module was written in a function of that name; the four fcpw
   schedules skip on offset_fcpw; ssa/skip-geometric-op shows the
   `reduce<any>` gadgets in a `distmin` leaf. Embree's
   `trace.skip(intersectsp_ray_tri)` never had the problem: that helper is
   not overloaded.
3. *Negation spelled `0 - x`.* SSA/Convert.cpp subtracted a float from
   positive zero, which is `+0.0` at `x = +0.0` (a third place for the
   signed-zero bug the simplifier's folds had) and a subtraction LLVM can
   only keep: `-(a - b)` came out as `(b - a) + 0.0`, three adds of zero a
   packet. It subtracts from `-0.0` now, `-x` for every x, `fneg` in LLVM
   and `neg.f32` in PTX; 52 goldens re-blessed, every change that spelling.
   (The sort key's sign flip, the fourth item read off the kernel, is the
   known-open interval fact and stands.)

Measured after the three (fcpw4w16, side 1024, least of 5; the full table
at the new commit follows below): every closest-point cell is over FCPW,
ganesha's volume batch 1.01 -> 1.16x, dragon's 0.97 -> 1.13x, the w8
configurations 0.93-0.98 -> 1.05-1.19x -- and the near batches gave back
5-20% (head 1.40 -> 1.14x). Separating the two leaf changes on head and
ganesha under fcpw4w16 (near, volume):

| leaf | head | ganesha |
|---|---|---|
| FCPW's shape with its exits (the matching schedule) | 1.14, 1.19 | 1.28, 1.16 |
| FCPW's shape, no exits | 1.43, 1.57 | 1.38, 1.05 |
| the seven-root leaf with exits | 1.09, 1.13 | 1.23, 1.12 |
| the seven-root leaf, no exits (the table's state) | 1.39, 1.53 | 1.40, 1.01 |

The one root is a small gain; the exits are the swing, and the counters say
why: with them our kernel takes 11.5M branch mispredictions on head's near
batch against FCPW's own 11.8M on the same queries -- the same wager FCPW
makes, at the same price -- where the straight-line leaf took 5.1M. In
cycles our kernel is 1.41x FCPW's traversal without the exits and 1.24x
with them; the exits pay only where whole packets settle early, which is
the volume batch on the large meshes. The user's ruling: the matching
schedules keep FCPW's exits, since the two sides must compute the same
thing under the same schedule; the departures go into tuned schedules, as
with Embree (the next section).

**Embree four-wide, pavilion and head, diffuse rays.** The perf split for
these cells is polluted by the driver's setup, which traces every primary
ray through Embree to build the diffuse batch, so Embree's symbol carries
that too. What the history says: these cells stood at 0.99-1.00 the night
before, the layout's lane predicate took two to three points (measured the
night it went in), and the eight-wide residual below took one or two.

**The eight-wide residual, A/B'd.** The compiler just before the reciprocal
change (a0b863cf, `rcp` a primitive) against the current one, alternated
twice so the day's drift fell on both:

| mesh, rays | pre-change | current |
|---|---|---|
| pavilion primary | 79.5, 78.0 | 78.5, 78.3 |
| pavilion diffuse | 27.9, 27.6 | 27.4, 27.3 |
| crown primary | 106.8, 108.3 | 105.2, 107.1 |
| crown diffuse | 11.57, 11.56 | 11.17, 11.29 |

One to three percent, real on the diffuse rays, not the six the two sweeps
suggested (the rest was drift). The two `embree` modules differ in ten
instructions, all in the per-ray reciprocal of the direction: `rcp_embree_vec3`
over a `vec3f` now lowers its Newton step as a pair of `<3 x float>` fmas
where the intrinsic worked on four lanes. Whether that alone costs 3% on
diffuse rays, or the machine code around it moved, is for the assembly
diff; it is small enough to leave for a quiet moment.

## Tuned schedules over FCPW's trees, and the plots (2026-10-06, night)

The user's direction, after the leaf fixes: the matching FCPW schedules
keep FCPW's exits, since the two sides must compute the same thing under
the same schedule; the departures that measure better go into tuned
schedules, as tuned.bonsai and tuned4.bonsai are to Embree's. So
schedules/fcpw4w16t, fcpw8w16t, fcpw4w8t and fcpw8w8t (compare.sh reads the
`t` as the tuned schedule over the same tree, and builds FCPW's tree of
that shape as for the matching one), each the matching schedule without
`closest.skip(offset_fcpw)`: the leaf's regions as straight-line masked
arithmetic, no branch to mispredict. Three departures were tried on head,
ganesha and dragon (side 1024, least of 5), each over the two sixteen-lane
trees:

| departure | result |
|---|---|
| the closest point's exits dropped | taken: head near 1.13 -> 1.36x and volume 1.18 -> 1.50x (four-wide), 1.13 -> 1.35x and 1.33 -> 1.68x (eight-wide); ganesha near 1.28 -> 1.39x / 1.42 -> 1.52x but volume 1.16 -> 1.05x / 1.26 -> 1.07x; dragon near 1.24 -> 1.30x / 1.36 -> 1.39x, volume 1.13 -> 1.01x / 1.25 -> 1.05x; the ray queries within a point |
| the children prefetched (Embree's directive) | not yet possible: the prefetch lowering fetches an arm that is a row of a group or a range of tiles, and FCPW's Interior is an arm of the node row's switch on its first slot's sign, with no storage of its own to name (Lower/Prefetches.cpp refuses it). The row is what to fetch; a lowering extension, known-open |
| the ray test's early exits (`trace.skip(intersectsp_ray_tri)`) | untried for the same reason, having been bundled with the prefetch in the probe; to be measured on its own |

The exits pay where whole packets settle early, the volume batch on the
largest meshes, and cost everywhere else; the tuned schedule takes the
larger side. The table below (the eight-wide trees at this commit) has
both schedules per tree, and the plots are drawn from it.

The plots (apps/rtq/plot.py, from the results CSV): three figures over
the thirteen meshes in order of triangle count -- the first hit and the
any hit with six bars a mesh, the three ray sets by hatching and the two
references by colour, and the closest point with four, the near and
volume batches by hatching -- the eight-wide trees only (Embree's BVH8,
FCPW's eight-wide Mbvh with sixteen-lane leaves), bonsai's rate over the
reference's with the line at 1 the reference; `--tuned` draws the same
from the tuned schedules. The colours are the paper's (graphs.py at the
root), the Okabe-Ito colour-blind palette: sky blue for Embree, orange for
FCPW. A cell the reference has no number for (FCPW finds no hit on
lte-orb, so its ao, diffuse and point batches are empty) leaves its slot
empty. The figures land beside the CSV, under build-rtq2/results/plots.

## The eight-wide table at 95544783: matching and tuned, both references (2026-10-06, night)

One exclusive block, 21:41-22:06, the compiler at 95544783 (the three
fixes above) from a clean worktree; one thread each on cpu 11, least of
five; Embree's BVH8 under `embree` and `tuned` (rays and any hit at side
2048, the closest point at 1024) and FCPW's eight-wide Mbvh at both leaf
widths under its matching and tuned schedules (side 1024, all queries).
The CSV holds this run alone (600 rows); the four-wide trees were left
out tonight by the user's direction and are the next block's. Every ray
and point agrees with Embree up to ties; against FCPW the known residue
stands (rays on at most 30 of a million per mesh, points on ivy and
dambreak through degenerate triangles).

Embree, bonsai's rate over Embree's (nearest hit of primary and diffuse
rays, any hit of ao rays; then the closest point's two batches):

| mesh | triangles | embree: nearest pri dif, any ao | tuned: nearest pri dif, any ao | embree: near vol | tuned: near vol |
|---|---|---|---|---|---|
| pavilion | 11366 | 1.05 1.03, 1.00 | 1.02 1.04, 1.02 | 3.04 4.25 | 3.04 4.26 |
| head | 17674 | 1.04 1.01, 0.99 | 1.03 1.11, 1.11 | 3.86 4.36 | 3.85 4.35 |
| zero-day | 48960 | 1.10 1.03, 1.01 | 1.13 1.16, 1.15 | 3.74 4.06 | 3.74 4.07 |
| bmw | 110592 | 1.05 1.04, 1.02 | 1.05 1.11, 1.13 | 4.05 4.56 | 4.04 4.55 |
| crown | 155520 | 1.10 1.06, 1.01 | 1.09 1.20, 1.17 | 4.40 4.61 | 4.38 4.56 |
| ivy | 179603 | 1.06 1.03, 0.98 | 1.05 1.16, 1.11 | 5.03 4.46 | 5.03 4.46 |
| villa | 390784 | 1.04 1.05, 1.01 | 1.04 1.10, 1.10 | 4.33 4.87 | 4.33 4.90 |
| dambreak | 1015024 | 1.03 1.01, 0.99 | 1.04 1.07, 1.07 | 3.78 5.77 | 3.78 5.76 |
| sportscar | 1091232 | 1.04 1.03, 1.04 | 1.05 1.13, 1.13 | 4.64 5.48 | 4.66 5.52 |
| landscape | 1916928 | 1.05 1.15, 1.13 | 1.09 1.22, 1.16 | 3.47 3.14 | 3.45 3.13 |
| lte-orb | 3423232 | 1.07 1.02, 1.00 | 1.06 1.08, 1.06 | 4.70 5.42 | 4.70 5.45 |
| ganesha | 4323658 | 1.05 1.03, 0.99 | 1.04 1.08, 1.06 | 5.65 8.24 | 5.68 8.18 |
| dragon | 7219045 | 1.05 1.02, 0.99 | 1.05 1.08, 1.06 | 6.07 7.77 | 6.06 7.83 |

| schedule | nearest, primary | nearest, diffuse | any, ao | closest, near | closest, volume | cells under 1.00 |
|---|---|---|---|---|---|---|
| embree | 1.03-1.10 | 1.01-1.15 | 0.98-1.13 | 3.04-6.07 | 3.14-8.24 | ivy ao 0.98; ao 0.99 on head, dambreak, ganesha, dragon |
| tuned | 1.02-1.13 | 1.04-1.22 | 1.02-1.17 | 3.04-6.06 | 3.13-8.18 | none |

The matching schedule's any hit of ao rays is the one column still level
with Embree, as it has been since the arms-by-count work; the nearest
hit of primary rays stands 3-10% ahead on every mesh, where this
afternoon's table (the regressed compiler, then the fold alone) had it at
0.99-1.10.

FCPW, bonsai's rate over FCPW's, the eight-wide tree at both leaf widths:

| mesh | fcpw8w16: nearest pri dif, any ao | fcpw8w16t | fcpw8w8 | fcpw8w8t |
|---|---|---|---|---|
| pavilion | 1.97 1.96, 2.13 | 1.97 1.93, 2.11 | 1.94 1.84, 2.01 | 1.94 1.84, 2.01 |
| head | 2.01 1.86, 1.96 | 2.01 1.84, 1.96 | 2.00 1.80, 1.90 | 2.01 1.81, 1.91 |
| zero-day | 3.36 2.07, 2.25 | 3.54 2.06, 2.26 | 3.53 1.99, 2.18 | 3.16 1.97, 2.11 |
| bmw | 1.84 1.80, 2.06 | 1.85 1.81, 2.04 | 1.83 1.81, 2.06 | 1.91 1.83, 2.12 |
| crown | 2.30 1.98, 2.09 | 2.29 1.96, 2.08 | 2.32 1.93, 2.03 | 2.58 1.97, 2.05 |
| ivy | 2.01 1.71, 1.77 | 1.90 1.72, 1.75 | 2.05 1.71, 1.73 | 1.92 1.70, 1.73 |
| villa | 1.70 1.86, 1.93 | 1.76 1.88, 1.96 | 1.70 1.86, 1.92 | 1.71 1.86, 1.92 |
| dambreak | 1.62 1.64, 1.78 | 1.66 1.64, 1.79 | 1.60 1.61, 1.70 | 1.61 1.58, 1.71 |
| sportscar | 1.63 1.69, 1.80 | 1.63 1.69, 1.79 | 1.67 1.65, 1.81 | 1.69 1.67, 1.80 |
| landscape | 1.51 2.32, 2.64 | 1.49 2.30, 2.62 | 1.36 2.19, 2.47 | 1.37 2.18, 2.44 |
| lte-orb | 1.64 -, - | 1.62 -, - | 1.55 -, - | 1.52 -, - |
| ganesha | 1.38 1.62, 1.67 | 1.41 1.61, 1.66 | 1.37 1.54, 1.58 | 1.37 1.54, 1.58 |
| dragon | 1.34 1.51, 1.53 | 1.35 1.50, 1.52 | 1.40 1.45, 1.49 | 1.35 1.45, 1.49 |

| mesh | fcpw8w16: near vol | fcpw8w16t | fcpw8w8 | fcpw8w8t |
|---|---|---|---|---|
| pavilion | 1.15 1.25 | 1.20 1.67 | 1.13 1.19 | 1.19 1.60 |
| head | 1.14 1.33 | 1.37 1.71 | 1.10 1.28 | 1.43 1.74 |
| zero-day | 1.42 1.53 | 1.65 1.64 | 1.37 1.42 | 1.55 1.50 |
| bmw | 1.17 1.43 | 1.47 1.63 | 1.12 1.31 | 1.53 1.58 |
| crown | 1.43 1.53 | 1.74 1.64 | 1.35 1.42 | 1.65 1.53 |
| ivy | 1.42 1.50 | 1.69 1.59 | 1.30 1.41 | 1.63 1.52 |
| villa | 1.19 1.39 | 1.61 1.66 | 1.12 1.28 | 1.56 1.53 |
| dambreak | 1.32 1.33 | 1.56 1.40 | 1.29 1.33 | 1.53 1.35 |
| sportscar | 1.29 1.36 | 1.58 1.36 | 1.22 1.23 | 1.48 1.24 |
| landscape | 1.26 1.30 | 1.44 1.28 | 1.27 1.35 | 1.35 1.36 |
| lte-orb | - - | - - | - - | - - |
| ganesha | 1.42 1.27 | 1.54 1.08 | 1.35 1.17 | 1.40 1.01 |
| dragon | 1.36 1.26 | 1.39 1.06 | 1.23 1.16 | 1.25 1.01 |

| schedule | nearest, primary | nearest, diffuse | any, ao | closest, near | closest, volume | cells under 1.00 |
|---|---|---|---|---|---|---|
| fcpw8w16 | 1.34-3.36 | 1.51-2.32 | 1.53-2.64 | 1.14-1.43 | 1.25-1.53 | none |
| fcpw8w16t | 1.35-3.54 | 1.50-2.30 | 1.52-2.62 | 1.20-1.74 | 1.06-1.71 | none |
| fcpw8w8 | 1.36-3.53 | 1.45-2.19 | 1.49-2.47 | 1.10-1.37 | 1.16-1.42 | none |
| fcpw8w8t | 1.35-3.16 | 1.45-2.18 | 1.49-2.44 | 1.19-1.65 | 1.01-1.74 | none |

Reading it. With FCPW's shape in the leaf, the matching schedules are over
FCPW on every cell, the closest point by 1.10-1.53x where this afternoon's
had the volume batch of the two largest meshes under it. The tuned
schedules' one departure, the leaf exits dropped, is the near batch's gain
(1.20-1.74x) and the large meshes' volume batch's loss (ganesha 1.27 ->
1.08, dragon 1.26 -> 1.06), as the probe said; the ray queries, which the
departure does not touch, agree between the two to within noise. The
plots in build-rtq2/results/plots are drawn from these rows: rtq-firsthit,
rtq-anyhit and rtq-closestpoint for the matching schedules, the same with
-tuned for the tuned ones.

## The prefetch over FCPW's layout: implemented, measured, and not taken (2026-10-06, night)

The tuned FCPW schedules' next departure was Embree's: the hit children
prefetched. The directive could not be lowered over FCPW's layout, because
its Interior is an arm of the node row's own switch on the first slot's
sign -- not a lookup into another group, as Embree's `Interior from
Nodes[ref >> 4]`, nor a range of tiles, as a leaf's `data = range(...)` --
so the arm had no storage of its own for the field resolver to name
(Lower/Layouts.cpp, field_in_layout, kRowOfArm). Now an inline arm that
names none resolves to the row the switch sits in, the node row at the
child's index, when the switch is inside a group's row; at a layout's
root there is no row to name and the refusal stands. Tests at the three
levels: lower/prefetch-switch-arm (the prefetch lowered as the switch on
the child row's kind, the interior arm fetching `group1_index[child]` and
the leaf arm its first packet, both at the widest arm's bytes),
ssa/prefetch-switch-arm and backends/llvm/prefetch-switch-arm (the gang's
prefetch of the hit children's rows, `llvm.prefetch` of the lines),
correctness/cpp/bvh4_fcpw_rows (bvh4_fcpw_slabs's scene and answers over
FCPW's node-row layout with both queries prefetching; a bad address would
fault).

Measured on the eight-wide sixteen-lane tree (fcpw8w16t with and without
`prefetch(triangles.Interior.children)` on all three queries, alternated
twice; side 1024, least of 5; bonsai over FCPW):

| mesh | cell | without | with |
|---|---|---|---|
| head | nearest primary / diffuse | 1.87-2.05 / 1.78-1.83 | 1.58 / 1.39 |
| head | any ao | 1.91-1.97 | 1.45 |
| head | closest near / volume | 1.37-1.44 / 1.71-1.78 | 1.11-1.15 / 1.42 |
| ganesha | nearest primary / diffuse | 1.38-1.43 / 1.60-1.62 | 1.36-1.38 / 1.51 |
| ganesha | any ao | 1.66-1.67 | 1.53 |
| ganesha | closest near / volume | 1.54-1.55 / 1.08-1.09 | 1.35-1.41 / 1.32-1.34 |

A loss on every cell but ganesha's volume batch, and a heavy one on the
small mesh. The reason is the directive's one policy: every arm fetches
the widest arm's bytes, so that the arms differ in the address alone and
the backend folds them into one fetch -- Embree's own `BVH::prefetch`,
which fetches the same four lines whatever the reference points at. Over
Embree's layout the widest arm is a Triangle4 tile of 144-176 bytes, three
lines; over FCPW's sixteen-lane layout it is a packet of 640 bytes, ten
lines per hit child, and a node row is 224. Ten lines a child is more than
the memory system returns in time for a traversal this short, and it
crowds out the lines the walk needs now. So the tuned FCPW schedules stay
without the prefetch, and what the measurement asks for is the size of a
prefetch as the schedule's to say -- a byte or line count beside the
cursor, `prefetch(triangles.Interior.children, 256)` -- since whether a
leaf's bytes are worth fetching ahead depends on the layout's tile and
the machine, not on anything the lowering can know. That is a scheduling
construct and waits for the user's ruling (ask before adding one);
until then the widest-arm policy stands, as it always has, and the
prefetch over FCPW's layout is available but not scheduled.

## The prefetch's line count, Embree's four-wide count, and the prefetch that read what it fetched (2026-10-07)

The user's ruling on the section above: a count of cache lines as an
optional argument of the directive is reasonable (no architecture
prefetches at a finer grain than its line; the count is the schedule's,
the line is the backend's). So `prefetch(cursor, n)`: ir::Prefetch carries
`lines`, the parser takes a positive constant after the cursor,
Lower/Prefetches.cpp passes it on the intrinsic, and the layout lowering
fetches `n * 64` bytes in place of the widest storage's
(LowerReferencePrefetches::kCacheLine, the LLVM backend's line, x86's; a
backend with another line would be asked for its own there). Zero is an
error. Tests: lower/prefetch-lines (the child's row at `256u`; the test
was first blessed with the two arms of the lowering as it then was, see
below), backends/llvm/prefetch-lines (four `llvm.prefetch` a hit child),
error/prefetch-lines-zero, and correctness/cpp/bvh4_fcpw_rows with a
count on both queries.

Embree's count, read from bvh.h (`BVH::prefetch`, the non-KNL branch): two
lines for every N over aligned nodes, `prefetchL1(ptr + 0*64)` and `+
1*64`, and the third and fourth only for N >= 8 or unaligned node types.
The directive without a count fetched the widest arm: over Embree's BVH8
layout the node's 256 bytes, Embree's four; over the BVH4 layout a
Triangle4 tile's 144 bytes, three, where Embree fetches two. The matching
four-wide schedule now says two, and the measurement of three against two
(embree4 on head, pavilion, ganesha; rays at side 2048, points at 1024;
least of 5, cpu 11, alternated twice; bonsai over Embree):

| mesh | lines | primary / ao / diffuse | any ao | near / volume |
|---|---|---|---|---|
| head | 3 | 1.06 / 0.98 / 0.98 | 0.98 | 3.69 / 4.17 |
| head | 2 | 1.06-1.07 / 0.97-0.98 / 0.98 | 0.98-0.99 | 3.69 / 4.16 |
| pavilion | 3 | 1.07 / 0.96-0.97 / 0.96-0.97 | 0.98 | 3.05 / 4.12-4.13 |
| pavilion | 2 | 1.05-1.07 / 0.96 / 0.96 | 0.98-0.99 | 3.07-3.10 / 4.12-4.13 |
| ganesha | 3 | 1.04-1.05 / 1.02 / 1.02 | 0.99 | 5.41 / 7.03-7.04 |
| ganesha | 2 | 1.05-1.06 / 1.02-1.03 / 1.02-1.03 | 1.00 | 5.38-5.39 / 6.79-6.80 |

Within a point on every ray cell; ganesha's volume batch, the one that
visits the most leaves, is 3.5% faster with the third line, a Triangle4
tile's last. So embree4.bonsai writes Embree's two, tuned4.bonsai writes
three with that measurement, and the eight-wide pair write their four
explicitly (no change in what they fetched). The four-wide cells are not
in the CSV yet (the eight-wide table is what the plots draw); the count
goes in with the four-wide block.

Then FCPW's tuned eight-wide tree with the count: fcpw8w16t with 2, 4 and
6 lines against none, head, ganesha, dragon, side 1024, least of 5,
alternated twice (bonsai over FCPW):

| mesh | lines | primary / diffuse | any ao | near / volume |
|---|---|---|---|---|
| head | 0 | 1.86-1.87 / 1.78 | 1.91 | 1.36-1.37 / 1.70-1.71 |
| head | 2 | 1.62-1.63 / 1.38 | 1.45 | 1.25 / 1.52-1.53 |
| head | 4 | 1.64-1.75 / 1.39-1.42 | 1.44-1.48 | 1.20-1.21 / 1.49-1.50 |
| head | 6 | 1.72-1.75 / 1.41-1.43 | 1.48-1.50 | 1.14-1.19 / 1.44-1.50 |
| ganesha | 0 | 1.38-1.40 / 1.60-1.61 | 1.66 | 1.54 / 1.08 |
| ganesha | 2 | 1.23-1.24 / 1.39 | 1.44-1.45 | 1.37-1.38 / 1.20 |
| ganesha | 4 | 1.23 / 1.40 | 1.45 | 1.37-1.38 / 1.27-1.29 |
| ganesha | 6 | 1.22-1.24 / 1.41 | 1.44-1.45 | 1.34-1.36 / 1.23-1.24 |
| dragon | 0 | 1.33-1.37 / 1.50-1.51 | 1.52-1.53 | 1.40 / 1.06-1.07 |
| dragon | 2 | 1.22-1.24 / 1.37 | 1.38-1.39 | 1.31 / 1.19 |
| dragon | 4 | 1.23-1.25 / 1.37-1.38 | 1.38-1.40 | 1.31-1.33 / 1.25-1.27 |
| dragon | 6 | 1.24-1.26 / 1.40 | 1.40 | 1.31-1.32 / 1.23-1.24 |

Two lines cost 13-25% on every ray cell and 9-12% on the near batch, and
no count did better than four; only the large meshes' volume batch gained
(1.08 -> 1.27-1.29, 1.07 -> 1.25-1.27 at four). No prefetch instruction
costs that: over Embree's layout, three lines against two is a wash. So
the lowered code was read (tests/bonsai/lower/prefetch-switch-arm.expect
as it was): over FCPW's layout the prefetch of a hit child was

    if (((group1_index[child].child0 >> 31) & 1) == 0) {
        prefetch(&group1_index[child], 160u)
    } else {
        prefetch(&group0_packets[(-group1_index[child].child0 - 1) ...], 160u)
    }

-- a load of the child row's first slot to choose the arm, and a second
through it for the leaf's packet: a demand miss on the very line the
prefetch was to bring in early, serialized into the parent's children
loop once per hit child, and a branch on a value that is not there yet.
Embree's layout never pays this, its arm being the reference's own low
bits, in a register. The rule, and the fix, in Lower/Layouts.cpp
(ReadsThrough, LowerReferencePrefetches): a prefetch never reads memory.
Its address is a function of the reference's bits, and where the arm's
kind, or an arm's address, would be read out of the row the reference
names, the row read through is what every arm's visit reads first and
the one thing fetchable without reading, so it is fetched in the read's
place -- an arm whose address reads through the reference fetches the
rows it reads instead of its storage, and when the switch's own
conditions read through it there is no switch, and one fetch of each row
read and of every arm's load-free storage, whatever the kind, as Embree
fetches the same lines whatever the NodeRef points at. The bytes without
a count are the widest storage fetched: the row's 224 over FCPW's
eight-wide layout (four lines, the row being 32-aligned), 112 in the
tests. The goldens: the switch gone, one `prefetch(&group1_index[child],
112u)` under the hit mask; the LLVM of the FCPW-shaped test from 25
`llvm.prefetch` calls and 44 branches to 9 and 38, loads 71 either way
(the child row's were gathers). Embree's shape (prefetch-children) is
unchanged: its arms keep their switch, their addresses being load-free.
The correctness test over FCPW's node-row layout passes with the fetch
(bvh4_fcpw_rows). The measurement of the fixed prefetch over FCPW's tree
follows in the next section once the machine is free (bonsai-69's sweep
runs until about 04:15).

## The fixed prefetch measured, the any hit FCPW sorts, and the audit (2026-10-07)

The machine crashed at about 01:09, with the commit above in and the
session's scratch -- the CSV folding script, the probe scripts, six pinned
worktrees under /tmp -- gone with the reboot. The user's rule from it: an
app's scratch lives under apps/<app>/scratch, gitignored (`apps/*/scratch/`),
never under /tmp. The scripts were rewritten there: setup-wt.sh pins a
worktree of the compiler at a commit with the shared checkout's dependencies
linked in (measurements run inside it, so an edit in the live tree cannot
change the compiler under a sweep); probe-prefetch.sh alternates a tuned
FCPW schedule with and without the directive; sweep.sh runs a block of the
table; tocsv.sh folds compare.sh logs into the CSV, carrying over the cells
an earlier CSV has that the run did not measure. The CSV and the plots moved
out of the build directory, which gets erased, to apps/rtq/results and
apps/rtq/plots (gitignored data).

**The fixed prefetch over FCPW's eight-wide tree.** fcpw8w16t with
`prefetch(triangles.Interior.children)` at the row (no count: 224 bytes,
four lines) and at 2 and 6 lines, against none, on all three queries; head,
ganesha, dragon; side 1024, least of 5, alternated twice; the compiler at
b4c292e0 from the pinned worktree (bonsai over FCPW, both rounds):

| mesh | lines | primary / diffuse | any ao | near / volume |
|---|---|---|---|---|
| head | 0 | 1.86-1.89 / 1.78-1.79 | 1.91 | 1.37 / 1.70-1.71 |
| head | 2 | 1.82-1.93 / 1.73-1.76 | 1.87-1.93 | 1.33 / 1.65-1.67 |
| head | row (4) | 1.83-1.98 / 1.71-1.75 | 1.87-1.93 | 1.33-1.35 / 1.69-1.72 |
| head | 6 | 1.94-1.97 / 1.74-1.78 | 1.91-1.96 | 1.26-1.34 / 1.63-1.65 |
| ganesha | 0 | 1.39-1.40 / 1.62 | 1.67 | 1.53-1.55 / 1.08 |
| ganesha | 2 | 1.37-1.39 / 1.64 | 1.68-1.69 | 1.51-1.58 / 1.15 |
| ganesha | row (4) | 1.39-1.40 / 1.64-1.65 | 1.70-1.71 | 1.57-1.65 / 1.28 |
| ganesha | 6 | 1.40-1.41 / 1.66-1.67 | 1.72 | 1.52 / 1.25-1.26 |
| dragon | 0 | 1.33-1.39 / 1.50-1.52 | 1.53-1.54 | 1.40-1.44 / 1.06-1.07 |
| dragon | 2 | 1.41 / 1.57-1.58 | 1.55-1.58 | 1.41-1.45 / 1.14 |
| dragon | row (4) | 1.38-1.39 / 1.59-1.60 | 1.58-1.59 | 1.44-1.46 / 1.25-1.28 |
| dragon | 6 | 1.41-1.45 / 1.60-1.62 | 1.60 | 1.42-1.49 / 1.22-1.24 |

With the demand load out of it, the prefetch over FCPW's layout behaves as
it does over Embree's: on the large meshes a gain on every cell -- rays 1-6%,
the near batch 2-7%, the volume batch 1.08 -> 1.28 and 1.07 -> 1.25-1.28
(18-20%) -- and on the small mesh a wash on the rays (within 3% both ways)
and 2-4% off the points. Two lines fetch too little for the volume batch
(1.15, 1.14); six do no more than four. The volume-batch gain is the tuned
closest point's known-open loss recovered: with the row fetched, the tuned
schedule's volume batch stands level with the matching one's (1.27, 1.26 in
the eight-wide table) instead of 15-19% behind it, and its near batch keeps
its lead. The same probe over the four-wide tree with sixteen-lane leaves
(fcpw4w16t, where the row is 112 bytes, two lines) and the eight-wide tree
with eight-lane leaves (fcpw8w8t), the row against none, same meshes and
method:

| schedule | mesh | prefetch | primary / diffuse | any ao | near / volume |
|---|---|---|---|---|---|
| fcpw4w16t | head | none | 1.78-1.80 / 1.85 | 2.00 | 1.35-1.36 / 1.50 |
| fcpw4w16t | head | row | 1.74-2.09 / 1.79-1.88 | 1.95-2.07 | 1.39 / 1.55-1.56 |
| fcpw4w16t | ganesha | none | 1.43-1.44 / 1.53-1.54 | 1.61-1.62 | 1.35-1.45 / 0.95-1.04 |
| fcpw4w16t | ganesha | row | 1.44-1.55 / 1.58-1.61 | 1.65-1.67 | 1.38-1.39 / 1.15-1.20 |
| fcpw4w16t | dragon | none | 1.36-1.39 / 1.42-1.47 | 1.40-1.50 | 1.29-1.32 / 1.00-1.01 |
| fcpw4w16t | dragon | row | 1.39-1.41 / 1.50-1.52 | 1.51-1.52 | 1.32-1.38 / 1.11-1.20 |
| fcpw8w8t | head | none | 1.80-1.96 / 1.74-1.79 | 1.84-1.90 | 1.34-1.36 / 1.65-1.68 |
| fcpw8w8t | head | row | 1.76-1.92 / 1.68-1.71 | 1.82-1.88 | 1.33-1.34 / 1.67 |
| fcpw8w8t | ganesha | none | 1.35-1.40 / 1.51-1.55 | 1.57-1.60 | 1.39-1.42 / 1.03-1.04 |
| fcpw8w8t | ganesha | row | 1.40 / 1.61-1.62 | 1.63-1.64 | 1.44-1.45 / 1.21 |
| fcpw8w8t | dragon | none | 1.34 / 1.43-1.45 | 1.47-1.48 | 1.24-1.25 / 1.00-1.01 |
| fcpw8w8t | dragon | row | 1.40-1.50 / 1.51-1.57 | 1.54-1.58 | 1.34-1.35 / 1.19-1.21 |

The same shape on both: the large meshes gain on every cell, the volume
batch most (0.95-1.04 -> 1.15-1.21 on ganesha and 1.00 -> 1.11-1.21 on
dragon, where the tuned four-wide schedule without it had stood under the
matching one's 1.16), head within noise both ways. So the four tuned FCPW
schedules write `prefetch(triangles.Interior.children)` on all three
queries -- the row's bytes, no count -- and keep the unsorted any hit.

**The answers matched: a hit record on both sides.** The user's decision
on the audit's open item: match the answer first, then one full re-run of
every cell, and whether the record stays is decided on the plots. So the
exports write what the reference writes. `trace_all` fills a `Hit` per ray
and `closest_all` a `ClosestHit` per point, both elements of the metric
file, since their fields and their arithmetic are the reference's:
metrics/embree.bonsai's Hit is RTCRayHit's (primID, t, u, v, the
unnormalized Ng = cross(e2, e1)) and its ClosestHit the tutorial callback's
(primID, p, d); metrics/fcpw.bonsai's Hit is the Interaction's (primID, d,
p = pa + v1 v + v2 w, the unit normal, uv = (1 - v - w, v)) and its
ClosestHit (primID, d, p, uv). Each is made once per query after the
traversal, for the element the argmin settled on, by the reference's test
run once more over that one element -- Embree's finalize for the last
accepted hit (Embree finalizes every hit that improved on the best, in the
leaf), FCPW's point, normal and uv for one lane (FCPW makes them for all
sixteen lanes of every packet before the scan). Two things came out of
writing it, both in the metrics: the closest-point routines had shortened
the reference's `p = a + ab v; d = |q - p|` to `|ap - ab v|`, the same
value algebraically but not the same roundings or the same operations
(the comment in metrics/fcpw.bonsai even said so), and they now compute the
point as the reference does (closest_point_embree, closest_point_fcpw -- the
latter returning FCPW's barycentric pair too, dead in the leaf, alive in the
record -- and `distmin` takes `q - p`); and the language has no
`rsqrt_approx`, so FCPW's `normalize` (enoki's approximate reciprocal root
with a Newton step) is a true root and a division in the record, a scalar
per ray that hits, the one place the arithmetic is not FCPW's instruction
for instruction. The driver reads both sides' records into one shape
(RefHit, RefClosest, as_ref) and reports, beside the id agreement, the
largest deviation of each field over the queries both sides answered alike
-- a ray's t relative, the normal relative to the reference's length, uv
absolute, the point and the closest point's distance as fractions of the
diagonal -- which should read as last bits except where the reference's own
arithmetic is ill-conditioned. The first compile said `unknown type: Hit`
at the export: the parser resolves an element's name when it reads the
declaration that uses it, file by file in input order, and the metric
file, which declares the records, came after the program. So the elements
moved to apps/rtq/elements.bonsai, compiled first (elements, metric,
program, schedule: every file's declarations before the file that names
them), and compare.sh's input list says so. The check on head (b4c292e0,
side 1024, one repeat): every row of all three queries agrees with both
references, the records within t 4-7e-6 relative, uv 3e-6 (Embree) and
up to 5e-4 on a primary ray (FCPW -- a sliver's barycentrics), Ng
bit-identical to Embree's, FCPW's unit normal within 2e-7 (the root), the
points within 1e-7 of the diagonal; and the SSA of `trace_all` shows the
argmin still carrying the leaf pointer and lane through the traversal, the
triangle loaded from its tile only after it -- the record's cost is the
scalar re-test and a 32-byte store per ray, nothing in the loop.

**The standard cells, and only those.** The user asked why the table had
ao rays under the first hit and primary and diffuse rays under the any hit.
There is a standard -- Aila and Laine (2009), kept by every traversal paper
since, Embree's included: three ray sets, each under the query it is for,
primary and diffuse rays the nearest hit, ambient-occlusion rays the any
hit with early termination -- and the other pairings were my
over-completeness, answering nothing a renderer asks and costing sweep
time. The driver now runs the standard pairing by default (`--batch` with
`--query` still runs any pairing for a probe), the plots draw the first
hit with primary and diffuse rays and the any hit with ao rays, and the
table below is those cells.

**The any hit FCPW sorts.** Asked why the ray queries stand 1.3-3.5x over
FCPW on FCPW's own tree, I read mbvh.inl's intersectFromNode again, and the
user pointed out that our any hit does not sort -- true of `occluded` in
every schedule here, Embree's traverseAnyHit pushing its children unsorted.
FCPW's does: intersectFromNode is one traversal for both queries,
`checkForOcclusion` only changing what happens at a hit, and every interior
node goes through enqueueNodes -- sortOrder4's five compare-and-swaps at
branching 4, the nearest child swapped to the top at 8. So the matching FCPW
schedules had under-matched FCPW's any hit since 2026-10-05, Embree's shape
copied where FCPW's should have been read, and every FCPW any-hit cell in
the tables above compares a sorted traversal against an unsorted one. The
four matching schedules now write `occluded.sort(...)` with the nearest
hit's key; the tuned ones keep the unsorted push (the sort on the any hit
measured a loss everywhere over Embree's trees, 2026-10-05, and the table
below says what it is over FCPW's). The driver was checked while at it:
the any hit does call FCPW with `checkForOcclusion = true` (rtq_hook.cpp,
reference_occluded), and FCPW's own occluded rate exceeds its intersect
rate on every batch, the early return and the skipped hit-point work.

**The audit, by the user's rule.** "Our schedules must match the reference
engine for everything except the tuned schedules. Our layouts must match.
Our queries must match. Only the tuned files can change the schedule, and
only the schedule." Every matching schedule, layout and query walked against
the reference's code, step by step:

| step | Embree (embree, embree4) | FCPW (fcpw4w16, fcpw8w16, fcpw4w8, fcpw8w8) |
|---|---|---|
| nearest hit, children's order | traverseClosestHit sorts by entry distance: `trace.sort` | enqueueNodes sorts (4: all four; 8: nearest first, rest in lane order): `trace.sort`, full at 8 -- the one departure, a form the language lacks |
| any hit, children's order | traverseAnyHit, unsorted: no sort | the same enqueueNodes: `occluded.sort` (fixed today) |
| leaf early exit | `early_out` after the edge tests on both paths: `trace.skip`, `occluded.skip` | intersectWideTriangle has none: no skip |
| prefetch | BVH::prefetch per child, 4 lines at N = 8, 2 at N = 4: `prefetch(.., 4)`, `(.., 2)` | none: none |
| stack | stackSizeSingle 564 / 244: `loopify` | FCPW_MBVH_MAX_DEPTH 154 / 96: `loopify` |
| node test | Embree's slab test on the bits, `tFar` folded (metrics/embree) | intersectWideBox with `neq(child, maxInt)` (metrics/fcpw) |
| leaf test | Triangle4 Moeller-Trumbore, `valid` lanes by the layout's `where` | sixteen- or eight-lane Moeller-Trumbore, the count's lanes |
| closest point, order and bounds | pointQuery's traverseClosestHit: `closest.sort`; the children's farthest-distance bound is ours alone (Embree's callback API cannot use it; the user's ruling: an exact bound is the lowering's) | findClosestPointFromNode: `closest.sort`, the six region exits (`closest.skip(offset_fcpw)`), `tMaxMin` tightening |
| layout, tree | Embree's bytes, Embree's builder | FCPW's bytes, FCPW's builder |
| the answer | rtcIntersect1 writes t, u, v, Ng, primID, geomID; ours the primID -- OPEN | Interaction: d, p, n, uv, indices; ours the primID -- OPEN |

Two items open from it, both put to the user: the answer -- our exports
return the id where both references produce a hit record, so a hit record
(t, u, v, Ng; FCPW's point, normal, uv) written by the program after the
argmin is the match, about Embree's epilog's work; and FCPW's eight-wide
partial order, which would be a new scheduling form (ask first). The
user's rulings: the answers are matched first (the hit record above) and
every cell re-run after it, the record's fate decided on the plots; the
partial order is not to become a scheduling form -- the two eight-wide
FCPW schedules note the difference and it stays a documented one.

**The FCPW comparison is the four-wide tree.** The user's direction: FCPW's
author says the MBVH4 is the better-tested BVH, so the comparison against
FCPW is made on it -- fcpw4w16, FCPW's default branching factor with the
leaf width its build picks on this machine (sixteen lanes, AVX-512), FCPW
as it is built here with no option set -- and the plots draw `fcpw4w16`
beside Embree's `embree` (each reference on its own default tree, BVH8 and
MBVH4; plot.py's defaults, `--fcpw` names another). The eight-wide
configurations (FCPW_USE_EIGHT_WIDE_BRANCHING) stay in the schedules and
the CSV as secondary cells, measured after the four-wide ones in a sweep.
The eight-wide partial-order difference above therefore touches only the
secondary cells.

**The table at b4c292e0 with the records, the standard cells (2026-10-07,
09:55-10:35).** One block after the driver reboot, from the pinned
worktree; Embree's four schedules (rays at side 2048, points at 1024) and
FCPW's four four-wide ones (side 1024); least of five on cpu 11; 504 rows
in apps/rtq/results/rtq-results.csv, the plots in apps/rtq/plots drawn
from `embree` and `fcpw4w16`. Every Embree cell agrees to ties; against
FCPW the known residue (1-12 primary rays of a million on bmw, crown, ivy,
villa, dambreak through the reciprocals' last bits; ivy's and dambreak's
degenerate triangles on the point batches). Bonsai over the reference:

| mesh | triangles | embree: nearest pri dif, any ao | near vol | fcpw4w16: nearest pri dif, any ao | near vol |
|---|---|---|---|---|---|
| pavilion | 11366 | 0.85 0.99, 0.99 | 2.75 3.93 | 1.46 1.89, 2.02 | 1.05 1.12 |
| head | 17674 | 0.89 1.01, 0.98 | 3.58 4.03 | 1.48 1.78, 1.96 | 1.10 1.15 |
| zero-day | 48960 | 1.04 1.06, 0.99 | 3.51 3.82 | 1.73 1.89, 2.06 | 1.28 1.31 |
| bmw | 110592 | 0.92 1.02, 1.01 | 3.87 4.36 | 1.52 1.73, 2.07 | 1.11 1.20 |
| crown | 155520 | 0.96 1.05, 1.00 | 4.30 4.52 | 1.70 1.82, 2.00 | 1.25 1.30 |
| ivy | 179603 | 0.95 1.02, 0.97 | 4.67 4.16 | 1.65 1.62, 1.72 | 1.22 1.29 |
| villa | 390784 | 0.92 1.03, 1.00 | 4.13 4.65 | 1.60 1.80, 2.05 | 1.11 1.19 |
| dambreak | 1015024 | 0.99 1.00, 0.98 | 3.59 5.53 | 1.50 1.47, 1.77 | 1.17 1.16 |
| sportscar | 1091232 | 0.97 1.03, 1.01 | 4.42 5.29 | 1.45 1.56, 1.78 | 1.17 1.19 |
| landscape | 1916928 | 0.88 1.05, 1.07 | 3.31 2.99 | 1.26 2.11, 2.51 | 1.12 1.16 |
| lte-orb | 3423232 | 0.95 1.01, 0.99 | 4.77 5.46 | 1.48 - , - | - - |
| ganesha | 4323658 | 0.98 1.02, 0.98 | 5.43 8.02 | 1.37 1.51, 1.62 | 1.27 1.17 |
| dragon | 7219045 | 0.99 1.02, 0.98 | 6.10 7.85 | 1.31 1.40, 1.44 | 1.22 1.12 |

The tuned and four-wide Embree schedules and the other four-wide FCPW
configurations, the same cells (pri dif, ao; near vol):

| mesh | tuned | embree4 | tuned4 | fcpw4w16t | fcpw4w8 | fcpw4w8t |
|---|---|---|---|---|---|---|
| pavilion | 0.85 1.02, 1.02; 2.75 3.92 | 0.92 0.92, 0.99; 2.83 3.94 | 0.90 0.95, 1.03; 2.87 3.86 | 1.47 1.87, 2.11; 1.08 1.42 | 1.52 1.91, 1.98; 1.08 1.04 | 1.49 1.82, 2.02; 1.10 1.32 |
| head | 0.90 1.10, 1.09; 3.74 4.05 | 0.95 0.96, 0.98; 3.46 3.89 | 0.95 1.05, 1.09; 3.50 3.92 | 1.46 1.75, 1.98; 1.34 1.50 | 1.49 1.78, 1.95; 1.07 1.10 | 1.76 1.79, 2.05; 1.31 1.43 |
| zero-day | 0.95 1.14, 1.12; 3.54 3.85 | 1.01 0.95, 0.99; 3.28 3.55 | 1.05 1.08, 1.12; 3.50 3.78 | 1.76 1.88, 2.11; 1.36 1.38 | 1.87 1.80, 1.97; 1.22 1.23 | 1.84 1.79, 1.99; 1.33 1.31 |
| bmw | 0.93 1.10, 1.11; 3.87 4.36 | 0.92 0.97, 1.01; 3.73 4.12 | 0.93 1.03, 1.10; 3.77 4.18 | 1.50 1.72, 2.10; 1.33 1.40 | 1.52 1.69, 1.99; 1.05 1.11 | 1.50 1.69, 2.06; 1.32 1.39 |
| crown | 0.98 1.19, 1.14; 4.10 4.34 | 1.00 0.97, 1.00; 4.00 4.02 | 0.99 1.08, 1.13; 4.09 4.06 | 1.94 1.83, 2.00; 1.40 1.40 | 1.69 1.74, 1.89; 1.18 1.20 | 1.61 1.75, 1.91; 1.36 1.34 |
| ivy | 0.98 1.15, 1.09; 4.67 4.17 | 1.01 0.98, 1.00; 4.60 4.19 | 1.00 1.09, 1.10; 4.87 4.25 | 1.59 1.62, 1.75; 1.43 1.42 | 1.58 1.56, 1.68; 1.12 1.21 | 1.58 1.60, 1.72; 1.33 1.31 |
| villa | 0.94 1.10, 1.08; 4.12 4.63 | 0.91 0.97, 1.01; 3.89 4.36 | 0.92 1.03, 1.09; 3.90 4.38 | 1.41 1.74, 1.99; 1.36 1.45 | 1.47 1.70, 1.95; 1.04 1.09 | 1.42 1.71, 1.94; 1.30 1.35 |
| dambreak | 0.96 1.04, 1.04; 3.59 5.52 | 0.95 0.99, 1.00; 3.63 5.00 | 0.96 1.02, 1.06; 3.64 5.12 | 1.51 1.54, 1.84; 1.38 1.30 | 1.49 1.49, 1.73; 1.12 1.12 | 1.48 1.49, 1.76; 1.32 1.07 |
| sportscar | 0.99 1.10, 1.12; 4.52 5.24 | 0.99 0.98, 1.02; 4.25 4.71 | 0.99 1.04, 1.10; 4.27 4.86 | 1.45 1.62, 1.86; 1.36 1.29 | 1.43 1.53, 1.68; 1.12 1.12 | 1.46 1.61, 1.80; 1.27 1.19 |
| landscape | 0.89 1.09, 1.14; 3.38 3.00 | 0.96 1.16, 1.23; 3.24 2.80 | 0.96 1.09, 1.20; 3.36 2.85 | 1.42 2.11, 2.34; 1.32 1.21 | 1.26 2.09, 2.49; 1.13 1.18 | 1.36 2.07, 2.30; 1.26 1.18 |
| lte-orb | 0.96 1.07, 1.06; 4.76 5.43 | 0.99 0.98, 1.00; 4.23 4.51 | 0.99 1.04, 1.07; 4.29 4.71 | 1.54 - , - ; - - | 1.44 - , - ; - - | 1.52 - , - ; - - |
| ganesha | 0.89 0.99, 1.04; 5.44 7.99 | 0.98 0.98, 1.00; 5.39 6.75 | 0.97 1.02, 1.05; 5.26 7.01 | 1.36 1.55, 1.66; 1.37 1.13 | 1.34 1.43, 1.50; 1.19 1.07 | 1.37 1.52, 1.58; 1.27 1.08 |
| dragon | 1.00 1.06, 1.07; 6.10 7.85 | 1.00 0.99, 1.00; 5.57 6.58 | 1.00 1.03, 1.05; 5.65 6.88 | 1.36 1.50, 1.51; 1.32 1.09 | 1.30 1.35, 1.36; 1.12 1.03 | 1.37 1.48, 1.48; 1.23 1.03 |

Reading it against last night's table (no record, unsorted FCPW any hit,
eight-wide FCPW tree). The record's cost falls where the hits are: Embree's
primary first hit went from 1.03-1.10x to 0.85-1.04x (the scalar re-test
and the 32-byte store on the one batch of a million rays with a 17-30% hit
rate), its diffuse first hit from 1.01-1.15x to 0.99-1.06x, its any hit --
no record -- unmoved at 0.97-1.07x; the closest point, whose record is one
closest-point routine per query, unmoved at 2.75-8.02x. Against FCPW's
MBVH4 the first hit stands 1.26-1.73x on primary rays and 1.40-2.11x on
diffuse, the any hit, now sorted as FCPW's, 1.44-2.51x, and the closest
point 1.05-1.31x on every cell. The tuned FCPW schedules' prefetch holds
the volume batch level with or above the matching one's on every mesh
(ganesha 1.13 vs 1.17, dragon 1.09 vs 1.12 -- within noise, where last
night's tuned had lost 15-19%), and their near batch leads by 0.1-0.3. The
tuned Embree schedule is level with the matching one on primary rays and
3-14% ahead on diffuse and ao, as before. The user decides on the record
from these plots; the profiles (both sides, first hit and any hit) and the
eight-wide FCPW block are the next machine blocks.

**Why the ray queries beat FCPW on its own MBVH4: the profile.** The user
asked for a profile-guided answer. perf record of cycles, instructions,
branches and branch-misses on one cell at a time, both sides in one
process pinned to cpu 11, fcpw4w16 (the matching schedule, the sorted any
hit), the driver built with -g so FCPW's inlined code attributes to its
source lines (scratch/profile.sh; reports under scratch/profile-fcpw4w16-*).
FCPW's traversal is three symbols -- intersectFromNode, enqueueNodes<4>
and intersectRayPrimitives<16> -- and ours one (`trace_all`,
`occluded_all`); the ratio of the sums, FCPW over bonsai, per event:

| mesh, cell | cycles | instructions | branches | branch-misses |
|---|---|---|---|---|
| head, first hit primary | 1.78 | 1.91 | 2.20 | 1.97 |
| head, first hit diffuse | 2.27 | 2.70 | 3.41 | 2.16 |
| head, any hit ao | 2.44 | 2.93 | 4.23 | 2.32 |
| dragon, first hit primary | 1.91 | 1.73 | 1.76 | 1.78 |
| dragon, first hit diffuse | 2.07 | 2.20 | -- | 2.24 |
| dragon, any hit ao | 2.23 | 2.51 | -- | 2.28 |

(the sampled shares of each side; dragon's branches event fell under the
report's 1% floor for our symbol.) FCPW retires two to three times the
instructions and takes two to four times the branches for the same rays
and the same pruning, and misses twice as often -- the gap is instruction
count and control flow, not the node or leaf arithmetic, which both sides
run as the same vector operations. Where FCPW's time goes, by source line
(head; the shares are of all cycles or misses in the process):

- *enqueueNodes, 29% of FCPW's cycles and 30% of its mispredictions on
  primary rays, 36% and 33% on ao rays.* Its hottest lines are sortOrder4's
  compare-and-swaps (mbvh.inl:779-780, `if (t[b] < t[d])`, `if (t[b] <
  t[c])`: each a lane extracted from an SSE register and a data-dependent
  branch; array_sse42.h:1030 is the extract) and the push loop
  (mbvh.inl:796-800: `if (mask[W])` per lane, the child and its distance
  extracted lane by lane, `tMaxMin = min(..)` per lane -- `stub` on the
  ray path, computed and thrown away). Five sort branches and four mask
  branches a node, all on values the ray decides. Ours: the sort lowered
  as Embree's chain, one branch on the hit count and then straight-line
  min/max over packed key|index words, the push a vector compress; the
  whole any-hit kernel has 12 conditional branches.
- *intersectRayPrimitives, 19% of FCPW's cycles and 33% of its
  mispredictions on primary rays, 33% of its mispredictions on ao rays.*
  After the sixteen-lane Moeller-Trumbore, the per-lane scan
  (mbvh.inl:910-914, `if (mask[w] && d[w] <= rtMax)`): the lines
  array_avx512.h:147-148 and array_kmask.h:121 are the extraction of one
  lane of `d` and one bit of the kmask, up to sixteen times a packet, each
  with a branch; on the nearest hit the point, normal (a normalize: a root
  and a division) and uv for all sixteen lanes before the scan. Ours: one
  horizontal min with the lane index beside the key, no per-lane branch,
  and the record made once per ray after the traversal.
- *intersectFromNode, 52% of FCPW's cycles on primary rays.* The box test
  itself (bvh.h:52 is the slab test's `tMin <= tMax`; array_sse42.h:174-175
  the SSE min/max) is the one part both sides share; the rest is the stack
  pop with its skip (mbvh.inl:978, 3.7% of the ao misses alone), the
  `flatTree[nodeIndex]` vector indexing (stl_vector.h:1163), the gathers of
  the ray into enoki vectors per call.

So the answer to "why faster": the same tree, the same bytes, the same
arithmetic and the same pruning, but FCPW spends between the vector
operations -- sorting, pushing and scanning lanes one at a time with a
branch each -- what our lowering spends in a handful of vector
instructions. The any hit, now sorted as FCPW's, keeps the largest margin
(2.4x in cycles) because FCPW sorts there too and its leaf scan runs for
nothing on a hit; the primary first hit has the smallest (1.8x) because
coherent rays predict FCPW's branches best.

**The record made in the leaf: the map inside the argmin.** The record as
first written -- a second triangle test on the winner after the traversal
-- cost 6-18% on primary rays against Embree (pavilion 77.9 -> 64.0
Mrays/s, head 67.3 -> 57.2, landscape 61.6 -> 52.4, dragon 36.4 -> 34.1;
Embree's own rates and our diffuse rates unchanged, so the traversal was
not what moved), where Embree makes its record in the leaf for each hit
that improves on the best. The user's direction: no new compiler pass --
apps/pbrt's `trace` already says it, `argmin(key, filter(pred, map(|p, g|
(p, g, intersection(q, ...)), set)))`, the map annotating every candidate
with its record so that the argmin's element is the pair, held as a
reference to the primitive and the record by value, and the leaf evaluates
key, predicate and record together with CSE keeping one test. rtq's
queries now read the same way: `trace` is `argmin(|t, h| distmin(r, t),
filter(|t, h| intersects(r, t), map(|t| (t, hit_record(r, t)), triangles)))`
answering `option[(Triangle, Hit)]`, `closest` the same with
`closest_record` and no filter, and the exports write the pair's record.
One lowering bug came out of it, at its cause: build_traversal's `map`
case did not pass the argmin's carry -- the bound each `from` carries and
the loop over the children that tests and tightens it -- down to the tree,
where the `filter` case does; under a filter the fused filter consumed the
carry first, so pbrt's shape never showed it, but the filterless closest
point over a map lost its children loop (`vectorize(triangles.Interior.
children)` found no loop named children; `prefetch` no loop to sit in).
The map passes the carry through now, as Algorithm 1's map rule says the
recursion is left alone. Tests lower/argmin-over-map (the closest point
annotated with a record, the children loop with its mask and tightening,
the leaf's `argmin=` of `(key, (ref, record))`; trace's pair under a
filter) and backends/llvm/argmin-over-map (the loop vectorized, the
record's fields selected at the winning lane).

**The table with the leaf-made record (12:12-12:55, compiler 27a9636f).**
Embree: first hit on primary rays 0.95-1.03x (pavilion 0.96, head 0.97,
zero-day 0.96, landscape 0.95, the rest 1.00-1.03), diffuse 1.01-1.12x,
any hit on ao rays 0.96-1.15x, closest point 2.97-5.82x near / 3.06-8.09x
volume. FCPW's MBVH4: first hit 1.35-1.95x primary / 1.40-2.13x diffuse,
any hit 1.45-2.48x, closest point 1.11-1.28x / 1.13-1.31x. The post-
traversal record's 0.85-0.92 on primary rays is recovered to 0.95-1.03;
against the id-only kernel's 1.03-1.10 the remaining 4-7% is below.

**The primary-ray slowdown against Embree, read off the assembly.** The
user asked for the reason, concretely. perf on head's primary first hit,
both sides in one process, for the current kernel and the id-only kernel
of b4c292e0 (scratch/profile.sh --app, the b4c292e0 tree extracted beside
it); the three kernels' disassembly read side by side (scratch/
embree-intersect.asm from libembree4's symbol table, ours-{new,old}-
cycles.annot.txt). Static counts of the whole kernels:

| kernel | instructions | stack operands | spill stores | four-lane ops |
|---|---|---|---|---|
| Embree BVHNIntersector1<8>, Triangle4 Moeller | 604 | 108 | -- | 183 |
| ours, id only (b4c292e0) | 603 | 63 | 10 | 137 |
| ours, record in the leaf | 651 | 85 | 23 | 182 |

and relative to Embree in the same run our kernel retires 7% more
instructions and takes 3% more cycles than the id-only one did: on head,
55.95 against 59.08 Mrays/s, which is 0.95 ns, four to five cycles, per
ray -- not a traversal regression but a per-ray tax. Where it is, path by
path (the leaf is the four-lane block test from `vsubps` through
`vpcmpneqd` -- the layout's `where` predicate -- and `kortestb; je`, the
same shape in all three kernels):

- *Per block passing the depth test:* two more masked multiplies (u and v
  scaled by the reciprocal beside t). Embree does the same three. Trivial.
- *Per improving hit:* the record's six fields selected at the winning
  lane -- six vector spills (`vmovaps %xmm, 0x160-0x1b0(%rsp)`), six scalar
  loads at the lane's offset, two `vinsertps` to reassemble Ng, one
  16-byte store into the accumulator's record slot -- about seventeen
  instructions where the id-only kernel did one indexed load of the
  packed lane word and one 8-byte store. Embree's own improving hit is no
  lighter: it spills U, V, T, absDen and Ng (seven `vmovaps`), takes the
  reciprocal and the three products into the stack, runs select_min, the
  primID load, the geometry mask test and the filter test, then spills
  seven ymm and five GPRs before writing the hit into the ray-hit struct.
  Parity, not a loss.
- *Per ray, the tax:* the accumulator grew from a 16-byte (key,
  reference) to a 48-byte (key, reference, record) tuple on the stack,
  and it is (a) initialized whole on every ray -- `no_hit`'s zeros, three
  `vxorps`/`vmovaps` stores per ray, the hottest single instruction of
  the new kernel at 2.9% of its samples -- and (b) copied out at the end
  through a second temporary, `_best1_result` (an option of the pair,
  built from the accumulator) and then `hits[i] = found[1]` -- loads and
  stores of the 32-byte record twice. Embree has neither: its hit lives
  in the caller's RTCRayHit from the start (the caller resets geomID off
  the clock; our driver does the same for Embree) and is written in place
  on an improvement. At about fifteen instructions and the stores' latency
  per ray, this is the four to five cycles.
- *Pre-existing, both kernels:* the leaf test reads the ray's constants
  from the stack on every block (`vsubps 0x150(%rsp)`, `vmulps
  0x110(%rsp)`: six stack operands per block) where Embree keeps them in
  registers; the eight-wide node test's constants hold ymm23-31 and the
  four-lane leaf's got spilled. Store-forwarded, off the critical chain
  (the earlier payload probe found such spills free), but a register
  allocation worth revisiting with the tuned schedule.

The plan, in order:

1. *The record accumulates in its destination.* The argmin's by-value
   held component (the record) is only ever read once, after the
   traversal, to be stored into `hits[i]`; so lower it to live there --
   the accumulate writes the record's fields into the output element on
   an improvement, the footer's copy and the second temporary vanish, and
   so does the per-ray identity init of the record (the `else` branch
   writes `no_hit` as it does now, and nothing reads the record unless the
   key improved). This is store sinking into a reduction, a cousin of
   Opt/Sink's assignment sinking (2278cf71, Knoop-Ruething-Steffen) and
   of the "argmin payload in storage" probe of 2026-10-05 (9440bea3),
   which found the reference's storage level because the reference is 8
   bytes; the record is 32 and comes with an init and a copy. Site: after
   the traversal is inlined (SSA), the pattern is one alloca whose record
   component has one reader, a store to an addressable destination not
   otherwise read -- replace the component's slot by the destination.
   Tests at the lower/ssa/llvm levels on the argmin-over-map program, and
   the rtq execution check. Expected: the per-ray tax gone, primary rays
   back to about 1.00-1.03x; a day.
2. *The key not carried twice.* The record's `t` is the key: the held
   tuple stores both. After inlining the two are one SSA value in the
   build of the held tuple; a component equal to the key component is
   dropped and the result reads the key. Hours.
3. *The ray's leaf constants in registers.* The six stack operands per
   block in both kernels; the schedule's own concern (which constants the
   eight-wide test and the four-lane test share) and LLVM's allocation.
   To probe with the tuned schedule, after 1 and 2.

Not in the plan: the per-hit extraction through spilled vectors, which
Embree does the same way, and the two masked multiplies, which are the
record's honest work.

**Fused against unfused, both against Embree (14:30-15:20, 27a9636f).**
The user's ask: stop comparing with the kernel that makes no record, and
measure the record's two placements against Embree -- fused (the map
inside the argmin, the record made in the leaf: the committed program)
and unfused (the argmin finds the triangle, `hit_record` on the winner
after the traversal: the morning's program, scratch/unfused-app) -- the
`embree` schedule, the standard cells, back to back (scratch/ab-fusion.sh,
sweep-ab-{fused,unfused}). Bonsai over Embree, primary / diffuse / ao:

| mesh | fused | unfused |
|---|---|---|
| pavilion | 0.95 / 1.01 / 0.99 | 0.85 / 0.99 / 0.99 |
| head | 0.99 / 1.02 / 0.97 | 0.89 / 1.01 / 0.98 |
| zero-day | 0.97 / 1.05 / 0.98 | 0.90 / 1.03 / 0.99 |
| bmw | 1.01 / 1.04 / 1.04 | 0.92 / 1.01 / 1.00 |
| crown | 1.00 / 1.07 / 0.99 | 0.93 / 1.06 / 1.00 |
| ivy | 1.02 / 1.05 / 0.96 | 0.98 / 1.03 / 0.97 |
| villa | 1.01 / 1.06 / 0.98 | 0.90 / 1.01 / 1.00 |
| dambreak | 1.02 / 1.01 / 0.97 | 0.95 / 0.99 / 0.98 |
| sportscar | 1.02 / 1.04 / 1.00 | 0.97 / 1.03 / 1.01 |
| landscape | 0.95 / 1.15 / 1.11 | 0.89 / 1.09 / 1.16 |
| lte-orb | 1.01 / 1.02 / 0.98 | 0.95 / 1.01 / 0.98 |
| ganesha | 1.01 / 1.02 / -- | 0.96 / 1.02 / 0.99 |
| dragon | 1.02 / 1.02 / 0.97 | 0.97 / 1.01 / 0.98 |

The fusion is worth 4-11% on primary rays on every mesh and 1-5% on
diffuse; the any hit, with no record, is level between the two. The
fused program stays. What is left below Embree with it -- pavilion,
zero-day and landscape at 0.95-0.97 on primary rays -- is the per-ray
tax of the accumulator above; the user's direction is to take plan step
1 (the record accumulated in its destination), and to never carry the
key twice, which step 1 absorbs: the record's `t` becomes the
destination's field and the key the loop's register copy, Embree's own
arrangement (tfar in a register, written into the ray on each
improvement).

**The record stored, not carried: what was built (2026-10-07, afternoon).**
LLVM's own O3 on the module had already said where the tax was: after it
the accumulator is no alloca at all -- scalar replacement promotes the
whole (key, reference, record) tuple, and the record's eight fields ride
the traversal as loop-carried values, 49 phis against the id-only
kernel's 32, which is what the 23 spill stores against 10 were. Embree
carries tfar alone and stores u, v, Ng and primID into the ray-hit memory
when a hit improves. So the lowering now does the same, and the user's
rule was no new pass and nothing hacky:

- `ir::Accumulate` has `alongside`: places an argmin or argmax writes
  beside its pair when the key improves, each taking the value given,
  under the same comparison and -- in a vectorized leaf -- from the same
  winning lane (printed `_best argmin= (key, held) with _record = value`).
  Every pass that rebuilds an Accumulate passes them through (Mutator,
  Visitor, Canonicalize, RenamePointerToExpr, CSE, DCE, Inline), the
  free-variable and mutation analyses count the places, DCE counts their
  values as uses on the places' behalf and keeps an accumulate whose
  alongside places are read. In the SSA an AccArgmin's operands close
  with the (place, value) pairs after the place, the value and the mask
  (`Instruction::alongside`, with `has_mask()` and `alongside_begin()` so
  that no site counts operands; Linearize inserts a mask before the
  pairs); the vectorizer's by-key tail selects each alongside value at
  the winning lane as it does the held fields; the LLVM backend stores
  them under the comparison's branch (`select` for the pair, a branch for
  the stores, since nothing is to be written otherwise); the C++ source
  emitter refuses them plainly.
- `ir::Allocate::Memory::Stored`: stack storage that stays storage,
  written at some points and read once after, never carried as a value
  between. An SSA Alloca with `stored` set, which PromoteAllocas leaves
  alone and the LLVM backend emits with volatile accesses -- the one way
  LLVM offers to keep an alloca out of scalar replacement; one volatile
  load of the slot at the end is enough to pin it, the per-hit stores
  stay plain -- and the C++ emitter as `volatile`.
- `build_arg_extremum` gives each by-value annotation component of an
  argmin whose element is held by reference (the map's computed tail,
  annotation_arity) a Stored slot of its own, `_recordN`, written
  alongside the pair and read once in the footer in place of the carried
  component; the carried tuple is the key and the reference, as it was
  before the record. An element held by value with no reference beside
  it stays as it was.

Two bugs came out on the way, each fixed at its cause: build_traversal's
`map` case had dropped the argmin's carry, and the three places that ask
whether a metric is over one level counted its annotation parameters as
levels (27a9636f); and DCE's use counter handled Accumulate by hand and
so never saw the alongside values, letting the inliner's result temps die
under them (the `Var ... not in current block` failure). The head check
with the slot (one repeat): every row agrees with both references as
before; Embree's primary first hit 0.98x against 0.99x carried and 0.92x
unfused, the rest within noise. The four-mesh measurement at side 2048,
least of five, follows.

## The record written where the program puts it, and the any hit's exit at the hit (2026-10-07 night, 2026-10-08)

The morning's record work (the section above) left the hit record carried
through the traversal as loop-carried values: LLVM's SROA promoted the
accumulator's slot after inlining, 49 phis against 32 in a kernel with no
record, and the spills around the node test were the primary-ray tax
(0.94-0.97x on head, pavilion, zero-day, landscape). Two attempts and what
each measured, in order.

**Storage that stays storage, rejected.** `Allocate::Memory::Stored` kept
the record's slot in memory by emitting its accesses volatile. Read in the
optimized LLVM before any run (the rule now: the assembly first, then the
sweep): LICM's store promotion had put the slot's stores back into loop
phis (`.promoted`, `.lcssa` names), so the record was still 55 phis through
the traversal, and the ray paid loads of the slot before the loop and stores
after it besides. Level at best on the four small meshes, zero-day primary
lost. Removed, with the volatile emission and the promotion exception.

**Destination passing, kept** (SSA/DestinationPassing.h, the `pass
destinations` phase after the promotion and before the storage passes).
Embree pays nothing for its record because its leaf writes the caller's
RayHit through a pointer, and LLVM may not promote a conditional store to
memory another thread could see. Our export does `hits[i] = found[1]`: a
slot filled by the traversal and copied once. The pass replaces such a slot
with the destination's address -- `gep(hits, i)` where the alloca was,
handed to the traversal in its place -- and deletes the copy, when the slot
is only written (by stores, alongside places, or callees that only write
their parameter), loaded once, the loaded value flows through struct
packing and block arguments into nothing but stores at one address, that
address is computable where the slot is made, the destination's memory is
read nowhere and stored only at that address, every path from the slot to
the region's end stores it, and no write of the slot can follow a store to
it (LLVM's MemCpyOpt call-slot optimization and Shaikhha et al.'s
destination-passing style, cited in the header). `BONSAI_EXPLAIN_DESTINATIONS=1`
prints every refusal. In rtq it fires on `trace_all` and `closest_all`; the
leaf's five record stores land in the hit element and the found path has no
copy. The hit kernel: 55 phis -> 28, 27 spills -> 15, the 12 stack reloads
gone, 677 -> 633 instructions. Measured on the four small meshes (least of
five, pinned): primary 0.95/0.99/0.97/0.95 -> 0.98/1.01/0.97/0.99
(pavilion/head/zero-day/landscape), diffuse 1.01-1.15 -> 1.02-1.19, any hit
and closest point unchanged (no record carried, or the same change).

Not every slot is a candidate, by design: the single-query `trace` returns
its record inside an option (the loaded value is read, not just stored), and
`_best` is read by the callee (the key). The test
tests/bonsai/correctness/cpp/destination_passing runs the batch exports and
the single queries over the tiled-elements scene and checks every batch
record against its single query's, field by field; tests/bonsai/ssa/
destination-passing shows the three shapes (replaced, read as well, returned).

**The any hit's systematic 2-3%, profiled.** Block-level cycles and retired
instructions on head and ivy, both kernels, after the pass (perf, -F 15000,
30 repeats): on head our kernel retires the same instructions as Embree's
(44.66% vs 44.56% of the run) and takes 2.7% more cycles, with 3.6% more
branches and 8.8% more mispredictions; on ivy 26% more branches. The leaf
and the node test cost the same or less than Embree's (11.6% vs 11.7%, 12.3%
vs 13.7% of the run); the excess is in the small control blocks (6.7% of the
run in 35 blocks under 1% each, against Embree's 3.3% in 29). Two per-block
branches Embree does not have: a `vptest`+branch guarding the layout's
padding predicate (`where id != -1`, which Embree folds into its hit mask),
and a `setne`+`test`+branch materializing the found flag and testing it at
the next block's head, where Embree branches on the hit test's `kortest`
straight to `return true`. The ray's leaf constants read from the stack
(six operands) are not a gap: Embree's leaf broadcasts the ray's origin from
memory too (three `vbroadcastss`, nine stack operands).

**The exit at the hit, kept.** The quantifier's per-element guard was `if
(undecided) body` at the top of each element; it is now `body; if (decided)
break` (Lower/Trees.cpp, guard_iteration). In a tiled leaf the lanes are a
parallel batch, so the ForEach lowering (Lower/ForEachs.cpp,
split_trailing_exit) makes an element-independent trailing exit test once
per tile, after the lane loop, where the loop it leaves is the one over
tiles; the SSA conversion now lowers `Break` (a jump to the innermost loop's
exit; a merge decision by `never_falls_through`, which counts a break as a
return does; refused inside a parallel body). The any-hit kernel: 423 -> 416
instructions, 34 -> 32 branches, no `setne`. Measured: pavilion/head/ivy any
hit 0.97/0.97/0.97 -> 0.99/0.98/0.98; first hit and closest point unchanged.

**The padding guard, left.** Removing the any-lane guard from arms whose
only memory effect is a masked write (ispc's rule: a masked store is safe
with every lane off) took the padding guard off -- and the guard off the
argmin's update arm with it, so the winning lane's reduction and the five
record stores ran on every leaf block instead of the blocks where a lane
improved: diffuse rays -5 to -10%, closest point -5%, for no gain on the any
hit. Embree guards that update itself (`if (none(valid)) return` before its
epilog). Reverted; the measurement is in the comment in Linearize.cpp's
must_skip. The padding guard stays: one `vptest` and a predicted branch per
block, the one per-block instruction Embree does not have, worth about a
percent on the any hit; a layout-level statement of the mask is the way to
remove it, which is a language question.

**The table under this compiler** (13 meshes, least of five, cpu 11, rays
at side 2048, points at 1024, FCPW at 1024; apps/rtq/results/rtq-results.csv
rows `1b14228e` (measured as that commit's compiler before its hash existed), the plots in apps/rtq/plots). Every Embree cell agrees
to ties; against FCPW the known residue, unchanged cell for cell (bmw 6,
crown 1, ivy 2, villa 7+1, dambreak 12, sportscar 7, landscape 15, ganesha
11+1 rays of a million through the reciprocals' last bits; ivy's and
dambreak's degenerate triangles on the point batches). Bonsai over the
reference:

| mesh | embree: pri dif, ao | near vol | fcpw4w16: pri dif, ao | near vol |
|---|---|---|---|---|
| pavilion | 0.97 1.02, 0.99 | 2.90 4.09 | 1.67 2.07, 1.94 | 1.10 1.13 |
| head | 1.04 1.04, 0.98 | 3.77 4.32 | 1.65 2.00, 1.81 | 1.12 1.17 |
| zero-day | 0.97 1.05, 0.99 | 3.53 3.85 | 1.83 2.07, 1.92 | 1.29 1.32 |
| bmw | 1.01 1.06, 1.00 | 3.90 4.40 | 1.69 2.04, 1.77 | 1.12 1.20 |
| crown | 1.00 1.07, 1.07 | 4.41 4.61 | 1.85 1.99, 1.86 | 1.26 1.30 |
| ivy | 1.01 1.06, 0.97 | 5.03 4.49 | 1.64 1.75, 1.63 | 1.25 1.32 |
| villa | 1.02 1.06, 0.99 | 4.18 4.76 | 1.58 1.87, 1.84 | 1.15 1.24 |
| dambreak | 1.02 1.02, 0.99 | 3.72 5.71 | 1.57 1.76, 1.49 | 1.20 1.17 |
| sportscar | 1.04 1.05, 1.01 | 4.47 5.36 | 1.54 1.76, 1.61 | 1.17 1.18 |
| landscape | 0.98 1.16, 1.16 | 3.39 3.11 | 1.37 2.48, 2.21 | 1.15 1.22 |
| lte-orb | 1.01 1.03, 0.99 | 4.52 5.33 | 1.50 | |
| ganesha | 1.04 1.04, 0.98 | 5.51 8.14 | 1.43 1.60, 1.51 | 1.26 1.14 |
| dragon | 1.04 1.03, 0.98 | 5.86 7.67 | 1.35 1.43, 1.39 | 1.21 1.11 |

Geomeans over the twelve plotted meshes (lte-orb excluded as plot.py does):
Embree primary 1.011, diffuse 1.054, any hit 1.008, near points 4.14, volume
points 4.86; FCPW primary 1.59, diffuse 1.73, any hit 1.88, near 1.19,
volume 1.21. The user's finishing rule -- no cell more than 3% under its
reference, the geomean a win -- holds: the cells at the line are pavilion
and zero-day primary (0.97) and ivy's any hit (0.97), none beyond it. What
each remaining 3% is, from the profiles above: on primary rays of the
small meshes, the per-ray prologue (81 instructions against Embree's 51:
the ray's splats stored, the stack and the key set up) and the per-block
padding guard; on the any hit, the padding guard and the by-count arms'
dispatch, whose branches Embree does not have.

## Known-open, smaller

- The tuned FCPW schedules' closest point: without FCPW's region exits
  the near batch gains (head 1.14 -> 1.37x, ganesha 1.42 -> 1.54x) and the
  large meshes' volume batch loses (ganesha 1.27 -> 1.08x, dragon 1.26 ->
  1.06x) -- the exits pay where far points settle whole packets in a
  vertex or edge region. The tuned schedule should never be worse than
  the matching one; whether the fixed prefetch recovers the volume batch
  without costing the near one is the measurement pending above, and
  failing that the tuned closest point keeps FCPW's exits.
- The eight-wide residual (the profiling section): 1-3% on diffuse rays
  on pavilion and crown against the compiler before the reciprocal change,
  the two kernels differing only in the per-ray `rcp_embree_vec3` lowering
  its Newton step on three lanes where the intrinsic used four. The
  assembly diff is the next look.
- The closest point's sort key, a squared distance, is not known
  non-negative to the sort's lowering: the chain flips the sign bit of
  each key before comparing (`select(isneg, 0x7fffffff, 0) ^ key`, four
  times a node in the LLVM), where the ray's key (`max(.., tnear)` with
  `tnear` clamped) is ordered on its bits directly. The interval facts
  that carry the ray's non-negativity (SSA/ConstantIntervals) do not
  derive it for `dot(v, v)` or for `max(.., 0)` squared. A few
  instructions a node on the closest point, for both references.
- The checker's closest-point distance (rtq_hook.cpp, distance_to for a
  point) is Ericson's test in double and goes to NaN on a degenerate
  triangle; the ivy and dambreak meshes have them, and a disagreement
  that involves one cannot be classified. To be made robust: the distance
  to the three edges, the face only where the area is not zero.
- The near-point batch is built from the reference's primary hits, so a
  reference whose ray test finds nothing (FCPW on lte-orb) leaves it
  empty. To be made reference-independent: surface samples, chosen by
  area, pushed off along the normal -- the same for every reference and
  independent of either side being right.
- FCPW's scalar binary Bvh (vectorize off) is not matched: the layout is
  easy (BvhNode rows of 32 bytes, a leaf's references into the primitive
  array with the vertices gathered through the soup's indices) but its
  children are ordered by a lexicographic key (d2Min, then d2Max where
  both are zero), which `sort` cannot take. The user's TODO.
- FCPW's eight-wide order -- the strictly nearest child first, the rest in
  stored order -- is a partial sort the scheduling language has no form
  for; the fcpw8* schedules sort fully. Expected to matter little, the
  measured `sort` being FCPW's four-wide order and Embree's.
- ir::TiledArray, the "group of elements in tiles", is a one-level special
  case: an indirect group whose body is one anonymous constant-size group
  of fields. It does not compose with the rest of the layout language --
  a tile of tiles, a switch or a lookup inside a tile -- and the derived
  fields added for FCPW inherit that. The composable form: every
  constant-size group along the way to an element is a lane dimension,
  the element's index decomposed mixed-radix through the nesting, a field
  stored as a vector over the product of the constant widths inside the
  level it is declared at, and the row machinery (stored and derived
  fields, switches, lookups) the one lowering for rows and tiles alike.
  About a day's work replacing the special case; the derived-tile tests
  would pass through it unchanged. The user's call (option 1, 2026-10-05:
  keep the one-level form for now, generalize afterwards).
- The exported batch answers with the primitive id alone; Embree also
  writes `t`, `u`, `v` and the normal. Recovering `t` from the argmin
  without a second triangle test wants the key beside the element.
- A mesh that fits in one leaf gets a one-child root from the driver, where
  Embree's root would be the leaf.
- `--side 2048` was used for the tables above so that a single-threaded
  Embree run lasts over a tenth of a second; the default 1024 is fine for
  checking agreement.

## Ray packets against rtcIntersect16 (2026-10-08)

The user's ask: one more benchmark, our packet tracing against Embree's --
on a tree we already have if Embree traces packets on one, else a list of
the trees it does, for the user to choose from. It does: Embree's packet
queries, rtcIntersect4/8/16 and rtcOccluded4/8/16, run over the same
BVH8Triangle4 the single-ray comparison is made on (and over BVH4Triangle4;
bvh8_factory.cpp registers `BVH8Triangle4Intersector{4,8,16}HybridMoeller`
for them), through one kernel, BVHNIntersectorKHybrid<N, K>
(kernels/bvh/bvh_intersector_hybrid.cpp). Our Embree build has
EMBREE_RAY_PACKETS=ON with the AVX2 and AVX-512 kernels, and
rtcGetDeviceProperty says packets of 16 are native on this machine. So no
new tree: the packet comparison is the program's rays in gangs of sixteen
over Embree's layout against rtcIntersect16 and rtcOccluded16 -- item 4 of
the plan of 2026-10-05 ("Ray packets: rtcIntersect8/16 against the parfor
over the rays vectorized"). The point query has no packet form
(rtcPointQuery4/8/16 are scalar loops, rtcore.cpp), so the packet
comparison is the two ray queries.

### Embree's packet kernel, step by step (v4.4.1, the audit)

`BVHNIntersectorKHybrid<8, 16, BVH_AN1, false, ArrayIntersectorK_1<16,
TriangleMIntersectorKMoeller<4, 16, true>>, single = true>::intersect`,
what rtcIntersect16 with its default arguments runs on this tree:

1. Entry. The valid mask from the caller's sixteen ints (`*valid_i == -1`);
   `tnear` and `tfar` clamped to `max(.., 0)`; TravRayK: `rdir =
   rcp_safe(dir)`, `org_rdir = org * rdir`, `nearXYZ` (the near-slab
   offsets per axis, for the single-ray fallback).
2. With RTC_RAY_QUERY_FLAG_COHERENT in the arguments, a different kernel,
   `intersectCoherent`: a frustum over the packet's rays tested against the
   node's eight boxes at once (`intersectNodeFrustum`), each frustum-hit
   child then tested per ray, a scalar stack of 564 carrying
   `toScalar(childDist)`, `BVH::prefetch` on every hit child. The default
   flag is INCOHERENT; this is the kernel below. (The driver's `--coherent`
   runs the frustum kernel for a probe; it is not the default comparison.)
3. `switchThreshold`: 14 for K = 16 ("seems to work best for KNL"), 7 for
   K = 8 over N = 8 (5 over N = 4), 3 for K = 4; 2 under the coherent flag.
4. The octant partition. `octant = ray.octant()` (the three sign bits of
   the direction). If any two valid rays have opposing signs on every axis
   (`split`), the packet is traversed as sub-packets: the first valid ray's
   octant, and every ray within one sign bit of it (`count_diff_octant <=
   1`), is one sub-packet, `tnear = +inf` and `tfar = -inf` for the rays
   not in it; the next sub-packet from the first ray left; and so on. A
   packet with no opposing pair is one sub-packet. Each sub-packet gets
   its own stack and its own traversal, one after another.
5. The stack: `stack_node[561]` of NodeRefs and `stack_near[561]` of
   sixteen-wide floats (stackSizeChunk = 1 + (N - 1) * maxDepth, no +3),
   an invalid sentinel at the bottom, the root pushed with `tray.tnear`.
6. Pop. `curDist = *sptr_near`; `active = curDist < tray.tfar`; `none` ->
   next pop. The hybrid switch: if `popcnt(active) <= switchThreshold`,
   for each active lane i, `intersect1(bvh, cur, i, ...)` -- the single
   ray's kernel from this node: TravRay built from lane i, a stack of 564
   with `dist` culling, BVHNNodeIntersector1 (the eight-wide node test of
   one ray), BVHNNodeTraverser1Hit::traverseClosestHit (the sorted descent
   with `BVH::prefetch`), the leaf as `TriangleMIntersectorKMoeller::
   intersect(pre, ray, k, ...)`, the four-wide Triangle4 test of ray k with
   its `early_out`, its epilog writing lane k of the packet's hit -- then
   `tray.tfar = min(tray.tfar, ray.tfar)` and the next pop.
7. Down traversal (`while (!cur.isLeaf())`): `valid_node = tfar > curDist`;
   `cur = empty; curDist = +inf`; for i in 0..N: `child = children[i]`,
   break at the first empty; `lhit = valid_node`, then
   `BVHNNodeIntersectorK::intersect(nodeRef, i, tray, .., lnearP, lhit)` =
   `intersectNodeK`: child i's six bounds broadcast against the sixteen
   rays, `lclipMin = msub(lower, rdir, org_rdir)` and so on, `lnearP =
   maxi(min(..), min(..), min(..))`, `lfarP = mini(max(..), ..)` (the
   mixed float/int form for K = 16 on AVX-512), `lhit = asInt(maxi(lnearP,
   tnear)) <= asInt(mini(lfarP, tfar))`. If `any(lhit)`: `childDist =
   select(lhit, lnearP, inf)`; if `any(childDist < curDist)` { if cur
   not empty, push (cur, curDist); `cur = child; curDist = childDist` }
   else push (child, childDist). No prefetch anywhere in this loop.
8. After the children: `cur` empty -> pop. If two or more were pushed,
   `if (any(near[-2] < near[-1])) swap`; if three or more, the same for
   [-3] vs [-1], then [-3] vs [-2] -- a partial sort of the stack's top by
   lane-wise any-compares. Then SWITCH_DURING_DOWN_TRAVERSAL: if `popcnt(
   tfar > curDist) <= switchThreshold`, push (cur, curDist) and go to pop,
   where step 6 hands the lanes to single rays.
9. Leaf. `valid_leaf = tfar > curDist`; none -> pop. `TriangleMIntersectorK
   Moeller<4, 16>::intersect(valid_leaf, ..)` over the leaf's blocks: for
   each slot i of a block, `if (!tri.valid(i)) break`, then
   `MoellerTrumboreIntersectorK::intersectK`: the slot broadcast (`v0, e1,
   e2, Ng = cross(e2, e1)` as vectors of sixteen), `C = v0 - org`, `R =
   cross(C, dir)`, `den = dot(Ng, dir)`, `U = dot(e2, R) ^ sgnDen` with
   `valid &= U >= 0; if (none(valid)) return`, the same for `V`, for
   `W = absDen - U - V`, the depth test `absDen * tnear < T <= absDen *
   tfar`, `den != 0`; the epilog (IntersectKEpilogM): the ray mask test,
   masked stores of t (into tfar), Ng, u, v, primID, geomID per lane. Then
   `tray.tfar = select(valid_leaf, ray.tfar, tray.tfar)`.
10. `occluded`: the same entry and stack; `terminated = !valid`; the switch
    at a pop to `occluded1` per active lane (the single ray's any hit:
    traverseAnyHit, prefetch, stack of 564), `set(terminated, i)` on a hit;
    the down traversal with no ordering at all -- for each hit child, push
    the previous `cur` and take this one, so the last hit child is
    descended into and the earlier ones wait, the single ray's
    traverseAnyHit order; the leaf OccludedKEpilogM, `terminated |= hits`,
    `if (all(terminated)) break`, `tfar = select(terminated, -inf, tfar)`;
    at the end `tfar = -inf` stored for the terminated lanes.

### What the schedule says of it, and what it cannot

schedules/packet16.bonsai, over the same layout (layouts/embree8.bonsai):

| Embree | Directive |
|---|---|
| K = 16, a packet of consecutive rays (the driver's grouping) | `trace_all.split(i, i_gang, i_lane, 16, true).vectorize(i_lane)`, the same on `occluded_all`; the tail gang's guard is the driver's `valid` ints |
| step 7, each child against the sixteen rays, the children in stored order | the children's loop left a loop inside the gang (not `vectorize(triangles.Interior.children)`) |
| step 7-8, the nearest hit child descended into, the rest pushed, by entry distance | `trace.sort(triangles.Interior, \|i, r, lo, hi\| distmin(r, AABB{lo[i], hi[i]}))` -- each lane's own key, `inf` where the lane misses, the order one decision for the gang by a vote per comparison |
| step 5, the stack of 561 | `.loopify(561)`, after the gang and the sort: one stack of nodes and one of masks for the gang |
| step 6, the cull at a pop | the argmin lowering's bound test at a node's entry, per lane |
| step 9, a block's slots one at a time against the sixteen rays | the leaf's loop left a loop inside the gang |
| step 9, `none(valid) return` after each edge test | `trace.skip(intersectsp_ray_tri)`: the any-lane test before what follows each `if` |
| step 9, the epilog's masked update of each lane's best | the argmin's own |
| no prefetch in the chunk loop | no `prefetch` |
| step 10, no order in the any hit, the last hit child descended into | `occluded.loopify(561)` with no sort (the lowering's order for an unsorted run is Embree's single-ray one, matched since 2026-10-02) |
| step 10, a lane done at its first hit, the packet done when every lane is | the gang's `any`: one bool per lane |
| step 1, the clamp of tnear and tfar | `clamped` in queries.bonsai, as for every schedule |

Not said by any directive, and recorded in the schedule file so the table
is read right:

- Steps 3, 6 and 8: the hybrid. At fourteen or fewer active rays the packet
  is put back and each active ray traversed alone with the single-ray
  kernel (prefetch, sorted eight-wide descent, four-wide leaf), the packet
  resumed after. Our gang traverses as a packet to the end. This is the
  construct the comparison most wants and does not have: "when fewer than
  T lanes are active at this recursion, continue each active lane under
  the scalar schedule" -- Benthin et al. 2012's switch, a scheduling
  construct, and so the user's call (bonsai-ask-before-new-scheduling-
  constructs).
- Step 4: the octant partition of a packet into sub-packets of like
  direction. A partition of a gang by a key before the traversal, each
  part traversed in turn: no directive says it.
- Steps 7-8: the order is a scan with lane-wise any-compares and a partial
  sort of the stack's top three, not a full sort; here the vote settles
  each comparison of a full network, and the user decided (2026-10-08,
  with bonsai-58's change) that the vote is the majority of the on-lanes:
  the gang descends in the order most of its rays want, where Embree
  descends in the first order any ray wants. A decided departure, recorded
  so the table is read with it.
- Step 5: Embree's stack carries the sixteen distances per entry and
  re-derives the active lanes from them at a pop; ours carries the mask.

### The driver

rtq_hook.cpp, `--packet K` (4, 8 or 16) and `--coherent`: the batch is laid
in packets of K consecutive rays -- Embree's struct of arrays
(RTCRayHit16: one array of sixteen per field, 64-byte aligned, a `valid`
array of sixteen ints beside it, -1 for a ray and 0 for a lane past the
batch's end) -- made before the clock, reset between runs off the clock
(tfar, geomID, primID of the live lanes), and answered with
rtcIntersect<K> or rtcOccluded<K> per packet, the arguments left null as
the single-ray calls leave them (INCOHERENT) or the one flag set. The
answers are read back lane by lane into the same records as the single
rays (RefHit: primID, t, u, v, Ng; blocked from `tfar < 0`), so the same
checks run. The primary hits the secondary batches are made from stay
rtcIntersect1's, so every schedule is measured over the same rays.
compare.sh reads the width off a `packet<K>` schedule's name and passes
`--packet K`, as it reads FCPW's tree off an `fcpw<b>w<w>` name;
benchmark.sh runs a packet schedule for the two ray queries only;
tocsv.sh gives a packet row the tree's width (8), the packet's being in
the schedule's name; plot.py draws `rtq-packet` from the `packet16` rows
and prints their geomeans.

Checked on head at side 256 and 512 (cpu 13, one repeat, a check and not
a measurement): every packet answer agrees with the program's, 0 ties, 0
differ, at K = 16, 8 and 4, with and without the coherent flag -- the same
Moeller-Trumbore per lane, the same tree.

### The compiler, today: the packet traversal of a wide node is refused

The schedule does not compile: `vectorize() over a traversal whose nodes
bound their children (`with ... on <child>`): the run in !par_cont_6!sortall
has a condition per call, and a gang would have to make each call under its
own mask` (src/SSA/Vectorize.cpp:1631). bonsai-58, who hit the same wall
with apps/wosx's packet schedule over FCPW's four-wide tree, has traced it
to two causes and proposed the fix to the user: (1) SSA/Convert.cpp applies
every `sort` before any other directive whatever the textual order, so the
sorted run is built on the scalar function and the vectorizer then runs it
per lane; (2) a run with a condition per call (a wide node's children) is
refused outright. The change: `sort` in textual order (a sort after a
loopify refused), and a sorted lanes run in a gang made with votes -- the
per-lane compare of two children's keys (`inf` where a lane missed), then
the vote across the on-lanes, the lanes' own hit bits as the child's mask
-- the vote being the majority of the on-lanes, as the binary packet
traversal's is, by the user's decision of 2026-10-08 (an any-vote would
have been Embree's `any(near_a < near_b)` scan). Owned by bonsai-58
(src/SSA/{Convert,SortRecursion,Vectorize,QueueRecursion}.cpp and tests);
nothing of it is touched here. The packet cells of the table wait on it.

### First numbers: Embree's packets against its single rays

Before the compiler change there is one measurement to make: what Embree's
own packets are worth over its single rays on this machine, which is the
context the packet table will be read in. (The sweep's logs are
apps/rtq/scratch/packet/embree16/*.log: the `embree` schedule, i.e. bonsai
scalar, against Embree answering in packets of sixteen, side 2048, least
of three, cpu 11; Embree16 / Embree1 follows from each cell's bonsai rate
being the same program as the CSV's `embree` rows.) The probe on head at
side 512 (cpu 13, one repeat) says what to expect:

| head, side 512 | Embree16 incoherent (default) | Embree1 (via bonsai scalar, ~1.0x) | Embree16 coherent flag (side 256) |
|---|---|---|---|
| primary, nearest hit | 53.2 Mr/s, 1.02x of scalar | 52.2 | 70.4 Mr/s, 1.50x |
| diffuse, nearest hit | 13.4, 0.99x | 13.5 | 5.7, 0.45x |
| ao, any hit | 20.9, 1.30x | 16.0 | -- |

With its default flag Embree's packet of sixteen is level with its single
rays on the nearest hit -- the hybrid at threshold 14 is single-ray
traversal below the first few levels -- and 1.3x on the any hit; the
coherent flag's frustum kernel is 1.5x on primary rays and loses more than
half on diffuse rays. Packets of 8 (threshold 7) and 4 (threshold 3)
measured level with or under single rays on the same probe.

The sweep (12 meshes, side 2048, least of three, cpu 11, every packet
answer agreeing with the program's): Embree's rtcIntersect16 and
rtcOccluded16 with the default flag over its rtcIntersect1 and
rtcOccluded1, the single-ray rate taken from the CSV's `embree` rows at
a4cd033e (the bonsai scalar rate measured in both runs agrees to 1-4%, the
drift check), million rays a second:

| mesh | primary E16 / E1 | diffuse E16 / E1 | ao E16 / E1 |
|---|---|---|---|
| pavilion | 85.7 / 76.1 = 1.13 | 29.0 / 27.0 = 1.07 | 68.4 / 30.1 = 2.27 |
| head | 70.5 / 64.5 = 1.09 | 14.0 / 14.2 = 0.99 | 26.5 / 17.9 = 1.48 |
| zero-day | 208.7 / 159.8 = 1.31 | 14.7 / 14.1 = 1.04 | 19.3 / 16.4 = 1.18 |
| bmw | 50.0 / 47.3 = 1.06 | 12.4 / 12.2 = 1.01 | 21.7 / 17.8 = 1.22 |
| crown | 116.6 / 99.6 = 1.17 | 11.1 / 10.8 = 1.03 | 15.1 / 13.0 = 1.17 |
| ivy | 68.9 / 63.6 = 1.08 | 5.0 / 4.9 = 1.01 | 7.0 / 6.7 = 1.04 |
| villa | 38.6 / 36.9 = 1.05 | 11.4 / 11.2 = 1.02 | 19.4 / 15.7 = 1.24 |
| dambreak | 27.9 / 27.2 = 1.03 | 7.9 / 8.0 = 0.99 | 15.4 / 14.4 = 1.07 |
| sportscar | 46.5 / 43.9 = 1.06 | 7.7 / 7.7 = 1.01 | 11.6 / 10.1 = 1.15 |
| landscape | 66.7 / 59.1 = 1.13 | 56.9 / 48.0 = 1.19 | 76.1 / 53.0 = 1.44 |
| ganesha | 25.3 / 24.0 = 1.05 | 6.2 / 6.2 = 1.01 | 7.7 / 7.2 = 1.06 |
| dragon | 36.2 / 34.7 = 1.04 | 4.1 / 4.2 = 0.98 | 5.3 / 5.1 = 1.04 |
| geomean | 1.10 | 1.03 | 1.25 |

So on this machine Embree's packet of sixteen, as shipped and with its
default flag, is worth a tenth on primary rays over its single rays, nothing
on diffuse rays, and a quarter on the ambient-occlusion any hit (where a
terminated lane costs nothing and the short rays keep the packet together
longest; pavilion's 2.27 is its small tree). That is the bar the packet
schedule is measured against: a `packet16` cell at 1.0 is a traversal as
good as Embree's hybrid, and the `embree` cells already say what the same
program does at single rays.

Also probed, since the stack-first order (loopify, then vectorize: each
lane walking the tree for its own ray over a stack of its own, the nodes
gathered -- apps/pbrt's perlane schedule, apps/wosx's packet schedule over
FCPW's four-wide tree) is the other way a gang can traverse: over Embree's
eight-wide layout it does not compile either today, `src/SSA/Linearize.cpp
:1243: hop.inside == NO_BLOCK -- The region of !par_cont_6!sortall leaves
for !par_cont_6!sort4 from both !break_2 and !par_cont_6!sortall!push`
(the schedule is apps/rtq/scratch/packet/perlane/perlane16.bonsai). Reported
to bonsai-58 with the other two; a linearizer bug over the loopified sorted
run of a wide node, not the per-lane form's cost. It is not the matching
schedule in any case -- Embree's packet kernel shares one node across the
packet -- and would be a secondary cell at most.

### The night's split, and the two things found past the vectorizer (2026-10-09)

The user, in the evening: get the packet tracer going over the night --
"sort + vectorize + loopify -> majority-vote packet tracing" -- and report
by morning. bonsai-58 owns the sort/vectorize side (Convert applying sort in
textual order; the gang's sorted run peeled lane by lane with each compare
decided by the majority; the conditioned run's calls each under their own
mask) and had the rtq packet16 schedule through the vectorizer within the
hour, against their in-progress build. What it then hit, both of which are
this side's and are fixed in the pinned worktree (apps/rtq/scratch/wt, off
fc1dd1e6, to be rebased onto their commit):

1. **A loop's counter made per lane by a per-lane exit.** The any hit's
   leaf (`occluded`) leaves its loop over a leaf's blocks, and the loop over
   a block's four slots, at the first hit: `if (decided) break` after the
   lane loop (Trees.cpp's guard_iteration), and in a gang `decided` is each
   lane's own. UniformizeLoops folds such an exit into a live mask, as it
   should, and linearization then froze *every* carried value for the lanes
   that had left -- `select(live, next, old)` -- which is right for a value
   read after the loop (a lane that left on iteration three keeps its own)
   and wrong for the counter: nothing past the loop reads it, the live lanes
   step it alike, and frozen per lane it made the tile's address per lane,
   `const u32[4]x16`, which no backend gathers and no packet traversal
   wants (Embree's chunk loop indexes one block for the packet). The fix is
   ispc's rule for a uniform loop variable under a varying `break`:
   `read_outside_loop` (SSA/Analysis.h) says whether anything past a loop
   reads a carried argument; UniformizeLoops records the ones nothing does
   (`UniformLoop::local`, decided after the preheader is inserted, since
   that renames the carried arguments) and hands the pure latch an
   undefined value for them on the break edges; the divergence analysis
   passes over an undefined incoming value, and at a loop header applies
   its temporal rule only to a header with one latch (several latches are a
   join of the paths inside the loop too -- the loopified traversal's pops
   and pushes -- and stay as they were) and only to an argument read past
   the loop; and Linearize leaves such an argument unfrozen, the break
   sources out of its blend. The uniform-counter test
   (tests/bonsai/ssa/vectorize-uniform-counter.bonsai) shows the two forms
   side by side: `first_past`'s counter an `i32` with `xs[j]` one load,
   `first_index`'s -- the counter being the answer -- an `i32x8` as before.
   Two latent bugs came out with it, both in Linearize.cpp. The blend of a
   join compared values by *name*, and a uniform loop nested in a masked
   region carries the outer value as a header argument under the outer
   name (a leaf loop's `total` under the traversal's), so the freeze was
   skipped as "a select between a value and itself": compared by defining
   block now (definition_block). And a uniform loop that runs under a
   partial mask had no freeze of its own at all: the lanes that are off go
   round it like everyone else and came out of it changed -- the
   vectorize_nested_loops test printed every lane's answer for n + 1 the
   moment its inner loop stopped being divergent. Such a loop now freezes,
   at each latch, every carried value something past it reads,
   `select(mask, next, current)`, exactly what a uniformized loop does
   through its live mask; its counter, read by nothing past it, stays a
   scalar.
2. **Reading a lane's best element out of a Triangle4 tile.** After the
   traversal the program dereferences the per-lane best (`found = *h;
   hits[i] = found[1]`, and the test's `t.p0.x`), which is a per-lane
   reference into an AoSoA tile: a gather of the tile's fields through
   per-lane addresses. CodeGen_LLVM::gather_elements had the struct case and
   the split-vector case but not a vector field widened as a vector of
   vectors, the lanes outermost -- `f32x3x8 is not f32x3 widened` -- which
   the Vector_t visitor lays out as the same struct of component vectors;
   it gathers them component by component now (test
   correctness/llvm/vectorize-gather-nested-vector). For rtq's trace the
   element's fields are dead (`found[1]` is all the program keeps) and
   LLVM's SROA drops the gathers after the fact; the SSA pipeline cannot,
   since the whole option is passed through a block argument that is read
   whole. A pre-existing LLVM bug surfaced beside it: SLP's vectorizeGEPIndices
   asserts (`toScalarizedStructTy: expected unpacked struct literal`) on a
   plain scalar loop reading `tris[which[i]].p.y` out of an array of
   structs with a vector field -- nothing to do with the gang, left open
   below.

The measurement of packet16 waits on bonsai-58's commit, onto which the
worktree's changes are rebased.

### The any-hit packet cell, measured (2026-10-09, 07:55)

bonsai-58's commit had not landed by the small hours (their session sat on
a prompt), so their eight dirty compiler files were snapshotted as they
stood at 07:36 (apps/rtq/scratch/packet/combined/bonsai-58-wip-0736.patch)
and built in the worktree on top of this side's two fixes -- a labelled,
preliminary compiler, to be re-run once the commit is in. With it, the
any hit's packet schedule compiles end to end and agrees with Embree's
rtcOccluded16 on every ray of every mesh. The nearest hit still fails after
the traversal, at the gang's read of a lane's best triangle out of its
Triangle4 tile: three things in the expression layer and the backend --
`Extract::make` and `PtrTo::make` do not type a vector of array references
indexed per lane as a per-lane place (each lane its own tile, through the
reference the traversal kept), CodeGen_Stmt's GEP case refuses such a base,
and CodeGen_LLVM's Extract visitor routes it to `extract_lane` rather than
a gather through the lanes' pointers -- all written up in this section's
first part and left for the day. So the any hit was measured on its own,
under schedules/packet16-anyhit.bonsai (the any hit as packet16.bonsai has
it, the nearest hit and the closest point as embree.bonsai has them; only
its `any` rows are packet cells), side 2048, least of five, pinned to cpu
11 with the GPU sweep paused:

| mesh | rtcOccluded16 Mr/s | packet16 Mr/s | speedup |
|---|---|---|---|
| pavilion | 65.6 | 49.8 | 0.76 |
| head | 26.1 | 25.6 | 0.98 |
| zero-day | 18.7 | 14.8 | 0.79 |
| bmw | 20.9 | 19.1 | 0.91 |
| crown | 14.7 | 13.1 | 0.89 |
| ivy | 6.7 | 4.7 | 0.70 |
| villa | 19.2 | 17.0 | 0.88 |
| dambreak | 15.1 | 11.2 | 0.75 |
| sportscar | 11.5 | 8.8 | 0.76 |
| landscape | 75.0 | 79.2 | 1.06 |
| ganesha | 7.6 | 5.5 | 0.73 |
| dragon | 5.4 | 3.9 | 0.72 |
| geomean | | | 0.82 |

(apps/rtq/results/logs/*.packet-anyhit.occluded.log; the rows are in the
CSV under schedule `packet16-anyhit`, commit `fc1dd1e6+`, and
apps/rtq/plots/rtq-packet.png draws them.) Read against the two tables
above it: Embree's rtcOccluded16 is 1.25x its rtcOccluded1 on these meshes
(the table of 2026-10-08), and the scalar matching schedule's any hit is
1.02x of rtcOccluded1, so the gang's any hit is level with the same
program's scalar any hit, a shade under -- 0.82 x 1.25 / 1.02 -- while
Embree's hybrid gains a quarter over its single rays. That is what a pure
packet traversal costs against a hybrid on rays this incoherent: the gang
visits the union of its sixteen rays' paths to the end, where Embree, with
fourteen or fewer rays still active, puts the node back and traces each
alone (step 6 of the audit), and short ambient-occlusion rays from
neighbouring pixels keep a packet together only at the top of the tree.
landscape, whose ambient-occlusion rays are the most coherent of the set
(the terrain's normals agree), is the one cell over 1; the open meshes
(pavilion, ivy, dambreak) and the two largest (ganesha, dragon), whose rays
scatter the gang's lanes soonest, are the lowest.

What the generated code is (apps/rtq/scratch/packet/anyhit-occ.ll, the
any-hit gang inlined into `occluded_all`, read before the sweep as the rule
asks): no gather and no scatter anywhere in the traversal -- its memory
operations are the node reference and the eight-wide bound vectors loaded
once per node, the tile's four-wide slot vectors loaded once per block, and
two stacks, `[561 x i64]` of node references and `[561 x <16 x i1>]` of
lane masks, with `[8 x <16 x i1>]` for the eight children's masks; the
ray test is sixteen-wide fused multiply-adds (23 `fma.v16f32`, the three
`rcp14.ps.512` of the reciprocal direction) with no per-lane scalar
arithmetic; the only gathers are the eight that read a gang's sixteen rays
out of the array of structs the driver hands over, once per gang, before
the traversal. The rays could be handed over as a struct of arrays, as the
driver already lays them out for Embree; a measurement for the morning,
not a fault of the traversal.

Open for the day, in order: (1) the nearest hit's packet cell (the
per-lane tile read above); (2) bonsai-58's commit, the rebase of the
worktree's fixes onto it, the suite, and the re-measurement of both ray
queries under packet16.bonsai proper; (3) the three steps of Embree's
kernel no directive says -- the hybrid switch above all, which is where
the quarter is -- as a question for the user, since each is a scheduling
construct; (4) the per-lane order (Linearize.cpp:1243) as a secondary
cell; (5) the SLP assertion on a scalar struct-array read, an LLVM-side
bug to reduce and report.

### The any-hit packet cell, profiled (2026-10-09, 10:30)

The user asked why the packet is slower for occlusion, and for the
generated code of both sides. Two meshes, head (the cell at 0.98) and
dragon (0.72), ambient-occlusion rays at side 2048, `perf record` of
cycles, instructions and branch misses on cpu 11 (bonsai-76's GPU profile
pinned elsewhere), the same rays and repeats for every kernel, our kernel
the packet16-anyhit schedule's `occluded_all` and, for the baseline, the
embree schedule's; Embree's two symbols are the chunk loop of
`BVHNIntersectorKHybrid<8,16>::occluded` and its single-ray fallback
`occluded1`. Everything below is under apps/rtq/scratch/packet/prof:
the perf data (head-ao, dragon-ao, scalar/), the per-symbol counts
(symbols.txt), the annotated disassembly of the three kernels (ours-annot,
embree-occluded16-annot, embree-occluded1-annot), our LLVM IR (rtq.ll),
and visits.sh with its output visits.txt.

Per ray, the same count of rays for every kernel:

| | head instr | head cycles | dragon instr | dragon cycles |
|---|---|---|---|---|
| ours, packet16-anyhit | 476 | 236 | 1919 | 1461 |
| Embree rtcOccluded16, both kernels | 392 | 213 | 648 | 1057 |
| -- of which the single-ray fallback | 238 (61%) | 122 (57%) | 545 (84%) | 936 (88%) |
| -- of which the chunk loop | 154 | 91 | 103 | 121 |
| ours, scalar (embree schedule) | 471 | 320 | 620 | 1126 |
| Embree rtcOccluded1 | 485 | 315 | 641 | 1105 |

IPC: ours 2.01 on head and 1.31 on dragon; Embree's fallback 1.96 and
0.58, its chunk 1.69 and 0.85. Branch misses per ray, head: ours 1.5,
Embree 2.3 (the gang's control flow is uniform). So on head we are 1.11x
Embree's cycles at 1.21x its instructions, and on dragon 1.38x its cycles
at 3.0x its instructions, where Embree's single rays are memory bound
(0.58 instructions a cycle, a 7M-triangle tree walked one node at a time,
four prefetches a node) and our gang's one path through it is not.

**Where the time is: node visits.** The instruction samples, summed over
a loop body of known length, say how often the body ran (visits.sh; skid
averages out over 25-45 instructions, take the counts as +-10%). Our
children loop runs eight times a node; Embree's chunk tests the eight
children of a node in a loop too; its fallback tests a node's eight boxes
in one eight-wide step per ray. Per sixteen rays:

| per 16 rays | head | dragon |
|---|---|---|
| ours: nodes the gang visits | 7.9 | 30.3 |
| Embree: nodes the packet visits in the chunk loop | 2.6 | 2.1 |
| Embree: nodes the sixteen rays then visit one at a time | 20.8 | 109 |

and what a visit costs, counted in the generated code: ours 43
instructions a child, 344 a node, plus about 70 after the loop (below);
Embree's chunk 31 a child plus about 12 for each child hit; Embree's
single ray 26 a node plus the push. So on head the gang's seven visits
cost about 2700 instructions in node tests where Embree's 2.6 + 20.8 cost
about 1200, and on dragon 10,400 against 3300 -- the 3.1x of the table.
The union of sixteen paths is shorter than their sum (7.9 against 23.4
node steps on head, 30 against 111 on dragon), but a step of the gang
costs thirteen single-ray steps, so it pays only while most lanes are on:
at 26 instructions a lone ray's node against 344 for the gang's, the gang
breaks even at about ten active lanes, and Embree switches at fourteen.
Dividing Embree's steps by our union gives the lanes that are useful at
one of our nodes, roughly: eight of sixteen on head, five on dragon.
That is the dominant cause, and it is the hybrid switch (step 6 of the
audit) that answers it, as Embree's own split says: 57% of its packet
time on head and 88% on dragon is spent tracing rays one at a time.

**The generated code, ours against Embree's chunk loop, per child.** The
IR (rtq.ll, `!par_body_5.i`) and the assembly (ours-occluded.asm,
0x24af0-0x24bfd) say where the 43 go against Embree's 31
(embree-occluded16.asm, 0x394b20-0x394bf7):

1. *The child's bounds.* We load the node's six eight-wide bound vectors
   once per node and read the child's with `extractelement <8 x float>
   %v, i32 %child` -- a variable index, which LLVM lowers through the
   stack, inside the loop, after any hoisting: six `vmovaps ymm ->
   stack` and six `vbroadcastss (stack)` per child, 12 of the 43. Embree
   broadcasts from the node, `vbroadcastss 0x40(%rax,%rdx,4)` and
   `vfmsub132ps 0x60(%rax,%rdx,4){1to16}`, 3 instructions and 3 memory
   operands. Ours to fix: a one-element read at a variable index of a
   vector that was loaded is a load at the element's address.
2. *The slab test's form.* Ours keeps the single-ray form, the plane
   chosen by the direction's sign: three `vcmpltps` (the sign tests,
   recomputed every child because all eight mask registers are live),
   three `vblendmps` and three masked `vmovaps`, then six FMAs and six
   integer min/max -- 21. Embree's chunk computes both planes and takes
   their min and max (`intersectNodeK`): six FMS with the broadcast
   folded in, six `vminps`/`vmaxps`, four integer min/max with the near
   and far bounds, 16, no select. Its single-ray kernel uses the
   sign-chosen form through precomputed near/far offsets, as our scalar
   schedule does. A question for the user: the packet's slab test
   written as Embree's packet kernel writes it, under the ruling that
   the slab test is on bits.
3. *The children's masks.* Each child's test is written into a per-lane
   `bool[8]` that lives in memory, `[8 x <16 x i1>]`, as a masked
   read-modify-write: `kmovw` load, `kandnw`, `korw`, `kmovw` store, 4
   a child -- 16% of our samples sit on that `korw`/`kmovw` pair, part
   of it the skid of the compare chain before them -- and after the
   loop the eight masks are loaded back, and-ed with the active mask
   again (the array holds stale lanes from earlier nodes), spilled to a
   second slot because eight masks, the active mask and the running
   union do not fit eight `k` registers, tested one by one for a lane on,
   and pushed one by one in reverse (0x24c03-0x24dfb): about 70 a node.
   Embree keeps no per-child mask: a child any lane hits is pushed in the
   loop as it is found, with the sixteen distances as the entry
   (`vmovaps %zmm0,(%r12)`), and the lanes come back at the pop as
   `curDist < tfar` (`vcmpltps 0x440(%rsp),%zmm1,%k0`). Ours to fix for
   the unsorted run: loopify's lowering of a gang's run with no sort can
   push each child inside the children loop as its test completes, as
   Embree's does, and the array goes away; for the sorted run the chain
   needs the eight tests first, and whether the entry carries distances
   (Embree) or a mask (ours) is the design question already open.
4. *The leaf.* 28% of our cycles, 0x24e40-0x25024: a tile's four slots in
   a loop, each slot's triangle broadcast from the four-wide block (no
   gather), the Moller test sixteen wide, the t-range skipped when no lane
   passes the barycentric test (the `!else_19` branch). Two things Embree
   does that we do not: it narrows the slot's valid lanes by the hits so
   far and leaves the tile when none are left (`occludedK`, `if
   (none(valid0)) break`), and it stops at the first empty slot where we
   test each slot's validity word. On these rays hits are rare (3.6% of
   head's ambient-occlusion rays are blocked, 13.6% of dragon's), so this
   is small here; it is ours to fix for the any hit all the same.
5. *The rays.* Eight `vgatherdps` per gang read sixteen rays out of the
   array of structs; Embree is handed a struct of arrays. Eight
   instructions in about 900 per gang on head; the measurement stands
   from the morning's list.

What the two kinds of cost would return, by the shares above: our per-node
cost taken from about 414 instructions to Embree's chunk's roughly 290 by
items 1 and 3 moves head from 0.98 to about 1.15 and dragon from 0.72 to
about 0.85, since node processing is 57% of our cycles (children loop
43%, the masks and pushes after it 14%); the rest of dragon's gap, and
most of the geomean's, is the hybrid switch.

Done this morning besides: the worktree's compiler change is committed
and cherry-picked onto ajr/ssa as 761f13d1 (the uniform counter, the two
Linearize fixes, the nested gather), after bonsai-58 confirmed nothing of
theirs was building; the ptr-arena packet test stays out until the
nearest hit's tile read works.

For the user, in order: (a) the hybrid switch -- a traversal that, at a
pop with fourteen or fewer lanes on, traces each lane's node with the
scalar traversal and returns to the gang's loop after; a scheduling
construct, so a question first; (b) the stack entry's contents, distances
or a mask (item 3); (c) the packet's slab test in the min/max form (item
2). Ours without asking: items 1, 3 (the unsorted push), 4 and 5, then
the nearest hit's tile read and the packet16 cells proper.

### The hybrid switch as a specialize (2026-10-09, the user's question)

The user asked whether the schedule can say the single-ray fallback, and
whether it is a `specialize()`: "something like `specialize(count >=
14).vectorize(forloop).loopify().else().loopify().vectorize(leaf)`", and
how Halide represents the if/else. Halide's `Func::specialize(cond)`
(src/Func.h): it makes a copy of the stage's definition whose schedule
"inherits every directive made about the parent Func's schedule so far",
returns a handle to that copy, and the directives written on the handle
apply to the copy alone; the condition is a boolean over the pipeline's
parameters and constants, tested once around the stage's loop nest, so
the generated code is `if (cond) { the copy's nest } else { the
parent's }`; several specializations of one stage are tested in the order
written, an if / else-if chain whose final else is the parent as it is
scheduled -- directives written on the parent after the specialize apply
to that else alone; a specialization may itself be specialized, a nested
if; and `specialize_fail(message)` ends the chain with a call to
halide_error in place of the else. Our `specialize()` is that shape
already (SSA/Specialize.h): a copy per variant or bool of a parameter,
the copies named `f[Label]`, dispatched where the loop is entered, the
directives after it addressed to one copy or to all.

What Embree's switch is in those terms, and the one thing it adds. Its
condition is not a parameter: it is the count of lanes still on at a node
of the gang's traversal, which changes at every node, so the test cannot
sit once around the loop nest -- it sits at the recursive call, per node.
And its two arms are the two schedules of the same recursion the user
wrote: above the count, the rays as a gang over one node (the packet:
`vectorize(i_lane)`, then the sort and `loopify` over one stack), and at
or below it, each lane's ray alone from that node (the matching
schedule: `loopify`, `vectorize(children)`, `prefetch`), the gang's loop
resuming after. That is Halide's `f.specialize(width >= 8).vectorize(x,
8)` -- the stage vectorized under a condition on the loop's extent, the
scalar loop otherwise -- with the extent being the lanes that are live,
which is a quantity the schedule can name once the loop is a gang's, and
with the test repeated at each call of the recursion because the extent
shrinks as lanes finish. So: yes, a specialize, written on the recursion
with a condition over the live count of the vectorized loop:

    occluded_all.split(i, i_gang, i_lane, 16, true).vectorize(i_lane);
    occluded.specialize(count(i_lane) > 14);     // at each recursive call
    occluded[True].loopify(561);                  // the gang: one node, one stack
    occluded[False].loopify(564)                  // each live lane alone, the matching schedule
                   .vectorize(triangles.Interior.children)
                   .prefetch(triangles.Interior.children, 4);

(`count(i_lane) > 14` is Embree's `popcnt(valid) > switchThreshold` for
K = 16; the nearest hit adds its sort to the `True` arm and the `False`
arm keeps the matching schedule's sort.) What the compiler would do with
it: the vectorizer makes the gang copy of `occluded` as now; at the gang
copy's recursive call it tests `popcount(mask) > 14` and, under the
condition, calls the gang copy (which loopify then puts on the gang's
stack); otherwise it runs a loop over the lanes that are on and calls the
scalar `occluded` for each with that lane's ray and the node -- the
scalar function as its own directives schedule it, which exists already,
since a vectorized function's scalar original is kept for whoever calls
it that way -- and continues. Embree's SWITCH_DURING_DOWN_TRAVERSAL is
the same test at the same place. Three things are new to the language,
each small: a condition that is an expression over a loop's live count
rather than a parameter's variant; a specialize whose test is at a
recursive call rather than a loop's entry (the loop form,
`run.specialize(j, flag)`, is the nearest existing one); and the `False`
arm meaning "the scalar function, once per live lane", which is what the
unvectorized parent is for a loop that was vectorized. A new construct,
so it waits for the user's word on the syntax; the stack entry's
contents (distances, Embree's, or a mask, ours) come with it, since the
fallback starts each lane at the node it popped with whatever the entry
holds.

### The child's box read off the node (2026-10-09, 11:45)

Item 1 of the profile, done: SSA/Simplify.cpp's ExtractIdx rule now
takes a run-time lane of a vector of vectors -- `lo[i]` with `lo` the
node's eight boxes and `i` the children loop's counter -- as the lane of
each component, and the rule it already had for a lane of a stored
vector turns each into a load at the lane's address (the commit in
apps/rtq/scratch/wt, d5e79877: a vector built from its components, the
layout's lower_x, lower_y, lower_z, and a vector of vectors stored whole
and packed, whose components follow one another). The any-hit packet's
children loop went from 43 instructions a child to 35 (scratch/packet/
prof/anyhit-new.asm): six `vbroadcastss 0x40(%r12,%rax,4)` straight off
the node, no store, where Embree's chunk loop has 31. What is left in the
loop against Embree's is item 2 (the sign tests and blends, 9 of the 35)
and item 3 (the mask read-modify-write, 4, and the reload, re-and and
eight pushes after the loop). rtq's scalar schedules are unchanged, the
same LLVM, since their children are vectorized and read the vectors
whole.

Measured, head and dragon, side 2048, least of five on cpu 11 (scratch/
packet/rule1/*.log; the morning's cells in brackets):

| mesh | rtcOccluded16 Mr/s | packet16 Mr/s | speedup |
|---|---|---|---|
| head | 26.55 | 26.00 | 0.98 (0.98) |
| dragon | 5.49 | 4.22 | 0.77 (0.72) |

Every ray agrees. Dragon gains 8% in our rate (3.89 to 4.22 Mr/s) and
head 1.6%, Embree's own rates having moved by 1 to 1.5% between the two
runs: the stack round trip cost most where the gang visits most nodes
per ray (30 a gang on dragon against 8 on head). Tests: ssa and
backends/llvm child-bounds-at-lane, correctness/cpp/bvh4_ptr_arena_
children; nineteen ssa and seventeen LLVM goldens re-blessed, each diff
read (the lane extracts of a node's boxes and a tile's vertices now
loads at their addresses marked invariant; the vectorized-children
tests read the components as dense loads off the row where they loaded
the row whole; nothing else but numbering). Suite 1506 of 1506 in the
worktree.

### The push moved into the children loop, three ways (2026-10-09, 12:20)

Item 3 of the profile: the gang's run of conditioned calls wrote each
child's test into a per-lane `bool[8]` in memory (a masked
read-modify-write a child), read the eight back after the loop, and-ed
them with the mask again, tested each for a lane on and pushed each
behind its test, about seventy instructions a node after the loop. The
rewrite (SSA/QueueRecursion.cpp, fuse_into_children_loop) recognizes
that shape exactly as the vectorizer and the tree lowering make it --
every call's mask `M && A[k]`, A written once in a loop at the loop
header's own counter, the children the lanes of one vector -- and moves
the push to where the test is made. Three forms were built and measured,
dragon, pinned, two rounds alternating with the same driver object
(scratch/packet/item3/ab.sh, ab.txt), against the state after item 1:

| form | dragon Mr/s | our kernel's instructions | branch misses |
|---|---|---|---|
| item 1 alone (the array, pushes after the loop) | 4.27, 4.26 | 4.19e9 | 2.22e7 |
| Embree's chunk loop: push behind `if any(lanes)`, the last hit kept as current | 3.99, 3.98 | 3.78e9 | 2.03e7 |
| the same with the children in reverse (lowest first off the stack) | 4.02, 3.97 | 3.97e9 | 2.03e7 |
| branch-free: the slot written every child, the count moved by `any` | 4.48, 4.41 | 3.70e9 | 0.82e7 |

The first lesson cost an hour: Embree's own loop shape lost 7% with 10%
fewer instructions and fewer mispredictions. Its branch on the child's
test is decided off the end of the box test's chain, some twenty-five
cycles into the child, so a misprediction there is found late and costs
most of a child's work in flight; the array form branched after the loop
on values long since ready. The hottest instructions of that kernel were
the first two at the hit branch's target (10% and 9% of its cycles).
Embree pays that too, but its chunk loop runs at the top of the tree
only. The order of the children -- last hit first, as Embree's chunk
loop has it, against lowest first, as its single rays and our lowering
have it -- made no difference. The branch-free form is what a pure
packet wants: the child's index (a scalar load off the row) and the
lanes that hit it are stored at the stack's top whether or not a lane
did, and the count moves on by `cast<u32>(any(lanes))`; the stack's
slack past its capacity takes the write past the last push. LLVM unrolls
the loop by two of its own accord once it is straight-line. The children
go in reverse so that the pops come lowest first, the order the traversal
had, since a lowering should not change a traversal's order; it measured
the same either way.

The cells, side 2048, least of five on cpu 11 (scratch/packet/item3/
*.log; before item 3 in brackets, the morning's in the second pair):

| mesh | rtcOccluded16 Mr/s | packet16 Mr/s | speedup |
|---|---|---|---|
| head | 26.42 | 31.12 | 1.18 (0.98, 0.98) |
| dragon | 5.48 | 4.42 | 0.81 (0.77, 0.72) |

Every ray agrees. Head gains 20% in our rate -- a small tree whose gang
visits eight nodes a sixteen rays, so the seventy instructions a node
after the loop were a third of its traversal -- and dragon 4.7% on top of
item 1's 8%. Per child the loop is the box test, two stores, a `kortest`
and a `setne`: what is left against Embree's chunk loop is item 2, the
sign tests and blends. Tests: ssa/vectorize-packet-any and backends/llvm/
vectorize_packet_any (the four-wide tree's `any` query in a gang of
eight: the push in the loop, no branch, no array, no pushes after), and
correctness/cpp/bvh4_child_volumes_packet runs the same lowering against
the scalar answers. The sorted run is untouched: its order needs every
test first, and its goldens did not move.

The twelve meshes again, after both fixes (results/logs/*.packet-anyhit.
occluded.log, the morning's at scratch/packet/anyhit-logs-0755; side
2048, least of five on cpu 11; the morning's cell in brackets):

| mesh | rtcOccluded16 Mr/s | packet16 Mr/s | speedup |
|---|---|---|---|
| pavilion | 67.70 | 57.78 | 0.85 (0.76) |
| head | 26.42 | 30.64 | 1.16 (0.98) |
| zero-day | 19.27 | 20.02 | 1.04 (0.79) |
| bmw | 21.45 | 24.61 | 1.15 (0.91) |
| crown | 15.17 | 18.27 | 1.20 (0.89) |
| ivy | 6.94 | 6.60 | 0.95 (0.70) |
| villa | 19.92 | 21.92 | 1.10 (0.88) |
| dambreak | 15.45 | 15.04 | 0.97 (0.75) |
| sportscar | 11.78 | 11.26 | 0.96 (0.76) |
| landscape | 73.87 | 108.87 | 1.47 (1.06) |
| ganesha | 7.71 | 6.83 | 0.89 (0.73) |
| dragon | 5.50 | 4.42 | 0.80 (0.72) |
| geomean | | | 1.03 (0.82) |

Every ray agrees on every mesh. The pure packet is now level with
Embree's hybrid on the set, ahead on the coherent and the mid-sized
meshes (landscape by half, crown by a fifth) and behind on the open
ones and the two largest, where the gang's lanes thin out soonest and
Embree's single rays take over: the shape the profile predicted, with
the two code-quality costs paid. What remains is the switch itself,
which is the user's question of specialize.

### The hybrid switch, tried and dropped (2026-10-09 evening)

The user kept this language's specialize notation for it, and the whole
hybrid was built and measured in one evening: `occluded.specialize(count(
i_lane) > 14)`, a specialize on a gang's state, tested at every call of
the gang copies of occluded's recursion (SSA/Specialize.h,
specialize_gang: the gang copy a dispatcher on the popcount of its mask,
the arms `occluded[True]` and `occluded[False]` copies of the gang's
traversal as it stood, each re-testing at its own entry, so that a
loopified arm decides at every pop, where Embree decides); and a second
directive, `occluded[False].serialize(i_lane)`, ispc's foreach_active,
which ran the False arm's lanes one at a time through a scalar copy of the
traversal scheduled as embree.bonsai's occluded. Correct on every ray of
every mesh; slower than the pure packet everywhere but dragon: geomean
0.83 against the pure packet's 1.03 over the twelve meshes, head 0.80
against 1.16, landscape 0.94 against 1.47, dragon 0.89 against 0.80. The
profile on head said why: our fallback visited fewer nodes than Embree's
single-ray fallback (4.9M against 6.2M) and still cost 1.14x its
instructions, because an entry below the switch does about two scalar
node visits and each lane paid a full ray setup per entry (the inverse
direction with its Newton step, the signs, the origin products), which
Embree's occluded1 reads out of a TravRayK precomputed once per packet.
LICM cannot hoist it, since the lane varies at every pop, and LLVM's
vector passes scalarize rather than widen a single-lane extract; the fix
would have been a memoizing rewrite of our own (a lane of a pure function
of the gang's arguments is that lane of the gang-wide function) behind an
SSA-level inline of the scalar copy. The user's ruling: a bad rabbit
hole. The serialize directive, the hybrid schedule
(packet16-hybrid-anyhit) and its logs were removed; the specialize on a
count stays, with its tests (ssa/specialize-count, error/specialize-count-
no-gang), as a schedule point over a gang's occupancy with both arms gangs.
The any-hit packet cell of the morning (1.03) stands as the result.

## The queries the walks make, isolated: FCPW's scalar trees and its GPU (2026-10-25)

apps/wosx's walks run over FCPW's scalar build (enoki off, the build WoSX
ships): the walk on spheres over its binary BVH, the walk on stars over
that and the normal cone hierarchy, where it casts its rays and finds its
star radius by the closest silhouette point. This table had measured FCPW
only as its vectorized Mbvh (fcpw4w16), on deps/fcpw, an older checkout
than the one under deps/wosx; and the walk on stars' profile had our
silhouette query cheaper than WoSX's and our closest point and ray cast
dearer, from perf samples with libm split by guess. The user's ruling: no
change to the traversals until isolated benchmarks say better or worse,
on the CPU and on the GPU, the silhouette query included.

### What was built

- `schedules/layouts/fcpwb.bonsai` and `fcpw_bvh.h` moved here from
  apps/wosx (the binary tree in FCPW's bytes, the copy of FCPW's tree into
  it), and the silhouette query with them: its elements (`Edge`,
  `ConeBox`, `SilhouetteQuery`, once the walk's `StarQuery`) into
  elements.bonsai, FCPW's cone and edge arithmetic into
  metrics/fcpw_scalar.bonsai's silhouette section with FCPW's record of the
  closest point on the edge (`SilhouetteHit`, which the leaf computes once
  for the test, the key and the record), the query into silhouette.bonsai
  with its extern set, which wost.bonsai imports; the normal cone
  hierarchy's rows written by `fcpw_snch.h` (templates over the generated
  row structs, since each program spells them differently), which
  apps/wosx's fcpw_stars_trees.h calls for its reflecting pair.
- `schedules/fcpwb.bonsai`: the three queries over the binary tree as
  bvh.inl's traversals run them (both children tested at the parent and
  written out with `unroll`, the nearer first, the far-corner rule for the
  closest point, a stack of sixty-four, the leaf's Voronoi exits); the
  driver's RTQ_FCPW_BINARY builds FCPW scalar. `schedules/layouts/
  fcpws.bonsai` and `schedules/fcpws.bonsai`: the SNCH as two trees over
  one node array of GPUSnchNode rows (the triangles the rays and points
  query, the edges the silhouette query searches, the driver handing the
  same node rows to both), `ConeBox` the bounding element whose metrics are
  the cone test for the silhouette query and the box alone for the rest;
  the fourth query's traversal ordered by the box distance alone with the
  cones tested at the parent; snch.bonsai the program with the silhouette
  batch; the driver's RTQ_FCPW_SNCH computes FCPW's silhouettes first,
  leaving out the convex edges WoSX's reflecting handler leaves out
  (getIgnoreCandidateSilhouetteCallback for a single-sided problem), and
  asks `findClosestSilhouettePoint` about each point of the two point sets
  within a tenth of the diagonal, no minimum radius so that both sides
  search to the exact closest (the walk hands FCPW its reflecting shell
  and FCPW stops at the first silhouette within it; apps/wosx/PLAN.md,
  "The silhouette query read against FCPW's"), WoSX's precision and normal
  flip; the agreement the same edge or none on both sides, a tie where two
  edges are equally far, the records' distance and point.
- `schedules/fcpwb-gpu.bonsai`, `schedules/fcpws-gpu.bonsai`: each batch's
  loop split by 256 and bound to the GPU's blocks and threads (one launch,
  one query a thread, FCPW's group size), the traversals bvh.slang's (the
  closest point by d2Min alone, as the kernel orders it); the reference
  FCPW's GPUScene, run by the driver under RTQ_GPU (`fcpw_gpu_reference.h`:
  transferToGPU's setup step for step on the CUDA backend; FCPW's batch
  calls taken apart so that the upload of the queries and the read of the
  answers are off the clock and the dispatch and the wait are what is
  timed, FCPW's own runShader), and on the program's side the buffers
  staged on the device before the clock with a copy inside it forbidden
  (runtime/bonsai_buffer.h), the clock around the launch and the wait.
- compare.sh: `metric_of` (fcpw_scalar for the scalar trees), `program_of`
  (snch.bonsai for fcpws), `fcpw_dir_of` (WoSX's FCPW for the scalar
  trees: a later checkout whose bvh.inl differs from deps/fcpw's), the
  -gpu rule with slang-rhi's headers and libraries as apps/wosx's, and
  RTQ_SKIP_CMAKE=1 to run the compiler as built while a peer's measurement
  owned build/. benchmark.sh runs the scalar pair and the GPU pair as
  sections of their own per mesh under the sessions' lock; tocsv.sh reads
  the new names and the silhouette rows (reference `fcpw-gpu` for the GPU
  cells).

### The arithmetic, to the bit

metrics/fcpw_scalar.bonsai's dot products and norms are Eigen's as the
build FCPW ships compiles them, measured rather than read
(apps/wosx/PLAN.md has the measurement): `dot` sums `(p0 + p1) + p2` with
the newer product fused, `squaredNorm` and the box distances `p0 + (p1 +
p2)` with the left one fused, both written as `fma` chains. On the head
every ray, point and silhouette answer agrees with FCPW's scalar build up
to ties, the hit's t, uv and point and the silhouette's distance and point
to the bit (0.0e+00 in the records columns); the one field off is the hit
record's unit normal (1.7e-07), Eigen's `normalized()` under clang's SLP
pass, a record field and never a key. Against FCPW's GPU the records are
within 4e-6 (the GPU compiler's own fusing) and the answers agree up to
ties.

### The miss record, and what the first GPU sweep taught

The first GPU sweep had pavilion's million primary rays at 0.62x while
every other row read 1.03x to 1.37x. The batches wrote a record for every
query -- `hits[i] = no_hit()` on a miss, forty bytes with the invalid id
and an infinite distance -- so that the driver could compare answers
without initializing anything. Neither reference does that inside its
call: rtcIntersect1 leaves a missed ray's RTCRayHit as the caller set it
(geomID invalid beforehand), FCPW's intersect returns false and leaves its
Interaction as constructed, and FCPW's GPUScene uploads the interactions
pre-filled and its kernel writes only hits. On the CPU the extra stores
cost nothing visible; on the GPU, pavilion's primary rays hit 16% of the
time, so our kernel stored some 35 MB that FCPW's never touched, inside
our clock and outside theirs. The batches now write only hits and the
driver fills the records with the miss record off the clock -- before each
run on the CPU, as it resets the references' structs; once before the
first run on the GPU, as FCPW's buffer is uploaded once, since a run
writes only hits and every run leaves the same records. Pavilion's primary
row went to 1.36x, the other rows unchanged within noise. (Copying the
records back and up again before each run, tried first, read every row
10-30% slower: tens of megabytes through the card between runs, where
FCPW's side runs with whatever the last run left in the card's caches.)
The user's rule, restated: no more work than the reference, and the same
stores on both sides. The Embree and fcpw4w16 rows of the table were
measured with the miss stores in the program; on the CPU they are a
fraction of a percent at most, but the rows are to be measured again
under the new batches before the figures are redrawn, so that every row
is the same program.

The same sweep showed two kinds of GPU disagreement to be read against the
CPU rows: on ivy, FCPW's GPU closest point answers with a triangle whose
distance the driver computes as NaN -- a degenerate triangle, where ours
answers the finite neighbour (23 and 27 of 26110 points; the CPU rows
agree to the bit) -- and on sportscar two of 146599 silhouette queries
classify an edge differently (FCPW's edge at 0.10 against ours at 0.20 of
a radius of 0.52; the cone and edge tests go through `acos`, `asin` and
`atan2`, whose GPU implementations differ between Slang's and ours).

### Measured (2026-10-25, 16:04-16:26, the lock held)

The thirteen meshes at side 1024, least of five after a warm-up, pinned to
cpu 11 (the host thread for the GPU cells); lte-orb left out of the
geomeans as the other figures leave it out (FCPW finds no hit on it).
Geomeans of bonsai over the reference, with the least and greatest cell
(apps/rtq/plots/rtq-fcpw-scalar-cpu.txt and -gpu.txt have every cell):

    over FCPW's scalar build (CPU)         geomean   min            max
      first hit, primary rays   (fcpwb)      0.81    0.59 pavilion  0.98 zero-day
      first hit, diffuse rays   (fcpwb)      0.88    0.58 pavilion  1.08 ivy
      any hit, ambient occlusion (fcpwb)     0.86    0.56 pavilion  1.08 ivy
      closest point, near       (fcpwb)      1.10    0.82 pavilion  1.21 ivy
      closest point, volume     (fcpwb)      1.21    1.11 landscape 1.37 dambreak
      silhouette, near          (fcpws)      1.14    1.00 pavilion  1.47 ivy
      silhouette, volume        (fcpws)      1.24    1.02 crown     1.60 landscape

    over FCPW's GPUScene (GPU)
      first hit, primary rays   (fcpwb-gpu)  1.19    0.97 landscape 1.39 head
      first hit, diffuse rays   (fcpwb-gpu)  1.33    1.19 dragon    1.48 head
      any hit, ambient occlusion (fcpwb-gpu) 1.35    1.19 dragon    1.56 landscape
      closest point, near       (fcpwb-gpu)  1.16    1.09 villa     1.31 pavilion
      closest point, volume     (fcpwb-gpu)  1.08    1.01 landscape 1.18 ivy
      silhouette, near          (fcpws-gpu)  1.22    1.06 crown     1.83 ivy
      silhouette, volume        (fcpws-gpu)  1.25    1.03 pavilion  1.92 ivy

What it says. On the CPU, against FCPW's scalar binary tree -- the tree
the walks traverse -- the ray queries are behind (0.81-0.88 by geomean,
pavilion and landscape at 0.56-0.59, the large meshes at 0.86-0.99), and
the closest point and the silhouette query are ahead (1.10-1.24). That
inverts the fcpw4w16 table, where the rays over FCPW's vectorized tree
were ahead: FCPW's scalar ray traversal, Eigen scalars and fused
multiply-adds over a 32-byte node, is the one our two-child visit does
not yet beat, and pavilion -- 11366 triangles, a tree that fits the
caches -- is where the visit's instruction count shows undiluted by
memory (apps/wosx/PLAN.md, "The binary CPU walk's gap", has the machine
code side by side: the `vec3f` loads and horizontal sums). It also reads
the walk on stars' profile right: the silhouette query was the cheaper
side there and the ray cast the dearer, as here; the closest point, 1.10-
1.21 here on points, was dearer in the walk, whose query is a bounded
sphere with WoSX's radius rule rather than an unbounded point. On the GPU
every query is ahead of FCPW's shaders on every mesh but landscape's
primary rays at 0.97, the silhouette query by 1.22-1.25 and up to 1.9 on
ivy.

Agreement. On the CPU every ray answer and every silhouette answer agrees
with FCPW's up to ties, the records to the bit; the closest point too,
once the driver's tie test took a degenerate triangle as the segment it is
(the first fold had 21-130 "differ" on ivy and dambreak, every one a
triangle with two corners coincident, whose distance Ericson's regions in
double divided by zero on). On the GPU the arithmetic is the GPU
compiler's on each side, so the records agree to 1e-5 rather than to the
bit, and a few rays in a million (bmw 6, villa 7, dragon 11, ganesha 11,
landscape 15, sportscar 3, crown 1, ivy 2) are grazing hits -- a
barycentric within 1e-5 of an edge, one side hitting and the other
missing, both ways -- and on sportscar two silhouette edges of 146599, on
crown and landscape one, classify differently through the GPU's `acos`,
`asin` and `atan2`.

### The machine code behind the table (2026-10-25, evening)

The user's questions: why the first hit fell so far behind on the CPU
when the three older figures had it level or ahead; whether the CPU cells
are one thread; why the GPU is ahead everywhere; and whether `acos`, `asin`
and `atan2` compile the same on both GPU sides. Read off the objects
(apps/rtq/scratch/binary: ours-trace_all.asm, fcpw-ray-subtree.asm; the
PTX and SASS of our four kernels and of FCPW's closestSilhouettePoint,
FCPW's compiled with slangc -target cuda and nvcc, ptxas -v and cuobjdump
on both).

A different reference program, not a worse tree. The older figures are
against FCPW's vectorized Mbvh -- four-wide nodes, sixteen-triangle
leaves, enoki's AVX-512 kernels -- with our schedule vectorizing the same
loops; this one is against FCPW's scalar build, a binary BVH of 32-byte
nodes and leaves of at most four triangles, Eigen scalars one box and one
triangle at a time, and a later checkout. Both sides one pinned thread on
the CPU (cpu 11, a plain loop each), one query a thread in blocks of 256
on the GPU.

The CPU ray visit. Static size is not it: our whole trace_all is 287
instructions, visit, leaf and record; FCPW's subtree loop alone is 423
with a call per triangle. The structures differ. FCPW decides the two
children's hits, their order and the push with branches (48 branches, no
cmov, 8 compares in the loop), which the predictor runs ahead of on
coherent rays; ours decides them through mask registers and conditional
moves (vcmpleps, kandw, kmovd twice, kshiftrb, vblendmps; four cmova, one
of them the next node), a data dependency through the box tests' full
latency on every visit. Beside it, 36 shuffles to handle the three-lane
vectors (vpermt2ps 7, vinsertps 8, vmovshdup 4) where Eigen loads scalars,
and the wide metric's "proper box" compare kept in the scalar metric
(vinsertps twice, vcmpleps, kandw a visit) for a box that is never the
inverted empty slot here. Primary rays on a cache-resident mesh are where
a dependency chain shows undiluted: pavilion 0.59. The leaf goes the other
way: FCPW's CPU reads each triangle's corners through the soup's indices
and dispatches `intersect` virtually (the subtree loop's six indirect
calls), ours reads FCPW's own 40-byte GPU rows inlined -- a layout
advantage on our side for the CPU comparison, to be said plainly: the
fcpwb layout is FCPW's GPU bytes, not its CPU soup, and the exactly
matching CPU layout would read through the soup, which the layout
language does not express (apps/wosx/PLAN.md, the layout extension). That
is why the leaf-heavy queries, closest point and silhouette, are ahead
while the node-heavy rays are behind. The branch-against-mask reading is
a hypothesis from the code; the measurement that decides it is perf stat
on pavilion's primary rays on both sides (cycles, instructions, branch
misses), and the experiment that fixes it, if it holds, is a visit whose
decision is branchy -- a codegen or schedule question for the user, no
change made.

The GPU. The two silhouette kernels are nearly the same arithmetic: FFMA
233 against 248, MUFU.RCP 21 against 21, MUFU.RSQ 11 against 11, eighteen
pi/2 constants and seven copysigns each, 1232 against 1168 SASS
instructions, no spills (ours 64 registers and a 512-byte stack, FCPW's 52
and 256). The one systematic difference is the loads: FCPW's tree reads
are 38 generic-address loads (LD.E, Slang's buffer pointers) and 21
global; ours 43 read-only global loads (LDG.E from `ld.global.nc`), the
texture-cache path. No branch predictor on a GPU, so the visit structure
that costs on the CPU costs nothing there. The load path is the likely
source of the 1.1-1.4x; confirming it is a profiler on the two kernels,
which the user's rule reserves to their say-so.

acos, asin, atan2: the same algorithms in single precision on both GPU
sides -- FCPW through Slang's CUDA prelude (`::acosf`, `::asinf`, the
float overload of `atan2`; slang-rhi's NVRTC path passes no fast-math
flag), ours through libdevice's `__nv_acosf`, `__nv_asinf`, `__nv_atan2f`
inlined (CodeGen_PTX::codegen_math_call; --fast-math would switch only
sin, cos, exp, log and pow to their approximate forms, as nvcc's
--use_fast_math does, and compare.sh does not pass it). The SASS counts
above are the evidence. On the CPU both sides call glibc's.

### Genuine apples-to-apples: the bytes FCPW's CPU reads (2026-10-25, evening)

The user's ruling: the comparison means nothing unless it is the same
algorithm over the same tree in the same layout on the same schedule, and
"our rows are FCPW's GPU layout" was already one case where it was not.
The audit of the rest, and what was done about each:

- The leaf layout (the one above). FCPW's CPU leaf reads `primitives[k]`,
  a `Triangle*`, the three indices out of the object and the corners out
  of `soup->positions` by index; our CPU layout held FCPW's GPUTriangle
  rows, the corners stored. Fixed: layouts/fcpwb.bonsai and fcpws.bonsai
  are now FCPW's own arrays -- flatTree (SnchNode in its CPU field order,
  which differs from GPUSnchNode's in the last twelve bytes), the pointer
  arrays, the Triangle and SilhouetteEdge objects reached by address
  through a `ptr group`, the positions -- wrapped where they lie by
  fcpw_bvh.h's wrap_fcpw_bvh and fcpw_snch.h's wrap_snch (nothing copied;
  the scene outlives the tree; every reference checked to point into the
  one object vector and every object to index the one soup). The tile
  stores the pointer and derives the rest: `t = objects[tri]`, `pa =
  positions[t.i0].p`, `e1 = pa - pb`. That needed the layout language to
  let a tile's derived field read another named group, by index or by
  address (commit 75282086: the parser puts a named plain group in the
  layout's scope as the array of its row struct; Lower/Layouts.cpp's fill
  binds the read to the group's storage; a derived member may be a step
  towards the element's fields rather than one of them). The GPU bytes,
  which FCPW's GPU reads, are fcpwb_gpu.bonsai and fcpws_gpu.bonsai, for
  the -gpu schedules and -- for now -- apps/wosx (its switch is pending).
- The edge's two face normals. FCPW's SilhouetteEdge::normal(0/1)
  recomputes them from the positions at every two-face test (the cross of
  two corner differences, normalized); the GPU row stores them. Derived in
  the CPU layout as FCPW computes them -- Eigen's cross, its squared norm
  `fma(x, x, fma(y, y, z*z))`, and normalized()'s division guarded by the
  squared norm being positive -- and read only where FCPW reads them. For
  the reads to sit there, the edge's `intersects` is written as FCPW's two
  statements (`if one_face { true } ; is_silhouette_edge(...)`) rather
  than one `||`: a call inside a short circuit stays a call taking the
  whole edge, which pins every field's value above the test.
- Eigen's normalized() has a `z > 0` guard (Dot.h), returning a zero vector
  as it is; the metric's `unit` and the hit normal (`v1.cross(v2)
  .normalized()`) now carry it, and the hit normal uses squaredNorm's fma
  order rather than dot's.
- The wide metric's "proper box" compare, kept in the scalar slab test for
  a box that is never the inverted empty slot here: dropped.
- The edge index is loaded per edge and per triangle where FCPW reads
  pIndex on a hit only (findClosestPoint reads it every time): the same
  cache line as the indices, one 4-byte load; Opt/Sink will not move a
  value past a statement holding a call, and the record is under one. Left
  as the known remainder.
- Kept by the user's ruling, measured as they are: our silhouette node
  test's order of conjuncts (cone before the running bound) and our direct
  entry of the nearer child where FCPW pushes both and pops the nearer.

The compiler, beyond the layout extension. Making the derived fields lazy
exposed that an element read out of a tile is one `let e = build<Edge>(
...)` whose fields are computed where it is built, so Opt/NarrowLoads now
splits a pure build into one let per field read (scalar replacement of
aggregates on a build, beside the merge-temp case), a field that is a bare
variable substituted rather than aliased, and the build erased once
nothing reads it; Opt/ValueClass treats a dereference through a read-only
pointer -- a layout's storage -- as a value rather than a read of memory,
so it crosses the stores of inlined temps; Opt/Sink enters a single using
arm whatever the arm holds (a call inside it is no obstacle to a pure
value) and sinks the arm again after entering, so a let descends through
nested branches to its use. With these the two gathers and the two
normalizations of an edge sit in the two-face arm after the radius test,
where FCPW's are. Tests narrow-loads-build (opt, backends/llvm,
correctness/llvm); goldens of some sixty tests moved (lets reordered and
renamed, builds split, constants folded through split fields, the any-hit
accept a select), each diff read before blessing.

A compiler bug found on the way, and fixed (2026-10-25, late): a `let a =
b` alias of a lane-indexed per-child value inside the children loop
(`b$low = _t67129`, `_t67129 = build<f32x3x2>(...)[children]`), surviving
to the SSA conversion and the unroll, made both children's sort keys one
SSA value -- the swap vanished, the silhouette query's tie pattern changed
and it ran 25% slower on head. The cause: a `let` whose value is a block
argument is a Set instruction carrying the program's name, and SSA/Defer.
cpp's clone_region keeps a Set's name in every copy by design (the
relooper binds the name; the defer rewrite finds a copied continuation's
values by it -- suffixing them broke every defer test), so an unrolled
children loop has two same-named Sets in dominating blocks; and the SSA
simplifier's value numbering (SSA/Simplify.cpp key_value) keyed an
operand by its NAME, so the second copy's `b$low - p` was numbered as the
first's. The fix: key_value keys an instruction by its identity
(key_instruction, the address, the key being a text for the table's
lifetime only). Tests unroll-let-names at the three levels (ssa/blocks,
backends/llvm, correctness/cpp) pin the shape -- two same-named Sets per
copy, two distinct keys, the swap present; the collision form itself (a
Set not behind a call) cannot be written in the source language, since
Opt/CSE's copy propagation removes every `let a = b` and a call result's
Set sits in a continuation the numbering walk does not cross, which is
why only a pass's alias ever produced it. Opt/NarrowLoads's bare-variable
substitution stays as the pass's own hygiene, not as the fix.

Head, the check before the sweep (repeats 3, the machine quiet, FCPW's
own arrays): rays 0.67/0.79/0.84 over the binary tree and 0.65/0.77/0.82
over the cone hierarchy, closest 0.99/1.13 and 0.96/1.10, silhouette
1.02/1.05; every query agrees to the bit up to ties; the lazy normals
measure the same as the eager ones on this mesh (0.32/0.64 M queries/s,
FCPW 0.31/0.61). Against the GPU-rows CPU layout the leaf-heavy queries
fell by what the layout gave them: closest 1.10/1.21 to 0.99/1.13,
silhouette 1.14/1.24 to 1.02/1.05, the rays 0.81/0.88/0.86 to
0.67/0.79/0.84.

Measured (2026-10-25, 19:34-19:53, the lock held, `benchmark.sh
--schedules fcpwb,fcpws`, five repeats, the least of each; plots/
rtq-fcpw-scalar-cpu.*; the earlier logs in results/logs/before-2026-10-25-
cpu-bytes/ and the earlier CSV beside the new one). Bonsai over FCPW's
scalar build, one pinned thread a side, every cell agreeing to the bit up
to ties (0 differ in all 85 cells; the GPU-rows table had the same
agreement):

    mesh       primary  diffuse  ao     near cp  vol cp  near sil  vol sil
    pavilion    0.60     0.59    0.57    0.79    1.10     0.94      1.05
    head        0.67     0.85    0.80    0.98    1.13     1.02      1.05
    zero-day    0.98     0.94    0.94    1.09    1.14     1.03      1.05
    bmw         0.76     0.91    0.86    1.05    1.08     1.08      1.05
    crown       0.92     0.95    0.94    1.10    1.10     0.97      0.98
    ivy         0.89     1.04    1.04    1.12    1.15     1.28      1.36
    villa       0.73     0.88    0.84    1.06    1.08     0.98      1.02
    dambreak    0.82     0.93    0.94    1.04    1.01     1.24      1.09
    sportscar   0.84     0.94    0.95    1.07    1.06     1.04      1.04
    landscape   0.74     0.62    0.63    1.02    0.97     1.05      1.14
    ganesha     0.81     0.88    0.88    1.09    1.01     1.03      0.96
    dragon      0.78     0.81    0.80    1.09    1.01     0.99      1.01
    geomean     0.79     0.85    0.84    1.04    1.07     1.05      1.06
    was (GPU rows on the CPU, 16:04)
                0.81     0.88    0.86    1.10    1.21     1.14      1.24

(lte-orb's primary rays 0.83; FCPW finds no hit on it for the rest, so it
stays out of the figure as before.) What moved is the leaf-heavy side: the
closest point and the silhouette query, which the stored corners had
carried to 1.1-1.24, are now level with FCPW (1.04-1.07), the gather and
the indirection paid on both sides alike; the rays, whose cost is the
visit, hardly moved (0.79/0.85/0.84 against 0.81/0.88/0.86) and still
read as the previous section says, a mask-and-cmov visit against FCPW's
branchy one, pavilion and landscape the extremes. The lazy normals and
the two-statement edge test measure the same as the eager form did on
head (0.32/0.64 M queries/s, FCPW 0.31/0.61), so nothing of the silhouette
column is the sinking's; it is the layout's. Both tables are now of the
same algorithm over the same bytes with the same arithmetic; what is left
between the sides is the visit's code shape and the two orderings the user
chose to keep, which these numbers can now be read against.

### What is left between the sides, as code (2026-10-25, night; corrected the same night)

The user asked for the remaining codegen differences as pseudocode and
what each would take to close. A first version of this section, written
from the 10-24 reading of the comb's assembly, described the two-lane
visit the sort's lowering made before that day's changes (apps/wosx/
PLAN.md, "The binary node's visit made FCPW's": a two-lane run sorted as
scalars, `unroll`, the simplifier's `fold_built_lanes`) and was wrong
about what is there now. This one is read off the LLVM the compiler
emits today for `fcpwb`'s trace_all and closest_all, against FCPW's
bvh.inl.

What the two-child visit is now. The two boxes are read from the
children's own rows and each tested on its own; the two keys are compared
once, as floats; the near child, the far child, their keys and their hit
conditions are chosen by five selects (the compare-and-swap; FCPW's `if
(d1 < d0) swap`); a branch on the far child's hit pushes its index and key
from registers onto the two stack arrays; a branch on the near child's
hit enters it or pops. That is FCPW's visit up to the kept ruling (we
enter the near child where FCPW pushes both and pops). No transpose of the
two children's boxes remains, no sortable-integer key, no mask word, no
lane select: the peel chain is for four lanes and up. Nothing of the sort
is left to fix at two lanes.

What differs, in the order of its likely cost:

1. `vec3f` is LLVM's `<3 x float>` in scalar code. A 12-byte, 4-aligned
   vector load becomes `load float` + `load <2 x float>` + `insertelement`
   + two `shufflevector`s; a scalar consumer of a component is an
   `extractelement`; a vector rebuilt from scalars is three inserts; a
   broadcast an insert and a shuffle. In the ray visit that is 8 loads, 4
   inserts and 8 shuffles for the two boxes where Eigen's FCPW does twelve
   scalar loads; per leaf triangle 6 loads, 3 inserts and 6 shuffles for
   the corners and 6 extracts to feed the scalar fma chains after them
   (the triangle test is scalar after a vector subtract); the hit normal 3
   inserts, a splat and a vector divide. trace_all has 24 shuffles, 14
   inserts and 18 extracts in 426 lines; closest_all 19, 12 and 34.

        ours:  lo = <load f, load <2 x f>, insert, shuffle, shuffle>
               t  = (lo - o) * inv            # <3 x float>
               d  = max(t[0], t[1]), ...      # extract, extract, ...
        FCPW:  lx, ly, lz = three loads; tx = (lx - ox) * ix; ...
               d  = max(max(tx, ty), tz)      # scalars throughout

   The fix: in non-gang code the LLVM backend lowers a geometric vector
   (f32x3, f32x2, f32x4) as its scalars -- a vector type is for a gang's
   lanes only, as Halide keeps vectors to the vectorized dimension -- so
   that loads are scalar loads, components are values and reductions are
   scalar chains; LLVM's SLP vectorizer re-packs isomorphic chains where
   its cost model says so. CodeGen_LLVM_SSA's type and shuffle lowering;
   the C++ backend's ext_vector types untouched. This is what clang makes
   of Eigen's Vector3f, so it is the representation the reference is
   measured with, not a tuning. Expected to be most of what remains on the
   cache-resident meshes (comb, pavilion, head); to be measured, not
   promised.

2. LLVM's SLP vectorizer then packs the two children's per-axis reductions
   into `<2 x float>` operations (`fcmp olt <2 x float>`, `select <2 x
   i1>`), whose two-lane compares are k-mask registers on this AVX-512
   machine, extracted lane by lane (`kshiftrb`, `kmovd`): the "mask
   registers" the evening's reading saw are LLVM's, not the sort's.
   Whether it still packs once the inputs are scalars (item 1) is to be
   seen; if it does and the extraction stays, the function's vector-width
   attributes are the lever, with a measurement.

3. The closest point's key computes the right child's distance twice on
   the common path. The key is FCPW's far-corner rule, `select(distmin(
   left) == 0 && distmin(right) == 0, distmax(child), distmin(child))`;
   the `&&` is a short circuit, so `distmin(right)` is computed in the
   `&&`'s right-hand arm when the left child contains the point, and
   again, unconditionally, for the right child's own key after the merge
   -- the LLVM has the two identical chains (subtract, max, max with zero,
   squares, two fmas) in `!and_rhs_7` and `!and_merge_8`, and GVN leaves
   them since the arm does not dominate the merge. A point inside the left
   child's box is the usual case of a descent, so the walk pays a box
   distance a visit for it. FCPW computes both overlaps once, before the
   compare (bvh.inl findClosestPointFromNode, 1339-1343: `overlap(s,
   d2Min, d2Max)` a child, `s.r2 = min(s.r2, d2Max)` between them, then
   the far-corner test on the two d2Min). The carried bound is right on
   our side -- the push stores distmin, not the key (closest_all's `%68`
   selects `%53`/`%30`), so the far-corner key orders and never prunes.
   The general fix is partial redundancy
   elimination of a pure, read-only value computed in an arm and again
   after the merge: hoist it to the branch (SSA/Simplify, or LLVM's
   GVNHoist in our pipeline). The transcription could also write a
   non-short-circuit `&`, since both operands are computed regardless; the
   compiler should see it either way.

4. The leaf's eager index load: `pIndex` (offset 20 of the Triangle
   object) is loaded with the three indices for every triangle tested;
   FCPW's intersect reads it on a hit only. Opt/Sink refuses to move a
   value past a statement holding a call (the record's); Opt/ValueClass
   already knows a read-only load for a value; Sink's legality for such a
   load across a call is the one check to add. A load and a live register
   per triangle, on the line already fetched.

5. Not a difference of the compiler's making: the control flow. The scalar
   schedules see no linearizer (Moll & Hack's partial linearization is the
   vectorizer's, for gang code); the visit's decisions are branches, and
   its selects are the swap's and the metric's value selects (`distmin`'s
   infinity where the ray misses, the clamp at zero), which FCPW's clang
   build makes of the same source. Whether a branch beats a select is the
   predictor's question -- coherent primary rays favour branches, diffuse
   rays and random walks the selects -- and LLVM carries heuristics both
   ways; nothing general to change here.

6. Kept by the user's ruling, not to be closed: our silhouette node test's
   order of conjuncts (the cone test before the running bound) and our
   direct entry of the nearer child where FCPW pushes both and pops it.

7. The GPU. The two kernels are the same arithmetic (FFMA 248 against 233,
   the same MUFU counts); the one systematic difference is the load path,
   ours `ld.global.nc` (LDG.E, the read-only cache) for the tree, FCPW's
   Slang buffer pointers generic (LD.E) plus some global. In our favour
   (1.1-1.4x); the profiler would confirm it, with the user's say-so.

Known-open beside these, general and not at two lanes: ConstantIntervals
does not derive non-negativity for `dot(v, v)` or `max(.., 0)` squared,
so the four-lane-and-up peel chain flips the closest-point keys' sign
bits (`sort_lanes`, `key_range.sign_clear`).

What the layout work did not change is the per-triangle leaf (about 130
instructions a side on the comb) and the traversal counts (ours 17% fewer
node visits on the comb, from the argmin's strict improvement against
FCPW's `d2Min <= r2`); what remains is items 1-4, all in the LLVM
lowering and the optimizer, none in the schedule or the layout.

### The ray gap read from the machine, and closed (2026-10-26)

The user, on the 10-26 table (rays over FCPW's scalar binary tree 0.80,
0.85 and 0.84 geomean, pavilion 0.57-0.60): the microbenchmarks have to be
at parity or better before any more time goes into the walks that are
built from them. Read on pavilion's first hit (`fcpwb`, `--query
intersect`, the scratch under scratch/rays-gap/; `compare.sh` gained
`RTQ_KEEP=1`, which leaves rtq.ll and rtq.out behind for exactly this):

- **Not instruction count.** `perf record` of cycles and instructions per
  symbol, the driver's run counts taken into account (it runs FCPW's
  primary rays three times -- the hits the secondary rays are generated
  from, the warm-up, the timed run -- and ours twice): per run ours
  retired 0.85x of FCPW's instructions in 2.0x its cycles. Our phase ran
  at IPC 1.5, FCPW's at 2.5 (`perf stat -I 50`).
- **Not memory, not branches.** Per symbol, L1 load misses and branch
  misses: ours 1.3x FCPW's of each per run, and both about one per ray
  -- a few dozen cycles of some four hundred a ray. The same tree bytes,
  wrapped, in the same visit order.
- **Not the vector form.** `vec3f` is LLVM's `<3 x float>` in scalar code
  (item 1 above): a 12-byte box corner is a float load, a `<2 x float>`
  load and a permute, and the node test had 13 shuffles in 79
  instructions. LLVM's Scalarizer run on every function no vectorize
  widened (an experiment, behind an environment variable in the worktree
  and not committed) was worth 8-10% on the rays before the SLP
  vectorizer, which re-pairs x and y and leaves z scalar -- clang's shape
  for FCPW's Eigen code -- and 10% *worse* after it, fully scalar. Not a
  general win, and not the gap; the user's reservation stands (ARM's and
  the GPU's short vectors are their own question; NVPTX was excluded).
- **Not the tie order.** The two-lane network kept the earlier child only
  where it was strictly the smaller, so a tie -- every ray whose origin
  lies in both children's boxes, most rays near the root of an interior
  scene -- entered the right child first where FCPW enters the left
  (`boxHits[2] < boxHits[0]`, strict). Fixed (compare_and_swap: the later
  child strictly smaller swaps; the tie test
  tests/bonsai/correctness/cpp/sort_tie_left_first): bmw's primary rays
  went from 5 ties and 6 differ against FCPW to 1 tie and 0 differ, and no
  rate moved.
- **The dependence.** FCPW picks the nearer child with a branch, `if
  (boxHits[2] < boxHits[0]) swap`, which the machine predicts and runs
  ahead of: the next node's loads issue while this node's slab arithmetic
  is in flight. ('The machine code behind the table' above had seen
  exactly this on 2025-10-25 -- FCPW's 48 branches and no cmov against our
  four cmova, "one of them the next node, a data dependency through the
  box tests' full latency on every visit" -- and the night's correction
  then set it aside as not an item, if-conversion being the linearizer's
  and the scalar schedules seeing none. The selects were the sort's own,
  and the dependence was the whole gap.) Our swap was five selects -- three cmov in the machine code
  -- so the next node's index was a value computed from the whole test,
  loads to compares, and every node waited for the previous one's chain:
  about forty cycles a node, serialized. The proof: the kept LLVM with the
  swap rewritten by hand into a branch with two arms (scratch/rays-gap/
  cmov/branch.ll, through llc) took pavilion's primary rays from 0.66x to
  1.37x of FCPW and the diffuse rays from 0.67x to 1.33x, every ray
  agreeing; the occlusion function, left as it was, stayed at 0.63x.

The lowering (src/SSA/SortRecursion.cpp, `sort_two`): a run of two is
sorted by a branch on its one comparison, a vote as before, between two
arms that each hold the run in one order -- the calls as written, with
their conditions, keys cleared -- meeting in one block before the run's
continuation (a call's continuation has one predecessor, for the
structuring of a recursion left recursive; a gang's region leaves for its
landing from the arms, which Linearize.cpp's hops now carry as a list).
Where each lane of a gang orders the run for itself -- the schedule
vectorizes a loop of the traversal's function or of a caller of it, so
that loopify's stack is per lane and its votes are dropped -- the network
of selects stays, since a lane cannot branch: `sort_recursion` takes
`per_lane` from the schedule (SSA/Convert.cpp reads the vectorize
directives off the whole schedule, as it does for a queue's leaves).
A gang that decides by the lanes' majority (vectorized, then sorted)
branches on the uniform vote, Embree's shape too. Loopify needed one more
fact: the arms' runs are lanes of the node's vectors read in order, which
it took for a run in no order of its own and wrote with its compaction's
convention (the right child first); `Terminator::MultiCall::ordered` says
a sort fixed the order, and the lanes path keeps off it.

Measured (five repeats, FCPW's rates matching the 10-26 table's, the lock
held, scratch/rays-gap/table.csv; the two-child visit as a branch, the tie
fix, nothing else), geomeans over the twelve meshes FCPW hits:

| over                       | primary | diffuse | ao   | closest near | closest volume | silhouette near | silhouette volume |
|----------------------------|---------|---------|------|--------------|----------------|-----------------|-------------------|
| FCPW's binary tree (fcpwb) | 1.16    | 1.13    | 1.09 | 1.20         | 1.17           |                 |                   |
| its normal cone hierarchy (fcpws) | 1.19 | 1.15 | 1.06 | 1.19       | 1.16           | 1.16            | 1.15              |

The least cells: dragon's occlusion rays 1.00 (fcpwb) and landscape's
1.00 (fcpws), crown's silhouettes 1.01 and 1.02, landscape's rays 1.06
and 1.07; the greatest pavilion's primary rays 1.51 and ivy's silhouettes
1.43 and 1.48.

**The GPU, the same afternoon.** The lowering reaches the GPU kernels
too (a schedule's `bind(i_thr, GPUThread)` recursion is scalar code per
thread), and the GPU table remeasured with the branch lost 3-5% on every
query but the occlusion rays (fcpwb-gpu primary 1.19 to 1.15, closest
near 1.16 to 1.10; FCPW's rates within 0.4%): a GPU thread is a lane of
its warp, and where the threads of a warp disagree on the order the branch
runs both arms under masks, which the selects never did. So `per_lane`
also holds for a function a loop bound to GPUThread, GPUBlock, OptixThread
or RTCore reaches (Convert.cpp's `lane_runs`), and the GPU kernels keep the
network of selects; the table remeasured once more below.

**Landscape's first hit on the GPU (the user's question).** 0.91x this
morning, 0.92x with the branch, the one GPU cell under 1. The mesh is a
slab 0.9 thick and 1240 by 1640 across, 1.9M triangles; 87,454 of the
1,048,576 primary rays hit it, so 92% leave after a few node tests and
both kernels run at 12 billion rays a second, 80-90 microseconds a
launch, where per-ray fixed work is the cost and traversal is not (the
deeper meshes are 1.1-1.5x ours). Read off the SASS of both (scratch/
rays-gap/gpu/: ours out of the driver binary's embedded PTX through
ptxas and cuobjdump, FCPW's `rayIntersection` through `slangc -target cuda
-D FCPW_BVH_TYPE=2` and nvcc, both sm_120): the kernels are the same size
and shape -- 688 instructions against 616, 61 registers against 55, the
same 512-byte stack, the same twelve MUFU (nine reciprocals, three
rsqrt), 53 branches against 50, 20 selects each. What differs per ray is
the record: FCPW's `GPURay` is 40 bytes, origin, direction, the
reciprocal direction and tMax, ten 32-bit loads, with the reciprocals
computed by its constructor on the host before the upload, off the clock;
ours is the program's `Ray`, 64 bytes with 16-byte `vec3f` fields,
sixteen loads of the eight words that matter and a 64-byte line fetched
for 32 bytes of payload, and a `Hit` of 64 bytes against FCPW's 40-byte
`GPUInteraction`, both written on a hit only. Over a million rays that is
67 against 41 megabytes read per launch, 800 against 540 GB/s on a card
that can do 1,790, with latency the rest. FCPW reads its buffers through
generic loads (LD.E, twenty of them) where ours go through the read-only
path (LDG.E.CONSTANT), in our favour on the other meshes. The record's
layout is the exported ABI's, not the layout language's: closing this cell
means a packed input record for the query batch, a layout question for the
user, not a codegen one. The 10-26 table had the rays at 0.80, 0.85 and 0.84 with
pavilion at 0.57-0.60 and head and landscape under 0.75. The Embree and
four-wide cells of pavilion, bmw and dragon, measured beside them as the
audit of what the change must not touch, match the 10-26 table to the
second decimal (the gang's runs go through the peel chain, which is
unchanged; the Embree schedules have no sort of two). Tests: ssa/blocks/sort and sort-recursion
(the branch where the selects were, pbrt's `if (dirIsNeg)` folded out of
`cast(!c) < cast(c)`), the two-child goldens (child-volumes-sorted-two,
-unrolled-two, unroll-let-names: the relooper's output of the unrolled
visit fell from 7280 lines to 508), the vectorized sorted traversals
unchanged in form (selects) with the tie fix's operand order, and
sort_tie_left_first run.

### The records, the reference's to the byte (2026-10-26, afternoon)

The user, on the landscape reading above ("why do we have larger
ray/answer records?? We need this to be an apples-to-apples comparison"),
and the rule that came with it: when we compare to Embree we match Embree,
when we compare to FCPW we match FCPW -- the system compared to, exactly,
for the algorithm and the layout, and for the base comparison the schedule
too. The records a batch is handed and hands back are part of the layout,
and ours were the program's: a `Ray` of 64 bytes with 16-byte `vec3f`
fields where FCPW's GPURay is 40, a `Hit` of 64 against its 40, and no
reciprocal direction carried where FCPW's constructor computes one on the
host, off the clock.

**What a batch is now.** Each metric file declares its reference's records
to the byte, and rtq.bonsai's and snch.bonsai's exports take and return
those, converted at the boundary (`ray_of`, `point_of`, `query_of` read a
record into the program's Ray, Point and SilhouetteQuery; `record_of_hit`,
`record_of_closest`, `record_of_silhouette` build the record from the
answer; the program computes on `vec3f` as before):

- FCPW (metrics/fcpw.bonsai and fcpw_scalar.bonsai, FCPW's GPU structs on
  both targets, fcpw/gpu/bvh_interop_structures.h): `RayRecord` is GPURay
  -- origin, direction, the reciprocal direction, tMax -- 40 bytes;
  `HitRecord`, `ClosestHitRecord` and `SilhouetteHitRecord` are
  GPUInteraction -- point, normal, uv, distance, index -- 40 bytes, written
  whole on a hit, the fields a query does not compute left zero as FCPW
  leaves them; `PointRecord` is GPUBoundingSphere -- center, squared radius
  -- 16 bytes; `SilhouetteQueryRecord` the sphere and WoSX's flip word, 20
  bytes, with the minimum radius and the precision passed once for the
  batch, as FCPW's kernel takes them.
- Embree (metrics/embree.bonsai): `RayRecord` is RTCRay -- org, tnear,
  dir, time, tfar, mask, id, flags -- 48 bytes, read as Embree reads it:
  RTCRay is `RTC_ALIGN(16)` and Embree's kernels hold it as two `Vec3ff`,
  "3 floats for ray origin, 1 float for tnear" and "3 floats for ray
  direction, 1 float for time" (kernels/common/ray.h), each one aligned
  sixteen-byte load, so the record says `org : vec4f` and `dir : vec4f`
  with tnear and time in the fourth lanes. (A first form with `[[packed]]
  vec3f` fields and scalar tnear and time was the same bytes at alignment
  four, loaded a float at a time -- eight loads and six inserts a ray where
  Embree makes two loads -- and cost Embree's first hit 5% geomean, 11-12%
  on zero-day and crown, where a ray is thirty cycles and the record's
  loads are a tenth of it.) `HitRecord` is RTCHit -- Ng, u, v, primID,
  geomID, instID -- 32 bytes, geomID the one geometry's 0 and instID the
  invalid id, as Embree writes a hit under no instance; `PointRecord` is
  RTCPointQuery -- x, y, z, time, radius, `RTC_ALIGN(16)`, twenty bytes of
  fields in a thirty-two-byte stride -- `p : vec4f` (x, y, z, time) and the
  radius, the twelve bytes of tail padding Embree's own; `ClosestHitRecord`
  is the driver's callback record, primID, distance, point, 20 bytes,
  Embree having no record for a point query's answer.

**The reciprocal direction, each side's own.** FCPW's GPURay constructor
computes dInv on the host before the upload, off the clock, and the kernel
reads it; the driver's `make_ray` does the same for FCPW's record, the
`Ray` element carries `dinv`, and the slab test reads it (elements.bonsai,
metrics/fcpw*.bonsai). Embree computes `rcp_safe` of the direction inside
rtcIntersect1, so for Embree's record `ray_of` computes it inside the timed
call.

**What is written besides the answer, and only that.** rtcIntersect1
shortens its ray's tfar to the hit and the tutorial's callback shrinks its
query's radius to the distance found; FCPW's GPU kernel writes nothing into
its ray or its sphere (the ray is a copy in registers, bvh.slang's
`intersect` shortening its own tMax). The metric says which (`ray_written`,
`point_written`: Embree's write the field, FCPW's are empty), and the
exports call them after the hit is written. For an occluded ray Embree
writes its tfar, set to -inf, and FCPW's GPU writes the interaction's index,
the first hit's -- four bytes either way -- so the one flag word of
`occluded_all` stands for both.

**The field type that made it possible: `[[packed]] vec3f`** (compiler
commit 858c6d45). The vector's storage form (Vector_t::packed, what a layout
row's `vec3f` has been since the layouts: twelve bytes aligned to four,
where the vector the program computes with takes sixteen), now writable as
a type in an element; the parser converts a read of such a field to the
computing vector and a value built or assigned into one to the storage
form, as Lower/Layouts.cpp has done for rows. Two other routes were tried
and are wrong: `array[f32, 3]` as the field (the user's first suggestion)
is a pointer in the LLVM struct where the generated header prints
`float[3]` -- every array is a pointer to the LLVM backend (CPP.cpp's own
note on Array_t), so the two sides would have disagreed about the record;
and `element [[packed]]`, which lands a 40-byte record by accident (16 + 4
+ 16 + 4), is C's packed -- one-byte alignment, under which NVPTX loads a
record a byte at a time where Slang's float3 loads floats. The records are
ordinary elements of packed vectors and scalars, aligned to four, no
padding; the generated headers spell the field `std::array<float, 3>`, and
the driver's static assertions against the generated headers give 40, 40,
40, 16 and 20 for FCPW's and 48, 32, 20, 20 for Embree's.

**The driver** (rtq_hook.cpp): the batches are the records, made by
`make_ray` and `make_point` per reference; the rays and points the program
writes into are the batch's working copies, reset off the clock between
runs as reset_rayhits and reset_queries restore Embree's own tfar and
radius; a hit record is read with its ray (`as_ref`), Embree's distance
being the ray's tfar; the silhouette batch's minimum radius and precision
are the two constants both sides take.

**What remains different, and is kept on record.** Embree's RTCRayHit
holds the hit beside its ray, one 80-byte record, where ours are two arrays
of 48 and 32: the same bytes read and written per ray, in two streams
rather than one; and RTCHit is aligned to sixteen where our 32-byte
HitRecord is aligned to four, which changes no store, both sides writing
Ng, u, v and the ids a scalar at a time. FCPW's CPU `Interaction` is larger
than its GPU one (it carries pointers and indices the GPU struct does not)
and its CPU `intersect` writes tMax into its own ray copy; our record is
the GPU one on both targets, so against FCPW's CPU trees we write 40 bytes
a hit where it writes its Interaction. The occluded word means tfar on one
side and an index on the other, four bytes on both.

Every configuration agrees with its reference on head, one repeat, before
the tables (embree, fcpwb, fcpws, fcpw4w16, fcpwb-gpu, fcpws-gpu: every ray
up to ties); the two wosx programs compile against the Ray that carries
dinv.

**Measured** (2026-10-26: the six schedules 17:05-17:40, the Embree rows
again 18:32-18:39 with the two-vector record; five repeats, the harness's
own lock -- benchmark.sh resumes from the logs it finds, so the six kinds'
logs were deleted first and the old ones kept in
results/logs/before-2026-10-26-records/, the CSV as
rtq-results.before-2026-10-26-records.csv), geomeans over the twelve
meshes, before the records and with them; the references' own rates within
1% of before throughout:

| over                              | primary                  | diffuse | ao   | closest near | closest volume | silhouette near | silhouette volume |
|-----------------------------------|--------------------------|---------|------|--------------|----------------|-----------------|-------------------|
| FCPW's GPUScene, binary (fcpwb-gpu) | 1.19 -> 1.34 (landscape 0.92 -> 1.58) | 1.37 -> 1.37 | 1.35 -> 1.36 | 1.15 -> 1.17 | 1.08 -> 1.11 | | |
| FCPW's GPUScene, SNCH (fcpws-gpu)   | 1.12 -> 1.23 (landscape 0.99 -> 1.34) | 1.27 -> 1.29 | 1.27 -> 1.29 | 1.11 -> 1.12 | 1.10 -> 1.12 | 1.22 -> 1.22 | 1.25 -> 1.25 |
| FCPW's scalar build, binary (fcpwb) | 1.13 -> 1.10 | 1.13 -> 1.10 | 1.09 -> 1.16 | 1.20 -> 1.20 | 1.17 -> 1.17 | | |
| its normal cone hierarchy (fcpws)   | 1.18 -> 1.19 | 1.14 -> 1.14 | 1.06 -> 1.06 | 1.18 -> 1.19 | 1.16 -> 1.16 | 1.16 -> 1.15 | 1.15 -> 1.13 |
| FCPW's four-wide Mbvh (fcpw4w16)    | 1.58 -> 1.62 | 1.64 -> 1.63 | 1.91 -> 1.91 | 1.17 -> 1.17 | 1.18 -> 1.19 | | |
| Embree's BVH8 (embree)              | 1.05 -> 1.03 | 1.05 -> 1.06 | 1.03 -> 1.02 | 2.35 -> 2.36 | 2.93 -> 2.94 | | |

What moved, in our rates: the GPU first hit +13% over the binary tree and
+10% over the cone hierarchy, the 40-byte ray with its reciprocal
precomputed against the 64-byte Ray with three divisions in the kernel
(landscape +75%, sportscar +35%, villa +20%, dambreak +16%: the meshes
most rays miss, where the record was the cost); the binary tree's
occlusion rays on the CPU +7%, the same 40 bytes and no hit to write; its
first hit -2%, the record's three vectors read a float at a time, as FCPW
reads its Eigen vectors, where the program's own Ray was two 16-byte
loads -- the cost FCPW pays, now paid on both sides; Embree's first hit
-3% (zero-day 1.12 -> 1.00, crown 1.10 -> 1.03), the tfar written back on
every hit and RTCRay's 48 bytes against the Ray's 40, the diffuse and
occlusion rays and both references' closest points within 1%. The
four-aligned Embree record measured in between put the first hit at 1.00
with zero-day and crown at 0.88: the load form, not the bytes. The
figures: apps/rtq/plots/rtq-fcpwb-fcpws-cpu and -gpu, rtq-firsthit,
rtq-anyhit, rtq-closestpoint (the packet figure is gone at the user's
word, 18:00; the packet16 rows still fold into the CSV).

### What Nsight Systems says of the GPU win (2026-10-26, evening)

The user's question: why are we ahead of FCPW on the binary tree at all,
on both targets, and what would profiling tell. Nsight Systems over the
fcpwb-gpu driver (the user's say-so; `nsys profile --trace=cuda` with GPU
metrics sampled at 100 kHz, 20 repeats, landscape and pavilion;
results/nsys/, read by nsys_kernels.py) -- both sides' kernels in one
trace, FCPW's through slang-rhi's CUDA backend, ours through the driver
API, the same 256-thread blocks over the same batches:

| cell | FCPW kernel ms | ours | kernel x | FCPW period ms | ours | wall x | dispatch us FCPW / ours |
|---|---|---|---|---|---|---|---|
| landscape primary | 0.069 | 0.046 | 1.49 | 0.098 | 0.057 | 1.71 | 25 / 7 |
| landscape ao | 0.023 | 0.019 | 1.22 | 0.046 | 0.026 | 1.79 | 22 / 6 |
| landscape diffuse | 0.026 | 0.021 | 1.21 | 0.048 | 0.029 | 1.66 | 21 / 7 |
| landscape closest near | 0.486 | 0.431 | 1.13 | 0.510 | 0.444 | 1.15 | 21 / 9 |
| landscape closest volume | 0.278 | 0.269 | 1.03 | 0.302 | 0.279 | 1.08 | 21 / 7 |
| pavilion primary | 0.083 | 0.064 | 1.29 | 0.112 | 0.073 | 1.53 | 26 / 6 |
| pavilion ao | 0.040 | 0.034 | 1.19 | 0.062 | 0.041 | 1.53 | 21 / 7 |
| pavilion diffuse | 0.046 | 0.035 | 1.33 | 0.068 | 0.043 | 1.60 | 21 / 7 |
| pavilion closest near | 0.098 | 0.078 | 1.26 | 0.122 | 0.088 | 1.38 | 23 / 9 |
| pavilion closest volume | 0.454 | 0.411 | 1.10 | 0.477 | 0.420 | 1.14 | 21 / 7 |

(kernel: the least GPU duration of the 21 launches; period: the mean
start-to-start gap between a run's launches, what the driver's clock sees
per repeat; dispatch: the period less the kernel.) Two findings:

- **A fixed 21-26 microseconds per dispatch on FCPW's side against 6-9 on
  ours**, the cost of slang-rhi's path from the host call to the kernel
  (command encoding, parameter binding, the fence) against a driver-API
  launch and a synchronize. The harness times both sides around the
  dispatch and the wait, as the user's rule of 10-24 has it (one query a
  thread in groups of 256, the same call on both sides), so this cost is
  in the table -- and for the 20-80 microsecond ray kernels it is a third
  to a half of the gap: landscape's ao rays are 1.22x kernel to kernel and
  1.79x on the clock. The closest-point cells, 0.3-0.5 ms a launch, carry
  it as 2-5%. Kernel to kernel the first hit is 1.3-1.5x, the any hit
  1.2x, the closest point 1.03-1.26x. A question for the user: which of
  the two is the number the figure should show -- the API as a walk pays
  for it, or the kernel the compiler made.
- **Neither kernel is bandwidth-bound, and both run at low occupancy; ours
  issues more and hides more.** DRAM read bandwidth inside the kernels is
  0.4-1.5% of peak on pavilion (the tree sits in L2) and 11% against 18%
  on landscape's primary rays, where we drain the same bytes faster.
  SM issue utilization (instructions issued per cycle, percent of peak)
  is 16% against 21% on landscape's primary rays and 37% against 40% on
  pavilion's; 25% against 32% and 44% against 53% on the closest point.
  Warps in flight average 16 against 31 (landscape primary) and 24 against
  39 (pavilion primary), 20-40 against 24-48 on the closest point, under
  the same register counts (63 against 64, 54 against 52) and the same
  block size, so the difference is not an occupancy limit but how long
  each warp sits waiting: ours keep more warps busy and finish sooner. The
  issue rates times the durations say the first-hit kernels issue about
  the same number of instruction slots (ours 10-15% fewer on landscape)
  and the closest-point kernels 10-15% more on our side, finishing 10-13%
  sooner -- the win on the point queries is latency hidden, not work
  saved. What hides it is what the SASS reading named (read-only loads
  through the texture path where Slang's buffers are generic loads; the
  nearer child entered directly where FCPW pushes both and pops) and the
  record once per query where FCPW fills its Interaction per improving
  candidate; apportioning the three needs Nsight Compute's stall reasons
  and L1 hit rates, which Nsight Systems does not sample.

Registers at run time: 63 (FCPW) against 64 (ours) for the first hit, 45
for our any hit, 54 against 52 for the closest point; CUPTI reports no
local memory allocated per thread for either (the stacks ptxas counted as
512- and 256-byte frames). The GPC clock column of the sampled metrics
exports as a wrapped integer in this nsys version and is not read.

FCPW's GPU has one ray kernel: bvh.cs.slang's `rayIntersection` takes
`uniform uint checkForOcclusion` and its leaf returns at the first hit
under it, so its any hit runs at the first hit's 63 registers (the trace:
every FCPW ray launch one function, one register count). Ours are three
kernels, the any hit at 45. And WoSX's solvers never ask for an any hit:
on the CPU (wosx/utils/fcpw_geometric_queries.h) every ray query is
`intersect(ray, interaction, false)`, the first hit, beside the
all-intersections form for the inside test and the sphere intersection
that samples the boundary; on the GPU the library has one occlusion form,
`intersectsWithBoundary` (a line of sight between two points,
fcpw-geometric-queries.slang), wrapped once in core/geometric-queries.slang
and called by no solver. The any-hit cells of the fcpwb figures are rtq's
(Aila and Laine's ambient-occlusion set), not a walk's query.

### What Nsight Compute says of the two ray kernels (2026-10-26, evening)

The user's say-so. `ncu --set full` over landscape's primary rays (the
first-hit pair: FCPW's rayIntersection and our trace_all kernel) and its
ambient-occlusion rays (the any-hit pair: the same FCPW kernel under
checkForOcclusion, our occluded_all kernel), the second profiled launch of
each; results/ncu/ (the .ncu-rep, the raw and details pages as CSV, a
tidy CSV of every metric per launch, landscape.txt), read by
apps/rtq/ncu_kernels.py. Nsight Compute replays under clock control, so
its durations (106.6 against 88.3 us, 1.21x; 40.3 against 34.7, 1.16x)
are not the harness's (69 against 46 us at the card's own clocks, 1.49x);
the counters are per launch and compare directly:

| landscape, FCPW / ours                 | first hit (primary)      | any hit (ao)            |
|----------------------------------------|--------------------------|-------------------------|
| warp instructions executed             | 31.2M / 28.5M (1.09)     | 7.76M / 7.43M (1.05)    |
| branch instructions                    | 3.02M / 1.51M (2.0)      | 707k / 184k (3.8)       |
| threads active per instruction, of 32  | 13.9 / 19.1              | 19.8 / 17.9             |
| global load sectors                    | 22.6M / 22.6M            | 2.36M / 2.33M           |
| L1/tex hit rate                        | 86.8% / 86.7%            | 92.6% / 90.3%           |
| L2 sectors read from L1/tex            | 3.07M / 3.02M            | 227k / 228k             |
| local (stack) load / store instructions| 103k / 13.5k, 221k / 61k | 48k / 2.4k, 77k / 4.2k  |
| local load / store sectors             | 951k / 46k, 1.35M / 89k  | 344k / 3.9k, 386k / 4.4k|
| registers; block limit by registers    | 63 / 64; 4 / 4 blocks    | 63 / 45; 4 / 5 blocks   |
| theoretical / achieved occupancy       | 67% / 67%; 38.6% / 61.9% | 67% / 83%; 26.4% / 32.9%|
| issue slot utilization                 | 25.9% / 24.7%            | 30.4% / 30.7%           |
| cycles between a warp's issues         | 18.9 / 30.5              | 11.4 / 12.5             |
| of them stalled on memory (long scoreboard) | 13.8 / 15.2         | 6.5 / 6.6               |
| of them stalled at a barrier           | 0 / 9.8                  | 0 / 2.0                 |
| memory throughput, % of peak           | 43% / 53%                | 13% / 12%               |

What it says, against the three suspects:

- **Not the load path.** The two kernels fetch the same bytes -- the same
  global load sectors to within 0.2%, the same L1/tex hit rate, the same
  L2 sector reads -- and the read-only loads buy nothing measurable here:
  the hypothesis of 10-25 does not hold. (FCPW's higher L2 hit rate, 42%
  against 20%, is its stack: local memory lives in L1 and L2, and the hits
  are its own pushes read back.)
- **The stack, and the shape of the visit.** FCPW's kernel makes 7.6x our
  stack loads and 3.6x our stack stores on the first hit, 20x and 18x on
  the any hit (its push-both-pop-nearer discipline, and Slang keeping the
  stack in local memory, against the nearer child entered directly and a
  stack that mostly stays in registers); it executes twice our branch
  instructions on the first hit and four times on the any hit; and on the
  first hit it keeps 13.9 of 32 lanes busy per instruction against our
  19.1 -- its warps diverge more over the same rays in the same order. The
  instruction count differs less (9% and 5%); the work is organized
  differently, not smaller.
- **The barrier is ours, and it is a cost.** Our kernels stall 9.8 of
  every 30.5 cycles between issues at a barrier on the first hit, 2.0 of
  12.5 on the any hit; FCPW's stall none. It is CodeGen_PTX's
  `barrier.sync 0` before and after a thread loop (emit_bound_parfor, "a
  barrier on either side orders the block-level effects"), emitted whether
  or not the block body has any effect outside the loop; these kernels
  have none (no shared allocation, nothing before or after the thread
  loop), so the barrier after the loop only holds a finished warp until the
  slowest of its block is done, and the block's registers with it (four
  blocks a multiprocessor), which is also why our achieved occupancy reads
  62% against FCPW's 39%: warps resident and parked. The condition to drop
  it is exact, not a heuristic -- a block body with one thread phase and no
  block-level effect needs no barrier (SSA/ThreadPhases.h already knows the
  phases) -- and pbrt's kernels with shared-memory queues keep theirs. Not
  changed here: a codegen change, with its PTX goldens and the gpu-optix
  compile the 10-25 rule asks for, for the user to call; on the traversal
  kernels it is worth up to the stalled share, a quarter to a third of the
  cycles.
- The any hit at 45 registers lifts the occupancy limit from four blocks to
  five; achieved occupancy follows (32.9% against 26.4%) but the kernel is
  only 1.16x, since both sides issue at the same 30% and wait the same 6.5
  cycles on memory: the rays are short, half a wave per multiprocessor,
  and the kernel is latency-bound at a tail no occupancy fills.

### The barrier, removed where nothing is ordered (2026-10-26, night)

The user's ruling: barriers belong between sequenced thread loops of one
block body, and around code only the leader runs; nowhere else. Who
emitted them: CodeGen_PTX's own `emit_bound_parfor`, a `barrier()` before
and at the join of every loop bound to GPUThread since the first PTX
backend (bd1f5442, 09-21), LLVM only lowering the intrinsic; nothing in the
IR asks for one, the block body of `trace_all` being one thread loop whose
continuation yields. The rule now (CodeGen_PTX.cpp, `phase_barrier_pending`
and `phase_is_followed`): a phase begins with a barrier only if a phase came
before it in the body and nothing has ordered the block since -- a leader
effect in between brought its own (after_once_effect) and cleared the need
-- and ends with one only if block-level code follows that may read what
the threads wrote (anything in the continuation but a yield, a return, or
another thread loop directly, which puts the one barrier between them
itself). Tests, written red first and counting the PTX's `barrier.sync`
lines (tests/bonsai/backends/ptx/thread-phase-*.bonsai): one phase alone,
0 (was 2); two phases in sequence, 1 (was 4); a leader store then a phase,
1 (was 3); a phase then a leader store reading the threads' writes, 2 (was
3). Eight PTX goldens moved by exactly their dropped barriers and what
follows from them (the intrinsic call and its attribute group gone, two
kernels now inferred `nosync willreturn memory(argmem: write)`).

pbrt's gpu-optix schedule compiled with both compilers (apps/pbrt/scratch/
barrier-check/, the embedded PTX read kernel by kernel): every one of its
wavefront kernels -- the material drains, the sample, medium, emissive,
escaped and exit kernels, the queue kernels -- had the pair and nothing
else, 2 to 0, 118 kernels; the pixel-loop kernels had a leader effect's
barrier beside the pair, 3 to 1; the OptiX programs have none either way;
230 barriers to 28 in all, no other line of PTX changed (the kernels are
two lines shorter each). bonsai-f9 runs pbrt's GPU tests and the
three-scene compare on the commit before building on it (same images
expected; the kernel milliseconds telling what the join's barrier cost the
drains).

Measured (23:05-23:10, five repeats, the GPU tables alone; before-copies
rtq-results.before-2026-10-26-barrier.csv and logs/before-2026-10-26-
barrier/), geomeans over twelve meshes, before the rule and with it, FCPW's
own rates within 0.8% of before throughout:

| over FCPW's GPUScene | primary | diffuse | ao | closest near | closest volume | silhouette near | silhouette volume |
|---|---|---|---|---|---|---|---|
| binary tree (fcpwb-gpu) | 1.36 -> 1.44 | 1.37 -> 1.44 | 1.36 -> 1.43 | 1.17 -> 1.23 | 1.11 -> 1.16 | | |
| cone hierarchy (fcpws-gpu) | 1.25 -> 1.33 | 1.29 -> 1.33 | 1.29 -> 1.33 | 1.12 -> 1.17 | 1.12 -> 1.15 | 1.22 -> 1.29 | 1.25 -> 1.33 |

Our rates moved +3% to +7% geomean on every query (first hit +6.5%, any
hit +5%, closest point +5%, silhouettes +6% and +7%), every mesh but two
cells within noise (sportscar's primary rays -5%, landscape's over the
cone hierarchy -3%; bmw's and villa's first hit +15-17%). Less than the
stall share Nsight Compute showed (a quarter to a third of the issue
cycles): a warp parked at the join was a slot another block's warp could
partly fill, so the counter overstated what the barrier cost in time.
Every ray agrees as before.

### The GPU tables under the device's min and max (2026-10-27, night)

The two GPU schedules measured again at 21:40 under 39c67ab9 -- the
compiler of the evening: the device's own min and max (aa4dbaaa) and
loopify's exit funnel with its slots fixed (9846e3e6, 9dbdebc9, 39c67ab9)
-- against the 10-26 table (before-copy
rtq-results.before-2026-10-27-funnel.csv, logs/../logs-gpu-before-2026-10-
27-funnel/), five repeats, FCPW's own rates within 1% throughout. The
funnel is not in these kernels: rtq's traversals are loops with one way
out (nine loopified recursions, no `!exit` block in any schedule's SSA), so
what moved is the min/max lowering. Geomeans over the twelve meshes, before
-> after, with our rate's ratio:

| over FCPW's GPUScene | primary | diffuse | ao | closest near | closest volume | silhouette near | silhouette volume |
|---|---|---|---|---|---|---|---|
| binary tree (fcpwb-gpu) | 1.42 -> 1.49 (x1.04) | 1.44 -> 1.49 (x1.04) | 1.43 -> 1.49 (x1.04) | 1.23 -> 1.26 (x1.02) | 1.16 -> 1.18 (x1.01) | | |
| cone hierarchy (fcpws-gpu) | 1.31 -> 1.33 (x1.03) | 1.33 -> 1.37 (x1.03) | 1.33 -> 1.37 (x1.02) | 1.17 -> 1.18 (x1.01) | 1.15 -> 1.18 (x1.02) | 1.29 -> 1.27 (x0.985) | 1.33 -> 1.31 (x0.975) |

The ray and point queries gained one to four percent -- `min.f32` and
`max.f32` where a compare and a select were -- and the silhouette queries
lost 1.5% and 2.5% as a group, ivy's 6% (2.01 -> 1.89 and 1.97 -> 1.86),
dambreak's 4%, head's 4%; no other cell under 0.97 of before. The
silhouette kernel is the one to read for it: what `maxnum`/`minnum` does
to its cone tests that the compare-and-select did not (the NaN rule, or the
scheduling around the paired f32 ops), tomorrow. The CPU table reruns
tonight (~05:15, after the wosx solver table and bonsai-f9's pbrt window),
the four plotted schedules; until then note that the folding stamps every
row of the CSV with the tree's commit, so the CPU rows say 39c67ab9 while
measured on the 26th.

**The CPU table under 39c67ab9** (03:25-03:54 on 10-28, the four plotted
schedules, 65 logs and 272 cells, five repeats, the machine idle; the 10-26
logs in logs-cpu-before-2026-10-27-funnel/, the same before-copy). A tie on
every query of every schedule within one percent -- our rate after/before
as a geomean over the meshes: embree 1.000 to 1.002, fcpw4w16 0.995 to
1.002, fcpws 1.001 to 1.004, the silhouette queries 1.001 (so their GPU loss
above is the device's min/max alone), Embree's and FCPW's own rates 0.99 to
1.01 as a group -- except the binary tree, fcpwb: its nearest query gained
one to four percent on every mesh (x1.018, 1.104 -> 1.139) and its any-hit
query lost on four (villa 0.908, ganesha 0.907, head 0.948, ivy 0.957;
x0.985 as a group, 1.156 -> 1.147) while FCPW's rate on those cells held
(0.99 to 1.01); its closest-point query 0.995. No compiler commit between
the two tables touches what these schedules compile to on the CPU: the
min/max change is the device's alone (the CPU keeps its compare and select),
the funnel is in none of rtq's loops, the barrier change is the GPU's, the
field form of specialize is pbrt's. So either the two compilers emit the
same machine code for fcpwb and the difference is the machine's hour, or
they do not and something unaccounted for moved. Settled at 08:08: fcpwb
compiled through build-rtq's compiler and through the one at aa4dbaaa
(apps/wosx/scratch/wt-aa4dbaaa) gives the same object to the byte (rtq.o
and rtq.h the same checksum, the LLVM text and the SSA identical, no
`!exit` block), so the drift is the measurement's conditions and not the
compiler's: the 10-26 rows were taken over that day (logs from 03:25 to
18:39) with both sessions at work on the machine, tonight's on a machine
with nothing else running. Tonight's table is the one to keep; the drift
is not read further. Twelve cells fell under 0.97 of their speedup: the four
above ours, the other eight the reference's -- FCPW faster by 5 to 32% than
on the 10-26 cell (pavilion's primary rays 1.63 -> 1.23 with our rate 0.996
of before and FCPW's 1.32), the 10-26 reference low on those cells. The
figures (plots/rtq-*.png, .pdf, .txt) are redrawn from this table at 03:54
and current. Its rows are stamped 2026-10-28 and e20b6ebe, the tree's head
when folded, while the compiler is 39c67ab9's, built at 21:26 (the commits
since touch apps/ and the PLAN files only).

### One protocol for every system and query (2026-10-30, evening)

The user, 14:00, on how the figures were measured: "We should probably be
consistent across the two systems. Please rerun with one consistent
methodology. And use the minimum of 3 runs (after one warmup), like the
pbrt/wosx benchmarks should be doing." Until now Embree's ray queries ran
over a 2048 by 2048 camera image (so that its runs took over a tenth of a
second), its point queries, every FCPW query and every GPU query over 1024
by 1024 (FCPW at a third of Embree's rate), each cell the least of five
runs; so the two panels of rtq-queries-cpu stood on query sets four times
apart on the secondary rays. dcdcb2ea makes benchmark.sh's three sides one
--side (1024) and its REPEATS 3, Embree's three compare.sh runs a mesh one
run over every query, and the lock it writes names the session
(RTQ_HOLDER); the driver's default repeats and its usage, and the README,
say so.

**The run**: 17:48 to 18:14, 27 minutes for the six plotted schedules
(embree, fcpw4w16, fcpwb, fcpws, fcpwb-gpu, fcpws-gpu) over the thirteen
meshes, 52 logs, 418 rows, the machine idle (bonsai-f9's 'done' at 17:46
after its pbrt sweeps, bonsai-ac's wosx table after my 'done' at 18:15).
The logs in results/logs/; the 10-28 table and its logs set aside as
rtq-results.before-2026-10-30-one-side.csv and
logs-before-2026-10-30-one-side/ (the exploratory schedules -- tuned,
embree4, fcpw4w8 and the rest -- live on there and were not remeasured).
The rows stamp febd378d, the tree's head when folded, while the compiler
is caa9eb9b's: build/compiler was built at 08:49 from the sources
committed at 09:00, cmake rebuilt nothing at 17:48, and no commit since
touches CMakeLists.txt, include, src or runtime. Every cell agrees with
its reference up to ties except the 36 cells that disagreed on 10-28, with
the same counts to the ray: FCPW's wide tree and its GPUScene on the
primary rays of nine meshes (1 to 15 rays of 1,048,576, ours finding the
hit FCPW misses), and single rays or points on five secondary cells.

**Against the 10-28 table**, geomeans over the twelve plotted meshes
(apps/rtq/scratch/compare_tables_1030.py, before -> after, then our rate's
and the reference's ratio after/before as geomeans):

| reference, schedule | primary | diffuse | ao | closest near | closest volume | silhouette near | silhouette volume |
|---|---|---|---|---|---|---|---|
| Embree, embree (2048 -> 1024 on the rays) | 1.026 -> 1.022 (0.81, 0.81) | 1.050 -> 1.053 (0.89, 0.89) | 1.026 -> 1.040 (0.90, 0.89) | 2.359 -> 2.363 (0.99, 0.99) | 2.948 -> 2.947 (0.99, 0.97) | | |
| FCPW Mbvh, fcpw4w16 | 1.632 -> 1.621 (0.99, 1.00) | 1.635 -> 1.630 (0.99, 0.99) | 1.907 -> 1.903 (0.99, 0.99) | 1.169 -> 1.171 (0.99, 0.98) | 1.183 -> 1.189 (1.00, 1.00) | | |
| FCPW binary, fcpwb | 1.143 -> 1.109 (0.99, 1.02) | 1.133 -> 1.124 (0.99, 0.99) | 1.147 -> 1.141 (0.99, 0.99) | 1.177 -> 1.176 (0.98, 0.99) | 1.154 -> 1.154 (0.99, 0.98) | | |
| FCPW cones, fcpws | 1.177 -> 1.176 | 1.135 -> 1.138 | 1.054 -> 1.058 | 1.184 -> 1.187 | 1.160 -> 1.164 | 1.145 -> 1.143 | 1.132 -> 1.133 |
| GPUScene binary, fcpwb-gpu | 1.510 -> 1.533 (1.01, 0.99) | 1.489 -> 1.489 | 1.488 -> 1.491 | 1.262 -> 1.250 (0.99, 1.00) | 1.181 -> 1.181 | | |
| GPUScene cones, fcpws-gpu | 1.363 -> 1.360 | 1.369 -> 1.370 | 1.367 -> 1.377 | 1.177 -> 1.177 | 1.175 -> 1.178 | 1.266 -> 1.271 | 1.306 -> 1.299 |

(fcpws's and the GPU's ratios not shown are 0.99 to 1.00 on both sides.)
Four readings:

1. *Embree's rays at 1024 in place of 2048.* Both sides' rates fall
   together -- the primary rays to 0.81 of before, the diffuse and
   ambient-occlusion rays to 0.89 and 0.90 -- and the speedups hold (1.02,
   1.05, 1.04). Half the angular density means neighbouring rays share
   fewer nodes, and the fall is largest on the biggest trees, whose nodes
   stream from memory: landscape, ganesha and dragon run at 0.62 to 0.66 of
   their 2048 rate on our side and 0.65 to 0.68 on Embree's. Ours falls two
   to four points more than Embree's on each of pavilion, villa, landscape,
   ganesha and dragon (0.921 against 0.946 on pavilion, 0.640 against 0.677
   on landscape), so the primary rays over the four large meshes read 3 to
   6% lower against Embree than at 2048 (villa 1.07 -> 1.03, landscape 1.01
   -> 0.95, ganesha 1.04 -> 1.00, dragon 1.04 -> 1.01) while head, zero-day
   and sportscar read 2 to 6% higher; the geomean 1.026 -> 1.022. Three
   cells stand under 0.97 against Embree: pavilion's primary rays 0.96
   (0.99 at 2048; a 15 ms run), landscape's 0.95 (1.01 at 2048; 26 ms), and
   ivy's ambient occlusion 0.95 (0.96 at 2048). Whether Embree's as-shipped
   traversal (its software prefetch of the children it is about to visit)
   carries it on the less coherent rays, or the three cells are the spread
   of runs this short, a rerun of the Embree cells (two minutes) and a read
   of the two traversals' memory stalls would say. Open.
2. *The 10-28 table's reference rates were low on the large meshes.*
   FCPW's binary BVH on landscape ran 18.2 Mr/s on 10-28 against 22.0 to
   22.6 in every other table (10-25, 10-26 twice, today 22.63), its
   ambient occlusion 14.2 against 15.7 to 15.9; its Mbvh on dragon's
   primary rays 11.8 against 13.0 to 13.2. Today's cells read 10-26's again
   (landscape fcpwb 1.06x, 1.05x, 1.14x where 10-26 had 1.09, 1.04, 1.15 and
   10-28 1.32, 1.18, 1.28; dragon fcpw4w16 1.36x where 10-26 had 1.34 and
   10-28 1.49), and the fcpwb first-hit geomean's move, 1.143 -> 1.109, is
   this alone: ours 0.99 of before, the reference 1.02. The reference's
   run-to-run spread on the large meshes is 10 to 25% between tables, ours
   1 to 2%; the 10-28 section above found the same of its 10-26 cells.
3. *The gather fix* (caa9eb9b, a gang's lane offsets read as signed) left
   fcpw4w16's rates at 0.99 of 10-28's on every query: it changed the code
   of every CPU gang gather, not its speed here.
4. *The GPU tables*, at the same side as before, hold within 1.5% (fcpwb-gpu
   primary +1.5%, its near point -1%), ours 0.99 to 1.01 of before on every
   query. The clock numbers stand as before (the dispatch and the wait
   alone, each API's launch overhead included; see the Nsight Systems
   section of 10-26).

The three figures are redrawn from this table at 18:14 and current
(plots/rtq-queries-cpu, rtq-fcpwb-cpu, rtq-fcpwb-gpu, .pdf/.png/.txt), the
three older per-query figures with them. The mesh table gains a camera
hits column -- the pixels of the primary image the mesh covers, the size
of each secondary set: 9,527 on zero-day to 344,769 on villa -- so the
paper's methodology paragraph can point at it for the set sizes; plot.py
regenerated it at 20:00 (the column takes Embree's hit count, f5dcc003).

**The three Embree cells rechecked** (21:12-21:17, the machine idle after
bonsai-ac's 'done' at 21:12; compare.sh's embree schedule at side 1024 and
three repeats, twice over each of pavilion, ivy, villa, landscape, ganesha
and dragon, into scratch/embree-recheck-1030/, the table untouched). Two of
the three reproduce and one was spread:

| cell | table 18:14 | pass 1 | pass 2 |
|---|---|---|---|
| pavilion, primary rays | 0.96 (70.7 / 68.0 Mr/s) | 0.99 (73.2 / 72.2) | 0.99 (72.5 / 71.5) |
| landscape, primary rays | 0.95 (39.9 / 38.0) | 0.95 (39.9 / 37.8) | 0.95 (40.2 / 38.0) |
| ivy, ambient occlusion | 0.95 (6.51 / 6.20) | 0.95 (6.66 / 6.33) | 0.95 (6.62 / 6.30) |

Pavilion's 0.96 was the spread of a 15 ms run (both sides ran 3 to 6%
faster on the recheck, ours more); its cell is 0.99, as at 2048. Villa's
primary rays moved the same way (1.03 in the table, 1.07 and 1.06 on the
recheck, our rate 32.3 against 33.4 and 33.5); ganesha and dragon held to
a hundredth on every query. Landscape's primary rays and ivy's ambient
occlusion are 0.95 three times over: real, and the 3% rule's open items.
Landscape's primary rays stood at 1.01 over the 2048 image, so the loss is
to the coarser image alone (a 1.9M-triangle terrain whose nodes stream
from memory; half the angular density, neighbouring rays sharing fewer
nodes -- the reading of 1. above, now with one mesh to read it on);
ivy's ambient occlusion was 0.96 at 2048 too (26,110 rays, a 4 ms run,
the smallest ray set of the twelve after zero-day's). The two traversals'
memory stalls on those two cells are the thing to read next; the table
stands as measured (one run's cells are not swapped for another's).

**The table against the compiler since (2026-10-31, 23:42).** The
compiler the table was measured with was build/compiler of 10-30 08:49,
caa9eb9b's sources; the tree has since taken b72c215b (10-31 02:19, two
SSA fixes for apps/pbrt's GPU wavefront variants: the hoist of
loop-invariant allocations carries a size's arithmetic out of the loop
with it, and a hit continuation that makes a reordering hint is refused).
Whether either moves an rtq cell is a compile, not a measurement: the six
plotted schedules (embree, fcpw4w16, fcpwb, fcpws, fcpwb-gpu, fcpws-gpu)
compiled as compare.sh compiles them (`-p ssa --no-heap --ffp-contract`,
the reference's metrics file, snch.bonsai for fcpws) with
`--dump-ssa-final -b llvm` under build-rtq's compiler of 10-29 14:16 and
under build/compiler of 10-30 23:45, which carries b72c215b. The final SSA
is byte-identical for all six, and so is the LLVM module, so nothing
between the two builds -- d14da45b, caa9eb9b, b72c215b -- changes rtq's
code, and the table stands for the tree's head. (No rtq schedule has a
deferral queue, the one construct the hoist's new case reaches; the two
`queue` matches in schedules/ are comments on FCPW's enqueueNodes.) The
dumps and the compared modules are in scratch/ssa-check-b72c215b/
(`<schedule>.<build>.ssa`, `.ll`), 12 seconds of one core under a lock of
my own, nothing timed on the machine. bonsai-ac ran the same check over
apps/wosx's fourteen deferral schedules with the same result.

### Where the 5% goes: the two Embree cells under the hardware counters (2026-11-01)

The user, 00:10: "please do profile to see where that 5% goes. Don't try
to fix it until after I read your profile report and I suggest a fix." The
cells: landscape's primary rays (0.95x) and ivy's ambient occlusion (0.95x),
real three times over (above). Nothing is changed here; this section is
the report.

**Method** (00:29-00:40, the machine idle after bonsai-f9's 'done', the
lock in my name; scratch/profile-1101/run.sh, run2.sh): compare.sh's embree
schedule built and run over landscape as the table runs it (RTQ_KEEP=1
leaves rtq.out), then the driver narrowed to one kernel over one ray set
(`--batch primary --query intersect`, `--batch ao --query occluded`) with
the repeats raised so that each side runs about four seconds, under `perf
record -F 10000` with six groups of five of the Zen 5 counters, pinned as
compare.sh pins (core 11). Each cell at side 1024 and, as the control, at
2048, where landscape read 1.01x. The driver runs Embree's side and ours in
one process and one thread; aggregate.py splits the samples at the first
sample in our generated function (trace_all / occluded_all) and sums each
event in Embree's window and ours; the "traversal only" rows below take
Embree's intersector with its rtcIntersect1 wrapper and PLT stub against
our generated function (which holds the loop over the rays too), leaving
out each window's untimed setup (the hook's reset of the RTCRayHit array,
our staging copies) -- the sum the driver's clock sees. annotate.py lays
the samples over objdump's listing per instruction and per address range.
Under perf the cells read 0.93-0.96x and 0.96-0.98x; the 2048 controls
1.00x and 0.98x.

**Per ray, traversal only** (Embree / ours; a ray-pass is one ray of one
pass; landscape 159M / 158M of them, ivy 26.1M each):

| counter, per ray | landscape 1024 | landscape 2048 | ivy ao 1024 |
|---|---|---|---|
| cycles | 146.1 / 153.3 (1.049) | 98.1 / 98.0 (1.000) | 870 / 891 (1.023) |
| instructions retired | 177.2 / 172.4 (0.973) | 177.7 / 173.0 | 983 / 951 (0.968) |
| ops retired | 184.6 / 175.9 | | 964 / 945 |
| ops dispatched (op cache + decoder) | 215.0 / 204.4 | | 2940 / 3076 (1.046) |
| cycles with retire blocked, any op | 106.6 / 114.7 (+8.1) | 57.1 / 59.6 | 345 / 299 |
| ... blocked by a load | 97.6 / 102.4 (+4.8) | 49.5 / 48.3 | 146 / 93 |
| branch mispredicts | 0.121 / 0.119 | 0.101 / 0.100 | 13.78 / 14.03 (1.019) |
| branches retired, taken | 16.5, 12.9 / 8.9, 4.1 | | 89.5, 44.4 / 97.5, 36.2 |
| L1 demand fills (misses) | 0.087 / 0.083 | 0.048 / 0.050 | 4.12 / 4.17 |
| ... from DRAM | 0.060 / 0.057 | 0.026 / 0.024 | 0.002 / 0.000 |
| L2 misses (ls_rd_blk_c) | 0.71 / 0.74 | 0.35 / 0.35 | 7.0 / 7.0 |
| prefetch instructions dispatched | 3.09 / 3.08 | | 209 / 251 |
| software prefetch fills | 0.72 / 0.69 | 0.43 / 0.41 | 25.3 / 24.5 |
| ineffective software prefetches | 1.80 / 1.80 | | 36.6 / 36.7 |
| hardware prefetch fills | 1.39 / 1.04 | 1.36 / 0.96 | 8.2 / 11.3 |
| dTLB misses | 0.021 / 0.018 | 0.021 / 0.016 | 2.27 / 0.28 |
| dispatch slots empty, front end | 63.6 / 40.3 | 62.0 / 29.0 | 1962 / 2017 |
| dispatch slots stalled, back end | 884 / 985 | 509 / 553 | 2026 / 2030 |
| op-cache misses | 1.5 / 1.2 | | 0.83 / 0.93 |

(ivy at 2048: cycles 866 / 872, instructions 1005 / 953, load-blocked 137
/ 80, mispredicts 13.5 / 13.7 -- the same picture.)

**Landscape.** Ours retires 3% fewer instructions and 5% fewer ops, with
the same mispredicts, the same demand misses, L2 misses and DRAM fills,
the same prefetch instructions, prefetch fills and ineffective prefetches,
fewer TLB misses and a less empty front end, and takes 7.2 more cycles a
ray -- every one of them a cycle with retirement blocked: 4.8 waiting on a
load, 3.4 on another incomplete op. At the 2048 image the same two
binaries over the same mesh are even (98.1 against 98.0, with the same
instruction counts), and per ray the 1024 image costs both sides twice the
misses (L2 misses 0.71 against 0.35 a ray, DRAM fills 0.060 against
0.026): the loss is the cost of those extra misses, which Embree's
instruction stream hides better than ours. By region (cycles a ray, Embree
/ ours; Embree's wrapper and PLT 4.8 added to its setup, its hit-record
write 3.4 to its leaf):

| region | landscape 1024 | landscape 2048 |
|---|---|---|
| ray setup | 17.4 / 14.3 | 15.6 / 15.6 |
| stack pop, ray end | 4.8 / 6.7 | 5.0 / 5.0 |
| node test, child selection | 80.9 / 85.3 | 54.7 / 53.1 |
| children sort (two or more hits) | 0.6 / 1.6 | 0.6 / 1.1 |
| leaf (with Embree's hit write) | 42.2 / 45.3 | 22.2 / 23.3 |
| the traversal proper (the last three) | 123.7 / 132.2 | 77.5 / 77.5 |

The fixed per-ray parts are even at both sides; the traversal proper is
even at 2048 and 8.5 cycles (7%) dearer for ours at 1024, 4.4 of them in
the node loop and 3.1 in the leaf. (The per-region instruction counts are
not reliable to better than about ten percent -- the retirement samples
land in bursts after a stall clears -- so the split is read from cycles
and the totals, which are.) The load-blocked cycles by region: node loop
59.6 / 57.6, leaf 32.1 / 38.0 at 1024 (33.1 / 29.7 and 12.5 / 14.8 at
2048): in the node loop ours waits less on loads than Embree and spends
more cycles retiring; in the leaf ours waits 5.9 cycles a ray more, 0.2
of them at the first touch of a triangle block (29.3 / 29.5 -- the
dominant stall on both sides, a quarter of all cycles) and 5.7 after it.
The hottest instructions: Embree's leaf `vmovaps 0x40(%r14,%rax,1),%xmm5`
(the block's second line, after its 0x80 load) 24.7% of cycles, 30% of the
load-blocked samples; ours `vmovdqa (%rcx),%xmm1; vptest %xmm13,%xmm1`
22.9% and 28.8%. The node test's first use of the bounds: Embree's
`vfmadd132ps` 11.1%, our two bounds loads 10.1% and 4.1%. kortest/je: 5.4
+ 5.3% against 7.4 + 3.3%. The child extraction and its four prefetcht0:
Embree's vpcompressd/vpermt2q/vmovq/prefetch 7.2 + 6.8 + 4.2 + 0.5%, our
tzcnt/blsr/load/prefetch 4.6 + 4.6%. The one hardware-counter difference
of size, hardware-prefetch fills 1.39 against 1.04 a ray, is the same at
2048 where the sides are even (1.36 against 0.96) -- Embree's 96-byte
RTCRayHit stream against our 32-byte ray records, not the tree.

What the listings show different in those two loops (scratch/profile-1101/
trace_all.dis, embree-intersect.dis), for the user to weigh:

1. *The leaf's empty-block test.* Ours loads each triangle block's geomIDs
   first (`vmovdqa (%rcx)`, rcx the block's 0x90) and tests the four
   against -1 (`vptest %xmm13` with xmm13 all ones, `jb` skips the block)
   before the nine vertex and edge loads; Embree's leaf loads the nine
   (0x80, 0x40, 0x70, 0x50, 0x60, 0x00, 0x10, 0x20, 0x30) and touches the
   ids only in the hit path, the block count coming off the pointer's low
   bits (`and $0xf; add -8`). Both first-touch the block's third cache
   line; ours adds a load, a test and a branch to each block ahead of the
   test, and the branch's samples carry 36.5% of our mispredict samples
   against 33.6% at Embree's leaf top. (Read closer, 11-01 midday: ours
   has Embree's `den != 0` test too, `vcmpneqps` at 2b166, and that test
   is what excludes Embree's padding lanes, zero triangles; the geomID
   test, `vpcmpneqd` into k2, excludes the same lanes a second time, and
   its block-skip branch never fires, no block being empty. The ids sit
   at +0x90..0x9f of the 176-byte block: in the line of the +0x80 load for
   three of the four 16-byte alignments, in a line of their own for the
   fourth, base = 48 mod 64, where our first touch is to a line the
   vertex loads never need; L2 misses are +4%, so that line is mostly an
   L2 hit, overlapped with the block's other lines.)
2. *The hit record.* Both compilers take a lane out of a vector the same
   way, a store of the vector to the stack and a scalar load of the lane:
   ours `vmovaps %xmm15,0x190(%rsp); vmovss 0x190(%rsp,%rdx,1)`, six of
   them; Embree `vmovaps %xmm4,0xd0(%rsp)` ... `vmovss
   0x120(%rsp,%rcx,1)`, eight. Embree then writes the hit into the
   RTCRayHit at once, in the leaf (`vmovss %xmm2,0x20(%rsi)` tfar,
   0x30..0x4c the hit; the first reading here, "once after the
   traversal", was wrong), and its next block reads tfar back from the
   record (`vmulps 0x20(%rsi){1to4}`); ours parks t, u, v and Ng in the
   frame (0x4c, 0x48, 0x44, 0xd0(%rsp)), keeps tfar in xmm6, and writes
   the output record at the ray's end. Rare on landscape (8.3% of rays
   hit); not the loss on either side.
3. *The child pointer.* Ours loads all eight (`vmovdqu64 (%rax),%zmm3`)
   and then the first hit child's slot again by tzcnt (`mov
   (%rax,%rdx,8),%rbp`) before the four prefetches; Embree's closest-hit
   extracts it from two ymm halves in registers (vpcompressd, vpermt2q,
   vmovq). Not a CSE miss: the index is a run-time tzcnt, and a load of
   lane i from memory is what LLVM makes of an `extractelement` with a
   variable index when it can (its other lowering is a store of the
   vector to the stack and a load of the lane, which is what our node
   loop gets for the hit children's *distances*: `vmovdqa32
   %ymm8,%ymm0{%k1}{z}; vmovdqu %ymm0,0x90(%rsp); vmovdqu %ymm0,0xb0(%rsp)`
   then four `mov 0x90(%rsp,%rdx,1)`-style loads, the one vector stored
   to two slots, on every node with two or more hits). Embree's own
   any-hit traversal (embree-occluded.dis, 2f83a8) fetches the first
   child exactly as ours does, `tzcnt; mov (%rax,%r8,8),%r15`; the
   register path belongs to its closest-hit sort, where the compress
   that orders the children leaves the first in lane 0 for free. Our
   zmm3 feeds only the four-or-more path (`vpcompressq` at 2ad77). The
   samples put our extraction below Embree's (4.6 + 4.6% against 7.2 +
   6.8 + 4.2%). One more frame slot of ours: the stack depth (`mov
   %rax,(%rsp)` at each node, `mov (%rsp),%rax` at the pop) where Embree
   keeps its stack pointer in r9.
4. *The ray setup* is 108 instructions (17 mov, 16 vbroadcastss, 7 shl, 6
   or, 6 and: the six near/far byte offsets built and parked in stack
   slots) against Embree's 78 plus its wrapper; even in cycles at both
   sides, so not the loss.

**Ivy.** No memory component: ours waits 36% less on loads (93 against
146 cycles a ray), has 13% fewer retire-blocked cycles, 3% fewer
instructions, the same misses, and the TLB misses are Embree's (2.27 a
ray against 0.28). Both traversals are bound by branch mispredicts: 13.8
and 14.0 a ray, 8.1 and 8.6 of them on taken branches, and two thirds of
every op dispatched is thrown away (2940 dispatched against 964 retired;
3076 against 945). Ours has 1.9% more mispredicts and 4.6% more ops
dispatched a ray, so 155 more discarded ops a ray (2131 against 1976:
152 against 143 an event). At about 37 cycles an event (2017 empty
front-end slots a ray over 14 events is 18 cycles of bubble each, and
the 152 discarded ops took 19 cycles to dispatch) the extra 0.25
mispredicts are 9 of the 20 cycles; the rest is the node region's
longer path. By region (cycles a ray, Embree / ours,
Embree's 15.4 after a hit added to its leaf): ray setup 43.3 / 47.6, stack
pop and ray end 47.1 / 49.8, node test and child selection 344.5 / 380.6,
children push (two or more hits) 77.8 / 71.0, leaf 352.2 / 341.8.
Mispredicts a ray by region: node 5.36 / 5.94, leaf 5.47 / 5.45, push
1.17 / 1.06, setup 0.68 / 0.77, pop 0.76 / 0.82. The node region holds
the whole difference: +36 cycles, +0.58 mispredicts, 13.5 fewer
load-blocked cycles a ray. Its branches: the node hit-or-miss (kortest,
je), one child or several (`blsr; je`), then our `popcnt; cmp $2; je; cmp
$3; je; cmp $4; jne` chain to one code path for two, three and four or
more children (the schedule's any-hit pushes in stored order, nothing
sorted, as Embree's traverseAnyHit does; Embree pushes the rest in a
loop); node-or-leaf (`test $0x8; jne`). The samples spread over the same
four decisions on both sides; the per-branch attribution is a sample
behind the branch and is not finer than that.

**How Embree hides the misses** (11-01 midday, the user's question). The
counters say where the time goes and not through which mechanism: they
count fills, and a load that finds its line already in flight -- a
prefetch's, the hardware prefetcher's, a neighbouring load's -- waits and
fills nothing. Inside a node or a block the two sides request the same
lines within a few instructions of each other (the bounds' three lines;
the block's three or four), so one miss overlaps the next the same way
there, and the chain from a node's data to its child's first load is the
same length to a few cycles (Embree: compress, permute, movq; ours: kmov,
tzcnt, a load from L1). What is left to hide a miss under is the work of
the rays that follow: the out-of-order window (448 ops on Zen 5) runs
past a stalled ray's predictable end into the next ray's setup and root,
and the next ray's misses start under this one's. Between two
traversals ours spends more ops a ray -- the setup of 108 instructions
against 78, the record copied out at the ray's end -- so the window
reaches fewer of the next ray's nodes; the slot counters show it: our
back end stalls dispatch on 11% more slots a ray (985 against 884) with
37% fewer empty front-end slots (40 against 64), the window fuller with
ops that wait. This reading fits every number (proportional to the
misses; equal counts; the extra cycles at the bounds' first use and in
the block after its first touch; even at 2048, where our fewer ops offset
it), and it is a reading, not a measurement. The measurement that settles
it is AMD's instruction-based sampling (`perf mem record` over the ibs_op
PMU, checked usable unprivileged on this kernel): for each sampled load
its latency, its data source (L1, L2, L3, DRAM, in flight) and its
tag-to-retire time, per instruction. Queued behind bonsai-f9's hold on
the machine ('timing' 10:20, about an hour, a CPU sweep after).

**Ivy and `likely()`** (the user's question). Embree's hints change the
block layout only, the likely arm falling through; Zen takes no static
hint. Our layout already falls through more: 36.2 taken branches a ray
against Embree's 44.4, with as empty a front end (2017 against 1962
slots). The loss is in the data-dependent branches no hint predicts.
By instruction (occluded_all.dis under ivy-ao-1024.g1, the samples a line
behind the branch): the one-or-more test `blsr; je` carries 7.6% of our
mispredict samples (3.5% at its taken target, 4.1% at the fall-through),
the node's hit-any `kortest; je` 6.3%, the count chain's compares about
3%, the leaf's hit test 8%, the block loop 3%; the rest is spread thin.
The two encodings of the hit count -- Embree's loop exit, taken once
after k-1 falls, against our one-hot compares `cmp $2; je; cmp $3; je;
cmp $4; jne` -- give the predictor the same information in a different
history, and Embree's history predicts 11% better in that region (5.36
against 5.94 a ray). The node region's other cost: our node code is four
copies (two, three, four, more children), each with its own loads,
prefetches and jump back, and the op cache misses 12% more (0.93 against
0.83 a ray).

**What it is not**, on either cell: miss counts (demand, L2, DRAM) within
5%; prefetch instructions, their fills and the ineffective ones equal;
TLB misses fewer on ours; op-cache misses and decoder ops negligible on
both (op cache 99.9% of ops); instruction-cache misses fewer on ours.

Data: scratch/profile-1101/ -- `<cell>.g<1-6>.data` (perf), `.agg.txt`
(the window sums with the top symbols per event), `*.dis`, run.log,
run2.log; aggregate.py and annotate.py reproduce every number above
from them. The table stands; no fix before the user's word.
