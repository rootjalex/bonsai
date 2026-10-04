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

## Known-open, smaller

- The exported batch answers with the primitive id alone; Embree also
  writes `t`, `u`, `v` and the normal. Recovering `t` from the argmin
  without a second triangle test wants the key beside the element.
- A mesh that fits in one leaf gets a one-child root from the driver, where
  Embree's root would be the leaf.
- `--side 2048` was used for the tables above so that a single-threaded
  Embree run lasts over a tenth of a second; the default 1024 is fine for
  checking agreement.
