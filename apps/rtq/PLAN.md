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
