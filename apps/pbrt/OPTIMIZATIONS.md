# apps/pbrt: optimizations noticed but not taken

A running list of things that would make this renderer faster, written down when
they are noticed rather than when they are done. Kept separate from PLAN.md
because the plan is about *what the renderer does* and this is about *what it
costs*, and the two have different rules: a plan item has to be faithful to
pbrt, and an item here has to change no pixel at all.

That is the entry condition for this file. **An optimization that changes the
image is not an optimization, it is a different renderer.** The one exception is
a change to visitation *order* inside a query whose answer does not depend on
order -- which is what most of the tree items below are, and which is exactly
what the schedule language exists to decide.

Ordered by what a measurement suggests they are worth, with the evidence.

## 1. The BVH does not order its children front to back -- DONE

**Implemented and on**, as `trace.sort(primitives.Interior, ...)` in
render.bonsai. It is pbrt's `if (dirIsNeg[node->axis])`, keyed on
`(1 / d)[axis] < 0` exactly as pbrt writes it, and every pixel is identical with
and without it -- an `argmin` over a set does not depend on the order the set is
visited in, which is why this is a schedule at all.

One measurement, on one scene, recorded because it was taken and not because it
decides anything:

    killeroo-simple, best of three, same pbrt run each time
      no sort            15.35 s
      front to back      16.50 s
      back to front      18.20 s

The middle row against the last says the ordering works: front to back really is
finding the near child, and getting it backwards costs 1.7 s. Against the first
row it says that on *this* scene the ordering does not pay for itself, and there
is a plausible story about why -- `left = index + 1`, so descending left first
walks the node array forwards and streams, while sorting jumps to
`index + offset` about half the time; 66k triangles is small enough to be close
to cache-resident, which is the case where locality beats pruning.

**That is not a reason to turn it off.** One small scene is not evidence about
the schedule a renderer should ship, and this renderer's first job is to compute
what pbrt computes -- the point of matching is to have a baseline that is
trustworthy before anything is tuned against it. The scenes that would actually
test the question, ganesha and crown, need mesh area lights and `volpath` first.
Revisit when there is a real workload to revisit it with.

Two things turned up on the way to it. `sort()` used to be lowered at the Stmt
level, before the layout ran, so it selected between whole subtree *values*
rather than node indices; the ordering is an SSA rewrite now
(`SSA/SortRecursion.h`) and costs a compare and two selects on `u32`s. And the
simplifier folded `select(c, 1, 0)` to `cast(!c)` -- inverted -- which is what
the third row above was measuring before it was fixed.

### The original note



**pbrt does and we do not**, which means our traversal tests more primitives
than pbrt's for the same ray. From the lowered IR:

    | Interior(low, high, left, right, axis : u8) ->
      if (...) { from ((primitives as Interior).left, (primitives as Interior).right) }

`from (left, right)` is a fixed order. pbrt's `BVHAggregate::Intersect`:

    if (dirIsNeg[node->axis]) { push(nodeIndex + 1); nodeIndex = secondChildOffset; }
    else                      { push(secondChildOffset); nodeIndex = nodeIndex + 1; }

so it descends into whichever child the ray reaches first. That makes the
running `best` tighten sooner, and every subtree whose `distmin` is already
past `best` is skipped -- which is the whole point of carrying `best` at all.

The `axis` a node was split on is **already in the layout** and already
unused by anything:

    switch nPrims {
        0 => Interior { offset : u32; left = index + 1; right = index + offset; };
    ...
    | Interior(left : PrimitiveBVH, right : PrimitiveBVH, axis : u8)

so this is a scheduling question and not a data one. It changes no answer: an
`argmin` over a set does not depend on the order the set is visited in. It is
the largest single item here, because the traversal plus its leaf tests is about
30% of a path render.

Two things to work out. The `from` construct takes an ordered pair, so
expressing "near child first" needs either a conditional `from` or a way for the
schedule to say that a tree's children are ordered by a field. And it is
strictly a *pruning* win, so it does nothing for `trace_any`, which has no
running bound.

## 2. `sphere_roots` is 13% of killeroo-simple, for one sphere

Measured, `perf` on the path render:

    _traverse_tree0     15.0
    dielectric_sample_f 14.7
    sphere_roots        13.1   <--
    halton_dimension     8.3
    triangle_hit         7.3
    _traverse_tree1      7.1

The scene has 66,533 triangles and **one** sphere, which is the light. Every
shadow ray is aimed at it, so every shadow ray reaches its leaf and solves the
quadratic -- and that quadratic is pbrt's interval-arithmetic one, which is a
dozen outward-rounded operations where a plain float version is four.

pbrt pays this too, so it is not a divergence. What it suggests is a *fast
path*: the intervals exist to make the sign of the discriminant trustworthy at
grazing incidence, and away from grazing the plain-float discriminant decides
the same way. Computing the cheap one first and falling back to intervals only
when the result is near zero would be exactly equal and much cheaper. The
"near zero" test has to be conservative enough to be provably safe, which is
the work.

Worth saying: this profile is not typical. A scene whose emitter is not also
the thing every ray points at would spend this differently.

## 3. The renderer computes a gbuffer pbrt's integrator would not

Found the moment the comparison started using the `pbrt` binary, and invisible
before it.

pbrt fills a `VisibleSurface` from inside `Li`, and **only its `path` and
`volpath` integrators do**. `RandomWalkIntegrator` and `SimplePathIntegrator`
take the parameter and ignore it, so a gbuffer of one of those renders is all
zeros -- pbrt spends nothing on normals or reflectance. This renderer computes
all three channels whatever the integrator is, which on those scenes is a
second traversal and a sixteen-sample `rho` per camera sample that pbrt does
not spend at all. Measured, against the binary:

    area-light       randomwalk   1.14x slower than pbrt
    area-light-path  simplepath   1.05x slower
    envmap-walk      randomwalk   1.75x slower

Those are the only scenes here where this renderer is slower than pbrt, and
this is why. It is the same class of finding as the diffuse `rho` shortcut, in
the other direction: work done that pbrt does not do.

The fix is the same one as the item below -- fill the visible surface from
inside the integrator, where pbrt fills it, and only in the integrator that
fills it.

## 4. The camera ray is traced twice

`render` calls `visible_surface`, which traces, and then `integrator_li`, which
traces the same ray again. pbrt fills its `VisibleSurface` from *inside* `Li` at
depth zero, off the intersection it already has.

There is no third renderer here and "the reference" is not one: it is pbrt-v4.
`render_reference` in scene_dump.cpp uses pbrt's own camera, aggregate, BSDFs
and integrator -- what it replaces is only pbrt's `RenderCPU` loop around them.
But it is *our* loop, and it is the thing that traces twice: it calls `Li` for
the radiance and then intersects again for the gbuffer. So both sides pay a
traversal that **pbrt itself would not** -- `RenderCPU` fills its
`VisibleSurface` from inside `Li`, off the intersection it already has.

The comparison between the two sides is therefore fair, and neither side is
what pbrt does. That second point is the one worth fixing.

Not free to fix, and the reason is in pbrt: only its **path** integrator fills a
`VisibleSurface`. `RandomWalkIntegrator` and `SimplePathIntegrator` ignore the
parameter, so a gbuffer of a random-walk render would be empty, and the whole
comparison rests on having normals for every scene. Filling it in all three
would be a divergence from pbrt in order to go faster, which this file does not
allow. The honest version is to fill it where pbrt does and have `compare.sh`
fall back to a separate pass only for the integrators where pbrt leaves it
empty.

## 5. A leaf's bounding-box test is inside its element loop -- DONE

Both traversals evaluate the leaf's own guard per element:

    foreach _iter0 in data {
      if (intersects(r, leafbox) && distmax(r, leafbox) > 0.001 &&
          distmin(r, leafbox) < best) {
        ...
      }
    }

`intersects` and `distmax` against the leaf's box do not depend on the element.
`distmin(...) < best` does, because `best` changes inside the loop.

**Fixed**, and it was a mistake in the lowering rather than something for LLVM
to clean up. `RewriteFilter` emitted the predicate's bound over the node's
volume at each *yield*, which inside a leaf means once per element;
`visit(Scan)` had always emitted the interior's once per node. It is emitted
around the loop now.

Hoisting is exact even though the bound contains the running accumulator:
`best` only ever decreases, so the condition evaluated at loop entry is the
weakest it will be and a leaf is never skipped that should have been entered.
What is given up is abandoning the rest of a leaf part-way through, which for
four primitives is not worth one box intersection each.

## 6. Nothing is vectorized -- DONE, and the traversal is a packet

The renderer was entirely scalar; the note said `vectorize()` was the schedule
directive with the most headroom and the least applied. It is applied now:
`render.split(s, s_gang, s_lane, 16, false).vectorize(s_lane)` runs a pixel's
samples as a sixteen-lane gang, the whole path trace at once, with divergence
-- a layered BSDF's random walk taking a different number of steps per lane --
handled by ISPC-style masking and Moll & Hack's partial linearization.

The traversal's form is decided by the order of two directives, and the order
was the second half of the work. Written `trace.loopify(64)` and then
`render.vectorize(s_lane)`, the traversal is a loop each lane walks at its own
pace over a stack of its own: sixteen node indices gathered at every step,
which for the coherent gang a pixel's samples make is one address fetched
sixteen times, and was two thirds of the vectorized time on book. Written the
other way round -- `vectorize` first, `trace.sort(...).loopify(64)` after --
the recursion is made by the gang as a whole, on one node the lanes share with
a mask of the lanes whose ray reached it, and loopify puts *that* on a stack:
one of scalar node indices and one of masks, each node's bounds and children
loaded once and broadcast, the sort's order settled by a vote of the lanes, and
a child pushed only if some lane is on. That is Wald's packet traversal
(Wald, Slusallek, Benthin & Wagner, Eurographics 2001), for the top-level tree
and for every instance's tree beneath it, derived rather than written. See
PLAN.md, "What is next", item 6, for the measurements.

## 7. The sample loop is deliberately sequential

    parfor s in 0u:spp { ... }

is unbound, so the compiler drops the atomic and the sum runs in sample order.
That is what makes a pixel agree with pbrt to the last bit, since pbrt adds a
pixel's samples in that order too. Binding it would parallelize a second axis
and cost that agreement.

Worth listing because it is a real choice and not an oversight: the schedule can
make it, and a comparison run and a production run might reasonably differ.

## 8. Memory the film and the lights stream

    environment map, fitted     67 MB   (4 floats per texel, 2048x2048)
    its two distributions       68 MB
    halton digit permutations   26 MB

The fitted texels are `(c0, c1, c2, scale)` and every one of them is a value the
sigmoid tolerates being approximate in; half floats would halve the 67 MB and
the distributions with it. Layout is explicitly the schedule's business here, so
this is in scope in a way that changing an algorithm would not be.

Unmeasured. A 2048x2048 map read by an escaped ray is a near-random access into
67 MB, which is a cache miss per escaped ray either way -- halving the footprint
may or may not change the miss rate.

## 9. Keep `RayDifferential` off the `Ray`

Not a saving so much as a cost not to incur, written down before the mistake is
made. Textures need a ray to carry `rxOrigin`, `rxDirection`, `ryOrigin`,
`ryDirection` and a flag -- 49 more bytes -- and the obvious place to put them
is on `Ray`, which is passed by value into the tree query.

The traversal never reads any of them. pbrt separates `RayDifferential` from
`Ray` and passes the base to `Intersect`; the same split here keeps the ray the
BVH copies at 28 bytes. Worth stating because the fields are only ever used
together with the ray, which makes putting them on it look natural.

## 10. A load per lane and a transpose in place of gathers -- tried, a loss

Written down so it is not tried again on its own. A gang reading an N-word
element from each lane's own index -- a BVH node, a triangle's vertices --
takes N gathers, and profiling book at depth 1 (sixteen lanes, Zen 5) put two
thirds of the vectorized time inside gather instructions at an IPC of 0.66. So
the backend was taught to read such an element as a masked 128-bit load per
lane and block of four words followed by a transpose (Halide's interleave
network), chosen per read by a cost model with per-target constants, only for
the words the gang goes on to read.

A microbenchmark of the reads in isolation favoured the transpose by 15 to 35
percent (a twelve-word node at sixteen lanes, random indices: 25.8 ns per gang
against 34.9 for gathers; coherent: 24.4 against 33.4). The render said the
opposite, at depth 1 and 64 spp: killeroo's vectorized time went from 1.465 s
to 3.08 s, and still 2.78 s once the transpose was limited to reads whose
fields were statically known; ganesha 10.2 to 14.3 s; bvh-lights 99 to 141 ms;
book, the one divergent scene, gained 5 percent. Against pbrt, killeroo went
from 1.42x faster to 1.21x slower. Reverted.

Two things the microbenchmark could not show. The traversal loop already
saturates the shuffle port with its blends and mask moves and is under
register pressure, so sixty extra shuffles and sixteen live 128-bit loads per
node cost far more there than in an empty loop. And on a coherent gang --
sixteen samples of one pixel at the same node -- a gather's sixteen addresses
fall in one or two cache lines and it is cheap; the profile's two thirds
"inside gathers" was the gang fetching the same address sixteen times, which
the packet traversal (item 6 above) removes outright: the gang shares the
node, so there is no gather to transpose. If a transpose is ever worth
revisiting it is for the per-lane schedule alone, judged on the render.

## Done, kept for the record

- **A leaf's volume bound hoisted out of its element loop** -- item 5.
- **`any()` for shadow rays** rather than `argmin` over the same predicate, and
  a fix to the quantifier lowering so a leaf's box is tested before its
  primitives. 14173 ms to 13299 ms on killeroo-simple, every pixel identical.
- **One `GetBSDF` per intersection**, where four or five were being built per
  vertex.
- **`tagged_index` for `Shape`**, taking a `Primitive` from 208 bytes to 16.
- **A TBB-balanced `bind`**, and the Halton digit permutation table.
