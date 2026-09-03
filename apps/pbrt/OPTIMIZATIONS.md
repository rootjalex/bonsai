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

## 1. The BVH does not order its children front to back

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

## 3. The camera ray is traced twice

`render` calls `visible_surface`, which traces, and then `integrator_li`, which
traces the same ray again. pbrt fills its `VisibleSurface` from *inside* `Li` at
depth zero, off the intersection it already has.

The reference render does the same double trace, so the comparison is honest --
but a pbrt render of these scenes into an `rgb` film traces once, and the second
traversal is a real cost on every camera sample.

Not free to fix, and the reason is in pbrt: only its **path** integrator fills a
`VisibleSurface`. `RandomWalkIntegrator` and `SimplePathIntegrator` ignore the
parameter, so a gbuffer of a random-walk render would be empty, and the whole
comparison rests on having normals for every scene. Filling it in all three
would be a divergence from pbrt in order to go faster, which this file does not
allow. The honest version is to fill it where pbrt does and have `compare.sh`
fall back to a separate pass only for the integrators where pbrt leaves it
empty.

## 4. A leaf's bounding-box test is inside its element loop

Both traversals evaluate the leaf's own guard per element:

    foreach _iter0 in data {
      if (intersects(r, leafbox) && distmax(r, leafbox) > 0.001 &&
          distmin(r, leafbox) < best) {
        ...
      }
    }

`intersects` and `distmax` against the leaf's box do not depend on the element.
`distmin(...) < best` does, because `best` changes inside the loop. So the first
two are loop-invariant and the third is not. Whether LLVM hoists them has not
been checked; if it does not, hoisting the invariant part of a leaf guard out of
the `foreach` is a compiler-side change that helps every tree query in the
language, not just this app.

## 5. Nothing is vectorized

The renderer is entirely scalar. `vectorize()` is the schedule directive with
the most headroom here and the least applied: a wavefront of suspended paths, or
the sample loop of a pixel, are both wide and independent. The obstacle is
divergence -- a layered BSDF's random walk takes a different number of steps per
lane -- which is exactly the case ISPC-style vectorization is built for.

## 6. The sample loop is deliberately sequential

    parfor s in 0u:spp { ... }

is unbound, so the compiler drops the atomic and the sum runs in sample order.
That is what makes a pixel agree with pbrt to the last bit, since pbrt adds a
pixel's samples in that order too. Binding it would parallelize a second axis
and cost that agreement.

Worth listing because it is a real choice and not an oversight: the schedule can
make it, and a comparison run and a production run might reasonably differ.

## 7. Memory the film and the lights stream

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

## Done, kept for the record

- **`any()` for shadow rays** rather than `argmin` over the same predicate, and
  a fix to the quantifier lowering so a leaf's box is tested before its
  primitives. 14173 ms to 13299 ms on killeroo-simple, every pixel identical.
- **One `GetBSDF` per intersection**, where four or five were being built per
  vertex.
- **`tagged_index` for `Shape`**, taking a `Primitive` from 208 bytes to 16.
- **A TBB-balanced `bind`**, and the Halton digit permutation table.
