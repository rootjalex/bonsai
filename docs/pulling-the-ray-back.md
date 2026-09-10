# Pulling the ray back: what an instanced traversal costs, and the rewrite

A query over a tree held in an element lowers, lays out, loopifies and sorts,
and it answers correctly. Until `pull-queries` it was not fast: the arithmetic
per node of an instance's tree was an order of magnitude more than pbrt's, for
one reason that was anticipated when the query language was settled. This is a
note on what the cost was, why the query is written the way it is, what the
rewrite that removes the cost is, and what it took from the predicate analysis
to let the query be written pbrt's way. The rule that governs all of it: the
generated code matches or beats pbrt, always. Where it did not, the question
was what in the specification or the compiler inserted the work, and the answer
turned out to be one of each.

## The measurement

`tests/bonsai/correctness/cpp/blas-tlas.bonsai`, compiled `-p ssa -b llvm
--triple x86_64-unknown-linux-gnu --mcpu generic`, in `_recloop_func1` -- the
walk over one instance's tree.

| instruction                      | before | motion in the set, pulled | motion in the tests, pulled |
| -------------------------------- | -----: | ------------------------: | --------------------------: |
| `fmul <3 x float>`               |     17 |                        11 |                           2 |
| `fadd <3 x float>`               |     29 |                         9 |                           0 |
| `minnum`/`maxnum` v3f32          |     14 |                         0 |                           0 |
| `getelementptr %struct.Instance` |      2 |                         1 |                           1 |

Before: `transform(i.render_from_instance, node_box)` -- the eight-corner box
transform of `Transform::operator()(const Bounds3f &)`, plus the min/max fold
back to a box -- at **every node**, on top of the slab test the node needed.

The middle column is the first thing tried: the query as a `map` of placed
triangles, with the ray pulled back. The node test became the slab test, but
nine multiplies and adds remained in the hit path, placing the triangle the
query returns each time a hit was recorded. That is what the last column
removes, by changing the specification rather than the compiler -- see below.
What is left is the slab test (two vector multiplies), one triangle test per
leaf element, and one read of the instance to record the hit, which pbrt does
too when it copies its `SurfaceInteraction`. `_recloop_func1` takes
`%struct.Ray %_pulled0` and the world-space ray not at all.

pbrt's cost at the same place is the slab test and nothing else; it pays about
three multiplies and three adds *once per instance*, in
`TransformedPrimitive::Intersect`. That is now our count too.

## How the query is written, and why

```
argmin(|i, tri| distmin(r, transform(i.render_from_instance, tri)),
       filter(|i, tri| intersects(r, transform(i.render_from_instance, tri)) && ..,
              flatten(|i| i.blas, instances)))
```

The set is the raw triangles -- each instance's tree as it holds them -- and
the *tests* place them: `intersects(r, transform(m, tri))`, the triangle where
the instance puts it, against the world-space ray. pbrt says
`intersects(untransform(m, r), tri)`, the ray moved back. Said our way there is
one uniform ray and one varying geometric object, and the ordinary bounding
rule applies to it; said pbrt's way the frame change hides inside a function
predicate analysis cannot see into, the triangle's box and the node's box are
in different spaces, and nothing relates them. That is a much worse program
than a slow one, so the algorithm says what it means and the lowering gets to
pbrt's arithmetic.

The set being the *raw* triangles is the specification half of the finding.
The first version wrote the set as the placed triangles, `flatten(|i|
map(|tri| transform(m, tri), i.blas), instances)`, with plain `intersects(r,
tri)` over them. Both forms bound identically and both are rewritten
identically; the difference is what `argmin` returns. A reduction has to store
the element of the set it ranges over, so the map form stores a placed triangle
at every recorded hit -- the nine multiplies and adds in the middle column --
for a result pbrt never computes. pbrt's inner `Intersect` returns the hit in
instance space and `TransformedPrimitive` transforms the *result*, once. The
raw form returns `(instance, triangle)` in the instance's frame, which is that,
and the one transform of the result is the caller's. (It also happens to be
what apps/pbrt will need: the interaction is computed from the untransformed
triangle and the pulled ray, then transformed, and that is the only way to
match pbrt's arithmetic to the bit.)

The map form is still a legitimate query -- when placed triangles *are* what is
wanted back -- and `tests/bonsai/lower/nested-trees.bonsai` keeps it, since it
is the form in which a tree's bounds have to be carried through the map.

## The rewrite

Three pieces. All three are built; `Opt/PullQueries.h` is the pass.

### 1. `untransform` is `transform` undone, and supplying one is a declaration

`transform` was already a geometric intrinsic: a *motion*, taking a motion
value and an extent and answering the extent moved, dispatched on the pair of
types so that `transform(t : Transform, b : AABB)` and `transform(t :
Transform, tri : Triangle)` are two implementations of one operation.

`untransform` is its sibling: the same motion undone, dispatched the same way.
The program supplies `untransform(t : Transform, r : Ray)` exactly as it
supplies the two above, and that pair is pbrt's `operator()` and
`ApplyInverse`. No new syntax.

The identity the rewrite applies is

```
rel(q, transform(m, x))  ==  rel(untransform(m, q), x)
```

For the topological relations it holds under any bijection and needs nothing
from the program. For the metrics it holds only if the query type's distances
are unchanged by moving both operands -- and *that* is what a program declares
by supplying an `untransform` for a query type. For a ray it means the
direction is not renormalised, so that the parametric `t` of a hit is the same
on both sides of the frame change. This is exactly why pbrt's
`Transform::ApplyInverse(const Ray &, Float *tMax)` (`util/transform.h:416`)
does not renormalise `d`, and it is the fact `tMax` rests on: an instance can
prune against a hit found in another one because both `t`s mean the same thing.
The ordering predicates -- `lex`, `ltx` and the rest -- ask about the world's
axes, which a rotation changes, so the rewrite never touches them.

The declaration is implicit in supplying the function. It could be made
explicit -- an attribute on the `untransform`, say -- as a parser-level change
on top of this; the contract itself would not move.

### 2. `pull-queries` applies it where a relation sees a `transform`

A pass after `LowerTrees`, so the pruning conditions predicate analysis emitted
exist to be rewritten; after `LowerSorts`, so the sort keys are on the
recursion; and before `LowerGeometrics`, so a motion is still a `GeomOp` rather
than a call. Wherever a relation or metric has a `transform(m, x)` on one side
and the program supplies the `untransform` for the other side's type, the motion
moves across, undone. Where the program supplies nothing the query stays as
written: correct, and slow.

This is not loop-invariant code motion and LICM could not do it. `transform(m,
node_box)` varies at every node, because the box does. The rewrite is what
turns a per-node varying term into a per-instance invariant one; only then is
there anything to hoist.

### 3. And hoists what it made

The same pass then moves each distinct `untransform(m, q)` ahead of every
recursion whose variables it does not mention -- nothing the recursion carries,
nothing its body binds, nothing its body writes. That check is what puts the
pulled ray where pbrt puts it, inside the loop over a leaf's instances and
ahead of the walk of that instance's tree, once per instance: the outer
recursion binds the instance, so the term cannot move past it, and the inner
one does not, so it can. Nothing knows where instances are.

A schedule's `sort()` keys follow the query into the pulled frame. pbrt's
front-to-back rule reads the sign of the direction along the node's split axis,
and inside an instance that axis is an axis of the instance's frame: pbrt
applies the rule after `ApplyInverse`, and a key left on the world ray would be
the right rule on the wrong ray -- never wrongly answered, since an argmin does
not depend on the order, and so never visible. `pull-the-ray-back.bonsai` pins
the inner key on `_pulled0.d` and the outer on `r.d`.

## What the analysis needed, to bound pbrt's form

The raw form was unavailable at first: predicate analysis refused any
geometric op whose operand was not a bare variable, so `intersects(r,
transform(m, tri))` stopped it cold. Three facts were missing, and each is a
fact about the traversal rather than a special case for instancing.

- **A motion maps a bound to a bound.** `x` within `V` puts `transform(m, x)`
  within `transform(m, V)` -- the same motion on the volume's type, which is
  what the type-dispatched `transform` is for -- provided `m` holds still over
  the subtree. That is `PredicateAnalysis::operand_view`.

- **Inside an instance's tree, the instance is fixed.** A flatten opens the
  inner traversal per element of the outer leaf, so while it runs the outer
  parameter names one element, not a subtree; its matrix has one value, and
  `m` does hold still. The volume map leaves such parameters out of the varying
  set and records what each stands for (`VolumeMap::fixed`), because the bound
  it emits has to name the element the traversal knows -- `_iter0` -- and not
  the lambda's `i`, which is not in scope where the bound is tested. A
  coiterated `product` reaches its second tree without passing through an
  element loop, so nothing is fixed there and nothing changes for it.

- **At the outer level, `with extent` bounds the placed expression.** The
  promise is that everything reachable through an instance lies within its
  extent, and the extent is `transform(render_from_instance, blas.AABB)` --
  the tree's box *placed*. So what the node's volume bounds is
  `transform(i.render_from_instance, tri)`, taken whole (`VolumeMap::exprs`),
  and not `tri`, which is in the instance's frame and about which the box says
  nothing. Binding the raw parameter would have been unsound; it was only ever
  right for the map form, where the parameter *is* the placed element, and
  that is still what happens there.

## The smaller things behind it

In rough order of what they are worth.

- **`any` still runs its leaf loop's counter to the end.** The body is guarded,
  so the predicate is skipped once the answer is settled, but the loop does not
  stop. Closing it is either a `While` with a compound condition -- which costs
  the countable-loop shape `vectorize` wants -- or extending the monotone
  accumulator exit in `SSA/QueueRecursion.cpp` from the stack loop to leaf
  loops, which additionally needs a check that the loop body writes nothing but
  the accumulator. A decision, not a bug.

- **The hit record loads the instance from the leaf's array.** `_recloop_func1`
  takes `(instances_index, _idx0)` and reads `instances.insts[..]` when it
  records a hit, since the pair it stores holds the instance by value. pbrt
  copies a `SurfaceInteraction` at the same moment, so this is the same class
  of work; handing the function the instance would save the address arithmetic.

- **`guard_leaf` on a leaf holding a single element** reorders the arms: `maybe`
  outside, `always` within. One test fewer on the miss path and one more on the
  always path, so a wash. It is uniform with the array case, which is why it is
  written that way.

- **A `transform` in a pruning condition with no `untransform` to move it** is
  the slow form, and nothing says so. A note under `-v` would be cheap.

## What is already good, so nobody re-derives it

Measured on the same build, all of these hold:

- The transform is fully inlined; there is no call left.
- `1 / r.d` is computed once per instance, not once per node.
- The slab test is shared between `intersects`, `distmin` and `distmax` within
  an arm, and `intersects` and `distmin` on a triangle are one intersection --
  LLVM's CSE gets both after inlining, so there is nothing to do at the IR
  level.
- The accumulator stays in registers as phis rather than a pointer, which is
  what `Allocate::unaliased` is for, and it survives crossing into the nested
  traversal.
- Both levels of the traversal prune, at every arm, with the node's bound tested
  once per leaf rather than once per element.
- Under `loopify`, each level gets a stack of its own, seeded from the row its
  element's field names.
- The ray is pulled back once per instance, the walk of the instance's tree
  tests stored geometry against it, its sort keys read the pulled direction,
  and nothing inside the walk reads the matrix.
