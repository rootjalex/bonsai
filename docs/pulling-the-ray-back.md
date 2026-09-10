# Pulling the ray back: what an instanced traversal costs, and the rewrite

A query over a tree held in an element lowers, lays out, loopifies and sorts,
and it answers correctly. Until `pull-queries` it was not fast: the arithmetic
per node of an instance's tree was an order of magnitude more than pbrt's, for
one reason that was anticipated when the query language was settled. This is a
note on what the cost was, why the query is written the way it is anyway, what
the rewrite that removes the cost is, and why it is an *algebraic* rewrite
rather than an optimisation anyone would fall into while doing something else.

## The measurement

`tests/bonsai/correctness/cpp/blas-tlas.bonsai`, compiled `-p ssa -b llvm
--triple x86_64-unknown-linux-gnu --mcpu generic`, in `_recloop_func1` -- the
walk over one instance's tree.

| instruction                      | before | after |
| -------------------------------- | -----: | ----: |
| `fmul <3 x float>`               |     17 |    11 |
| `fadd <3 x float>`               |     29 |     9 |
| `minnum`/`maxnum` v3f32          |     14 |     0 |
| `getelementptr %struct.Instance` |      2 |     1 |

Before: `transform(i.render_from_instance, node_box)` -- the eight-corner box
transform of `Transform::operator()(const Bounds3f &)`, plus the min/max fold
back to a box -- at **every node**, on top of the slab test the node needed.

After: the node test is the slab test. The remaining multiplies and adds are
the two arms' slab tests (only one runs per node) and the yielded triangle
being placed in the hit path, once per hit. The one instance read is the same:
the matrix is read to build the value the query returns, not to test a node.
`_recloop_func1` takes `%struct.Ray %_pulled0` as a parameter and the
world-space ray not at all.

pbrt's cost at the same place is the slab test and nothing else; it pays about
three multiplies and three adds *once per instance*, in
`TransformedPrimitive::Intersect`. That is now our count too.

## Why the query is written the way it is

Not by accident, and the shape should not be changed. The query says

```
argmin(.., filter(|i, tri| intersects(r, tri) && ..,
                  flatten(|i| map(|tri| transform(i.render_from_instance, tri),
                                  i.blas),
                          instances)))
```

-- the triangle moved forward into render space -- where pbrt says
`intersects(untransform(m, r), tri)`, the ray moved back into instance space.

Said this way there is one uniform ray and one varying geometric object, and the
ordinary bounding rule applies to it: a varying geometric argument is replaced
by its bounding volume, `with extent` relates the inner tree's box to the outer
tree's, and the whole pruning chain is three annotations and no new concept.

Said pbrt's way, the frame change hides inside a function predicate analysis
cannot see into. The triangle's box is in the instance's frame and the enclosing
node's is in the world's, and nothing relates them, so the outer level cannot
prune at all. That is a much worse program than a slow one.

So the algorithm stays as it is and the lowering is what gets to pbrt's
arithmetic. That is the standing rule -- the IR does not have to match pbrt's
code shape, it has to reach it after lowering and simplification.

## The rewrite

Three pieces. All three are built; `Opt/PullQueries.h` is the pass.

### 1. `untransform` is `transform` undone, and supplying one is a declaration

`transform` was already a geometric intrinsic: a *motion*, taking a motion
value and an extent and answering the extent moved, dispatched on the pair of
types so that `transform(t : Transform, b : AABB)` and `transform(t :
Transform, tri : Triangle)` are two implementations of one operation. That is
what lets a mapped tree's bounds be the mapped bounds without a second
derivation.

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
exist to be rewritten, and before `LowerGeometrics`, so a motion is still a
`GeomOp` rather than a call to `transform_Transform_AABB`. Wherever a relation
or metric has a `transform(m, x)` on one side and the program supplies the
`untransform` for the other side's type, the motion moves across, undone. Where
the program supplies nothing the query stays as written: correct, and slow.

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

A schedule's `sort()` keys follow the query into the pulled frame, which is
why the pass runs after `LowerSorts`. pbrt's front-to-back rule reads the sign
of the direction along the node's split axis, and inside an instance that axis
is an axis of the instance's frame: pbrt applies the rule after `ApplyInverse`,
and a key left on the world ray would be the right rule on the wrong ray --
never wrongly answered, since an argmin does not depend on the order, and so
never visible. `pull-the-ray-back.bonsai` pins the inner key on `_pulled0.d`
and the outer on `r.d`.

`tests/bonsai/lower/pull-the-ray-back.bonsai` is the rewrite on its own;
`tests/bonsai/backends/llvm/tree-traversal-instanced.bonsai` is what it
compiles to; `blas-tlas{,-loopified,-sorted}` run it and print the same answers
they printed before.

## The smaller things behind it

In rough order of what they are worth.

- **`any` still runs its leaf loop's counter to the end.** The body is guarded,
  so the predicate is skipped once the answer is settled, but the loop does not
  stop. Closing it is either a `While` with a compound condition -- which costs
  the countable-loop shape `vectorize` wants -- or extending the monotone
  accumulator exit in `SSA/QueueRecursion.cpp` from the stack loop to leaf
  loops, which additionally needs a check that the loop body writes nothing but
  the accumulator. A decision, not a bug.

- **The instance is re-derived in the hit path.** `_recloop_func1` still takes
  `(instances_index, _idx0)` and reconstructs the instance to build the yielded
  triangle. Once per hit rather than once per node now, so it stopped
  mattering, but handing the function the instance would be tidier.

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
  an arm -- LLVM's CSE gets it after inlining, so there is nothing to do at the
  IR level.
- The accumulator stays in registers as phis rather than a pointer, which is
  what `Allocate::unaliased` is for, and it survives crossing into the nested
  traversal.
- Both levels of the traversal prune, at every arm, with the node's bound tested
  once per leaf rather than once per element.
- Under `loopify`, each level gets a stack of its own, seeded from the row its
  element's field names.
- The ray is pulled back once per instance, and the walk of the instance's tree
  tests stored geometry against it.
