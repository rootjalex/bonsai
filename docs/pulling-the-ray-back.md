# What an instanced traversal costs, and the rewrite that fixes it

A query over a tree held in an element lowers, lays out, loopifies and sorts,
and it answers correctly. It is not fast: the arithmetic per node of an
instance's tree is an order of magnitude more than pbrt's, and for one reason
that was anticipated when the query language was settled and has not been built
yet. **Nothing here is being done now.** It is written down because the fix is
an *algebraic* rewrite rather than an ordinary optimisation, so nobody is going
to fall into it while doing something else, and because the second half of it
only works if the first half is stated exactly right.

## The measurement

`tests/bonsai/correctness/cpp/blas-tlas.bonsai`, compiled `-p ssa -b llvm
--triple x86_64-unknown-linux-gnu --mcpu generic`, in `_recloop_func1` -- the
walk over one instance's tree. Per node visited:

| instruction              | count |
| ------------------------ | ----: |
| `fmul <3 x float>`       |    17 |
| `fadd <3 x float>`       |    29 |
| `minnum`/`maxnum` v3f32  |    14 |
| `getelementptr %struct.Instance` | 2 |

That is `transform(i.render_from_instance, node_box)` -- the eight-corner box
transform of `Transform::operator()(const Bounds3f &)`, plus the min/max fold
back to a box -- evaluated at **every node**, on top of the slab test the node
actually needed.

pbrt's cost at the same place is the slab test and nothing else. It pays about
three multiplies and three adds *once per instance*, in
`TransformedPrimitive::Intersect`, and then tests the stored boxes untouched. An
instance whose tree visits a hundred nodes costs us some six thousand vector
operations against pbrt's six.

## Why the query is written the way it is

Not by accident, and the shape should not be changed. The query says

```
argmin(.., filter(|i, tri| intersects(r, tri) && .., 
                  flatten(|i| map(|tri| place(i, tri), i.blas), instances)))
```

-- the triangle moved forward into render space -- where pbrt says
`intersects(pull(i, r), tri)`, the ray moved back into instance space.

Said this way there is one uniform ray and one varying geometric object, and the
ordinary bounding rule applies to it: a varying geometric argument is replaced
by its bounding volume, `with extent` relates the inner tree's box to the outer
tree's, and the whole pruning chain is three annotations and no new concept.

Said pbrt's way, the frame change hides inside a function predicate analysis
cannot see into. The triangle's box is in the instance's frame and the enclosing
node's is in the world's, and nothing relates them, so the outer level cannot
prune at all. That is a much worse program than a slow one.

So the algorithm stays as it is and the lowering is what has to get to pbrt's
arithmetic. That is the standing rule -- the IR does not have to match pbrt's
code shape, it has to reach it after lowering and simplification.

## The rewrite

Three pieces, in order.

### 1. `place` and `pull` are inverse, and something has to say so

The identity is

```
geomop(r, place(i, x))  ==  geomop(pull(i, r), x)
```

for `intersects`, `distmin`, `distmax` and the rest of the geometric
predicates, where `place` and `pull` are inverse affine maps.

It holds because an affine map takes a line to a line and preserves the
parameter along it, *provided the direction is transformed rather than
renormalised*. That is exactly why pbrt's `Transform::ApplyInverse(const Ray &,
Float *tMax)` (`util/transform.h:416`) does not renormalise `d`: a hit at
parametric `t` in instance space is at the same `t` in render space, which is
what makes `tMax` comparable across the frame change and what lets one instance
prune against a hit found in another. The same fact is what makes this rewrite
value-preserving for the distance predicates and not only for the boolean ones.

Where the identity should live is the open question. It is not something
predicate analysis should infer -- it is a property of a pair of user functions,
so the user has to declare it. Some form of

```
inverse place(i, _) = pull(i, _);
```

in the tree/ADT language, checked no further than taking the program's word for
it, is the smallest thing that works. Note that `place` is already the thing the
ADT language knows about: `with extent = transform(render_from_instance,
blas.AABB)` names the same map, applied to a volume rather than to an element.
Deriving the pair from the extent, the way `set_nested_volume_maps` already
derives the volume map from it, is worth looking at before adding syntax.

### 2. Apply it where a geometric op sees a `place`

A rewrite over the lowered traversal: wherever a geometric operation has a
uniform first argument and a `place(i, x)` second, rewrite to `pull(i, r)` and
`x`. This is not LICM and LICM cannot do it -- `place(i, node_box)` varies per
node, because the box does. The rewrite is what turns a per-node varying term
into a per-instance invariant one.

It has to reach the *bounds* as well as the elements. The expensive term is
`transform(i.rfi, build<AABB>(node.low, node.high))` in the pruning condition,
which predicate analysis emitted from the `with extent` annotation, not
something the query wrote. So the rewrite belongs after predicate analysis has
run and before the layout turns the boxes into loads -- or, if that is awkward,
on the SSA where the shape is uniform and the geometric ops are still calls.

### 3. Then LICM hoists it

`pull(i, r)` depends on the instance and not on the node, so once step 2 has run
it is loop-invariant across the whole inner traversal. Hoisting it to the top of
`_recloop_func1` gives one ray transform per instance, which is pbrt's cost.
Better still is hoisting it into the outer leaf's element loop body, where the
instance is already loaded -- same count, one less argument threaded.

There is no LICM pass on the SSA today. A traversal function is entered once per
instance, so hoisting to the function's entry block is enough here and is
nearly free to write; a general one is a bigger project and is not required for
this.

## The smaller things behind it

In rough order of what they are worth.

- **The instance is re-derived at every node.** `_recloop_func1` takes
  `(instances_index, _idx0)` and reconstructs
  `instances.insts[reinterpret_cast<..>(..).pOffset + _idx0]` wherever it needs
  the transform -- two dependent loads per node. It should be handed the
  instance. Mostly moot after the rewrite above, since the transform becomes the
  only reader and it runs once.

- **`any` still runs its leaf loop's counter to the end.** The body is guarded,
  so the predicate is skipped once the answer is settled, but the loop does not
  stop. Closing it is either a `While` with a compound condition -- which costs
  the countable-loop shape `vectorize` wants -- or extending the monotone
  accumulator exit in `SSA/QueueRecursion.cpp` from the stack loop to leaf
  loops, which additionally needs a check that the loop body writes nothing but
  the accumulator. A decision, not a bug.

- **`guard_leaf` on a leaf holding a single element** reorders the arms: `maybe`
  outside, `always` within. One test fewer on the miss path and one more on the
  always path, so a wash. It is uniform with the array case, which is why it is
  written that way, but if the always path ever matters it is worth revisiting.

## What is already good, so nobody re-derives it

Measured on the same build, all of these hold today:

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
