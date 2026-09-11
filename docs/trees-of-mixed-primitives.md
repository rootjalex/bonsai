# A tree of mixed primitives

pbrt's top-level `BVHAggregate` holds primitives of two kinds side by side:
`GeometricPrimitive`, a shape with a material, and `TransformedPrimitive`, a
transform and a whole tree of them. `Primitive::Intersect` dispatches on which
one it has, and the two arms cost what they cost: a shape test for the first,
`ApplyInverse` and a walk of the held tree for the second. This is a note on
how that scene is written in bonsai, what the compiler needed to lower it, and
the measurement that says the traversal does what pbrt's does and no more.
pulling-the-ray-back.md is the half of the story where every element is an
instance; this is the half where some are not.

## The program

```
element Prim =
    | Solo(tri : Triangle)
    | Inst(render_from_instance : Transform, blas : set[Triangle]);

func geometry(p : Prim) -> set[Triangle] =
    match p { Solo(tri) => set[Triangle]{tri}, Inst(m, blas) => blas };

element Prim with extent = |p| transform(p, geometry(p).AABB);

func transform(p : Prim, b : AABB) -> AABB { match p { Solo(tri) => return b;
                                                       Inst(m, blas) => return transform(m, b); } }
func untransform(p : Prim, r : Ray) -> Ray { match p { Solo(tri) => return r;
                                                       Inst(m, blas) => return untransform(m, r); } }

func trace(r : Ray) =
    argmin(|p, tri| distmin(r, transform(p, tri)),
           filter(|p, tri| intersects(r, transform(p, tri)) && ..,
                  flatten(|p| geometry(p), prims)));
```

The element is a variant, because pbrt's is (`Primitive` is a
`TaggedPointer`). What a query reaches through one depends on the arm --
one triangle, or an instance's tree -- and `geometry` says which, once. The
queries and the extent all name it, as pbrt's `Bounds()` and `Intersect()`
dispatch on the same tag. The primitive itself is the motion: a
`TransformedPrimitive` moves what it holds and a `GeometricPrimitive` holds it
where it is, so `transform(p, ..)` and `untransform(p, ..)` are a match on `p`
that does the one or the other, and the query is written for both arms at once
as `intersects(r, transform(p, tri))`, the triangle where its primitive puts
it.

## What the language needed

Five things, each small, none specific to instancing.

- **A match that is a value.** `match p { Solo(tri) => .., Inst(m, blas) =>
  .. }` as an expression, one arm per variant, all arms of one type. Its arms
  bind nothing in the IR: the parser resolves `tri` to `(p as Solo).tri`, a
  read off the value known to be that variant (`Unwrap`, which now applies to
  a variant as it did to a tree node), so an arm is a plain expression over
  the enclosing scope and every analysis sees through it with nothing to scope.
  A set-typed one is opened by the traversal into a match statement with one
  traversal per arm. A value-typed one is a branch -- an arm reads the fields
  of the variant the value is, so it cannot be a `Select` -- and LowerADTs
  gives it a function of its own whose body is the same match as a statement;
  the backend inlines the call. lower/match-expr.bonsai shows the lifted
  function and correctness/llvm/match-expr.bonsai runs it.

- **A set literal**, `set[Triangle]{tri}`, spelled the way a vector literal is.
  A set is never a value at runtime; this one is traversed by yielding what it
  lists.

- **A set-valued function** as a name for part of a query. `geometry` is called
  from inside `flatten` and from the extent, and a traversal is built over the
  whole query at once, so LowerSetFunctions puts its expression back where it
  is called, before anything reads a set expression, and drops the definition
  once it is spent. An exported set-valued function is a query of its own,
  materialised for its caller, and is left alone.

- **An extent as a function of the element.** A struct's extent is written
  over its fields; a variant has no fields in common across its arms, so it is
  written `|p| ..` and dispatches inside -- and it is recorded as a function
  either way, which is what every reader wanted. `element Prim with extent =
  ..;` after the element lets the extent name `geometry`, which takes a `Prim`
  and so cannot precede it. What the extent says is `Primitive::Bounds()`:
  the geometry the primitive holds, placed by the primitive.

- **A variant as a tree's element and as the owner of a nested tree.** The
  schedule binds the arm's field, `Inst.blas : BLAS from BlasNodes`; the layout
  stores the variant `tagged_index`, one pool per arm, with the `Inst` pool
  holding the row its tree is rooted at in place of the set.

## What the analysis had to see

The outer tree's node bounds `transform(p, tri)` for every element beneath it
and every triangle it reaches, whichever arm the element is: that is what the
extent promises, said about the set the flatten reaches. `placed_shape` finds
the set's root bound in the extent -- `geometry(p).AABB`, the `.AABB` of a
set-typed expression that mentions the field the nested match was reached
through -- and puts the query's inner parameter in its place, giving
`transform(p, tri)` as the expression the node's box bounds. That is the same
rule the struct form used, with the field access generalised to any set
expression that reaches the field.

Inside an instance's tree the instance is fixed, as before; the element the
level was reached through is `p` itself and not `(p as Inst)`, since the bound
is stated over the query's parameter. Nothing else changed. The Solo arm has
no tree and no node: the filter's test is emitted directly against the
triangle, guarded, like everything in the leaf, by the leaf's own box tested
once ahead of the element loop.

## The measurement

backends/llvm/tree-traversal-mixed.bonsai, `_recloop_func0`, the walk over
primitives. Per element of a leaf:

- one tag test, `icmp ult i64 %handle, 2^56`, and nothing else the arms share.
- Solo: `intersectsp_ray_tri(%r, tri)` -- the ray as it arrived. The source
  has `untransform(p, r)` in every test of this arm, because the query says
  `transform(p, tri)` for both kinds of primitive and pulling the ray back
  rewrites both; it is a match on `p` that answers `r` here. Inlined into a
  block the tag test already put in the Solo case, LLVM folds the second test
  of the same tag, and the arm reads the ray it was given. No transform, no
  copy, no call.
- Inst: one `untransform_Transform_Ray`, then `_recloop_func1` with the pulled
  ray and the running best. The walk of the instance's tree is the one
  pulling-the-ray-back.md measures: slab tests, one triangle test per leaf
  element, and one read of the primitive to record a hit.

pbrt's `Primitive::Intersect` at the same place: the tag dispatch, then
`Triangle::Intersect` for one arm and `ApplyInverse` plus `BVHAggregate::
Intersect` for the other. The same count.

correctness/cpp/mixed-tlas.bonsai runs it with a standalone triangle in front
of an instance's triangle in one lane and behind one in another, so the
running best is checked crossing both ways between the arms; loopified and
sorted schedules print the same lines.
