# Finishing the move of scheduling into SSA

Every scheduling transformation in bonsai is meant to be an SSA rewrite. Most
are: `loopify`, `split`, `collapse`, `vectorize`, `defer`, `specialize` and
`bind` all run off the block CFG, driven by `ConvertToSSA`. Two things are not,
and this is a note
about what it would take to finish them. **Neither is being done now.** It is
written down because the reasons are not obvious from the code, and because the
second one is a large enough project that it should be started deliberately
rather than fallen into.

## Where `sort()` still is, and why

`sort()` is half moved. The reordering -- the part that decides which recursive
call happens first -- is `SSA/SortRecursion.cpp`, and it works on the block CFG
like everything else. What is still at the Stmt level, in `Lower/Sorts.cpp`, is
**name resolution**: turning the schedule's `sort(primitives.Interior, |i, r,
axis| ...)` into an expression.

That part needs two things which `LowerLayouts` consumes:

- **The `Match` and its named arm.** The schedule names a tree variable and a
  variant, `primitives` and `Interior`. After `LowerLayouts` there is no
  `Match`: the arms have become a dispatch on `nPrims == 0`, and which arm you
  are in survives only as a `reinterpret_cast<_tree_layout2>`.

- **The key lambda's captured parameters.** `axis` is bound to
  `Access("axis", Unwrap(Interior, primitives))` -- a variant field named by the
  tree declaration. `Lower/Layouts.cpp` builds a per-arm `field_map` from
  `field_in_layout(tree, arm, field)`, rewrites exactly those accesses with it,
  and drops it. After that pass the name-to-offset mapping has been compiled out
  of the program.

So the constraint is not that the *ordering* wants to happen early. It is that
the *binding* has to happen while the tree is still a tree. The reordering
genuinely wants to be late, because post-layout a branch is a `u32` index and
ordering two of them is a compare and two selects; pre-layout a branch is a
subtree reference and selecting between two of them loads both nodes.

### The cheap way to remove `Lower/Sorts.cpp`

Nothing above requires a separate Stmt pass. It requires the binding facts to
still be reachable from SSA. Two pieces:

1. **Keep what `field_in_layout` worked out.** Either export the per-arm field
   map from `LowerLayouts`, or have it plant a named binding per variant field
   at the recursion point and let DCE drop the ones nothing reads. `axis`
   already survives post-layout as a struct field, because the layout block
   declares it; arm-specific fields like `offset` need the arm's reinterpret
   recorded alongside them.

2. **An `Expr` to instructions emitter usable from an SSA rewrite.**
   `ConvertToSSA` already has one, inside `FunctionBuilder`. Today
   `SortRecursion` can only hand-emit through `append`, which is fine for a
   compare and a few selects and hopeless for an arbitrary key lambda.

With both, `SortRecursion` resolves `axis` itself, emits the key, and orders the
run: one SSA pass, and `Lower/Sorts.cpp` is deleted. Roughly 200-300 lines, and
it changes no other pass's assumptions.

## Moving `LowerLayouts` after SSA conversion

The larger question. Layouts are declared in the `schedule` block, so choosing
one *is* a scheduling decision -- and it is currently the earliest and least
reversible one in the compiler, about a thousand lines running before the
recursion has even been built into a function.

### What it buys

- Every scheduling decision at one level, with one place to test them.
  `Lower/Sorts.cpp` dies as a side effect rather than as a special case.
- **Layout and vectorization could be co-designed.** Today the layout is fixed
  long before `vectorize()` runs, so a layout chosen to suit a vector width is
  not expressible. That is the interesting one.
- The layout experiments in `apps/pbrt/OPTIMIZATIONS.md` -- half-float texels,
  AoSoA node packing -- become rewrites that can be applied and compared,
  instead of front-end changes that re-run everything downstream.

### What it costs

- **SSA loses the invariant that every value has a known representation.** A
  tree reference has no size until a layout picks one. `SplitAggregates`,
  `PromoteAllocas` and `Vectorize` all reason about widths and component counts
  today and would each need a "not yet lowered" case.
- **New tree-shaped SSA ops** -- match-on-arm, field-of-arm, child-of -- which
  is a real expansion of `Instruction::Op`. Every pass that switches on ops
  grows a case or an ordering constraint, and the relooper (`SSA/CodeGen_Stmt`)
  has to be able to render them or SSA stops being readable.
- **The ordering constraint does not disappear, it moves.** `queue_recursion`
  allocates `[64 x i32]`; it has to know a branch is a `u32`. So loopify still
  has to run after the layout rewrite, and the key binding still has to run
  before it. The result is bind-key, layout, order-run, loopify as four ordered
  SSA rewrites -- cleaner than today, but the same shape, not a simplification.
- **Six passes sit between `LowerLayouts` and `ConvertToSSA` that assume
  concrete data.** `LowerForEachs` iterating a leaf's
  `range(prims, pOffset, nPrims)` is the clearest; `LowerDynamicSets`,
  `LowerYields`, `LowerScans`, `LowerTuples` and `LowerDynamicArrays` all need
  reordering or moving too.
- **No obviously-correct intermediate state.** The pbrt comparison is what tells
  us the compiler is honest. A migration that leaves the tree abstract through
  most of the pipeline has a long window in which that check says nothing.

### The reading

The cheap path gets the stated goal: all scheduling in SSA, `Lower/Sorts.cpp`
gone. Moving layout lowering is worth doing for a different reason -- making
layout schedulable at the same level as everything else, and co-designable with
`vectorize()` -- and that reason is not pressing until something wants it. Do
the cheap one when `sort()` is next touched; keep the big one as a deliberate
project, not a side effect of tidying up.
