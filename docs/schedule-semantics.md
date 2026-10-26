# Scheduling Language Semantics

A Bonsai program says what to compute. A schedule says how: in what order
the loops run, on what hardware, and when the work behind each call is
done and where its state lives. A schedule never changes the result of a
program. This document defines each directive by the loop nest it
produces, in the manner of the Halide paper (Ragan-Kelley et al., PLDI
2013): a small renderer is the running example, and every directive is
shown as the change it makes to that renderer's loops.

The directives fall into three groups. *Loop transformations* change the
order in which iterations run (Section 2). *Hardware bindings* say which
unit of the machine runs a loop or a function (Section 3). *Continuation
styles* say when the work behind a call is done and where the state it
needs is kept while it waits (Section 4). Section 5 covers what fits none
of the three. The grammar is in `grammar_schedule.tex`; the headers under
`include/SSA/` document each implementation.

## 1. Programs and their loop nests

A program is a set of functions. A body has bindings of pure values, loads
and stores, atomic accumulates (`atomic x += e`), calls, `if` and `match`,
`for` loops, `parfor` loops, and `return`. A `for` loop runs its iterations
in order; an iteration may depend on the one before. A `parfor` loop says
its iterations are independent: distinct iterations write distinct
locations, or write through atomic accumulates. The compiler may run the
iterations of a `parfor` in any order, in parallel, or interleaved at any
granularity, and may reassociate its accumulates (Halide's guarantee for
reductions). That freedom is what every directive uses, and the condition
under which each is valid is stated as a condition on the loops involved.

The running example renders an image of `W x H` pixels, one path per
pixel:

    func shade(h : Hit, ray : Ray, depth : i32, max_depth : i32,
               rad : reduce(+) Float) -> void {
        rad += emitted(h);
        if depth + 1 >= max_depth {
            return;
        }
        step(bounce(h, ray), depth + 1, max_depth, rad);
        return;
    }
    func step(ray : Ray, depth : i32, max_depth : i32,
              rad : reduce(+) Float) -> void {
        hit = trace(ray);                  // the nearest intersection, an option
        if hit {
            h = *hit;
            shade(h, ray, depth, max_depth, rad);
            return;
        }
        rad += sky(ray);
        return;
    }
    func render(max_depth : i32, film : mut array[Float, W*H]) -> void {
        parfor p in 0:W*H {
            ray = camera(p);
            rad : reduce(+) Float = 0.0;
            step(ray, 0, max_depth, rad);
            film[p] = rad;
        }
        return;
    }

The radiance is one number, to keep the listings short. This program, with
eight pixels and a scene of one wall on a line, is
`tests/bonsai/correctness/llvm/semantics-render.bonsai`. Every schedule
below is a test beside it that prints the same film, and an SSA golden of
the same name under `tests/bonsai/ssa` shows the rewrite.

With no schedule the program runs as written: one loop, and a recursion
of depth at most `max_depth` inside each iteration.

    par for each p in 0..W*H:
        ray = camera(p); rad = 0
        step(ray, 0, max_depth, rad)       // recursion: trace, shade, trace, ...
        film[p] = rad

Loop nests are written as in the Halide paper: `for each` is a sequential
loop, `par for each` a parallel one, `vec for each` a loop whose
iterations are the lanes of one SIMD gang.

Inside the compiler a program is in SSA form, and every directive is a
rewrite of it. A function is a list of blocks. A block takes arguments in
place of phi nodes, holds instructions, and ends in one terminator: a
jump to a block with arguments; a dispatch on a condition to one of
several jumps; a return; a call, which names the callee and the block to
continue at, the call's value arriving as that block's first argument; or
a parallel loop, which names its index and bounds, a body block ending in
a yield, and the block after the loop. Loops are not lowered to branches,
so a directive that names a loop finds it. A call is in tail position
when its continuation block returns the call's value and does nothing
else. Where a directive's rewrite is given below, it is in the form the
compiler prints: `let @n = e in` binds a value, `f(a) -> R` is a call
continuing at block `R`, a call on a line of its own drops its value, and
`parfor i in [lo:hi:1] { .. }` is a parallel loop.

## 2. Loop transformations

A loop transformation changes the order in which the iterations of one
or more `parfor` loops run. The body of each iteration is unchanged.
Every transformation here is valid because the loops it touches are
`parfor` loops, whose iterations may run in any order (Allen and Kennedy;
Wolf and Lam).

**split(l, lo, li, n, guard).** Loop `l` becomes an outer loop `lo` that
strides through the range by `n` and an inner loop `li` over `[0, n)`; the
body sees `lo + li` where it saw `l`. With `guard` true the body is
guarded by `lo + li < end`, so a range that is not a multiple of `n` runs
exactly its own iterations; with `guard` false the range is asserted to
divide by `n`. `n` is a constant, or the name of a value that is the same
for the whole loop. `render.split(p, pb, pi, 256, true)`:

    par for each pb in 0..W*H step 256:
        par for each pi in 0..256:
            p = pb + pi
            if p < W*H:
                ray = camera(p); ...

**collapse(lo, li, l).** The inverse of split: two nested loops become
one loop over the rectangle they cover, each index recovered from the step
number. `lo` must run `li` and nothing else.

**reorder(li, lo).** Interchange of two nested `parfor` loops, named from
the innermost out as in Halide: afterwards `li` is inside `lo`. The nest
need not be perfect. What the outer body does before its inner loop (the
prologue) runs in a loop of its own before the nest; what it does after
(the epilogue) runs in a loop of its own after it; pure arithmetic on the
outer index is recomputed where it is used; any other value the inner loop
reads from the prologue is kept in an array with one entry per outer
iteration. Loop distribution followed by loop interchange. With a sample
loop added to the example,

    par for each p in 0..W*H:
        film[p] = 0                        // prologue
        par for each s in 0..spp:
            body(p, s)
        weight[p] = spp                    // epilogue

`render.reorder(p, s)` produces

    par for each p in 0..W*H: film[p] = 0            // p!prologue
    par for each s in 0..spp:
        par for each p in 0..W*H: body(p, s)
    par for each p in 0..W*H: weight[p] = spp        // p!epilogue

The prologue and epilogue loops are variants of `p`, so a later
`bind(p, ..)` covers them.

**persistent(l, lw, n).** `n` workers `lw` claim the iterations of `l`
from a shared counter until the range is used up; the persistent threads
of Aila and Laine (2009). The chunk is the stride, so to hand out several
iterations at a time, split first and make the outer loop persistent. `l`
must not be bound to hardware and must not be a queue's drain (Section
4); a later bind of `lw` says what runs the workers.

    counter = 0
    par for each w in 0..n:
        p = atomicAdd(counter, 1)
        while p < W*H:
            body(p)
            p = atomicAdd(counter, 1)

**vectorize(l).** The extent of `l` must be a constant; it is the gang
width. The iterations of `l` become the lanes of one SIMD gang and the
loop disappears. Each lane computes what its iteration would have. A
branch is run under a mask of the lanes that take it (partial
linearization, Moll and Hack 2018); a call under a mask is made if any
lane is on; a loop nested inside is run by each lane for itself; an
accumulate from several lanes is reduced across them first. To vectorize
a wide loop, split it to the gang width and vectorize the inner loop; the
split's guard becomes the gang's mask, so the last gang runs partly full.
`render.split(p, pb, pi, 8, true).vectorize(pi)`:

    par for each pb in 0..W*H step 8:
        vec for each pi in 0..8:               // one gang of eight lanes
            p = pb + pi
            mask = p < W*H
            ray = camera(p)                    // eight rays, one per lane
            ...

Whether `vectorize` is a loop transformation or a hardware binding is a
fair question: it reorders nothing, it maps iterations onto the lanes of
a SIMD unit. It sits here because its effect on the program is the
Halide one, a loop of constant extent replaced by one gang-wide
statement; Section 3 says how it combines with the bindings.

**unroll(l).** The extent of `l` must be a constant; the loop is replaced
by that many copies of its body one after another, the index a constant
in each (Halide's `unroll`). Nothing is reordered and nothing runs in
parallel that did not: the copies run in the loop's order. What it is
for is the loop a tree's node tests its children in, which lowering
writes as a loop over the children whatever their number so that a
`vectorize` of it is one test of all of them at once: unrolled, a binary
node's two tests are written out in a row, each child's box read off its
own row, which is how FCPW's and pbrt's traversals are written.
`closest.unroll(tris.Interior.children)`:

    for each children in 0..2:          // the children's tests
        test(children)
    ==>
    test(0)
    test(1)

**loopify(), loopify(n).** Recursion becomes iteration. A tail recursion
becomes a loop: every call of the function by itself in tail position is a
jump back to its entry. A recursion that goes through other functions
first (`step` calls `shade`, which calls `step`) is made direct by
inlining the calls on the way back, unless the schedule holds a function
as a function of its own (`[[noinline]]`, or a `defer` or `stage` written
on it). `step.loopify()`:

    par for each p in 0..W*H:
        ray = camera(p); rad = 0; depth = 0
        loop:
            hit = trace(ray)
            if not hit: rad += sky(ray); break
            rad += emitted(*hit)
            if depth + 1 >= max_depth: break
            ray = bounce(*hit, ray); depth += 1
        film[p] = rad

A recursion that calls itself more than once per step (a tree traversal:
`from (left, right)`) cannot become a loop that way, because the second
call has to happen after the first returns. `loopify(n)` puts the pending
calls on an explicit stack of `n` entries instead: the first call is made
at once and the rest are pushed, last in first out, so the visit order is
the program's. The recursion must be tail-modulo-recursion: after a
recursive call returns, nothing remains but more recursive calls and the
return. `n` is the schedule's promise about the depth.

**sort(c, lambda).** On a recursion's children, `trace.sort(tris.Interior,
|child| key)`: the calls of a `from (a, b)` are made in ascending order of
the key the lambda gives each child. The sort must come before `loopify`,
which replaces the run of calls by stack pushes and leaves nothing to
permute. In a gang the comparisons vote by the lanes' majority, so the
gang descends into one child order. On a call, `vol_route.sort(
vol_surface, |prim| key)`, the threads are reordered by the key just
before the call (OptiX's `optixReorder`); nothing an iteration computes
changes.

**specialize(x).** Halide's `specialize`, from a boolean condition to a
variant. `x` is a parameter of the function of variant type (an algebraic
type, or a bool). The function's outermost loops are copied once per
variant, the parameter's tag is a constant in each copy so that every
`match` on it folds to one arm, and the loops are entered through a
dispatch on the tag. `render.specialize(integrator)` for an integrator
with two variants:

    switch integrator.tag:
        Path:    par for each p in 0..W*H: body with integrator = Path(..)
        VolPath: par for each p in 0..W*H: body with integrator = VolPath(..)

Each copy computes what the original computes under its tag, so the
dispatch is exhaustive and no general copy is kept. Written on one copy,
`render[VolPath].specialize(sampler)`, it copies that variant's loops
alone. Written with a loop, `render.specialize(shadow, have_media)`, it
copies one loop wherever it sits and dispatches where the loop was.
Written with a field of one arm, `render[VolPath].specialize(VolPath::
have_subsurface)`, the condition is a bool the arm of a variant parameter
carries -- pbrt's `haveSubsurface`, kept in the VolPath integrator as pbrt
keeps it in its integrator -- and the loops copied are that arm's, the
copies an earlier `specialize` on the parameter made; the copies are
`False` and `True` as a bool's are, and in each the arm's value has the
field set to the copy's, so the arm's binding of it reads a constant.
Written on a gang's state, `occluded.specialize(count(lane) > 14)`, the
condition is the number of live lanes at each call of the gang's copies,
and the two copies `occluded[True]` and `occluded[False]` may be scheduled
differently from there (Embree's hybrid traversal). Because later
directives that name a loop find every copy of it, `specialize` goes
first among a function's directives, after other specializes only; the
loop form goes after the directives that made its loop.

### 2.1 The rewrites

Each directive above, as the rewrite of the SSA it is (Section 1 for the
notation). A loop is a `parfor` terminator naming its index, bounds, a
body block and a continuation block; a rewrite that makes a loop makes
that, and one that changes a loop's body edits the body block's region.

- **split(l, lo, li, n, guard).** The terminator of `l`'s block becomes a
  `parfor lo` over `start:end:n*stride` whose body is a new block holding
  `parfor li` over `0:n:stride`; the old body becomes `li`'s body, and
  where it read `l` it reads `lo + li`. With `guard`, the old body is
  entered through `if lo + li < end`, else it is entered directly. A
  run-time chunk is an expression over the function's values emitted in
  the loop's block (SSA/Rewrite.cpp, split).
- **collapse(lo, li, l).** The two terminators become one `parfor l` over
  `0:(trip(lo) * trip(li))`, and the old body's reads of `lo` and `li`
  become `start_o + (l / trip(li)) * stride_o` and `start_i + (l mod
  trip(li)) * stride_i`, under `if` tests where the ranges do not divide
  by their strides. `li`'s bounds must be the same on every step of `lo`
  (SSA/CollapseLoops.cpp).
- **reorder(li, lo).** The outer loop is distributed at the inner one:
  the instructions before `li` go to a prologue loop `lo!prologue` when
  they have effects, are recomputed in the new inner body when they are
  pure, and are stored into an array `<name>!expanded` of one entry per
  iteration of `lo` when they are reads; the instructions after `li` go to
  an epilogue loop `lo!epilogue`. Then the perfect nest left is
  interchanged: `li`'s terminator moves to the outer position with `lo`'s
  inside it (SSA/ReorderLoops.h).
- **persistent(l, lw, n).** A counter is allocated and set to `start`
  before the loop; `parfor lw in 0:n` is the new terminator, its body a
  `while` whose header claims `i = atomic_add(counter, stride)` and tests
  `i < end`, and whose body is the old body with `i` for `l`
  (SSA/Rewrite.cpp, persistent).
- **vectorize(l).** Every `parfor` nested in `l`'s body is made the
  sequential loop it also is (header, body, latch). Then each value of the
  region is widened to `lanes` of its type, the loop's index becoming the
  ramp `start + stride * <0..lanes-1>`; the divergent branches are
  linearized in Moll and Hack's manner, each block gaining a lane mask,
  each block argument becoming selects over the masks of the paths that
  defined it; a call from the gang goes to a copy of the callee widened
  the same way (SSA/Vectorize.cpp, SSA/Linearize.h). The loop's terminator
  is replaced by a jump: the body runs once.
- **loopify().** Each call of the function to itself in tail position
  becomes a jump to the function's entry block with the call's arguments,
  after the callees on a path back to the function are copied into it at
  their calls (inline_call) so that every call back is of the function
  itself. **loopify(n):** a run of calls (a `multicall`) becomes a loop
  over a stack of `n` entries: the first call's arguments become the next
  iteration's, the others are pushed, and an iteration with no call pops
  one or ends (SSA/QueueRecursion.h).
- **sort(c, lambda).** The run's calls are kept as one `multicall`
  terminator until here; its per-call argument lists are permuted into
  ascending order of their keys by a network of compare-and-select
  instructions, each comparison wrapped in a `vote` so that a gang orders
  them by its lanes' majority; a run whose calls are the lanes of a vector
  is sorted over the lanes that are hit, one peel at a time
  (SSA/SortRecursion.h). Before `loopify`, which replaces the run.
- **specialize(x).** For each of the function's outermost `parfor`s whose
  body refers to `x`, the body's region is copied once per variant; a
  selector block per copy holds the loop as it was over the copy; the
  loop's block reads `x`'s tag and ends in a dispatch on it to the
  selectors. In each copy the uses of `x` (and of a load of `x` hoisted
  before the loop) are renamed to a value defined at the body's head: `x`
  with its tag field a constant, so every match on it folds. **specialize(l,
  x)** does the same for the one loop named. **specialize(A::f)**, `f` a
  bool field of the arm `A` of a variant parameter, copies the arm's loops
  for `false` and `true`: the loop's block reads the field's byte out of
  the payload's words (Lower/WordStorage.h) and dispatches on it, and each
  copy's value is the given one with the arm's tag and that byte set to
  the copy's value, so every read of the field folds (the known-bits rule
  of SSA/Simplify.cpp). **specialize(count(l) > k)** copies a gang's
  region under a test of the count of live lanes (SSA/Specialize.cpp).

## 3. Hardware binding

A binding names the unit of the machine that runs each iteration of a
loop. It changes no computation; code generation turns a bound loop into a
launch, a thread, a kernel, or a program for a fixed-function unit.

**bind(l, R)** with `R` one of `CPUThread`, `GPUBlock`, `GPUThread`,
`OptixThread`. An iteration of `l` is one unit of `R`. The bindings nest as
the hardware does: a `GPUThread` loop sits directly inside a `GPUBlock`
loop and the two form one kernel launch; a gang (Section 2, `vectorize`)
sits inside a `CPUThread` loop, and under a GPU bind the warp is the gang;
a `CPUThread` loop and a GPU loop do not nest. These rules are checked
when the bind is applied.

    render.split(p, pb, pi, 256, true);
    render.bind(pb, GPUBlock);
    render.bind(pi, GPUThread);

    launch kernel over W*H/256 blocks of 256 threads:   // one launch
        p = block * 256 + thread
        if p < W*H:
            ray = camera(p); ...

A loop over a queue (Section 4) bound to the GPU is launched over the
queue's capacity, each thread testing its index against the count on the
device, so that the host neither reads the count back nor waits between
launches. That is how the launch is lowered, not a change of meaning.

**bind(l, OptixThread)** and **q.bind(RTCore)** together put ray tracing
on the RT cores: a query `q` declared over an `OptixTree` runs on the
hardware, and a loop bound to `OptixThread` whose body reaches the query
is that launch's ray-generation program. The part of the body after the
nearest-hit query becomes the launch's closest-hit program, run as each
ray's traversal completes (the continuation after the trace, cut as a
`stage` cuts a function; Section 4). This is pbrt's shape: the ray
generation program traces and handles the miss, the closest-hit program
handles the hit.

**bind(TextureUnit, lambda).** The one bind of a function rather than a
loop. `texture_filter.bind(TextureUnit, |t, st, dstdx, dstdy| ...)`: on
this hardware the function is computed by the texture unit as the lambda
says, the lambda taking the function's parameters and giving its value
through the unit's intrinsic. The program keeps the function as the exact
algorithm; the schedule asserts that the unit's answer is acceptable in
its place.

**prefetch(c, n).** In the loop that tests a node's children, bring the
storage each hit child refers to into cache, `n` lines of it:
`trace.prefetch(tris.Interior.children, 4)`. A hint; no change of
meaning. It is a hardware directive because what it moves is a cache
line, whose size is the machine's.

### 3.1 The rewrites

- **bind(l, R).** No change to the graph: the `parfor` terminator records
  `R` in its `binding`, after a check that `R` may nest inside the binding
  of every loop around it. Code generation makes the launch. For a loop
  over a queue bound to the GPU, the bind adds to the loop's block the
  launch's extent (the queue's capacity), the guard `index < count` read
  on the device, and the count's address, from the facts the deferral left
  on the terminator (SSA/Bind.cpp).
- **bind(l, OptixThread)**, with **q.bind(RTCore)** on the query. The
  query's traversal is lowered to a call of a trace function wrapping
  `optixTrace`; for the bound loop, the chain of tail calls from its body
  to that call is copied, the copy's call redirected to the trace
  function's hit-side variant, and the continuation after the trace runs
  in the copy as the launch's closest-hit program, while in the loop's own
  chain the continuation becomes `if hit { return } else { continuation }`
  (SSA/HitContinuations.h). The region before the trace must have no
  effect but loads and writes to its own locals.
- **bind(TextureUnit, lambda).** The function's body is replaced by the
  lambda's, whose intrinsic the GPU backend lowers to the unit's fetch.
- **prefetch(c, n).** In the loop that tests the arm's children, the body
  gains `if mask[i] { prefetch(children[i]) }`, the intrinsic lowered by
  the layout to the address and size of the row the reference names
  (Lower/Prefetches.h).

## 4. Continuation styles

Halide's call schedule decides, for each function a stage calls, where in
the caller's loops the callee is computed and how long its values are
stored. The analogue here is the call. By default a call is made where it
is written and its state lives on the stack. The directives of this
section change that: the work behind a call is done later, in a loop of
its own over all the calls that are waiting, and the state each waiting
call needs is kept in a queue. Each waiting call is a *continuation*: the
rest of the computation from that point, as data.

Write a function with a call in it as

    f(..) { P; r = g(a); K(r, L) }

`P` is what `f` does before the call, `a` the arguments, `K` the rest of
`f` from the call's return, and `L` the values of `f` defined before the
call and used in `K`. A directive that defers the call stores `a` and `L`
and runs `g` and `K` later; one that stages it runs `g` now and stores `r`
and `L` for `K` later. In the literature `K` as a function of `r` and `L`
is the CPS conversion of the one call site (Nielsen 2001; Kelsey 1995, for
why this is natural in SSA), and the stored data is the defunctionalized
continuation (Reynolds 1972). A wavefront path tracer is a program run
this way (Laine, Karras and Aila 2013), and the generated loops are meant
to read as pbrt's wavefront integrator does.

### 4.1 queue(l, n, M)

`q = f.queue(l)` declares a queue owned by function `f`, one instance per
iteration of `f`'s loop `l` (`root`: one per call of `f`). What an entry
is and which calls push one is said by the `defer` and `stage` directives
that name `q`. The storage is made at the start of the iteration and is
dead at its end; the compiler makes it once and shares it across
iterations that never overlap in time. A queue declared at a loop bound
to `CPUThread` is that thread's queue: its storage comes from the thread's
arena, taken as the iteration starts and given back as it ends, so that a
thread allocates nothing after its first iteration and a queue of a chunk
of pixels times the sample count costs no stack (runtime/bonsai_parallel.h,
"The arena").

The capacity is the number of entries an iteration of `l` can push. For a
linear deferral each entry's run pushes at most one successor, so the
capacity is the trip count of the loop that pushes the first entries; a
queue pushed by another queue's drain takes that drain's capacity. A
capacity `n` written in the schedule is checked against the inferred one
where both are constants and refused where the inferred one is not, since
an overflow would be silent. `M` is the memory the queue lives in
(`Heap`, `Stack`, `DeviceGlobal`, `DeviceShared`, `Managed`), or the
caller's memory handed in as buffer parameters (`ExternHost`,
`ExternDevice`, `ExternManaged`; the capacity must then be a constant and
`f` exported). Every reader and writer of the queue must run at a level of
the hardware that can reach `M`: host memory is never touched inside a
GPU-bound loop, device memory is never read entry by entry by host code,
shared memory belongs to one block.

A queue is stored as a struct of arrays: a count, and one array per scalar
of the entry, an entry's structs and short vectors taken apart so that a
gang reads each scalar as one vector load. Bools are packed eight to a
byte. A push claims a slot by advancing the count (atomically; a gang
advances it once by the number of lanes that push) and writes the entry's
scalars.

*The rewrite.* A `queue` declaration alone changes nothing; it is read by
the first deferral onto the queue, which makes the storage: the types
`Entry_q` and `Queue_q` (4.2), the count and one array per scalar of the
entry, allocated where the owner's loop `l` begins an iteration (or once
per call for `root`), sized by the capacity given or by the producer's
trip count, and marked for the memory `M` names, which the storage pass
and the backends honour (SSA/QueueStorage.h); `ExternDevice` makes them
parameters of the function that the caller fills.

### 4.2 defer(g, q)

`f.defer(g, q)` is written on `f` about its call to `g`. The call is
replaced by a push of the continuation's data onto `q`, and `q`'s owner
runs a loop, the *drain*, that makes the waiting calls. In the SSA that
push is one terminator, `defer q entry` (`Terminator::Defer`): the store
of the entry and the end of the activity in one, which every analysis
sees the block end at, lowered after the last directive has run into the
count's atomic add, the entry's stores and the return or yield
(`lower_pushes`; a gang variant's deferral becomes the predicated push and
a yield before it is linearized). A spawned call's push and the owner's
own first push stay instructions, since their blocks go on. The drain is a
`parfor` named after the queue, so later directives can split, vectorize
or bind it. It runs immediately after the loop that pushed the entries
and before anything the program ordered after that loop; where several
queues are pushed from one loop, their drains run in the order the queues'
directives were written.

The call `step(ray, 0, max_depth, rad)` in `render` and the recursive
call in `shade` are the same chain, and deferring both puts every path
step on the queue (`render.defer(step, rays)` says the first call is a
push too; test `semantics-defer`):

    rays = render.queue(root);
    render.defer(step, rays);
    shade.defer(step, rays);

    alloc rays[2]: W*H entries of (ray, depth, slot)
    alloc record.rad[W*H]
    par for each p in 0..W*H:                            // the producer loop
        ray = camera(p); record.rad[p] = 0
        push rays[0] <- (ray, 0, p)
    round = 0
    while rays[round & 1].count > 0:                     // a round per bounce
        cur = round & 1; nxt = cur ^ 1
        rays[nxt].count = 0
        par for each e in 0..rays[cur].count:            // the drain, `rays`
            (ray, depth, slot) = rays[cur][e]
            hit = trace(ray)
            if hit:
                record.rad[slot] += emitted(*hit)
                if depth + 1 < max_depth:
                    push rays[nxt] <- (bounce(*hit, ray), depth + 1, slot)
            else:
                record.rad[slot] += sky(ray)
        round += 1
    par for each p in 0..W*H:                            // the rest pass, `rays_rest`
        film[p] = record.rad[p]

Three things to read off this nest.

*The entry* holds the call's arguments that vary from one call to the
next: here the ray and the depth. Values that are the same for every
entry (the scene, `max_depth`, a camera) are not stored; the drain has
them in scope. A pure function of stored or constant values, of the
entry's slot, or of values the drain has in scope is recomputed at the
drain rather than stored -- and so is one that reaches the push as a
parameter of the pushing function, when every call the owner makes into
that function hands the same such function (the pixel from the sample's
slot, through every queue of a path). A value selected from a read-only
array by an index is stored as the index -- an array parameter without
`mut`, read again at the drain -- and a value loaded from the record at
the entry's slot in a round in which nothing writes it is loaded again
(4.5). An argument that is a pointer must point at a mutable local of the
producer's iteration (`rad`), and is stored as that local's slot in the
record. The analysis that decides this
is polynomial and conservative.

*The record* is one slot per producer iteration (pbrt's
`pixelSampleState`): the iteration's reducers and mutable locals that the
chain touches, the values the rest of the iteration needs after its call,
and the call's value when the rest uses it. An entry carries the slot
index and nothing else of the frame. The rest of the producer's iteration
after the call, `film[p] = rad`, runs in the *rest pass* once every
drain of the queue's family has run. So no drain runs a continuation of
the owner, and nothing returns up the chain.

*The record is stored as the entry is*, and so is every other array a
directive induces (a cut's set, 4.5; a loop interchange's expansion of a
value per iteration): a struct of arrays, one array per leaf of the stored
type -- a scalar, a short vector kept whole where no gang touches it (the
vector itself for two or four components of sixteen bytes or fewer; packed
for a 3-vector), a bool a byte of its own in a slot the chain writes field
by field, an ADT's padding nowhere -- and a leaf nothing reads has no
array and is stored nowhere, as pbrt's `SOA<PixelSampleState>` is an
array per member and its `visibleSurface` is untouched for an RGB film. A
`mut Record` local of the producer's iteration is therefore
`q_rec_<leaf>_slots` per leaf, the chain function that is handed it takes
a pointer per leaf in the parameter's place (`rec_p`, `rec_hits`, ...),
and a field store through the parameter is a store into that leaf's array.
Each such storage has a handle named for the stored thing -- `q.rec`,
`q.result`, `q.rest.<value>`, `s!expanded` -- which is what a later
directive takes to lay it out otherwise; nothing here is the layout
language's promise (SSA/InducedStorage.h).

*Rounds.* When running an entry can push onto the queue it came from, the
drain runs in rounds, each draining the entries the round before pushed,
until a round pushes nothing. There are two buffers when the drain itself
pushes the successors, as here (pbrt's `rayQueues[depth & 1]`), and one
when another queue's drain pushes them after this pass is over (4.3).
When no entry's run can push onto the queue, there is one pass.

A deferred call may sit in tail position, as `shade`'s does: `shade` does
nothing after the call but return. A call that is not in tail position,
`f = P; r = g(a); K(r, L)`, is deferred together with its continuation:
the entry holds `a` and `L`, and the drain runs `g` and then `K`. This is
pbrt's medium-sample queue, work whose routing is known before the call.
A spawned call, `spawn g(.., x);` in the program, says the call may run
beside the rest of its activity and reports through the reducers it is
handed, `x` among them (`docs/parallel-semantics.md`, section 1); the
entry carries `x` -- the address of the reducer, or of its record slot --
the drain makes the call, which adds into it, and the caller does not
wait until the `join;` the program wrote: the owner's iteration runs on in
place after its call into the chain, up to the join, is suspended there,
and what follows the join runs in the pass over the record after the
drain (`docs/parallel-semantics.md`, section 4). A return joins nothing,
so the join is the owner's to write, and the parser refuses a read of the
reducer before it. With no join before the iteration's end nothing is
suspended and the pass has nothing to run.

Conditions, each refused with an error: the loop that pushes is a
`parfor` (deferring across a `for` would reorder what the `for` promised);
every caller between `f` and the queue's owner does nothing after its
call into the chain but return (the complete continuation is captured; the
owner's own rest goes to the record); one call to `g` in `f`; each step
makes at most one deferred call (a branching recursion is not bounded by
the producer's count); the chain is not recursive apart from the deferred
call and its functions are not called from outside it.

The owner's call into the chain may sit under a test, `if c { f(a); } K`:
the deferral ends the iteration only when the call is made. An iteration
whose test fails runs `K` where it stands, at once, with its values in
scope; an iteration that called runs `K` in the pass over the record after
the drain, from what the record kept. `K` is therefore in the program
twice, once as written and once copied into the pass, and the record's
byte per slot (`<q>_ran`, set at the call) tells the pass which iterations
pushed (pbrt's camera kernel: a ray is pushed only when the lens gives one,
and the film is written for every sample). A deferred call under a test
*inside* the chain -- `f` returning without having called `g`, or a
recursion's step ending its path -- is told apart by the chain itself,
which then *reports*: every function of the chain returns a bool, or the
option of its value (`{ value, set }`, `set` false for "deferred"), the
deferral answering `true` or the empty option and every other return
`false` or `some(value)`, a tail call into the chain returning the
callee's answer onward. The owner's call becomes `callc f(..) q!deferred`,
which dispatches: "deferred" sets the slot's byte and ends the iteration,
"returned" runs `K` where it stands with the value out of the option. The
byte is then set at the dispatch rather than at the call. A chain whose
every path defers somewhere -- onto this queue or onto another directive's
(`QueueSpec::deferred_elsewhere`) -- does not report; nor does one whose
owner makes the initial push, after which every slot holds an entry.

*The rewrite* (SSA/Defer.cpp). Let `n` be the producer's trip count and
`A` the stored part of the call's arguments, by the rules above.

1. Two types: `Entry_q { A }`, and `Queue_q { count : u32; .. }` with one
   array per scalar of `A`. A queue is a struct of arrays, as pbrt's are.
2. Every function on the chain from the owner to `f` gains a parameter
   `_queue_q : mut Queue_q*`, handed down each call, and returns nothing:
   a return that ended the path stores the path's value through a second
   parameter `_result_q` when the owner's rest used it.
3. The deferred call, a terminator `g(a) -> R` with `R` returning, becomes
   `push _queue_q <- build<Entry_q>(A)` and `return`.
4. In the owner, before the producer loop: `q_queue : mut Queue_q[2]`
   built over arrays `q_<leaf>_0`, `q_<leaf>_1 : mut T[n]`, and a record
   array `q_<local>_<leaf>_slots : mut T[n]` per leaf of each record value
   (one array, `q_<local>_slots`, for a scalar). In the producer's
   body the record values are stored at `[i]`, the call is handed
   `&q_queue[0]`, `&q_result[i]` and `i` (or is itself the push, when the
   owner's directive made it one), and the blocks after the call are cut
   out.
5. After the producer loop, the nest above: `q!round = 0`, a `while` on
   `q_queue[q!round & 1].count`, the reset of the other buffer, `parfor q
   in [0:count:1]` whose body loads each leaf at `q` and calls `f` with
   them, the uniform arguments and `&q_queue[nxt]`, and `parfor q_rest in
   [0:n:1]` holding the cut-out blocks with `q_result[q_rest]` and the
   record in place of the locals. A queue its drain cannot push onto gets
   one buffer, no round loop, and `q_queue.count = 0` after its pass.
6. When the whole schedule has run, each `push` is lowered to `let s =
   atomic_add(&count, 1u)` and a store of each leaf at `s`.

For `shade.defer(step, rays)` the call site goes from

    step(next, @d, rad) -> R                 // R: return

to

    push _queue_rays <- build<Entry_rays>(next, @d, slot)
    return

and the drain's body is `step(ray, depth, max_depth, &rays_rad_slots[slot],
&rays_queue[nxt])` with `ray`, `depth`, `slot` loaded from the leaves at
index `rays`.

### 4.3 stage(g, q)

`f.stage(g, q)` cuts `f` immediately after the call rather than before
it. `g` runs where it is; the push follows the call at once, for every
entry, carrying the call's value whatever it is and the rest's live
values; the drain runs the rest, which is the whole of `f` from the
call's return. Nothing of `f` after the call stays in the first loop.
With the recursion deferred as above, `step.stage(trace, hits)` makes
the rays drain trace and push, and the hits drain test the hit and run
either arm (test `semantics-stage`, which also splits the queue as 4.4
does):

    alloc rays: W*H entries of (ray, depth, slot)         // one buffer: see below
    alloc hits: W*H entries of (hit, ray, depth, slot)
    ...
    while rays.count > 0:
        par for each e in 0..rays.count:                 // `rays`: trace
            (ray, depth, slot) = rays[e]
            hit = trace(ray)
            push hits <- (hit, ray, depth, slot)         // the cut: a miss is pushed too
        rays.count = 0                                   // read for the pass, then emptied
        par for each e in 0..hits.count:                 // `hits`: the rest of step
            (hit, ray, depth, slot) = hits[e]
            if hit:
                record.rad[slot] += emitted(*hit)
                if depth + 1 < max_depth:
                    push rays <- (bounce(*hit, ray), depth + 1, slot)
            else:
                record.rad[slot] += sky(ray)
        hits.count = 0

The ray queue has one buffer now: nothing pushes onto it while its pass
runs, since the pushes come from the hits drain afterwards. The hits
queue's capacity is the rays drain's, since each entry pushes exactly
once. The conditions are those of a deferral with a continuation (4.2);
`stage` is applied after the deferral of `f`'s own recursion, so the
chain through the rest has no other recursion. A deferral written after
the stage whose call lies in the rest finds it there: the schedule's name
for `f` stands for `f` and for the parts cut off from it.

pbrt's trace kernel routes by the result: a miss onto the escaped-ray
queue, a hit onto a material queue. That is this stage with the queue
split on the call's value, `hits.specialize(hit)` (4.4): the push after
the trace computes the tag and goes to `hits[None]` or `hits[Some]`, and
`hits[None]`'s drain is the miss arm alone, its entries holding only what
that arm reads (the ray's direction and the slot, in the golden). A push
that sits inside an arm comes from a different directive. Deferring the tail call of an
arm (4.2) puts the push where that call was; the renderer's schedule
routes that way today, the routing function's tail calls deferred onto
the escaped, emissive and hit queues, with no stage in it.

*The rewrite* (SSA/Stage.cpp).

1. Split `f` at the call `g(a) -> R`. The blocks from `R` on are copied
   into a new function `f!after`, whose parameters are `R`'s arguments
   (the call's value first) and then the values of `f` defined before the
   call and used in those blocks.
2. In `f`, the call continues at a new block `R!stage(r)` holding one tail
   call, `f!after(r, live..)`, whose continuation returns.
3. `f.defer(f!after, q)` as in 4.2: the entry is the stored part of `(r,
   live..)`, the push takes the tail call's place in `R!stage`, and the
   drain calls `f!after`.

For `step.stage(trace, hits)`:

    func step(ray, depth, max_depth, rad, _queue_rays, _slot_rays, _queue_hits, _slot_hits) -> void {
      trace(ray) -> R!stage
      R!stage(hit): push _queue_hits <- build<Entry_hits>(hit, ray, depth, _slot_hits); return
    }
    func step!after(hit, ray, depth, max_depth, rad, _queue_rays, _slot_rays) -> void {
      if hit .. }                            // R and what followed it, as they were

The deferral of a call that is not in tail position (4.2) is the same
split with a different step 2: the call and the rest become one function,
`f!g!k(a, live..) { g(a) -> R'; R'(r): f!after(r, live..) }`, the site
tail-calls it, and `defer` queues that call.

### 4.4 Q.specialize(x)

`q.specialize(x)` splits a queue by a value of variant type that the
drained function computes from an entry: one queue, one drain and one copy
of the drained function per variant, the tag a constant in the copy so
every `match` on it folds, the push computing the tag and going to the
variant's queue, the drains running in variant order. Each variant's
queue has the parent's capacity. `hits.specialize(h.material)` with two
materials:

    alloc hits[Diffuse], hits[Glass]: W*H entries each
    ...
            switch material(h).tag:
                Diffuse: push hits[Diffuse] <- (h, ray, depth, slot)
                Glass:   push hits[Glass]   <- (h, ray, depth, slot)
        par for each e in 0..hits[Diffuse].count: shade with material = Diffuse
        par for each e in 0..hits[Glass].count:   shade with material = Glass

A variant's queue may be split again (`hits[Basic].specialize(material)`).
A drain that no entry of the scene can reach is not launched, by facts the
driver supplies (`SSA/GateDrains.h`); that is a lowering.

*The rewrite* (inside defer(), SSA/Defer.cpp with SSA/Specialize.cpp). The
split is a tree of keys under the queue. For each leaf, a sub-queue: its
own count and arrays under the path's name (`hits!Some!Diffuse`), the
arrays of the leaves its drained copy never reads left out; a copy of the
drained function with each key on the path replaced by the value with
that variant's tag (specialize_function), so its matches fold; and a
drain loop named by the path, the loops run in variant order. The push
computes each key's tag from the values it is about to store, dispatching
on it block by block down the tree, and the leaf block pushes onto its
sub-queue. The entry type is one for the whole split.

### 4.5 stage(g, q, l) and defer(g, q, l): a cut inside a drain, on the same queue

Both forms above put the continuation on a queue of its own. pbrt's sample
kernel is a cut that keeps the queue: it reads the ray queue, draws the
bounce's random numbers and stores them in an array of its own, indexed
by the pixel; the trace kernel reads the same ray queue again, and the
material kernel reads the array. Nothing of the ray is copied. A `stage`
onto a new queue copies the ray's state into that queue; measured on
10-09, that copy made the staged sample kernel 2.2x slower than pbrt's.

The three-argument form is for this. Its arguments are

- `g`, the call in `f` at which `f` is cut, as in 4.2 and 4.3;
- `q`, a queue that exists already, made by `queue` and given a drain by
  deferrals, from whose drain `f` is reached by calls each in tail
  position. In 4.2 and 4.3 the second argument is a new queue that the
  continuation is pushed onto. Here it is the queue whose drain is
  already running `f`, and nothing is pushed;
- `l`, a fresh name for the new loop the cut makes.

Write the drain of `q` as `par for each e in 0..q.count: C(q[e])`, where
`C` is the chain of calls from the drain's body down to `f`, and `f = P;
r = g(a); K(r, L)` as before. `f.stage(g, q, l)` replaces the drain with
two loops over the same buffer and the same count, one after the other,
in the place the drain had:

    par for each e in 0..q.count:          // `l`: the first pass
        C down to f, then P, then r = g(a)
        record.S[slot] = S                 // new arrays of the record
    par for each e in 0..q.count:          // `q`: the second pass
        r and L from q[e] and record.S[slot]
        K(r, L)                            // f!after

`S` is what the second pass needs and cannot get otherwise: `r`, and each
value of `L` that is not a field of the entry, not the same for every
entry, and not a pure function of those (which the second pass recomputes,
the rule of 4.2). The record gains, for each value of `S`, one array per
scalar of it -- the entry's rule of 4.2, a struct of arrays: a short vector
one array (the vector itself for two or four components of sixteen bytes or
fewer, aligned as the vector is; packed, its components back to back, for a
3-vector), bools eight to a byte, an ADT's padding nowhere -- one
slot per producer iteration as a reducer's is (4.2). The first pass writes
them at the entry's slot, which the entry carries already (pbrt's
`pixelIndex`), and the second pass reads back the scalars it uses. No
entry grows and no push changes.
The first pass takes the name `l`. The second keeps the queue's name `q`,
so a directive on `q` written after this one addresses the pass that runs
`K`.

In the running example, let `step` draw the bounce's random numbers
before it traces, as pbrt does, and let `shade` use them:

    func shade(h : Hit, ray : Ray, u : Float, depth : i32, p : i32,
               max_depth : i32, rad : reduce(+) Float) -> void {
        rad += emitted(h);
        if depth + 1 >= max_depth {
            return;
        }
        step(bounce(h, ray, u), depth + 1, p, max_depth, rad);
        return;
    }
    func step(ray : Ray, depth : i32, p : i32, max_depth : i32,
              rad : reduce(+) Float) -> void {
        u = draw(p, depth);                // the bounce's random number
        hit = trace(ray);
        if hit {
            h = *hit;
            shade(h, ray, u, depth, p, max_depth, rad);
            return;
        }
        rad += sky(ray);
        return;
    }
    func render(max_depth : i32, film : mut array[Float, W*H]) -> void {
        parfor p in 0:W*H {
            ray = camera(p);
            rad : reduce(+) Float = 0.0;
            step(ray, 0, p, max_depth, rad);
            film[p] = rad;
        }
        return;
    }

With the recursion deferred as in 4.2, `rays` is the queue made by
`render.queue(root)`. Its entries are `(ray, depth, p)`, `p` being the
pixel and also the record's slot. Its drain is the loop named `rays`,
whose body is the deferred call `step(ray, depth, p, max_depth, &rad[p])`,
so `C` is that one call. The schedule

    rays = render.queue(root);
    render.defer(step, rays);
    shade.defer(step, rays);
    step.stage(draw, rays, samples);

cuts `step` after `draw`. `K` is the trace and the test of the hit with
both arms; `L` is `ray`, `depth` and `p`, all fields of the entry; `S` is
`u` alone. The nest of 4.2 becomes

    alloc rays[2]: W*H entries of (ray, depth, p)        // written by the pushes
    alloc record.rad[W*H]
    alloc record.u[W*H]                                  // written by `samples`
    par for each p in 0..W*H:                            // the producer loop
        ray = camera(p); record.rad[p] = 0
        push rays[0] <- (ray, 0, p)
    round = 0
    while rays[round & 1].count > 0:
        cur = round & 1; nxt = cur ^ 1
        rays[nxt].count = 0
        par for each e in 0..rays[cur].count:            // `samples`: the first pass
            (_, depth, p) = rays[cur][e]
            record.u[p] = draw(p, depth)
        par for each e in 0..rays[cur].count:            // `rays`: the second pass
            (ray, depth, p) = rays[cur][e]; u = record.u[p]
            hit = trace(ray)
            if hit:
                record.rad[p] += emitted(*hit)
                if depth + 1 < max_depth:
                    push rays[nxt] <- (bounce(*hit, ray, u), depth + 1, p)
            else:
                record.rad[p] += sky(ray)
        round += 1
    par for each p in 0..W*H:                            // `rays_rest`
        film[p] = record.rad[p]

`record.u` is one array: a pixel has one path in flight, whose draw the
first pass writes and the second reads within the round, and the next
round overwrites it. The count, the capacity, the two buffers, the rounds
and the rest pass are those of 4.2. pbrt's sample kernel is the `samples` loop and its
trace kernel the `rays` loop, and the schedule binds them as pbrt builds
them, a CUDA kernel and a ray-generation program:

    render.split(samples, samples_blk, samples_thr, 256, true);
    render.bind(samples_blk, GPUBlock);
    render.bind(samples_thr, GPUThread);
    render.bind(rays, OptixThread);

This is why the first pass needs a name of its own: a bind has to say
which of the two loops it means, and a generated name cannot be written
in a schedule. The optional third argument is the one addition to the
grammar; with it present, the second argument is read as the queue above
rather than as a queue to push onto.

`f.defer(g, q, l)` is the same construction with the cut before the call:
the first pass runs `P` and stores `g`'s varying arguments and `K`'s live
values, the second runs `g` and then `K`. The two name one cut from either
side. Where `g1` is followed at once by `g2`, `f.stage(g1, q, l)` and
`f.defer(g2, q, l)` produce the same loops: `step.defer(trace, rays,
samples)` gives the nest above, since `trace`'s argument `ray` is a field
of the entry and `u` is live after the call.

*The rewrite.* Steps 1 and 2 of 4.3 split `f` at the call and leave
`R!stage(r)` holding the tail call `f!after(r, live..)`. Then, in place of
a deferral onto a new queue:

3. Each argument of that tail call is followed back: one that is a
   parameter of `f`, and at every call of the chain a parameter of the
   caller, back to the drain's own call, is a value the drain hands the
   chain (an entry field, a uniform, a slot address, the queue), and the
   second pass hands it again. The others are `S`. For each, an array of
   the record per scalar of its type, `q_<name>_<scalar>_slots : mut
   T[n]` (`q_<name>_slots` for a scalar value), is made where the record's
   arrays are and sized as they are, the scalars those of 4.2's struct of
   arrays. `Entry_q` and `Queue_q` are unchanged, so every push onto `q`
   is.
4. Every function on the chain from the drain to `f` gains a parameter
   `_<name>_<scalar>_q : mut T*` per array, a pointer to this entry's slot
   of it, handed down as `_queue_q` is. In `R!stage` the tail call becomes
   a store of each scalar through its pointer, then `return`.
5. The drain's block and body are copied. The copy, its body as it was,
   is named `l` and placed where the drain was, the round entering it
   first; it takes the block's side effects with it, the reset of the
   other buffer's count, so that they come before both passes. Its call
   hands the chain `&q_<name>_<scalar>_slots[slot]` for each array, `slot`
   the one the entry carries (`_slot_q`). The drain follows the copy,
   keeping the name `q`, the bounds, and the capacity and count a GPU
   launch is built from (Section 3), with a new body: the entry's leaves
   loaded at `q` as before, each value of `S` put back together from its
   scalars loaded at `slot`, and the call `f!after(r, live..)` with them
   and the values handed again. That is the call the drain of a new queue
   would have made, with loads from the record in place of the new
   queue's; a scalar the rest never reads is a dead load once the rest is
   folded in, so each pass reads what it uses.
6. The rest, `f!after`, is folded into the second pass as a drained
   callee is folded into its drain (4.2).

For `f.defer(g, q, l)`, step 2 makes `f!g!k` as in 4.3's last paragraph
and the stored tail call is `f!g!k(a, live..)`. For `step.stage(draw,
rays, samples)` (golden `tests/bonsai/ssa/semantics-stage-same-queue`):

    func step(ray, depth, p, max_depth, rad, _queue_rays, _slot_rays, _draw_rays : mut f32*) -> void {
      draw(p, depth) -> R!stage
      R!stage(u): *_draw_rays = u; return
    }
    func step!after(u, ray, depth, p, max_depth, rad, _queue_rays, _slot_rays) -> void {
      trace(ray) -> R; R(hit): if hit .. }     // as it was

        rays_queue[nxt].count = 0
        parfor samples in [0u:@count:1u] {   // the copy: the drain as it was
          let ray, depth, slot = <the leaves at samples> in
          step(ray, depth, slot, max_depth, &rays_rad_slots[slot], &rays_queue[nxt], slot,
               &rays_draw_slots[slot])
        }
        parfor rays in [0u:@count:1u] {      // the drain, its body replaced
          let ray, depth, slot = <the leaves at rays> in
          let u = rays_draw_slots[slot] in
          step!after(u, ray, depth, slot, max_depth, &rays_rad_slots[slot], &rays_queue[nxt], slot)
        }

Why the result is the same. For each entry, `P`, `g` and `K` run once
each, in that order, as before. Across entries the only new ordering is
that every entry's first half runs before any entry's second half, which
a `parfor` admits: its iterations may be interleaved at any granularity.
`K` reads `r` and `L` as it did, from the record at this entry's slot,
which nothing writes between the two passes, and from the entry's own
fields. A push made in `P` is unchanged: the other buffer is reset before
the first pass, a push onto `q` goes to it in either pass, and any other
queue pushed is drained after the second pass, as it was drained after
the drain.

If some path through `C` and `P` returned before reaching the call, the
first pass would run that entry to its return, and one bit per slot, set
by the first pass, would tell the second pass to skip it. That bit is not
built: the compiler refuses such a cut today. The example's draw is
reached by every path.

Conditions, each refused with an error: `f` is reached from `q`'s drain
by calls each in tail position, and every function of the chain is called
from the chain alone, the owner's own call being a push (a call that ran
one of them whole would stop at the cut); one call to `g` in `f`, reached
by every path; the entries carry the record's slot; `l` names no existing
loop; the directive is written after the deferrals that gave `q` its
drain and before any directive on the loop `q`, since it splits that
loop; and `q` is not yet split by a `specialize`.

What the drains after it carry. With `step.defer(shade, hits)` added, as
the renderer has it, the push in the `Some` arm hands `shade` the draw
`u`, and a deferral stores a value it is handed (4.2). pbrt's material
work item carries the pixel index instead, and its material kernel reads
the samples from the array. The entry rule that does the same: a value
the pusher was handed that the drain it runs in loaded from an array of
the record at the entry's slot, where that array is one a cut's first
pass writes and this drain runs in the same round after the second pass,
is not stored. The entry carries the slot already, and the drain loads
`record.u[slot]` itself, a recipe as 4.2's recomputed values are (test
`semantics-reload-by-slot`: the hits entry is the hit, the ray, the depth
and the slot). The array holds still from the second pass to the round's
end because only the first pass writes it, and a drain placed in that
stretch is the only kind a deferral made there can have (4.2, where the
drain sits); a push onto the queue of the cut itself is for the next
round, whose first pass writes the array again, and reads nothing from it.

### 4.6 A queue on a specialized loop: one queue per arm

A `specialize(x)` written before a loop's queue copies the loop once per
variant of `x` and dispatches on `x`'s tag above the copies (section 2).
A queue declared on that loop is then one queue *per copy*, each owned by
its arm's loop and named with the arm's variant, and every directive on
the queue -- the deferrals, the cut, the splits -- is applied once per
arm:

    render[VolPath].specialize(film_visible_surface);
    rays = render[VolPath].queue(p_band, ExternDevice);
    li_vol_path.defer(vol_path_step, rays);

    switch film_visible_surface:
        False: par for each p in p_band!VolPath!False: ...; push rays!False
               drains of rays!False, rest pass of rays!False
        True:  par for each p in p_band!VolPath!True:  ...; push rays!True
               drains of rays!True, rest pass of rays!True

The dispatch is taken once, before anything runs, and only the arm taken
runs its loops, its drains and its rest pass: a specialized flag never
launches a kernel of the arm it did not choose. Each arm's queue, record
and rest are its own, so what one arm never reads of its record is stored
nowhere (4.2's storage rule): with an RGB film, the `False` arm's record
has no array for the visible surface and its camera kernel writes none,
which is pbrt's `if (initializeVisibleSurface)` -- a runtime branch in
pbrt -- resolved per arm. The price is code: the chain's functions exist
once per arm.

The chain is the arm's own. A function the specialize copied for the arm,
because the arm's key is handed to it (`integrator_li!True`, and whatever
it hands the key on to), is the arm's already. A function of the chain
the key never reaches -- a pusher such as `vol_exit`, which pushes onto
the queue and so takes the queue, whose type is the arm's -- is copied
for the arm by the deferral and the arm's calls retargeted
(`vol_exit!True`). A function off the chain, one that neither reaches the
callee by calls nor pushes onto the queue, stays one function. The copies
carry the arm's variant in their names as every copy does, so a later
directive naming the function finds each arm's (`vol_path_step.stage(...)`
cuts `vol_path_step!True` and `vol_path_step!False`), and the loops made
beside the drain take the variant last (`rays_rest!True`), so that
`render.bind(rays_rest, GPUThread)` binds every arm's and `bind(rays,
...)` does not bind the rest passes. A queue on `root` of a specialized
owner has an arm per variant of the owner's outermost loops, each arm's
region beginning at the loop copy the dispatch enters. A queue on a loop
with one copy, or on an owner nobody specialized, is one queue as before.

An arm that never calls into the chain has no queue of it: with
`render.specialize(integrator)`, a queue of `vol_path_step` exists in the
VolPath arm alone, which is why such a queue is declared on the arm,
`render[VolPath].queue(p_band)`.

A split of the queue (`hits.specialize(material)`, section 4.4) is a
split of each arm's queue, `hits!False` and `hits!True` each with the
split's leaves under it and the two sharing the mask of which variants
the scene has; the launch gates (SSA/GateDrains.h) are read per arm's
queue, so a queue fed by an arm's leaf drains is gated on their bits as
before. For a queue on `root`, an arm's drains sit beside the arm's loop,
under the arm's selector, and the arm's chain is reached through them as
through the loop.

A deferral's callee is one function per arm. The schedule's name for it
stands for the function as the arm calls it: the function, the one copy
a queue's specialize upstream made of it for the drain that calls it
(`finish!Big`, when `step` hands `finish` the key `hits` was specialized
by and only the Big drain calls it), and the continuations a stage split
off that copy. Two copies of the callee called side by side in one arm
(`finish!Big` and `finish!Small`, when both drains call it) are refused,
since a queue's drain calls one function: the argument has to be the same
for every copy, which is why pbrt's BSSRDF kernel takes the basic texture
evaluator for every entry and `vol_subsurface` takes none.

Reading a slot of a record nothing wrote is undefined, as reading
unwritten storage is in C++. The rule above never produces such a read:
a leaf is dropped only when no load of it exists anywhere in the program,
and a leaf that is loaded is stored wherever the program stores it.

## 5. Other

**skip(c*).** Not a transform: a policy for a gang's branches. A
vectorized branch is linearized, every arm computed under a mask; an arm
no lane takes is still paid for unless an any-lane test precedes it. The
linearizer puts such a test before an arm that touches memory or makes a
call; `skip` names the arms of pure arithmetic that get one too.

**layout.** `layout Shape = tagged_index;` and the tree layouts a
schedule block may carry belong to the layout language: they choose how a
value is represented and change nothing a program observes.

**Order.** Directives apply in the order they are written, across
functions as within one, and the order is part of the meaning: the drains
of queues pushed from one loop run in the order the queues were deferred;
`loopify` then `vectorize` is a loop each lane walks at its own pace,
`vectorize` then `loopify` is a traversal the gang makes together on one
stack of nodes and one of masks; `stage` comes after the deferral of the
function's own recursion; the whole-function `specialize` comes first
among a function's directives and the loop form after the directives that
made its loop. A function's name in a schedule stands for the function,
the parts a `stage` or `defer` cut off from it, and the copies a
`specialize` made of it; a loop's name stands for every copy of the loop.
A queue's declaration is not ordered: a `defer` may name it wherever the
block declared it.
