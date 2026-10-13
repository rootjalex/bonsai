# Parallel Semantics: parfor, spawn, reducers and atomic

A Bonsai program says where it may run in parallel: a `parfor` loop says
its iterations may, a `spawn` says a call may run apart from the code
that made it. A schedule says where it does. This document defines what
the program promises when it says so, what a schedule may make of the
promise, and what the compiler checks and takes off on the strength of
it. The scheduling directives themselves are defined by the loop nests
they produce in `docs/schedule-semantics.md`; this document is about the
one thing that document states as a condition and does not define: the
independence a `parfor` claims (its section 1) and the freedom a `spawn`
grants (its section 4.2), and the `atomic` accumulates that make both
true.

The short form. `parfor` and `spawn` both express *possible*
parallelism. A program is written so that it has a meaning however much
of that parallelism a schedule takes up, which is what makes a schedule
free to take up any of it without changing the result. An accumulate into
a location that another concurrently running piece of the program may
write is written `atomic`, or the program has a race and no meaning. The
compiler checks the promise where it can decide it, and takes the
`atomic` off wherever the schedule chosen leaves nothing running beside
the accumulate.

## 1. Activities

An *activity* is one sequential thread of control: a sequence of
statements run in order. Running a program is running a set of
activities. `main` runs in one. A call runs in its caller's activity: a
called function is not a fork. A `for` loop runs in its activity.

**parfor.** `parfor i in lo:hi { B }` runs `B` once per `i` in an
activity of its own, one per iteration, and the activity that reached
the loop waits at the loop's end for all of them. This is fork-join
parallelism, `finish { for (i in lo..hi) async B }` in the notation of
X10 (Lee and Palsberg, "Featherweight X10", PPoPP 2010, whose calculus
is the one this section follows).

**spawn.** `spawn x op= g(a)`, where `x` is a reducer declared
`reduce(op)` (the parser refuses anything else), evaluates the arguments
`a` in the spawning activity and makes a *task*: an activity that runs
`g(a)` and then accumulates the value into `x` with `op`. The task runs
at a point the schedule chooses, under two constraints.

- It runs no earlier than the spawn and no later than the *join* of `x`
  (below).
- It runs as a whole with respect to its spawner: the spawner's own
  statements are either all before it or all after it. A task is a
  deferred call, not a fork of the spawner's continuation; the two run
  one after the other, in an order the schedule picks. This is the
  weaker of the two readings of `spawn` in the literature -- Cilk's
  `spawn` (Frigo, Leiserson and Randall, PLDI 1998) runs the continuation
  beside the spawned call, which Bonsai does not promise and no schedule
  produces.

Apart from its spawner a task is unordered: it may run beside every
other task spawned while it is outstanding, and beside every activity
its spawner may run beside. Two tasks spawned by one activity, or by two
iterations of one `parfor`, or one by an activity and one by a task of
it, may all run at once.

**The join of a reducer.** A reducer is accumulated into wherever it is
in scope, and read as a value only in the scope that declared it (the
parser refuses a read of a reducer parameter; `tests/bonsai/error/
reducer-read`). A read of its value, and the end of its scope, are its
joins: every task feeding it has completed by then. Nothing else waits
for a task. In X10 terms the declaring scope's read is a `finish` over
the tasks that name the reducer; `docs/schedule-semantics.md` section
4.2 shows the drain and the pass over the record that implement it.

A function that spawns onto a reducer it was handed therefore returns
before the task runs, and the task's value reaches the reducer before
the function's caller -- or whoever declared the reducer -- reads it.
That is the meaning the existing tests check: `tests/bonsai/correctness/
llvm/defer-spawn` reads `total` in `main` after `run` has returned;
`defer-spawn-join` reads the iteration's `total` after its call.

## 2. Memory, happens-before and races

Memory is a set of *locations*: a scalar variable, an element of an
array, a field. An activity's statements perform *events* on locations:
a read, a write, an accumulate (`x op= e`: a read, the operation, and a
write of the result, three steps another activity may run between), and
an *atomic accumulate* (`atomic x op= e`: the same three as one
indivisible step). A reducer's location is written only by accumulates
and by its initialisation.

*Happens-before* is the smallest transitive order containing: program
order within an activity; the fork edges, from the point before a
`parfor` to the start of each iteration and from a spawn to the start of
its task; and the join edges, from the end of each iteration to the
point after its `parfor`, and from the end of a task to each join of the
reducer it feeds. Two events of different activities with neither
ordered before the other *may happen in parallel*.

Two events *conflict* when they touch the same location and at least one
of them writes. A *race* is a pair of conflicting events that may happen
in parallel, unless both are atomic accumulates with the same operation
on the same location -- those commute, and an accumulate order a program
cannot observe is what a reducer declares (`docs/schedule-semantics.md`
section 1: a `parfor`'s accumulates may be reassociated, Halide's
guarantee for reductions). A read beside an atomic accumulate is a race:
the value it sees depends on which contributions are in.

**A program with a race in any of its executions has no meaning.** This
is the rule of the C++ memory model (Boehm and Adve, PLDI 2008, and
[intro.races]), chosen for the same reason: a program without a race
behaves as if its activities were interleaved one statement at a time,
whatever the hardware does (the data-race-free guarantee, Adve and Hill,
ISCA 1990), so the schedule and the backends are free to run it on any
machine as long as they introduce no race of their own.

## 3. The program's promise

> **Rule.** For every two events that may happen in parallel and
> conflict, both are atomic accumulates with the reducer's operation on
> the reducer's location.

Equivalently, for the writes a program makes: an accumulate into a
location that another activity may write or read at the same time is
written `atomic`, and nothing reads a reducer while an activity that
accumulates into it may be running. The *may* is the one of section 1:
every `parfor` iteration and every task is taken to run at once with
everything it is unordered with. This is independent of any schedule, so
the promise is checkable on the program alone, and a program that keeps
it has the same meaning under every schedule.

The cases, with the tests that pin each:

| the accumulate | shared? | must be |
|---|---|---|
| `total += v` in a `parfor` body, `total` declared outside the loop | yes: every iteration writes the one word | `atomic` (`tests/bonsai/correctness/gpu/reduce-param`; `pending/race-parfor-reducer` is the plain form) |
| the same, in a function the body calls, `total` a `reduce` parameter | yes: a call is in its caller's iteration | `atomic` (`ssa/demote-atomics-callee`) |
| a spawned function writing a `reduce` parameter handed from outside the loop | yes: the tasks write the one word, and may run at once | `atomic` (`ssa/demote-atomics-spawn-*`, `correctness/gpu/demote-atomics-spawn`; `pending/race-spawned-function` is the plain form) |
| `rad += e` into a reducer declared inside the iteration, by the iteration's own calls, no task | no: one word per iteration, one activity | plain (the renderer; `correctness/llvm/semantics-render`) |
| the spawn's own `x op= v`, performed by the task | shared whenever `x` is: declared outside the loop, or fed by two tasks of one iteration (two spawns on a path, or one spawned function spawning onward) | `atomic` -- with no spelling for it today (section 6; `pending/race-spawn-accumulate`) |
| `total += x` in the spawner after `spawn total += g(..)` | no: a task never interleaves its spawner (section 1) | plain is right beside the task; `atomic` iff the loop shares `total` |
| a read of `total` inside a `parfor` that accumulates into it | a read beside writes | refused (`pending/race-reducer-read`) |

What the promise does not cover is unchanged: a `parfor`'s plain stores
to distinct locations (`film[p] = rad`) are the independence the loop
claims, and the compiler takes the claim as given, as `docs/schedule-
semantics.md` section 1 says. The rule above is about reducers -- one
word per declaration, written by accumulates alone -- where sharing is
decidable: a reducer's address reaches an iteration or a task either
from outside the fork, and is then one word for all of them, or from an
allocation inside it, and is then that activity's own.

## 4. What a schedule may do

A schedule chooses, for each fork, how it is run.

- A `parfor` runs its iterations one at a time in index order (no
  directive), as threads (`bind(l, CPUThread)`, `bind(l, GPUThread)`),
  or as the lanes of one gang (`vectorize(l)`), whose accumulates to one
  location are combined across the lanes before the store, so a gang is
  one activity for the purposes of section 2.
- A task runs where it was spawned, in the spawner's activity (no
  directive: "run as written, it is the accumulate it looks like"), or in
  a *drain* (`f.defer(g, q)`): a `parfor` over the queue's entries that
  runs after the loop that pushed them and before the pass that runs the
  reducers' joins, itself run one entry at a time or bound. A drain's
  iteration is the task: the call, and the accumulate into the reducer's
  slot in the record (or into the reducer, when it is one word for all).

Every one of these is an execution section 1 allows: the iterations'
interleaving is one the `parfor` admits, and a task's placement is inside
its window and does not interleave its spawner. So **a schedule never
changes the result of a program that keeps the promise**: the program
has no race under the maximal interleaving, the schedule's executions
are among those interleavings, and by the data-race-free guarantee each
of them computes what some statement-by-statement interleaving does,
every one of which the reducer's reassociation freedom makes equal.

What a schedule does change is which pairs of events *actually* happen in
parallel: a subset of the pairs of section 3. Two iterations of a loop
nothing bound never do; two tasks of a drain nothing bound never do; a
task run inline is in its spawner's activity and runs beside exactly
what the spawner does. That subset is what the demotion reads.

## 5. What the compiler checks and takes off

Both are one analysis asked two questions. `SSA/Contention.h` answers, for
the address an accumulate writes and a set of forks taken as parallel,
whether two activities running at once can write it: `Disjoint` (no),
`Shared` (yes), or `Unknown` (not shown either way). An address is
Disjoint when it is storage the activity made for itself (an allocation
inside the loop's body or inside the called function), or an element of
an array at a subscript that is one-to-one in the iteration's index; it
is Shared when it is one word reaching every activity from outside the
fork -- a reducer parameter, a reducer declared outside the loop; and
otherwise Unknown. The two questions differ in which forks are taken as
parallel.

**The check (to be built).** Take every `parfor` and every spawn as
parallel -- the `may` of section 3. An accumulate that is not `atomic`
and whose address is Shared is a race: refuse the program, naming the
accumulate, the reducer and the fork that shares it, as the `.expect`
files under `tests/bonsai/pending/race-*` propose. A read of a reducer
inside a fork that accumulates into it is refused the same way. Unknown
on a reducer's location does not arise, since a reducer's address is a
declaration's or a parameter's; Unknown on a `mut` array element is the
program's independence claim and is left to it. The check is a
diagnostic: it changes no program it accepts.

**The demotion (built: `SSA/DemoteAtomics.h`).** Take as parallel only
the forks the schedule made so: the loops bound to threads
(`parallel_loops_by_block`), and every function such a loop calls,
directly or through other calls (`functions_called_in_parallel`), whose
each call is one iteration's. A drain is a `parfor`, so a bound drain is
such a loop and the spawned function is a function it calls; an unbound
drain is sequential and a spawned function it alone calls is called in
parallel by nothing. An `atomic` accumulate whose address is Disjoint
under those forks is made plain. Unknown keeps it: being wrong here is a
race, being cautious is only slow. The demotion is an optimisation: an
indivisible read-modify-write beside which no other activity touches the
location computes the same value as a divisible one, so no execution's
result changes.

The two together are the user's two statements about spawn. *A spawned
function that writes a reducer must write it atomic*: the check. *The
atomic comes off when the spawn is not parallelised*: the demotion, which
asks what runs the spawned call -- the drain, bound or not; the spawner's
own loop, when the task is inline -- and not what spawned it.

The tests, one program under five schedules (`wet`, spawned once per
iteration of `run`'s loop, writes the run's `total`; `step` writes it
too; the spawn's own value goes into the iteration's `n`):

| schedule | `step`'s atomic | `wet`'s atomic | test |
|---|---|---|---|
| none: the loop sequential, the task inline | off | off | `ssa/demote-atomics-spawn-inline`, `correctness/llvm/demote-atomics-spawn-inline` |
| `step.defer(wet, q)`, nothing bound | off | off | `ssa/demote-atomics-spawn-drain`, `backends/llvm/demote-atomics-spawn-drain`, `correctness/llvm/demote-atomics-spawn-drain` |
| the same, `run.bind(i, CPUThread)`: the loop parallel, the drain sequential | on | off | `ssa/demote-atomics-spawn-bound`, `backends/llvm/demote-atomics-spawn-bound` |
| `run.bind(i, CPUThread)` alone: the task inline in a parallel loop | on | on | (the SSA, by hand: both kept) |
| the loop, the drain and the pass bound to GPU threads | on | on | `backends/ptx/demote-atomics-spawn`, `correctness/gpu/demote-atomics-spawn` |

Every schedule prints the same: 97.5, and eight ones. `ssa/demote-
atomics-callee` and `correctness/gpu/reduce-param` are the same facts
for a plain call under a bound loop.

## 6. Open

1. **The spelling for the spawn's own accumulate.** The task's `x op= v`
   is an accumulate like any other and section 3's rule applies to it;
   when `x` is shared it must be atomic, and the parser gives a spawn's
   accumulate no `atomic` and the drain copies the flag it is given
   (`src/Parser/Parser.cpp`, the `SPAWN` branch; `SSA/Defer.cpp`). The
   spelling proposed is `spawn atomic x op= g(..)`: `spawn` qualifies
   the statement `atomic x op= g(..)`, the `atomic` next to the
   accumulate it is about, as in the plain form. The alternative is to
   make the spawn's accumulate the compiler's -- the drain emits an
   atomic wherever the demotion's analysis cannot show the slot Disjoint
   -- which keeps the program unchanged and gives the one accumulate in
   the language that the compiler locks silently. The first is this
   document's reading of "atomic is the program's statement".
2. **What that costs the renderer, and what takes the cost back.**
   `apps/pbrt/render.bonsai` spawns four contributions into the sample's
   `radiance`, a reducer declared inside the render loop's iteration and
   fed by several tasks of one path (`shadow_contribution`,
   `emissive_contribution`, `escaped_contribution`,
   `exit_shadow_contribution`); under the rule they are shared and the
   four spawns would carry `atomic`. Every schedule in use sequences
   them -- different queues drain in order, a path's shadow rays of
   different bounces are different rounds -- so the demotion should take
   all four off, and today it cannot: a drain's accumulate indexes the
   record by a slot loaded from the entry, which the analysis cannot show
   one-to-one. The deferral knows it: a queue pushed once per entry of
   one producer per pass has distinct slots in a pass. Recording that
   fact on the drain for `contention_of` to read is what makes the rule
   cost the renderer nothing; without it the four spawns are device
   atomics on four floats each. The check should land with it.
3. **The spawned entry's reducer address on the GPU.** When the spawn's
   reducer is one word for all (`run`'s parameter), the entry stores its
   address, and on the JIT runtime's GPU path that is the producer
   launch's device copy, freed before the drain's kernel runs
   (`runtime/bonsai_cuda.h`, "Memory"): `pending/race-spawn-accumulate`
   prints 65 for 97.5 with the binds, every spawned half lost, with or
   without the atomic. Separate from the spelling, and needed before a
   shared-reducer spawn can run bound at all.
4. **What the check would refuse today.** Fourteen test programs write
   plainly into a reducer shared across a `parfor` (every one a deferral
   test: `defer-spawn`, `defer-spawn-two-producers`, `defer-two-drains`,
   `defer-primary-pusher`, `defer-mutual-producers`, `gate-drains`,
   `gate-drains-nested`, `gate-drains-under-specialize`,
   `defer-single-buffer`, `error/defer-drained-before-pushed`, in their
   `ssa`, `backends/llvm` and `correctness/llvm` copies), and eighteen
   spawn onto a shared reducer with the spawn's own accumulate, which
   needs the spelling. None of the three applications does either: the
   renderer's film and wosx's walk sums are already `atomic`. The tests
   are right as programs run sequentially and would gain `atomic` where
   the rule asks, the demotion taking it off again under their
   schedules, so no golden output changes.
