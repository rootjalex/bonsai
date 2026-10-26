# Parallel Semantics: parfor, spawn, join, reducers and atomic

A Bonsai program says where it may run in parallel: a `parfor` loop says
its iterations may, a `spawn` says a call may run beside the rest of the
function that made it. A schedule says where it does. This document
defines what the program promises when it says so, what a schedule may
make of the promise, and what the compiler checks and takes off on the
strength of it. The scheduling directives themselves are defined by the
loop nests they produce in `docs/schedule-semantics.md`; this document is
about the one thing that document states as a condition and does not
define: the independence a `parfor` claims (its section 1) and the freedom
a `spawn` grants (its section 4.2), and the reducers that make both true.

The short form. `parfor` and `spawn` both express *possible* parallelism,
and `spawn` is a `parfor` of two iterations. A program is written so that
it has a meaning however much of that parallelism a schedule takes up,
which is what makes a schedule free to take up any of it without changing
the result. What several activities add into at once is a *reducer*, whose
every accumulate is atomic by its declaration and which is read only once
everything adding into it has joined; what the program otherwise knows two
iterations may share is written `atomic`. The compiler refuses a read that
comes too early, and takes the atomics off wherever the schedule chosen
leaves nothing running beside the accumulate.

## 1. Activities

An *activity* is one sequential thread of control: a sequence of
statements run in order. Running a program is running a set of
activities. `main` runs in one. A call runs in its caller's activity: a
called function is not a fork. A `for` loop runs in its activity.

**parfor.** `parfor i in lo:hi { B }` runs `B` once per `i` in an
activity of its own, one per iteration, and the activity that reached
the loop waits at the loop's end for all of them. This is fork-join
parallelism, `finish { for (i in lo..hi) async B }` in the notation of
X10 (Lee and Palsberg, "Featherweight X10", PPoPP 2010).

**spawn and join.** `spawn f(a);` evaluates the arguments `a` in the
spawning activity and forks: the call `f(a)` runs in an activity of its
own, and the spawning activity goes on with the statements after the
spawn, its *continuation*, which may run at the same time as the call.
`join;` waits for every call the activity has spawned since its last
join -- through the functions it called as well, since a call runs in
its caller's activity. A function's return joins nothing: a spawn the
callee made and did not join is outstanding in the caller after the
return, and the caller writes the `join;` before it reads what the spawn
adds into (the parser refuses the read before one, and knows which
calls leave a spawn outstanding, `src/Parser/Parser.cpp`
`Outstanding`). The end of a `parfor` body joins the calls spawned in
it, since an iteration is an activity whose end is its own join, and so
does the end of the program. This is Cilk's `spawn` and `sync` (Frigo,
Leiserson and Randall, "The Implementation of the Cilk-5 Multithreaded
Language", PLDI 1998; Blumofe et al., "Cilk: An Efficient Multithreaded
Runtime System", PPoPP 1995) without Cilk's rule that a procedure cannot
return until its children have; the nearer model is X10's `async` and
`finish` (Lee and Palsberg, PPoPP 2010), where an `async` is bounded by
the enclosing `finish` and not by the procedure that made it -- here a
`parfor` body is a `finish`, and a `join;` closes one early. What this
buys is that the program alone says where every join is, and a schedule
only places the calls (section 4).

A spawn is a `parfor` of two iterations, and that is its definition:

    spawn f(a); K; join; J        is        parfor t in 0:2 {
                                                if t == 0 { f(a); } else { K; }
                                            }
                                            J;

where `K` is the continuation up to the join -- through the return of
the function that spawned, into its caller, when the join is written
there -- and `J` what follows it.
Everything this document says of two iterations of a `parfor` therefore
holds of a spawned call and its continuation, and of two calls spawned
before one join. The SSA keeps the spawn as a terminator of its own
(`Terminator::Spawn`, `Terminator::Join` in `include/SSA/SSA.h`) rather
than lowering it to that loop, so that the structure -- which call, which
continuation, where the join is -- stays visible to the schedule and to
every analysis; the parser lowers nothing either (`ir::CallStmt::spawned`,
`ir::Join`).

A spawned function returns nothing: nothing waits for its value. What it
does reaches the spawner through its arguments: the reducers it is handed
(section 3), which it adds into, and nothing else of the spawner's, since
a `mut` argument written by the call and read by the continuation would be
the race of section 2 (the deferral refuses one, `SSA/Defer.cpp`).

## 2. Memory, happens-before and races

Memory is a set of *locations*: a scalar variable, an element of an
array, a field. An activity's statements perform *events* on locations:
a read, a write, an accumulate (`x op= e`: a read, the operation, and a
write of the result, three steps another activity may run between), and
an *atomic accumulate* (the same three as one indivisible step). A
reducer's location is written only by accumulates and by its
initialisation, and every accumulate into it is atomic.

*Happens-before* is the smallest transitive order containing: program
order within an activity; the fork edges, from the point before a
`parfor` to the start of each iteration, and from a spawn to the start
of the spawned call and to the start of the continuation; and the join
edges, from the end of each iteration to the point after its `parfor`,
and from the end of a spawned call and the end of the continuation to
the join. Two events of different activities with neither ordered
before the other *may happen in parallel*.

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

## 3. Reducers, and the program's promise

A *reducer* is a variable or parameter declared `reduce(op) T`. It is the
location several activities add into at once, and the language makes it
safe by construction:

- It is only accumulated into, with `op`, never assigned; the parser
  refuses an assignment (`tests/bonsai/error/reducer-assign`) and any
  other operation.
- Every accumulate into it is atomic, by the declaration. Writing
  `atomic` on one is refused as a second spelling of one thing
  (`error/atomic-reducer`). Whether the atomic costs anything is the
  schedule's, and the compiler's to take off (section 5).
- It is read as a value only in the scope that declared it, and only
  once everything that may be adding into it has joined: not in a
  function it was handed to (`error/reducer-read`), not inside a `parfor`
  it was declared outside of (`error/reducer-read-parfor`), and not while
  a spawned call that was handed it is outstanding, before a `join;` or
  the return (`error/reducer-read-spawned`). Each refused read is a read
  beside writes under section 2. The parser decides all three
  syntactically: the declaration's `parfor` depth against the read's, and
  the set of reducers handed to spawned calls since the last join
  (`src/Parser/Parser.cpp`, refuse_reducer_read).
- It is handed only to a reducer parameter of a called or spawned
  function (`error/reducer-argument`, `error/reducer-mut-argument`), so
  the rules above hold of every function that can reach it.

> **Rule.** For every two events that may happen in parallel and
> conflict, both are atomic accumulates with the same operation on the
> same location.

The *may* is the one of section 1: every `parfor` iteration, and both
arms of every spawn, are taken to run at once with everything they are
unordered with. The rule is independent of any schedule, so a program
that keeps it has the same meaning under every schedule. For reducers the
language keeps it on the program's behalf: the accumulates are atomic and
the reads come after the joins. What is left to the program is its `mut`
storage. Distinct iterations writing distinct locations -- `film[p] =
rad`, with `p` the iteration's -- is the independence a `parfor` claims,
and the compiler takes the claim as given (`docs/schedule-semantics.md`
section 1); where the program knows two iterations may write one `mut`
location, it writes `atomic x op= v` (the renderer's film, wosx's walk
sums), and that too is a statement the compiler takes as given and makes
cheap where it can.

## 4. What a schedule may do

A schedule chooses, for each fork, how it is run.

- A `parfor` runs its iterations one at a time in index order (no
  directive), as threads (`bind(l, CPUThread)`, `bind(l, GPUThread)`),
  or as the lanes of one gang (`vectorize(l)`), whose accumulates to one
  location are combined across the lanes before the store, so a gang is
  one activity for the purposes of section 2.
- A spawn runs as written -- the call, then the continuation, in the
  spawning activity (no directive) -- or its call goes onto a queue
  (`f.defer(g, q)`) drained by a `parfor` over the entries after the loop
  that pushed them, itself run one entry at a time or bound, with the
  continuation run in place up to the join the program wrote. There the
  activity waits, which under a deferral means it is suspended -- its
  iteration ends -- and resumes after the drain, in the pass over the
  record (`docs/schedule-semantics.md` section 4.2), where what follows
  the join runs. The join stays where the program put it: a schedule
  never moves one, it only decides that the wait is a suspension across
  the drain rather than a wait on a running task. A spawn with no join
  before its iteration's end leaves nothing to resume.

Every one of these is an execution section 1 allows: the iterations'
interleaving is one the `parfor` admits, a spawned call run inline or in a
drain is one placement of iteration zero, and what follows a join still
follows it. So **a schedule never changes the result of a program that
keeps the promise**: the program has no race under the maximal
interleaving, the schedule's executions are among those interleavings,
and by the data-race-free guarantee each of them computes what some
statement-by-statement interleaving does, every one of which the
reducer's reassociation freedom makes equal.

What a schedule does change is which pairs of events *actually* happen in
parallel: a subset of the pairs of section 3. Two iterations of a loop
nothing bound never do; two entries of a drain nothing bound never do; a
spawned call run inline runs beside nothing of its spawner. That subset is
what the demotion reads.

## 5. What the compiler takes off

Every reducer accumulate is born atomic, so the compiler's job is to make
plain the ones the schedule leaves nothing running beside, and to keep the
rest: being wrong is a race, being cautious is only slow. The demotion
(`SSA/DemoteAtomics.h`) runs once every directive has been applied, since
that is what settles which loops carry a binding, and asks the contention
analysis (`SSA/Contention.h`) of each atomic accumulate whether two
activities running at once can write its address: `Disjoint` (no, and the
atomic comes off), `Shared` (yes) or `Unknown` (not shown; kept). The forks
taken as parallel are the loops bound to threads
(parallel_loops_by_block), every function such a loop calls, directly or
through other calls, whose each call is one iteration's
(functions_called_in_parallel), and nothing else: an unbound `parfor`
runs one iteration at a time, and a spawn still standing at this point is
run as written (a schedule that took up its parallelism rewrote it into a
push, and the drain is a `parfor` of its own, bound or not).

An address is Disjoint under a loop when it is storage the iteration made
for itself (an allocation inside the loop's body, or inside the called
function), an element at a subscript that is the index or one-to-one in
it, or -- for a function called in parallel -- a parameter every call hands
storage of one of those kinds, followed through the call graph to the
loops that run the calls (`parameter_contention`, a fixed point over every
call site of the program), each entry of a drain included. A drain's
entry carries the record slot of the sample it came from, and the slots
of one pass are distinct when no sample's work pushed onto the queue
twice before the pass: a count over paths, from each activity of the
owner -- an iteration of the sample loop, an entry of a drain -- through
the calls it makes and the queues it pushes onto that are drained earlier
in the same round, where a push onto a queue drained in rounds counts
nothing (its entry runs in the next round), a loop around a push counts
many, a recursion counts many when a push lies on its cycle and nothing
when none does (a tree traversal), and the greatest number along any
path is at most one (`SSA/DistinctSlots.h`). Two drains feeding one queue pass when no path
reaches both, which is how the renderer's material kernels and medium
kernel both push shadow rays and no sample sends two in a round. The
deferral asks this when it makes the drain and records the answer on the
drain's slot reads (`Function::injective_slots`) for the analysis to read,
through the splits a schedule makes of the drain loop. An address is
Shared when one word reaches every activity from outside the fork -- a
reducer the loop's owner was handed, a reducer declared outside the loop
-- and otherwise Unknown.

The tests, one program under five schedules (`wet`, spawned once per
iteration of `run`'s loop with the run's `total` and the iteration's `n`,
adds half its value into the first and one into the second; `step`, which
the loop calls, adds into `total` too):

| schedule | `step`'s `total +=` | `wet`'s `total +=` | `wet`'s `n +=` | test |
|---|---|---|---|---|
| none: the loop sequential, the spawn as written | plain | plain | plain | `ssa/demote-atomics-spawn-inline`, `correctness/llvm/demote-atomics-spawn-inline` |
| `step.defer(wet, q)`, nothing bound | plain | plain | plain | `ssa/demote-atomics-spawn-drain`, `backends/llvm/demote-atomics-spawn-drain`, `correctness/llvm/demote-atomics-spawn-drain` |
| the same, `run.bind(i, CPUThread)`: the loop parallel, the drain sequential | atomic | plain | plain | `ssa/demote-atomics-spawn-bound`, `backends/llvm/demote-atomics-spawn-bound` |
| the loop, the drain and the pass bound to GPU threads | atomic | atomic | plain: the slot is the entry's own | `backends/ptx/demote-atomics-spawn`, `correctness/gpu/demote-atomics-spawn` |

Every schedule prints the same: 97.5, and eight ones. `ssa/demote-
atomics-callee` and `correctness/gpu/reduce-param` are the same facts
for a plain call under a bound loop. The spawn's parallelism is decided by
what runs the spawned call -- the drain, bound or not; the spawner's own
loop, when the call is inline -- and not by what spawned it.

## 6. Open

1. **The spawned entry's reducer address on the GPU.** When a reducer
   handed to a spawned call is one word for all (`run`'s parameter), the
   entry stores its address, and on the JIT runtime's GPU path that is
   the producer launch's device copy, freed before the drain's kernel
   runs (`runtime/bonsai_cuda.h`, "Memory"): the shared-reducer spawn of
   2026-10-13's `race-spawn-accumulate` lost every spawned contribution
   under GPU binds, where `correctness/gpu/demote-atomics-spawn`, whose
   entry holds the same parameter's address, prints the right sum. Which
   route through the deferral stores a stale address is to be found
   under the new form, where every reducer a spawned call adds into is
   one of its arguments.
2. **Generated C++ source (`-b cppx`)** expresses an atomic accumulate as
   `std::atomic_ref`'s fetch_add or fetch_sub on a scalar; a vector
   reducer's accumulate is refused there (`-b cpp` and `-b llvm` have
   it).
