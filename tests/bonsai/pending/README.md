# Pending: programs the compiler gets wrong today

Each `.bonsai` here is a small program the compiler fails on, and the
`.expect` beside it is the output it should produce, written so that
moving the pair into the suite's directory (`correctness/llvm`, or
`correctness/gpu` for the GPU one) and adding the name to that
directory's `CMakeLists.txt` is the whole act of closing it. Nothing here
is registered with ctest, so the suite stays green. Run one by hand from
the repository root, inside the conda env `bonsai`:

    ./build/compiler -p ssa -b llvm -e -i tests/bonsai/pending/<name>.bonsai

Each file's header says what it does, what comes out today and why.

| test | the construct | today | expected |
|---|---|---|---|
| `specialize-recursive-callee` | a queue's specialize (`hits.specialize(r)`) copies a function handed the key that hands it to itself | the compiler never finishes: `finish!Big`, `finish!Big!Big`, `finish!Big!Big!Big`, ... until the stack is gone (`ulimit -s 2048` makes it die in seconds; the loop form `run.specialize(flag)` over such a function is fine) | `-1 0 1 2 3 3.5 4 4.5` |
| `spawn-then-spawn` | a spawned call (`spawn l += dampen(x, l)`, deferred onto `dries`) whose drained function spawns onto a second queue (`spawn l += soak(x)`, onto `soaks`) | `[internal] Error: src/SSA/SSA.cpp:748 -- soaks!end lists predecessor ^ that does not point to it` | half of each value |
| `spawn-then-spawn-after-defer` | the same beside an ordinary deferral with a rest (`step.defer(shade, hits)`, the rest `out[i] = l`) | compiles and runs: the wet scenes lose every soak the `dries` drain spawned and the first sample's soak from the `hits` drain; with `hits.specialize(s)` added, SIGBUS after printing the sums without any soak | the four lines of the `.expect` |

All three fail the same way with the compiler of 29e15282, before the
per-arm queues of 2026-10-12 (checked against a build of that commit):
none is a regression of that work, which is what found them. A fourth,
`reduce-param-gpu` -- `atomic total += x` into a `reduce(+)` parameter
inside a function the GPU-bound loop calls, printing 1.0 for 65 -- was
the atomics demotion's, which asked only the function's own loops, and is
fixed: the program is correctness/gpu/reduce-param, with backends/ptx/
reduce-param (the `atom.global.add.f32` kept) and the ssa, backends/llvm
and correctness/llvm demote-atomics-callee tests (SSA/DemoteAtomics.h,
SSA/Contention.h: functions_called_in_parallel). The renderer is exposed
to none of the three: its radiance is a per-sample slot of the ray queue's
record, not a reducer of the run, and no spawned call of its chain spawns
onward.

Where to look, each read off the compiler's own output (the dumps and the
runs are reproduced by the commands in each file's header):

- `specialize-recursive-callee`: `specialize_callees` in SSA/Specialize.cpp
  -- the loop over a copy's calls that coins `clone_name = call->call.name
  + "!" + v.label` -- called from the split in SSA/Defer.cpp (`grow`).
  `specialize_function` has renamed the copy's parameter `r` to `r!Big`
  and retargeted the copy's call of itself to `finish!Big`; the walk then
  reads `finish!Big(r!Big, k, n - 1)` as a call handing the specialized
  value to a callee, coins `finish!Big!Big`, copies, and walks the copy,
  whose self-call is `finish!Big!Big(r!Big!Big, ...)`, and so on (the
  names, under gdb with a breakpoint on specialize_function). A copy's
  call of itself is the one call to leave alone: its callee already is
  the copy. Mutual recursion is fine as it stands -- the other function's
  copy exists by the time the walk comes back round to it, and the call
  is retargeted without a copy.
- `spawn-then-spawn`: SSA/Defer.cpp, the single-producer path that ends
  the producer's iteration at its call (`end = fresh_block(prefix +
  "end")`, under "The producer passes the queue"). `dampen.defer(soak,
  soaks)` finds one call into its chain from the owner: the `dries`
  drain's `callc dampen(x, l, ...) dries!ran(l)`, a spawned call whose
  continuation adds dampen's value into `l` -- the accumulate that drain
  owes. The path reads that continuation as the producer's rest: it
  makes `soaks!end` with `dries!run` as its predecessor and, since the
  continuation takes the call's value (`keeps_result`), stores the value
  into a record slot it reaches from `soaks!end` -- while `dries!run`'s
  call still continues to `dries!ran` (the repointing, `pcall.cont = ...`,
  comes after), which is what SSA.cpp's predecessor check reports (the
  block it dumps is `dries!run`, with the slot GEPs already added).
  Repointing earlier would not do: the drain's accumulate would be cut
  off. A producer that is another queue's drain has no rest to move --
  its iteration is the call and, for a spawned queue, the accumulate --
  and should hand the queue down and keep its continuation, as the
  several-producers path does for each of its producers; and a spawned
  deferral has no value flowing up the chain to keep a result record for
  ("nothing flows up", a few hundred lines above).
- `spawn-then-spawn-after-defer`: SSA/Defer.cpp, the queue's capacity when
  its producer is a drain (`producer_is_drain`, "the producer loop is
  another queue's drain"): each entry of that drain pushes at most once,
  so its capacity bounds this queue's -- and several producers that are
  drains "share that bound", checked as their capacities being the same
  value. `soaks` has two producers, the `hits` drain (through shade and
  wetten) and the `dries` drain (through dampen and wetten), of capacity
  8 and 8, so `soaks` gets 8 slots (`soaks_x = alloca<f32[8]>` in
  `--dump-ssa-postschedule`) for up to 16 pushes. Evidence: with shade
  pushing only for Fuzz (3 of 8) and the dries pushes limited to 4, 5 or
  8 by the `wet` test -- 7, 8 and 11 pushes against the 8 slots -- the
  sums come out right, right, and a segfault. The shared bound is right
  for the leaves of one split family, whose entries total the family's;
  two drains of different queues bound the pushes by the sum of their
  capacities. The lost first sample and the SIGBUS with `hits.specialize(
  s)` are the overflow landing on whatever the layout puts after the
  arrays.
