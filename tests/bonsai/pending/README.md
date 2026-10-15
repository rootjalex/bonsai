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
| `spawn-then-spawn-after-defer` | a spawned call (`spawn dampen(x, l)`, deferred onto `dries`) whose drained function spawns onto a second queue (`spawn soak(x, l)`, onto `soaks`), beside an ordinary deferral with a rest (`step.defer(shade, hits)`, the rest `out[i] = l`) | compiles and runs: the wet scenes lose every soak the `dries` drain spawned and the first sample's soak from the `hits` drain; with `hits.specialize(s)` added, SIGBUS after printing the sums without any soak | the four lines of the `.expect` |

Both fail the same way with the compiler of 29e15282, before the per-arm
queues of 2026-10-12 (checked against a build of that commit): neither is
a regression of that work, which is what found them. Two more were here
and are closed. `reduce-param-gpu` -- `atomic total += x` into a
`reduce(+)` parameter inside a function the GPU-bound loop calls,
printing 1.0 for 65 -- was the atomics demotion's, which asked only the
function's own loops, and is fixed: the program is
correctness/gpu/reduce-param, with backends/ptx/reduce-param (the
`atom.global.add.f32` kept) and the ssa, backends/llvm and
correctness/llvm demote-atomics-callee tests (SSA/DemoteAtomics.h,
SSA/Contention.h: functions_called_in_parallel). `spawn-then-spawn` -- the
construct above without the ordinary deferral beside it, which stopped
with `[internal] Error: src/SSA/SSA.cpp:748 -- soaks!end lists
predecessor ^ that does not point to it` -- went away with the `spawn
f(args);` form of 2026-10-15: the second deferral's one producer, the
dries drain's call, had the accumulate for its continuation under `spawn
x op= g()`, which the single-producer path of SSA/Defer.cpp read as the
producer's rest; a spawned call's continuation owes nothing now. It is
ssa/spawn-then-spawn and correctness/llvm/spawn-then-spawn. The renderer
is exposed to neither of the two left: its radiance is a per-sample slot
of the ray queue's record, not a reducer of the run, and no spawned call
of its chain spawns onward.

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
