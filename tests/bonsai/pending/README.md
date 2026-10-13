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
| `reduce-param-gpu` | a `reduce(+)` parameter of the run accumulated from a loop bound to the GPU, no queues | `1.0`, the first value alone (a drain bound to the GPU adding into it: `1.0` too; a spawned call's accumulate: `0.0`) | `65` |
| `spawn-then-spawn` | a spawned call (`spawn l += dampen(x, l)`, deferred onto `dries`) whose drained function spawns onto a second queue (`spawn l += soak(x)`, onto `soaks`) | `[internal] Error: src/SSA/SSA.cpp:748 -- soaks!end lists predecessor ^ that does not point to it` | half of each value |
| `spawn-then-spawn-after-defer` | the same beside an ordinary deferral with a rest (`step.defer(shade, hits)`, the rest `out[i] = l`) | compiles and runs: the wet scenes lose every soak the `dries` drain spawned and the first sample's soak from the `hits` drain; with `hits.specialize(s)` added, SIGBUS after printing the sums without any soak | the four lines of the `.expect` |

All four fail the same way with the compiler of 29e15282, before the
per-arm queues of 2026-10-12 (checked against a build of that commit):
none is a regression of that work, which is what found them. The
renderer is exposed to none of them: its radiance is a per-sample slot of
the ray queue's record, not a reducer of the run, and no spawned call of
its chain spawns onward.

Where to look:

- `specialize-recursive-callee`: `specialize_callees` in SSA/Specialize.cpp,
  called from the split in SSA/Defer.cpp (`grow`). `specialize_function`
  has already retargeted a copy's call of itself to the copy; the walk
  that follows the key into callees reads that call as handing the
  specialized value to a callee and copies again. A copy's call of itself
  is the one call to leave alone.
- `reduce-param-gpu`: CodeGen/CodeGen_GPU_Host.cpp, a reducer parameter
  reaching a loop placed on the device. The GPU reduction the suite covers
  is `atomic total[0] += ..` into a `mut array` word
  (correctness/gpu/block-reduce); a reducer parameter has no test and does
  not work. With the loop on the GPU and the drain on the host, the
  compiler refuses the program instead (`[unimplemented] the loop bound to
  the GPU reads hits_queue ... whose elements hold pointers of their own`).
- `spawn-then-spawn`: SSA/Defer.cpp, the blocks a spawned deferral makes
  around the call (`<queue>!end`) when the drained function pushes onto
  another queue; the predecessor list of the second queue's end block
  names a block that does not jump to it.
