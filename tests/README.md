### Cheatsheet

CTest is used to manage tests. Run `ctest` in the root of the build directory to
run all tests. Some flags might be useful:

- `-R`: select tests that match a particular regular expression.
- `-L`: select a label to run. Existing labels include `backends`,
  `correctness`, `error`, `llvm`, `lower`, `opt`, and `parsing`.
- `-j`: run on multiple processors

To update the golden outputs during a test run, set the environment variable
`BONSAI_UPDATE_EXPECT` to `1` while running CTest:

```console
$ BONSAI_UPDATE_EXPECT=1 ctest -L llvm
```

### Goldens of the SSA block graph

Most tests under `tests/bonsai/ssa/` diff the *relooper's* output: `-p ssa`
converts to SSA, applies the schedule, and turns the result back into
statements, and those statements are what gets printed. That is worth pinning,
since it is what code generation consumes for any function that is not
vectorized -- but it is a reconstruction. Two different block graphs can reloop
to the same statements, so a golden of the statements is a lossy witness for
what a schedule actually did to the graph.

`tests/bonsai/ssa/blocks/` holds goldens of the graph itself, using two flags:

- `--dump-ssa-preschedule`: the SSA as built, before any rewrite runs.
- `--dump-ssa-postschedule`: what the schedule left, immediately before the
  relooper.

Both print to stdout, ahead of whatever else the compile prints, so a test can
ask for either or both. Asking for both is usually what you want for a
transform: the golden then says what the rewrite *changed*, not only what it
ended at. Imported functions are left out unless `-v` is also given, matching
what the program printer does.

### Architecture-specific goldens

A test's expected output normally lives in `<name>.expect`. A few of the
`backends` goldens record what LLVM's optimizer produced, and that genuinely
differs between target architectures -- LLVM 19 SLP vectorizes a pair of `i32`
operations on AArch64 but not on x86-64, and SimplifyCFG names the same select
`%spec.select` on one and `%common.ret.op` on the other.

Where that happens, an optional `<name>.<arch>.expect` overrides the base
golden on that architecture, with `<arch>` being `CMAKE_SYSTEM_PROCESSOR`
lower-cased (`x86_64`, `arm64`). The base file stays whatever CI blesses, so
adding an override never disturbs the other platform. `BONSAI_UPDATE_EXPECT`
re-blesses the override where one exists and the base golden where one does
not.

Only add an override once you have confirmed the difference is the optimizer
targeting a different architecture, and not the compiler being wrong. Three
differences that looked like this turned out to be real bugs: a target machine
built from an empty triple, a struct field sized with the host's
`sizeof(pthread_mutex_t)`, and two `IRBuilder` calls whose evaluation order was
unspecified.

Also consult the official CTest
documentation: https://cmake.org/cmake/help/latest/manual/ctest.1.html
