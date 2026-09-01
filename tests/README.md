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

### Codegen goldens name their target

A test's expected output lives in `<name>.expect`, and there is exactly one of
them per test. The `backends/llvm` and `backends/cpp` goldens record what
LLVM's optimizer produced, which genuinely differs between architectures --
LLVM 19 SLP vectorizes a pair of `i32` operations on AArch64 but not on
x86-64, and SimplifyCFG names the same select `%spec.select` on one and
`%common.ret.op` on the other.

Those tests therefore name the architecture they are testing rather than
inheriting the machine that happens to run them:

```
//! flags: -b llvm --target x86_64-unknown-linux-gnu
```

The backend generates code for that triple on any host, so one golden is
correct everywhere and a test blessed on a laptop is still right in CI. To test
another architecture, add a test that names its triple and record its own
golden; do not add a second golden for an existing test.

Tests that compile and run the generated code -- everything under
`correctness` -- deliberately omit `--target`, since they need code for the
machine executing them.

Before recording a codegen difference, confirm it is the optimizer targeting a
different architecture and not the compiler being wrong. Three differences that
looked architectural turned out to be real bugs: a target machine built from an
empty triple, a struct field sized with the host's `sizeof(pthread_mutex_t)`,
and two `IRBuilder` calls whose evaluation order was unspecified.

Also consult the official CTest
documentation: https://cmake.org/cmake/help/latest/manual/ctest.1.html
