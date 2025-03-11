### Cheatsheet

CTest is used to manage tests. Run `ctest` in the root of the build directory to
run all tests. Some flags might be useful:

- `-R`: select tests that match a particular regular expression.
- `-L`: select a label to run. Existing labels include `backends`,
  `correctness`, `errors`, `llvm`, and `parsing`.
- `-j`: run on multiple processors

To update the golden outputs during a test run, set the environment variable
`BONSAI_UPDATE_EXPECT` to `1` while running CTest:

```console
$ BONSAI_UPDATE_EXPECT=1 ctest -L llvm
```

Also consult the official CTest
documentation: https://cmake.org/cmake/help/latest/manual/ctest.1.html
