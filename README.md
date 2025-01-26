# bonsai
DSL for Recursive Geometric Queries

### Setup.

1. This project uses the [CMake](https://cmake.org/) build system. On MacOS,

```bash
brew install cmake
```

2. There is a dependency on LLVM; this currently uses [version 19.1.6](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.6).

3. Build `bonsai`:

```bash
export LLVM_ROOT=/path/to/llvm-install # e.g., /Users/ajroot/projects/llvm-install-19
export LLVM_CONFIG=$LLVM_ROOT/bin/llvm-config

# Option 1: normal
cmake 
cmake -S . -B build
cmake --build build

# Option 2: debug
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --config Debug
```

4. Run `bonsai` tests:

```bash
make run_tests
```

### Acknowledgements

A significant portion of the code in this repository is modeled after, or directly taken from, the [Halide](https://github.com/halide/Halide) compiler. That is because they both have done incredible work, and because it is the compiler that I (AJR) am most familiar with navigating and understanding. As a result, this repository benefits heavily from over a decade of hard work from the Halide developers.

The lexer/parser was largely modeled after the [Simit](https://github.com/simit-lang/simit) lexer/parser, because the frontend language looks similar enough. There are a few important changes, including support for inline functions, and interfaces.
