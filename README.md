# bonsai
DSL for Recursive Geometric Queries

### Setup.

1. This project uses the [CMake](https://cmake.org/) build system. On MacOS,

```bash
brew install cmake
```

2. There is a dependency on LLVM; this currently uses [version 19.1.6](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.6).

If you already have this version installed somewhere, set the following parameters:
```bash
export LLVM_ROOT=/path/to/llvm-install # e.g., /Users/ajroot/projects/llvm-install-19
export LLVM_CONFIG=$LLVM_ROOT/bin/llvm-config
```

Otherwise, clone the submodule we have set up, and build it:
```bash
git submodule update --init --depth 1 deps/llvm-project
cmake -G "Unix Makefiles" -S deps/llvm-project/llvm -B deps/llvm-build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
      -DLLVM_ENABLE_RUNTIMES=compiler-rt \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_EH=ON \
      -DLLVM_ENABLE_RTTI=ON \
      -DLLVM_ENABLE_HTTPLIB=OFF \
      -DLLVM_ENABLE_LIBEDIT=OFF \
      -DLLVM_ENABLE_LIBXML2=OFF \
      -DLLVM_ENABLE_ZLIB=OFF \
      -DLLVM_ENABLE_ZSTD=OFF \
      -DLLVM_BUILD_32_BITS=OFF
cmake --build deps/llvm-build -j<N PARALLELISM>
cmake --install deps/llvm-build --prefix deps/llvm-install
```

3. Build `bonsai`:

```bash
# Option 1: normal
cmake 
cmake -S . -B build
cmake --build build -j<N PARALLELISM>

# Option 2: debug
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --config Debug -j<N PARALLELISM>
```

4. Run `bonsai` tests:

```bash
cd build # or build-dbg
make run_tests # runs cpp tests
```

### Acknowledgements

A significant portion of the code in this repository is modeled after, or directly taken from, the [Halide](https://github.com/halide/Halide) compiler. That is because they both have done incredible work, and because it is the compiler that I (AJR) am most familiar with navigating and understanding. As a result, this repository benefits heavily from over a decade of hard work from the Halide developers.

The lexer/parser was largely modeled after the [Simit](https://github.com/simit-lang/simit) lexer/parser, because the frontend language looks similar enough. There are a few important changes, including support for inline functions, and interfaces.
