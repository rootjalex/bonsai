# bonsai

**Click [here for the PLDI 2026 Bonsai branch](https://github.com/rootjalex/bonsai/tree/ajr/pldi-benchmarks) and [here for the PLDI 2026 Scion branch](https://github.com/rootjalex/bonsai/tree/cpg/pldi-artifact).**

DSL for Recursive Geometric Queries

### Setup.

1. This project uses the [CMake](https://cmake.org/) build system. On macOS,
   ```bash
   brew install cmake
   ```
2. There is a dependency on LLVM; this currently
   uses [version 23.1.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.1).
   On macOS, `brew install llvm@23` is sufficient.

If you have a custom version installed somewhere, set the following environment
variable:

```bash
export LLVM_ROOT=/path/to/llvm-install # e.g., /Users/ajroot/projects/llvm-install-23
```

To build LLVM yourself, clone the submodule we have set up, and build it.
Only LLVM itself is needed (no clang, lld or runtimes: nothing here links
them, and the apps use the conda environment's clang++); the three targets
are the host, the AArch64 the tests cross-compile for, and NVPTX for the
PTX backend:

```bash
git submodule update --init --depth 1 deps/llvm-project
cmake -G "Unix Makefiles" -S deps/llvm-project/llvm -B deps/llvm-build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_PROJECTS="" \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_EH=ON \
      -DLLVM_ENABLE_RTTI=ON \
      -DLLVM_ENABLE_HTTPLIB=OFF \
      -DLLVM_ENABLE_LIBEDIT=OFF \
      -DLLVM_ENABLE_LIBXML2=OFF \
      -DLLVM_ENABLE_ZLIB=OFF \
      -DLLVM_ENABLE_ZSTD=OFF \
      -DLLVM_BUILD_32_BITS=OFF \
      -DLLVM_USE_LINKER=lld \
      -DBUILD_SHARED_LIBS=ON
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

You may need to point `cmake` to the `deps` install, e.g. via:
```bash
cmake -S . -B build -DLLVM_DIR=/Users/ajroot/projects/bonsai/deps/llvm-install/lib/cmake/llvm
```

# Testing

Just run `ctest --test-dir build`! To regenerate `.expect` files, set the
environment variable `BONSAI_UPDATE_EXPECT=1` when running `ctest`.

### Acknowledgements

A significant portion of the code in this repository is modeled after, or
directly taken from, the [Halide] compiler. That is because they both have done
incredible work, and because it is the compiler that I (AJR) am most familiar
with navigating and understanding. As a result, this repository benefits heavily
from over a decade of hard work from the Halide developers.

The lexer/parser was largely modeled after the [Simit] lexer/parser, because the
frontend language looks similar enough. There are a few important changes,
including support for inline functions, and interfaces.

[Halide]: https://github.com/halide/Halide

[Simit]: https://github.com/simit-lang/simit
