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
3. Two optional, header-detected dependencies of the generated code's
   runtime (`runtime/bonsai_parallel.h`), looked for beside the C++
   compiler, which is where a conda environment puts them: Intel TBB
   (`conda install tbb-devel`), for the parallel loop and the parallel
   sort; and Intel's header-only
   [x86-simd-sort](https://github.com/intel/x86-simd-sort) (BSD-3), for the
   sort of a sorted queue, installed by copying its `src/` to
   `$CONDA_PREFIX/include/x86simdsort` and taken only when the driver is
   compiled with `-march=native`. Without them the runtime uses
   `std::thread` and its own radix sort.

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

# Benchmarks

`bash apps/benchmark.sh` remakes every figure of the paper from the compiler
built out of the checkout, in one go: the pbrt GPU and CPU sweeps and the GPU
heatmap (`apps/pbrt`), the rtq query figures (`apps/rtq`) and the wosx GPU
figures (`apps/wosx`). `--stages` picks a subset (`--stages pbrt`, `--stages
"rtq wosx"`), `--dry-run` prints the commands. A full run takes about 12 h
20 min on the paper's machine (measured 11-02/03; the script's header has
the stages' times). The run is attributed to the
commit it was built from and refuses uncommitted compiler sources;
`apps/pbrt/results/provenance.txt` records it. The run ends by gathering
the figures -- each one's .pdf, .png and the .txt that reads it, the
tables' .tex -- and that provenance into `apps/plots`, the last run's in
one place (`--stages gather` only gathers). `apps/benchmarks.md` says
which schedule file each bar of each figure is.

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
