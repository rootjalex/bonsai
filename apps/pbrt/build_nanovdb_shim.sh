#!/bin/bash
set -euo pipefail

# Build nanovdb_shim.cpp -- the implementation of the renderer's foreign
# functions over NanoVDB (docs/foreign-functions.md) -- in the forms the
# compiler links: host bitcode always, and PTX for a GPU architecture when
# one is named.
#
#     apps/pbrt/build_nanovdb_shim.sh <outdir> [sm_NN]
#
# Writes <outdir>/nanovdb_shim.bc, and <outdir>/nanovdb_shim.ptx when sm_NN
# is given, and prints the `--link` flags to hand the compiler, one per
# line. NanoVDB's header is pbrt's copy, found through pbrt's own build
# flags as build_scene_dump.sh finds pbrt's (PBRT= points at a built pbrt
# elsewhere). The driver's own copy of the shim -- the one checked against
# the generated header -- is compiled by the script that compiles the
# driver, with -DBONSAI_SHIM_CHECK and the include flag this prints as
# NANOVDB_INCLUDE= on its last line.

if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"
OUTDIR="${1:?usage: build_nanovdb_shim.sh <outdir> [sm_NN]}"
ARCH="${2:-}"
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
PBRT_BUILD="$(dirname "$PBRT")"
FLAGS="$PBRT_BUILD/CMakeFiles/pbrt_lib.dir/flags.make"
if [[ ! -f "$FLAGS" ]]; then
  echo "cannot read $FLAGS -- is $PBRT_BUILD a pbrt cmake build directory?" >&2
  exit 1
fi
NANOVDB_INCLUDE="$(grep -o -- '-I[^ ]*/nanovdb' "$FLAGS" | head -1)"
if [[ -z "$NANOVDB_INCLUDE" ]]; then
  echo "pbrt's build flags name no nanovdb include directory" >&2
  exit 1
fi
# The same compiler the driver is built with (see render.sh): clang, for
# the bitcode.
BONSAI_CXX="${BONSAI_CXX:-clang++}"
mkdir -p "$OUTDIR"
"$BONSAI_CXX" -std=c++17 -O2 -emit-llvm -c "$NANOVDB_INCLUDE" \
    "$PREFIX/nanovdb_shim.cpp" -o "$OUTDIR/nanovdb_shim.bc"
echo "--link"
echo "$OUTDIR/nanovdb_shim.bc"
if [[ -n "$ARCH" ]]; then
  # Device BITCODE, not PTX. PTX is opaque to the generated module's
  # optimizer: the shim stays an outlined `.visible .func` and the shadow
  # march pays a real call per density read -- the loop's live state
  # spilled around every step, the grid transform recomputed inside each
  # (explosion's march measured ~40% over pbrt's per loaded launch).
  # Bitcode links into the device and OptiX modules BEFORE they are
  # optimized (link_foreign_implementations routes it by its nvptx64
  # triple), the linked functions are marked always-inline
  # (CodeGen_PTX::finish), and the read inlines into the march loop --
  # nvcc's own shape for pbrt's medium code, reached through the same
  # --link mechanism. Compiled as plain C++ against the nvptx64 target
  # rather than clang's CUDA mode, which this machine's CUDA headers are
  # too new for; the bare target configures no standard headers, so the
  # toolchain's are named explicitly. If any of that is missing, the old
  # nvcc PTX is emitted instead, loudly: the renderer still works, the
  # march just keeps the call per step.
  CLANG_DIR="$(dirname "$(command -v "$BONSAI_CXX")")"
  GXX_GLOB=("$CLANG_DIR"/../lib/gcc/*/*/include/c++)
  GXX="${GXX_GLOB[0]:-}"
  SYSROOT_GLOB=("$CLANG_DIR"/../*/sysroot/usr/include)
  SYSROOT="${SYSROOT_GLOB[0]:-}"
  GPU_BC=""
  if [[ -d "$GXX" && -d "$SYSROOT" ]]; then
    if "$BONSAI_CXX" -std=c++17 -O2 --target=nvptx64-nvidia-cuda \
        -nostdinc++ -isystem "$GXX" -isystem "$GXX/x86_64-conda-linux-gnu" \
        -isystem "$SYSROOT" -emit-llvm -c "$NANOVDB_INCLUDE" \
        "$PREFIX/nanovdb_shim.cpp" -o "$OUTDIR/nanovdb_shim.gpu.bc"; then
      GPU_BC="$OUTDIR/nanovdb_shim.gpu.bc"
    fi
  fi
  if [[ -n "$GPU_BC" ]]; then
    echo "--link"
    echo "$GPU_BC"
  else
    echo "WARNING: no device bitcode ($BONSAI_CXX against nvptx64 failed);" \
         "falling back to nvcc PTX, the density read stays an outlined call" >&2
    NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
    if [[ ! -x "$NVCC" ]]; then
      echo "no nvcc at $NVCC -- set CUDA_HOME or NVCC" >&2
      exit 1
    fi
    # -rdc=true: the function is to be called from another module's code,
    # so it is emitted as a `.visible .func` rather than inlined away into
    # nothing.
    "$NVCC" -ptx -rdc=true -O3 -std=c++17 -arch="$ARCH" "$NANOVDB_INCLUDE" \
        -x cu "$PREFIX/nanovdb_shim.cpp" -o "$OUTDIR/nanovdb_shim.ptx"
    echo "--link"
    echo "$OUTDIR/nanovdb_shim.ptx"
  fi
fi
echo "NANOVDB_INCLUDE=$NANOVDB_INCLUDE"
