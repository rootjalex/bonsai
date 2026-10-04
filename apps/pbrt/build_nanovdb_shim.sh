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
echo "NANOVDB_INCLUDE=$NANOVDB_INCLUDE"
