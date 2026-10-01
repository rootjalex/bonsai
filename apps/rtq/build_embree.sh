#!/bin/bash

set -euo pipefail

# Builds Embree from the deps/embree submodule into deps/embree-build and
# installs it under deps/embree-install, which is where compare.sh looks.
#
#     git submodule update --init deps/embree
#     apps/rtq/build_embree.sh
#
# Run from the repository root, inside the `bonsai` conda environment, whose
# clang and TBB are what Embree is built with -- the same TBB the program's
# runtime spreads its loops with, so both sides share one thread pool.
#
# What is built: the triangle kernels only, for AVX2 and AVX-512, which is
# what Embree uses on this machine (its ISA is chosen at run time, and an
# AVX-512 machine takes the AVX-512 kernels). The other geometry kinds and
# the tutorials are left out to keep the build short; nothing about how
# triangles are traversed depends on them. Everything else is Embree's
# default, as shipped: filter functions on, ray masks on, no backface
# culling.
if [[ "$(pwd)" == */apps/rtq ]]; then
  cd ../..
fi

if [[ ! -f deps/embree/CMakeLists.txt ]]; then
  echo "deps/embree is empty: run \`git submodule update --init deps/embree\` first" >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the bonsai conda environment first (TBB and clang come from it)" >&2
  exit 1
fi

cmake -S deps/embree -B deps/embree-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX="$PWD/deps/embree-install" \
    -DEMBREE_TUTORIALS=OFF -DEMBREE_ISPC_SUPPORT=OFF \
    -DEMBREE_TASKING_SYSTEM=TBB -DEMBREE_TBB_ROOT="$CONDA_PREFIX" \
    -DEMBREE_MAX_ISA=NONE \
    -DEMBREE_ISA_SSE2=OFF -DEMBREE_ISA_SSE42=OFF -DEMBREE_ISA_AVX=OFF \
    -DEMBREE_ISA_AVX2=ON -DEMBREE_ISA_AVX512=ON \
    -DEMBREE_GEOMETRY_QUAD=OFF -DEMBREE_GEOMETRY_CURVE=OFF \
    -DEMBREE_GEOMETRY_SUBDIVISION=OFF -DEMBREE_GEOMETRY_USER=OFF \
    -DEMBREE_GEOMETRY_INSTANCE=OFF -DEMBREE_GEOMETRY_INSTANCE_ARRAY=OFF \
    -DEMBREE_GEOMETRY_GRID=OFF -DEMBREE_GEOMETRY_POINT=OFF
cmake --build deps/embree-build -j
cmake --install deps/embree-build

echo "Embree installed under deps/embree-install"
