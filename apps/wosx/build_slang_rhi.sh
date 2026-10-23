#!/bin/bash

set -euo pipefail

# Builds slang-rhi -- the GPU runtime WoSX's and FCPW's GPU code run on --
# from the submodule FCPW pins (deps/wosx/deps/fcpw/deps/slang-rhi) into
# deps/slang-rhi-build, which is where compare.sh looks for it when a
# schedule runs on the GPU. slang-rhi fetches the Slang compiler it needs as
# a prebuilt release at configure time (the version FCPW's CMake pins,
# SLANG_RHI_FETCH_SLANG_VERSION), so this needs the network once.
#
#     git -C deps/wosx/deps/fcpw submodule update --init deps/slang-rhi
#     apps/wosx/build_slang_rhi.sh
#
# Run from the repository root, inside the `bonsai` conda environment: the
# library is built with its clang, the compiler the driver is built with.
# Only the CUDA backend is built -- WoSX's persistent-thread kernels need
# it (its demos say so and disable them on any other), and the driver asks
# for it by name -- and the Vulkan, WebGPU and CPU backends, OptiX, the
# tests and the examples are left out: nothing about how a compute shader
# runs on this machine depends on them, and they would pull in headers the
# machine does not have.
if [[ "$(pwd)" == */apps/wosx ]]; then
  cd ../..
fi

SLANG_RHI="deps/wosx/deps/fcpw/deps/slang-rhi"
if [[ ! -f "$SLANG_RHI/CMakeLists.txt" ]]; then
  echo "$SLANG_RHI is empty: run \`git -C deps/wosx/deps/fcpw submodule update --init deps/slang-rhi\` first" >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the bonsai conda environment first (clang comes from it)" >&2
  exit 1
fi

# FCPW's pin of the Slang release (deps/wosx/deps/fcpw/CMakeLists.txt).
SLANG_VERSION="$(sed -n 's/.*SLANG_RHI_FETCH_SLANG_VERSION "\([0-9.]*\)".*/\1/p' deps/wosx/deps/fcpw/CMakeLists.txt | head -1)"
if [[ -z "$SLANG_VERSION" ]]; then
  echo "could not read FCPW's SLANG_RHI_FETCH_SLANG_VERSION from deps/wosx/deps/fcpw/CMakeLists.txt" >&2
  exit 1
fi

cmake -S "$SLANG_RHI" -B deps/slang-rhi-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DSLANG_RHI_FETCH_SLANG_VERSION="$SLANG_VERSION" \
    -DSLANG_RHI_BUILD_TESTS=OFF -DSLANG_RHI_BUILD_TESTS_WITH_GLFW=OFF \
    -DSLANG_RHI_BUILD_EXAMPLES=OFF -DSLANG_RHI_INSTALL=OFF \
    -DSLANG_RHI_ENABLE_CUDA=ON -DSLANG_RHI_ENABLE_OPTIX=OFF \
    -DSLANG_RHI_ENABLE_VULKAN=OFF -DSLANG_RHI_ENABLE_WGPU=OFF \
    -DSLANG_RHI_ENABLE_CPU=OFF
cmake --build deps/slang-rhi-build -j

echo "slang-rhi built under deps/slang-rhi-build (Slang $SLANG_VERSION under deps/slang-rhi-build/_deps/slang-src)"
