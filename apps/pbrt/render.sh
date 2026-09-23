#!/bin/bash

set -euo pipefail

# Build and run PBRT milestone 0. See render.bonsai.
#
# Runs from either the repository root or this directory.
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi

PREFIX="apps/pbrt"

# PBRT's own `--spp <n>`, under PBRT's own name, taken before the positional
# arguments so it can appear anywhere on the line. It is passed straight to
# scene_dump, which sets it on PBRTOptions before parsing -- so every sampler's
# Create reads it and there is no second implementation here of what the flag
# means.
#
# Worth having because a scene in `scenes/` is sized for a comparison that has
# to finish, and 64 samples with the pixel jitter off is a noisy picture. This
# script is the one for looking at an image, so this is the knob it wants.
# `--disable-pixel-jitter` is pbrt's other option and goes the same way: it pins
# every sample of a pixel to the pixel's centre, which is what this renderer did
# before it had a reconstruction filter.
#
# `--schedule <name>` picks how the program is run: one of the files in
# schedules/, compiled alongside render.bonsai (see schedules/packet.bonsai,
# the default).
DUMP_OPTS=()
SCHEDULE="${SCHEDULE:-packet}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spp)
      if [[ $# -lt 2 ]]; then
        echo "--spp needs a sample count" >&2
        exit 1
      fi
      DUMP_OPTS+=(--spp "$2")
      shift 2
      ;;
    --disable-pixel-jitter)
      DUMP_OPTS+=(--disable-pixel-jitter)
      shift
      ;;
    --schedule)
      if [[ $# -lt 2 ]]; then
        echo "--schedule needs a name from $PREFIX/schedules/" >&2
        exit 1
      fi
      SCHEDULE="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# The driver has to be built with clang: the generated header declares its
# vectors with ext_vector_type, and only clang has it. gcc's vector_size takes a
# size in bytes and requires a power of two, so it cannot spell a three-element
# vector laid out the way LLVM lays one out. See src/CodeGen/CPP.cpp.
#
# Deliberately not $CXX, which conda's compiler packages export as their gcc
# wrapper -- honouring it would mean this picks gcc inside an activated
# environment and fails a hundred lines later.
BONSAI_CXX="${BONSAI_CXX:-clang++}"
# Asked rather than assumed: gcc merely warns that it is ignoring the attribute
# and then fails on every use of the type. The swizzle is what makes this fail
# there -- with the attribute ignored, f3 is a plain float and .y is not a
# member of it.
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header: it needs clang's" >&2
  echo "ext_vector_type. Set BONSAI_CXX to a clang++." >&2
  exit 1
fi
# See compare.sh: runtime/bonsai_parallel.h uses tbb::parallel_for where TBB is
# reachable and std::thread where it is not, and TBB lives beside the compiler
# in a conda environment. Silent here rather than warning, because this script
# is for looking at an image and not for timing one.
TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
TBB_FLAGS=()
if [[ -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  TBB_FLAGS=(-I"$TBB_PREFIX/include" -L"$TBB_PREFIX/lib"
             -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb)
fi
SCENE="${1:-$PREFIX/scenes/three-spheres.pbrt}"
# The renderer writes linear float, as pbrt does; the PNG beside it is the
# post-processing step, and is what to actually look at.
OUT="${2:-$PREFIX/pbrt.pfm}"
PNG="${OUT%.pfm}.png"

# The compiler's build directory, as compare.sh takes it: `build` unless
# BONSAI_BUILD_DIR names another (build-23, for the LLVM 23 build).
BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
cmake --build "$BONSAI_BUILD_DIR" -j

# The scene comes from a .pbrt file, read by PBRT's own parser. Needs a built
# pbrt; see build_scene_dump.sh.
bash $PREFIX/build_scene_dump.sh
# The spectral tables are generated from pbrt's source, and one of them -- the
# film's D65 -- is reproduced by following what pbrt does rather than copied.
# Ask a running pbrt whether the result is right before rendering with it.
./$PREFIX/scene_dump --check-tables
./$PREFIX/scene_dump ${DUMP_OPTS[@]+"${DUMP_OPTS[@]}"} "$SCENE" \
    "$PREFIX/scene.txt"

# The spectral data is generated (see make_spectrum_tables.py) and the fit that
# uses it is a port, so check the round trip before rendering with it: a fit
# that is subtly wrong still produces plausible numbers, just the wrong colour.
"$BONSAI_CXX" -std=c++20 -O2 -I$PREFIX $PREFIX/rgb2spec_check.cpp \
    -o $PREFIX/rgb2spec_check
./$PREFIX/rgb2spec_check
rm $PREFIX/rgb2spec_check

# `-p ssa` because bind is an SSA rewrite. The other two outputs are here to
# be looked at when something renders wrong: the .bir is what the schedule
# left behind, the .ll is what it became.
#
# The flags match compare.sh, because a render that is not the one being
# compared against pbrt is a render nothing checks. `--no-heap` refuses any
# allocation, which a renderer's inner loop must not make; `--ffp-contract`
# fuses `a * b + c`, which is what pbrt's gcc build does.
FLAGS=(-p ssa --no-heap --ffp-contract)
# The program and its schedule, two inputs making one program.
INPUTS=(-i $PREFIX/render.bonsai -i "$PREFIX/schedules/$SCHEDULE.bonsai")
if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
  echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
  exit 1
fi
# `--fast-math` and a 128-register cap for a schedule that runs on the GPU,
# as compare.sh and the eval harness give them: pbrt's GPU build is nvcc's
# --use_fast_math with -maxrregcount 128, its CPU build is exact (see
# CompilerOptions::fast_math and gpu_max_registers).
if grep -q "GPUBlock\|GPUThread" "$PREFIX/schedules/$SCHEDULE.bonsai"; then
  FLAGS+=(--fast-math --gpu-max-registers 128)
fi
"./$BONSAI_BUILD_DIR/compiler" -p ssa "${INPUTS[@]}" -o $PREFIX/render.bir
"./$BONSAI_BUILD_DIR/compiler" "${FLAGS[@]}" "${INPUTS[@]}" -b llvm -o $PREFIX/render.ll
"./$BONSAI_BUILD_DIR/compiler" "${FLAGS[@]}" "${INPUTS[@]}" -b cpp -o $PREFIX/render

# -I. so that the generated header can find the runtime it includes.
"$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $PREFIX/render_hook.cpp \
    $PREFIX/render.o "${TBB_FLAGS[@]}" -o $PREFIX/render.out

./$PREFIX/render.out --no-implicit-copies "$PREFIX/scene.txt" "$OUT"

# The image holds normals rather than radiance, so the encoding is the remap
# that makes a direction visible rather than pbrt's sRGB curve.
python3 $PREFIX/to_png.py "$OUT" "$PNG" --normals

# The albedo beside it is a colour, so it gets the sRGB curve. One spectral
# sample per pixel, so what this shows is a reflectance seen at four
# wavelengths rather than the colour a converged render would give -- see the
# note on `albedo` in render.bonsai.
ALBEDO="${OUT%.pfm}-albedo.pfm"
python3 $PREFIX/to_png.py "$ALBEDO" "${ALBEDO%.pfm}.png"

# The radiance, which is the render -- the other two are what the gbuffer
# records on the way. Also an sRGB curve, being a colour.
#
# It is very noisy, and that is the integrator rather than a fault: a random
# walk never samples a light, it scatters uniformly over the sphere and finds
# one by chance, so on a scene lit by one small sphere most paths find nothing.
# pbrt's own randomwalk on the same scene is exactly as noisy.
RADIANCE="${OUT%.pfm}-radiance.pfm"
python3 $PREFIX/to_png.py "$RADIANCE" "${RADIANCE%.pfm}.png"

# render.h is left where it was written, because render_hook.cpp includes it
# and the next build wants it there. It is generated rather than committed, so
# there is nothing to keep in step.
rm $PREFIX/render.bir
rm $PREFIX/render.ll
rm $PREFIX/render.o
rm $PREFIX/render.out
rm $PREFIX/scene.txt
rm $PREFIX/scene_dump
# Only clang on Apple platforms leaves one of these behind.
rm -rf $PREFIX/render.out.dSYM

exit 0
