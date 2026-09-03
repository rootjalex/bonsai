#!/bin/bash

set -euo pipefail

# Render the same scene with pbrt and with this app, and compare the normals.
#
# This is the app's reason for existing made testable. The claim is that the
# bonsai source is a faithful transcription of pbrt and that the schedule only
# makes it faster; the way to find out is to ask pbrt what the answer is. What
# is compared is pbrt's gbuffer normals, which come from the camera ray's first
# hit, so a difference means the camera transform, the ray generation, or an
# intersection disagrees -- the parts most likely to be subtly wrong and least
# likely to look wrong.
#
# Needs a built pbrt. Point PBRT at it if it is somewhere else:
#
#     PBRT=~/src/pbrt-v4/build/pbrt apps/pbrt/compare.sh

if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi

PREFIX="apps/pbrt"
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
# The driver has to be built with clang: the generated header declares its
# vectors with ext_vector_type, which is what makes the C++ side's float3 the
# same thing as LLVM's <3 x float>, and only clang has it. gcc's vector_size is
# not a substitute -- it takes a size in bytes and requires a power of two, so
# there is no way to spell a three-element vector that is laid out the way LLVM
# lays one out. See src/CodeGen/CPP.cpp.
#
# Deliberately not $CXX. Conda's compiler packages export that as their gcc
# wrapper, so honouring it means a plain `apps/pbrt/compare.sh` inside an
# activated environment silently picks gcc and fails a hundred lines later.
BONSAI_CXX="${BONSAI_CXX:-clang++}"
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
WORK="${WORK:-$PREFIX/compare-out}"
# The scene both sides render. Any .pbrt this renderer understands will do.
SCENE="${1:-$PREFIX/scenes/three-spheres.pbrt}"

if [[ ! -x "$PBRT" ]]; then
  echo "no pbrt at $PBRT -- set PBRT to a built one" >&2
  exit 1
fi
if [[ ! -f "$SCENE" ]]; then
  echo "no scene at $SCENE" >&2
  exit 1
fi
# pbrt is run from the working directory, so that the .exr it is told to write
# lands there rather than beside the repository. That means it needs the scene
# by a path that survives the change of directory, and a relative one does not.
SCENE="$(cd "$(dirname "$SCENE")" && pwd)/$(basename "$SCENE")"
# Asked rather than assumed. Existing is not the same as being able to compile
# the header: gcc merely warns that it is ignoring ext_vector_type and then
# fails on every use of the type, which is a hundred lines of template noise
# saying nothing about what is actually wrong. The swizzle is what makes this
# fail on gcc -- with the attribute ignored, f3 is a plain float and .y is not a
# member of it.
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header: it needs clang's" >&2
  echo "ext_vector_type. Set BONSAI_CXX to a clang++." >&2
  exit 1
fi

# TBB, looked for beside the compiler -- which is where a conda environment
# puts both. runtime/bonsai_parallel.h runs a bind(p, CPUThread) loop through
# tbb::parallel_for when that header is reachable and through std::thread when
# it is not, and the difference is worth the search: the pixels of a render
# cost wildly different amounts, so a loop balanced by work stealing finishes
# when its work does rather than when its unluckiest thread does. On
# killeroo-simple that is 15.7 seconds against 10.9.
TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
TBB_FLAGS=()
if [[ -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  TBB_FLAGS=(-I"$TBB_PREFIX/include" -L"$TBB_PREFIX/lib"
             -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb)
else
  echo "no TBB beside $BONSAI_CXX -- the render will be timed with the" >&2
  echo "std::thread fallback, which balances the loop less well." >&2
fi

mkdir -p "$WORK"

cmake --build build -j

# The scene is read once, by PBRT's parser, and both sides render what it says.
# There is no second description of it to keep in step -- which is the point,
# and what makes adding a scene to this comparison a matter of writing one file.
#
# PBRT_TREE=1 also takes the BVH PBRT built, so that what is being timed is the
# traversal the schedule produced rather than whose builder found a better
# tree. Off by default, because building our own is the general case: the day a
# schedule asks for a tree PBRT has no equivalent of, PBRT has none to give.
bash $PREFIX/build_scene_dump.sh "$WORK/scene_dump"
# See render.sh: the film's D65 is reproduced from pbrt's construction rather
# than copied out of it, so it is checked against a running pbrt.
"$WORK/scene_dump" --check-tables
DUMP_FLAGS=()
if [[ -n "${PBRT_TREE:-}" ]]; then
  DUMP_FLAGS+=(--pbrt-tree)
fi
# --reference renders the gbuffer with pbrt's own camera, aggregate and BSDFs
# rather than running the pbrt binary and pulling channels out of its EXR. The
# binary can only produce those channels for a scene that asks for
# `Film "gbuffer"`, and no scene anyone else wrote does -- they say
# `Film "rgb"`, and pbrt has a command-line override for the sample count but
# none for the film. Verified against the binary on a scene that does ask for a
# gbuffer: the normals are identical once ours are rounded to the half floats
# the binary stores.
# Both sides are timed as the best of several runs. Every source of noise on a
# shared machine adds time and none removes it, so the minimum is the closest
# estimate of how long the work actually takes; a mean would be an estimate of
# how busy the machine was. REPEATS=1 to skip it.
REPEATS="${REPEATS:-5}"

# pbrt's side is timed inside scene_dump, rendering the same three channels
# with pbrt's own code, rather than by running the pbrt binary.
#
# The binary was the wrong thing to time and had been for a while. It renders
# whatever integrator the scene names -- none, in killeroo-simple, so pbrt's
# default of volpath -- into an `rgb` film, which computes no VisibleSurface and
# no reflectance. This renderer computes normals, a sixteen-sample reflectance
# and a random walk. Comparing those two says nothing about the schedule: it
# says volpath and randomwalk are different algorithms, and that one side was
# also filling in a gbuffer.
#
# What is timed now is pbrt's intersection, pbrt's rho and pbrt's
# RandomWalkIntegrator over the same samples of the same pixels -- the same
# work, so the difference is the thing being measured.
#
# Wavelength jitter is left on, and pixel jitter off, both by the options
# main() sets before parsing; the reference render reads pbrt's own options, so
# there is no second implementation of what those flags mean.
DUMP_OUT=$("$WORK/scene_dump" "${DUMP_FLAGS[@]}" --reference "$WORK/pbrt" \
    --repeats "$REPEATS" "$SCENE" "$WORK/scene.txt")
echo "$DUMP_OUT"
PBRT_SECONDS=$(echo "$DUMP_OUT" | sed -n 's/^scene_dump: reference seconds: //p')

# This renderer. The resolution comes from the scene along with everything
# else, so there is no longer a second place for it to disagree. It repeats
# inside one process, so there is no per-run startup to pay.
# --no-heap: a renderer's inner loop must not allocate, and nothing frees a
# heap allocation, so this is checked rather than hoped for. Three constant
# tables written where they were used once cost a fifth of the render and four
# gigabytes of resident memory before anybody noticed; this is what noticing
# looks like now.
# --ffp-contract: pbrt is built with gcc and no -ffp-contract flag, so it gets
# gcc's default of `fast` and its `a*b + c` rounds once. This app cannot agree
# with pbrt bit for bit while rounding twice. See include/SSA/Contract.h.
./build/compiler -p ssa --no-heap --ffp-contract \
    -i $PREFIX/render.bonsai -b cpp -o $PREFIX/render
"$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $PREFIX/render_hook.cpp \
    $PREFIX/render.o "${TBB_FLAGS[@]}" -o "$WORK/render.out"
BONSAI_OUT=$(BONSAI_REPEATS="$REPEATS" "$WORK/render.out" "$WORK/scene.txt" \
    "$WORK/bonsai.pfm")
echo "$BONSAI_OUT"
BONSAI_SECONDS=$(echo "$BONSAI_OUT" | sed -n 's/^render seconds: //p')

# The rendered image, both sides, as something to actually look at. The
# comparison below reports numbers; these are what to open when the numbers say
# something is wrong and it is not obvious what.
python3 $PREFIX/to_png.py "$WORK/pbrt-radiance.pfm" "$WORK/pbrt-radiance.png"
python3 $PREFIX/to_png.py "$WORK/bonsai-radiance.pfm" \
    "$WORK/bonsai-radiance.png"

python3 $PREFIX/compare_gbuffer.py "$WORK/pbrt.pfm" "$WORK/bonsai.pfm" \
    --albedo "$WORK/pbrt-albedo.pfm" "$WORK/bonsai-albedo.pfm" \
    --radiance "$WORK/pbrt-radiance.pfm" "$WORK/bonsai-radiance.pfm" \
    --pbrt-seconds "${PBRT_SECONDS:-0}" --bonsai-seconds "${BONSAI_SECONDS:-0}" \
    --repeats "$REPEATS"

rm -f $PREFIX/render.o "$WORK/render.out"
rm -rf "$WORK/render.out.dSYM"
