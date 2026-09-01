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
"$WORK/scene_dump" "${DUMP_FLAGS[@]}" "$SCENE" "$WORK/scene.txt"

# Both sides are timed as the best of several runs. Every source of noise on a
# shared machine adds time and none removes it, so the minimum is the closest
# estimate of how long the work actually takes; a mean would be an estimate of
# how busy the machine was. REPEATS=1 to skip it.
REPEATS="${REPEATS:-5}"

# pbrt's reference. --disable-pixel-jitter puts the sample at the pixel centre,
# which is where this renderer puts its one sample.
#
# pbrt renders once per invocation, so repeating means running it again -- and
# paying for parsing and the BVH build each time, which is why its own reported
# time is what gets compared rather than the wall clock out here.
PBRT_SECONDS=""
for _ in $(seq 1 "$REPEATS"); do
  # Wavelength jitter is left on. Both sides now draw the wavelength from the
  # scene's sampler as its sample's first 1D value, and this renderer
  # reproduces pbrt's stream exactly -- see sampler.bonsai -- so the two draw
  # the same four wavelengths for the same pixel and sample. Turning the jitter
  # off would fix them both at u = 0.5 and so hide any disagreement about the
  # sampler, which is the thing worth checking.
  (cd "$WORK" && "$PBRT" --disable-pixel-jitter --quiet "$SCENE")
  # pbrt's own render time, which its progress reporter measures from after the
  # scene and the BVH are built.
  RUN=$("$IMGTOOL" info "$WORK/ref.exr" |
      sed -n 's/.*render time:.*(total \([0-9.]*\)s).*/\1/p')
  PBRT_SECONDS=$(awk -v best="$PBRT_SECONDS" -v now="$RUN" \
      'BEGIN { if (best == "" || now + 0 < best + 0) print now; else print best }')
done

# imgtool warns that the three channels are not R, G and B, which is the point:
# what is being extracted is a vector field, not a colour. Dropped rather than
# left to look like something went wrong, since it is expected every run.
"$IMGTOOL" convert --channels N.X,N.Y,N.Z --outfile "$WORK/pbrt.pfm" "$WORK/ref.exr" \
    2>&1 | grep -v "they are not R, G, and B" || true
# The gbuffer's other half: the reflectance of whatever the camera ray found,
# which is what the spectral pipeline has to reproduce.
"$IMGTOOL" convert --channels Albedo.R,Albedo.G,Albedo.B \
    --outfile "$WORK/pbrt-albedo.pfm" "$WORK/ref.exr" \
    2>&1 | grep -v "they are not R, G, and B" || true

# This renderer. The resolution comes from the scene along with everything
# else, so there is no longer a second place for it to disagree. It repeats
# inside one process, so there is no per-run startup to pay.
./build/compiler -p ssa -i $PREFIX/render.bonsai -b cpp -o $PREFIX/render
"$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $PREFIX/render_hook.cpp \
    $PREFIX/render.o -o "$WORK/render.out"
BONSAI_OUT=$(BONSAI_REPEATS="$REPEATS" "$WORK/render.out" "$WORK/scene.txt" \
    "$WORK/bonsai.pfm")
echo "$BONSAI_OUT"
BONSAI_SECONDS=$(echo "$BONSAI_OUT" | sed -n 's/^render seconds: //p')

python3 $PREFIX/compare_gbuffer.py "$WORK/pbrt.pfm" "$WORK/bonsai.pfm" \
    --albedo "$WORK/pbrt-albedo.pfm" "$WORK/bonsai-albedo.pfm" \
    --pbrt-seconds "${PBRT_SECONDS:-0}" --bonsai-seconds "${BONSAI_SECONDS:-0}" \
    --repeats "$REPEATS"

rm -f $PREFIX/render.o "$WORK/render.out"
rm -rf "$WORK/render.out.dSYM"
