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
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
WORK="${WORK:-$PREFIX/compare-out}"

if [[ ! -x "$PBRT" ]]; then
  echo "no pbrt at $PBRT -- set PBRT to a built one" >&2
  exit 1
fi

mkdir -p "$WORK"

cmake --build build -j

# pbrt's reference. --disable-pixel-jitter puts the sample at the pixel centre,
# which is where this renderer puts its one sample.
(cd "$WORK" && "$PBRT" --disable-pixel-jitter --quiet "$OLDPWD/$PREFIX/scene.pbrt")
# imgtool warns that the three channels are not R, G and B, which is the point:
# what is being extracted is a vector field, not a colour. Dropped rather than
# left to look like something went wrong, since it is expected every run.
"$IMGTOOL" convert --channels N.X,N.Y,N.Z --outfile "$WORK/pbrt.pfm" "$WORK/ref.exr" \
    2>&1 | grep -v "they are not R, G, and B" || true

# pbrt's own render time, which its progress reporter measures from after the
# scene and the BVH are built. Taken from the EXR rather than timed from
# outside so that parsing the scene and building the tree stay out of it.
PBRT_SECONDS=$("$IMGTOOL" info "$WORK/ref.exr" |
    sed -n 's/.*render time:.*(total \([0-9.]*\)s).*/\1/p')

# This renderer.
./build/compiler -p ssa -i $PREFIX/render.bonsai -b cpp -o $PREFIX/render
clang++ -g -std=c++20 -O3 -I. $PREFIX/render_hook.cpp $PREFIX/render.o \
    -o "$WORK/render.out"
# Must match the resolution in scene.pbrt. compare_normals.py refuses to
# compare images of different sizes, so a drift here is caught rather than
# quietly compared against nothing.
BONSAI_OUT=$(PBRT_WIDTH=1600 PBRT_HEIGHT=900 "$WORK/render.out" "$WORK/bonsai.pfm")
echo "$BONSAI_OUT"
BONSAI_SECONDS=$(echo "$BONSAI_OUT" | sed -n 's/^render seconds: //p')

python3 $PREFIX/compare_normals.py "$WORK/pbrt.pfm" "$WORK/bonsai.pfm" \
    --pbrt-seconds "${PBRT_SECONDS:-0}" --bonsai-seconds "${BONSAI_SECONDS:-0}"

rm -f $PREFIX/render.o "$WORK/render.out"
rm -rf "$WORK/render.out.dSYM"
