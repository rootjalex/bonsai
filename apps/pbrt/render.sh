#!/bin/bash

set -euo pipefail

# Build and run PBRT milestone 0. See render.bonsai.
#
# Runs from either the repository root or this directory.
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi

PREFIX="apps/pbrt"
SCENE="${1:-$PREFIX/scenes/three-spheres.pbrt}"
# The renderer writes linear float, as pbrt does; the PNG beside it is the
# post-processing step, and is what to actually look at.
OUT="${2:-$PREFIX/pbrt.pfm}"
PNG="${OUT%.pfm}.png"

cmake --build build -j

# The scene comes from a .pbrt file, read by PBRT's own parser. Needs a built
# pbrt; see build_scene_dump.sh.
bash $PREFIX/build_scene_dump.sh
# The spectral tables are generated from pbrt's source, and one of them -- the
# film's D65 -- is reproduced by following what pbrt does rather than copied.
# Ask a running pbrt whether the result is right before rendering with it.
./$PREFIX/scene_dump --check-tables
./$PREFIX/scene_dump "$SCENE" "$PREFIX/scene.txt"

# The spectral data is generated (see make_spectrum_tables.py) and the fit that
# uses it is a port, so check the round trip before rendering with it: a fit
# that is subtly wrong still produces plausible numbers, just the wrong colour.
clang++ -std=c++20 -O2 -I$PREFIX $PREFIX/rgb2spec_check.cpp -o $PREFIX/rgb2spec_check
./$PREFIX/rgb2spec_check
rm $PREFIX/rgb2spec_check

# `-p ssa` because bind is an SSA rewrite. The other two outputs are here to
# be looked at when something renders wrong: the .bir is what the schedule
# left behind, the .ll is what it became.
./build/compiler -p ssa -i $PREFIX/render.bonsai -o $PREFIX/render.bir
./build/compiler -p ssa -i $PREFIX/render.bonsai -b llvm -o $PREFIX/render.ll
./build/compiler -p ssa -i $PREFIX/render.bonsai -b cpp -o $PREFIX/render

# -I. so that the generated header can find the runtime it includes.
clang++ -g -std=c++20 -O3 -I. -I$PREFIX $PREFIX/render_hook.cpp $PREFIX/render.o \
    -o $PREFIX/render.out

./$PREFIX/render.out "$PREFIX/scene.txt" "$OUT"

# The image holds normals rather than radiance, so the encoding is the remap
# that makes a direction visible rather than pbrt's sRGB curve.
python3 $PREFIX/to_png.py "$OUT" "$PNG" --normals

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
