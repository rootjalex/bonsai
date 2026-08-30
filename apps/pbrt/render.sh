#!/bin/bash

set -euo pipefail

# Build and run PBRT milestone 0. See render.bonsai.
#
# Runs from either the repository root or this directory.
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi

PREFIX="apps/pbrt"
OUT="${1:-$PREFIX/pbrt-m0.ppm}"

cmake --build build -j

# `-p ssa` because bind is an SSA rewrite. The other two outputs are here to
# be looked at when something renders wrong: the .bir is what the schedule
# left behind, the .ll is what it became.
./build/compiler -p ssa -i $PREFIX/render.bonsai -o $PREFIX/render.bir
./build/compiler -p ssa -i $PREFIX/render.bonsai -b llvm -o $PREFIX/render.ll
./build/compiler -p ssa -i $PREFIX/render.bonsai -b cpp -o $PREFIX/render

# -I. so that the generated header can find the runtime it includes.
clang++ -g -std=c++20 -O3 -I. $PREFIX/render_hook.cpp $PREFIX/render.o \
    -o $PREFIX/render.out

time ./$PREFIX/render.out "$OUT"

# render.h is generated too, but it is checked in: regenerating it is how it
# stays current, so it is left where it was written.
rm $PREFIX/render.bir
rm $PREFIX/render.ll
rm $PREFIX/render.o
rm $PREFIX/render.out
# Only clang on Apple platforms leaves one of these behind.
rm -rf $PREFIX/render.out.dSYM

exit 0
