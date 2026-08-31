#!/bin/bash

set -euo pipefail

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/rtiow/cpu ]]; then
  cd ../../..
fi

PREFIX="apps/rtiow/cpu"

# Compile
cmake --build build --config Debug -j
# -p ssa because the schedule's `split` and `bind` are SSA rewrites; see the
# flags header in main.bonsai.
./build/compiler -p ssa -i $PREFIX/main.bonsai -o $PREFIX/main.bir
./build/compiler -p ssa -i $PREFIX/main.bonsai -b llvm -o $PREFIX/main.ll
./build/compiler -p ssa -i $PREFIX/main.bonsai -b cpp -o $PREFIX/main
# -I. so that the generated header can find the runtime it includes.
clang++ -g -std=c++20 -O3 -I. $PREFIX/main_hook.cpp $PREFIX/main.o -o $PREFIX/bonsai.out
# Run
time ./$PREFIX/bonsai.out $PREFIX/rtiow-cpu-image.ppm

# Clean up. main.h is left where it was written, because main_hook.cpp includes
# it and the next build wants it there. It is generated rather than committed,
# so there is nothing to keep in step.
rm $PREFIX/main.bir
rm $PREFIX/main.ll
rm $PREFIX/main.o
rm $PREFIX/bonsai.out
# Only clang on Apple platforms leaves one of these behind.
rm -rf $PREFIX/bonsai.out.dSYM

exit 0
