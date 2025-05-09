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
./build/compiler -i $PREFIX/main.bonsai -o $PREFIX/main.bir
./build/compiler -i $PREFIX/main.bonsai -b llvm -o $PREFIX/main.ll
./build/compiler -i $PREFIX/main.bonsai -b cpp -o $PREFIX/main
clang++ -g -std=c++20 -O3 $PREFIX/main_hook.cpp $PREFIX/main.o -o $PREFIX/bonsai.out
# Run
time ./$PREFIX/bonsai.out $PREFIX/rtiow-cpu-image.ppm

# Clean up
rm $PREFIX/main.bir
rm $PREFIX/main.ll
rm $PREFIX/main.o
rm $PREFIX/bonsai.out
rm -r $PREFIX/bonsai.out.dSYM

exit 0
