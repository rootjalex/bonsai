#!/bin/bash 

set -ex

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/wos ]]; then
  cd ../..
fi

PREFIX="apps/wos"

# Compile
cmake --build build --config Debug -j
./build/compiler -i $PREFIX/solve.bonsai -o $PREFIX/solve.bir
./build/compiler -i $PREFIX/solve.bonsai -b llvm -o $PREFIX/solve.ll
./build/compiler -i $PREFIX/solve.bonsai -b cpp -o $PREFIX/solve
# clang++ -g -std=c++20 -O3 $PREFIX/solve.cpp $PREFIX/solve.o -o $PREFIX/bonsai.out
# Run
# time ./$PREFIX/bonsai.out $PREFIX/rtiow-cpu-image.ppm

# Clean up
# rm $PREFIX/solve.bir
# rm $PREFIX/solve.ll
# rm $PREFIX/solve.o
# rm $PREFIX/bonsai.out
# rm -r $PREFIX/bonsai.out.dSYM

exit 0
