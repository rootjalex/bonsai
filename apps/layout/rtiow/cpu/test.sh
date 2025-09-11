#!/bin/bash 

set -euo pipefail

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/layout/rtiow/cpu ]]; then
  cd ../../..
fi

PREFIX="apps/layout/rtiow/cpu"

# Compile
cmake --build build --config Debug -j
./build/compiler -i $PREFIX/main.bonsai -b cppx -o $PREFIX/rtiow
# clang++ -g -std=c++20 -O3 $PREFIX/main_hook.cpp $PREFIX/main.o -o $PREFIX/bonsai.out
# Run
# time ./$PREFIX/bonsai.out $PREFIX/rtiow-cpu-image.ppm
clang++ -std=c++20 -O3 -g -o $PREFIX/rtiow.out $PREFIX/rtiow_main.cpp $PREFIX/rtiow.cpp -Iruntime/CPP

# Clean up

exit 0
