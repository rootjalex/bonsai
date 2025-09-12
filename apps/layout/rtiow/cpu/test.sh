#!/bin/bash 

set -euo pipefail

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/layout/rtiow/cpu ]]; then
  cd ../../..
fi

PREFIX="apps/layout/rtiow/cpu"

# 1. Build the Bonsai compiler.
cmake --build build --config Debug -j
# 2. Lower to C++.
./build/compiler -i $PREFIX/main.bonsai -b cppx -o $PREFIX/rtiow
# 3. Compile the lowered C++.
clang++ -std=c++20 -O3 -g -o $PREFIX/rtiow.out $PREFIX/rtiow_main.cpp $PREFIX/rtiow.cpp -Iruntime/CPP -I.
# 4. Run it.
time ./$PREFIX/rtiow.out $PREFIX/rtiow-cpu-image.ppm

# Clean up

exit 0
