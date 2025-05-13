#!/bin/bash 

set -euo pipefail

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/rtiow/cuda ]]; then
  cd ../../..
fi

PREFIX="apps/rtiow/cuda"

# Compile
cmake --build build --config Debug -j
./build/compiler -i $PREFIX/rtiow.bonsai -b cuda -o $PREFIX/rtiow.h
module load cuda
nvcc -Iruntime/CUDA -O3 $PREFIX/main.cu -o $PREFIX/main

# Run
time ./$PREFIX/main $PREFIX/rtiow-cuda-image.ppm

# Clean up
rm $PREFIX/rtiow.h
rm $PREFIX/main

exit 0
