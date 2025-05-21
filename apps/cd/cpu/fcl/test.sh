#!/bin/bash 

set -euo pipefail

# MUST be run from this directory
PREFIX="apps/cd/cpu/fcl"

if [[ "$(pwd)" == */apps/cd/cpu/fcl ]]; then
  cd ../../../..
fi

# build and compile bonsai
cmake --build build --config Debug -j
./build/compiler -i $PREFIX/main.bonsai -o $PREFIX/main.bir
./build/compiler -i $PREFIX/main.bonsai -b llvm -o $PREFIX/main.ll
./build/compiler -i $PREFIX/main.bonsai -b cpp -o $PREFIX/main
cd apps/cd/cpu/fcl

# build the main hook (requires fcl)
mkdir -p build
cd build
cmake ..

# compile and run
make -j

# run
./main

# clean up
rm ../main.bir
rm ../main.ll
rm ../main.o

exit 0
