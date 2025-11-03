#!/bin/bash 

set -euo pipefail

# MUST be run from this directory
PREFIX="apps/cd/cpu/fcl"

if [[ "$(pwd)" == */apps/cd/cpu/fcl ]]; then
  cd ../../../..
fi

# build and compile bonsai
cmake --build build --config Release -j
time ./build/compiler -i $PREFIX/main.bonsai -p canonicalize -p lower-trees -p lower-sorts -p lower-externs -p lower-geometrics -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p lower-generic -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $PREFIX/cd_gen
# ./build/compiler -i $PREFIX/main.bonsai -o $PREFIX/main.bir
# ./build/compiler -i $PREFIX/main.bonsai -b llvm -o $PREFIX/main.ll
# ./build/compiler -i $PREFIX/main.bonsai -b cpp -o $PREFIX/main
cd apps/cd/cpu/fcl

# echo "Build with profiling"
# # build the main hook (requires fcl)
# rm -rf build
# cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_PROFILING=ON
# cmake --build build -j

# # run
# ./build/main hairball hairball_60_70_10
# ./build/main hairball dragon
# ./build/main dragon_60_70_10 hairball_60_70_10

# Now run with profiling
echo "Build without profiling"

# TODO: figure out how to programmatically enable profiling instead of
# cd_profiled.cpp getting stale possibly.
# rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

if [[ "$(uname)" == "Linux" ]]; then
    BIND="numactl --physcpubind 0-15"
    OBJ_DIR="/scratch/cpg/bonsai/apps/cd/data"
else
    echo "Warning: numactl only available on Linux; skipping CPU binding"
    BIND=""
    OBJ_DIR="/Users/ajroot/Downloads/xxx"
fi

# run
$BIND ./build/main $OBJ_DIR dragon_60_70_10 dragon
$BIND ./build/main $OBJ_DIR dragon_60_70_10 hairball_60_70_10
$BIND ./build/main $OBJ_DIR hairball dragon
$BIND ./build/main $OBJ_DIR hairball hairball_60_70_10

exit 0
