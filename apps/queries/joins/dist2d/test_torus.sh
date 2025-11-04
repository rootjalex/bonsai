#!/bin/bash

set -e

cmake --build build --config Debug -j

SRC_DIR=apps/queries/joins/dist2d
DIST_FLAGS="$DIST_DEFINE"

time ./build/compiler -i $SRC_DIR/joins.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -b cppx -o $SRC_DIR/joins_gen
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/main_torus.out $SRC_DIR/main_torus.cpp $SRC_DIR/joins_gen.cpp -I. $DIST_DEFINE
if [[ "$(uname)" == "Linux" ]]; then
    numactl --physcpubind 0-15 ./$SRC_DIR/main_torus.out 1 &> $SRC_DIR/bonsai_torus1_results.txt
    numactl --physcpubind 0-15 ./$SRC_DIR/main_torus.out 100 &> $SRC_DIR/bonsai_torus100_results.txt
    numactl --physcpubind 0-15 ./$SRC_DIR/main_torus.out 1000 &> $SRC_DIR/bonsai_torus1000_results.txt
else
    ./$SRC_DIR/main_torus.out 1 &> $SRC_DIR/bonsai_torus1_results.txt
    ./$SRC_DIR/main_torus.out 100 &> $SRC_DIR/bonsai_torus100_results.txt
    ./$SRC_DIR/main_torus.out 1000 &> $SRC_DIR/bonsai_torus1000_results.txt
fi
