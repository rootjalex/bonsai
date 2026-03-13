#!/bin/bash

set -e

cmake --build build --config Debug -j

SRC_DIR=apps/queries/joins/dist2d

time ./build/compiler -i $SRC_DIR/torus.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/torus_gen

time clang++ -std=c++20 -O3 -g -o $SRC_DIR/main_torus.out $SRC_DIR/main_torus.cpp $SRC_DIR/torus_gen.cpp -I.

if [[ "$(uname)" == "Linux" ]]; then
    BIND="numactl --physcpubind 0-15"
else
    BIND=""
fi

$BIND ./$SRC_DIR/main_torus.out 1 &> $SRC_DIR/torus1_results.txt
# $BIND ./$SRC_DIR/main_torus.out 5 &> $SRC_DIR/torus5_results.txt
$BIND ./$SRC_DIR/main_torus.out 100 &> $SRC_DIR/torus100_results.txt
#$BIND ./$SRC_DIR/main_torus.out 1000 &> $SRC_DIR/torus1000_results.txt
$BIND ./$SRC_DIR/main_torus.out 100000 &> $SRC_DIR/torus100000_results.txt
