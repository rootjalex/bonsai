#!/bin/bash

set -e

cmake --build build --config Debug -j

SRC_DIR=apps/queries/joins/dist2d
# time ./build/compiler -i $SRC_DIR/joins.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -p lower-logical-operation -p cse -p dce 
time ./build/compiler -i $SRC_DIR/joins.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/joins_gen

time clang++ -std=c++20 -O3 -g -o $SRC_DIR/main_torus.out $SRC_DIR/main_torus.cpp $SRC_DIR/joins_gen.cpp -I.
# exit
if [[ "$(uname)" == "Linux" ]]; then
    BIND="numactl --physcpubind 0-15"
else
    BIND=""
fi

$BIND ./$SRC_DIR/main_torus.out 1 &> $SRC_DIR/torus1_results.txt
$BIND ./$SRC_DIR/main_torus.out 100 &> $SRC_DIR/torus100_results.txt
$BIND ./$SRC_DIR/main_torus.out 1000 &> $SRC_DIR/torus1000_results.txt
