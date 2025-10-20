#!/bin/bash

set -e

cmake --build build --config Release -j

SRC_DIR=apps/queries/joins/salary

# TODO: run python generate.py with script arguments
python $SRC_DIR/generate.py "$@" > $SRC_DIR/data.csv
time ./build/compiler -i $SRC_DIR/joins.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -b cppx -o $SRC_DIR/joins_gen
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/joins_gen.cpp -I.

./$SRC_DIR/main.out $SRC_DIR/data.csv

python $SRC_DIR/compare.py $SRC_DIR/data.csv

# rm $SRC_DIR/data.csv
