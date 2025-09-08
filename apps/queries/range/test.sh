#!/bin/bash

set -e

cmake --build build --config Debug -j
./build/compiler -i apps/queries/range/range.bonsai -p canonicalize -p lower-externs -b cppx -o apps/queries/range/range_gen
./build/compiler -i apps/queries/range/range_fast.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -b cppx -o apps/queries/range/range_fast_gen
clang++ -std=c++20 -O3 -g -o apps/queries/range/range_main.out apps/queries/range/range_main.cpp apps/queries/range/range_gen.cpp apps/queries/range/range_fast_gen.cpp -I.
./apps/queries/range/range_main.out
