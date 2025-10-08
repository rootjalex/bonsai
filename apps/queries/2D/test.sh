#!/bin/bash

set -e

cmake --build build --config Debug -j
./build/compiler -i apps/queries/2D/queries.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -b cppx -o apps/queries/2D/queries_gen
clang++ -std=c++20 -O3 -g -o apps/queries/2D/main.out apps/queries/2D/main.cpp apps/queries/2D/queries_gen.cpp -I.
./apps/queries/2D/main.out
