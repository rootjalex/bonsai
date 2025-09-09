#!/bin/bash

set -e

# parse distribution flag
DIST="${1:-uniform}"  # default to "uniform" if no argument is provided

case "$DIST" in
    uniform)
        DIST_DEFINE="-DUSE_UNIFORM"
        ;;
    normal)
        DIST_DEFINE="-DUSE_NORMAL"
        ;;
    exponential)
        DIST_DEFINE="-DUSE_EXPONENTIAL"
        ;;
    lognormal)
        DIST_DEFINE="-DUSE_LOGNORMAL"
        ;;
    cauchy)
        DIST_DEFINE="-DUSE_CAUCHY"
        ;;
    weibull)
        DIST_DEFINE="-DUSE_WEIBULL"
        ;;
    *)
        echo "Unknown distribution: $DIST"
        echo "Supported: uniform, normal, exponential, lognormal, cauchy, weibull"
        exit 1
        ;;
esac

cmake --build build --config Debug -j
./build/compiler -i apps/queries/range/range.bonsai -p canonicalize -p lower-externs -b cppx -o apps/queries/range/range_gen
./build/compiler -i apps/queries/range/range_fast.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -b cppx -o apps/queries/range/range_fast_gen
clang++ -std=c++20 -O3 -g -o apps/queries/range/range_main.out apps/queries/range/range_main.cpp apps/queries/range/range_gen.cpp apps/queries/range/range_fast_gen.cpp -I. $DIST_DEFINE
./apps/queries/range/range_main.out
