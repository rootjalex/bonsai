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

SRC_DIR=apps/queries/joins/dist2d
DIST_FLAGS="$DIST_DEFINE"

time ./build/compiler -i $SRC_DIR/joins.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p simplify -b cppx -o $SRC_DIR/joins_gen
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/joins_gen.cpp -I. $DIST_DEFINE
./$SRC_DIR/main.out
