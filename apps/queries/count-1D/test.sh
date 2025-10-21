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

SRC_DIR=apps/queries/count-1D
DIST_FLAGS="$DIST_DEFINE"

# echo "Compiling main"
# time clang++ -std=c++20 -O3 -g -c $SRC_DIR/count1d_main.cpp -I. $DIST_FLAGS -o $SRC_DIR/count1d_main.o
# echo "Compiling linear scan"
# time clang++ -std=c++20 -O3 -g -c $SRC_DIR/count1d_gen.cpp -I. $DIST_FLAGS -o $SRC_DIR/count1d_gen.o
# echo "Compiling tree traversal"
# time clang++ -std=c++20 -O3 -g -c $SRC_DIR/count1d_fast_gen.cpp -I. $DIST_FLAGS -o $SRC_DIR/count1d_fast_gen.o
# echo "Compiling tree traversal + count aug"
# time clang++ -std=c++20 -O3 -g -c $SRC_DIR/count1d_fast_aug_gen.cpp -I. $DIST_FLAGS -DUSE_AUG=1 -o $SRC_DIR/count1d_fast_aug_gen.o

# echo "Linking (no aug)"
# # Link for non-augmented main
# time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_main.out \
#     $SRC_DIR/count1d_main.o \
#     $SRC_DIR/count1d_gen.o \
#     $SRC_DIR/count1d_fast_gen.o

# echo "Linking (aug)"
# # Link for augmented main
# time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_aug_main.out \
#     $SRC_DIR/count1d_main.o \
#     $SRC_DIR/count1d_gen.o \
#     $SRC_DIR/count1d_fast_aug_gen.o \
#     -DUSE_AUG=1

time ./build/compiler -i $SRC_DIR/count1d.bonsai -p canonicalize -p lower-externs -b cppx -o $SRC_DIR/count1d_gen
time ./build/compiler -i $SRC_DIR/count1d_fast.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields  -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/count1d_fast_gen
time ./build/compiler -i $SRC_DIR/count1d_fast_aug.bonsai -p canonicalize -p lower-trees -p lower-externs -p lower-dynamic-sets -p lower-scans -p lower-layouts -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields  -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/count1d_fast_aug_gen
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_main.out $SRC_DIR/count1d_main.cpp $SRC_DIR/count1d_gen.cpp $SRC_DIR/count1d_fast_gen.cpp -I. $DIST_DEFINE
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_aug_main.out $SRC_DIR/count1d_main.cpp $SRC_DIR/count1d_gen.cpp $SRC_DIR/count1d_fast_aug_gen.cpp -I. $DIST_DEFINE -DUSE_AUG=1
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_main_ablation.out $SRC_DIR/count1d_main.cpp $SRC_DIR/count1d_gen.cpp $SRC_DIR/count1d_fast_gen.cpp -I. $DIST_DEFINE -DPROFILE_UNFUSED
time clang++ -std=c++20 -O3 -g -o $SRC_DIR/count1d_main_ablation2.out $SRC_DIR/count1d_main.cpp $SRC_DIR/count1d_fast_aug_gen.cpp -I. $DIST_DEFINE -DPROFILE_UNFUSED -DUSE_AUG=1

if [[ "$(uname)" == "Darwin" ]]; then
    sudo purge               # clear caches (optional)
    killall -STOP Spotlight  # pause indexing
fi

if [[ "$(uname)" == "Linux" ]]; then
    numactl --physcpubind 0-15 ./$SRC_DIR/count1d_main.out
else
    ./$SRC_DIR/count1d_main.out
fi

if [[ "$(uname)" == "Darwin" ]]; then
    sudo purge               # clear caches (optional)
    killall -STOP Spotlight  # pause indexing
fi

if [[ "$(uname)" == "Linux" ]]; then
    numactl --physcpubind 0-15 ./$SRC_DIR/count1d_aug_main.out
else
    ./$SRC_DIR/count1d_aug_main.out
fi

if [[ "$(uname)" == "Darwin" ]]; then
    sudo purge               # clear caches (optional)
    killall -STOP Spotlight  # pause indexing
fi

# if [[ "$(uname)" == "Linux" ]]; then
#     numactl --physcpubind 0-15 ./$SRC_DIR/count1d_main_ablation.out
# else
#     ./$SRC_DIR/count1d_main_ablation.out
# fi

if [[ "$(uname)" == "Darwin" ]]; then
    sudo purge               # clear caches (optional)
    killall -STOP Spotlight  # pause indexing
fi

if [[ "$(uname)" == "Linux" ]]; then
   numactl --physcpubind 0-15 ./$SRC_DIR/count1d_main_ablation2.out
else
    ./$SRC_DIR/count1d_main_ablation2.out
fi
