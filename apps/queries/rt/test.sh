#!/bin/bash 

set -euo pipefail

cmake --build build --config Release -j

SRC_DIR="apps/queries/rt"


time ./build/compiler -i $SRC_DIR/rt.bonsai -p canonicalize -p lower-trees -p lower-sorts -p lower-externs -p lower-geometrics -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p lower-generic -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/rt
# -p cse -b cppx -o $SRC_DIR/rt
# exit

if [[ "$(uname)" == "Darwin" ]]; then
    EIGEN_INCLUDE="-I/opt/homebrew/include/eigen3"
else
    EIGEN_INCLUDE="-I/usr/include/eigen3"
fi

INCLUDES="-I. -I$SRC_DIR -Ideps/fcpw/include $EIGEN_INCLUDE -I/Users/ajroot/projects/cgal/install/include -I/opt/homebrew/opt/boost/include"

# COMPILE_FLAGS="-g -O0 -fsanitize=address,undefined"
COMPILE_FLAGS="-O3 -march=native"

time clang++ -std=c++20 $COMPILE_FLAGS $INCLUDES -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/rt.cpp

./$SRC_DIR/main.out

# time clang++ -std=c++20 -O3 -march=native -I$SRC_DIR -I. -Ideps/fcpw/include $EIGEN_INCLUDE -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/rt3.cpp -DAJR_PROFILE

# ./$SRC_DIR/main.out