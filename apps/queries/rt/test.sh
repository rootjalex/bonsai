#!/bin/bash 

set -euo pipefail

cmake --build build --config Release -j

SRC_DIR="apps/queries/rt"


time ./build/compiler -i $SRC_DIR/rt.bonsai -p canonicalize -p lower-trees -p lower-sorts -p lower-externs -p lower-geometrics -p lower-dynamic-sets -p lower-scans -p lower-layouts -p loop-transforms -p lower-recloops -p lower-foreachs -p unswitch -p lower-yields -p lower-generic -p simplify -p lower-logical-operation -p cse -p dce -b cppx -o $SRC_DIR/rt_gen
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

time clang++ -std=c++20 $COMPILE_FLAGS $INCLUDES -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/rt_gen.cpp
# time clang++ -std=c++20 -S -emit-llvm $COMPILE_FLAGS $INCLUDES -o $SRC_DIR/rt.ll $SRC_DIR/rt_gen.cpp
echo "white_oak = ["
./$SRC_DIR/main.out /Users/ajroot/projects/pldi-bonsai/apps/queries/rt/ white-oak /Users/ajroot/projects/pldi-bonsai/apps/queries/rt/rays camera 17 19
echo "]"
# time clang++ -std=c++20 $COMPILE_FLAGS $INCLUDES -o $SRC_DIR/main.out $SRC_DIR/main.cpp $SRC_DIR/rt.cpp -DAJR_PROFILE

# ./$SRC_DIR/main.out
