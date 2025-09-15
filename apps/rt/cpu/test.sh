#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("aosoa" "soa" "eq" "pbrt")
OBJECTS=("env" "dragon")

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

for OBJECT in "${OBJECTS[@]}"; do
  echo "--- ${OBJECT} ---"
  for LAYOUT in "${LAYOUTS[@]}"; do
    echo "--- ${APPLICATION} - ${TARGET} - ${LAYOUT} ---"
    # 0. Remove any previously built image.
    rm -f ${PREFIX}/${APPLICATION}-${LAYOUT}.ppm
    # 1. Build the Bonsai compiler.
    cmake --build build --config Debug -j > /dev/null
    # 2. Lower to C++.
    ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
    # 3. Compile the lowered C++.
    clang++ -std=c++20 -O3 -g -fsanitize=address -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/main.cpp ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/${APPLICATION} -Iruntime/CPP 
    # 4. Run it.
    ./${PREFIX}/${APPLICATION}_${LAYOUT}.out ${OBJECT}
    # 5. Clean up
    rm ${PREFIX}/${APPLICATION}.h
    rm ${PREFIX}/${APPLICATION}.cpp
    rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
    rm -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
  done
  echo -e "\n"
done

exit 0
