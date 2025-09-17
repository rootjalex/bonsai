#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("aosoa" "soa" "eq" "pbrt")
OBJECTS=("cornell-box" "dragon" "hairball" "sponza")
RAY_COUNT=65536 # (2^16)
RAY_FILE="RANDOM_GENERATED_RAYS"

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# Save a set of random rays.
clang++ -std=c++20 -O3 -o ${KERNEL_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/generate.cpp
./${KERNEL_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/${RAY_FILE}.txt ${RAY_COUNT}

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
    clang++ -std=c++20 -O3 -g -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/main.cpp ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/${APPLICATION} -Iruntime/CPP 
    # 4. Run it.
    ./${PREFIX}/${APPLICATION}_${LAYOUT}.out ${OBJECT} ${KERNEL_PATH}/${RAY_FILE}.txt
    # 5. Clean up
    rm ${PREFIX}/${APPLICATION}.h
    rm ${PREFIX}/${APPLICATION}.cpp
    rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
    rm -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
  done
  echo -e "\n"
done

rm ${KERNEL_PATH}/${RAY_FILE}.out
rm ${KERNEL_PATH}/${RAY_FILE}.txt

exit 0
