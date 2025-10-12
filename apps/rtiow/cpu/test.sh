#!/bin/bash 

set -euo pipefail

APPLICATION="rtiow"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("pbrt" "soaos" "ptr")

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

for LAYOUT in "${LAYOUTS[@]}"; do
  echo "-- ${APPLICATION} - ${TARGET} - ${LAYOUT} --"
  # 0. Remove any previously built image.
  rm -f ${PREFIX}/${APPLICATION}-${LAYOUT}.ppm
  # 1. Build the Bonsai compiler.
  cmake --build build --config Debug -j > /dev/null
  # 2. Lower to C++.
  ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
  # 3. Compile the lowered C++.
  clang++ -std=c++20 -O3 -march=native -g -fsanitize=address -Wl,-stack_size -Wl,0x10000000 -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/main.cpp ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/rtiow -Iruntime/CPP 
  # 4. Run it.
  ./${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/${APPLICATION}-${LAYOUT}.ppm
  # 5. Clean up
  rm ${PREFIX}/${APPLICATION}.h
  rm ${PREFIX}/${APPLICATION}.cpp
  rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
  rm -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
done

exit 0
