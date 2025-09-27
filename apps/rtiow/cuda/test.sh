#!/bin/bash 

set -euo pipefail

APPLICATION="rtiow"
TARGET="cuda"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("pbrt")

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

for LAYOUT in "${LAYOUTS[@]}"; do
  echo "-- ${APPLICATION} - ${TARGET} - ${LAYOUT} --"
  # Compile
  rm -f ${PREFIX}/${APPLICATION}-${LAYOUT}.ppm
  cmake --build build --config Debug -j
  ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cuda -o ${PREFIX}/rtiow.h
  module load cuda
  nvcc -Iapps/rtiow -Iruntime/CUDA -O3 ${PREFIX}/main.cu -o ${PREFIX}/main
  # Run
  compute-sanitizer ./${PREFIX}/main ${PREFIX}/${APPLICATION}-${LAYOUT}.ppm

  # Clean up
  rm $PREFIX/rtiow.h
  rm $PREFIX/main
done


exit 0
