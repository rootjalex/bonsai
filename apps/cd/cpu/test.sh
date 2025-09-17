#!/bin/bash 

set -euo pipefail

APPLICATION="cd"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("pbrt")
OBJECTS=("cornell-box" "hairball" "sponza" "dragon")

if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../..
fi

for OBJECT_A in "${OBJECTS[@]}"; do
  for OBJECT_B in "${OBJECTS[@]}"; do
    if [[ "$OBJECT_A" == "$OBJECT_B" ]]; then
      continue
    fi
    for LAYOUT in "${LAYOUTS[@]}"; do
      # 1. build and compile bonsai
      cmake --build build --config Debug -j # > /dev/null
      # 2. lower to c++
      ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
      # 3. compile the lowered c++

      # ${PREFIX}/main.cpp
      clang++ -std=c++20 -O3 -g -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/${APPLICATION} -Iruntime/CPP 

      # 4. build the main hook (requires fcl)
      cd ${PREFIX} 
      mkdir -p build
      cd build
      cmake ..
      make -j
      
      # run
      ./${PREFIX}/${APPLICATION}_${LAYOUT}.out ${OBJECT_A} ${OBJECT_B}

      # clean up
      rm ${PREFIX}/${APPLICATION}.h
      rm ${PREFIX}/${APPLICATION}.cpp
      rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      rm -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
    done
  done
done

exit 0
