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

      # 3. build the main hook and final executable (requires fcl)
      cd ${PREFIX} 
      mkdir -p build
      cd build
      
      # Configure CMake with layout parameter
      cmake -DLAYOUT=${LAYOUT} ..
      
      # Build main library first
      make -j main_library
      
      # Go back to reconfigure CMake now that generated files exist
      cmake -DLAYOUT=${LAYOUT} -DAPPLICATION=${APPLICATION} ..
      
      # Build final executable
      make -j
      
      # run (executable is now in the build directory)
      cd ..       # build directory
      cd ../../.. # PREFIX
      ./${PREFIX}/build/${APPLICATION}_${LAYOUT}.out ${OBJECT_A} ${OBJECT_B}
      
      # clean up
      rm -f ${PREFIX}/${APPLICATION}.h
      rm -f ${PREFIX}/${APPLICATION}.cpp
      rm -rf ${PREFIX}/build
    done
  done
done

exit 0
