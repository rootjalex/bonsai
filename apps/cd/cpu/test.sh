#!/bin/bash 

set -euo pipefail

APPLICATION="cd"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("pbrt-align32")
N="${1:-10}"
# *NOTE* we are using the FCL OBJ loader for this benchmark, and it doesn't seem to
#  work with all OBJ files. Therefore, we only choose a select few.
OBJECTS_A=("hairball" "hairball" "suzanne")
OBJECTS_B=("bunny" "teapot" "hairball")

if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../..
fi

rm -rf ${PREFIX}/results
mkdir ${PREFIX}/results

for ((i=0; i<${#OBJECTS_A[@]}; i++)); do
  OBJECT_A="${OBJECTS_A[i]}"
  OBJECT_B="${OBJECTS_B[i]}"
  for LAYOUT in "${LAYOUTS[@]}"; do
    echo "--- ${OBJECT_A} - ${OBJECT_B} - ${LAYOUT} ---"
    echo "--- ${OBJECT_A} - ${OBJECT_B} - ${LAYOUT} ---" >> ${PREFIX}/results/${LAYOUT}.txt
    # 1. build and compile bonsai
    cmake --build build --config Debug -j > /dev/null
    
    # 2. lower to c++
    ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}

    # 3. build the main hook and final executable (requires fcl)
    cd ${PREFIX} 
    mkdir -p build
    cd build
    
    # Configure CMake with layout parameter
    export LDFLAGS="-Wl,-no_warn_duplicate_libraries"
    cmake -DLAYOUT=${LAYOUT} .. > /dev/null
    
    # Build main library first
    make -j main_library > /dev/null
    
    # Go back to reconfigure CMake now that generated files exist
    cmake -DLAYOUT=${LAYOUT} -DAPPLICATION=${APPLICATION} .. > /dev/null
    
    # Build final executable
    make -j > /dev/null 2>&1
    
    # run (executable is now in the build directory)
    cd ..       # build directory
    cd ../../.. # PREFIX
    for ((k=0; k < N; k++)); do
      ./${PREFIX}/build/${APPLICATION}_${LAYOUT}.out ${OBJECT_A} ${OBJECT_B} >> ${PREFIX}/results/${LAYOUT}.txt
    done
    
    # clean up
    rm -f ${PREFIX}/${APPLICATION}.h
    rm -f ${PREFIX}/${APPLICATION}.cpp
    rm -rf ${PREFIX}/build
  done
done

for LAYOUT in "${LAYOUTS[@]}"; do
  python3.11 ${PREFIX}/collect.py ${LAYOUT} ${PREFIX}/results/${LAYOUT}.txt >> ${PREFIX}/results/aggregate.txt
done

exit 0
