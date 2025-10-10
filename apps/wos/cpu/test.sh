#!/bin/bash 

set -euo pipefail

APPLICATION="wos"
TARGET="cpu"
KERNEL_PATH="apps/wos"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="${KERNEL_PATH}/layouts"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      break
      ;;
    *)
      shift
      ;;
  esac
done


N="${1:-3}"
NUM_QUERIES="${2:-100}"

OBJECTS=("white-oak")
# OBJECTS=("lucy" "sheep" "san-miguel-x35-y22-z47" "hairball" "white-oak" "sponza") # "power-plant"

DATA_PATH=${PREFIX}/results
DATA_FILE="data"

# only run on performance cores for the Fredwood.
FREDWOOD_FLAG="numactl --physcpubind 0-15" 

if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${MIN_POWER} only ***"
  MAX_POWER=${MIN_POWER}
  N=1
fi

if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# Delete previous data
rm -rf ${PREFIX}/results
mkdir ${PREFIX}/results

run_tests() {
  local BVH_SUFFIX="$1"
  local SPECIFIC_LAYOUT="${2:-}" # optional: specific layout to test

  LAYOUTS=()
  if [[ -n "${SPECIFIC_LAYOUT}" ]]; then
    # (debug mode) test a single layout
    LAYOUTS=("${SPECIFIC_LAYOUT}")
  else
    # test *all* layouts
    for file in "${LAYOUT_PATH}/${BVH_SUFFIX}"/*.bonsai; do
      NAME=$(basename "$file" .bonsai)
      LAYOUTS+=("${NAME}")
    done
  fi
  echo "-- with layouts: ${LAYOUTS[@]}"

  for OBJECT in "${OBJECTS[@]}"; do
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "--- ${OBJECT} - ${LAYOUT} ---"
      echo "--- ${OBJECT} - ${LAYOUT} ---" >> ${PREFIX}/results/${LAYOUT}.txt

      # 0. Combine the layout and schedule into a single file.
      LAYOUT_FILE=$(mktemp).bonsai
      cat ${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai > ${LAYOUT_FILE}
      cat ${PREFIX}/schedule.bonsai >> ${LAYOUT_FILE}
      echo "}" >> "${LAYOUT_FILE}"
      
      # 1. build and compile bonsai
      cmake --build build --config Debug -j # > /dev/null
      
      # 2. lower to c++
      ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${LAYOUT_FILE} -b cppx -o ${PREFIX}/${APPLICATION}

      # 3. build the main hook and final executable
      cd ${PREFIX}
      mkdir -p build
      cd build
      
      # Configure CMake with layout parameter
      export LDFLAGS="-Wl,-no_warn_duplicate_libraries"
      cmake -DLAYOUT=${LAYOUT} -DAPPLICATION=${APPLICATION} .. # > /dev/null
      
      # Build the executable
      make -j # > /dev/null 2>&1
      
      cd ..       # back to PREFIX
      cd ../../.. # back to root
      
      # run (executable is now in the build directory)
      for ((k=0; k < N; k++)); do
        ./${PREFIX}/build/${APPLICATION}_${LAYOUT}.out "${OBJ}" ${NUM_QUERIES} >> ${PREFIX}/results/${LAYOUT}.txt
      done
      
      # clean up
      rm -f ${PREFIX}/${APPLICATION}.h
      rm -f ${PREFIX}/${APPLICATION}.cpp
      rm -rf ${PREFIX}/build
    done
  done
}

run_tests "2"

exit 0