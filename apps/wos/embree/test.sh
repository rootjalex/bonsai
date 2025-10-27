#!/bin/bash 

set -euo pipefail

APPLICATION="wos"
TARGET="embree"
KERNEL_PATH="apps/wos"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="apps/rt/layouts" # (just share layouts)

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


N="${1:-9}"
N_QUERIES="${2:-10000}"
OBJECTS=("lucy" "white-oak" "sheep" "san-miguel-x35-y22-z47" "hairball" "sponza" "power-plant")

# only run on performance cores for the Fredwood.
# TODO(cgyurgyik): this was causing performance regressions.
FREDWOOD_FLAG="" # "numactl --physcpubind 0-15" 

if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${MIN_POWER} only ***"
  N=1
  N_QUERIES=1
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

  
  if [[ "${BVH_SUFFIX}" == "2" ]]; then
    LAYOUTS=("pbrt-align16" "pbrt" "pbrt-q16" "sg-eq")
  else
    LAYOUTS=("bvh8-q8-ci" "bvh8" "bvh8-q16-ci")
  fi
  if [[ -n "${SPECIFIC_LAYOUT}" ]]; then
    # (debug mode) test a single layout
    LAYOUTS=("${SPECIFIC_LAYOUT}")
  fi
  echo "-- with layouts: ${LAYOUTS[@]}"

  # replace `$N$` with BVH_SUFFIX.
  MAIN_FILE="main"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" ${PREFIX}/${MAIN_FILE}.cpp > ${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  # insert the canonical tree functions (we do it in this hacky way since they're shared between CPU / GPU.
  # a better approach might be using macros, similar to PBRT).
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cpp
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cpp
  fi

  for OBJECT in "${OBJECTS[@]}"; do
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "--- ${OBJECT} - ${LAYOUT} ---"
      echo "--- ${OBJECT} - ${LAYOUT} - ${N_QUERIES} ---" >> ${PREFIX}/results/${LAYOUT}.txt

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
      mkdir -p build && cd build
      BUILD_DIRECTORY="build_${BVH_SUFFIX}_${LAYOUT}"  # unique per {branching factor, layout}.
      mkdir -p ${BUILD_DIRECTORY}
      cd ${BUILD_DIRECTORY}
      
      CLANG_FLAG=""
      if [[ "$(uname)" == "Linux" ]]; then
        CLANG_FLAG="-DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/clang++"
      fi
      cmake -DLAYOUT=${LAYOUT} -DAPPLICATION=${APPLICATION} -DBVH_SUFFIX=${BVH_SUFFIX} ../.. ${CLANG_FLAG} > /dev/null
      
      make -j # > /dev/null 2>&1
      
      cd ../..    # back to PREFIX
      cd ../../.. # back to root
      
      for ((k=0; k < N; k++)); do
        ./${PREFIX}/build/${BUILD_DIRECTORY}/${APPLICATION}_${LAYOUT}.out "${OBJECT}" ${N_QUERIES} | tee -a ${PREFIX}/results/${LAYOUT}.txt
      done
      
      rm -f ${PREFIX}/${APPLICATION}.h
      rm -f ${PREFIX}/${APPLICATION}.cpp
    done
  done

  rm ${PREFIX}/${MAIN_FILE}.cpp
}

run_tests "8"
run_tests "2"

exit 0