#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="${KERNEL_PATH}/layouts"

OBJECTS=("hairball" "white-oak" "power-plant" "sponza" "sheep") # san-miguel-x35-y22-z47

# Parse flags
DRY_RUN=false
DEBUG_MODE=false
DEBUG_LAYOUT=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      POSITIONAL_ARGS=()
      break
      ;;
    --debug)
      DEBUG_MODE=true
      DEBUG_LAYOUT="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

N="${1:-9}" # drop lowest 2 and highest 2 runs in processing
SCHEDULE="${2:-parallel}" # or single-thread
HIT_RATIO="${3:-75}" # n%, e.g., 75% is the default
TYPE="${4:-COMPARISON}" # other option, PERFORMANCE
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"
PARTITION="sah"

MIN_POWER=10
MAX_POWER=20

# only run on performance cores for the Fredwood.
FREDWOOD_FLAG="numactl --physcpubind 0-15" 

# Override for dry run.
if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${MIN_POWER} only ***"
  MAX_POWER=${MIN_POWER}
  N=1
fi

# Override for debug mode
if [[ "${DEBUG_MODE}" == true ]]; then
  echo "*** DEBUG MODE: testing layout ${DEBUG_LAYOUT} only ***"
  MAX_POWER=${MIN_POWER}
  N=2
  OBJECTS=("${OBJECTS[0]}")
  
  # Find which folder contains this layout
  DEBUG_BVH_SUFFIX=""
  for folder in "${LAYOUT_PATH}"/*; do
    if [[ -d "$folder" && -f "${folder}/${DEBUG_LAYOUT}.bonsai" ]]; then
      DEBUG_BVH_SUFFIX=$(basename "$folder")
      break
    fi
  done
  
  if [[ -z "${DEBUG_BVH_SUFFIX}" ]]; then
    echo "ERROR: Layout '${DEBUG_LAYOUT}' not found in ${LAYOUT_PATH}"
    exit 1
  fi
  echo "found ${DEBUG_LAYOUT} in ${LAYOUT_PATH}/${DEBUG_BVH_SUFFIX}/"
fi

RAY_COUNTS=()
for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
done

# Argument for ray counts passed to main.
ARGV="${#RAY_COUNTS[@]}"
for COUNT in "${RAY_COUNTS[@]}"; do
    ARGV="${ARGV} ${COUNT}"
done

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# For saving a set of random rays.
clang++ -std=c++20 -O3 -march=native -o ${RAY_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/generate.cpp

for RAY_COUNT in "${RAY_COUNTS[@]}"; do
  echo ${RAY_COUNT} >> ${DATA_PATH}/${DATA_FILE}.txt
  for OBJECT in "${OBJECTS[@]}"; do
    if [ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays" ]; then
      echo "no rays found for ${OBJECT} with count ${RAY_COUNT} and ratio ${HIT_RATIO}; generating now..."
      FLAG=""
      if [[ "$(uname)" == "Linux" ]]; then
        FLAG="${FLAG} ${FREDWOOD_FLAG}"  
      fi
      ${FLAG} ./${RAY_PATH}/${RAY_FILE}.out ${OBJECT} ${RAY_PATH} ${RAY_COUNT} 0.${HIT_RATIO}
      echo "...${RAY_COUNT} rays generated for ${OBJECT} with hit ratio: 0.${HIT_RATIO}"
    fi
  done
done

# Delete previous data.
rm -f -r ${DATA_PATH}
mkdir ${DATA_PATH}

# Install python dependencies for data processing.
pip install -r ${KERNEL_PATH}/requirements.txt

if [[ "$(uname)" == "Linux" ]]; then
  echo "Running on Linux (presumably Redwood)!"
fi

echo "runs: ${N}, schedule: ${SCHEDULE}"
> ${DATA_PATH}/${DATA_FILE}.txt # clear

# Function to run tests for a given main file and layouts
run_tests() {
  local BVH_SUFFIX="$1"
  local SPECIFIC_LAYOUT="${2:-}" # optional: specific layout to test
  
  LAYOUTS=()
  if [[ -n "${SPECIFIC_LAYOUT}" ]]; then
    # (debug mode) test a single layout
    LAYOUTS=("${SPECIFIC_LAYOUT}")
  else
    # test *all* layouts in the folder
    for file in "${LAYOUT_PATH}/${BVH_SUFFIX}"/*.bonsai; do
      NAME=$(basename "$file" .bonsai)
      LAYOUTS+=("${NAME}")
    done
  fi
  echo "-- with layouts: ${LAYOUTS[@]}"
  
  # replace `$N$` with BVH_SUFFIX.
  MAIN_FILE="main_trace"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" ${PREFIX}/${MAIN_FILE}.cpp > ${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  # insert the canonical tree functions (we do it in this hacky way since they're shared between CPU / GPU.
  # a better approach might be using macros, similar to PBRT).
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${KERNEL_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cpp
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${KERNEL_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cpp
  fi

  for OBJECT in "${OBJECTS[@]}"; do
    echo "object: ${OBJECT}" 
    echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT} (${MAIN_FILE})"
      echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
      # 0. Combine the layout and schedule into a single file.
      LAYOUT_FILE=$(mktemp).bonsai
      cat ${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai > ${LAYOUT_FILE}
      cat ${PREFIX}/schedule.bonsai >> ${LAYOUT_FILE}
      echo "}" >> "${LAYOUT_FILE}"

      # 1. Build the Bonsai compiler.
      cmake --build build --config Debug -j > /dev/null
      # 2. Lower to C++.
      ./build/compiler -i ${PREFIX}/main.bonsai -l ${LAYOUT_FILE} -b cppx -o ${PREFIX}/${APPLICATION}
      # 3. Compile the lowered C++.
      COMMON_FLAGS="-std=c++20 -O3 -march=native -I. -Iapps/${APPLICATION} -Iruntime/CPP"
      if [[ "${SCHEDULE}" == "parallel" ]]; then
        COMMON_FLAGS="-fopenmp ${COMMON_FLAGS}"
      fi
      # if [[ "${DEBUG_MODE}" == true ]]; then
      #   COMMON_FLAGS="-fsanitize=address -fno-omit-frame-pointer ${COMMON_FLAGS}"
      # fi
      
      if [[ "${DEBUG_MODE}" == true ]]; then
        # Generate LLVM IR for combined sources
        clang++ ${COMMON_FLAGS} -S -emit-llvm ${PREFIX}/${MAIN_FILE}.cpp ${PREFIX}/${APPLICATION}.cpp
        MAIN_LL="${MAIN_FILE}.ll"
        cat ${MAIN_LL} ${APPLICATION}.ll > ${DATA_PATH}/${APPLICATION}_${LAYOUT}.ll
        rm ${MAIN_LL}
        rm ${APPLICATION}.ll
        
        # Generate assembly for combined sources
        clang++ ${COMMON_FLAGS} -S ${PREFIX}/${MAIN_FILE}.cpp ${PREFIX}/${APPLICATION}.cpp
        MAIN_S="${MAIN_FILE}.s"
        cat ${MAIN_S} ${APPLICATION}.s > ${DATA_PATH}/${APPLICATION}_${LAYOUT}.asm
        rm ${MAIN_S}
        rm ${APPLICATION}.s
      fi 
      
      # Compile executable
      COMPILE="clang++ ${COMMON_FLAGS} -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/${MAIN_FILE}.cpp ${PREFIX}/${APPLICATION}.cpp"
      if [[ "$(uname)" == "Linux" ]]; then 
        COMPILE="${COMPILE} -Wl,-rpath,$CONDA_PREFIX/lib"
      fi
      ${COMPILE}
      
      # 4. Run it.
      EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      EXECUTE="./${EXECUTABLE} ${OBJECT} ${PARTITION} ${SCHEDULE} ${ARGV}"
      echo "${EXECUTE}"
      if [[ "$(uname)" == "Linux" ]]; then
        EXECUTE="${FREDWOOD_FLAG} ${EXECUTE}"
      fi
      for ((i=0; i < N; i++)); do
        ${EXECUTE} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done

      # 5. Clean up
      rm ${PREFIX}/${APPLICATION}.h
      rm ${PREFIX}/${APPLICATION}.cpp
      rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      rm -f -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
      rm ${LAYOUT_FILE}
    done
    echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
  done

  rm ${PREFIX}/${MAIN_FILE}.cpp # remove the old c++ file
}

if [[ "${DEBUG_MODE}" == true ]]; then
  echo "running debug test for ${DEBUG_LAYOUT} in ${DEBUG_BVH_SUFFIX}..."
  run_tests "${DEBUG_BVH_SUFFIX}" "${DEBUG_LAYOUT}"
  echo "... debug test complete"
else
  echo "running tests with 8-mixed-BVH..."
  run_tests "8_mixed"
  echo "... tests complete for 8-mixed-BVH"

  echo "running tests with 8-BVH..."
  run_tests "8"
  echo "... tests complete for 8-BVH"

  echo "running tests with 2-BVH..."
  run_tests "2"
  echo "... tests complete for 2-BVH"
fi

rm ${RAY_PATH}/${RAY_FILE}.out

exit 0
