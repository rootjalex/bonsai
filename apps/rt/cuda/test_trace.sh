#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cuda"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="${KERNEL_PATH}/layouts"

OBJECTS=("white-oak" "hairball" "power-plant" "sponza")
# "sheep"

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

N="${1:-3}"
HIT_RATIO="${2:-75}"
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"
PARTITION="sah"

MIN_POWER=10
MAX_POWER=25

# Override for dry run
if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${MIN_POWER} only ***"
  MAX_POWER=${MIN_POWER}
  N=1
  OBJECTS=("${OBJECTS[0]}")
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

ARGV="${#RAY_COUNTS[@]}"
for COUNT in "${RAY_COUNTS[@]}"; do
    ARGV="${ARGV} ${COUNT}"
done

if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# Save a set of random rays.
clang++ -std=c++20 -O3 -march=native -o ${RAY_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/generate.cpp

for RAY_COUNT in "${RAY_COUNTS[@]}"; do
  echo ${RAY_COUNT} >> ${DATA_PATH}/${DATA_FILE}.txt
  for OBJECT in "${OBJECTS[@]}"; do
    if [ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays" ]; then
      echo "no rays found for ${OBJECT} with count ${RAY_COUNT} and ratio ${HIT_RATIO}; generating now..."
      FLAG=""
      if [[ "$(uname)" == "Linux" ]]; then
        FLAG="${FLAG} numactl --physcpubind 0-15"
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

echo "runs: ${N}"
> ${DATA_PATH}/${DATA_FILE}.txt

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
      if [[ "$(uname)" == "ptr" ]]; then
        continue
      fi
      LAYOUTS+=("${NAME}")
    done
  fi
  echo "-- with layouts: ${LAYOUTS[@]}"
  
  MAIN_FILE="main_trace"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" ${PREFIX}/${MAIN_FILE}.cu > ${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cu
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  # insert the canonical tree functions (we do it in this hacky way since they're shared between CPU / GPU.
  # a better approach might be using macros, similar to PBRT).
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${KERNEL_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${KERNEL_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
  fi
  
  for OBJECT in "${OBJECTS[@]}"; do
    echo "object: ${OBJECT}" 
    echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT} (${MAIN_FILE})"
      echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
      
      LAYOUT_FILE=$(mktemp).bonsai
      cat ${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai > ${LAYOUT_FILE}
      cat ${PREFIX}/schedule.bonsai >> ${LAYOUT_FILE}
      echo "}" >> "${LAYOUT_FILE}"

      cmake --build build --config Debug -j > /dev/null
      ./build/compiler -i ${PREFIX}/main.bonsai -l ${LAYOUT_FILE} -b cuda -o ${PREFIX}/${APPLICATION}.h
      module load cuda
      nvcc -Iapps/rt -Iruntime/CUDA -O3 ${PREFIX}/${MAIN_FILE}.cu -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      
      EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      EXECUTE="./${EXECUTABLE} ${OBJECT} ${PARTITION} ${ARGV}"
      if [[ "${DEBUG_MODE}" == true ]]; then
        EXECUTE="compute-sanitizer ${EXECUTE}"
      fi
      echo "${EXECUTE}"
      for ((i=0; i < N; i++)); do
        ${EXECUTE} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done
      
      rm ${PREFIX}/${APPLICATION}.h
      rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      rm ${LAYOUT_FILE}
    done
    echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
  done

  rm ${PREFIX}/${MAIN_FILE}.cu
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
