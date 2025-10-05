#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cuda"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"

LAYOUTS_2BVH=("eq" "eq-align32" "soa" "soa-align16" "soa-align32" "pbrt" "pbrt-align16" "pbrt-align32" "ptr")
LAYOUTS_8BVH=("bvh8" "cl-bvh8" "bvh8-align32" "cl-bvh8-align32")
LAYOUTS_8_MIXED_BVH=("ebq" "eb")

OBJECTS=("sponza" "power-plant" "hairball")

# Parse flags
DRY_RUN=false
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      POSITIONAL_ARGS=()
      break
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

N="${1:-14}" # drop lowest 2 and highest 2 runs in processing
HIT_RATIO="${2:-75}" # n%, e.g., 75% is the default
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"

MIN_POWER=10
MAX_POWER=25

# Override for dry run
if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${MIN_POWER} only ***"
  MAX_POWER=${MIN_POWER}
  N=2  # Only 2 iterations for dry run.
  LAYOUTS_2BVH=("${LAYOUTS_2BVH[0]}" "${LAYOUTS_2BVH[1]}")
  LAYOUTS_8BVH=("${LAYOUTS_8BVH[0]}" "${LAYOUTS_8BVH[1]}")
  LAYOUTS_8_MIXED_BVH=("${LAYOUTS_8_MIXED_BVH[0]}" "${LAYOUTS_8_MIXED_BVH[1]}")
  OBJECTS=("${OBJECTS[0]}")  # Only first object.
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

# Save a set of random rays.
clang++ -std=c++20 -O3 -march=native -o ${RAY_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/generate.cpp

for RAY_COUNT in "${RAY_COUNTS[@]}"; do
  echo ${RAY_COUNT} >> ${DATA_PATH}/${DATA_FILE}.txt
  for OBJECT in "${OBJECTS[@]}"; do
    if [ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays" ]; then
      echo "no rays found for ${OBJECT} with count ${RAY_COUNT} and ratio ${HIT_RATIO}; generating now..."
      FLAG=""
      if [[ "$(uname)" == "Linux" ]]; then
        FLAG="${FLAG} numactl --physcpubind 0-15" # only run on performance cores for the Fredwood.
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
> ${DATA_PATH}/${DATA_FILE}.txt # clear

# Function to run tests for a given main file and layouts
run_tests() {
  local BVH_SUFFIX="$1" # e.g., `2` or `8_mixed`. 
  shift
  local LAYOUTS=("$@")
  
  MAIN_FILE="main_trace"
  # replace `$N$` with BVH_SUFFIX.
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" ${PREFIX}/${MAIN_FILE}.cu > ${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cu
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  # Replace `// AUTO-GENERATED canonical_tree_8_mixed.h` with the respective tree construction code.
  # We need this diabolical hack because there's cyclic dependencies between the generated code and 
  # construction code. The correct fix is to generate header and source files instead of just a header.
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
  else
    # macos
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
  fi
  
  for OBJECT in "${OBJECTS[@]}"; do
    echo "object: ${OBJECT}" 
    echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT} (${MAIN_FILE})"
      echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
      # 1. Build the Bonsai compiler.
      cmake --build build --config Debug -j > /dev/null
      # 2. Lower to cuda.
      ./build/compiler -i ${PREFIX}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cuda -o ${PREFIX}/${APPLICATION}.h
      # 3. Compile the lowered cuda.
      module load cuda
      nvcc -Iapps/rt -Iruntime/CUDA -O3 ${PREFIX}/${MAIN_FILE}.cu -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      
      # 4. Run it.
      EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      EXECUTE="./${EXECUTABLE} ${OBJECT} ${ARGV}"
      for ((i=0; i < N; i++)); do
        ${EXECUTE} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done
      
      # 5. Clean up
      rm ${PREFIX}/${APPLICATION}.h
      rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
    done
    echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
  done

  rm ${PREFIX}/${MAIN_FILE}.cu # remove the old cu file
}

echo "running tests with 8-mixed-BVH..."
run_tests "8_mixed" "${LAYOUTS_8_MIXED_BVH[@]}"
echo "... tests complete for 8-mixed-BVH"

echo "running tests with 8-BVH..."
run_tests "8" "${LAYOUTS_8BVH[@]}"
echo "... tests complete for 8-BVH"

echo "running tests with 2-BVH..."
run_tests "2" "${LAYOUTS_2BVH[@]}"
echo "... tests complete for 2-BVH"

# Process data
python3.11 ${KERNEL_PATH}/collect_trace.py ${DATA_PATH}/${DATA_FILE}.txt

rm ${RAY_PATH}/${RAY_FILE}.out

exit 0