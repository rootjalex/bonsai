#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="embree"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="${KERNEL_PATH}/layouts"

OBJECTS=("hairball" "white-oak" "power-plant" "sponza") # sheep, san-miguel-x35-y22-z47

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
rm -rf ${PREFIX}/build
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
  LAYOUTS=("bvh4" "bvh8")
  echo "-- with layouts: ${LAYOUTS[@]}"
  

  for OBJECT in "${OBJECTS[@]}"; do
    echo "object: ${OBJECT}" 
    echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "${APPLICATION}, ${TARGET}, embree-${LAYOUT}" | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      # 3. Build it.
      mkdir -p "${PREFIX}/build"
      cd "${PREFIX}/build"
      cmake ..        > /dev/null
      cmake --build . > /dev/null

      cd ../../../..
      
      # 4. Run it.
      EXECUTE="./${PREFIX}/build/main_trace ${OBJECT} ${LAYOUT} ${SCHEDULE} ${ARGV}"
      echo "${EXECUTE}"
      if [[ "$(uname)" == "Linux" ]]; then
        EXECUTE="${FREDWOOD_FLAG} ${EXECUTE}"
      fi
      for ((i=0; i < N; i++)); do
        ${EXECUTE} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done

      # 5. Clean up
    done
    echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
  done

}

run_tests

rm -rf ${PREFIX}/build
rm ${RAY_PATH}/${RAY_FILE}.out

exit 0
