#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"

LAYOUTS_2BVH=("ptr" "soa" "soa-align16" "soa-align32" "pbrt" "pbrt-align16" "pbrt-align32")
LAYOUTS_8BVH=("bvh8" "qbvh8" "bvh8-align32" "qbvh8-align32")

OBJECTS=("power-plant" "hairball" "sponza")

# Parse flags
DRY_RUN=false
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      # If dry-run is passed, clear any previously collected args and break
      POSITIONAL_ARGS=()
      break
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done


TYPE="${1:-COMPARISON}" # other option, PERFORMANCE
N="${2:-9}" # drop lowest 2 and highest 2 runs in processing
SCHEDULE="${3:-parallel}" # or single-thread
HIT_RATIO="${4:-75}" # n%, e.g., 75% is the default
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"

MIN_POWER=10
MAX_POWER=24 # these should be aligned with the C++ file

# Override for dry run.
if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with MIN_POWER=${MIN_POWER} only ***"
  MAX_POWER=${MIN_POWER}
  N=2  # Only 2 iterations for dry run.
  LAYOUTS_2BVH=("${LAYOUTS_2BVH[0]}" "${LAYOUTS_2BVH[1]}")  # Only first two layouts.
  LAYOUTS_8BVH=("${LAYOUTS_8BVH[0]}" "${LAYOUTS_8BVH[1]}")  # Only first two layouts.
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

if [[ "$(uname)" == "Linux" ]]; then
  echo "Running on Linux (presumably Redwood)!"
fi

echo "runs: ${N}, schedule: ${SCHEDULE}"
> ${DATA_PATH}/${DATA_FILE}.txt # clear

# Function to run tests for a given main file and layouts
run_tests() {
  local MAIN_FILE=$1
  shift
  local LAYOUTS=("$@")
  
  for OBJECT in "${OBJECTS[@]}"; do
    echo "object: ${OBJECT}" 
    echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    for LAYOUT in "${LAYOUTS[@]}"; do
      echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT} (${MAIN_FILE})"
      echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
      # 1. Build the Bonsai compiler.
      cmake --build build --config Debug -j > /dev/null
      # 2. Lower to C++.
      ./build/compiler -i ${PREFIX}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
      # 3. Compile the lowered C++.
      COMMON_FLAGS="-std=c++20 -O3 -march=native -I. -Iapps/${APPLICATION} -Iruntime/CPP"
      if [[ "${SCHEDULE}" == "parallel" ]]; then
        COMMON_FLAGS="-fopenmp ${COMMON_FLAGS}"
      fi
      
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
      
      # Compile executable
      clang++ ${COMMON_FLAGS} -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/${MAIN_FILE}.cpp ${PREFIX}/${APPLICATION}.cpp
      
      # 4. Run it.
      EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      COMMAND="./${EXECUTABLE} ${OBJECT} ${SCHEDULE} ${ARGV}"
      if [[ "$(uname)" == "Linux" ]]; then
        COMMAND="numactl --physcpubind 0-15 ${COMMAND}" # only run on performance cores for the Fredwood.
      fi
      #   if [[ "${TYPE}" == "PERFORMANCE" ]]; then
      #     # collect
      #     perf record -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ${COMMAND}
      #     # report
      #     perf report --symbol-filter=*trace* --sort=overhead,symbol >> ${DATA_PATH}/${OBJECT}_${LAYOUT}.txt
      #   else
      for ((i=0; i < N; i++)); do
        ${COMMAND} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done
      #  fi

      # 5. Clean up
      rm ${PREFIX}/${APPLICATION}.h
      rm ${PREFIX}/${APPLICATION}.cpp
      rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      rm -f -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
    done
    echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
  done
}

echo "running tests for 2-BVH..."
run_tests "main_trace" "${LAYOUTS_2BVH[@]}"
echo "... tests complete for 2-BVH"

echo "running tests with 8-BVH..."
run_tests "main_trace_8" "${LAYOUTS_8BVH[@]}"
echo "... tests complete for 8-BVH"

# Process data
python3.11 ${KERNEL_PATH}/collect_trace.py ${DATA_PATH}/${DATA_FILE}.txt

rm ${RAY_PATH}/${RAY_FILE}.out

exit 0
