#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("eq" "ptr" "soa" "soa-align16" "soa-align32" "pbrt" "pbrt-align16" "pbrt-align32")
OBJECTS=("power-plant" "hairball")
TYPE="${1:-COMPARISON}" # other option, PERFORMANCE
N="${2:-14}" # drop lowest 2 and highest 2 runs in processing
HIT_RATIO="${3:-75}" # n%, e.g., 75% is the default
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"

MIN_POWER=15
MAX_POWER=20 # these should be aligned with the C++ file
RAY_COUNTS=()
for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
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
      echo "no rays found for ${OBJECT}; generating now..."
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


echo "runs: ${N}"
> ${DATA_PATH}/${DATA_FILE}.txt # clear

for OBJECT in "${OBJECTS[@]}"; do
  echo "object: ${OBJECT}" 
  echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
  for LAYOUT in "${LAYOUTS[@]}"; do
    echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT}"
    echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    # 1. Build the Bonsai compiler.
    cmake --build build --config Debug -j > /dev/null
    # 2. Lower to C++.
    ./build/compiler -i ${PREFIX}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
    # 3. Compile the lowered C++.
    clang++ -std=c++20 -O3 -march=native -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/main_trace.cpp ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/${APPLICATION} -Iruntime/CPP 
    # 4. Run it.
    EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
    COMMAND="./${EXECUTABLE} ${OBJECT}"
    if [[ "$(uname)" == "Linux" ]]; then
      COMMAND="numactl --physcpubind 0-15 ${COMMAND}" # only run on performance cores for the Fredwood.
    fi
    if [[ "${TYPE}" == "PERFORMANCE" ]]; then
      # collect
      perf record -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ${COMMAND}
      # report
      perf report --symbol-filter=*trace* --sort=overhead,symbol >> ${DATA_PATH}/${OBJECT}_${LAYOUT}.txt
    else
      for ((i=0; i < N; i++)); do
        ${COMMAND} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      done
    fi
    # 5. Clean up
    rm ${PREFIX}/${APPLICATION}.h
    rm ${PREFIX}/${APPLICATION}.cpp
    rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
    rm -f -r ${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM
  done
  echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
done

# Process data
python3.11 ${KERNEL_PATH}/collect_trace.py ${DATA_PATH}/${DATA_FILE}.txt

rm ${RAY_PATH}/${RAY_FILE}.out

exit 0
