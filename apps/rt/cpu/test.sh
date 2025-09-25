#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("soa-align16-p-v2" "soa-align16-p" "soa-align16-v2" "soa-align16" "soa-align32-p-v2" "soa-align32-p" "soa-align32-v2" "soa-align32" "pbrt-align16-p" "pbrt-align16" "pbrt-align32-p" "pbrt-align32")
OBJECTS=("san-miguel" "hairball" "dragon" "sponza")
TYPE="${1:-COMPARISON}" # other option, PERFORMANCE
RAY_COUNT="${2:-65536}"   # default 2^16
N="${3:-14}" # drop lowest 2 and highest 2 runs in processing
HIT_RATIO="${4:-75}" # n%, e.g., 75% is the default
RAY_PATH="${PREFIX}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# Delete previous data.
rm -f -r ${DATA_PATH}
mkdir ${DATA_PATH}

# If this is the first ever run, make the ray directory.
mkdir -p ${RAY_PATH}

# Install python dependencies for data processing.
pip install -r ${KERNEL_PATH}/requirements.txt

# Save a set of random rays.
clang++ -std=c++20 -O3 -o ${RAY_PATH}/${RAY_FILE}.out ${KERNEL_PATH}/generate.cpp

echo "rays: ${RAY_COUNT}, runs: ${N}, hit ratio: 0.${HIT_RATIO}"
echo ${RAY_COUNT} >> ${DATA_PATH}/${DATA_FILE}.txt
if [[ "$(uname)" == "Linux" ]]; then
  echo "Running on Linux (presumably Redwood)!"
fi

for OBJECT in "${OBJECTS[@]}"; do
   echo "object: ${OBJECT}" 
  if [ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays" ]; then
    echo "no rays found for ${OBJECT}; generating now..."
    FLAG=""
    if [[ "$(uname)" == "Linux" ]]; then
      FLAG="${FLAG} numactl --physcpubind 0-15" # only run on performance cores for the Fredwood.
    fi
    ${FLAG} ./${RAY_PATH}/${RAY_FILE}.out ${OBJECT} ${RAY_PATH} ${RAY_COUNT} 0.${HIT_RATIO}
    echo "...${RAY_COUNT} rays generated for ${OBJECT} with hit ratio: 0.${HIT_RATIO}"
  fi
  echo "${OBJECT}" >> ${DATA_PATH}/${DATA_FILE}.txt
  for LAYOUT in "${LAYOUTS[@]}"; do
    echo "  ${APPLICATION}, ${TARGET}, ${LAYOUT}"
    echo "${APPLICATION}, ${TARGET}, ${LAYOUT}" >> ${DATA_PATH}/${DATA_FILE}.txt
    # 1. Build the Bonsai compiler.
    cmake --build build --config Debug -j > /dev/null
    # 2. Lower to C++.
    ./build/compiler -i ${KERNEL_PATH}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cppx -o ${PREFIX}/${APPLICATION}
    # 3. Compile the lowered C++.
    clang++ -std=c++20 -O3 -g -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/main.cpp ${PREFIX}/${APPLICATION}.cpp -I. -Iapps/${APPLICATION} -Iruntime/CPP 
    # 4. Run it.
    EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
    COMMAND="./${EXECUTABLE} ${OBJECT} ${RAY_COUNT} ${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays"
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
        ${COMMAND} >> ${DATA_PATH}/${DATA_FILE}.txt
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

rm ${RAY_PATH}/${RAY_FILE}.out

# Process data
python3.11 ${PREFIX}/collect.py ${DATA_PATH}/${DATA_FILE}.txt

exit 0
