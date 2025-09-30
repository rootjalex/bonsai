#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cuda"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("eq" "soa" "soa-align16" "soa-align32" "pbrt" "pbrt-align16" "pbrt-align32" "ptr")
OBJECTS=("power-plant" "hairball")
N="${1:-14}" # drop lowest 2 and highest 2 runs in processing
HIT_RATIO="${2:-75}" # n%, e.g., 75% is the default
RAY_PATH="${KERNEL_PATH}/rays"
RAY_FILE="kernel"
DATA_PATH=${PREFIX}/results
DATA_FILE="data"

MIN_POWER=16
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

# Delete previous data.
rm -f -r ${DATA_PATH}
mkdir ${DATA_PATH}

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

# Install python dependencies for data processing.
pip install -r ${KERNEL_PATH}/requirements.txt

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
    # 2. Lower to cuda.
    ./build/compiler -i ${PREFIX}/main.bonsai -l ${PREFIX}/${LAYOUT}.bonsai -b cuda -o ${PREFIX}/${APPLICATION}.h
    # 3. Compile the lowered cuda.
    module load cuda
    nvcc -Iapps/rt -Iruntime/CUDA -O3 ${PREFIX}/main_trace.cu -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out
    # 4. Run it.
    EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
    COMMAND="./${EXECUTABLE} ${OBJECT}"
    for ((i=0; i < N; i++)); do
      ${COMMAND} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
    done
    # 5. Clean up
    rm ${PREFIX}/${APPLICATION}.h
    rm ${PREFIX}/${APPLICATION}_${LAYOUT}.out
  done
  echo -e "---\n" >> ${DATA_PATH}/${DATA_FILE}.txt
done

# Process data
python3.11 ${KERNEL_PATH}/collect_trace.py ${DATA_PATH}/${DATA_FILE}.txt

rm ${RAY_PATH}/${RAY_FILE}.out

exit 0
