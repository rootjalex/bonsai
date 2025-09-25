#!/bin/bash 

set -euo pipefail

APPLICATION="rt"
TARGET="cpu"
KERNEL_PATH="apps/${APPLICATION}"
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUTS=("ptr" "soa" "pbrt")
OBJECTS=("san-miguel" "hairball" "dragon" "sponza")
RAY_COUNT="${1:-65536}"   # default 2^16
N="${2:-14}" # drop lowest 2 and highest 2 runs in processing
HIT_RATIO="${3:-75}" # n%, e.g., 75% is the default 
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
for OBJECT in "${OBJECTS[@]}"; do
   echo "object: ${OBJECT}" 
  if [ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays" ]; then
    echo "no rays found for ${OBJECT}; generating now..."
    ./${RAY_PATH}/${RAY_FILE}.out ${OBJECT} ${RAY_PATH} ${RAY_COUNT} 0.${HIT_RATIO}
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
    for ((i=0; i < N; i++)); do
      ./${PREFIX}/${APPLICATION}_${LAYOUT}.out ${OBJECT} ${RAY_COUNT} ${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${HIT_RATIO}.rays >> ${DATA_PATH}/${DATA_FILE}.txt
    done
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
