#!/bin/bash 

set -euo pipefail

APPLICATION="wos"
TARGET="cuda"
KERNEL_PATH="apps/${APPLICATION}"
RAY_TRACING_PATH="apps/rt" # share same trees as 'rt'.
PREFIX="${KERNEL_PATH}/${TARGET}"
LAYOUT_PATH="${RAY_TRACING_PATH}/layouts" # share same layouts as 'rt'.

OBJECTS=("white-oak" "lucy" "sheep" "san-miguel-x35-y22-z47" "hairball" "sponza" "power-plant")

DRY_RUN=false
DEBUG_MODE=false
DEBUG_LAYOUT=""
POSITIONAL_ARGS=()

# parse command line arguments
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

N="${POSITIONAL_ARGS[0]:-9}"
N_QUERIES="${POSITIONAL_ARGS[1]:-1048576}" # 2^20
DATA_PATH="${PREFIX}/results"
DATA_FILE="closest_point"
PARTITION="sah"

echo "${N}, ${N_QUERIES}"

# Override for dry run
if [[ "${DRY_RUN}" == true ]]; then
  echo "*** DRY RUN MODE: testing with count=${N_QUERIES} only ***"
  N=1
  N_QUERIES=1000
  OBJECTS=("${OBJECTS[0]}")
fi

# Override for debug mode
if [[ "${DEBUG_MODE}" == true ]]; then
  echo "*** DEBUG MODE: testing layout ${DEBUG_LAYOUT} only ***"
  N=2
  N_QUERIES=1000
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

if [[ "$(pwd)" == */${PREFIX} ]]; then
  cd ../../..
fi

# Delete previous data.
rm -f -r ${DATA_PATH}
mkdir ${DATA_PATH}

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
      echo ${NAME}
      if [[ "${NAME}" == "ptr" ]]; then
        continue
      fi
      LAYOUTS+=("${NAME}")
    done
  fi
  echo "-- with layouts: ${LAYOUTS[@]}"
  
  MAIN_FILE="main"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" ${PREFIX}/${MAIN_FILE}.cu > ${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cu
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  # insert the canonical tree functions (we do it in this hacky way since they're shared between CPU / GPU.
  # a better approach might be using macros, similar to PBRT).
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${RAY_TRACING_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${RAY_TRACING_PATH}/canonical_tree_${BVH_SUFFIX}.h" ${PREFIX}/${MAIN_FILE}.cu
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
      
      # Prepend closest_point import to main.bonsai
      MAIN_BONSAI_TEMPORARY=$(mktemp).bonsai
      cat ${PREFIX}/main.bonsai > ${MAIN_BONSAI_TEMPORARY}
      
      ./build/compiler -i ${MAIN_BONSAI_TEMPORARY} -l ${LAYOUT_FILE} -b cuda -o ${PREFIX}/${APPLICATION}.h
      
      # Clean up temp main file
      rm ${MAIN_BONSAI_TEMPORARY}
      
      module load cuda
      nvcc -Iapps/wos -Iapps/rt -Iruntime/CUDA -O3 ${PREFIX}/${MAIN_FILE}.cu -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out
      
      EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      # Pass N and N_QUERIES as arguments to the executable
      EXECUTE="./${EXECUTABLE} ${OBJECT} ${PARTITION} ${LAYOUT} ${N} ${N_QUERIES}"
      if [[ "${DEBUG_MODE}" == true ]]; then
        EXECUTE="compute-sanitizer ${EXECUTE}"
      fi
      echo "${EXECUTE}"
      # Run the executable only once - it will handle N runs internally
      ${EXECUTE} | tee -a ${DATA_PATH}/${DATA_FILE}.txt
      
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
  echo "running tests with 8-BVH..."
  run_tests "8"
  echo "... tests complete for 8-BVH"

  echo "running tests with 2-BVH..."
  run_tests "2"
  echo "... tests complete for 2-BVH"
fi

exit 0
