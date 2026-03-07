#!/bin/bash
#
# PLDI 2026 Artifact: Data Layout Polymorphism for Bounding Volume Hierarchies.
#
# We use three hardware platforms,
#   (1) Intel Core i9-14900K with 24 CPU cores (8 performacne, 16 efficiency) and AVX-512 (512-bit)
#       996 KiB L1, 32 MiB L2, and 36 MiB L3
#   (2) Apple MacBook M2 Pro with 10 CPU cores, Neon (128-bit)
#       16 GIB of unified memory
#   (3) Nvidia GeForce RTX 4090 
#       24 GiB of GDDR6X memory
#
#
# Usage:
#   ./artifact/run.sh              # Full evaluation (several hours)
#   ./artifact/run.sh --short      # Quick smoke test (~10 minutes)
#   ./artifact/run.sh --step N     # Run only step N (1=build, 2=DSE RT, 3=DSE CPQ,
#                                  #   4=Embree comparison, 5=FCL comparison,
#                                  #   6=FCPW comparison, 7=generate figures)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/artifact/results"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# ── Parse arguments ──────────────────────────────────────────────────────────
SHORT=false
STEP=""
GPU=false
show_help() {
  cat <<'HELPEOF'
Usage: ./artifact/run.sh [OPTIONS]

PLDI 2026 Artifact: Data Layout Polymorphism for Bounding Volume Hierarchies

Options:
  --short        Quick smoke test (~10 minutes). Runs 1 scene, 1 repetition,
                 and reduced ray/query counts for each step. Use this to verify
                 that everything compiles and executes correctly.

  --step N       Run only step N. Steps can be run independently and in any
                 order (though step 1 must complete before steps 2-6).

  --gpu          Enable GPU (CUDA) benchmarks. When set, steps 2 and 3 compile
                 and run on an NVIDIA GPU using nvcc. Steps 4-6 (Embree, FCL,
                 FCPW comparisons) are CPU-only and are skipped with --gpu.
                 Requires: nvcc (CUDA toolkit) on PATH.

  --help, -h     Show this help message and exit.

Steps:
  0   Install dependencies (brew/conda/apt, Python venv)
  1   Build the Scion compiler
  2   Design Space Exploration: Ray Tracing (Table 1, Figure 7)
      CPU: 26 BVH layouts × 7 scenes × 2 ray types × 2 intersections
      GPU (--gpu): same layouts (excluding 'ptr') compiled with nvcc
  3   Design Space Exploration: Closest Point Queries (Table 1, Figure 7)
      CPU: BVH2 layouts × 7 scenes, with FCPW comparison
      GPU (--gpu): BVH2+BVH8 layouts via CUDA (no FCPW comparison)
  4   Embree comparison: Ray Tracing (Figure 8)       [CPU only]
  5   FCL comparison: Collision Detection (Figure 9a)  [CPU only]
  6   FCPW comparison: Closest Point Queries (Figure 9b) [CPU only]
  7   Generate figures from collected data (Figures 6-9)

Examples:
  ./artifact/run.sh                    # Full CPU evaluation (several hours)
  ./artifact/run.sh --short            # Quick smoke test (~10 min)
  ./artifact/run.sh --gpu              # Full GPU evaluation
  ./artifact/run.sh --gpu --short      # Quick GPU smoke test
  ./artifact/run.sh --step 2           # Run only RT DSE (CPU)
  ./artifact/run.sh --step 2 --gpu     # Run only RT DSE (GPU)
  ./artifact/run.sh --step 7           # Generate figures from existing data
HELPEOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --short)    SHORT=true;  shift ;;
    --step)     STEP="$2";   shift 2 ;;
    --gpu)      GPU=true;    shift ;;
    --help|-h)  show_help ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Short-run configuration ─────────────────────────────────────────────────
if [[ "${SHORT}" == true ]]; then
  RT_N=1                           # number of repetitions
  RT_OBJECTS=("hairball")          # single scene
  RT_RAY_TYPES=("camera")         # single ray type
  RT_INTERSECTIONS=("mt")         # single intersection method
  RT_2_LAYOUTS=("pbrt" "pbrt-q16")
  RT_8_LAYOUTS=("bvh8" "bvh8-q8")
  EMBREE_LAYOUTS=("qbvh8i" "bvh8i")
  CD_N=1
  CD_OBJECTS_A=("hairball")
  CD_OBJECTS_B=("hairball_60_70_10")
  CD_LAYOUTS=("pbrt" "fcl")
  CPQ_N=1
  CPQ_OBJECTS=("hairball")
  CPQ_N_QUERIES=1000
  CPQ_LAYOUTS=("pbrt")
  MIN_POWER=18
  MAX_POWER=18
  # GPU-specific (CUDA skips 'ptr' layout)
  CUDA_RT_2_LAYOUTS=("pbrt" "pbrt-q16")
  CUDA_RT_8_LAYOUTS=("bvh8" "bvh8-q8")
  CUDA_CPQ_N_QUERIES=1000
else
  RT_N=9
  RT_OBJECTS=("lucy" "sheep" "san-miguel-x35-y22-z47" "hairball" "white-oak" "sponza" "power-plant")
  RT_RAY_TYPES=("camera" "secondary")
  RT_INTERSECTIONS=("mt" "pc")
  RT_2_LAYOUTS=("pbrt" "pbrt-align16" "pbrt-q16" "pbrt-q16-soaos" "pbrt-soaos" "pbrt-soaos-align16" "ptr" "sg-eq" "sg-eq-align16")
  RT_8_LAYOUTS=("bvh8" "bvh8-align16" "bvh8-q8" "bvh8-q8-align16" "bvh8-q8-ci" "bvh8-q8-ci-align16" "bvh8-q16" "bvh8-q16-align16" "bvh8-q16-ci" "bvh8-q16-ci-align16")
  EMBREE_LAYOUTS=("qbvh8i" "bvh8i")
  CD_N=9
  CD_OBJECTS_A=("hairball" "hairball" "dragon_60_70_10")
  CD_OBJECTS_B=("hairball_60_70_10" "dragon" "hairball_60_70_10")
  CD_LAYOUTS=("pbrt" "pbrt-q16" "fcl")
  CPQ_N=9
  CPQ_OBJECTS=("lucy" "hairball" "white-oak" "sheep" "san-miguel-x35-y22-z47" "sponza" "power-plant")
  CPQ_N_QUERIES=100000
  CPQ_LAYOUTS=("pbrt")
  MIN_POWER=18
  MAX_POWER=23
  # GPU-specific (CUDA skips 'ptr' layout; uses all others)
  CUDA_RT_2_LAYOUTS=("pbrt" "pbrt-align16" "pbrt-q16" "pbrt-q16-soaos" "pbrt-soaos" "pbrt-soaos-align16" "sg-eq" "sg-eq-align16")
  CUDA_RT_8_LAYOUTS=("bvh8" "bvh8-align16" "bvh8-q8" "bvh8-q8-align16" "bvh8-q8-ci" "bvh8-q8-ci-align16" "bvh8-q16" "bvh8-q16-align16" "bvh8-q16-ci" "bvh8-q16-ci-align16")
  CUDA_CPQ_N_QUERIES=1048576
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[artifact]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[  OK  ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

should_run() {
  [[ -z "${STEP}" ]] || [[ "${STEP}" == "$1" ]]
}

detect_platform() {
  case "$(uname -s)" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      fail "Unsupported platform: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    arm64|aarch64) ARCH="arm" ;;
    x86_64|amd64)  ARCH="x86" ;;
    *)              fail "Unsupported architecture: $(uname -m)" ;;
  esac
  log "Platform: ${PLATFORM}/${ARCH} ($(nproc 2>/dev/null || sysctl -n hw.logicalcpu) cores)"
}

# ── Step 0: Install dependencies ────────────────────────────────────────────
install_deps() {
  log "Step 0: Installing dependencies..."

  if [[ "${PLATFORM}" == "macos" ]]; then
    # Ensure Homebrew is available
    if ! command -v brew &>/dev/null; then
      fail "Homebrew is required on macOS. Install from https://brew.sh"
    fi

    # Core build dependencies
    brew install cmake llvm@19 libomp embree eigen 2>/dev/null || true

    # Set LLVM root for CMake's find_package(ModernLLVM) only.
    # Do NOT put Homebrew LLVM on PATH or set CC/CXX — the Homebrew clang
    # lacks the macOS SDK sysroot and will fail on standalone compilations.
    # The system /usr/bin/clang++ (Apple Clang) is used for app compilation.
    LLVM_PREFIX="$(brew --prefix llvm@19)"
    export LLVM_ROOT="${LLVM_PREFIX}"

    # Detect macOS SDK sysroot for any clang invocations that need it
    MACOS_SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "")"

  elif [[ "${PLATFORM}" == "linux" ]]; then
    # Check for conda environment (used on evaluation machines)
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      log "Conda environment detected: ${CONDA_PREFIX}"
      # Install via conda-forge. Split into separate installs so that a
      # conflict in one group doesn't block the others.
      log "  Installing cmake >= 3.30..."
      conda install -y -c conda-forge "cmake>=3.30" || \
        fail "Failed to install cmake >= 3.30 via conda. Install manually."
      # Install each group separately to isolate solver conflicts.
      log "  Installing LLVM 19, clang, Embree, Eigen, OpenMP..."
      conda install -y -c conda-forge llvmdev=19.1.6 clangdev=19.1.6 \
        embree eigen libomp 2>/dev/null || {
        log "  WARNING: Full conda install failed; trying packages individually..."
        conda install -y -c conda-forge clangdev=19.1.6 llvmdev=19.1.6 2>/dev/null || \
          conda install -y -c conda-forge clangdev llvmdev 2>/dev/null || true
        conda install -y -c conda-forge embree eigen libomp 2>/dev/null || true
      }

      # If conda clang or libomp not available, try installing them individually.
      if [[ ! -x "${CONDA_PREFIX}/bin/clang++" ]]; then
        log "  Retrying clang install..."
        conda install -y -c conda-forge clang_linux-64 2>/dev/null || true
      fi
      if [[ ! -f "${CONDA_PREFIX}/include/omp.h" ]]; then
        log "  Installing libomp separately..."
        conda install -y -c conda-forge llvm-openmp 2>/dev/null || \
          conda install -y -c conda-forge openmp 2>/dev/null || true
      fi
      # Ensure eigen is installed (needed by FCPW/FCL builds)
      local _eigen_found=false
      for _d in "${CONDA_PREFIX}/share/eigen3/cmake" \
                "${CONDA_PREFIX}/lib/cmake/eigen3" \
                "${CONDA_PREFIX}/share/cmake/eigen3"; do
        if [[ -f "${_d}/Eigen3Config.cmake" ]]; then _eigen_found=true; break; fi
      done
      if [[ "${_eigen_found}" == false ]]; then
        log "  Installing Eigen3..."
        conda install -y -c conda-forge eigen 2>/dev/null || true
      fi
      # Note: Polyscope/GLFW is built with mock GL backend (headless benchmark),
      # so X11/GL dev headers are not required.

      if [[ -x "${CONDA_PREFIX}/bin/clang++" ]]; then
        export CC="${CONDA_PREFIX}/bin/clang"
        export CXX="${CONDA_PREFIX}/bin/clang++"
      elif command -v clang++ &>/dev/null; then
        log "  Using system clang++: $(command -v clang++)"
        export CC="clang"
        export CXX="clang++"
      else
        fail "clang++ not found. Install via: conda install -c conda-forge clangdev, or: sudo apt install clang"
      fi
      export LLVM_ROOT="${CONDA_PREFIX}"
    else
      # No conda — expect dependencies to be pre-installed.
      log "No conda environment detected. Ensure the following are installed:"
      log "  cmake (>= 3.30), clang, llvm-dev (19.x), libomp-dev, embree (4.x), eigen3"
      if command -v clang++ &>/dev/null; then
        export CC="clang"
        export CXX="clang++"
      fi
      if [[ -d "/usr/lib/llvm-19" ]]; then
        export LLVM_ROOT="/usr/lib/llvm-19"
      fi
    fi
  fi

  # Python dependencies (for analysis scripts).
  # Create a venv to avoid PEP 668 issues with system Python.
  if [[ ! -d "${ROOT_DIR}/artifact/.venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv "${ROOT_DIR}/artifact/.venv"
  fi
  source "${ROOT_DIR}/artifact/.venv/bin/activate"
  pip install --quiet matplotlib numpy pandas

  ok "Dependencies installed"
}

# ── Step 1: Build Scion (Bonsai compiler) ────────────────────────────────────
build_compiler() {
  log "Step 1: Building Scion compiler..."
  cd "${ROOT_DIR}"

  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
  cmake --build build -j"${NPROC}" 2>&1 | tail -3

  # Verify compiler works
  if [[ ! -x build/compiler ]]; then
    # Try alternate location
    if [[ -x build/bin/compiler ]]; then
      ln -sf "${ROOT_DIR}/build/bin/compiler" "${ROOT_DIR}/build/compiler" 2>/dev/null || true
    else
      fail "Compiler binary not found after build"
    fi
  fi

  ok "Scion compiler built successfully"
}

# ── Ray generation helper ────────────────────────────────────────────────────
generate_rays() {
  local RAY_TYPE="$1"
  log "Generating rays (type=${RAY_TYPE})..."

  cd "${ROOT_DIR}"
  local RAY_PATH="apps/rt/rays"
  local KERNEL_PATH="apps/rt"

  # Build ray generator (use system clang on macOS to avoid sysroot issues)
  local GEN_CXX="/usr/bin/clang++"
  local GEN_FLAGS="-std=c++20 -O3 -march=native"
  if [[ "${PLATFORM}" == "macos" ]]; then
    # Apple Clang doesn't use -fopenmp by default; libomp from brew
    local OMP_PREFIX
    OMP_PREFIX="$(brew --prefix libomp 2>/dev/null || echo "")"
    if [[ -n "${OMP_PREFIX}" ]]; then
      GEN_FLAGS="${GEN_FLAGS} -Xpreprocessor -fopenmp -I${OMP_PREFIX}/include -L${OMP_PREFIX}/lib -lomp"
    fi
  else
    GEN_CXX="${CXX:-clang++}"
    GEN_FLAGS="${GEN_FLAGS} -fopenmp"
    # If libomp is in conda, add its include/lib paths
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "${CONDA_PREFIX}/include/omp.h" ]]; then
      GEN_FLAGS="${GEN_FLAGS} -I${CONDA_PREFIX}/include -L${CONDA_PREFIX}/lib"
    fi
  fi
  ${GEN_CXX} ${GEN_FLAGS} -o "${RAY_PATH}/kernel.out" "${KERNEL_PATH}/generate.cpp"

  local RAY_COUNTS=()
  for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
  done

  for RAY_COUNT in "${RAY_COUNTS[@]}"; do
    for OBJECT in "${RT_OBJECTS[@]}"; do
      if [[ ! -f "${RAY_PATH}/${OBJECT}_${RAY_COUNT}_${RAY_TYPE}.rays" ]]; then
        ./"${RAY_PATH}/kernel.out" "${OBJECT}" "${RAY_PATH}" "${RAY_COUNT}" "${RAY_TYPE}"
      fi
    done
  done
}

# ── Step 2: DSE - Ray Tracing ───────────────────────────────────────────────
run_dse_rt() {
  log "Step 2: Design Space Exploration - Ray Tracing (Table 1, Figure 7)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/rt"

  local RAY_COUNTS=()
  for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
  done
  local ARGV="${#RAY_COUNTS[@]}"
  for COUNT in "${RAY_COUNTS[@]}"; do
    ARGV="${ARGV} ${COUNT}"
  done

  local APPLICATION="rt"
  local PREFIX="apps/rt/cpu"
  local LAYOUT_PATH="apps/rt/layouts"
  local PARTITION="sah"

  for RAY_TYPE in "${RT_RAY_TYPES[@]}"; do
    generate_rays "${RAY_TYPE}"

    for INTERSECT in "${RT_INTERSECTIONS[@]}"; do
      # Name files as <machine>-<ray_type>.txt for hygiene_rt.py compatibility.
      # ray_type "camera" is used as-is (hygiene_rt.py converts to "primary").
      local OUTFILE="${RESULTS_DIR}/rt/${ARCH}-${RAY_TYPE}.txt"
      > "${OUTFILE}"

      # -- Run 2-BVH layouts --
      run_bonsai_rt_layouts "2" "${RT_2_LAYOUTS[*]}" "${RAY_TYPE}" "${INTERSECT}" \
        "${ARGV}" "${PARTITION}" "${OUTFILE}"

      # -- Run 8-BVH layouts --
      run_bonsai_rt_layouts "8" "${RT_8_LAYOUTS[*]}" "${RAY_TYPE}" "${INTERSECT}" \
        "${ARGV}" "${PARTITION}" "${OUTFILE}"
    done
  done

  ok "DSE RT complete. Results in ${RESULTS_DIR}/rt/"
}

run_bonsai_rt_layouts() {
  local BVH_SUFFIX="$1"
  local LAYOUTS_STR="$2"
  local RAY_TYPE="$3"
  local INTERSECT="$4"
  local ARGV="$5"
  local PARTITION="$6"
  local OUTFILE="$7"

  local APPLICATION="rt"
  local PREFIX="apps/rt/cpu"
  local LAYOUT_PATH="apps/rt/layouts"
  local SCHEDULE="parallel"

  read -ra LAYOUTS <<< "${LAYOUTS_STR}"

  cd "${ROOT_DIR}"

  # Prepare main_trace with canonical tree
  local MAIN_FILE="main_trace"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" "${PREFIX}/${MAIN_FILE}.cpp" > "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp"
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r apps/rt/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cpp"
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r apps/rt/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cpp"
  fi

  for OBJECT in "${RT_OBJECTS[@]}"; do
    echo "${OBJECT}" >> "${OUTFILE}"
    for LAYOUT in "${LAYOUTS[@]}"; do
      # Check layout file exists
      if [[ ! -f "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" ]]; then
        log "WARNING: Layout ${LAYOUT} not found in ${LAYOUT_PATH}/${BVH_SUFFIX}/, skipping"
        continue
      fi

      echo "${APPLICATION}, cpu, ${LAYOUT}, ${INTERSECT}" >> "${OUTFILE}"
      log "  RT: ${OBJECT} / ${LAYOUT} (bvh${BVH_SUFFIX}, ${INTERSECT})"

      # Combine layout + schedule
      local LAYOUT_FILE
      LAYOUT_FILE=$(mktemp).bonsai
      cat "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" > "${LAYOUT_FILE}"
      cat "${PREFIX}/schedule.bonsai" >> "${LAYOUT_FILE}"
      echo "}" >> "${LAYOUT_FILE}"

      # Build compiler (incremental)
      cmake --build build -j"${NPROC}" > /dev/null 2>&1

      # Prepend intersection import
      local MAIN_BONSAI
      MAIN_BONSAI=$(mktemp).bonsai
      echo "import apps/rt/cpu/intersect/${INTERSECT};" > "${MAIN_BONSAI}"
      cat "${PREFIX}/main.bonsai" >> "${MAIN_BONSAI}"

      # Lower to C++
      ./build/compiler -i "${MAIN_BONSAI}" -l "${LAYOUT_FILE}" -b cppx -o "${PREFIX}/${APPLICATION}"
      rm "${MAIN_BONSAI}"

      # Compile (use system clang on macOS to avoid sysroot issues)
      local APP_CXX="/usr/bin/clang++"
      local COMMON_FLAGS="-std=c++20 -O3 -march=native -I. -Iapps/${APPLICATION} -Iruntime/CPP"
      if [[ "${PLATFORM}" == "macos" ]]; then
        local OMP_PREFIX
        OMP_PREFIX="$(brew --prefix libomp 2>/dev/null || echo "")"
        if [[ -n "${OMP_PREFIX}" ]]; then
          COMMON_FLAGS="${COMMON_FLAGS} -Xpreprocessor -fopenmp -I${OMP_PREFIX}/include -L${OMP_PREFIX}/lib -lomp"
        fi
      else
        APP_CXX="${CXX:-clang++}"
        COMMON_FLAGS="${COMMON_FLAGS} -fopenmp"
        if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/include" ]]; then
          COMMON_FLAGS="${COMMON_FLAGS} -I${CONDA_PREFIX}/include -L${CONDA_PREFIX}/lib"
        fi
      fi
      local COMPILE_CMD="${APP_CXX} ${COMMON_FLAGS} -o ${PREFIX}/${APPLICATION}_${LAYOUT}.out ${PREFIX}/${MAIN_FILE}.cpp ${PREFIX}/${APPLICATION}.cpp"
      if [[ "$(uname)" == "Linux" ]] && [[ -n "${CONDA_PREFIX:-}" ]]; then
        COMPILE_CMD="${COMPILE_CMD} -Wl,-rpath,${CONDA_PREFIX}/lib"
      fi
      ${COMPILE_CMD}

      # Run
      local EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      for ((i=0; i < RT_N; i++)); do
        ./"${EXECUTABLE}" "${OBJECT}" "${PARTITION}" "${SCHEDULE}" "${RAY_TYPE}" "${LAYOUT}" ${ARGV} \
          | tee -a "${OUTFILE}"
      done

      # Clean up
      rm -f "${PREFIX}/${APPLICATION}.h" "${PREFIX}/${APPLICATION}.cpp"
      rm -f "${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      rm -f -r "${PREFIX}/${APPLICATION}_${LAYOUT}.out.dSYM"
      rm -f "${LAYOUT_FILE}"
    done
    echo -e "---\n" >> "${OUTFILE}"
  done

  rm -f "${PREFIX}/${MAIN_FILE}.cpp"
}

# ── Step 3: DSE - Closest Point Queries ──────────────────────────────────────
run_dse_cpq() {
  log "Step 3: Design Space Exploration - Closest Point Queries (Table 1, Figure 7)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/cpq"

  local APPLICATION="wos"
  local PREFIX="apps/wos/fcpw"
  local LAYOUT_PATH="apps/wos/fcpw/layouts"
  local BVH_SUFFIX="2"
  local OUTFILE="${RESULTS_DIR}/cpq/dse.txt"
  > "${OUTFILE}"

  for OBJECT in "${CPQ_OBJECTS[@]}"; do
    for LAYOUT in "${CPQ_LAYOUTS[@]}"; do
      log "  CPQ: ${OBJECT} / ${LAYOUT}"
      echo "--- ${OBJECT} - ${LAYOUT} - ${CPQ_N_QUERIES} ---" >> "${OUTFILE}"

      # Combine layout + schedule
      local LAYOUT_FILE
      LAYOUT_FILE=$(mktemp).bonsai
      cat "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" > "${LAYOUT_FILE}"
      cat "${PREFIX}/schedule.bonsai" >> "${LAYOUT_FILE}"
      echo "}" >> "${LAYOUT_FILE}"

      # Build compiler (incremental)
      cmake --build build -j"${NPROC}" > /dev/null 2>&1

      # Lower to C++ (generate wos.h and wos.cpp)
      ./build/compiler -i "apps/wos/main.bonsai" -l "${LAYOUT_FILE}" -b cppx -o "${PREFIX}/${APPLICATION}"

      # Patch: replace the generated _traverse_tree0 with an optimized version
      # that uses priority-queue child ordering (visiting the nearer child first).
      # The current version of Scion does not yet support this scheduling
      # optimization natively. The traversal algorithm is orthogonal to the
      # data layout contributions of this paper; see artifact/cpq_traverse_patch.cpp.
      python3 -c "
import re, sys
with open('${PREFIX}/${APPLICATION}.cpp', 'r') as f:
    src = f.read()
with open('artifact/cpq_traverse_patch.cpp', 'r') as f:
    patch = f.read()
# Replace the _traverse_tree0 function body (from signature to its closing brace)
pattern = r'Triangle _traverse_tree0\(const Point\s*\*\s*__restrict__\s+p,\s*const Triangles\s*\*\s*__restrict__\s+triangles\)\s*\{.*?\n\}\n'
result = re.sub(pattern, patch.strip(), src, count=1, flags=re.DOTALL)
if result == src:
    print('ERROR: _traverse_tree0 pattern not found in generated code; cannot apply CPQ patch', file=sys.stderr)
    sys.exit(1)
else:
    with open('${PREFIX}/${APPLICATION}.cpp', 'w') as f:
        f.write(result)
    print('Patched _traverse_tree0 with priority-queue child ordering')
"

      # Prepare main file with canonical tree
      local MAIN_FILE="main"
      sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" "${PREFIX}/${MAIN_FILE}.cpp" > "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp"
      if [[ "$(uname)" == "Linux" ]]; then
        sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp"
      else
        sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${PREFIX}/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp"
      fi

      # Build the WOS executable (fetches FCPW/Polyscope on first run)
      cd "${PREFIX}"
      mkdir -p build && cd build
      local BUILD_DIRECTORY="build_${BVH_SUFFIX}_${LAYOUT}"
      mkdir -p "${BUILD_DIRECTORY}" && cd "${BUILD_DIRECTORY}"

      local EXTRA_CMAKE_FLAGS=""
      if [[ "$(uname)" == "Linux" ]] && [[ -x "${CONDA_PREFIX:-}/bin/clang++" ]]; then
        EXTRA_CMAKE_FLAGS="-DCMAKE_CXX_COMPILER=${CONDA_PREFIX}/bin/clang++"
      fi
      if [[ -n "${CONDA_PREFIX:-}" ]]; then
        EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS} -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}"
        # Eigen3 from conda may be in share/eigen3/cmake or lib/cmake/eigen3
        for _eigen_dir in "${CONDA_PREFIX}/share/eigen3/cmake" \
                          "${CONDA_PREFIX}/lib/cmake/eigen3" \
                          "${CONDA_PREFIX}/share/cmake/eigen3"; do
          if [[ -f "${_eigen_dir}/Eigen3Config.cmake" ]]; then
            EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS} -DEigen3_DIR=${_eigen_dir}"
            break
          fi
        done
      fi
      # Force mock GL backend — this is a headless benchmark, no rendering needed.
      cmake -DLAYOUT="${LAYOUT}" -DAPPLICATION="${APPLICATION}" -DBVH_SUFFIX="${BVH_SUFFIX}" \
        -DPOLYSCOPE_BACKEND_OPENGL3_GLFW=OFF -DPOLYSCOPE_BACKEND_OPENGL_MOCK=ON \
        ../.. ${EXTRA_CMAKE_FLAGS}
      make -j"${NPROC}"

      cd "${ROOT_DIR}"

      for ((k=0; k < CPQ_N; k++)); do
        ./"${PREFIX}/build/${BUILD_DIRECTORY}/${APPLICATION}_${LAYOUT}.out" "${OBJECT}" ${CPQ_N_QUERIES} \
          | tee -a "${OUTFILE}"
      done

      rm -f "${PREFIX}/${APPLICATION}.h" "${PREFIX}/${APPLICATION}.cpp"
      rm -f "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cpp"
    done
  done

  ok "DSE CPQ complete. Results in ${RESULTS_DIR}/cpq/"
}

# ── GPU: CUDA RT DSE ─────────────────────────────────────────────────────────
run_dse_rt_cuda() {
  log "Step 2 (GPU): Design Space Exploration - Ray Tracing (CUDA)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/rt"

  local RAY_COUNTS=()
  for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
  done
  local ARGV="${#RAY_COUNTS[@]}"
  for COUNT in "${RAY_COUNTS[@]}"; do
    ARGV="${ARGV} ${COUNT}"
  done

  local APPLICATION="rt"
  local PREFIX="apps/rt/cuda"
  local LAYOUT_PATH="apps/rt/layouts"
  local PARTITION="sah"

  # Try to load CUDA module (Linux HPC systems)
  module load cuda 2>/dev/null || true

  # Verify nvcc is available
  if ! command -v nvcc &>/dev/null; then
    fail "nvcc not found. Install the CUDA toolkit or load the cuda module."
  fi

  for RAY_TYPE in "${RT_RAY_TYPES[@]}"; do
    generate_rays "${RAY_TYPE}"

    for INTERSECT in "${RT_INTERSECTIONS[@]}"; do
      local OUTFILE="${RESULTS_DIR}/rt/cuda-${RAY_TYPE}.txt"
      > "${OUTFILE}"

      # -- Run 2-BVH layouts --
      run_cuda_rt_layouts "2" "${CUDA_RT_2_LAYOUTS[*]}" "${RAY_TYPE}" "${INTERSECT}" \
        "${ARGV}" "${PARTITION}" "${OUTFILE}"

      # -- Run 8-BVH layouts --
      run_cuda_rt_layouts "8" "${CUDA_RT_8_LAYOUTS[*]}" "${RAY_TYPE}" "${INTERSECT}" \
        "${ARGV}" "${PARTITION}" "${OUTFILE}"
    done
  done

  ok "DSE RT (CUDA) complete. Results in ${RESULTS_DIR}/rt/"
}

run_cuda_rt_layouts() {
  local BVH_SUFFIX="$1"
  local LAYOUTS_STR="$2"
  local RAY_TYPE="$3"
  local INTERSECT="$4"
  local ARGV="$5"
  local PARTITION="$6"
  local OUTFILE="$7"

  local APPLICATION="rt"
  local PREFIX="apps/rt/cuda"
  local LAYOUT_PATH="apps/rt/layouts"

  read -ra LAYOUTS <<< "${LAYOUTS_STR}"

  cd "${ROOT_DIR}"

  # Prepare main_trace with canonical tree
  local MAIN_FILE="main_trace"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" "${PREFIX}/${MAIN_FILE}.cu" > "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cu"
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r apps/rt/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cu"
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r apps/rt/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cu"
  fi

  for OBJECT in "${RT_OBJECTS[@]}"; do
    echo "${OBJECT}" >> "${OUTFILE}"
    for LAYOUT in "${LAYOUTS[@]}"; do
      if [[ ! -f "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" ]]; then
        log "WARNING: Layout ${LAYOUT} not found in ${LAYOUT_PATH}/${BVH_SUFFIX}/, skipping"
        continue
      fi

      echo "${APPLICATION}, cuda, ${LAYOUT}, ${INTERSECT}" >> "${OUTFILE}"
      log "  RT CUDA: ${OBJECT} / ${LAYOUT} (bvh${BVH_SUFFIX}, ${INTERSECT})"

      # Combine layout + schedule
      local LAYOUT_FILE
      LAYOUT_FILE=$(mktemp).bonsai
      cat "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" > "${LAYOUT_FILE}"
      cat "${PREFIX}/schedule.bonsai" >> "${LAYOUT_FILE}"
      echo "}" >> "${LAYOUT_FILE}"

      # Build compiler (incremental)
      cmake --build build -j"${NPROC}" > /dev/null 2>&1

      # Prepend intersection import
      local MAIN_BONSAI
      MAIN_BONSAI=$(mktemp).bonsai
      echo "import apps/rt/cuda/intersect/${INTERSECT};" > "${MAIN_BONSAI}"
      cat "${PREFIX}/main.bonsai" >> "${MAIN_BONSAI}"

      # Lower to CUDA header
      ./build/compiler -i "${MAIN_BONSAI}" -l "${LAYOUT_FILE}" -b cuda -o "${PREFIX}/${APPLICATION}.h"
      rm "${MAIN_BONSAI}"

      # Compile with nvcc
      nvcc -Iapps/rt -Iruntime/CUDA -O3 \
        "${PREFIX}/${MAIN_FILE}.cu" -o "${PREFIX}/${APPLICATION}_${LAYOUT}.out"

      # Run (no --schedule arg on GPU)
      local EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      for ((i=0; i < RT_N; i++)); do
        ./"${EXECUTABLE}" "${OBJECT}" "${PARTITION}" "${RAY_TYPE}" "${LAYOUT}" ${ARGV} \
          | tee -a "${OUTFILE}"
      done

      # Clean up
      rm -f "${PREFIX}/${APPLICATION}.h"
      rm -f "${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      rm -f "${LAYOUT_FILE}"
    done
    echo -e "---\n" >> "${OUTFILE}"
  done

  rm -f "${PREFIX}/${MAIN_FILE}.cu"
}

# ── GPU: CUDA CPQ DSE ────────────────────────────────────────────────────────
run_dse_cpq_cuda() {
  log "Step 3 (GPU): Design Space Exploration - Closest Point Queries (CUDA)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/cpq"

  local APPLICATION="wos"
  local PREFIX="apps/wos/cuda"
  local RAY_TRACING_PATH="apps/rt"
  local LAYOUT_PATH="${RAY_TRACING_PATH}/layouts"  # shares layouts with RT
  local PARTITION="sah"
  local OUTFILE="${RESULTS_DIR}/cpq/cuda-dse.txt"
  > "${OUTFILE}"

  # Try to load CUDA module (Linux HPC systems)
  module load cuda 2>/dev/null || true

  if ! command -v nvcc &>/dev/null; then
    fail "nvcc not found. Install the CUDA toolkit or load the cuda module."
  fi

  # Run BVH8 first (matching original script order), then BVH2
  run_cuda_cpq_layouts "8" "${CUDA_RT_8_LAYOUTS[*]}" "${PARTITION}" "${OUTFILE}"
  run_cuda_cpq_layouts "2" "${CUDA_RT_2_LAYOUTS[*]}" "${PARTITION}" "${OUTFILE}"

  ok "DSE CPQ (CUDA) complete. Results in ${RESULTS_DIR}/cpq/"
}

run_cuda_cpq_layouts() {
  local BVH_SUFFIX="$1"
  local LAYOUTS_STR="$2"
  local PARTITION="$3"
  local OUTFILE="$4"

  local APPLICATION="wos"
  local PREFIX="apps/wos/cuda"
  local RAY_TRACING_PATH="apps/rt"
  local LAYOUT_PATH="${RAY_TRACING_PATH}/layouts"

  read -ra LAYOUTS <<< "${LAYOUTS_STR}"

  cd "${ROOT_DIR}"

  # Prepare main with canonical tree
  local MAIN_FILE="main"
  sed "s/\\\$N\\\$/${BVH_SUFFIX}/g" "${PREFIX}/${MAIN_FILE}.cu" > "${PREFIX}/${MAIN_FILE}_${BVH_SUFFIX}.cu"
  MAIN_FILE="${MAIN_FILE}_${BVH_SUFFIX}"
  if [[ "$(uname)" == "Linux" ]]; then
    sed -i "/\/\/ AUTO-GENERATED canonical_tree/r ${RAY_TRACING_PATH}/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cu"
  else
    sed -i '' "/\/\/ AUTO-GENERATED canonical_tree/r ${RAY_TRACING_PATH}/canonical_tree_${BVH_SUFFIX}.h" "${PREFIX}/${MAIN_FILE}.cu"
  fi

  for OBJECT in "${CPQ_OBJECTS[@]}"; do
    echo "${OBJECT}" >> "${OUTFILE}"
    for LAYOUT in "${LAYOUTS[@]}"; do
      if [[ ! -f "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" ]]; then
        log "WARNING: Layout ${LAYOUT} not found in ${LAYOUT_PATH}/${BVH_SUFFIX}/, skipping"
        continue
      fi

      echo "${APPLICATION}, cuda, ${LAYOUT}" >> "${OUTFILE}"
      log "  CPQ CUDA: ${OBJECT} / ${LAYOUT} (bvh${BVH_SUFFIX})"

      # Combine layout + schedule
      local LAYOUT_FILE
      LAYOUT_FILE=$(mktemp).bonsai
      cat "${LAYOUT_PATH}/${BVH_SUFFIX}/${LAYOUT}.bonsai" > "${LAYOUT_FILE}"
      cat "${PREFIX}/schedule.bonsai" >> "${LAYOUT_FILE}"
      echo "}" >> "${LAYOUT_FILE}"

      # Build compiler (incremental)
      cmake --build build -j"${NPROC}" > /dev/null 2>&1

      # Lower to CUDA header
      local MAIN_BONSAI
      MAIN_BONSAI=$(mktemp).bonsai
      cat "${PREFIX}/main.bonsai" > "${MAIN_BONSAI}"
      ./build/compiler -i "${MAIN_BONSAI}" -l "${LAYOUT_FILE}" -b cuda -o "${PREFIX}/${APPLICATION}.h"
      rm "${MAIN_BONSAI}"

      # Compile with nvcc
      nvcc -Iapps/wos -Iapps/rt -Iruntime/CUDA -O3 \
        "${PREFIX}/${MAIN_FILE}.cu" -o "${PREFIX}/${APPLICATION}_${LAYOUT}.out"

      # Run: WOS CUDA handles N runs internally
      local EXECUTABLE="${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      ./"${EXECUTABLE}" "${OBJECT}" "${PARTITION}" "${LAYOUT}" "${CPQ_N}" "${CUDA_CPQ_N_QUERIES}" \
        | tee -a "${OUTFILE}"

      # Clean up
      rm -f "${PREFIX}/${APPLICATION}.h"
      rm -f "${PREFIX}/${APPLICATION}_${LAYOUT}.out"
      rm -f "${LAYOUT_FILE}"
    done
    echo -e "---\n" >> "${OUTFILE}"
  done

  rm -f "${PREFIX}/${MAIN_FILE}.cu"
}

# ── Step 4: Embree Comparison ────────────────────────────────────────────────
run_embree_comparison() {
  log "Step 4: Embree Comparison - Ray Tracing (Figure 8)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/embree"

  local RAY_COUNTS=()
  for ((p=MIN_POWER; p<=MAX_POWER; p++)); do
    RAY_COUNTS+=($((2**p)))
  done
  local ARGV="${#RAY_COUNTS[@]}"
  for COUNT in "${RAY_COUNTS[@]}"; do
    ARGV="${ARGV} ${COUNT}"
  done

  local PREFIX="apps/rt/embree"
  local SCHEDULE="parallel"

  # Build Embree benchmark
  mkdir -p "${PREFIX}/build"
  cd "${PREFIX}/build"
  cmake .. > /dev/null 2>&1
  cmake --build . -j"${NPROC}" > /dev/null 2>&1
  cd "${ROOT_DIR}"

  for RAY_TYPE in "${RT_RAY_TYPES[@]}"; do
    generate_rays "${RAY_TYPE}"
    # Name as <machine>-<ray_type>.txt for hygiene_rt.py compatibility
    local OUTFILE="${RESULTS_DIR}/embree/${ARCH}-${RAY_TYPE}.txt"
    > "${OUTFILE}"

    for OBJECT in "${RT_OBJECTS[@]}"; do
      echo "${OBJECT}" >> "${OUTFILE}"
      for LAYOUT in "${EMBREE_LAYOUTS[@]}"; do
        # Embree lines need 4 fields for hygiene_rt.py: rt, target, layout, intersect
        # Determine intersect from layout (qbvh8i -> pc, bvh8i -> mt)
        local EMBREE_INTERSECT="mt"
        if [[ "${LAYOUT}" == *"qbvh8"* ]]; then
          EMBREE_INTERSECT="pc"
        fi
        echo "rt, embree, embree-${LAYOUT}, ${EMBREE_INTERSECT}" >> "${OUTFILE}"
        log "  Embree: ${OBJECT} / ${LAYOUT} (${RAY_TYPE})"

        for ((i=0; i < RT_N; i++)); do
          ./"${PREFIX}/build/main_trace" "${OBJECT}" "${LAYOUT}" "${SCHEDULE}" "${RAY_TYPE}" ${ARGV} \
            | tee -a "${OUTFILE}"
        done
      done
      echo -e "---\n" >> "${OUTFILE}"
    done
  done

  rm -rf "${PREFIX}/build"
  ok "Embree comparison complete. Results in ${RESULTS_DIR}/embree/"
}

# ── Step 5: FCL Comparison (Collision Detection) ─────────────────────────────
run_fcl_comparison() {
  log "Step 5: FCL Comparison - Collision Detection (Figure 9)"
  cd "${ROOT_DIR}"

  mkdir -p "${RESULTS_DIR}/cd"

  local APPLICATION="cd"
  local PREFIX="apps/cd/cpu"
  local OUTFILE="${RESULTS_DIR}/cd/fcl_comparison.txt"
  > "${OUTFILE}"

  for ((i=0; i<${#CD_OBJECTS_A[@]}; i++)); do
    local OBJECT_A="${CD_OBJECTS_A[i]}"
    local OBJECT_B="${CD_OBJECTS_B[i]}"
    for LAYOUT in "${CD_LAYOUTS[@]}"; do
      log "  CD: ${OBJECT_A} vs ${OBJECT_B} / ${LAYOUT}"
      echo "--- ${OBJECT_A} - ${OBJECT_B} - ${LAYOUT} ---" >> "${OUTFILE}"

      # Build compiler
      cmake --build build -j"${NPROC}" > /dev/null 2>&1

      # Lower to C++
      ./build/compiler -i "apps/${APPLICATION}/main.bonsai" \
        -l "${PREFIX}/${LAYOUT}.bonsai" -b cppx -o "${PREFIX}/${APPLICATION}"

      # Build executable
      cd "${PREFIX}"
      mkdir -p build && cd build

      local EXTRA_CMAKE_FLAGS=""
      if [[ "$(uname)" == "Linux" ]] && [[ -x "${CONDA_PREFIX:-}/bin/clang++" ]]; then
        EXTRA_CMAKE_FLAGS="-DCMAKE_CXX_COMPILER=${CONDA_PREFIX}/bin/clang++"
      fi
      if [[ -n "${CONDA_PREFIX:-}" ]]; then
        EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS} -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}"
        for _eigen_dir in "${CONDA_PREFIX}/share/eigen3/cmake" \
                          "${CONDA_PREFIX}/lib/cmake/eigen3" \
                          "${CONDA_PREFIX}/share/cmake/eigen3"; do
          if [[ -f "${_eigen_dir}/Eigen3Config.cmake" ]]; then
            EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS} -DEigen3_DIR=${_eigen_dir}"
            break
          fi
        done
      fi
      cmake -DLAYOUT="${LAYOUT}" .. ${EXTRA_CMAKE_FLAGS} > /dev/null 2>&1
      make -j"${NPROC}" main_library > /dev/null 2>&1
      cmake -DLAYOUT="${LAYOUT}" -DAPPLICATION="${APPLICATION}" .. ${EXTRA_CMAKE_FLAGS} > /dev/null 2>&1
      make -j"${NPROC}" > /dev/null 2>&1

      cd "${ROOT_DIR}"

      for ((k=0; k < CD_N; k++)); do
        ./"${PREFIX}/build/${APPLICATION}_${LAYOUT}.out" "${OBJECT_A}" "${OBJECT_B}" \
          | tee -a "${OUTFILE}"
      done

      rm -f "${PREFIX}/${APPLICATION}.h" "${PREFIX}/${APPLICATION}.cpp"
      rm -rf "${PREFIX}/build"
    done
  done

  ok "FCL comparison complete. Results in ${RESULTS_DIR}/cd/"
}

# ── Step 6: FCPW Comparison (Closest Point Queries) ─────────────────────────
run_fcpw_comparison() {
  log "Step 6: FCPW Comparison - Closest Point Queries (Figure 9)"
  cd "${ROOT_DIR}"

  # FCPW comparison is built into the WOS executable (step 3).
  # The WOS benchmark already includes FCPW timing data in its output.
  # If step 3 was run, results are already available.

  if [[ -f "${RESULTS_DIR}/cpq/dse.txt" ]]; then
    ok "FCPW comparison data already collected in step 3"
  else
    log "Running CPQ benchmarks (includes FCPW comparison)..."
    run_dse_cpq
  fi
}

# ── Step 7: Convert raw data to CSV and generate figures ─────────────────────
generate_figures() {
  log "Step 7: Converting raw data to CSV and generating figures..."
  cd "${ROOT_DIR}"

  # Activate venv if it exists (needed for standalone --step 7)
  if [[ -d "${ROOT_DIR}/artifact/.venv" ]]; then
    source "${ROOT_DIR}/artifact/.venv/bin/activate"
  fi

  local SCRIPTS="apps/scripts"
  mkdir -p "${RESULTS_DIR}/figures"

  local RT_CSV="${RESULTS_DIR}/rt-results.csv"
  local CPQ_CSV="${RESULTS_DIR}/cpq-results.csv"
  local CD_CSV="${RESULTS_DIR}/cd-results.csv"

  # ── 7a: Hygiene RT data (raw .txt -> CSV) ──────────────────────────────
  rm -f "${RT_CSV}"
  if [[ -d "${RESULTS_DIR}/rt" ]]; then
    log "  Converting RT results to CSV..."
    # Process CPU results (<arch>-*.txt) and CUDA results (cuda-*.txt)
    for txt_file in "${RESULTS_DIR}/rt/"*.txt; do
      [[ -f "${txt_file}" ]] || continue
      log "    hygiene_rt.py ${txt_file}"
      cd "${SCRIPTS}"
      python3 hygiene_rt.py "${txt_file}" "${RT_CSV}" || \
        log "    WARNING: hygiene_rt.py failed on ${txt_file}"
      cd "${ROOT_DIR}"
    done
  fi

  # ── 7b: Hygiene Embree data (append to same RT CSV) ────────────────────
  if [[ -d "${RESULTS_DIR}/embree" ]]; then
    log "  Converting Embree results to CSV..."
    for txt_file in "${RESULTS_DIR}/embree/"*.txt; do
      [[ -f "${txt_file}" ]] || continue
      log "    hygiene_rt.py ${txt_file}"
      cd "${SCRIPTS}"
      python3 hygiene_rt.py "${txt_file}" "${RT_CSV}" || \
        log "    WARNING: hygiene_rt.py failed on ${txt_file} (Embree format may need manual handling)"
      cd "${ROOT_DIR}"
    done
  fi

  # ── 7c: Hygiene CD data ────────────────────────────────────────────────
  rm -f "${CD_CSV}"
  if [[ -d "${RESULTS_DIR}/cd" ]] && [[ -f "${RESULTS_DIR}/cd/fcl_comparison.txt" ]]; then
    log "  Converting CD results to CSV..."
    cd "${SCRIPTS}"
    python3 hygiene_cd.py "${RESULTS_DIR}/cd/fcl_comparison.txt" "${CD_CSV}" \
      --machine "${ARCH}" || \
      log "  WARNING: hygiene_cd.py failed"
    cd "${ROOT_DIR}"
  fi

  # ── 7d: Hygiene CPQ data ───────────────────────────────────────────────
  rm -f "${CPQ_CSV}"
  if [[ -d "${RESULTS_DIR}/cpq" ]]; then
    log "  Converting CPQ results to CSV..."
    # Process CPU CPQ results
    if [[ -f "${RESULTS_DIR}/cpq/dse.txt" ]]; then
      cd "${SCRIPTS}"
      python3 hygiene_cpq.py "${RESULTS_DIR}/cpq/dse.txt" "${CPQ_CSV}" \
        --machine "${ARCH}" || \
        log "  WARNING: hygiene_cpq.py failed (CPU)"
      cd "${ROOT_DIR}"
    fi
    # Process CUDA CPQ results
    if [[ -f "${RESULTS_DIR}/cpq/cuda-dse.txt" ]]; then
      cd "${SCRIPTS}"
      python3 hygiene_cpq.py "${RESULTS_DIR}/cpq/cuda-dse.txt" "${CPQ_CSV}" \
        --machine "cuda" || \
        log "  WARNING: hygiene_cpq.py failed (CUDA)"
      cd "${ROOT_DIR}"
    fi
  fi

  # ── 7e: Generate plots from CSVs ───────────────────────────────────────
  # These mirror the exact commands used to produce the paper's figures.
  local OUT="${RESULTS_DIR}/figures"

  if [[ -f "${RT_CSV}" ]]; then
    # -- DSE Pareto frontier (Figure 7) --
    log "  Generating DSE Pareto frontier (Figure 7)..."
    python3 "${SCRIPTS}/collect_dse.py" "${RT_CSV}" \
      --machines "${ARCH}" \
      --output_filename "${OUT}/dse" || \
      log "  WARNING: DSE plot generation failed"

    # -- Data-dependent: Pareto frontier (Figure 6a) --
    log "  Generating data-dependent Pareto plot..."
    python3 "${SCRIPTS}/collect_dse.py" "${RT_CSV}" \
      --machines "${ARCH}" --ray-types primary \
      --scenes lucy power-plant \
      --layout-groups "bvh2:pbrt-q16,pbrt-q16-soaos,pbrt,sg-eq" \
      --label-dominated --remove-legend --memory-type bvh \
      --output_filename "${OUT}/data-dependent1" || \
      log "  WARNING: data-dependent1 plot failed"

    # -- Data-dependent: Speedup bar chart (Figure 6b) --
    log "  Generating data-dependent speedup bar chart..."
    python3 "${SCRIPTS}/speedup_bar_chart.py" "${RT_CSV}" \
      --scene lucy --machine "${ARCH}" \
      --layouts sg-eq --baseline pbrt-q16 \
      --ray-types primary secondary \
      --output "${OUT}/data-dependent2" || \
      log "  WARNING: data-dependent2 plot failed"

    # -- Machine-dependent: Pareto frontier (Figure 6c) --
    log "  Generating machine-dependent Pareto plot..."
    python3 "${SCRIPTS}/collect_dse.py" "${RT_CSV}" \
      --machines "${ARCH}" --ray-types secondary \
      --scenes sponza \
      --layout-groups "bvh2:pbrt,ptr,pbrt-q16-soaos,pbrt-q16,sg-eq" \
      --memory-type bvh \
      --output_filename "${OUT}/machine-dependent1" || \
      log "  WARNING: machine-dependent1 plot failed"

    # -- Embree comparison (Figure 8) --
    log "  Generating Embree comparison (Figure 8)..."
    python3 "${SCRIPTS}/embree_comparison_rt.py" "${RT_CSV}" \
      --scenes "${RT_OBJECTS[@]}" \
      --layouts bvh8-q8-ci bvh8-q16-ci bvh8 embree-bvh8i embree-qbvh8i \
      --machines "${ARCH}" \
      --best \
      --output "${OUT}/embree-comparison-rt" || \
      log "  WARNING: Embree comparison plot failed"

    # -- Novel layout dominating table --
    log "  Generating novel layout dominating table..."
    python3 "${SCRIPTS}/is_pareto_optimal.py" "${RT_CSV}" pbrt-q16 bvh2 \
      --latex-output "${OUT}/novel-layout-table.tex" || \
      log "  WARNING: novel layout table failed"
  fi

  if [[ -f "${CD_CSV}" ]]; then
    # -- FCL comparison (Figure 9a) --
    log "  Generating FCL comparison (Figure 9a)..."
    for ((idx=0; idx<${#CD_OBJECTS_A[@]}; idx++)); do
      local S1="${CD_OBJECTS_A[idx]}"
      local S2="${CD_OBJECTS_B[idx]}"
      python3 "${SCRIPTS}/fcl_comparison_cd.py" "${CD_CSV}" \
        --layouts "${CD_LAYOUTS[@]}" \
        --scene1 "${S1}" --scene2 "${S2}" \
        --machines "${ARCH}" \
        --output_filename "${OUT}/fcl-comparison-cd_${S1}_${S2}" || \
        log "  WARNING: FCL comparison plot failed for ${S1} vs ${S2}"
    done
  fi

  if [[ -f "${CPQ_CSV}" ]]; then
    # -- FCPW comparison (Figure 9b) --
    log "  Generating FCPW comparison (Figure 9b)..."
    python3 "${SCRIPTS}/fcpw_comparison_cpq.py" "${CPQ_CSV}" \
      --scenes "${CPQ_OBJECTS[@]}" \
      --output "${OUT}/fcpw-comparison-cpq" || \
      log "  WARNING: FCPW comparison plot failed"
  fi

  if [[ -f "${RT_CSV}" ]] && [[ -f "${CPQ_CSV}" ]]; then
    # -- Algorithm-dependent (Figure 6d) --
    log "  Generating algorithm-dependent plot (Figure 6d)..."
    python3 "${SCRIPTS}/collect_dse.py" "${RT_CSV}" \
      --compare-cpq "${CPQ_CSV}" \
      --machines "${ARCH}" --ray-types primary \
      --layouts bvh2 \
      --scenes san-miguel-x35-y22-z47 \
      --memory-type bvh \
      --output_filename "${OUT}/algorithm-dependent" || \
      log "  WARNING: algorithm-dependent plot failed"
  fi

  ok "Figures generated in ${OUT}/"
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
  detect_platform
  mkdir -p "${RESULTS_DIR}"

  if [[ "${SHORT}" == true ]]; then
    log "=== SHORT RUN MODE: quick smoke test ==="
  fi
  if [[ "${GPU}" == true ]]; then
    log "=== GPU MODE: CUDA benchmarks ==="
  fi

  if should_run 0; then install_deps;          fi
  if should_run 1; then build_compiler;         fi
  if [[ "${GPU}" == true ]]; then
    if should_run 2; then run_dse_rt_cuda;        fi
    if should_run 3; then run_dse_cpq_cuda;       fi
    if should_run 4; then log "Step 4: Embree comparison — skipped (CPU only, not available with --gpu)"; fi
    if should_run 5; then log "Step 5: FCL comparison — skipped (CPU only, not available with --gpu)"; fi
    if should_run 6; then log "Step 6: FCPW comparison — skipped (CPU only, not available with --gpu)"; fi
  else
    if should_run 2; then run_dse_rt;             fi
    if should_run 3; then run_dse_cpq;            fi
    if should_run 4; then run_embree_comparison;  fi
    if should_run 5; then run_fcl_comparison;     fi
    if should_run 6; then run_fcpw_comparison;    fi
  fi
  if should_run 7; then generate_figures;       fi

  log "=== All steps complete ==="
  log "Results are in: ${RESULTS_DIR}/"
  log "  rt/       - Ray tracing DSE results"
  log "  cpq/      - Closest point query DSE results"
  log "  embree/   - Embree comparison results"
  log "  cd/       - FCL collision detection comparison results"
  log "  figures/  - Generated PDF figures"
}

main
