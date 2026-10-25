#!/bin/bash

set -euo pipefail

# Build the program with a schedule and run it against WoSX on a mesh.
#
#     apps/wosx/compare.sh [--schedule S[,S...]] [driver options] <mesh.ply[.gz]>
#
# Two comparison points by default, `fcpw4w16,fcpw4w16-threads`.
# schedules/fcpw4w16.bonsai runs the walk as WoSX's loop and the distance
# query as FCPW's findClosestPoint over FCPW's four-wide tree with
# sixteen-lane leaves -- the tree WoSX's boundary handler builds when asked
# to vectorize -- one thread against one thread, and its table says whether
# the compiler makes of that structure what WoSX's and FCPW's hand-written
# code is; schedules/fcpw4w16-threads.bonsai is the same with the points
# spread over the threads on both sides (WoSX's tbb::parallel_for solve
# against the parfor bound to the CPU threads), pinned to the performance
# cores; schedules/fcpw4w16-persistent.bonsai is the threaded one with the
# points handed to sixteen persistent workers instead of TBB's partition,
# run the same way. The packet schedules (fcpw4w16-packet, the walks as the
# lanes; fcpw4w16-packet-points, the points) are run the single-threaded
# way. A schedule named fcpwb* runs over WoSX's default tree instead --
# FCPW's scalar binary BVH, FCPW_USE_ENOKI off, the layout in
# apps/rtq/schedules/layouts/fcpwb.bonsai -- and the driver is built to have
# FCPW build that tree for WoSX's side too (WOSX_BINARY_BVH). A schedule named
# fcpwb2* runs over the same binary tree declared as FCPW traverses it,
# each node holding its children's boxes (schedules/layouts/fcpwb2.bonsai,
# WOSX_BINARY_BVH2). A schedule named *-gpu* is the GPU comparison: WoSX's
# GPU solver -- its walk on spheres as a Slang compute shader through
# slang-rhi, over FCPW's binary tree, the one tree FCPW puts on a GPU --
# against the schedule's kernels, both on the card; the driver is built
# with WOSX_GPU against slang-rhi (deps/slang-rhi-build, built once by
# build_slang_rhi.sh) and run with --gpu, pinned like a single-thread run
# (the host thread only; the GPU is the GPU). Each schedule named is built
# and run in turn, its table under a `=== schedule` line; the other
# arguments go to the driver (wosx_hook.cpp: --side, --walks, --epsilon,
# --repeats, --open, --image, --gpu-reference ...).
#
# Run from the repository root, inside the `bonsai` conda environment. WoSX
# is an optional dependency: a submodule at deps/wosx, header-only, with
# FCPW (and FCPW's enoki and Eigen) and pcg32 as submodules of its own, so
# there is nothing to build; TBB, which WoSX's solver includes, comes from
# the conda environment. Without the checkout this stops and says so;
# nothing else in the repository needs WoSX. The GPU comparison needs one
# more thing built, slang-rhi (FCPW's submodule deps/slang-rhi, and the
# Slang release it fetches): build_slang_rhi.sh, once.
#
# The measurement is single-threaded and pinned, as apps/rtq's is: both
# sides solve their points as one plain loop on one core (WoSX's
# runSingleThreaded; the program's parfor over the points is left unbound
# by the schedule), on the core the kernel ranks best, with the memory of
# its NUMA node, under numactl. RTQ_CPUS=<cpu> pins to that CPU instead.
#
# Environment: BONSAI_BUILD_DIR names the compiler's build directory (by
# default the first of `build`, `build-*` that CMake has configured with the
# LLVM the compiler needs, which is said when it is picked); BONSAI_CXX a
# clang++ (default `clang++`; the generated header needs clang's
# ext_vector_type); RTQ_CPUS as above; WOSX_KEEP set keeps the built
# program and driver (wosx.out, wosx.ll, ...) after the run, for a profile.
if [[ "$(pwd)" == */apps/wosx ]]; then
  cd ../..
fi

PREFIX="apps/wosx"
SCHEDULES="fcpw4w16,fcpw4w16-threads"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedule)
      if [[ $# -lt 2 ]]; then
        echo "--schedule needs a name from $PREFIX/schedules/, or several separated by commas" >&2
        exit 1
      fi
      SCHEDULES="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

IFS=, read -r -a SCHEDULE_LIST <<< "$SCHEDULES"
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
    exit 1
  fi
done

WOSX="deps/wosx"
FCPW="$WOSX/deps/fcpw"
if [[ ! -f "$WOSX/include/wosx/point_estimation/walk_on_spheres.h" ||
      ! -f "$WOSX/deps/pcg32/pcg32.h" ||
      ! -f "$FCPW/include/fcpw/fcpw.h" || ! -f "$FCPW/deps/enoki/include/enoki/array.h" ||
      ! -f "$FCPW/deps/eigen/Eigen/Core" ]]; then
  echo "WoSX is not checked out with the submodules it needs: this app compares against it." >&2
  echo "    git submodule update --init deps/wosx" >&2
  echo "    git -C deps/wosx submodule update --init deps/fcpw deps/pcg32" >&2
  echo "    git -C deps/wosx/deps/fcpw submodule update --init deps/enoki deps/eigen" >&2
  exit 1
fi

BONSAI_CXX="${BONSAI_CXX:-clang++}"
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header: it needs clang's" >&2
  echo "ext_vector_type. Set BONSAI_CXX to a clang++." >&2
  exit 1
fi
# zlib (for gzipped meshes) and TBB (which WoSX's solver includes) live
# beside the compiler in the conda environment.
TOOLCHAIN_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
if [[ ! -f "$TOOLCHAIN_PREFIX/include/oneapi/tbb/parallel_for.h" ]]; then
  echo "TBB is not in the toolchain's prefix ($TOOLCHAIN_PREFIX): activate the bonsai conda environment" >&2
  exit 1
fi

if [[ -z "${BONSAI_BUILD_DIR:-}" ]]; then
  for d in build build-*; do
    if [[ -f "$d/CMakeCache.txt" ]] && grep -q '^LLVM_DIR:[A-Z]*=.*/cmake' "$d/CMakeCache.txt"; then
      BONSAI_BUILD_DIR="$d"
      break
    fi
  done
  if [[ -z "${BONSAI_BUILD_DIR:-}" ]]; then
    echo "no configured build directory among build, build-*: configure one" >&2
    echo "(cmake -S . -B build -DLLVM_DIR=<llvm>/lib/cmake/llvm) or set BONSAI_BUILD_DIR" >&2
    exit 1
  fi
  echo "compiler build directory: $BONSAI_BUILD_DIR (BONSAI_BUILD_DIR names another)"
fi
cmake --build "$BONSAI_BUILD_DIR" -j

# The program with one schedule, as $PREFIX/wosx.out. The flags are
# apps/rtq/compare.sh's: `-p ssa` because sort and loopify are SSA
# rewrites; `--no-heap` refuses any allocation in the compiled program;
# `--ffp-contract` fuses `a * b + c` as FCPW's build does (enoki's dot is
# an explicit fmadd, and its build passes -ffp-contract=fast). The driver
# is compiled against WoSX's headers and the FCPW WoSX pins, with the flags
# FCPW's own CMake would set for a vectorized build on this machine --
# FCPW_USE_ENOKI, FCPW_SIMD_WIDTH 16 and the four-wide branching of the
# schedule's tree, -march=native for enoki's vectors, NDEBUG as a release
# build, enoki's -fno-math-errno and -ffp-contract=fast.
build_schedule() {
  local schedule="$1"
  local flags=(-p ssa --no-heap --ffp-contract)
  # The program: the walk on spheres (wosx.bonsai) unless the schedule is
  # named wost-*, which is the walk on stars (wost.bonsai, over apps/rtq's
  # silhouette query and the FCPW scalar metric's cone and edge arithmetic),
  # compared against WoSX's WalkOnStars by a driver built with WOSX_STARS;
  # the rest of the name says the tree and how it is run, as below.
  local program="wosx" tree_name="$schedule"
  local program_inputs=(-i $PREFIX/wosx.bonsai)
  local program_defines=()
  if [[ "$schedule" == wost-* ]]; then
    program="wost"
    tree_name="${schedule#wost-}"
    program_inputs=(-i $PREFIX/wost.bonsai)
    program_defines=(-DWOSX_STARS)
  fi
  # The arithmetic of the FCPW build the schedule's tree is compared
  # against, compiled before the program as apps/rtq compiles it before
  # rtq.bonsai: metrics/fcpw.bonsai, the vectorized build's (enoki's
  # reciprocal estimate), for the four-wide tree; metrics/fcpw_scalar.bonsai,
  # the scalar build's and the GPU's (true divisions), for the binary tree.
  local metric="fcpw"
  if [[ "$tree_name" == fcpwb* ]]; then
    metric="fcpw_scalar"
  fi
  local inputs=(-i apps/rtq/elements.bonsai -i "apps/rtq/metrics/$metric.bonsai"
                "${program_inputs[@]}" -i "$PREFIX/schedules/$schedule.bonsai")
  "./$BONSAI_BUILD_DIR/compiler" -p ssa "${inputs[@]}" -o $PREFIX/wosx.bir
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b llvm -o $PREFIX/wosx.ll
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b cpp -o $PREFIX/wosx

  # The tree the schedule's name says: fcpw<branching>w<width> is FCPW's
  # vectorized Mbvh at that branching and width (apps/rtq's layouts), fcpwb
  # FCPW's scalar binary BVH -- WoSX's default, FCPW_USE_ENOKI off
  # (apps/rtq/schedules/layouts/fcpwb.bonsai, the driver's WOSX_BINARY_BVH),
  # fcpwb2 the same binary tree with each node holding its children's boxes
  # (layouts/fcpwb2.bonsai, WOSX_BINARY_BVH2). The suffixes say how it is
  # run: -threads, -persistent and -packet* as above, -gpu* on the GPU. The
  # walk on stars runs on the binary trees alone (layouts/fcpw_stars.bonsai;
  # its own header copies them, so WOSX_BINARY_BVH is not set beside
  # WOSX_STARS).
  local branching width defines
  local tail='(-threads|-persistent|-packet(-[a-z0-9]+)*|-gpu(-[a-z0-9]+)*)?'
  if [[ "$tree_name" =~ ^fcpw([48])w([0-9]+)t?$tail$ && "$program" == wosx ]]; then
    branching="${BASH_REMATCH[1]}"
    width="${BASH_REMATCH[2]}"
    defines=(-DNDEBUG -DFCPW_USE_ENOKI "-DFCPW_SIMD_WIDTH=$width")
    if [[ "$branching" == 8 ]]; then
      defines+=(-DFCPW_USE_EIGHT_WIDE_BRANCHING)
    fi
  elif [[ "$tree_name" =~ ^fcpwb2$tail$ && "$program" == wosx ]]; then
    branching=2
    width=1
    defines=(-DNDEBUG -DWOSX_BINARY_BVH2)
  elif [[ "$tree_name" =~ ^fcpwb$tail$ ]]; then
    branching=2
    width=1
    defines=(-DNDEBUG)
    if [[ "$program" == wosx ]]; then
      defines+=(-DWOSX_BINARY_BVH)
    fi
  else
    echo "a schedule is named fcpw<branching>w<width>[t], fcpwb or fcpwb2, then [-threads|-persistent|-packet[-variant...]|-gpu[-variant...]], after FCPW's tree (fcpw4w16, fcpw4w16-threads, fcpw4w16-packet, fcpwb, fcpwb-packet, fcpwb2-gpu, ...), or wost-fcpwb[-...] for the walk on stars: $schedule" >&2
    exit 1
  fi
  defines+=(${program_defines[@]+"${program_defines[@]}"})
  # A schedule on the GPU: the driver runs WoSX's GPU solver as the
  # reference, which is compiled against slang-rhi's headers and the Slang
  # release slang-rhi fetched, and linked with both (the Slang library is
  # found at run time through the rpath; Slang itself finds CUDA's NVRTC).
  local gpu=0
  local gpu_defines=() gpu_includes=() gpu_libs=()
  if [[ "$schedule" == *-gpu* ]]; then
    gpu=1
    local rhi_src="$FCPW/deps/slang-rhi" rhi_build="deps/slang-rhi-build"
    local slang_src="deps/slang-rhi-build/_deps/slang-src"
    local rhi_lib slang_lib
    rhi_lib="$(ls "$rhi_build"/libslang-rhi.a "$rhi_build"/libslang-rhi.so 2>/dev/null | head -n 1 || true)"
    # The compiler library is libslang-compiler.so since Slang 2025.21 and
    # libslang.so before; slang-rhi links whichever the release has, the
    # newer name first, and so does this.
    slang_lib="$(ls "$slang_src"/lib/libslang-compiler.so "$slang_src"/lib/libslang.so 2>/dev/null | head -n 1 || true)"
    if [[ ! -f "$rhi_src/include/slang-rhi.h" || -z "$rhi_lib" || -z "$slang_lib" ||
          ! -f "$slang_src/include/slang.h" ]]; then
      echo "a GPU schedule compares against WoSX's GPU solver, which runs on slang-rhi:" >&2
      echo "build it once with apps/wosx/build_slang_rhi.sh (into $rhi_build)" >&2
      exit 1
    fi
    # WoSX's GPU headers reach its nearest-neighbour finder, which is
    # nanoflann's, another header-only submodule of WoSX.
    if [[ ! -f "$WOSX/deps/nanoflann/include/nanoflann.hpp" ]]; then
      echo "WoSX's GPU headers need its nanoflann submodule:" >&2
      echo "    git -C deps/wosx submodule update --init deps/nanoflann" >&2
      exit 1
    fi
    gpu_defines=(-DWOSX_GPU -DWOSX_USE_GPU -DFCPW_USE_GPU)
    # slang-rhi's headers, its configuration header the build generated,
    # the Slang release's, and nanoflann's.
    gpu_includes=(-isystem "$rhi_src/include" -isystem "$rhi_build/include"
                  -isystem "$slang_src/include" -isystem "$WOSX/deps/nanoflann/include")
    # slang-rhi's own library, then the resources its build embeds through
    # CMakeRC (libslang-rhi-resources.a, which the first refers to), then
    # Slang's shared library, found at run time through the rpath.
    local rhi_resources
    rhi_resources="$(ls "$rhi_build"/libslang-rhi-resources.a 2>/dev/null | head -n 1 || true)"
    gpu_libs=("$rhi_lib" ${rhi_resources:+"$rhi_resources"} "$slang_lib"
              -Wl,-rpath,"$(cd "$slang_src/lib" && pwd)" -ldl -lpthread)
  fi
  # The driver compiled once per (header, defines) within a run and linked
  # per schedule, keyed by the generated header's checksum: compiling it
  # against WoSX's and FCPW's headers takes longer than a measurement.
  local key
  key="$(cat $PREFIX/wosx.h | md5sum | cut -c1-16)-$program-$branching-$width-$gpu"
  if [[ ! -f "$PREFIX/.wosx_hook-$key.o" ]]; then
    "$BONSAI_CXX" -std=c++20 -O3 -march=native -fno-math-errno -ffp-contract=fast \
        "${defines[@]}" ${gpu_defines[@]+"${gpu_defines[@]}"} -I. -I$PREFIX \
        -isystem "$WOSX/include" -isystem "$WOSX/deps/pcg32" \
        -isystem "$FCPW/include" -isystem "$FCPW/deps/enoki/include" \
        -isystem "$FCPW/deps/eigen" -isystem "$TOOLCHAIN_PREFIX/include" \
        ${gpu_includes[@]+"${gpu_includes[@]}"} \
        -c $PREFIX/wosx_hook.cpp -o "$PREFIX/.wosx_hook-$key.o"
  fi
  "$BONSAI_CXX" "$PREFIX/.wosx_hook-$key.o" $PREFIX/wosx.o \
      -L"$TOOLCHAIN_PREFIX/lib" -Wl,-rpath,"$TOOLCHAIN_PREFIX/lib" -ltbb -lz \
      ${gpu_libs[@]+"${gpu_libs[@]}"} \
      -o $PREFIX/wosx.out
}

# The cores to pin to. One thread: the physical core the kernel ranks
# highest, one hardware thread of it, with the memory of its node, chosen
# as apps/rtq/compare.sh chooses it (its comment says why). The threads (a
# `-threads` schedule): the performance cores, which are the cores that
# share that core's last-level cache -- on a part with two chiplets the
# frequency chiplet, all its hardware threads -- so that both pools, WoSX's
# and the program's, run on the same cores and none on the slower ones.
# RTQ_CPUS names a set instead, for either.
if ! command -v numactl >/dev/null 2>&1; then
  echo "numactl is needed to pin the measurement to one core" >&2
  exit 1
fi
if [[ -n "${RTQ_CPUS:-}" ]]; then
  CPU="$RTQ_CPUS"
  HOW="RTQ_CPUS"
else
  HOW="the lowest-numbered physical core (no preferred-core ranking here)"
  CPU="$(for d in /sys/devices/system/cpu/cpu[0-9]*; do
      cpu="${d##*cpu}"
      siblings="$d/topology/core_cpus_list"
      [[ -f "$siblings" ]] || siblings="$d/topology/thread_siblings_list"
      first="$(sed 's/[,-].*//' "$siblings")"
      [[ "$first" == "$cpu" ]] || continue
      rank=0
      if [[ -f "$d/cpufreq/amd_pstate_prefcore_ranking" ]]; then
        rank="$(cat "$d/cpufreq/amd_pstate_prefcore_ranking")"
      fi
      printf '%s %s\n' "$rank" "$cpu"
    done | sort -k1,1nr -k2,2n | head -n 1 | awk '{print $2}')"
  if [[ -f "/sys/devices/system/cpu/cpu${CPU%%,*}/cpufreq/amd_pstate_prefcore_ranking" ]]; then
    HOW="the physical core with the highest amd_pstate_prefcore_ranking"
  fi
fi
NODE="$(basename "$(ls -d /sys/devices/system/cpu/cpu${CPU%%,*}/node[0-9]* 2>/dev/null | head -n 1)" 2>/dev/null)"
NODE="${NODE#node}"
CORES="$CPU"
CORES_HOW="$HOW"
if [[ -z "${RTQ_CPUS:-}" ]]; then
  L3="/sys/devices/system/cpu/cpu${CPU%%,*}/cache/index3/shared_cpu_list"
  if [[ -f "$L3" ]]; then
    CORES="$(cat "$L3")"
    CORES_HOW="the cores sharing cpu $CPU's last-level cache"
  fi
fi

STATUS=0
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  echo "=== schedule $SCHEDULE ($PREFIX/schedules/$SCHEDULE.bonsai)"
  build_schedule "$SCHEDULE"
  if [[ "$SCHEDULE" == *-gpu* ]]; then
    # Both sides on the GPU; the host thread pinned as a single-thread run
    # is, so that what it does between the kernels is the same every time.
    echo "pinned to cpu $CPU, memory of NUMA node ${NODE:-0}: $HOW (the host thread; both sides run on the GPU)"
    numactl --physcpubind="$CPU" --membind="${NODE:-0}" ./$PREFIX/wosx.out --gpu "$@" || STATUS=$?
  elif [[ "$SCHEDULE" == *-threads || "$SCHEDULE" == *-persistent ]]; then
    echo "pinned to cpus $CORES, memory of NUMA node ${NODE:-0}: $CORES_HOW"
    numactl --physcpubind="$CORES" --membind="${NODE:-0}" ./$PREFIX/wosx.out --threads "$@" || STATUS=$?
  else
    echo "pinned to cpu $CPU, memory of NUMA node ${NODE:-0}: $HOW"
    numactl --physcpubind="$CPU" --membind="${NODE:-0}" ./$PREFIX/wosx.out "$@" || STATUS=$?
  fi
  # WOSX_KEEP leaves the built program and driver in place (wosx.out, wosx.ll,
  # ...) for a profile or a read of the code after the run.
  if [[ -z "${WOSX_KEEP:-}" ]]; then
    rm -f $PREFIX/wosx.bir $PREFIX/wosx.ll $PREFIX/wosx.h $PREFIX/wosx.o $PREFIX/wosx.out
  fi
done
rm -f $PREFIX/.wosx_hook-*.o
exit $STATUS
