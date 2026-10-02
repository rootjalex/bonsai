#!/bin/bash

set -euo pipefail

# Build the program with each schedule and run it against Embree on a mesh.
#
#     apps/rtq/compare.sh [--schedule S[,S...]] [--side N] [--repeats N] <mesh.ply[.gz]>
#
# Two comparison points by default, `embree,tuned`: schedules/embree.bonsai
# follows Embree's traversal step for step and says whether the compiler
# makes of that structure what Embree's hand-written code is; schedules/
# tuned.bonsai departs from it where a step measures worse here and says what
# the schedule language can do beyond it. Each is built and run in turn, its
# table under a `=== schedule` line.
#
# Run from the repository root, inside the `bonsai` conda environment. Embree
# is an optional dependency: a submodule at deps/embree, built by
# build_embree.sh into deps/embree-install. Without both this stops and says
# so; nothing else in the repository needs Embree.
#
# What is compared is in rtq_hook.cpp: both sides traverse the tree Embree's
# builder makes, over the same rays, each the least of several runs after a
# warm-up; every answer is checked against Embree's.
#
# The measurement is single-threaded and pinned. Both sides run their rays
# as one plain loop on one core -- the program's parfor over the rays is
# left unbound by the schedule and so is a sequential loop, and Embree's
# rays go through a loop of rtcIntersect1 -- and that core is the one the
# kernel ranks best: on an AMD part with preferred-core ranking
# (amd_pstate_prefcore_ranking), a core of the frequency chiplet rather than
# the cache one; without a ranking, the lowest-numbered physical core -- with
# the memory of its NUMA node, under numactl. A multi-threaded number mixes
# the kernel's cost with a pool's balancing, and the two sides can land on
# unlike cores; it is not the comparison. RTQ_CPUS=<cpu> pins to that CPU
# instead.
#
# Environment: BONSAI_BUILD_DIR names the compiler's build directory (by
# default the first of `build`, `build-*` that CMake has configured with the
# LLVM the compiler needs, which is said when it is picked); BONSAI_CXX a
# clang++ (default `clang++`; the generated header needs clang's
# ext_vector_type); RTQ_CPUS as above.
if [[ "$(pwd)" == */apps/rtq ]]; then
  cd ../..
fi

PREFIX="apps/rtq"
SCHEDULES="embree,tuned"
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

if [[ ! -f deps/embree/CMakeLists.txt ]]; then
  echo "Embree is not checked out: this app compares against it and needs it." >&2
  echo "    git submodule update --init deps/embree" >&2
  echo "    $PREFIX/build_embree.sh" >&2
  exit 1
fi
EMBREE="deps/embree-install"
if [[ ! -f "$EMBREE/include/embree4/rtcore.h" || ! -f "$EMBREE/lib/libembree4.so" ]]; then
  echo "Embree is checked out but not built: run $PREFIX/build_embree.sh" >&2
  exit 1
fi
IFS=, read -r -a SCHEDULE_LIST <<< "$SCHEDULES"
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
    exit 1
  fi
done

BONSAI_CXX="${BONSAI_CXX:-clang++}"
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header: it needs clang's" >&2
  echo "ext_vector_type. Set BONSAI_CXX to a clang++." >&2
  exit 1
fi
# zlib, for gzipped meshes, lives beside the compiler in the conda
# environment.
TOOLCHAIN_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"

# The compiler's build directory: the one named, or else the first of the
# usual ones that is configured -- whose CMake cache found an LLVM. A
# directory configured before the LLVM the compiler needs was installed has
# `LLVM_DIR-NOTFOUND` in its cache and fails to configure when built, so it
# is passed over rather than tried.
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

# The program with one schedule, as $PREFIX/rtq.out. `-p ssa` because sort
# and loopify are SSA rewrites; `--no-heap` refuses any allocation in the
# compiled program; `--ffp-contract` fuses `a * b + c` as Embree's build does
# (its node test is written with msub, its dot products with madd).
build_schedule() {
  local schedule="$1"
  local flags=(-p ssa --no-heap --ffp-contract)
  local inputs=(-i $PREFIX/rtq.bonsai -i "$PREFIX/schedules/$schedule.bonsai")
  "./$BONSAI_BUILD_DIR/compiler" -p ssa "${inputs[@]}" -o $PREFIX/rtq.bir
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b llvm -o $PREFIX/rtq.ll
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b cpp -o $PREFIX/rtq

  "$BONSAI_CXX" -std=c++20 -O3 -I. -I$PREFIX -isystem "$EMBREE/include" \
      $PREFIX/rtq_hook.cpp $PREFIX/rtq.o \
      -L"$EMBREE/lib" -Wl,-rpath,"$EMBREE/lib" -lembree4 \
      -L"$TOOLCHAIN_PREFIX/lib" -Wl,-rpath,"$TOOLCHAIN_PREFIX/lib" -lz \
      -o $PREFIX/rtq.out
}

# The core to pin to (see the top of the file): the physical core the kernel
# ranks highest, one hardware thread of it. A CPU is the first of its core
# when its number is the first in its core's sibling list.
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
echo "pinned to cpu $CPU, memory of NUMA node ${NODE:-0}: $HOW"

STATUS=0
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  echo "=== schedule $SCHEDULE ($PREFIX/schedules/$SCHEDULE.bonsai)"
  build_schedule "$SCHEDULE"
  numactl --physcpubind="$CPU" --membind="${NODE:-0}" ./$PREFIX/rtq.out "$@" || STATUS=$?
  rm -f $PREFIX/rtq.bir $PREFIX/rtq.ll $PREFIX/rtq.h $PREFIX/rtq.o $PREFIX/rtq.out
done
exit $STATUS
