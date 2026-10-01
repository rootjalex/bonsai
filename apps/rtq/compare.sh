#!/bin/bash

set -euo pipefail

# Build the program with a schedule and run it against Embree on a mesh.
#
#     apps/rtq/compare.sh [--schedule S] [--side N] [--repeats N] [--threads N] <mesh.ply[.gz]>
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
# The measurement is single-threaded and pinned: by default (--threads 1)
# both sides run their rays as one plain loop on one core, and that core is
# the one the kernel ranks best -- on an AMD part with preferred-core ranking
# (amd_pstate_prefcore_ranking), a core of the frequency chiplet rather than
# the cache one; without a ranking, the lowest-numbered physical core -- with
# the memory of its NUMA node, under numactl. A multi-threaded number mixes
# the kernel's cost with the pool's balancing, and the two sides can land on
# unlike cores; it is not the comparison. --threads N pins to the N
# best-ranked physical cores (no SMT siblings); --threads 0 runs on every
# core, unpinned. RTQ_CPUS=<cpu list> pins to exactly those CPUs instead.
#
# Environment: BONSAI_BUILD_DIR names the compiler's build directory (default
# `build`); BONSAI_CXX a clang++ (default `clang++`; the generated header
# needs clang's ext_vector_type); RTQ_CPUS as above.
if [[ "$(pwd)" == */apps/rtq ]]; then
  cd ../..
fi

PREFIX="apps/rtq"
SCHEDULE="embree"
THREADS=1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedule)
      if [[ $# -lt 2 ]]; then
        echo "--schedule needs a name from $PREFIX/schedules/" >&2
        exit 1
      fi
      SCHEDULE="$2"
      shift 2
      ;;
    --threads)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "--threads needs a count: 1 (one core, pinned), N (N cores, pinned), 0 (every core)" >&2
        exit 1
      fi
      THREADS="$2"
      ARGS+=("$1" "$2")
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
if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
  echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
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
# TBB and zlib live beside the compiler in the conda environment; the
# runtime's parallel loop runs on TBB where its header is reachable (see
# runtime/bonsai_parallel.h), and the driver spreads Embree's rays with it.
TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
if [[ ! -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  echo "no TBB beside $BONSAI_CXX; activate the bonsai conda environment" >&2
  exit 1
fi

BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
cmake --build "$BONSAI_BUILD_DIR" -j

# `-p ssa` because sort and loopify are SSA rewrites; `--no-heap` refuses any
# allocation in the compiled program; `--ffp-contract` fuses `a * b + c` as
# Embree's build does (its node test is written with msub, its dot products
# with madd).
FLAGS=(-p ssa --no-heap --ffp-contract)
INPUTS=(-i $PREFIX/rtq.bonsai -i "$PREFIX/schedules/$SCHEDULE.bonsai")
"./$BONSAI_BUILD_DIR/compiler" -p ssa "${INPUTS[@]}" -o $PREFIX/rtq.bir
"./$BONSAI_BUILD_DIR/compiler" "${FLAGS[@]}" "${INPUTS[@]}" -b llvm -o $PREFIX/rtq.ll
"./$BONSAI_BUILD_DIR/compiler" "${FLAGS[@]}" "${INPUTS[@]}" -b cpp -o $PREFIX/rtq

"$BONSAI_CXX" -std=c++20 -O3 -I. -I$PREFIX -isystem "$EMBREE/include" \
    -isystem "$TBB_PREFIX/include" \
    $PREFIX/rtq_hook.cpp $PREFIX/rtq.o \
    -L"$EMBREE/lib" -Wl,-rpath,"$EMBREE/lib" -lembree4 \
    -L"$TBB_PREFIX/lib" -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb -lz \
    -o $PREFIX/rtq.out

# The cores to pin to (see the top of the file): the THREADS physical cores
# the kernel ranks highest, one hardware thread each. A CPU is the first of
# its core when its number is the first in its core's sibling list.
PIN=()
if [[ "$THREADS" -gt 0 ]]; then
  if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl is needed to pin the measurement (or pass --threads 0 for an unpinned run)" >&2
    exit 1
  fi
  if [[ -n "${RTQ_CPUS:-}" ]]; then
    CPUS="$RTQ_CPUS"
    HOW="RTQ_CPUS"
  else
    HOW="the lowest-numbered physical cores (no preferred-core ranking here)"
    CPUS="$(for d in /sys/devices/system/cpu/cpu[0-9]*; do
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
      done | sort -k1,1nr -k2,2n | head -n "$THREADS" | awk '{print $2}' | paste -sd,)"
    if [[ -f "/sys/devices/system/cpu/cpu${CPUS%%,*}/cpufreq/amd_pstate_prefcore_ranking" ]]; then
      HOW="the physical cores with the highest amd_pstate_prefcore_ranking"
    fi
  fi
  NODE="$(basename "$(ls -d /sys/devices/system/cpu/cpu${CPUS%%,*}/node[0-9]* 2>/dev/null | head -n 1)" 2>/dev/null)"
  NODE="${NODE#node}"
  PIN=(numactl --physcpubind="$CPUS" --membind="${NODE:-0}")
  echo "pinned to cpu(s) $CPUS, memory of NUMA node ${NODE:-0}: $HOW"
fi

STATUS=0
"${PIN[@]}" ./$PREFIX/rtq.out "$@" || STATUS=$?

rm -f $PREFIX/rtq.bir $PREFIX/rtq.ll $PREFIX/rtq.h $PREFIX/rtq.o $PREFIX/rtq.out
exit $STATUS
