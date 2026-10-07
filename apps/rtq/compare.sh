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
# table under a `=== schedule` line. The same pair over Embree's four-wide
# tree is `embree4,tuned4` (schedules/trees/bvh4.bonsai): the driver reads
# the width off the layout it is compiled against and asks Embree's device
# for the tree of that width (`tri_accel=bvh4.triangle4`), so each side of
# a table traverses a tree of the same width as the other.
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
# ext_vector_type); RTQ_CPUS as above; RTQ_PAGES=4k maps the tree's storage
# on plain pages instead of Embree's 2 MB ones (see rtq_hook.cpp, OsMemory).
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

# Which reference a schedule compares against: the fcpw* schedules (over
# FCPW's trees, schedules/trees/mbvh*.bonsai) against FCPW, every other
# against Embree. An fcpw schedule's name says which of FCPW's trees it is
# over -- `fcpw<branching>w<width>`, FCPW's compile-time branching factor and
# leaf width, with a `t` after it for the tuned schedule over the same tree
# -- which is how the driver is told to build FCPW's tree of that shape
# (FCPW_USE_EIGHT_WIDE_BRANCHING, FCPW_SIMD_WIDTH).
reference_of() {
  if [[ "$1" == fcpw* ]]; then echo fcpw; else echo embree; fi
}
IFS=, read -r -a SCHEDULE_LIST <<< "$SCHEDULES"
NEEDS_EMBREE=0
NEEDS_FCPW=0
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
    exit 1
  fi
  case "$(reference_of "$SCHEDULE")" in
    fcpw) NEEDS_FCPW=1 ;;
    *) NEEDS_EMBREE=1 ;;
  esac
done

EMBREE="deps/embree-install"
if [[ $NEEDS_EMBREE == 1 ]]; then
  if [[ ! -f deps/embree/CMakeLists.txt ]]; then
    echo "Embree is not checked out: this app compares against it and needs it." >&2
    echo "    git submodule update --init deps/embree" >&2
    echo "    $PREFIX/build_embree.sh" >&2
    exit 1
  fi
  if [[ ! -f "$EMBREE/include/embree4/rtcore.h" || ! -f "$EMBREE/lib/libembree4.so" ]]; then
    echo "Embree is checked out but not built: run $PREFIX/build_embree.sh" >&2
    exit 1
  fi
fi
# FCPW is header-only, with Eigen and enoki as submodules of its own; there
# is nothing to build.
FCPW="deps/fcpw"
if [[ $NEEDS_FCPW == 1 ]]; then
  if [[ ! -f "$FCPW/include/fcpw/fcpw.h" || ! -f "$FCPW/deps/enoki/include/enoki/array.h" ||
        ! -f "$FCPW/deps/eigen-git-mirror/Eigen/Core" ]]; then
    echo "FCPW is not checked out with its own submodules: the fcpw* schedules compare against it." >&2
    echo "    git submodule update --init --recursive deps/fcpw" >&2
    exit 1
  fi
fi

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
# (its node test is written with msub, its dot products with madd) and as
# FCPW's does (enoki's dot is an explicit fmadd, and its build passes
# -ffp-contract=fast). The driver is compiled against the one reference the
# schedule is for: Embree's headers and library, or FCPW's headers with the
# flags FCPW's own CMake would set -- FCPW_USE_ENOKI, the leaf width and the
# branching of the schedule's tree, -march=native for enoki's vectors,
# NDEBUG as a release build, enoki's -fno-math-errno and -ffp-contract=fast.
build_schedule() {
  local schedule="$1"
  local flags=(-p ssa --no-heap --ffp-contract)
  # The program, the reference's arithmetic (metrics/<reference>.bonsai:
  # its slab test, its triangle test, its point distances, transcribed), and
  # the schedule over the reference's tree. RTQ_METRICS names another
  # metrics file in place of the reference's, for measuring what an
  # arithmetic differs by against the same reference.
  local metrics="${RTQ_METRICS:-$PREFIX/metrics/$(reference_of "$schedule").bonsai}"
  local inputs=(-i $PREFIX/rtq.bonsai -i "$metrics" -i "$PREFIX/schedules/$schedule.bonsai")
  "./$BONSAI_BUILD_DIR/compiler" -p ssa "${inputs[@]}" -o $PREFIX/rtq.bir
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b llvm -o $PREFIX/rtq.ll
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b cpp -o $PREFIX/rtq

  if [[ "$(reference_of "$schedule")" == fcpw ]]; then
    local branching width
    # `fcpw<branching>w<width>` is the schedule matching FCPW's over that
    # tree; a `t` after it (fcpw4w16t) is the tuned schedule over the same
    # tree, as tuned.bonsai is to embree.bonsai.
    if [[ "$schedule" =~ ^fcpw([48])w([0-9]+)t?$ ]]; then
      branching="${BASH_REMATCH[1]}"
      width="${BASH_REMATCH[2]}"
    else
      echo "an fcpw schedule is named fcpw<branching>w<width>[t] (fcpw4w16, fcpw8w8, fcpw4w16t, ...): $schedule" >&2
      exit 1
    fi
    local defines=(-DRTQ_FCPW -DNDEBUG -DFCPW_USE_ENOKI "-DFCPW_SIMD_WIDTH=$width")
    if [[ "$branching" == 8 ]]; then
      defines+=(-DFCPW_USE_EIGHT_WIDE_BRANCHING)
    fi
    # The driver compiled once per (header, defines) within a run and linked
    # per schedule: it includes the generated rtq.h, which two schedules over
    # the same layout generate alike (the matching one and the tuned one),
    # and compiling it against FCPW's headers takes longer than the
    # measurement. Keyed by the header's checksum, so an unlike header
    # compiles afresh; the objects go at the end of the run.
    local key
    key="fcpw-$(cat $PREFIX/rtq.h | md5sum | cut -c1-16)-$branching-$width"
    if [[ ! -f "$PREFIX/.rtq_hook-$key.o" ]]; then
      "$BONSAI_CXX" -std=c++20 -O3 -march=native -fno-math-errno -ffp-contract=fast \
          "${defines[@]}" -I. -I$PREFIX \
          -isystem "$FCPW/include" -isystem "$FCPW/deps/enoki/include" \
          -isystem "$FCPW/deps/eigen-git-mirror" \
          -c $PREFIX/rtq_hook.cpp -o "$PREFIX/.rtq_hook-$key.o"
    fi
    "$BONSAI_CXX" "$PREFIX/.rtq_hook-$key.o" $PREFIX/rtq.o \
        -L"$TOOLCHAIN_PREFIX/lib" -Wl,-rpath,"$TOOLCHAIN_PREFIX/lib" -lz \
        -o $PREFIX/rtq.out
  else
    local key
    key="embree-$(cat $PREFIX/rtq.h | md5sum | cut -c1-16)"
    if [[ ! -f "$PREFIX/.rtq_hook-$key.o" ]]; then
      "$BONSAI_CXX" -std=c++20 -O3 -I. -I$PREFIX -isystem "$EMBREE/include" \
          -c $PREFIX/rtq_hook.cpp -o "$PREFIX/.rtq_hook-$key.o"
    fi
    "$BONSAI_CXX" "$PREFIX/.rtq_hook-$key.o" $PREFIX/rtq.o \
        -L"$EMBREE/lib" -Wl,-rpath,"$EMBREE/lib" -lembree4 \
        -L"$TOOLCHAIN_PREFIX/lib" -Wl,-rpath,"$TOOLCHAIN_PREFIX/lib" -lz \
        -o $PREFIX/rtq.out
  fi
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
rm -f $PREFIX/.rtq_hook-*.o
exit $STATUS
