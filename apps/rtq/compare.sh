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
# a table traverses a tree of the same width as the other. A packet
# schedule, `packet<K>` (schedules/packet16.bonsai), runs the rays in gangs
# of K over the eight-wide tree, and Embree's side is told to answer them in
# packets of K too (the driver's `--packet K`: rtcIntersect<K> and
# rtcOccluded<K> in place of rtcIntersect1, the same K consecutive rays a
# packet on both sides); its closest-point rows are the scalar query's, since
# Embree has no packet point query.
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
# on plain pages instead of Embree's 2 MB ones (see rtq_hook.cpp, OsMemory);
# RTQ_SKIP_CMAKE=1 runs the compiler as built, without rebuilding it first;
# RTQ_METRICS=<file> compiles the program over that metrics file;
# RTQ_KEEP=1 leaves the last schedule's rtq.bir, rtq.ll, rtq.h, rtq.o and
# rtq.out under apps/rtq after the run, for reading the generated code and
# profiling the driver (perf needs the binary the samples came from).
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
# The arithmetic a schedule's program is compiled with (metrics/<name>.bonsai):
# the reference's, and FCPW has two -- metrics/fcpw.bonsai, its vectorized
# build's (enoki's reciprocal estimate, one root per packet), for the
# fcpw<branching>w<width> schedules over its Mbvh; metrics/fcpw_scalar.bonsai,
# its scalar build's (true divisions, Eigen's two sums), for fcpwb, the
# schedule over its binary BVH.
metric_of() {
  if [[ "$1" == fcpwb* || "$1" == fcpws* ]]; then echo fcpw_scalar; else reference_of "$1"; fi
}
# The program: rtq.bonsai, the three queries over a tree of triangles; for
# fcpws, FCPW's normal cone hierarchy, snch.bonsai, which adds the
# silhouette query over the tree's edges.
program_of() {
  if [[ "$1" == fcpws* ]]; then echo "$PREFIX/snch.bonsai"; else echo "$PREFIX/rtq.bonsai"; fi
}
# Which FCPW checkout: deps/fcpw for the vectorized Mbvh, the one the
# fcpw4w16 table has always been measured against; for the binary tree the
# FCPW WoSX ships (deps/wosx/deps/fcpw, a later checkout whose bvh.inl
# differs), since that schedule isolates the queries apps/wosx's walks make,
# against the build they make them against.
fcpw_dir_of() {
  if [[ "$1" == fcpwb* || "$1" == fcpws* ]]; then echo deps/wosx/deps/fcpw; else echo deps/fcpw; fi
}
# The packet width a schedule's name asks of Embree's side: `packet<K>` is
# the program's rays in gangs of K against rtcIntersect<K>, and so is a
# `packet<K>-<variant>` (packet16-anyhit: the any hit alone in gangs, the
# rest scalar); every other schedule is single rays against rtcIntersect1
# (empty).
packet_of() {
  if [[ "$1" =~ ^packet([0-9]+)(-.*)?$ ]]; then echo "${BASH_REMATCH[1]}"; fi
}
IFS=, read -r -a SCHEDULE_LIST <<< "$SCHEDULES"
NEEDS_EMBREE=0
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  if [[ ! -f "$PREFIX/schedules/$SCHEDULE.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$SCHEDULE.bonsai" >&2
    exit 1
  fi
  if [[ "$(reference_of "$SCHEDULE")" != fcpw ]]; then
    NEEDS_EMBREE=1
  fi
  K="$(packet_of "$SCHEDULE")"
  if [[ -n "$K" && "$K" != 4 && "$K" != 8 && "$K" != 16 ]]; then
    echo "a packet schedule is named packet<K> for one of Embree's packet widths, 4, 8 or 16: $SCHEDULE" >&2
    exit 1
  fi
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
# is nothing to build. Each fcpw schedule's checkout (fcpw_dir_of) is
# checked, its Eigen under deps/eigen-git-mirror or deps/eigen as the
# checkout names it.
for SCHEDULE in "${SCHEDULE_LIST[@]}"; do
  if [[ "$(reference_of "$SCHEDULE")" != fcpw ]]; then continue; fi
  FCPW="$(fcpw_dir_of "$SCHEDULE")"
  if [[ ! -f "$FCPW/include/fcpw/fcpw.h" || ! -f "$FCPW/deps/enoki/include/enoki/array.h" ||
        -z "$(ls -d "$FCPW"/deps/eigen*/Eigen/Core 2>/dev/null)" ]]; then
    echo "FCPW is not checked out with its own submodules at $FCPW: $SCHEDULE compares against it." >&2
    echo "    git submodule update --init --recursive $FCPW" >&2
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
# RTQ_SKIP_CMAKE=1 leaves the compiler as built: for a working tree whose
# build directory must not be touched while another measurement rebuilds
# from it (the shared checkout's lock), or whose compiler is known current.
if [[ -z "${RTQ_SKIP_CMAKE:-}" ]]; then
  cmake --build "$BONSAI_BUILD_DIR" -j
fi

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
  # The elements, the reference's arithmetic (metrics/<reference>.bonsai:
  # its slab test, its triangle test, its point distances and its answer
  # records, transcribed), the program, and the schedule over the
  # reference's tree -- in that order, each file's declarations before the
  # file that names them (the metric's records are declared over the
  # elements, the program's exports over the records). RTQ_METRICS names
  # another metrics file in place of the reference's, for measuring what an
  # arithmetic differs by against the same reference.
  local metrics="${RTQ_METRICS:-$PREFIX/metrics/$(metric_of "$schedule").bonsai}"
  local inputs=(-i $PREFIX/elements.bonsai -i "$metrics" -i "$(program_of "$schedule")"
                -i "$PREFIX/schedules/$schedule.bonsai")
  "./$BONSAI_BUILD_DIR/compiler" -p ssa "${inputs[@]}" -o $PREFIX/rtq.bir
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b llvm -o $PREFIX/rtq.ll
  "./$BONSAI_BUILD_DIR/compiler" "${flags[@]}" "${inputs[@]}" -b cpp -o $PREFIX/rtq

  if [[ "$(reference_of "$schedule")" == fcpw ]]; then
    local branching width defines fcpw eigen
    fcpw="$(fcpw_dir_of "$schedule")"
    eigen="$(ls -d "$fcpw"/deps/eigen* | head -n 1)"
    # `fcpw<branching>w<width>` is the schedule matching FCPW's over that
    # tree; a `t` after it (fcpw4w16t) is the tuned schedule over the same
    # tree, as tuned.bonsai is to embree.bonsai; `fcpwb` is the schedule
    # matching FCPW's scalar build over its binary BVH (the driver's
    # RTQ_FCPW_BINARY; FCPW_USE_ENOKI off, as WoSX's CMake leaves it);
    # `fcpws` the one matching it over its normal cone hierarchy, with the
    # silhouette query (RTQ_FCPW_SNCH, snch.bonsai); `-gpu` after either of
    # those is the GPU comparison, FCPW's compute shaders over the same
    # tree as the reference (RTQ_GPU; the tree's rows are the ones FCPW
    # itself puts on the card).
    local base="$schedule" gpu=0
    if [[ "$schedule" == *-gpu ]]; then
      base="${schedule%-gpu}"
      gpu=1
    fi
    if [[ "$base" =~ ^fcpw([48])w([0-9]+)t?$ && $gpu == 0 ]]; then
      branching="${BASH_REMATCH[1]}"
      width="${BASH_REMATCH[2]}"
      defines=(-DRTQ_FCPW -DNDEBUG -DFCPW_USE_ENOKI "-DFCPW_SIMD_WIDTH=$width")
      if [[ "$branching" == 8 ]]; then
        defines+=(-DFCPW_USE_EIGHT_WIDE_BRANCHING)
      fi
    elif [[ "$base" == fcpwb ]]; then
      branching=2
      width=1
      defines=(-DRTQ_FCPW -DRTQ_FCPW_BINARY -DNDEBUG)
    elif [[ "$base" == fcpws ]]; then
      branching=2
      width=1
      defines=(-DRTQ_FCPW -DRTQ_FCPW_SNCH -DNDEBUG)
    else
      echo "an fcpw schedule is named fcpw<branching>w<width>[t] (fcpw4w16, fcpw8w8, fcpw4w16t, ...), fcpwb or fcpws, or fcpwb-gpu or fcpws-gpu: $schedule" >&2
      exit 1
    fi
    # The GPU comparison: FCPW's GPUScene runs on slang-rhi and the Slang
    # release it fetched (deps/slang-rhi-build, built once by
    # apps/wosx/build_slang_rhi.sh from the submodule under WoSX's FCPW),
    # which the driver is compiled against and linked with (the Slang
    # library found at run time through the rpath; Slang itself finds
    # CUDA's NVRTC). The driver is told where FCPW's Slang sources are.
    local gpu_includes=() gpu_libs=()
    if [[ $gpu == 1 ]]; then
      local rhi_src="$fcpw/deps/slang-rhi" rhi_build="deps/slang-rhi-build"
      local slang_src="deps/slang-rhi-build/_deps/slang-src"
      local rhi_lib slang_lib rhi_resources
      rhi_lib="$(ls "$rhi_build"/libslang-rhi.a "$rhi_build"/libslang-rhi.so 2>/dev/null | head -n 1 || true)"
      slang_lib="$(ls "$slang_src"/lib/libslang-compiler.so "$slang_src"/lib/libslang.so 2>/dev/null | head -n 1 || true)"
      rhi_resources="$(ls "$rhi_build"/libslang-rhi-resources.a 2>/dev/null | head -n 1 || true)"
      if [[ ! -f "$rhi_src/include/slang-rhi.h" || -z "$rhi_lib" || -z "$slang_lib" ||
            ! -f "$slang_src/include/slang.h" ]]; then
        echo "a GPU schedule compares against FCPW's GPUScene, which runs on slang-rhi:" >&2
        echo "build it once with apps/wosx/build_slang_rhi.sh (into $rhi_build)" >&2
        exit 1
      fi
      defines+=(-DRTQ_GPU -DFCPW_USE_GPU "-DRTQ_FCPW_DIR=\"$(cd "$fcpw" && pwd)\"")
      gpu_includes=(-isystem "$rhi_src/include" -isystem "$rhi_build/include"
                    -isystem "$slang_src/include")
      gpu_libs=("$rhi_lib" ${rhi_resources:+"$rhi_resources"} "$slang_lib"
                -Wl,-rpath,"$(cd "$slang_src/lib" && pwd)" -ldl -lpthread)
    fi
    # The driver compiled once per (header, defines) within a run and linked
    # per schedule: it includes the generated rtq.h, which two schedules over
    # the same layout generate alike (the matching one and the tuned one),
    # and compiling it against FCPW's headers takes longer than the
    # measurement. Keyed by the header's checksum, so an unlike header
    # compiles afresh; the objects go at the end of the run.
    local key
    key="fcpw-$(cat $PREFIX/rtq.h | md5sum | cut -c1-16)-$branching-$width-$gpu"
    if [[ ! -f "$PREFIX/.rtq_hook-$key.o" ]]; then
      "$BONSAI_CXX" -std=c++20 -O3 -march=native -fno-math-errno -ffp-contract=fast \
          "${defines[@]}" -I. -I$PREFIX \
          -isystem "$fcpw/include" -isystem "$fcpw/deps/enoki/include" \
          -isystem "$eigen" ${gpu_includes[@]+"${gpu_includes[@]}"} \
          -c $PREFIX/rtq_hook.cpp -o "$PREFIX/.rtq_hook-$key.o"
    fi
    "$BONSAI_CXX" "$PREFIX/.rtq_hook-$key.o" $PREFIX/rtq.o \
        -L"$TOOLCHAIN_PREFIX/lib" -Wl,-rpath,"$TOOLCHAIN_PREFIX/lib" -lz \
        ${gpu_libs[@]+"${gpu_libs[@]}"} \
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
  # Embree answers a packet schedule's rays in packets of the same width.
  PACKET=()
  K="$(packet_of "$SCHEDULE")"
  if [[ -n "$K" ]]; then PACKET=(--packet "$K"); fi
  numactl --physcpubind="$CPU" --membind="${NODE:-0}" ./$PREFIX/rtq.out "${PACKET[@]}" "$@" || STATUS=$?
  if [[ -z "${RTQ_KEEP:-}" ]]; then
    rm -f $PREFIX/rtq.bir $PREFIX/rtq.ll $PREFIX/rtq.h $PREFIX/rtq.o $PREFIX/rtq.out
  fi
done
rm -f $PREFIX/.rtq_hook-*.o
exit $STATUS
