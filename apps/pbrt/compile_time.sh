#!/usr/bin/env bash
# apps/pbrt/compile_time.sh: how long the compiler takes on each schedule of
# the sweeps, and how long pbrt-v4 takes to build, measured for the figure
# pbrt-compile-time (the user, 11-03).
#
#   apps/pbrt/compile_time.sh --out DIR [--schedules "s1 s2 ..."] [--repeats N]
#       [--resume] [--no-pbrt | --pbrt-only] [--pbrt-source DIR] [--pbrt-build DIR]
#
# Each schedule's renderer is built once (--repeats, 1 by default: the
# user, 11-03, three builds of a schedule agreed to a third of a percent)
# exactly as a comparison builds it (compare.sh --build-renderer: the
# compiler on render.bonsai with the schedule, then clang on the driver),
# and compare.sh's two timing lines are kept -- `compiler seconds`, the
# schedule's own: parsing, scheduling, the SSA passes, LLVM's optimization
# and the object, and for a GPU schedule its PTX and OptiX modules; and
# `driver seconds`, clang on the driver, the same host code whatever the
# schedule. One CSV row per build into DIR/compile-times.csv:
#
#   schedule,side,repeat,compiler_seconds,driver_seconds,disturbed,date,commit,host
#
# `side` is cpu or gpu by the schedule file (a binding to GPUBlock,
# GPUThread or OptixThread makes it gpu -- compare.sh's own test, which also
# gives such a build --fast-math and the register cap); `commit` is the
# compiler's last commit, as results/provenance.txt records it. The build's
# log is DIR/logs/<schedule>.<repeat>.log and the binary is removed once
# built: the time is what is measured. --resume skips a schedule that has
# its rows already, so a stopped run is continued by the same command, as
# the sweeps are.
#
# Then pbrt itself, the reference the figure's dashed line stands for (the
# user, 11-03: "not an exact comparison", and so marked): pbrt-v4 built
# from a clean build directory as a user builds it -- configured from the
# reference build's own cache (--pbrt-build, the directory of compare.sh's
# PBRT binary: the same compilers, OptiX path, shader model and options),
# `cmake --build` with a job per core, its bundled dependencies and its
# GPU wavefront included -- in its own conda environment (the one its
# compilers live in). Its row is `pbrt-v4,pbrt,1,<wall seconds>,0,...`,
# with pbrt's commit; DIR/pbrt-build.txt has the wall and CPU seconds,
# the jobs, the source and the commit; the build directory is removed
# once timed. --no-pbrt leaves it out, --pbrt-only measures it alone.
#
# Before each build the machine is checked idle the way the sweeps check it:
# another process above half a core delays the build, in steps of ten
# seconds up to ten minutes, after which the build goes ahead and its row
# says `disturbed` 1. The schedules default to the two sweeps' sides
# (figures.sh GPU_SIDES and CPU_SIDES, pbrt's own side left out).
set -euo pipefail
cd "$(dirname "$0")/../.."
PREFIX=apps/pbrt
source $PREFIX/figures.sh
REPEATS=1   # figures.sh's count is the sweeps'; a build is measured once

OUT=""
SCHEDULES=""
RESUME=0
WITH_PBRT=1
PBRT_ONLY=0
PBRT_BIN="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
PBRT_BUILD="$(dirname "$PBRT_BIN")"
PBRT_SOURCE="$(dirname "$PBRT_BUILD")"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-pbrt) WITH_PBRT=0; shift ;;
    --pbrt-only) PBRT_ONLY=1; shift ;;
    --pbrt-source) PBRT_SOURCE="$2"; shift 2 ;;
    --pbrt-build) PBRT_BUILD="$2"; shift 2 ;;
    -h|--help) sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "compile_time.sh: --out DIR is needed" >&2; exit 1; }
[[ -n "$SCHEDULES" ]] || SCHEDULES="$(spaced "$GPU_SIDES") $(cpu_schedules)"
(( PBRT_ONLY )) && SCHEDULES=""
mkdir -p "$OUT/logs" "$OUT/renderers" "$OUT/work"
CSV="$OUT/compile-times.csv"
[[ -f "$CSV" ]] || echo "schedule,side,repeat,compiler_seconds,driver_seconds,disturbed,date,commit,host" > "$CSV"
COMMIT=$(git log -1 --format=%h -- src include runtime stdlib CMakeLists.txt)
HOST=$(hostname)
NOW() { date '+%Y-%m-%d %H:%M:%S'; }

side_of() {
  if grep -v '^\s*//' "$PREFIX/schedules/$1.bonsai" | grep -q 'GPUBlock\|GPUThread\|OptixThread'; then
    echo gpu
  else
    echo cpu
  fi
}

# Another process above half a core, ours and this shell's aside.
busy() {
  ps -eo pcpu=,pid=,ppid= | awk -v me=$$ '$1 > 50 && $2 != me && $3 != me { n++ } END { exit n > 0 ? 0 : 1 }'
}
wait_idle() {  # prints 0 when the machine was idle in time, 1 when not
  local waited=0
  while busy; do
    if (( waited >= 600 )); then echo 1; return; fi
    (( waited == 0 )) && echo "[$(date +%T)] another process is busy; waiting" >&2
    sleep 10
    waited=$((waited + 10))
  done
  echo 0
}

seconds_in() {  # seconds_in LOG "compiler seconds"
  sed -n "s/^$2: //p" "$1" | tail -1
}

for s in $SCHEDULES; do
  if [[ ! -f "$PREFIX/schedules/$s.bonsai" ]]; then
    echo "compile_time.sh: no schedule $PREFIX/schedules/$s.bonsai" >&2; exit 1
  fi
  have=$(grep -c "^$s," "$CSV" || true)
  if (( RESUME && have >= REPEATS )); then
    echo "[$(date +%T)] $s: built already"
    continue
  fi
  side=$(side_of "$s")
  for (( r = have + 1; r <= REPEATS; r++ )); do
    disturbed=$(wait_idle)
    log="$OUT/logs/$s.$r.log"
    bin="$OUT/renderers/$s.out"
    echo "[$(date +%T)] $s $r/$REPEATS ($side)"
    if ! WORK="$OUT/work" bash $PREFIX/compare.sh --schedule "$s" --build-renderer "$bin" > "$log" 2>&1; then
      echo "  the build failed; see $log" >&2; exit 1
    fi
    compiler=$(seconds_in "$log" "compiler seconds")
    driver=$(seconds_in "$log" "driver seconds")
    [[ -n "$compiler" && -n "$driver" ]] || { echo "  no timing lines in $log" >&2; exit 1; }
    echo "$s,$side,$r,$compiler,$driver,$disturbed,$(NOW),$COMMIT,$HOST" >> "$CSV"
    echo "  compiler $compiler s, driver $driver s"
    rm -f "$bin"
  done
done

if (( WITH_PBRT )); then
  have=$(grep -c "^pbrt-v4," "$CSV" || true)
  if (( RESUME && have >= 1 )); then
    echo "[$(date +%T)] pbrt-v4: built already"
  else
    [[ -f "$PBRT_SOURCE/CMakeLists.txt" ]] || { echo "compile_time.sh: no pbrt source at $PBRT_SOURCE (--pbrt-source)" >&2; exit 1; }
    [[ -f "$PBRT_BUILD/CMakeCache.txt" ]] || { echo "compile_time.sh: no pbrt build to take the configuration from at $PBRT_BUILD (--pbrt-build)" >&2; exit 1; }
    build="$OUT/pbrt-build"
    rm -rf "$build"
    mkdir -p "$build"
    # The reference build's configuration, every option the user set and
    # every path cmake found there, as an initial cache for the clean one;
    # the entries that name the build directory itself stay out.
    grep -E '^[A-Za-z0-9_]+:(STRING|BOOL|PATH|FILEPATH)=' "$PBRT_BUILD/CMakeCache.txt" \
      | grep -vE '^(CMAKE_INSTALL_PREFIX|CMAKE_CACHEFILE_DIR|CMAKE_HOME_DIRECTORY|CMAKE_MAKE_PROGRAM):' \
      | sed -E 's/^([A-Za-z0-9_]+):([A-Z]+)=(.*)$/set(\1 "\3" CACHE \2 "")/' > "$build/init.cmake"
    # Its compilers' conda environment, from where the cache says they are.
    cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$PBRT_BUILD/CMakeCache.txt")
    env_dir=""
    case "$cxx" in */envs/*/bin/*) env_dir="${cxx%/bin/*}" ;; esac
    jobs=$(nproc)
    pbrt_commit=$(git -C "$PBRT_SOURCE" log -1 --format=%h 2>/dev/null || echo unknown)
    echo "[$(date +%T)] pbrt-v4: configuring a clean build from $PBRT_BUILD's cache${env_dir:+ in $env_dir}"
    (
      if [[ -n "$env_dir" && -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate "$env_dir"
      fi
      cmake -S "$PBRT_SOURCE" -B "$build" -C "$build/init.cmake" > "$OUT/logs/pbrt-v4.configure.log" 2>&1
    ) || { echo "  the configure failed; see $OUT/logs/pbrt-v4.configure.log" >&2; exit 1; }
    disturbed=$(wait_idle)
    echo "[$(date +%T)] pbrt-v4: building with $jobs jobs"
    (
      if [[ -n "$env_dir" && -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate "$env_dir"
      fi
      /usr/bin/time -f '%e %U %S' -o "$build/time.txt" cmake --build "$build" -j"$jobs" > "$OUT/logs/pbrt-v4.build.log" 2>&1
    ) || { echo "  the build failed; see $OUT/logs/pbrt-v4.build.log" >&2; exit 1; }
    read -r wall user sys < "$build/time.txt"
    cpu=$(awk -v u="$user" -v s="$sys" 'BEGIN { printf "%.2f", u + s }')
    echo "pbrt-v4,pbrt,1,$wall,0.00,$disturbed,$(NOW),$pbrt_commit,$HOST" >> "$CSV"
    {
      echo "pbrt-v4 built from a clean build directory, configured from $PBRT_BUILD's cache"
      echo "source      $PBRT_SOURCE ($pbrt_commit)"
      echo "jobs        $jobs (cmake --build -j, a job per core)"
      echo "wall        $wall s"
      echo "cpu         $cpu s (user $user + sys $sys)"
      echo "date        $(NOW)"
      echo "host        $HOST"
    } > "$OUT/pbrt-build.txt"
    echo "  wall $wall s, cpu $cpu s over $jobs jobs"
    rm -rf "$build"
  fi
fi
echo "rows: $CSV"
