#!/usr/bin/env bash
# apps/pbrt/compile_time.sh: how long the compiler takes on each schedule of
# the sweeps, measured for the figure pbrt-compile-time (the user, 11-03).
#
#   apps/pbrt/compile_time.sh --out DIR [--schedules "s1 s2 ..."] [--repeats N] [--resume]
#
# Each schedule's renderer is built REPEATS times exactly as a comparison
# builds it (compare.sh --build-renderer: the compiler on render.bonsai
# with the schedule, then clang on the driver), and compare.sh's two timing
# lines are kept -- `compiler seconds`, the schedule's own: parsing,
# scheduling, the SSA passes, LLVM's optimization and the object, and for a
# GPU schedule its PTX and OptiX modules; and `driver seconds`, clang on the
# driver, the same host code whatever the schedule. The figure draws the
# least of a schedule's compiler seconds, as the sweeps draw the least of
# their renders (figures.sh REPEATS, the same count). One CSV row per build
# into DIR/compile-times.csv:
#
#   schedule,side,repeat,compiler_seconds,driver_seconds,disturbed,date,commit,host
#
# `side` is cpu or gpu by the schedule file (a binding to GPUBlock,
# GPUThread or OptixThread makes it gpu -- compare.sh's own test, which also
# gives such a build --fast-math and the register cap); `commit` is the
# compiler's last commit, as results/provenance.txt records it. The build's
# log is DIR/logs/<schedule>.<repeat>.log and the binary is removed once
# built: the time is what is measured. --resume skips a schedule that has
# its REPEATS rows already, so a stopped run is continued by the same
# command, as the sweeps are.
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

OUT=""
SCHEDULES=""
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "compile_time.sh: --out DIR is needed" >&2; exit 1; }
[[ -n "$SCHEDULES" ]] || SCHEDULES="$(spaced "$GPU_SIDES") $(cpu_schedules)"
mkdir -p "$OUT/logs" "$OUT/renderers" "$OUT/work"
CSV="$OUT/compile-times.csv"
[[ -f "$CSV" ]] || echo "schedule,side,repeat,compiler_seconds,driver_seconds,disturbed,date,commit,host" > "$CSV"
COMMIT=$(git log -1 --format=%h -- src include runtime stdlib CMakeLists.txt)
HOST=$(hostname)

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
    echo "[$(date +%T)] $s: $have builds already"
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
    echo "$s,$side,$r,$compiler,$driver,$disturbed,$(date '+%Y-%m-%d %H:%M:%S'),$COMMIT,$HOST" >> "$CSV"
    echo "  compiler $compiler s, driver $driver s"
    rm -f "$bin"
  done
done
echo "rows: $CSV"
