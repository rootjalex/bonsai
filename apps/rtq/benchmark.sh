#!/bin/bash
# The rtq table, end to end: every mesh against both references under the
# matching schedules, the results folded into apps/rtq/results/rtq-results.csv,
# the plots drawn into apps/rtq/plots, and the geomeans printed. One command
# to re-run after a compiler change, from the repository root, inside the
# `bonsai` conda environment, with the compiler built:
#
#     apps/rtq/benchmark.sh
#     apps/rtq/benchmark.sh --meshes "head pavilion" --schedules embree
#     apps/rtq/benchmark.sh --plot-only          # fold and plot what is logged
#
# Options: --schedules (default embree,fcpw4w16, the two matching schedules
# the plots draw; a tuned or four-wide schedule may be added, each a `===
# schedule` section of the same runs), --meshes (default the thirteen, in
# order of triangle count), --out DIR for the logs (default
# apps/rtq/results/logs), --repeats (default 5, each number the least of
# them after a warm-up), --rays-side / --points-side / --fcpw-side (2048 /
# 1024 / 1024: Embree's rays run at a side where a single-threaded run
# lasts over a tenth of a second, the point batches at FCPW's side so both
# references answer the same points, FCPW's every query at 1024 since its
# rate is a third of Embree's), --fresh to measure every cell again rather
# than resuming, --plot-only to skip the measuring, --commit C to stamp the
# rows with that commit rather than the tree's (logs measured under a
# compiler since committed, folded again). RTQ_SCENES names the
# pbrt-v4-scenes checkout (default ~/projects/pbrt-v4-scenes);
# BONSAI_BUILD_DIR the compiler's build directory (compare.sh's rule
# otherwise); RTQ_CPUS the core to pin to (compare.sh's rule otherwise).
#
# What a run does. Before each mesh the machine is checked idle -- nothing
# but this at over half a core -- and a busy machine is waited for, up to
# ten minutes, then the run stops and says so: it never measures beside
# another load. `touch <out>/PAUSE` holds it at the next mesh boundary until
# the file goes. Each cell's log is written as `<log>.partial` and renamed
# when complete, so a log that exists is a complete run and a stopped sweep
# resumes with the same command (--fresh clears them first). The logs are
# folded by tocsv.sh, each row stamped with the commit the compiler was
# built from (`<hash>+` when the tree has uncommitted compiler changes), the
# rows of cells not measured this time carried over from the CSV as it was;
# then plot.py draws the three figures and prints, per reference and ray or
# point set, the geomean over the plotted meshes with the cells under 0.97
# named -- the rule the comparison is judged by.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCENES="${RTQ_SCENES:-$HOME/projects/pbrt-v4-scenes}"
ALL_MESHES="pavilion head zero-day bmw crown ivy villa dambreak sportscar landscape lte-orb ganesha dragon"
MESHES="$ALL_MESHES"
SCHEDULES="embree,fcpw4w16"
OUT="apps/rtq/results/logs"
CSV="apps/rtq/results/rtq-results.csv"
PLOTS="apps/rtq/plots"
RAYS_SIDE=2048
POINTS_SIDE=1024
FCPW_SIDE=1024
REPEATS=5
FRESH=0
PLOT_ONLY=0
COMMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --meshes) MESHES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --rays-side) RAYS_SIDE="$2"; shift 2 ;;
    --points-side) POINTS_SIDE="$2"; shift 2 ;;
    --fcpw-side) FCPW_SIDE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --plot-only) PLOT_ONLY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
for s in $(tr , ' ' <<< "$SCHEDULES"); do
  [[ -f "apps/rtq/schedules/$s.bonsai" ]] || { echo "no schedule apps/rtq/schedules/$s.bonsai" >&2; exit 1; }
done
mkdir -p "$OUT" "$PLOTS" "$(dirname "$CSV")"

mesh_path() {
  case "$1" in
    pavilion) echo "$SCENES/barcelona-pavilion/geometry/mesh_00099.ply" ;;
    head) echo "$SCENES/head/geometry/head.ply" ;;
    zero-day) ls "$SCENES"/zero-day/geometry/9dbbf607*.ply | head -1 ;;
    bmw) echo "$SCENES/bmw-m6/geometry/mesh_00037.ply" ;;
    crown) echo "$SCENES/crown/geometry/mesh_00397.ply" ;;
    ivy) echo "$SCENES/sanmiguel/geometry/enredadera_00001.ply" ;;
    villa) echo "$SCENES/villa/geometry/mesh_00072.ply" ;;
    dambreak) echo "$SCENES/dambreak/geometry/dambreak1.ply" ;;
    sportscar) echo "$SCENES/sportscar/geometry/OSD_CarPoly_0000_m028.ply" ;;
    landscape) echo "$SCENES/landscape/geometry/mesh_00370.ply" ;;
    lte-orb) echo "$SCENES/lte-orb/geometry/mesh-3.ply.gz" ;;
    ganesha) echo "$SCENES/ganesha/geometry/ganesha.ply.gz" ;;
    dragon) echo "$SCENES/sssdragon/geometry/dragon.ply.gz" ;;
    *) echo "unknown mesh $1 (the thirteen: $ALL_MESHES)" >&2; exit 1 ;;
  esac
}

# Anything but this run's own processes and the sessions' shells at over
# half a core means someone else is measuring or compiling.
machine_busy() {
  ps -eo pid,pcpu,comm --no-headers | awk '
    $2 > 50 && $3 !~ /^(claude|node|tmux|sshd|ps|awk|bash|rtq\.out)$/ { busy = 1 }
    END { exit !busy }'
}
wait_idle() {
  local waited=0
  while machine_busy; do
    if (( waited == 0 )); then echo "[$(date +%T)] machine busy, waiting:"; ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -4; fi
    sleep 30; waited=$((waited + 30))
    if (( waited >= 600 )); then echo "[$(date +%T)] the machine did not go idle in ten minutes; stopping before $1" >&2; exit 3; fi
  done
}

EMBREE_SCHEDULES=""
FCPW_SCHEDULES=""
for s in $(tr , ' ' <<< "$SCHEDULES"); do
  if [[ "$s" == fcpw* ]]; then FCPW_SCHEDULES="$FCPW_SCHEDULES${FCPW_SCHEDULES:+,}$s"; else EMBREE_SCHEDULES="$EMBREE_SCHEDULES${EMBREE_SCHEDULES:+,}$s"; fi
done

run() {  # run <log> <compare.sh args...>
  local log="$1"; shift
  if [[ -s "$log" ]]; then echo "  have $log"; return; fi
  echo "[$(date +%T)] -> $log"
  apps/rtq/compare.sh "$@" > "$log.partial" 2>&1 || echo "  (exit $?: the driver's disagreements or a failure; see $log)"
  mv "$log.partial" "$log"
}

if (( ! PLOT_ONLY )); then
  for mesh in $MESHES; do
    mesh_path "$mesh" > /dev/null
    if (( FRESH )); then rm -f "$OUT/$mesh".s*.log; fi
  done
  for mesh in $MESHES; do
    if [[ -e "$OUT/PAUSE" ]]; then
      echo "[$(date +%T)] paused before $mesh ($OUT/PAUSE exists)"
      while [[ -e "$OUT/PAUSE" ]]; do sleep 10; done
      echo "[$(date +%T)] resumed"
    fi
    wait_idle "$mesh"
    path="$(mesh_path "$mesh")"
    if [[ -n "$EMBREE_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$RAYS_SIDE.intersect.log" --schedule "$EMBREE_SCHEDULES" --side "$RAYS_SIDE" --repeats "$REPEATS" --query intersect "$path"
      run "$OUT/$mesh.s$RAYS_SIDE.occluded.log" --schedule "$EMBREE_SCHEDULES" --side "$RAYS_SIDE" --repeats "$REPEATS" --query occluded "$path"
      run "$OUT/$mesh.s$POINTS_SIDE.closest.log" --schedule "$EMBREE_SCHEDULES" --side "$POINTS_SIDE" --repeats "$REPEATS" --query closest "$path"
    fi
    if [[ -n "$FCPW_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$FCPW_SIDE.fcpw.log" --schedule "$FCPW_SCHEDULES" --side "$FCPW_SIDE" --repeats "$REPEATS" "$path"
    fi
  done
  echo "[$(date +%T)] measured: $OUT"
fi

# The compiler the rows were measured under: the commit, marked when the
# tree carries uncommitted compiler changes, unless --commit said.
if [[ -z "$COMMIT" ]]; then
  COMMIT="$(git rev-parse --short HEAD)"
  if [[ -n "$(git status --porcelain -- CMakeLists.txt include src 2>/dev/null)" ]]; then
    COMMIT="$COMMIT+"
  fi
fi
shopt -s nullglob
LOGS=("$OUT"/*.log)
shopt -u nullglob
(( ${#LOGS[@]} > 0 )) || { echo "no logs in $OUT" >&2; exit 1; }
KEEP=()
if [[ -s "$CSV" ]]; then
  cp "$CSV" "$CSV.prev"
  KEEP=(--keep "$CSV.prev")
fi
bash apps/rtq/tocsv.sh --commit "$COMMIT" --repeats "$REPEATS" "${KEEP[@]}" -o "$CSV" "${LOGS[@]}"
python3 apps/rtq/plot.py "$CSV" -o "$PLOTS"
