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
# schedule` section of the same runs; a packet schedule, packet16, runs its
# own sections -- the two ray queries only, against Embree's rtcIntersect16
# and rtcOccluded16, which compare.sh asks of the driver by the schedule's
# name -- since Embree has no packet point query; the schedules over FCPW's
# scalar trees, fcpwb and fcpws, run their own section against FCPW built
# scalar, and the GPU schedules, fcpwb-gpu and fcpws-gpu, theirs against
# FCPW's GPUScene: `--schedules fcpwb,fcpws,fcpwb-gpu,fcpws-gpu`), --meshes (default the thirteen, in
# order of triangle count), --out DIR for the logs (default
# apps/rtq/results/logs), --repeats (default 3, each number the least of
# them after a warm-up, as the pbrt and wosx benchmarks take theirs),
# --side (default 1024: the primary image's side for every system and
# every query, so that the figures stand on one set of queries; until
# 10-30 Embree's rays ran at 2048, for runs over a tenth of a second, and
# FCPW's at 1024 for its slower rate), --fresh to measure every cell again rather
# than resuming, --plot-only to skip the measuring, --commit C to stamp the
# rows with that commit rather than the tree's (logs measured under a
# compiler since committed, folded again). RTQ_SCENES names the
# pbrt-v4-scenes checkout (default ~/projects/pbrt-v4-scenes);
# BONSAI_BUILD_DIR the compiler's build directory (compare.sh's rule
# otherwise); RTQ_CPUS the core to pin to (compare.sh's rule otherwise);
# RTQ_HOLDER the session the lock names while this measures (default rtq).
#
# What a run does. Before each mesh the machine is checked idle -- nothing
# but this at over half a core, no other process on the GPU, and no other
# benchmark's claim in /tmp/benchmark.lock (the sessions' convention: one
# line naming the holder, written while a measurement runs; this script
# writes its own while it measures and removes it when done) -- and a busy
# machine is waited for, up to ten minutes, then the run stops and says so:
# it never measures beside another load. `touch <out>/PAUSE` holds it at the next mesh boundary until
# the file goes. Each cell's log is written as `<log>.partial` and renamed
# when complete, so a log that exists is a complete run and a stopped sweep
# resumes with the same command (--fresh clears them first). The logs are
# folded by tocsv.sh, each row stamped with the commit the compiler was
# built from (`<hash>+` when the tree has uncommitted compiler changes), the
# rows of cells not measured this time carried over from the CSV as it was;
# then plot.py draws the figures (the paper's three: rtq-queries-cpu, the
# three CPU queries over Embree and FCPW, and rtq-fcpwb-cpu and -gpu, the
# walks' queries over FCPW's scalar trees; and the three older ones, a query
# each) and prints, per reference and ray or point set, the geomean over the
# plotted meshes with the cells under 0.97 named -- the rule the comparison
# is judged by.
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
SIDE=1024
REPEATS=3
FRESH=0
PLOT_ONLY=0
COMMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --meshes) MESHES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --side) SIDE="$2"; shift 2 ;;
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
# half a core means someone else is measuring or compiling; so does a
# process on the GPU (the GPU cells share the card with the other sessions'
# GPU runs), and so does another benchmark's claim in the lock file.
LOCK=/tmp/benchmark.lock
OWN_LOCK=0
machine_busy() {
  if [[ -e "$LOCK" && "$OWN_LOCK" -eq 0 ]]; then
    echo "  $LOCK is held: $(cat "$LOCK" 2>/dev/null)"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 &&
     [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    echo "  a process is on the GPU: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';')"
    return 0
  fi
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
  # The claim, while this measures: the holder and the start time, as the
  # sessions write it; removed when the sweep ends or is stopped.
  if [[ "$OWN_LOCK" -eq 0 ]]; then
    echo "${RTQ_HOLDER:-rtq} apps/rtq/benchmark.sh since $(date +%F\ %T)" > "$LOCK"
    OWN_LOCK=1
    trap 'if [[ "$OWN_LOCK" -eq 1 ]]; then rm -f "$LOCK"; fi' EXIT
  fi
}

# The schedules by the section they run in: Embree's (embree, tuned,
# embree4, tuned4), the packet ones, FCPW's over its Mbvh (fcpw<b>w<w>[t]),
# FCPW's over its scalar trees (fcpwb, fcpws: FCPW built scalar, the driver
# RTQ_FCPW_BINARY or RTQ_FCPW_SNCH) and the GPU ones (*-gpu, against FCPW's
# GPUScene), each group one compare.sh run per mesh with its own log.
EMBREE_SCHEDULES=""
FCPW_SCHEDULES=""
SCALAR_SCHEDULES=""
GPU_SCHEDULES=""
PACKET_SCHEDULES=""
for s in $(tr , ' ' <<< "$SCHEDULES"); do
  if [[ "$s" == *-gpu ]]; then GPU_SCHEDULES="$GPU_SCHEDULES${GPU_SCHEDULES:+,}$s"
  elif [[ "$s" == fcpwb || "$s" == fcpws ]]; then SCALAR_SCHEDULES="$SCALAR_SCHEDULES${SCALAR_SCHEDULES:+,}$s"
  elif [[ "$s" == fcpw* ]]; then FCPW_SCHEDULES="$FCPW_SCHEDULES${FCPW_SCHEDULES:+,}$s"
  elif [[ "$s" == packet* ]]; then PACKET_SCHEDULES="$PACKET_SCHEDULES${PACKET_SCHEDULES:+,}$s"
  else EMBREE_SCHEDULES="$EMBREE_SCHEDULES${EMBREE_SCHEDULES:+,}$s"; fi
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
    # Embree's schedules, every query in one run (until 10-30 three runs,
    # the rays at their own side).
    if [[ -n "$EMBREE_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$SIDE.embree.log" --schedule "$EMBREE_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" "$path"
    fi
    # The packet schedules: the ray queries against Embree's packets; no
    # closest-point run, which Embree has no packet form of.
    if [[ -n "$PACKET_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$SIDE.packet.intersect.log" --schedule "$PACKET_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" --query intersect "$path"
      run "$OUT/$mesh.s$SIDE.packet.occluded.log" --schedule "$PACKET_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" --query occluded "$path"
    fi
    if [[ -n "$FCPW_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$SIDE.fcpw.log" --schedule "$FCPW_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" "$path"
    fi
    # FCPW's scalar trees and its GPU, every query of each (the silhouette
    # query included, for fcpws).
    if [[ -n "$SCALAR_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$SIDE.fcpw-scalar.log" --schedule "$SCALAR_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" "$path"
    fi
    if [[ -n "$GPU_SCHEDULES" ]]; then
      run "$OUT/$mesh.s$SIDE.gpu.log" --schedule "$GPU_SCHEDULES" --side "$SIDE" --repeats "$REPEATS" "$path"
    fi
  done
  echo "[$(date +%T)] measured: $OUT"
  if [[ "$OWN_LOCK" -eq 1 ]]; then rm -f "$LOCK"; OWN_LOCK=0; fi
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
