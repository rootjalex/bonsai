#!/bin/bash
# The wosx table, end to end: every mesh against WoSX in parallel under the
# schedule WoSX runs -- its tbb::parallel_for over the points against the
# program's parfor on the same threads, and the persistent workers against
# the same -- the results folded into apps/wosx/results/wosx-results.csv,
# and the table printed with its geomeans. One command to re-run after a
# compiler change, from the repository root, inside the `bonsai` conda
# environment, with the compiler built:
#
#     apps/wosx/benchmark.sh
#     apps/wosx/benchmark.sh --meshes "whale dragon" --schedules fcpw4w16
#     apps/wosx/benchmark.sh --table-only        # fold, print and plot what is logged
#     apps/wosx/benchmark.sh --gpu               # the GPU table: WoSX's GPU solver against the GPU schedules
#     apps/wosx/benchmark.sh --stars             # the walk on stars' table (wost.bonsai against WoSX's WalkOnStars)
#     apps/wosx/benchmark.sh --stars --gpu       # the walk on stars on the GPU
#
# The figures (plot.py) go into apps/wosx/plots, one per tree
# (wosx-cpu-fcpw4w16-speedup.*, wosx-cpu-fcpwb-speedup.*): the meshes along
# the x axis by triangle count, two bars each -- the threads against the
# threads, the persistent workers against the threads -- and on WoSX's
# default tree the exploration's bars beside them (plot.py's
# CPU_EXTRA_BARS: the layout where a node holds its children's boxes,
# threaded and persistent; the walk vectorized over a point's walks in
# gangs of 32 and of 16 lanes, threaded). One thread against
# one thread (the bare schedules, `fcpwb`, `fcpw4w16`, `wost-fcpwb`) is
# measured by name alone (--schedules) and drawn by none of the figures:
# an application runs in parallel under the reference's schedule, and the
# single thread's question -- does the compiled traversal match the
# hand-written one -- is apps/rtq's, asked there (the user, 2026-10-28; on
# the large meshes that cell ran WoSX's single thread for an hour).
#
# --gpu runs the GPU comparison instead: the schedules default to the GPU
# ones (fcpwb-gpu, the schedule matching WoSX's persistent-thread kernel,
# its variants and the launches through OptiX, README.md's list;
# fcpwb-gpu-point, a point a claim, left the defaults and the figure on
# 2026-10-28 at a tenth of WoSX's rate; compare.sh's *-gpu* rule builds the driver against
# slang-rhi and runs both sides on the card), the logs go to
# apps/wosx/results/gpu-logs, the CSV to apps/wosx/results/wosx-gpu-results.csv
# and the plot to apps/wosx/plots/wosx-gpu-speedup.*; the idle check also
# waits for the GPU to be empty of every other process.
#
# --stars runs the walk on stars' table instead: wost.bonsai against WoSX's
# WalkOnStars on the same meshes, each split at the slice the sample points
# lie on into an absorbing lower half carrying the Dirichlet data and a
# reflecting upper half with zero flux (compare.sh's wost-* rule builds
# the driver for it). The schedules default to the walk on stars' matching
# ones, wost-fcpwb-threads and wost-fcpwb-persistent -- WoSX's default
# trees, the threads and the persistent workers -- or
# with --gpu to wost-fcpwb-gpu, the schedule matching WoSX's GPU walk on
# stars, and the walk on spheres' GPU variants on this walk (wost-fcpwb-gpu-*,
# the same names); the logs go to apps/wosx/results/stars-logs (stars-gpu-logs), the
# CSV to apps/wosx/results/wosx-stars-results.csv (wosx-stars-gpu-
# results.csv) and the figures to wosx-stars-cpu-fcpwb-speedup.* and
# wosx-stars-gpu-speedup.*. The error column is the root-mean-square
# difference between the two sides' estimates, since a problem with a
# reflecting boundary has no closed-form solution to measure against.
#
# The meshes: WoSX's own demo meshes (deps/wosx/demo_apps/*/data -- the comb
# electrodes, the whale, the car, the octopus, the beast, the rover, which
# is unzipped once into the results directory) and the thirteen pbrt
# meshes apps/rtq measures, in order of triangle count. Whether a surface
# is closed the driver reads off the mesh (an edge of one triangle is a
# rim): closed, the sample points are the slice's interior; open, the
# whole slice.
#
# Options: --schedules (default fcpw4w16-threads,fcpw4w16-persistent,
# fcpwb-threads,fcpwb-persistent, then the exploration's fcpwb2-threads,
# fcpwb2-persistent, fcpwb-packet-threads, fcpwb-packet-16-threads: on
# each of the two trees -- FCPW's four-wide Mbvh, and WoSX's default,
# FCPW's scalar binary BVH -- the threads each, and the threads with the
# points claimed by sixteen persistent workers instead of TBB's partition;
# on WoSX's default tree also the other layout and the two packets), --meshes
# (default all, names as below), --out DIR for the logs (default
# apps/wosx/results/logs) and --csv FILE for the table folded from them
# (default apps/wosx/results/wosx-results.csv; an exploration names both,
# so that the results directory keeps the table), --side / --walks /
# --repeats (default 128 / 64 /
# 3: the driver's slice, walks per point, and runs per side with the least
# kept), --fresh to measure every cell again rather than resuming,
# --table-only to skip the measuring. RTQ_SCENES names the pbrt-v4-scenes
# checkout (default ~/projects/pbrt-v4-scenes); BONSAI_BUILD_DIR the
# compiler's build directory (compare.sh's rule otherwise); RTQ_CPUS the
# cores to pin to (compare.sh's rule otherwise).
#
# What a run does. Before each mesh the machine is checked idle -- nothing
# but this at over half a core, no other process on the GPU, and no other
# benchmark's claim in /tmp/benchmark.lock (the sessions' convention: one
# line naming the holder, written while a measurement runs; this script
# writes its own while it measures and removes it when done) -- and a busy
# machine is waited for, up to ten minutes, then the run stops and says so:
# it never measures beside another load. `touch <out>/PAUSE` holds it at
# the next mesh boundary until the file goes. Each (mesh, schedule) cell's
# log is written as `<log>.partial` and renamed when complete, so a log
# that exists is a complete run and a stopped sweep resumes with the same
# command (--fresh clears the meshes' logs first), and a schedule added
# later is measured on its own without the others being run again (the
# logs of the first runs, one per mesh with every schedule of the run in
# it, fold the same). The logs are
# folded into the CSV, one row per (mesh, schedule), each stamped with the
# date and the commit the compiler was built from (`<hash>+` when the tree
# has uncommitted compiler changes); then the table is printed -- per
# schedule, each mesh's rates, speedup, walk lengths, errors and z, and the
# geomean of the speedups with the least and the greatest named.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCENES="${RTQ_SCENES:-$HOME/projects/pbrt-v4-scenes}"
WOSX="deps/wosx/demo_apps"
WOSX_MESHES="comb whale car octopus beast rover"
RTQ_MESHES="pavilion head zero-day bmw crown ivy villa dambreak sportscar landscape lte-orb ganesha dragon"
ALL_MESHES="$WOSX_MESHES $RTQ_MESHES"
MESHES="$ALL_MESHES"
SCHEDULES=""
OUT=""
CSV=""
PLOTS="apps/wosx/plots"
SIDE=128
WALKS=64
REPEATS=3
FRESH=0
TABLE_ONLY=0
GPU=0
STARS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --meshes) MESHES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --side) SIDE="$2"; shift 2 ;;
    --walks) WALKS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --table-only) TABLE_ONLY=1; shift ;;
    --gpu) GPU=1; shift ;;
    --stars) STARS=1; shift ;;
    -h|--help) sed -n '2,88p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
# The CPU table's defaults -- the two parallel schedules on each of the two
# trees, FCPW's four-wide Mbvh (both sides vectorized) and WoSX's default,
# FCPW's scalar binary BVH (both sides as WoSX ships), and on WoSX's tree
# the exploration's four (the fcpwb2 layout threaded and persistent, the
# packets of 32 and 16 lanes threaded) -- or the GPU table's: the matching
# schedule, its variants, and the launches through OptiX (the plain
# ray-generation launch, the same with the threads reordered by position
# before every step, and the persistent claim under that reordering); or
# the walk on stars' tables, on WoSX's default trees alone, its GPU table
# the walk on spheres' schedules on that walk with the per-point deferral.
if (( STARS && GPU )); then
  SCHEDULES="${SCHEDULES:-wost-fcpwb-gpu,wost-fcpwb-gpu-parallel,wost-fcpwb-gpu-reorder,wost-fcpwb-gpu-256k,wost-fcpwb-gpu-reorder-256k,wost-fcpwb-gpu-parallel-reorder-blk64,wost-fcpwb-gpu-optix,wost-fcpwb-gpu-optix-ser,wost-fcpwb-gpu-optix-ser-persistent,wost-fcpwb-gpu-defer-block}"
  OUT="${OUT:-apps/wosx/results/stars-gpu-logs}"
  CSV="${CSV:-apps/wosx/results/wosx-stars-gpu-results.csv}"
elif (( STARS )); then
  SCHEDULES="${SCHEDULES:-wost-fcpwb-threads,wost-fcpwb-persistent}"
  OUT="${OUT:-apps/wosx/results/stars-logs}"
  CSV="${CSV:-apps/wosx/results/wosx-stars-results.csv}"
elif (( GPU )); then
  SCHEDULES="${SCHEDULES:-fcpwb-gpu,fcpwb-gpu-parallel,fcpwb-gpu-reorder,fcpwb-gpu-256k,fcpwb-gpu-reorder-256k,fcpwb-gpu-parallel-reorder-blk64,fcpwb-gpu-optix,fcpwb-gpu-optix-ser,fcpwb-gpu-optix-ser-persistent}"
  OUT="${OUT:-apps/wosx/results/gpu-logs}"
  CSV="${CSV:-apps/wosx/results/wosx-gpu-results.csv}"
else
  SCHEDULES="${SCHEDULES:-fcpw4w16-threads,fcpw4w16-persistent,fcpwb-threads,fcpwb-persistent,fcpwb2-threads,fcpwb2-persistent,fcpwb-packet-threads,fcpwb-packet-16-threads}"
  OUT="${OUT:-apps/wosx/results/logs}"
  CSV="${CSV:-apps/wosx/results/wosx-results.csv}"
fi
for s in $(tr , ' ' <<< "$SCHEDULES"); do
  [[ -f "apps/wosx/schedules/$s.bonsai" ]] || { echo "no schedule apps/wosx/schedules/$s.bonsai" >&2; exit 1; }
done
mkdir -p "$OUT" "$(dirname "$CSV")"

mesh_path() {
  case "$1" in
    comb) echo "$WOSX/electrostatics/data/mems-comb-electrodes.obj" ;;
    whale) echo "$WOSX/potential_flow/data/whale.obj" ;;
    car) echo "$WOSX/potential_flow/data/car.obj" ;;
    octopus) echo "$WOSX/geometric_deformation/data/octopus.obj" ;;
    beast) echo "$WOSX/geometric_deformation/data/beast.obj" ;;
    rover)
      # Shipped zipped; unzipped once beside the results.
      if [[ ! -f "apps/wosx/results/meshes/rover.obj" ]]; then
        mkdir -p apps/wosx/results/meshes
        unzip -q -o "$WOSX/thermal_conduction/data/rover.zip" rover.obj -d apps/wosx/results/meshes >&2
      fi
      echo "apps/wosx/results/meshes/rover.obj" ;;
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
    *) echo "unknown mesh $1 (the meshes: $ALL_MESHES)" >&2; exit 1 ;;
  esac
}

# Anything but this run's own processes and the sessions' shells at over
# half a core means someone else is measuring or compiling; so does any
# process on the GPU (a GPU measurement shares the card with nothing, and
# the other sessions' GPU runs count as the machine in use either way), and
# so does another benchmark's claim in the lock file.
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
    $2 > 50 && $3 !~ /^(claude|node|tmux|sshd|ps|awk|bash|wosx\.out)$/ { busy = 1 }
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
    echo "bonsai-ac apps/wosx/benchmark.sh since $(date +%F\ %T)" > "$LOCK"
    OWN_LOCK=1
    trap 'if [[ "$OWN_LOCK" -eq 1 ]]; then rm -f "$LOCK"; fi' EXIT
  fi
}

run() {  # run <log> <compare.sh args...>
  local log="$1"; shift
  if [[ -s "$log" ]]; then echo "  have $log"; return; fi
  echo "[$(date +%T)] -> $log"
  apps/wosx/compare.sh "$@" > "$log.partial" 2>&1 || echo "  (exit $?: a failure; see $log)"
  mv "$log.partial" "$log"
}

if (( ! TABLE_ONLY )); then
  for mesh in $MESHES; do
    mesh_path "$mesh" > /dev/null
    if (( FRESH )); then rm -f "$OUT/$mesh".s*.log "$OUT/$mesh"__*.log; fi
  done
  # A cell is one (mesh, schedule): its log `<mesh>__<schedule>.s<side>.w<walks>.log`.
  # A mesh whose log of an earlier run (`<mesh>.s<side>.w<walks>.log`, every
  # schedule of that run in it) already holds the schedule is not run again
  # for it.
  for mesh in $MESHES; do
    if [[ -e "$OUT/PAUSE" ]]; then
      echo "[$(date +%T)] paused before $mesh ($OUT/PAUSE exists)"
      while [[ -e "$OUT/PAUSE" ]]; do sleep 10; done
      echo "[$(date +%T)] resumed"
    fi
    path="$(mesh_path "$mesh")"
    for schedule in $(tr , ' ' <<< "$SCHEDULES"); do
      old="$OUT/$mesh.s$SIDE.w$WALKS.log"
      if [[ -s "$old" ]] && grep -q "^=== schedule $schedule " "$old"; then
        echo "  have $old ($schedule)"
        continue
      fi
      log="$OUT/${mesh}__$schedule.s$SIDE.w$WALKS.log"
      if [[ -s "$log" ]]; then echo "  have $log"; continue; fi
      wait_idle "$mesh ($schedule)"
      run "$log" --schedule "$schedule" --side "$SIDE" --walks "$WALKS" --repeats "$REPEATS" "$path"
    done
  done
  echo "[$(date +%T)] measured: $OUT"
  if [[ "$OWN_LOCK" -eq 1 ]]; then rm -f "$LOCK"; OWN_LOCK=0; fi
fi

# The compiler the rows were measured under: the commit, marked when the
# tree carries uncommitted compiler changes.
COMMIT="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain -- CMakeLists.txt include src 2>/dev/null)" ]]; then
  COMMIT="$COMMIT+"
fi
shopt -s nullglob
LOGS=("$OUT"/*.log)
shopt -u nullglob
(( ${#LOGS[@]} > 0 )) || { echo "no logs in $OUT" >&2; exit 1; }

# Fold the logs: one row per schedule section of each log, the fields read
# off the driver's lines (`  N triangles, ... closed|K rim edges`, `sample
# points: N of ...`, `pinned to cpu(s) ...`, and the table's one row).
echo "date,commit,machine,cpus,schedule,mesh,triangles,closed,points,walks,wosx_rate,bonsai_rate,speedup,wosx_ended,bonsai_ended,wosx_walk_length,bonsai_walk_length,wosx_rms_error,bonsai_rms_error,z,units" > "$CSV"
for log in "${LOGS[@]}"; do
  # `<mesh>.s..` for a run's log of every schedule, `<mesh>__<schedule>.s..`
  # for one cell's.
  mesh="$(basename "$log")"; mesh="${mesh%%.s*}"; mesh="${mesh%%__*}"
  awk -v date="$(date -r "$log" +%F)" -v commit="$COMMIT" -v machine="$(hostname -s)" -v mesh="$mesh" '
    /^=== schedule / { schedule = $3 }
    /^pinned to cpus? / { cpus = $4; sub(/,$/, "", cpus) }
    /^  [0-9]+ triangles, / { triangles = $1; closed = ($0 ~ /; closed/) ? 1 : 0 }
    /^sample points: / { points = $3 }
    /^ *[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+ +[0-9.]+x / {
      row = $0
      gsub(/[;,]/, " ", row)
      n = split(row, f, /[ \t]+/)
      # (a leading empty field) points walks wosx bonsai speedup endedw endedb lenw lenb errw errb z
      walks = f[3]; wosx = f[4]; bonsai = f[5]; speedup = f[6]; sub(/x$/, "", speedup)
      # cpus is quoted: a set of cores ("8-15,24-31") carries commas of its own.
      printf "%s,%s,%s,\"%s\",%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,Msteps/s\n",
        date, commit, machine, cpus, schedule, mesh, triangles, closed, points, walks,
        wosx, bonsai, speedup, f[7], f[8], f[9], f[10], f[11], f[12], f[13]
    }' "$log" >> "$CSV"
done

# The table and the geomeans, in order of triangle count. The error column
# is the rms error against the solution, or for the walk on stars the rms
# difference from WoSX's estimate.
python3 - "$CSV" "$STARS" <<'EOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
error = "rms difference" if sys.argv[2] == "1" else "rms error"
# The schedules in the order they were run (one thread, the threads, the
# persistent workers), not alphabetical.
for schedule in dict.fromkeys(r["schedule"] for r in rows):
    sel = sorted((r for r in rows if r["schedule"] == schedule), key=lambda r: int(r["triangles"]))
    print(f"\n=== {schedule} ({sel[0]['cpus'] if sel else ''}; rates in million steps per second)")
    print(f"{'mesh':<10} {'triangles':>9} {'closed':>6} {'points':>7} {'WoSX':>7} {'bonsai':>7} {'speedup':>8}  "
          f"{'walk length':>15}  {error:>19}  {'z':>5}")
    logs = []
    for r in sel:
        s = float(r["speedup"]); logs.append(math.log(s))
        print(f"{r['mesh']:<10} {int(r['triangles']):>9} {'yes' if r['closed']=='1' else 'no':>6} {int(r['points']):>7} "
              f"{float(r['wosx_rate']):>7.2f} {float(r['bonsai_rate']):>7.2f} {s:>7.2f}x  "
              f"{float(r['wosx_walk_length']):>7.2f} {float(r['bonsai_walk_length']):>7.2f}  "
              f"{float(r['wosx_rms_error']):>9.2e} {float(r['bonsai_rms_error']):>9.2e}  {float(r['z']):>5.2f}")
    if logs:
        lo = min(sel, key=lambda r: float(r["speedup"])); hi = max(sel, key=lambda r: float(r["speedup"]))
        print(f"geomean speedup {math.exp(sum(logs)/len(logs)):.3f} over {len(logs)} meshes "
              f"(least {float(lo['speedup']):.2f} {lo['mesh']}, greatest {float(hi['speedup']):.2f} {hi['mesh']})")
EOF
echo "rows: $CSV"

# The plot: one bar per (mesh, schedule), bonsai's rate over WoSX's, into
# apps/wosx/plots (plot.py prints its own geomeans over the plotted meshes;
# the landscape is left out of the plot, since nearly every walk on it
# escapes on its first step). The GPU table draws its own figure, and the
# walk on stars' tables theirs.
PLOT_FLAGS=()
if (( GPU )); then PLOT_FLAGS+=(--gpu); fi
if (( STARS )); then PLOT_FLAGS+=(--stars); fi
python3 apps/wosx/plot.py "$CSV" -o "$PLOTS" ${PLOT_FLAGS[@]+"${PLOT_FLAGS[@]}"}
