#!/bin/bash
# The wosx table, end to end: every mesh against WoSX under both matching
# schedules -- one thread against one thread, the threads against the
# threads -- the results folded into apps/wosx/results/wosx-results.csv, and
# the table printed with its geomeans. One command to re-run after a
# compiler change, from the repository root, inside the `bonsai` conda
# environment, with the compiler built:
#
#     apps/wosx/benchmark.sh
#     apps/wosx/benchmark.sh --meshes "whale dragon" --schedules fcpw4w16
#     apps/wosx/benchmark.sh --table-only        # fold, print and plot what is logged
#
# The plot (plot.py) goes into apps/wosx/plots: the meshes along the x axis
# by triangle count, two bars each, one thread against one thread and the
# threads against the threads.
#
# The meshes: WoSX's own demo meshes (deps/wosx/demo_apps/*/data -- the comb
# electrodes, the whale, the car, the octopus, the beast, the rover, which
# is unzipped once into the results directory) and the thirteen pbrt
# meshes apps/rtq measures, in order of triangle count. Whether a surface
# is closed the driver reads off the mesh (an edge of one triangle is a
# rim): closed, the sample points are the slice's interior; open, the
# whole slice.
#
# Options: --schedules (default fcpw4w16,fcpw4w16-threads), --meshes
# (default all, names as below), --out DIR for the logs (default
# apps/wosx/results/logs), --side / --walks / --repeats (default 128 / 64 /
# 3: the driver's slice, walks per point, and runs per side with the least
# kept), --fresh to measure every cell again rather than resuming,
# --table-only to skip the measuring. RTQ_SCENES names the pbrt-v4-scenes
# checkout (default ~/projects/pbrt-v4-scenes); BONSAI_BUILD_DIR the
# compiler's build directory (compare.sh's rule otherwise); RTQ_CPUS the
# cores to pin to (compare.sh's rule otherwise).
#
# What a run does. Before each mesh the machine is checked idle -- nothing
# but this at over half a core -- and a busy machine is waited for, up to
# ten minutes, then the run stops and says so: it never measures beside
# another load. `touch <out>/PAUSE` holds it at the next mesh boundary until
# the file goes. Each mesh's log is written as `<log>.partial` and renamed
# when complete, so a log that exists is a complete run and a stopped sweep
# resumes with the same command (--fresh clears them first). The logs are
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
SCHEDULES="fcpw4w16,fcpw4w16-threads"
OUT="apps/wosx/results/logs"
CSV="apps/wosx/results/wosx-results.csv"
PLOTS="apps/wosx/plots"
SIDE=128
WALKS=64
REPEATS=3
FRESH=0
TABLE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --meshes) MESHES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --side) SIDE="$2"; shift 2 ;;
    --walks) WALKS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --table-only) TABLE_ONLY=1; shift ;;
    -h|--help) sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
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
# half a core means someone else is measuring or compiling.
machine_busy() {
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
    run "$OUT/$mesh.s$SIDE.w$WALKS.log" --schedule "$SCHEDULES" --side "$SIDE" --walks "$WALKS" --repeats "$REPEATS" "$path"
  done
  echo "[$(date +%T)] measured: $OUT"
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
  mesh="$(basename "$log")"; mesh="${mesh%%.s*}"
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

# The table and the geomeans, in order of triangle count.
python3 - "$CSV" <<'EOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for schedule in sorted({r["schedule"] for r in rows}):
    sel = sorted((r for r in rows if r["schedule"] == schedule), key=lambda r: int(r["triangles"]))
    print(f"\n=== {schedule} ({sel[0]['cpus'] if sel else ''}; rates in million steps per second)")
    print(f"{'mesh':<10} {'triangles':>9} {'closed':>6} {'points':>7} {'WoSX':>7} {'bonsai':>7} {'speedup':>8}  "
          f"{'walk length':>15}  {'rms error':>19}  {'z':>5}")
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
# escapes on its first step).
python3 apps/wosx/plot.py "$CSV" -o "$PLOTS"
