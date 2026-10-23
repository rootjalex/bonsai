#!/usr/bin/env bash
# The CPU comparison: pbrt's CPU wavefront against every CPU schedule of this
# renderer, on the scenes given, at the sample counts given -- one command,
# one CSV, which plot_cpu_wavefront.py draws.
#
#     apps/pbrt/cpu_compare.sh [--spp "N ..."] [--repeats N] [--out DIR] [--resume]
#                              [--schedules "a b c"] [--scenes "d/s ..."]
#
# Every number in the CSV is written by this script from a compare.sh run
# whose log sits beside it; nothing is typed in. A cell is one scene at one
# sample count. For each cell the script runs compare.sh once per schedule,
# one at a time (compare.sh shares apps/pbrt/render.{h,o}), scene-major so
# that a scene's whole column is in hand before the next scene starts:
#
# - `scalar` (the path-per-sample schedule, pbrt's own CPU structure) against
#   pbrt's CPU volpath integrator, which gives the rows `pbrt-volpath` and
#   `scalar`;
# - every other schedule against pbrt's CPU wavefront (compare.sh
#   --pbrt-wavefront), the algorithm they share, which gives the rows
#   `pbrt-wavefront` (once per cell) and one row per schedule.
#
# Each side is the least of REPEATS renders, as compare.sh takes them (pbrt as
# REPEATS processes reading its own render clock, ours as one process
# rendering REPEATS times). pbrt's wavefront is rendered once per cell and
# kept for the cell's other schedules (compare.sh --pbrt-cache, under
# --out/pbrt-cache), so that every schedule of a cell is measured against the
# same reference time and the same reference image. Every image is checked
# against pbrt's as compare.sh checks it, and the CSV carries the verdict.
#
# The CSV (--out/cpu-results.csv) has one row per (scene, spp, side):
#
#     scene,spp,side,seconds,speedup,verdict,repeats,load,measured,log
#
# `side` is `pbrt-wavefront`, `pbrt-volpath` or a schedule's name; `seconds`
# is that side's render time; `speedup` is pbrt's wavefront time over this
# side's, blank on pbrt's wavefront row; `verdict` is compare.sh's
# (`ok ...` when the image matched pbrt's, `-` on pbrt's rows); `load` is the
# machine's one-minute load average before the run, to be read against the
# core count when a number looks slow; `measured` is when the run finished;
# `log` the compare.sh log the row was read from. `--resume` continues a
# run into the same --out: a row already in the CSV is not measured again.
#
# Scenes are named `<dir>/<name>` under $SCENES_DIR (~/projects/
# pbrt-v4-scenes); the default five are of rising cost -- two shapes and a
# diffuse floor (killeroo-simple), the same under a conductor (killeroo-gold),
# a textured mesh (book), a million-triangle mesh (ganesha), and a large
# architectural scene of many materials (sanmiguel-courtyard) -- and
# `--scenes` takes any list scene_dump converts. The default schedules are
# the CPU wavefront study's (REPORT.md, 2026-10-21); `--schedules` takes any
# list from schedules/ and `scalar` is always run first against pbrt's
# volpath. Everything the run acts on is checked before any of it starts.
set -uo pipefail
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"

SPPS="16 64"
REPEATS=3
OUT="$PREFIX/cpu-compare-out"
SCHEDULES="scalar wavefront-frame wavefront-frame-packet wavefront-frame-packet-reorder \
wavefront-thread-packet wavefront-thread-packet-reorder"
SCENES="killeroos/killeroo-simple killeroos/killeroo-gold pbrt-book/book \
ganesha/ganesha sanmiguel/sanmiguel-courtyard"
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spp) SPPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
done

PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
SCENES_DIR="${SCENES_DIR:-$HOME/projects/pbrt-v4-scenes}"
BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
fail=0
for tool in "$PBRT" "$IMGTOOL" "$BONSAI_BUILD_DIR/compiler"; do
  if [[ ! -x "$tool" ]]; then echo "missing: $tool" >&2; fail=1; fi
done
for s in $SCHEDULES; do
  if [[ ! -f "$PREFIX/schedules/$s.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$s.bonsai" >&2; fail=1
  fi
done
for sc in $SCENES; do
  if [[ ! -f "$SCENES_DIR/$sc.pbrt" ]]; then
    echo "no scene $SCENES_DIR/$sc.pbrt" >&2; fail=1
  fi
done
for SPP in $SPPS; do
  if ! [[ "$SPP" =~ ^[0-9]+$ && "$SPP" -ge 1 ]]; then
    echo "--spp takes positive integers, not '$SPP'" >&2; fail=1
  fi
done
if ! [[ "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 1 ]]; then
  echo "--repeats takes a positive integer" >&2; fail=1
fi
if [[ -e "$OUT" && "$RESUME" == 0 ]]; then
  echo "$OUT exists; --resume continues into it, or name another --out" >&2; fail=1
fi
if [[ "$fail" == 1 ]]; then exit 1; fi

mkdir -p "$OUT/logs" "$OUT/images"
CSV="$OUT/cpu-results.csv"
if [[ ! -f "$CSV" ]]; then
  echo "scene,spp,side,seconds,speedup,verdict,repeats,load,measured,log" > "$CSV"
fi

# `scalar` first (against pbrt's volpath), then the rest in the order given.
WAVEFRONT_SCHEDULES=""
for s in $SCHEDULES; do
  if [[ "$s" != scalar ]]; then WAVEFRONT_SCHEDULES="$WAVEFRONT_SCHEDULES $s"; fi
done

have_row() {  # scene spp side
  awk -F, -v a="$1" -v b="$2" -v c="$3" '$1==a && $2==b && $3==c {found=1} END {exit !found}' "$CSV"
}

# A field of a compare.sh log. The lines read are the ones compare.sh prints:
# `pbrt: rendered <scene> in <s>s (...)`, the driver's `render seconds: <s>`,
# and compare_gbuffer.py's verdict, the line starting `ok` or `FAILED`.
pbrt_seconds_in() { sed -n 's/^pbrt: rendered .* in \([0-9.]*\)s (.*/\1/p' "$1" | head -1; }
ours_seconds_in() { sed -n 's/^render seconds: \([0-9.]*\)$/\1/p' "$1" | head -1; }
verdict_in() { grep -m1 -E '^(ok|FAILED)' "$1" | tr ',' ';' | cut -c1-160; }
speedup_of() { awk -v a="$1" -v b="$2" 'BEGIN { if (a != "" && b != "" && b > 0) printf "%.3f", a / b; }'; }
load_now() { cut -d' ' -f1 /proc/loadavg; }

csv_row() {  # scene spp side seconds speedup verdict load log
  echo "$1,$2,$3,$4,$5,\"$6\",$REPEATS,$7,$(date '+%Y-%m-%d %H:%M:%S'),$8" >> "$CSV"
}

# One compare.sh run: the log, the images copied out, the rows written.
# $1 scene, $2 spp, $3 schedule, $4 pbrt side name, $5... extra compare.sh flags.
run_cell() {
  local sc="$1" spp="$2" sd="$3" pside="$4"; shift 4
  local name; name="$(basename "$sc")-s$spp-$sd"
  local log="$OUT/logs/$name.log"
  local load; load="$(load_now)"
  local t0; t0=$(date +%s)
  echo "== $name (load $load) at $(date +%H:%M:%S)"
  REPEATS="$REPEATS" WORK="$OUT/work" BONSAI_BUILD_DIR="$BONSAI_BUILD_DIR" \
    bash "$PREFIX/compare.sh" "$@" --spp "$spp" --schedule "$sd" \
    "$SCENES_DIR/$sc.pbrt" > "$log" 2>&1
  local st=$?
  local p o v
  p="$(pbrt_seconds_in "$log")"; o="$(ours_seconds_in "$log")"; v="$(verdict_in "$log")"
  if [[ $st -ne 0 || -z "$p" || -z "$o" || -z "$v" ]]; then
    echo "   compare.sh exit $st; pbrt '$p' ours '$o' verdict '$v' -- see $log"
    v="${v:-FAILED: compare.sh exit $st}"
  fi
  cp "$OUT/work/bonsai-radiance.png" "$OUT/images/$name.png" 2>/dev/null
  cp "$OUT/work/pbrt-radiance.png" "$OUT/images/$(basename "$sc")-s$spp-$pside.png" 2>/dev/null
  if ! have_row "$(basename "$sc")" "$spp" "$pside"; then
    csv_row "$(basename "$sc")" "$spp" "$pside" "$p" "" "-" "$load" "$log"
  fi
  # The speedup is over pbrt's wavefront: this run's pbrt when that is what
  # it ran against, else the cell's wavefront time already in the table.
  local ref="$PBRT_WAVEFRONT_SECONDS"
  if [[ "$pside" == pbrt-wavefront ]]; then ref="$p"; fi
  csv_row "$(basename "$sc")" "$spp" "$sd" "$o" "$(speedup_of "$ref" "$o")" "$v" "$load" "$log"
  echo "   pbrt $pside ${p}s, $sd ${o}s, $v; $(( $(date +%s) - t0 )) s"
}

echo "cpu_compare.sh: scenes [$SCENES], spp [$SPPS], schedules [$SCHEDULES], repeats $REPEATS, out $OUT, from $(date '+%Y-%m-%d %H:%M:%S')"
for sc in $SCENES; do
  for spp in $SPPS; do
    scene="$(basename "$sc")"
    # pbrt's wavefront time for the speedup column, rendered by the first
    # wavefront run of the cell and read back by the rest from the cache.
    PBRT_WAVEFRONT_SECONDS=""
    for sd in $WAVEFRONT_SCHEDULES; do
      if have_row "$scene" "$spp" "$sd"; then
        PBRT_WAVEFRONT_SECONDS="$(awk -F, -v a="$scene" -v b="$spp" '$1==a && $2==b && $3=="pbrt-wavefront" {print $4}' "$CSV" | head -1)"
        continue
      fi
      run_cell "$sc" "$spp" "$sd" pbrt-wavefront --pbrt-wavefront --pbrt-cache "$OUT/pbrt-cache"
      if [[ -z "$PBRT_WAVEFRONT_SECONDS" ]]; then
        PBRT_WAVEFRONT_SECONDS="$(awk -F, -v a="$scene" -v b="$spp" '$1==a && $2==b && $3=="pbrt-wavefront" {print $4}' "$CSV" | head -1)"
      fi
    done
    if [[ " $SCHEDULES " == *" scalar "* ]] && ! have_row "$scene" "$spp" scalar; then
      run_cell "$sc" "$spp" scalar pbrt-volpath --pbrt-cache "$OUT/pbrt-cache"
    fi
  done
done
echo "done $(date '+%Y-%m-%d %H:%M:%S'); the table is $CSV"
