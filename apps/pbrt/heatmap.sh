#!/usr/bin/env bash
# The speedup heatmap: every scene the converter takes, rendered by
# `pbrt --gpu` and by this renderer's gpu-optix schedule (the matching one,
# pbrt's wavefront on the RT cores), at several sample counts, drawn as the
# grid plot_gpu_heatmap.py makes -- scenes across, counts down, each cell
# the wall-time speedup, hatched when the images disagree.
#
#     apps/pbrt/heatmap.sh [--scenes "d/s ..."] [--spp "N ..."]
#                          [--out DIR] [--repeats N] [--title TEXT]
#
# Defaults: every scene known to convert (the 29 of the 2026-10-04 sweep
# and the three verification additions), sample counts 16 64 128, repeats 3
# (each cell the least of three on each side, the sweep's precedent), out
# apps/pbrt/compare-out-heatmap. The run is resumable: gpu_compare.sh is
# called once per scene with --resume into one directory, so a scene with
# every row already in the table is skipped and an interrupted scene is
# redone -- rerun the same command after an update and only the missing or
# begun cells are measured. Each scene runs under a watchdog (45 minutes),
# and no new scene starts after the cutoff (5 hours), so an overnight run
# ends by morning whatever a scene does; both override by environment
# (HEATMAP_SCENE_TIMEOUT, HEATMAP_CUTOFF, seconds). The figure is drawn at
# the end either way: OUT/heatmap-gpu-optix.png.
set -uo pipefail
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"

# bunny-cloud and kroken were held out while the 595 driver's nvidia_uvm
# lazy-free fault (the crashes of 2026-09-29, 10-04 and 10-07) stood;
# the user lifted the hold on 2026-10-07 once driver 615.71.09 was in.
ALL_SCENES="killeroos/killeroo-simple killeroos/killeroo-gold \
killeroos/killeroo-coated-gold pbrt-book/book ganesha/ganesha \
barcelona-pavilion/pavilion-day zero-day/frame25 lte-orb/lte-orb-simple-ball \
lte-orb/lte-orb-silver lte-orb/lte-orb-rough-glass head/head \
sssdragon/dragon_10 villa/villa-lights-on smoke-plume/plume \
bunny-cloud/bunny-cloud explosion/explosion disney-cloud/disney-cloud \
landscape/view-0 zero-day/frame380 crown/crown kroken/camera-1 \
sportscar/sportscar-sky bmw-m6/bmw-m6 dambreak/dambreak0 \
transparent-machines/frame542 villa/villa-daylight clouds/clouds \
sanmiguel/sanmiguel-courtyard bistro/bistro_cafe bistro/bistro_vespa \
watercolor/camera-1 sanmiguel/sanmiguel-courtyard-second"

SCENES="$ALL_SCENES"
SPPS="16 64 128"
OUT="$PREFIX/compare-out-heatmap"
REPEATS=3
TITLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes) SCENES="$2"; shift 2 ;;
    --spp) SPPS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

SCENE_TIMEOUT="${HEATMAP_SCENE_TIMEOUT:-2700}"
CUTOFF="${HEATMAP_CUTOFF:-18000}"
START=$(date +%s)
mkdir -p "$OUT"

for sc in $SCENES; do
  now=$(date +%s)
  if (( now - START > CUTOFF )); then
    echo "heatmap: the cutoff passed; no new scene (resume to finish)" \
      | tee -a "$OUT/heatmap.log"
    break
  fi
  echo "=== $sc ($(date +%H:%M:%S))" | tee -a "$OUT/heatmap.log"
  timeout "$SCENE_TIMEOUT" bash "$PREFIX/gpu_compare.sh" \
      --spp "$SPPS" --repeats "$REPEATS" --out "$OUT" --resume \
      --schedules gpu-optix --scenes "$sc" \
      >> "$OUT/heatmap.log" 2>&1
  status=$?
  if (( status == 124 )); then
    echo "    $sc hit the scene watchdog; its begun cells rerun on resume" \
      | tee -a "$OUT/heatmap.log"
  elif (( status != 0 )); then
    echo "    $sc failed (exit $status); see $OUT/heatmap.log" \
      | tee -a "$OUT/heatmap.log"
  fi
done

python3 "$PREFIX/plot_gpu_heatmap.py" "$OUT/heatmap-gpu-optix.png" \
    "$OUT/table.tsv" --schedule gpu-optix --sort name \
    ${TITLE:+--title "$TITLE"}
echo "heatmap: $OUT/heatmap-gpu-optix.png"
