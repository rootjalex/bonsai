#!/usr/bin/env bash
# The figures of apps/pbrt, drawn into plots/ under fixed names from the
# runs that are the current measurements (the user, 10-24: "the plots
# directory should only have pbrt-cpu-sweep.pdf, pbrt-gpu-sweep.pdf, and the
# heatmap"). Nothing else belongs in plots/, and nothing in it is
# committed (.gitignore); this script is how every figure is redrawn.
#
#   bash apps/pbrt/plots.sh [--png]
#
#   plots/pbrt-cpu-sweep.pdf   the CPU schedules' speedup over pbrt's CPU
#                              wavefront, five scenes x 16/64/128 spp, from
#                              cpu_compare.sh's CSVs (plot_cpu_wavefront.py)
#   plots/pbrt-gpu-sweep.pdf   every measured OptiX schedule's speedup over
#                              pbrt --gpu on the same cells, from
#                              gpu_compare.sh's tables (plot_gpu_wavefront.py)
#   plots/pbrt-gpu-heatmap.pdf the matching GPU wavefront (gpu-optix) over
#                              pbrt --gpu on all thirty scenes x 16/64/128/256
#                              spp, from the last heatmap run
#                              (plot_gpu_heatmap.py)
#
# Each plotter also writes <figure>.txt beside its figure: the summary
# statistics the paper quotes (geomeans, min and max and which cell each
# is, what each scheduling decision bought, the trend with spp) and where
# the measured causes are written up (the user, 10-24).
#
# --png draws a .png beside each .pdf to look at. The runs each figure is
# drawn from are the variables below; override one in the environment to
# draw from another run (CPU_CSVS="..." bash apps/pbrt/plots.sh). Several
# CSVs or tables are joined as the plotters join them: a later run's row
# replaces an earlier one's, every side against the reference beside it.
set -euo pipefail
PREFIX=$(cd "$(dirname "$0")" && pwd)
OUT="$PREFIX/plots"
CPU_CSVS="${CPU_CSVS:-$PREFIX/cpu-compare-out/cpu-results.csv $PREFIX/cpu-compare-out-sortfix/cpu-results.csv}"
# The GPU sweep of 10-27 evening: the paper figure's seven sides on the
# five scenes at 16/64/128 spp under one compiler (9846e3e6, the loopify
# exit funnel on top of the specialize fix and the device min/max). The
# exploration sides of 10-24 (wf-b128, wf-b512, wf-fused, wf-fused-ser,
# wf-sort36, mk-ser-ray) are in gpu-compare-out-explore-b/table.tsv, an
# older compiler's; to draw them, list that table here too and name them
# in GPU_SIDES.
GPU_TABLES="${GPU_TABLES:-$PREFIX/gpu-compare-out-c/table.tsv}"
# The heatmap's redo of the same evening under the same compiler.
HEATMAP_TABLE="${HEATMAP_TABLE:-$PREFIX/compare-out-heatmap10/table.tsv}"
# The GPU sweep's sides, in drawing order (plot_gpu_wavefront.py's SIDES
# names them): the megakernel family, then the wavefront's.
GPU_SIDES="${GPU_SIDES:-gpu-optix-mega,gpu-optix-mega-pass,gpu-optix-mega-flat,gpu-optix-mega-ser,gpu-optix-mega-ser-flat,gpu-optix,gpu-optix-sort}"
png=0
[[ "${1:-}" == "--png" ]] && png=1

for f in $CPU_CSVS $GPU_TABLES $HEATMAP_TABLE; do
  [[ -f "$f" ]] || { echo "plots.sh: no $f" >&2; exit 1; }
done
mkdir -p "$OUT"

formats=(pdf)
(( png )) && formats+=(png)
for ext in "${formats[@]}"; do
  python3 "$PREFIX/plot_cpu_wavefront.py" $CPU_CSVS -o "$OUT/pbrt-cpu-sweep.$ext"
  # The paper's four GPU sides (10-24) and the three megakernel shapes of
  # 10-27 (the user: the GPU figures other than the heatmap gain them):
  # seven, within the ramp's nine; the exploration sides the tables also
  # hold (wf-b128, wf-b512, wf-fused, wf-sort36, mk-ser-ray) are drawn by
  # naming them in GPU_SIDES.
  python3 "$PREFIX/plot_gpu_wavefront.py" $GPU_TABLES --sides "$GPU_SIDES" -o "$OUT/pbrt-gpu-sweep.$ext"
  python3 "$PREFIX/plot_gpu_heatmap.py" "$OUT/pbrt-gpu-heatmap.$ext" "$HEATMAP_TABLE" --sort geometry
done
ls -la "$OUT"
