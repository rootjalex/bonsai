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
#                              pbrt --gpu on all thirty scenes x 16/64/128
#                              spp, from the last heatmap run
#                              (plot_gpu_heatmap.py)
#   plots/pbrt-gpu-profile.pdf the wavefront's trace kernel against the
#                              megakernels under Nsight Compute, a row of
#                              three panels per scene (kernel size, lanes
#                              active, where a warp's cycles go), from
#                              profile_kernels.sh's reports in profile-out
#                              (plot_gpu_profile.py; the user, 10-28)
#   plots/scene_statistics.txt every scene of the figures described -- its
#                              triangles, meshes, instances, materials and
#                              their kinds, textures, lights, resolution,
#                              depth, rays per camera ray, pbrt --gpu's time
#                              -- from pbrt's .stats beside the tables'
#                              images and the converter's logs, the sweep's
#                              scenes first (scene_statistics.py; the user,
#                              10-29), with a .tex twin the paper can input
#
# Each plotter also writes <figure>.txt beside its figure: the tables it
# was drawn from, the summary statistics the paper quotes (geomeans, min
# and max and which cell each is, what each scheduling decision bought,
# the trend with spp), for the sweeps the best schedule on each scene and
# the measured reasons it is best there (the user, 10-29), and where the
# measured causes are written up (the user, 10-24). The colours are one
# scheme across the four figures (figure_style.SCHEDULE_COLOURS).
#
# --png draws a .png beside each .pdf to look at. The runs each figure is
# drawn from are the variables below; override one in the environment to
# draw from another run (CPU_CSVS="..." bash apps/pbrt/plots.sh). Several
# CSVs or tables are joined as the plotters join them: a later run's row
# replaces an earlier one's, every side against the reference beside it.
set -euo pipefail
PREFIX=$(cd "$(dirname "$0")" && pwd)
OUT="$PREFIX/plots"
# The CPU sweep of the night of 10-28 (02:28-07:41): cpu_compare.sh's nine
# schedules on the five scenes at 16/64/128 spp under the compiler at
# 7af5bd86 (the exit merge in place of the funnel). The 10-23/24 tables
# it replaces are cpu-compare-out/ and cpu-compare-out-sortfix/.
CPU_CSVS="${CPU_CSVS:-$PREFIX/cpu-compare-out-head/cpu-results.csv}"
# The GPU sweep of the same night (23:15-00:45): the paper figure's eight
# sides on the five scenes at 16/64/128 spp under the same compiler,
# every schedule's registers at OptiX's own choice (BONSAI_GPU_MAX_REGISTERS=0,
# as pbrt compiles its own modules; the raygens land at 128, the SER
# pipeline at 255 -- see the profiles). The night of 10-27's tables under
# 39c67ab9 (the exit funnel) are gpu-compare-out-d/. The exploration sides
# of 10-24 (wf-b128, wf-b512, wf-fused, wf-fused-ser, wf-sort36,
# mk-ser-ray) are in gpu-compare-out-explore-b/table.tsv, an older
# compiler's; to draw them, list that table here too and name them in
# GPU_SIDES.
GPU_TABLES="${GPU_TABLES:-$PREFIX/gpu-compare-out-e/table.tsv}"
# The heatmap of the same night (00:45-02:22), the gpu-optix renderer
# reused from the sweep; compare-out-heatmap11/ is 10-27's under 39c67ab9.
HEATMAP_TABLE="${HEATMAP_TABLE:-$PREFIX/compare-out-heatmap12/table.tsv}"
# The GPU sweep's sides, in drawing order (plot_gpu_wavefront.py's SIDES
# names them): the megakernel family, then the wavefront's.
GPU_SIDES="${GPU_SIDES:-gpu-optix-mega,gpu-optix-mega-pass,gpu-optix-mega-flat,gpu-optix-mega-blocked,gpu-optix-mega-ser,gpu-optix-mega-ser-flat,gpu-optix,gpu-optix-sort}"
# The profile figure's launches (the user, 10-28): profile_kernels.sh's
# reports under profile-out-e (02:22-02:28 the same night, on the sweep's
# renderers) -- the wavefront's trace kernel and the two flat megakernels
# on ganesha, where the wavefront wins, and on killeroo-gold, where the SER
# megakernel does, at 16 spp. profile-out/ holds 10-28 daytime's under
# 39c67ab9.
PROFILES="${PROFILES:-$PREFIX/profile-out-e}"
PROFILE_CELLS="${PROFILE_CELLS:-ganesha:16 killeroo-gold:16}"
PROFILE_SIDES="${PROFILE_SIDES:-gpu-optix,gpu-optix-mega-flat,gpu-optix-mega-ser-flat}"
png=0
[[ "${1:-}" == "--png" ]] && png=1

for f in $CPU_CSVS $GPU_TABLES $HEATMAP_TABLE; do
  [[ -f "$f" ]] || { echo "plots.sh: no $f" >&2; exit 1; }
done
[[ -d "$PROFILES" ]] || { echo "plots.sh: no $PROFILES (profile_kernels.sh writes it)" >&2; exit 1; }
cells=()
for c in $PROFILE_CELLS; do cells+=(--cell "$c"); done
mkdir -p "$OUT"

# The scenes described: the sweep's (the GPU tables') first, then the
# heatmap's, from pbrt's .stats beside each table's pbrt image.
python3 "$PREFIX/scene_statistics.py" $HEATMAP_TABLE --sweep $GPU_TABLES -o "$OUT/scene_statistics.txt"

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
  python3 "$PREFIX/plot_gpu_profile.py" "${cells[@]}" --sides "$PROFILE_SIDES" --profiles "$PROFILES" -o "$OUT/pbrt-gpu-profile.$ext"
done
ls -la "$OUT"
