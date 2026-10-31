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
#                              wavefront, six scenes x 16/128 spp (the
#                              user's set, 10-30), from cpu_compare.sh's
#                              CSVs (plot_cpu_wavefront.py)
#   plots/pbrt-gpu-sweep.pdf   every measured OptiX schedule's speedup over
#                              pbrt --gpu, the same six scenes x 16/128 spp,
#                              from gpu_compare.sh's tables
#                              (plot_gpu_wavefront.py)
#   plots/pbrt-gpu-heatmap.pdf the matching GPU wavefront (gpu-optix) over
#                              pbrt --gpu on all thirty scenes x 16/64/128
#                              spp, from the last heatmap run
#                              (plot_gpu_heatmap.py)
#   plots/pbrt-gpu-profile.pdf the wavefront's trace kernel against the
#                              megakernels under Nsight Compute, a row of
#                              three panels per scene (SASS instructions,
#                              active threads per warp, warp stall
#                              reasons), from
#                              profile_kernels.sh's reports in profile-out
#                              (plot_gpu_profile.py; the user, 10-28)
#   plots/pbrt-gpu-depth.pdf   why the wavefronts lose on a deep scene:
#                              frame542's share of paths alive at each
#                              bounce (pbrt --gpu's statistics) beside the
#                              wall time of pbrt --gpu, the matching
#                              wavefront and the flat megakernel with
#                              reordering at depth caps 4, 16, 64, 256
#                              (the depth sweep of 10-30, depth-out/;
#                              plot_gpu_depth.py; 10-31)
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
# The CPU sweep of 10-30 (11:13-17:12, scratch/run_six_scenes.sh
# --then-cpu): cpu_compare.sh's nine sides on the six scenes at 16 and 128
# spp, three repeats, under the compiler at caa9eb9b (a gang's 32-bit
# lane offsets read unsigned -- the first CPU table on which the packet
# schedules render bistro at all). The night of 10-28's table under
# 7af5bd86 (killeroo-simple, killeroo-gold, book, sanmiguel-courtyard,
# ganesha at 16/64/128 spp) is cpu-compare-out-head/, which the four
# shared scenes repeat within two percent on every side; 10-29/30's
# stopped eight-scene run is cpu-compare-out-eight/ (108 rows, pre-fix
# compiler); the 10-23/24 tables were cpu-compare-out/ (deleted 10-30 with
# every run directory older than a week, at the user's word; PLAN has its
# numbers) and cpu-compare-out-sortfix/.
# cpu-compare-out-two/ (10-30/31 night, 23:34-02:12, the same nine sides
# and compiler) holds smoke-plume's and frame542's cells, the sweep's
# seventh and eighth scenes; its two plume scalar rows were measured again
# at 02:29 under the spread rule (cpu_compare.sh's header), the rest of
# the table as the sweep left it.
CPU_CSVS="${CPU_CSVS:-$PREFIX/cpu-compare-out-six/cpu-results.csv $PREFIX/cpu-compare-out-two/cpu-results.csv}"
# The GPU sweep of 10-30 morning (09:37-11:13, scratch/run_six_scenes.sh):
# the paper figure's eight sides on the six scenes at 16 and 128 spp (the
# user's set of 10-30: killeroo-gold, book, sanmiguel-courtyard, ganesha,
# dragon_10, bistro_vespa -- landscape out as too big, kroken and
# watercolor out so that the CPU sweep of the same scenes fits eight
# hours) under the compiler at caa9eb9b, every schedule's registers at
# OptiX's own choice (BONSAI_GPU_MAX_REGISTERS=0, as pbrt compiles its own
# modules; the raygens land at 128, the SER pipeline at 255 -- see the
# profiles). The afternoon of 10-29's eight scenes (these five, less
# dragon_10, plus landscape, kroken, watercolor) under d14da45b are
# gpu-compare-out-eight/, whose shared scenes this run repeats within a
# percent; the night of 10-28's five scenes at 16/64/128 spp under
# 7af5bd86 are gpu-compare-out-e/, the night of 10-27's under 39c67ab9
# (the exit funnel) gpu-compare-out-d/. The exploration sides of 10-24
# (wf-b128, wf-b512, wf-fused, wf-fused-ser, wf-sort36, mk-ser-ray) are
# in gpu-compare-out-explore-b/table.tsv, an older compiler's; to draw
# them, list that table here too and name them in GPU_SIDES.
# gpu-compare-out-two/ (10-30, 17:20-17:45, the same eight schedules and
# compiler) holds frame542's cells, the sweep's seventh scene (the user,
# 10-30 evening), beside clouds', which SWEEP_SCENES leaves out;
# gpu-compare-out-plume/ (22:09-22:17, the six's renderers reused) holds
# smoke-plume's, the eighth (the user: a participating medium, the one
# axis the surface scenes lack, at a fifth of clouds' cost).
# gpu-compare-out-wf3/ (10-31, 02:48-02:59) holds the fused wavefront
# (gpu-optix-fused) on the eight scenes under the compiler at b72c215b --
# the hoist carrying a size's arithmetic and a continuation with a
# reordering hint keeping the raygen shape, which change nothing of the
# eight schedules above (their renderers' key is the compiler's sources,
# and the tables of 10-30 stand); gpu-compare-out-wf3b/ (09:06-09:20) the
# fused wavefront with reordering by material (gpu-optix-fused-ser) on the
# eight, every cell matching, beside the pass-of-four wavefront's rows
# (gpu-optix-pass), which hold no time: it matched pbrt at 1 spp and dies
# at 16 spp on every scene with an illegal address (PLAN, 10-31), so the
# figure leaves that side out with a note until it renders.
GPU_TABLES="${GPU_TABLES:-$PREFIX/gpu-compare-out-six/table.tsv $PREFIX/gpu-compare-out-two/table.tsv $PREFIX/gpu-compare-out-plume/table.tsv $PREFIX/gpu-compare-out-wf3/table.tsv $PREFIX/gpu-compare-out-wf3b/table.tsv}"
# The sweeps' scenes, in the figures' order (the user, 10-29: pbrt --gpu's
# time at 128 spp, ascending), keyed by path under the scene set; both
# sweep figures draw these and only these from their tables (the
# plotters' --scenes). A scene whose CPU sweep has not run yet is left out
# of the CPU figure with a note.
SWEEP_SCENES="${SWEEP_SCENES:-killeroos/killeroo-gold,pbrt-book/book,sanmiguel/sanmiguel-courtyard,ganesha/ganesha,sssdragon/dragon_10,smoke-plume/plume,bistro/bistro_vespa,transparent-machines/frame542}"
# Where each sweep's speedup axis ends (the user, 10-30: a linear axis, cut
# at 6x on the GPU sweep, never a log axis for speedups); a bar past the
# cutoff is drawn to it with a break mark, and its value stands upright
# over the bars with a thin leader to its bar (the user's choice among six
# trials, 10-30 night: the values on one line, never stacked; frame542's
# megakernels, 6.2x to 20.4x over pbrt --gpu). The CPU sweep's bars run to
# 7.7x with thirteen of them between 6x and 8x (killeroo-gold's and book's
# per-thread wavefronts and megakernels), so its axis ends at 8x and cuts
# nothing; the same flags for the day a bar passes it.
GPU_YMAX="${GPU_YMAX:-6}"
CPU_YMAX="${CPU_YMAX:-8}"
CUT_FLAGS="--leaders --rotate-cut-labels"
# The heatmap of the same night (00:45-02:22), the gpu-optix renderer
# reused from the sweep; compare-out-heatmap11/ is 10-27's under 39c67ab9.
HEATMAP_TABLE="${HEATMAP_TABLE:-$PREFIX/compare-out-heatmap12/table.tsv}"
# The GPU sweep's sides, in drawing order (plot_gpu_wavefront.py's SIDES
# names them): the wavefront that matches pbrt's first, then its variants
# -- the ray sort, the material stage fused into the trace program, that
# with shader execution reordering by material, and passes of four samples
# -- then the megakernels: the naive shape, the flat launch, the flat
# launch with reordering (the user, 10-30 evening: five wavefronts and
# three megakernels; mk-pass, mk-blk and mk-ser dropped as never best and
# each below a sibling at every cell -- their rows stay in the tables and
# the .txt of 10-30). A side with no rows yet is left out (the fused and
# pass wavefronts until their run of the night of 10-30 lands).
GPU_SIDES="${GPU_SIDES:-gpu-optix,gpu-optix-sort,gpu-optix-fused,gpu-optix-fused-ser,gpu-optix-pass,gpu-optix-mega,gpu-optix-mega-flat,gpu-optix-mega-ser-flat}"
# The profile figure's launches (the user, 10-28): profile_kernels.sh's
# reports under profile-out-e (02:22-02:28 the same night, on the sweep's
# renderers) -- the wavefront's trace kernel and the two flat megakernels
# on ganesha, where the wavefront wins, and on killeroo-gold, where the SER
# megakernel does, at 16 spp. profile-out/ holds 10-28 daytime's under
# 39c67ab9.
PROFILES="${PROFILES:-$PREFIX/profile-out-e}"
PROFILE_CELLS="${PROFILE_CELLS:-ganesha:16 killeroo-gold:16}"
PROFILE_SIDES="${PROFILE_SIDES:-gpu-optix,gpu-optix-mega-flat,gpu-optix-mega-ser-flat}"
# The depth figure's runs (10-30, 22:17-22:26, scratch/post_confirm_gpu.sh;
# the user: why frame542's megakernels are 18-20x): the depth sweep's
# table (16 spp, the scene's maxdepth rewritten to 4, 16, 64 and 256, the
# least of three on each side, every image checked against pbrt's CPU
# integrator at that depth), pbrt --gpu's statistics of the full-depth
# frame at 128 spp, and the sweep's log for each bar's verdict.
DEPTH_TABLE="${DEPTH_TABLE:-$PREFIX/depth-out/frame542-depth.tsv}"
DEPTH_STATS="${DEPTH_STATS:-$PREFIX/depth-out/frame542-s128.pbrt.stats}"
DEPTH_LOG="${DEPTH_LOG:-$PREFIX/depth-out/post_confirm_gpu.log}"
png=0
[[ "${1:-}" == "--png" ]] && png=1

for f in $CPU_CSVS $GPU_TABLES $HEATMAP_TABLE $DEPTH_TABLE $DEPTH_STATS $DEPTH_LOG; do
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
  python3 "$PREFIX/plot_cpu_wavefront.py" $CPU_CSVS --scenes "$SWEEP_SCENES" --ymax "$CPU_YMAX" $CUT_FLAGS -o "$OUT/pbrt-cpu-sweep.$ext"
  # The paper's four GPU sides (10-24) and the three megakernel shapes of
  # 10-27 (the user: the GPU figures other than the heatmap gain them):
  # seven, within the ramp's nine; the exploration sides the tables also
  # hold (wf-b128, wf-b512, wf-fused, wf-sort36, mk-ser-ray) are drawn by
  # naming them in GPU_SIDES.
  python3 "$PREFIX/plot_gpu_wavefront.py" $GPU_TABLES --sides "$GPU_SIDES" --scenes "$SWEEP_SCENES" --ymax "$GPU_YMAX" $CUT_FLAGS -o "$OUT/pbrt-gpu-sweep.$ext"
  python3 "$PREFIX/plot_gpu_heatmap.py" "$OUT/pbrt-gpu-heatmap.$ext" "$HEATMAP_TABLE" --sort geometry
  python3 "$PREFIX/plot_gpu_profile.py" "${cells[@]}" --sides "$PROFILE_SIDES" --profiles "$PROFILES" -o "$OUT/pbrt-gpu-profile.$ext"
  python3 "$PREFIX/plot_gpu_depth.py" "$DEPTH_TABLE" --stats "$DEPTH_STATS" --log "$DEPTH_LOG" -o "$OUT/pbrt-gpu-depth.$ext"
done
ls -la "$OUT"
