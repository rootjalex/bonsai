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
#   plots/pbrt-compile-time.pdf how long the compiler takes on each schedule
#                              of the two sweeps, a bar per schedule in the
#                              sweeps' colours, the least of the benchmark's
#                              repeats, from compile_time.sh's CSV
#                              (plot_compile_time.py; the user, 11-03)
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
# What the figures are made of -- the sweep scenes, the sides, the sample
# counts, the repeats, and where apps/benchmark.sh writes each measurement
# -- is in figures.sh, shared with the script that measures them.
source "$PREFIX/figures.sh"
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
# cpu-compare-out-six-v2/ and -two-v2/ (11-01 evening, 18:52-) hold the
# three drawn vectorized wavefronts (wavefront-thread-packet, -chunk256,
# -sort) measured again under the compiler at 6488f426, whose vectorizer
# clones a callee per constant argument and so changes their code (the
# afternoon's sweep had built two of its eight renderers under that
# compiler and six under the one before, PLAN "11-01 evening"); their pbrt
# caches are the six's and the two's copied, so every cell is timed
# against the figure's own reference. A later file's row replaces an
# earlier one's for the same (scene, spp, side): these three sides draw
# from the -v2 tables, everything else from the first two.
# Since 11-02 the default is apps/benchmark.sh's CPU sweep (figures.sh);
# the tables of 10-30 and 11-01 named above are those of the figure before
# the compiler fixes of 11-02, kept on disk and in this file's history.
CPU_CSVS="${CPU_CSVS:-$CPU_SWEEP_OUT/cpu-results.csv}"
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
# (gpu-optix-pass) of that run, which hold no time: it died at 16 spp on
# every scene with an illegal address, the driver having sized its queues
# for one sample a pixel (PLAN, 10-31); gpu-compare-out-wf3c/ (11-01
# 00:07-00:22) the pass-of-four wavefront again under e08e0195 -- the
# header stating each queue array's capacity and the driver allocating by
# it -- on the eight, every cell matching, which is the side the figure
# draws as wf-pass.
# gpu-compare-out-explore3/ (11-01, 10:54-12:07) holds the schedule
# exploration's first five (the user: "what sometimes-optimal schedules can
# you find", from the directives the language has, no layout change): the
# wavefront with the escaped and emissive stages in the trace program too
# (gpu-optix-fused-route), with the shadow ray traced there too
# (gpu-optix-fused-shadow: a loss everywhere and a launch failure on plume,
# its two cells without a time), reduced to the ray queue alone
# (gpu-optix-bounce), the flat megakernel with a pixel's samples adjacent in
# the launch (gpu-optix-mega-flat-samples) and the SER flat megakernel with a
# reordering by ray before every trace (gpu-optix-mega-ser-ray-flat);
# gpu-compare-out-explore4/ (12:13-13:04) the two SER shapes in the samples
# order (gpu-optix-mega-ser-flat-samples, gpu-optix-mega-ser-ray-flat-
# samples) beside gpu-optix, gpu-optix-mega-flat and gpu-optix-mega-ser-flat
# measured again in the same run, which reproduced their 10-30 cells within
# 2% (every kernel time within 1%); the later table's rows are the ones
# drawn for those three, as the plotter takes them. All 160 cells of the two
# runs match pbrt's image but fused-shadow's two on plume.
# Since 11-02 the default is apps/benchmark.sh's GPU sweep (figures.sh);
# the tables named above are the figure's before the compiler fixes of
# 11-02, kept on disk and in this file's history.
GPU_TABLES="${GPU_TABLES:-$GPU_SWEEP_OUT/table.tsv}"
# The sweeps' scenes, in the figures' order (the user, 10-29: pbrt --gpu's
# time at 128 spp, ascending), keyed by path under the scene set; both
# sweep figures draw these and only these from their tables (the
# plotters' --scenes). A scene whose CPU sweep has not run yet is left out
# of the CPU figure with a note. Set in figures.sh (SWEEP_SCENES).
# Where each sweep's speedup axis ends (the user, 10-30: a linear axis, cut
# on the GPU sweep, never a log axis for speedups; 11-01 morning: cut at
# 4x rather than 6x, "it looks better"; 11-01 afternoon: 5x, once the SER
# megakernel in the samples order put killeroo-gold at 4.06x with a label
# over it -- "remove that pesky 4.1x speedup label" -- so only frame542's
# megakernel bars, 7.5x to 21.2x, are cut); a bar past the cutoff is
# drawn to it with a break mark, and its value stands upright over the
# bars with a thin leader to its bar (the user's choice among six trials,
# 10-30 night: the values on one line, never stacked; frame542's
# megakernels, 4.6x to 20.4x over pbrt --gpu). The CPU sweep's bars run to
# 9.2x (frame542's megakernels and per-thread wavefronts, 8.0x to 9.2x,
# since its CPU sweep of 10-31; every other scene's under 7.9x), so its
# axis ends at 10x and cuts nothing (the user, 11-01: "Definitely use 10x
# for the CPU cut off"; at 8x seven of frame542's bars were cut); the same
# flags for the day a bar passes it.
GPU_YMAX="${GPU_YMAX:-5}"
CPU_YMAX="${CPU_YMAX:-10}"
CUT_FLAGS="--leaders --rotate-cut-labels"
# The heatmap of the same night (00:45-02:22), the gpu-optix renderer
# reused from the sweep; compare-out-heatmap11/ is 10-27's under 39c67ab9.
# Since 11-02 the default is apps/benchmark.sh's heatmap (figures.sh).
HEATMAP_TABLE="${HEATMAP_TABLE:-$HEATMAP_OUT/table.tsv}"
# The GPU sweep's sides, in drawing order (plot_gpu_wavefront.py's SIDES
# names them). The user's rule of 11-01: every schedule that is the best on
# at least one cell, the matching one among them, the rest filled to
# balance the wavefronts and the megakernels, and seven in all (wf-sort,
# never best, out: "it makes the legend look weird"). The winners of the
# two exploration runs: the matched wavefront (ganesha, dragon_10, plume),
# the wavefront with the material, escaped and emissive stages in the trace
# program (bistro), the SER flat megakernel in the samples order (killeroo-
# gold), that with the reordering by ray (book, sanmiguel, frame542 at 16
# spp) and the same in the pixel order (frame542 at 128 spp); filled with
# the one-launch-per-bounce wavefront (the best wavefront on frame542) and
# the flat megakernel in the samples order without reordering (the step
# the SER ones are measured from). Out since 11-01: wf-sort, wf-fused (the
# fused-route wavefront is the same decision and ahead on every scene),
# wf-fused-ser, wf-pass (the bounce wavefront matches it on frame542 at 128
# spp and beats it everywhere else), mk, mk-flat and mk-ser-flat (each
# behind its samples-order sibling at every cell but frame542 at 128 spp
# for mk-ser-flat); their rows stay in the tables and the .txt's
# comparisons. The 10-30 choice (five wavefronts, three megakernels) is in
# plots.sh's history. A side with no timed rows is left out.
# Set in figures.sh (GPU_SIDES).
# The CPU sweep's sides, in drawing order (plot_cpu_wavefront.py's SIDES
# names them; the user's rules of 11-01: every schedule best on at least
# one cell, both of pbrt's integrators matched, the megakernel match the
# LAST Bonsai bar with PBRT mk right after it "so these two are easily
# visually comparable", at most eight). The wavefront match first, the
# per-thread vectorized wavefront and its two winning variants of 11-01
# (256 pixels a thread; the material stage in the trace drain's gang),
# the sorted one, then the untiled megakernel and the tiled match, then
# pbrt's megakernel. Set when the CPU sweep of 11-01 finished; the
# never-winning sides (the band wavefronts' vectorized and sorted forms,
# the per-thread scalar one, the samples-outer order, the 4,096-pixel
# chunk, the megakernel's sample-outer, waves, chunk and collapsed
# variants) stay in the CSVs and the .txt's comparisons.
# Set in figures.sh (CPU_SIDES).
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
# The compile times of the sweeps' schedules (compile_time.sh, the
# pbrt-compile stage of apps/benchmark.sh).
COMPILE_TABLE="${COMPILE_TABLE:-$COMPILE_OUT/compile-times.csv}"

png=0
[[ "${1:-}" == "--png" ]] && png=1

for f in $CPU_CSVS $GPU_TABLES $HEATMAP_TABLE $DEPTH_TABLE $DEPTH_STATS $DEPTH_LOG $COMPILE_TABLE; do
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
  python3 "$PREFIX/plot_cpu_wavefront.py" $CPU_CSVS --sides "$CPU_SIDES" --scenes "$SWEEP_SCENES" --ymax "$CPU_YMAX" $CUT_FLAGS -o "$OUT/pbrt-cpu-sweep.$ext"
  # The paper's four GPU sides (10-24) and the three megakernel shapes of
  # 10-27 (the user: the GPU figures other than the heatmap gain them):
  # seven, within the ramp's nine; the exploration sides the tables also
  # hold (wf-b128, wf-b512, wf-fused, wf-sort36, mk-ser-ray) are drawn by
  # naming them in GPU_SIDES.
  python3 "$PREFIX/plot_gpu_wavefront.py" $GPU_TABLES --sides "$GPU_SIDES" --scenes "$SWEEP_SCENES" --ymax "$GPU_YMAX" $CUT_FLAGS -o "$OUT/pbrt-gpu-sweep.$ext"
  python3 "$PREFIX/plot_gpu_heatmap.py" "$OUT/pbrt-gpu-heatmap.$ext" "$HEATMAP_TABLE" --sort geometry
  python3 "$PREFIX/plot_gpu_profile.py" "${cells[@]}" --sides "$PROFILE_SIDES" --profiles "$PROFILES" -o "$OUT/pbrt-gpu-profile.$ext"
  python3 "$PREFIX/plot_gpu_depth.py" "$DEPTH_TABLE" --stats "$DEPTH_STATS" --log "$DEPTH_LOG" -o "$OUT/pbrt-gpu-depth.$ext"
  python3 "$PREFIX/plot_compile_time.py" $COMPILE_TABLE --cpu-order "$CPU_SIDES" --gpu-order "$GPU_SIDES" -o "$OUT/pbrt-compile-time.$ext"
done
ls -la "$OUT"
