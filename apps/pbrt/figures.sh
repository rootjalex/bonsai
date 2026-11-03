# apps/pbrt/figures.sh: what the paper's pbrt figures are made of, in one
# place, sourced by the script that measures them (apps/benchmark.sh) and
# the script that draws them (apps/pbrt/plots.sh), so that the two cannot
# drift apart. Every value is a default an environment variable overrides.
#
# The sweeps' scenes, in the figures' order (the user, 10-29: pbrt --gpu's
# time at 128 spp, ascending), keyed by path under the scene set
# (~/projects/pbrt-v4-scenes). Comma-separated, as the plotters take them;
# the harnesses take them space-separated (see `spaced` below).
: "${SWEEP_SCENES:=killeroos/killeroo-gold,pbrt-book/book,sanmiguel/sanmiguel-courtyard,ganesha/ganesha,sssdragon/dragon_10,smoke-plume/plume,bistro/bistro_vespa,transparent-machines/frame542}"
# The sample counts of both sweeps, and the heatmap's (the user, 10-29 and
# 10-30); the repeats every side is the least of, after a warm-up
# (one protocol across every system of a figure, the user, 10-30).
: "${SWEEP_SPP:=16 128}"
: "${HEATMAP_SPP:=16 64 128}"
: "${REPEATS:=3}"
# The GPU sweep's sides, in drawing order (the user's rule of 11-01: every
# schedule that is the best on at least one cell, the match of pbrt's
# wavefront first, at most seven; plot_gpu_wavefront.py names them).
: "${GPU_SIDES:=gpu-optix,gpu-optix-fused-route,gpu-optix-bounce,gpu-optix-mega-flat-samples,gpu-optix-mega-ser-flat-samples,gpu-optix-mega-ser-ray-flat-samples,gpu-optix-mega-ser-ray-flat}"
# The CPU sweep's sides, in drawing order (the user's rules of 11-01: every
# schedule best on at least one cell, both of pbrt's integrators matched,
# the megakernel match the last Bonsai bar with PBRT mk right after it, at
# most eight). `pbrt-volpath` is pbrt's own CPU integrator, which the
# harness measures beside `scalar`; it is a side of the figure and not a
# schedule to compile.
: "${CPU_SIDES:=wavefront-frame,wavefront-thread-packet,wavefront-thread-packet-chunk256,wavefront-thread-packet-fused,wavefront-thread-packet-sort,scalar,scalar-tiled,pbrt-volpath}"
# Where apps/benchmark.sh writes and plots.sh reads: one directory per
# measurement, each a harness's own --out (its table or CSV, renderers,
# logs, images), under results/. Not committed (.gitignore: compare-out*,
# *.csv, *.tsv); plots.sh draws the figures from them.
: "${RESULTS:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"
: "${GPU_SWEEP_OUT:=$RESULTS/gpu-sweep}"
: "${CPU_SWEEP_OUT:=$RESULTS/cpu-sweep}"
: "${HEATMAP_OUT:=$RESULTS/heatmap}"

# A comma-separated list as the harnesses take it, space-separated.
spaced() { echo "$1" | tr ',' ' '; }
# The CPU sides that are schedules of ours: pbrt's own integrator left out.
cpu_schedules() { spaced "$CPU_SIDES" | tr ' ' '\n' | grep -v '^pbrt-' | tr '\n' ' ' | sed 's/ $//'; }
