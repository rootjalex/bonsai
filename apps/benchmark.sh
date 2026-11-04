#!/usr/bin/env bash
# apps/benchmark.sh: the paper's benchmarks, end to end, from the compiler
# built out of this checkout -- eight figures, each as its .pdf, its .png
# and the .txt that reads it, and nothing else measured or drawn on the way
# (the user, 11-03: these and no other figure):
#
#     pbrt-gpu-sweep  pbrt-cpu-sweep  pbrt-gpu-heatmap  pbrt-compile-time   (apps/pbrt)
#     rtq-queries-cpu  rtq-fcpwb-gpu                                         (apps/rtq)
#     wosx-gpu-speedup  wosx-stars-gpu-speedup                               (apps/wosx)
#
# so that every figure can be remade in one go under one compiler, for
# reproducibility. Run from the repository root, inside the `bonsai` conda
# environment:
#
#     apps/benchmark.sh                      # everything, in the order below
#     apps/benchmark.sh --stages "pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-plots"
#     apps/benchmark.sh --stages pbrt        # the five pbrt stages
#     apps/benchmark.sh --stages "rtq wosx"  # the other two apps' runners
#     apps/benchmark.sh --fresh              # measure every cell again: each pbrt stage's
#                                            # results set aside first under a dated name
#     apps/benchmark.sh --dry-run            # print the commands, run nothing
#     apps/benchmark.sh --stages gather      # only copy the figures into apps/plots
#
# A full run takes about 13 h on the 32-thread host with one RTX 5090: the
# pbrt sweeps and heatmap 9 h 31 min and the compile times 1 h 25 min as
# measured 11-02/03; rtq about 12 min and wosx about 1 h 55 min, the same
# day's runs (27 min, 2 h 20 min) less the sections and schedules the
# figures do not draw, left out since (the user, 11-03). The durations
# below are from those runs.
#
# Stages, in order, and what each makes:
#   pbrt-gpu      apps/pbrt/gpu_compare.sh: pbrt --gpu against the GPU sweep's
#                 schedules on the sweep scenes at the sweep's sample counts
#                 -> apps/pbrt/results/gpu-sweep/table.tsv (1 h 29 min)
#   pbrt-cpu      apps/pbrt/cpu_compare.sh: pbrt's CPU wavefront and CPU
#                 integrator against the CPU sweep's schedules, same scenes
#                 and counts -> apps/pbrt/results/cpu-sweep/cpu-results.csv
#                 (6 h 23 min on every core)
#   pbrt-heatmap  apps/pbrt/heatmap.sh: the matching GPU wavefront (gpu-optix)
#                 against pbrt --gpu on every scene the converter takes, at
#                 the heatmap's counts, the renderer reused from the GPU sweep
#                 -> apps/pbrt/results/heatmap/table.tsv (1 h 39 min, under
#                 its 5 h cutoff)
#   pbrt-compile  apps/pbrt/compile_time.sh: the compiler's time on every
#                 schedule of the two sweeps, each built once as a
#                 comparison builds it, and pbrt-v4's own build from a
#                 clean build directory -> apps/pbrt/results/compile-time/
#                 compile-times.csv (the user, 11-03; 1 h 25 min: fourteen
#                 builds of 3 to 10 min on one core, pbrt's a minute)
#   pbrt-plots    apps/pbrt/plots.sh --png, FIGURES naming the four pbrt
#                 figures above (the compile-time figure once its stage has
#                 run) into apps/pbrt/plots/; the profile, depth and
#                 scene-statistics figures that tool also draws are not the
#                 reproduction's
#   rtq           apps/rtq/benchmark.sh --fresh --schedules embree,fcpw4w16,
#                 fcpwb-gpu,fcpws-gpu: the matching CPU schedules against
#                 Embree and FCPW (rtq-queries-cpu) and the GPU schedules
#                 against FCPW's GPUScene (rtq-fcpwb-gpu) on the thirteen
#                 meshes -> apps/rtq/plots/ (about 12 min: 11-03's 27 min
#                 less the 14 min of the scalar-tree section, which fed
#                 rtq-fcpwb-cpu alone)
#   wosx          apps/wosx/benchmark.sh --fresh --gpu and --fresh --stars
#                 --gpu, each with --schedules the eight its figure draws
#                 (plot.py's GPU_DEFAULT_SCHEDULES, STARS_GPU_DEFAULT_SCHEDULES):
#                 WoSX's GPU solvers against those schedules, every cell
#                 measured again -> apps/wosx/plots/wosx-gpu-speedup,
#                 wosx-stars-gpu-speedup (about 38 min and 1 h 15 min: 11-03's
#                 47 min and 1 h 34 min were over ten schedules each)
#   the end       the eight figures -- each one's .pdf, .png and the .txt
#                 that reads it -- and the run's provenance copied from the
#                 apps' plots directories into apps/plots: one place, the
#                 last run's (a full run empties it first; `--stages gather`
#                 copies alone, measuring nothing)
#
# What the figures are made of -- scenes, sides, sample counts, repeats --
# lives in apps/pbrt/figures.sh, which plots.sh reads too. The rtq and wosx
# runners carry their own definitions (their --schedules defaults) and
# their own lock handling, so they are called and not wrapped.
#
# The machine is one shared by the sessions that measure on it (the lock
# protocol of /tmp/benchmark.lock): the script refuses to start on another's
# lock, on a running harness, on a process holding the GPU, or on
# uncommitted compiler sources -- a run is attributed to the commit it was
# built from, and `results/provenance.txt` records it, as does the head of
# the run's log, since the next run writes provenance.txt afresh. It holds the lock
# through the pbrt stages, removed on exit however the script ends, and
# hands the machine to the rtq and wosx runners for theirs. Every pbrt
# stage resumes: a table's finished cells are kept and the missing ones
# measured, so an interrupted run is continued by the same command.
# --fresh measures every cell again: before the pbrt stages each one's
# results directory is set aside under a dated name
# (results/<stage>.before-<date>, the convention of the rtq and wosx
# tables), the old tables kept for comparison; the rtq and wosx runners
# are called with their own --fresh in every run, since their logs do not
# resume cleanly across an older run's (a log that exists passes for a
# finished cell). To continue an interrupted fresh run, repeat the command
# without --fresh. BENCHMARK_HOLDER names the holder in the lock (default:
# the user).
#
# While it runs, the compiler sources of this checkout are frozen: the
# pbrt harnesses rebuild the compiler before each renderer they compile
# (compare.sh, gpu_compare.sh), so an edit made in this checkout during the
# run would be built into the middle of a sweep. Compiler work during a
# run goes in a worktree of its own with its own build directory.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate bonsai
fi
source apps/pbrt/figures.sh

DEFAULT_STAGES="pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-compile pbrt-plots rtq wosx"
STAGES="$DEFAULT_STAGES"
DRY=0
FRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "benchmark.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
# `pbrt` names its four stages.
# `pbrt` alone names its five stages; a stage token is matched whole (a
# word-boundary match took the `pbrt` of `pbrt-compile` too, 11-03).
expanded=""
for stage in $STAGES; do
  if [[ "$stage" == pbrt ]]; then
    expanded+=" pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-compile pbrt-plots"
  else
    expanded+=" $stage"
  fi
done
STAGES="${expanded# }"
HOLDER="${BENCHMARK_HOLDER:-$USER}"
LOCK=/tmp/benchmark.lock
HARNESS='^(/bin/)?bash apps/(pbrt|wosx|rtq)/(gpu_compare|heatmap|cpu_compare|compare|benchmark)\.sh'
mkdir -p "$RESULTS"
LOG="$RESULTS/benchmark-$(date +%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== apps/benchmark.sh starts $(date '+%F %T'); stages: $STAGES"

# A stage's ending line reads the status into `rc` first: `$?` inside the
# line would be the status of its `$(date)` (the stage lines of 11-02/03
# said rc=0 of a stage whose runner had failed).
run() {
  echo "+ $*"
  (( DRY )) || "$@"
}

#--- the paper's figures in one place --------------------------------------
# Each app's plotting step writes its figures beside its results
# (apps/pbrt/plots, apps/rtq/plots, apps/wosx/plots). The run ends by
# copying the eight figures of the header -- each one's .pdf, .png and the
# .txt that reads it -- and the run's provenance into apps/plots, so that
# the last run's figures are in one place (the user, 11-03). No versions: a
# full run empties the directory first, a run of some stages replaces what
# it remade and leaves the rest. `--stages gather` does this copy and
# nothing else.
PLOTS=apps/plots
gather_plots() {  # full | part
  mkdir -p "$PLOTS"
  [[ "$1" == full ]] && rm -f "$PLOTS"/*
  local f
  for f in apps/pbrt/plots/{pbrt-gpu-sweep,pbrt-cpu-sweep,pbrt-gpu-heatmap,pbrt-compile-time}.{pdf,png,txt} \
           apps/rtq/plots/{rtq-queries-cpu,rtq-fcpwb-gpu}.{pdf,png,txt} \
           apps/wosx/plots/{wosx-gpu-speedup,wosx-stars-gpu-speedup}.{pdf,png,txt} \
           "$RESULTS/provenance.txt"; do
    [[ -f "$f" ]] && cp -f "$f" "$PLOTS/"
  done
  echo "=== $PLOTS: $(ls "$PLOTS" | wc -l) files, the last run's figures and its provenance"
}
if [[ "$STAGES" == "gather" ]]; then
  gather_plots part
  echo "=== apps/benchmark.sh ends $(date '+%F %T'); log $LOG"
  exit 0
fi

#--- the machine, the sources, the compiler --------------------------------
if [[ -f $LOCK ]] && ! grep -q "^$HOLDER " "$LOCK"; then
  echo "refused: a lock not ours: $(cat "$LOCK")"; exit 1
fi
if pgrep -f "$HARNESS" | grep -qv "^$$\$"; then
  echo "refused: a harness is running:"; pgrep -af "$HARNESS"; exit 1
fi
if command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "refused: something is on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv; exit 1
fi
if [[ -n "$(git status --short -- src include runtime stdlib CMakeLists.txt)" ]]; then
  echo "refused: uncommitted compiler sources (a run is attributed to a commit):"
  git status --short -- src include runtime stdlib CMakeLists.txt; exit 1
fi
HEAD=$(git rev-parse --short HEAD)
COMPILER_COMMIT=$(git log -1 --format=%h -- src include runtime stdlib CMakeLists.txt)
echo "HEAD $HEAD; last compiler commit $COMPILER_COMMIT"
# Every target, no --target: the compiler and the suite runner out of one
# build, so what measures is what the tests passed on.
run cmake --build "${BONSAI_BUILD_DIR:-build}" -j"$(nproc)"
(( DRY )) || {
  {
    echo "date        $(date '+%F %T')"
    echo "HEAD        $HEAD"
    echo "compiler    $COMPILER_COMMIT (last commit under src include runtime stdlib CMakeLists.txt)"
    echo "binary      $(ls -la --time-style=+%F\ %T "${BONSAI_BUILD_DIR:-build}/compiler" | awk '{print $6, $7}')"
    echo "stages      $STAGES"
    echo "scenes      $SWEEP_SCENES"
    echo "spp         sweeps $SWEEP_SPP; heatmap $HEATMAP_SPP; repeats $REPEATS"
    echo "gpu sides   $GPU_SIDES"
    echo "cpu sides   $CPU_SIDES"
    echo "host        $(hostname); $(nproc) threads; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
  } | tee "$RESULTS/provenance.txt"
}

#--- the pbrt stages, under our lock ---------------------------------------
take_lock() {
  echo "$HOLDER timing: apps/benchmark.sh -- $1 since $(date '+%F %T'); 'done' follows" > "$LOCK"
  trap 'rm -f "$LOCK"; echo "=== lock removed $(date +%T)"' EXIT
}
drop_lock() { rm -f "$LOCK"; trap - EXIT; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has pbrt-gpu || has pbrt-cpu || has pbrt-heatmap || has pbrt-compile; then
  (( DRY )) || take_lock "the pbrt stages ($(echo "$STAGES" | tr ' ' '\n' | grep pbrt- | grep -v plots | tr '\n' ' '))"
fi
# --fresh: a stage's results of an earlier run set aside under a dated
# name, so that the stage measures every cell again (its --resume finds
# nothing) and the old table stays for comparison.
set_aside() {  # set_aside DIR
  [[ -d "$1" ]] || return 0
  local aside="$1.before-$(date +%F)" n=1
  while [[ -e "$aside" ]]; do aside="$1.before-$(date +%F)-$n"; n=$((n + 1)); done
  echo "+ mv $1 $aside"
  (( DRY )) || mv "$1" "$aside"
}
if (( FRESH )); then
  has pbrt-gpu && set_aside "$GPU_SWEEP_OUT"
  has pbrt-cpu && set_aside "$CPU_SWEEP_OUT"
  has pbrt-heatmap && set_aside "$HEATMAP_OUT"
  has pbrt-compile && set_aside "$COMPILE_OUT"
fi
if has pbrt-gpu; then
  echo "=== pbrt-gpu $(date +%T): the GPU sweep into $GPU_SWEEP_OUT"
  run bash apps/pbrt/gpu_compare.sh --out "$GPU_SWEEP_OUT" --resume \
      --spp "$SWEEP_SPP" --repeats "$REPEATS" \
      --schedules "$(spaced "$GPU_SIDES")" --scenes "$(spaced "$SWEEP_SCENES")"
  rc=$?; echo "=== pbrt-gpu ends $(date +%T) rc=$rc"
fi
if has pbrt-cpu; then
  echo "=== pbrt-cpu $(date +%T): the CPU sweep into $CPU_SWEEP_OUT"
  run bash apps/pbrt/cpu_compare.sh --out "$CPU_SWEEP_OUT" --resume \
      --spp "$SWEEP_SPP" --repeats "$REPEATS" \
      --schedules "$(cpu_schedules)" --scenes "$(spaced "$SWEEP_SCENES")"
  rc=$?; echo "=== pbrt-cpu ends $(date +%T) rc=$rc"
fi
if has pbrt-heatmap; then
  echo "=== pbrt-heatmap $(date +%T): every scene, gpu-optix against pbrt --gpu, into $HEATMAP_OUT"
  REUSE=()
  [[ -d "$GPU_SWEEP_OUT" ]] && REUSE=(--reuse "$GPU_SWEEP_OUT")
  run bash apps/pbrt/heatmap.sh --out "$HEATMAP_OUT" --spp "$HEATMAP_SPP" --repeats "$REPEATS" "${REUSE[@]}"
  rc=$?; echo "=== pbrt-heatmap ends $(date +%T) rc=$rc"
fi
if has pbrt-compile; then
  # Every schedule of the two sweeps built once, the compiler's seconds of
  # the build kept (compare.sh prints them), then pbrt-v4's own build; the
  # machine is quiet here, between the heatmap and the plots.
  echo "=== pbrt-compile $(date +%T): the compiler's time on the sweeps' schedules and pbrt's build, into $COMPILE_OUT"
  run bash apps/pbrt/compile_time.sh --out "$COMPILE_OUT" --resume \
      --schedules "$(spaced "$GPU_SIDES") $(cpu_schedules)"
  rc=$?; echo "=== pbrt-compile ends $(date +%T) rc=$rc"
fi
if has pbrt-plots; then
  echo "=== pbrt-plots $(date +%T)"
  # The reproduction's four pbrt figures and no other (plots.sh's FIGURES);
  # the compile-time figure once its stage has run, since a run of the
  # sweeps alone has no table for it.
  PBRT_FIGURES="pbrt-gpu-sweep pbrt-cpu-sweep pbrt-gpu-heatmap"
  if [[ -f "$COMPILE_OUT/compile-times.csv" ]]; then
    PBRT_FIGURES+=" pbrt-compile-time"
  else
    echo "  pbrt-compile-time left out: no $COMPILE_OUT/compile-times.csv (the pbrt-compile stage writes it)"
  fi
  run env FIGURES="$PBRT_FIGURES" GPU_TABLES="$GPU_SWEEP_OUT/table.tsv" CPU_CSVS="$CPU_SWEEP_OUT/cpu-results.csv" \
      HEATMAP_TABLE="$HEATMAP_OUT/table.tsv" bash apps/pbrt/plots.sh --png
  rc=$?; echo "=== pbrt-plots ends $(date +%T) rc=$rc"
fi
(( DRY )) || drop_lock

#--- the other apps' runners, each taking the lock for itself -------------
if has rtq; then
  # One invocation for the two rtq figures (rtq-c0, 11-02: one fold, one
  # plot): each mesh measured under three sections in turn -- Embree and
  # FCPW's four-wide tree (rtq-queries-cpu; plot.py's --embree and --fcpw
  # defaults) and FCPW's GPUScene (rtq-fcpwb-gpu; its SCALAR_FIGURES). The
  # section over FCPW's scalar CPU trees (fcpwb, fcpws) fed rtq-fcpwb-cpu
  # alone, half the stage's time, and is left out (the user, 11-03); its
  # rows stay in the CSV with their own dates, tocsv.sh carrying over what
  # a run did not measure. --fresh, since the runner takes a log that
  # exists for a finished run (and removes every section's logs of a mesh,
  # so it goes on the one call); RTQ_SKIP_CMAKE, since the compiler was
  # built above and compare.sh would otherwise rebuild it before every
  # cell.
  echo "=== rtq $(date +%T): every mesh against Embree, FCPW's four-wide tree and FCPW's GPUScene"
  run env RTQ_HOLDER="$HOLDER" RTQ_SKIP_CMAKE=1 bash apps/rtq/benchmark.sh --fresh \
      --schedules embree,fcpw4w16,fcpwb-gpu,fcpws-gpu
  rc=$?; echo "=== rtq ends $(date +%T) rc=$rc"
fi
if has wosx; then
  # The schedules each wosx figure draws, from the plotter's own lists (one
  # source: plot.py's GPU_DEFAULT_SCHEDULES and STARS_GPU_DEFAULT_SCHEDULES).
  # The runner's defaults hold two more each -- the launch through OptiX
  # without reordering and the persistent claim under the reordering --
  # that fed the .txt's step-by-step readings and no bar, a fifth of the
  # stage's time, left out since (the user, 11-03).
  wosx_schedules() {  # wosx_schedules --gpu | --stars
    python3 - "$1" <<'EOF'
import sys
sys.path.insert(0, "apps/wosx")
import plot
print(",".join(plot.STARS_GPU_DEFAULT_SCHEDULES if sys.argv[1] == "--stars" else plot.GPU_DEFAULT_SCHEDULES))
EOF
  }
  WOSX_SPHERES=$(wosx_schedules --gpu) && [[ -n "$WOSX_SPHERES" ]] || { echo "benchmark.sh: apps/wosx/plot.py's GPU_DEFAULT_SCHEDULES could not be read" >&2; exit 1; }
  WOSX_STARS=$(wosx_schedules --stars) && [[ -n "$WOSX_STARS" ]] || { echo "benchmark.sh: apps/wosx/plot.py's STARS_GPU_DEFAULT_SCHEDULES could not be read" >&2; exit 1; }
  # --fresh, as for rtq: the runner takes a cell's log that exists for a
  # finished run and would fold the old logs into new tables and figures
  # without measuring anything (the 11-03 03:29 run did just that).
  echo "=== wosx $(date +%T): WoSX's GPU solvers against the figures' GPU schedules, the walk on spheres and the walk on stars"
  run bash apps/wosx/benchmark.sh --fresh --gpu --schedules "$WOSX_SPHERES"
  run bash apps/wosx/benchmark.sh --fresh --stars --gpu --schedules "$WOSX_STARS"
  echo "=== wosx ends $(date +%T)"
fi
if (( DRY )); then
  echo "+ gather_plots"
elif [[ "$STAGES" == "$DEFAULT_STAGES" ]]; then
  gather_plots full
else
  gather_plots part
fi
echo "=== apps/benchmark.sh ends $(date '+%F %T'); log $LOG"
