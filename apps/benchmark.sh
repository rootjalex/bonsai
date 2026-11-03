#!/usr/bin/env bash
# apps/benchmark.sh: the paper's benchmarks, end to end, from the compiler
# built out of this checkout -- the pbrt GPU and CPU sweeps and the GPU
# heatmap (apps/pbrt), the rtq query figures (apps/rtq) and the wosx GPU
# figures (apps/wosx) -- so that every figure can be remade in one go under
# one compiler, for reproducibility. Run from the repository root, inside
# the `bonsai` conda environment:
#
#     apps/benchmark.sh                      # everything, in the order below
#     apps/benchmark.sh --stages "pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-plots"
#     apps/benchmark.sh --stages pbrt        # the four pbrt stages
#     apps/benchmark.sh --stages "rtq wosx"  # the other two apps' runners
#     apps/benchmark.sh --dry-run            # print the commands, run nothing
#
# Stages, in order, and what each makes:
#   pbrt-gpu      apps/pbrt/gpu_compare.sh: pbrt --gpu against the GPU sweep's
#                 schedules on the sweep scenes at the sweep's sample counts
#                 -> apps/pbrt/results/gpu-sweep/table.tsv (~2 h)
#   pbrt-cpu      apps/pbrt/cpu_compare.sh: pbrt's CPU wavefront and CPU
#                 integrator against the CPU sweep's schedules, same scenes
#                 and counts -> apps/pbrt/results/cpu-sweep/cpu-results.csv
#                 (~8 h on every core)
#   pbrt-heatmap  apps/pbrt/heatmap.sh: the matching GPU wavefront (gpu-optix)
#                 against pbrt --gpu on every scene the converter takes, at
#                 the heatmap's counts, the renderer reused from the GPU sweep
#                 -> apps/pbrt/results/heatmap/table.tsv (up to its 5 h cutoff)
#   pbrt-plots    apps/pbrt/plots.sh --png: pbrt-gpu-sweep, pbrt-cpu-sweep,
#                 pbrt-gpu-heatmap (and the profile and depth figures from
#                 their own tools' last runs) into apps/pbrt/plots/
#   rtq           apps/rtq/benchmark.sh: the matching CPU schedules against
#                 Embree and FCPW (rtq-queries-cpu), then the GPU schedules
#                 against FCPW's GPUScene (rtq-fcpwb-gpu) -> apps/rtq/plots/
#   wosx          apps/wosx/benchmark.sh --gpu and --stars --gpu: WoSX's GPU
#                 solvers against the GPU schedules -> apps/wosx/plots/
#                 wosx-gpu-speedup, wosx-stars-gpu-speedup
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
# built from, and `results/provenance.txt` records it. It holds the lock
# through the pbrt stages, removed on exit however the script ends, and
# hands the machine to the rtq and wosx runners for theirs. Every pbrt
# stage resumes: a table's finished cells are kept and the missing ones
# measured, so an interrupted run is continued by the same command.
# BENCHMARK_HOLDER names the holder in the lock (default: the user).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate bonsai
fi
source apps/pbrt/figures.sh

STAGES="pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-plots rtq wosx"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "benchmark.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
# `pbrt` names its four stages.
STAGES=$(echo "$STAGES" | sed 's/\bpbrt\b/pbrt-gpu pbrt-cpu pbrt-heatmap pbrt-plots/')
HOLDER="${BENCHMARK_HOLDER:-$USER}"
LOCK=/tmp/benchmark.lock
HARNESS='^(/bin/)?bash apps/(pbrt|wosx|rtq)/(gpu_compare|heatmap|cpu_compare|compare|benchmark)\.sh'
mkdir -p "$RESULTS"
LOG="$RESULTS/benchmark-$(date +%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== apps/benchmark.sh starts $(date '+%F %T'); stages: $STAGES"

run() {
  echo "+ $*"
  (( DRY )) || "$@"
}

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
  } > "$RESULTS/provenance.txt"
}

#--- the pbrt stages, under our lock ---------------------------------------
take_lock() {
  echo "$HOLDER timing: apps/benchmark.sh -- $1 since $(date '+%F %T'); 'done' follows" > "$LOCK"
  trap 'rm -f "$LOCK"; echo "=== lock removed $(date +%T)"' EXIT
}
drop_lock() { rm -f "$LOCK"; trap - EXIT; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has pbrt-gpu || has pbrt-cpu || has pbrt-heatmap; then
  (( DRY )) || take_lock "the pbrt stages ($(echo "$STAGES" | tr ' ' '\n' | grep pbrt- | grep -v plots | tr '\n' ' '))"
fi
if has pbrt-gpu; then
  echo "=== pbrt-gpu $(date +%T): the GPU sweep into $GPU_SWEEP_OUT"
  run bash apps/pbrt/gpu_compare.sh --out "$GPU_SWEEP_OUT" --resume \
      --spp "$SWEEP_SPP" --repeats "$REPEATS" \
      --schedules "$(spaced "$GPU_SIDES")" --scenes "$(spaced "$SWEEP_SCENES")"
  echo "=== pbrt-gpu ends $(date +%T) rc=$?"
fi
if has pbrt-cpu; then
  echo "=== pbrt-cpu $(date +%T): the CPU sweep into $CPU_SWEEP_OUT"
  run bash apps/pbrt/cpu_compare.sh --out "$CPU_SWEEP_OUT" --resume \
      --spp "$SWEEP_SPP" --repeats "$REPEATS" \
      --schedules "$(cpu_schedules)" --scenes "$(spaced "$SWEEP_SCENES")"
  echo "=== pbrt-cpu ends $(date +%T) rc=$?"
fi
if has pbrt-heatmap; then
  echo "=== pbrt-heatmap $(date +%T): every scene, gpu-optix against pbrt --gpu, into $HEATMAP_OUT"
  REUSE=()
  [[ -d "$GPU_SWEEP_OUT" ]] && REUSE=(--reuse "$GPU_SWEEP_OUT")
  run bash apps/pbrt/heatmap.sh --out "$HEATMAP_OUT" --spp "$HEATMAP_SPP" --repeats "$REPEATS" "${REUSE[@]}"
  echo "=== pbrt-heatmap ends $(date +%T) rc=$?"
fi
if has pbrt-plots; then
  echo "=== pbrt-plots $(date +%T)"
  run env GPU_TABLES="$GPU_SWEEP_OUT/table.tsv" CPU_CSVS="$CPU_SWEEP_OUT/cpu-results.csv" \
      HEATMAP_TABLE="$HEATMAP_OUT/table.tsv" bash apps/pbrt/plots.sh --png
  echo "=== pbrt-plots ends $(date +%T) rc=$?"
fi
(( DRY )) || drop_lock

#--- the other apps' runners, each taking the lock for itself -------------
if has rtq; then
  echo "=== rtq $(date +%T): the matching CPU schedules against Embree and FCPW, then the GPU schedules against FCPW's GPUScene"
  run env RTQ_HOLDER="$HOLDER" bash apps/rtq/benchmark.sh
  run env RTQ_HOLDER="$HOLDER" bash apps/rtq/benchmark.sh --schedules fcpwb-gpu,fcpws-gpu
  echo "=== rtq ends $(date +%T)"
fi
if has wosx; then
  echo "=== wosx $(date +%T): WoSX's GPU solvers against the GPU schedules, the walk on spheres and the walk on stars"
  run bash apps/wosx/benchmark.sh --gpu
  run bash apps/wosx/benchmark.sh --stars --gpu
  echo "=== wosx ends $(date +%T)"
fi
echo "=== apps/benchmark.sh ends $(date '+%F %T'); log $LOG"
