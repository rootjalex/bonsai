#!/usr/bin/env bash
# The GPU comparison: pbrt --gpu against every GPU schedule of this renderer,
# on the scenes given, at the sample counts given -- one command, one table.
#
#     apps/pbrt/gpu_compare.sh [--spp "N ..."] [--repeats N] [--out DIR] [--resume]
#                              [--schedules "a b c"] [--scenes "d/s ..."]
#                              [--reuse DIR]
#
# `--spp` takes one count or several ("16 64 256"): every scene is run at
# each, and the table holds a row per (scene, count, side), which is what
# plot_gpu_heatmap.py draws as scenes across and counts down.
#
# A number is never one run's. For each scene, pbrt --gpu is run REPEATS
# times as separate processes (it cannot repeat a render in one) and its own
# render timer is read from the image it writes; the least of them is pbrt's
# time. Each schedule is compiled once and run as one process that renders
# REPEATS times (BONSAI_REPEATS), and the least of the driver's timer is
# ours. The first render is the warm-up, and it absorbs what is not the
# renderer's speed: module loading, OptiX finishing its pipelines, and the
# GPU's climb from the idle clocks every process starts at (180 MHz on this
# machine; the first hundred-odd milliseconds of a cold process's kernels
# run at a fraction of speed), which pbrt's timed render escapes because its
# prefetch of every allocation runs ahead of its first kernel. Beside each
# time is the kernel time: pbrt's from `--stats` (its "Wavefront Kernel
# Profile"), ours from BONSAI_KERNEL_STATS (runtime/bonsai_cuda.h) of the
# fastest of REPEATS renders in one process -- the device's busy time, apart
# from whatever either side's timer counts around it (pbrt's timer includes
# its prefetch and first launches, some sixty milliseconds on this machine;
# ours includes the host's launch gaps). Every image is checked against
# pbrt's as compare.sh checks it, converted to a PNG, and the table says
# where each is. A cell whose image fails the check is held against pbrt's
# own spread -- the scene rendered by pbrt's CPU integrator, once per cell,
# untimed -- and passes when pbrt's two renderers disagree with each other
# more than ours disagrees with the GPU one (the user's rule of 2026-10-18;
# compare_gbuffer.py --pbrt-spread): the verdict then reads `ok: within
# pbrt's own disagreement`.
#
# The GPU's clock is read before the run (gpu_clock.sh) and the table's
# last line says whether it was locked (`--clock N` asks for a lock at N
# MHz and refuses to run without it; BONSAI_GPU_CLOCK_REQUIRED=1 does the
# same for the default 2400) or unlocked, which is the default for now:
# the card is shared with another project's benchmarks that must not run
# locked (the user, 2026-10-10). Locking takes root (`nvidia-smi -lgc`).
#
# The machine is checked idle before every timed run, and watched during it:
# another process on the GPU, or someone else's process at the top of the
# CPU, and the run is redone (three tries at most); a set of our runs more
# than a quarter apart is redone as a set. The table's `note` column counts
# what a cell had redone and what it kept disturbed. `--resume` continues a
# run into the same --out directory: its binaries are reused, scenes with
# every row already in the table are skipped, a scene begun and not finished
# is redone.
#
# The schedules are compiled at once, each into its own directory
# (OUT/<schedule>/: the generated render.h, the driver linked with it as
# render.out, compile.log, and render.key), so that seven take the ten
# minutes one takes, where they took ten minutes each in turn; the
# driver's sources take the header's path from the macro BONSAI_RENDER_H
# (render_hook.cpp, nanovdb_shim.cpp), so the compiles share nothing
# under apps/pbrt. render.key says what a renderer is made of -- the
# compiler binary and its flags, every .bonsai of the renderer and of
# its schedule, the driver's sources and headers, the runtime headers,
# the GPU -- and `--reuse DIR` copies a schedule's renderer from an
# earlier run's directory when its key is this run's, compiling only
# what has no match (two runs under one compiler, the second adding a
# scene or a count, compile once); `--resume` reuses this run's own
# renderers under the same test, and recompiles one whose key moved.
# The converter (OUT/scene_dump, scene_dump.key) is kept the same way.
#
# Scenes are named as `<dir>/<name>` under $SCENES_DIR (~/projects/
# pbrt-v4-scenes): killeroos/killeroo-simple, pbrt-book/book. Only scenes
# scene_dump converts can be given; the survey of the scene set is in
# PLAN.md. Each scene is converted once, into apps/pbrt/scenes/converted/
# <dir>/<name>.txt (convert_scene.sh), and that conversion serves every
# count of this run and every later run until the converter's sources or
# the scene change; the sample count is the renderer's own `--spp`
# (pbrt's override transcribed, render_hook.cpp), as `--spp` is pbrt's on
# the other side. A run is refused with less than 50 GB free under --out:
# a cell writes a few hundred megabytes of renders, and the sweep of
# 2026-10-11 filled the disk and lost its last twelve scenes.
set -euo pipefail
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"

SPPS="16"
REPEATS=3
OUT="$PREFIX/gpu-compare-out"
# The schedules on the ray tracing hardware (the user, 10-23: the GPU
# figure compares techniques that use OptiX): the matching wavefront, the
# same with its ray queue sorted on the device, the megakernel, and the
# megakernel with shader execution reordering.
SCHEDULES="gpu-optix gpu-optix-sort gpu-optix-mega gpu-optix-mega-ser"
SCENES="killeroos/killeroo-simple killeroos/killeroo-gold pbrt-book/book"
RESUME=0
REUSE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spp) SPPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --reuse) REUSE="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --clock) export BONSAI_GPU_CLOCK="$2" BONSAI_GPU_CLOCK_REQUIRED=1; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
done
source "$PREFIX/gpu_clock.sh"

# Everything the run will act on, checked before any of it starts: a run is
# minutes long and a missing scene should not be found at the end of it.
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
SCENES_DIR="${SCENES_DIR:-$HOME/projects/pbrt-v4-scenes}"
BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
BONSAI_CXX="${BONSAI_CXX:-clang++}"
# The register cap every schedule is compiled under. 0, the default, is
# none: OptiX picks the count for each pipeline (128 for a ray-generation
# program without reordering, 255 for one with it, on this card), which is
# how pbrt builds its own OptiX modules (OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
# in src/pbrt/gpu/optix/aggregate.cpp; the `-maxrregcount 128` of its
# CMakeLists reaches only its CUDA kernels), so that the wavefront is
# compared like with like. BONSAI_GPU_MAX_REGISTERS=N compiles this run's
# schedules under a cap of N -- a megakernel's trade between spills and
# warps is not the wavefront's -- which reaches OptiX since 66b6590d (before
# it OptiX ignored the PTX's .maxnreg, and every run was at its choice
# whatever the harness asked); the compiler line of the log says which, and
# the table's last line when it is a cap.
GPU_MAX_REGISTERS="${BONSAI_GPU_MAX_REGISTERS:-0}"
REGISTER_FLAGS=()
if [[ "$GPU_MAX_REGISTERS" != 0 ]]; then
  REGISTER_FLAGS=(--gpu-max-registers "$GPU_MAX_REGISTERS")
fi
fail=0
for tool in "$PBRT" "$IMGTOOL" "$BONSAI_BUILD_DIR/compiler"; do
  if [[ ! -x "$tool" ]]; then echo "missing: $tool" >&2; fail=1; fi
done
for s in $SCHEDULES; do
  if [[ ! -f "$PREFIX/schedules/$s.bonsai" ]]; then
    echo "no schedule $PREFIX/schedules/$s.bonsai" >&2; fail=1
  fi
done
if [[ -n "$REUSE" && ! -d "$REUSE" ]]; then
  echo "--reuse: no directory $REUSE" >&2; fail=1
fi
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
# Room for the run: the nearest existing parent of --out, since --out may
# not exist yet.
room_dir="$(realpath -m "$OUT")"
while [[ ! -d "$room_dir" ]]; do room_dir="$(dirname "$room_dir")"; done
room_gb=$(df --output=avail -B1G "$room_dir" | tail -1 | tr -d ' ')
if (( room_gb < 50 )); then
  echo "only ${room_gb} GB free under $OUT; a run needs 50 GB of room" >&2; fail=1
fi
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header; set BONSAI_CXX to a clang++" >&2
  fail=1
fi
OPTIX_SDK="${BONSAI_OPTIX_SDK:-$HOME/installs/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -f "$OPTIX_SDK/include/optix.h" || ! -f "$CUDA_DIR/include/cuda.h" ]]; then
  echo "no OptiX SDK at $OPTIX_SDK or no CUDA at $CUDA_DIR (BONSAI_OPTIX_SDK, CUDA_HOME)" >&2
  fail=1
fi
if ! command -v nvidia-smi > /dev/null; then
  echo "no nvidia-smi: no GPU to compare on" >&2; fail=1
elif ! require_locked_gpu_clock; then
  fail=1
fi
if [[ "$fail" != 0 ]]; then exit 1; fi
echo "$GPU_CLOCK_NOTE"

TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
TBB_FLAGS=()
if [[ -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  TBB_FLAGS=(-I"$TBB_PREFIX/include" -L"$TBB_PREFIX/lib"
             -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb)
fi
OPTIX_FLAGS=(-isystem "$OPTIX_SDK/include" -isystem "$CUDA_DIR/include")

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# A build tree is brought up to date; a directory that only holds a compiler
# (a copy kept of an earlier build, to measure a change against) is used as
# it is.
if [[ -f "$BONSAI_BUILD_DIR/CMakeCache.txt" ]]; then
  cmake --build "$BONSAI_BUILD_DIR" -j > /dev/null
fi
# Which compiler this run's numbers are of: the path, and the commit its
# tree is at (a kept copy's is the commit it was built from), for the log.
echo "compiler: $BONSAI_BUILD_DIR/compiler ($(git -C "$(dirname "$BONSAI_BUILD_DIR")" rev-parse --short HEAD 2>/dev/null || echo 'no git tree')); registers $([[ "$GPU_MAX_REGISTERS" == 0 ]] && echo "at OptiX's choice (no cap, as pbrt's own modules)" || echo "capped at $GPU_MAX_REGISTERS")"
if [[ -n "$REUSE" ]]; then REUSE="$(cd "$REUSE" && pwd)"; fi
# What a binary is made of, for --resume and --reuse: a key over the
# sources and tools behind it, written beside it when it is built. A
# binary whose key is not this run's is of another compiler or of other
# sources, and is built again rather than reused.
key_of() { sha256sum "$@" | sha256sum | cut -c1-16; }
# apps/pbrt's own headers, which the converter and the driver include;
# not the generated ones (render.h is a compile's output, the schema's
# header is made from the .fbs, which is in every key).
APP_HEADERS=()
for h in "$PREFIX"/*.h; do
  case "$h" in */render.h|*_generated.h) ;; *) APP_HEADERS+=("$h") ;; esac
done
# The converter is one of the binaries --resume reuses, as the schedules'
# are below: a sweep that calls this once per scene would otherwise rebuild
# it every time. Its key: its source and the headers, the schema, its
# build script, and pbrt itself, whose parser and library it links.
converter_key="$(key_of "$PREFIX/scene_dump.cpp" "$PREFIX/scene_geometry.fbs" \
    "$PREFIX/build_scene_dump.sh" "${APP_HEADERS[@]}" "$PBRT")"
if [[ "$RESUME" -eq 1 && -x "$OUT/scene_dump" && \
      "$(cat "$OUT/scene_dump.key" 2>/dev/null)" == "$converter_key" ]]; then
  echo "reusing $OUT/scene_dump"
elif [[ -n "$REUSE" && -x "$REUSE/scene_dump" && \
        "$(cat "$REUSE/scene_dump.key" 2>/dev/null)" == "$converter_key" ]]; then
  echo "reusing $REUSE/scene_dump (the same sources and pbrt)"
  cp "$REUSE/scene_dump" "$OUT/scene_dump"
  echo "$converter_key" > "$OUT/scene_dump.key"
else
  rm -f "$OUT/scene_dump.key"
  bash $PREFIX/build_scene_dump.sh "$OUT/scene_dump"
  echo "$converter_key" > "$OUT/scene_dump.key"
fi

# The machine has to be idle of everything before a timed run: the user's
# own benchmarks share it. Waits while it is not.
idle_check() {
  # A file named PAUSE in the output directory holds the run between timed
  # runs for as long as it exists (`touch DIR/PAUSE`, `rm DIR/PAUSE`): the
  # way to borrow the machine for a measurement of one's own without
  # stopping the process -- a stopped child ends a shell's wait, and the
  # run with it.
  if [[ -e "$OUT/PAUSE" ]]; then
    echo "paused: $OUT/PAUSE exists; waiting for it to go" >&2
    while [[ -e "$OUT/PAUSE" ]]; do sleep 5; done
    echo "resumed" >&2
  fi
  for _ in $(seq 90); do
    local busy gpu
    busy=$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR==2{print int($1)}')
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "${busy:-0}" -lt 50 && "$gpu" -eq 0 ]]; then return 0; fi
    echo "machine busy (top cpu ${busy}%, gpu processes $gpu); waiting" >&2
    sleep 20
  done
  echo "the machine never went idle" >&2; return 1
}
pbrt_seconds_of() {
  "$IMGTOOL" info "$1" 2>/dev/null | sed -n 's/.*(total \([0-9.]*\)s).*/\1/p' | head -1
}
min() { sort -g | head -1; }

# Runs a command while watching the machine: anything else on the GPU
# (nvidia-smi's list of compute processes, sampled every half second) or
# any process of someone else's at the top of the CPU is a disturbance,
# and the run's number is noise. The machine's other user runs short
# benchmarks that start and end inside a render, which the idle check
# before the run cannot see. Sets WATCHED_SEEN and WATCHED_WHAT.
watched() {
  WATCHED_SEEN=0; WATCHED_WHAT=""
  eval "$1" &
  local pid=$! gpu top
  while kill -0 "$pid" 2>/dev/null; do
    # (`|| true`: under pipefail a grep that finds nothing fails the
    # pipeline, and finding nothing is the good case.)
    # (`[No data]`, `[Not Found]`, `[Insufficient Permissions]`: what
    # nvidia-smi prints for a process it cannot name -- one of ours on its
    # way out, in practice -- and not someone else's.)
    gpu=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null |
          grep -v 'pbrt\|render.out\|^\[' | head -1 || true)
    if [[ -n "$gpu" ]]; then WATCHED_SEEN=1; WATCHED_WHAT="$gpu on the GPU"; fi
    top=$(ps -eo pcpu,comm --sort=-pcpu |
          awk 'NR > 1 && $2 !~ /^(pbrt|render\.out|nvidia-smi|ps|awk|bash|sort|grep|head|tail|sleep)$/ { if (int($1) >= 50) print $2 " at " int($1) "% cpu"; exit }' || true)
    if [[ -n "$top" ]]; then WATCHED_SEEN=1; WATCHED_WHAT="${WATCHED_WHAT:+$WATCHED_WHAT, }$top"; fi
    sleep 0.5
  done
  wait "$pid"
}

# A timed run: waits for the machine to be idle, runs the command watched,
# and redoes it -- up to three tries -- when something else ran meanwhile.
# Counts the redos in REDONE, and a run kept disturbed after three tries in
# KEPT_DISTURBED, for the cell's note in the table. Returns the command's
# status.
timed_run() {
  local attempt rc
  for attempt in 1 2 3; do
    idle_check
    watched "$1"; rc=$?
    if [[ "$WATCHED_SEEN" -eq 0 ]]; then return $rc; fi
    if [[ "$attempt" -lt 3 ]]; then
      echo "another process ran during that ($WATCHED_WHAT); redoing" >&2
      REDONE=$((REDONE + 1))
    else
      echo "another process ran during that ($WATCHED_WHAT); three tries, keeping this one" >&2
      KEPT_DISTURBED=$((KEPT_DISTURBED + 1))
    fi
  done
  return $rc
}
# A profiled run, watched. The command (`$2`) writes its profile to `$1.try`,
# which becomes `$1`; `$3` is a pipeline reading the total off the profile,
# left in PROFILE (empty when the run wrote none).
profile_run() {
  local file="$1" cmd="$2" extract="$3"
  PROFILE=""
  timed_run "$cmd" || true
  if [[ -f "$file.try" ]]; then
    PROFILE=$(eval "$extract" < "$file.try" 2>/dev/null || true)
    mv "$file.try" "$file"
  fi
}
# What the counters say of a side's runs, for the table's note column.
note_of() {
  local n=""
  [[ "$REDONE" -gt 0 ]] && n="redone $REDONE"
  [[ "$KEPT_DISTURBED" -gt 0 ]] && n="${n:+$n, }$KEPT_DISTURBED kept disturbed"
  echo "${n:--}"
}

# The schedules, compiled at once, each into its own directory: the
# compiler writes OUT/<s>/render.{h,o}, and the driver is compiled with
# BONSAI_RENDER_H naming that header (render_hook.cpp and nanovdb_shim.cpp
# include the macro, "render.h" beside the source when it is not given),
# so that no compile reads another's header; compile.log holds each one's
# output and render.key what it was made of. What the compiles would
# share under apps/pbrt is made once, before them: the NanoVDB shim
# (docs/foreign-functions.md, build_nanovdb_shim.sh: host bitcode and
# device bitcode for this GPU, linked into the generated module, and the
# same source compiled into the driver against the header) and the
# scene schema's header (scene_schema.sh, which flatc writes beside the
# .fbs).
SHIM_ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
SHIM_OUT=$(bash $PREFIX/build_nanovdb_shim.sh "$OUT" "$SHIM_ARCH")
NANOVDB_INCLUDE=$(echo "$SHIM_OUT" | sed -n 's/^NANOVDB_INCLUDE=//p')
mapfile -t LINK_FLAGS < <(echo "$SHIM_OUT" | grep -v '^NANOVDB_INCLUDE=')
SCHEMA_FLAGS=$(bash $PREFIX/scene_schema.sh)
COMPILER_FLAGS=(-p ssa --no-heap --ffp-contract --fast-math "${REGISTER_FLAGS[@]}")
# A renderer's key: the compiler binary and its flags; the renderer's
# .bonsai files (render.bonsai and what it imports, apps/pbrt's and the
# stdlib's) and the trees a schedule may import; the driver's sources and
# headers, the runtime headers the generated code includes, the shim's
# source and build script, the schema; the GPU, the shim's include path
# and the C++ compiler -- and the schedule's own file.
common_key="$( { key_of "$BONSAI_BUILD_DIR/compiler" "$PREFIX"/*.bonsai stdlib/*.bonsai \
        "$PREFIX"/schedules/trees/*.bonsai "$PREFIX/render_hook.cpp" \
        "$PREFIX/nanovdb_shim.cpp" "${APP_HEADERS[@]}" "$PREFIX/scene_geometry.fbs" \
        "$PREFIX/build_nanovdb_shim.sh" runtime/*.h
      echo "${COMPILER_FLAGS[*]} $SHIM_ARCH $NANOVDB_INCLUDE"
      "$BONSAI_CXX" --version | head -1; } | sha256sum | cut -c1-16)"
renderer_key() {
  { echo "$common_key"; sha256sum "$PREFIX/schedules/$1.bonsai"; } | sha256sum | cut -c1-16
}
compile_schedule() {
  local s="$1" start=$SECONDS
  "$BONSAI_BUILD_DIR/compiler" "${COMPILER_FLAGS[@]}" "${LINK_FLAGS[@]}" \
      -i $PREFIX/render.bonsai -i "$PREFIX/schedules/$s.bonsai" -b cpp -o "$OUT/$s/render"
  "$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $SCHEMA_FLAGS \
      -DBONSAI_RENDER_H="\"$OUT/$s/render.h\"" $PREFIX/render_hook.cpp \
      -DBONSAI_SHIM_CHECK "$NANOVDB_INCLUDE" $PREFIX/nanovdb_shim.cpp \
      "$OUT/$s/render.o" "${TBB_FLAGS[@]}" "${OPTIX_FLAGS[@]}" -o "$OUT/$s/render.out"
  rm -f "$OUT/$s/render.o"
  echo "compiled in $((SECONDS - start)) s"
}
declare -A RENDERER_KEY
to_compile=()
for s in $SCHEDULES; do
  mkdir -p "$OUT/$s"
  RENDERER_KEY[$s]="$(renderer_key "$s")"
  if [[ "$RESUME" -eq 1 && -x "$OUT/$s/render.out" ]]; then
    if [[ "$(cat "$OUT/$s/render.key" 2>/dev/null)" == "${RENDERER_KEY[$s]}" ]]; then
      echo "resuming with the compiled $s"
      continue
    fi
    echo "the compiled $s in $OUT is not of this compiler and these sources (no key, or another); compiling it again"
  fi
  if [[ -n "$REUSE" ]]; then
    if [[ -x "$REUSE/$s/render.out" && \
          "$(cat "$REUSE/$s/render.key" 2>/dev/null)" == "${RENDERER_KEY[$s]}" ]]; then
      echo "reusing $REUSE/$s/render.out (the same compiler and sources)"
      cp "$REUSE/$s/render.out" "$OUT/$s/render.out"
      cp "$REUSE/$s/render.h" "$OUT/$s/render.h"
      echo "${RENDERER_KEY[$s]}" > "$OUT/$s/render.key"
      echo "copied from $REUSE/$s/render.out on $(date)" > "$OUT/$s/compile.log"
      continue
    fi
    echo "nothing of $s to reuse in $REUSE (no renderer there, or one of another compiler or other sources); compiling"
  fi
  rm -f "$OUT/$s/render.key" "$OUT/$s/render.out"
  to_compile+=("$s")
done
if (( ${#to_compile[@]} > 0 )); then
  echo "compiling ${#to_compile[@]} schedule(s) at once: ${to_compile[*]} (each one's log is $OUT/<schedule>/compile.log)"
  compile_pids=()
  for s in "${to_compile[@]}"; do
    compile_schedule "$s" > "$OUT/$s/compile.log" 2>&1 &
    compile_pids+=("$!")
  done
  compile_failed=0
  for i in "${!to_compile[@]}"; do
    s="${to_compile[$i]}"
    if wait "${compile_pids[$i]}"; then
      echo "${RENDERER_KEY[$s]}" > "$OUT/$s/render.key"
      echo "compiled $s ($(tail -1 "$OUT/$s/compile.log"))"
    else
      echo "compiling $s failed; the end of $OUT/$s/compile.log:" >&2
      tail -20 "$OUT/$s/compile.log" >&2
      compile_failed=1
    fi
  done
  if (( compile_failed )); then exit 1; fi
fi

TABLE="$OUT/table.tsv"
TAB=$'\t'
if [[ "$RESUME" -eq 1 && -f "$TABLE" ]]; then
  echo "resuming $TABLE: a scene with every row is skipped, one begun is redone"
else
  printf 'scene\tspp\tside\twall_s\tkernel_ms\tspeedup\tagree\tverdict\tnote\timage\tpath\n' > "$TABLE"
fi
for SPP in $SPPS; do
for sc in $SCENES; do
  name=$(basename "$sc"); dir="$SCENES_DIR/$(dirname "$sc")"
  # The cell's files and its rows are keyed by the scene's directory as
  # well as its name: kroken and watercolor both have a camera-1, and a
  # sweep that resumed into one table took the second for the first.
  cell="$OUT/$(basename "$(dirname "$sc")")-$name-s$SPP"
  if [[ "$RESUME" -eq 1 ]]; then
    have=$(grep -c "^$name$TAB$SPP$TAB.*$TAB$sc\$" "$TABLE" || true)
    if [[ "$have" -ge $((1 + $(echo $SCHEDULES | wc -w))) ]]; then
      echo "== $sc at $SPP spp: done already"
      continue
    fi
    grep -v "^$name$TAB$SPP$TAB.*$TAB$sc\$" "$TABLE" > "$TABLE.tmp" || true
    mv "$TABLE.tmp" "$TABLE"
  fi
  echo "== $name at $SPP spp"
  # This renderer's scene, converted once for every count and every run
  # (convert_scene.sh: apps/pbrt/scenes/converted/<dir>/<name>.txt, redone
  # only when the converter or the scene changed). `--gpu`: the scene as
  # `pbrt --gpu` builds it (a PLY's quads as triangles, which the hardware
  # traces), since that is the reference. The count is not the
  # conversion's: it is the renderer's own `--spp` below, as it is pbrt's
  # own `--spp` on the other side.
  if ! conv=$(SCENES_DIR="$SCENES_DIR" PBRT="$PBRT" bash $PREFIX/convert_scene.sh "$OUT/scene_dump" "$sc" --gpu); then
    continue
  fi
  # pbrt: REPEATS timed renders (the least), one to a full-float image for the
  # comparison, and one under --stats for its kernel profile. A scene that
  # `pbrt --gpu` cannot render still gets its row and its cells (10-25/26:
  # it segfaults inside the driver's acceleration-structure build, called
  # from pbrt's OptiXAggregate::buildOptixBVH -- on the hair scenes for the
  # bilinear patches it turns curves into (buildBVHForBLPs), and on
  # zero-day's two frames for their 65 885 triangle meshes
  # (buildBVHForTriangles); the run used to end there with the scenes after
  # it unmeasured). The reference image is then pbrt's wavefront integrator on the CPU
  # (`pbrt --wavefront`: the same integrator, the CPU's intersection),
  # rendered once and untimed: the row's side says so, its wall, kernel and
  # speedup are `-`, as is every speedup of the scene, and the comparison
  # of the images is the same.
  REDONE=0; KEPT_DISTURBED=0
  pbrt_secs=""
  pbrt_side="pbrt --gpu"; pbrt_ref=("$PBRT" --gpu); pbrt_gpu_ok=1
  for _ in $(seq "$REPEATS"); do
    if ! timed_run "(cd '$dir' && '$PBRT' --gpu --spp '$SPP' --outfile '$cell.pbrt.exr' '$name.pbrt' > '$cell.pbrt.log' 2>&1)"; then
      echo "  pbrt --gpu failed on $name (see $cell.pbrt.log); the reference image is pbrt --wavefront on the CPU, untimed" >&2
      pbrt_side="pbrt --wavefront (cpu)"; pbrt_ref=("$PBRT" --wavefront); pbrt_gpu_ok=0
      pbrt_secs=""
      break
    fi
    pbrt_secs="$pbrt_secs $(pbrt_seconds_of "$cell.pbrt.exr")"
  done
  pbrt_wall=$(echo $pbrt_secs | tr ' ' '\n' | min)
  # compare_gbuffer.py is told pbrt's time only when there is one.
  pbrt_secs_arg=()
  if [[ -n "$pbrt_wall" ]]; then pbrt_secs_arg=(--pbrt-seconds "$pbrt_wall"); fi
  idle_check
  # pbrt's radiance as full floats for the comparison: a PFM straight from an
  # `rgb` film, and from a `gbuffer` film -- which writes EXR and nothing
  # else -- the R, G, B channels of its EXR, pulled out by imgtool as
  # compare.sh pulls them. Full floats means telling the film so: pbrt's
  # `savefp16` is true by default and quantizes the pixels to halves before
  # any file is written, PFM included, which rounds everything below 6e-8 to
  # zero and costs a comparison within 1e-3 its last digit (see compare.sh's
  # film_full_floats, which this follows: the scene on pbrt's standard
  # input from its own directory, its Film told `"bool savefp16" false`
  # unless it says something itself).
  film=$(sed -n 's/^[[:space:]]*Film[[:space:]]*"\([a-z]*\)".*/\1/p' "$dir/$name.pbrt" | head -1)
  full_floats() {
    if grep -q 'savefp16' "$1"; then
      cat "$1"
    else
      awk '/^[[:space:]]*Film[[:space:]]/ {
             sub(/^[[:space:]]*Film[[:space:]]+"[a-z]+"/, "& \"bool savefp16\" false")
           }
           { print }' "$1"
    fi
  }
  # (The image files first removed: a cell redone after a failure must not
  # be compared against the image of an earlier run.)
  rm -f "$cell.pbrt-radiance.exr" "$cell.pbrt-radiance.pfm"
  if [[ "$film" == "gbuffer" ]]; then
    (cd "$dir" && full_floats "$name.pbrt" | "${pbrt_ref[@]}" --spp "$SPP" --outfile "$cell.pbrt-radiance.exr" >> "$cell.pbrt.log" 2>&1) || true
    if [[ -f "$cell.pbrt-radiance.exr" ]]; then
      "$IMGTOOL" convert --channels R,G,B --outfile "$cell.pbrt-radiance.pfm" "$cell.pbrt-radiance.exr" > /dev/null
    fi
  else
    (cd "$dir" && full_floats "$name.pbrt" | "${pbrt_ref[@]}" --spp "$SPP" --outfile "$cell.pbrt-radiance.pfm" >> "$cell.pbrt.log" 2>&1) || true
  fi
  pbrt_kernel=""
  if [[ "$pbrt_gpu_ok" -eq 1 ]]; then
    profile_run "$cell.pbrt.stats" \
      "(cd '$dir' && '$PBRT' --gpu --stats --spp '$SPP' --outfile '$cell.pbrt-stats.exr' '$name.pbrt' > '$cell.pbrt.stats.try' 2>&1)" \
      "grep 'Total rendering time' | awk '{print \$4}'"
    pbrt_kernel="$PROFILE"
  fi
  if [[ -f "$cell.pbrt-radiance.pfm" ]]; then
    python3 $PREFIX/to_png.py "$cell.pbrt-radiance.pfm" "$cell.pbrt.png" > /dev/null
  else
    echo "  pbrt wrote no reference image for $name (see $cell.pbrt.log); the scene's cells have no verdict" >&2
  fi
  pbrt_speedup="-"
  if [[ -n "$pbrt_wall" ]]; then pbrt_speedup="1.00x"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t-\t-\t%s\t%s\t%s\n' "$name" "$SPP" "$pbrt_side" "${pbrt_wall:--}" "${pbrt_kernel:--}" "$pbrt_speedup" "$(note_of)" "$cell.pbrt.png" "$sc" | tee -a "$TABLE"
  # Ours: each schedule as one process rendering REPEATS times, the least of
  # them (the first is the warm-up), and once more with the kernel profile
  # on, which is of the last render.
  for s in $SCHEDULES; do
    REDONE=0; KEPT_DISTURBED=0
    wall=""
    if timed_run "BONSAI_REPEATS=$REPEATS '$OUT/$s/render.out' --spp '$SPP' --no-implicit-copies '$conv' '$cell.$s.pfm' > '$cell.$s.log' 2>&1"; then
      wall=$(sed -n 's/^render seconds: //p' "$cell.$s.log" | head -1)
    else
      echo "$s failed on $name: see $cell.$s.log" >&2
    fi
    profile_run "$cell.$s.stats" \
      "BONSAI_REPEATS=$REPEATS BONSAI_KERNEL_STATS=1 '$OUT/$s/render.out' --spp '$SPP' --no-implicit-copies '$conv' '$cell.$s.pfm' > '$cell.$s.stats.try' 2>&1" \
      "sed -n 's/.*total kernel time *\([0-9.]*\) ms.*/\1/p' | head -1"
    kernel="$PROFILE"
    # (`|| true`: a FAILED verdict is an exit status, and a row of the table
    # rather than the end of the run.)
    cmp=$(python3 $PREFIX/compare_gbuffer.py --radiance-only --radiance "$cell.pbrt-radiance.pfm" "$cell.$s-radiance.pfm" \
          "${pbrt_secs_arg[@]}" --bonsai-seconds "${wall:-1}" --repeats "$REPEATS" 2>&1 | tee "$cell.$s.compare" || true)
    agree=$(echo "$cmp" | sed -n 's/.*agree to 1e-03 relative (\([0-9.]*%\)).*/\1/p' | head -1)
    # The verdict line, `ok: ...` or `FAILED: ...` (`|| true`: a grep that
    # matches nothing is an exit status, and must not end the run).
    verdict=$(echo "$cmp" | grep -E '^ok|^FAILED|mismatch|differ' | head -1 | cut -c1-200 || true)
    if [[ ! -f "$cell.pbrt-radiance.pfm" ]]; then verdict="no reference image"; fi
    # The user's rule of 2026-10-18: a scene whose two pbrt renderers -- the
    # CPU integrator and the GPU wavefront the reference is -- disagree with
    # each other more than ours disagrees with the reference is rendered
    # correctly, whatever the lit-pixel and mean tolerances say of it
    # (pavilion-day at 16 spp: a glass scene under an environment map,
    # where pbrt's two renders part six times further than ours parts from
    # either). A cell that fails is held against that spread: pbrt renders
    # the scene on the CPU once per cell, untimed, at the same count, by
    # `volpath` with the scene's integrator parameters -- what the wavefront
    # runs whatever the scene names (wavefront/integrator.cpp) -- and the
    # comparison is redone with it (compare_gbuffer.py --pbrt-spread).
    if [[ "$verdict" == FAILED* ]]; then
      spread="$cell.pbrt-cpu-radiance.pfm"
      if [[ ! -f "$spread" ]]; then
        echo "  $s differs from $pbrt_side on $name at $SPP spp; rendering pbrt's CPU integrator for the spread" >&2
        as_volpath() {
          sed 's/^\([[:space:]]*Integrator[[:space:]]*\)"[a-z]*"/\1"volpath"/'
        }
        if [[ "$film" == "gbuffer" ]]; then
          (cd "$dir" && full_floats "$name.pbrt" | as_volpath | "$PBRT" --spp "$SPP" --outfile "$cell.pbrt-cpu-radiance.exr" > "$cell.pbrt-cpu.log" 2>&1) || true
          if [[ -f "$cell.pbrt-cpu-radiance.exr" ]]; then
            "$IMGTOOL" convert --channels R,G,B --outfile "$spread" "$cell.pbrt-cpu-radiance.exr" > /dev/null || true
          fi
        else
          (cd "$dir" && full_floats "$name.pbrt" | as_volpath | "$PBRT" --spp "$SPP" --outfile "$spread" > "$cell.pbrt-cpu.log" 2>&1) || true
        fi
      fi
      if [[ -f "$spread" ]]; then
        cmp=$(python3 $PREFIX/compare_gbuffer.py --radiance-only --radiance "$cell.pbrt-radiance.pfm" "$cell.$s-radiance.pfm" \
              --pbrt-spread "$spread" \
              "${pbrt_secs_arg[@]}" --bonsai-seconds "${wall:-1}" --repeats "$REPEATS" 2>&1 | tee "$cell.$s.compare" || true)
        agree=$(echo "$cmp" | sed -n 's/.*agree to 1e-03 relative (\([0-9.]*%\)).*/\1/p' | head -1)
        verdict=$(echo "$cmp" | grep -E '^ok|^FAILED|mismatch|differ' | head -1 | cut -c1-200 || true)
      else
        echo "  pbrt's CPU render failed (see $cell.pbrt-cpu.log); the verdict stands" >&2
      fi
    fi
    # (A schedule that failed wrote no image: its row says so, and the run
    # goes on to the next schedule rather than ending here.)
    if [[ -f "$cell.$s-radiance.pfm" ]]; then
      python3 $PREFIX/to_png.py "$cell.$s-radiance.pfm" "$cell.$s.png" > /dev/null
    fi
    speedup=$(awk -v p="${pbrt_wall:-0}" -v o="${wall:-0}" 'BEGIN{ if (o > 0 && p > 0) printf "%.2fx", p/o; else print "-" }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$SPP" "$s" "${wall:--}" "${kernel:--}" "$speedup" "${agree:-?}" "${verdict:-?}" "$(note_of)" "$cell.$s.png" "$sc" | tee -a "$TABLE"
  done
done
done
echo
echo "table: $TABLE (wall in seconds, the least of $REPEATS -- pbrt's of $REPEATS processes, ours of $REPEATS renders in one process, the first the warm-up; kernel time in ms, pbrt's from one --stats run, ours the profile of the fastest render; speedup is pbrt's wall over ours; the note says how many runs were redone because something else ran alongside, and how many were kept disturbed after three tries; $GPU_CLOCK_NOTE$([[ "$GPU_MAX_REGISTERS" != 0 ]] && echo "; our schedules compiled with registers capped at $GPU_MAX_REGISTERS, where pbrt's modules are at OptiX's choice"))"
column -t -s $'\t' "$TABLE"
