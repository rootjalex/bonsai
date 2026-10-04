#!/usr/bin/env bash
# The GPU comparison: pbrt --gpu against every GPU schedule of this renderer,
# on the scenes given, at the sample counts given -- one command, one table.
#
#     apps/pbrt/gpu_compare.sh [--spp "N ..."] [--repeats N] [--out DIR] [--resume]
#                              [--schedules "a b c"] [--scenes "d/s ..."]
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
# where each is.
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
# Scenes are named as `<dir>/<name>` under $SCENES_DIR (~/projects/
# pbrt-v4-scenes): killeroos/killeroo-simple, pbrt-book/book. Only scenes
# scene_dump converts can be given; the survey of the scene set is in
# PLAN.md.
set -euo pipefail
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"

SPPS="16"
REPEATS=3
OUT="$PREFIX/gpu-compare-out"
SCHEDULES="gpu-optix gpu-optix-mega gpu-optix-mega-ser"
SCENES="killeroos/killeroo-simple killeroos/killeroo-gold pbrt-book/book"
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

# Everything the run will act on, checked before any of it starts: a run is
# minutes long and a missing scene should not be found at the end of it.
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
SCENES_DIR="${SCENES_DIR:-$HOME/projects/pbrt-v4-scenes}"
BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
BONSAI_CXX="${BONSAI_CXX:-clang++}"
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
fi
if [[ "$fail" != 0 ]]; then exit 1; fi

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
bash $PREFIX/build_scene_dump.sh "$OUT/scene_dump"

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
    gpu=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null |
          grep -v 'pbrt\|render.out' | head -1 || true)
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

# The schedules, compiled once each. The compiler writes $PREFIX/render.h
# and render.o, which the driver's include finds first, so one at a time.
for s in $SCHEDULES; do
  if [[ "$RESUME" -eq 1 && -x "$OUT/$s/render.out" ]]; then
    echo "resuming with the compiled $s"
    continue
  fi
  echo "compiling schedule $s"
  mkdir -p "$OUT/$s"
  # The NanoVDB shim (docs/foreign-functions.md, build_nanovdb_shim.sh):
  # host bitcode and PTX for this GPU, linked into the generated module,
  # and the same source compiled into the driver against the header.
  if [[ -z "${SHIM_OUT:-}" ]]; then
    SHIM_ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
    SHIM_OUT=$(bash $PREFIX/build_nanovdb_shim.sh "$OUT" "$SHIM_ARCH")
    NANOVDB_INCLUDE=$(echo "$SHIM_OUT" | sed -n 's/^NANOVDB_INCLUDE=//p')
    mapfile -t LINK_FLAGS < <(echo "$SHIM_OUT" | grep -v '^NANOVDB_INCLUDE=')
  fi
  "$BONSAI_BUILD_DIR/compiler" -p ssa --no-heap --ffp-contract --fast-math \
      --gpu-max-registers 128 "${LINK_FLAGS[@]}" -i $PREFIX/render.bonsai \
      -i "$PREFIX/schedules/$s.bonsai" -b cpp -o $PREFIX/render
  "$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $(bash $PREFIX/scene_schema.sh) \
      $PREFIX/render_hook.cpp \
      -DBONSAI_SHIM_CHECK "$NANOVDB_INCLUDE" $PREFIX/nanovdb_shim.cpp \
      $PREFIX/render.o "${TBB_FLAGS[@]}" "${OPTIX_FLAGS[@]}" -o "$OUT/$s/render.out"
  rm -f $PREFIX/render.o
done

TABLE="$OUT/table.tsv"
TAB=$'\t'
if [[ "$RESUME" -eq 1 && -f "$TABLE" ]]; then
  echo "resuming $TABLE: a scene with every row is skipped, one begun is redone"
else
  printf 'scene\tspp\tside\twall_s\tkernel_ms\tspeedup\tagree\tverdict\tnote\timage\n' > "$TABLE"
fi
for SPP in $SPPS; do
for sc in $SCENES; do
  name=$(basename "$sc"); dir="$SCENES_DIR/$(dirname "$sc")"
  cell="$OUT/$name-s$SPP"
  if [[ "$RESUME" -eq 1 ]]; then
    have=$(grep -c "^$name$TAB$SPP$TAB" "$TABLE" || true)
    if [[ "$have" -ge $((1 + $(echo $SCHEDULES | wc -w))) ]]; then
      echo "== $name at $SPP spp: done already"
      continue
    fi
    grep -v "^$name$TAB$SPP$TAB" "$TABLE" > "$TABLE.tmp" || true
    mv "$TABLE.tmp" "$TABLE"
  fi
  echo "== $name at $SPP spp"
  # This renderer's scene, converted once.
  (cd "$dir" && "$OUT/scene_dump" --spp "$SPP" "$name.pbrt" "$cell.txt" > "$cell.dump.log" 2>&1) ||
    { echo "scene_dump refused $sc: $(tail -1 "$cell.dump.log")" >&2; continue; }
  # pbrt: REPEATS timed renders (the least), one to a full-float image for the
  # comparison, and one under --stats for its kernel profile.
  REDONE=0; KEPT_DISTURBED=0
  pbrt_secs=""
  for _ in $(seq "$REPEATS"); do
    timed_run "(cd '$dir' && '$PBRT' --gpu --spp '$SPP' --outfile '$cell.pbrt.exr' '$name.pbrt' > '$cell.pbrt.log' 2>&1)"
    pbrt_secs="$pbrt_secs $(pbrt_seconds_of "$cell.pbrt.exr")"
  done
  pbrt_wall=$(echo $pbrt_secs | tr ' ' '\n' | min)
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
  if [[ "$film" == "gbuffer" ]]; then
    (cd "$dir" && full_floats "$name.pbrt" | "$PBRT" --gpu --spp "$SPP" --outfile "$cell.pbrt-radiance.exr" >> "$cell.pbrt.log" 2>&1)
    "$IMGTOOL" convert --channels R,G,B --outfile "$cell.pbrt-radiance.pfm" "$cell.pbrt-radiance.exr" > /dev/null
  else
    (cd "$dir" && full_floats "$name.pbrt" | "$PBRT" --gpu --spp "$SPP" --outfile "$cell.pbrt-radiance.pfm" >> "$cell.pbrt.log" 2>&1)
  fi
  profile_run "$cell.pbrt.stats" \
    "(cd '$dir' && '$PBRT' --gpu --stats --spp '$SPP' --outfile '$cell.pbrt-stats.exr' '$name.pbrt' > '$cell.pbrt.stats.try' 2>&1)" \
    "grep 'Total rendering time' | awk '{print \$4}'"
  pbrt_kernel="$PROFILE"
  python3 $PREFIX/to_png.py "$cell.pbrt-radiance.pfm" "$cell.pbrt.png" > /dev/null
  printf '%s\t%s\tpbrt --gpu\t%s\t%s\t1.00x\t-\t-\t%s\t%s\n' "$name" "$SPP" "$pbrt_wall" "$pbrt_kernel" "$(note_of)" "$cell.pbrt.png" | tee -a "$TABLE"
  # Ours: each schedule as one process rendering REPEATS times, the least of
  # them (the first is the warm-up), and once more with the kernel profile
  # on, which is of the last render.
  for s in $SCHEDULES; do
    REDONE=0; KEPT_DISTURBED=0
    wall=""
    if timed_run "BONSAI_REPEATS=$REPEATS '$OUT/$s/render.out' --no-implicit-copies '$cell.txt' '$cell.$s.pfm' > '$cell.$s.log' 2>&1"; then
      wall=$(sed -n 's/^render seconds: //p' "$cell.$s.log" | head -1)
    else
      echo "$s failed on $name: see $cell.$s.log" >&2
    fi
    profile_run "$cell.$s.stats" \
      "BONSAI_REPEATS=$REPEATS BONSAI_KERNEL_STATS=1 '$OUT/$s/render.out' --no-implicit-copies '$cell.txt' '$cell.$s.pfm' > '$cell.$s.stats.try' 2>&1" \
      "sed -n 's/.*total kernel time *\([0-9.]*\) ms.*/\1/p' | head -1"
    kernel="$PROFILE"
    # (`|| true`: a FAILED verdict is an exit status, and a row of the table
    # rather than the end of the run.)
    cmp=$(python3 $PREFIX/compare_gbuffer.py --radiance-only --radiance "$cell.pbrt-radiance.pfm" "$cell.$s-radiance.pfm" \
          --pbrt-seconds "$pbrt_wall" --bonsai-seconds "${wall:-1}" --repeats "$REPEATS" 2>&1 | tee "$cell.$s.compare" || true)
    agree=$(echo "$cmp" | sed -n 's/.*agree to 1e-03 relative (\([0-9.]*%\)).*/\1/p' | head -1)
    # The verdict line, `ok: ...` or `FAILED: ...` (`|| true`: a grep that
    # matches nothing is an exit status, and must not end the run).
    verdict=$(echo "$cmp" | grep -E '^ok|^FAILED|mismatch|differ' | head -1 | cut -c1-60 || true)
    python3 $PREFIX/to_png.py "$cell.$s-radiance.pfm" "$cell.$s.png" > /dev/null
    speedup=$(awk -v p="$pbrt_wall" -v o="${wall:-0}" 'BEGIN{ if (o > 0) printf "%.2fx", p/o; else print "-" }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$SPP" "$s" "${wall:--}" "${kernel:--}" "$speedup" "${agree:-?}" "${verdict:-?}" "$(note_of)" "$cell.$s.png" | tee -a "$TABLE"
  done
done
done
echo
echo "table: $TABLE (wall in seconds, the least of $REPEATS -- pbrt's of $REPEATS processes, ours of $REPEATS renders in one process, the first the warm-up; kernel time in ms, pbrt's from one --stats run, ours the profile of the fastest render; speedup is pbrt's wall over ours; the note says how many runs were redone because something else ran alongside, and how many were kept disturbed after three tries)"
column -t -s $'\t' "$TABLE"
