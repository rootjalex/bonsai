#!/usr/bin/env bash
# The GPU comparison: pbrt --gpu against every GPU schedule of this renderer,
# on the scenes given, at one sample count -- one command, one table.
#
#     apps/pbrt/gpu_compare.sh [--spp N] [--repeats N] [--out DIR]
#                              [--schedules "a b c"] [--scenes "d/s ..."]
#
# For each scene, pbrt --gpu is run REPEATS times and its own render timer is
# read from the image it writes; the least of them is pbrt's time. Each
# schedule is compiled once, then run REPEATS times as separate processes
# (each a cold start, as pbrt's is) and the mean of the driver's timer is
# ours. Beside each time is the kernel time: pbrt's from `--stats` (its
# "Wavefront Kernel Profile"), ours from BONSAI_KERNEL_STATS
# (runtime/bonsai_cuda.h) -- the device's busy time, apart from whatever
# either side's timer counts around it (pbrt's timer includes its memory
# prefetch and first launches, some sixty milliseconds on this machine; ours
# includes the host's launch gaps). Every image is checked against pbrt's as
# compare.sh checks it, converted to a PNG, and the table says where each is.
#
# The machine is checked idle before every timed run: another process on the
# GPU, or a busy CPU, means waiting rather than timing.
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

SPP=16
REPEATS=3
OUT="$PREFIX/gpu-compare-out"
SCHEDULES="gpu-optix gpu-optix-mega gpu-optix-mega-ser"
SCENES="killeroos/killeroo-simple killeroos/killeroo-gold pbrt-book/book"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spp) SPP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
if ! [[ "$SPP" =~ ^[0-9]+$ && "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 1 ]]; then
  echo "--spp and --repeats take positive integers" >&2; fail=1
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
cmake --build "$BONSAI_BUILD_DIR" -j > /dev/null
bash $PREFIX/build_scene_dump.sh "$OUT/scene_dump"

# The machine has to be idle of everything before a timed run: the user's
# own benchmarks share it. Waits while it is not.
idle_check() {
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
mean() { awk '{ s += $1; n++ } END { if (n) printf "%.4f", s / n }'; }
min() { sort -g | head -1; }

# The schedules, compiled once each. The compiler writes $PREFIX/render.h
# and render.o, which the driver's include finds first, so one at a time.
for s in $SCHEDULES; do
  echo "compiling schedule $s"
  mkdir -p "$OUT/$s"
  "./$BONSAI_BUILD_DIR/compiler" -p ssa --no-heap --ffp-contract --fast-math \
      --gpu-max-registers 128 -i $PREFIX/render.bonsai \
      -i "$PREFIX/schedules/$s.bonsai" -b cpp -o $PREFIX/render
  "$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX $PREFIX/render_hook.cpp \
      $PREFIX/render.o "${TBB_FLAGS[@]}" "${OPTIX_FLAGS[@]}" -o "$OUT/$s/render.out"
  rm -f $PREFIX/render.o
done

TABLE="$OUT/table.tsv"
printf 'scene\tspp\tside\twall_s\tkernel_ms\tspeedup\tagree\tverdict\timage\n' > "$TABLE"
for sc in $SCENES; do
  name=$(basename "$sc"); dir="$SCENES_DIR/$(dirname "$sc")"
  cell="$OUT/$name-s$SPP"
  echo "== $name at $SPP spp"
  # This renderer's scene, converted once.
  (cd "$dir" && "$OUT/scene_dump" --spp "$SPP" "$name.pbrt" "$cell.txt" > "$cell.dump.log" 2>&1) ||
    { echo "scene_dump refused $sc: $(tail -1 "$cell.dump.log")" >&2; continue; }
  # pbrt: REPEATS timed renders (the least), one to a full-float image for the
  # comparison, and one under --stats for its kernel profile.
  pbrt_secs=""
  for _ in $(seq "$REPEATS"); do
    idle_check
    (cd "$dir" && "$PBRT" --gpu --spp "$SPP" --outfile "$cell.pbrt.exr" "$name.pbrt" > "$cell.pbrt.log" 2>&1)
    pbrt_secs="$pbrt_secs $(pbrt_seconds_of "$cell.pbrt.exr")"
  done
  pbrt_wall=$(echo $pbrt_secs | tr ' ' '\n' | min)
  idle_check
  (cd "$dir" && "$PBRT" --gpu --spp "$SPP" --outfile "$cell.pbrt-radiance.pfm" "$name.pbrt" >> "$cell.pbrt.log" 2>&1)
  idle_check
  (cd "$dir" && "$PBRT" --gpu --stats --spp "$SPP" --outfile "$cell.pbrt-stats.exr" "$name.pbrt" > "$cell.pbrt.stats" 2>&1)
  pbrt_kernel=$(grep "Total rendering time" "$cell.pbrt.stats" | awk '{print $4}')
  python3 $PREFIX/to_png.py "$cell.pbrt-radiance.pfm" "$cell.pbrt.png" > /dev/null
  printf '%s\t%s\tpbrt --gpu\t%s\t%s\t1.00x\t-\t-\t%s\n' "$name" "$SPP" "$pbrt_wall" "$pbrt_kernel" "$cell.pbrt.png" | tee -a "$TABLE"
  # Ours: each schedule REPEATS times as its own process (the mean), and once
  # more with the kernel profile on.
  for s in $SCHEDULES; do
    secs=""
    for _ in $(seq "$REPEATS"); do
      idle_check
      BONSAI_REPEATS=1 "$OUT/$s/render.out" --no-implicit-copies "$cell.txt" "$cell.$s.pfm" > "$cell.$s.log" 2>&1 ||
        { echo "$s failed on $name: see $cell.$s.log" >&2; break; }
      secs="$secs $(sed -n 's/^render seconds: //p' "$cell.$s.log")"
    done
    wall=$(echo $secs | tr ' ' '\n' | mean)
    idle_check
    BONSAI_REPEATS=1 BONSAI_KERNEL_STATS=1 "$OUT/$s/render.out" --no-implicit-copies "$cell.txt" "$cell.$s.pfm" > "$cell.$s.stats" 2>&1 || true
    kernel=$(sed -n 's/.*total kernel time *\([0-9.]*\) ms.*/\1/p' "$cell.$s.stats" | head -1)
    cmp=$(python3 $PREFIX/compare_gbuffer.py --radiance-only --radiance "$cell.pbrt-radiance.pfm" "$cell.$s-radiance.pfm" \
          --pbrt-seconds "$pbrt_wall" --bonsai-seconds "${wall:-1}" --repeats "$REPEATS" 2>&1 | tee "$cell.$s.compare")
    agree=$(echo "$cmp" | sed -n 's/.*agree to 1e-03 relative (\([0-9.]*%\)).*/\1/p' | head -1)
    verdict=$(echo "$cmp" | grep -E '^ok|mismatch|differ' | head -1 | cut -c1-40)
    python3 $PREFIX/to_png.py "$cell.$s-radiance.pfm" "$cell.$s.png" > /dev/null
    speedup=$(awk -v p="$pbrt_wall" -v o="${wall:-0}" 'BEGIN{ if (o > 0) printf "%.2fx", p/o; else print "-" }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$SPP" "$s" "${wall:--}" "${kernel:--}" "$speedup" "${agree:-?}" "${verdict:-?}" "$cell.$s.png" | tee -a "$TABLE"
  done
done
echo
echo "table: $TABLE (wall in seconds -- pbrt's the least of $REPEATS, ours the mean of $REPEATS cold runs; kernel time in ms; speedup is pbrt's wall over ours)"
column -t -s $'\t' "$TABLE"
