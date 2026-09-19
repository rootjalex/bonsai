#!/bin/bash

set -euo pipefail

# The schedules against each other and against pbrt, over path depth and
# sample count, on the scenes this renderer is measured on.
#
#     apps/pbrt/bench.sh [options] [scene.pbrt ...]
#
#     --schedules "scalar perlane packet"   the files in schedules/ to run
#     --depths "1 2 3 4 5"                  the depth matrix, at --depth-spp
#     --depth-spp 64                        samples per pixel for the depth matrix
#     --spps "16 64 256"                    the sample-count matrix, at --spp-depth
#     --spp-depth 5                         path depth for the sample-count matrix
#     --repeats 3                           best of this many runs, both sides
#     --out DIR                             where everything is written
#                                           (default apps/pbrt/bench-out)
#
# Two matrices rather than their product: how the schedules scale with depth,
# where the gangs diverge after the first bounce, and how they scale with
# samples, where more lanes share a pixel's rays. The cell the two share is
# rendered once.
#
# For every cell the scene is converted once (scene_dump, at that depth and
# count), pbrt renders it, each schedule renders it, and every image is checked
# against pbrt's the way compare.sh checks one -- the lit pixels, the mean, the
# fraction of pixels agreeing closely -- and against the other schedules' bit
# for bit. What comes out is a TSV, `results.tsv` in the output directory, one
# row per (scene, depth, spp, schedule), and the tables printed at the end.
# The compile time of each schedule is measured too, once, since it is part of
# what a schedule costs.
#
# The tree is pbrt's own (scene_dump --pbrt-tree) unless PBRT_TREE=0, so that
# what is timed is the traversal each schedule makes of one tree rather than
# whose builder found the better one; compare.sh, whose question is whether
# the whole renderer is right, builds its own by default.
#
# Needs a built pbrt, as compare.sh does: PBRT=... to point at it. Runs one
# render at a time, and nothing else should be running beside it: every source
# of noise adds time and the minimum of a few runs only removes some of it.

if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"

SCHEDULES="scalar perlane packet"
DEPTHS="1 2 3 4 5"
DEPTH_SPP=64
SPPS="16 64 256"
SPP_DEPTH=5
REPEATS=3
WORK="$PREFIX/bench-out"
SCENES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedules) SCHEDULES="$2"; shift 2 ;;
    --depths) DEPTHS="$2"; shift 2 ;;
    --depth-spp) DEPTH_SPP="$2"; shift 2 ;;
    --spps) SPPS="$2"; shift 2 ;;
    --spp-depth) SPP_DEPTH="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) WORK="$2"; shift 2 ;;
    --*) echo "unknown option $1" >&2; exit 1 ;;
    *) SCENES+=("$1"); shift ;;
  esac
done
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=("$HOME/projects/pbrt-v4-scenes/pbrt-book/book.pbrt"
          "$HOME/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt"
          "$HOME/projects/pbrt-v4-scenes/ganesha/ganesha.pbrt"
          "$PREFIX/scenes/bvh-lights.pbrt"
          "$PREFIX/scenes/instances.pbrt")
fi

PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
IMGTOOL="${IMGTOOL:-$(dirname "$PBRT")/imgtool}"
BONSAI_CXX="${BONSAI_CXX:-clang++}"
PBRT_TREE="${PBRT_TREE:-1}"
if [[ ! -x "$PBRT" ]]; then
  echo "no pbrt at $PBRT -- set PBRT to a built one" >&2
  exit 1
fi
for scene in "${SCENES[@]}"; do
  if [[ ! -f "$scene" ]]; then
    echo "no scene at $scene" >&2
    exit 1
  fi
done

# See compare.sh for both of these: the header needs clang's ext_vector_type,
# and TBB lives beside the compiler.
if ! echo 'typedef float f3 __attribute__((ext_vector_type(3)));
           float pick(f3 v) { return v.y; }' |
    "$BONSAI_CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
  echo "$BONSAI_CXX cannot compile the generated header: set BONSAI_CXX to a" >&2
  echo "clang++." >&2
  exit 1
fi
TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
TBB_FLAGS=()
if [[ -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  TBB_FLAGS=(-I"$TBB_PREFIX/include" -L"$TBB_PREFIX/lib"
             -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb)
else
  echo "no TBB beside $BONSAI_CXX; the renders will balance their loops" >&2
  echo "with std::thread, which is slower." >&2
fi

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
RESULTS="$WORK/results.tsv"
COMPILES="$WORK/compile.tsv"

cmake --build build -j
bash $PREFIX/build_scene_dump.sh "$WORK/scene_dump"
"$WORK/scene_dump" --check-tables

# Every schedule's renderer, built once. The compile is timed as the whole
# `compiler` run, which is what a user of the schedule waits for.
printf 'schedule\tcompile_seconds\n' > "$COMPILES"
for schedule in $SCHEDULES; do
  file="$PREFIX/schedules/$schedule.bonsai"
  if [[ ! -f "$file" ]]; then
    echo "no schedule $file" >&2
    exit 1
  fi
  echo "== compiling $schedule"
  started=$(date +%s.%N)
  ./build/compiler -p ssa --no-heap --ffp-contract \
      -i $PREFIX/render.bonsai -i "$file" -b cpp -o "$WORK/render_$schedule"
  finished=$(date +%s.%N)
  printf '%s\t%.2f\n' "$schedule" "$(echo "$finished - $started" | bc -l)" \
      >> "$COMPILES"
  # render_hook.cpp includes "render.h": each schedule's header in a
  # directory of its own.
  mkdir -p "$WORK/inc_$schedule"
  cp "$WORK/render_$schedule.h" "$WORK/inc_$schedule/render.h"
  "$BONSAI_CXX" -g -std=c++20 -O3 -I. -I$PREFIX -I"$WORK/inc_$schedule" \
      $PREFIX/render_hook.cpp "$WORK/render_$schedule.o" "${TBB_FLAGS[@]}" \
      -o "$WORK/render_$schedule.out"
done
cat "$COMPILES"

# The cells: the depth matrix, then the sample-count matrix, the shared cell
# once.
CELLS=()
for depth in $DEPTHS; do
  CELLS+=("$depth $DEPTH_SPP")
done
for spp in $SPPS; do
  cell="$SPP_DEPTH $spp"
  if [[ ! " ${CELLS[*]} " == *" $cell "* ]]; then
    CELLS+=("$cell")
  fi
done

# pbrt's side, as compare.sh does it: the scene's `Integrator` directive
# rewritten with the depth (and put in front of a scene that names none, as
# `path`, which is what scene_dump resolves such a scene to), read on standard
# input from the scene's directory; the render time from the EXR's metadata,
# the best of REPEATS runs; the radiance out of the last EXR.
pbrt_scene() {
  local scene="$1" depth="$2" integrator
  integrator=$(sed -n 's/^[[:space:]]*Integrator[[:space:]]*"\([a-z]*\)".*/\1/p' \
               "$scene" | head -1)
  if [[ -z "$integrator" ]]; then
    echo "Integrator \"path\" \"integer maxdepth\" [ $depth ]"
    cat "$scene"
  else
    awk -v depth="$depth" '
      /^[[:space:]]*Integrator[[:space:]]/ {
        gsub(/"integer maxdepth"[[:space:]]*\[[^]]*\]/, "")
        sub(/^[[:space:]]*Integrator[[:space:]]+"[a-z]+"/,
            "&" " \"integer maxdepth\" [ " depth " ]")
        in_integrator = 1
        print
        next
      }
      in_integrator && /^[[:space:]]*"/ {
        if ($0 ~ /"integer maxdepth"/) { next }
        print
        next
      }
      { in_integrator = 0; print }' "$scene"
  fi
}
pbrt_seconds_of() {
  "$IMGTOOL" info "$1" 2>/dev/null |
    sed -n 's/.*(total \([0-9.]*\)s).*/\1/p' | head -1
}
min_seconds() {
  awk -v a="$1" -v b="$2" \
    'BEGIN { if (a == "" || (b != "" && b < a)) print b; else print a }'
}

printf 'scene\tdepth\tspp\tschedule\tseconds\tpbrt_seconds\tspeedup_over_pbrt\tmean_ratio\tagree_percent\tverdict\tsame_image_as\n' \
    > "$RESULTS"
for scene in "${SCENES[@]}"; do
  scene="$(cd "$(dirname "$scene")" && pwd)/$(basename "$scene")"
  base=$(basename "$scene" .pbrt)
  for cell in "${CELLS[@]}"; do
    read -r depth spp <<< "$cell"
    tag="$base-d$depth-s$spp"
    echo "== $base, depth $depth, $spp spp"

    DUMP_FLAGS=(--spp "$spp" --maxdepth "$depth")
    if [[ "$PBRT_TREE" != "0" ]]; then
      DUMP_FLAGS+=(--pbrt-tree)
    fi
    "$WORK/scene_dump" "${DUMP_FLAGS[@]}" "$scene" "$WORK/$tag.txt" \
        > "$WORK/$tag-dump.log" 2>&1

    pbrt_seconds=""
    for _ in $(seq "$REPEATS"); do
      rm -f "$WORK/$tag-pbrt.exr"
      ( cd "$(dirname "$scene")" &&
          pbrt_scene "$scene" "$depth" |
          "$PBRT" --outfile "$WORK/$tag-pbrt.exr" --spp "$spp" \
          > "$WORK/$tag-pbrt.log" 2>&1 ) || true
      if [[ ! -s "$WORK/$tag-pbrt.exr" ]]; then
        echo "pbrt did not write $WORK/$tag-pbrt.exr:" >&2
        cat "$WORK/$tag-pbrt.log" >&2
        exit 1
      fi
      pbrt_seconds=$(min_seconds "$pbrt_seconds" \
                     "$(pbrt_seconds_of "$WORK/$tag-pbrt.exr")")
    done
    "$IMGTOOL" convert --channels R,G,B --outfile "$WORK/$tag-pbrt.pfm" \
        "$WORK/$tag-pbrt.exr" > /dev/null
    echo "   pbrt: ${pbrt_seconds}s"

    first=""
    for schedule in $SCHEDULES; do
      out=$(BONSAI_REPEATS="$REPEATS" "$WORK/render_$schedule.out" \
            "$WORK/$tag.txt" "$WORK/$tag-$schedule.pfm")
      seconds=$(echo "$out" | sed -n 's/^render seconds: //p')
      report=$(python3 $PREFIX/compare_gbuffer.py --radiance-only \
               --radiance "$WORK/$tag-pbrt.pfm" \
               "$WORK/$tag-$schedule-radiance.pfm" \
               --pbrt-seconds "$pbrt_seconds" --bonsai-seconds "$seconds" \
               --repeats "$REPEATS" 2>&1 || true)
      ratio=$(echo "$report" | sed -n 's/.*(\([0-9.infa]*\)x)$/\1/p' | head -1)
      agree=$(echo "$report" | sed -n 's/.*(\([0-9.]*\)%)$/\1/p' | head -1)
      verdict=$(echo "$report" | rg -q '^FAILED' && echo differs || echo ok)
      speedup=$(awk -v p="$pbrt_seconds" -v b="$seconds" \
                'BEGIN { if (b > 0) printf "%.2f", p / b; else print "?" }')
      # Bit for bit against the first schedule's image: the schedules only
      # change how the program runs, so they had better agree exactly, and
      # where they differ by a last bit the report says so.
      same="-"
      if [[ -z "$first" ]]; then
        first="$schedule"
      elif cmp -s "$WORK/$tag-$first-radiance.pfm" \
                  "$WORK/$tag-$schedule-radiance.pfm"; then
        same="$first"
      else
        same="differs"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$base" "$depth" "$spp" "$schedule" "$seconds" "$pbrt_seconds" \
          "$speedup" "${ratio:-?}" "${agree:-?}" "$verdict" "$same" >> "$RESULTS"
      echo "   $schedule: ${seconds}s (${speedup}x pbrt), image ${ratio:-?}x pbrt's mean, ${agree:-?}% close, $verdict${same:+, vs $first: $same}"
    done
  done
done

echo
echo "== compile seconds"
column -t -s $'\t' "$COMPILES"
echo
echo "== results ($RESULTS)"
column -t -s $'\t' "$RESULTS"
