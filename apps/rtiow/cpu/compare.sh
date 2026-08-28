#!/bin/bash
#
# Renders the Ray Tracing in One Weekend scene twice -- once through the
# Stmt-level pipeline and once through the SSA one (`-p ssa`) -- reports how
# long each took, and checks that they drew the same image.
#
# The SSA pipeline is where `loopify` is an SSA rewrite: `sample` is a
# value-returning tail recursion turned into a loop, and `trace` is a
# branching tree traversal turned into a loop over an explicit stack. So this
# is the end-to-end check that those transforms preserve the picture.
#
#   ./apps/rtiow/cpu/compare.sh                 # the full 1200x675, 50 spp
#   RTIOW_WIDTH=200 RTIOW_SAMPLES=4 ./apps/rtiow/cpu/compare.sh   # quick
#
set -euo pipefail

# Runnable from the repository root or from this directory.
if [[ "$(pwd)" == */apps/rtiow/cpu ]]; then
  cd ../../..
fi

PREFIX="apps/rtiow/cpu"
OUT="${OUT:-$PREFIX/compare-out}"
# From scratch each time: a half-written image left by an interrupted run
# would otherwise be compared against a fresh one.
rm -rf "$OUT"
mkdir -p "$OUT"

# The checked-in schedule splits the image loop across threads with
# `cpu_thread`, which generates calls to libdispatch -- a macOS library, so it
# does not link elsewhere -- and which the SSA pipeline does not implement
# anyway. Both sides are built from a copy without it, so that what is being
# compared is the two pipelines and not two different schedules.
SRC="$OUT/scene.bonsai"
sed '/image.split/,/cpu_thread/d' "$PREFIX/main.bonsai" > "$SRC"

build() { # <label> <extra compiler args...>
  local label="$1"; shift
  # The header goes where the driver includes it from, which is beside the
  # driver: a quoted include is looked for next to the including file first,
  # so writing it anywhere else would leave whatever is there shadowing it.
  # Both pipelines produce the same header, and it is the checked-in one --
  # regenerating it is how it stays current.
  ./build/compiler -i "$SRC" -b cpp -o "$PREFIX/main" "$@"
  mv "$PREFIX/main.o" "$OUT/$label.o"
  clang++ -std=c++20 -O2 -I. \
      "$PREFIX/main_hook.cpp" "$OUT/$label.o" -o "$OUT/$label.out"
}

echo "building..."
build baseline
build ssa -p ssa

for label in baseline ssa; do
  echo
  echo "=== $label ==="
  # The driver prints its own setup/render/write breakdown.
  "$OUT/$label.out" "$OUT/$label.ppm"
done

echo
if cmp -s "$OUT/baseline.ppm" "$OUT/ssa.ppm"; then
  echo "images are identical"
else
  echo "IMAGES DIFFER -- $(cmp -l "$OUT/baseline.ppm" "$OUT/ssa.ppm" | wc -l) bytes"
fi

# PNGs, if there is anything around to make them with.
if command -v convert > /dev/null; then
  for label in baseline ssa; do
    convert "$OUT/$label.ppm" "$OUT/$label.png"
  done
  echo "wrote $OUT/baseline.png and $OUT/ssa.png"
else
  echo "wrote $OUT/baseline.ppm and $OUT/ssa.ppm"
fi
