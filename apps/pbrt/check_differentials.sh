#!/usr/bin/env bash
# The numbers a texture is filtered by, this renderer's against pbrt's.
#
# Worth its own script because a wrong footprint is invisible in an image. It
# moves no edge and changes no colour -- it only makes a texture slightly too
# blurry or too sharp -- so no pixel comparison against pbrt would fail on it.
# Nothing here renders anything.
#
# Both sides print the same four camera rays and the same synthetic hit, and
# both branches of ComputeDifferentials at it: the one that uses the ray's own
# differentials, which only a first hit has, and the camera approximation, which
# is what every hit after a diffuse bounce falls back on.
#
# Usage: bash apps/pbrt/check_differentials.sh [scene.pbrt]
set -euo pipefail

PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PREFIX/../.." && pwd)"
SCENE="${1:-$PREFIX/scenes/three-spheres.pbrt}"
WORK="$PREFIX/compare-out"
mkdir -p "$WORK"
cd "$ROOT"

BONSAI_CXX="${BONSAI_CXX:-clang++}"
TBB_PREFIX="$(dirname "$(dirname "$(command -v "$BONSAI_CXX")")")"
TBB_FLAGS=()
if [[ -f "$TBB_PREFIX/include/tbb/parallel_for.h" ]]; then
  TBB_FLAGS=(-I"$TBB_PREFIX/include" -L"$TBB_PREFIX/lib"
             -Wl,-rpath,"$TBB_PREFIX/lib" -ltbb)
fi

bash "$PREFIX/build_scene_dump.sh" "$WORK/scene_dump"
./build/compiler -p ssa --no-heap --ffp-contract \
    -i "$PREFIX/render.bonsai" -b cpp -o "$PREFIX/render"
"$BONSAI_CXX" -g -std=c++20 -O3 -I. -I"$PREFIX" "$PREFIX/render_hook.cpp" \
    "$PREFIX/render.o" ${TBB_FLAGS[@]+"${TBB_FLAGS[@]}"} -o "$WORK/render.out"

"$WORK/scene_dump" --print-differentials "$SCENE" "$WORK/diff-scene.txt" \
    | grep -E '^(camdiff|dudxy|spawn) ' > "$WORK/pbrt-differentials.txt"
"$WORK/render.out" --print-differentials "$WORK/diff-scene.txt" \
    > "$WORK/bonsai-differentials.txt"

# Compared with a tolerance, and the tolerance is argued rather than picked.
#
# `Approximate_dp_dxy` ends in `px - pDownZ`, a difference of two vectors of
# magnitude |p_camera| -- about 3.8 for this hit -- that agree to within 1e-4.
# That subtraction cancels about 4.6 decimal digits, and float32 starts with
# 7.2, so roughly 2.6 remain: a relative uncertainty near 5e-3 whatever either
# side does. pbrt's own answer has no more digits than that. Bit-exactness is
# therefore not available here and not worth chasing, and what is checked is
# that the disagreement stays at that floor rather than growing past it.
#
# The well-conditioned lines are checked exactly, and they are the ones that
# say the transcription is right: all four camera rays and their scaled
# differentials come out bit for bit.
python3 - "$WORK/pbrt-differentials.txt" "$WORK/bonsai-differentials.txt" <<'PY'
import sys

def read(path):
    rows = {}
    for line in open(path):
        key, _, rest = line.partition(':')
        rows[key.strip()] = [float(x) for x in rest.replace('|', ' ').split()]
    return rows

a, b = read(sys.argv[1]), read(sys.argv[2])
if a.keys() != b.keys():
    sys.exit('the two sides printed different rows')

TOL = 5e-3
worst, worst_key, exact, bad = 0.0, '', 0, []
for key in a:
    if a[key] == b[key]:
        exact += 1
    for i, (x, y) in enumerate(zip(a[key], b[key])):
        if x == y:
            continue
        rel = abs(x - y) / max(abs(x), abs(y))
        if rel > worst:
            worst, worst_key = rel, '%s field %d' % (key, i)
        if rel > TOL:
            bad.append('%s field %d: pbrt %.9g, here %.9g (%.1e)'
                       % (key, i, x, y, rel))
    # The camera's own differentials are not the output of a cancelling
    # subtraction, so there is no reason for them to differ at all.
    if key.startswith('camdiff') and a[key] != b[key]:
        bad.append('%s differs, and a camera ray should be exact' % key)

print('%d of %d rows bit-exact; worst relative difference %.2e (%s)'
      % (exact, len(a), worst, worst_key))
if bad:
    print('\n'.join(bad))
    sys.exit('differentials disagree by more than float32 leaves here')
print('ok: matches pbrt to what float32 retains through the cancellation')
PY
