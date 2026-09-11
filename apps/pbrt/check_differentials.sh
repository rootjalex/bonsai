#!/usr/bin/env bash
# The numbers a texture is filtered by, and the numbers it filters to, this
# renderer's against pbrt's -- bit for bit where nothing cancels.
#
# Worth its own script because a wrong footprint is invisible in an image. It
# moves no edge and changes no colour -- it only makes a texture slightly too
# blurry or too sharp -- so no pixel comparison against pbrt would fail on it.
# Until it is differenced by a bump map: then an ulp in the footprint is a
# different normal, which is how most of the rows below were earned. Nothing
# here renders anything.
#
# The rows, each printed by both sides in the same format so a plain diff is
# the comparison:
#
#   camray, camdiff   the camera ray for four pixels, and its two companions,
#                     through an off-centre lens sample so a camera with a
#                     lens takes its lens branch in earnest
#   invpoint, invvec  the camera transform applied backwards to a point and a
#                     vector -- pbrt's ApplyInverse, which associates its sums
#                     differently from the forward transform
#   dudxy             both branches of ComputeDifferentials at a synthetic hit:
#                     the ray's own differentials, which only a first hit has,
#                     and the camera approximation every hit after a diffuse
#                     bounce falls back on
#   spawn             what a specular bounce leaves of the differentials, and
#                     that a rough one drops them
#   lambda            the four wavelengths the spectrum rows are sampled at
#   texrgb            every texture the scene converted, filtered through
#                     pbrt's own MIPMap at three points and three footprints
#   texf, texs        the same lookups through pbrt's texture objects, read as
#                     a float or as a spectrum according to how the scene
#                     declared them; texsr is pbrt's spectrum recomposed from
#                     texrgb, which is what told the fold of a constant `scale`
#                     texture apart from a scale of the spectrum
#   sig               an RGB fitted to a spectrum by table lookup and sampled,
#                     for near-grey colours where the table changes fastest
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
    | grep -E '^(camray|camdiff|invpoint|invvec|dudxy|spawn|texrgb|texf|texs|texsr|sig) |^lambda:' \
    > "$WORK/pbrt-differentials.txt"
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
# A texture is read as a float or as a spectrum according to how the scene
# declared it, which pbrt knows and this renderer's scene file does not carry;
# so this side prints both readings of every texture and pbrt's decides which
# one is compared. Every other kind of row has to appear on both sides.
b = {k: v for k, v in b.items() if not k.startswith('tex') or k in a}
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
    # The camera's own rays and differentials, and its transform applied
    # backwards, are not the output of a cancelling subtraction, so there is
    # no reason for them to differ at all. Nor is a texture lookup: it is
    # pbrt's arithmetic over pbrt's own pyramid, and a bump map differences
    # two of them.
    if (key.startswith(('camray', 'camdiff', 'invpoint', 'invvec', 'tex',
                        'sig', 'lambda'))
            and a[key] != b[key]):
        bad.append('%s differs, and it should be exact' % key)

print('%d of %d rows bit-exact; worst relative difference %.2e (%s)'
      % (exact, len(a), worst, worst_key))
if bad:
    print('\n'.join(bad))
    sys.exit('differentials disagree by more than float32 leaves here')
print('ok: matches pbrt to what float32 retains through the cancellation')
PY
