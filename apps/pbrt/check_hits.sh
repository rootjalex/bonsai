#!/usr/bin/env bash
# The camera ray of a pixel sample and the distance to what it hits, this
# renderer's against pbrt's own -- bit for bit, for every eighth pixel each
# way and the first four samples of each.
#
# Worth its own check because of what pbrt's wavefront does with these three
# numbers: it seeds a medium's random stream from them, `RNG rng(Hash(ray.o,
# tMax), Hash(ray.d))` (wavefront/media.cpp), and draws every distance and
# every absorb / scatter / null choice along the ray from that stream. A last
# bit's difference in the origin, the direction or the hit distance is a
# different stream for that ray and, some of the time, a different choice --
# which an image comparison sees only as a few pixels' albedo off by a
# sample's worth, and cannot attribute. This prints the inputs and says which
# one it was. Nothing here renders anything.
#
# `hit` rows, each printed by both sides in the same format: the pixel, the
# sample, the ray's origin, its direction, the hit distance (`inf` for a
# miss), and the camera sample the ray came from -- film point, lens point,
# time -- so a direction that differs is laid at the sampler's door or the
# camera's. pbrt's come from its sampler, camera and aggregate as its own render
# builds them (`scene_dump --print-hits`); this renderer's from `hit_at`
# (render.bonsai), which starts the sampler on the pixel and sample and draws
# as `render` draws.
#
# Usage: bash apps/pbrt/check_hits.sh [scene.pbrt] [schedule]
set -euo pipefail

PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PREFIX/../.." && pwd)"
SCENE="${1:-$PREFIX/scenes/camera-medium.pbrt}"
# Absolute, since scene_dump is run from the scene's own directory below.
SCENE="$(cd "$(dirname "$SCENE")" && pwd)/$(basename "$SCENE")"
SCHEDULE="${2:-scalar}"
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
# The compiler from the build directory compare.sh uses, and its flags: the
# contraction of a multiply and an add into one rounding is part of the
# answer, and pbrt's CPU build contracts.
BONSAI_BUILD_DIR="${BONSAI_BUILD_DIR:-build}"
"./$BONSAI_BUILD_DIR/compiler" -p ssa --no-heap --ffp-contract \
    -i "$PREFIX/render.bonsai" -i "$PREFIX/schedules/$SCHEDULE.bonsai" \
    -b cpp -o "$PREFIX/render"
"$BONSAI_CXX" -g -std=c++20 -O3 -I. -I"$PREFIX" "$PREFIX/render_hook.cpp" \
    "$PREFIX/render.o" ${TBB_FLAGS[@]+"${TBB_FLAGS[@]}"} -o "$WORK/render.out"

# From the scene's directory, so its own `Include`s resolve.
ROWS='^(hit|filtf|condcdf|margfunc|margcdf) |^filtertables:'
(cd "$(dirname "$SCENE")" &&
 "$WORK/scene_dump" --print-hits "$SCENE" "$WORK/hits-scene.txt") \
    | grep -E "$ROWS" > "$WORK/pbrt-hits.txt"
"$WORK/render.out" --print-hits "$WORK/hits-scene.txt" \
    | grep -E "$ROWS" > "$WORK/bonsai-hits.txt"

# `%.9g` round-trips a float exactly, and so does pbrt's FloatToString, so
# equal text is equal bits and a plain comparison is the test. The report
# says which quantity disagreed, and how often, which is what the reader
# needs next.
#
# pbrt's filter tables arrive as one `filtertables` line, its FilterSampler's
# ToString: the tabulated filter with every value round-trip exact, and each
# PiecewiseConstant1D's integral likewise -- but the CDFs inside to six
# digits only, which cannot be compared. They are not needed: a CDF is a
# running sum of its table's values and a division by its integral, so with
# the values and the integrals equal to the bit and the arithmetic pbrt's,
# the CDFs are too, and the `hit` rows check the whole chain end to end.
python3 - "$WORK/pbrt-hits.txt" "$WORK/bonsai-hits.txt" <<'PY'
import re
import struct
import sys

NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

def as_float32(token):
    # Both sides print floats that round-trip -- `%.9g` here, the shortest
    # form that does on pbrt's -- so the same value can be spelt two ways,
    # and the comparison is of the float each spelling names.
    try:
        return struct.pack('<f', float(token))
    except (ValueError, OverflowError):
        return token

def read(path):
    rows = {}
    for line in open(path):
        key, _, rest = line.partition(':')
        if key.startswith('filtertables'):
            rows[key.strip()] = rest.split()
        else:
            rows[key.strip()] = [as_float32(token) for token in rest.split()]
    return rows

def tables_of(printed):
    # `f: [ Array2D extent: [ [ 0, 0 ] - [ nx, ny ] ] values: [ [ ... ], ... ] ]`
    # rows of x for each y, then one `funcInt:` per conditional row and a
    # last one for the marginal.
    rows = {}
    extent = re.search(r'extent: \[ \[ 0, 0 \] - \[ (\d+), (\d+) \] \]', printed)
    values = re.search(r'values: \[(.*?)\] \] distrib:', printed)
    if not extent or not values:
        sys.exit('could not read the filter tables out of pbrt\'s ToString')
    nx, ny = int(extent.group(1)), int(extent.group(2))
    numbers = re.findall(NUMBER, values.group(1))
    if len(numbers) != nx * ny:
        sys.exit('pbrt printed %d filter values for a %dx%d table'
                 % (len(numbers), nx, ny))
    for y in range(ny):
        for x in range(nx):
            rows['filtf %d %d' % (x, y)] = [as_float32(numbers[y * nx + x])]
    integrals = re.findall(r'funcInt: (' + NUMBER + ')', printed)
    if len(integrals) != ny + 1:
        sys.exit('pbrt printed %d integrals for %d rows' % (len(integrals), ny))
    for y in range(ny):
        rows['margfunc %d' % y] = [as_float32(integrals[y])]
    return rows

def spelt(tokens):
    # A row's values back as text, for the report.
    return ' '.join('%.9g' % struct.unpack('<f', t)[0] if isinstance(t, bytes)
                    else t for t in tokens)

a, b = read(sys.argv[1]), read(sys.argv[2])
if 'filtertables' in a:
    a.update(tables_of(' '.join(a.pop('filtertables'))))
# What pbrt did not print at comparable precision is not compared.
b = {key: value for key, value in b.items()
     if not key.startswith(('condcdf', 'margcdf'))}
if a.keys() != b.keys():
    only_a = sorted(set(a) - set(b))[:5]
    only_b = sorted(set(b) - set(a))[:5]
    sys.exit('the two sides printed different rows: pbrt only %s; here only %s'
             % (only_a, only_b))
if not a:
    sys.exit('no hit rows printed')

names = ['origin', 'direction', 'distance', 'film point', 'lens point',
         'time', 'pixel draw', 'filter table']
wrong = {name: 0 for name in names}
examples = []
for key in a:
    if a[key] == b[key]:
        continue
    if not key.startswith('hit '):
        wrong['filter table'] += 1
    else:
        # origin | direction | distance | film | lens | time | draw: which
        # fields of the row differ.
        fields = {'origin': a[key][0:3] != b[key][0:3],
                  'direction': a[key][4:7] != b[key][4:7],
                  'distance': a[key][8:9] != b[key][8:9],
                  'film point': a[key][10:12] != b[key][10:12],
                  'lens point': a[key][13:15] != b[key][13:15],
                  'time': a[key][16:17] != b[key][16:17],
                  'pixel draw': a[key][18:20] != b[key][18:20]}
        for name, differs in fields.items():
            if differs:
                wrong[name] += 1
    if len(examples) < 8:
        examples.append('%s\n  pbrt %s\n  here %s'
                        % (key, spelt(a[key]), spelt(b[key])))

total = len(a)
bad = sum(1 for key in a if a[key] != b[key])
print('%d of %d rows bit-exact (%d hit rows, %d filter table entries)'
      % (total - bad, total, sum(1 for k in a if k.startswith('hit ')),
         sum(1 for k in a if not k.startswith('hit '))))
if bad:
    print('differing rows: ' + ', '.join('%s in %d' % (name, wrong[name])
                                          for name in names if wrong[name]))
    print('\n'.join(examples))
    sys.exit('camera rays or hit distances differ from pbrt')
print('ok: every camera ray and hit distance is pbrt\'s to the bit')
PY
