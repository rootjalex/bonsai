#!/bin/bash

set -euo pipefail

# Build apps/pbrt/scene_dump, the tool that reads a .pbrt scene with PBRT's own
# parser. See the comment at the top of scene_dump.cpp for why it is a separate
# program rather than part of the renderer's driver.
#
# Every flag comes out of PBRT's own build tree rather than being written down
# here. That is not tidiness: PBRT's headers only agree with its library when
# they see the same defines it was compiled with -- `-DPBRT_BUILD_GPU_RENDERER`
# in particular changes struct layouts -- and a program that guesses them
# compiles, links, and then corrupts memory at a distance. Reading them from
# flags.make means they cannot drift.
#
# Point PBRT at a built pbrt if it is somewhere else:
#
#     PBRT=~/src/pbrt-v4/build/pbrt apps/pbrt/build_scene_dump.sh

if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi

PREFIX="apps/pbrt"
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"
PBRT_BUILD="$(dirname "$PBRT")"
OUT="${1:-$PREFIX/scene_dump}"

if [[ ! -x "$PBRT" ]]; then
  echo "no pbrt at $PBRT -- set PBRT to a built one" >&2
  exit 1
fi

FLAGS="$PBRT_BUILD/CMakeFiles/pbrt_lib.dir/flags.make"
LINK="$PBRT_BUILD/CMakeFiles/pbrt_exe.dir/link.txt"
for f in "$FLAGS" "$LINK"; do
  if [[ ! -f "$f" ]]; then
    echo "cannot read $f -- is $PBRT_BUILD a pbrt cmake build directory?" >&2
    exit 1
  fi
done

# The compiler pbrt was built with. Its headers do not compile with an
# arbitrary one, and objects from two toolchains should not meet in a link.
CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$PBRT_BUILD/CMakeCache.txt")
if [[ ! -x "$CXX" ]]; then
  echo "cannot find the compiler pbrt was built with ($CXX)" >&2
  exit 1
fi

# Defines and include paths, verbatim from the library's own compile flags.
CXX_DEFS=$(sed -n 's/^CXX_DEFINES = //p' "$FLAGS")
CXX_INCS=$(sed -n 's/^CXX_INCLUDES = //p' "$FLAGS")
# -march=native and friends are in here too, which is what pbrt's own
# translation units were compiled with.
CXX_OPTS=$(sed -n 's/^CXX_FLAGS = //p' "$FLAGS")

# The link line, minus pbrt's own main and the output it names. What is left is
# the set of libraries pbrt needs, in the order it needs them.
LINK_ARGS=$(tr ' ' '\n' < "$LINK" |
  grep -v '^CMakeFiles/pbrt_exe' |
  grep -v '^-o$' |
  grep -v '^pbrt$' |
  tail -n +2 |
  tr '\n' ' ')

# libcuda lives with the driver rather than in the conda prefix, and pbrt's
# link line reaches it through a -L that cmake supplies separately.
STUBS="$(dirname "$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$PBRT_BUILD/CMakeCache.txt")")/../lib/stubs"

# shellcheck disable=SC2086
"$CXX" $CXX_OPTS $CXX_DEFS $CXX_INCS -I"$PREFIX" \
    -c "$PREFIX/scene_dump.cpp" -o "$PREFIX/scene_dump.o"

# The link runs from pbrt's build directory, because the library paths in its
# link line are relative to it. So everything of ours named on that command has
# to be absolute first -- including an output path the caller may already have
# given as absolute, which is why this resolves rather than prepends.
OBJ_ABS="$(realpath "$PREFIX/scene_dump.o")"
OUT_ABS="$(realpath -m "$OUT")"
mkdir -p "$(dirname "$OUT_ABS")"

# shellcheck disable=SC2086
(cd "$PBRT_BUILD" && "$CXX" -o "$OUT_ABS" "$OBJ_ABS" -L"$STUBS" $LINK_ARGS)

rm -f "$PREFIX/scene_dump.o"
echo "built $OUT_ABS"
