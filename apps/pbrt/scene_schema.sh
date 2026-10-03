#!/bin/bash
# Generate the C++ header for the scene's geometry sidecar from its FlatBuffers
# schema (scene_geometry.fbs), and print the compiler flags a program that
# includes scene_io.h needs because of it.
#
#     "$CXX" ... $(bash apps/pbrt/scene_schema.sh) ... scene_dump.cpp
#
# The header goes beside the schema, as render.h goes beside render.bonsai,
# and is not committed (.gitignore): it is a build product of a checked-in
# source, regenerated here on every build, which is a second a time. The
# flags are the FlatBuffers include directory -- found from `flatc` itself,
# so the headers are the version the schema was compiled with -- and nothing
# else, since every caller already passes -I apps/pbrt.
#
# flatc comes with the `flatbuffers` package of the conda environment
# (conda-forge, 25.x; the 64-bit vectors the schema uses need 23.5 or later);
# FLATC names another.
set -e
PREFIX="$(dirname "$0")"
FLATC="${FLATC:-flatc}"
if ! command -v "$FLATC" > /dev/null; then
  echo "scene_schema.sh: no flatc on the path; conda install -c conda-forge flatbuffers" >&2
  exit 1
fi
"$FLATC" --cpp --cpp-std c++17 -o "$PREFIX" "$PREFIX/scene_geometry.fbs" >&2
INCLUDE="$(cd "$(dirname "$(command -v "$FLATC")")/../include" && pwd)"
echo "-isystem $INCLUDE"
