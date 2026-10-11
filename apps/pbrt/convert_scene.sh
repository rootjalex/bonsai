#!/usr/bin/env bash
# One conversion per scene, shared by every sweep and every sample count.
#
#     apps/pbrt/convert_scene.sh <scene_dump> <dir/name> [scene_dump options]
#
# Prints the path of the scene converted for this renderer,
# apps/pbrt/scenes/converted/<dir>/<name>.txt with its sidecars beside it,
# converting first when it is missing or stale. A conversion is a function of
# the converter's sources (scene_dump.cpp and the headers and schema it is
# built from), of the pbrt it is linked against, of the options, and of the
# scene file; a stamp beside the conversion (`.txt.stamp`) records all of
# them, and a conversion whose stamp is not today's is redone. (The files a
# scene includes -- its geometry, its PLYs, its textures -- are not in the
# stamp; a scene set edited in place wants the conversion removed by hand.)
#
# The sample count and the path depth are not among the options: a
# conversion carries the scene's own, and the renderer's `--spp` and
# `--maxdepth` change them at render time (render_hook.cpp, sampler_at:
# pbrt's override transcribed, each sampler overridden the way its own Create
# overrides it) as pbrt's own `--spp` does on the other side -- so one
# conversion serves a sweep's every count. The converter's output goes to
# `.txt.log` beside the conversion; a refusal leaves no stamp, so the next
# call converts again.
#
# Scenes are `<dir>/<name>` under $SCENES_DIR (~/projects/pbrt-v4-scenes), as
# gpu_compare.sh names them; PBRT names the pbrt the converter was linked
# against (build_scene_dump.sh's default). The folder is data, not source
# (.gitignore): 37 GB for the thirty-two scenes of the heatmap's list,
# kroken's displaced triangles ten of them, regenerated from pbrt-v4-scenes
# in a few minutes (kroken's the slowest at about four).
#
# Until 2026-10-11 every cell of a sweep converted its scene again -- three
# sample counts, three copies of kroken's eleven gigabytes of displaced
# triangles -- and the copies left by thirty sweeps filled the disk.
set -euo pipefail
if [[ "$(pwd)" == */apps/pbrt ]]; then
  cd ../..
fi
PREFIX="apps/pbrt"
if [[ $# -lt 2 ]]; then
  sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
  exit 1
fi
DUMP="$1"
SCENE="$2"
shift 2
OPTS=("$@")
SCENES_DIR="${SCENES_DIR:-$HOME/projects/pbrt-v4-scenes}"
PBRT="${PBRT:-$HOME/projects/pbrt-v4/build/pbrt}"

dir="$SCENES_DIR/$(dirname "$SCENE")"
name="$(basename "$SCENE")"
src="$dir/$name.pbrt"
if [[ ! -x "$DUMP" ]]; then
  echo "convert_scene.sh: no converter at $DUMP" >&2; exit 1
fi
# Absolute, since the converter runs from the scene's directory below.
DUMP="$(realpath "$DUMP")"
if [[ ! -f "$src" ]]; then
  echo "convert_scene.sh: no scene $src" >&2; exit 1
fi
if [[ ! -f "$PBRT" ]]; then
  echo "convert_scene.sh: no pbrt at $PBRT (PBRT names it)" >&2; exit 1
fi

out_dir="$PREFIX/scenes/converted/$(dirname "$SCENE")"
mkdir -p "$out_dir"
out="$(cd "$out_dir" && pwd)/$name.txt"

# The stamp: everything the conversion is a function of, hashed together.
key=$( {
  cat "$PREFIX/scene_dump.cpp" "$PREFIX/scene_io.h" "$PREFIX/measured_io.h" \
      "$PREFIX/scene_geometry.fbs" "$PREFIX/cie_tables.h" \
      "$PREFIX/sobol_tables.h" "$PREFIX/srgb_table.h"
  stat -c '%s %Y' "$PBRT" "$src"
  echo "${OPTS[*]-}"
} | sha256sum | cut -d' ' -f1)

if [[ -f "$out" && -f "$out.stamp" && "$(cat "$out.stamp")" == "$key" ]]; then
  echo "$out"
  exit 0
fi

rm -f "$out.stamp"
# From the scene's own directory, as pbrt reads it: its includes are relative.
if ! (cd "$dir" && "$DUMP" ${OPTS[@]+"${OPTS[@]}"} "$name.pbrt" "$out" > "$out.log" 2>&1); then
  echo "convert_scene.sh: scene_dump refused $SCENE: $(tail -1 "$out.log")" >&2
  exit 1
fi
echo "$key" > "$out.stamp"
echo "$out"
