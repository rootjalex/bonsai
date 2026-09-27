# apps/pbrt

A pbrt-v4 renderer written in bonsai: `render.bonsai` and the files it
imports are the program, `schedules/*.bonsai` are the ways of running it, and
`render_hook.cpp` is the C++ driver that reads a converted scene, builds the
acceleration structures and calls the compiled render. `PLAN.md` is the
working record of what was built, measured and decided, and why.

## Schedules

Every schedule file imports one of `schedules/trees/`: `bvh.bonsai` lays the
scene's sets out as pbrt's BVH (a tree the compiler traverses), `optix.bonsai`
as the ray tracing hardware's structures (`OptixTree`, searched by
`trace.bind(RTCore)`). The CPU schedules (`scalar`, `packet`, `perlane`, the
`wavefront*` family) and the software GPU schedules (`gpu`, `gpu-wavefront`)
take the first; `gpu-optix` (pbrt's GPU wavefront on the RT cores),
`gpu-optix-mega` (a megakernel tracing through OptiX) and
`gpu-optix-mega-ser` (the same with shader execution reordering before the
material stage) take the second.

## Scripts

Run from the repository root, inside the `bonsai` conda environment. Each
takes `PBRT=<path to a built pbrt>` (default `~/projects/pbrt-v4/build/pbrt`)
and `BONSAI_BUILD_DIR=<compiler build directory>` (default `build`).

- `render.sh [--spp N] [--schedule S] <scene.pbrt> <out.pfm>`: build the
  renderer with one schedule and render a scene.
- `compare.sh [--spp N] <scene.pbrt>`: one scene, pbrt against one schedule
  (`SCHEDULE=`), both timed, the images checked. `--wavefront` and `--gpu`
  pick which pbrt to compare against.
- `gpu_compare.sh [--spp N] [--repeats N] [--schedules "..."] [--scenes "..."]
  [--out DIR]`: the GPU table. pbrt `--gpu` and every GPU schedule on every
  scene: pbrt's time is the least of its runs, ours the mean of ours, each
  beside its kernel time (pbrt's `--stats` profile and our
  `BONSAI_KERNEL_STATS`), every image checked against pbrt's and written as a
  PNG whose path the table gives. Scenes are `<dir>/<name>` under
  `~/projects/pbrt-v4-scenes` (`SCENES_DIR=`).
- `build_scene_dump.sh <out>`: builds `scene_dump`, which reads a `.pbrt`
  scene with pbrt's own parser and writes it in the driver's format.
- `check_hits.sh`, `check_differentials.sh`: per-hit and per-differential
  probes against pbrt, for finding where a bit differs.

Environment variables the compiled renderer and its runtime read:
`BONSAI_REPEATS` (how many times the driver renders, reporting the best),
`BONSAI_KERNEL_STATS` (a per-kernel GPU time profile at exit),
`BONSAI_OPTIX_VALIDATION` (OptiX's validation mode, for a fault in a
launch), `BONSAI_EXPLAIN_DEVICE` (at compile time: where each buffer is
needed and what each launch captures).
