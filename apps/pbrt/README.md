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
and `BONSAI_BUILD_DIR=<compiler build directory>` (default `build`). A build
tree is brought up to date first; `gpu_compare.sh` also accepts a directory
that only holds a `compiler` binary -- a copy kept of an earlier build -- and
uses it as it is, which is how a change is measured against the compiler
before it.

- `render.sh [--spp N] [--schedule S] <scene.pbrt> <out.pfm>`: build the
  renderer with one schedule and render a scene.
- `compare.sh [--spp N] <scene.pbrt>`: one scene, pbrt against one schedule
  (`SCHEDULE=`), both timed, the images checked. `--wavefront` and `--gpu`
  pick which pbrt to compare against. Both this and `gpu_compare.sh` hand
  pbrt its scene with the Film told `"bool savefp16" false` unless the
  scene says something itself: pbrt's film quantizes its pixels to halves
  before writing any file, a PFM included, and a float render checked
  against a half-quantized one loses its last digit of agreement and, on
  an emissive medium, counts every pixel lit by 1e-20 here and rounded to
  black there as a pixel lit on one side only.
- `gpu_compare.sh [--spp N] [--repeats N] [--schedules "..."] [--scenes "..."]
  [--out DIR] [--resume]`: the GPU table. pbrt `--gpu` and every GPU
  schedule on every scene, no number ever one run's: pbrt's time is the
  least of three processes, ours the least of three renders in one process
  (the first is the warm-up, absorbing module loading and the GPU's climb
  from the idle clocks every process starts at), each beside its kernel
  time (pbrt's `--stats` profile; our `BONSAI_KERNEL_STATS` profile of the
  fastest of the three renders), every image checked against pbrt's and
  written as a PNG whose path the table gives.
  Scenes are `<dir>/<name>` under `~/projects/pbrt-v4-scenes`
  (`SCENES_DIR=`). Every timed run waits for the machine to be idle and is
  watched while it runs (anything else on the GPU, or another user's
  process at the top of the CPU); a disturbed run is redone, three tries
  at most, and a set of our runs more than a quarter apart is redone as a
  set. The table's `note` column says how many runs a cell had redone and
  how many it kept disturbed. `--resume` continues a run into the same
  `--out` directory, reusing its binaries and skipping the scenes whose
  rows are all in the table. A file named `PAUSE` in the output directory
  holds the run between timed runs for as long as it exists, which is how
  to borrow the machine for a measurement of one's own. `--spp` takes one
  count or several (`--spp "16 64 256"`): every scene at each, one table.
- `plot_gpu_table.py OUT.png TABLE.tsv...`: the tables gpu_compare.sh
  wrote, plotted -- each schedule's speedup over `pbrt --gpu` and every
  side's kernel time, a group of bars per scene and a row of panels per
  table (sample count); a bar whose image did not match pbrt's is hatched.
- `plot_gpu_heatmap.py OUT.png TABLE.tsv... [--schedule S] [--sort
  table|name|speedup]`: the same tables as a heatmap -- scenes across,
  sample counts down, each cell one schedule's speedup over `pbrt --gpu`
  (default `gpu-optix`), coloured on a log scale centred on parity with the
  ratio printed in the cell, the GraphIt paper's figure for a schedule
  against its baselines. A cell whose image did not match pbrt's is hatched
  with its number struck through; a cell no table has is blank. Several
  counts come from `gpu_compare.sh --spp "16 64 256"` into one table.
- `build_scene_dump.sh <out>`: builds `scene_dump`, which reads a `.pbrt`
  scene with pbrt's own parser and writes it in the driver's format: a text
  file for the small parts (camera, materials, lights, textures, spectra)
  and binary sidecars beside it -- `.geo` for the geometry (meshes,
  vertices, shapes, trees, instances; a FlatBuffer, `scene_geometry.fbs`),
  `.tex` for texels, `.env` for environment maps, `.pl` for measured BRDFs,
  `.vol` for the grid media's voxels, `.smp` for the tables a `sobol` or
  `pmj02bn` sampler reads (pbrt's full Sobol' matrices, its blue-noise
  point sets and textures, copied from pbrt's own arrays), `.vdb` for the
  `nanovdb` media's grid buffers, copied out of their `.nvdb` files as
  they are and read by NanoVDB's own accessor (below). `--gpu` converts
  the scene as `pbrt --gpu` builds it where that differs from pbrt's CPU:
  a PLY's quads become two triangles each (pbrt's OptiX aggregate splits
  them so the hardware traces them; its CPU keeps bilinear patches), which
  `gpu_compare.sh` passes, since `pbrt --gpu` is its reference.
- `build_nanovdb_shim.sh <outdir> [sm_NN]`: builds `nanovdb_shim.cpp`, the
  implementation of the renderer's two foreign functions over NanoVDB
  (`docs/foreign-functions.md`), to host bitcode and, given a GPU
  architecture, to PTX with nvcc, and prints the `--link` flags that fold
  them into the generated module so that a voxel read is NanoVDB's accessor
  inlined rather than a call. The scripts above run it and also compile the
  shim into the driver against the generated header, where a prototype
  mismatch is a compile error.
- `scene_schema.sh`: runs `flatc` on `scene_geometry.fbs` and prints the
  compiler flags the scene reader needs; every script that compiles
  `scene_dump.cpp` or `render_hook.cpp` calls it. Needs the `flatbuffers`
  package of the conda environment.
- `check_hits.sh`, `check_differentials.sh`: per-hit and per-differential
  probes against pbrt, for finding where a bit differs.

Environment variables the compiled renderer and its runtime read:
`BONSAI_REPEATS` (how many times the driver renders, reporting the best),
`BONSAI_KERNEL_STATS` (a per-kernel GPU time profile at exit),
`BONSAI_OPTIX_VALIDATION` (OptiX's validation mode, for a fault in a
launch), `BONSAI_EXPLAIN_DEVICE` (at compile time: where each buffer is
needed and what each launch captures).
