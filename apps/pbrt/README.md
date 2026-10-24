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
`gpu-optix-sort` (the same with the ray queue sorted by the two-point
Morton key before each trace, the device sort of
runtime/bonsai_cuda_sort.cu), `gpu-optix-mega` (a megakernel tracing
through OptiX) and `gpu-optix-mega-ser` (the same with shader execution
reordering before the material stage) take the second.

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
  black there as a pixel lit on one side only. `--build-renderer <path>`
  compiles the schedule's renderer to a file and stops; `--renderer <path>`
  compares with that file instead of compiling one, which is how a sweep
  compiles each schedule once (the renderer does not depend on the scene).
- `gpu_clock.sh`: sourced by `gpu_compare.sh` and `heatmap.sh`; reads the
  idle GPU's clocks before a run and the table's last line says whether
  they were locked or not. An unlocked GeForce boosts by load and a
  speedup moves with it, so timing at a locked clock is the sound way
  (`--clock N`, or `BONSAI_GPU_CLOCK_REQUIRED=1` for the default 2400 MHz
  SM with the memory clock at its maximum, refuse to run without the
  lock); the default since 2026-10-10 records the clock and runs either
  way, since the card is shared with another project's benchmarks that
  must not run locked. Locking takes root: `sudo nvidia-smi -lgc 2400,2400
  && sudo nvidia-smi -lmc 14001,14001`; `-rgc`/`-rmc` release.
- `gpu_compare.sh [--spp N] [--repeats N] [--schedules "..."] [--scenes "..."]
  [--out DIR] [--resume] [--clock MHZ]`: the GPU table. pbrt `--gpu` and every GPU
  schedule on every scene, no number ever one run's: pbrt's time is the
  least of three processes, ours the least of three renders in one process
  (the first is the warm-up, absorbing module loading and the GPU's climb
  from the idle clocks every process starts at), each beside its kernel
  time (pbrt's `--stats` profile; our `BONSAI_KERNEL_STATS` profile of the
  fastest of the three renders), every image checked against pbrt's and
  written as a PNG whose path the table gives. A cell whose image fails
  the check is held against pbrt's own spread: the scene rendered by
  pbrt's CPU integrator, once per cell and untimed (`<cell>.pbrt-cpu-
  radiance.pfm`), and the cell passes -- `ok: within pbrt's own
  disagreement` -- when pbrt's two renderers disagree with each other more
  than ours disagrees with the GPU one, on each check that failed (the
  user's rule of 2026-10-18; `compare_gbuffer.py --pbrt-spread`). A pixel
  of ours that is not a number (NaN or infinite) -- pbrt's film admits
  none -- is left out of the mean rather than making it NaN (which passed
  every tolerance once) and counted as one lit-pixel disagreement against
  the lit-pixel allowance, named in the verdict either way (the user,
  10-24: a cell that differs from pbrt by less than pbrt's own two
  renderers differ is not marked; a handful of NaN pixels in two million
  is not, thousands are). `recompare.py OUT_DIR` re-derives a table's
  `agree` and `verdict` columns from the images a run kept, under the
  rules as they stand, without re-rendering (the old table is kept as
  `table.tsv.before-recompare`).
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
- `cpu_compare.sh [--spp "N ..."] [--repeats N] [--schedules "..."]
  [--scenes "..."] [--out DIR] [--resume]`: the CPU comparison -- pbrt's
  CPU wavefront against every CPU schedule, on the scenes given (five of
  rising cost by default) at the sample counts given, each schedule's
  renderer compiled once up front (`DIR/renderers`), then one compare.sh run
  per schedule per cell, `scalar` against pbrt's volpath and the wavefront
  schedules against `pbrt --wavefront`, pbrt rendered once per cell and
  kept for the cell's schedules (`compare.sh --pbrt-cache`). Writes
  `DIR/cpu-results.csv` (a row per scene, spp and side: seconds, speedup,
  verdict, load, time, log), the logs it was read from, and both sides'
  images; `--resume` skips rows already in the CSV. Every number in the
  CSV comes from a log beside it.
- `plot_cpu_wavefront.py DIR/cpu-results.csv -o OUT.pdf [--sides a,b,...]
  [--log] [--values] [--width W --height H]`: that CSV as one figure --
  every side's speedup over pbrt's CPU wavefront (the dashed line at 1), a
  group of bars per scene and a subgroup per sample count; PBRT mk black,
  each Bonsai side its own colour from ColorBrewer's nine-class YlGnBu
  ramp laid over the drawing order (pale yellow for mk-b-s-ps to deep blue
  for the last wavefront), colour-blind safe and print-friendly. The image checks are not marked
  on the figure (the CSV's verdict column has them). The axis reads
  "Speedup over PBRT wf";
  pbrt's bars are "PBRT mk" (its megakernel, the CPU volpath) and the
  reference line "PBRT wf"; the bars run PBRT mk, mk-b-s-ps, then the
  wavefronts; a Bonsai schedule is named
  type-parallelism-SIMD-order[-sort] -- `wf`/`mk` (wavefront, megakernel),
  `b`/`t` (bulk, per-thread), `s`/`v` (scalar, vectorized drains),
  `ps`/`sp` (pixels outer or samples outer), `-sort` when the rays are
  sorted before the trace -- so `scalar` is mk-b-s-ps,
  `wavefront-thread-packet` is wf-t-v-ps and `wavefront-thread-packet-sort`
  is wf-t-v-ps-sort (the table is in the script).
- `plots.sh [--png]`: every figure of this directory, drawn into `plots/`
  under its one fixed name from the runs that are the current
  measurements (the user, 10-24: no guessing what is what). `plots/` holds
  exactly these and is not committed:
  - `pbrt-cpu-sweep.pdf`: the CPU schedules over pbrt's CPU wavefront,
    five scenes x 16/64/128 spp (plot_cpu_wavefront.py from
    cpu-compare-out and cpu-compare-out-sortfix);
  - `pbrt-gpu-sweep.pdf`: every measured OptiX schedule over `pbrt --gpu`
    on the same cells (plot_gpu_wavefront.py from gpu-compare-out,
    gpu-compare-out-sort and gpu-compare-out-explore);
  - `pbrt-gpu-heatmap.pdf`: the matching GPU wavefront (gpu-optix) over
    `pbrt --gpu` on all thirty scenes x 16/64/128/256 spp
    (plot_gpu_heatmap.py from the last heatmap run, compare-out-heatmap7).

  The runs are variables at the top of the script (`CPU_CSVS`,
  `GPU_TABLES`, `HEATMAP_TABLE`), overridable in the environment when a
  figure is to be drawn from another run; `--png` draws a `.png` beside
  each `.pdf` to look at.
- `plot_gpu_wavefront.py DIR/table.tsv -o OUT.pdf [--kernel] [--sides a,b,...]
  [--log] [--values] [--width W --height H]`: the GPU twin of that figure
  from gpu_compare.sh's table -- every OptiX schedule's speedup over
  `pbrt --gpu` (the dashed line at 1, "PBRT wf": pbrt's GPU renderer is
  its wavefront, and it has no GPU megakernel), wall-clock as the table's
  speedup column is, or kernel time with `--kernel`; the bars in order
  `Bonsai mk` (gpu-optix-mega), `Bonsai mk-ser`, `Bonsai mk-ser-ray`,
  `Bonsai wf` (gpu-optix), `Bonsai wf-b128`, `Bonsai wf-b512`,
  `Bonsai wf-fused`, `Bonsai wf-sort` (gpu-optix-sort) and
  `Bonsai wf-sort36`, whichever the tables hold -- the megakernels then
  the wavefronts, each family from plain to variants -- coloured with the
  same YlGnBu ramp spread over the sides drawn (four sides: classes 1, 3,
  5, 7; nine: the whole ramp). Only schedules on the ray tracing hardware
  are drawn (the user, 10-23): the software traversals never won a cell.
  The drawing is plot_cpu_wavefront.py's.
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
  counts come from `gpu_compare.sh --spp "16 64 256"` into one table. No
  title unless `--title`; the axes read "scene" and "spp", the colour bar
  "Speedup" (the user, 10-24).
- `report_stages.py OUT_DIR [--spp 16] [--report REPORT.md]`: the
  stage-by-stage tables of REPORT.md from a heatmap run's `.stats` files
  -- our kernel profile and pbrt's wavefront kernel profile paired into
  stages (trace, shadow, each material, media, camera, film), per scene and
  per stage, worst ratio first; with `--report` it replaces the section
  between the `stages:begin` and `stages:end` markers of that file.
- `convert_scene.sh <scene_dump> <dir/name> [options]`: the scene converted
  once, into `scenes/converted/<dir>/<name>.txt` with its sidecars, shared
  by every sweep and every sample count; prints the path. The conversion is
  redone only when what it is a function of changed -- the converter's
  sources, the pbrt it links, the options, the scene file -- which a stamp
  beside it records. The sample count and path depth are not a
  conversion's: the renderer's `--spp` and `--maxdepth` set them at render
  time (pbrt's override, transcribed in `render_hook.cpp`), as pbrt's own
  `--spp` does on the other side. `gpu_compare.sh` converts through it. The
  folder is data, not source (.gitignore). Until 2026-10-11 every cell of a
  sweep converted its scene again, and the copies of thirty sweeps filled
  the disk.
- `build_scene_dump.sh <out>`: builds `scene_dump`, which reads a `.pbrt`
  scene with pbrt's own parser and writes it in the driver's format: a text
  file for the small parts (camera, materials, lights, textures, spectra)
  and binary sidecars beside it -- `.geo` for the geometry (a FlatBuffer,
  `scene_geometry.fbs`: the meshes, each carrying its statement's material,
  alpha texture, media and first light as pbrt's GPU keeps them per mesh,
  with no record per triangle; the vertex pools; the shapes that are not
  triangle meshes; the instance definitions and instances; no trees -- the
  driver builds them, as pbrt's GPU builds its own),
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
