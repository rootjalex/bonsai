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
  [--out DIR] [--resume] [--reuse DIR] [--clock MHZ]`: the GPU table. pbrt `--gpu` and every GPU
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
  The schedules compile at once, each into `DIR/<schedule>/` (the
  generated `render.h`, the driver linked with it as `render.out`,
  `compile.log`, and `render.key`: a hash of the compiler binary and its
  flags, the renderer's and the schedule's `.bonsai` files, the driver's
  sources and headers and the runtime headers), seven in the ten minutes
  one takes; the driver's sources take the header's path from the macro
  `BONSAI_RENDER_H`, `render.h` beside them by default, so the compiles
  share nothing under `apps/pbrt`. `--reuse DIR2` copies a schedule's
  renderer, and the converter (`scene_dump`, `scene_dump.key`), from an
  earlier run's directory whose key is this run's, compiling only what
  has no match; `--resume` recompiles a renderer whose key moved.
- `heatmap.sh [--scenes "..."] [--spp "N ..."] [--out DIR] [--repeats N]
  [--title TEXT] [--clock MHZ] [--reuse DIR]`: the heatmap's run --
  `gpu_compare.sh` once per scene with `--resume` into one directory, the
  gpu-optix schedule alone, on every scene the converter takes (thirty) at
  16, 64 and 128 spp, each scene under a watchdog and no new scene after a
  cutoff (`HEATMAP_SCENE_TIMEOUT`, `HEATMAP_CUTOFF`, seconds), the figure
  drawn at the end; rerun the same command to finish an interrupted run.
  `--reuse` is `gpu_compare.sh`'s: a heatmap after a sweep under the same
  compiler skips its one compile.
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
  each Bonsai side the colour its schedule has on every figure here
  (figure_style.SCHEDULE_COLOURS: the megakernel warm, the wavefronts cool
  and darker with each variant; the user, 10-29). The image checks are not marked
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
- `compare_runs.py OLD_TABLE NEW_TABLE [--images] [--threshold 0.03]`: two
  runs of `gpu_compare.sh` or `heatmap.sh` cell by cell -- the kernel and
  wall times as new/old ratios, a verdict that matched pbrt before and does
  not now, and with `--images` the rendered .pfm against the old one byte
  for byte (a compiler change that is not meant to change what is computed
  leaves every image identical) -- with a summary of the cells slower,
  faster and within the threshold and the geometric mean of the kernel
  ratio per side. Exit status 1 on a verdict regression or a differing
  image, for a wrapper to read. What an overnight redo of the figures is
  checked with.
- `plots.sh [--png]`: every figure of this directory, drawn into `plots/`
  under its one fixed name from the runs that are the current
  measurements (the user, 10-24: no guessing what is what). `plots/` holds
  exactly these and is not committed:
  - `pbrt-cpu-sweep.pdf`: the CPU schedules over pbrt's CPU wavefront,
    five scenes x 16/64/128 spp (plot_cpu_wavefront.py from
    cpu-compare-out-head, the night of 10-28's run);
  - `pbrt-gpu-sweep.pdf`: every measured OptiX schedule over `pbrt --gpu`
    on the same cells (plot_gpu_wavefront.py from gpu-compare-out-e, the
    same night's run under the same compiler);
  - `pbrt-gpu-heatmap.pdf`: the matching GPU wavefront (gpu-optix) over
    `pbrt --gpu` on all thirty scenes x 16/64/128 spp, the scenes in
    order of the geometry pbrt built for them, volumes first (`--sort
    geometry`; plot_gpu_heatmap.py from the last heatmap run,
    compare-out-heatmap12), one band of thirty, the ratio in each cell to
    two digits, the colour bar along the top;
  - `pbrt-gpu-profile.pdf`: the wavefront's trace kernel against the two
    flat megakernels under Nsight Compute, a row of three panels per
    scene -- the launch's SASS instructions, the lanes active per warp
    instruction, and where a warp's cycles go (instruction fetch, the
    wait on the traversal hardware, memory, the rest) -- on ganesha,
    where the wavefront wins, and killeroo-gold, where the SER megakernel
    does, at 16 spp (plot_gpu_profile.py from profile-out-e, the reports
    profile_kernels.sh writes; the user, 10-28).

  All four are drawn for the paper's page (figure_style.py): ACM's
  acmsmall format, a 5.478 in text block, so each PDF is exactly that
  wide and `\includegraphics[width=\textwidth]` does not rescale it;
  labels and legends 8 pt and tick labels 7 pt as printed (the user,
  10-24: readable, not necessarily the body's 10 pt); one colour scheme
  for every figure (the user, 10-29: the sweeps had each laid the one ramp
  over its own bars) -- the megakernel schedules on a warm ramp
  (ColorBrewer YlOrBr) and the wavefront schedules on the cool one
  (YlGnBu, the heatmap's), each darker with every scheduling step, the
  same schedule the same colour on the CPU sweep, the GPU sweep and the
  profile figure (`mk` and `mk-b-s-ps` alike, `wf` and `wf-b-s-sp` alike),
  pbrt's bars black; figure_style.py's SCHEDULE_COLOURS is the table and
  says how the shades were chosen; the sweeps' y axis reads
  "Speedup" with the dashed PBRT line named in the legend, the legend in
  two rows with the schedule codes alone (pbrt's two say "PBRT"; the rest
  are Bonsai's). Beside each figure the plotter writes `<figure>.txt`,
  the statistics the paper quotes (the user, 10-24): for every series the
  geometric mean, the least and greatest value and which scene and sample
  count each is, over everything and per sample count (the heatmap also
  per scene); then a reading of the figure -- what each scheduling
  decision bought, as the ratio of the two sides that differ by it, bar by
  bar; the trend from the fewest samples to the most; the mechanisms
  behind the figure and, for every scene, the best schedule at each sample
  count (computed) with the measured reasons it is best there (the user,
  10-29; a scene without a written reading is marked); the heatmap's
  volume-against-surface split, the rank correlation of the speedup
  with the geometry built and its highest and lowest scenes -- the tables
  each figure was drawn from, and the PLAN.md sections where the causes
  were measured. Liberation Sans embedded as TrueType (fonttype 42), since ACM's
  production rejects the Type 3 fonts matplotlib emits by default (check a
  PDF with `pdffonts`). The two sweeps are the same size so they stack as
  one float's subfigures.

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
  the wavefronts, each family from plain to variants -- each in its
  schedule's colour (figure_style.SCHEDULE_COLOURS: the megakernels warm,
  `mk` palest to `mk-ser-flat` darkest; the wavefronts cool, `wf` the
  palest shade and `wf-sort` the darkest, as the CPU figure's wf-b-s-sp
  and wf-t-v-ps-sort are). Only schedules on the ray tracing hardware
  are drawn (the user, 10-23): the software traversals never won a cell.
  The drawing is plot_cpu_wavefront.py's.
- `profile_kernels.sh [--out DIR] [--renderers DIR] [--single-pass-timing]
  SCHEDULE SCENE SPP LAUNCH` and `profile_kernels.sh --list N SCHEDULE
  SCENE SPP`: one OptiX launch of a renderer gpu_compare.sh compiled
  (RENDERERS/SCHEDULE/render.out, gpu-compare-out-e by default; SCENE as
  gpu_compare.sh names it, `killeroos/killeroo-gold`, a bare name being
  `<name>/<name>`) under Nsight Compute's full section set, kernel replay,
  written to `profile-out-e/<name>-SPP-SCHEDULE-L<LAUNCH>.*`: the report,
  its details page, every metric as one CSV row and the launch's SASS
  with per-instruction counts. Nsight names every OptiX launch
  `optixLaunch`, so the launch is chosen by its ordinal in the process;
  `--list N` prints the first N (duration, grid, threads, lanes) to tell
  them apart: the wavefront's first is its first band's closest-hit
  trace, the flat megakernels' first is the per-pixel prologue and their
  second the frame (a pipeline with shader execution reordering is listed
  as one block of 512 threads at 255 registers: Nsight's view of its
  dispatch, not a grid). `--single-pass-timing` re-measures the timing
  metrics the figure draws in renderer runs of one Nsight pass each and
  merges them over the full set's (kept as `.multipass.raw.csv`): kernel
  replay restores the launch's memory between passes, and on ganesha that
  made the wavefront's trace launch read 250 MB from system memory and
  run 47x slow (10-28); a stall ratio that needs two passes even alone
  (the load latency) is the remainder over every other stall reason,
  each in a one-pass run, as the `bonsai__timing_provenance` column
  says; `--merge-only` redoes the merge from the single-pass files
  without the card. The profiler runs on an OptiX process only with the
  user's say-so (10-20, after a driver crash); the lock is the caller's.
- `plot_gpu_profile.py --cell SCENE:SPP [--cell ...] [--sides a,b,...]
  [--profiles DIR] -o OUT.pdf`: the profile figure from those reports, a
  row per cell and three panels across; `--sides` names the schedules in
  drawing order (gpu-optix drawn `wf trace`, then `mk-flat` and
  `mk-ser-flat`), a side without a profile left out of its row with a
  note. Beside it `<figure>.txt`: every launch's duration, instructions,
  registers, achieved warps, issue-slot use, the four stall components
  in cycles, lanes, SASS size and the lanes of each function in the
  launch (the raygen's continuations before and after the first trace,
  the hit and miss programs), and where the reading is in PLAN.md.
- `plot_gpu_table.py OUT.png TABLE.tsv...`: the tables gpu_compare.sh
  wrote, plotted -- each schedule's speedup over `pbrt --gpu` and every
  side's kernel time, a group of bars per scene and a row of panels per
  table (sample count); a bar whose image did not match pbrt's is hatched.
- `figure_style.py`: the paper's figure style the three plotters import --
  `TEXT_WIDTH` (5.478 in, acmsmall's text block), `paper_style()` (TrueType
  embedding, Liberation Sans, 8 pt labels, 7 pt ticks, thin axes),
  `YLGNBU` (the nine-class ramp), `describe`, `ratios` and `write_summary`
  (the `.txt` beside a figure). apps/rtq/plot.py and apps/wosx/plot.py
  import it too.
- `plot_gpu_heatmap.py OUT.png TABLE.tsv... [--schedule S] [--sort
  table|name|speedup|geometry] [--per-band N] [--width IN]`: the same
  tables as a heatmap -- scenes across (one band by default; `--per-band
  N` cuts them into bands of N stacked down the figure),
  sample counts down, each cell one schedule's speedup over `pbrt --gpu`
  (default `gpu-optix`), coloured on a log scale centred on parity with the
  ratio printed in the cell, the GraphIt paper's figure for a schedule
  against its baselines. A cell whose image did not match pbrt's is hatched
  with its number struck through; a cell no table has is blank. Several
  counts come from `gpu_compare.sh --spp "16 64 256"` into one table. No
  title unless `--title`; the axes read "scene" and "spp", the colour bar
  "Speedup" (the user, 10-24). `--sort geometry` orders the scenes by the
  shapes pbrt built for them -- the Geometry section of the `.pbrt.stats`
  beside each scene's pbrt image: triangles (displacement included),
  bilinear patches, spheres, disks, cylinders, curves; base geometry, not
  instances expanded -- so a purely volumetric scene (one or two shapes)
  comes first and kroken (288M triangles) last; the order and counts are
  printed.
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
  them so the hardware traces them; its CPU keeps bilinear patches), and a
  scene whose integrator this renderer lacks (`bdpt`, `sppm`) is converted
  as `volpath`, with the warning pbrt's wavefront prints, since that is
  what `pbrt --gpu` renders it with (without `--gpu` such a scene is
  refused: pbrt's CPU would run the scene's own integrator, and two
  integrators compared measure the integrators). `gpu_compare.sh` passes
  `--gpu`, since `pbrt --gpu` is its reference. The cameras are pbrt's
  `perspective`, `orthographic` and `realistic` (the lens system, read off
  the camera pbrt built: its focused interfaces and its 64 exit pupil
  bounds; an `aperture` image or shape is not supported yet), the films
  `rgb` and `gbuffer`; anything else pbrt has and this renderer lacks is
  refused by name rather than rendered as something else.
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
