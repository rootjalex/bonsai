# apps/pbrt against pbrt's GPU wavefront: progress by scene and by stage

The standing question of this directory: does the renderer the compiler
makes of `render.bonsai` under `schedules/gpu-optix.bonsai` beat
`pbrt --gpu` (the wavefront volpath integrator, same spp, same image) on
every scene, stage by stage? This file is the running answer. Every number
is a kernel time in milliseconds at 16 spp, the least of three renders,
ours from `BONSAI_KERNEL_STATS` and pbrt's from its `--stats` kernel
profile, read off the compare harness's `.stats` files (`gpu_compare.sh`;
protocol in PLAN.md). A ratio is ours over pbrt's: below 1 is a win. The
compiler and driver are named with each table so that a row can be redone.

Rules: a row is only updated from a run whose image matched pbrt's; a stage
is "beaten" when ours is below pbrt's on every scene that has the stage;
nothing is tuned per scene. The wall-clock speedup the harness prints
(pbrt's process over ours, both warm) is reported beside the kernel total
where it was measured.

## Where it stands (2026-10-09, 07:00)

The surface-scene deficit of 10-05 to 10-08 -- every ray, probe and
alpha-tested shadow launch 1.7x to 9x behind pbrt's while the material,
camera and film kernels were at par or ahead -- had one cause, found 10-08
22:10: the acceleration structure's element arrays reached every OptiX
launch as host addresses (a loaded copy of the tree's layout struct,
threaded under an anonymous name past the launch marshaling's lookup), so
each hit's element loads went over the system-memory path. With the fix
(CodeGen_GPU_Host.cpp) every trace launch measured so far is faster than
pbrt's, and all three scenes measured are faster overall with matching
images. The remaining stage behind on every scene that has it is the
subsurface material kernel.

Measured 10-09 06:55 (`scratch/surface/compare-fixed`): the fix plus the
uncommitted mesh-entity series of 10-08, driver 615.71.09, OptiX 9.1 (pbrt:
OptiX 7.7); ours / pbrt in ms, then the ratio.

### head (head/head) -- wall speedup 2.06x

| stage | ours 10-07 | ours 10-09 | pbrt | ratio |
|---|---|---|---|---|
| closest-hit trace + routing (`__raygen__rays` / Trace closest hit rays) | 20.4 | 8.2 | 10.0 | 0.82 |
| subsurface probe trace (`__raygen__probes` / Tracing subsurface scattering probe rays) | 37.3 | 5.8 | 7.0 | 0.83 |
| shadow traces (`__raygen__shadow$False` + `__raygen__exit_shadow$False` / Trace shadow rays, both passes) | 2.0 | 2.0 | 2.2 | 0.92 |
| subsurface material (`hits_blk$Subsurface` / SubsurfaceMaterial + BxDF eval) | 9.0 | 9.0 | 5.3 | 1.70 |
| everything else (camera rays, exits, escaped, emissive, film) | 9.3 | 9.5 | 21.0 | 0.45 |
| **total kernel time** | 78 | **34.5** | 45.5 | **0.76** |

### killeroo-simple (killeroos/killeroo-simple) -- wall speedup 2.84x

| stage | ours 10-09 | pbrt | ratio |
|---|---|---|---|
| closest-hit trace + routing | 12.7 | 16.6 | 0.77 |
| shadow trace | 3.2 | 4.7 | 0.68 |
| coated-diffuse material (`hits_blk$CoatedDiffuse` / CoatedDiffuseMaterial + BxDF eval) | 6.0 | 7.5 | 0.80 |
| everything else | 12.3 | 20.6 | 0.60 |
| **total kernel time** | **34.2** | 49.4 | **0.69** |

### dragon_10 (sssdragon/dragon_10) -- wall speedup 1.68x

| stage | ours 10-07 | ours 10-09 | pbrt | ratio |
|---|---|---|---|---|
| closest-hit trace + routing | 140 | 36.8 | 42.0 | 0.88 |
| subsurface probe trace | 236 | 41.6 | 48.6 | 0.86 |
| shadow traces (both passes) | 6.9 | 13.9 | 15.5 | 0.90 |
| coated-diffuse material | | 33.7 | 41.9 | 0.80 |
| subsurface material | | 14.5 | 6.1 | 2.4 |
| everything else | | 23.6 | 56.9 | 0.41 |
| **total kernel time** | 464 | **164** | 211 | **0.78** |

Open on these three: the subsurface material kernel (1.7x on head, 2.4x on
dragon; pbrt's evaluates the BSSRDF and the BxDF in one kernel of 32
launches where ours launches 48 and does more per entry -- to be read
against pbrt's `SubsurfaceMaterial` kernel in the pbrt-GPU audit).

### The 10-07 matrix, the "before" (driver 615.71.09, compiler of 10-07)

| stage | head | dragon_10 | pavilion-day | book | sanmiguel | landscape |
|---|---|---|---|---|---|---|
| closest-hit trace + routing | 20.4 / 10.1 | 140 / 42 | 151 / 53 | 89 / 52 | 263 / 55 | 3796 / 419 |
| shadow trace | 1.0 / 2.1 | 6.9 / 15.6 | 45 / 17 | 8.8 / 12.6 | 63 / 21 | 1634 / 236 |
| subsurface probe trace | 37.3 / 7.0 | 236 / 49 | | | | |
| material kernels, summed | 8.9 / 5.7 | 49 / 50 | 74 / 83 | 40 / 59 | 51 / 51 | 275 / 273 |
| camera rays + samples | 3.5 / 6.1 | 8.4 / 18.5 | 8.0 / 14.5 | 12.4 / 19.5 | 6.7 / 13.4 | 49 / 75 |
| escaped, emissive, film, rest | 2.8 / 4.4 | 7.0 / 11.5 | 5.5 / 9.1 | 5.4 / 7.8 | 3.7 / 7.0 | 29 / 52 |
| total kernel time | 78 / 45 | 464 / 211 | 284 / 178 | 156 / 153 | 387 / 149 | 5784 / 1062 |

Each cell is ours / pbrt's. The three trace rows are the ones the fix
touches (the shadow row only where the scene has alpha-tested geometry,
whose any-hit program binds elements: pavilion, sanmiguel, landscape); the
rest should be unchanged by it.

## To do, in order

1. The 29-scene, three-sample-count sweep with the fix
   (`scratch/surface/sweep3.sh` -> `compare-out-heatmap3`, started 10-09
   07:05 in a window agreed with the other sessions; its heatmap replaces
   the 10-07 one), then this file's tables for every scene from its
   `.stats` files.
2. The stages still behind after that, per scene, in order of their share
   of the total; the subsurface material kernel first.
3. The pbrt-GPU behavior audit (PLAN.md, "Design: the mesh is the entity",
   last paragraph): every place pbrt's wavefront deviates from its CPU path.
