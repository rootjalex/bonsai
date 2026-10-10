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
where it was measured. The per-scene and per-stage tables below are
generated: `python3 apps/pbrt/report_stages.py <heatmap dir> --report
apps/pbrt/REPORT.md` rewrites them from a heatmap run's `.stats` files
(README.md), so a new sweep updates them in one command.

## Where it stands (2026-10-10, 13:40)

**The fresh sweep: every cell faster than pbrt --gpu, by 1.04x to 2.56x.**
The 30-scene, three-sample-count sweep the user asked for ran this
morning (`heatmap.sh`, 10:20 to 12:40, `compare-out-heatmap4`; driver
615.71.09, OptiX 9.1 against pbrt's 7.7; the compiler and renderer of
76e3b4e5, pbrt's structure and representation matched end to end; the
GPU clock unlocked and recorded so, at the user's direction, since
another project's benchmarks share the card). Wall speedups at 16 spp
run from 1.13x (landscape) to 2.56x (killeroo-simple), geometric mean
1.64x; at 64 spp 1.06x to 2.28x, mean 1.42x; at 128 spp 1.04x to 2.27x,
mean 1.36x. Kernel time is 0.43 (watercolor) to 0.95 (landscape) of
pbrt's at every count. The heatmap is `compare-out-heatmap4/heatmap-gpu-
optix.png`. Images: 84 of the 90 cells match pbrt; the six that do not
are the known ones (pavilion-day at 16 spp, 0.17% fewer lit pixels
against an allowance of 0.1%; bistro_cafe at every count and bistro_vespa
at 16 and 64, the lit-pixel bias open since 10-05), unchanged in kind and
count from the sweeps before.

Against yesterday's sweep (`compare-out-heatmap3`, built with the then
uncommitted mesh-entity series) this one is 5% slower in wall time and
1.6% slower in kernel time across the board (geometric means over the 30
scenes at each count). That is the structure commit of last night
(c73d5f60): the sampler draws moved from the material kernels into the
trace launch by the decision of 10-09, the material queues split by
texture evaluator into 22 kernels, the BSSRDF kernel split out, and with
them 54 one-word memsets a round on the host. The per-stage tables below
show it as the closest-hit trace row behind on 26 of 30 scenes at 1.27 of
pbrt's sum -- but that row sets our trace launch *with* the draws against
pbrt's trace kernel *without* them, pbrt's sample kernel being a row of
its own ("sample draws: pbrt only", 798 ms over the 30 scenes). Against
pbrt's trace kernel plus its sample kernel, our launch was 1.02 of pbrt's
summed, ahead on 13 scenes and behind on the heavy ones (disney-cloud
1.20, landscape 1.19); and the draws' own addition to our launch, read as
its growth since yesterday's sweep, was 1.44x what pbrt's sample kernel
costs pbrt for the same draws (scratch/surface/reset/trace-plus-
draws.txt).

**Two changes since, measured over the 30 scenes at 16 spp** (`compare-
out-reset16`, 12:50 to 13:35, against this morning's cells): wall time
faster on all 30, geometric mean 1.074x (frame542 1.29x, dambreak 1.17x,
rough-glass 1.16x, the rest 1.02x to 1.13x); kernel time 1.017x; no cell
slower; images byte-identical to the morning's on the scenes checked and
the same 27 of 30 matching pbrt.

1. *The draws made pbrt's* (the user, 12:00: "the draw should be exactly
   the same"). Audited function by function against pbrt's samplers.h,
   lowdiscrepancy.h, rng.h and wavefront/samples.cpp (PLAN.md, "The draws
   made pbrt's"): every draw's arithmetic already matched; what differed
   was the sampler's life. pbrt's sample kernel makes a fresh local copy
   of the sampler per ray per bounce and starts it from the pixel and
   sample index at dimension 6 + 7 * depth; ours threaded a 40-byte `mut
   SamplerState` down the path and through the recursion, so the deferral
   kept it per ray in a record slot that every bounce read, restarted,
   drew through (each draw's dimension step a global load and store) and
   wrote back -- the same restart as pbrt's, some forty global memory
   operations more per ray per bounce on the raygen's critical path. Now
   the path carries the ray's pixel and the pass's sample index, each
   bounce starts a local sampler as pbrt does, the per-ray state array is
   gone, the generator's jump-ahead is one jump that stops at the delta's
   top bit as pbrt's does, and the draw code compiled as a kernel of its
   own is 4096 SASS instructions and 48 registers against pbrt's sample
   kernel's 3576 and 55 (Halton, both). The trace launch with the draws
   in it is now 0.96 of pbrt's trace plus sample kernel summed (3815
   against 3975 ms), ahead on 26 of 30 scenes; the draws' addition to it
   is 1.17x pbrt's sample kernel (932 against 798 ms), landscape 2.0x,
   book 1.9x, crown 1.7x, villa-lights-on 1.6x at the far end and most
   scenes near or under one. What is left is the placement -- the ten
   values and the ray live across optixTrace, the raygen's occupancy --
   which the user is weighing (the options in the conversation of 13:20:
   a per-pixel record for the drawn values with the schedule as it is;
   or a stage onto the drained queue, pbrt's sample kernel without the
   copy of the ray's state).

2. *One reset launch per round* (the first general codegen fix of the
   profiling the user asked for; PLAN.md, "One reset launch per round").
   A round's host side issued 54 `cuMemsetD32Async` of one word -- a
   queue's count zeroed -- where pbrt resets every queue a depth fills
   in one launch. Twenty-two were a bug in the round-guard pass (the
   material queues' counts zeroed twice a round, the exit's reset not
   recognized for an array of headers); the rest were resets strewn
   through the round, one stream operation each, on the critical path
   behind the round guard's synchronous read. Now the round guard moves
   every inner queue's reset to the top of the round beside the ray
   queue's own, and the backend issues a run of host word-stores into
   device memory as one launch of a generated one-thread kernel
   (`_bonsai_store_words_32` here), pbrt's `GPUDo("Reset queues")` as a
   lowering: 32 words at 7.3 us of host issue a round where the memsets
   took 33.6 (the microbenchmark: a microsecond a word against a flat
   7.3 for the launch, so the launch pays from four words up). On
   killeroo-simple the host's gap over the kernels went from 7.1 to 3.8
   ms a render.

Stages, at 16 spp over the 30 scenes (the tables below, from this
morning's sweep): beaten on every scene that has them are the shadow
traces (0.83, the two bistros excepted), escaped rays (0.50), emitters
(0.38), every material but one (coated-diffuse 0.55, diffuse 0.42,
conductor 0.22, coated-conductor 0.49, dielectric 0.51, diffuse-
transmission 0.62, thin-dielectric 0.68, the universal-texture variants
0.51 to 0.69), the subsurface probe trace and exit (0.85, 0.51), medium
sampling and scattering (0.66, 0.60) and the film (0.32). Behind: the
camera rays on 29 of 30 (1.35 summed, worst lte-orb-silver 1.79: the
next stage to profile), the trace launch as read above, the measured
material on 1 of 2 (1.01), the BSSRDF sampling on 2 of 2 (1.08) and the
subsurface material on 1 of 2 (1.05).

What still differs from pbrt: the draws' placement (the user's decision,
under review); the material kernels launched for the last depth, where
pbrt leaves its loop before them; the round loop's per-round synchronous
read of the ray queue's count, where pbrt runs a fixed number of rounds
-- the host's gap over the kernels that remains (3.8 ms of killeroo-
simple's 38.7) is this read and the launches it serializes; the 31 queue
headers restaged once a pass (PLAN.md); and the 8-byte triangle record
per triangle that pbrt's device does not have (the synthesized triangle
set, a layout construct to ask for). All are open items in PLAN.md.

## Where it stood (2026-10-09, 09:30; the evening's addendum first)

**Evening of 10-09: pbrt's structure is matched.** The gpu-optix schedule
now has pbrt's queues and kernels one for one -- the material queues split
by texture evaluator and by material under each, the BSSRDF kernel over a
queue of its own -- with the one difference of the sampler calls made at
the top of the trace launch rather than in a kernel of their own (the
decision taken after "The sample stage" below). Images are byte-identical
to the sweep's on the seven scenes checked. Against the morning's sweep it
costs nothing on the villas and killeroo, 3.8 ms on head and 132 ms (17%)
on kroken -- but that comparison is not like for like: the sweep's
binaries carried the uncommitted mesh-entity series of 10-08
(scratch/wt-fix), which the committed tree does not, and that series
moved the ray launches by a third to a half and takes the per-triangle
primitive records out of the scene. Kroken's loss was memory, and it is
fixed (10-10, 03:10). The scene is the same on both sides -- 287.9
million triangles in 248 meshes by pbrt's own count, 247.7 million of
them made by pbrt's displacement tessellation at load, and the dump holds
the same 287.9 million -- but every triangle carried a primitive record
of its own (material, light, alpha, the two media and the shape: 32
bytes, 9 GB in all, plus 8 bytes of triangle record, 2.3 GB) where pbrt
keeps one such record per mesh in its hit records, so the scene took
about 38 GB of device memory against pbrt's 27.2 GB, the card's 32.6 GB
filled, and the managed queue memory thrashed. Three changes since, all
pbrt's representation: the mesh-entity series landed (1ed2eabb), the
scene file is per mesh (6aba4023: the placement on the mesh, no record
per triangle, no tree of pbrt's), and an instanced mesh is a primitive of
its own, placed by its instance, with the definition's structure shared
by every instance of it as pbrt shares it. With them kroken renders at
779 ms of kernels (the sweep's 764, pbrt's 1733) with the host gap gone
(0.80 s of wall), within pbrt's memory, and kroken, landscape, head and
killeroo render pixel for pixel with pbrt ("pbrt's structure, matched"
below). The GPU clock stays unlocked for now (the user, 10-10: another
project's benchmarks share the card); the scripts record what it read.

**Every cell of the sweep is faster than pbrt --gpu.** The 30-scene, three-
sample-count sweep run after the fix below (`heatmap.sh`, 10-09 07:00 to
09:17, `compare-out-heatmap3`; driver 615.71.09, OptiX 9.1 against pbrt's
7.7; the compiler of d5d4a3ee plus the uncommitted mesh-entity series of
10-08) puts the wall speedup between 1.06x (landscape at 128 spp) and
2.88x (killeroo-simple at 16 spp), and the kernel-time ratio at 16 spp
between 0.41 (watercolor) and 0.95 (landscape). The heatmap is
`compare-out-heatmap3/heatmap-gpu-optix.png`; it replaces the 10-07 one,
whose surface scenes were at 0.67x to 0.86x.

Images: 27 of the 30 scenes match pbrt at all three counts. The three that
do not are not new: pavilion-day at 16 spp lights 0.17% fewer pixels than
pbrt where the rule allows 0.1% (the same count as on 10-07; it passes at
64 and 128); bistro_cafe at every count and bistro_vespa at 16 and 64 light
more pixels than pbrt -- at 16 spp 176,546 pixels lit only here against
12,167 lit only in pbrt, the mean radiance agreeing to 0.04%, the extra
pixels dim (median luminance 1.7e-4) and all over the left facade's ivy and
awning -- the lit-pixel bias open since the 10-05 investigation (PLAN.md,
"The bistro and villa lit-pixel verdicts"), with the same counts in the
10-04 sweep and under both GPU schedules, so untouched by anything since.
vespa at 128 passes as the gap closes with spp.

Stages, at 16 spp over all 30 scenes (the tables below, which from 12:50
pair pbrt's sample kernel with a sample-draws row of its own rather than
with the camera rays): beaten on every scene that has them are the
escaped-ray and emitter kernels (0.50, 0.40), the film and the rest
(0.10), the coated-diffuse, conductor and coated-conductor materials
(0.57, 0.39, 0.55), the subsurface probe trace and exit (0.85, 0.51) and
medium sampling (0.66). Behind somewhere: the camera rays, against pbrt's
camera kernel alone, on 29 of 30 (1.34 summed; the earlier 0.32 of this row
had pbrt's sample kernel in it), the closest-hit trace on 2 of 30
(landscape 1.08, disney-cloud 1.02; 0.91 summed), the shadow traces on
the two bistros (1.23, 1.11; 0.83 summed), the dielectric material on 11
of 16 in the sweep (worst sanmiguel 1.53; frame542 194 against 146 ms is
most of it; 1.12 summed -- the sample stage below turns this round), the
diffuse material on 7 of 22 (worst sportscar 1.33; 0.55 summed),
diffuse-transmission on 4 of 8, measured on 3 of 3 (1.09 summed),
subsurface on 2 of 2 (1.17), thin-dielectric on 2 of 2 (1.35) and medium
scattering on 3 of 9 (plume 1.11, explosion 1.10, bunny-cloud 1.07; 0.80
summed). Two caveats on the material rows of the sweep: ours drew a
bounce's samples in the material kernel where pbrt has a separate sample
kernel (the "materials with their sample draws" view adds the two), and
pbrt's profile lists the launches it made for a material no ray reached
(empty launches at its floor of about 0.015 ms each), which ours never
makes.

### Why the speedup moves with the sample count (Matt Pharr's question, 10-09)

Both renderers scale linearly in their kernels: per sample, pbrt's kernel
time at 128 spp over its time at 16 spp has a median of 1.002 across the
30 scenes and ours 1.001, and the ratio of our kernel time to pbrt's is the
same at both counts (median 0.697 and 0.696). pbrt's two exceptions are
kroken (0.81) and watercolor (0.98), the texture-heaviest scenes, whose
16-spp kernels carry a cold-texture cost. The GPU's clocks are not locked
by the harness (the RTX 5090 idles at 180 MHz and boosts to about 3 GHz;
`nvidia-smi -lgc` needs root), but these numbers show no clock effect.

The variation is in the wall times' overheads. pbrt's render timer carries
a fixed cost beyond its kernels, a median of 75 ms per render at any count;
ours carries about 8 ms at 16 spp that grows with the count, 20 to 40 µs
per wavefront round: the round loop reads the ray queue's count back from
the device each round to decide whether to go on (a synchronization per
round), where pbrt runs maxDepth rounds blind and launches every kernel,
empty or not. So at 16 spp pbrt's fixed cost flatters our speedup and at
128 spp our per-round readback eats into it: killeroo is 2.88x by wall at
16 spp, 1.55x at 128, and 1.45x by kernels at both. The 64 and 128 spp
rows of the heatmap are the ones to trust; the remedy is pbrt's own loop,
a fixed round count from the program's `max_depth` with no readback (PLAN.md,
deferred).

### The fix behind it

The surface-scene deficit of 10-05 to 10-08 -- every ray, probe and
alpha-tested shadow launch 1.7x to 9x behind pbrt's while the material,
camera and film kernels were at par or ahead -- had one cause, found 10-08
22:10: the acceleration structure's element arrays reached every OptiX
launch as host addresses (a loaded copy of the tree's layout struct,
threaded under an anonymous name past the launch marshaling's lookup), so
each hit's element loads went over the system-memory path. Fixed in
CodeGen_GPU_Host.cpp (d5d4a3ee); the first three scenes measured 10-09
06:55 are kept below as the before-and-after.

### Profiled: the dielectric material kernel (2026-10-09, 09:45-10:15)

Nsight Compute on `_kernel_hits_blk$Dielectric_7` against pbrt's
`EvaluateMaterialAndBSDF<DielectricMaterial>` kernel, on sanmiguel-courtyard
(the worst ratio, 1.53) and frame542 (the largest gap, 194 against 146 ms):
a per-launch pass over every launch of both, and the full section set with
SASS stall sampling on the first two launches of sanmiguel
(`scratch/surface/dielectric/`: `perlaunch.py`, `sass_path.py`,
`sass_tier.py`, the `.ncu-rep` and `.csv` files). Clocks locked by the
profiler, so the times are not the sweep's but are comparable to each other.

| sanmiguel, 16 spp, under the profiler | ours | pbrt |
|---|---|---|
| launches | 192 | 160 |
| kernel time, all launches | 6.32 ms | 3.80 ms |
| warp instructions, all launches | 298 M | 140 M |
| an empty launch | 3.6 µs | 23 µs |
| a launch with glass hits in it | 34 to 50 µs | 22 to 30 µs |
| threads launched, registers per thread | 552 K, 128 | 552 K, 128 |
| achieved occupancy | 20% | 30% |
| instructions per warp that had work (depth 1) | 3,830 | 1,121 to 1,356 |
| spill traffic, all launches | 0.74 GB | 1.54 GB |

frame542 is the same picture at scale: ours 8,224 launches in 303 ms
against pbrt's 8,192 in 217 ms; our 32 empty launches cost 4.2 µs each and
pbrt's 1,952 empty ones 25 µs each, but our 8,192 launches with work
average 36.9 µs (618 K instructions) against pbrt's 6,240 at 26.9 µs
(372 K).

**What the time is.** A launch of this kernel has very few items -- in
sanmiguel's second launch, 160 of the 17,370 warps have a hit to handle,
and the GPU sits at 1.5% of its compute throughput -- so its duration is
the dependent chain of one working warp, not throughput. Of our 3,830
instructions per working warp, 1,865 (49%) and most of the serialized
memory latency are the bounce's seven sampler draws: `vol_surface` draws
`direct_uc`, `direct_u`, `uc`, `u` and `u_rr` for every hit, specular or
not, as pbrt's wavefront consumes seven dimensions per bounce; each draw
of the Halton sampler (`scrambled_radical_inverse`) is an out-of-line
64-bit division followed by about six digit iterations, each with a
dependent 16-bit gather from the permutation table -- some 42 serialized
gathers per hit -- and each draw begins by reloading the sampler state's
dimension from global memory and ends by storing it back (`state : mut
SamplerState` lives in memory, not registers). The other 1,560
instructions (41%) are the straight path: the entry's loads and their
immediate spills (46 reloads, 49 spills on the path), the material, the
BSDF sample, the next ray.

**Why pbrt does not pay it here.** pbrt's wavefront draws the same seven
dimensions in a kernel of its own over the whole ray queue before the
trace (`GenerateRaySamples`, wavefront/samples.cpp: dimension 6 + 7·depth,
the seven values stored with the ray), where half a million rays overlap
the chain and the material kernels read seven floats. That kernel costs
pbrt 7.98 ms on sanmiguel, which this report's stage table charges to
"camera rays + samples" (hence our 0.32 there). Our schedule has no sample
stage: the draws run inside each material kernel, which is fine where the
kernel has many items and is the whole cost where it has few. The same
accounts for the other cheap-per-item kernels behind (thin dielectric,
diffuse on its scenes, measured, medium scattering), to be confirmed.

**Patterns of ours worth fixing regardless, in the compiler:** the
sampler state round-tripping through global memory on every draw; the
layout's 28 array base pointers loaded from a params struct in global
memory at the start of every working warp, one extra dependent level
where pbrt's parameters sit in the constant bank; the 64-bit division as
a call made seven times. Our empty launch at 3.6 µs against pbrt's 23 µs
is the gated launch at work and is the one place we are ahead here.

**The decision taken (the user, 10-09 10:20): match pbrt's schedule.** The
sample stage is in (below), and the audit that follows is of everything
else.

**The like-for-like sum.** Our material kernels did the draws that pbrt's
sample kernel does, so the fair comparison of the material rows is ours
against pbrt's material kernels plus its `Generate ray samples`. The
generated tables below carry that as "materials with their sample draws
(a view, not a stage)": summed over the 30 scenes ours is 3,476 ms against
pbrt's 6,565 (0.53), and no scene is above 0.93 (sportscar). On the scenes
the dielectric row singled out: sanmiguel 50.9 against 50.7 + 8.0 = 58.7
(0.87, where the dielectric row alone said 1.53), frame542 194 against
259 + 182 = 440 (0.44, against 1.33), crown 439 against 565 + 79 = 644
(0.68, against 1.26). The dielectric row was an accounting artifact of
where the draws ran; the latency chain it exposed was real, and the sample
stage removes it from every material kernel at once.

### Audit: pbrt's wavefront kernels against the gpu-optix schedule (2026-10-09)

pbrt's `WavefrontPathIntegrator::Render` (wavefront/integrator.cpp), one
sample of one scanline band at a time (our `p_band` pass), against the
drains the schedule makes, in pbrt's order. A kernel is "matched" when ours
does the same work over the same queue at the same point of the round.

| pbrt | ours | matched |
|---|---|---|
| Reset ray queue; Reset queues before tracing rays; Reset shadowRayQueue | the counts are reset by the round's own kernels | no kernel needed |
| Generate camera rays (with the camera's sample dimensions, `StartPixelSample(pPixel, sampleIndex, 0)`) | `p_blk$VolPath` | yes |
| Generate ray samples (the bounce's seven dimensions per ray, `6 + 7·depth`, plus three with subsurface) | the `rays` drain, `vol_ray_samples` -- **added 10-09**; until then drawn inside each material and medium kernel | yes, from 10-09 |
| Trace closest hit rays (`__raygen__findClosest`; the closest-hit program builds the interaction and routes the ray: escaped, emissive, a material's queue, the medium-sample queue, or straight on across a medium boundary) | `__raygen__traced`: the trace and `vol_route` | yes; ours routes in the raygen after the trace returns, pbrt in its closest-hit program (the uncommitted mesh-entity series of 10-08 moves ours there too) |
| Update camera / indirect ray stats | -- | statistics only |
| Sample medium interaction | `medium_samples_blk` | yes |
| Sample direct/indirect - Henyey-Greenstein | `medium_scatters_blk` | yes |
| Handle escaped rays | `escaped_blk` | yes |
| Handle emitters hit by indirect rays | `emissive_blk` | yes |
| `if (wavefrontDepth == maxDepth) break` -- the loop ends here, before the material kernels | ours runs the material drains at the last depth too, on empty queues (head: 48 launches against 32, 0.016 ms each) | nearly: a gate on the depth would drop them |
| `<Material>Material + BxDF eval`, one kernel per material **and per texture evaluator** (Basic / Universal queues) | `hits_blk$Basic$<Material>` and `hits_blk$Universal$<Material>` -- **from 10-09 evening**: `hits.specialize(evaluator)` then `hits[Basic].specialize(material)` and `hits[Universal].specialize(material)`; the trace routes a hit by `material_evaluator` (bxdf.bonsai), pbrt's per-material `CanEvaluateTextures` with the displacement, and the kernels read textures through `eval_*_by(evaluator, ...)` (textures.bonsai) | yes, from 10-09 evening (until then one kernel per material) |
| Trace shadow rays / Tracing shadow Tr rays | `__raygen__shadow$<have_media>` | yes |
| Get BSSRDF and enqueue probe ray (a kernel over `bssrdfEvalQueue`; the BSSRDF's textures read with `BasicTextureEvaluator()` whichever material queue the hit came through, wavefront/subsurface.cpp) | `bssrdfs_blk`, the drain of `bssrdfs` -- **from 10-09 evening**: `vol_surface.defer(vol_subsurface, bssrdfs)`, pushed by both evaluators' subsurface kernels; `vol_subsurface` reads the material's textures with `Basic()` as pbrt's kernel does (until then it was the subsurface material kernel's own tail) | yes, from 10-09 evening |
| Tracing subsurface scattering probe rays (`__raygen__randomHit`) | `__raygen__probes` | yes |
| Handle out-scattering after SSS | `exits_blk` | yes |
| Trace shadow rays, the second pass | `__raygen__exit_shadow$<have_media>` | yes |
| Update film (once per sample) | `rays_rest_blk` | yes |
| the image from the film, after the last sample | `q_blk` | yes |

After the sample stage two structural differences remained, the BSSRDF
kernel folded into the subsurface material kernel and the material queues
not split by texture evaluator; both are matched from the evening of
10-09 (the rows above; the measurement is in "pbrt's structure, matched"
below). What differs from pbrt now, by the user's decision: the sample
kernel. pbrt draws a bounce's sampler values in a kernel of its own over
the ray queue (GenerateRaySamples) and the trace kernel then reads the ray
queue again; ours draws them at the top of the trace launch, one pass over
the queue instead of two -- "The sample stage" below measured the two-pass
form and it cost more than it saved. The empty material launches at the
last depth are the one small difference left on pbrt's side.

**The sample stage (10-09, 10:30).** `vol_ray_samples` draws the seven
(or ten) dimensions at the top of `vol_path_step`, right after the sampler
is restarted at the bounce's dimensions, and the material, medium and
subsurface kernels read the `RaySamples` value instead of drawing; the
values are the same dimensions in the same order, so every image is
unchanged bit for bit. `vol_path_step.stage(vol_ray_samples, traced)` in
gpu-optix.bonsai and gpu-wavefront.bonsai makes the `rays` drain the
sample kernel and the new `traced` queue's drain the trace, in pbrt's
order. It needed one compiler fix: the drains run in the order the
schedule defers their queues, so the stage has to come before the
medium-sample deferral that the same function makes, and that deferral
then found no call in `vol_path_step` -- the stage had moved it into
`vol_path_step!after`. A deferral now looks in the continuations split off
from the function it names (SSA/Stage.h, `call_holder`; tests
ssa/stage-then-defer and correctness/llvm/stage-then-defer). Two more
compiler fixes followed before it rendered: the C header's identifiers
(an anonymous SSA name reached it), and a round queue with one buffer
clearing its count ahead of its pass, which a GPU drain, launched over the
capacity and testing each index against the count on the device, read as
empty -- the first one-buffer round queue a GPU schedule had, since the
ray queue stopped being pushed by its own drain once the trace moved to
`traced` (eddc0297, with the two before it).

**Measured (10-09 12:40, `scratch/surface/compare-stage`, 16 spp, kernel
ms; images identical to the sweep's pixel for pixel):**

| scene | before (the sweep) | with the stage | pbrt |
|---|---|---|---|
| killeroo-simple | 34.1 | 40.6 | 49.4 |
| head | 34.6 | 40.7 | 45.5 |
| sanmiguel-courtyard | 129.5 | 138.8 | 149.4 |
| frame542 | 711 | 749 | 1206 |

| kernel | before | with the stage | pbrt |
|---|---|---|---|
| frame542, dielectric material | 194.0 | 71.5 | 145.7 |
| sanmiguel, dielectric material | 4.32 | 2.10 | 2.83 |
| sanmiguel, diffuse material | 25.7 | 21.0 | 24.5 |
| sanmiguel, coated-diffuse material | 14.6 | 12.7 | 17.8 |
| sanmiguel, the trace launch | 49.9 | 52.0 | 55.3 |
| sanmiguel, the sample kernel | -- | 17.4 | 8.0 |
| killeroo-simple, the sample kernel | -- | 8.6 | 3.6 |
| frame542, the sample kernel | -- | 189 | 182 |

So the stage does to the material kernels what the profile said it would
-- the dielectric kernel is ahead of pbrt on every scene now, frame542 at
0.49 -- and the stage itself costs more than it saves on a dense scene: our
sample kernel is 2.2x pbrt's on sanmiguel and 2.4x on killeroo. The reason
is the stage's form, not the draws: a stage's continuation is a new queue,
so every ray's state (about 200 bytes: ray, throughput, the two
probabilities, wavelengths, the previous vertex, depth, flags) is written
onto `traced` by the sample kernel and read back by the trace, once more
per bounce, where pbrt writes 40 bytes of samples into a per-pixel array
and traces the same ray queue it sampled. On frame542, whose launches are
small, the per-launch floor dominates both and the two kernels cost the
same. The matching structure is a stage *in place*: the rest of the
function run as a second pass over the same queue, the staged call's value
kept in a per-entry side array (pbrt's `pixelSampleState.samples`), which
is compiler work -- the decision pending below. The other point, the
draws made in the trace launch itself ahead of `optixTrace` with no stage
(`schedules/gpu-optix-draws-in-trace.bonsai`: the same program under the
committed schedule), measured 13:05 (`scratch/surface/compare-stage-c`,
images identical again):

| 16 spp, kernel ms | before | stage | draws in the trace launch | pbrt |
|---|---|---|---|---|
| killeroo-simple | 34.1 | 40.6 | 35.8 | 49.3 |
| head | 34.6 | 40.7 | 37.9 | 45.2 |
| sanmiguel-courtyard | 129.5 | 138.8 | 129.2 | 148.1 |
| frame542 | 711 | 749 | 724 | 1205 |

Per kernel the draws cost about the same wherever they run and only move:
on sanmiguel the trace launch grows 49.9 to 60.0 ms while the material
kernels lose 8.8 (diffuse 25.7 to 21.2, coated-diffuse 14.6 to 12.4,
dielectric 4.3 to 2.1); on frame542 the trace launch grows 285 to 433 ms
while the dielectric kernel loses 122 (194 to 72). So the sum is close to
conserved: a dense material kernel already hides the chain's latency, and
the one that did not (the dielectric kernel) was small in absolute terms
except on frame542, where pbrt's own structure costs pbrt 182 ms of sample
kernel at its launch floor. What the stage buys is parity of structure and
of per-stage accounting, not time: an in-place stage would cost about what
pbrt's sample kernel costs (8 ms on sanmiguel, 3.6 on killeroo) and break
about even with the draws left in the material kernels. The decision
(14:30): no in-place stage; the draws run at the top of the trace launch,
one pass over the ray queue, and everything else follows pbrt's structure
-- the next section.

### pbrt's structure, matched (2026-10-09, 14:30-23:30)

A *draw* here is one call to the sampler for a random number (pbrt's
`Get1D`/`Get2D`). The decision was pbrt's queues and kernels one for one,
with the one difference that the bounce's draws are made at the top of
the trace launch rather than in a kernel of their own, since the two-pass
form above costs more than it saves. The two structural differences the
audit left are in now.

**The texture-evaluator split.** pbrt keeps two material queues per
material type and a kernel for each: a hit whose material's textures are
all constants or images goes to the basic queue, whose kernel instantiates
the material with `BasicTextureEvaluator` and so carries only the leaf
texture readers; any other hit goes to the universal queue
(`basicEvalMaterialQueue`, `universalEvalMaterialQueue`, wavefront/
integrator.h; the test is each material's `CanEvaluateTextures`, run in
the closest-hit program). The program carries the same fact: `element
TextureEvaluator = | Basic | Universal` (textures.bonsai), a reader for
each, `material_evaluator` (bxdf.bonsai) transcribing pbrt's per-material
lists with the displacement, the trace routing each hit by it, and the
material functions taking the evaluator as a parameter -- which is what
lets the specialize shrink the basic kernels, as pbrt's `GetBxDF<TexEval>`
does. Schedule: `hits.specialize(evaluator); hits[Basic].specialize(
material); hits[Universal].specialize(material);`, so the kernels are
`hits_blk$Basic$<Material>` and `hits_blk$Universal$<Material>`. The
driver passes `textureevaluator_arms`, `basic_material_arms` and
`universal_material_arms`, pbrt's `haveBasicEvalMaterial[]` and
`haveUniversalEvalMaterial[]`, so a kernel of a (material, evaluator) pair
the scene does not have is never launched (the compiler's launch gates
had to learn a split under a split for this: 1d7b3aa6). Check: on
villa-daylight the nine (material, evaluator) kernels with work are
exactly pbrt's nine, on kroken the eight are pbrt's eight.

**The BSSRDF kernel.** `vol_surface.defer(vol_subsurface, bssrdfs)`
makes pbrt's "Get BSSRDF and enqueue probe ray" the drain of a queue the
material kernels push (`bssrdfs_blk`), ahead of the probe trace, instead
of the subsurface material kernel's tail. It reads the material's textures
with the basic evaluator for every entry, as pbrt's kernel does
(`GetBSSRDF(BasicTextureEvaluator(), ..)`, wavefront/subsurface.cpp),
whichever queue the hit came through. On head the two kernels cost 4.87
(subsurface material) + 3.34 (BSSRDF) = 8.2 ms against the folded
kernel's 9.1 in the sweep and pbrt's 5.37 + 2.95 = 8.3.

**Images:** byte-identical to the sweep's on killeroo-simple, head,
dragon_10, villa-lights-on, villa-daylight, kroken and watercolor at 16
and 64 spp, each matching pbrt by the compare's rule.

**Measured, first run (23:00, `scratch/surface/compare-struct`, 16 spp,
kernel ms; before the nested launch gates, so with every material kernel
under an evaluator the scene has launched empty every round):**

| scene | the sweep (10-09 morning) | pbrt's structure, ungated | pbrt |
|---|---|---|---|
| killeroo-simple | 34.1 | 36.5 | 49.3 |
| head | 34.6 | 40.7 | 45.4 |
| dragon_10 | 164 | 178 | 210 |
| villa-lights-on | 50.5 | 55.3 | 84.3 |
| villa-daylight | 185 | 209 | 291 |
| kroken | 764 | 949 | 1731 |
| watercolor | 838 | 931 | 2073 |

Where head's 6.1 ms went: the trace launch 8.17 to 11.87 ms (+3.7, the
draws now run there; pbrt's own sample kernel on head is 3.17 ms, so this
is the cost of pbrt's structure in our placement of it), ten empty
basic-material launches a round (+2.4, the gating gap, fixed), the probe
trace 5.78 to 6.68 (+0.9), the subsurface chain -0.9. On kroken and
watercolor the empty launches were 22 kernels a round at 4 us each.

**Second run, with the nested launch gates (23:40, `scratch/surface/
compare-struct2`, kernel ms; images identical again; the GPU clock
unlocked, as every number before 10-10 is):**

| scene | spp | the sweep | pbrt's structure | pbrt | pbrt's / ours |
|---|---|---|---|---|---|
| killeroo-simple | 16 | 34.1 | 32.8 | 49.2 | 1.50 |
| killeroo-simple | 64 | 137 | 131 | 197 | 1.51 |
| head | 16 | 34.6 | 38.4 | 45.2 | 1.18 |
| head | 64 | 138 | 154 | 181 | 1.18 |
| dragon_10 | 16 | 164 | 168 | 210 | 1.25 |
| dragon_10 | 64 | 656 | 674 | 843 | 1.25 |
| villa-lights-on | 16 | 50.5 | 50.1 | 84.5 | 1.69 |
| villa-lights-on | 64 | 202 | 201 | 337 | 1.68 |
| villa-daylight | 16 | 185 | 186 | 290 | 1.56 |
| villa-daylight | 64 | 740 | 743 | 1165 | 1.57 |
| kroken | 16 | 764 | 896 | 1714 | 1.91 |
| kroken | 64 | 3061 | 3211 | 5827 | 1.81 |
| watercolor | 16 | 838 | 877 | 2058 | 2.35 |
| watercolor | 64 | 3350 | 3507 | 8084 | 2.30 |

So with the gates back, pbrt's structure costs what the sweep's did on
the villas and less on killeroo (the ungated launches were the whole of
their first-run loss), head is +3.8 ms (its trace launch 8.2 to 11.9),
dragon +4.7, watercolor +39 (4.6%) and kroken +132 (17%), its trace launch
193 to 299 ms. **But the sweep is not a like-for-like baseline.** Its
binaries were built in scratch/wt-fix with the uncommitted mesh-entity
series of 10-08 (PLAN.md, "The mesh entity built and measured"), which
the committed tree does not have: that series made the mesh the entity
the hit refers to, as pbrt's GPU does, moved the ray launches by a third
to a half (head's 22.7 to 16.2 ms, dragon's 170 to 106 at the time), and
took the per-triangle primitive records out of the scene. So the trace
launches above compare a tree without the series against one with it,
and the draws' share of the difference is not known from these numbers;
it will be once the series is in.

Kroken's loss was traced (00:20-01:00, nsys and a memory sampler on one
render of each side). The scene is the same on both sides: pbrt's own
statistics count 287.9 million triangles in 248 meshes, 247.7 million of
them made by its displacement tessellation at load, and the dump holds
2.6 million shapes plus ten instances of objects holding 285.3 million --
the same 287.9 million. What differs is the record per triangle. Here each
triangle is a primitive of its own: a Geometric record (the shape's
tagged index, material, light, alpha, two media; 32 bytes) and a triangle
record (mesh and index, 8 bytes), 11.5 GB over the 288 million, where
pbrt's GPU keeps one material-and-media record per mesh in the shader
binding table and lets the hardware name the triangle. So the scene takes
about 38 GB of device memory here against pbrt's 27.2 GB (nvidia-smi's
peak during a render: 32.2 of 32.6 GB on our side, which is the card
full, 27.2 on pbrt's), and with the material queues doubled (22 queues of
302 bytes an entry at the band's capacity, 7.7 GiB of queue arrays) the
managed memory oversubscribes and thrashes: the per-render prefetch of
the 1097 queue arrays takes 293 ms of host time (cuMemPrefetchAsync, 267
us each on average) and the launches page-fault -- the 0.3 s of kroken's
wall time that its kernels do not account for (1.20 s against 0.90), and
some part of its trace launch's growth. With the sweep's eleven material
queues, and the mesh-entity series in those binaries, the scene fit.
Watercolor, with 5 GB of geometry, does not thrash (47 ms of host time per
render). The mesh entity is the per-mesh record, as pbrt has it: it
removes the 11.5 GB. (The dump file on disk wrote the placement per
triangle too, 21 GB on kroken; since 01:30 on 10-10 it is per mesh, as
pbrt's TriangleMesh carries it, with no record per triangle and no tree
of pbrt's: landscape's file went from 2.16 to 1.12 GB, head's to half a
megabyte, kroken's from 21.2 to 10.6, and killeroo, head and landscape
render pixel for pixel as before.)

**Resolved (10-10, 03:10): the instanced mesh as a primitive.** The
mesh-entity series left an instance's object per triangle, and kroken's
285 million triangles are all inside ten instances, so the series alone
left the card full (32.2 GB again). pbrt's GPU places one structure per
definition under each OptixInstance and reaches the mesh's own record
through the instance's SBT offset; the same here, without a third tree
level in the program (PLAN.md, "The instanced mesh as a primitive"): a
`Placed` primitive per instance use and mesh -- the instance's two
transforms, the mesh, the mesh's run of `tris` -- the definition's
structure built once with a build input per mesh and placed by every
instance of it, its records naming the instance's placed meshes
consecutively (runtime/bonsai_optix.h, `holder_per_input`). The
definition's other shapes stay the instance's `blas`. So no record per
triangle is left on the device, and a hit through an instanced mesh costs
what a hit through a scene's own mesh does plus one pool row for the
transforms, as pbrt's costs its instance node.

| 16 spp, kernel ms | the sweep (10-09 morning) | pbrt's structure, 23:40 | pbrt's representation, 03:10 | pbrt |
|---|---|---|---|---|
| killeroo-simple | 34.1 | 32.8 | 36.1 | 49.5 |
| head | 34.6 | 38.4 | 39.6 | 45.7 |
| landscape | 1008 | -- | 1012 | 1064 |
| kroken | 764 | 896 | 779 | 1733 |

Kroken's wall time is 0.80 s for 0.78 s of kernels now (1.20 for 0.90
before), its camera-ray kernel back from 29 to 14 ms, its trace launch
236 ms against the sweep's 193 -- the draws' placement, still to be read
off a sweep whose two sides are the same tree. The card's peak over the
whole compare, pbrt's renders included, is 27.2 GB, pbrt's own figure.
Every image matches pbrt. Tests: correctness/gpu/rtcore-placed (the
definition of two meshes placed twice, with consecutive holders) and
correctness/cpp/placed-tlas (the same on the CPU trees).

Two smaller findings from the same trace. Each round's host work is 54
device writes of a queue's count (cuMemsetD32Async, 0.9 us each) plus the
round guard's synchronous read (4 us idle); 38 ms per kroken render, where
pbrt zeroes every queue in one "Reset queues" launch -- the material
queues' counts are written twice a round, before the trace and again after
their pass. And the camera-ray kernel doubled on kroken (14 to 29 ms)
while unchanged on head, which the thrash explains as well. The decision
on the draws stands as taken; what it costs is to be read from a sweep
whose two sides are the same tree.

What still differs from pbrt: the draws' placement (the decision above);
the material kernels launched for the last depth, where pbrt leaves its
loop before them (ours makes one round of launches more per sample:
dragon 192 against 160 per kernel, at the empty-launch floor); and the
round loop's per-round readback of the ray queue's count, where pbrt runs
a fixed number of rounds (the question above). All three are open items
in PLAN.md.

<!-- stages:begin -->
Generated by `report_stages.py` from `compare-out-heatmap4` (30 scenes; kernel times at 16 spp).

#### Every scene

| scene | image | kernel ours | kernel pbrt | ratio | wall 16 spp | wall 64 spp | wall 128 spp |
|---|---|---|---|---|---|---|---|
| killeroos/killeroo-simple | ok | 35.8 | 49.2 | 0.73 | 2.56x | 1.58x | 1.28x |
| killeroos/killeroo-gold | ok | 92.0 | 148.4 | 0.62 | 1.62x | 1.51x | 1.50x |
| killeroos/killeroo-coated-gold | ok | 105.4 | 163.3 | 0.65 | 1.87x | 1.55x | 1.54x |
| pbrt-book/book | ok | 116.3 | 153.9 | 0.76 | 1.77x | 1.44x | 1.32x |
| ganesha/ganesha | ok | 119.8 | 177.7 | 0.67 | 1.72x | 1.59x | 1.47x |
| barcelona-pavilion/pavilion-day | **FAILED** | 139.2 | 179.1 | 0.78 | 1.66x | 1.43x | 1.32x |
| lte-orb/lte-orb-simple-ball | ok | 176.9 | 212.7 | 0.83 | 1.46x | 1.24x | 1.22x |
| lte-orb/lte-orb-silver | ok | 95.8 | 161.3 | 0.59 | 1.99x | 1.70x | 1.57x |
| lte-orb/lte-orb-rough-glass | ok | 218.1 | 336.7 | 0.65 | 1.43x | 1.39x | 1.39x |
| head/head | ok | 39.4 | 45.2 | 0.87 | 1.75x | 1.33x | 1.25x |
| sssdragon/dragon_10 | ok | 173.1 | 210.4 | 0.82 | 1.54x | 1.28x | 1.18x |
| villa/villa-lights-on | ok | 53.9 | 84.8 | 0.64 | 2.17x | 1.62x | 1.58x |
| smoke-plume/plume | ok | 230.5 | 301.3 | 0.76 | 1.60x | 1.39x | 1.32x |
| bunny-cloud/bunny-cloud | ok | 2157 | 3069 | 0.70 | 1.41x | 1.39x | 1.39x |
| explosion/explosion | ok | 538.0 | 686.5 | 0.78 | 1.33x | 1.29x | 1.28x |
| disney-cloud/disney-cloud | ok | 5280 | 7029 | 0.75 | 1.34x | 1.33x | 1.32x |
| landscape/view-0 | ok | 1016 | 1064 | 0.95 | 1.13x | 1.06x | 1.04x |
| crown/crown | ok | 834.5 | 1225 | 0.68 | 1.45x | 1.37x | 1.37x |
| kroken/camera-1 | ok | 778.0 | 1742 | 0.45 | 2.11x | 1.82x | 1.78x |
| sportscar/sportscar-sky | ok | 187.5 | 222.4 | 0.84 | 1.49x | 1.21x | 1.18x |
| bmw-m6/bmw-m6 | ok | 131.6 | 213.7 | 0.62 | 2.00x | 1.64x | 1.61x |
| dambreak/dambreak0 | ok | 467.0 | 690.5 | 0.68 | 1.54x | 1.36x | 1.33x |
| transparent-machines/frame542 | ok | 826.3 | 1207 | 0.68 | 1.17x | 1.13x | 1.12x |
| villa/villa-daylight | ok | 191.5 | 291.6 | 0.66 | 1.70x | 1.54x | 1.52x |
| clouds/clouds | ok | 677.0 | 903.8 | 0.75 | 1.38x | 1.25x | 1.26x |
| sanmiguel/sanmiguel-courtyard | ok | 125.5 | 149.6 | 0.84 | 1.74x | 1.37x | 1.24x |
| bistro/bistro_cafe | **FAILED** | 217.3 | 316.1 | 0.69 | 1.91x | 1.53x | 1.48x |
| bistro/bistro_vespa | **FAILED** | 227.2 | 325.3 | 0.70 | 1.82x | 1.57x | 1.47x |
| watercolor/camera-1 | ok | 888.0 | 2048 | 0.43 | 2.48x | 2.28x | 2.27x |
| sanmiguel/sanmiguel-courtyard-second | ok | 166.1 | 193.9 | 0.86 | 1.40x | 1.21x | 1.18x |

The wall speedup is pbrt's process over ours, both warm, from table.tsv. Images: 27 of 30 scenes match pbrt at 16 spp; a FAILED row's times are reported but do not count as beaten.

#### Every stage

| stage | scenes | behind on | worst | sum ours | sum pbrt | ratio of sums |
|---|---|---|---|---|---|---|
| closest-hit trace + routing | 30 | 26 | 1.73 (transparent-machines/frame542) | 4036 | 3176 | 1.27 |
| shadow traces (every pass) | 30 | 2 | 1.23 (bistro/bistro_cafe) | 4415 | 5297 | 0.83 |
| sample draws (the bounce's dimensions, per ray) | 30 | 0 | - | 0 | 798.2 | pbrt only |
| camera rays | 30 | 29 | 1.79 (lte-orb/lte-orb-silver) | 331.1 | 245.4 | 1.35 |
| escaped rays | 30 | 0 | 0.65 (villa/villa-lights-on) | 205.1 | 407.5 | 0.50 |
| emitters hit | 30 | 0 | 0.61 (kroken/camera-1) | 146.2 | 385.7 | 0.38 |
| material: CoatedDiffuse (basic textures) | 21 | 0 | 0.70 (landscape/view-0) | 913.2 | 1647 | 0.55 |
| material: Diffuse (basic textures) | 22 | 0 | 0.70 (sanmiguel/sanmiguel-courtyard) | 399.0 | 947.4 | 0.42 |
| film + the rest | 30 | 0 | 0.53 (pbrt-book/book) | 83.6 | 817.7 | 0.10 |
| materials with their sample draws (a view, not a stage) | 30 | 0 | 0.81 (lte-orb/lte-orb-simple-ball) | 2709 | 6566 | 0.41 |
| material: Conductor (basic textures) | 11 | 0 | 0.30 (villa/villa-daylight) | 141.2 | 550.6 | 0.26 |
| material: CoatedConductor (basic textures) | 6 | 0 | 0.56 (bistro/bistro_cafe) | 224.5 | 472.8 | 0.47 |
| material: Dielectric (basic textures) | 16 | 0 | 0.75 (barcelona-pavilion/pavilion-day) | 223.5 | 423.5 | 0.53 |
| material: DiffuseTransmission (basic textures) | 8 | 0 | 0.69 (landscape/view-0) | 158.1 | 320.9 | 0.49 |
| material: Measured (basic textures) | 3 | 2 | 1.07 (sportscar/sportscar-sky) | 180.8 | 175.5 | 1.03 |
| subsurface probe trace | 2 | 0 | 0.85 (sssdragon/dragon_10) | 46.9 | 55.3 | 0.85 |
| subsurface exit | 2 | 0 | 0.53 (head/head) | 11.7 | 22.9 | 0.51 |
| BSSRDF sampling (the probe enqueued) | 2 | 2 | 1.16 (head/head) | 9.31 | 8.63 | 1.08 |
| material: Subsurface (basic textures) | 2 | 1 | 1.15 (sssdragon/dragon_10) | 12.0 | 11.4 | 1.05 |
| material: Diffuse (pbrt only) | 8 | 0 | - | 0 | 177.2 | pbrt only |
| material: ThinDielectric (basic textures) | 2 | 0 | 0.68 (villa/villa-lights-on) | 6.84 | 10.5 | 0.65 |
| material: CoatedDiffuse (universal textures) | 4 | 0 | 0.61 (kroken/camera-1) | 234.9 | 544.9 | 0.43 |
| material: Dielectric (universal textures) | 2 | 0 | 0.68 (villa/villa-lights-on) | 8.29 | 12.4 | 0.67 |
| material: Diffuse (universal textures) | 4 | 0 | 0.70 (villa/villa-daylight) | 186.3 | 473.6 | 0.39 |
| medium sampling | 9 | 0 | 0.70 (clouds/clouds) | 4093 | 6153 | 0.67 |
| medium scattering | 9 | 0 | 0.69 (smoke-plume/plume) | 216.7 | 467.4 | 0.46 |
| material: Mixed (basic textures) | 4 | 0 | - | 20.4 | 0 | ours only |

A stage is beaten when ours is below pbrt's on every scene that has it. "behind on" counts the scenes where it is not; the sums are over all the scenes that have the stage on either side.

##### closest-hit trace + routing

| scene | ours | pbrt | ratio |
|---|---|---|---|
| transparent-machines/frame542 | 540.5 | 312.7 | **1.73** |
| disney-cloud/disney-cloud | 360.9 | 244.1 | **1.48** |
| crown/crown | 342.4 | 235.4 | **1.45** |
| head/head | 13.5 | 10.0 | **1.35** |
| clouds/clouds | 192.7 | 146.1 | **1.32** |
| bunny-cloud/bunny-cloud | 200.4 | 152.5 | **1.31** |
| sssdragon/dragon_10 | 54.7 | 41.8 | **1.31** |
| landscape/view-0 | 544.5 | 419.9 | **1.30** |
| smoke-plume/plume | 87.3 | 67.4 | **1.30** |
| kroken/camera-1 | 236.3 | 196.1 | **1.21** |
| pbrt-book/book | 63.2 | 52.6 | **1.20** |
| explosion/explosion | 92.5 | 77.8 | **1.19** |
| watercolor/camera-1 | 174.5 | 148.6 | **1.17** |
| ganesha/ganesha | 56.9 | 49.0 | **1.16** |
| villa/villa-lights-on | 21.6 | 18.7 | **1.15** |
| sanmiguel/sanmiguel-courtyard-second | 84.4 | 73.7 | **1.14** |
| dambreak/dambreak0 | 235.8 | 208.2 | **1.13** |
| sanmiguel/sanmiguel-courtyard | 62.6 | 55.4 | **1.13** |
| killeroos/killeroo-simple | 18.4 | 16.6 | **1.11** |
| killeroos/killeroo-gold | 52.1 | 47.2 | **1.10** |
| killeroos/killeroo-coated-gold | 50.1 | 45.9 | **1.09** |
| lte-orb/lte-orb-simple-ball | 46.1 | 42.6 | **1.08** |
| barcelona-pavilion/pavilion-day (image FAILED) | 57.0 | 52.7 | **1.08** |
| villa/villa-daylight | 66.3 | 62.8 | **1.06** |
| bistro/bistro_vespa (image FAILED) | 79.6 | 76.4 | **1.04** |
| bmw-m6/bmw-m6 | 47.1 | 46.7 | **1.01** |
| sportscar/sportscar-sky | 64.9 | 66.0 | 0.98 |
| bistro/bistro_cafe (image FAILED) | 68.8 | 73.9 | 0.93 |
| lte-orb/lte-orb-rough-glass | 81.9 | 90.2 | 0.91 |
| lte-orb/lte-orb-silver | 39.2 | 45.5 | 0.86 |

##### shadow traces (every pass)

ours: `__raygen__shadow$<alpha>` and `__raygen__exit_shadow$<alpha>`; pbrt: the shadow and shadow-Tr trace kernels.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| bistro/bistro_cafe (image FAILED) | 27.3 | 22.2 | **1.23** |
| bistro/bistro_vespa (image FAILED) | 30.6 | 27.6 | **1.11** |
| transparent-machines/frame542 | 136.6 | 140.8 | 0.97 |
| head/head | 2.13 | 2.23 | 0.96 |
| explosion/explosion | 139.9 | 148.2 | 0.94 |
| barcelona-pavilion/pavilion-day (image FAILED) | 15.6 | 17.0 | 0.91 |
| clouds/clouds | 194.0 | 214.6 | 0.90 |
| sssdragon/dragon_10 | 14.0 | 15.6 | 0.90 |
| smoke-plume/plume | 40.5 | 45.2 | 0.90 |
| sanmiguel/sanmiguel-courtyard | 18.4 | 20.8 | 0.88 |
| landscape/view-0 | 206.0 | 236.3 | 0.87 |
| disney-cloud/disney-cloud | 2118 | 2453 | 0.86 |
| sanmiguel/sanmiguel-courtyard-second | 24.9 | 28.9 | 0.86 |
| watercolor/camera-1 | 63.2 | 73.4 | 0.86 |
| lte-orb/lte-orb-simple-ball | 8.73 | 10.2 | 0.86 |
| ganesha/ganesha | 7.36 | 8.90 | 0.83 |
| villa/villa-daylight | 9.67 | 11.8 | 0.82 |
| villa/villa-lights-on | 2.75 | 3.38 | 0.81 |
| lte-orb/lte-orb-rough-glass | 22.7 | 28.7 | 0.79 |
| kroken/camera-1 | 93.0 | 117.7 | 0.79 |
| bunny-cloud/bunny-cloud | 1057 | 1416 | 0.75 |
| sportscar/sportscar-sky | 9.65 | 12.9 | 0.75 |
| dambreak/dambreak0 | 42.5 | 57.7 | 0.74 |
| crown/crown | 93.1 | 127.6 | 0.73 |
| bmw-m6/bmw-m6 | 7.13 | 9.91 | 0.72 |
| lte-orb/lte-orb-silver | 6.34 | 9.11 | 0.70 |
| killeroos/killeroo-coated-gold | 6.81 | 9.84 | 0.69 |
| killeroos/killeroo-gold | 6.95 | 10.1 | 0.69 |
| killeroos/killeroo-simple | 3.15 | 4.65 | 0.68 |
| pbrt-book/book | 7.43 | 12.7 | 0.59 |

##### sample draws (the bounce's dimensions, per ray)

ours: the `rays` drain of the sample stage (10-09); a run before it has no kernel here, its draws having run inside the material and medium kernels, so pbrt's stands alone there.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| killeroos/killeroo-simple | 0 | 3.68 | pbrt only |
| killeroos/killeroo-gold | 0 | 9.16 | pbrt only |
| killeroos/killeroo-coated-gold | 0 | 8.91 | pbrt only |
| pbrt-book/book | 0 | 10.4 | pbrt only |
| ganesha/ganesha | 0 | 9.74 | pbrt only |
| barcelona-pavilion/pavilion-day (image FAILED) | 0 | 8.07 | pbrt only |
| lte-orb/lte-orb-simple-ball | 0 | 8.80 | pbrt only |
| lte-orb/lte-orb-silver | 0 | 6.13 | pbrt only |
| lte-orb/lte-orb-rough-glass | 0 | 18.1 | pbrt only |
| head/head | 0 | 3.20 | pbrt only |
| sssdragon/dragon_10 | 0 | 11.9 | pbrt only |
| villa/villa-lights-on | 0 | 3.34 | pbrt only |
| smoke-plume/plume | 0 | 13.4 | pbrt only |
| bunny-cloud/bunny-cloud | 0 | 49.7 | pbrt only |
| explosion/explosion | 0 | 21.7 | pbrt only |
| disney-cloud/disney-cloud | 0 | 56.9 | pbrt only |
| landscape/view-0 | 0 | 38.8 | pbrt only |
| crown/crown | 0 | 78.5 | pbrt only |
| kroken/camera-1 | 0 | 34.6 | pbrt only |
| sportscar/sportscar-sky | 0 | 11.3 | pbrt only |
| bmw-m6/bmw-m6 | 0 | 10.7 | pbrt only |
| dambreak/dambreak0 | 0 | 60.6 | pbrt only |
| transparent-machines/frame542 | 0 | 181.8 | pbrt only |
| villa/villa-daylight | 0 | 12.7 | pbrt only |
| clouds/clouds | 0 | 58.8 | pbrt only |
| sanmiguel/sanmiguel-courtyard | 0 | 7.75 | pbrt only |
| bistro/bistro_cafe (image FAILED) | 0 | 11.6 | pbrt only |
| bistro/bistro_vespa (image FAILED) | 0 | 11.9 | pbrt only |
| watercolor/camera-1 | 0 | 26.7 | pbrt only |
| sanmiguel/sanmiguel-courtyard-second | 0 | 9.20 | pbrt only |

##### camera rays

both sides draw the camera's own dimensions here.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| lte-orb/lte-orb-silver | 8.52 | 4.77 | **1.79** |
| lte-orb/lte-orb-rough-glass | 8.44 | 4.73 | **1.78** |
| ganesha/ganesha | 12.7 | 9.16 | **1.38** |
| bunny-cloud/bunny-cloud | 12.5 | 9.08 | **1.38** |
| sportscar/sportscar-sky | 12.6 | 9.16 | **1.37** |
| dambreak/dambreak0 | 12.4 | 9.08 | **1.37** |
| pbrt-book/book | 12.4 | 9.14 | **1.35** |
| landscape/view-0 | 49.7 | 36.8 | **1.35** |
| villa/villa-daylight | 9.55 | 7.13 | **1.34** |
| bistro/bistro_cafe (image FAILED) | 12.7 | 9.46 | **1.34** |
| clouds/clouds | 12.7 | 9.52 | **1.34** |
| transparent-machines/frame542 | 9.77 | 7.34 | **1.33** |
| bistro/bistro_vespa (image FAILED) | 12.7 | 9.63 | **1.31** |
| killeroos/killeroo-coated-gold | 8.59 | 6.58 | **1.31** |
| sanmiguel/sanmiguel-courtyard-second | 9.33 | 7.20 | **1.30** |
| killeroos/killeroo-gold | 8.64 | 6.67 | **1.30** |
| crown/crown | 8.52 | 6.59 | **1.29** |
| explosion/explosion | 14.3 | 11.1 | **1.29** |
| smoke-plume/plume | 8.73 | 6.78 | **1.29** |
| sssdragon/dragon_10 | 8.33 | 6.53 | **1.28** |
| watercolor/camera-1 | 17.2 | 13.5 | **1.27** |
| bmw-m6/bmw-m6 | 8.37 | 6.60 | **1.27** |
| barcelona-pavilion/pavilion-day (image FAILED) | 8.13 | 6.58 | **1.24** |
| lte-orb/lte-orb-simple-ball | 8.84 | 7.17 | **1.23** |
| sanmiguel/sanmiguel-courtyard | 6.75 | 5.50 | **1.23** |
| head/head | 3.52 | 2.89 | **1.22** |
| kroken/camera-1 | 14.0 | 11.7 | **1.20** |
| killeroos/killeroo-simple | 3.02 | 2.53 | **1.19** |
| villa/villa-lights-on | 2.69 | 2.43 | **1.11** |
| disney-cloud/disney-cloud | 5.50 | 0 | ours only |

##### escaped rays

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 0.70 | 1.08 | 0.65 |
| head/head | 1.77 | 2.88 | 0.61 |
| sssdragon/dragon_10 | 3.90 | 6.60 | 0.59 |
| explosion/explosion | 5.74 | 9.87 | 0.58 |
| landscape/view-0 | 14.0 | 24.5 | 0.57 |
| barcelona-pavilion/pavilion-day (image FAILED) | 2.49 | 4.36 | 0.57 |
| bmw-m6/bmw-m6 | 4.16 | 7.52 | 0.55 |
| smoke-plume/plume | 4.19 | 7.95 | 0.53 |
| lte-orb/lte-orb-simple-ball | 3.76 | 7.40 | 0.51 |
| ganesha/ganesha | 4.48 | 8.93 | 0.50 |
| villa/villa-daylight | 2.72 | 5.49 | 0.50 |
| clouds/clouds | 17.2 | 34.9 | 0.49 |
| sanmiguel/sanmiguel-courtyard | 1.49 | 3.04 | 0.49 |
| sportscar/sportscar-sky | 5.63 | 11.6 | 0.48 |
| disney-cloud/disney-cloud | 12.4 | 26.8 | 0.46 |
| sanmiguel/sanmiguel-courtyard-second | 1.47 | 3.23 | 0.46 |
| watercolor/camera-1 | 11.2 | 25.0 | 0.45 |
| bistro/bistro_cafe (image FAILED) | 1.90 | 4.26 | 0.45 |
| bunny-cloud/bunny-cloud | 15.1 | 34.2 | 0.44 |
| bistro/bistro_vespa (image FAILED) | 1.89 | 4.34 | 0.44 |
| dambreak/dambreak0 | 16.7 | 40.1 | 0.42 |
| kroken/camera-1 | 6.82 | 17.0 | 0.40 |
| transparent-machines/frame542 | 41.0 | 116.4 | 0.35 |
| killeroos/killeroo-simple | 0.39 | 0 | ours only |
| killeroos/killeroo-gold | 2.42 | 0 | ours only |
| killeroos/killeroo-coated-gold | 2.39 | 0 | ours only |
| pbrt-book/book | 0.98 | 0 | ours only |
| lte-orb/lte-orb-silver | 0.85 | 0 | ours only |
| lte-orb/lte-orb-rough-glass | 4.30 | 0 | ours only |
| crown/crown | 13.1 | 0 | ours only |

##### emitters hit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 9.90 | 16.1 | 0.61 |
| bistro/bistro_vespa (image FAILED) | 3.10 | 5.06 | 0.61 |
| bistro/bistro_cafe (image FAILED) | 2.98 | 4.90 | 0.61 |
| crown/crown | 26.4 | 46.6 | 0.57 |
| lte-orb/lte-orb-rough-glass | 7.46 | 13.3 | 0.56 |
| ganesha/ganesha | 2.15 | 3.89 | 0.55 |
| lte-orb/lte-orb-silver | 1.79 | 3.35 | 0.53 |
| killeroos/killeroo-simple | 0.52 | 0.98 | 0.53 |
| villa/villa-lights-on | 0.48 | 0.94 | 0.51 |
| head/head | 0.25 | 0.51 | 0.49 |
| barcelona-pavilion/pavilion-day (image FAILED) | 1.08 | 2.29 | 0.47 |
| sssdragon/dragon_10 | 1.08 | 2.30 | 0.47 |
| bmw-m6/bmw-m6 | 1.55 | 3.46 | 0.45 |
| smoke-plume/plume | 1.03 | 2.42 | 0.43 |
| sanmiguel/sanmiguel-courtyard | 0.81 | 1.92 | 0.42 |
| sanmiguel/sanmiguel-courtyard-second | 1.00 | 2.58 | 0.39 |
| villa/villa-daylight | 1.85 | 4.84 | 0.38 |
| killeroos/killeroo-coated-gold | 1.31 | 3.46 | 0.38 |
| killeroos/killeroo-gold | 1.32 | 3.50 | 0.38 |
| pbrt-book/book | 1.18 | 3.19 | 0.37 |
| sportscar/sportscar-sky | 1.20 | 3.26 | 0.37 |
| watercolor/camera-1 | 4.25 | 11.6 | 0.37 |
| lte-orb/lte-orb-simple-ball | 0.86 | 2.36 | 0.36 |
| disney-cloud/disney-cloud | 8.89 | 25.2 | 0.35 |
| explosion/explosion | 1.38 | 3.96 | 0.35 |
| dambreak/dambreak0 | 11.5 | 33.7 | 0.34 |
| clouds/clouds | 8.99 | 26.6 | 0.34 |
| bunny-cloud/bunny-cloud | 8.98 | 26.9 | 0.33 |
| landscape/view-0 | 4.00 | 12.8 | 0.31 |
| transparent-machines/frame542 | 28.9 | 113.7 | 0.25 |

##### material: CoatedDiffuse (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 72.3 | 103.0 | 0.70 |
| crown/crown | 134.0 | 204.7 | 0.65 |
| explosion/explosion | 19.7 | 30.1 | 0.65 |
| sanmiguel/sanmiguel-courtyard-second | 14.2 | 21.9 | 0.65 |
| killeroos/killeroo-simple | 4.83 | 7.52 | 0.64 |
| sssdragon/dragon_10 | 26.5 | 41.5 | 0.64 |
| sanmiguel/sanmiguel-courtyard | 11.4 | 17.8 | 0.64 |
| sportscar/sportscar-sky | 8.72 | 14.2 | 0.61 |
| pbrt-book/book | 13.4 | 22.6 | 0.59 |
| barcelona-pavilion/pavilion-day (image FAILED) | 23.8 | 40.5 | 0.59 |
| lte-orb/lte-orb-rough-glass | 57.7 | 99.7 | 0.58 |
| bunny-cloud/bunny-cloud | 67.5 | 118.4 | 0.57 |
| villa/villa-daylight | 37.9 | 68.5 | 0.55 |
| bistro/bistro_cafe (image FAILED) | 82.3 | 148.9 | 0.55 |
| lte-orb/lte-orb-silver | 27.4 | 50.5 | 0.54 |
| bistro/bistro_vespa (image FAILED) | 75.8 | 143.9 | 0.53 |
| kroken/camera-1 | 85.9 | 166.1 | 0.52 |
| bmw-m6/bmw-m6 | 29.6 | 58.9 | 0.50 |
| ganesha/ganesha | 23.0 | 48.3 | 0.48 |
| villa/villa-lights-on | 9.57 | 22.2 | 0.43 |
| watercolor/camera-1 | 87.8 | 218.0 | 0.40 |

##### material: Diffuse (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sanmiguel/sanmiguel-courtyard | 17.0 | 24.3 | 0.70 |
| landscape/view-0 | 16.7 | 24.2 | 0.69 |
| sanmiguel/sanmiguel-courtyard-second | 23.0 | 33.6 | 0.68 |
| villa/villa-daylight | 6.54 | 10.3 | 0.64 |
| bmw-m6/bmw-m6 | 4.36 | 6.97 | 0.63 |
| barcelona-pavilion/pavilion-day (image FAILED) | 3.85 | 6.18 | 0.62 |
| sportscar/sportscar-sky | 2.63 | 4.27 | 0.62 |
| smoke-plume/plume | 9.02 | 14.7 | 0.61 |
| lte-orb/lte-orb-silver | 2.45 | 4.56 | 0.54 |
| dambreak/dambreak0 | 39.0 | 74.4 | 0.52 |
| lte-orb/lte-orb-rough-glass | 9.78 | 19.3 | 0.51 |
| villa/villa-lights-on | 1.40 | 2.85 | 0.49 |
| crown/crown | 55.6 | 114.6 | 0.49 |
| disney-cloud/disney-cloud | 21.5 | 46.1 | 0.47 |
| killeroos/killeroo-simple | 4.80 | 11.4 | 0.42 |
| lte-orb/lte-orb-simple-ball | 12.4 | 29.6 | 0.42 |
| pbrt-book/book | 14.5 | 37.3 | 0.39 |
| watercolor/camera-1 | 66.9 | 204.0 | 0.33 |
| killeroos/killeroo-gold | 12.7 | 39.7 | 0.32 |
| killeroos/killeroo-coated-gold | 12.5 | 39.2 | 0.32 |
| kroken/camera-1 | 52.2 | 166.1 | 0.31 |
| ganesha/ganesha | 10.2 | 33.9 | 0.30 |

##### film + the rest

ours: `rays_rest_blk` and `q_blk`; pbrt: Update film, the queue resets, the stats updates, Evaluate Displacement, Other.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| pbrt-book/book | 3.22 | 6.08 | 0.53 |
| ganesha/ganesha | 3.15 | 6.00 | 0.53 |
| sportscar/sportscar-sky | 3.06 | 5.89 | 0.52 |
| bistro/bistro_vespa (image FAILED) | 3.06 | 5.95 | 0.51 |
| bistro/bistro_cafe (image FAILED) | 2.98 | 5.97 | 0.50 |
| landscape/view-0 | 10.8 | 22.4 | 0.49 |
| sanmiguel/sanmiguel-courtyard-second | 2.34 | 5.03 | 0.47 |
| explosion/explosion | 3.38 | 7.42 | 0.46 |
| lte-orb/lte-orb-silver | 2.10 | 4.68 | 0.45 |
| head/head | 0.75 | 1.71 | 0.44 |
| smoke-plume/plume | 2.09 | 4.94 | 0.42 |
| killeroos/killeroo-coated-gold | 2.05 | 4.86 | 0.42 |
| killeroos/killeroo-gold | 2.05 | 4.87 | 0.42 |
| barcelona-pavilion/pavilion-day (image FAILED) | 1.94 | 4.67 | 0.42 |
| sanmiguel/sanmiguel-courtyard | 1.59 | 4.13 | 0.38 |
| sssdragon/dragon_10 | 2.05 | 5.34 | 0.38 |
| villa/villa-daylight | 2.32 | 6.45 | 0.36 |
| bmw-m6/bmw-m6 | 1.96 | 5.45 | 0.36 |
| killeroos/killeroo-simple | 0.66 | 1.93 | 0.34 |
| villa/villa-lights-on | 0.63 | 1.85 | 0.34 |
| lte-orb/lte-orb-simple-ball | 3.53 | 13.3 | 0.26 |
| lte-orb/lte-orb-rough-glass | 2.07 | 12.2 | 0.17 |
| clouds/clouds | 2.95 | 19.9 | 0.15 |
| bunny-cloud/bunny-cloud | 2.83 | 20.6 | 0.14 |
| dambreak/dambreak0 | 2.99 | 23.9 | 0.13 |
| watercolor/camera-1 | 5.99 | 68.4 | 0.09 |
| crown/crown | 1.99 | 33.1 | 0.06 |
| disney-cloud/disney-cloud | 1.30 | 25.0 | 0.05 |
| transparent-machines/frame542 | 2.29 | 76.0 | 0.03 |
| kroken/camera-1 | 5.40 | 409.7 | 0.01 |

##### materials with their sample draws (a view, not a stage)

the material kernels above summed with the sample draws on each side -- pbrt's `Generate ray samples`, and our sample kernel where the run has one (until the sample stage of 10-09 the bounce's draws ran inside our material kernels) -- the like-for-like sum either way; a second view of rows already counted, left out of the totals.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| lte-orb/lte-orb-simple-ball | 105.1 | 129.6 | 0.81 |
| sportscar/sportscar-sky | 90.4 | 113.5 | 0.80 |
| landscape/view-0 | 186.6 | 311.7 | 0.60 |
| sanmiguel/sanmiguel-courtyard-second | 42.6 | 73.2 | 0.58 |
| barcelona-pavilion/pavilion-day (image FAILED) | 52.9 | 91.4 | 0.58 |
| sanmiguel/sanmiguel-courtyard | 33.9 | 58.7 | 0.58 |
| head/head | 4.98 | 8.89 | 0.56 |
| sssdragon/dragon_10 | 33.5 | 61.5 | 0.55 |
| bistro/bistro_cafe (image FAILED) | 100.7 | 195.5 | 0.51 |
| villa/villa-daylight | 99.1 | 193.1 | 0.51 |
| bistro/bistro_vespa (image FAILED) | 96.3 | 196.3 | 0.49 |
| lte-orb/lte-orb-rough-glass | 91.4 | 187.6 | 0.49 |
| bmw-m6/bmw-m6 | 61.3 | 134.1 | 0.46 |
| villa/villa-lights-on | 25.1 | 56.5 | 0.44 |
| crown/crown | 281.7 | 649.0 | 0.43 |
| dambreak/dambreak0 | 91.1 | 212.9 | 0.43 |
| killeroos/killeroo-simple | 9.63 | 22.6 | 0.43 |
| kroken/camera-1 | 382.9 | 906.0 | 0.42 |
| pbrt-book/book | 27.9 | 70.2 | 0.40 |
| lte-orb/lte-orb-silver | 37.0 | 93.9 | 0.39 |
| killeroos/killeroo-coated-gold | 34.1 | 92.8 | 0.37 |
| ganesha/ganesha | 33.1 | 91.9 | 0.36 |
| watercolor/camera-1 | 582.2 | 1623 | 0.36 |
| explosion/explosion | 19.7 | 55.2 | 0.36 |
| bunny-cloud/bunny-cloud | 67.5 | 194.9 | 0.35 |
| smoke-plume/plume | 10.9 | 37.2 | 0.29 |
| killeroos/killeroo-gold | 18.6 | 76.2 | 0.24 |
| disney-cloud/disney-cloud | 21.5 | 103.0 | 0.21 |
| transparent-machines/frame542 | 67.2 | 440.5 | 0.15 |
| clouds/clouds | 0 | 85.4 | pbrt only |

##### material: Conductor (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-daylight | 6.12 | 20.7 | 0.30 |
| kroken/camera-1 | 22.2 | 76.2 | 0.29 |
| sportscar/sportscar-sky | 2.96 | 10.8 | 0.27 |
| villa/villa-lights-on | 1.60 | 5.94 | 0.27 |
| bmw-m6/bmw-m6 | 4.71 | 17.6 | 0.27 |
| crown/crown | 53.3 | 200.6 | 0.27 |
| watercolor/camera-1 | 33.4 | 140.9 | 0.24 |
| lte-orb/lte-orb-silver | 7.22 | 32.7 | 0.22 |
| barcelona-pavilion/pavilion-day (image FAILED) | 1.91 | 8.73 | 0.22 |
| killeroos/killeroo-gold | 5.90 | 27.3 | 0.22 |
| smoke-plume/plume | 1.92 | 9.12 | 0.21 |

##### material: CoatedConductor (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| bistro/bistro_cafe (image FAILED) | 16.0 | 28.4 | 0.56 |
| bmw-m6/bmw-m6 | 17.0 | 31.7 | 0.54 |
| bistro/bistro_vespa (image FAILED) | 17.1 | 32.1 | 0.53 |
| kroken/camera-1 | 76.3 | 149.2 | 0.51 |
| killeroos/killeroo-coated-gold | 21.7 | 44.7 | 0.49 |
| watercolor/camera-1 | 76.3 | 186.7 | 0.41 |

##### material: Dielectric (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| barcelona-pavilion/pavilion-day (image FAILED) | 4.75 | 6.33 | 0.75 |
| dambreak/dambreak0 | 52.1 | 77.8 | 0.67 |
| sanmiguel/sanmiguel-courtyard-second | 1.90 | 2.96 | 0.64 |
| villa/villa-lights-on | 0.89 | 1.40 | 0.64 |
| sanmiguel/sanmiguel-courtyard | 1.88 | 2.99 | 0.63 |
| bistro/bistro_cafe (image FAILED) | 2.47 | 3.96 | 0.62 |
| sportscar/sportscar-sky | 2.16 | 3.55 | 0.61 |
| bistro/bistro_vespa (image FAILED) | 3.43 | 5.65 | 0.61 |
| villa/villa-daylight | 2.75 | 4.90 | 0.56 |
| bmw-m6/bmw-m6 | 4.52 | 8.24 | 0.55 |
| watercolor/camera-1 | 6.43 | 12.2 | 0.53 |
| crown/crown | 26.6 | 50.6 | 0.52 |
| kroken/camera-1 | 17.3 | 35.1 | 0.49 |
| lte-orb/lte-orb-rough-glass | 23.8 | 50.4 | 0.47 |
| transparent-machines/frame542 | 67.2 | 145.7 | 0.46 |
| landscape/view-0 | 5.31 | 11.6 | 0.46 |

##### material: DiffuseTransmission (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 92.2 | 134.0 | 0.69 |
| barcelona-pavilion/pavilion-day (image FAILED) | 4.45 | 6.73 | 0.66 |
| sanmiguel/sanmiguel-courtyard-second | 3.51 | 5.61 | 0.63 |
| sanmiguel/sanmiguel-courtyard | 3.59 | 5.82 | 0.62 |
| villa/villa-daylight | 4.38 | 8.35 | 0.52 |
| villa/villa-lights-on | 1.33 | 2.54 | 0.52 |
| watercolor/camera-1 | 21.2 | 69.0 | 0.31 |
| kroken/camera-1 | 27.4 | 88.9 | 0.31 |

##### material: Measured (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sportscar/sportscar-sky | 74.0 | 69.4 | **1.07** |
| lte-orb/lte-orb-simple-ball | 92.7 | 91.2 | **1.02** |
| barcelona-pavilion/pavilion-day (image FAILED) | 14.2 | 14.9 | 0.95 |

##### subsurface probe trace

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sssdragon/dragon_10 | 41.2 | 48.5 | 0.85 |
| head/head | 5.72 | 6.87 | 0.83 |

##### subsurface exit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 3.27 | 6.22 | 0.53 |
| sssdragon/dragon_10 | 8.43 | 16.7 | 0.51 |

##### BSSRDF sampling (the probe enqueued)

ours: the `bssrdfs` drain; pbrt: Get BSSRDF and enqueue probe ray.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 3.41 | 2.95 | **1.16** |
| sssdragon/dragon_10 | 5.90 | 5.68 | **1.04** |

##### material: Subsurface (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sssdragon/dragon_10 | 7.02 | 6.10 | **1.15** |
| head/head | 4.98 | 5.32 | 0.94 |

##### material: Diffuse (pbrt only)

pbrt launches a kernel for every material of the scene at every depth, reached by a ray or not; ours launches only for a non-empty queue, so a material no ray of ours reached has no kernel here. Each row says how many launches pbrt made and what each cost: at pbrt's launch floor of about 0.015 ms they were empty.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head (32 launches, 0.012 ms each) | 0 | 0.37 | pbrt only |
| sssdragon/dragon_10 (160 launches, 0.012 ms each) | 0 | 1.92 | pbrt only |
| bunny-cloud/bunny-cloud (1600 launches, 0.017 ms each) | 0 | 26.8 | pbrt only |
| explosion/explosion (240 launches, 0.014 ms each) | 0 | 3.31 | pbrt only |
| transparent-machines/frame542 (8192 launches, 0.014 ms each) | 0 | 112.9 | pbrt only |
| clouds/clouds (1600 launches, 0.017 ms each) | 0 | 26.5 | pbrt only |
| bistro/bistro_cafe (image FAILED) (160 launches, 0.017 ms each) | 0 | 2.67 | pbrt only |
| bistro/bistro_vespa (image FAILED) (160 launches, 0.017 ms each) | 0 | 2.70 | pbrt only |

##### material: ThinDielectric (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 1.33 | 1.95 | 0.68 |
| villa/villa-daylight | 5.51 | 8.56 | 0.64 |

##### material: CoatedDiffuse (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 67.8 | 111.0 | 0.61 |
| villa/villa-daylight | 22.4 | 39.5 | 0.57 |
| villa/villa-lights-on | 6.18 | 12.2 | 0.51 |
| watercolor/camera-1 | 138.5 | 382.2 | 0.36 |

##### material: Dielectric (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 1.56 | 2.29 | 0.68 |
| villa/villa-daylight | 6.73 | 10.1 | 0.67 |

##### material: Diffuse (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-daylight | 6.76 | 9.59 | 0.70 |
| villa/villa-lights-on | 1.19 | 1.73 | 0.69 |
| watercolor/camera-1 | 148.1 | 383.5 | 0.39 |
| kroken/camera-1 | 30.2 | 78.8 | 0.38 |

##### medium sampling

| scene | ours | pbrt | ratio |
|---|---|---|---|
| clouds/clouds | 210.1 | 298.7 | 0.70 |
| explosion/explosion | 255.9 | 364.7 | 0.70 |
| disney-cloud/disney-cloud | 2682 | 4010 | 0.67 |
| bunny-cloud/bunny-cloud | 759.3 | 1156 | 0.66 |
| crown/crown | 52.2 | 88.2 | 0.59 |
| watercolor/camera-1 | 11.9 | 20.2 | 0.59 |
| kroken/camera-1 | 15.8 | 27.1 | 0.58 |
| smoke-plume/plume | 69.2 | 120.0 | 0.58 |
| dambreak/dambreak0 | 37.7 | 67.3 | 0.56 |

##### medium scattering

| scene | ours | pbrt | ratio |
|---|---|---|---|
| smoke-plume/plume | 6.45 | 9.32 | 0.69 |
| explosion/explosion | 5.26 | 8.25 | 0.64 |
| bunny-cloud/bunny-cloud | 33.5 | 58.1 | 0.58 |
| clouds/clouds | 38.5 | 68.1 | 0.57 |
| disney-cloud/disney-cloud | 70.1 | 142.0 | 0.49 |
| dambreak/dambreak0 | 16.2 | 37.6 | 0.43 |
| crown/crown | 15.2 | 38.4 | 0.40 |
| kroken/camera-1 | 13.9 | 41.1 | 0.34 |
| watercolor/camera-1 | 17.5 | 64.5 | 0.27 |

##### material: Mixed (basic textures)

pbrt resolves a mix material inside its closest-hit program and has no kernel for it; ours is the choice step.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| crown/crown | 12.2 | 0 | ours only |
| kroken/camera-1 | 3.57 | 0 | ours only |
| bmw-m6/bmw-m6 | 1.13 | 0 | ours only |
| watercolor/camera-1 | 3.56 | 0 | ours only |

<!-- stages:end -->

## The three scenes measured right after the fix (2026-10-09, 06:55)

`scratch/surface/compare-fixed`: the fix plus the uncommitted mesh-entity
series of 10-08, driver 615.71.09, OptiX 9.1 (pbrt: OptiX 7.7); ours /
pbrt in ms, then the ratio. The 10-07 column is the compiler before the
fix.

### head (head/head) -- wall speedup 2.06x

| stage | ours 10-07 | ours 10-09 | pbrt | ratio |
|---|---|---|---|---|
| closest-hit trace + routing (`__raygen__rays` / Trace closest hit rays) | 20.4 | 8.2 | 10.0 | 0.82 |
| subsurface probe trace (`__raygen__probes` / Tracing subsurface scattering probe rays) | 37.3 | 5.8 | 7.0 | 0.83 |
| shadow traces (`__raygen__shadow$False` + `__raygen__exit_shadow$False` / Trace shadow rays, both passes) | 2.0 | 2.0 | 2.2 | 0.92 |
| subsurface material + BSSRDF sampling (`hits_blk$Subsurface` / SubsurfaceMaterial + BxDF eval, plus Get BSSRDF and enqueue probe ray: ours does both in one kernel) | 9.0 | 9.0 | 8.3 | 1.09 |
| subsurface exit (`exits_blk` / Handle out-scattering after SSS) | 3.2 | 3.9 | 6.5 | 0.60 |
| everything else (camera rays, escaped, emissive, film) | 6.1 | 5.6 | 11.5 | 0.49 |
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
| subsurface material + BSSRDF sampling (one kernel of ours against two of pbrt's) | | 14.5 | 11.8 | 1.23 |
| subsurface exit (`exits_blk` / Handle out-scattering after SSS) | | 8.4 | 16.6 | 0.51 |
| everything else | | 15.2 | 40.3 | 0.38 |
| **total kernel time** | 464 | **164** | 211 | **0.78** |

The 48 launches of ours against pbrt's 32 on these scenes are the final
depth's empty launches (the queue is empty there, as pbrt's loop ends
before its material pass), 0.016 ms each.

## The 10-07 matrix, the "before" (driver 615.71.09, compiler of 10-07)

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
touched (the shadow row only where the scene has alpha-tested geometry,
whose any-hit program binds elements: pavilion, sanmiguel, landscape).

## To do, in order

1. The stages behind, by the time behind: the dielectric material kernel
   (frame542 +48 ms, crown +13 ms, behind on 11 of 16 scenes; profiled
   above -- the sampler draws inside a kernel with few items, pending the
   user's decision on a sample stage, plus three compiler patterns), the
   closest-hit trace on landscape (+33 ms) and disney-cloud (+4 ms), the
   measured material (sportscar, 3 of 3), the diffuse material on its 7
   scenes, then subsurface, thin-dielectric, diffuse-transmission, medium
   scattering on plume, explosion and bunny-cloud, and the shadow traces
   on the bistros (which first need their image verdict settled).
2. The pbrt-GPU behavior audit (PLAN.md, "Design: the mesh is the entity",
   last paragraph): every place pbrt's wavefront deviates from its CPU
   path; the bistro lit-pixel bias sits there.
3. After each fix, the three-scene check (`gpu_compare.sh`) on the scene
   the fix was for plus head and killeroo-simple; a full sweep only in a
   window agreed with the other sessions on the machine.
