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

## Where it stands (2026-10-09, 09:30; the evening's addendum first)

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
Generated by `report_stages.py` from `compare-out-heatmap3` (30 scenes; kernel times at 16 spp).

#### Every scene

| scene | image | kernel ours | kernel pbrt | ratio | wall 16 spp | wall 64 spp | wall 128 spp |
|---|---|---|---|---|---|---|---|
| killeroos/killeroo-simple | ok | 34.1 | 49.4 | 0.69 | 2.88x | 1.41x | 1.55x |
| killeroos/killeroo-gold | ok | 88.7 | 149.2 | 0.59 | 1.76x | 1.66x | 1.61x |
| killeroos/killeroo-coated-gold | ok | 102.2 | 164.2 | 0.62 | 2.00x | 1.66x | 1.64x |
| pbrt-book/book | ok | 105.8 | 152.7 | 0.69 | 2.02x | 1.63x | 1.49x |
| ganesha/ganesha | ok | 120.9 | 177.2 | 0.68 | 1.79x | 1.63x | 1.51x |
| barcelona-pavilion/pavilion-day | **FAILED** | 148.5 | 179.1 | 0.83 | 1.61x | 1.39x | 1.26x |
| lte-orb/lte-orb-simple-ball | ok | 175.6 | 212.6 | 0.83 | 1.52x | 1.28x | 1.26x |
| lte-orb/lte-orb-silver | ok | 99.6 | 161.5 | 0.62 | 2.03x | 1.71x | 1.57x |
| lte-orb/lte-orb-rough-glass | ok | 226.8 | 336.2 | 0.67 | 1.48x | 1.43x | 1.42x |
| head/head | ok | 34.6 | 45.5 | 0.76 | 2.06x | 1.55x | 1.46x |
| sssdragon/dragon_10 | ok | 163.8 | 211.1 | 0.78 | 1.67x | 1.38x | 1.35x |
| villa/villa-lights-on | ok | 50.5 | 84.2 | 0.60 | 2.40x | 1.79x | 1.73x |
| smoke-plume/plume | ok | 218.6 | 301.4 | 0.73 | 1.73x | 1.49x | 1.42x |
| bunny-cloud/bunny-cloud | ok | 2145 | 3068 | 0.70 | 1.44x | 1.44x | 1.41x |
| explosion/explosion | ok | 525.9 | 689.0 | 0.76 | 1.38x | 1.34x | 1.33x |
| disney-cloud/disney-cloud | ok | 5253 | 7048 | 0.75 | 1.35x | 1.33x | 1.34x |
| landscape/view-0 | ok | 1008 | 1064 | 0.95 | 1.16x | 1.09x | 1.06x |
| crown/crown | ok | 846.8 | 1220 | 0.69 | 1.49x | 1.41x | 1.39x |
| kroken/camera-1 | ok | 763.6 | 1739 | 0.44 | 2.19x | 1.87x | 1.83x |
| sportscar/sportscar-sky | ok | 191.3 | 220.7 | 0.87 | 1.51x | 1.21x | 1.18x |
| bmw-m6/bmw-m6 | ok | 142.1 | 212.1 | 0.67 | 1.94x | 1.59x | 1.56x |
| dambreak/dambreak0 | ok | 470.6 | 694.0 | 0.68 | 1.64x | 1.44x | 1.41x |
| transparent-machines/frame542 | ok | 711.3 | 1206 | 0.59 | 1.50x | 1.45x | 1.43x |
| villa/villa-daylight | ok | 185.4 | 292.3 | 0.63 | 1.85x | 1.64x | 1.60x |
| clouds/clouds | ok | 635.5 | 902.5 | 0.70 | 1.54x | 1.39x | 1.39x |
| sanmiguel/sanmiguel-courtyard | ok | 129.5 | 149.4 | 0.87 | 1.75x | 1.37x | 1.24x |
| bistro/bistro_cafe | **FAILED** | 229.3 | 317.3 | 0.72 | 1.86x | 1.48x | 1.43x |
| bistro/bistro_vespa | **FAILED** | 234.9 | 326.2 | 0.72 | 1.77x | 1.55x | 1.45x |
| watercolor/camera-1 | ok | 838.0 | 2051 | 0.41 | 2.67x | 2.44x | 2.43x |
| sanmiguel/sanmiguel-courtyard-second | ok | 168.3 | 194.3 | 0.87 | 1.43x | 1.22x | 1.19x |

The wall speedup is pbrt's process over ours, both warm, from table.tsv. Images: 27 of 30 scenes match pbrt at 16 spp; a FAILED row's times are reported but do not count as beaten.

#### Every stage

| stage | scenes | behind on | worst | sum ours | sum pbrt | ratio of sums |
|---|---|---|---|---|---|---|
| closest-hit trace + routing | 30 | 2 | 1.08 (landscape/view-0) | 2883 | 3181 | 0.91 |
| shadow traces (every pass) | 30 | 2 | 1.23 (bistro/bistro_cafe) | 4422 | 5303 | 0.83 |
| sample draws (the bounce's dimensions, per ray) | 30 | 0 | - | 0 | 798.5 | pbrt only |
| camera rays | 30 | 29 | 1.77 (lte-orb/lte-orb-rough-glass) | 330.9 | 246.2 | 1.34 |
| escaped rays | 30 | 0 | 0.64 (villa/villa-lights-on) | 205.1 | 407.4 | 0.50 |
| emitters hit | 30 | 0 | 0.62 (kroken/camera-1) | 155.6 | 387.3 | 0.40 |
| material: CoatedDiffuse | 21 | 0 | 0.88 (landscape/view-0) | 1241 | 2195 | 0.57 |
| material: Diffuse | 22 | 7 | 1.33 (sportscar/sportscar-sky) | 783.4 | 1421 | 0.55 |
| film + the rest | 30 | 0 | 0.54 (pbrt-book/book) | 83.8 | 814.5 | 0.10 |
| materials with their sample draws (a view, not a stage) | 30 | 0 | 0.93 (sportscar/sportscar-sky) | 3476 | 6565 | 0.53 |
| material: Conductor | 11 | 0 | 0.50 (villa/villa-daylight) | 212.8 | 539.3 | 0.39 |
| material: CoatedConductor | 6 | 0 | 0.67 (bistro/bistro_cafe) | 261.1 | 470.7 | 0.55 |
| material: Dielectric | 16 | 11 | 1.53 (sanmiguel/sanmiguel-courtyard) | 487.9 | 436.5 | 1.12 |
| material: DiffuseTransmission | 8 | 4 | 1.12 (sanmiguel/sanmiguel-courtyard) | 241.3 | 320.6 | 0.75 |
| material: Measured | 3 | 3 | 1.14 (sportscar/sportscar-sky) | 190.1 | 175.2 | 1.09 |
| subsurface probe trace | 2 | 0 | 0.85 (sssdragon/dragon_10) | 47.0 | 55.6 | 0.85 |
| subsurface exit | 2 | 0 | 0.51 (head/head) | 11.7 | 23.1 | 0.51 |
| material: Subsurface | 2 | 2 | 1.22 (sssdragon/dragon_10) | 23.4 | 20.1 | 1.17 |
| material: Diffuse (pbrt only) | 8 | 0 | - | 0 | 177.1 | pbrt only |
| material: ThinDielectric | 2 | 2 | 1.35 (villa/villa-lights-on) | 14.1 | 10.5 | 1.35 |
| medium sampling | 9 | 0 | 0.70 (explosion/explosion) | 4060 | 6170 | 0.66 |
| medium scattering | 9 | 3 | 1.11 (smoke-plume/plume) | 373.9 | 465.4 | 0.80 |
| material: Mixed | 4 | 0 | - | 20.2 | 0 | ours only |

A stage is beaten when ours is below pbrt's on every scene that has it. "behind on" counts the scenes where it is not; the sums are over all the scenes that have the stage on either side.

##### closest-hit trace + routing

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 450.9 | 417.4 | **1.08** |
| disney-cloud/disney-cloud | 245.9 | 242.1 | **1.02** |
| smoke-plume/plume | 66.8 | 67.2 | 0.99 |
| kroken/camera-1 | 192.8 | 198.2 | 0.97 |
| ganesha/ganesha | 46.0 | 48.7 | 0.94 |
| transparent-machines/frame542 | 285.2 | 312.0 | 0.91 |
| clouds/clouds | 133.1 | 145.9 | 0.91 |
| sanmiguel/sanmiguel-courtyard-second | 66.8 | 73.8 | 0.90 |
| sanmiguel/sanmiguel-courtyard | 49.9 | 55.3 | 0.90 |
| explosion/explosion | 69.1 | 77.6 | 0.89 |
| sssdragon/dragon_10 | 37.4 | 42.2 | 0.89 |
| bunny-cloud/bunny-cloud | 136.1 | 154.2 | 0.88 |
| watercolor/camera-1 | 137.2 | 155.7 | 0.88 |
| barcelona-pavilion/pavilion-day (image FAILED) | 45.6 | 52.8 | 0.86 |
| crown/crown | 201.5 | 234.5 | 0.86 |
| lte-orb/lte-orb-simple-ball | 36.2 | 42.4 | 0.85 |
| sportscar/sportscar-sky | 54.3 | 65.1 | 0.83 |
| bistro/bistro_vespa (image FAILED) | 63.4 | 76.5 | 0.83 |
| killeroos/killeroo-gold | 38.4 | 47.4 | 0.81 |
| head/head | 8.17 | 10.1 | 0.81 |
| villa/villa-lights-on | 15.1 | 18.8 | 0.80 |
| lte-orb/lte-orb-rough-glass | 72.3 | 90.1 | 0.80 |
| lte-orb/lte-orb-silver | 36.9 | 46.0 | 0.80 |
| villa/villa-daylight | 50.0 | 62.8 | 0.80 |
| killeroos/killeroo-coated-gold | 36.5 | 46.1 | 0.79 |
| dambreak/dambreak0 | 162.3 | 208.3 | 0.78 |
| pbrt-book/book | 40.6 | 52.4 | 0.78 |
| killeroos/killeroo-simple | 12.8 | 16.6 | 0.77 |
| bmw-m6/bmw-m6 | 36.1 | 46.9 | 0.77 |
| bistro/bistro_cafe (image FAILED) | 55.9 | 74.0 | 0.76 |

##### shadow traces (every pass)

ours: `__raygen__shadow$<alpha>` and `__raygen__exit_shadow$<alpha>`; pbrt: the shadow and shadow-Tr trace kernels.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| bistro/bistro_cafe (image FAILED) | 27.5 | 22.3 | **1.23** |
| bistro/bistro_vespa (image FAILED) | 30.7 | 27.6 | **1.11** |
| transparent-machines/frame542 | 136.6 | 140.9 | 0.97 |
| explosion/explosion | 141.3 | 149.5 | 0.95 |
| barcelona-pavilion/pavilion-day (image FAILED) | 15.6 | 16.8 | 0.93 |
| head/head | 2.03 | 2.21 | 0.92 |
| clouds/clouds | 193.4 | 214.3 | 0.90 |
| smoke-plume/plume | 40.4 | 45.1 | 0.90 |
| sanmiguel/sanmiguel-courtyard | 18.4 | 20.8 | 0.89 |
| sssdragon/dragon_10 | 13.8 | 15.6 | 0.89 |
| watercolor/camera-1 | 64.0 | 73.6 | 0.87 |
| landscape/view-0 | 204.5 | 235.8 | 0.87 |
| disney-cloud/disney-cloud | 2128 | 2459 | 0.87 |
| sanmiguel/sanmiguel-courtyard-second | 24.9 | 28.9 | 0.86 |
| lte-orb/lte-orb-simple-ball | 8.73 | 10.2 | 0.86 |
| ganesha/ganesha | 7.43 | 8.94 | 0.83 |
| villa/villa-daylight | 9.70 | 11.7 | 0.83 |
| villa/villa-lights-on | 2.71 | 3.33 | 0.81 |
| lte-orb/lte-orb-rough-glass | 22.7 | 28.7 | 0.79 |
| kroken/camera-1 | 92.6 | 117.3 | 0.79 |
| sportscar/sportscar-sky | 9.66 | 12.9 | 0.75 |
| bunny-cloud/bunny-cloud | 1053 | 1416 | 0.74 |
| dambreak/dambreak0 | 42.8 | 57.7 | 0.74 |
| bmw-m6/bmw-m6 | 7.27 | 9.85 | 0.74 |
| crown/crown | 93.2 | 127.7 | 0.73 |
| killeroos/killeroo-gold | 7.00 | 10.0 | 0.70 |
| killeroos/killeroo-coated-gold | 6.82 | 9.81 | 0.70 |
| lte-orb/lte-orb-silver | 6.34 | 9.16 | 0.69 |
| killeroos/killeroo-simple | 3.16 | 4.70 | 0.67 |
| pbrt-book/book | 7.44 | 12.5 | 0.59 |

##### sample draws (the bounce's dimensions, per ray)

ours: the `rays` drain of the sample stage (10-09); a run before it has no kernel here, its draws having run inside the material and medium kernels, so pbrt's stands alone there.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| killeroos/killeroo-simple | 0 | 3.63 | pbrt only |
| killeroos/killeroo-gold | 0 | 9.12 | pbrt only |
| killeroos/killeroo-coated-gold | 0 | 9.03 | pbrt only |
| pbrt-book/book | 0 | 10.4 | pbrt only |
| ganesha/ganesha | 0 | 9.74 | pbrt only |
| barcelona-pavilion/pavilion-day (image FAILED) | 0 | 8.55 | pbrt only |
| lte-orb/lte-orb-simple-ball | 0 | 8.81 | pbrt only |
| lte-orb/lte-orb-silver | 0 | 6.09 | pbrt only |
| lte-orb/lte-orb-rough-glass | 0 | 18.1 | pbrt only |
| head/head | 0 | 3.17 | pbrt only |
| sssdragon/dragon_10 | 0 | 12.0 | pbrt only |
| villa/villa-lights-on | 0 | 3.34 | pbrt only |
| smoke-plume/plume | 0 | 13.0 | pbrt only |
| bunny-cloud/bunny-cloud | 0 | 48.9 | pbrt only |
| explosion/explosion | 0 | 21.8 | pbrt only |
| disney-cloud/disney-cloud | 0 | 57.3 | pbrt only |
| landscape/view-0 | 0 | 40.7 | pbrt only |
| crown/crown | 0 | 78.8 | pbrt only |
| kroken/camera-1 | 0 | 34.4 | pbrt only |
| sportscar/sportscar-sky | 0 | 10.9 | pbrt only |
| bmw-m6/bmw-m6 | 0 | 10.5 | pbrt only |
| dambreak/dambreak0 | 0 | 60.0 | pbrt only |
| transparent-machines/frame542 | 0 | 181.5 | pbrt only |
| villa/villa-daylight | 0 | 12.8 | pbrt only |
| clouds/clouds | 0 | 58.6 | pbrt only |
| sanmiguel/sanmiguel-courtyard | 0 | 7.98 | pbrt only |
| bistro/bistro_cafe (image FAILED) | 0 | 11.7 | pbrt only |
| bistro/bistro_vespa (image FAILED) | 0 | 11.8 | pbrt only |
| watercolor/camera-1 | 0 | 26.6 | pbrt only |
| sanmiguel/sanmiguel-courtyard-second | 0 | 9.17 | pbrt only |

##### camera rays

both sides draw the camera's own dimensions here.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| lte-orb/lte-orb-rough-glass | 8.38 | 4.73 | **1.77** |
| lte-orb/lte-orb-silver | 8.46 | 4.84 | **1.75** |
| sportscar/sportscar-sky | 12.6 | 9.10 | **1.38** |
| ganesha/ganesha | 12.5 | 9.09 | **1.38** |
| dambreak/dambreak0 | 12.6 | 9.13 | **1.38** |
| pbrt-book/book | 12.4 | 9.08 | **1.37** |
| bunny-cloud/bunny-cloud | 12.3 | 9.09 | **1.35** |
| bistro/bistro_cafe (image FAILED) | 12.6 | 9.37 | **1.34** |
| landscape/view-0 | 50.6 | 37.7 | **1.34** |
| clouds/clouds | 12.8 | 9.52 | **1.34** |
| transparent-machines/frame542 | 9.73 | 7.34 | **1.33** |
| villa/villa-daylight | 9.55 | 7.21 | **1.32** |
| sanmiguel/sanmiguel-courtyard-second | 9.40 | 7.11 | **1.32** |
| bistro/bistro_vespa (image FAILED) | 12.6 | 9.68 | **1.30** |
| killeroos/killeroo-gold | 8.44 | 6.56 | **1.29** |
| killeroos/killeroo-coated-gold | 8.44 | 6.59 | **1.28** |
| sssdragon/dragon_10 | 8.40 | 6.58 | **1.28** |
| smoke-plume/plume | 8.56 | 6.72 | **1.27** |
| watercolor/camera-1 | 16.9 | 13.4 | **1.26** |
| explosion/explosion | 14.2 | 11.2 | **1.26** |
| crown/crown | 8.48 | 6.74 | **1.26** |
| bmw-m6/bmw-m6 | 8.32 | 6.62 | **1.26** |
| kroken/camera-1 | 14.3 | 11.5 | **1.25** |
| barcelona-pavilion/pavilion-day (image FAILED) | 8.21 | 6.69 | **1.23** |
| lte-orb/lte-orb-simple-ball | 8.80 | 7.26 | **1.21** |
| head/head | 3.52 | 2.91 | **1.21** |
| sanmiguel/sanmiguel-courtyard | 6.66 | 5.51 | **1.21** |
| killeroos/killeroo-simple | 3.03 | 2.54 | **1.19** |
| villa/villa-lights-on | 2.73 | 2.38 | **1.15** |
| disney-cloud/disney-cloud | 5.51 | 0 | ours only |

##### escaped rays

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 0.69 | 1.08 | 0.64 |
| head/head | 1.77 | 2.84 | 0.62 |
| sssdragon/dragon_10 | 3.84 | 6.62 | 0.58 |
| explosion/explosion | 5.70 | 9.93 | 0.57 |
| landscape/view-0 | 13.6 | 24.6 | 0.55 |
| bmw-m6/bmw-m6 | 4.11 | 7.44 | 0.55 |
| barcelona-pavilion/pavilion-day (image FAILED) | 2.48 | 4.54 | 0.55 |
| smoke-plume/plume | 4.11 | 7.94 | 0.52 |
| ganesha/ganesha | 4.50 | 8.86 | 0.51 |
| lte-orb/lte-orb-simple-ball | 3.74 | 7.42 | 0.50 |
| sportscar/sportscar-sky | 5.63 | 11.5 | 0.49 |
| clouds/clouds | 16.7 | 34.8 | 0.48 |
| villa/villa-daylight | 2.71 | 5.77 | 0.47 |
| disney-cloud/disney-cloud | 12.0 | 26.8 | 0.45 |
| watercolor/camera-1 | 11.1 | 24.9 | 0.45 |
| sanmiguel/sanmiguel-courtyard | 1.37 | 3.07 | 0.45 |
| bistro/bistro_cafe (image FAILED) | 1.89 | 4.24 | 0.45 |
| bistro/bistro_vespa (image FAILED) | 1.89 | 4.32 | 0.44 |
| bunny-cloud/bunny-cloud | 14.7 | 33.8 | 0.44 |
| sanmiguel/sanmiguel-courtyard-second | 1.36 | 3.23 | 0.42 |
| dambreak/dambreak0 | 16.6 | 40.3 | 0.41 |
| kroken/camera-1 | 6.78 | 17.0 | 0.40 |
| transparent-machines/frame542 | 43.4 | 116.4 | 0.37 |
| killeroos/killeroo-simple | 0.39 | 0 | ours only |
| killeroos/killeroo-gold | 2.42 | 0 | ours only |
| killeroos/killeroo-coated-gold | 2.39 | 0 | ours only |
| pbrt-book/book | 1.00 | 0 | ours only |
| lte-orb/lte-orb-silver | 0.85 | 0 | ours only |
| lte-orb/lte-orb-rough-glass | 4.29 | 0 | ours only |
| crown/crown | 12.9 | 0 | ours only |

##### emitters hit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 9.97 | 16.1 | 0.62 |
| bistro/bistro_vespa (image FAILED) | 3.09 | 5.13 | 0.60 |
| bistro/bistro_cafe (image FAILED) | 2.97 | 4.94 | 0.60 |
| lte-orb/lte-orb-rough-glass | 7.36 | 13.3 | 0.55 |
| ganesha/ganesha | 2.15 | 3.90 | 0.55 |
| lte-orb/lte-orb-silver | 1.79 | 3.33 | 0.54 |
| crown/crown | 25.2 | 48.3 | 0.52 |
| killeroos/killeroo-simple | 0.51 | 0.99 | 0.52 |
| villa/villa-lights-on | 0.47 | 0.94 | 0.50 |
| barcelona-pavilion/pavilion-day (image FAILED) | 1.02 | 2.28 | 0.45 |
| bmw-m6/bmw-m6 | 1.52 | 3.44 | 0.44 |
| smoke-plume/plume | 1.06 | 2.43 | 0.44 |
| sssdragon/dragon_10 | 0.95 | 2.29 | 0.41 |
| head/head | 0.21 | 0.51 | 0.41 |
| lte-orb/lte-orb-simple-ball | 0.92 | 2.35 | 0.39 |
| killeroos/killeroo-coated-gold | 1.29 | 3.38 | 0.38 |
| villa/villa-daylight | 1.81 | 4.81 | 0.38 |
| killeroos/killeroo-gold | 1.30 | 3.48 | 0.37 |
| watercolor/camera-1 | 4.28 | 11.7 | 0.37 |
| sanmiguel/sanmiguel-courtyard | 0.70 | 1.94 | 0.36 |
| pbrt-book/book | 1.13 | 3.15 | 0.36 |
| disney-cloud/disney-cloud | 8.91 | 25.2 | 0.35 |
| sanmiguel/sanmiguel-courtyard-second | 0.91 | 2.58 | 0.35 |
| transparent-machines/frame542 | 40.1 | 113.7 | 0.35 |
| explosion/explosion | 1.38 | 3.97 | 0.35 |
| clouds/clouds | 8.98 | 26.5 | 0.34 |
| dambreak/dambreak0 | 11.4 | 33.7 | 0.34 |
| bunny-cloud/bunny-cloud | 8.99 | 26.9 | 0.33 |
| sportscar/sportscar-sky | 1.08 | 3.25 | 0.33 |
| landscape/view-0 | 4.21 | 12.8 | 0.33 |

##### material: CoatedDiffuse

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 90.7 | 103.1 | 0.88 |
| explosion/explosion | 26.4 | 30.2 | 0.87 |
| crown/crown | 174.7 | 204.8 | 0.85 |
| bunny-cloud/bunny-cloud | 99.2 | 119.6 | 0.83 |
| sanmiguel/sanmiguel-courtyard | 14.6 | 17.8 | 0.82 |
| sanmiguel/sanmiguel-courtyard-second | 17.6 | 21.7 | 0.81 |
| sssdragon/dragon_10 | 33.3 | 41.6 | 0.80 |
| killeroos/killeroo-simple | 5.94 | 7.48 | 0.79 |
| sportscar/sportscar-sky | 11.2 | 14.2 | 0.79 |
| pbrt-book/book | 17.4 | 22.5 | 0.77 |
| barcelona-pavilion/pavilion-day (image FAILED) | 30.7 | 40.5 | 0.76 |
| bistro/bistro_cafe (image FAILED) | 101.1 | 149.3 | 0.68 |
| lte-orb/lte-orb-rough-glass | 65.2 | 99.6 | 0.65 |
| bistro/bistro_vespa (image FAILED) | 93.8 | 144.9 | 0.65 |
| bmw-m6/bmw-m6 | 37.5 | 59.6 | 0.63 |
| lte-orb/lte-orb-silver | 30.3 | 50.8 | 0.60 |
| ganesha/ganesha | 28.7 | 48.6 | 0.59 |
| villa/villa-daylight | 56.2 | 107.9 | 0.52 |
| villa/villa-lights-on | 15.0 | 34.1 | 0.44 |
| kroken/camera-1 | 115.2 | 276.9 | 0.42 |
| watercolor/camera-1 | 176.7 | 599.8 | 0.29 |

##### material: Diffuse

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sportscar/sportscar-sky | 5.68 | 4.28 | **1.33** |
| bmw-m6/bmw-m6 | 8.63 | 6.62 | **1.30** |
| landscape/view-0 | 30.2 | 23.9 | **1.26** |
| barcelona-pavilion/pavilion-day (image FAILED) | 7.28 | 6.09 | **1.20** |
| disney-cloud/disney-cloud | 51.2 | 46.0 | **1.11** |
| sanmiguel/sanmiguel-courtyard | 25.6 | 24.5 | **1.05** |
| sanmiguel/sanmiguel-courtyard-second | 34.3 | 34.2 | **1.00** |
| smoke-plume/plume | 13.9 | 14.8 | 0.94 |
| dambreak/dambreak0 | 70.0 | 74.5 | 0.94 |
| crown/crown | 104.1 | 115.6 | 0.90 |
| lte-orb/lte-orb-silver | 3.45 | 4.58 | 0.75 |
| villa/villa-daylight | 14.0 | 20.0 | 0.70 |
| lte-orb/lte-orb-rough-glass | 13.1 | 19.3 | 0.68 |
| villa/villa-lights-on | 2.96 | 4.43 | 0.67 |
| killeroos/killeroo-simple | 7.63 | 11.5 | 0.66 |
| lte-orb/lte-orb-simple-ball | 18.5 | 29.4 | 0.63 |
| pbrt-book/book | 22.5 | 36.7 | 0.61 |
| killeroos/killeroo-gold | 20.0 | 40.2 | 0.50 |
| ganesha/ganesha | 16.5 | 33.3 | 0.49 |
| killeroos/killeroo-coated-gold | 19.4 | 39.5 | 0.49 |
| watercolor/camera-1 | 213.3 | 585.9 | 0.36 |
| kroken/camera-1 | 81.1 | 246.0 | 0.33 |

##### film + the rest

ours: `rays_rest_blk` and `q_blk`; pbrt: Update film, the queue resets, the stats updates, Evaluate Displacement, Other.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| pbrt-book/book | 3.24 | 6.05 | 0.54 |
| ganesha/ganesha | 3.14 | 5.95 | 0.53 |
| sportscar/sportscar-sky | 3.05 | 5.85 | 0.52 |
| bistro/bistro_vespa (image FAILED) | 3.00 | 5.92 | 0.51 |
| landscape/view-0 | 11.1 | 22.4 | 0.50 |
| bistro/bistro_cafe (image FAILED) | 2.94 | 5.94 | 0.49 |
| sanmiguel/sanmiguel-courtyard-second | 2.39 | 4.99 | 0.48 |
| head/head | 0.77 | 1.70 | 0.45 |
| lte-orb/lte-orb-silver | 2.13 | 4.71 | 0.45 |
| explosion/explosion | 3.32 | 7.40 | 0.45 |
| killeroos/killeroo-coated-gold | 2.06 | 4.83 | 0.43 |
| killeroos/killeroo-gold | 2.05 | 4.83 | 0.42 |
| barcelona-pavilion/pavilion-day (image FAILED) | 1.96 | 4.63 | 0.42 |
| smoke-plume/plume | 2.06 | 4.91 | 0.42 |
| sssdragon/dragon_10 | 2.08 | 5.25 | 0.40 |
| sanmiguel/sanmiguel-courtyard | 1.63 | 4.15 | 0.39 |
| bmw-m6/bmw-m6 | 1.97 | 5.39 | 0.37 |
| villa/villa-daylight | 2.30 | 6.45 | 0.36 |
| villa/villa-lights-on | 0.62 | 1.84 | 0.34 |
| killeroos/killeroo-simple | 0.66 | 1.98 | 0.33 |
| lte-orb/lte-orb-simple-ball | 3.54 | 13.4 | 0.26 |
| lte-orb/lte-orb-rough-glass | 2.05 | 12.1 | 0.17 |
| clouds/clouds | 2.95 | 19.8 | 0.15 |
| bunny-cloud/bunny-cloud | 2.78 | 20.5 | 0.14 |
| dambreak/dambreak0 | 3.02 | 24.0 | 0.13 |
| watercolor/camera-1 | 6.00 | 68.3 | 0.09 |
| crown/crown | 2.01 | 33.0 | 0.06 |
| disney-cloud/disney-cloud | 1.30 | 24.8 | 0.05 |
| transparent-machines/frame542 | 2.32 | 75.6 | 0.03 |
| kroken/camera-1 | 5.41 | 407.8 | 0.01 |

##### materials with their sample draws (a view, not a stage)

the material kernels above summed with the sample draws on each side -- pbrt's `Generate ray samples`, and our sample kernel where the run has one (until the sample stage of 10-09 the bounce's draws ran inside our material kernels) -- the like-for-like sum either way; a second view of rows already counted, left out of the totals.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sportscar/sportscar-sky | 105.0 | 113.0 | 0.93 |
| lte-orb/lte-orb-simple-ball | 113.7 | 129.6 | 0.88 |
| landscape/view-0 | 272.9 | 313.5 | 0.87 |
| sanmiguel/sanmiguel-courtyard | 50.9 | 58.7 | 0.87 |
| sanmiguel/sanmiguel-courtyard-second | 62.5 | 73.7 | 0.85 |
| barcelona-pavilion/pavilion-day (image FAILED) | 73.6 | 91.4 | 0.81 |
| head/head | 9.12 | 11.9 | 0.77 |
| dambreak/dambreak0 | 159.0 | 212.4 | 0.75 |
| sssdragon/dragon_10 | 47.6 | 67.3 | 0.71 |
| crown/crown | 439.4 | 643.5 | 0.68 |
| bistro/bistro_cafe (image FAILED) | 125.5 | 196.5 | 0.64 |
| bmw-m6/bmw-m6 | 82.8 | 132.4 | 0.63 |
| bistro/bistro_vespa (image FAILED) | 120.1 | 197.0 | 0.61 |
| killeroos/killeroo-simple | 13.6 | 22.6 | 0.60 |
| lte-orb/lte-orb-rough-glass | 109.6 | 187.2 | 0.59 |
| pbrt-book/book | 39.9 | 69.6 | 0.57 |
| villa/villa-daylight | 109.4 | 193.6 | 0.57 |
| bunny-cloud/bunny-cloud | 99.2 | 195.3 | 0.51 |
| villa/villa-lights-on | 28.2 | 55.8 | 0.50 |
| disney-cloud/disney-cloud | 51.2 | 103.3 | 0.50 |
| ganesha/ganesha | 45.2 | 91.7 | 0.49 |
| killeroos/killeroo-coated-gold | 44.8 | 93.4 | 0.48 |
| smoke-plume/plume | 17.8 | 37.1 | 0.48 |
| explosion/explosion | 26.4 | 55.2 | 0.48 |
| lte-orb/lte-orb-silver | 43.2 | 93.5 | 0.46 |
| kroken/camera-1 | 400.9 | 905.0 | 0.44 |
| transparent-machines/frame542 | 194.0 | 440.1 | 0.44 |
| killeroos/killeroo-gold | 29.1 | 77.0 | 0.38 |
| watercolor/camera-1 | 561.0 | 1619 | 0.35 |
| clouds/clouds | 0 | 85.0 | pbrt only |

##### material: Conductor

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-daylight | 10.2 | 20.6 | 0.50 |
| barcelona-pavilion/pavilion-day (image FAILED) | 4.11 | 8.47 | 0.49 |
| kroken/camera-1 | 36.1 | 74.9 | 0.48 |
| sportscar/sportscar-sky | 5.07 | 11.0 | 0.46 |
| villa/villa-lights-on | 2.61 | 5.78 | 0.45 |
| bmw-m6/bmw-m6 | 7.42 | 16.8 | 0.44 |
| crown/crown | 84.2 | 193.2 | 0.44 |
| smoke-plume/plume | 3.85 | 9.33 | 0.41 |
| killeroos/killeroo-gold | 9.11 | 27.7 | 0.33 |
| lte-orb/lte-orb-silver | 9.42 | 32.0 | 0.29 |
| watercolor/camera-1 | 40.7 | 139.5 | 0.29 |

##### material: CoatedConductor

| scene | ours | pbrt | ratio |
|---|---|---|---|
| bistro/bistro_cafe (image FAILED) | 19.2 | 28.8 | 0.67 |
| bmw-m6/bmw-m6 | 20.6 | 31.0 | 0.67 |
| bistro/bistro_vespa (image FAILED) | 20.6 | 31.9 | 0.64 |
| kroken/camera-1 | 91.8 | 149.3 | 0.62 |
| killeroos/killeroo-coated-gold | 25.3 | 44.9 | 0.56 |
| watercolor/camera-1 | 83.6 | 184.9 | 0.45 |

##### material: Dielectric

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sanmiguel/sanmiguel-courtyard | 4.32 | 2.83 | **1.53** |
| sanmiguel/sanmiguel-courtyard-second | 4.43 | 2.95 | **1.50** |
| transparent-machines/frame542 | 194.0 | 145.7 | **1.33** |
| bistro/bistro_cafe (image FAILED) | 5.24 | 4.00 | **1.31** |
| barcelona-pavilion/pavilion-day (image FAILED) | 8.33 | 6.41 | **1.30** |
| crown/crown | 64.3 | 51.0 | **1.26** |
| sportscar/sportscar-sky | 4.44 | 3.55 | **1.25** |
| watercolor/camera-1 | 15.0 | 12.6 | **1.18** |
| dambreak/dambreak0 | 89.0 | 77.9 | **1.14** |
| landscape/view-0 | 11.9 | 11.5 | **1.04** |
| bistro/bistro_vespa (image FAILED) | 5.80 | 5.59 | **1.04** |
| bmw-m6/bmw-m6 | 7.57 | 7.89 | 0.96 |
| kroken/camera-1 | 30.4 | 35.3 | 0.86 |
| villa/villa-lights-on | 2.52 | 3.73 | 0.68 |
| lte-orb/lte-orb-rough-glass | 31.3 | 50.4 | 0.62 |
| villa/villa-daylight | 9.25 | 15.2 | 0.61 |

##### material: DiffuseTransmission

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sanmiguel/sanmiguel-courtyard | 6.27 | 5.61 | **1.12** |
| sanmiguel/sanmiguel-courtyard-second | 6.17 | 5.73 | **1.08** |
| barcelona-pavilion/pavilion-day (image FAILED) | 7.02 | 6.69 | **1.05** |
| landscape/view-0 | 140.2 | 134.3 | **1.04** |
| villa/villa-daylight | 8.25 | 8.38 | 0.98 |
| villa/villa-lights-on | 2.50 | 2.54 | 0.98 |
| kroken/camera-1 | 42.7 | 88.2 | 0.48 |
| watercolor/camera-1 | 28.2 | 69.1 | 0.41 |

##### material: Measured

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sportscar/sportscar-sky | 78.6 | 69.1 | **1.14** |
| barcelona-pavilion/pavilion-day (image FAILED) | 16.2 | 14.7 | **1.10** |
| lte-orb/lte-orb-simple-ball | 95.2 | 91.4 | **1.04** |

##### subsurface probe trace

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sssdragon/dragon_10 | 41.2 | 48.6 | 0.85 |
| head/head | 5.78 | 7.00 | 0.83 |

##### subsurface exit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 3.25 | 6.36 | 0.51 |
| sssdragon/dragon_10 | 8.44 | 16.7 | 0.50 |

##### material: Subsurface

one kernel of ours against two of pbrt's: the material evaluation and the BSSRDF sampling that enqueues the probe.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sssdragon/dragon_10 | 14.3 | 11.8 | **1.22** |
| head/head | 9.12 | 8.32 | **1.10** |

##### material: Diffuse (pbrt only)

pbrt launches a kernel for every material of the scene at every depth, reached by a ray or not; ours launches only for a non-empty queue, so a material no ray of ours reached has no kernel here. Each row says how many launches pbrt made and what each cost: at pbrt's launch floor of about 0.015 ms they were empty.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head (32 launches, 0.012 ms each) | 0 | 0.37 | pbrt only |
| sssdragon/dragon_10 (160 launches, 0.012 ms each) | 0 | 1.91 | pbrt only |
| bunny-cloud/bunny-cloud (1600 launches, 0.017 ms each) | 0 | 26.8 | pbrt only |
| explosion/explosion (240 launches, 0.014 ms each) | 0 | 3.31 | pbrt only |
| transparent-machines/frame542 (8192 launches, 0.014 ms each) | 0 | 112.9 | pbrt only |
| clouds/clouds (1600 launches, 0.017 ms each) | 0 | 26.4 | pbrt only |
| bistro/bistro_cafe (image FAILED) (160 launches, 0.017 ms each) | 0 | 2.70 | pbrt only |
| bistro/bistro_vespa (image FAILED) (160 launches, 0.017 ms each) | 0 | 2.70 | pbrt only |

##### material: ThinDielectric

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 2.58 | 1.91 | **1.35** |
| villa/villa-daylight | 11.6 | 8.59 | **1.34** |

##### medium sampling

| scene | ours | pbrt | ratio |
|---|---|---|---|
| explosion/explosion | 255.5 | 365.9 | 0.70 |
| clouds/clouds | 205.5 | 297.8 | 0.69 |
| disney-cloud/disney-cloud | 2665 | 4025 | 0.66 |
| bunny-cloud/bunny-cloud | 757.7 | 1156 | 0.66 |
| kroken/camera-1 | 15.1 | 26.8 | 0.56 |
| smoke-plume/plume | 67.2 | 120.4 | 0.56 |
| watercolor/camera-1 | 11.2 | 20.4 | 0.55 |
| crown/crown | 47.6 | 87.8 | 0.54 |
| dambreak/dambreak0 | 34.9 | 70.4 | 0.50 |

##### medium scattering

| scene | ours | pbrt | ratio |
|---|---|---|---|
| smoke-plume/plume | 10.7 | 9.59 | **1.11** |
| explosion/explosion | 9.10 | 8.28 | **1.10** |
| bunny-cloud/bunny-cloud | 60.0 | 56.2 | **1.07** |
| disney-cloud/disney-cloud | 135.7 | 141.7 | 0.96 |
| clouds/clouds | 62.1 | 69.0 | 0.90 |
| dambreak/dambreak0 | 28.0 | 38.1 | 0.73 |
| kroken/camera-1 | 25.8 | 39.5 | 0.65 |
| crown/crown | 16.4 | 38.3 | 0.43 |
| watercolor/camera-1 | 26.1 | 64.7 | 0.40 |

##### material: Mixed

pbrt resolves a mix material inside its closest-hit program and has no kernel for it; ours is the choice step.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| crown/crown | 12.1 | 0 | ours only |
| kroken/camera-1 | 3.45 | 0 | ours only |
| bmw-m6/bmw-m6 | 1.10 | 0 | ours only |
| watercolor/camera-1 | 3.54 | 0 | ours only |

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
