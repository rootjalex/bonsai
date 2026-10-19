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

## Where it stands (2026-10-19, 09:15)

**The subsurface entry is 180 bytes against pbrt's 168, from 349; the
material is stored as its index and rebuilt at the drain, and the queue
has 2 producers where it had 22** (PLAN.md, "The entry stores what it is
handed" and "The material by index"). Three compiler changes, each a
general one: the deferral's recipes extended -- a read of a read-only
array at an index the entry holds is stored as the index, the slot is a
leaf a recipe may stand on, a pusher's parameter is followed to the
owner's calls and a recipe built at each, all required to agree, and the
budget and the reload-only guard gone for a cost argument; the
simplifier's trivial-phi rule for constants -- an argument every edge
hands one constant is that constant, which folds the subsurface dispatch
out of the twenty material kernels whose material is not Subsurface; and
the definitions analysis resolving a name a block reads without taking
at the block that binds it, for the readers that want the value behind
it (the simplifier's key, the deferral's recipe builder), which also
found common subexpressions the key had missed. In the program, the
probe walk's reservoir starts empty, as pbrt's does. Entries, counted
array by array (a vec3 as 12 bytes; the `False` family, the `True` one
identical):

| queue | before | after | pbrt |
|---|---|---|---|
| bssrdfs | 349 | 180 | 168 (GetBSSRDFAndProbeRayWorkItem) |
| probes | 320 | 184 | 332 (SubsurfaceScatterWorkItem) |
| hits, per material | 255 | 247 | 252 (MaterialEvalWorkItem) |
| exits | 228 | 220 | -- |
| medium_samples | 223 | 215 | -- |
| medium_scatters | 120 | 112 | -- |
| emissive, escaped, shadow, exit_shadow | 169, 145, 96, 96 | unchanged | -- |

What remains over pbrt in bssrdfs is the hit-derived copies -- medium,
the primitive's media, mat, the hit point and shading normal, the texture
context, beta, r_u, depth, wave, eta_scale, lambda -- which the
cross-queue reload by position (to-do item 1) would take out. 1626 tests
pass; 54 goldens moved, each read before it was blessed (PLAN.md has the
categories). The images are checked on the GPU in nacho-9b's window
(head, dragon_10, killeroo-simple at 16 spp), about two hours after this
is written; the compiler is committed on the suite, as b6b09dc9 was.

## To do, in order (2026-10-19, 09:15)

0. The GPU check above; then the heatmap's subsurface and media rows at
   16 spp for the kernel time of the smaller entries (the protocol: the
   least of three).
1. The cross-queue reload by position (item 1 of 07:00, unchanged).
2. The hit chain (item 2 of 07:00, unchanged; the range set for the
   user's approval).
3. The NaN pixels and the probe's 1.9% (item 3 of 07:00).
4. The megakernel's 52 ms (item 4 of 07:00).
5. Tooling debts (item 5 of 07:00).

## Where it stands (2026-10-19, 07:00)

**The 30-scene heatmap at 16, 64, 128 and 256 spp, 117 of 120 cells
matching pbrt's image, every cell faster** (compare-out-heatmap6,
heatmap-gpu-optix.png; the compiler of b6b09dc9, the renderer of
c9025e99). The 256 row is new: the wavefront schedule matches pbrt's
kernel for kernel (the stage tables below pair them one to one), which is
what the user asked before adding it; what still differs from pbrt is the
work-item layouts and the hit chain, to-do items 1 and 2.

| spp | wall speedup, mean | least | most | kernel speedup, mean |
|---|---|---|---|---|
| 16 | 2.01x | 1.29x view-0 | 3.13x killeroo-simple | 1.68x |
| 64 | 1.75x | 1.19x view-0 | 2.43x watercolor | 1.66x |
| 128 | 1.69x | 1.19x view-0 | 2.43x watercolor | 1.65x |
| 256 | 1.65x | 1.18x view-0 | 2.40x watercolor | 1.65x |

Against heatmap5 (10-11): 1.94x, 1.65x and 1.59x at the three counts it
had. The three cells that fail are not transport: kroken at 128 spp,
watercolor at 128 and at 256 hold one, two and ten NaN pixels, which
every earlier heatmap passed because a NaN mean is within every tolerance
(PLAN.md, "A pixel that is not a number passed the image check"); the
harness fails them now (5728a546), and the NaN itself is open, the two
scenes being the only two lit through a portal.

**Three fixes on the way to it.** The subsurface scenes had hung since
10-11 (head, dragon_10: the read analysis took the probe walk's material
for unread, the drain handed `undef`, LLVM deleted the walk's hit path;
b6b09dc9, PLAN.md "The probe walk's material handed undefined"), and the
same blind spot had every media scene's shadow rays on undefined
wavelengths. kroken came out 13% dark in the first rerun: its sun is a
sphere light with a constant alpha of zero, pbrt's special case for an
invisible light -- unmasked, fully emissive, typed as a delta light so
direct lighting uses no MIS -- and the 10-17 alpha mask had silenced it
(c9025e99, PLAN.md "kroken's invisible sun"; a probe scene for the case,
probes/invisible_light_d1, matches). And the NaN rule above.

## To do, in order (2026-10-19, 07:00)

0. In flight: the deferral's recipes extended (SSA/Defer.cpp; PLAN.md
   "The entry stores what it is handed") -- a read of a read-only array
   stored as its index, the slot as a recipe leaf, a pure function of the
   slot or of a uniform recognized through the chain's calls -- and the
   probe walk's reservoir starting empty (render.bonsai). Expected: the
   subsurface material stored as `mat` (44 bytes to none beyond the index
   already there), the pixel coordinates stored in no queue, the probes
   entry without the entry surface (116 bytes); bssrdfs from 345 bytes to
   under 200 against pbrt's 168, probes from 316 to under 200 against
   332. Measured when the card and the cores are free.
1. The work items still over pbrt after 0: the hits entry's 14 bytes for
   recomputing the outgoing direction (store `wo`), the round counter
   and the hit-derived copies every later queue carries (wave, depth,
   medium, the primitive's media) -- the cross-queue reload by position,
   designed in the same PLAN section, needs a stability argument for the
   earlier queue's arrays across the round.
2. The hit chain: six dependent loads to pbrt's four. Half needs no new
   information: the record-constant part of the hit program (the arm, the
   mesh row) hoisted into the hit-group record by partial evaluation in
   Lower/Trees.cpp. Half does: that a mesh's triangles are the integers 0
   to count, which the program can say with a range-valued set
   (PLAN.md, "The triangle range as a set the type describes"), for the
   user's approval.
3. The NaN pixels (kroken and watercolor at 128 and 256 spp; the portal
   light suspected) and the invisible-light probe's 1.9% (whole samples
   lost in a ring; PLAN.md "Open: the probe's 1.9%"): both want a 1-spp
   run and two probe variants, a few GPU minutes.
4. The megakernel's 52 ms on killeroo-simple at 64 spp against 10-02:
   10-03's shapes and materials in one raygen, which no scene flag names.
5. Tooling debts: convert_scene.sh's stamp does not see an `Include`d
   file change; check_differentials.sh aborts in `--print-differentials`;
   scene_dump refuses the orthographic camera; the renderer has no
   `--pixel` switch to replay one pixel's samples.

## Where it stands (2026-10-18, 12:00)

**The megakernels are 1.24x to 1.71x faster than on 10-15, by schedule.**
The slowdown since 10-02 was the program's growth into one raygen (below,
10-17); the schedule's answer needed a form of `specialize` that could
name pbrt's `haveSubsurface`, which the program keeps in the VolPath
variant of the integrator as pbrt keeps it in its integrator. The user
chose the syntax `render[VolPath].specialize(VolPath::have_subsurface)`,
the compiler has it (9c823901: the field form, PLAN.md "The field form of
specialize"), and both megakernel schedules copy the volpath pixel loop
per scene flag, `have_media` and `VolPath::have_subsurface`, so a scene
launches a raygen holding neither media nor subsurface code when it has
neither -- 219,744 PTX lines against the 606,298 of the one raygen before.
Measured on the protocol's three scenes (apps/pbrt/scratch/
mega-field-1018; kernel milliseconds, the least of three):

| scene | spp | pbrt | gpu-optix-mega (10-15) | gpu-optix-mega | mega over pbrt | gpu-optix-mega-ser (10-15) | gpu-optix-mega-ser | mega-ser over pbrt |
|---|---|---|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.5 | 116.2 | 88.3 | 0.56x | 73.4 | 49.3 | 1.00x |
| killeroo-gold | 16 | 149.4 | 317.4 | 183.2 | 0.82x | 266.4 | 159.6 | 0.94x |
| book | 16 | 152.8 | 338.8 | 240.9 | 0.63x | 277.7 | 210.1 | 0.73x |
| killeroo-simple | 64 | 198.0 | 459.5 | 349.8 | 0.57x | 298.2 | 197.8 | 1.00x |
| killeroo-gold | 64 | 597.6 | 1259.2 | 736.1 | 0.81x | 1073.4 | 652.2 | 0.92x |
| book | 64 | 617.0 | 1365.1 | 1001.1 | 0.62x | 1126.9 | 837.2 | 0.74x |

Every image matches pbrt's. The sorted megakernel is at pbrt's kernel
time on killeroo-simple and within 8% on killeroo-gold; book is a quarter
behind. killeroo-simple at 64 spp is 349.8 ms where the wired variant of
10-18 predicted 346 and 10-02's compiler renders 298, so what is left of
the climb is the 52 ms of shapes and materials every megakernel carries
and no scene flag names. The wavefront needs no such flag: its subsurface
stages are kernels of their own, launched when a scene has the material,
and its material kernels are one per material arm (PLAN.md, "The
wavefront does not need it"). The standing schedule stays gpu-optix.

**The pavilion rule.** The user's ruling of 10-18: a scene pbrt's own two
renderers disagree on more than we disagree with pbrt is rendered
correctly by us. The harness applies it (9bef802f): a cell whose image
fails the check is held against pbrt's CPU render of the same scene, and
passes when pbrt's disagreement with itself is the larger -- pavilion-day
at 16 spp, 2,058 lit pixels from pbrt --gpu where pbrt's CPU render is
5,529 from it, reads `ok: within pbrt's own disagreement`. The 30-scene
heatmap was redone under it twice -- the first run found the subsurface
hang and the dark kroken -- and the result is the section above (2026-10-19,
07:00).

## Where it stands (2026-10-17, 23:40)

**Every one of the 30 scenes matches pbrt's image now except pavilion-day
at 16 spp, the calibrated tolerance artifact, and every one is faster.**
The two bistro scenes had lit far more pixels than pbrt at every sample
count since the first sweep (10-04). The cause was not the sampler, the
light tree, the sky or the cutout test, each of which a probe scene
cleared (PLAN.md, "The bistro lit-pixel bias"): pbrt masks an *area light*
by the alpha texture its shape carries -- a point of the light the texture
cuts out emits nothing when a ray lands on it and is no sample at all when
a shadow ray is aimed at it (DiffuseAreaLight::L, ::SampleLi) -- and
render.bonsai's AreaLight had been written without that mask. Bistro's
emissive streetlight glass has an alpha of zero everywhere, so in pbrt
those lamps light nothing, and in ours they lit the whole street dimly.
The mask is in (render.bonsai: `AreaLight.alpha`, `area_light_masked`;
the emissive hit carries the point and uv as pbrt's HitAreaLightWorkItem
does; render_hook.cpp sets each light's alpha from its shape). Measured
(apps/pbrt/scratch/alphafix-1016; kernel milliseconds, the least of
three; the 10-11 columns are heatmap5's, under which the images failed):

| scene | spp | pbrt | ours (10-11) | ours now | kernel speedup | image |
|---|---|---|---|---|---|---|
| bistro_cafe | 16 | 315.3 | 207.7 | 181.7 | 1.73x | matches |
| bistro_vespa | 16 | 327.4 | 216.9 | 188.9 | 1.73x | matches |
| bistro_cafe | 64 | 1269.2 | 830.6 | 727.3 | 1.75x | matches |
| bistro_vespa | 64 | 1309.7 | 863.1 | 751.5 | 1.74x | matches |

The wall speedups the harness printed are 2.27x and 2.16x at 16 spp,
1.85x and 1.92x at 64. The kernels are faster than on 10-11 as well
(the spawn/join and queue work since), and the masked lamps' samples,
declined now as pbrt declines them, cost nothing. A probe built for the
rule itself -- a quad whose alpha is a gradient, as the light -- lands
1.0% under pbrt's lit count and mean, uniformly over the image, with 78%
of the shared pixels exact and the rest at ratios near 2 and 1/2; an
open detail, noted below, not a transport defect of the scenes.

The megakernel schedules' 1.5x slowdown since 10-02 is not the driver:
10-02's compiler rebuilt under today's driver renders killeroo-simple at
64 spp in 298 ms against 459 ms for the current one. Nor is it the
texture graph's non-inlined evaluator calls, the raygen's one structural
difference from then (448 real call sites): a renderer with the calls
removed takes 468 ms and one with the calls kept and the callees made
trivial 460 ms. A bisect over the 210 commits between, each step built
from clean, found the first bad commit to be the subsurface material
(3206864a, 10-03 20:27: 358 ms before it, 428 after, on a scene with no
subsurface material), and the rest of the climb to be the same thing in
smaller steps -- the shapes, media and samplers of 10-03's morning +36,
the foreign types and the nanovdb medium +21, the cylinder and the hair
material +30, nothing since 10-07 (PLAN.md, "The megakernel's 1.5x since
10-02"). A megakernel pays for every arm compiled into its raygen whether
or not a scene reaches it; the wavefront's kernels are per stage and per
arm, and it got faster over the same commits. The schedule's answer is to
copy the megakernel's pixel loop per scene flag as gpu-optix copies its
own per `film_visible_surface` -- but `have_subsurface` is a field of the
VolPath variant rather than a parameter of `render`, which `specialize`
cannot name yet. Measured with the flags wired off in the arm,
killeroo-simple at 64 spp: 459 ms now, 378 without the subsurface code,
346 without the media code too, against 10-02's 298 -- the two flags
would give back 113 of the 161 ms, and the 48 left are the shapes and
materials of 10-03 that every megakernel carries. The two ways to make
the flags schedulable are to-do item 0's.

## Where it stands (2026-10-15, 16:15)

**The megakernel schedules render book again; the crash was the
compiler's mirror pass.** SSA/MirrorLoads.cpp computes a pure value a
loop derives from a cell in memory where the cell is written instead,
and with more than one such value per cell it placed the second's entry
update before the cell's first store, so the light tree's descent
indexed the tree with an undefined value; book's tree is the protocol's
one with an interior node. The narrowing of element reads (10-13,
2b29f205) is what left three such chains in `bvh_descend`. Fixed at the
root (each update placed where its write stands when it is placed),
tests at every level (mirror-derived-loads-twice), the suite green;
PLAN.md, "Found and fixed". Measured on the protocol's three scenes
(apps/pbrt/scratch/gpu-compare-1015-mega; kernel milliseconds, the
least of three; gpu-optix's column is 08:45's run):

| scene | spp | pbrt | gpu-optix | gpu-optix-mega | gpu-optix-mega-ser | mega over pbrt | mega-ser over pbrt |
|---|---|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.5 | 28.0 | 116.2 | 73.4 | 0.43x | 0.67x |
| killeroo-gold | 16 | 148.8 | 68.9 | 317.4 | 266.4 | 0.47x | 0.56x |
| book | 16 | 153.2 | 84.7 | 338.8 | 277.7 | 0.45x | 0.55x |
| killeroo-simple | 64 | 197.5 | 111.7 | 459.5 | 298.2 | 0.43x | 0.66x |
| killeroo-gold | 64 | 596.0 | 275.4 | 1259.2 | 1073.4 | 0.47x | 0.56x |
| book | 64 | 613.3 | 338.2 | 1365.1 | 1126.9 | 0.45x | 0.54x |

Every image matches pbrt's; the megakernel images are byte-identical to
each other's, and book's to gpu-optix's of 08:45. The megakernels are
1.5x to 1.8x slower than on 10-02, when they last ran on book, while
pbrt and gpu-optix are faster than then; the cause is open, the
driver's closest-hit cost of 10-07 the first place to look (to-do item
4).

## Where it stands (2026-10-15, 08:45)

**The renderer's four spawned contributions are void functions adding
into the sample's radiance, every accumulate into a reducer is atomic
by declaration, and the demotion takes all of them off.** The rulings of
10-13 (PLAN.md, "`spawn f(args);` and `join;`"): spawn is `spawn
f(args);`, a call of a void function reporting through its `reduce(op)`
arguments, `join;` a statement, the join implicit at a function's
return as Cilk's sync is; the SSA keeps Spawn and Join as terminators,
never lowered to the parfor they mean. The demotion asks the whole
program which storage each reducer parameter is handed
(parameter_contention) and reads a drain's slots as distinct within a
pass when no sample's work can push onto the queue twice before the
pass (SSA/DistinctSlots.h: a count over paths through the calls and the
queues drained earlier in the round, where a recursion counts many only
if a push lies on its cycle -- the BVH traversal pushes nothing), and
follows a loopified loop header's name of a parameter to the parameter
(the medium kernel's `l`). The gpu-wavefront PTX, which had sixteen
`atom.relaxed.gpu.global.add.f32` under the first slot rule, has none;
the 98 device atomics left are the queues' counts. Measured on the
protocol's three scenes against 10-12's queue-per-arm run (kernel
milliseconds, least of three on each side):

| scene | spp | pbrt | queue per arm (10-12) | spawn form | kernel speedup over pbrt |
|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.3 | 28.02 | 27.97 | 1.76x |
| killeroo-gold | 16 | 149.5 | 69.03 | 68.89 | 2.17x |
| book | 16 | 152.9 | 84.76 | 84.73 | 1.80x |
| killeroo-simple | 64 | 198.4 | 111.70 | 111.73 | 1.78x |
| killeroo-gold | 64 | 597.2 | 275.65 | 275.39 | 2.17x |
| book | 64 | 612.2 | 338.35 | 338.19 | 1.81x |

Nothing moved -- 1.001x kernel as a geometric mean at 16 spp, 1.000x at
64, every cell within 0.2%, 1.90x over pbrt at 16 spp and 1.91x at 64 --
and all six radiance images are byte-identical to 10-12's: under
gpu-optix the contributions' accumulates were plain before and are plain
again, the same kernels. What the compare found on the way: the
megakernel schedules (gpu-optix-mega, gpu-optix-mega-ser) abort on book
with CUDA_ERROR_ILLEGAL_ADDRESS at the first synchronize, under the
last committed compiler as well, so it predates this work (they were
last measured on book on 10-02); to-do item 4 -- fixed at 16:15, the
section above.

## Where it stands (2026-10-12, 21:25)

**The surface's initializer is written only for a gbuffer film, by
specializing the pixel loop on the film's flag, and a queue on a
specialized loop is one queue per arm.** The ruling of 16:40 is built
(da37462e, 2356177e; docs/schedule-semantics.md 4.6): `render[VolPath]
.specialize(film_visible_surface)` ahead of the queues copies the pixel
loop per value of the flag, every queue below is one per arm with its
own record, drains and rest pass under the arm's dispatch, and the host
tests the flag once and launches one arm's kernels -- 102 device kernels
and twelve OptiX modules in the program, one set per film kind, of which
a scene runs one. The RGB arm's record has no array of the visible
surface at all (`rays!False.visible : VisibleSurface as nothing
(unread)`), so its camera kernel writes 41 bytes a pixel sample fewer
than before (26 stores, 8 of them 128-bit and 3 bytes, against 32, 9
and 4): pbrt's `if (initializeVisibleSurface)` as code that is not
there. The first compare of the build drew every scene dark and found
two bugs in how the passes after the deferral read an arm's queue, both
fixed and tested (PLAN.md, "The first compare of the arms build
failed"): the launch gates matched a split to its queue by exact name and
lost the arm queues' splits, so the shadow drain was gated on
`have_media` and skipped in every scene without media; and a root queue's
arm chain was walked from the loop alone, missing the drains beside it.
Measured on the protocol's five scenes against the wavelengths run
(kernel milliseconds, least of three on each side):

| scene | spp | pbrt | wavelengths beside the surface | queue per arm |
|---|---|---|---|---|
| killeroo-simple | 16 | 49.3 | 28.41 | 28.02 |
| killeroo-gold | 16 | 149.6 | 69.94 | 69.03 |
| book | 16 | 152.9 | 85.59 | 84.76 |
| lte-orb-silver | 16 | 161.4 | 79.80 | 78.90 |
| lte-orb-rough-glass | 16 | 335.5 | 199.81 | 198.36 |
| killeroo-simple | 64 | 198.3 | 113.06 | 111.70 |
| killeroo-gold | 64 | 598.2 | 278.86 | 275.65 |
| book | 64 | 612.2 | 341.60 | 338.35 |
| lte-orb-silver | 64 | 645.2 | 319.15 | 314.51 |
| lte-orb-rough-glass | 64 | 1344.3 | 799.08 | 792.77 |

1.011x kernel as a geometric mean at both counts, every cell faster
(1.007x to 1.015x), 1.88x over pbrt at 16 spp and 1.89x at 64; all ten
radiance images byte-identical to heatmap5's. The camera-ray stage, where
the bytes were: 19.3 ms against pbrt's 27.8 summed over the five scenes
at 16 spp (0.69 of pbrt's; 22.0 and 0.80 in the wavelengths run), 76.0
against 111.6 at 64 (0.68; was 0.77), the worst scene lte-orb-silver 0.79
(was 0.94). With this the rulings of 11:50 and 16:40 are all built,
measured and committed; the day's kernel speedups: 1.012x (vector
leaves), 1.047x (the record a struct of arrays), 1.003x (the
wavelengths), 1.011x (the arm), which with the 1.110x over heatmap5 the
record's section reports makes 1.13x over heatmap5 for the day.

## Where it stands (2026-10-12, 15:50)

**The two remaining rulings of 11:50 are built, tested and measured.**
The wavelengths are their own per-sample value beside the visible
surface, as pbrt's `pixelSampleState.lambda` is (render.bonsai:
`path_lambda`, handed down the chain next to `visible` through the
seventeen functions that carry the surface; `VisibleSurface` is pbrt's
`{n, ns, albedo, set}`; the gbuffer's accumulation is under the film's
flag). A layout group's base is aligned to its largest member and its
rows are read and written at that alignment (09e3b202): the primer's
20-byte node went from twenty byte loads to five word loads and a
half-word, 26 tree and layout goldens gained the aligned loads and the
boundary check `bonsai_buffer_require_aligned`, which refuses a buffer
that does not start where the group's rows need it rather than letting a
kernel fault; this renderer's kernels are untouched by it (the RT cores
traverse, the scene arrays are plain arrays), so apps/rtq's and
apps/wosx's GPU traversals are where it lands. Measured together on the
five scenes against the record run (kernel milliseconds, least of three
on each side):

| scene | spp | pbrt | struct-of-arrays record | wavelengths beside the surface |
|---|---|---|---|---|
| killeroo-simple | 16 | 49.5 | 28.29 | 28.41 |
| killeroo-gold | 16 | 148.7 | 69.43 | 69.94 |
| book | 16 | 153.5 | 85.68 | 85.59 |
| lte-orb-silver | 16 | 161.4 | 81.24 | 79.80 |
| lte-orb-rough-glass | 16 | 336.2 | 201.02 | 199.81 |
| killeroo-simple | 64 | 197.0 | 113.18 | 113.06 |
| killeroo-gold | 64 | 596.2 | 277.29 | 278.86 |
| book | 64 | 611.5 | 341.77 | 341.60 |
| lte-orb-silver | 64 | 645.2 | 324.93 | 319.15 |
| lte-orb-rough-glass | 64 | 1342.7 | 801.77 | 799.08 |

1.003x kernel as a geometric mean at both counts, within the noise, as a
change of faithfulness should be (lte-orb-silver 1.018x, the material
kernels storing two 16-byte vectors fewer); all ten radiance images
byte-identical to heatmap5's. What the ruling's other half asked --
the surface's initializer written only for a gbuffer film, pbrt's `if
(initializeVisibleSurface)` -- is not done, and `specialize()` on the
film's flag alone would not do it: the record is one storage for every
copy of the loop and the rest pass reads it under the flag at run time,
so the unread rule keeps the arrays and the RGB copy keeps its 41 bytes
of stores a pixel sample. PLAN.md ("The wavelengths beside the surface,
and the surface's initialization") lays out the three honest ways --
specialize before the deferral (a queue, a record and a set of kernels
per film kind), a declaration without an initializer (a language
addition), or pbrt's `set` flag as the tag of an ADT `Empty | Seen` whose
empty variant stores one byte -- and recommends the third; the user's
call.

## Where it stands (2026-10-12, 14:40)

**Every array the compiler induces is a struct of arrays, and the
camera-ray deficit is closed.** The user's ruling of 11:50 -- the 10-11
rule for a cut's stored set was for every storage a directive makes, with
a handle named for the stored thing -- is built (SSA/InducedStorage.h,
a6edb516): the record a deferral keeps per pixel sample, until now one
array of `VisibleSurface` structs at a 96-byte stride, is six arrays (the
two normals, the albedo, the two halves of the wavelengths, the set
byte), every chain function's `visible : mut VisibleSurface` is a pointer
per leaf at every call, and a leaf nothing reads has no array and no
store. The transforms register their storage under `rays.visible`,
`rays.l`, `rays.result`, `rays.rest.<value>`, `rays.vol_ray_samples`,
`<name>!expanded`, which the post-schedule dump lists and a later
directive can take. The camera kernel writes the record in nine stores
into arrays at 12- and 16-byte strides, three of them 128-bit, where it
wrote eight into the struct -- about half the sectors per warp -- and the
rest pass reads one load per leaf. Measured on the protocol's five scenes
against the morning's vector-leaf run (kernel milliseconds, least of
three on each side, pbrt rerun beside ours):

| scene | spp | pbrt | step 1 | vector leaves | struct-of-arrays record | kernel speedup over pbrt | wall |
|---|---|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.4 | 29.97 | 29.62 | 28.29 | 1.75x | 3.33x |
| killeroo-gold | 16 | 149.4 | 75.02 | 73.94 | 69.43 | 2.15x | 2.81x |
| book | 16 | 154.0 | 92.82 | 92.06 | 85.68 | 1.80x | 2.43x |
| lte-orb-silver | 16 | 161.2 | 85.41 | 84.34 | 81.24 | 1.98x | 2.45x |
| lte-orb-rough-glass | 16 | 336.1 | 206.50 | 203.88 | 201.02 | 1.67x | 1.73x |
| killeroo-simple | 64 | 198.3 | 120.09 | 118.39 | 113.18 | 1.75x | 2.07x |
| killeroo-gold | 64 | 598.2 | 299.50 | 295.00 | 277.29 | 2.16x | 2.39x |
| book | 64 | 618.0 | 370.62 | 367.47 | 341.77 | 1.81x | 1.99x |
| lte-orb-silver | 64 | 646.1 | 341.96 | 338.17 | 324.93 | 1.99x | 2.08x |
| lte-orb-rough-glass | 64 | 1344.0 | 822.93 | 815.00 | 801.77 | 1.68x | 1.68x |

Every cell faster: 1.047x kernel as a geometric mean at 16 spp and
1.048x at 64 over the vector-leaf run (book 1.074x, killeroo-gold
1.065x), 1.110x over heatmap5 with the three fixes since 10-11; all ten
radiance images byte-identical to heatmap5's. **The camera-ray stage is
ahead of pbrt on all five scenes**: 21.8 ms against pbrt's 27.7 at 16
spp (0.79x), 86.2 against 111.9 at 64 (0.77x), lte-orb-silver the worst
at 0.92x, where the 20:30 tables had it behind on 29 of 30 scenes and
this morning's run at 1.32x; film and the rest moved from 0.52x to 0.40x
of pbrt's with it. The deficit was the record's layout, not the draws.
The two remaining rulings of 11:50 are built and in test: the
wavelengths as their own per-sample value beside the surface, as pbrt
keeps them (`path_lambda`, render.bonsai), with the gbuffer's
accumulation under the film's flag; and a layout group's base aligned to
its largest member with the row loads and stores emitted at that
alignment (CodeGen_LLVM.cpp, `bonsai_buffer_require_aligned`). Their
measurement follows in this file.

## Where it stands (2026-10-12, 11:50)

**The camera-ray kernel read against pbrt's GenerateCameraRays, and the
first of its causes fixed.** The two kernels do the same work in the same
order -- the pixel from the launch index, the sampler's start of the
pixel sample, the wavelength draw, the filter sampled by two binary
searches, the time and lens draws, the ray through the same transforms
-- and the times said so before the code did: ours costs a flat 0.33 to
0.35 ns a ray on all thirty scenes whatever the sampler, where pbrt's
moves with the sampler (0.20 on the two PMJ02BN scenes, 0.27 to 0.34 on
Halton). A kernel that cannot tell a table lookup from a dozen 64-bit
divisions is not spending its time on arithmetic. What differs is the
stores: pbrt writes 212 bytes a pixel sample in 25 instructions, nine of
them 128-bit; ours wrote 276 in 58, forty-eight of them scalar. Twenty of
the scalar stores were the five spectra of the entry -- beta, the two
path densities, the two halves of the wavelengths -- each stored as four
scalars because a 4-vector leaf of a queue was the *packed* vector,
sixteen bytes aligned to four, which no backend can write in one
instruction; pbrt's `SOA<SampledSpectrum>` keeps a `Float4` per entry. A
2- or 4-vector leaf of sixteen bytes or fewer is now stored as the vector
type itself, aligned as it is, so the push's store and every drain's load
are one 128-bit instruction (stored_vector_type, SSA/Defer.cpp; a 3-vector
stays packed). In the assembly the camera kernel's stores went to 28
scalar and nine 128-bit on every sampler, and the Diffuse material kernel's
from 67 scalar to 35. A codegen rule reading alignment off an array's
stride was tried first and withdrawn: fourteen test drivers hand interior
pointers C++ does not align to sixteen, so that promise was unsound.

Measured on the protocol's five scenes against the step-1 run (kernel
milliseconds, least of three on each side, pbrt rerun beside ours):

| scene | spp | pbrt | step 1 (e07e53b4) | vector leaves | kernel speedup over pbrt | wall |
|---|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.5 | 29.97 | 29.62 | 1.67x | 3.39x |
| killeroo-gold | 16 | 149.5 | 75.02 | 73.94 | 2.02x | 2.67x |
| book | 16 | 153.0 | 92.82 | 92.06 | 1.66x | 2.28x |
| lte-orb-silver | 16 | 161.6 | 85.41 | 84.34 | 1.92x | 2.37x |
| lte-orb-rough-glass | 16 | 335.5 | 206.50 | 203.88 | 1.65x | 1.69x |
| killeroo-simple | 64 | 197.7 | 120.09 | 118.39 | 1.67x | 1.99x |
| killeroo-gold | 64 | 598.3 | 299.50 | 295.00 | 2.03x | 2.25x |
| book | 64 | 617.2 | 370.62 | 367.47 | 1.68x | 1.87x |
| lte-orb-silver | 64 | 644.9 | 341.96 | 338.17 | 1.91x | 2.00x |
| lte-orb-rough-glass | 64 | 1336.7 | 822.93 | 815.00 | 1.64x | 1.65x |

Every cell faster, 1.012x kernel as a geometric mean at both counts
(1.060x over heatmap5 with the two steps of 10-11), all ten radiance
images byte-identical to heatmap5's. The stage itself: the camera rays
over the five scenes 37.6 -> 36.8 ms against pbrt's 27.8 at 16 spp
(1.36x -> 1.32x), 149.5 -> 146.1 against 112.0 at 64 (1.34x -> 1.31x);
lte-orb-rough-glass, the worst, 1.64x -> 1.60x. A fifth of the camera
kernel's deficit, then; the rest is in what PLAN.md's section lists and
the user ruled on at 11:50: the visible-surface record slot, a 96-byte
struct per pixel sample where pbrt's pixel state is an array per member
(the ruling: every storage the compiler induces is a struct of arrays,
with a handle named after the stored thing); the wavelengths kept inside
the surface so that the surface is initialized for every film (the
ruling: follow pbrt -- the wavelengths their own per-sample value, the
surface's initialization under `specialize()` on the film's flag); and,
outside this renderer, a layout group's rows read on the GPU at alignment
one (the ruling: a group's base is aligned to its largest member and the
loads emitted aligned). The 48 bytes of constant previous-vertex context
at the camera push and the 3-vectors' 12-byte stride stay listed.

## Where it stands (2026-10-11, 20:30)

**The stage tables were misfiling two of our kernels; regenerated.**
`report_stages.py` matched our camera-ray kernel as `_kernel_p_blk$VolPath`
and the sample pass as `_kernel_rays_blk`. Since the kernels per sampler
of 10-10 (482c0de1) they are `_kernel_p_blk$VolPath$<Sampler>` and
`_kernel_samples_blk$<Sampler>`, which neither pattern matched, so both
fell into "film + the rest". The 13:30 tables therefore read "camera rays:
pbrt only", "sample draws: pbrt only" and "film + the rest: behind on 28
scenes, 1.47x" -- an artifact of the names, not a measurement. With the
patterns fixed, the tables below regenerated from heatmap5 at 16 spp: the
camera rays are behind on 29 of 30 scenes, 303.6 ms against pbrt's 246.7
over the sweep (1.23x), lte-orb-rough-glass 1.66x (7.81 against 4.70 ms)
and lte-orb-silver 1.64x, the halton scenes 1.23x to 1.28x (book 11.4
against 9.09); the sample draws behind on 5 (disney-cloud 1.09x,
lte-orb-silver 1.06x; 0.90x over the sweep); film and the rest ahead on
every scene (0.20x). No other row moved. The stage ours loses on almost
every scene is therefore the kernel that starts a pixel sample: pbrt's
GenerateCameraRays draws the camera's dimensions and makes the ray, ours
does the same work in a kernel specialized per sampler, and the gap is
widest on the two pmj02bn scenes. Not yet looked into; it is the next
kernel to read against pbrt's. The rows behind elsewhere: the shadow
traces on bistro_cafe (1.23x, image FAILED), the closest-hit trace on
landscape (1.01x), the measured material on sportscar (1.08x), BSSRDF
sampling on head (1.18x of 8.5 ms) and the subsurface material on
dragon_10 (1.12x of 11 ms).

**Dead-argument elimination is not a step**, though the 15:55 section
below names it next: the seven draw loads for the hit continuation's dead
parameter exist in the SSA alone. The optimized device IR, read tonight
(`compiler ... -b ptx -o file` writes it beside the PTX), has the
closest-hit program loading none of the slot arrays and the material
kernels never dereferencing the subsurface ones; LLVM's own pass does the
job on our internalized, inlined modules. PLAN.md, step 1's section
(16acbb62).

## Where it stands (2026-10-11, 15:55)

**Two steps of the work-item plan, each measured on its own.** The user's
rulings this afternoon: a cut's stored set is a struct of arrays by
default (step 6 of PLAN.md's plan, commit 28bd4aac), and step 1, a call's
argument read as the callee reads it (e07e53b4). Each was timed against
heatmap5 on killeroo-simple, killeroo-gold, book and the two lte-orb
scenes at 16 and 64 spp, the protocol's set with the two scenes step 6 was
for, pbrt --gpu rerun beside ours each time (least of three on each
side), the first with a compiler kept at 28bd4aac, the second with HEAD.
Kernel time in milliseconds, since pbrt's wall time moved between the
runs (killeroo-simple at 64 spp: 0.21 s in one, 0.26 s in the next) while
the kernel times held:

| scene | spp | pbrt | heatmap5 | +SoA (28bd4aac) | +reads (e07e53b4) |
|---|---|---|---|---|---|
| killeroo-simple | 16 | 49.2 | 31.2 | 30.5 | 30.0 |
| killeroo-gold | 16 | 149.3 | 77.9 | 77.3 | 75.0 |
| book | 16 | 153.9 | 96.0 | 95.7 | 92.8 |
| lte-orb-silver | 16 | 161.5 | 93.0 | 89.2 | 85.4 |
| lte-orb-rough-glass | 16 | 334.1 | 213.7 | 211.0 | 206.5 |
| killeroo-simple | 64 | 198.2 | 124.8 | 122.0 | 120.1 |
| killeroo-gold | 64 | 599.0 | 310.9 | 308.3 | 299.5 |
| book | 64 | 615.9 | 382.6 | 381.2 | 370.6 |
| lte-orb-silver | 64 | 646.3 | 371.6 | 356.3 | 342.0 |
| lte-orb-rough-glass | 64 | 1341.4 | 854.5 | 845.3 | 822.9 |

The struct of arrays alone: kernel 1.018x faster than heatmap5 as a
geometric mean at both counts, every cell faster, lte-orb-silver 1.043x
-- the two cells that had slowed after the sample pass (0.966x and
0.996x on 10-10) are ahead now, the 44-byte struct gathered per entry
being seven arrays of which the exit and scatter kernels read four. The
reads on top: another 1.029x at both counts, every cell faster
(killeroo-gold 1.031x, book 1.030x, silver 1.044x), from entries that
shrank -- hits 271 -> 255 bytes against pbrt's 252, bssrdfs 382 -> 345,
probes 341 -> 316, medium_samples 255 -> 223, the two shadow queues 112
-> 80, emissive 165 -> 149, escaped 161 -> 145. Together 1.047x kernel
over heatmap5 at both counts, 1.050x and 1.044x wall. All ten images are
byte-identical to heatmap5's radiance, and the four GPU tests of the cut
and the reload (the scalar pair and the struct pair) pass. Open from the
two: the material kernels and the closest-hit program still load all
seven draw scalars for a parameter dead once pushes are lowered (PLAN.md,
dead-argument elimination); `cook`'s instruction budget is a heuristic to
replace.

## Where it stands (2026-10-11, 13:30)

**The sample pass over the full sweep: 30 scenes at 16, 64 and 128 spp**
(`compare-out-heatmap5`, 10:24 to 12:48 with a stop of twenty-seven
minutes when the disk filled, GPU clock unlocked and recorded, every cell
pbrt's least of three processes against our least of three renders in one
process, as before). Over `pbrt --gpu`, geometric means: 1.89x wall at 16
spp (1.55x kernel), 1.63x at 64 (1.54x), 1.58x at 128 (1.54x); over the 90
cells 1.70x wall and 1.54x kernel, where heatmap4 -- the sweep of the
10-10 morning, before the sample pass -- was 1.48x and 1.41x. Every cell is
faster than pbrt: the least landscape at 1.24x, 1.15x and 1.14x by count,
the most killeroo-simple at 3.35x (16 spp) and watercolor at 2.39x and
2.38x (64 and 128). Against heatmap4 cell by cell: wall 1.135x faster as a
geometric mean at each of the three counts, kernel 1.089x, no cell slower
by more than 1%; the largest gains frame542 1.40x wall (1.15x kernel),
crown 1.26x, book 1.22x to 1.25x, dambreak 1.24x, killeroo-gold 1.21x; the
least disney-cloud 1.03x and bunny-cloud 1.04x, the media scenes the
sample pass touches least. Images: 84 of 90 match pbrt, and the six that
miss are the cells heatmap4 missed (pavilion-day at 16, bistro_cafe at all
three counts, bistro_vespa at 16 and 64 -- the lit-pixel rule, PLAN.md).
The figure: `compare-out-heatmap5/heatmap-gpu-optix.png`. The stage
tables below are regenerated from this sweep at 16 spp.

**The disk, and one conversion per scene.** The sweep lost its last
twelve scenes at 11:30 to a full disk: every cell of every sweep had
converted its scene again, and 1.7 TB of identical converted copies sat
under the compare-out directories, scratch and eval. They are deleted
(with the 25 root-level compare-out experiment directories of 10-04 to
10-06 and eval's float renders; every table, stats file, log, PNG and
render of the sweeps this file cites stays), the disk is at 44%, and each
scene is now converted once: `apps/pbrt/convert_scene.sh` into the
gitignored `apps/pbrt/scenes/converted/<dir>/<name>.txt`, reused by every
sweep and every count and redone only when the converter, pbrt, the options
or the scene change; gpu_compare.sh converts through it and renders with
the renderer's own `--spp`, pbrt's override transcribed, as `--spp` is
pbrt's on the other side -- verified byte-identical to the sweep's images
on killeroo-simple at 64 (halton), lte-orb-silver at 64 (pmj02bn) and
clouds at 16 (zsobol). The texture sidecars are three times pbrt's
texture memory (landscape 3.04 GB against pbrt's 990 MiB) because the
converter builds a pyramid per imagemap declaration where pbrt caches one
per image file; the fix is a converter change for the user to call
(PLAN.md "The disk, the duplicate conversions, and the full sweep").

**Open.** The plan of this morning (PLAN.md "The work items matched, and
the hit chain"), nothing of it built, awaiting the user's choice: the hits
entry at 271 bytes against pbrt's 252, bssrdfs at 382 against 168, probes
at 369 against 332; the cut's draws stored as one struct per slot where
pbrt's are a struct of arrays (the lte-orb cells); the hit chain's six
dependent loads against pbrt's four; and the two structural differences
that remain by ruling -- the round loop reads the queue's count where pbrt
runs maxDepth rounds blind, and the material drains launch at the last
depth over empty queues where pbrt breaks before them.

## Where it stands (2026-10-10, 21:50)

**pbrt's sample kernel, as a schedule directive, and the structure now
matched kernel for kernel but one.** The user's direction this afternoon:
the GPU schedule should match pbrt's wavefront kernel for kernel, with the
bounce's sample draws a kernel of their own over the ray queue ahead of
the trace kernel and the queue reused by both -- not a copy of the ray's
state onto a second queue, which is what a `stage` onto a new queue does
and what cost 2.2x on 10-09 -- and before building it, a rigorous
semantics of every directive. docs/schedule-semantics.md is that, written
the way the Halide paper writes its model: a running example (one path per
pixel), the loop nest each directive produces, the conditions as lists,
and for defer, stage and the new cut the rewrite in the compiler's
printed SSA; the example and each of its schedules are tests
(tests/bonsai/correctness/llvm/semantics-*, with SSA goldens of the same
names). The directive is `f.stage(g, q, l)` -- and `f.defer(g, q, l)`,
the same cut before the call -- where `q` is the queue whose drain already
runs `f` and `l` names the first pass: the drain becomes two passes over
the same entries in the same round, the first running `f` up to the call
and storing what the rest needs in arrays of the record at the entry's
slot (pbrt's pixelSampleState, indexed by pixelIndex), the second the
drain as it was, reading them back; no entry grows. The downstream
queues do not carry the draws either: a deferral whose pusher was handed a
value the drain loaded from such an array reloads it at its own drain
(pbrt's material kernel reads pixelSampleState.samples[pixelIndex]), so
the material entry went from 311 to 271 bytes against pbrt's 252. And
the camera and sample kernels are one per sampler, as pbrt's templates
are. In gpu-optix.bonsai: `vol_path_step.stage(vol_ray_samples, rays,
samples)`, the split and binds of `samples`, `render.specialize(p_blk,
sampler)` and `render.specialize(samples_blk, sampler)`. Commits f61df4ee
(doc, tests), 4fb63990 (the cut), 4fc8ef18 (the reload), f803bc35 (three
compiler fixes the renderer found: the loop form of specialize seeing a
parameter through a hoisted load and reading the tag before copying the
loop; the reload following every route to the owner's call;
Definitions::parameter seeing through a `let`).

**The 30 scenes at 16 spp** (`compare-out-samples16`, 21:15 to 21:47,
GPU clock unlocked and recorded, against `compare-out-reset16` of this
afternoon cell by cell): wall time 1.057x faster, geometric mean, kernel
time 1.072x; faster on 28 of 30 cells (book 1.18x kernel, killeroo-gold
1.17x, frame542 1.16x, killeroo-simple 1.13x, coated-gold 1.13x, crown,
plume and head 1.11x), the two lte-orb cells slower (silver 0.966x
kernel, rough-glass 0.996x). Against pbrt --gpu the geometric mean over
the 30 cells is 1.87x (from 1.80x), every cell faster, from 1.25x
(landscape) to 2.91x (killeroo-simple); kernel time 0.43 (kroken) to 0.88
(landscape) of pbrt's. Images: the same 27 of 30 match pbrt and the same
three (pavilion-day, bistro_cafe, bistro_vespa) miss the lit-pixel rule
as before; killeroo-simple is byte-identical to the morning's render
after each of the three changes. The two zero-day frames are absent from
both sweeps because pbrt --gpu itself segfaults on them. The 64 and 128
spp rows below are this morning's (`compare-out-heatmap4`); they were not
rerun tonight.

Kernel by kernel on killeroo-simple (pbrt's in brackets): the trace
launch 17.5 to 9.8 ms (16.6) once the draws left it and the hit program
stopped loading and pushing them; the sample kernel 3.6 (3.7); the camera
kernel 2.8 (2.6, from 1.35x pbrt's on 29 scenes to 1.07x here); the
material kernels unchanged. On lte-orb-silver the raygen fell 35.0 to
27.1 and the PMJ02BN sample kernel added 6.6 (6.1), but the CoatedDiffuse
material kernel rose 25.7 to 29.6: it gathers the bounce's samples from
the record by slot, a 44-byte struct per entry over two or three cache
lines, where it had read them from its own compacted arrays, and where
pbrt's SOA<RaySamples> is three Float4 arrays and two floats. Storing a
cut's values in the record as the queue stores an entry, a struct of
arrays, is the next step for those cells.

**The entries against pbrt's work items**, bytes an entry stores: rays 181
(pbrt's RayWorkItem 184), escaped 161 (164), emissive 165 (192), shadow
112 (124), medium_samples 295 (372), exits 252 (330), hits 271 (252),
probes 369 (about 330), bssrdfs 382 (168: the SubsurfaceMaterial by value
where pbrt carries a pointer). What hits still carries over pbrt: the
pixel coordinates (8 bytes, recomputable from the slot), the round counter
`wave` (4, uniform per round), a duplicated material index (4), the
primitive's tag and index for the outgoing direction (5), the alpha and
light indices (8).

**The hit chain.** The closest-hit program reads the record, then
`tris[base + prim]` for the triangle's mesh, then the mesh's record, its
indices and its positions: four dependent loads to pbrt's three, whose
record holds the mesh pointer. The fix is the Mesh arm's `tris`
synthesized from the mesh and an index range, so the triangle is
arithmetic, and the record carrying the mesh index; both are language and
record changes for the user to approve (PLAN.md, "The sample pass on the
same queue").

**The one structural difference left**, by the user's ruling: pbrt runs
`maxDepth` rounds blind, launching over empty queues, where our round
loop reads the ray queue's count on the host each round.

## Where it stood (2026-10-10, 13:40)

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
| Trace closest hit rays (`__raygen__findClosest`; the closest-hit program builds the interaction and routes the ray: escaped, emissive, a material's queue, the medium-sample queue, or straight on across a medium boundary) | `__raygen__traced`: the trace and `vol_route` | yes. Until 10-10 ours routed in the raygen after the trace returned; since the mesh-entity series landed (1ed2eabb, SSA/HitContinuations) the closest-hit program `__closesthit__trace$rays` runs the continuation `vol_path_step!after!hit` -- the hit built and routed inside the program, the raygen keeping the miss path -- as pbrt's does. What still differs inside it is the hit chain: six dependent loads from the record to a vertex against pbrt's four (PLAN.md, 10-11) |
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
Generated by `report_stages.py` from `compare-out-heatmap6` (30 scenes; kernel times at 16 spp).

#### Every scene

| scene | image | kernel ours | kernel pbrt | ratio | wall 16 spp | wall 64 spp | wall 128 spp | wall 256 spp |
|---|---|---|---|---|---|---|---|---|
| killeroos/killeroo-simple | ok | 28.2 | 49.4 | 0.57 | 3.13x | 2.04x | 1.68x | 1.65x |
| killeroos/killeroo-gold | ok | 69.7 | 149.5 | 0.47 | 2.66x | 2.36x | 2.02x | 2.10x |
| killeroos/killeroo-coated-gold | ok | 84.1 | 163.5 | 0.51 | 2.28x | 1.98x | 1.97x | 1.90x |
| pbrt-book/book | ok | 86.0 | 154.3 | 0.56 | 2.36x | 1.98x | 1.82x | 1.74x |
| ganesha/ganesha | ok | 97.9 | 177.1 | 0.55 | 2.09x | 1.99x | 1.85x | 1.78x |
| barcelona-pavilion/pavilion-day | ok | 121.4 | 178.6 | 0.68 | 1.83x | 1.68x | 1.55x | 1.48x |
| lte-orb/lte-orb-simple-ball | ok | 157.0 | 211.7 | 0.74 | 1.69x | 1.41x | 1.40x | 1.36x |
| lte-orb/lte-orb-silver | ok | 81.4 | 161.4 | 0.50 | 2.38x | 2.05x | 2.05x | 1.98x |
| lte-orb/lte-orb-rough-glass | ok | 200.3 | 334.6 | 0.60 | 1.70x | 1.68x | 1.68x | 1.64x |
| head/head | ok | 31.1 | 45.2 | 0.69 | 2.08x | 1.66x | 1.58x | 1.55x |
| sssdragon/dragon_10 | ok | 147.7 | 210.4 | 0.70 | 1.79x | 1.52x | 1.50x | 1.45x |
| villa/villa-lights-on | ok | 45.9 | 84.4 | 0.54 | 2.54x | 1.99x | 1.93x | 1.90x |
| smoke-plume/plume | ok | 190.3 | 300.6 | 0.63 | 1.94x | 1.71x | 1.63x | 1.59x |
| bunny-cloud/bunny-cloud | ok | 2112 | 3067 | 0.69 | 1.47x | 1.45x | 1.46x | 1.45x |
| explosion/explosion | ok | 505.5 | 688.0 | 0.73 | 1.42x | 1.39x | 1.38x | 1.38x |
| disney-cloud/disney-cloud | ok | 5157 | 7050 | 0.73 | 1.37x | 1.37x | 1.37x | 1.37x |
| landscape/view-0 | ok | 898.6 | 1064 | 0.84 | 1.29x | 1.19x | 1.19x | 1.18x |
| crown/crown | ok | 733.0 | 1210 | 0.61 | 1.83x | 1.74x | 1.75x | 1.74x |
| kroken/camera-1 | ok | 740.0 | 1905 | 0.39 | 2.27x | 1.96x | 1.92x | 1.89x |
| sportscar/sportscar-sky | ok | 159.7 | 220.7 | 0.72 | 1.75x | 1.44x | 1.42x | 1.41x |
| bmw-m6/bmw-m6 | ok | 114.4 | 212.4 | 0.54 | 2.35x | 1.97x | 1.94x | 1.88x |
| dambreak/dambreak0 | ok | 397.5 | 690.1 | 0.58 | 2.00x | 1.79x | 1.75x | 1.73x |
| transparent-machines/frame542 | ok | 705.0 | 1209 | 0.58 | 1.65x | 1.61x | 1.59x | 1.60x |
| villa/villa-daylight | ok | 169.5 | 291.2 | 0.58 | 2.02x | 1.84x | 1.80x | 1.78x |
| clouds/clouds | ok | 631.7 | 902.8 | 0.70 | 1.58x | 1.45x | 1.46x | 1.42x |
| sanmiguel/sanmiguel-courtyard | ok | 109.2 | 149.3 | 0.73 | 2.00x | 1.59x | 1.46x | 1.39x |
| bistro/bistro_cafe | ok | 181.2 | 317.4 | 0.57 | 2.27x | 1.85x | 1.80x | 1.77x |
| bistro/bistro_vespa | ok | 188.0 | 326.0 | 0.58 | 2.19x | 1.91x | 1.80x | 1.74x |
| watercolor/camera-1 | ok | 849.5 | 2050 | 0.41 | 2.64x | 2.43x | 2.43x | 2.40x |
| sanmiguel/sanmiguel-courtyard-second | ok | 142.5 | 193.4 | 0.74 | 1.62x | 1.42x | 1.39x | 1.37x |

The wall speedup is pbrt's process over ours, both warm, from table.tsv. Images: 30 of 30 scenes match pbrt at 16 spp; a FAILED row's times are reported but do not count as beaten.

#### Every stage

| stage | scenes | behind on | worst | sum ours | sum pbrt | ratio of sums |
|---|---|---|---|---|---|---|
| closest-hit trace + routing | 30 | 0 | 0.99 (landscape/view-0) | 2311 | 3171 | 0.73 |
| shadow traces (every pass) | 30 | 0 | 0.97 (transparent-machines/frame542) | 4396 | 5307 | 0.83 |
| sample draws (the bounce's dimensions, per ray) | 30 | 2 | 1.09 (disney-cloud/disney-cloud) | 703.3 | 798.8 | 0.88 |
| camera rays | 30 | 0 | 0.80 (lte-orb/lte-orb-silver) | 157.0 | 245.7 | 0.64 |
| escaped rays | 30 | 0 | 0.59 (head/head) | 201.4 | 408.2 | 0.49 |
| emitters hit | 30 | 0 | 0.62 (kroken/camera-1) | 145.6 | 379.4 | 0.38 |
| material: CoatedDiffuse (basic textures) | 21 | 0 | 0.72 (landscape/view-0) | 911.9 | 1646 | 0.55 |
| material: Diffuse (basic textures) | 22 | 0 | 0.71 (sanmiguel/sanmiguel-courtyard) | 396.9 | 948.3 | 0.42 |
| film + the rest | 30 | 2 | 2.26 (head/head) | 143.8 | 979.9 | 0.15 |
| materials with their sample draws (a view, not a stage) | 30 | 0 | 0.79 (lte-orb/lte-orb-simple-ball) | 2705 | 6560 | 0.41 |
| material: Conductor (basic textures) | 11 | 0 | 0.32 (kroken/camera-1) | 144.5 | 537.3 | 0.27 |
| material: CoatedConductor (basic textures) | 6 | 0 | 0.54 (bmw-m6/bmw-m6) | 224.8 | 471.0 | 0.48 |
| material: Dielectric (basic textures) | 16 | 0 | 0.75 (barcelona-pavilion/pavilion-day) | 220.9 | 423.5 | 0.52 |
| material: DiffuseTransmission (basic textures) | 8 | 0 | 0.70 (landscape/view-0) | 162.2 | 321.0 | 0.51 |
| material: Measured (basic textures) | 3 | 2 | 1.06 (sportscar/sportscar-sky) | 178.5 | 174.7 | 1.02 |
| subsurface probe trace | 2 | 0 | 0.83 (sssdragon/dragon_10) | 45.8 | 55.3 | 0.83 |
| subsurface exit | 2 | 0 | 0.53 (head/head) | 11.8 | 22.9 | 0.52 |
| material: Subsurface (basic textures) | 2 | 0 | 0.55 (head/head) | 10.8 | 20.1 | 0.54 |
| material: Diffuse (pbrt only) | 8 | 0 | - | 0 | 176.6 | pbrt only |
| material: ThinDielectric (basic textures) | 2 | 0 | 0.62 (villa/villa-lights-on) | 6.54 | 10.7 | 0.61 |
| material: CoatedDiffuse (universal textures) | 4 | 0 | 0.63 (kroken/camera-1) | 237.5 | 545.5 | 0.44 |
| material: Dielectric (universal textures) | 2 | 0 | 0.66 (villa/villa-lights-on) | 8.10 | 12.5 | 0.65 |
| material: Diffuse (universal textures) | 4 | 0 | 0.72 (villa/villa-daylight) | 182.3 | 473.9 | 0.38 |
| medium sampling | 9 | 0 | 0.70 (clouds/clouds) | 4090 | 6171 | 0.66 |
| medium scattering | 9 | 0 | 0.67 (smoke-plume/plume) | 223.8 | 464.9 | 0.48 |
| material: Mixed (basic textures) | 4 | 0 | - | 20.0 | 0 | ours only |

A stage is beaten when ours is below pbrt's on every scene that has it. "behind on" counts the scenes where it is not; the sums are over all the scenes that have the stage on either side.

##### closest-hit trace + routing

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 417.5 | 420.1 | 0.99 |
| kroken/camera-1 | 162.6 | 196.2 | 0.83 |
| sanmiguel/sanmiguel-courtyard-second | 57.9 | 73.4 | 0.79 |
| transparent-machines/frame542 | 246.1 | 313.5 | 0.79 |
| watercolor/camera-1 | 115.9 | 149.1 | 0.78 |
| sanmiguel/sanmiguel-courtyard | 42.8 | 55.4 | 0.77 |
| crown/crown | 175.8 | 236.6 | 0.74 |
| bistro/bistro_vespa | 55.3 | 76.6 | 0.72 |
| disney-cloud/disney-cloud | 171.9 | 239.3 | 0.72 |
| barcelona-pavilion/pavilion-day | 36.1 | 52.6 | 0.69 |
| bunny-cloud/bunny-cloud | 104.0 | 152.2 | 0.68 |
| ganesha/ganesha | 32.9 | 48.8 | 0.67 |
| clouds/clouds | 95.3 | 145.7 | 0.65 |
| sssdragon/dragon_10 | 26.5 | 41.8 | 0.63 |
| lte-orb/lte-orb-rough-glass | 56.6 | 90.0 | 0.63 |
| explosion/explosion | 48.1 | 76.8 | 0.63 |
| villa/villa-lights-on | 11.5 | 18.6 | 0.62 |
| smoke-plume/plume | 41.3 | 67.1 | 0.62 |
| bistro/bistro_cafe | 45.7 | 74.4 | 0.61 |
| lte-orb/lte-orb-simple-ball | 25.2 | 41.3 | 0.61 |
| villa/villa-daylight | 37.7 | 62.8 | 0.60 |
| dambreak/dambreak0 | 121.3 | 208.6 | 0.58 |
| pbrt-book/book | 29.8 | 52.5 | 0.57 |
| sportscar/sportscar-sky | 36.7 | 65.4 | 0.56 |
| killeroos/killeroo-coated-gold | 25.7 | 45.8 | 0.56 |
| killeroos/killeroo-gold | 26.6 | 47.5 | 0.56 |
| lte-orb/lte-orb-silver | 25.1 | 45.4 | 0.55 |
| head/head | 5.50 | 10.1 | 0.54 |
| bmw-m6/bmw-m6 | 25.1 | 46.9 | 0.53 |
| killeroos/killeroo-simple | 8.84 | 16.7 | 0.53 |

##### shadow traces (every pass)

ours: `__raygen__shadow$<alpha>` and `__raygen__exit_shadow$<alpha>`; pbrt: the shadow and shadow-Tr trace kernels.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| transparent-machines/frame542 | 136.7 | 140.8 | 0.97 |
| head/head | 2.10 | 2.20 | 0.95 |
| explosion/explosion | 139.8 | 149.2 | 0.94 |
| barcelona-pavilion/pavilion-day | 15.6 | 16.9 | 0.92 |
| clouds/clouds | 194.7 | 214.8 | 0.91 |
| smoke-plume/plume | 40.4 | 44.8 | 0.90 |
| sssdragon/dragon_10 | 14.0 | 15.6 | 0.90 |
| sanmiguel/sanmiguel-courtyard | 18.3 | 21.0 | 0.88 |
| landscape/view-0 | 205.2 | 236.2 | 0.87 |
| disney-cloud/disney-cloud | 2117 | 2460 | 0.86 |
| sanmiguel/sanmiguel-courtyard-second | 24.7 | 28.8 | 0.86 |
| watercolor/camera-1 | 63.1 | 73.5 | 0.86 |
| lte-orb/lte-orb-simple-ball | 8.52 | 10.2 | 0.84 |
| ganesha/ganesha | 7.43 | 8.93 | 0.83 |
| villa/villa-lights-on | 2.77 | 3.34 | 0.83 |
| villa/villa-daylight | 9.55 | 11.7 | 0.82 |
| lte-orb/lte-orb-rough-glass | 22.8 | 28.7 | 0.80 |
| kroken/camera-1 | 92.7 | 117.2 | 0.79 |
| bistro/bistro_cafe | 17.4 | 22.3 | 0.78 |
| bistro/bistro_vespa | 21.4 | 27.8 | 0.77 |
| bunny-cloud/bunny-cloud | 1057 | 1418 | 0.75 |
| dambreak/dambreak0 | 42.8 | 58.0 | 0.74 |
| sportscar/sportscar-sky | 9.47 | 12.9 | 0.73 |
| crown/crown | 93.5 | 127.8 | 0.73 |
| bmw-m6/bmw-m6 | 7.13 | 9.82 | 0.73 |
| lte-orb/lte-orb-silver | 6.47 | 9.15 | 0.71 |
| killeroos/killeroo-gold | 6.89 | 9.96 | 0.69 |
| killeroos/killeroo-coated-gold | 6.77 | 9.79 | 0.69 |
| killeroos/killeroo-simple | 3.19 | 4.67 | 0.68 |
| pbrt-book/book | 7.76 | 12.5 | 0.62 |

##### sample draws (the bounce's dimensions, per ray)

ours: the first pass of the `rays` cut, `samples_blk$<Sampler>` since the kernels per sampler of 10-10 (`rays_blk` in the runs before); a run before the sample stage (10-09) has no kernel here, its draws having run inside the material and medium kernels, so pbrt's stands alone there.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| disney-cloud/disney-cloud | 61.7 | 56.8 | **1.09** |
| kroken/camera-1 | 34.9 | 34.4 | **1.01** |
| watercolor/camera-1 | 25.8 | 26.6 | 0.97 |
| barcelona-pavilion/pavilion-day | 7.46 | 7.98 | 0.93 |
| landscape/view-0 | 35.2 | 38.1 | 0.92 |
| dambreak/dambreak0 | 54.7 | 60.2 | 0.91 |
| bunny-cloud/bunny-cloud | 44.3 | 48.8 | 0.91 |
| clouds/clouds | 53.0 | 58.8 | 0.90 |
| smoke-plume/plume | 11.7 | 13.1 | 0.90 |
| sportscar/sportscar-sky | 9.83 | 11.0 | 0.90 |
| explosion/explosion | 19.5 | 21.9 | 0.89 |
| ganesha/ganesha | 8.62 | 9.72 | 0.89 |
| villa/villa-lights-on | 2.95 | 3.35 | 0.88 |
| killeroos/killeroo-simple | 3.20 | 3.65 | 0.88 |
| villa/villa-daylight | 11.0 | 12.7 | 0.87 |
| pbrt-book/book | 9.10 | 10.5 | 0.87 |
| bistro/bistro_cafe | 10.1 | 11.7 | 0.86 |
| bmw-m6/bmw-m6 | 9.00 | 10.6 | 0.85 |
| killeroos/killeroo-coated-gold | 7.67 | 9.02 | 0.85 |
| lte-orb/lte-orb-simple-ball | 7.51 | 8.84 | 0.85 |
| sanmiguel/sanmiguel-courtyard-second | 7.91 | 9.35 | 0.85 |
| killeroos/killeroo-gold | 7.71 | 9.14 | 0.84 |
| bistro/bistro_vespa | 9.96 | 11.9 | 0.84 |
| head/head | 2.65 | 3.17 | 0.84 |
| transparent-machines/frame542 | 151.9 | 183.6 | 0.83 |
| crown/crown | 65.4 | 79.7 | 0.82 |
| sanmiguel/sanmiguel-courtyard | 6.66 | 8.15 | 0.82 |
| sssdragon/dragon_10 | 9.23 | 11.9 | 0.77 |
| lte-orb/lte-orb-silver | 4.60 | 6.15 | 0.75 |
| lte-orb/lte-orb-rough-glass | 10.1 | 18.1 | 0.56 |

##### camera rays

both sides draw the camera's own dimensions here; ours one kernel per sampler since 10-10.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| lte-orb/lte-orb-silver | 3.80 | 4.76 | 0.80 |
| lte-orb/lte-orb-rough-glass | 3.55 | 4.77 | 0.74 |
| killeroos/killeroo-simple | 1.75 | 2.54 | 0.69 |
| lte-orb/lte-orb-simple-ball | 4.83 | 7.28 | 0.66 |
| watercolor/camera-1 | 8.81 | 13.3 | 0.66 |
| villa/villa-lights-on | 1.57 | 2.39 | 0.66 |
| pbrt-book/book | 5.96 | 9.09 | 0.66 |
| killeroos/killeroo-gold | 4.21 | 6.60 | 0.64 |
| ganesha/ganesha | 5.75 | 9.09 | 0.63 |
| killeroos/killeroo-coated-gold | 4.21 | 6.66 | 0.63 |
| villa/villa-daylight | 4.50 | 7.16 | 0.63 |
| sportscar/sportscar-sky | 5.70 | 9.09 | 0.63 |
| kroken/camera-1 | 7.37 | 11.8 | 0.63 |
| dambreak/dambreak0 | 5.75 | 9.19 | 0.63 |
| explosion/explosion | 7.03 | 11.2 | 0.63 |
| sanmiguel/sanmiguel-courtyard | 3.43 | 5.49 | 0.62 |
| sssdragon/dragon_10 | 4.06 | 6.53 | 0.62 |
| bunny-cloud/bunny-cloud | 5.65 | 9.09 | 0.62 |
| sanmiguel/sanmiguel-courtyard-second | 4.42 | 7.13 | 0.62 |
| bmw-m6/bmw-m6 | 4.10 | 6.63 | 0.62 |
| clouds/clouds | 5.80 | 9.39 | 0.62 |
| crown/crown | 4.07 | 6.62 | 0.61 |
| head/head | 1.80 | 2.93 | 0.61 |
| barcelona-pavilion/pavilion-day | 3.99 | 6.52 | 0.61 |
| landscape/view-0 | 22.5 | 37.1 | 0.60 |
| bistro/bistro_cafe | 5.71 | 9.49 | 0.60 |
| transparent-machines/frame542 | 4.45 | 7.42 | 0.60 |
| smoke-plume/plume | 4.08 | 6.81 | 0.60 |
| bistro/bistro_vespa | 5.72 | 9.59 | 0.60 |
| disney-cloud/disney-cloud | 2.46 | 0 | ours only |

##### escaped rays

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 1.73 | 2.91 | 0.59 |
| villa/villa-lights-on | 0.60 | 1.03 | 0.58 |
| sssdragon/dragon_10 | 3.81 | 6.59 | 0.58 |
| explosion/explosion | 5.69 | 9.91 | 0.57 |
| barcelona-pavilion/pavilion-day | 2.30 | 4.33 | 0.53 |
| landscape/view-0 | 12.9 | 24.5 | 0.53 |
| smoke-plume/plume | 4.15 | 7.93 | 0.52 |
| clouds/clouds | 17.4 | 34.5 | 0.50 |
| lte-orb/lte-orb-simple-ball | 3.64 | 7.33 | 0.50 |
| bmw-m6/bmw-m6 | 3.89 | 7.85 | 0.50 |
| ganesha/ganesha | 4.33 | 8.85 | 0.49 |
| sportscar/sportscar-sky | 5.39 | 11.6 | 0.46 |
| bunny-cloud/bunny-cloud | 15.1 | 33.6 | 0.45 |
| sanmiguel/sanmiguel-courtyard | 1.35 | 3.04 | 0.44 |
| disney-cloud/disney-cloud | 11.7 | 26.9 | 0.43 |
| bistro/bistro_cafe | 1.80 | 4.22 | 0.43 |
| villa/villa-daylight | 2.47 | 5.89 | 0.42 |
| dambreak/dambreak0 | 16.7 | 39.9 | 0.42 |
| bistro/bistro_vespa | 1.79 | 4.29 | 0.42 |
| watercolor/camera-1 | 10.8 | 26.5 | 0.41 |
| kroken/camera-1 | 6.60 | 16.4 | 0.40 |
| sanmiguel/sanmiguel-courtyard-second | 1.33 | 3.33 | 0.40 |
| transparent-machines/frame542 | 41.9 | 116.9 | 0.36 |
| killeroos/killeroo-simple | 0.41 | 0 | ours only |
| killeroos/killeroo-gold | 2.40 | 0 | ours only |
| killeroos/killeroo-coated-gold | 2.36 | 0 | ours only |
| pbrt-book/book | 0.96 | 0 | ours only |
| lte-orb/lte-orb-silver | 0.88 | 0 | ours only |
| lte-orb/lte-orb-rough-glass | 4.34 | 0 | ours only |
| crown/crown | 12.7 | 0 | ours only |

##### emitters hit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 9.87 | 16.0 | 0.62 |
| bistro/bistro_cafe | 2.92 | 4.95 | 0.59 |
| ganesha/ganesha | 2.24 | 3.87 | 0.58 |
| bistro/bistro_vespa | 3.03 | 5.27 | 0.57 |
| lte-orb/lte-orb-silver | 1.84 | 3.33 | 0.55 |
| crown/crown | 22.2 | 41.6 | 0.54 |
| lte-orb/lte-orb-rough-glass | 6.86 | 13.4 | 0.51 |
| head/head | 0.25 | 0.50 | 0.50 |
| villa/villa-lights-on | 0.44 | 0.93 | 0.47 |
| killeroos/killeroo-simple | 0.46 | 0.98 | 0.47 |
| barcelona-pavilion/pavilion-day | 1.05 | 2.28 | 0.46 |
| smoke-plume/plume | 1.04 | 2.36 | 0.44 |
| bmw-m6/bmw-m6 | 1.49 | 3.46 | 0.43 |
| sssdragon/dragon_10 | 0.97 | 2.27 | 0.43 |
| watercolor/camera-1 | 4.45 | 11.5 | 0.39 |
| killeroos/killeroo-coated-gold | 1.28 | 3.36 | 0.38 |
| sanmiguel/sanmiguel-courtyard | 0.72 | 1.89 | 0.38 |
| villa/villa-daylight | 1.83 | 4.82 | 0.38 |
| pbrt-book/book | 1.20 | 3.18 | 0.38 |
| disney-cloud/disney-cloud | 9.35 | 25.3 | 0.37 |
| killeroos/killeroo-gold | 1.29 | 3.50 | 0.37 |
| explosion/explosion | 1.46 | 3.97 | 0.37 |
| sanmiguel/sanmiguel-courtyard-second | 0.92 | 2.55 | 0.36 |
| lte-orb/lte-orb-simple-ball | 0.85 | 2.36 | 0.36 |
| bunny-cloud/bunny-cloud | 9.45 | 26.3 | 0.36 |
| dambreak/dambreak0 | 12.1 | 33.6 | 0.36 |
| clouds/clouds | 9.45 | 26.5 | 0.36 |
| landscape/view-0 | 4.44 | 12.6 | 0.35 |
| sportscar/sportscar-sky | 1.03 | 3.25 | 0.32 |
| transparent-machines/frame542 | 31.1 | 113.5 | 0.27 |

##### material: CoatedDiffuse (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 74.2 | 102.7 | 0.72 |
| crown/crown | 136.6 | 203.9 | 0.67 |
| explosion/explosion | 20.1 | 30.4 | 0.66 |
| sanmiguel/sanmiguel-courtyard-second | 14.3 | 21.8 | 0.65 |
| sanmiguel/sanmiguel-courtyard | 11.5 | 17.8 | 0.65 |
| sssdragon/dragon_10 | 26.8 | 41.5 | 0.65 |
| killeroos/killeroo-simple | 4.83 | 7.51 | 0.64 |
| sportscar/sportscar-sky | 8.77 | 14.2 | 0.62 |
| pbrt-book/book | 13.7 | 22.5 | 0.61 |
| barcelona-pavilion/pavilion-day | 24.0 | 40.5 | 0.59 |
| lte-orb/lte-orb-rough-glass | 57.9 | 98.7 | 0.59 |
| bunny-cloud/bunny-cloud | 67.9 | 119.3 | 0.57 |
| villa/villa-daylight | 38.7 | 68.3 | 0.57 |
| lte-orb/lte-orb-silver | 26.9 | 50.6 | 0.53 |
| kroken/camera-1 | 87.7 | 166.0 | 0.53 |
| bistro/bistro_cafe | 77.2 | 148.9 | 0.52 |
| bmw-m6/bmw-m6 | 30.2 | 59.1 | 0.51 |
| ganesha/ganesha | 23.6 | 48.5 | 0.49 |
| bistro/bistro_vespa | 68.3 | 144.2 | 0.47 |
| villa/villa-lights-on | 9.75 | 22.2 | 0.44 |
| watercolor/camera-1 | 89.0 | 217.4 | 0.41 |

##### material: Diffuse (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sanmiguel/sanmiguel-courtyard | 17.2 | 24.1 | 0.71 |
| sanmiguel/sanmiguel-courtyard-second | 23.5 | 33.5 | 0.70 |
| landscape/view-0 | 16.7 | 23.9 | 0.70 |
| villa/villa-daylight | 6.46 | 10.1 | 0.64 |
| bmw-m6/bmw-m6 | 4.18 | 6.63 | 0.63 |
| smoke-plume/plume | 9.13 | 14.8 | 0.62 |
| sportscar/sportscar-sky | 2.62 | 4.40 | 0.60 |
| barcelona-pavilion/pavilion-day | 3.67 | 6.29 | 0.58 |
| lte-orb/lte-orb-silver | 2.52 | 4.54 | 0.56 |
| dambreak/dambreak0 | 38.4 | 73.9 | 0.52 |
| villa/villa-lights-on | 1.44 | 2.79 | 0.52 |
| lte-orb/lte-orb-rough-glass | 9.99 | 19.4 | 0.52 |
| crown/crown | 54.3 | 114.7 | 0.47 |
| disney-cloud/disney-cloud | 19.9 | 46.4 | 0.43 |
| killeroos/killeroo-simple | 4.80 | 11.5 | 0.42 |
| pbrt-book/book | 14.8 | 37.9 | 0.39 |
| lte-orb/lte-orb-simple-ball | 10.5 | 29.4 | 0.36 |
| watercolor/camera-1 | 70.1 | 203.8 | 0.34 |
| killeroos/killeroo-coated-gold | 12.5 | 39.2 | 0.32 |
| killeroos/killeroo-gold | 12.8 | 40.2 | 0.32 |
| ganesha/ganesha | 10.4 | 33.4 | 0.31 |
| kroken/camera-1 | 51.1 | 167.3 | 0.31 |

##### film + the rest

ours: `rays_rest_blk` and `q_blk`; pbrt: Update film, the queue resets, the stats updates, Evaluate Displacement, Other.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 3.80 | 1.68 | **2.26** |
| sssdragon/dragon_10 | 7.23 | 5.29 | **1.37** |
| sportscar/sportscar-sky | 2.70 | 5.84 | 0.46 |
| bistro/bistro_vespa | 2.68 | 5.87 | 0.46 |
| ganesha/ganesha | 2.67 | 5.91 | 0.45 |
| landscape/view-0 | 9.90 | 22.2 | 0.45 |
| bistro/bistro_cafe | 2.60 | 5.89 | 0.44 |
| lte-orb/lte-orb-silver | 2.03 | 4.67 | 0.43 |
| explosion/explosion | 3.16 | 7.30 | 0.43 |
| pbrt-book/book | 2.62 | 6.08 | 0.43 |
| sanmiguel/sanmiguel-courtyard-second | 2.12 | 4.98 | 0.43 |
| barcelona-pavilion/pavilion-day | 1.93 | 4.55 | 0.42 |
| smoke-plume/plume | 2.05 | 4.86 | 0.42 |
| villa/villa-daylight | 2.63 | 6.30 | 0.42 |
| killeroos/killeroo-coated-gold | 1.99 | 4.79 | 0.42 |
| sanmiguel/sanmiguel-courtyard | 1.67 | 4.08 | 0.41 |
| killeroos/killeroo-gold | 1.96 | 4.87 | 0.40 |
| villa/villa-lights-on | 0.74 | 1.85 | 0.40 |
| bmw-m6/bmw-m6 | 2.14 | 5.37 | 0.40 |
| killeroos/killeroo-simple | 0.73 | 1.90 | 0.38 |
| lte-orb/lte-orb-rough-glass | 4.28 | 12.0 | 0.36 |
| clouds/clouds | 6.93 | 19.5 | 0.35 |
| dambreak/dambreak0 | 8.25 | 23.3 | 0.35 |
| bunny-cloud/bunny-cloud | 6.91 | 19.8 | 0.35 |
| transparent-machines/frame542 | 24.4 | 73.1 | 0.33 |
| crown/crown | 10.5 | 32.1 | 0.33 |
| lte-orb/lte-orb-simple-ball | 3.78 | 13.3 | 0.28 |
| disney-cloud/disney-cloud | 6.35 | 25.5 | 0.25 |
| watercolor/camera-1 | 7.77 | 69.5 | 0.11 |
| kroken/camera-1 | 7.34 | 577.4 | 0.01 |

##### materials with their sample draws (a view, not a stage)

the material kernels above summed with the sample draws on each side -- pbrt's `Generate ray samples`, and our sample kernel where the run has one (until the sample stage of 10-09 the bounce's draws ran inside our material kernels) -- the like-for-like sum either way; a second view of rows already counted, left out of the totals.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| lte-orb/lte-orb-simple-ball | 102.6 | 130.0 | 0.79 |
| sportscar/sportscar-sky | 88.9 | 112.6 | 0.79 |
| landscape/view-0 | 191.0 | 310.9 | 0.61 |
| sanmiguel/sanmiguel-courtyard-second | 43.3 | 73.2 | 0.59 |
| sanmiguel/sanmiguel-courtyard | 34.2 | 58.5 | 0.58 |
| barcelona-pavilion/pavilion-day | 53.0 | 91.4 | 0.58 |
| villa/villa-daylight | 99.8 | 192.6 | 0.52 |
| lte-orb/lte-orb-rough-glass | 91.7 | 185.8 | 0.49 |
| sssdragon/dragon_10 | 33.1 | 67.2 | 0.49 |
| bistro/bistro_cafe | 95.0 | 196.2 | 0.48 |
| bmw-m6/bmw-m6 | 61.6 | 132.4 | 0.47 |
| villa/villa-lights-on | 25.3 | 56.2 | 0.45 |
| bistro/bistro_vespa | 88.2 | 196.6 | 0.45 |
| crown/crown | 285.2 | 639.4 | 0.45 |
| kroken/camera-1 | 389.4 | 902.8 | 0.43 |
| killeroos/killeroo-simple | 9.63 | 22.6 | 0.43 |
| dambreak/dambreak0 | 85.9 | 212.0 | 0.41 |
| pbrt-book/book | 28.5 | 70.9 | 0.40 |
| lte-orb/lte-orb-silver | 36.7 | 94.1 | 0.39 |
| head/head | 4.53 | 11.8 | 0.38 |
| ganesha/ganesha | 33.9 | 91.7 | 0.37 |
| killeroos/killeroo-coated-gold | 34.1 | 93.1 | 0.37 |
| explosion/explosion | 20.1 | 55.6 | 0.36 |
| watercolor/camera-1 | 583.4 | 1622 | 0.36 |
| bunny-cloud/bunny-cloud | 67.9 | 194.1 | 0.35 |
| smoke-plume/plume | 11.1 | 37.1 | 0.30 |
| killeroos/killeroo-gold | 18.7 | 77.1 | 0.24 |
| disney-cloud/disney-cloud | 19.9 | 103.2 | 0.19 |
| transparent-machines/frame542 | 68.6 | 443.7 | 0.15 |
| clouds/clouds | 0 | 85.4 | pbrt only |

##### material: Conductor (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 23.1 | 72.8 | 0.32 |
| villa/villa-daylight | 6.20 | 20.3 | 0.31 |
| crown/crown | 55.1 | 191.0 | 0.29 |
| villa/villa-lights-on | 1.63 | 5.77 | 0.28 |
| bmw-m6/bmw-m6 | 4.67 | 16.8 | 0.28 |
| sportscar/sportscar-sky | 2.98 | 11.2 | 0.26 |
| watercolor/camera-1 | 33.7 | 141.2 | 0.24 |
| barcelona-pavilion/pavilion-day | 1.90 | 8.46 | 0.22 |
| lte-orb/lte-orb-silver | 7.32 | 32.7 | 0.22 |
| killeroos/killeroo-gold | 5.90 | 27.8 | 0.21 |
| smoke-plume/plume | 1.95 | 9.27 | 0.21 |

##### material: CoatedConductor (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| bmw-m6/bmw-m6 | 16.9 | 31.3 | 0.54 |
| bistro/bistro_cafe | 15.4 | 28.8 | 0.54 |
| kroken/camera-1 | 78.0 | 148.8 | 0.52 |
| bistro/bistro_vespa | 16.4 | 32.2 | 0.51 |
| killeroos/killeroo-coated-gold | 21.6 | 44.8 | 0.48 |
| watercolor/camera-1 | 76.4 | 185.1 | 0.41 |

##### material: Dielectric (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| barcelona-pavilion/pavilion-day | 4.86 | 6.44 | 0.75 |
| sanmiguel/sanmiguel-courtyard | 1.87 | 2.77 | 0.68 |
| sanmiguel/sanmiguel-courtyard-second | 1.95 | 2.92 | 0.67 |
| villa/villa-lights-on | 0.88 | 1.39 | 0.63 |
| dambreak/dambreak0 | 47.5 | 77.9 | 0.61 |
| villa/villa-daylight | 2.97 | 4.88 | 0.61 |
| bistro/bistro_vespa | 3.40 | 5.61 | 0.61 |
| sportscar/sportscar-sky | 2.12 | 3.53 | 0.60 |
| bistro/bistro_cafe | 2.34 | 4.08 | 0.57 |
| bmw-m6/bmw-m6 | 4.54 | 7.95 | 0.57 |
| crown/crown | 27.1 | 50.1 | 0.54 |
| watercolor/camera-1 | 6.19 | 12.3 | 0.50 |
| kroken/camera-1 | 17.5 | 35.6 | 0.49 |
| lte-orb/lte-orb-rough-glass | 23.9 | 49.6 | 0.48 |
| transparent-machines/frame542 | 68.6 | 147.0 | 0.47 |
| landscape/view-0 | 5.18 | 11.4 | 0.45 |

##### material: DiffuseTransmission (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| landscape/view-0 | 94.9 | 134.7 | 0.70 |
| barcelona-pavilion/pavilion-day | 4.52 | 7.02 | 0.64 |
| sanmiguel/sanmiguel-courtyard-second | 3.57 | 5.58 | 0.64 |
| sanmiguel/sanmiguel-courtyard | 3.60 | 5.63 | 0.64 |
| villa/villa-lights-on | 1.41 | 2.59 | 0.54 |
| villa/villa-daylight | 4.48 | 8.54 | 0.52 |
| kroken/camera-1 | 28.3 | 88.0 | 0.32 |
| watercolor/camera-1 | 21.5 | 68.9 | 0.31 |

##### material: Measured (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sportscar/sportscar-sky | 72.4 | 68.3 | **1.06** |
| lte-orb/lte-orb-simple-ball | 92.2 | 91.7 | **1.00** |
| barcelona-pavilion/pavilion-day | 14.0 | 14.7 | 0.95 |

##### subsurface probe trace

| scene | ours | pbrt | ratio |
|---|---|---|---|
| sssdragon/dragon_10 | 40.3 | 48.4 | 0.83 |
| head/head | 5.47 | 6.87 | 0.80 |

##### subsurface exit

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 3.27 | 6.22 | 0.53 |
| sssdragon/dragon_10 | 8.56 | 16.7 | 0.51 |

##### material: Subsurface (basic textures)

one kernel of ours against two of pbrt's: the material evaluation and the BSSRDF sampling that enqueues the probe.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head | 4.53 | 8.30 | 0.55 |
| sssdragon/dragon_10 | 6.29 | 11.8 | 0.53 |

##### material: Diffuse (pbrt only)

pbrt launches a kernel for every material of the scene at every depth, reached by a ray or not; ours launches only for a non-empty queue, so a material no ray of ours reached has no kernel here. Each row says how many launches pbrt made and what each cost: at pbrt's launch floor of about 0.015 ms they were empty.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| head/head (32 launches, 0.011 ms each) | 0 | 0.35 | pbrt only |
| sssdragon/dragon_10 (160 launches, 0.012 ms each) | 0 | 1.90 | pbrt only |
| bunny-cloud/bunny-cloud (1600 launches, 0.016 ms each) | 0 | 26.0 | pbrt only |
| explosion/explosion (240 launches, 0.014 ms each) | 0 | 3.32 | pbrt only |
| transparent-machines/frame542 (8192 launches, 0.014 ms each) | 0 | 113.1 | pbrt only |
| clouds/clouds (1600 launches, 0.017 ms each) | 0 | 26.6 | pbrt only |
| bistro/bistro_cafe (160 launches, 0.017 ms each) | 0 | 2.70 | pbrt only |
| bistro/bistro_vespa (160 launches, 0.017 ms each) | 0 | 2.72 | pbrt only |

##### material: ThinDielectric (basic textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 1.23 | 1.97 | 0.62 |
| villa/villa-daylight | 5.31 | 8.77 | 0.61 |

##### material: CoatedDiffuse (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| kroken/camera-1 | 69.5 | 110.8 | 0.63 |
| villa/villa-daylight | 22.2 | 39.4 | 0.57 |
| villa/villa-lights-on | 6.18 | 12.2 | 0.51 |
| watercolor/camera-1 | 139.5 | 383.1 | 0.36 |

##### material: Dielectric (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-lights-on | 1.56 | 2.36 | 0.66 |
| villa/villa-daylight | 6.54 | 10.1 | 0.65 |

##### material: Diffuse (universal textures)

| scene | ours | pbrt | ratio |
|---|---|---|---|
| villa/villa-daylight | 6.87 | 9.60 | 0.72 |
| villa/villa-lights-on | 1.20 | 1.69 | 0.71 |
| kroken/camera-1 | 30.8 | 79.1 | 0.39 |
| watercolor/camera-1 | 143.4 | 383.6 | 0.37 |

##### medium sampling

| scene | ours | pbrt | ratio |
|---|---|---|---|
| clouds/clouds | 209.5 | 299.5 | 0.70 |
| explosion/explosion | 255.4 | 365.8 | 0.70 |
| disney-cloud/disney-cloud | 2681 | 4027 | 0.67 |
| bunny-cloud/bunny-cloud | 767.5 | 1157 | 0.66 |
| smoke-plume/plume | 68.2 | 120.0 | 0.57 |
| crown/crown | 49.3 | 87.3 | 0.56 |
| watercolor/camera-1 | 11.2 | 20.2 | 0.56 |
| kroken/camera-1 | 14.8 | 26.9 | 0.55 |
| dambreak/dambreak0 | 33.5 | 68.3 | 0.49 |

##### medium scattering

| scene | ours | pbrt | ratio |
|---|---|---|---|
| smoke-plume/plume | 6.32 | 9.49 | 0.67 |
| explosion/explosion | 5.26 | 8.27 | 0.64 |
| bunny-cloud/bunny-cloud | 33.4 | 56.8 | 0.59 |
| clouds/clouds | 39.6 | 67.6 | 0.59 |
| disney-cloud/disney-cloud | 75.4 | 142.8 | 0.53 |
| dambreak/dambreak0 | 16.6 | 37.3 | 0.44 |
| crown/crown | 14.3 | 38.4 | 0.37 |
| kroken/camera-1 | 14.5 | 39.9 | 0.36 |
| watercolor/camera-1 | 18.4 | 64.5 | 0.29 |

##### material: Mixed (basic textures)

pbrt resolves a mix material inside its closest-hit program and has no kernel for it; ours is the choice step.

| scene | ours | pbrt | ratio |
|---|---|---|---|
| crown/crown | 12.1 | 0 | ours only |
| kroken/camera-1 | 3.24 | 0 | ours only |
| bmw-m6/bmw-m6 | 1.08 | 0 | ours only |
| watercolor/camera-1 | 3.61 | 0 | ours only |

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

## To do, in order (2026-10-17, 23:40)

0. The 30-scene gate is met but for pavilion-day's 16-spp cell, whose
   lit-pixel verdict is the 09-29 calibration artifact (pbrt's own two
   renderers disagree there six times more than we do with either): the
   tolerance is right for the simple scenes it was set on, so the
   question is whether the table should carry a per-scene or spp-aware
   allowance for a glass scene under an environment map at 16 spp, or keep
   printing FAILED against the calibration note -- the user's call. Open
   from the fix: the synthetic masked-light probe (probes/masked_light_d1)
   sits 1.0% under pbrt's lit count and mean. The megakernel: the bisect
   put the slowdown on the program's growth into its one raygen, the
   subsurface material first (PLAN.md, "The megakernel's 1.5x since
   10-02"); the schedule fix is to copy its pixel loop per scene flag, and
   two ways would let `specialize` name `have_subsurface` -- make it a
   parameter of `render` as `have_media` and `film_visible_surface` are
   (pbrt keeps all three as WavefrontPathIntegrator members, so the
   program would be no less pbrt's), or teach the loop form of specialize
   to name a field the integrator's arm binds -- the user's call, with the
   recoverable milliseconds measured in PLAN.md. Tooling
   debts found on the way: convert_scene.sh's stamp does not see an
   `Include`d file change; check_differentials.sh aborts in our
   `--print-differentials` (a host buffer without a size); scene_dump
   refuses the orthographic camera.

## To do, in order (2026-10-12, 21:25; items 1-5 as they stood)

1. The user's rulings of 11:50 and 16:40 are done and measured (PLAN.md,
   the four sections of 10-12): the record and every other induced
   storage a struct of arrays (a6edb516, 1.047x kernel, the camera rays
   ahead of pbrt everywhere), the wavelengths beside the surface as pbrt
   keeps them (1.003x, within noise), a layout group's base aligned to
   its largest member (09e3b202, the renderer untouched), and the
   surface's initializer written only for a gbuffer film by one queue
   per arm of the specialized pixel loop (da37462e, 2356177e, 1.011x,
   the camera-ray stage 0.69 of pbrt's).
2. The work-item plan (PLAN.md, "The work items matched, and the hit
   chain"): step 5's program changes (hits 255 -> 241 bytes against pbrt's
   252; the probe's empty start; the subsurface context), then the hit
   chain (step 8: six dependent loads from the record to a vertex against
   pbrt's four -- the closest-hit program's remaining difference, and
   landscape the one scene the trace is behind on), then the recipes of
   steps 2-4 for bssrdfs (345 against 168) and probes (316 against 332);
   the 48 bytes of constant context at the camera push (a per-push-site
   read analysis, deferred) and the 3-vectors as component arrays on the
   GPU (to decide with the camera kernel's number after 1).
3. The sample draws on disney-cloud (1.09x) and lte-orb-silver (1.06x),
   after 1, since the two kernels share the sampler code.
4. The standing debts: `cook`'s `budget = 16` replaced by a cost rule; the
   texture pyramid per declaration in scene_dump (landscape's .tex three
   times pbrt's); compare.sh and render.sh onto the shared conversion
   folder; the bistro and pavilion image verdicts (the lit-pixel rule,
   with the pbrt-GPU behavior audit of PLAN.md, "Design: the mesh is the
   entity", last paragraph); and the compiler debts the arms found
   (PLAN.md, the arms section; each a program with its expected output in
   tests/bonsai/pending, whose README says where each lives, all present
   at 29e15282): a queue's specialize clones a recursive callee handed
   the key without end (specialize_callees copying a copy's retargeted
   call of itself), and a spawned call's drained function spawning onward
   onto a second queue -- an internal error alone (the single-producer
   path of the deferral reading a drain's accumulate as the producer's
   rest) and lost sums beside a deferral with a rest (a queue fed by two
   drains of different queues sized as one drain's capacity). The third,
   the atomics demotion taking an `atomic` off inside a function a
   GPU-bound loop calls, is fixed (the pass asks the whole program which
   functions bound loops call; correctness/gpu/reduce-param). Two more
   for a ruling, found when the gate test's GPU twin was run with its
   accumulates written `atomic`: `spawn` has no `atomic` spelling, so a
   spawn into the run's reducer from a bound drain is a race the program
   cannot say, and a spawned call's entry holds the reducer's address,
   which on the JIT runtime's GPU path is a per-launch device copy's.
   The semantics behind both -- spawn as possible parallelism like
   parfor, a spawned function's reducer writes atomic, a check to build
   and the demotion that takes the atomic off where the spawn is not
   parallelised -- are docs/parallel-semantics.md, with the demotion's
   side in test (demote-atomics-spawn-* at every level) and the check's
   side as four pending programs with proposed diagnostics
   (tests/bonsai/pending/race-*); the spelling of an atomic spawn, and
   its cost to the renderer's four spawns into `radiance` unless the
   demotion learns a drain's slots are distinct within a pass, are the
   rulings asked there. (10-15) Both ruled and built: the spelling is
   `spawn f(args);` with `join;`, every accumulate into a reducer is
   atomic by declaration and the parser refuses the races the pending
   programs wrote (they are gone), and the demotion has the slot fact
   (PLAN.md, "`spawn f(args);` and `join;`"; "Where it stands
   (2026-10-15)"). The spawned entry's reducer address on the JIT
   runtime's GPU path stays open (docs/parallel-semantics.md, section
   6). New: the megakernel schedules abort on book with
   CUDA_ERROR_ILLEGAL_ADDRESS, under the last committed compiler too --
   last measured there on 10-02; book is the protocol's one
   coateddiffuse scene, and walk_step.loopify() is in those schedules
   alone (PLAN.md, "Found, not this work's"). (10-15, 16:15) Fixed:
   SSA/MirrorLoads.cpp placed a second mirror's entry update before the
   cell's first store, and the light tree's descent indexed the tree
   with an undefined value (PLAN.md, "Found and fixed"); both megakernel
   schedules render every protocol scene with images matching pbrt's
   and each other's. Open from the same compare: the megakernels run
   1.5x to 1.8x slower than on 10-02 while pbrt and gpu-optix got
   faster. Not measured; the driver's closest-hit cost of 10-07 is the
   first place to look.
5. After each fix, the compare on the fix's scenes plus head and
   killeroo-simple at 16 and 64 spp, the least of three on each side, one
   table per fix in this file; a full sweep only in a window agreed with
   the other sessions on the machine.
