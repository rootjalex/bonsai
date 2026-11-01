#!/usr/bin/env python3
"""The GPU schedule study as one figure, the CPU figure's twin: every
schedule's speedup over pbrt's GPU wavefront (`pbrt --gpu`), a group of
bars per scene, a subgroup per sample count, from the table gpu_compare.sh
writes.

    python3 apps/pbrt/plot_gpu_wavefront.py apps/pbrt/gpu-compare-out/table.tsv \
        [apps/pbrt/gpu-compare-out-sort/table.tsv ...] \
        -o apps/pbrt/plots/pbrt-gpu-sweep.pdf

(plots.sh draws it under that name from the three tables below, the
figure's one name since 10-24.)

A table is tab-separated, one row per measured cell and side (scene, spp,
side, wall_s, kernel_ms, speedup, agree, verdict, note, image, path;
gpu_compare.sh's header says what each is). The side `pbrt --gpu` is the
reference and draws no bar: the dashed line at 1 is it, labelled "PBRT wf"
since pbrt's GPU renderer is its wavefront (pbrt has no GPU megakernel, so
the GPU figure has no black bar). The bars are the wall-clock speedup, the
least of the run's repeats on each side, as the table's own speedup column
is; `--kernel` draws the kernel-time speedup instead (pbrt's from
`--stats`, ours from BONSAI_KERNEL_STATS), the device's busy time apart
from either side's launch gaps. Several tables are several runs joined
(a schedule added later and measured on its own, as gpu-optix-sort was on
10-23): every side's bar is its speedup over the pbrt that ran in the same
run, so a run's own reference never stands in for another's; a side in
more than one table takes the last table's row (a re-measurement
replaces what it re-measures), as the CPU figure's join does.

The sides, in the user's order (10-30: the matched wavefront first, its
modifications, then this renderer's megakernels, on both sweeps): pbrt's
wavefront on the RT cores (`wf`, gpu-optix, the matching schedule), that
wavefront with the ray queue sorted by the two-point Morton key before
each trace (`wf-sort`, gpu-optix-sort), the OptiX megakernel (`mk`,
gpu-optix-mega: a path per sample in one raygen program, tracing through
OptiX) and the same with shader execution reordering before the material
stage (`mk-ser`, gpu-optix-mega-ser). Every side traces through OptiX: the
user's rule for the GPU is to compare techniques on the ray tracing
hardware alone, the software traversals (gpu.bonsai, gpu-wavefront.bonsai)
having never won a cell. Since 10-27 the megakernel has three more shapes
in the figure (the user: the GPU figures other than the heatmap gain
them): one thread per path with a launch per sample (`Bonsai mk-pass`,
gpu-optix-mega-pass), the same as one launch per frame (`Bonsai mk-flat`,
gpu-optix-mega-flat), and that launch with the reordering (`Bonsai
mk-ser-flat`, gpu-optix-mega-ser-flat), and since the night of 10-27 the
flat launch in blocks of 2^18 paths from a host loop (`Bonsai mk-blk`,
gpu-optix-mega-blocked); plots.sh names these eight as the paper
figure's sides (GPU_SIDES). `--sides` names a subset in its own order.
The sides take the paper's one ramp (ColorBrewer YlGnBu, the heatmap's)
at eight equal perceptual steps from its class 1 to its class 8 in the
legend's order, as the CPU figure does (figure_style.SCHEDULE_COLOURS;
the user, 10-29): `mk` the palest, as the CPU's mk-b-s-ps is, `wf-sort`
the darkest, as its wf-t-v-ps-sort, every step down the legend a step
darker; the exploration's sides borrow shades, so a figure naming them
names a set without a repeat. All the measured schedules fit on one
figure that way, as the user asked on 10-24:

    python3 apps/pbrt/plot_gpu_wavefront.py apps/pbrt/gpu-compare-out/table.tsv \
        apps/pbrt/gpu-compare-out-sort/table.tsv \
        apps/pbrt/gpu-compare-out-explore/table.tsv \
        -o apps/pbrt/plots/pbrt-gpu-sweep.pdf

Everything else about the drawing is plot_cpu_wavefront.py's `draw`,
including the size: the paper's text width, the same as the CPU figure's,
so that the two stack as subfigures of one float at one scale.
"""
import argparse
import csv
import sys

from figure_style import SCHEDULE_COLOURS
from plot_cpu_wavefront import add_figure_arguments, choose_scenes, choose_sides, draw

REFERENCE = "pbrt --gpu"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup"

# (side in the table, label in the legend, colour): the drawing order (the
# user, 10-30 evening) -- the wavefront that matches pbrt's first, then its
# sorted variant, then the megakernels from the plain shape to the SER flat
# one; the CPU figure reads the same way. The colour is the schedule's on
# every figure (figure_style.SCHEDULE_COLOURS, keyed by the label).
SIDES = [(side, label, SCHEDULE_COLOURS[label]) for side, label in [
    ("gpu-optix", "wf"),              # pbrt's wavefront on the RT cores, matched
    ("gpu-optix-sort", "wf-sort"),    # its ray queue sorted before each trace
    # The wavefront's variants of 10-30 night (the user: five wavefronts
    # against three megakernels): the material stage inside the trace
    # program (gpu-optix-fused, the 10-24 exploration's), that with shader
    # execution reordering by material before the switch (gpu-optix-fused-
    # ser), and passes of four samples in place of one (gpu-optix-pass: a
    # quarter of pbrt's rounds of launches).
    ("gpu-optix-fused", "wf-fused"),
    ("gpu-optix-fused-ser", "wf-fused-ser"),
    ("gpu-optix-pass", "wf-pass"),
    ("gpu-optix-mega", "mk"),
    # The megakernel's shapes of 10-27 (PLAN, 'the megakernel against
    # Dr.Jit's'): one thread per path with a launch per sample
    # (`reorder(p, s)`), the same collapsed into one launch per frame
    # (`collapse(s, p, ps)`), and that launch with the SER sort before the
    # material stage. The user (10-27): the GPU figures other than the
    # heatmap gain them.
    ("gpu-optix-mega-pass", "mk-pass"),
    ("gpu-optix-mega-flat", "mk-flat-px"),
    # The flat launch in blocks of 2^18 paths, the host looping over the
    # blocks (the user, 10-27 evening); drawn when a table has it.
    ("gpu-optix-mega-blocked", "mk-blk"),
    ("gpu-optix-mega-ser", "mk-ser"),
    ("gpu-optix-mega-ser-flat", "mk-flat-ser-px"),
    # The exploration of 10-24 (apps/pbrt/schedules/gpu-optix-*.bonsai):
    # variants of the megakernel and the wavefront, drawn when a table has
    # them or `--sides` asks; more than nine sides exceed the ramp, which
    # is why plots.sh names the paper's sides (GPU_SIDES).
    ("gpu-optix-mega-ser-ray", "mk-ser-ray"),
    ("gpu-optix-b128", "wf-b128"),
    ("gpu-optix-b512", "wf-b512"),
    ("gpu-optix-sort36", "wf-sort36"),
    # The exploration of 11-01 (the user: "what sometimes-optimal schedules
    # can you find", from the directives the language has): the wavefront
    # with the escaped and emissive stages inside the trace program as well
    # as the material (fused-route), with the shadow ray traced there too
    # (fused-shadow), and reduced to the ray queue alone, one launch per
    # bounce (bounce); the flat megakernel with a pixel's samples adjacent in
    # the launch (flat-samples), the SER flat megakernel with a reordering by
    # ray before every trace (ser-ray-flat), and those two SER shapes in the
    # samples order.
    # The labels are the user's short ones (11-01): a flat megakernel is in
    # the samples order unless it says -px, and wf-fuse-mat is the material
    # stage in the trace program with the two one-line routings beside it.
    ("gpu-optix-fused-route", "wf-fuse-mat"),
    ("gpu-optix-fused-shadow", "wf-fused-shadow"),
    ("gpu-optix-bounce", "wf-bounce"),
    ("gpu-optix-mega-flat-samples", "mk-flat"),
    ("gpu-optix-mega-ser-ray-flat", "mk-flat-ser-ray-px"),
    ("gpu-optix-mega-ser-flat-samples", "mk-flat-ser"),
    ("gpu-optix-mega-ser-ray-flat-samples", "mk-flat-ser-ray"),
]]
# Pairs of sides differing by one decision, for the summary's reading of
# the figure (plot_cpu_wavefront.COMPARISONS explains).
COMPARISONS = [
    ("wf", "mk", "pbrt's wavefront schedule, matched, against the OptiX megakernel"),
    ("mk-pass", "mk", "one thread per path and a launch per sample instead of a pixel's samples in turn"),
    ("mk-flat-px", "mk-pass", "the sample and pixel loops collapsed into one launch per frame"),
    ("mk-blk", "mk-flat-px", "the frame's paths launched in blocks of 2^18 from a host loop instead of all at once"),
    ("mk-ser", "mk", "shader execution reordering before the material stage"),
    ("mk-flat-ser-px", "mk-flat-px", "shader execution reordering before the material stage, on the flat launch"),
    ("mk-ser-ray", "mk-ser", "reordering keyed by the ray instead of the material"),
    ("wf-b128", "wf", "bands of 128 pixel rows instead of the matching band size"),
    ("wf-b512", "wf", "bands of 512 pixel rows instead of the matching band size"),
    ("wf-fused", "wf", "the material stage inside the trace program, the hit queue gone"),
    ("wf-fused-ser", "wf-fused", "shader execution reordering by material before the fused material switch"),
    ("wf-pass", "wf", "passes of four samples instead of one: a quarter of the rounds of launches"),
    ("wf-sort", "wf", "the ray queue sorted by the 60-bit two-point Morton key before each trace"),
    ("wf-sort36", "wf-sort", "the coarse 36-bit sort key instead of the 60-bit one"),
    ("wf-fuse-mat", "wf", "the material stage inside the trace program (with the miss and emissive routing), the hit queue gone"),
    ("wf-fuse-mat", "wf-fused", "the miss and emissive routing inside the trace program as well as the material"),
    ("wf-fused-shadow", "wf-fused", "the shadow ray traced inside the trace program as well as the material"),
    ("wf-bounce", "wf", "one launch per bounce: every stage inside the trace program, the ray queue alone"),
    ("wf-bounce", "wf-fuse-mat", "every remaining stage inside the trace program too, the ray queue alone"),
    ("mk-flat", "mk-flat-px", "the flat launch with a pixel's samples adjacent (samples order) instead of a row's pixels"),
    ("mk-flat-ser", "mk-flat", "shader execution reordering by material before the material stage, on the samples-order flat launch"),
    ("mk-flat-ser", "mk-flat-ser-px", "the SER flat launch in the samples order instead of the pixel order"),
    ("mk-flat-ser-ray", "mk-flat-ser", "a reordering by ray before every trace, on the SER flat launch"),
    ("mk-flat-ser-ray-px", "mk-flat-ser-px", "a reordering by ray before every trace, on the pixel-order SER flat launch"),
    ("mk-flat-ser-ray", "mk-flat-ser-ray-px", "the ray-reordered SER flat launch in the samples order instead of the pixel order"),
]
# Why the schedules stand where they do, as measured (the user, 10-29;
# plot_cpu_wavefront.WHY and READINGS explain the form): the mechanisms,
# then each scene's reading under the computed best side of each cell.
WHY = [
    "what the megakernel pays: one program holding every material, shape, light and texture of",
    "  pbrt (330k SASS instructions against the wavefront's trace kernel's 15k), fetched by warps",
    "  whose lanes empty as their paths end and diverge -- 9 to 15 lanes of 32 after the first",
    "  bounce on the two profiled scenes, 31 for the trace kernel (pbrt-gpu-profile). What the",
    "  wavefront pays: a launch per queue per bounce and the queues' traffic (the ray read, the",
    "  hit record written: about 900 bytes a ray through the trace kernel), which pbrt pays too.",
    "  The balance is the scene's: a cheap trace with few materials favours the megakernel, a",
    "  heavy trace or a program whose lanes diverge the wavefront.",
    "the megakernel's shapes: a pixel's samples in turn (mk) cost 1.7-1.9x against one thread per",
    "  path (mk-pass); one launch per frame (mk-flat-px) saves 2-6% more; blocks of 2^18 paths from a",
    "  host loop (mk-blk) cost 0.83x of mk-flat-px, each launch draining its block's longest path with",
    "  the card part empty, 30 to 1013 times a frame; shader execution reordering before the",
    "  material stage (mk-flat-ser-px) buys 1.14x on ganesha to 1.46x on killeroo-gold over mk-flat-px",
    "  by kernel time, by refilling the warps. The exploration of 11-01 (gpu-compare-out-explore3 and",
    "  -explore4) added two decisions. The launch's order: with a pixel's samples adjacent in the",
    "  launch (the samples order, the labels without -px) a warp of 32 is one pixel's samples at 128",
    "  spp and two pixels' at 16, whose camera rays leave through one footprint and hit one surface,",
    "  where the flat launch of 10-27 had a row's pixels adjacent (-px). It is ahead at every cell but",
    "  frame542 at 128 spp: by kernel time 4-80% on the flat launch (frame542 the 80%, book the 4%),",
    "  2-37% on the SER one and 2-26% on the ray-reordered one; at frame542's 128 spp the pixel order",
    "  is ahead by 4-8%. The reordering by ray before every trace (mk-flat-ser-ray: optixReorder on",
    "  the next ray's origin and direction cells, the top sixteen bits of the coarse Morton key, on",
    "  top of the reordering by material; the idea of mk-ser-ray on 10-24): over mk-flat-ser by",
    "  kernel time 1.2-1.4x on book, sanmiguel and bistro, 1.4x on ganesha, 1.6x on dragon_10, 1.9x",
    "  on plume -- the heavier the trace, the more a warp of rays walking one part of the tree is",
    "  worth -- and 0.92-0.93x on killeroo-gold, where the conductor leaves little for a second sort",
    "  to regroup (a reading). It is the best megakernel on seven of eight scenes and takes book and",
    "  sanmiguel from the wavefronts; on bistro the plain samples-order flat launch is the best",
    "  megakernel, SER costing 0.72x of it there as it did on the pixel order.",
    "the figure since 11-01 draws seven sides (the user: every schedule that is the best on at least",
    "  one cell, the matched one, the rest filling out the two families, wf-sort out): the matched",
    "  wavefront is best on ganesha, dragon_10 and plume, wf-fuse-mat on bistro, mk-flat-ser on",
    "  killeroo-gold (4.06x, 3.13x), mk-flat-ser-ray on book (3.28x, 2.59x), sanmiguel (2.62x,",
    "  1.97x) and frame542 at 16 spp (20.1x), its pixel-order twin mk-flat-ser-ray-px on frame542 at",
    "  128 spp (21.2x); wf-bounce and mk-flat fill. The incumbents re-measured beside the new sides",
    "  (gpu-optix, gpu-optix-mega-flat and gpu-optix-mega-ser-flat in explore4) reproduced their",
    "  10-30 cells within 2% by wall and 1% by kernel time, so the earlier tables stand.",
    "the ray sort (wf-sort) never pays on the GPU: 9-15% over the unsorted wavefront by kernel",
    "  time and 20-31% by wall -- a dozen small launches before every trace (96 sorts in a 30 ms",
    "  frame at 16 spp) for coherence the traversal hardware's own scheduling already had.",
    "with more samples every speedup falls, to 0.75-0.80x of its 16-spp value at 128 spp over the",
    "  first six scenes (0.82-0.86x over the eight of 10-29, whose three large scenes fell least;",
    "  about 0.7x on the five cheap scenes of 10-28 alone): pbrt's per-frame costs (launch setup,",
    "  queue resets, camera rays) are a smaller share of a longer render. frame542's hold",
    "  (0.98-1.11x): its frame is rounds of launches at both counts.",
    "over the first six scenes (10-30), the wavefront is best on five and the SER flat megakernel",
    "  on killeroo-gold alone; where the trace is heavy (ganesha, dragon_10, bistro) every",
    "  megakernel is below pbrt's wavefront at both counts, and SER's gain on the flat launch runs",
    "  from a loss (bistro, 0.76x) to 1.47x (killeroo-gold at 128 spp) -- the figure's spread.",
    "  frame542, the seventh (10-30 evening), is the other extreme: depth 256 with short paths, so",
    "  the wavefronts pay launches for bounces almost nobody reaches and every megakernel is far",
    "  ahead (the SER flat 18-20x). killeroo-coated-gold (gpu-compare-out-coated, not drawn) is",
    "  killeroo-gold with CoatedConductor for the Conductor: every megakernel 2.0-2.7x slower, the",
    "  wavefront 1.2x, so the megakernel's one win on the six is the conductor's closed-form BSDF.",
    "  The eight of 10-29 (gpu-compare-out-eight: the six less dragon_10, plus landscape, kroken,",
    "  watercolor) had SER doubling the flat megakernel on kroken and watercolor; those cells",
    "  stand as measurements and are not drawn.",
    "the wavefront's variants (10-31, under the compiler at b72c215b): the material stage inside",
    "  the trace program (wf-fused) is 0.85-0.97x of the matched wavefront where the trace is cheap",
    "  (killeroo-gold, book, ganesha, dragon_10, plume: the raygen's continuation, which runs once",
    "  the warp's traversals are back, grows by a material's evaluation) and 1.04-1.13x where it is",
    "  heavy or deep (sanmiguel, bistro, frame542: the hit queue's round trip was the larger cost),",
    "  so it is the best side on sanmiguel and bistro at both counts. Reordering by material before",
    "  the fused material switch (wf-fused-ser) is 0.76-1.09x of wf-fused: a gain on book and",
    "  dragon_10 (3-9%), a wash on sanmiguel and plume, a loss on ganesha, bistro (0.85x) and",
    "  frame542 (0.81x) -- one bounce's work after the reorder (a trace, one material, a push) is too",
    "  little to repay a sort of the launch's rays every bounce, where the SER megakernel sorts once",
    "  for a whole path's divergence (a reading from the walls; not profiled). The pass-of-four",
    "  wavefront (wf-pass: a band of pixels times four samples per round of launches, a quarter of",
    "  pbrt's rounds) is 0.09-0.79x of pbrt --gpu at 16 spp and 0.45-2.62x at 128, below every",
    "  other wavefront on every scene but frame542 at 128 spp: its queues do not fit the card. Every",
    "  queue array is allocated at the band's capacity -- 1,950 arrays, 15.6 KB an entry across",
    "  them -- and four samples a pixel makes 4.1M entries on book (1920x1080): 60 GB of managed",
    "  memory on a 32 GB card, where the matched wavefront's 14 GB fit, so the pages migrate as the",
    "  kernels touch them. The wall time exceeds the kernels' own by 0.6-2.4 s a render, the same at",
    "  16 and 128 spp (a cost paid once a render, not per sample), and a single trace launch takes",
    "  57-126 ms where the matched wavefront's longest is 0.4 ms (the renderers' kernel statistics).",
    "  pbrt's cap of a million paths a pass is the memory bound this schedule crossed; a quarter of",
    "  the rounds is bought with four times the queue memory, and only frame542 at 128 spp, where",
    "  256 bounces of rounds are the whole cost, comes out ahead (2.62x, the best wavefront there).",
    "the wavefront's variants of 11-01 (explore3): the material stage in the trace program with the",
    "  routing of a miss and of an emissive hit beside it (wf-fuse-mat: the two one-line stages pbrt",
    "  gives kernels of their own, HandleEscapedRays and HandleEmissiveIntersection) is 1.00-1.04x of",
    "  wf-fused by kernel time on every scene and 1.14x on frame542, so it is the best wavefront",
    "  where wf-fused was, bistro at both counts (2.44x, 1.94x), and draws in its place. The shadow",
    "  ray traced in the trace program too (wf-fused-shadow, not drawn) is 0.64-0.88x of wf-fused by",
    "  kernel time everywhere but frame542 (1.27x) -- the trace program's lanes wait on a second",
    "  traversal before they push the next ray -- and fails with an unspecified launch failure on",
    "  plume, the media scene (open, PLAN 11-01). One launch per bounce (wf-bounce: the ray queue",
    "  alone, every stage in the trace program, Laine et al.'s one-logic-kernel shape without their",
    "  path regeneration) is the best wavefront on frame542 (2.65x, 2.57x against the matched 1.66x,",
    "  1.59x: a fifth of the launches a round) and level with the matched one on killeroo-gold at 16",
    "  spp; elsewhere it is below wf-fuse-mat, worst where the fused program's size cost most",
    "  (ganesha 0.69x, dragon_10 0.67x of wf at 128 spp): it carries every stage's code, as the",
    "  megakernel does, without the megakernel's one launch. wf-pass is out of the figure: wf-bounce",
    "  matches it on frame542 at 128 spp (2.57x against 2.62x) without the paging and is ahead",
    "  everywhere else.",
]
READINGS = {
    "killeroo-simple": [
        "66k triangles, two materials (CoatedDiffuse, Diffuse), the cheapest trace of the five: the",
        "wavefront's launches and queue traffic are the larger share of a short frame, and the SER",
        "flat megakernel pays neither and runs full warps (on killeroo-gold's profile the reordering",
        "lifts the frame's lanes from 14.7 to 23.3 of 32 and cuts the warp instructions from 28.5 G",
        "to 18.6 G), so it wins every cell; at 16 spp pbrt's per-frame costs are the largest share of",
        "its 0.11 s, hence the 4.65x.",
    ],
    "killeroo-gold": [
        "532k triangles, Conductor and Diffuse; the profile figure's second row: the flat megakernel's",
        "frame is 341k SASS instructions at 14.7 lanes of 32 with 18% of its warp-cycles on the",
        "instruction fetch, 101 ms; with SER 23.3 lanes and 65 ms; the wavefront's trace kernel 15k",
        "instructions at 31.6 lanes, 64% of its cycles waiting on memory (the queues) and 23% on the",
        "traversal -- so the SER flat megakernel is best at every count and the plain flat one is",
        "within 20% of the wavefront (10-30, the pixel-order launches). Measured on 11-01, the SER flat",
        "megakernel in the samples order (mk-flat-ser) is best at both counts, 4.06x and 3.13x (kernel",
        "54 and 422 ms against the pixel order's 59 and 470), and the reordering by ray costs it 7-8%",
        "here, the one scene where it does: the conductor's closed-form BSDF leaves the warps little",
        "divergence for a second sort to regroup (a reading, not measured).",
    ],
    "book": [
        "21k triangles, CoatedDiffuse and Diffuse over textured pages: the material kernels take as",
        "long as the trace (10-24's profiles), and the fused raygen of 10-24 cost book the most of",
        "the five (29-31% over trace + hits), a bigger program lowering the trace's occupancy where",
        "the trace is cheap -- the wavefront wins every cell, the flat megakernel within 7% at 16 spp",
        "(10-30). SER bought nothing here on the pixel-order launch (mk-flat-ser-px below mk-flat-px at",
        "every count), the one scene where it did not; why is not measured (no megakernel profile of",
        "book). Measured on 11-01, the samples-order SER flat megakernel with the reordering by ray",
        "(mk-flat-ser-ray) takes both cells, 3.28x and 2.59x against the wavefront's 2.44x and 1.88x:",
        "in the samples order SER by material is worth 1.03-1.22x by kernel time after all, and the",
        "reordering by ray another 1.22-1.34x -- the textured pages' CoatedDiffuse and the trace both",
        "regrouped (a reading from the kernel times; not profiled).",
    ],
    "ganesha": [
        "one statue of 4.3M triangles, CoatedDiffuse and Diffuse; the profile figure's first row says",
        "where the megakernel's time goes: the frame's raygen runs at 9.2 lanes of 32 (6.0 in the",
        "continuations after the first trace) with 44% of its warp-cycles fetching instructions over a",
        "330k-instruction program, while the wavefront's trace kernel is 15k instructions at 31.2",
        "lanes, its cycles on memory (53%) and the traversal wait (39%). SER lifts the lanes to 15.0",
        "and removes the traversal wait, but the fetch grows to 84% of the cycles (the sorted warps run",
        "different arms of the one raygen; the counters say the fetch grew, the why is a reading). The",
        "megakernel's loss here is the raygen's size and the empty lanes, not the traversal, which",
        "both sides do on the RT cores (10-28) -- every megakernel is below pbrt, the wavefront 2x.",
        "On 11-01 the wavefront keeps both cells (2.08x, 1.85x); the nearest megakernel is the",
        "samples-order SER one with the reordering by ray (mk-flat-ser-ray, 1.25x and 1.20x), the ray",
        "reorder buying it 1.4x by kernel time over SER alone and leaving it a third short.",
    ],
    "sanmiguel": [
        "2.5M triangles in 796 meshes, instanced (a two-level traversal), four material kinds",
        "(CoatedDiffuse, Dielectric, Diffuse, DiffuseTransmission -- the most of 10-28's five) and the",
        "heaviest trace per sample (0.22 ms a trace launch at 64 spp against 0.15 on book): both sides",
        "spend most of the frame in the same RT-core traversal, so what the schedule saves around it",
        "is the smallest share here and the wavefront's margin the smallest of the six (landscape's,",
        "measured on 10-29 and no longer drawn, was smaller), falling fastest with spp. The fused",
        "wavefront of 10-24 cost nothing on this scene where it cost the others a fifth to a third --",
        "the reading being that the trace is bound by the traversal's latency, not by occupancy, so a",
        "bigger program rides along -- which is also why the SER flat megakernel is nearest the",
        "wavefront here (1.81x against 2.03x at 16 spp). pbrt's own 16-spp wall is not steady between",
        "runs: 0.24 s on 10-28, 0.19 s on 10-29, 0.24 s on 10-30 (the least of three processes each",
        "time) while ours stayed at 0.118 s, so this cell's speedups read 2.03x, 1.61x and 2.03x.",
        "Measured on 10-31, the fused wavefront is the best wavefront here at both counts (2.19x, 1.54x:",
        "1.05-1.08x of the matched wavefront) -- the hit queue's round trip saved on a trace this",
        "heavy outweighs the longer raygen continuation; reordering by material adds nothing (0.98x).",
        "Measured on 11-01, the samples-order SER flat megakernel with the reordering by ray",
        "(mk-flat-ser-ray) is the best side at both counts, 2.62x and 1.97x against wf-fuse-mat's 2.19x",
        "and 1.55x: the two-level trace that bounded the wavefront's margin is what a warp of rays",
        "walking one part of the tree speeds up, the reordering by ray 1.37-1.39x over mk-flat-ser by",
        "kernel time (a reading from the kernel times; not profiled).",
    ],
    # The scenes added on 10-29 (the user: eight scenes at 16 and 128 spp)
    # and on 10-30 (dragon_10 in landscape's place; landscape, kroken and
    # watercolor out of the sweeps since 10-30, their readings kept for
    # gpu-compare-out-eight's table). What the table shows; the causes are
    # not yet measured (no profile of these scenes), and are said so.
    "dragon_10": [
        "7.2M triangles in two meshes, three materials (Subsurface on the dragon, CoatedDiffuse and",
        "Diffuse), one infinite light, no textures, depth 5 and 2.76 rays a camera ray: ganesha's shape",
        "at a larger scale -- the wavefront wins every cell (1.93x, 1.58x; the heatmap's 1.91x and 1.59x)",
        "with the sort behind it (1.56x, 1.28x), every megakernel below pbrt at both counts, the SER flat",
        "one nearest at 0.90x and 0.71x (SER buying 1.28x over the flat launch at both counts), and the",
        "per-sample megakernel 0.36x and 0.29x, the lowest of the six for that side. Not profiled; a",
        "subsurface hit adds pbrt's subsurface sample to the bounce -- a probe ray traced through the",
        "dragon and a second material evaluation where it lands -- so the lane's program is longer and",
        "more divergent than ganesha's, and the megakernel's loss is read as its empty lanes and its",
        "size, as on ganesha, not the traversal; a reading, not a measurement. On 11-01 the wavefront",
        "keeps both cells (1.90x, 1.59x); the nearest megakernel is mk-flat-ser-ray (1.48x, 1.23x), the",
        "reordering by ray buying it 1.6x by kernel time over SER alone.",
    ],
    "bistro": [
        "2.9M triangles in 1,693 meshes, 133 materials, 20,638 area lights (the light BVH's scene): the",
        "wavefront wins every cell (2.16x, 1.76x) by 2.5x over the best megakernel, and this is the one",
        "scene where shader execution reordering costs -- mk-flat-ser-px 0.67x against mk-flat-px's 0.85x",
        "at 16 spp (0.52x against 0.66x at 128), and the per-sample SER megakernel 0.33x and 0.25x, the worst",
        "cells of the sweep. Why SER loses here is not measured (no profile of bistro); the suspicion is",
        "the light sampling's share of the frame -- a sort by material does nothing for a stage that",
        "walks a light BVH of twenty thousand emitters -- and it is a reading, not a measurement.",
        "Measured on 10-31, the fused wavefront is the best side here at both counts (2.42x, 1.92x:",
        "1.09-1.13x of the matched wavefront, its largest gain of the eight), and reordering by",
        "material costs it 15% (2.06x, 1.64x) -- the same stage that SER cannot help in the megakernel.",
        "On 11-01 wf-fuse-mat, the fused wavefront with the miss and emissive routing beside the",
        "material, is the best side (2.44x, 1.94x, 1% over wf-fused); every megakernel stays below pbrt,",
        "the samples-order flat launch nearest (0.93x, 0.75x) and SER 0.72x of it, as on the pixel order.",
    ],
    # The seventh scene (the user, 10-30 evening), measured after the CPU
    # sweep into gpu-compare-out-two with the six's renderers.
    "frame542": [
        "1.3M triangles in 229 meshes (5 instances), Dielectric and Diffuse, one light, depth 256,",
        "2.81 rays a camera ray, 1920x840: the paths are short but the depth is deep, and a wavefront",
        "runs its round of launches for every bounce up to the depth while any path of the pass lives,",
        "so both wavefronts pay rounds of nearly empty launches -- pbrt's kernel time is 47 ns a camera",
        "path here against 6.6 on killeroo-gold, ours 27 against 3.0 -- while a megakernel's thread",
        "leaves when its path dies (the SER flat one 2.8 ns a path against 2.6). So every megakernel",
        "is far ahead: the SER flat 18.3x and 20.4x, the largest speedup of any scene, the SER 8.1x and",
        "8.7x, the per-sample 6.2x and 6.5x, the flat 4.7x and 4.6x; the wavefront 1.66x and 1.62x (the",
        "heatmap's 1.66x and 1.59x) and the sort below pbrt (0.92x, 0.90x: a sort a round on those",
        "near-empty queues). The reading is not verified by a launch count (nsys would give it).",
        "Measured on 11-01, the pass-of-four wavefront is the best wavefront here at 128 spp (2.62x",
        "against the matched 1.62x and the fused 1.68x) and 0.79x at 16: the only scene where a quarter",
        "of the rounds repays the paging of its four-times-larger queues (WHY, the pass paragraph);",
        "the one-launch-per-bounce wavefront then matched it there (2.57x) and beat it at 16 spp (2.65x",
        "against 0.79x), and is the best wavefront drawn. Among the megakernels the samples-order SER",
        "ones lead at 16 spp (mk-flat-ser-ray 20.1x, mk-flat-ser 19.0x) and the pixel-order",
        "ray-reordered one at 128 spp (mk-flat-ser-ray-px 21.2x against 19.2x), the one cell of the",
        "sweep where the pixel order is ahead (by 4-8% of kernel time; why is not measured).",
    ],
    # The eighth scene (the user, 10-30 evening: a participating medium),
    # measured 22:09-22:17 into gpu-compare-out-plume with the six's renderers.
    "plume": [
        "a smoke plume: a uniformgrid medium in a scaled sphere with an interface boundary, a",
        "Conductor sphere and a Diffuse floor (two triangles), one light, depth 5, 5.6 rays a camera",
        "ray, 1200x1200 -- the one participating medium of the sweep, so the only scene whose paths",
        "run volpath's medium stage (the delta-tracking walk through the grid) rather than a surface",
        "shading alone. The wavefront is best (2.15x, 1.81x; the heatmap's 2.16x, 1.81x), the sort",
        "behind it (1.76x, 1.47x), and every megakernel below pbrt at both counts, 0.28-0.71x at 16",
        "spp and 0.20-0.57x at 128: the medium walk is a loop of per-ray variable length inside the",
        "bounce loop, so a warp's lanes diverge on it, and SER has no surface hit to regroup by before",
        "it -- the SER per-sample megakernel is the worst cell (0.20x) and the SER flat one gains",
        "nothing over the flat (0.57x against 0.56x). A reading from the walls; not profiled. On 11-01",
        "the wavefront keeps both cells (2.18x, 1.81x); the reordering by ray is worth the most here of",
        "any scene to the megakernel (1.9x by kernel time over SER alone: the rays leaving the medium",
        "walk regrouped before their next trace, a reading) and mk-flat-ser-ray still reads 1.41x and",
        "1.24x; the fused wavefronts are 0.95-0.97x of the matched one.",
    ],
    "landscape": [
        "27.9M triangles, 30,034 instances of 32 objects (a two-level traversal through the deepest",
        "instancing of the set), 638 materials, 1,611 textures, 3840x2160: the flattest scene -- the",
        "wavefront 1.30x and 1.20x, the heatmap's lowest, the SER flat megakernel within 5% of it (1.25x,",
        "1.14x) and the flat megakernels at parity (1.02x, 0.92x), only the per-sample forms well below.",
        "Both sides spend the frame in the same instanced RT-core trace, so the schedule has the least to",
        "save (sanmiguel's reading at a larger scale); not profiled.",
    ],
    "kroken": [
        "288M triangles (248M from displacement, 10 instances of 10 objects), 80 materials, 2,175 area",
        "lights, depth 15 and 6.75 rays a camera ray, the deepest paths of the set: the wavefront wins",
        "every cell (2.30x, 1.94x) with the sort behind it; SER doubles the flat megakernel (0.64x ->",
        "1.33x at 16 spp, 0.53x -> 1.10x at 128), the largest SER gain of the eight with watercolor's,",
        "and still leaves it 1.7x behind the wavefront. Not profiled; the long paths (a lane alive to",
        "bounce 15 holds its warp) are the reading for the megakernels' loss.",
    ],
    "watercolor": [
        "58M triangles (34M from displacement), 944 meshes, 261 materials, 412 textures, depth 15, 4.1",
        "rays a camera ray: the highest speedups of the sweep -- the wavefront 2.83x and 2.59x (the",
        "heatmap's highest), the SER flat megakernel 2.61x and 2.38x, within 9% of it, SER doubling the",
        "flat launch (1.31x -> 2.61x, 1.19x -> 2.38x), the per-sample megakernels 0.73-1.35x. Why pbrt's",
        "wavefront is slowest here relative to ours is not measured; the 1.86 GiB of image maps and the",
        "261 materials make the material stage the suspect.",
    ],
}
CAUSES = [
    "causes measured, apps/pbrt/PLAN.md: 'The GPU matrix measured: five scenes, three sample",
    "  counts, four OptiX sides (10-24)' -- the reading of the four paper sides, the sort's own",
    "  time against what coherence saves on the RT cores; 'The GPU schedule space explored: is",
    "  the best schedule data-dependent? (10-24)' -- the band sizes, the fused wavefront and the",
    "  reorder keys; 'The fused wavefront's trade read off the profiles (10-24)' -- fusing costs",
    "  a fifth to a third where the trace is cheap and nothing on sanmiguel, whose two-level",
    "  trace is latency-bound. REPORT.md 10-23..10-24 has each run's numbers. 'The megakernel",
    "  against Dr.Jit's (10-27)' -- a pixel's samples in turn cost 1.74x against one thread per",
    "  path, one launch per frame 2-6% more, the SER sort 1.14-1.46x on the flat launch; '10-28,",
    "  the night' -- the blocked megakernel's tail; 'The profile figure (10-28)' -- the lanes and",
    "  the fetch on ganesha and killeroo-gold (pbrt-gpu-profile.txt has the numbers).",
]


def load(paths, kernel=False):
    """gpu_compare.sh's tables as rows with `scene`, `key` (the scene's
    `path` under the scene set, else its name: kroken's and watercolor's
    scenes are both camera-1), `spp` (int), `side`, `seconds` (float: the
    wall time, or the kernel time in seconds with `kernel`), `verdict` and
    `reference` (the seconds of the `pbrt --gpu` row of the same cell in
    the same table); a row without a time draws nothing, and a (scene,
    spp, side) a later table repeats replaces the earlier row."""
    joined = {}
    order = []
    for path in paths:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        table = []
        for r in rows:
            try:
                r["spp"] = int(r["spp"])
                r["seconds"] = (float(r["kernel_ms"]) / 1000.0 if kernel
                                else float(r["wall_s"]))
            except (ValueError, KeyError):
                continue
            r["key"] = r.get("path") or r["scene"]
            table.append(r)
        references = {}
        for r in table:
            if r["side"] == REFERENCE:
                key = (r["key"], r["spp"])
                references[key] = min(r["seconds"], references.get(key, r["seconds"]))
        for r in table:
            key = (r["key"], r["spp"], r["side"])
            r["reference"] = references.get((r["key"], r["spp"]))
            if key not in joined:
                order.append(key)
            joined[key] = r
    return [joined[key] for key in order]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+")
    add_figure_arguments(ap)
    ap.add_argument("--kernel", action="store_true",
                    help="the kernel-time speedup instead of the wall-clock one")
    args = ap.parse_args(argv[1:])
    if args.out is None:
        args.out = "pbrt-gpu-sweep.pdf"

    rows = choose_scenes(load(args.tables, kernel=args.kernel), args.scenes)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args, COMPARISONS, CAUSES,
         why=WHY, readings=READINGS, sources=args.tables)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
