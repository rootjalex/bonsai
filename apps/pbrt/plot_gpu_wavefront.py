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

The sides, in the user's order (10-23): the OptiX megakernel (`Bonsai mk`,
gpu-optix-mega: a path per sample in one raygen program, tracing through
OptiX), the same with shader execution reordering before the material
stage (`Bonsai mk-ser`, gpu-optix-mega-ser), pbrt's wavefront on the RT
cores (`Bonsai wf`, gpu-optix, the matching schedule) and that wavefront
with the ray queue sorted by the two-point Morton key before each trace
(`Bonsai wf-sort`, gpu-optix-sort). Every side traces through OptiX: the
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
from plot_cpu_wavefront import add_figure_arguments, choose_sides, draw

REFERENCE = "pbrt --gpu"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup"

# (side in the table, label in the legend, colour): the drawing order, the
# megakernels first and then the wavefronts, each family from its plain
# schedule to its variants. The colour is the schedule's on every figure
# (figure_style.SCHEDULE_COLOURS, keyed by the label).
SIDES = [(side, label, SCHEDULE_COLOURS[label]) for side, label in [
    ("gpu-optix-mega", "mk"),
    # The megakernel's shapes of 10-27 (PLAN, 'the megakernel against
    # Dr.Jit's'): one thread per path with a launch per sample
    # (`reorder(p, s)`), the same collapsed into one launch per frame
    # (`collapse(s, p, ps)`), and that launch with the SER sort before the
    # material stage. The user (10-27): the GPU figures other than the
    # heatmap gain them.
    ("gpu-optix-mega-pass", "mk-pass"),
    ("gpu-optix-mega-flat", "mk-flat"),
    # The flat launch in blocks of 2^18 paths, the host looping over the
    # blocks (the user, 10-27 evening); drawn when a table has it.
    ("gpu-optix-mega-blocked", "mk-blk"),
    ("gpu-optix-mega-ser", "mk-ser"),
    ("gpu-optix-mega-ser-flat", "mk-ser-flat"),
    # The exploration of 10-24 (apps/pbrt/schedules/gpu-optix-*.bonsai):
    # variants of the megakernel and the wavefront, drawn when a table has
    # them or `--sides` asks; more than nine sides exceed the ramp, which
    # is why plots.sh names the paper's sides (GPU_SIDES).
    ("gpu-optix-mega-ser-ray", "mk-ser-ray"),
    ("gpu-optix", "wf"),
    ("gpu-optix-b128", "wf-b128"),
    ("gpu-optix-b512", "wf-b512"),
    ("gpu-optix-fused", "wf-fused"),
    ("gpu-optix-fused-ser", "wf-fused-ser"),
    ("gpu-optix-sort", "wf-sort"),
    ("gpu-optix-sort36", "wf-sort36"),
]]
# Pairs of sides differing by one decision, for the summary's reading of
# the figure (plot_cpu_wavefront.COMPARISONS explains).
COMPARISONS = [
    ("wf", "mk", "pbrt's wavefront schedule, matched, against the OptiX megakernel"),
    ("mk-pass", "mk", "one thread per path and a launch per sample instead of a pixel's samples in turn"),
    ("mk-flat", "mk-pass", "the sample and pixel loops collapsed into one launch per frame"),
    ("mk-blk", "mk-flat", "the frame's paths launched in blocks of 2^18 from a host loop instead of all at once"),
    ("mk-ser", "mk", "shader execution reordering before the material stage"),
    ("mk-ser-flat", "mk-flat", "shader execution reordering before the material stage, on the flat launch"),
    ("mk-ser-ray", "mk-ser", "reordering keyed by the ray instead of the material"),
    ("wf-b128", "wf", "bands of 128 pixel rows instead of the matching band size"),
    ("wf-b512", "wf", "bands of 512 pixel rows instead of the matching band size"),
    ("wf-fused", "wf", "the camera-ray generation fused into the first trace"),
    ("wf-sort", "wf", "the ray queue sorted by the 60-bit two-point Morton key before each trace"),
    ("wf-sort36", "wf-sort", "the coarse 36-bit sort key instead of the 60-bit one"),
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
    "  path (mk-pass); one launch per frame (mk-flat) saves 2-6% more; blocks of 2^18 paths from a",
    "  host loop (mk-blk) cost 0.83x of mk-flat, each launch draining its block's longest path with",
    "  the card part empty, 30 to 1013 times a frame; shader execution reordering before the",
    "  material stage (mk-ser-flat) buys 1.14x on ganesha to 1.46x on killeroo-gold over mk-flat",
    "  by kernel time, by refilling the warps.",
    "the ray sort (wf-sort) never pays on the GPU: 9-15% over the unsorted wavefront by kernel",
    "  time and 20-31% by wall -- a dozen small launches before every trace (96 sorts in a 30 ms",
    "  frame at 16 spp) for coherence the traversal hardware's own scheduling already had.",
    "with more samples every speedup falls, to 0.82-0.86x of its 16-spp value at 128 spp over the",
    "  eight scenes (about 0.7x on the five cheap scenes of 10-28 alone): pbrt's per-frame costs",
    "  (launch setup, queue resets, camera rays) are a smaller share of a longer render.",
    "over the eight scenes (10-29), the wavefront is best on seven and the SER flat megakernel on",
    "  killeroo-gold alone; where the trace is heavy (ganesha, bistro, kroken) every megakernel",
    "  is below pbrt's wavefront at 128 spp, and SER's gain on the flat launch runs from a loss",
    "  (bistro, 0.79x) to a doubling (kroken 2.1x, watercolor 2.0x) -- the figure's spread.",
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
        "within 20% of the wavefront.",
    ],
    "book": [
        "21k triangles, CoatedDiffuse and Diffuse over textured pages: the material kernels take as",
        "long as the trace (10-24's profiles), and the fused raygen of 10-24 cost book the most of",
        "the five (29-31% over trace + hits), a bigger program lowering the trace's occupancy where",
        "the trace is cheap -- the wavefront wins every cell, the flat megakernel within 7% at 16 spp.",
        "SER buys nothing here (mk-ser-flat below mk-flat at every count), the one scene where it does",
        "not; why is not measured (no megakernel profile of book).",
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
    ],
    "sanmiguel": [
        "2.5M triangles in 796 meshes, instanced (a two-level traversal), four material kinds",
        "(CoatedDiffuse, Dielectric, Diffuse, DiffuseTransmission -- the most of the five) and the",
        "heaviest trace per sample (0.22 ms a trace launch at 64 spp against 0.15 on book): both sides",
        "spend most of the frame in the same RT-core traversal, so what the schedule saves around it",
        "is the smallest share here and the wavefront's margin the smallest of the five of 10-28",
        "(landscape's is smaller), falling fastest with spp. The fused wavefront of 10-24 cost nothing",
        "on this scene where it cost the others a fifth to a third -- the reading being that the trace",
        "is bound by the traversal's latency, not by occupancy, so a bigger program rides along --",
        "which is also why the SER flat megakernel is nearest the wavefront here (1.49x against 1.61x",
        "at 16 spp on 10-29; pbrt's own 16-spp wall moved from 0.24 s to 0.19 s between the two runs",
        "while ours did not, so the 16-spp speedups read 2.03x on 10-28 and 1.61x now).",
    ],
    # The four scenes added on 10-29 (the user: eight scenes at 16 and 128
    # spp). What the table shows; the causes are not yet measured (no
    # profile of these scenes), and are said so.
    "bistro": [
        "2.9M triangles in 1,693 meshes, 133 materials, 20,638 area lights (the light BVH's scene): the",
        "wavefront wins every cell (2.16x, 1.76x) by 2.5x over the best megakernel, and this is the one",
        "scene where shader execution reordering costs -- mk-ser-flat 0.67x against mk-flat's 0.85x at 16",
        "spp (0.52x against 0.66x at 128), and the per-sample SER megakernel 0.33x and 0.25x, the worst",
        "cells of the sweep. Why SER loses here is not measured (no profile of bistro); the suspicion is",
        "the light sampling's share of the frame -- a sort by material does nothing for a stage that",
        "walks a light BVH of twenty thousand emitters -- and it is a reading, not a measurement.",
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

    rows = load(args.tables, kernel=args.kernel)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args, COMPARISONS, CAUSES,
         why=WHY, readings=READINGS, sources=args.tables)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
