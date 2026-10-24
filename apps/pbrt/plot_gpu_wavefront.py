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
having never won a cell. `--sides` names a subset in its own order. The
colours are classes of the CPU figure's YlGnBu ramp spread evenly over the
sides drawn, in drawing order, so that neighbours differ in lightness: four
sides take the classes 1, 3, 5 and 7 (the paper figure), up to seven sides
stay within 1..7, and eight or nine sides use the whole ramp, the palest
class told from the page by the bar's edge. All nine measured schedules
(the four above and the exploration's five) fit on one figure that way,
as the user asked on 10-24:

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

from figure_style import YLGNBU
from plot_cpu_wavefront import add_figure_arguments, choose_sides, draw

REFERENCE = "pbrt --gpu"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup"

# (side in the table, label in the legend): the drawing order, the
# megakernels first and then the wavefronts, each family from its plain
# schedule to its variants. The colour is assigned by `colour_sides` from the
# number of sides drawn.
SIDES = [
    ("gpu-optix-mega", "mk"),
    ("gpu-optix-mega-ser", "mk-ser"),
    # The exploration of 10-24 (apps/pbrt/schedules/gpu-optix-*.bonsai):
    # variants of the megakernel and the wavefront, drawn when a table has
    # them or `--sides` asks; more than nine sides exceed the ramp.
    ("gpu-optix-mega-ser-ray", "mk-ser-ray"),
    ("gpu-optix", "wf"),
    ("gpu-optix-b128", "wf-b128"),
    ("gpu-optix-b512", "wf-b512"),
    ("gpu-optix-fused", "wf-fused"),
    ("gpu-optix-fused-ser", "wf-fused-ser"),
    ("gpu-optix-sort", "wf-sort"),
    ("gpu-optix-sort36", "wf-sort36"),
]
# Pairs of sides differing by one decision, for the summary's reading of
# the figure (plot_cpu_wavefront.COMPARISONS explains).
COMPARISONS = [
    ("wf", "mk", "pbrt's wavefront schedule, matched, against the OptiX megakernel"),
    ("mk-ser", "mk", "shader execution reordering before the material stage"),
    ("mk-ser-ray", "mk-ser", "reordering keyed by the ray instead of the material"),
    ("wf-b128", "wf", "bands of 128 pixel rows instead of the matching band size"),
    ("wf-b512", "wf", "bands of 512 pixel rows instead of the matching band size"),
    ("wf-fused", "wf", "the camera-ray generation fused into the first trace"),
    ("wf-sort", "wf", "the ray queue sorted by the 60-bit two-point Morton key before each trace"),
    ("wf-sort36", "wf-sort", "the coarse 36-bit sort key instead of the 60-bit one"),
]
CAUSES = [
    "causes measured, apps/pbrt/PLAN.md: 'The GPU matrix measured: five scenes, three sample",
    "  counts, four OptiX sides (10-24)' -- the reading of the four paper sides, the sort's own",
    "  time against what coherence saves on the RT cores; 'The GPU schedule space explored: is",
    "  the best schedule data-dependent? (10-24)' -- the band sizes, the fused wavefront and the",
    "  reorder keys; 'The fused wavefront's trade read off the profiles (10-24)' -- fusing costs",
    "  a fifth to a third where the trace is cheap and nothing on sanmiguel, whose two-level",
    "  trace is latency-bound. REPORT.md 10-23..10-24 has each run's numbers.",
]


def colour_sides(sides):
    """`sides` ((side, label) in drawing order) with a YlGnBu class each:
    the classes spread evenly over 1..7 for up to seven sides (four sides:
    1, 3, 5, 7) and over the whole ramp 0..8 for eight or nine; more than
    nine is an error, since the ramp has nine classes."""
    n = len(sides)
    if n > len(YLGNBU):
        raise ValueError(f"{n} sides for a ramp of {len(YLGNBU)} colours")
    low, high = (1, 7) if n <= 7 else (0, 8)
    if n == 1:
        classes = [high]
    else:
        classes = [round(low + (high - low) * i / (n - 1)) for i in range(n)]
    return [(side, label, YLGNBU[c]) for (side, label), c in zip(sides, classes)]


def load(paths, kernel=False):
    """gpu_compare.sh's tables as rows with `scene`, `spp` (int), `side`,
    `seconds` (float: the wall time, or the kernel time in seconds with
    `kernel`), `verdict` and `reference` (the seconds of the `pbrt --gpu`
    row of the same cell in the same table); a row without a time draws
    nothing, and a (scene, spp, side) a later table repeats replaces the
    earlier row."""
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
            table.append(r)
        references = {}
        for r in table:
            if r["side"] == REFERENCE:
                key = (r["scene"], r["spp"])
                references[key] = min(r["seconds"], references.get(key, r["seconds"]))
        for r in table:
            key = (r["scene"], r["spp"], r["side"])
            r["reference"] = references.get((r["scene"], r["spp"]))
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
    draw(rows, colour_sides(sides), REFERENCE, REFERENCE_LABEL, Y_LABEL, args, COMPARISONS, CAUSES)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
