#!/usr/bin/env python3
"""The GPU schedule study as one figure, the CPU figure's twin: every
schedule's speedup over pbrt's GPU wavefront (`pbrt --gpu`), a group of
bars per scene, a subgroup per sample count, from the table gpu_compare.sh
writes.

    python3 apps/pbrt/plot_gpu_wavefront.py apps/pbrt/gpu-compare-out/table.tsv \
        -o apps/pbrt/plots/gpu-wavefront-speedups.pdf

The table is tab-separated, one row per measured cell and side (scene,
spp, side, wall_s, kernel_ms, speedup, agree, verdict, note, image, path;
gpu_compare.sh's header says what each is). The side `pbrt --gpu` is the
reference and draws no bar: the dashed line at 1 is it, labelled "PBRT wf"
since pbrt's GPU renderer is its wavefront (pbrt has no GPU megakernel, so
the GPU figure has no black bar). The bars are the wall-clock speedup, the
least of the run's repeats on each side, as the table's own speedup column
is; `--kernel` draws the kernel-time speedup instead (pbrt's from
`--stats`, ours from BONSAI_KERNEL_STATS), the device's busy time apart
from either side's launch gaps.

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
colours are four classes of the CPU figure's YlGnBu ramp, spread so that
neighbours differ in lightness; everything else about the drawing is
plot_cpu_wavefront.py's `draw`.
"""
import argparse
import csv
import sys

from plot_cpu_wavefront import YLGNBU, add_figure_arguments, choose_sides, draw

REFERENCE = "pbrt --gpu"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup over PBRT wf"

SIDES = [
    ("gpu-optix-mega", "Bonsai mk", YLGNBU[1]),
    ("gpu-optix-mega-ser", "Bonsai mk-ser", YLGNBU[3]),
    ("gpu-optix", "Bonsai wf", YLGNBU[5]),
    ("gpu-optix-sort", "Bonsai wf-sort", YLGNBU[7]),
]


def load(path, kernel=False):
    """gpu_compare.sh's table as rows with `scene`, `spp` (int), `side`,
    `seconds` (float: the wall time, or the kernel time in seconds with
    `kernel`) and `verdict`; a row without a time draws nothing."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        try:
            r["spp"] = int(r["spp"])
            r["seconds"] = (float(r["kernel_ms"]) / 1000.0 if kernel
                            else float(r["wall_s"]))
        except (ValueError, KeyError):
            continue
        out.append(r)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    add_figure_arguments(ap)
    ap.add_argument("--kernel", action="store_true",
                    help="the kernel-time speedup instead of the wall-clock one")
    args = ap.parse_args(argv[1:])
    if args.out is None:
        args.out = "gpu-wavefront-speedups.pdf"

    rows = load(args.table, kernel=args.kernel)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
