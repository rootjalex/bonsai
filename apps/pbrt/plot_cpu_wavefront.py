#!/usr/bin/env python3
"""The CPU wavefront study as one figure: every schedule's speedup over
pbrt's CPU wavefront (`pbrt --wavefront` without `--gpu`), a group of bars
per scene, a subgroup per sample count, from the CSV cpu_compare.sh writes.

    python3 apps/pbrt/plot_cpu_wavefront.py apps/pbrt/cpu-compare-out/cpu-results.csv \
        -o apps/pbrt/plots/cpu-wavefront-speedups.pdf

The CSV has one row per measured cell and side (scene, spp, side, render
seconds, speedup, verdict, ...; cpu_compare.sh's header says what each
column is). The side `pbrt-wavefront` is the reference and draws no bar:
the dashed line at 1 is it. The other sides are drawn in the order of SIDES
below -- pbrt's megakernel first, then this renderer's schedules named by
the encoding SIDES explains (type-parallelism-SIMD-order[-sort]: mk-b-s-ps
is the path-per-sample `scalar` schedule, wf-b-v-sp-sort the band
wavefront with vectorized drains and the rays sorted, wf-t-v-ps the
per-thread vectorized wavefront). `--sides` names a subset (comma-separated
side names) and keeps that order, so the five-bar figure of the first study is
`--sides pbrt-volpath,scalar,wavefront-frame,wavefront-frame-packet,wavefront-frame-packet-reorder`.

Each side has its own colour from ColorBrewer's nine-class YlGnBu ramp
(the user's choice, 10-23, after the hatched one-blue figure proved hard to
read): a sequential ramp, pale yellow to deep blue, laid over the sides in
their drawing order, so neighbouring bars differ in lightness as well as
hue and the figure reads in greyscale and under the common colour
deficiencies (ColorBrewer marks the ramp colour-blind safe and
print-friendly; no nine-colour qualitative palette is). Every bar has a
thin dark edge so the pale end stays visible on white. The figure does not
mark the image checks: the CSV's
verdict column has them, and the one that ever failed (sanmiguel 16 spp
against pbrt's volpath) was pbrt's own two integrators disagreeing by more
than we differ from either, which the user ruled fine (10-23). The bars carry no
numbers (the user, 10-22: they made the figure noisy; the CSV has them;
`--values` puts them back). `--width` and `--height` are the figure
size in inches (the defaults fit a two-column page's full width); `--log`
puts the speedup axis on a log scale, where the line at 1 sits mid-axis.
The output format follows the extension (.pdf for the paper, .png to look
at); fonts are embedded as TrueType (fonttype 42).
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REFERENCE = "pbrt-wavefront"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup over PBRT wf"   # the user's wording (10-23)
EDGE = "#333333"             # every bar's thin edge

# ColorBrewer YlGnBu, nine classes, lightest first (the user's palette, 10-23).
YLGNBU = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
          "#1D91C0", "#225EA8", "#253494", "#081D58"]

# (side in the CSV, label in the legend, colour): the drawing order.
# pbrt has two schedules and is named by them: its megakernel (the CPU
# volpath integrator, a path per sample) and its wavefront, the reference.
# A Bonsai schedule is named by the user's encoding (10-22),
# type-parallelism-SIMD-order[-sort]: type `wf` (wavefront) or `mk`
# (megakernel, a path per sample); parallelism `b` (bulk: the frame's
# pixels shared by the threads, as pbrt's bands and a megakernel's pixel
# loop are) or `t` (per-thread: a thread owns a chunk of pixels and its
# queues); SIMD `s` (scalar drains) or `v` (vectorized, sixteen-wide gangs);
# order `ps` (pixels outer, samples inner, the program's order) or `sp`
# (samples outer, pixels inner: `reorder(p, s)` or `reorder(p_in, s)`);
# `-sort` when the rays queue is sorted by the two-point Morton key before
# each trace pass. The code is read from the schedule file's directives.
SIDES = [
    ("pbrt-volpath", "PBRT mk", YLGNBU[0]),
    ("scalar", "Bonsai mk-b-s-ps", YLGNBU[1]),
    ("wavefront-frame", "Bonsai wf-b-s-sp", YLGNBU[2]),
    ("wavefront-frame-packet", "Bonsai wf-b-v-sp", YLGNBU[3]),
    ("wavefront-frame-packet-reorder", "Bonsai wf-b-v-sp-sort", YLGNBU[4]),
    ("wavefront-thread", "Bonsai wf-t-s-ps", YLGNBU[5]),
    ("wavefront-thread-packet", "Bonsai wf-t-v-ps", YLGNBU[6]),
    ("wavefront-thread-packet-reorder", "Bonsai wf-t-v-sp", YLGNBU[7]),
    ("wavefront-thread-packet-sort", "Bonsai wf-t-v-ps-sort", YLGNBU[8]),
]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            r["spp"] = int(r["spp"])
            r["seconds"] = float(r["seconds"])
        except (ValueError, KeyError):
            continue       # a row whose run produced no time draws nothing
        out.append(r)
    return out


def ordered(values):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="cpu-wavefront-speedups.pdf")
    ap.add_argument("--sides", default=None,
                    help="comma-separated side names to draw, in this order")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.4)
    ap.add_argument("--log", action="store_true", help="log-scale speedup axis")
    ap.add_argument("--values", action="store_true", help="the speedup written on each bar")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv[1:])

    rows = load(args.csv)
    scenes = ordered(r["scene"] for r in rows)
    spps = sorted(ordered(r["spp"] for r in rows))
    sides = SIDES
    if args.sides:
        wanted = args.sides.split(",")
        known = {s[0]: s for s in SIDES}
        missing = [w for w in wanted if w not in known]
        if missing:
            print(f"unknown sides: {', '.join(missing)}", file=sys.stderr)
            return 1
        sides = [known[w] for w in wanted]
    present = {r["side"] for r in rows}
    sides = [s for s in sides if s[0] in present]
    if not sides:
        print("no side of the CSV is one SIDES knows", file=sys.stderr)
        return 1

    def cell(scene, spp, side):
        return next((r for r in rows if r["scene"] == scene and r["spp"] == spp
                     and r["side"] == side), None)

    def reference(scene, spp):
        # The least of the cell's reference rows, should a run have more than one.
        times = [r["seconds"] for r in rows
                 if r["scene"] == scene and r["spp"] == spp and r["side"] == REFERENCE]
        return min(times) if times else None

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    bar = 1.0                     # one bar's width in x units
    gap_spp = 2.0                 # between the sample counts of a scene
    gap_scene = 4.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            ref = reference(scene, spp)
            group_start = x
            for side, label, colour in sides:
                c = cell(scene, spp, side)
                if c is not None and ref is not None:
                    speedup = ref / c["seconds"]
                    ax.bar(x, speedup, bar * 0.9, color=colour,
                           edgecolor=EDGE, linewidth=0.4, zorder=2)
                    if args.values:
                        ax.text(x, speedup, f"{speedup:.2f}", ha="center", va="bottom",
                                fontsize=5.5, rotation=90, zorder=4)
                x += bar
            spp_centres.append((group_start + x - bar) / 2)
            spp_labels.append(f"{spp}")       # "spp" is written once, at the left
            if j + 1 < len(spps):
                x += gap_spp
        scene_centres.append((scene_start + x - bar) / 2)
        x += gap_scene

    right = x - gap_scene
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=1.2, zorder=3)
    ax.text(right + 0.3, 1.0, REFERENCE_LABEL, ha="left", va="center", fontsize=6.5)
    ax.set_xticks(spp_centres)
    ax.set_xticklabels(spp_labels)
    ax.tick_params(axis="x", length=0)
    ax.text(-bar, -0.045, "spp", ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8)
    for centre, scene in zip(scene_centres, scenes):
        ax.text(centre, -0.16, scene, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylabel(Y_LABEL)
    if args.log:
        ax.set_yscale("log")
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * (1.12 if args.values else 1.04))
    ax.set_xlim(-bar, right + 0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    handles = [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.4, label=label)
               for _, label, colour in sides]
    ncol = 3 if len(handles) > 6 else 2 if len(handles) > 4 else len(handles)
    legend_rows = (len(handles) + ncol - 1) // ncol
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0 + 0.02), handlelength=2.2, handleheight=1.2,
              borderaxespad=0.0, columnspacing=1.2)

    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out} ({len(scenes)} scenes, {len(spps)} sample counts, "
          f"{len(sides)} sides, {legend_rows} legend rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
