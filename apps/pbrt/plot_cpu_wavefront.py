#!/usr/bin/env python3
"""The CPU wavefront study as one figure: every schedule's speedup over
pbrt's CPU wavefront (`pbrt --wavefront` without `--gpu`), a group of bars
per scene, a subgroup per sample count, from the CSV cpu_compare.sh writes.

    python3 apps/pbrt/plot_cpu_wavefront.py apps/pbrt/cpu-compare-out/cpu-results.csv \
        [apps/pbrt/cpu-compare-out-sortfix/cpu-results.csv ...] \
        -o apps/pbrt/plots/pbrt-cpu-sweep.pdf

(plots.sh draws it under that name, the figure's one name since 10-24.)

A CSV has one row per measured cell and side (scene, spp, side, render
seconds, speedup, verdict, ...; cpu_compare.sh's header says what each
column is). The side `pbrt-wavefront` is the reference and draws no bar:
the dashed line at 1 is it. Several CSVs are several runs joined: a later
file's row replaces an earlier one's for the same (scene, spp, side), and
every side's bar is its speedup over the pbrt that ran in the same CSV
(the sorted schedules were re-measured on 10-24 with the ray key fixed,
into a CSV of their own). The other sides are drawn in the order of SIDES
below -- pbrt's megakernel first, then this renderer's schedules named by
the encoding SIDES explains (type-parallelism-SIMD-order[-sort]: mk-b-s-ps
is the path-per-sample `scalar` schedule, wf-b-v-sp-sort the band
wavefront with vectorized drains and the rays sorted, wf-t-v-ps the
per-thread vectorized wavefront). `--sides` names a subset (comma-separated
side names) and keeps that order, so the five-bar figure of the first study is
`--sides pbrt-volpath,scalar,wavefront-frame,wavefront-frame-packet,wavefront-frame-packet-reorder`.

pbrt's bar is black and each Bonsai side has its own colour from
ColorBrewer's nine-class YlGnBu ramp, the first eight classes (the user's
choice, 10-23, after the hatched one-blue figure proved hard to read): a
sequential ramp, pale yellow to deep blue, laid over the Bonsai sides in
their drawing order, so neighbouring bars differ in lightness as well as
hue and the figure reads in greyscale and under the common colour
deficiencies (ColorBrewer marks the ramp colour-blind safe and
print-friendly; no nine-colour qualitative palette is). Every bar has a
thin dark edge so the pale end stays visible on white. The figure does not
mark the image checks: the CSV's verdict column has them, and the one that
ever failed (sanmiguel 16 spp against pbrt's volpath) was pbrt's own two
integrators disagreeing by more than we differ from either, which the user
ruled fine (10-23). The bars carry no numbers (the user, 10-22: they made
the figure noisy; the CSV has them; `--values` puts them back). `--width`
and `--height` are the figure size in inches; the defaults are the paper's
text width (acmsmall, 5.478 in; figure_style.py has the format, the fonts
and their sizes, so a figure included at `\\textwidth` is not rescaled and
its 10 pt labels stay 10 pt) by 2.7 in, the legend in three columns above
the axes (three rows for nine sides; the user asked for five over four on
10-24, which the nine labels at 10 pt do not fit into 5.478 in). `--log`
puts the speedup axis on a log scale, where the line at 1 sits mid-axis.
The output format follows the extension (.pdf for the paper, .png to look
at); fonts are embedded as TrueType (fonttype 42, as ACM requires).

The drawing is `draw`, shared with plot_gpu_wavefront.py, which draws the
GPU study's table the same way with its own sides and reference.
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from figure_style import FONT, TEXT_WIDTH, TICK_FONT, paper_style  # noqa: E402

REFERENCE = "pbrt-wavefront"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup over\nPBRT wf"   # the user's wording (10-23), on two lines at 10 pt
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
PBRT_COLOUR = "#000000"      # pbrt's bar is black; the ramp is Bonsai's (the user, 10-23)
SIDES = [
    ("pbrt-volpath", "PBRT mk", PBRT_COLOUR),
    ("scalar", "Bonsai mk-b-s-ps", YLGNBU[0]),
    ("wavefront-frame", "Bonsai wf-b-s-sp", YLGNBU[1]),
    ("wavefront-frame-packet", "Bonsai wf-b-v-sp", YLGNBU[2]),
    ("wavefront-frame-packet-reorder", "Bonsai wf-b-v-sp-sort", YLGNBU[3]),
    ("wavefront-thread", "Bonsai wf-t-s-ps", YLGNBU[4]),
    ("wavefront-thread-packet", "Bonsai wf-t-v-ps", YLGNBU[5]),
    ("wavefront-thread-packet-reorder", "Bonsai wf-t-v-sp", YLGNBU[6]),
    ("wavefront-thread-packet-sort", "Bonsai wf-t-v-ps-sort", YLGNBU[7]),
]


def load(paths, reference=REFERENCE):
    """cpu_compare.sh's CSVs as rows with `scene`, `spp` (int), `side`,
    `seconds` (float), `verdict` and `reference` (the seconds of the
    reference side of the same cell in the same CSV); a row whose run
    produced no time draws nothing and is dropped. Several CSVs are
    several runs joined, a later file's row replacing an earlier one's for
    the same (scene, spp, side) -- a schedule re-measured after a fix, as
    the sorted CPU schedules were on 10-24 -- and every side measured
    against the reference that ran beside it."""
    joined = {}
    order = []
    for path in paths:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        table = []
        for r in rows:
            try:
                r["spp"] = int(r["spp"])
                r["seconds"] = float(r["seconds"])
            except (ValueError, KeyError):
                continue
            table.append(r)
        references = {}
        for r in table:
            if r["side"] == reference:
                key = (r["scene"], r["spp"])
                references[key] = min(r["seconds"], references.get(key, r["seconds"]))
        for r in table:
            key = (r["scene"], r["spp"], r["side"])
            r["reference"] = references.get((r["scene"], r["spp"]))
            if key not in joined:
                order.append(key)
            joined[key] = r
    return [joined[key] for key in order]


def ordered(values):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def add_figure_arguments(ap):
    """The arguments the CPU and GPU figures share."""
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--sides", default=None,
                    help="comma-separated side names to draw, in this order")
    ap.add_argument("--width", type=float, default=TEXT_WIDTH,
                    help="inches; the default is the paper's text width (figure_style.py)")
    ap.add_argument("--height", type=float, default=2.7,
                    help="inches, the axes and the legend above them")
    ap.add_argument("--log", action="store_true", help="log-scale speedup axis")
    ap.add_argument("--values", action="store_true", help="the speedup written on each bar")
    ap.add_argument("--title", default=None)


def choose_sides(rows, all_sides, wanted):
    """The sides to draw: `wanted` (comma-separated names) in that order, or
    all of `all_sides`; either way only those the rows have. None, with a
    message on stderr, when nothing can be drawn."""
    sides = all_sides
    if wanted:
        names = wanted.split(",")
        known = {s[0]: s for s in all_sides}
        missing = [w for w in names if w not in known]
        if missing:
            print(f"unknown sides: {', '.join(missing)}", file=sys.stderr)
            return None
        sides = [known[w] for w in names]
    present = {r["side"] for r in rows}
    sides = [s for s in sides if s[0] in present]
    if not sides:
        print("no side of the table is one the figure knows", file=sys.stderr)
        return None
    return sides


def draw(rows, sides, reference, reference_label, y_label, args):
    """The figure: a bar per side of `sides` ((side, label, colour), in
    drawing order) in a subgroup per sample count in a group per scene, each
    bar the `reference` side's seconds over the side's, from `rows` as `load`
    returns them. Written to args.out; returns the number of legend rows."""
    scenes = ordered(r["scene"] for r in rows)
    spps = sorted(ordered(r["spp"] for r in rows))

    def cell(scene, spp, side):
        return next((r for r in rows if r["scene"] == scene and r["spp"] == spp
                     and r["side"] == side), None)

    def reference_seconds(scene, spp):
        # The least of the cell's reference rows, should a run have more than one.
        times = [r["seconds"] for r in rows
                 if r["scene"] == scene and r["spp"] == spp and r["side"] == reference]
        return min(times) if times else None

    def reference_of(c, scene, spp):
        # A row from a run of its own carries that run's reference seconds
        # (`reference`, set by the loader: the GPU figure joins tables), so
        # that a side is always measured against the pbrt that ran beside it.
        return c["reference"] if c.get("reference") is not None else reference_seconds(scene, spp)

    paper_style()
    # Constrained layout keeps the legend, the scene names and the y label
    # inside a figure of exactly the size asked, so the PDF is the text width.
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")

    bar = 1.0                     # one bar's width in x units
    gap_spp = 1.5                 # between the sample counts of a scene
    gap_scene = 3.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            group_start = x
            for side, label, colour in sides:
                c = cell(scene, spp, side)
                ref = reference_of(c, scene, spp) if c is not None else None
                if c is not None and ref is not None:
                    speedup = ref / c["seconds"]
                    ax.bar(x, speedup, bar * 0.9, color=colour,
                           edgecolor=EDGE, linewidth=0.3, zorder=2)
                    if args.values:
                        ax.text(x, speedup, f"{speedup:.2f}", ha="center", va="bottom",
                                fontsize=TICK_FONT - 2, rotation=90, zorder=4)
                x += bar
            spp_centres.append((group_start + x - bar) / 2)
            spp_labels.append(f"{spp}")       # "spp" is written once, at the left
            if j + 1 < len(spps):
                x += gap_spp
        scene_centres.append((scene_start + x - bar) / 2)
        x += gap_scene

    right = x - gap_scene
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.9, zorder=3)
    ax.text(right + 0.3, 1.0, reference_label, ha="left", va="center", fontsize=TICK_FONT)
    ax.set_xticks(spp_centres)
    ax.set_xticklabels(spp_labels)
    ax.tick_params(axis="x", length=0)
    ax.text(-bar, -0.03, "spp", ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=TICK_FONT)
    for centre, scene in zip(scene_centres, scenes):
        # A long name breaks at its last hyphen onto a second line, so that
        # five scenes' names fit side by side at text width.
        if len(scene) > 10 and "-" in scene:
            head, tail = scene.rsplit("-", 1)
            scene = f"{head}-\n{tail}"
        ax.text(centre, -0.15, scene, ha="center", va="top", linespacing=1.1,
                transform=ax.get_xaxis_transform(), fontsize=TICK_FONT)
    ax.set_ylabel(y_label)
    if args.log:
        ax.set_yscale("log")
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * (1.12 if args.values else 1.04))
    ax.set_xlim(-bar, right + 0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    handles = [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.3, label=label)
               for _, label, colour in sides]
    # Nine labels of "Bonsai wf-t-v-ps-sort" at 10 pt do not fit five across
    # the paper's 5.478 in (the user's two-row layout, 10-24, needs about
    # 8 in): three columns, three rows.
    ncol = len(handles) if len(handles) <= 4 else 3
    legend_rows = (len(handles) + ncol - 1) // ncol
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0 + 0.02),
              borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out} ({len(scenes)} scenes, {len(spps)} sample counts, "
          f"{len(sides)} sides, {legend_rows} legend rows)")
    return legend_rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    add_figure_arguments(ap)
    args = ap.parse_args(argv[1:])
    if args.out is None:
        args.out = "pbrt-cpu-sweep.pdf"

    rows = load(args.csvs)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
