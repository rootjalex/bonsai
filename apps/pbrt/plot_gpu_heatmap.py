#!/usr/bin/env python3
"""A heatmap of gpu_compare.sh's tables: scenes across, sample counts down,
each cell the speedup of one of this renderer's GPU schedules over
`pbrt --gpu` on that scene at that count -- the figure the GraphIt paper
draws for its schedules against the hand-written baselines, a grid of
numbers coloured by how far each is from parity.

    python3 apps/pbrt/plot_gpu_heatmap.py OUT.png TABLE.tsv [TABLE.tsv ...]
        [--schedule gpu-optix] [--sort name|speedup] [--title TEXT]

Each table is a `table.tsv` gpu_compare.sh wrote; several may be given, and
one may hold several sample counts (gpu_compare.sh --spp "16 64 256", or
--resume runs into one directory). The rows of the schedule asked for are
collected by (scene, spp); a cell no table has is left blank, and a cell
whose image did not match pbrt's is drawn hatched with its number struck
through, so that a speedup on a wrong picture cannot be read as a speedup.
Colour is log2 of the speedup on a diverging scale centred on parity --
green above one, red below -- so 2x and 0.5x are equally far from the
middle; the numbers are the ratios themselves, as the table prints them.
Scenes are in the tables' order, or by name, or by their mean speedup, or
(`--sort geometry`) by how much geometry pbrt built for them -- the shapes
counted in the Geometry section of the `.pbrt.stats` file gpu_compare.sh
keeps beside each pbrt image (`Triangles per mesh N / M` counts N; spheres,
disks, curves and bilinear patches count too), read from the first pbrt row
of each scene; a purely volumetric scene has one or two shapes and comes
first. The order and its counts are printed. The figure has no title
unless `--title` gives one, the axes read "scene" and "spp" and the colour
bar "Speedup" (the user, 10-24).

The figure is the paper's text width (figure_style.py: acmsmall, 5.478 in,
10 pt labels, TrueType fonts embedded as ACM requires) and the scenes are
cut into bands of at most `--per-band` (default 15) stacked down the
figure, each band the sample counts down and its scenes across, in the
order chosen, continuing from one band to the next; the one colour bar at
the right is for all of them. Thirty scenes in one band were 0.18 in wide
each at text width and unreadable (the user, 10-24); in two they are
0.3 in, room for the ratio in each cell at 8 pt (without its "x", which
the colour bar carries).
"""

import argparse
import csv
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from figure_style import TEXT_WIDTH, TICK_FONT, paper_style  # noqa: E402


def number(text):
    try:
        return float(text.rstrip("x"))
    except (ValueError, AttributeError):
        return None


def read_tables(paths, schedule):
    """(scene, spp) -> (speedup, matched), in the order the tables list scenes.

    A scene is keyed by its `path` column (`kroken/camera-1`) where the
    table has one -- two scenes may share a name -- and labelled by its
    name, or by the path where the name is not unique."""
    cells = {}
    scenes = []
    names = {}
    stats = {}
    spps = set()
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = r.get("path") or r["scene"]
                if r["side"].startswith("pbrt"):
                    # The pbrt side's `.stats` lies beside its image.
                    image = r.get("image") or ""
                    if key not in stats and image.endswith(".png"):
                        stats[key] = image[:-len(".png")] + ".stats"
                    continue
                if r["side"] != schedule:
                    continue
                v = number(r["speedup"])
                if v is None:
                    continue
                spp = int(r["spp"])
                if key not in scenes:
                    scenes.append(key)
                    names[key] = r["scene"]
                spps.add(spp)
                cells[(key, spp)] = (v, r["verdict"].startswith("ok"))
    labels = {}
    for key in scenes:
        shared = sum(1 for k in scenes if names[k] == names[key]) > 1
        labels[key] = key if shared else names[key]
    return scenes, sorted(spps), cells, labels, stats


def shape_count(stats_path):
    """The shapes pbrt built for a scene, summed over the Geometry section of
    its `--stats` output: the numerator of `Triangles per mesh N / M`, and
    the counts of spheres, disks, cylinders, curves and bilinear patches.
    Instances are not expanded -- the count is of base geometry. None when
    the file is missing or has no Geometry section."""
    try:
        with open(stats_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    total = None
    section = None
    for line in lines:
        if line.startswith("Statistics:"):
            section = ""
            continue
        if section is None:
            continue
        if line.startswith("  ") and not line.startswith("    "):
            section = line.strip()
            continue
        if section != "Geometry":
            continue
        # Only the shape counts: pbrt also lists its buffer cache here and
        # the triangles displacement mapping added (already in the total).
        m = re.match(r"\s+(Triangles per mesh|Bilinear patches per mesh|Spheres|Disks|"
                     r"Cylinders|Curves)\s+(\d+)\b", line)
        if m:
            total = (total or 0) + int(m.group(2))
    return total


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--schedule", default="gpu-optix",
                        help="the side of the table to plot (default gpu-optix)")
    parser.add_argument("--sort", choices=["table", "name", "speedup", "geometry"],
                        default="table",
                        help="the order of the scenes across; geometry = by the shapes "
                             "pbrt built, from the .pbrt.stats beside each pbrt image")
    parser.add_argument("--title", default=None)
    parser.add_argument("--per-band", type=int, default=15,
                        help="scenes across one band; more scenes make more bands")
    parser.add_argument("--width", type=float, default=TEXT_WIDTH,
                        help="inches; the default is the paper's text width")
    args = parser.parse_args(argv[1:])

    scenes, spps, cells, labels, stats = read_tables(args.tables, args.schedule)
    if not scenes:
        print(f"no rows of schedule {args.schedule} in the tables given", file=sys.stderr)
        return 1
    if args.sort == "name":
        scenes.sort()
    elif args.sort == "speedup":
        def mean_log(scene):
            vs = [math.log2(cells[(scene, s)][0]) for s in spps if (scene, s) in cells]
            return sum(vs) / len(vs) if vs else 0.0
        scenes.sort(key=mean_log, reverse=True)
    elif args.sort == "geometry":
        counts = {sc: shape_count(stats.get(sc, "")) for sc in scenes}
        missing = [sc for sc in scenes if counts[sc] is None]
        if missing:
            print("no pbrt .stats with a Geometry section for: " + ", ".join(
                f"{sc} ({os.path.basename(stats[sc])})" if sc in stats else sc
                for sc in missing), file=sys.stderr)
            return 1
        scenes.sort(key=lambda sc: (counts[sc], sc))
        for sc in scenes:
            print(f"{counts[sc]:>12,}  {labels[sc]}")

    # log2 of the speedup, parity at zero; the scale is symmetric about it
    # so that the colours mean the same thing on every plot of the same
    # range, and wide enough for whatever the tables hold.
    all_logs = [math.log2(cells[(sc, s)][0]) for sc in scenes for s in spps if (sc, s) in cells]
    extent = max((abs(x) for x in all_logs), default=1.0)
    extent = max(1.0, math.ceil(extent * 2) / 2)
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)

    per = max(1, args.per_band)
    bands = [scenes[i:i + per] for i in range(0, len(scenes), per)]
    across = min(per, len(scenes))
    cell_h = 0.22                                   # inches, one sample count's row
    label_h = 1.25                                  # the rotated scene names under a band
    height = len(bands) * (len(spps) * cell_h + label_h) + 0.2
    paper_style()
    # Constrained layout keeps the bands, their names and the colour bar
    # inside a figure of exactly the width asked, so the PDF is the text width.
    fig, axes = plt.subplots(len(bands), 1, figsize=(args.width, height), squeeze=False,
                             layout="constrained")
    axes = [row[0] for row in axes]
    image = None
    for ax, band in zip(axes, bands):
        logs = [[math.log2(cells[(sc, s)][0]) if (sc, s) in cells else float("nan")
                 for sc in band] for s in spps]
        image = ax.imshow(logs, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xlim(-0.5, across - 0.5)             # every band's cells the same width
        ax.set_xticks(range(len(band)))
        ax.set_xticklabels([labels[sc] for sc in band], rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.set_yticks(range(len(spps)))
        ax.set_yticklabels([str(s) for s in spps])
        ax.set_ylabel("spp")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Cell borders, as a table has.
        ax.set_xticks([i - 0.5 for i in range(len(band) + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(len(spps) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)

        for i, s in enumerate(spps):
            for j, sc in enumerate(band):
                cell = cells.get((sc, s))
                if cell is None:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee",
                                               edgecolor="white", linewidth=1.0))
                    continue
                v, matched = cell
                # Dark text on the pale middle of the scale, light on its ends.
                dark = abs(math.log2(v)) < 0.6 * extent
                # The ratio alone; the colour bar carries the "x", and
                # "1.60x" at 8 pt would overflow a 0.3 in cell.
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=TICK_FONT - 1,
                        color="black" if dark else "white",
                        fontweight="bold" if not dark else "normal")
                if not matched:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               hatch="///", edgecolor="black", linewidth=0))
                    ax.plot([j - 0.3, j + 0.3], [i, i], color="black", linewidth=0.9)
    axes[-1].set_xlabel("scene")
    if args.title:
        axes[0].set_title(args.title)

    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015, aspect=12 * len(bands))
    ticks = [t for t in range(-int(extent), int(extent) + 1)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{2.0 ** t:g}x" for t in ticks])
    bar.set_label("Speedup")
    bar.outline.set_linewidth(0.6)
    fig.savefig(args.out, dpi=200)
    blank = sum(1 for s in spps for sc in scenes if (sc, s) not in cells)
    wrong = sum(1 for c in cells.values() if not c[1])
    print(f"wrote {args.out}: {len(scenes)} scenes x {len(spps)} sample counts in "
          f"{len(bands)} band(s), {blank} blank, {wrong} not matching pbrt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
