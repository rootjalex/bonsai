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
Colour is log2 of the speedup on the paper's YlGnBu ramp (the wavefront
family's ramp on the sweep figures: this is the wavefront's figure),
parity its palest class and the top of the data (rounded up to a half
power of two) its darkest; a cell below parity would be red (none is). The numbers are
the ratios to two digits ("1.6"), what a cell holds at text width (the
user, 10-24). Beside the figure goes `<figure>.txt`, the summary the paper
quotes: geometric mean, least and greatest cell and which they are, over
everything, per sample count and per scene (figure_style.write_summary).
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

from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from figure_style import (TEXT_WIDTH, TICK_FONT, YLGNBU, geomean, lean_angle,  # noqa: E402
                          lean_height, lean_labels, paper_style, scene_label,
                          write_summary)


BELOW = "#d7301f"       # a cell below parity, off the ramp


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
        # A scene named by a camera or view number (kroken/camera-1,
        # watercolor/camera-1, landscape/view-0) is labelled by its
        # directory, the scene's own name (figure_style.scene_label; the
        # full names otherwise, this figure having room to lean them), as
        # is any scene whose name another scene shares.
        shared = sum(1 for k in scenes if names[k] == names[key]) > 1
        labels[key] = (key.split("/")[0] if shared and "/" in key
                       else scene_label(key, short=False))
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


def reading(scenes, spps, cells, labels, stats):
    """Lines that read the heatmap from its own data (the user, 10-24: the
    statistics should explain the speedups where they can): the trend from
    the fewest samples to the most and how many scenes follow it; the
    volume scenes against the surface scenes; the rank correlation of the
    speedup with the geometry pbrt built (from the `.pbrt.stats` beside
    each pbrt image, when present); and where the measured causes are."""
    notes = []
    if len(spps) > 1:
        lo, hi = spps[0], spps[-1]
        at_lo = geomean([(sc, cells[(sc, lo)][0]) for sc in scenes if (sc, lo) in cells])
        at_hi = geomean([(sc, cells[(sc, hi)][0]) for sc in scenes if (sc, hi) in cells])
        falling = sum(1 for sc in scenes if (sc, lo) in cells and (sc, hi) in cells
                      and cells[(sc, hi)][0] < cells[(sc, lo)][0])
        both = sum(1 for sc in scenes if (sc, lo) in cells and (sc, hi) in cells)
        if at_lo and at_hi:
            notes.append(f"with more samples the speedup falls: geomean {at_lo:.2f}x at {lo} spp, "
                         f"{at_hi:.2f}x at {hi} spp, lower at {hi} than at {lo} on {falling} of "
                         f"{both} scenes -- consistent with the per-frame costs we save (pbrt's "
                         f"launches, queue resets, camera rays) being a smaller share of a "
                         f"longer render")
    counts = {sc: shape_count(stats.get(sc, "")) for sc in scenes}
    with_counts = [sc for sc in scenes if counts[sc]]
    if len(with_counts) >= 4:
        volumes = [sc for sc in with_counts if counts[sc] <= 8]
        surfaces = [sc for sc in with_counts if counts[sc] > 8]
        per_scene = {sc: geomean([(s, cells[(sc, s)][0]) for s in spps if (sc, s) in cells])
                     for sc in with_counts}
        if volumes and surfaces:
            gv = geomean([(sc, per_scene[sc]) for sc in volumes])
            gs = geomean([(sc, per_scene[sc]) for sc in surfaces])
            notes.append(f"volume scenes (at most 8 shapes: {', '.join(labels[sc] for sc in volumes)})"
                         f" geomean {gv:.2f}x; surface scenes {gs:.2f}x -- a volume render is "
                         f"mostly ray marching in the medium stage, where our kernel and pbrt's "
                         f"do the same work, so the surface stages' gains weigh less")
        # Spearman's rank correlation of the speedup with the geometry.
        ranked = sorted(surfaces, key=lambda sc: counts[sc])
        if len(ranked) >= 4:
            def rank(order):
                return {sc: i for i, sc in enumerate(order)}
            r_geo = rank(ranked)
            r_speed = rank(sorted(surfaces, key=lambda sc: per_scene[sc]))
            n = len(surfaces)
            d2 = sum((r_geo[sc] - r_speed[sc]) ** 2 for sc in surfaces)
            rho = 1 - 6 * d2 / (n * (n * n - 1))
            notes.append(f"rank correlation of the speedup with the geometry built, over the "
                         f"{n} surface scenes: Spearman rho = {rho:+.2f} -- "
                         + ("the speedup does not follow scene size" if abs(rho) < 0.3 else
                            ("a weak" if abs(rho) < 0.6 else "a strong") + " tendency for larger "
                            "scenes to speed up " + ("more" if rho > 0 else "less"))
                         + "; the trace itself both sides do on the RT cores, so what the "
                           "schedule saves is around it (queues, the material dispatch, launches)")
        # The ends of the range, with the geometry each was built from.
        by_speed = sorted(with_counts, key=lambda sc: per_scene[sc])
        def named(scs):
            return ", ".join(f"{labels[sc]} {per_scene[sc]:.2f}x ({counts[sc]:,} shapes)" for sc in scs)
        notes.append(f"the lowest scenes: {named(by_speed[:4])}; the highest: "
                     f"{named(by_speed[-4:][::-1])} -- the media scenes and the largest instanced "
                     f"scene at the bottom, the small two-material scenes and watercolor at the top; "
                     f"beyond the volume split above, the per-scene causes are not measured for this "
                     f"figure (the sweep figure's .txt reads its five scenes)")
    notes.extend([
        "causes measured, apps/pbrt/PLAN.md: 'The GPU matrix measured: five scenes, three sample",
        "  counts, four OptiX sides (10-24)' -- what the matching wavefront saves over pbrt's;",
        "  'The fused wavefront's trade read off the profiles (10-24)' -- the per-scene kernel",
        "  profiles; the heatmap7 entries of REPORT.md (10-19..10-24) -- each cell's run.",
    ])
    return notes


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
    parser.add_argument("--per-band", type=int, default=0,
                        help="scenes across one band, more scenes making more bands "
                             "stacked down the figure; 0 (the default) is one band")
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
    # Colour: log2 of the speedup from parity (the ramp's palest class) up
    # to the top of the data rounded to a half power of two (its darkest),
    # on the paper's YlGnBu ramp; a cell below parity, should one appear,
    # is red (none of the 120 is).
    all_logs = [math.log2(cells[(sc, s)][0]) for sc in scenes for s in spps if (sc, s) in cells]
    top = max(1.0, math.ceil(max(all_logs, default=1.0) * 2) / 2)
    norm = Normalize(vmin=0.0, vmax=top)
    cmap = LinearSegmentedColormap.from_list("ylgnbu", YLGNBU)
    cmap.set_under(BELOW)

    per = args.per_band if args.per_band > 0 else len(scenes)
    bands = [scenes[i:i + per] for i in range(0, len(scenes), per)]
    across = min(per, len(scenes))
    cell_h = 0.2                                    # inches, one sample count's row
    # The scene names lean at the least angle at which neighbours do not
    # overlap (figure_style.lean_angle; the user, 10-24: upright names
    # wasted a third of the figure). Thirty columns at text width lean 45
    # degrees or so; a few columns lie flat.
    pitch = (args.width - 0.5) / across
    names = [labels[sc] for sc in scenes]
    angle = lean_angle(names, pitch)
    label_h = lean_height(names, angle)             # the leaning names under a band
    bar_h = 0.45                                    # the colour bar above, with its label
    height = len(bands) * (len(spps) * cell_h + label_h) + bar_h
    paper_style()
    # Constrained layout keeps the names and the colour bar inside a figure
    # of exactly the width asked, so the PDF is the text width.
    fig, axes = plt.subplots(len(bands), 1, figsize=(args.width, height), squeeze=False,
                             layout="constrained")
    axes = [row[0] for row in axes]
    image = None
    for ax, band in zip(axes, bands):
        logs = [[math.log2(cells[(sc, s)][0]) if (sc, s) in cells else float("nan")
                 for sc in band] for s in spps]
        image = ax.imshow(logs, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xlim(-0.5, across - 0.5)             # every band's cells the same width
        ax.set_xticks(range(len(band)))
        lean_labels(ax, [labels[sc] for sc in band], pitch)
        ax.set_yticks(range(len(spps)))
        ax.set_yticklabels([str(s) for s in spps])
        ax.set_ylabel("spp")
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Cell borders, as a table has.
        ax.set_xticks([i - 0.5 for i in range(len(band) + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(len(spps) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)

        for i, s in enumerate(spps):
            for j, sc in enumerate(band):
                cell = cells.get((sc, s))
                if cell is None:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee",
                                               edgecolor="white", linewidth=0.8))
                    continue
                v, matched = cell
                # Two digits ("1.6", "3.2"): what a 0.16 in cell holds at
                # 6.5 pt (the user, 10-24). Dark text on the pale half of
                # the ramp, white on the dark half and on red.
                dark = 0.0 <= math.log2(v) < 0.55 * top
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=TICK_FONT - 0.5,
                        color="black" if dark else "white")
                if not matched:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               hatch="///", edgecolor="black", linewidth=0))
                    ax.plot([j - 0.3, j + 0.3], [i, i], color="black", linewidth=0.8)
    if args.title:
        axes[0].set_title(args.title)

    # The colour bar lies along the top, so the cells have the whole width.
    bar = fig.colorbar(image, ax=axes, location="top", orientation="horizontal",
                       fraction=0.06, pad=0.02, shrink=0.45, aspect=35)
    ticks = [t for t in (0.0, math.log2(1.5), 1.0, math.log2(3), 2.0, math.log2(6), 3.0)
             if t <= top + 1e-9]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{2.0 ** t:g}x" for t in ticks])
    bar.set_label("Speedup over PBRT wf", labelpad=2)
    bar.outline.set_linewidth(0.5)
    bar.ax.tick_params(length=2, pad=1.5)
    fig.savefig(args.out, dpi=200)
    # The summary beside the figure: every cell, then each sample count's
    # row, then each scene's column.
    def points(keys):
        return [(f"{labels[sc]} {s} spp", cells[(sc, s)][0]) for sc, s in keys if (sc, s) in cells]
    sections = [("every scene and sample count",
                 [(args.schedule, points((sc, s) for sc in scenes for s in spps))]),
                ("by sample count", [(f"{s} spp", points((sc, s) for sc in scenes)) for s in spps]),
                ("by scene", [(labels[sc], points((sc, s) for s in spps)) for sc in scenes])]
    notes = reading(scenes, spps, cells, labels, stats)
    here = os.path.dirname(os.path.abspath(__file__))
    drawn_from = ", ".join(os.path.relpath(t, here) for t in args.tables)
    summary = write_summary(args.out, f"Speedup of {args.schedule} over pbrt --gpu "
                            f"({len(scenes)} scenes x {len(spps)} sample counts)\n"
                            f"drawn from apps/pbrt/{drawn_from}", sections, notes)
    blank = sum(1 for s in spps for sc in scenes if (sc, s) not in cells)
    wrong = sum(1 for c in cells.values() if not c[1])
    print(f"wrote {args.out}: {len(scenes)} scenes x {len(spps)} sample counts in "
          f"{len(bands)} band(s), {blank} blank, {wrong} not matching pbrt; and {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
