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
Scenes are in the tables' order, or by name, or by their mean speedup.
"""

import argparse
import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


def number(text):
    try:
        return float(text.rstrip("x"))
    except (ValueError, AttributeError):
        return None


def read_tables(paths, schedule):
    """(scene, spp) -> (speedup, matched), in the order the tables list scenes."""
    cells = {}
    scenes = []
    spps = set()
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["side"] != schedule:
                    continue
                v = number(r["speedup"])
                if v is None:
                    continue
                spp = int(r["spp"])
                if r["scene"] not in scenes:
                    scenes.append(r["scene"])
                spps.add(spp)
                cells[(r["scene"], spp)] = (v, r["verdict"].startswith("ok"))
    return scenes, sorted(spps), cells


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--schedule", default="gpu-optix",
                        help="the side of the table to plot (default gpu-optix)")
    parser.add_argument("--sort", choices=["table", "name", "speedup"], default="table",
                        help="the order of the scenes across")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv[1:])

    scenes, spps, cells = read_tables(args.tables, args.schedule)
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

    # log2 of the speedup, parity at zero; the scale is symmetric about it
    # so that the colours mean the same thing on every plot of the same
    # range, and wide enough for whatever the tables hold.
    values = [[cells[(sc, s)][0] if (sc, s) in cells else None for sc in scenes] for s in spps]
    logs = [[math.log2(v) if v else float("nan") for v in row] for row in values]
    extent = max((abs(x) for row in logs for x in row if not math.isnan(x)), default=1.0)
    extent = max(1.0, math.ceil(extent * 2) / 2)
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)

    width = max(6.0, 0.9 * len(scenes) + 2.5)
    height = max(2.6, 0.75 * len(spps) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(logs, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=35, ha="right")
    ax.set_yticks(range(len(spps)))
    ax.set_yticklabels([str(s) for s in spps])
    ax.set_ylabel("samples per pixel")
    ax.set_xlabel("scene")
    ax.set_title(args.title or f"{args.schedule}: speedup over pbrt --gpu")
    # Cell borders, as a table has.
    ax.set_xticks([i - 0.5 for i in range(len(scenes) + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(len(spps) + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    for i, s in enumerate(spps):
        for j, sc in enumerate(scenes):
            cell = cells.get((sc, s))
            if cell is None:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee",
                                           edgecolor="white", linewidth=1.5))
                continue
            v, matched = cell
            # Dark text on the pale middle of the scale, light on its ends.
            dark = abs(math.log2(v)) < 0.6 * extent
            text = f"{v:.2f}x"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black" if dark else "white",
                    fontweight="bold" if not dark else "normal")
            if not matched:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="black", linewidth=0))
                ax.plot([j - 0.3, j + 0.3], [i, i], color="black", linewidth=1.2)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    ticks = [t for t in range(-int(extent), int(extent) + 1)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{2.0 ** t:g}x" for t in ticks])
    bar.set_label("speedup (log scale, parity in the middle)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    blank = sum(1 for s in spps for sc in scenes if (sc, s) not in cells)
    wrong = sum(1 for c in cells.values() if not c[1])
    print(f"wrote {args.out}: {len(scenes)} scenes x {len(spps)} sample counts, "
          f"{blank} blank, {wrong} not matching pbrt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
