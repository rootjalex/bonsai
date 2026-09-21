#!/usr/bin/env python3
"""Draws the speedups of several scenes' render_matrix runs on one figure, so
that the trend across scenes, depths and sample counts can be read at once.

    python3 eval/summary.py killeroo-simple book pavilion-day
    python3 eval/summary.py killeroo-simple book pavilion-day --depths 5

Each scene is a name under eval/out/ that render_matrix.py has filled -- its
results.json is read, nothing is rendered -- given in the order the rows are
wanted, which is meant to be increasing complexity. One row per scene, one
panel per depth, and in each panel the speedup over pbrt against the sample
count, one line per schedule, with pbrt itself the line at 1x. Written as
eval/plots/summary-speedup.pdf and .png (summary-d5-speedup for a subset of
the depths), or --name for another stem.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_matrix  # noqa: E402  the style and the labels, so there is one set

ROOT = render_matrix.ROOT

# One colour and marker per schedule, the same in every panel and every
# figure drawn from these results.
STYLE = {
    "scalar": ("#7f7f7f", "o"),
    "perlane": ("#1f77b4", "s"),
    "packet": ("#d62728", "^"),
    "wavefront-perlane": ("#2ca02c", "D"),
    "wavefront": ("#9467bd", "v"),
}


def load(out, scene):
    path = os.path.join(out, scene, "results.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no results for {scene} at {path}: run "
                         f"eval/render_matrix.py on the scene first")
    with open(path) as f:
        return json.load(f)


def axes_of(results, depths, spps, schedules):
    """The depths, sample counts and schedules present in every asked-for
    cell, in the order given; a cell missing from the results is an error,
    since a plot with a hole would read as a measurement."""
    cells = results["cells"]
    for d in depths:
        for p in spps:
            tag = f"d{d}-s{p}"
            if tag not in cells:
                raise SystemExit(f"depth {d} at {p} spp was not rendered")
            for s in schedules:
                if s not in cells[tag]:
                    raise SystemExit(f"{s} was not rendered at depth {d}, {p} spp")


def main(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+",
                        help="scene names under --out, rows top to bottom")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--spps", type=int, nargs="+",
                        default=render_matrix.DEFAULT_SPPS)
    parser.add_argument("--schedules", nargs="+",
                        default=render_matrix.DEFAULT_SCHEDULES)
    parser.add_argument("--out", default=os.path.join(ROOT, "eval", "out"))
    parser.add_argument("--plots", default=os.path.join(ROOT, "eval", "plots"))
    parser.add_argument("--name", default=None,
                        help="the figure's file stem (default summary, or "
                             "summary-d<depths> for a subset of the depths)")
    parser.add_argument("--width", type=float, default=None,
                        help="figure width in inches (default 2 per depth)")
    args = parser.parse_args(argv[1:])

    results = {scene: load(args.out, scene) for scene in args.scenes}
    for scene, r in results.items():
        axes_of(r, args.depths, args.spps, args.schedules)

    plt = render_matrix.plot_style()
    rows, cols = len(args.scenes), len(args.depths)
    width = args.width or min(10.0, 2.0 * cols + 0.9)
    fig, axes = plt.subplots(rows, cols, figsize=(width, 1.55 * rows + 0.7),
                             squeeze=False, sharex=True, sharey="row")
    x = np.array(args.spps, dtype=float)
    for i, scene in enumerate(args.scenes):
        cells = results[scene]["cells"]
        for j, depth in enumerate(args.depths):
            ax = axes[i][j]
            ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
            for schedule in args.schedules:
                y = [cells[f"d{depth}-s{p}"][schedule]["speedup"] for p in args.spps]
                failed = [bool(cells[f"d{depth}-s{p}"][schedule]["failed"])
                          for p in args.spps]
                colour, marker = STYLE.get(schedule, ("black", "x"))
                ax.plot(x, y, color=colour, marker=marker, markersize=2.5,
                        linewidth=0.8,
                        label=render_matrix.RENDERER_LABELS.get(schedule, schedule))
                # A cell whose image failed the check against pbrt is marked,
                # as the heatmaps mark it.
                for xv, yv, bad in zip(x, y, failed):
                    if bad:
                        ax.annotate("†", (xv, yv), fontsize=7, ha="center",
                                    va="bottom")
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(p) for p in args.spps])
            ax.tick_params(length=2)
            ax.grid(True, linewidth=0.3, alpha=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i == 0:
                ax.set_title(f"depth {depth}", pad=4)
            if i == rows - 1:
                ax.set_xlabel("samples per pixel")
            if j == 0:
                ax.set_ylabel(f"{scene}\nspeedup over pbrt")
            ax.set_ylim(bottom=0.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(args.schedules),
               frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # What the numbers are the best of, as the heatmaps say it: once when
    # every scene was run the same way, per scene otherwise.
    captions = {scene: render_matrix.repeats_caption(args, results[scene])
                for scene in args.scenes}
    if len(set(captions.values())) == 1:
        note = next(iter(captions.values()))
    else:
        note = "; ".join(f"{scene}: {c}" for scene, c in captions.items())
    fig.text(0.0, -0.01, note, fontsize=6, ha="left", va="top",
             transform=fig.transFigure)

    stem = args.name
    if stem is None:
        stem = "summary"
        if sorted(args.depths) != [1, 2, 3, 4, 5]:
            stem += "-d" + "".join(str(d) for d in args.depths)
    os.makedirs(args.plots, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.plots}/{stem}-speedup.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.plots}/{stem}-speedup.pdf and .png")

    # The same numbers as text, one line per (scene, depth, spp), the
    # schedules across.
    head = "scene\tdepth\tspp\tpbrt_s\t" + "\t".join(args.schedules)
    lines = [head]
    for scene in args.scenes:
        cells = results[scene]["cells"]
        for depth in args.depths:
            for p in args.spps:
                cell = cells[f"d{depth}-s{p}"]
                lines.append("\t".join(
                    [scene, str(depth), str(p), f"{cell['pbrt_seconds']:.3f}"] +
                    [f"{cell[s]['speedup']:.2f}" for s in args.schedules]))
    with open(f"{args.plots}/{stem}-speedup.tsv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
