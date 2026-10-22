#!/usr/bin/env python3
"""The CPU wavefront study as one figure: every schedule's speedup over
pbrt's CPU wavefront (`pbrt --wavefront` without `--gpu`), a group of bars
per scene, a subgroup per sample count.

    python3 apps/pbrt/plot_cpu_wavefront.py apps/pbrt/results/cpu-wavefront.csv \
        -o apps/pbrt/plots/cpu-wavefront-speedups.pdf

The CSV has one row per measured cell: scene, spp, side, render seconds
(the least of three, compare.sh REPEATS=3, pbrt's own render time on its
side), the image verdict (`ok` when compare.sh matched pbrt's image) and
the date. The side `pbrt-wavefront` is the reference and draws no bar; the
line at 1 is it. The other sides are drawn in the order of SIDES below,
pbrt's own volpath first, then our schedules from the path-per-sample
`scalar` through the band wavefront, its vectorized form, the ray-sorted
form, and the per-thread wavefront in its two orders; `--sides` names a
subset (comma-separated side names) and keeps that order, so the five-bar
figure is `--sides pbrt-volpath,scalar,wavefront-frame,wavefront-frame-packet,wavefront-frame-packet-reorder`.
A bar whose image did not match pbrt's is hatched. Every bar carries its
speedup; `--no-values` drops them. The colours are the Okabe-Ito palette
(the one apps/rtq and apps/wosx draw with), safe for the common colour
deficiencies. `--width` and `--height` are the figure size in inches (the
defaults fit a two-column page's full width); `--log` puts the speedup axis
on a log scale. The output format follows the extension (.pdf for the
paper, .png to look at); fonts are embedded as TrueType (fonttype 42).
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

# (side in the CSV, label in the legend, colour): the drawing order.
SIDES = [
    ("pbrt-volpath", "pbrt volpath (CPU, scalar)", "#999999"),
    ("scalar", "Bonsai scalar", "#E69F00"),
    ("wavefront-frame", "Bonsai scalar wavefront", "#56B4E9"),
    ("wavefront-frame-packet", "Bonsai vectorized wavefront", "#009E73"),
    ("wavefront-frame-packet-reorder", "Bonsai vectorized + ray-reorder wavefront", "#F0E442"),
    ("wavefront-thread-packet", "Bonsai per-thread vectorized wavefront", "#0072B2"),
    ("wavefront-thread-packet-reorder", "Bonsai per-thread vectorized wavefront, sample-major", "#CC79A7"),
]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["spp"] = int(r["spp"])
        r["seconds"] = float(r["seconds"])
    return rows


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
    ap.add_argument("--height", type=float, default=3.2)
    ap.add_argument("--log", action="store_true", help="log-scale speedup axis")
    ap.add_argument("--no-values", action="store_true", help="no number on the bars")
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

    def cell(scene, spp, side):
        return next((r for r in rows if r["scene"] == scene and r["spp"] == spp
                     and r["side"] == side), None)

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "hatch.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    bar = 1.0                     # one bar's width in x units
    gap_spp = 1.5                 # between the sample counts of a scene
    gap_scene = 4.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            ref = cell(scene, spp, REFERENCE)
            group_start = x
            for side, label, colour in sides:
                c = cell(scene, spp, side)
                if c is not None and ref is not None:
                    speedup = ref["seconds"] / c["seconds"]
                    ok = c["verdict"].startswith("ok")
                    ax.bar(x, speedup, bar * 0.92, color=colour,
                           hatch=None if ok else "///",
                           edgecolor="black" if not ok else "none", linewidth=0.5)
                    if not args.no_values:
                        ax.text(x, speedup, f"{speedup:.2f}", ha="center", va="bottom",
                                fontsize=5.5, rotation=90)
                x += bar
            spp_centres.append((group_start + x - bar) / 2)
            spp_labels.append(f"{spp} spp")
            if j + 1 < len(spps):
                x += gap_spp
        scene_centres.append((scene_start + x - bar) / 2)
        x += gap_scene

    ax.axhline(1.0, color="black", linewidth=0.8, zorder=0)
    ax.set_xticks(spp_centres)
    ax.set_xticklabels(spp_labels)
    ax.tick_params(axis="x", length=0)
    for centre, scene in zip(scene_centres, scenes):
        ax.text(centre, -0.16, scene, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylabel("speedup over pbrt CPU wavefront")
    if args.log:
        ax.set_yscale("log")
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.12)       # room for the values
    ax.set_xlim(-bar, x - gap_scene)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    handles = [Patch(facecolor=colour, label=label) for _, label, colour in sides]
    if any(not r["verdict"].startswith("ok") for r in rows if r["side"] != REFERENCE):
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                             label="image differs from pbrt's"))
    ncol = 2 if len(handles) > 4 else len(handles)
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0 + 0.12 * ((len(handles) + ncol - 1) // ncol)))

    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
