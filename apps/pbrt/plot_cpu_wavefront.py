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
below -- pbrt's own scalar integrator first, then this renderer's schedules
from the path-per-sample `scalar` through the band wavefront (scalar and
vectorized drains, then the vectorized one with its rays reordered) and the
per-thread wavefront (scalar and vectorized drains, then the vectorized one
pushed sample-major). `--sides` names a subset (comma-separated side names)
and keeps that order, so the five-bar figure of the first study is
`--sides pbrt-volpath,scalar,wavefront-frame,wavefront-frame-packet,wavefront-frame-packet-reorder`.

Colour tells the renderer apart and hatching tells the schedules apart:
pbrt's bar is grey, every Bonsai bar is one blue, and each Bonsai schedule
has its own hatch, so the figure reads the same in print, in greyscale and
under the common colour deficiencies (the two colours are from the
Okabe-Ito palette). A bar whose image did not match pbrt's has a thick
vermilion edge; the legend says so when any does. Every bar carries its
speedup (`--no-values` drops them). `--width` and `--height` are the figure
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
REFERENCE_LABEL = "pbrt wavefront"
PBRT_COLOUR = "#7F7F7F"      # a mid grey
BONSAI_COLOUR = "#0072B2"    # Okabe-Ito blue
BAD_EDGE = "#D55E00"         # Okabe-Ito vermilion, for an image that did not match

# (side in the CSV, label in the legend, colour, hatch): the drawing order.
# The names follow one pattern: the renderer, then "per-thread" when the
# queues are a thread's, then the drains (scalar or vectorized), then the
# structure (wavefront, or nothing for a path per sample), then a variant.
SIDES = [
    ("pbrt-volpath", "pbrt scalar", PBRT_COLOUR, ""),
    ("scalar", "Bonsai scalar", BONSAI_COLOUR, ""),
    ("wavefront-frame", "Bonsai scalar wavefront", BONSAI_COLOUR, "////"),
    ("wavefront-frame-packet", "Bonsai vectorized wavefront", BONSAI_COLOUR, "xxxx"),
    ("wavefront-frame-packet-reorder", "Bonsai vectorized wavefront, rays reordered", BONSAI_COLOUR, "...."),
    ("wavefront-thread", "Bonsai per-thread scalar wavefront", BONSAI_COLOUR, "\\\\\\\\"),
    ("wavefront-thread-packet", "Bonsai per-thread vectorized wavefront", BONSAI_COLOUR, "++++"),
    ("wavefront-thread-packet-reorder", "Bonsai per-thread vectorized wavefront, sample-major", BONSAI_COLOUR, "oooo"),
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
                         "legend.fontsize": 7, "hatch.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    bar = 1.0                     # one bar's width in x units
    gap_spp = 1.5                 # between the sample counts of a scene
    gap_scene = 4.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    any_bad = False
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            ref = reference(scene, spp)
            group_start = x
            for side, label, colour, hatch in sides:
                c = cell(scene, spp, side)
                if c is not None and ref is not None:
                    speedup = ref / c["seconds"]
                    ok = c["verdict"].startswith("ok")
                    any_bad = any_bad or not ok
                    ax.bar(x, speedup, bar * 0.9, color=colour, hatch=hatch,
                           edgecolor="white" if ok else BAD_EDGE,
                           linewidth=0.5 if ok else 1.4, zorder=2)
                    if not args.no_values:
                        ax.text(x, speedup, f"{speedup:.2f}", ha="center", va="bottom",
                                fontsize=5.5, rotation=90, zorder=4)
                x += bar
            spp_centres.append((group_start + x - bar) / 2)
            spp_labels.append(f"{spp} spp")
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
    for centre, scene in zip(scene_centres, scenes):
        ax.text(centre, -0.16, scene, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylabel(f"speedup over {REFERENCE_LABEL} (CPU)")
    if args.log:
        ax.set_yscale("log")
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.12)       # room for the values
    ax.set_xlim(-bar, right + 0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    handles = [Patch(facecolor=colour, hatch=hatch, edgecolor="white", linewidth=0.5,
                     label=label) for _, label, colour, hatch in sides]
    if any_bad:
        handles.append(Patch(facecolor="white", edgecolor=BAD_EDGE, linewidth=1.4,
                             label="image differs from pbrt's"))
    ncol = 2 if len(handles) > 4 else len(handles)
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
