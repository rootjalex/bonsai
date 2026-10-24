#!/usr/bin/env python3
"""The wosx comparison as plots, from the results CSV benchmark.sh folds
its logs into (one row per measured cell: schedule, mesh, both rates in
steps per second, the speedup).

    python apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots/
    python apps/wosx/plot.py apps/wosx/results/wosx-gpu-results.csv -o apps/wosx/plots/ --gpu

Two CPU figures, one per tree, the meshes along the x axis in order of
triangle count and three bars a mesh; a bar is bonsai's rate over WoSX's
on that cell, and the line at 1 is WoSX:

  wosx-cpu-fcpw4w16-speedup  FCPW's four-wide tree, both sides vectorized
                             (`fcpw4w16*`)
  wosx-cpu-fcpwb-speedup     WoSX's default tree, FCPW's scalar binary BVH,
                             both sides as WoSX ships (`fcpwb*`)

The three bars, on either tree: one thread against one thread (`fcpwb`,
`fcpw4w16`); the threads against the threads (`-threads`: WoSX's
tbb::parallel_for over the points against the program's parfor bound to the
same pinned threads, the same TBB); and sixteen persistent workers against
those same threads (`-persistent`: the workers claim points from a counter;
WoSX has no persistent CPU solver, so its side is the threaded solve again).
Walk on spheres has one schedule per thread count and no parameter the
speedup is swept over: a step is one distance query, and the epsilon shell
and the walk count set how many steps a point takes, not what a step costs
(benchmark.sh fixes them at WoSX's demo defaults). The landscape is left out
by default (--exclude names the meshes left out): its normalized box is a
thousandth thick, nearly every walk escapes on its first step, and its bar
measures the cost of an escaping walk, not the traversal.

--gpu draws the GPU figure, wosx-gpu-speedup (benchmark.sh --gpu): every
bar is a GPU schedule of the program over the one reference, WoSX's GPU
solver -- its persistent-thread kernel on FCPW's binary tree, 131072
resident threads each claiming one walk at a time from a counter, a warp's
consecutive claims neighbouring points. The matching schedule (`fcpwb-gpu`)
does the same; the variants each depart from it in one thing
(--gpu-schedules names them): no persistent claim (`-flat`, one launch, a
thread per walk), a warp's claims one point's walks (`-walks`), a point a
claim with its walks in sequence (`-points`), the matching kernel over the
layout that stores the children's boxes in the parent's row (`fcpwb2-gpu`)
and over FCPW's four-wide tree (`fcpw4w16-gpu`), which FCPW cannot put on a
GPU.

The colours are apps/pbrt's figures' (the user's choice, 10-24):
ColorBrewer's nine-class YlGnBu ramp, a colour a schedule, no hatches, the
edge and the line at 1 as there. No titles: the file's name says which tree
and which side, the legend what each bar compares.
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EDGE = "#333333"  # every bar's thin edge, as apps/pbrt/plot_cpu_wavefront.py
# ColorBrewer YlGnBu, nine classes, lightest first (apps/pbrt's palette).
YLGNBU = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
          "#1D91C0", "#225EA8", "#253494", "#081D58"]

# The CPU figures' bars, on either tree: (suffix of the tree's schedule,
# legend label, colour), in drawing order. Three bars take classes 1, 4 and
# 7 of the ramp, far enough apart to tell at a glance.
CPU_BARS = [
    ("", "single-threaded", YLGNBU[1]),
    ("-threads", "parallel", YLGNBU[4]),
    ("-persistent", "persistent", YLGNBU[7]),
]
# (schedule base, figure name): the two trees.
CPU_TREES = [
    ("fcpw4w16", "wosx-cpu-fcpw4w16-speedup"),
    ("fcpwb", "wosx-cpu-fcpwb-speedup"),
]
CPU_REFERENCE = "WoSX"
# The GPU figure's bars, every one over WoSX's persistent-thread kernel:
# (schedule, legend label, colour), the labels apps/pbrt's figures' kind --
# the schedule's shape in a word or two; README.md's schedule list says
# what each is. Schedules only: the other layout and the other tree
# (fcpwb2-gpu, fcpw4w16-gpu) are a sweep of their own, not this figure's.
GPU_BARS = [
    ("fcpwb-gpu", "persistent", YLGNBU[1]),
    ("fcpwb-gpu-parallel", "parallel", YLGNBU[2]),
    ("fcpwb-gpu-reorder", "persistent-reorder", YLGNBU[3]),
    ("fcpwb-gpu-point", "persistent-point", YLGNBU[4]),
    ("fcpwb-gpu-256k", "persistent-256k", YLGNBU[5]),
    ("fcpwb-gpu-reorder-256k", "persistent-reorder-256k", YLGNBU[6]),
    ("fcpwb-gpu-parallel-reorder-blk64", "parallel-reorder-blk64", YLGNBU[7]),
]
GPU_REFERENCE = "WoSX's persistent kernel"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meshes_by_triangles(rows, exclude):
    tris = {}
    for r in rows:
        if r["mesh"] not in exclude:
            tris[r["mesh"]] = int(r["triangles"])
    return sorted(tris, key=lambda m: tris[m]), tris


def cells(rows, schedule):
    return {r["mesh"]: float(r["speedup"]) for r in rows if r["schedule"] == schedule}


def geomeans(rows, schedules, exclude):
    """The geometric mean of each schedule's bars over the plotted meshes, the
    least and the greatest named (apps/rtq's rule: every geomean a win, no
    cell more than 3% under the reference; the cells under 0.97 are named)."""
    meshes, _ = meshes_by_triangles(rows, exclude)
    print(f"geomeans, bonsai over WoSX, over {len(meshes)} meshes ({', '.join(meshes)}):")
    print(f"  {'schedule':<20} {'geomean':>8} {'min':>6} {'max':>6}  below 0.97")
    for schedule in schedules:
        data = cells(rows, schedule)
        values = {m: data[m] for m in meshes if m in data}
        if not values:
            continue
        g = math.exp(sum(math.log(v) for v in values.values()) / len(values))
        low = " ".join(f"{m} {v:.2f}" for m, v in values.items() if v < 0.97)
        print(f"  {schedule:<20} {g:8.3f} {min(values.values()):6.2f} {max(values.values()):6.2f}  {low}")


def tick_labels(meshes, tris):
    return [f"{m}\n{tris[m] / 1e6:.2g}M" if tris[m] >= 1e6
            else f"{m}\n{tris[m] // 1000}k" if tris[m] >= 1000
            else f"{m}\n{tris[m]}" for m in meshes]


def figure(rows, meshes, tris, bars, reference, ylabel):
    """One figure: a bar per (schedule, label, colour) of `bars` per mesh, the
    dashed line at 1 named `reference` at its right end, the legend above the
    axes. Returns the figure and the schedules drawn, or (None, [])."""
    data = {s: cells(rows, s) for s, _, _ in bars}
    drawn = [(s, label, colour) for s, label, colour in bars if data[s]]
    if not drawn:
        return None, []
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(13, 3.4))
    width = 0.8 / len(drawn)
    top = 1.0
    for i, mesh in enumerate(meshes):
        for k, (s, _, colour) in enumerate(drawn):
            v = data[s].get(mesh)
            if v is None:
                continue
            x = i - 0.4 + width * (k + 0.5)
            ax.bar(x, v, width, color=colour, edgecolor=EDGE, linewidth=0.4, zorder=2)
            top = max(top, v)
    right = len(meshes) - 0.4
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=1.2, zorder=3)
    ax.text(right + 0.05, 1.0, reference, ha="left", va="center", fontsize=6.5)
    ax.set_xticks(range(len(meshes)))
    ax.set_xticklabels(tick_labels(meshes, tris), fontsize=8)
    ax.set_xlim(-0.6, right)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.4, label=label)
               for _, label, colour in drawn]
    ncol = 3 if len(handles) > 3 else len(handles)
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=2.2, handleheight=1.2,
              borderaxespad=0.0, columnspacing=1.2)
    return fig, [s for s, _, _ in drawn]


def save(fig, out_dir, name, formats):
    fig.tight_layout()
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(path)
    plt.close(fig)


def draw_cpu(rows, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    for base, name in CPU_TREES:
        bars = [(base + suffix, label, colour) for suffix, label, colour in CPU_BARS]
        fig, drawn = figure(rows, meshes, tris, bars, CPU_REFERENCE, "Speedup over WoSX")
        if fig is None:
            print(f"{name}: no rows for {base}'s schedules; skipped", file=sys.stderr)
            continue
        save(fig, out_dir, name, formats)


def draw_gpu(rows, schedules, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    known = {s: (label, colour) for s, label, colour in GPU_BARS}
    bars = [(s, *known.get(s, (s, YLGNBU[8]))) for s in schedules]
    fig, drawn = figure(rows, meshes, tris, bars, GPU_REFERENCE, "Speedup over WoSX (GPU)")
    if fig is None:
        print(f"no rows for {schedules}; nothing drawn", file=sys.stderr)
        return
    save(fig, out_dir, "wosx-gpu-speedup", formats)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figures")
    p.add_argument("--gpu", action="store_true",
                   help="draw the GPU figure: the GPU schedules over WoSX's GPU solver")
    p.add_argument("--gpu-schedules", default=",".join(s for s, _, _ in GPU_BARS),
                   help="the GPU schedules, in the order of their bars (default: the matching "
                        "schedule and its variants, as benchmark.sh --gpu measures them)")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="landscape",
                   help="comma-separated meshes left out (default landscape, whose walks escape "
                        "on their first step)")
    a = p.parse_args()
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    if a.gpu:
        schedules = [s for s in a.gpu_schedules.split(",") if s]
        draw_gpu(rows, schedules, a.out, a.format.split(","), exclude)
    else:
        schedules = [base + suffix for base, _ in CPU_TREES for suffix, _, _ in CPU_BARS]
        draw_cpu(rows, a.out, a.format.split(","), exclude)
    geomeans(rows, schedules, exclude)


if __name__ == "__main__":
    main()
