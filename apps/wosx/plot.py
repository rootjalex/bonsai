#!/usr/bin/env python3
"""The wosx comparison as plots, from the results CSV benchmark.sh folds
its logs into (one row per measured cell: schedule, mesh, both rates in
steps per second, the speedup).

    python apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots/
    python apps/wosx/plot.py apps/wosx/results/wosx-gpu-results.csv -o apps/wosx/plots/ --gpu

The CPU figure has two panels, one per tree, the meshes along the x axis
in order of triangle count and three bars a mesh: sequential (one thread
against one thread), parallel (the threads against the threads) and
persistent (sixteen persistent workers against the threads). The upper
panel is FCPW's four-wide tree, both sides vectorized (`fcpw4w16*`); the
lower is WoSX's default tree, FCPW's scalar binary BVH, both sides as WoSX
ships (`fcpwb*`). A bar is bonsai's rate over WoSX's on that cell; the
line at 1 is WoSX. Walk on spheres has one schedule per thread count and
no parameter the speedup is swept over: a step is one distance query, and
the epsilon shell and the walk count set how many steps a point takes, not
what a step costs (benchmark.sh fixes them at WoSX's demo defaults). The
landscape is left out by default (--exclude names the meshes left out):
its normalized box is a thousandth thick, nearly every walk escapes on its
first step, and its bar measures the cost of an escaping walk, not the
traversal. The colour is apps/rtq's for FCPW (the Okabe-Ito orange), the
trees being FCPW's.

--gpu draws the GPU table (benchmark.sh --gpu): every bar is a GPU
schedule of the program over WoSX's GPU solver, its persistent-thread
kernel on FCPW's binary tree -- the matching schedule (`fcpwb-gpu`, solid)
and the variants that depart from it (--gpu-schedules names them), each a
bar.
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

COLOUR = "#E69F00"  # FCPW's orange in apps/rtq/plot.py (CUD)
HATCHES = ["", "///", "xxx", "...", "\\\\\\", "++", "oo"]
# The CPU table's bars, on either tree. sequential: bonsai on one thread
# over WoSX on one thread; parallel: bonsai on the threads over WoSX on the
# threads; persistent: bonsai's persistent workers over WoSX on the threads.
CPU_KINDS = ["sequential", "parallel", "persistent"]
CPU_TREES = [
    ("fcpw4w16", "FCPW's four-wide tree (both sides vectorized)"),
    ("fcpwb", "WoSX's default tree: FCPW's binary BVH (both sides as WoSX ships)"),
]
CPU_SUFFIX = {"sequential": "", "parallel": "-threads", "persistent": "-persistent"}
# The GPU table, every bar over WoSX's persistent-thread kernel: the
# schedule matching that kernel on FCPW's bytes (a walk a claim, the points
# fastest), the same walks as one flat launch, the walks fastest, a point a
# claim, the matching kernel over the layout that stores the children's
# boxes in the parent, and over FCPW's four-wide tree.
GPU_LABEL = {"fcpwb-gpu": "matching", "fcpwb-gpu-flat": "flat launch",
             "fcpwb-gpu-walks": "walks fastest", "fcpwb-gpu-points": "a point a claim",
             "fcpwb2-gpu": "children's boxes in the parent", "fcpw4w16-gpu": "four-wide tree"}


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


def bars(ax, rows, meshes, schedules, labels, ylabel, title=None):
    """One panel: the schedules' bars per mesh, a hatch per schedule, the
    line at 1, the legend above the axes. Returns the drawn schedules."""
    data = {s: cells(rows, s) for s in schedules}
    drawn = [s for s in schedules if data[s]]
    if not drawn:
        return []
    hatch = {s: HATCHES[k % len(HATCHES)] for k, s in enumerate(drawn)}
    width = 0.8 / len(drawn)
    top = 1.0
    for i, mesh in enumerate(meshes):
        for k, s in enumerate(drawn):
            v = data[s].get(mesh)
            if v is None:
                continue
            x = i - 0.4 + width * (k + 0.5)
            ax.bar(x, v, width, color=COLOUR, hatch=hatch[s], edgecolor="black", linewidth=0.5)
            top = max(top, v)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(meshes)))
    ax.set_xlim(-0.6, len(meshes) - 0.4)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLOUR, edgecolor="black", linewidth=0.5, hatch=hatch[s],
                     label=labels.get(s, s)) for s in drawn]
    if title:
        ax.text(0, 1.02, title, transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    # The legend above the axes, clear of the tallest bar, right of the
    # panel's title; no figure title (the caption says which tree and which
    # schedules).
    ax.legend(handles=handles, fontsize=8, ncol=len(handles), loc="lower right",
              frameon=False, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    return drawn


def save(fig, out_dir, name, formats):
    fig.tight_layout()
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


def draw_cpu(rows, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    trees = [(base, title) for base, title in CPU_TREES
             if any(cells(rows, base + CPU_SUFFIX[k]) for k in CPU_KINDS)]
    if not trees:
        print("no rows for either tree's schedules; nothing drawn", file=sys.stderr)
        return
    fig, axes = plt.subplots(len(trees), 1, figsize=(13, 3.2 * len(trees)), sharex=True,
                             squeeze=False)
    for ax, (base, title) in zip(axes[:, 0], trees):
        schedules = [base + CPU_SUFFIX[k] for k in CPU_KINDS]
        labels = {base + CPU_SUFFIX[k]: k for k in CPU_KINDS}
        bars(ax, rows, meshes, schedules, labels, "Speedup over WoSX", title)
    axes[-1, 0].set_xticklabels(tick_labels(meshes, tris), fontsize=8)
    save(fig, out_dir, "wosx-speedup", formats)


def draw_gpu(rows, schedules, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    fig, ax = plt.subplots(figsize=(13, 3.4))
    drawn = bars(ax, rows, meshes, schedules, GPU_LABEL, "Speedup over WoSX (GPU)")
    if not drawn:
        print(f"no rows for {schedules}; nothing drawn", file=sys.stderr)
        plt.close(fig)
        return
    ax.set_xticklabels(tick_labels(meshes, tris), fontsize=8)
    save(fig, out_dir, "wosx-gpu-speedup", formats)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figure")
    p.add_argument("--gpu", action="store_true",
                   help="draw the GPU table: the GPU schedules over WoSX's GPU solver")
    p.add_argument("--gpu-schedules", default=",".join(GPU_LABEL),
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
        schedules = [base + CPU_SUFFIX[k] for base, _ in CPU_TREES for k in CPU_KINDS]
        draw_cpu(rows, a.out, a.format.split(","), exclude)
    geomeans(rows, schedules, exclude)


if __name__ == "__main__":
    main()
