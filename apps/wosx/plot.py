#!/usr/bin/env python3
"""The wosx comparison as one plot, from the results CSV benchmark.sh folds
its logs into (one row per measured cell: schedule, mesh, both rates in
steps per second, the speedup).

    python apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots/

One figure, the meshes along the x axis in order of triangle count, two bars
a mesh: one thread against one thread (`fcpw4w16`, solid) and the threads
against the threads (`fcpw4w16-threads`, hatched), the same tree and the
same question on both sides of every bar. A bar is bonsai's rate over
WoSX's on that cell; the line at 1 is WoSX. Walk on spheres has one
schedule per thread count and no parameter the speedup is swept over: a
step is one distance query, and the epsilon shell and the walk count set
how many steps a point takes, not what a step costs (benchmark.sh fixes
them at WoSX's demo defaults). The landscape is left out by default
(--exclude names the meshes left out): its normalized box is a thousandth
thick, nearly every walk escapes on its first step, and its bar measures the
cost of an escaping walk, not the traversal. The colour is apps/rtq's for
FCPW (the Okabe-Ito orange), the tree being FCPW's.
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
HATCH = {"single": "", "threads": "///", "persistent": "xxx"}
LABEL = {"single": "one thread each",
         "threads": "the performance cores' threads each",
         "persistent": "the same, the points claimed by 16 persistent workers"}


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
    print(f"  {'schedule':<18} {'geomean':>8} {'min':>6} {'max':>6}  below 0.97")
    for kind, schedule in schedules.items():
        data = cells(rows, schedule)
        values = {m: data[m] for m in meshes if m in data}
        if not values:
            continue
        g = math.exp(sum(math.log(v) for v in values.values()) / len(values))
        low = " ".join(f"{m} {v:.2f}" for m, v in values.items() if v < 0.97)
        print(f"  {schedule:<18} {g:8.3f} {min(values.values()):6.2f} {max(values.values()):6.2f}  {low}")


def draw(rows, schedules, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    data = {kind: cells(rows, schedule) for kind, schedule in schedules.items()}
    if not any(data.values()):
        print(f"no rows for {schedules}; nothing drawn", file=sys.stderr)
        return
    # The schedules with rows: the persistent one is drawn when measured.
    kinds = [kind for kind in schedules if data[kind]]
    width = 0.8 / len(kinds)
    fig, ax = plt.subplots(figsize=(13, 3.4))
    top = 1.0
    for i, mesh in enumerate(meshes):
        for k, kind in enumerate(kinds):
            v = data[kind].get(mesh)
            if v is None:
                continue
            x = i - 0.4 + width * (k + 0.5)
            ax.bar(x, v, width, color=COLOUR, hatch=HATCH[kind], edgecolor="black", linewidth=0.5)
            top = max(top, v)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(meshes)))
    ax.set_xticklabels([f"{m}\n{tris[m] / 1e6:.2g}M" if tris[m] >= 1e6
                        else f"{m}\n{tris[m] // 1000}k" if tris[m] >= 1000
                        else f"{m}\n{tris[m]}" for m in meshes], fontsize=8)
    ax.set_xlim(-0.6, len(meshes) - 0.4)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("Speedup over WoSX")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLOUR, edgecolor="black", linewidth=0.5, hatch=HATCH[kind],
                     label=LABEL[kind]) for kind in kinds if data[kind]]
    # The legend above the axes, clear of the tallest bar; no title (the
    # figure's caption says which tree and which schedules).
    ax.legend(handles=handles, fontsize=8, ncol=len(handles), loc="lower left",
              frameon=False, bbox_to_anchor=(0, 1.0), borderaxespad=0)
    fig.tight_layout()
    for fmt in formats:
        path = os.path.join(out_dir, f"wosx-speedup.{fmt}")
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figure")
    p.add_argument("--single", default="fcpw4w16",
                   help="the one-thread schedule (default fcpw4w16)")
    p.add_argument("--threads", default="fcpw4w16-threads",
                   help="the threaded schedule (default fcpw4w16-threads)")
    p.add_argument("--persistent", default="fcpw4w16-persistent",
                   help="the threaded schedule with persistent workers, drawn when its rows "
                        "exist (default fcpw4w16-persistent)")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="landscape",
                   help="comma-separated meshes left out (default landscape, whose walks escape "
                        "on their first step)")
    a = p.parse_args()
    schedules = {"single": a.single, "threads": a.threads, "persistent": a.persistent}
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    draw(rows, schedules, a.out, a.format.split(","), exclude)
    geomeans(rows, schedules, exclude)


if __name__ == "__main__":
    main()
