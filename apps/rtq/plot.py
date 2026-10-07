#!/usr/bin/env python3
"""The rtq comparison as three plots, from the results CSV compare.sh's runs
are folded into (one row per measured cell: reference, schedule, mesh, query,
ray set, both rates, the speedup).

    python apps/rtq/plot.py build-rtq2/results/rtq-results.csv -o plots/
    python apps/rtq/plot.py results.csv -o plots/ --tuned      # the tuned schedules

Three figures, the meshes along the x axis in order of triangle count: the
first hit (`nearest`) and the any hit (`any`) with six bars a mesh -- the
three ray sets (primary, ao, diffuse) by hatching, the two references
(Embree, FCPW) by colour -- and the closest point with four, the near and
volume batches by hatching. Every bar is bonsai's rate over the reference's
on that cell; the line at 1 is the reference. lte-orb is left out by default
(--exclude names the meshes left out): FCPW's triangle test finds no hit at
all on that small-unit mesh, so its ao, diffuse and point batches are empty
and the mesh says nothing about the comparison.

The eight-wide trees only: Embree's BVH8 (`embree`, or `tuned` with --tuned)
and FCPW's MBVH8 with sixteen-lane leaves (`fcpw8w16`, or `fcpw8w16t`);
--embree and --fcpw name other schedules. The colours are the paper's
(graphs.py), the Okabe-Ito colour-blind palette: sky blue for Embree, orange
for FCPW.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COLOUR = {"embree": "#56B4E9", "fcpw": "#E69F00"}  # sky blue, orange (CUD)
HATCH = {"primary": "", "ao": "///", "diffuse": "xxx", "near": "", "volume": "///"}
LABEL = {"primary": "primary rays", "ao": "ambient-occlusion rays",
         "diffuse": "diffuse rays", "near": "points near surface",
         "volume": "points in volume"}
FIGURES = [
    ("firsthit", "nearest", ["primary", "ao", "diffuse"], "First hit"),
    ("anyhit", "any", ["primary", "ao", "diffuse"], "Any hit"),
    ("closestpoint", "closest", ["near", "volume"], "Closest point"),
]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meshes_by_triangles(rows, exclude):
    tris = {}
    for r in rows:
        if r["mesh"] not in exclude:
            tris[r["mesh"]] = int(r["triangles"])
    return sorted(tris, key=lambda m: tris[m]), tris


def cells(rows, reference, schedule, query):
    out = {}
    for r in rows:
        if r["reference"] == reference and r["schedule"] == schedule and r["query"] == query:
            out[(r["mesh"], r["rays"])] = float(r["speedup"])
    return out


def draw(rows, schedules, out_dir, suffix, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    refs = [("embree", schedules["embree"]), ("fcpw", schedules["fcpw"])]
    for name, query, sets, title in FIGURES:
        data = {ref: cells(rows, ref, sched, query) for ref, sched in refs}
        if not any(data.values()):
            print(f"{name}: no rows for {schedules}; skipped", file=sys.stderr)
            continue
        n = len(sets) * len(refs)
        width = 0.8 / n
        fig, ax = plt.subplots(figsize=(13, 3.4))
        top = 1.0
        for i, mesh in enumerate(meshes):
            for j, rays in enumerate(sets):
                for k, (ref, _) in enumerate(refs):
                    v = data[ref].get((mesh, rays))
                    if v is None:
                        continue
                    x = i - 0.4 + width * (j * len(refs) + k + 0.5)
                    ax.bar(x, v, width, color=COLOUR[ref], hatch=HATCH[rays],
                           edgecolor="black", linewidth=0.5)
                    top = max(top, v)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(meshes)))
        ax.set_xticklabels([f"{m}\n{tris[m] / 1e6:.2g}M" if tris[m] >= 1e6 else f"{m}\n{tris[m] // 1000}k"
                            for m in meshes], fontsize=8)
        ax.set_xlim(-0.6, len(meshes) - 0.4)
        ax.set_ylim(0, top * 1.08)
        ax.set_ylabel("Speedup over reference")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor=COLOUR[ref], edgecolor="black", linewidth=0.5,
                         label="Embree" if ref == "embree" else "FCPW")
                   for ref, _ in refs]
        handles += [Patch(facecolor="white", edgecolor="black", linewidth=0.5,
                          hatch=HATCH[s], label=LABEL[s]) for s in sets]
        # The legend above the axes, clear of the tallest bar; no title (the
        # figure's caption says which query it is, and which schedules).
        ax.legend(handles=handles, fontsize=8, ncol=len(handles), loc="lower left",
                  frameon=False, bbox_to_anchor=(0, 1.0), borderaxespad=0)
        fig.tight_layout()
        for fmt in formats:
            path = os.path.join(out_dir, f"rtq-{name}{suffix}.{fmt}")
            fig.savefig(path, dpi=160)
            print(path)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figures")
    p.add_argument("--tuned", action="store_true",
                   help="the tuned schedules (tuned, fcpw8w16t) instead of the matching ones")
    p.add_argument("--embree", help="Embree's schedule to plot (default embree, or tuned)")
    p.add_argument("--fcpw", help="FCPW's schedule to plot (default fcpw8w16, or fcpw8w16t)")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="lte-orb",
                   help="comma-separated meshes left out (default lte-orb, which FCPW finds no hit on)")
    a = p.parse_args()
    schedules = {"embree": a.embree or ("tuned" if a.tuned else "embree"),
                 "fcpw": a.fcpw or ("fcpw8w16t" if a.tuned else "fcpw8w16")}
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    draw(load(a.csv), schedules, a.out, "-tuned" if a.tuned else "", a.format.split(","), exclude)


if __name__ == "__main__":
    main()
