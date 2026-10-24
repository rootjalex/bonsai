#!/usr/bin/env python3
"""The rtq comparison as three plots, from the results CSV compare.sh's runs
are folded into (one row per measured cell: reference, schedule, mesh, query,
ray set, both rates, the speedup).

    python apps/rtq/plot.py apps/rtq/results/rtq-results.csv -o apps/rtq/plots
    python apps/rtq/plot.py results.csv -o plots/ --tuned      # the tuned schedules

Three figures, the meshes along the x axis in order of triangle count: the
first hit (`nearest`) and the any hit (`any`) with six bars a mesh -- the
three ray sets (primary, ao, diffuse) by hatching, the two references
(Embree, FCPW) by colour -- and the closest point with four, the near and
volume batches by hatching. Every bar is bonsai's rate over the reference's
on that cell; the line at 1 is the reference. lte-orb is left out by default
(--exclude names the meshes left out): FCPW's triangle test finds no hit at
all on that small-unit mesh, so its ao, diffuse and point batches are empty
and the mesh says nothing about the comparison. A fourth figure, the
packets, is drawn when the CSV has rows of the packet schedule (`packet16`;
--packet names another): three bars a mesh, the program's rays in gangs of
sixteen over Embree's rtcIntersect16 (primary and diffuse rays) and
rtcOccluded16 (ambient-occlusion rays), the ray sets by hatching, both sides
over the same BVH8 and the same sixteen consecutive rays a packet.

Each reference on its own default tree: Embree's BVH8 (`embree`) and FCPW's
four-wide Mbvh with sixteen-lane leaves (`fcpw4w16`, FCPW's default branching
and the leaf width its build picks on an AVX-512 machine; the four-wide tree
is the one FCPW's author calls the better tested); --embree and --fcpw name
other matching schedules. With --tuned the bars are the tuning's gain alone:
bonsai's rate under the tuned schedule (`tuned`, `fcpw4w16t`; --embree-tuned, --fcpw-tuned)
over its rate under the matching one, cell by cell, the reference out of it,
so the line at 1 is the matching schedule and the colour says which tree.
The colours are apps/pbrt's figures' (the user's choice, 10-24): two classes
of ColorBrewer's nine-class YlGnBu ramp, far enough apart to tell with the
hatches over them -- the light second for Embree, the fifth for FCPW -- and
the bars' edge and the line at 1 as there.
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
from matplotlib.patches import Patch  # noqa: E402

EDGE = "#333333"  # every bar's thin edge, as apps/pbrt/plot_cpu_wavefront.py
# ColorBrewer YlGnBu, nine classes, lightest first (apps/pbrt's palette).
YLGNBU = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
          "#1D91C0", "#225EA8", "#253494", "#081D58"]
COLOUR = {"embree": YLGNBU[1], "fcpw": YLGNBU[5]}
HATCH = {"primary": "", "ao": "///", "diffuse": "xxx", "near": "", "volume": "///"}
LABEL = {"primary": "primary rays", "ao": "ambient-occlusion rays",
         "diffuse": "diffuse rays", "near": "points near surface",
         "volume": "points in volume"}
# Aila and Laine's three ray sets, each under the query it is for: the first
# hit of primary and diffuse rays, the any hit of ambient-occlusion rays.
FIGURES = [
    ("firsthit", "nearest", ["primary", "diffuse"], "First hit"),
    ("anyhit", "any", ["ao"], "Any hit"),
    ("closestpoint", "closest", ["near", "volume"], "Closest point"),
]
# The packet figure's bars: the same three sets under the same queries, one
# figure, since the packet comparison is one comparison (Embree's
# rtcIntersect<K> and rtcOccluded<K> against the rays in gangs of K).
PACKET_SETS = [("nearest", "primary"), ("nearest", "diffuse"), ("any", "ao")]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meshes_by_triangles(rows, exclude):
    tris = {}
    for r in rows:
        if r["mesh"] not in exclude:
            tris[r["mesh"]] = int(r["triangles"])
    return sorted(tris, key=lambda m: tris[m]), tris


def cells(rows, reference, schedule, query, column="speedup"):
    out = {}
    for r in rows:
        if r["reference"] == reference and r["schedule"] == schedule and r["query"] == query:
            out[(r["mesh"], r["rays"])] = float(r[column])
    return out


def tuned_over_matching(rows, reference, matching, tuned, query):
    """Bonsai's own rate under the tuned schedule over its rate under the
    matching one, cell by cell: the tuning's gain alone, the reference out
    of it."""
    base = cells(rows, reference, matching, query, "bonsai_rate")
    over = cells(rows, reference, tuned, query, "bonsai_rate")
    return {k: over[k] / base[k] for k in over if k in base and base[k] > 0}


def packet_cells(rows, schedule):
    """The packet schedule's cells against Embree's packets, keyed by (mesh,
    rays): the ray sets are distinct across the two queries, so one map
    holds the nearest hit's primary and diffuse cells and the any hit's ao."""
    out = {}
    for query, rays in PACKET_SETS:
        for key, v in cells(rows, "embree", schedule, query).items():
            if key[1] == rays:
                out[key] = v
    return out


def geomeans(rows, schedules, exclude, tuned, packet=None):
    """The geometric mean of each figure's bars over the plotted meshes, per
    reference and ray or point set: the one number per cell kind the
    comparison is judged by (the user's rule of 2026-10-07: every geomean a
    win, no single cell more than 3% under its reference). Printed as a
    table, with the cells under 0.97 named, since a geomean can hide one.
    The packet schedule's cells, when there are any, follow the same way."""
    meshes, _ = meshes_by_triangles(rows, exclude)
    refs = [("embree", schedules["embree"]), ("fcpw", schedules["fcpw"])]
    what = "tuned over matching" if tuned else "bonsai over the reference"
    print(f"geomeans, {what}, over {len(meshes)} meshes "
          f"({', '.join(meshes)}):")
    print(f"  {'reference':<10} {'schedule':<10} {'query':<8} {'rays':<8} "
          f"{'geomean':>8} {'min':>6} {'max':>6}  below 0.97")

    def row(ref, sched, query, rays, data):
        values = {m: data[(m, rays)] for m in meshes if (m, rays) in data}
        if not values:
            return
        g = math.exp(sum(math.log(v) for v in values.values()) / len(values))
        low = " ".join(f"{m} {v:.2f}" for m, v in values.items() if v < 0.97)
        print(f"  {ref:<10} {sched:<10} {query:<8} {rays:<8} "
              f"{g:8.3f} {min(values.values()):6.2f} {max(values.values()):6.2f}"
              f"  {low}")

    for ref, sched in refs:
        for _, query, sets, _ in FIGURES:
            data = (tuned_over_matching(rows, ref, sched, tuned[ref], query) if tuned
                    else cells(rows, ref, sched, query))
            for rays in sets:
                row(ref, sched, query, rays, data)
    if packet and not tuned:
        data = packet_cells(rows, packet)
        for query, rays in PACKET_SETS:
            row("embree", packet, query, rays, data)


def draw_packet(rows, schedule, out_dir, formats, exclude):
    """The packet figure: the program's rays in gangs of K over Embree's
    packets of K, three bars a mesh (primary and diffuse rays under the
    nearest hit, ambient-occlusion rays under the any hit), Embree's colour,
    the sets by hatching; the line at 1 is Embree's packet kernel."""
    data = packet_cells(rows, schedule)
    if not data:
        print(f"packet: no rows for {schedule}; skipped", file=sys.stderr)
        return
    meshes, tris = meshes_by_triangles(rows, exclude)
    # The packet's width is the digits after `packet` in the schedule's name;
    # a suffix (packet16-anyhit, the any hit alone in gangs) is not part of it.
    width_match = re.match(r"packet(\d+)", schedule)
    k = width_match.group(1) if width_match else "K"
    n = len(PACKET_SETS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(13, 3.4))
    top = 1.0
    for i, mesh in enumerate(meshes):
        for j, (_, rays) in enumerate(PACKET_SETS):
            v = data.get((mesh, rays))
            if v is None:
                continue
            x = i - 0.4 + width * (j + 0.5)
            ax.bar(x, v, width, color=COLOUR["embree"], hatch=HATCH[rays],
                   edgecolor=EDGE, linewidth=0.4)
            top = max(top, v)
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=1.2, zorder=3)
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
    handles = [Patch(facecolor=COLOUR["embree"], edgecolor=EDGE, linewidth=0.4,
                     label=f"Embree, packets of {k} (rtcIntersect{k}, rtcOccluded{k})")]
    handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4,
                      hatch=HATCH[rays], label=LABEL[rays]) for _, rays in PACKET_SETS]
    ax.legend(handles=handles, fontsize=8, ncol=len(handles), loc="lower left",
              frameon=False, bbox_to_anchor=(0, 1.0), borderaxespad=0)
    fig.tight_layout()
    for fmt in formats:
        path = os.path.join(out_dir, f"rtq-packet.{fmt}")
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


def draw(rows, schedules, out_dir, suffix, formats, exclude, tuned):
    meshes, tris = meshes_by_triangles(rows, exclude)
    refs = [("embree", schedules["embree"]), ("fcpw", schedules["fcpw"])]
    for name, query, sets, title in FIGURES:
        if tuned:
            data = {ref: tuned_over_matching(rows, ref, schedules[ref], tuned[ref], query)
                    for ref, _ in refs}
        else:
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
                           edgecolor=EDGE, linewidth=0.4)
                    top = max(top, v)
        ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(meshes)))
        ax.set_xticklabels([f"{m}\n{tris[m] / 1e6:.2g}M" if tris[m] >= 1e6 else f"{m}\n{tris[m] // 1000}k"
                            for m in meshes], fontsize=8)
        ax.set_xlim(-0.6, len(meshes) - 0.4)
        ax.set_ylim(0, top * 1.08)
        ax.set_ylabel("Speedup over matching schedule" if tuned else "Speedup over reference")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor=COLOUR[ref], edgecolor=EDGE, linewidth=0.4,
                         label=("Embree" if ref == "embree" else "FCPW") + ("'s tree" if tuned else ""))
                   for ref, _ in refs]
        handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4,
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
                   help="the tuning's gain: bonsai under the tuned schedules over bonsai under "
                        "the matching ones, cell by cell")
    p.add_argument("--embree", default="embree", help="Embree's matching schedule (default embree)")
    p.add_argument("--fcpw", default="fcpw4w16", help="FCPW's matching schedule (default fcpw4w16)")
    p.add_argument("--embree-tuned", default="tuned", help="Embree's tuned schedule (default tuned)")
    p.add_argument("--fcpw-tuned", default="fcpw4w16t", help="FCPW's tuned schedule (default fcpw4w16t)")
    p.add_argument("--packet", default="packet16",
                   help="the packet schedule, against Embree's packets of its width (default packet16)")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="lte-orb",
                   help="comma-separated meshes left out (default lte-orb, which FCPW finds no hit on)")
    a = p.parse_args()
    schedules = {"embree": a.embree, "fcpw": a.fcpw}
    tuned = {"embree": a.embree_tuned, "fcpw": a.fcpw_tuned} if a.tuned else None
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    draw(rows, schedules, a.out, "-tuned" if a.tuned else "", a.format.split(","),
         exclude, tuned)
    if not a.tuned:
        draw_packet(rows, a.packet, a.out, a.format.split(","), exclude)
    geomeans(rows, schedules, exclude, tuned, None if a.tuned else a.packet)


if __name__ == "__main__":
    main()
