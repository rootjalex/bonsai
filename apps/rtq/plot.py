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
on that cell: the y axis reads Speedup, the legend says over which
reference each colour is, and the line at 1 is the reference. lte-orb is
left out by default (--exclude names the meshes left out): FCPW's triangle
test finds no hit at all on that small-unit mesh, so its ao, diffuse and
point batches are empty and the mesh says nothing about the comparison. A
fourth figure, the packets, is drawn when the CSV has rows of the packet
schedule (`packet16`; --packet names another): three bars a mesh, the
program's rays in gangs of sixteen over Embree's rtcIntersect16 (primary
and diffuse rays) and rtcOccluded16 (ambient-occlusion rays), the ray sets
by hatching, both sides over the same BVH8 and the same sixteen consecutive
rays a packet.

Each reference on its own default tree: Embree's BVH8 (`embree`) and FCPW's
four-wide Mbvh with sixteen-lane leaves (`fcpw4w16`, FCPW's default branching
and the leaf width its build picks on an AVX-512 machine; the four-wide tree
is the one FCPW's author calls the better tested); --embree and --fcpw name
other matching schedules. With --tuned the bars are the tuning's gain alone:
bonsai's rate under the tuned schedule (`tuned`, `fcpw4w16t`; --embree-tuned, --fcpw-tuned)
over its rate under the matching one, cell by cell, the reference out of it,
so the line at 1 is the matching schedule and the colour says which tree.

Beside each figure (the user's rule, 10-24) go the statistics the paper
quotes, `<figure>.txt`: per reference and ray or point set the geomean over
the plotted meshes, the least and greatest cell and which mesh each is, and
notes reading the figure; and the mesh facts go in one table rather than
on every plot, `rtq-meshes.txt` to read and `rtq-meshes.tex` to \\input
(a booktabs tabular: triangles, each reference's node count), so the tick
labels are the mesh names alone.

The colours are apps/pbrt's figures' (the user's choice, 10-24): two classes
of ColorBrewer's nine-class YlGnBu ramp, far enough apart to tell with the
hatches over them -- the light second for Embree, the fifth for FCPW -- and
the bars' edge and the line at 1 as there. The figures are drawn at the
size they are printed (apps/pbrt/figure_style.py has the format): 5.478 in
wide, acmsmall's text block, what \\includegraphics[width=\\textwidth]
scales a figure to in the paper, 1.5 in tall so that the three stack in
half a page, 8 pt labels and 7 pt ticks in Liberation Sans embedded as
TrueType (pdf.fonttype 42, which ACM's production accepts where
matplotlib's default Type 3 fonts are refused), the mesh names turned 45
degrees, and the legend in one row above the axes.
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

# The paper's figure style, apps/pbrt's module: acmsmall's text width, the
# font sizes, the YlGnBu ramp, Liberation Sans embedded as TrueType, and
# the summary written beside a figure.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrt"))
from figure_style import TEXT_WIDTH, YLGNBU, lean_labels, paper_style, write_summary  # noqa: E402

EDGE = "#333333"  # every bar's thin edge, as apps/pbrt's figures
COLOUR = {"embree": YLGNBU[1], "fcpw": YLGNBU[5]}
NAME = {"embree": "Embree", "fcpw": "FCPW"}
HATCH = {"primary": "", "ao": "///", "diffuse": "xxx", "near": "", "volume": "///"}
# The legend's words for the sets; the caption says they are rays or points.
LABEL = {"primary": "primary", "ao": "ambient occlusion", "diffuse": "diffuse",
         "near": "near surface", "volume": "in volume"}
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
# A height three of these stack in on half a page with their captions.
HEIGHT = 1.5


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


def callbacks_per_query(rows, meshes, schedule, rays):
    """The mean over the plotted meshes of the triangles Embree handed its
    rtcPointQuery callback per query (the CSV's callbacks_per_query), for
    the closest-point notes."""
    values = [float(r["callbacks_per_query"]) for r in rows
              if r["reference"] == "embree" and r["schedule"] == schedule
              and r["query"] == "closest" and r["rays"] == rays
              and r["mesh"] in meshes and r.get("callbacks_per_query")]
    return sum(values) / len(values) if values else None


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


def new_figure():
    paper_style()
    return plt.subplots(figsize=(TEXT_WIDTH, HEIGHT))


def mesh_ticks(ax, meshes):
    """The meshes along the x axis by name alone (their facts are in the
    mesh table), leaning only as far as they must not to overlap
    (figure_style.lean_labels; twelve names at text width lean about 20
    degrees -- the user, 10-24: 45 degrees left the figure mostly white)."""
    ax.set_xticks(range(len(meshes)))
    ax.set_xlim(-0.6, len(meshes) - 0.4)
    lean_labels(ax, meshes, (TEXT_WIDTH - 0.6) / len(meshes))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def legend_above(ax, handles):
    """The legend in one row above the axes, clear of the tallest bar; no
    title (the figure's caption says which query it is, and which
    schedules)."""
    ax.legend(handles=handles, ncol=len(handles), loc="lower left", frameon=False,
              bbox_to_anchor=(0, 1.0), borderaxespad=0, columnspacing=1.0)


def save(fig, out_dir, name, formats):
    """The page exactly the figure's size (no cropping to the content, which
    would leave LaTeX to rescale it): the layout fits the labels and the
    legend inside it. Returns the path of the first format written."""
    fig.tight_layout(pad=0.3)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=200)
        print(path)
        paths.append(path)
    plt.close(fig)
    return paths[0]


def tex_number(n):
    return f"{n:,}".replace(",", "\\,")


def mesh_table(rows, meshes, tris, schedules, out_dir, exclude):
    """The mesh facts in one table rather than on every plot (the user,
    10-24): rtq-meshes.txt to read and rtq-meshes.tex to \\input, a booktabs
    tabular. Each mesh's triangles and the node count of each reference's
    tree, read off its matching schedule's rows."""
    nodes = {}
    for r in rows:
        if r["schedule"] in schedules.values() and r.get("nodes"):
            nodes[(r["reference"], r["mesh"])] = int(r["nodes"])
    columns = [("mesh", lambda m: m), ("triangles", lambda m: tris[m]),
               ("Embree BVH8 nodes", lambda m: nodes.get(("embree", m), "")),
               ("FCPW Mbvh nodes", lambda m: nodes.get(("fcpw", m), ""))]
    table = [[f(m) for _, f in columns] for m in meshes]
    txt = os.path.join(out_dir, "rtq-meshes.txt")
    with open(txt, "w") as f:
        widths = [max(len(h), *(len(f"{row[i]:,}" if isinstance(row[i], int) else str(row[i]))
                                for row in table)) for i, (h, _) in enumerate(columns)]
        f.write("  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                          for i, ((h, _), w) in enumerate(zip(columns, widths))) + "\n")
        for row in table:
            f.write("  ".join((str(v) if i == 0 else f"{v:,}" if isinstance(v, int) else str(v))
                              .ljust(w) if i == 0 else
                              (f"{v:,}" if isinstance(v, int) else str(v)).rjust(w)
                              for i, (v, w) in enumerate(zip(row, widths))) + "\n")
        if exclude:
            f.write(f"\nleft out of the figures: {', '.join(sorted(exclude))}\n")
    print(txt)
    tex = os.path.join(out_dir, "rtq-meshes.tex")
    with open(tex, "w") as f:
        f.write("% apps/rtq/plot.py: the meshes of the rtq figures, in order of triangle\n"
                "% count, with the node count of each reference's tree. \\usepackage{booktabs}.\n")
        f.write("\\begin{tabular}{l" + "r" * (len(columns) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(h for h, _ in columns) + " \\\\\n\\midrule\n")
        for row in table:
            f.write(" & ".join(tex_number(v) if isinstance(v, int) else str(v) for v in row)
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(tex)


def series(data, meshes, rays):
    """The (mesh, speedup) points of one ray or point set, in mesh order."""
    return [(m, data[(m, rays)]) for m in meshes if (m, rays) in data]


def notes_for(name, rows, meshes, schedules, tuned):
    """What the paper says beside each figure: where the gains come from and
    where their causes are measured (apps/rtq/PLAN.md)."""
    if tuned:
        return ["every bar is bonsai under the tuned schedule over bonsai under the "
                "matching one, cell by cell: the tuning's gain alone, the reference out of it"]
    notes = []
    if name == "firsthit":
        notes += [
            "over Embree: Embree's own traversal order over Embree's BVH8 bytes against "
            "rtcIntersect1; level is the target of a matching schedule, and the bars are "
            "level within the run-to-run spread",
            "over FCPW: the same four-wide Mbvh with sixteen-lane leaves on both sides, FCPW's "
            "vectorized build (enoki); the gain is the generated traversal's, measured in "
            "apps/rtq/PLAN.md (the fcpw4w16 sections)",
        ]
    elif name == "anyhit":
        notes += [
            "over Embree: rtcOccluded1 against the any hit in Embree's children order, both "
            "leaving at the first hit; level by design",
            "over FCPW: FCPW's any hit is its closest-hit traversal stopped at the first hit over "
            "the same tree; the gain is the generated traversal's (apps/rtq/PLAN.md)",
        ]
    elif name == "closestpoint":
        near = callbacks_per_query(rows, meshes, schedules["embree"], "near")
        volume = callbacks_per_query(rows, meshes, schedules["embree"], "volume")
        handed = (f" -- {near:.0f} triangles a near query and {volume:.0f} a volume query on "
                  f"these meshes (the CSV's callbacks_per_query)" if near and volume else "")
        notes += [
            "over Embree: Embree offers the closest point only as rtcPointQuery with a user "
            "callback per leaf triangle, its tutorial's form, and hands the callback every "
            f"triangle of every leaf it reaches{handed}; its traversal cannot shrink the radius "
            "by a child's far corner, which ours does, so the gain grows with the mesh "
            "(apps/rtq/PLAN.md, 'The closest point against Embree's rtcPointQuery')",
            "over FCPW: FCPW's findClosestPoint over the same four-wide tree, the smallest of "
            "the gains over FCPW",
        ]
    return notes


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
    fig, ax = new_figure()
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
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3)
    mesh_ticks(ax, meshes)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("Speedup")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLOUR["embree"], edgecolor=EDGE, linewidth=0.4,
                     label=f"over Embree's packets of {k}")]
    handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4,
                      hatch=HATCH[rays], label=LABEL[rays]) for _, rays in PACKET_SETS]
    legend_above(ax, handles)
    path = save(fig, out_dir, "rtq-packet", formats)
    print(write_summary(
        path, f"Speedup over Embree's packets of {k} (rtcIntersect{k}, rtcOccluded{k}), "
              f"{len(meshes)} meshes",
        [("every mesh", [(f"{LABEL[rays]} ({query})", series(data, meshes, rays))
                         for query, rays in PACKET_SETS])],
        ["the program's rays in gangs of sixteen against Embree's packet kernels, both sides "
         "over the same BVH8 and the same sixteen consecutive rays a packet"]))


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
        fig, ax = new_figure()
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
        ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3)
        mesh_ticks(ax, meshes)
        ax.set_ylim(0, top * 1.08)
        ax.set_ylabel("Speedup")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        # The colours say what the bar is over (the user, 10-24: the y axis
        # reads Speedup, the legend what over); with --tuned, which tree.
        handles = [Patch(facecolor=COLOUR[ref], edgecolor=EDGE, linewidth=0.4,
                         label=(f"{NAME[ref]}'s tree, over matching" if tuned
                                else f"over {NAME[ref]}"))
                   for ref, _ in refs]
        handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4,
                          hatch=HATCH[s], label=LABEL[s]) for s in sets]
        legend_above(ax, handles)
        path = save(fig, out_dir, f"rtq-{name}{suffix}", formats)
        what = "tuned schedule over matching" if tuned else "bonsai over the reference"
        kind = "points" if query == "closest" else "rays"
        sections = [(f"over {NAME[ref]} ({schedules[ref]}{', tuned ' + tuned[ref] if tuned else ''})",
                     [(f"{LABEL[s]} {kind}", series(data[ref], meshes, s)) for s in sets])
                    for ref, _ in refs if data[ref]]
        notes = notes_for(name, rows, meshes, schedules, tuned)
        if exclude:
            notes.append(f"left out: {', '.join(sorted(exclude))} (FCPW finds no hit on lte-orb)")
        print(write_summary(path, f"{title}: {what}, {len(meshes)} meshes", sections, notes))


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
        meshes, tris = meshes_by_triangles(rows, exclude)
        mesh_table(rows, meshes, tris, schedules, a.out, exclude)
    geomeans(rows, schedules, exclude, tuned, None if a.tuned else a.packet)


if __name__ == "__main__":
    main()
