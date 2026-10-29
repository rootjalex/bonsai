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
point batches are empty and the mesh says nothing about the comparison.
Two more, the queries apps/wosx's walks make over FCPW's scalar trees, are
drawn when the CSV has their rows: `rtq-fcpwb-fcpws-cpu`,
bonsai over FCPW's scalar build, and `rtq-fcpwb-fcpws-gpu`, bonsai over
FCPW's GPUScene, seven bars a mesh -- the first hit, the any hit and the
closest point over FCPW's binary BVH (`fcpwb`, `fcpwb-gpu`) and the closest
silhouette point over its normal cone hierarchy (`fcpws`, `fcpws-gpu`), the
query by colour and the set by hatching.

The three are also drawn as one, `rtq-queries`, a figure the size of one
of them for the page that has not the room for three: ten bars a mesh,
adjacent, in the three figures' order -- the first hit's primary and
diffuse rays, the any hit's ambient-occlusion rays, the closest point's
near-surface and in-volume points, Embree's bar then FCPW's within each
set -- the colour the reference and each set's hatch the one it has in the
three, and the legend one row, the colours then the sets under the short
names the caption explains: ch (primary), ch (diffuse), ao, cp (near),
cp (in). The y axis is the closest point's, so the ray queries' bars over
Embree stand at the line. The three are drawn as before beside it.

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
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation, blended_transform_factory  # noqa: E402

# The paper's figure style, apps/pbrt's module: acmsmall's text width, the
# font sizes, the YlGnBu ramp, Liberation Sans embedded as TrueType, and
# the summary written beside a figure.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrt"))
from figure_style import TEXT_WIDTH, YLGNBU, lean_labels, paper_style, write_summary  # noqa: E402

# Every bar's thin edge, and so its hatch, which matplotlib draws in the
# edge colour: black (the user, 10-29: apps/pbrt's #333333 read as a light
# grey in the hatches of the combined figure's narrow bars).
EDGE = "black"
COLOUR = {"embree": YLGNBU[1], "fcpw": YLGNBU[5]}
NAME = {"embree": "Embree", "fcpw": "FCPW"}
# One hatch per set, no two alike, so that a set reads the same in every
# figure and the legend has no two swatches the same (the user, 10-26: the
# ambient-occlusion and in-volume bars had shared one).
HATCH = {"primary": "", "ao": "///", "diffuse": "xxx", "near": "...", "volume": "\\\\\\"}


def hatch_colour(face):
    """The colour a bar's hatch and edge are drawn in: the dark edge on
    every bar but the darkest blues, white there. A hatch is drawn in the
    patch's edge colour, and the dark grey vanished on the silhouette bars
    (YLGNBU[7]), where the two silhouette sets read as one bar (the user,
    10-26); the closest point's mid blue (YLGNBU[5]) still shows the dark
    hatch and keeps it, and white on the light yellow would vanish the same
    way."""
    r, g, b = matplotlib.colors.to_rgb(face)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.35 else EDGE
# The legend's words for the sets; the caption says they are rays or points.
LABEL = {"primary": "primary", "ao": "ambient occlusion", "diffuse": "diffuse",
         "near": "near surface", "volume": "in volume"}
# The combined figure's legend, one row (the user, 10-29): the sets under
# the short names its caption explains -- ch the closest (first) hit of
# the primary and diffuse rays, ao the any hit of the ambient-occlusion
# rays, cp the closest point of the near-surface and in-volume points.
SHORT = {"primary": "ch (primary)", "diffuse": "ch (diffuse)", "ao": "ao",
         "near": "cp (near)", "volume": "cp (in)"}
# Aila and Laine's three ray sets, each under the query it is for: the first
# hit of primary and diffuse rays, the any hit of ambient-occlusion rays.
FIGURES = [
    ("firsthit", "nearest", ["primary", "diffuse"], "First hit"),
    ("anyhit", "any", ["ao"], "Any hit"),
    ("closestpoint", "closest", ["near", "volume"], "Closest point"),
]
# The two figures of the queries apps/wosx's walks make, over FCPW's scalar
# trees (the build WoSX ships): one for the CPU, bonsai over FCPW's scalar
# build, one for the GPU, bonsai over FCPW's GPUScene. Seven bars a mesh,
# the query by colour -- four classes of the same ramp, far enough apart
# with the hatches -- and the set by hatching: the first hit of primary and
# diffuse rays and the any hit of ambient-occlusion rays over the binary
# BVH (fcpwb), the closest point of the near and volume points over it, and
# the closest silhouette point about the same points over the normal cone
# hierarchy (fcpws). Each platform's pair of schedules, and its reference's
# name in the CSV.
#
# Named by the two schedules whose cells they hold (the user, 10-26: the
# name says the data structure): `rtq-fcpwb-fcpws-cpu`, FCPW's binary BVH
# and its normal cone hierarchy on the CPU, and `rtq-fcpwb-fcpws-gpu`, the
# same two trees on the GPU -- FCPW's GPUScene takes only the binary trees
# (its README: the scene must be built with vectorization off before it is
# transferred), so there is no GPU counterpart of the four-wide figures.
SCALAR_FIGURES = [
    ("fcpwb-fcpws-cpu", "fcpw", "fcpwb", "fcpws", "FCPW's scalar build"),
    ("fcpwb-fcpws-gpu", "fcpw-gpu", "fcpwb-gpu", "fcpws-gpu", "FCPW's GPUScene"),
]
# (query, set, which of the two schedules, the legend's name of the query)
SCALAR_SETS = [
    ("nearest", "primary", 0, "First hit"), ("nearest", "diffuse", 0, "First hit"),
    ("any", "ao", 0, "Any hit"),
    ("closest", "near", 0, "Closest point"), ("closest", "volume", 0, "Closest point"),
    ("silhouette", "near", 1, "Silhouette"), ("silhouette", "volume", 1, "Silhouette"),
]
QUERY_COLOUR = {"First hit": YLGNBU[1], "Any hit": YLGNBU[3], "Closest point": YLGNBU[5],
                "Silhouette": YLGNBU[7]}
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


def callbacks_per_query(rows, meshes, schedule, rays):
    """The mean over the plotted meshes of the triangles Embree handed its
    rtcPointQuery callback per query (the CSV's callbacks_per_query), for
    the closest-point notes."""
    values = [float(r["callbacks_per_query"]) for r in rows
              if r["reference"] == "embree" and r["schedule"] == schedule
              and r["query"] == "closest" and r["rays"] == rays
              and r["mesh"] in meshes and r.get("callbacks_per_query")]
    return sum(values) / len(values) if values else None


def geomeans(rows, schedules, exclude, tuned):
    """The geometric mean of each figure's bars over the plotted meshes, per
    reference and ray or point set: the one number per cell kind the
    comparison is judged by (the user's rule of 2026-10-07: every geomean a
    win, no single cell more than 3% under its reference). Printed as a
    table, with the cells under 0.97 named, since a geomean can hide one."""
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


def finish_axes(ax, meshes, top):
    """What every figure's axes have once the bars are drawn: the line at 1
    (the reference), the meshes along the x axis, the y axis from 0 to just
    over the tallest bar reading Speedup, no top or right spine, a faint
    grid behind the bars."""
    ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3)
    mesh_ticks(ax, meshes)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("Speedup")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def legend_above(ax, handles, ncol=None, page_left=False, **spacing):
    """The legend in one row above the axes, clear of the tallest bar; no
    title (the figure's caption says which query it is, and which
    schedules). `ncol` makes it rows of that many, where one row of all
    the handles would be wider than the figure; `page_left` starts the row
    at the page's left edge rather than the axes' (the combined figure's
    seven entries, which in one row are wider than the axes); `spacing`
    overrides the legend's column spacing and handle padding. Returns the
    legend."""
    transform = ax.transAxes
    if page_left:
        fig = ax.figure
        transform = (blended_transform_factory(fig.transFigure, ax.transAxes)
                     + ScaledTranslation(0.03, 0, fig.dpi_scale_trans))
    return ax.legend(handles=handles, ncol=ncol or len(handles), loc="lower left",
                     frameon=False, bbox_to_anchor=(0, 1.0), bbox_transform=transform,
                     borderaxespad=0, **{"columnspacing": 1.0, **spacing})


def save(fig, out_dir, name, formats, legends=()):
    """The page exactly the figure's size (no cropping to the content, which
    would leave LaTeX to rescale it): the layout fits the labels and the
    legend inside it. With `legends`, those legends are kept out of
    tight_layout's reckoning and the height they measure together reserved
    above the axes instead: tight_layout takes a legend wider than the
    axes' first guess for a margin the axes must leave on the right, and
    squeezes them (the scalar figures' one-row legend halved theirs,
    3eeb61ad); left out, the legends run from the axes' left edge as wide
    as the page allows. Returns the path of the first format written."""
    pad = 0.3
    if not legends:
        fig.tight_layout(pad=pad)
    else:
        for legend in legends:
            legend.set_in_layout(False)
        fig.canvas.draw()
        boxes = [legend.get_window_extent(fig.canvas.get_renderer()) for legend in legends]
        height = (max(b.y1 for b in boxes) - min(b.y0 for b in boxes)) / fig.dpi
        gap = pad * plt.rcParams["font.size"] / 72
        fig.tight_layout(pad=pad, rect=(0, 0, 1, 1 - (height + gap) / fig.get_figheight()))
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
        # FCPW's binary trees (the scalar figures'): the BVH's node count, and
        # the normal cone hierarchy's, which has as many nodes over the same
        # triangles.
        if r["schedule"] == "fcpwb" and r.get("nodes"):
            nodes[("fcpwb", r["mesh"])] = int(r["nodes"])
    columns = [("mesh", lambda m: m), ("triangles", lambda m: tris[m]),
               ("Embree BVH8 nodes", lambda m: nodes.get(("embree", m), "")),
               ("FCPW Mbvh nodes", lambda m: nodes.get(("fcpw", m), ""))]
    if any(k[0] == "fcpwb" for k in nodes):
        columns.append(("FCPW binary BVH nodes", lambda m: nodes.get(("fcpwb", m), "")))
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


def scalar_cells(rows, reference, schedules):
    """The cells of one platform's figure, keyed by (mesh, query, rays): each
    set's speedup under the schedule it is measured with."""
    out = {}
    for query, rays, which, _ in SCALAR_SETS:
        for (mesh, r), v in cells(rows, reference, schedules[which], query).items():
            if r == rays:
                out[(mesh, query, rays)] = v
    return out


def draw_scalar(rows, out_dir, formats, exclude):
    """The two figures over FCPW's scalar trees, the CPU's and the GPU's
    (SCALAR_FIGURES): drawn only where the CSV has the rows, each with its
    summary beside it."""
    for name, reference, binary, snch, over in SCALAR_FIGURES:
        data = scalar_cells(rows, reference, (binary, snch))
        if not data:
            print(f"{name}: no rows for {binary}, {snch} over {reference}; skipped",
                  file=sys.stderr)
            continue
        meshes = [m for m in meshes_by_triangles(rows, exclude)[0]
                  if any(k[0] == m for k in data)]
        n = len(SCALAR_SETS)
        width = 0.8 / n
        # Taller than the three by the legend's second row: seven handles in
        # one row are wider than the text block.
        paper_style()
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, HEIGHT + 0.25))
        top = 1.0
        for i, mesh in enumerate(meshes):
            for j, (query, rays, _, kind) in enumerate(SCALAR_SETS):
                v = data.get((mesh, query, rays))
                if v is None:
                    continue
                x = i - 0.4 + width * (j + 0.5)
                ax.bar(x, v, width, color=QUERY_COLOUR[kind], hatch=HATCH[rays],
                       edgecolor=hatch_colour(QUERY_COLOUR[kind]), linewidth=0.4)
                top = max(top, v)
        finish_axes(ax, meshes, top)
        kinds = []
        for _, _, _, kind in SCALAR_SETS:
            if kind not in kinds:
                kinds.append(kind)
        handles = [Patch(facecolor=QUERY_COLOUR[k], edgecolor=EDGE, linewidth=0.4, label=k)
                   for k in kinds]
        sets = []
        for _, rays, _, _ in SCALAR_SETS:
            if rays not in sets:
                sets.append(rays)
        handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4, hatch=HATCH[s],
                          label=LABEL[s]) for s in sets]
        # Two rows of five, filled down the columns: the four queries in the
        # first two -- First hit over Any hit, Closest point over Silhouette
        # -- and the five sets in the other three (the user, 10-26).
        legend_above(ax, handles, ncol=5)
        path = save(fig, out_dir, f"rtq-{name}", formats)
        sections = []
        for kind in kinds:
            sections.append((kind, [(f"{LABEL[rays]} ({query})",
                                     [(m, data[(m, query, rays)]) for m in meshes
                                      if (m, query, rays) in data])
                                    for query, rays, _, k in SCALAR_SETS if k == kind]))
        notes = [
            f"every bar is bonsai over {over} on the same tree: the first hit, the any hit "
            f"and the closest point over FCPW's binary BVH ({binary}), the closest silhouette "
            f"point over its normal cone hierarchy ({snch}), the queries apps/wosx's walks make",
            "the silhouette query asks, about each point of the two point sets, for the "
            "closest silhouette edge within a tenth of the diagonal, no minimum radius, so "
            "that both sides search to the exact closest",
        ]
        if reference == "fcpw-gpu":
            notes.append("both sides timed around the dispatch and the wait alone, the "
                         "queries uploaded and the answers read back off the clock; one "
                         "query a thread in groups of 256 on both sides")
        if exclude:
            notes.append(f"left out: {', '.join(sorted(exclude))} (FCPW finds no hit on lte-orb)")
        print(write_summary(path, f"Speedup over {over}, the walks' queries, {len(meshes)} meshes",
                            sections, notes))


def scalar_geomeans(rows, exclude):
    """The geomeans of the two scalar figures' bars, printed as geomeans()
    prints the others, where the CSV has their rows."""
    for name, reference, binary, snch, over in SCALAR_FIGURES:
        data = scalar_cells(rows, reference, (binary, snch))
        if not data:
            continue
        meshes = [m for m in meshes_by_triangles(rows, exclude)[0]
                  if any(k[0] == m for k in data)]
        print(f"geomeans, bonsai over {over}, {len(meshes)} meshes:")
        for query, rays, which, _ in SCALAR_SETS:
            values = {m: data[(m, query, rays)] for m in meshes if (m, query, rays) in data}
            if not values:
                continue
            g = math.exp(sum(math.log(v) for v in values.values()) / len(values))
            low = " ".join(f"{m} {v:.2f}" for m, v in values.items() if v < 0.97)
            print(f"  {reference:<10} {(binary, snch)[which]:<10} {query:<10} {rays:<8} "
                  f"{g:8.3f} {min(values.values()):6.2f} {max(values.values()):6.2f}  {low}")


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
                           edgecolor=hatch_colour(COLOUR[ref]), linewidth=0.4)
                    top = max(top, v)
        finish_axes(ax, meshes, top)
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


def draw_combined(rows, schedules, out_dir, suffix, formats, exclude, tuned):
    """The three figures' cells in one figure, `rtq-queries` (the user,
    10-29: the three on one plot, to save the page), the size of one of
    them. Three things vary in a mesh's ten bars: the reference by colour,
    as in every figure (the legend says what the bar is over); the ray or
    point set by hatch, the five hatches the three figures use; and the
    query by the set's short name in the legend -- ch, ao, cp -- which the
    caption explains, the sets in the three figures' order. The set is
    nested in the query (a set belongs to one query, so there are five
    pairs, not fifteen), which is why the hatch alone fixes the query.
    Within a set Embree's bar then FCPW's, and the ten bars adjacent, as
    in the three (the user, 10-29: a space within a mesh reads as a
    missing bar; a headed legend block a query, and the sets' full names,
    took three legend rows and left the figure mostly white). The other
    ways considered: the query by hatch and the set by a tint of the
    reference's colour (two tints of Embree's light yellow are white and
    green, and the paper keeps one colour a system); three panels across
    the width (twelve meshes by four bars in 1.6 inches cannot be read,
    and the mesh names would take half the height)."""
    meshes, _ = meshes_by_triangles(rows, exclude)
    refs = [("embree", schedules["embree"]), ("fcpw", schedules["fcpw"])]
    data = {}
    for ref, sched in refs:
        for _, query, _, _ in FIGURES:
            data[(ref, query)] = (tuned_over_matching(rows, ref, sched, tuned[ref], query)
                                  if tuned else cells(rows, ref, sched, query))
    if not any(data.values()):
        print(f"queries: no rows for {schedules}; skipped", file=sys.stderr)
        return
    # A mesh's ten bars, adjacent as the three figures' are (the user,
    # 10-29: a space within a mesh reads as a missing bar): the queries in
    # the figures' order, each set's two references side by side.
    slots = [(query, s, ref) for _, query, sets, _ in FIGURES for s in sets for ref, _ in refs]
    width = 0.8 / len(slots)
    paper_style()
    # The bars are 2.6 pt wide here against the three figures' 4.4 to 6.6,
    # and matplotlib's 1 pt hatch lines fill them: the crosses and the
    # diagonals read as dark bars. Thinner lines for this figure alone (0.8
    # pt, in black: at 0.6 and the grey edge of before the user read them as
    # light grey, 10-29); the setting is read as the figure is drawn, so the
    # block runs to the save.
    with plt.rc_context({"hatch.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, HEIGHT))
        top = 1.0
        for i, mesh in enumerate(meshes):
            for j, (query, rays, ref) in enumerate(slots):
                v = data[(ref, query)].get((mesh, rays))
                if v is None:
                    continue
                ax.bar(i - 0.4 + width * (j + 0.5), v, width, color=COLOUR[ref],
                       hatch=HATCH[rays], edgecolor=hatch_colour(COLOUR[ref]), linewidth=0.4)
                top = max(top, v)
        finish_axes(ax, meshes, top)
        # The legend in one row (the user, 10-29): the colours, saying what
        # the bars are over, then the sets in the bars' order under the
        # caption's short names (SHORT). Seven entries at the three
        # figures' spacing run 5.65 in, wider than the page, so the row
        # starts at the page's left edge rather than the axes' and the
        # entries stand a little closer: 5.28 in, a quarter inch to spare.
        handles = [Patch(facecolor=COLOUR[ref], edgecolor=EDGE, linewidth=0.4,
                         label=(f"{NAME[ref]}'s tree, over matching" if tuned
                                else f"over {NAME[ref]}"))
                   for ref, _ in refs]
        handles += [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.4, hatch=HATCH[s],
                          label=SHORT[s]) for _, _, sets, _ in FIGURES for s in sets]
        legend = legend_above(ax, handles, page_left=True, handletextpad=0.5, columnspacing=0.8)
        fig.canvas.draw()
        if legend.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi > TEXT_WIDTH - 0.06:
            # Wider than the page even so (the tuned labels, or a wider
            # fallback font): two rows from the axes' left, filled down the
            # columns, the figure a quarter inch taller for the second, as
            # the scalar figures are.
            legend.remove()
            legend = legend_above(ax, handles, ncol=(len(handles) + 1) // 2)
            fig.set_size_inches(TEXT_WIDTH, HEIGHT + 0.25)
        path = save(fig, out_dir, f"rtq-queries{suffix}", formats, [legend])
    what = "tuned schedule over matching" if tuned else "bonsai over the reference"
    sections = []
    for ref, _ in refs:
        named = [(f"{title.lower()}, {LABEL[s]} {'points' if query == 'closest' else 'rays'}",
                  series(data[(ref, query)], meshes, s))
                 for _, query, qsets, title in FIGURES for s in qsets]
        if any(points for _, points in named):
            sections.append((f"over {NAME[ref]} ({schedules[ref]}"
                             f"{', tuned ' + tuned[ref] if tuned else ''})", named))
    if tuned:
        notes = notes_for("firsthit", rows, meshes, schedules, tuned)
    else:
        notes = [f"{title.lower()}, {line}" for name, _, _, title in FIGURES
                 for line in notes_for(name, rows, meshes, schedules, tuned)]
    if exclude:
        notes.append(f"left out: {', '.join(sorted(exclude))} (FCPW finds no hit on lte-orb)")
    print(write_summary(path, f"First hit, any hit and closest point: {what}, "
                        f"{len(meshes)} meshes", sections, notes))


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
    draw_combined(rows, schedules, a.out, "-tuned" if a.tuned else "", a.format.split(","),
                  exclude, tuned)
    if not a.tuned:
        draw_scalar(rows, a.out, a.format.split(","), exclude)
        meshes, tris = meshes_by_triangles(rows, exclude)
        mesh_table(rows, meshes, tris, schedules, a.out, exclude)
    geomeans(rows, schedules, exclude, tuned)
    if not a.tuned:
        scalar_geomeans(rows, exclude)


if __name__ == "__main__":
    main()
