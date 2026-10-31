#!/usr/bin/env python3
"""The rtq comparison as the paper's figures, from the results CSV
compare.sh's runs are folded into (one row per measured cell: reference,
schedule, mesh, query, ray set, both rates, the speedup).

    python apps/rtq/plot.py apps/rtq/results/rtq-results.csv -o apps/rtq/plots
    python apps/rtq/plot.py results.csv -o plots/ --tuned      # the tuned schedules

The paper's three figures (the user, 10-30), the meshes along the x axis
in order of triangle count, every bar bonsai's rate over the reference's
on that cell with the line at 1 the reference, and the ray or point set by
its fill -- one shade a set in every figure, from the paper's ramp
(SET_COLOUR), the legend one row under the short names the caption
explains: ch the closest (first) hit of the primary and diffuse rays, ao
the any hit of the ambient-occlusion rays, cp the closest point of the
near-surface and in-volume points, sq the closest silhouette point about
the same points.

`rtq-queries-cpu`: the first hit (`nearest`), the any hit (`any`) and the
closest point (`closest`) over Embree and over FCPW, two panels, one a
reference (named in the panel), five bars a mesh in the sets' order, each
panel its own y range since the points over Embree run to 5x where the
rays stand at the line.

`rtq-fcpwb-cpu` and `rtq-fcpwb-gpu`: the queries apps/wosx's walks make
over FCPW's scalar trees, bonsai over FCPW's scalar build and over its
GPUScene, drawn when the CSV has their rows: seven bars a mesh, the five
sets over FCPW's binary BVH (`fcpwb`, `fcpwb-gpu`) and the two silhouette
sets over its normal cone hierarchy (`fcpws`, `fcpws-gpu`), the y axis
Speedup over FCPW.

Also drawn, the three figures the paper had before, one a query
(`rtq-firsthit`, `rtq-anyhit`, `rtq-closestpoint`): six, six and four
bars a mesh, the reference by colour (the light yellow Embree, the blue
FCPW; the legend says over which) and the set by hatching, for the reading
of one query at the size of a page's width. lte-orb is left out of every
figure by default (--exclude names the meshes left out): FCPW's triangle
test finds no hit at all on that small-unit mesh, so its ao, diffuse and
point batches are empty and the mesh says nothing about the comparison.

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
(a booktabs tabular: triangles, camera hits, each reference's node count), so the tick
labels are the mesh names alone.

The colours are the paper's one ramp, ColorBrewer's nine-class YlGnBu as
apps/pbrt/figure_style.py has it (the user's palette, 10-23). The sets
take seven shades of it at equal perceptual steps from its class 1 to its
class 8 (RAMP7, by the method of figure_style's RAMP8), light to dark in
the bars' order, so that a set keeps one shade in every figure (the user's
rule for the pbrt figures, 10-29: one colour a schedule across every
figure) and a greyscale print keeps the order; neighbouring shades are
12.7 or more apart in OKLab x 100 under normal vision and 10.3 or more
under the protan, deutan and tritan simulations (apps/rtq/scratch/
cvd_check.py), the eight-step ramp's kind of separation, with the fixed
bar order under the legend the second reading. The three older figures
colour the references instead, two classes far enough apart to tell with
the hatches over them, the light second for Embree, the fifth for FCPW.
Every bar's edge is black, as is the line at 1. The figures are drawn at
the size they are printed (figure_style.py has the format): 5.478 in
wide, acmsmall's text block, what \\includegraphics[width=\\textwidth]
scales a figure to in the paper, 1.5 in tall (the two-panel queries
figure 2.0) so that three stack in half a page, 8 pt labels and 7 pt
ticks in Liberation Sans embedded as TrueType (pdf.fonttype 42, which
ACM's production accepts where matplotlib's default Type 3 fonts are
refused), the mesh names leaned by figure_style's rule, and the legend in
one row above the axes.
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
# The legend's words for the sets in the three older figures; the caption
# says they are rays or points.
LABEL = {"primary": "primary", "ao": "ambient occlusion", "diffuse": "diffuse",
         "near": "near surface", "volume": "in volume"}
# Aila and Laine's three ray sets, each under the query it is for: the first
# hit of primary and diffuse rays, the any hit of ambient-occlusion rays.
FIGURES = [
    ("firsthit", "nearest", ["primary", "diffuse"], "First hit"),
    ("anyhit", "any", ["ao"], "Any hit"),
    ("closestpoint", "closest", ["near", "volume"], "Closest point"),
]
# The paper's ramp cut in seven at equal perceptual steps from YlGnBu's
# class 1 to its class 8 -- figure_style's RAMP8's method, equal OKLab arc
# length along the ramp as a colour map interpolates it, the ends
# ColorBrewer's own (apps/rtq/scratch/cvd_check.py measures the steps) --
# one shade a set in every figure, light to dark in the bars' order (the
# user, 10-30; the rule of one colour a thing across every figure is the
# pbrt figures', 10-29). The lightness falls monotonically, so a greyscale
# print keeps the order; neighbours are 12.7 or more apart in OKLab x 100
# under normal vision and 10.3 or more under the protan, deutan and tritan
# simulations, the eight-step ramp's kind of separation (10.7 and 8.8),
# the fixed bar order under the legend the second reading. An orange for
# the any hit apart from the ramp was tried first (10-30): the same grey
# as the diffuse shade on a photocopy, and a second palette in the paper.
RAMP7 = ["#EDF8B1", "#A0DAB8", "#4DBBC2", "#1D91C0", "#2164AB", "#253493", "#081D58"]
# The seven sets in the bars' order, (query, set, the legend's short name).
# The short names are the caption's -- ch the closest (first) hit of the
# primary and diffuse rays, ao the any hit of the ambient-occlusion rays,
# cp the closest point of the near-surface and in-volume points, sq the
# closest silhouette point about the same points (the user, 10-29 and
# 10-30) -- and a set's shade is its place in RAMP7.
SETS = [
    ("nearest", "primary", "ch (primary)"), ("nearest", "diffuse", "ch (diffuse)"),
    ("any", "ao", "ao"),
    ("closest", "near", "cp (near)"), ("closest", "volume", "cp (in)"),
    ("silhouette", "near", "sq (near)"), ("silhouette", "volume", "sq (in)"),
]
SHORT = {(query, s): short for query, s, short in SETS}
SET_COLOUR = {(query, s): RAMP7[i] for i, (query, s, _) in enumerate(SETS)}
# The two figures of the queries apps/wosx's walks make, over FCPW's scalar
# trees (the build WoSX ships): one for the CPU, bonsai over FCPW's scalar
# build, one for the GPU, bonsai over FCPW's GPUScene. Seven bars a mesh,
# the set by its shade: the first hit of primary and diffuse rays and the
# any hit of ambient-occlusion rays over the binary BVH (fcpwb), the
# closest point of the near and volume points over it, and the closest
# silhouette point about the same points over the normal cone hierarchy
# (fcpws). Each platform's pair of schedules, and its reference's name in
# the CSV.
#
# Named `rtq-fcpwb-cpu` and `rtq-fcpwb-gpu` (the user, 10-30; until then
# rtq-fcpwb-fcpws-*, after both schedules): FCPW's binary trees on the CPU
# and on the GPU -- FCPW's GPUScene takes only the binary trees (its
# README: the scene must be built with vectorization off before it is
# transferred), so there is no GPU counterpart of the four-wide figures.
SCALAR_FIGURES = [
    ("fcpwb-cpu", "fcpw", "fcpwb", "fcpws", "FCPW's scalar build"),
    ("fcpwb-gpu", "fcpw-gpu", "fcpwb-gpu", "fcpws-gpu", "FCPW's GPUScene"),
]
# (query, set, which of the two schedules, the summary's name of the query)
SCALAR_SETS = [
    ("nearest", "primary", 0, "First hit"), ("nearest", "diffuse", 0, "First hit"),
    ("any", "ao", 0, "Any hit"),
    ("closest", "near", 0, "Closest point"), ("closest", "volume", 0, "Closest point"),
    ("silhouette", "near", 1, "Silhouette"), ("silhouette", "volume", 1, "Silhouette"),
]
# A height three of these stack in on half a page with their captions; the
# two-panel queries figure takes half an inch more.
HEIGHT = 1.5
PANELS_HEIGHT = 2.0


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


def legend_above(ax, handles, ncol=None, centre=False, **spacing):
    """The legend in one row above the axes, clear of the tallest bar; no
    title (the figure's caption says which query it is, and which
    schedules). `ncol` makes it rows of that many, where one row of all
    the handles would be wider than the figure; `centre` puts the row's
    middle over the axes' rather than its start at their left (the paper's
    three figures, the user 10-30); `spacing` overrides the legend's column
    spacing, handle length and padding. Returns the legend."""
    return ax.legend(handles=handles, ncol=ncol or len(handles),
                     loc="lower center" if centre else "lower left", frameon=False,
                     bbox_to_anchor=(0.5 if centre else 0, 1.0), borderaxespad=0,
                     **{"columnspacing": 1.0, **spacing})


def set_handles(pairs):
    """The legend's swatches for the sets `pairs` names, (query, set) each,
    in that order: the set's shade under its short name."""
    return [Patch(facecolor=SET_COLOUR[k], edgecolor=EDGE, linewidth=0.4, label=SHORT[k])
            for k in pairs]


def save(fig, out_dir, name, formats, legends=(), layout=True):
    """The page exactly the figure's size (no cropping to the content, which
    would leave LaTeX to rescale it): the layout fits the labels and the
    legend inside it. With `legends`, those legends are kept out of
    tight_layout's reckoning and the height they measure together reserved
    above the axes instead: tight_layout takes a legend wider than the
    axes' first guess for a margin the axes must leave on the right, and
    squeezes them (the scalar figures' one-row legend halved theirs,
    3eeb61ad); left out, the legends run from the axes' left edge as wide
    as the page allows. `layout=False` keeps the figure's own
    subplots_adjust (the two-panel figure, whose y label is a figure text
    tight_layout does not reckon). Returns the path of the first format
    written."""
    pad = 0.3
    if not layout:
        pass
    elif not legends:
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
    tree, read off its matching schedule's rows; and its camera hits, the
    pixels of the primary image the mesh covers, which is the size of each
    secondary set (one ao ray, one diffuse ray, one near point and one volume
    point a hit), read off Embree's ao row -- each reference's sets are made
    from its own primary hits, and FCPW's miss a few pixels of Embree's."""
    nodes = {}
    hits = {}
    for r in rows:
        if r["schedule"] in schedules.values() and r.get("nodes"):
            nodes[(r["reference"], r["mesh"])] = int(r["nodes"])
        if (r["schedule"] in schedules.values() and r["rays"] == "ao" and r.get("count")
                and (r["reference"] == "embree" or r["mesh"] not in hits)):
            hits[r["mesh"]] = int(r["count"])
        # FCPW's binary trees (the scalar figures'): the BVH's node count, and
        # the normal cone hierarchy's, which has as many nodes over the same
        # triangles.
        if r["schedule"] == "fcpwb" and r.get("nodes"):
            nodes[("fcpwb", r["mesh"])] = int(r["nodes"])
    columns = [("mesh", lambda m: m), ("triangles", lambda m: tris[m]),
               ("camera hits", lambda m: hits.get(m, "")),
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
                "% count, with the camera hits (the pixels of the 1024 by 1024 primary image\n"
                "% the mesh covers, the size of each secondary set) and the node count of\n"
                "% each reference's tree. \\usepackage{booktabs}.\n")
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
        width = 0.8 / len(SCALAR_SETS)
        fig, ax = new_figure()
        top = 1.0
        for i, mesh in enumerate(meshes):
            for j, (query, rays, _, _) in enumerate(SCALAR_SETS):
                v = data.get((mesh, query, rays))
                if v is None:
                    continue
                ax.bar(i - 0.4 + width * (j + 0.5), v, width, color=SET_COLOUR[(query, rays)],
                       edgecolor=EDGE, linewidth=0.4)
                top = max(top, v)
        finish_axes(ax, meshes, top)
        # One reference has the figure, so the y axis says what the bars are
        # over and nothing else does (the user, 10-30: no "over FCPW's ..."
        # label on a single-system plot; the summary's title names the build).
        ax.set_ylabel("Speedup over FCPW")
        # The seven sets in the bars' order, one row centred over the axes:
        # 4.9 in at these spacings, inside the page's 5.48.
        legend = legend_above(ax, set_handles([(q, r) for q, r, _, _ in SCALAR_SETS]),
                              centre=True, columnspacing=0.8, handletextpad=0.5,
                              handlelength=1.5)
        kinds = []
        for _, _, _, kind in SCALAR_SETS:
            if kind not in kinds:
                kinds.append(kind)
        path = save(fig, out_dir, f"rtq-{name}", formats, [legend])
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


def draw_queries(rows, schedules, out_dir, suffix, formats, exclude, tuned):
    """The paper's figure of the three CPU queries, `rtq-queries-cpu` (the
    user, 10-29: the three on one plot, to save the page; 10-30: this
    form): two panels, one a reference, named in the panel -- over Embree
    above over FCPW -- five bars a mesh in the sets' order filled by the
    set's shade, each panel its own y range since the points over Embree
    run to 5x where the rays stand at the line and one range squashed
    them, one y label for both, and the legend one row of the five sets
    centred over the axes. Nothing to decode but the fill. The forms tried
    before it (10-29, apps/rtq/scratch/queries_alternatives.py): both
    references in one panel, ten bars a mesh with the reference by colour
    and the set by hatch, which in 2.6 pt bars read as dark stripes
    (rtq-queries, bb8f05ed to 93f49783); the set by fill and the reference
    by outline; a tinted table of every speedup; the geomeans with
    whiskers."""
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
    pairs = [(query, s) for _, query, sets, _ in FIGURES for s in sets]
    width = 0.8 / len(pairs)
    paper_style()
    fig, axes = plt.subplots(len(refs), 1, sharex=True, figsize=(TEXT_WIDTH, PANELS_HEIGHT))
    for ax, (ref, _) in zip(axes, refs):
        top = 1.0
        for i, mesh in enumerate(meshes):
            for j, (query, rays) in enumerate(pairs):
                v = data[(ref, query)].get((mesh, rays))
                if v is None:
                    continue
                ax.bar(i - 0.4 + width * (j + 0.5), v, width, color=SET_COLOUR[(query, rays)],
                       edgecolor=EDGE, linewidth=0.4, zorder=2)
                top = max(top, v)
        # finish_axes's line, spines and grid, with the panel's own y range
        # at three ticks and a little more headroom for its name at the top
        # left; the mesh names on the lower panel alone.
        ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3)
        ax.set_ylim(0, top * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.004, 0.96, (f"{NAME[ref]}'s tree, over matching" if tuned
                              else f"over {NAME[ref]}"),
                transform=ax.transAxes, va="top", ha="left")
    mesh_ticks(axes[-1], meshes)
    legend = legend_above(axes[0], set_handles(pairs), centre=True, columnspacing=0.8,
                          handletextpad=0.5, handlelength=1.5)
    # The margins by hand (save's tight_layout would not reckon the y label,
    # a figure text spanning both panels): 0.2 in above the panels for the
    # legend, 0.38 in below for the leaned mesh names, the y label at the
    # panels' middle height.
    above, below = 1 - 0.2 / PANELS_HEIGHT, 0.38 / PANELS_HEIGHT
    fig.subplots_adjust(left=0.08, right=0.99, top=above, bottom=below, hspace=0.15)
    fig.text(0.012, (above + below) / 2, "Speedup", rotation=90, va="center", ha="left")
    path = save(fig, out_dir, f"rtq-queries-cpu{suffix}", formats, [legend], layout=False)
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
    draw_queries(rows, schedules, a.out, "-tuned" if a.tuned else "", a.format.split(","),
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
