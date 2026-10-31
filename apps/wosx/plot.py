#!/usr/bin/env python3
"""The wosx comparison as plots, from the results CSV benchmark.sh folds
its logs into (one row per measured cell: schedule, mesh, both rates in
steps per second, the speedup).

    python apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots/
    python apps/wosx/plot.py apps/wosx/results/wosx-gpu-results.csv -o apps/wosx/plots/ --gpu
    python apps/wosx/plot.py apps/wosx/results/wosx-stars-results.csv -o apps/wosx/plots/ --stars
    python apps/wosx/plot.py apps/wosx/results/wosx-stars-gpu-results.csv -o apps/wosx/plots/ --stars --gpu

Two CPU figures, one per tree, the meshes along the x axis in order of
triangle count and two bars a mesh; a bar is bonsai's rate over WoSX's
on that cell, the y axis reads Speedup, and the dashed line at 1, in the
legend, is WoSX:

  wosx-cpu-fcpw4w16-speedup  FCPW's four-wide tree, both sides vectorized
                             (`fcpw4w16*`)
  wosx-cpu-fcpwb-speedup     WoSX's default tree, FCPW's scalar binary BVH,
                             both sides as WoSX ships (`fcpwb*`)

The two bars, on either tree: the threads against the threads (`-threads`:
WoSX's tbb::parallel_for over the points against the program's parfor bound
to the same pinned threads, the same TBB); and sixteen persistent workers
against those same threads (`-persistent`: the workers claim points from a
counter; WoSX has no persistent CPU solver, so its side is the threaded
solve again). An application is measured in parallel under the schedule
the reference runs; one thread against one thread (the bare `fcpwb`,
`fcpw4w16`, a cell the tables once held) is a microbenchmark's question,
apps/rtq's, and is no bar (the user, 2026-10-28). On WoSX's default tree
the exploration's bars stand beside the two (CPU_EXTRA_BARS; the user,
2026-10-28: more schedules, vectorization among them): the same tree in the
layout where a node holds its children's boxes (`fcpwb2-threads`,
`fcpwb2-persistent`), the walk vectorized over a point's walks in gangs
of sixteen and eight lanes (`fcpwb-packet-16-threads`,
`fcpwb-packet-8-threads`; the gangs of thirty-two and of four are in the
table and not drawn), and the walks deferred step by step onto a queue
drained in gangs of sixteen, a point's queue and a chunk's queue sorted by
the walks' positions (`fcpwb-packet-defer-threads`,
`fcpwb-packet-defer-sort-threads`), each run in parallel under WoSX's
schedule.
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
consecutive claims neighbouring points. The matching schedule (`fcpwb-gpu`,
"persistent") does the same; the variants each change the schedule and
nothing else (--gpu-schedules names them; README.md's schedule list says
what each is): one launch with a thread per walk ("parallel"), a warp's
claims one point's walks ("persistent-reorder"), twice the threads so the
card is full ("persistent-256k", "persistent-reorder-256k"), one launch
with the walk fastest in blocks of 64 ("parallel-reorder-blk64"), and the
launch made through OptiX -- as it is ("optix"), with the threads
reordered by where their walks stand before every step, shader execution
reordering ("optix-ser"), and the persistent claim under that reordering
("optix-ser-persistent"). A point a claim with its walks in sequence
("persistent-point", `fcpwb-gpu-point`) was a bar until 2026-10-28: at a
tenth of WoSX's rate on every mesh it left most of the card's threads with
no point to claim, and the user had it removed; its rows stay in the CSV.

--stars draws the walk on stars' figures (benchmark.sh --stars): the same
comparison for wost.bonsai against WoSX's WalkOnStars, each mesh split at
the slice into an absorbing lower half and a reflecting upper half. On the
CPU, wosx-stars-cpu-fcpwb-speedup: WoSX's default trees -- FCPW's scalar
BVH of the absorbing boundary and its normal cone hierarchy of the
reflecting one -- and the same two bars (`wost-fcpwb*`); with --gpu,
wosx-stars-gpu-speedup: the matching schedule over WoSX's GPU walk on
stars (`wost-fcpwb-gpu`, the schedule of `fcpwb-gpu` on the other program)
and the walk on spheres' variants on this walk under the same labels, with
the per-point deferral beside them ("defer-block": a point per block, its
walks' steps on a queue in shared memory, a round running the next step of
every walk still going) and the compacting block wavefront ("defer-chunk2":
the same queue owned by a chunk of two points, 128 walks a block;
2026-10-30).
The mesh table is wosx-stars-meshes.*: the walk lengths differ from the
walk on spheres', since a star is bounded by the reflecting boundary's
silhouettes as well as by the absorbing boundary.

Beside each figure (the user's rule, 10-24) go the statistics the paper
quotes, `<figure>.txt`: per schedule one line beginning with the range
(least-greatest) and the geomean over the plotted meshes (the user, 10-30),
then the count and the least, quartile, median and greatest cells with
their meshes (figure_style.describe), what each change of schedule bought
cell by cell in the same form, and notes reading the figure; and the mesh
facts go in one table rather than on every plot, `wosx-meshes.txt` to read
and `wosx-meshes.tex` to \\input (a booktabs tabular: triangles, closed or
not, sample points, walks, WoSX's mean walk length), so the tick labels
are the mesh names alone.

The colours are apps/pbrt's figures' (the user's choice, 10-24 and
10-29): the paper's one ramp, ColorBrewer's YlGnBu, at eight equal
perceptual steps laid over each figure's bars in the legend's order,
palest first (figure_style.RAMP8; SCHEDULE_COLOURS below), `parallel`
and `persistent` the same shades on the CPU and GPU figures, no hatches,
the edge and the line at 1 as there. No titles: the file's name says which tree
and which side, the legend what each bar compares. The figures are drawn
at the size they are printed (apps/pbrt/figure_style.py has the format):
5.478 in wide, acmsmall's text block, what \\includegraphics[width=
\\textwidth] scales a figure to in the paper, 8 pt labels and 7 pt ticks in
Liberation Sans embedded as TrueType (pdf.fonttype 42, which ACM's
production accepts where matplotlib's default Type 3 fonts are refused),
the mesh names leaning only as far as eighteen of them must to clear each
other (figure_style.lean_labels), the legend above the axes in as many
columns as that width holds (legend_above; --width draws a wider figure).
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# The paper's figure style, apps/pbrt's module: acmsmall's text width, the
# font sizes, the YlGnBu ramp, Liberation Sans embedded as TrueType, and
# the summary written beside a figure.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrt"))
from figure_style import (RAMP8, TEXT_WIDTH, YLGNBU, describe, lean_labels,  # noqa: E402
                          paper_style, ratios, write_summary)

EDGE = "#333333"  # every bar's thin edge, as apps/pbrt's figures

# The schedules' colours, the scheme of apps/pbrt's figures (the user,
# 10-29: one palette across the paper, the same style on the wosx
# figures): the paper's YlGnBu ramp at eight equal perceptual steps
# (figure_style.RAMP8) laid over each figure's bars in the legend's
# order, palest first, darker with each step down; `parallel` and
# `persistent`, the two schedules both figures draw, keep their shades
# on the GPU figure (whose legend has them in the other order). The GPU
# figure's ninth default bar, optix-ser-persistent, takes the ramp's
# near-white class 0 past the eight, as it did before; the walk on
# stars' tenth, defer-block, a neutral grey. The CPU figures' gangs and
# the GPU figure's reorders are different schedules on the same shades:
# the two figures' legends name them, as they never share a figure.
SCHEDULE_COLOURS = {
    # the CPU figures' legend
    "parallel": RAMP8[0],
    "persistent": RAMP8[1],
    "parallel-fcpwb2": RAMP8[2],
    "persistent-fcpwb2": RAMP8[3],
    "parallel-packet16": RAMP8[4],
    "parallel-packet8": RAMP8[5],
    "parallel-defer16": RAMP8[6],
    "parallel-defer-sort16": RAMP8[7],
    # the GPU figures' legend, after persistent and parallel
    "persistent-reorder": RAMP8[2],
    "persistent-256k": RAMP8[3],
    "persistent-reorder-256k": RAMP8[4],
    "parallel-reorder-blk64": RAMP8[5],
    "optix": RAMP8[6],
    "optix-ser": RAMP8[7],
    "optix-ser-persistent": YLGNBU[0],
    "defer-block": "#9E9E9E",
    # the compacting block wavefront (2026-10-30): defer-block's queue
    # owned by a chunk of two points, the darker grey of the same family
    "defer-chunk2": "#616161",
    # drawn only when asked for (--gpu-schedules): shades borrowed
    "defer": YLGNBU[0],
    "persistent-point": "#9E9E9E",
}

# The CPU figures' bars, on either tree: (suffix of the tree's schedule,
# legend label, colour), in drawing order. The one-thread cell
# (the bare schedule, `fcpwb`: one thread against WoSX's runSingleThreaded)
# is no bar: an application is measured in parallel under the schedule the
# reference runs, and one thread against one thread is a microbenchmark's
# question, apps/rtq's (the user, 2026-10-28). Its rows stay in the CSV
# where they were measured; a table measured with it folds the same.
CPU_BARS = [
    ("-threads", "parallel", SCHEDULE_COLOURS["parallel"]),
    ("-persistent", "persistent", SCHEDULE_COLOURS["persistent"]),
]
# (schedule base, figure name, what the tree is): the two trees.
CPU_TREES = [
    ("fcpw4w16", "wosx-cpu-fcpw4w16-speedup",
     "FCPW's four-wide tree, both sides vectorized"),
    ("fcpwb", "wosx-cpu-fcpwb-speedup",
     "WoSX's default tree, FCPW's scalar binary BVH, both sides as WoSX ships"),
]
# The exploration's bars beside the two, per tree base: (schedule, legend
# label, colour, what it changes against the parallel bar), drawn after
# the two and only where the table has their rows. On WoSX's default tree
# (the user, 2026-10-28: more CPU schedules, vectorization among them):
# the same tree in the layout where each node's row holds its children's
# boxes, one row read a visit where FCPW's rows take three (fcpwb2*, WoSX's
# side unchanged on FCPW's own rows); and the walk vectorized over a
# point's walks -- the lanes are walks, the traversal one node for the
# gang, each child entered by the lanes whose query reaches it -- in gangs
# of sixteen lanes (the machine's vector width) and of eight (the user,
# 2026-10-28: "try packet4 and packet8"; the gangs of thirty-two and of
# four are measured and not drawn, below); and the walks deferred step by
# step onto a queue drained in gangs of sixteen (the user, 2026-10-29:
# "maybe a defer + vectorize would work"), a point's queue, which compacts
# the walks and changes nothing, and a chunk of points' queue sorted by the
# walks' positions, which gathers a gang's lanes in one cell of the
# problem's box and is worth 1.8x over the packet (PLAN.md, "Why the
# packets lose on the CPU": the lanes' spread over the tree, not the walks'
# lengths). The packet forms ran at a tenth of WoSX's rate one thread
# against one thread on 2026-10-09 (PLAN.md, "The packet over the
# points"): a point's walks spread over the mesh a few steps in, and a
# leaf any lane reaches is tested gang-wide for a lane or two.
CPU_EXTRA_BARS = {
    "fcpwb": [
        ("fcpwb2-threads", "parallel-fcpwb2", SCHEDULE_COLOURS["parallel-fcpwb2"],
         "the node's row holding its children's boxes (layouts/fcpwb2), the threads"),
        ("fcpwb2-persistent", "persistent-fcpwb2", SCHEDULE_COLOURS["persistent-fcpwb2"],
         "the node's row holding its children's boxes, the persistent workers"),
        ("fcpwb-packet-16-threads", "parallel-packet16", SCHEDULE_COLOURS["parallel-packet16"],
         "a point's walks vectorized in gangs of sixteen lanes, the threads"),
        ("fcpwb-packet-8-threads", "parallel-packet8", SCHEDULE_COLOURS["parallel-packet8"],
         "a point's walks vectorized in gangs of eight lanes, the threads"),
        ("fcpwb-packet-defer-threads", "parallel-defer16", SCHEDULE_COLOURS["parallel-defer16"],
         "a point's walks deferred step by step onto the point's queue, the queue drained "
         "in gangs of sixteen lanes (no lane idles behind a longer walk), the threads"),
        ("fcpwb-packet-defer-sort-threads", "parallel-defer-sort16",
         SCHEDULE_COLOURS["parallel-defer-sort16"],
         "the walks deferred onto a chunk of sixty-four points' queue, drained in gangs of "
         "sixteen in the order of the walks' positions (a Morton code), the threads"),
    ],
}
# Measured but not drawn by default, the ramp's eight shades spoken for
# and the table's rows keeping them: fcpwb-packet-threads (gangs of
# thirty-two, a third under the gangs of sixteen everywhere: two vector
# instructions for one) and fcpwb-packet-4-threads (0.57x over WoSX
# against the gangs of eight's 0.56x: below the register's width the
# gang's width does not matter). Both are drawn by naming them in
# CPU_EXTRA_BARS["fcpwb"] with a colour.
# The walk on stars' CPU figure (--stars): WoSX's default trees alone, the
# same two bars on the `wost-fcpwb*` schedules.
STARS_CPU_TREES = [
    ("wost-fcpwb", "wosx-stars-cpu-fcpwb-speedup",
     "the walk on stars on WoSX's default trees (FCPW's scalar BVH of the absorbing boundary, "
     "its normal cone hierarchy of the reflecting one), both sides as WoSX ships"),
]
# The GPU figure's bars, every one over WoSX's persistent-thread kernel:
# (schedule, legend label, colour), the labels apps/pbrt's figures' kind --
# the schedule's shape in a word or two; README.md's schedule list says
# what each is. Schedules only: the other layout and the other tree
# (fcpwb2-gpu, fcpw4w16-gpu) are a sweep of their own, not this figure's.
GPU_BARS = [(s, label, SCHEDULE_COLOURS[label]) for s, label in [
    ("fcpwb-gpu", "persistent"),
    ("fcpwb-gpu-parallel", "parallel"),
    ("fcpwb-gpu-reorder", "persistent-reorder"),
    ("fcpwb-gpu-256k", "persistent-256k"),
    ("fcpwb-gpu-reorder-256k", "persistent-reorder-256k"),
    ("fcpwb-gpu-parallel-reorder-blk64", "parallel-reorder-blk64"),
    # The launch through OptiX: the plain launch as a ray-generation
    # program (a control), the same with the threads reordered by where
    # their walks stand before every step (shader execution reordering,
    # `walk.sort(walk, ...)`), and the persistent claim under that
    # reordering.
    ("fcpwb-gpu-optix", "optix"),
    ("fcpwb-gpu-optix-ser", "optix-ser"),
    ("fcpwb-gpu-optix-ser-persistent", "optix-ser-persistent"),
    # The two deferrals: the walks' steps on a queue, a round of the drain
    # running the next step of every walk still going -- one queue across
    # the card, or one per point in the block's shared memory. Measured on
    # 2026-10-25 and not in the table's defaults since; drawn where the
    # table has their rows.
    ("fcpwb-gpu-defer", "defer"),
    ("fcpwb-gpu-defer-block", "defer-block"),
    # The compacting block wavefront (the user, 2026-10-30): defer-block's
    # queue owned by a chunk of two points, 128 walks a block -- the one
    # chunk size that came near the one-point block; the chunks of four,
    # eight and sixteen fall further behind (PLAN.md, "The compacting
    # block wavefront") and are measured by name, not drawn.
    ("fcpwb-gpu-defer-chunk2", "defer-chunk2"),
    # A point a claim, its walks in sequence: a tenth of WoSX's rate on
    # every mesh, the card left nearly empty; removed from the table's
    # defaults and the figure on 2026-10-28 (the user). Named here so that
    # a table carrying its rows is drawn with a label, if asked for by
    # --gpu-schedules.
    ("fcpwb-gpu-point", "persistent-point"),
]]
# The walk on stars' GPU figure (--stars --gpu): the matching schedule over
# WoSX's GPU walk on stars, the same persistent claim of a walk at a time,
# and the walk on spheres' variants on this walk under the same labels
# (wost-fcpwb-gpu-*.bonsai: each the same directives on solve_stars), with
# the per-point deferral, which a step of three traversals should reward
# more than the one-query walk did.
STARS_GPU_BARS = [(s, label, SCHEDULE_COLOURS[label]) for s, label in [
    ("wost-fcpwb-gpu", "persistent"),
    ("wost-fcpwb-gpu-parallel", "parallel"),
    ("wost-fcpwb-gpu-reorder", "persistent-reorder"),
    ("wost-fcpwb-gpu-256k", "persistent-256k"),
    ("wost-fcpwb-gpu-reorder-256k", "persistent-reorder-256k"),
    ("wost-fcpwb-gpu-parallel-reorder-blk64", "parallel-reorder-blk64"),
    ("wost-fcpwb-gpu-optix", "optix"),
    ("wost-fcpwb-gpu-optix-ser", "optix-ser"),
    ("wost-fcpwb-gpu-optix-ser-persistent", "optix-ser-persistent"),
    ("wost-fcpwb-gpu-defer", "defer"),
    ("wost-fcpwb-gpu-defer-block", "defer-block"),
    ("wost-fcpwb-gpu-defer-chunk2", "defer-chunk2"),
]]
# What each GPU schedule changed against another, for the summary: (label
# of the schedule, label of the one it differs from in one thing, the
# thing). The labels are the same on both walks' figures.
GPU_STEPS = [
    ("parallel", "persistent", "one launch, a thread per walk, instead of the persistent claim"),
    ("persistent-reorder", "persistent", "a warp's claims one point's walks instead of neighbouring points"),
    ("persistent-256k", "persistent", "twice the threads, the card full"),
    ("persistent-reorder-256k", "persistent-reorder", "twice the threads with the walk-fastest claims"),
    ("parallel-reorder-blk64", "parallel", "the walk fastest and blocks of 64 in the one launch"),
    ("optix", "parallel", "the same one launch made through OptiX as a ray-generation program, no reordering"),
    ("optix-ser", "optix", "the threads reordered by the cell their walks stand in before every step (shader execution reordering)"),
    ("optix-ser-persistent", "optix-ser", "the persistent claim of a walk at a time under the reordering"),
    ("optix-ser-persistent", "persistent", "the reordering alone, both with the persistent claim"),
    ("persistent-point", "persistent", "a point a claim, its walks in sequence"),
    ("defer", "parallel", "the walk's steps deferred onto one card-wide queue, a launch per round"),
    ("defer-block", "defer", "a queue per point in the block's shared memory, the rounds inside the kernel"),
    ("defer-block", "persistent", "a point per block, its walks' steps on a queue in shared memory, against the persistent claim"),
    ("defer-chunk2", "defer-block", "the block's queue owned by a chunk of two points, 128 walks a block instead of 64"),
    ("defer-chunk2", "persistent", "the compacting block wavefront of two points' walks, against the persistent claim"),
]
REFERENCE = "WoSX"


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


def points(data, meshes):
    """The (mesh, speedup) pairs of one schedule in mesh order."""
    return [(m, data[m]) for m in meshes if m in data]


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


def figure(rows, meshes, bars, ylabel, height, fig_width=TEXT_WIDTH):
    """One figure: a bar per (schedule, label, colour) of `bars` per mesh, the
    dashed line at 1 as WoSX in the legend above the axes, the meshes by
    name, `fig_width` inches wide (the paper's text block unless --width
    says otherwise). Returns the figure and the schedules drawn, or (None, [])."""
    data = {s: cells(rows, s) for s, _, _ in bars}
    drawn = [(s, label, colour) for s, label, colour in bars if data[s]]
    if not drawn:
        return None, []
    paper_style()
    fig, ax = plt.subplots(figsize=(fig_width, height))
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
    line = ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3,
                      label=REFERENCE)
    ax.set_xticks(range(len(meshes)))
    # The names lean only as far as they must to clear their neighbours
    # (the user, 10-24: no needless whitespace under the axes); the pitch
    # is the axes' width less what the y label takes, over the meshes.
    lean_labels(ax, meshes, (fig_width - 0.6) / len(meshes))
    ax.set_xlim(-0.6, len(meshes) - 0.4)
    ax.set_ylim(0, top * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The reference line leads the legend, then a patch per schedule, as
    # apps/pbrt's figures have it: one row up to five entries, else as many
    # columns as the figure is wide for.
    handles = [Line2D([], [], color=line.get_color(), linestyle=line.get_linestyle(),
                      linewidth=line.get_linewidth(), label=REFERENCE)]
    handles += [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.4, label=label)
                for _, label, colour in drawn]
    rows_above = legend_above(fig, ax, handles)
    # A legend of more than two rows takes its extra rows from the height,
    # not from the axes.
    fig.set_size_inches(fig_width, height + 0.16 * max(0, rows_above - 2))
    return fig, [s for s, _, _ in drawn]


def legend_above(fig, ax, handles):
    """The legend above the axes, measured to fit: one row up to five
    entries, else the most columns whose row is no wider than the figure --
    the layout pass does not see a legend outside the axes, and the nine
    GPU schedules in five columns ran past the page's right edge in the PDF
    (the user, 10-27). Returns the number of rows; save() makes room above
    the axes for them."""
    ncol = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    while True:
        legend = ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
                           bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0, columnspacing=1.2)
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        if box.width <= 0.97 * fig.bbox.width or ncol == 1:
            return math.ceil(len(handles) / ncol)
        ncol -= 1


def save(fig, out_dir, name, formats):
    """The page exactly the figure's size (no cropping to the content, which
    would leave LaTeX to rescale it): the layout fits the labels and the
    legend inside it. Returns the path of the first format written."""
    fig.tight_layout(pad=0.3)
    # The legend above the axes is outside what tight_layout lays out: the
    # axes' top comes down by whatever the legend overhangs the page.
    legend = fig.axes[0].get_legend()
    if legend is not None:
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        over = (box.y1 - fig.bbox.height) / fig.bbox.height + 0.02
        if over > 0:
            fig.subplots_adjust(top=fig.subplotpars.top - over)
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


def mesh_table(rows, meshes, out_dir, exclude, name="wosx-meshes"):
    """The mesh facts in one table rather than on every plot (the user,
    10-24): wosx-meshes.txt to read and wosx-meshes.tex to \\input, a
    booktabs tabular (wosx-stars-meshes.* for the walk on stars, whose walk
    lengths are its own). Triangles, whether the surface is closed, the
    sample points on the slice and their walks, and WoSX's mean walk length,
    read off the first row of each mesh (the same on every schedule)."""
    first = {}
    for r in rows:
        first.setdefault(r["mesh"], r)
    columns = [("mesh", lambda r: r["mesh"]), ("triangles", lambda r: int(r["triangles"])),
               ("closed", lambda r: "yes" if r["closed"] == "1" else "no"),
               ("points", lambda r: int(r["points"])), ("walks", lambda r: int(r["walks"])),
               ("walk length", lambda r: f"{float(r['wosx_walk_length']):.1f}")]
    table = [[f(first[m]) for _, f in columns] for m in meshes]

    def text(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    txt = os.path.join(out_dir, f"{name}.txt")
    with open(txt, "w") as f:
        widths = [max(len(h), *(len(text(row[i])) for row in table))
                  for i, (h, _) in enumerate(columns)]
        f.write("  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                          for i, ((h, _), w) in enumerate(zip(columns, widths))) + "\n")
        for row in table:
            f.write("  ".join(text(v).ljust(w) if i == 0 else text(v).rjust(w)
                              for i, (v, w) in enumerate(zip(row, widths))) + "\n")
        f.write("\nwalk length: WoSX's mean number of steps of a walk that ended, 64 walks a "
                "point, epsilon 0.001\n")
        if exclude:
            f.write(f"left out of the figures: {', '.join(sorted(exclude))}\n")
    print(txt)
    tex = os.path.join(out_dir, f"{name}.tex")
    with open(tex, "w") as f:
        f.write(f"% apps/wosx/plot.py: the meshes of the {name.replace('-meshes', '')} figures, in "
                "order of triangle\n"
                "% count; walk length is WoSX's mean number of steps of a walk that ended.\n"
                "% \\usepackage{booktabs}.\n")
        f.write("\\begin{tabular}{l" + "r" * (len(columns) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(h for h, _ in columns) + " \\\\\n\\midrule\n")
        for row in table:
            f.write(" & ".join(tex_number(v) if isinstance(v, int) else str(v) for v in row)
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(tex)


def cpu_notes(base, exclude):
    notes = [
        "parallel: WoSX's tbb::parallel_for over the points against the parfor bound to the "
        "same pinned threads; persistent: sixteen workers claiming points from a counter, "
        "against WoSX's threaded solve (it has no persistent CPU solver); one thread against "
        "one thread is no bar: an application is measured in parallel under the reference's "
        "schedule, the single thread's question is apps/rtq's (the user, 2026-10-28)",
    ]
    if base.startswith("wost-"):
        notes += [
            "the walk on stars: a step asks three queries where the walk on spheres asks one "
            "-- the closest silhouette edge of the reflecting boundary on FCPW's normal cone "
            "hierarchy, which bounds the star, the ray cast against the reflecting boundary on "
            "the same tree, and the distance to the absorbing boundary on its BVH; the mesh is "
            "split at the slice, the lower half absorbing with the Dirichlet data, the upper "
            "half reflecting with zero flux, WoSX's defaults for the reflecting shell and the "
            "silhouette precision (1e-3) on both sides",
            "no closed-form solution with a reflecting boundary: the table's error column is "
            "the rms difference between the two sides' estimates",
        ]
    elif base == "fcpwb":
        notes += [
            "the two-child visit is FCPW's since 2026-10-25 -- two box tests, "
            "one compare, a swap, the far child pushed and the near one entered (apps/wosx/"
            "PLAN.md, 'The binary node's visit made FCPW's'); what is left between the sides "
            "is in the generated code of the closest-point query itself, read in apps/rtq/"
            "PLAN.md, 'What is left between the sides, as code' (vec3f as LLVM's <3 x float> "
            "in scalar code, the far-corner key's redundant distmin, the eager pIndex load); "
            "the 2026-10-24 reading of a 130-instruction visit ('The binary CPU walk's gap, "
            "read from the machine') describes the visit that change removed",
            "the threaded and persistent forms are ahead on every mesh but the comb",
        ]
    else:
        notes += [
            "FCPW's vectorized build (enoki, FCPW_USE_ENOKI) on both sides; the tree WoSX's "
            "boundary handler builds when asked to vectorize",
        ]
    if exclude:
        notes.append(f"left out: {', '.join(sorted(exclude))} (nearly every walk escapes on its "
                     "first step; the bar measures an escaping walk, not the traversal)")
    return notes


def cpu_bars(base):
    """The two bars of a tree and, where the exploration has them, its extra
    bars: (schedule, label, colour) in drawing order."""
    bars = [(base + suffix, label, colour) for suffix, label, colour in CPU_BARS]
    bars += [(s, label, colour) for s, label, colour, _ in CPU_EXTRA_BARS.get(base, [])]
    return bars


def draw_cpu(rows, out_dir, formats, exclude, trees, width=TEXT_WIDTH):
    meshes, tris = meshes_by_triangles(rows, exclude)
    for base, name, what in trees:
        bars = cpu_bars(base)
        fig, drawn = figure(rows, meshes, bars, "Speedup", height=1.7, fig_width=width)
        if fig is None:
            print(f"{name}: no rows for {base}'s schedules; skipped", file=sys.stderr)
            continue
        path = save(fig, out_dir, name, formats)
        data = {s: cells(rows, s) for s in drawn}
        label = {s: lab for s, lab, _ in bars}
        sections = [("every mesh, over WoSX", [(label[s], points(data[s], meshes)) for s in drawn])]
        notes = cpu_notes(base, exclude)
        threads, persistent = base + "-threads", base + "-persistent"
        if threads in data and persistent in data:
            notes.append("the persistent workers over TBB's partition, the same reference: "
                         + describe(ratios(points(data[persistent], meshes),
                                           points(data[threads], meshes))))
        # Each exploration bar against the parallel one: what it changes,
        # and what the change bought cell by cell.
        for s, lab, _, thing in CPU_EXTRA_BARS.get(base, []):
            if s in data and threads in data:
                notes.append(f"{thing} ({lab} over parallel): "
                             + describe(ratios(points(data[s], meshes),
                                               points(data[threads], meshes))))
        print(write_summary(path, f"Speedup over WoSX on the CPU, {what}, {len(meshes)} meshes",
                            sections, notes))


def draw_gpu(rows, schedules, out_dir, formats, exclude, stars, width=TEXT_WIDTH):
    meshes, tris = meshes_by_triangles(rows, exclude)
    known = {s: (label, colour) for s, label, colour in (STARS_GPU_BARS if stars else GPU_BARS)}
    bars = [(s, *known.get(s, (s, RAMP8[-1]))) for s in schedules]
    # A little taller: seven bars a mesh and a legend of two rows above them.
    fig, drawn = figure(rows, meshes, bars, "Speedup", height=2.0 if len(bars) > 5 else 1.7,
                        fig_width=width)
    if fig is None:
        print(f"no rows for {schedules}; nothing drawn", file=sys.stderr)
        return
    path = save(fig, out_dir, "wosx-stars-gpu-speedup" if stars else "wosx-gpu-speedup", formats)
    data = {s: cells(rows, s) for s in drawn}
    label = {s: known.get(s, (s, None))[0] for s in drawn}
    by_label = {label[s]: points(data[s], meshes) for s in drawn}
    sections = [("every mesh, over WoSX's persistent kernel", [(label[s], by_label[label[s]]) for s in drawn])]
    if stars:
        notes = [
            "persistent is the matching schedule of the walk on stars (131072 threads claiming "
            "a walk at a time from a counter, a warp's claims neighbouring points), over WoSX's "
            "GPU walk on stars run the same way; a step is the three queries of the CPU figure, "
            "the distance bounded on both sides by 1.001 times the old distance plus the step; "
            "the others change one thing of it each, the walk on spheres' variants on this walk",
            "no closed-form solution with a reflecting boundary: the error column is the rms "
            "difference between the two sides' estimates",
        ]
    else:
        notes = [
            "persistent is the matching schedule (131072 threads claiming a walk at a time from "
            "a counter, a warp's claims neighbouring points); the others change one thing of it "
            "each",
        ]
    for over, base, thing in GPU_STEPS:
        if over in by_label and base in by_label:
            notes.append(f"{thing} ({over} over {base}): "
                         + describe(ratios(by_label[over], by_label[base])))
    best = {}
    for m in meshes:
        here = [(lab, dict(pts).get(m)) for lab, pts in by_label.items() if m in dict(pts)]
        if here:
            best[m] = max(here, key=lambda p: p[1])[0]
    counts = {}
    for m, lab in best.items():
        counts.setdefault(lab, []).append(m)
    for lab, ms in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        notes.append(f"{lab} is the best of the {len(drawn)} on {len(ms)} of {len(meshes)} meshes"
                     + (f" ({', '.join(ms)})" if len(ms) <= 4 else ""))
    if stars:
        if exclude:
            notes.append(f"left out: {', '.join(sorted(exclude))} (nearly every walk escapes on "
                         "its first step)")
        print(write_summary(path, f"Speedup over WoSX's GPU walk on stars, {len(meshes)} meshes, "
                                  f"{len(drawn)} schedules of the one program", sections, notes))
        return
    notes += [
        "crown and beast have fewer walks (29440, 86016) than the kernel has threads, so a "
        "thread claims one walk and the card runs nearly empty (15-25% achieved occupancy "
        "against 32-49% on a full mesh, 2026-10-24); the kernel's time is the longest walks' "
        "latency on both sides, and the two are the lowest bars of most schedules; with 1024 "
        "walks a point they were 1.16x and 1.24x on 2026-10-24. apps/wosx/PLAN.md, 'The crown "
        "and the beast'",
        "on every mesh both kernels keep about three of a warp's 32 lanes active per "
        "instruction (ncu, 2026-10-24, head and dragon: WoSX 3.1 and 2.4, ours 2.8 and 2.1): a "
        "warp holds its 32 walks until the longest ends, and the lanes' traversals diverge "
        "within a step; our lead is in the registers (64 against 82, 49% occupancy against "
        "32%), not the lanes. apps/wosx/PLAN.md, 'The lanes of a warp'",
    ]
    if "persistent-point" in by_label:
        notes.append("persistent-point: a point a claim leaves 460 to 16348 threads busy of "
                     "131072, an empty card")
    if exclude:
        notes.append(f"left out: {', '.join(sorted(exclude))} (nearly every walk escapes on its "
                     "first step)")
    print(write_summary(path, f"Speedup over WoSX's GPU solver, {len(meshes)} meshes, "
                              f"{len(drawn)} schedules of the one program", sections, notes))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figures")
    p.add_argument("--gpu", action="store_true",
                   help="draw the GPU figure: the GPU schedules over WoSX's GPU solver")
    p.add_argument("--gpu-schedules", default=None,
                   help="the GPU schedules, in the order of their bars (default: the matching "
                        "schedule and its variants, as benchmark.sh --gpu measures them; with "
                        "--stars the walk on stars' matching schedule)")
    p.add_argument("--stars", action="store_true",
                   help="draw the walk on stars' figures (the wost-* schedules) instead")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="landscape",
                   help="comma-separated meshes left out (default landscape, whose walks escape "
                        "on their first step)")
    p.add_argument("--width", type=float, default=TEXT_WIDTH,
                   help="the figure's width in inches (default %(default)s, the paper's text "
                        "block; wider for a slide or a look -- LaTeX scales a wider figure down "
                        "to the text width, its text with it)")
    a = p.parse_args()
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    trees = STARS_CPU_TREES if a.stars else CPU_TREES
    if a.gpu:
        gpu_bars = STARS_GPU_BARS if a.stars else GPU_BARS
        # The figure's bars by default: every schedule of the list but the
        # retired one, each drawn where the table has its rows.
        named = (a.gpu_schedules if a.gpu_schedules is not None
                 else ",".join(s for s, _, _ in gpu_bars if s != "fcpwb-gpu-point"))
        schedules = [s for s in named.split(",") if s]
        draw_gpu(rows, schedules, a.out, a.format.split(","), exclude, a.stars, a.width)
    else:
        schedules = [s for base, _, _ in trees for s, _, _ in cpu_bars(base)]
        draw_cpu(rows, a.out, a.format.split(","), exclude, trees, a.width)
    meshes, _ = meshes_by_triangles(rows, exclude)
    mesh_table(rows, meshes, a.out, exclude, "wosx-stars-meshes" if a.stars else "wosx-meshes")
    geomeans(rows, schedules, exclude)


if __name__ == "__main__":
    main()
