#!/usr/bin/env python3
"""The CPU wavefront study as one figure: every schedule's speedup over
pbrt's CPU wavefront (`pbrt --wavefront` without `--gpu`), a group of bars
per scene, a subgroup per sample count, from the CSV cpu_compare.sh writes.

    python3 apps/pbrt/plot_cpu_wavefront.py apps/pbrt/cpu-compare-out/cpu-results.csv \
        [apps/pbrt/cpu-compare-out-sortfix/cpu-results.csv ...] \
        -o apps/pbrt/plots/pbrt-cpu-sweep.pdf

(plots.sh draws it under that name, the figure's one name since 10-24.)

A CSV has one row per measured cell and side (scene, spp, side, render
seconds, speedup, verdict, ..., path; cpu_compare.sh's header says what
each column is). A scene is keyed by its `path` under the scene set
(`kroken/camera-1`: two scenes share the name camera-1) where the CSV has
the column, else by its name, and labelled on one line by
figure_style.scene_label (the directory for a camera or view number, the
short forms for sanmiguel-courtyard and bistro_vespa; the user, 10-29: no
name broken over two lines). The side `pbrt-wavefront` is the reference
and draws no bar: the dashed line at 1 is it. Several CSVs are several
runs joined: a later file's row replaces an earlier one's for the same
(scene, spp, side), and
every side's bar is its speedup over the pbrt that ran in the same CSV
(the sorted schedules were re-measured on 10-24 with the ray key fixed,
into a CSV of their own). The other sides are drawn in the order of SIDES
below -- pbrt's megakernel first, then this renderer's schedules named by
the encoding SIDES explains (type-parallelism-SIMD-order[-sort]: mk-b-s-ps
is the path-per-sample `scalar` schedule, wf-b-v-sp-sort the band
wavefront with vectorized drains and the rays sorted, wf-t-v-ps the
per-thread vectorized wavefront). `--sides` names a subset (comma-separated
side names) and keeps that order, so the five-bar figure of the first study is
`--sides pbrt-volpath,scalar,wavefront-frame,wavefront-frame-packet,wavefront-frame-packet-reorder`.

pbrt's bar is black and the Bonsai sides take the paper's one ramp
(ColorBrewer YlGnBu, the heatmap's, the rtq and wosx figures') at eight
equal perceptual steps from its class 1 to its class 8, in the legend's
order, the GPU sweep laid the same way (figure_style.SCHEDULE_COLOURS;
the user, 10-29): the palest shade the plain megakernel on both figures
(mk-b-s-ps here, `mk` there), the darkest the sorted wavefront
(wf-t-v-ps-sort here, `wf-sort` there), every step down a legend a step
darker; the shades between name each figure's own schedules. Every bar has a
thin dark edge so the palest shades stay visible on white. The figure does not
mark the image checks: the CSV's verdict column has them, and the one that
ever failed (sanmiguel 16 spp against pbrt's volpath) was pbrt's own two
integrators disagreeing by more than we differ from either, which the user
ruled fine (10-23). The bars carry no numbers (the user, 10-22: they made
the figure noisy; the CSV has them; `--values` puts them back). `--width`
and `--height` are the figure size in inches; the defaults are the paper's
text width (acmsmall, 5.478 in; figure_style.py has the format, the fonts
and their sizes, so a figure included at `\\textwidth` is not rescaled and
its 8 pt labels stay 8 pt) by 2.4 in, the legend in two rows above the
axes: the dashed reference line first, then the sides (ten entries, five
over five; the user's layout, 10-24). The legend labels are the schedule
codes alone -- pbrt's two carry "PBRT", everything else is Bonsai's, as the
caption says -- so that the rows fit at text width; the y axis reads just
"Speedup", over the dashed line (the user, 10-24). `--log`
puts the speedup axis on a log scale, where the line at 1 sits mid-axis.
The output format follows the extension (.pdf for the paper, .png to look
at); fonts are embedded as TrueType (fonttype 42, as ACM requires).

The drawing is `draw`, shared with plot_gpu_wavefront.py, which draws the
GPU study's table the same way with its own sides and reference.
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from figure_style import (PBRT_COLOUR, SCHEDULE_COLOURS, TEXT_WIDTH, TICK_FONT,  # noqa: E402
                          describe, geomean, paper_style, ratios, scene_label,
                          write_summary)

REFERENCE = "pbrt-wavefront"
REFERENCE_LABEL = "PBRT wf"
Y_LABEL = "Speedup"          # over PBRT wf, the dashed line, named in the legend (10-24)
EDGE = "#333333"             # every bar's thin edge

# (side in the CSV, label in the legend, colour): the drawing order. The
# colour is the schedule's on every figure (figure_style.SCHEDULE_COLOURS,
# keyed by the label).
# pbrt has two schedules and is named by them: its megakernel (the CPU
# volpath integrator, a path per sample) and its wavefront, the reference.
# A Bonsai schedule is named by the user's encoding (10-22),
# type-parallelism-SIMD-order[-sort]: type `wf` (wavefront) or `mk`
# (megakernel, a path per sample); parallelism `b` (bulk: the frame's
# pixels shared by the threads, as pbrt's bands and a megakernel's pixel
# loop are) or `t` (per-thread: a thread owns a chunk of pixels and its
# queues); SIMD `s` (scalar drains) or `v` (vectorized, sixteen-wide gangs);
# order `ps` (pixels outer, samples inner, the program's order) or `sp`
# (samples outer, pixels inner: `reorder(p, s)` or `reorder(p_in, s)`);
# `-sort` when the rays queue is sorted by the two-point Morton key before
# each trace pass. The code is read from the schedule file's directives.
SIDES = [(side, label, SCHEDULE_COLOURS[label]) for side, label in [
    # pbrt's own megakernel first, then the schedule that matches pbrt's
    # wavefront, its modifications by increasing departure (the band's
    # drains vectorized, then sorted; the per-thread queues, then their
    # drains vectorized, the samples-outer order, the sort), and this
    # renderer's megakernel last (the user, 10-30 evening).
    ("pbrt-volpath", "PBRT mk"),            # black (figure_style.PBRT_COLOUR)
    ("wavefront-frame", "wf-b-s-sp"),
    ("wavefront-frame-packet", "wf-b-v-sp"),
    ("wavefront-frame-packet-reorder", "wf-b-v-sp-sort"),
    ("wavefront-thread", "wf-t-s-ps"),
    ("wavefront-thread-packet", "wf-t-v-ps"),
    ("wavefront-thread-packet-reorder", "wf-t-v-sp"),
    ("wavefront-thread-packet-sort", "wf-t-v-ps-sort"),
    ("scalar", "mk-b-s-ps"),
]]
assert SIDES[0][2] == PBRT_COLOUR


def load(paths, reference=REFERENCE):
    """cpu_compare.sh's CSVs as rows with `scene`, `key` (the scene's path
    under the scene set where the CSV has one, else its name), `spp`
    (int), `side`, `seconds` (float), `verdict` and `reference` (the
    seconds of the reference side of the same cell in the same CSV); a row
    whose run produced no time draws nothing and is dropped. Several CSVs
    are several runs joined, a later file's row replacing an earlier one's
    for the same (scene, spp, side) -- a schedule re-measured after a fix,
    as the sorted CPU schedules were on 10-24 -- and every side measured
    against the reference that ran beside it."""
    joined = {}
    order = []
    for path in paths:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        table = []
        for r in rows:
            try:
                r["spp"] = int(r["spp"])
                r["seconds"] = float(r["seconds"])
            except (ValueError, KeyError):
                continue
            r["key"] = r.get("path") or r["scene"]
            table.append(r)
        references = {}
        for r in table:
            if r["side"] == reference:
                key = (r["key"], r["spp"])
                references[key] = min(r["seconds"], references.get(key, r["seconds"]))
        for r in table:
            key = (r["key"], r["spp"], r["side"])
            r["reference"] = references.get((r["key"], r["spp"]))
            if key not in joined:
                order.append(key)
            joined[key] = r
    return [joined[key] for key in order]


def ordered(values):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


# Pairs of sides that differ by one scheduling decision, and what it is:
# the summary reports what each bought, bar by bar (the user, 10-24: the
# statistics should explain the speedups where they can).
COMPARISONS = [
    ("mk-b-s-ps", "PBRT mk", "our megakernel against pbrt's (same algorithm, our codegen)"),
    ("wf-b-s-sp", "PBRT mk", "pbrt's wavefront schedule, matched, against its megakernel"),
    ("wf-b-v-sp", "wf-b-s-sp", "vectorizing the drains of the shared-band wavefront"),
    ("wf-b-v-sp-sort", "wf-b-v-sp", "sorting the rays before each trace (shared bands)"),
    ("wf-t-s-ps", "wf-b-s-sp", "per-thread queues instead of frame-wide bands (scalar drains)"),
    ("wf-t-v-ps", "wf-t-s-ps", "vectorizing the drains of the per-thread wavefront"),
    ("wf-t-v-sp", "wf-t-v-ps", "samples-outer order instead of pixels-outer (per-thread, vectorized)"),
    ("wf-t-v-ps-sort", "wf-t-v-ps", "sorting the rays before each trace (per-thread, vectorized)"),
    ("wf-t-v-ps", "mk-b-s-ps", "the best wavefront against our megakernel"),
]
# Why the schedules stand where they do, as measured (the user, 10-29: the
# .txt should say why different schedules are best on different scenes).
# First the mechanisms, then each scene's reading; `draw` writes the
# computed best side of every scene and sample count above each reading,
# so a reading that no longer matches its numbers shows. A scene without
# a reading is marked as such (keyed by the scene's label on the figure,
# figure_style.scene_label). The sections of apps/pbrt/PLAN.md these
# come from are named at the end.
WHY = [
    "what pbrt's CPU wavefront pays: frame-wide bands whose queues every thread shares -- it runs",
    "  2.1-6.7x slower than pbrt's own megakernel (PBRT mk), least on bistro and most on",
    "  killeroo-gold. The per-thread wavefront (wf-t-*) gives each thread its own chunk of pixels and",
    "  its own queues: 1.8-4.8x over the same drains on shared bands (wf-t-s-ps over wf-b-s-sp,",
    "  geomean 2.8x), the one decision that moves the figure; it buys most where pbrt's wavefront",
    "  loses most to its megakernel (killeroo-gold, book) and least where the drains dominate",
    "  (bistro, dragon_10). Our megakernel (mk-b-s-ps) beats pbrt's by 1.04-1.39x on the same",
    "  algorithm: our codegen.",
    "the drains vectorized (v over s) pay where the shading is heavy (bistro 29-30%, sanmiguel",
    "  22-27%, book 13-18%, dragon_10 14-15%, ganesha 8-13% in the per-thread form) and nothing on",
    "  killeroo-gold; the samples-outer order (sp over ps) is level or behind (0.87-1.00x); the ray",
    "  sort before each trace pays where the rays are incoherent and the shading heavy (bistro",
    "  1.11-1.18x, sanmiguel 1.02-1.06x per thread; 1.12-1.13x on the bands of both) and costs where",
    "  a thread's pass is large and the shading cheap (killeroo-gold, ganesha and book at 128 spp,",
    "  0.87-0.90x), since a chunk x spp queue sorted and read through the permutation leaves the",
    "  core's cache.",
    "the megakernel wins killeroo-gold at both counts and dragon_10 by 1-3%: a path's state stays in",
    "  registers where the wavefront writes and reads back its queues each stage -- on killeroo-gold",
    "  the shading is too cheap to hide that; on dragon_10 the subsurface bounce's probe trace and",
    "  second material evaluation are two more queue passes for the wavefront and straight-line code",
    "  for the megakernel (a reading; not profiled). The GPU's reading of dragon_10 (its megakernels",
    "  0.3-0.9x pbrt: lanes diverging under the subsurface sample) has no counterpart with one path a",
    "  thread. A smaller chunk at high spp stays the untested remedy for the wavefront (10-23).",
    "with more samples the speedups barely move (the wavefronts 0.97-1.01x from 16 to 128 spp, the",
    "  per-thread sort's 0.95x the most; the megakernels 0.99-1.00x), where every GPU speedup falls",
    "  to 0.75-0.80x (pbrt-gpu-sweep.txt).",
]
READINGS = {
    # killeroo-simple is 10-28's five-scene table's (cpu-compare-out-head,
    # 16/64/128 spp), out of the figure since 10-30; its reading is kept for
    # that table.
    "killeroo-simple": [
        "two materials (CoatedDiffuse, Diffuse) and the cheapest trace of the five (66k triangles):",
        "vectorizing the per-thread drains buys nothing (wf-t-v-ps over wf-t-s-ps 0.98-1.03x), and at 64",
        "and 128 spp the chunk's queues outgrow the cache where the megakernel's path-at-a-time state",
        "does not, so our megakernel takes those cells and the per-thread wavefronts edge it at 16 spp.",
        "The band's sort pays (1.83x against 1.77x); the per-thread sort costs most here at 128 spp",
        "(0.83x: 128k keys a pass per thread).",
    ],
    "killeroo-gold": [
        "Conductor and Diffuse over 532k triangles, one area light: the cheapest shading of the six, so",
        "the per-thread drains' vectorization is level (0.99-1.00x) and the per-thread sort costs most",
        "here (0.87x at 128 spp); pbrt's wavefront loses most to its megakernel (6.5-6.7x) and the",
        "per-thread queues buy most (4.6-4.8x over the bands); our megakernel is best at both counts",
        "(1.14-1.19x pbrt's megakernel) -- the one scene of the six where the shading never hides the",
        "wavefront's queue traffic.",
    ],
    "book": [
        "textured CoatedDiffuse pages over 21k triangles: heavy shading per hit, so vectorizing the",
        "per-thread drains pays 13-18% and the per-thread wavefront takes both cells (the sorted one at",
        "16 spp by 2%; the sort costs 10% at 128); pbrt's own megakernel is weakest here (3.7-3.8x over",
        "its wavefront) and ours beats it by 1.38-1.39x, the most of the six.",
    ],
    "ganesha": [
        "one statue of 4.3M triangles in 6 meshes, CoatedDiffuse and Diffuse, 5 lights: the per-thread",
        "vectorized drains pay 8-13% and win both cells; the band's vectorized drains lose 9% here, the",
        "one scene (on the list for perf stat, not done); the samples-outer order is 9-11% behind (cause",
        "not measured) and the per-thread sort costs 4-12%.",
    ],
    "sanmiguel": [
        "four material kinds (CoatedDiffuse, Dielectric, Diffuse, DiffuseTransmission; 2.5M triangles",
        "in 796 meshes, 69k instances, 1,069 textures) and incoherent rays: pbrt's wavefront loses only",
        "2.7x to its own megakernel (the drains and not the queues dominating), so the speedups over",
        "it are the second smallest of the six, after bistro's; the vectorized drains pay 22-27% per",
        "thread and the ray sort pays on both structures (per-thread 1.02-1.06x, band 1.12-1.13x: the",
        "rays regrouped by the two-point Morton key feed the packet traversal coherent rays), which",
        "makes the sorted per-thread wavefront the best at both counts (10-24, with the fixed key).",
        "The scalar schedule's 16 spp cell is the known check failure against pbrt's volpath (853,108",
        "lit pixels to its 856,648, where pbrt's own two integrators differ by more; fine, 10-23).",
    ],
    "dragon_10": [
        "7.2M triangles in two meshes, CoatedDiffuse, Diffuse and the one Subsurface material of the",
        "six, one light, no textures: pbrt's wavefront loses only 2.7x to its megakernel, the",
        "per-thread queues buy 2.0x (bistro's 1.8-1.9x the only smaller gain) and the vectorized drains",
        "14-15%, and our megakernel is best at both counts, by 1-3% over the per-thread vectorized",
        "wavefront (1.22-1.24x pbrt's megakernel, second to book's): the subsurface bounce's probe",
        "trace and second material evaluation are two more queue passes for the wavefront and",
        "straight-line code for the megakernel (a reading; not profiled). On the GPU the same scene's",
        "megakernels are 0.3-0.9x pbrt (pbrt-gpu-sweep.txt); that reading, lanes diverging under the",
        "subsurface sample, has no counterpart with one path a thread. The sort costs 3-6% and the",
        "samples-outer order 7-9%.",
    ],
    "bistro": [
        "2.9M triangles in 1,693 meshes, 133 materials of four kinds (CoatedConductor, CoatedDiffuse,",
        "Dielectric, Diffuse), 409 textures (1.7 GiB of image maps), 20,638 area lights, 1920x1080: the",
        "heaviest shading of the six. pbrt's wavefront loses least to its megakernel here (2.1x), so",
        "every speedup over it is the smallest of the six (our megakernel 2.3x, the best wavefront",
        "3.5-3.7x); the per-thread queues buy least (1.8-1.9x), the vectorized drains most (29-30% per",
        "thread) and the ray sort most (per-thread 1.11-1.18x, band 1.13x), so the sorted per-thread",
        "wavefront is best at both counts, 11-18% over the unsorted. Its texture pool is past 2 GB:",
        "the packet schedules crashed on it before caa9eb9b (10-30); every cell matches pbrt now.",
    ],
    "plume": [
        "a smoke plume: a uniform-grid medium under an interface material, a scaled sphere, one light,",
        "1.44M pixels -- the eight's one participating medium. pbrt's wavefront loses 6.2-6.5x to its",
        "megakernel (the medium march is a queue pass per step for a wavefront, straight-line code",
        "for a megakernel), so the per-thread queues buy the most of the eight (3.2-3.3x) and the",
        "vectorized drains 29-36% more; the per-thread vectorized wavefront is best at both counts",
        "(7.8x, 7.3x), the samples-outer order a wash, the sort 3-7% behind (nothing to sort for",
        "inside a medium). Our megakernel is the one CPU cell pair of the eight below pbrt's own",
        "(0.92x and 0.95x of pbrt's VolPath): the medium march, which the GPU megakernels lose on",
        "too (0.3-0.7x there). Its image is 0.9335x pbrt's VolPath and 0.9999x pbrt's wavefront --",
        "pbrt's two integrators differ on media by design and the program follows the wavefront, so",
        "the cell is checked against the wavefront's render (cpu_compare.sh, 10-31).",
    ],
    "frame542": [
        "1.3M triangles, Dielectric and Diffuse, paths of up to 256 bounces (0.06% of camera paths",
        "reach the last): pbrt's wavefront loses the most of the eight to its megakernel (8.1x), a",
        "round of queue passes for every bounce any path reaches. The per-thread queues buy 4.7x;",
        "the vectorized drains lose 2-9% here (dielectric paths split between reflection and",
        "refraction and few survive deep, so the gang's lanes are masked), the sort recovers 1-4% of",
        "that, the samples-outer order costs 11-12%; the plain per-thread wavefront is the best",
        "wavefront at both counts (8.3x, 7.9x). Our megakernel beats it, 9.2x and 8.9x, 1.12x and",
        "1.11x of pbrt's own megakernel -- the CPU's version of the GPU reading (pbrt-gpu-depth): a",
        "thread following its path to its end pays nothing for the bounces nobody reaches.",
    ],
}
# Where the measured causes are written up.
CAUSES = [
    "causes measured, apps/pbrt/PLAN.md: 'The per-thread wavefront (10-21)' -- why t beats b,",
    "  a thread's own queues against frame-wide bands shared by all; 'The matrix measured: five",
    "  scenes, three sample counts, nine sides (10-23)' -- the reading of every bar; 'The CPU",
    "  sorted schedules re-measured with the fixed key (10-24)' -- the sort pays where the",
    "  shading is heavy and costs where it is not; '10-30, morning' -- the gang gathers' lane",
    "  offsets read unsigned (the packet schedules on bistro) and 'The CPU sweep of the six' --",
    "  this table, its blocks and its agreement with 10-28's. REPORT.md 10-22..10-24 has each",
    "  earlier run's numbers.",
]


def add_figure_arguments(ap):
    """The arguments the CPU and GPU figures share."""
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--sides", default=None,
                    help="comma-separated side names to draw, in this order")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene keys (paths under the scene set, or the "
                         "figure's labels) to draw, in this order; the default is every "
                         "scene of the tables in the order they appear")
    ap.add_argument("--width", type=float, default=TEXT_WIDTH,
                    help="inches; the default is the paper's text width (figure_style.py)")
    ap.add_argument("--height", type=float, default=2.4,
                    help="inches, the axes and the legend above them")
    ap.add_argument("--log", action="store_true", help="log-scale speedup axis")
    ap.add_argument("--ymax", type=float, default=None,
                    help="cut the speedup axis at this value: a bar past it is drawn to the "
                         "cutoff with a break mark and its value written above (the user, "
                         "10-30: no log axis for speedups; frame542's 20x bars)")
    ap.add_argument("--leaders", action="store_true",
                    help="a thin line from each cut bar's value to its bar")
    ap.add_argument("--rotate-cut-labels", action="store_true",
                    help="the cut bars' values upright (rotated 90 degrees), spread over their group")
    ap.add_argument("--ybreak", type=float, default=None,
                    help="break the speedup axis at --ymax: the bars past it continue in a short "
                         "upper axis that runs to this value, the break marked on the axis and "
                         "across the bars, no value written (the user, 10-30)")
    ap.add_argument("--values", action="store_true", help="the speedup written on each bar")
    ap.add_argument("--title", default=None)


def choose_sides(rows, all_sides, wanted):
    """The sides to draw: `wanted` (comma-separated names) in that order, or
    all of `all_sides`; either way only those the rows have. None, with a
    message on stderr, when nothing can be drawn."""
    sides = all_sides
    if wanted:
        names = wanted.split(",")
        known = {s[0]: s for s in all_sides}
        missing = [w for w in names if w not in known]
        if missing:
            print(f"unknown sides: {', '.join(missing)}", file=sys.stderr)
            return None
        sides = [known[w] for w in names]
    present = {r["side"] for r in rows}
    sides = [s for s in sides if s[0] in present]
    if not sides:
        print("no side of the table is one the figure knows", file=sys.stderr)
        return None
    return sides


def choose_scenes(rows, wanted):
    """The rows of the scenes `wanted` (comma-separated keys -- paths under
    the scene set -- or figure labels), in that order, so that a figure
    drawn from several tables shows the paper's scenes in the paper's
    order whatever else the tables hold (the user, 10-30: frame542's cells
    share a table with clouds'). None or empty keeps every row in the
    tables' order. A wanted scene no table has is named on stderr and
    left out: the CPU figure may lag the GPU figure by a scene until its
    sweep runs."""
    if not wanted:
        return rows
    names = [w for w in wanted.split(",") if w]
    chosen = []
    for name in names:
        hit = [r for r in rows if r["key"] == name or scene_label(r["key"]) == name]
        if not hit:
            print(f"no rows for scene {name}; left out", file=sys.stderr)
        chosen.extend(hit)
    return chosen


def cut_mark(ax, x, top, half_width):
    """The break across a bar cut at the axis top, as papers break a bar: a
    gap in the page colour slanting across the bar a little under its cut
    top, with a short black slash along each edge of the gap, a little
    wider than the bar. After ~/projects/loom/plot_style.py's cut_mark (the
    user, 10-30: use that mark). `top` is the bar's drawn height, the
    cutoff; `half_width` half the bar's width in x units."""
    from matplotlib.transforms import offset_copy
    y = top - top * 0.07
    xs = [x - half_width * 1.1, x + half_width * 1.1]
    ys = [y - top * 0.012, y + top * 0.012]
    ax.plot(xs, ys, color="white", linewidth=1.8, solid_capstyle="butt", zorder=7)
    for pts in (-1.25, 1.25):
        ax.plot(xs, ys, color="black", linewidth=0.5, solid_capstyle="butt", zorder=8,
                transform=offset_copy(ax.transData, fig=ax.figure, x=0, y=pts, units="points"))


def best_where(points, scenes, spps):
    """For every scene label of `scenes`, the best side at each sample
    count and the one behind it, from the bars drawn: (label, `16 spp A
    4.65x (B 3.20x next); ...`) -- the computed half of the per-scene
    reading."""
    lines = []
    for scene in scenes:
        parts = []
        for spp in spps:
            key = f"{scene} {spp} spp"
            ranked = sorted(((v, label) for label, pts in points.items()
                             for k, v in pts if k == key), reverse=True)
            if not ranked:
                continue
            part = f"{spp} spp {ranked[0][1]} {ranked[0][0]:.2f}x"
            if len(ranked) > 1:
                part += f" ({ranked[1][1]} {ranked[1][0]:.2f}x next)"
            parts.append(part)
        if parts:
            lines.append((scene, "; ".join(parts)))
    return lines


def draw(rows, sides, reference, reference_label, y_label, args, comparisons=(), causes=(),
         why=(), readings=None, sources=()):
    """The figure: a bar per side of `sides` ((side, label, colour), in
    drawing order) in a subgroup per sample count in a group per scene, each
    bar the `reference` side's seconds over the side's, from `rows` as `load`
    returns them. Written to args.out, with `<out>.txt` beside it: the
    tables it was drawn from (`sources`); each side's geometric mean, least
    and greatest bar and which, over everything and per sample count; then
    what each of `comparisons` ((label, label, what differs between the two
    schedules)) bought, bar by bar; the trend of each side from the fewest
    samples to the most; `why`, the mechanisms behind the figure; for every
    scene the best side at each sample count (computed) and the scene's
    reading from `readings` ({scene label: [lines]}, "not yet written up"
    when a scene has none); and `causes`, lines naming where the measured
    explanations are written up. Returns the number of legend rows.

    A scene is its `key` (its path under the scene set, or its name) and is
    labelled by figure_style.scene_label, one line under its bars."""
    scenes = ordered(r["key"] for r in rows)
    spps = sorted(ordered(r["spp"] for r in rows))
    labels = {scene: scene_label(scene) for scene in scenes}

    def cell(scene, spp, side):
        return next((r for r in rows if r["key"] == scene and r["spp"] == spp
                     and r["side"] == side), None)

    def reference_seconds(scene, spp):
        # The least of the cell's reference rows, should a run have more than one.
        times = [r["seconds"] for r in rows
                 if r["key"] == scene and r["spp"] == spp and r["side"] == reference]
        return min(times) if times else None

    def reference_of(c, scene, spp):
        # A row from a run of its own carries that run's reference seconds
        # (`reference`, set by the loader: the GPU figure joins tables), so
        # that a side is always measured against the pbrt that ran beside it.
        return c["reference"] if c.get("reference") is not None else reference_seconds(scene, spp)

    paper_style()
    # Constrained layout keeps the legend, the scene names and the y label
    # inside a figure of exactly the size asked, so the PDF is the text width.
    broken = args.ybreak is not None and args.ymax is not None
    if broken:
        # A broken axis: the body below --ymax, a short upper axis from there
        # to --ybreak, one bar in four parts of the height; both hold every
        # bar and each clips it to its own range.
        fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(args.width, args.height),
                                         layout="constrained", sharex=True,
                                         gridspec_kw={"height_ratios": [1, 3], "hspace": 0.04})
        axes_all = [ax, ax_top]
    else:
        fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
        ax_top = None
        axes_all = [ax]

    bar = 1.0                     # one bar's width in x units
    gap_spp = 1.5                 # between the sample counts of a scene
    gap_scene = 3.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    points = {label: [] for _, label, _ in sides}      # the bars, for the summary
    cut_labels = []                                    # (x, text) of bars past --ymax
    label_top, label_headroom_pt = None, 0.0           # where the cut values sit
    checked, unverified = 0, []                        # the drawn bars' image checks
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            group_start = x
            cuts = 0                  # bars of this group past the cutoff
            for side, label, colour in sides:
                c = cell(scene, spp, side)
                ref = reference_of(c, scene, spp) if c is not None else None
                if c is not None and ref is not None:
                    speedup = ref / c["seconds"]
                    points[label].append((f"{labels[scene]} {spp} spp", speedup))
                    # Every drawn bar's image check, for the .txt (the user,
                    # 10-30: "make sure all results are being verified").
                    # pbrt's own sides (PBRT mk, pbrt's volpath on the CPU)
                    # are references and carry no verdict.
                    if not side.startswith("pbrt"):
                        verdict = str(c.get("verdict") or "")
                        checked += 1
                        if not verdict.startswith("ok"):
                            unverified.append(f"{labels[scene]} {spp} spp {label}: "
                                              f"{verdict[:90] or 'no verdict'}")
                    past = args.ymax is not None and speedup > args.ymax
                    cut = past and not broken
                    for a in axes_all:
                        a.bar(x, args.ymax if cut else speedup, bar * 0.9, color=colour,
                              edgecolor=EDGE, linewidth=0.3, zorder=2)
                    if cut:
                        cut_mark(ax, x, args.ymax, bar * 0.45)
                        cut_labels.append((x, f"{speedup:.1f}x"))
                        cuts += 1
                    elif past:
                        # Broken axis: the bar is broken where the axis is.
                        cut_mark(ax, x, args.ymax, bar * 0.45)
                    if args.values and not cut:
                        ax.text(x, speedup, f"{speedup:.2f}", ha="center", va="bottom",
                                fontsize=TICK_FONT - 2, rotation=90, zorder=4)
                x += bar
            spp_centres.append((group_start + x - bar) / 2)
            spp_labels.append(f"{spp}")       # "spp" is written once, at the left
            if j + 1 < len(spps):
                x += gap_spp
        scene_centres.append((scene_start + x - bar) / 2)
        x += gap_scene

    right = x - gap_scene
    if cut_labels:
        # The cut bars' values on one line just over the cutoff, each over
        # its bar; where neighbours would overprint, the left one moves
        # left and the right one right until they clear (the user, 10-30:
        # no stacking). The text's width is estimated from its characters
        # at the axes' scale, the axes taken as the figure less its margin.
        size = TICK_FONT - 1
        per_point = (right + 2 * bar) / ((args.width - 0.8) * 72.0)   # x units a point
        upright = args.rotate_cut_labels
        # An upright label is as wide as the text is tall; a flat one as its
        # characters.
        widths = [(1.1 * size if upright else 0.56 * size * len(text)) * per_point
                  for _, text in cut_labels]
        xs = [x for x, _ in cut_labels]
        for _ in range(100):
            moved = False
            for i in range(len(xs) - 1):
                need = (widths[i] + widths[i + 1]) / 2 + 1.5 * per_point
                gap = xs[i + 1] - xs[i]
                if gap < need:
                    xs[i] -= (need - gap) / 2
                    xs[i + 1] += (need - gap) / 2
                    moved = True
            if not moved:
                break
        # With leaders the values sit a little higher, so the line from a
        # bar's cut top to its value has a length to be seen.
        y_text = args.ymax * (1.07 if args.leaders else 1.015)
        # The headroom the values need above y_text, in points: an upright
        # value is as tall as its characters, a flat one as the type.
        longest = max(len(text) for _, text in cut_labels)
        label_headroom_pt = (0.6 * size * longest if upright else 1.2 * size) + 2.0
        label_top = y_text
        for xl, (xb, text) in zip(xs, cut_labels):
            if args.leaders:
                ax.plot([xb, xl], [args.ymax, y_text - 0.008 * args.ymax], color="#555555",
                        linewidth=0.4, zorder=6, clip_on=False)
            ax.text(xl, y_text, text, ha="center", va="bottom", fontsize=size,
                    rotation=90 if upright else 0, zorder=6, clip_on=False)
    reference_line = ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8,
                                zorder=3, label=reference_label)
    ax.set_xticks(spp_centres)
    ax.set_xticklabels(spp_labels)
    ax.tick_params(axis="x", length=0)
    ax.text(-bar, -0.03, "spp", ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=TICK_FONT)
    for centre, scene in zip(scene_centres, scenes):
        # One line each (the user, 10-29): the names are short enough for
        # eight scenes side by side at text width (figure_style.scene_label).
        ax.text(centre, -0.15, labels[scene], ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=TICK_FONT)
    if broken:
        fig.supylabel(y_label, fontsize=plt.rcParams["axes.labelsize"])
    else:
        ax.set_ylabel(y_label)
    if args.log:
        ax.set_yscale("log")
    elif broken:
        ax.set_ylim(0, args.ymax)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= args.ymax])
        ax_top.set_ylim(args.ymax, args.ybreak)
        from matplotlib.ticker import MaxNLocator
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
        ax_top.set_yticks([t for t in ax_top.get_yticks() if args.ymax < t <= args.ybreak])
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.spines["bottom"].set_visible(False)
        # The break marks on the axis: a short slash at the foot of the upper
        # axis and the head of the lower, as matplotlib's broken-axis recipe.
        slash = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none",
                     color="black", mec="black", mew=0.6, clip_on=False, zorder=9)
        ax_top.plot([0], [0], transform=ax_top.transAxes, **slash)
        ax.plot([0], [1], transform=ax.transAxes, **slash)
    elif args.ymax is not None:
        # Room above the cutoff for the cut bars' values and nothing more,
        # no tick past the cutoff.
        if label_top is None:
            top = args.ymax * 1.04
        else:
            axes_pt = max(60.0, (args.height - 0.55) * 72.0)
            top = label_top / max(0.5, 1.0 - label_headroom_pt / axes_pt)
        ax.set_ylim(0, top)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= args.ymax])
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * (1.12 if args.values else 1.04))
    for a in axes_all:
        a.set_xlim(-bar, right + bar)
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        a.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
        a.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    # The reference line leads the legend, then a patch per side; two rows
    # at most, filled evenly (ten entries: five over five; the user, 10-24).
    handles = [reference_line] + [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.3,
                                        label=label) for _, label, colour in sides]
    ncol = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    legend_rows = (len(handles) + ncol - 1) // ncol
    (ax_top if broken else ax).legend(
        handles=handles, ncol=ncol, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0 + 0.02),
        borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    # The summary beside the figure: every side over all its bars, then
    # over each sample count.
    sections = [("every scene and sample count",
                 [(label, points[label]) for _, label, _ in sides])]
    for spp in spps:
        sections.append((f"{spp} spp",
                         [(label, [p for p in points[label] if p[0].endswith(f" {spp} spp")])
                          for _, label, _ in sides]))
    notes = []
    if unverified:
        notes.append(f"image checks: {checked - len(unverified)} of {checked} drawn bars match pbrt's "
                     f"image; NOT matching or unchecked:")
        notes.extend(f"  {line}" for line in unverified)
    else:
        notes.append(f"image checks: every one of the {checked} drawn bars matches pbrt's image "
                     f"(the harness's verdict on each cell)")
    for a, b, what in comparisons:
        if a in points and b in points:
            notes.append(f"{what}: {a} over {b}  {describe(ratios(points[a], points[b]))}")
    if len(spps) > 1:
        lo, hi = spps[0], spps[-1]
        for _, label, _ in sides:
            at = {spp: geomean([p for p in points[label] if p[0].endswith(f" {spp} spp")])
                  for spp in (lo, hi)}
            if at[lo] and at[hi]:
                notes.append(f"{label}: geomean {at[lo]:.2f}x at {lo} spp, {at[hi]:.2f}x at {hi} spp"
                             f" ({at[hi] / at[lo]:.2f}x from the fewest samples to the most)")
    if why:
        notes.append("")
        notes.append("why: the mechanisms")
        notes.extend(f"  {line}" for line in why)
    notes.append("")
    notes.append("which schedule is best on each scene, and why")
    paths = {labels[scene]: scene for scene in scenes}
    for scene, line in best_where(points, [labels[s] for s in scenes], spps):
        where = f" ({paths[scene]})" if paths[scene] != scene else ""
        notes.append(f"  {scene}{where}: {line}")
        reading = (readings or {}).get(scene)
        if reading:
            notes.extend(f"      {r}" for r in reading)
        else:
            notes.append("      reading: not yet written up (plot's READINGS)")
    if causes:
        notes.append("")
        notes.extend(causes)
    here = os.path.dirname(os.path.abspath(__file__))
    drawn_from = ", ".join(os.path.relpath(p, here) for p in sources) if sources else None
    title = f"{y_label} over {reference_label} ({len(scenes)} scenes x {len(spps)} sample counts)"
    if drawn_from:
        title += f"\ndrawn from apps/pbrt/{{{drawn_from}}}" if len(sources) > 1 else f"\ndrawn from apps/pbrt/{drawn_from}"
    summary = write_summary(args.out, title, sections, notes)
    print(f"wrote {args.out} ({len(scenes)} scenes, {len(spps)} sample counts, "
          f"{len(sides)} sides, {legend_rows} legend rows) and {summary}")
    return legend_rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    add_figure_arguments(ap)
    args = ap.parse_args(argv[1:])
    if args.out is None:
        args.out = "pbrt-cpu-sweep.pdf"

    rows = choose_scenes(load(args.csvs), args.scenes)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args, COMPARISONS, CAUSES,
         why=WHY, readings=READINGS, sources=args.csvs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
