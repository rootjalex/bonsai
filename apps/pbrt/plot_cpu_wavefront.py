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

pbrt's bar is black and each Bonsai side has the colour its schedule has
on every figure of apps/pbrt (figure_style.SCHEDULE_COLOURS; the user,
10-29, after the CPU and GPU sweeps had each laid the one ramp over its
own bars): the wavefront family on the paper's one ramp (YlGnBu, the
heatmap's, its seven darkest classes) and the megakernel family in greys
beside pbrt's black, each darker with every scheduling step down the
legend, so that the family is read from the hue and the variant from the
lightness -- mk-b-s-ps is the GPU figure's `mk` grey and wf-b-s-sp its
`wf` class, the same schedules. Every bar has a
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
    ("pbrt-volpath", "PBRT mk"),            # black (figure_style.PBRT_COLOUR)
    ("scalar", "mk-b-s-ps"),
    ("wavefront-frame", "wf-b-s-sp"),
    ("wavefront-frame-packet", "wf-b-v-sp"),
    ("wavefront-frame-packet-reorder", "wf-b-v-sp-sort"),
    ("wavefront-thread", "wf-t-s-ps"),
    ("wavefront-thread-packet", "wf-t-v-ps"),
    ("wavefront-thread-packet-reorder", "wf-t-v-sp"),
    ("wavefront-thread-packet-sort", "wf-t-v-ps-sort"),
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
    "  2.7-6.8x slower than pbrt's own megakernel (PBRT mk). The per-thread wavefront (wf-t-*) gives",
    "  each thread its own chunk of pixels and its own queues: 3.7x over the same drains on shared",
    "  bands (wf-t-s-ps over wf-b-s-sp), the one decision that moves the figure. Our megakernel",
    "  (mk-b-s-ps) beats pbrt's by 1.05-1.4x on the same algorithm: our codegen.",
    "the drains vectorized (v over s) pay where the shading is heavy (sanmiguel 24-26%, book 12-18%,",
    "  ganesha 8-13% in the per-thread form) and nothing on the killeroos; the samples-outer order",
    "  (sp over ps) is level or behind; the ray sort before each trace pays where the rays are",
    "  incoherent and the shading heavy (sanmiguel) and costs where a thread's pass is large (the",
    "  killeroos at 128 spp, 0.83x), since a chunk x spp queue sorted and read through the",
    "  permutation leaves the core's cache.",
    "the megakernel wins the killeroos at 64 and 128 spp: a thread's chunk of 1024 pixels x spp",
    "  queue entries outgrows the cache there while a path's state stays in registers; a smaller",
    "  chunk at high spp is the untested remedy (10-23).",
    "with more samples the wavefronts' speedups fall a little (0.95-0.98x from 16 to 128 spp), the",
    "  per-thread sort's most (0.87x), and the megakernels' not at all.",
]
READINGS = {
    "killeroo-simple": [
        "two materials (CoatedDiffuse, Diffuse) and the cheapest trace of the five (66k triangles):",
        "vectorizing the per-thread drains buys nothing (wf-t-v-ps over wf-t-s-ps 0.98-1.03x), and at 64",
        "and 128 spp the chunk's queues outgrow the cache where the megakernel's path-at-a-time state",
        "does not, so our megakernel takes those cells and the per-thread wavefronts edge it at 16 spp.",
        "The band's sort pays (1.83x against 1.77x); the per-thread sort costs most here at 128 spp",
        "(0.83x: 128k keys a pass per thread).",
    ],
    "killeroo-gold": [
        "Conductor and Diffuse over 532k triangles: killeroo-simple's reading -- cheap shading, so the",
        "drains' vectorization is level (0.99-1.01x), and the chunk's queues at high spp cost the",
        "wavefront what the megakernel's registers do not; our megakernel is best at every count.",
    ],
    "book": [
        "textured CoatedDiffuse pages over 21k triangles: the heaviest shading per hit after sanmiguel's,",
        "so vectorizing the per-thread drains pays 12-18% and the per-thread wavefront takes every cell",
        "(the sorted one at 16 spp); pbrt's own megakernel is weakest here (3.8x over its wavefront) and",
        "ours beats it by 1.4x, the most of the five.",
    ],
    "ganesha": [
        "one statue of 4.3M triangles, CoatedDiffuse and Diffuse: the per-thread vectorized drains pay",
        "8-13% and win every cell; the band's vectorized drains lose 8-9% here, the one scene (on the",
        "list for perf stat, not done), and the samples-outer order is 11-15% behind (cause not",
        "measured).",
    ],
    "sanmiguel": [
        "the heaviest shading of the five (four material kinds: CoatedDiffuse, Dielectric, Diffuse,",
        "DiffuseTransmission; 2.5M triangles in 796 meshes, instanced) and the least coherent rays:",
        "pbrt's wavefront loses least to its own megakernel here (2.7x, the drains and not the queues",
        "dominating), so every speedup over it is the smallest of the five; the vectorized drains pay",
        "most (24-26% per thread) and the ray sort pays on both structures (per-thread 1.02-1.06x,",
        "band 1.13x: the rays regrouped by the two-point Morton key feed the packet traversal",
        "coherent rays), which makes the sorted per-thread wavefront the best at every count -- the",
        "one scene where the sort wins on the CPU (10-24, with the fixed key).",
    ],
}
# Where the measured causes are written up.
CAUSES = [
    "causes measured, apps/pbrt/PLAN.md: 'The per-thread wavefront (10-21)' -- why t beats b,",
    "  a thread's own queues against frame-wide bands shared by all; 'The matrix measured: five",
    "  scenes, three sample counts, nine sides (10-23)' -- the reading of every bar; 'The CPU",
    "  sorted schedules re-measured with the fixed key (10-24)' -- the sort pays where the",
    "  shading is heavy and costs where it is not. REPORT.md 10-22..10-24 has each run's numbers.",
]


def add_figure_arguments(ap):
    """The arguments the CPU and GPU figures share."""
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--sides", default=None,
                    help="comma-separated side names to draw, in this order")
    ap.add_argument("--width", type=float, default=TEXT_WIDTH,
                    help="inches; the default is the paper's text width (figure_style.py)")
    ap.add_argument("--height", type=float, default=2.4,
                    help="inches, the axes and the legend above them")
    ap.add_argument("--log", action="store_true", help="log-scale speedup axis")
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
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")

    bar = 1.0                     # one bar's width in x units
    gap_spp = 1.5                 # between the sample counts of a scene
    gap_scene = 3.0               # between scenes
    x = 0.0
    scene_centres, spp_centres, spp_labels = [], [], []
    points = {label: [] for _, label, _ in sides}      # the bars, for the summary
    for scene in scenes:
        scene_start = x
        for j, spp in enumerate(spps):
            group_start = x
            for side, label, colour in sides:
                c = cell(scene, spp, side)
                ref = reference_of(c, scene, spp) if c is not None else None
                if c is not None and ref is not None:
                    speedup = ref / c["seconds"]
                    points[label].append((f"{labels[scene]} {spp} spp", speedup))
                    ax.bar(x, speedup, bar * 0.9, color=colour,
                           edgecolor=EDGE, linewidth=0.3, zorder=2)
                    if args.values:
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
    ax.set_ylabel(y_label)
    if args.log:
        ax.set_yscale("log")
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * (1.12 if args.values else 1.04))
    ax.set_xlim(-bar, right + bar)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if args.title:
        ax.set_title(args.title)

    # The reference line leads the legend, then a patch per side; two rows
    # at most, filled evenly (ten entries: five over five; the user, 10-24).
    handles = [reference_line] + [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.3,
                                        label=label) for _, label, colour in sides]
    ncol = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    legend_rows = (len(handles) + ncol - 1) // ncol
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
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

    rows = load(args.csvs)
    sides = choose_sides(rows, SIDES, args.sides)
    if sides is None:
        return 1
    draw(rows, sides, REFERENCE, REFERENCE_LABEL, Y_LABEL, args, COMPARISONS, CAUSES,
         why=WHY, readings=READINGS, sources=args.csvs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
