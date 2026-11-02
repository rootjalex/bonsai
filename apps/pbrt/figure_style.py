"""The paper's figure style, shared by apps/pbrt's three plotters (and
imported by apps/rtq's and apps/wosx's if they want it).

The paper is in ACM's acmsmall format (PACMPL / PLDI): a single column,
10 pt text, 12 pt leading, a text block 5.478 in wide and 7.884 in tall.
A figure drawn `TEXT_WIDTH` inches wide and included at `\\textwidth` is
not scaled by LaTeX, so the sizes set here are the sizes on the page:
8 pt for labels and the legend (the user, 10-24: readable, not
necessarily the body's 10 pt; 8 pt is acmart's footnote size, a shade
under its 9 pt captions), 7 pt for tick labels. (acmart's two-column
formats, sigconf and sigplan, have a 7.0 in text width; a figure drawn for
them is shrunk to 78% here.) ACM's production (TAPS) requires every font embedded and
rejects Type 3 fonts, which matplotlib's PDF backend emits by default;
`fonttype 42` embeds TrueType instead. The face is Liberation Sans
(metrically Helvetica/Arial; installed as TrueType, so it embeds as
Type 42 -- the Libertine faces acmart uses for its text exist here only as
CFF OpenType, which matplotlib cannot embed that way), falling back to
Arial, Helvetica and DejaVu Sans when a machine lacks it.

    from figure_style import TEXT_WIDTH, paper_style
    paper_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3))

`lean_labels(ax, labels, pitch)` sets x tick labels at the least angle at
which they do not overlap (`lean_angle`, `lean_height` size a figure for
them). `write_summary` writes the statistics the paper quotes beside a figure,
`<figure>.txt`: for each series a line that begins with the range
(least-greatest) and the geometric mean, then the count, the least value,
the lower quartile, the median, the upper quartile and the greatest value,
and which point each is (the user, 10-24, 10-26 and 10-30; `describe` has
the form).
`YLGNBU` is the paper's one colour ramp (the heatmap's cells, the rtq and
wosx figures, the sweeps' bars); `RAMP8` is eight of its shades at equal
perceptual steps, and `SCHEDULE_COLOURS` lays them over the two sweeps
the same way (the user, 10-29). `scene_label` names a scene on a
figure from its path in the scene set, one line each.
"""

import math
import os
import re

import matplotlib.pyplot as plt

TEXT_WIDTH = 5.478      # inches, acmsmall's text block (the PLDI call for papers)
TEXT_HEIGHT = 7.884

FONT = 8                # labels, legend, the text at the axes
TICK_FONT = 7           # tick labels
SANS = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]


def paper_style(font=FONT, tick=TICK_FONT):
    """Set matplotlib up for a figure of the paper: TrueType embedding, the
    sans face, the sizes above, thin axes."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": SANS,
        "font.size": font, "axes.labelsize": font, "axes.titlesize": font,
        "legend.fontsize": font, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.handlelength": 1.4, "legend.handleheight": 0.8,
        "legend.borderpad": 0.2, "legend.labelspacing": 0.3,
    })


def text_width(text, font=TICK_FONT):
    """About how long `text` runs in inches at `font` points (half an em a
    character, Liberation Sans' average), a shade over for safety."""
    return 0.07 + 0.5 * font / 72 * len(text)


def lean_angle(labels, pitch, font=TICK_FONT, margin=4.0):
    """The least angle, in degrees, at which the tick labels `labels` set
    `pitch` inches apart do not run into each other: 0 when the longest
    lies within the pitch, else the angle at which neighbouring lines of
    text (font * 1.15 points high) are a line apart, sin(angle) = line /
    pitch, plus `margin` degrees; 90 at most. The least height the labels
    can take (the user, 10-24: upright names wasted a third of the heatmap,
    and 45 degrees most of the rtq figures)."""
    longest = max((text_width(t, font) for t in labels), default=0.0)
    if longest <= pitch:
        return 0.0
    line = font * 1.15 / 72
    return min(90.0, math.degrees(math.asin(min(1.0, line / pitch))) + margin)


def lean_height(labels, angle, font=TICK_FONT):
    """The inches the labels take below the axis at `angle` degrees."""
    longest = max((text_width(t, font) for t in labels), default=0.0)
    line = font * 1.15 / 72
    a = math.radians(angle)
    return 0.08 + longest * math.sin(a) + line * math.cos(a)


def lean_labels(ax, labels, pitch, font=TICK_FONT):
    """Set `labels` as the x tick labels of `ax`, leaning by `lean_angle`
    for `pitch` inches between ticks; returns the angle. The ticks
    themselves are set to 0..n-1 first if the axis has not that many."""
    if len(ax.get_xticks()) != len(labels):
        ax.set_xticks(range(len(labels)))
    angle = lean_angle(labels, pitch, font)
    if angle == 0.0:
        ax.set_xticklabels(labels, fontsize=font)
    else:
        ax.set_xticklabels(labels, fontsize=font, rotation=angle, ha="right", va="top",
                           rotation_mode="anchor")
    return angle


def quantile(points, q):
    """The q-quantile of `points`, (label, value) pairs, by linear
    interpolation between the two cells nearest the rank q * (n - 1)
    (numpy's default), and the cells it is computed from: the one cell the
    rank falls on, or the two it lies between."""
    ordered = sorted(points, key=lambda p: p[1])
    position = q * (len(ordered) - 1)
    below, above = math.floor(position), math.ceil(position)
    if below == above:
        return ordered[below][1], [ordered[below]]
    fraction = position - below
    value = ordered[below][1] * (1 - fraction) + ordered[above][1] * fraction
    return value, [ordered[below], ordered[above]]


def _statistic(name, value, cells):
    """`name value (where)`: the cell's label when the statistic is one
    cell's value, the two cells with their values when it lies between
    them."""
    if len(cells) == 1:
        where = cells[0][0]
    else:
        where = ", ".join(f"{label} {v:.2f}" for label, v in cells)
    return f"{name} {value:.2f}x ({where})"


def describe(points):
    """Summary statistics of `points`, (label, value) pairs of one series,
    as one line of text, which begins with the range (least-greatest) and
    the geometric mean (the user, 10-30: every summary starts with those
    two), then the count, then in order the least value, the lower
    quartile, the median, the upper quartile and the greatest value, each
    with the label of the cell it belongs to -- or, where a quartile or the
    median falls between two cells, both cells with their values. The paper
    quotes these beside every figure (the user, 10-24; the quartiles and
    the median with their cells, 10-26)."""
    points = [(label, v) for label, v in points if v is not None and v > 0]
    if not points:
        return "no points"
    geomean = math.exp(sum(math.log(v) for _, v in points) / len(points))
    low = min(points, key=lambda p: p[1])
    high = max(points, key=lambda p: p[1])
    return (f"range {low[1]:.2f}x-{high[1]:.2f}x  geomean {geomean:.2f}x  "
            f"n={len(points)}  "
            f"min {low[1]:.2f}x ({low[0]})  "
            f"{_statistic('lower quartile', *quantile(points, 0.25))}  "
            f"{_statistic('median', *quantile(points, 0.5))}  "
            f"{_statistic('upper quartile', *quantile(points, 0.75))}  "
            f"max {high[1]:.2f}x ({high[0]})")


def ratios(numerators, denominators):
    """(label, a / b) for every label the two series share: one series'
    gain over another, point by point, so that `describe` of it says how
    much the one thing that differs between them bought and where."""
    b = {label: v for label, v in denominators if v}
    return [(label, v / b[label]) for label, v in numerators if label in b and v]


def geomean(points):
    values = [v for _, v in points if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


# A scene's name on a figure. The tables key a scene by its path under the
# scene set (`kroken/camera-1`), since two scenes share the file name
# camera-1 (kroken's and watercolor's); the label is the file's name,
# except that a name which is only a camera or view number is replaced by
# the scene's directory (kroken, watercolor, landscape), and the long
# names below are shortened so that the sweeps' scenes stand side by side
# on one line at text width (the user, 10-29: no name broken over two
# lines) and the heatmap's thirty are not leaning paragraphs (the user,
# 10-30: the sanmiguel names). The paper never prints a file name (the
# user, 10-30); the scene table (scene_statistics.py) prints the label
# and, under it, which file each shortened label stands for.
SCENE_LABELS = {
    "sanmiguel/sanmiguel-courtyard": "sanmiguel",
    "sanmiguel/sanmiguel-courtyard-second": "sanmiguel-2",
    "bistro/bistro_vespa": "bistro",
}
GENERIC_NAME = re.compile(r"^(camera|view)-\d+$")


def scene_label(key, short=True):
    """The label of the scene `key` -- its `dir/name` path, or a bare name
    where a table has no path column: SCENE_LABELS' short form when
    `short` (the sweeps), the directory when the name is a camera or view
    number, else the name itself."""
    if short and key in SCENE_LABELS:
        return SCENE_LABELS[key]
    if "/" in key:
        directory, name = key.rsplit("/", 1)
        return directory if GENERIC_NAME.match(name) else name
    return key


def write_summary(figure_path, title, sections, notes=()):
    """Write the figure's summary beside it, `<figure stem>.txt`: `title`,
    then each section of `sections` ((heading, [(name, points)]) with one
    `describe` line per named series), then `notes`, lines of text that
    read the figure -- what each schedule change bought, trends, and where
    the measured causes are written up. Returns the path written."""
    path = os.path.splitext(figure_path)[0] + ".txt"
    with open(path, "w") as f:
        f.write(f"{title}\n")
        for heading, series in sections:
            f.write(f"\n{heading}\n")
            width = max((len(name) for name, _ in series), default=0)
            for name, points in series:
                f.write(f"  {name:<{width}}  {describe(points)}\n")
        if notes:
            f.write("\nreading the figure\n")
            for line in notes:
                f.write(f"  {line}\n")
    return path


# ColorBrewer YlGnBu, nine classes, lightest first: the paper's ramp (the
# user's palette, 10-23) -- the heatmap's cells, and the base of the
# wavefront family's colours below.
YLGNBU = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
          "#1D91C0", "#225EA8", "#253494", "#081D58"]

# The schedules' colours: the one ramp, laid over every figure the same
# way (the user, 10-29). Each sweep draws eight Bonsai schedules, and
# each takes RAMP8 -- eight shades of the YlGnBu ramp at equal perceptual
# steps from its class 1 to its class 8 (the ramp as the heatmap's
# colour map interpolates it; the ends are ColorBrewer's own) -- in its
# legend's order: the palest shade the plain megakernel (gpu-optix-mega
# on the GPU, the scalar schedule on the CPU: the same schedule), the
# darkest the sorted wavefront (gpu-optix-sort,
# wavefront-thread-packet-sort), every step down a legend a step darker.
# So the two figures read by one rule and share their ends; the shades
# between name each figure's own schedules (the legend says which),
# there being more schedules than shades. The near-white class 0 draws
# no bar. pbrt's own bars are black (the CPU's PBRT mk), pbrt's
# wavefront the dashed reference line. apps/wosx/plot.py lays RAMP8 over
# its figures the same way.
#
# What was tried and undone the same day, for the record: the noon
# commit (621343b4) put the megakernels on a second hue, a warm ramp, so
# that every schedule could keep one colour across both sweeps; the user:
# the paper has one palette, and the figures must match the heatmap's
# and the rtq and wosx figures'. Greys for the megakernels next (f8090ba4,
# one hue plus neutrals) made the GPU sweep, six of whose eight bars are
# megakernels, near unreadable. The ramp alone, as before noon, is the
# user's choice; this table keeps its one gain, the same shades in the
# same direction on both sweeps.
#
# Why equal steps rather than the raw classes 1..8: the dataviz
# validator's distances (OKLab x100) put the raw classes 1 and 2
# (#EDF8B1, #C7E9B4) 6.6 apart under normal vision and 5.2 under protan
# simulation, the ramp's pale end being its flattest; at equal steps
# every neighbouring pair of RAMP8 is 10.7 apart under normal vision and
# 8.8 under protan and deutan (8.0 tritan), in either sweep's drawing
# order. Ordered shades cannot reach the validator's categorical floor of
# 15 between neighbours; the fixed bar order within every group, keyed by
# the legend, is the second encoding.
PBRT_COLOUR = "#000000"
RAMP8 = ["#EDF8B1", "#ADDFB7", "#69C4BF", "#30A4C2", "#1F7EB7", "#2357A4", "#21318B", "#081D58"]

# Legend label -> colour, for every schedule a figure of apps/pbrt draws
# (the labels are plot_cpu_wavefront.py's and plot_gpu_wavefront.py's).
#
# The order (the user, 10-30 evening): on both sweeps the schedule that
# matches pbrt's wavefront comes first, then the wavefront's modifications
# by increasing departure from it, then this renderer's megakernels; the
# CPU figure's PBRT mk (pbrt's own megakernel, black) leads its legend.
# The ramp runs palest to darkest down each legend, so the matched
# wavefront is the palest bar of both sweeps and the last megakernel the
# darkest.
SCHEDULE_COLOURS = {
    "PBRT mk": PBRT_COLOUR,
    # The GPU sweep's legend, top to bottom (the user, 11-01: the winners
    # of at least one cell -- wf, wf-fuse-mat, mk-flat-ser, mk-flat-ser-ray
    # and mk-flat-ser-ray-px -- and the rest filled to three wavefronts and
    # four megakernels; wf-sort, which never wins, out: "it makes the
    # legend look weird"). The labels are the short ones the user chose
    # (11-01): every drawn megakernel but none of the wavefronts is a flat
    # launch, and every flat one is in the samples order (a pixel's samples
    # adjacent in the launch, schedules/gpu-optix-*-samples.bonsai) unless
    # it says -px, the pixel order; the .txt names each label's file.
    # Seven sides over the eight shades: the ramp skips one between the
    # families, so the wavefronts are the greens and the megakernels the
    # blues, and the ends still agree with the CPU sweep's.
    "wf": RAMP8[0],                 # gpu-optix, pbrt's wavefront matched
    "wf-fuse-mat": RAMP8[1],        # gpu-optix-fused-route: the material stage (and the miss and emissive routing) in the trace program
    "wf-bounce": RAMP8[2],          # gpu-optix-bounce: one launch per bounce, the ray queue alone
    "mk-flat": RAMP8[4],            # gpu-optix-mega-flat-samples: the flat launch, samples order
    "mk-flat-ser": RAMP8[5],        # gpu-optix-mega-ser-flat-samples: that with SER by material
    "mk-flat-ser-ray": RAMP8[6],    # gpu-optix-mega-ser-ray-flat-samples: that with the reordering by ray too
    "mk-flat-ser-ray-px": RAMP8[7], # gpu-optix-mega-ser-ray-flat: the same in the pixel order
    # Sides out of the figure (10-30 night: mk-pass, mk-blk, mk-ser, never
    # best and each below a sibling at every cell; 11-01: wf-sort, wf-fused,
    # wf-fused-ser, wf-pass, mk, the pixel-order mk-flat-px and
    # mk-flat-ser-px, and the exploration's wf-fused-shadow); drawn only
    # when --sides asks, and then they borrow shades. Before 11-01 the
    # labels mk-flat and mk-ser-flat named the pixel-order launches
    # (gpu-optix-mega-flat, gpu-optix-mega-ser-flat), which PLAN.md's
    # entries up to 10-31 still call them.
    "wf-sort": RAMP8[1],            # gpu-optix-sort
    "wf-fused": RAMP8[1],           # gpu-optix-fused: the material stage alone in the trace program
    "wf-fused-ser": RAMP8[2],       # gpu-optix-fused-ser: that with reordering by material
    "wf-fused-shadow": RAMP8[3],    # gpu-optix-fused-shadow: the shadow ray traced in the trace program too
    "wf-pass": RAMP8[3],            # gpu-optix-pass: passes of four samples
    "mk": RAMP8[3],                 # gpu-optix-mega
    "mk-flat-px": RAMP8[4],         # gpu-optix-mega-flat: the flat launch, pixel order
    "mk-flat-ser-px": RAMP8[5],     # gpu-optix-mega-ser-flat
    "mk-pass": RAMP8[4],
    "mk-blk": RAMP8[5],
    "mk-ser": RAMP8[6],
    # The CPU sweep's legend, top to bottom: the wavefront that matches
    # pbrt's first, the wavefronts that win cells, then the megakernels,
    # the one that matches pbrt's megakernel last among them and PBRT mk
    # (black) after it, so the two sit side by side (the user, 11-01). The
    # ramp skips one shade between the families, as the GPU sweep's does.
    "wf-b-s-sp": RAMP8[0],          # wavefront-frame, pbrt's wavefront matched
    "wf-t-v-ps": RAMP8[1],          # wavefront-thread-packet: per-thread queues, vectorized drains
    "wf-t256-v-ps": RAMP8[2],       # wavefront-thread-packet-chunk256: 256 pixels a thread
    "wf-t-v-ps-fuse": RAMP8[3],     # wavefront-thread-packet-fused: the material stage in the trace drain's gang
    "wf-t-v-ps-sort": RAMP8[4],     # wavefront-thread-packet-sort
    "mk-b-s-ps": RAMP8[6],          # scalar: a pixel's samples in turn, the pixels in the runtime's chunks
    "mk-b-s-ps-tiled": RAMP8[7],    # scalar-tiled: pbrt's megakernel matched, its 32x32 tiles
    # CPU sides out of the figure (11-01: never best on a cell), drawn only
    # when --sides asks; they borrow shades.
    "wf-b-v-sp": RAMP8[1],
    "wf-b-v-sp-sort": RAMP8[2],
    "wf-t-s-ps": RAMP8[3],
    "wf-t4096-v-ps": RAMP8[2],      # wavefront-thread-packet-chunk4096
    "wf-t-v-sp": RAMP8[5],
    "mk-b-s-sp": RAMP8[5],          # scalar-sample-outer
    "mk-b-s-w16": RAMP8[5],         # scalar-waves16
    "mk-b-s-ps-c1k": RAMP8[6],      # scalar-chunk1024
    "mk-b-s-flat": RAMP8[6],        # scalar-collapsed
    # The GPU exploration's sides of 10-24, drawn only when asked for
    # (--sides): they borrow shades, so a figure naming them should name
    # a set without a repeat (wf, wf-b128, wf-b512, wf-fused, wf-fused-ser,
    # wf-sort, wf-sort36 is one).
    "mk-ser-ray": "#9E9E9E",        # a neutral: the control of the reorder key
    "wf-b128": RAMP8[2],
    "wf-b512": RAMP8[3],
    "wf-sort36": RAMP8[1],
}
