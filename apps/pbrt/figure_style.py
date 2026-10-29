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
`<figure>.txt`: for each series the geometric mean, the least value, the
lower quartile, the median, the upper quartile and the greatest value, and
which point each is (the user, 10-24 and 10-26; `describe` has the form).
`YLGNBU` is the paper's colour ramp (the heatmap's cells); `MEGAKERNEL`,
`WAVEFRONT` and `SCHEDULE_COLOURS` give every schedule the one colour it
has on every figure (the user, 10-29). `scene_label` names a scene on a
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
    as one line of text: the count, the geometric mean, then in order the
    least value, the lower quartile, the median, the upper quartile and the
    greatest value, each with the label of the cell it belongs to -- or,
    where a quartile or the median falls between two cells, both cells with
    their values. The paper quotes these beside every figure (the user,
    10-24; the quartiles and the median with their cells, 10-26)."""
    points = [(label, v) for label, v in points if v is not None and v > 0]
    if not points:
        return "no points"
    geomean = math.exp(sum(math.log(v) for _, v in points) / len(points))
    low = min(points, key=lambda p: p[1])
    high = max(points, key=lambda p: p[1])
    return (f"n={len(points)}  geomean {geomean:.2f}x  "
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
# the scene's directory (kroken, watercolor, landscape), and the sweeps
# shorten the two long names below so that eight scenes' names stand
# side by side on one line at text width (the user, 10-29: no name
# broken over two lines).
SCENE_LABELS = {
    "sanmiguel/sanmiguel-courtyard": "sanmiguel",
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

# The schedules' colours, one scheme for every figure (the user, 10-29:
# the CPU and GPU sweeps had each laid the one ramp over its own bars in
# drawing order, so the same colour meant a different schedule on each).
# Two sequential ramps, one per family of schedules, each darker with
# every scheduling step down the figure's legend:
#
#   MEGAKERNEL  warm, ColorBrewer YlOrBr -- a path per thread in one program
#   WAVEFRONT   cool, ColorBrewer YlGnBu (the heatmap's ramp) -- pbrt's queue
#               structure and its variants
#
# so that the family is read from the hue and the variant from the
# lightness, and the same schedule has the same colour on every figure:
# the plain megakernel (gpu-optix-mega on the GPU, the scalar schedule on
# the CPU) is MEGAKERNEL[0] on both sweeps, the matching wavefront
# (gpu-optix, wavefront-frame) WAVEFRONT[0], the sorted wavefront of each
# sweep WAVEFRONT[-1]; the profile figure's bars take their launch's
# schedule colour. pbrt's own bars are black (PBRT_COLOUR; the CPU's PBRT
# mk), pbrt's wavefront the dashed reference line of both sweeps.
#
# The colours are at equal OKLab spacing along each ramp between its
# classes 2.5 and 8 (YlOrBr, six) and 3 and 8 (YlGnBu, seven), chosen by
# apps/pbrt/scratch/palette_pick.py with the dataviz validator's
# distances: bars drawn side by side are at least 8.0 apart (OKLab x100)
# under simulated protan and deutan vision and 9.4 under normal vision,
# and every warm colour at least 12.2 from every cool one under either
# deficiency -- the one-ramp scheme's two palest classes had been 4.7 and
# 5.6 apart. Six or seven ordered shades of a ramp cannot reach the
# validator's categorical floor of 15 between neighbours; the fixed bar
# order within every group, keyed by the legend, is the second encoding.
PBRT_COLOUR = "#000000"
MEGAKERNEL = ["#FED370", "#FEA332", "#ED7315", "#C94A02", "#983404", "#662506"]
WAVEFRONT = ["#7FCDBB", "#3FB4C4", "#1F93C0", "#2071B1", "#234EA0", "#1E2E85", "#081D58"]

# Legend label -> colour, for every schedule a figure of apps/pbrt draws
# (the labels are plot_cpu_wavefront.py's and plot_gpu_wavefront.py's).
SCHEDULE_COLOURS = {
    "PBRT mk": PBRT_COLOUR,
    # The megakernels, from the plain one down the GPU figure's legend.
    "mk": MEGAKERNEL[0],            # GPU gpu-optix-mega
    "mk-b-s-ps": MEGAKERNEL[0],     # CPU scalar: the same schedule, the same colour
    "mk-pass": MEGAKERNEL[1],
    "mk-flat": MEGAKERNEL[2],
    "mk-blk": MEGAKERNEL[3],
    "mk-ser": MEGAKERNEL[4],
    "mk-ser-flat": MEGAKERNEL[5],
    # The wavefronts: pbrt's structure matched, then the CPU figure's
    # variants in its legend's order; the GPU's sorted wavefront takes the
    # last shade, as the CPU's does.
    "wf": WAVEFRONT[0],             # GPU gpu-optix
    "wf-b-s-sp": WAVEFRONT[0],      # CPU wavefront-frame: the same structure
    "wf-b-v-sp": WAVEFRONT[1],
    "wf-b-v-sp-sort": WAVEFRONT[2],
    "wf-t-s-ps": WAVEFRONT[3],
    "wf-t-v-ps": WAVEFRONT[4],
    "wf-t-v-sp": WAVEFRONT[5],
    "wf-t-v-ps-sort": WAVEFRONT[6],
    "wf-sort": WAVEFRONT[6],        # GPU gpu-optix-sort
    # The GPU exploration's sides of 10-24, drawn only when asked for
    # (never beside the CPU variants whose shades they borrow).
    "mk-ser-ray": "#FEE391",        # YlOrBr's class 2, above the family's first
    "wf-b128": WAVEFRONT[1],
    "wf-b512": WAVEFRONT[2],
    "wf-fused": WAVEFRONT[3],
    "wf-fused-ser": WAVEFRONT[4],
    "wf-sort36": WAVEFRONT[5],
}
