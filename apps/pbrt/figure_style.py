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
`<figure>.txt`: for each series the geometric mean, the least and greatest
value and which point each is (the user, 10-24). `YLGNBU` is the paper's
one colour ramp.
"""

import math
import os

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


def describe(points):
    """Summary statistics of `points`, (label, value) pairs of one series:
    the count, the geometric mean, and the least and greatest value each
    with the label of the point it belongs to, as one line of text. The
    paper quotes these beside every figure (the user, 10-24)."""
    points = [(label, v) for label, v in points if v is not None and v > 0]
    if not points:
        return "no points"
    geomean = math.exp(sum(math.log(v) for _, v in points) / len(points))
    low = min(points, key=lambda p: p[1])
    high = max(points, key=lambda p: p[1])
    return (f"n={len(points)}  geomean {geomean:.2f}x  "
            f"min {low[1]:.2f}x ({low[0]})  max {high[1]:.2f}x ({high[0]})")


def ratios(numerators, denominators):
    """(label, a / b) for every label the two series share: one series'
    gain over another, point by point, so that `describe` of it says how
    much the one thing that differs between them bought and where."""
    b = {label: v for label, v in denominators if v}
    return [(label, v / b[label]) for label, v in numerators if label in b and v]


def geomean(points):
    values = [v for _, v in points if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


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


# ColorBrewer YlGnBu, nine classes, lightest first: the paper's one ramp
# (the user's palette, 10-23), used for the bars of the sweeps and the
# cells of the heatmap alike.
YLGNBU = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
          "#1D91C0", "#225EA8", "#253494", "#081D58"]
