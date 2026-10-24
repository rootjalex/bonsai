"""The paper's figure style, shared by apps/pbrt's three plotters (and
imported by apps/rtq's and apps/wosx's if they want it).

The paper is in ACM's acmsmall format (PACMPL / PLDI): a single column,
10 pt text, 12 pt leading, a text block 5.478 in wide and 7.884 in tall.
A figure drawn `TEXT_WIDTH` inches wide and included at `\\textwidth` is
not scaled by LaTeX, so the sizes set here are the sizes on the page:
10 pt for labels and the legend, the body text's size (the user, 10-24:
about 10 pt when printed), 9 pt for tick labels. (acmart's two-column
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
"""

import matplotlib.pyplot as plt

TEXT_WIDTH = 5.478      # inches, acmsmall's text block (the PLDI call for papers)
TEXT_HEIGHT = 7.884

FONT = 10               # labels, legend, the text at the axes
TICK_FONT = 9           # tick labels
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
        "legend.handlelength": 1.6, "legend.handleheight": 0.9,
    })
