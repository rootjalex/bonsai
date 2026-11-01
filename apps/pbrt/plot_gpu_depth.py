#!/usr/bin/env python3
"""Why the wavefronts lose on a deep scene, as one figure of two panels:
transparent-machines/frame542 -- dielectric machines, whose scene file
asks for paths of up to 256 bounces -- where the flat megakernel with
reordering renders 18-20x faster than pbrt --gpu and the matching
wavefront 1.6x (the GPU sweep).

    python3 apps/pbrt/plot_gpu_depth.py apps/pbrt/depth-out/frame542-depth.tsv \\
        --stats apps/pbrt/depth-out/frame542-s128.pbrt.stats \\
        [--log apps/pbrt/depth-out/post_confirm_gpu.log] \\
        -o apps/pbrt/plots/pbrt-gpu-depth.pdf

(plots.sh draws it under that name, the figure's one name since 10-31.)

The left panel is the scene's depth histogram from pbrt --gpu's own
statistics (`--stats`; the "Indirect rays, depth d" counters of its
wavefront, over its camera rays): of the paths a camera ray starts, the
share still alive at each bounce -- 13% leave the first bounce, 4.4% the
tenth, 0.7% the fiftieth, 0.06% the last. Dotted lines mark the four
depth caps of the right panel with the share alive at each.

The right panel is the wall time of the same frame at 16 spp with the
scene's depth capped at 4, 16, 64 and 256 (compare.sh --maxdepth rewrites
the Integrator's maxdepth in the scene text both sides read), the least of
three renders on each side, from the depth sweep's table: pbrt --gpu
(black), the matching wavefront (`Bonsai wf`, gpu-optix) and the flat
megakernel with reordering (`Bonsai mk-flat-ser-px`, gpu-optix-mega-ser-flat,
the pixel-order flat launch; the sweep's label since 11-01),
in the sweep figures' colours (figure_style.SCHEDULE_COLOURS). A
wavefront pays a round of launches for every bounce any path reaches --
seven kernels a round, every one launched whether its queue holds a
million rays or a hundred -- so its time grows with the depth cap long
after the queues have emptied: pbrt --gpu from 0.13 s at depth 4 to 1.33 s
at 256, ours from 0.05 s to 0.80 s. The megakernel's time is the longest
path's, 0.02 s to 0.07 s: one launch, each thread following its path to
its end. Seconds, not speedups, so that the growth itself is what the
panel shows; the speedups over pbrt --gpu at each cap are in the .txt.

The .txt beside the figure (figure_style.write_summary) has each
schedule's speedup over pbrt --gpu across the four caps (range and
geomean first), the share alive at each bounce drawn, every drawn bar's
seconds and its image check from the sweep's log (`--log`; pbrt's own
bars are pbrt's), pbrt's launch counts and trace-kernel times from the
statistics, and the reading.
"""

import argparse
import csv
import math
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import (TEXT_WIDTH, TICK_FONT, PBRT_COLOUR, RAMP8, SCHEDULE_COLOURS,
                          describe, paper_style, write_summary)

try:
    from plot_cpu_wavefront import EDGE
except ImportError:  # the CPU plotter's bar edge, should it move
    EDGE = "#333333"

REFERENCE = "pbrt --gpu"
# (table side, legend label, colour), in drawing order within a depth cap.
SIDES = [
    (REFERENCE, "PBRT wf", PBRT_COLOUR),
    ("gpu-optix", "Bonsai wf", SCHEDULE_COLOURS["wf"]),
    ("gpu-optix-mega-ser-flat", "Bonsai mk-flat-ser-px", SCHEDULE_COLOURS["mk-flat-ser-px"]),
]

CAMERA = re.compile(r"^\s*Camera rays\s+(\d+)\s*$")
INDIRECT = re.compile(r"^\s*Indirect rays, depth (\d+)\s+(\d+)\s*$")
# pbrt's kernel profile lines: name, launches, ms, share, (avg, min, max).
KERNEL = re.compile(r"^\s*(.+?)\s+(\d+) launches\s+([\d.]+) ms /\s*([\d.]+)%\s*\(avg\s*([\d.]+), min\s*([\d.]+), max\s*([\d.]+)\)")
# The depth sweep's log, one line per compare.sh run:
#   depth 4 gpu-optix: rc=0 render time: ... ok: the image matches pbrt
CHECK = re.compile(r"^depth (\d+) (\S+): rc=(\d+) .*?((?:ok|FAILED)[^\n]*?)\s*$")


def load_table(path):
    """The depth sweep's rows: scene, spp (int), maxdepth (int), side,
    wall_s (float), note."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                r["spp"] = int(r["spp"])
                r["maxdepth"] = int(r["maxdepth"])
                r["wall_s"] = float(r["wall_s"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def load_stats(path):
    """pbrt --gpu's statistics: the camera ray count, {depth: indirect
    rays}, and the kernel profile [(name, launches, ms, share, avg, min,
    max)]."""
    camera, alive, kernels = None, {}, []
    with open(path, errors="replace") as f:
        for line in f:
            m = CAMERA.match(line)
            if m:
                camera = int(m.group(1))
                continue
            m = INDIRECT.match(line)
            if m:
                alive[int(m.group(1))] = int(m.group(2))
                continue
            m = KERNEL.match(line)
            if m:
                kernels.append((m.group(1).strip(), int(m.group(2)), float(m.group(3)),
                                float(m.group(4)), float(m.group(5)), float(m.group(6)),
                                float(m.group(7))))
    if camera is None or not alive:
        raise SystemExit(f"{path}: no 'Camera rays' or 'Indirect rays, depth d' counters "
                         f"(pbrt --gpu --stats writes them)")
    return camera, alive, kernels


def load_checks(path):
    """{(maxdepth, side): verdict} from the depth sweep's log."""
    checks = {}
    if path is None:
        return checks
    with open(path, errors="replace") as f:
        for line in f:
            m = CHECK.match(line.rstrip("\n"))
            if m:
                checks[(int(m.group(1)), m.group(2))] = m.group(4)
    return checks


def seconds_label(s):
    return f"{s:.2f}" if s >= 0.1 else f"{s:.3f}"


def percent_label(p):
    return f"{p:.1f}%" if p >= 1.0 else f"{p:.2f}%"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="the depth sweep's table (scene, spp, maxdepth, side, wall_s, note)")
    ap.add_argument("--stats", required=True, help="pbrt --gpu's --stats output at the scene's full depth")
    ap.add_argument("--log", help="the depth sweep's log, for each bar's image check")
    ap.add_argument("-o", "--out", default="pbrt-gpu-depth.pdf")
    ap.add_argument("--width", type=float, default=TEXT_WIDTH)
    ap.add_argument("--height", type=float, default=2.0)
    args = ap.parse_args(argv[1:])

    rows = load_table(args.table)
    camera, alive, kernels = load_stats(args.stats)
    checks = load_checks(args.log)
    scenes = sorted({r["scene"] for r in rows})
    spps = sorted({r["spp"] for r in rows})
    if len(scenes) != 1 or len(spps) != 1:
        raise SystemExit(f"{args.table}: one scene at one sample count expected, "
                         f"got {scenes} at {spps}")
    scene, spp = scenes[0], spps[0]
    caps = sorted({r["maxdepth"] for r in rows})

    def wall(cap, side):
        times = [r["wall_s"] for r in rows if r["maxdepth"] == cap and r["side"] == side]
        return min(times) if times else None

    paper_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(args.width, args.height),
                                     layout="constrained", width_ratios=[1.15, 1.0])

    # (a) the share of paths alive at each bounce.
    depths = sorted(alive)
    share = {d: 100.0 * alive[d] / camera for d in depths}
    xs = depths
    ys = [share[d] for d in depths]
    ax_a.fill_between(xs, ys, color=RAMP8[1], alpha=0.7, linewidth=0, zorder=1)
    ax_a.plot(xs, ys, color=RAMP8[6], linewidth=1.1, zorder=2)
    top = max(ys) * 1.12
    # The caps of the right panel, with the share alive at each. The labels
    # step down the panel in the caps' order and say which bounce they
    # belong to; the deepest cap's sits at the right edge, the others to
    # the right of their line -- or of the last line within the first
    # fifteenth of the axis, where two caps' lines stand too close for a
    # label between them (4 and 16 of 256).
    label_y = [top * 0.80, top * 0.42, top * 0.22, top * 0.10]
    crowd = max(depths) / 15.0
    crowded = [c for c in caps if c <= crowd]
    for i, cap in enumerate(caps):
        if cap not in share:
            continue
        ax_a.axvline(cap, color="#777777", linestyle=(0, (2, 2)), linewidth=0.5, zorder=3)
        y = label_y[i] if i < len(label_y) else top * 0.1
        text = f"{percent_label(share[cap])} at {cap}"
        if cap >= 0.75 * max(depths):
            ax_a.text(cap - 3, y, text, ha="right", va="center", fontsize=TICK_FONT)
        else:
            x = max(cap, crowded[-1]) + 3 if cap in crowded else cap + 3
            ax_a.text(x, y, text, ha="left", va="center", fontsize=TICK_FONT)
    ax_a.set_xlim(0, max(depths) + 1)
    ax_a.set_ylim(0, top)
    ax_a.set_xlabel("bounce")
    ax_a.set_ylabel("paths alive (% of camera paths)")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax_a.set_axisbelow(True)

    # (b) wall time against the depth cap, a bar per side.
    n = len(SIDES)
    bar = 0.8 / n
    drawn = []  # (cap, side, label, seconds)
    for j, (side, label, colour) in enumerate(SIDES):
        for i, cap in enumerate(caps):
            s = wall(cap, side)
            if s is None:
                continue
            x = i + (j - (n - 1) / 2) * bar
            ax_b.bar(x, s, bar * 0.92, color=colour, edgecolor=EDGE, linewidth=0.3, zorder=2)
            drawn.append((cap, side, label, s, x))
    ymax = max(s for _, _, _, s, _ in drawn)
    headroom = 1.42  # the upright value labels' height over the tallest bar
    for cap, side, label, s, x in drawn:
        ax_b.text(x, s + ymax * 0.015, seconds_label(s), rotation=90, ha="center",
                  va="bottom", fontsize=TICK_FONT, zorder=5)
    ax_b.set_ylim(0, ymax * headroom)
    ax_b.set_xticks(range(len(caps)))
    ax_b.set_xticklabels([str(c) for c in caps])
    ax_b.set_xlabel("maximum depth")
    ax_b.set_ylabel(f"seconds ({spp} spp)")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    handles = [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.3, label=label)
               for _, label, colour in SIDES]
    ax_b.legend(handles=handles, frameon=False, loc="upper left", ncol=1)

    fig.savefig(args.out, dpi=200)

    # The summary beside the figure.
    sections = []
    speedups = []
    for side, label, _ in SIDES:
        if side == REFERENCE:
            continue
        points = []
        for cap in caps:
            ref, ours = wall(cap, REFERENCE), wall(cap, side)
            if ref and ours:
                points.append((f"depth {cap}", ref / ours))
        speedups.append((label, points))
    sections.append((f"speedup over {REFERENCE} across the depth caps ({scene}, {spp} spp, "
                     f"the least of three renders on each side)", speedups))

    notes = []
    notes.append(f"paths alive per bounce ({scene}, from {args.stats}: pbrt --gpu's "
                 f"'Indirect rays, depth d' over its {camera:,} camera rays):")
    marks = sorted(set([1, 2, 5, 10, 20, 50, 100] + caps))
    for d in marks:
        if d in share:
            notes.append(f"  bounce {d:>3}: {alive[d]:>12,} rays alive, {share[d]:.3f}% of the paths")
    notes.append(f"the right panel's bars ({args.table}; the image check of each against pbrt's "
                 f"CPU integrator at that depth, from {args.log or 'no log given'}; pbrt's own "
                 f"bars are pbrt's):")
    for cap, side, label, s, _ in drawn:
        if side == REFERENCE:
            check = "pbrt's own render"
        else:
            check = checks.get((cap, side), "no image check found in the log")
        notes.append(f"  depth {cap:>3}  {label:<18} {s:7.4f} s  {check}")
    for side, label, _ in SIDES:
        times = [wall(c, side) for c in caps]
        if all(times):
            notes.append(f"{label}: {seconds_label(times[0])} s at depth {caps[0]} to "
                         f"{seconds_label(times[-1])} s at depth {caps[-1]}, "
                         f"{times[-1] / times[0]:.1f}x the time for {caps[-1] // caps[0]}x the depth")
    trace = [k for k in kernels if k[0].startswith("Trace closest hit")]
    if trace:
        name, launches, ms, pct, avg, lo, hi = trace[0]
        notes.append(f"pbrt --gpu's trace kernel at the scene's full depth, from the statistics: "
                     f"{launches:,} launches, {ms:.0f} ms ({pct:.1f}% of its kernel time), "
                     f"{avg * 1e3:.0f} us a launch on average, {lo * 1e3:.0f} us the least, "
                     f"{hi * 1e3:.0f} us the most -- a round of launches for every bounce any "
                     f"path reaches, most of them over a nearly empty queue")
    per_round = [k for k in kernels if k[1] == max((k[1] for k in kernels), default=0)]
    if per_round:
        notes.append(f"kernels launched once a round: {', '.join(k[0] for k in per_round)} "
                     f"({per_round[0][1]:,} launches each)")
    notes.append("reading: a wavefront's time grows with the depth cap long after the queues have "
                 "emptied, since every bounce any path reaches costs a round of launches of every "
                 "stage's kernel; the megakernel's time is the longest path's, one launch, each "
                 "thread following its own path to its end. Causes measured: apps/pbrt/PLAN.md, "
                 "the 10-30 section's 'frame542's depth sweep' and the kernel profiles before it.")
    write_summary(args.out, f"{scene}: paths alive per bounce, and wall time against the depth cap "
                  f"(pbrt --gpu, the matching wavefront, the flat megakernel with reordering)",
                  sections, notes)
    print(f"wrote {args.out} and its .txt ({len(drawn)} bars, {len(depths)} bounces)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
