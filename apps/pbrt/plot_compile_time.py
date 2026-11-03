#!/usr/bin/env python3
"""The compiler's time on each schedule of the two sweeps, one bar per
schedule, the CPU sweep's then the GPU sweep's (the user, 11-03):

    python3 apps/pbrt/plot_compile_time.py apps/pbrt/results/compile-time/compile-times.csv \\
        -o apps/pbrt/plots/pbrt-compile-time.pdf

(plots.sh draws it under that name.)

The rows are compile_time.sh's, one per build; a bar is a schedule's
`compiler_seconds` (the least, when a schedule was built more than once).
The bars stand in the sweep figures' order and colours
(plot_cpu_wavefront.py's and plot_gpu_wavefront.py's SIDES, through
figure_style.SCHEDULE_COLOURS), under the sweep figures' labels, so a bar
here is the bar of the same colour there. The y axis is the compile time,
its ticks in minutes. A dashed line marks pbrt-v4's own build time, the
row compile_time.sh adds for it (the user, 11-03: "not an exact
comparison", and the .txt says what it is).

The .txt beside the figure has every drawn schedule's file, its compiler
seconds over the builds and the driver's, the machine and the compiler's
commit, pbrt's build described, and the reading: which schedules compile
slowest and what in their compilation takes the time.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from figure_style import (PBRT_COLOUR, TEXT_WIDTH, TICK_FONT, lean_height, lean_labels,
                          paper_style)
from plot_cpu_wavefront import SIDES as CPU_SIDES
from plot_gpu_wavefront import SIDES as GPU_SIDES

EDGE = "#333333"


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def minutes_label(seconds, _pos=None):
    if seconds == 0:
        return "0"
    minutes = seconds / 60.0
    return f"{minutes:g} min"


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", nargs="+", help="compile_time.sh's CSV(s)")
    parser.add_argument("-o", "--out", required=True, help="the figure (.pdf or .png)")
    parser.add_argument("--width", type=float, default=TEXT_WIDTH)
    parser.add_argument("--height", type=float, default=1.75)
    args = parser.parse_args(argv)

    rows = [row for path in args.csv for row in read_rows(path)]
    if not rows:
        sys.exit("no rows")
    compiler = defaultdict(list)   # schedule -> compiler seconds over the builds
    driver = defaultdict(list)
    disturbed = defaultdict(int)
    commits, hosts = set(), set()
    pbrt = []                      # pbrt-v4's own build, wall seconds, if measured
    pbrt_commit = None
    for row in rows:
        if row["side"] == "pbrt":
            pbrt.append(float(row["compiler_seconds"]))
            pbrt_commit = row["commit"]
            hosts.add(row["host"])
            continue
        compiler[row["schedule"]].append(float(row["compiler_seconds"]))
        driver[row["schedule"]].append(float(row["driver_seconds"]))
        disturbed[row["schedule"]] += int(row.get("disturbed", "0") or 0)
        commits.add(row["commit"])
        hosts.add(row["host"])
    pbrt_note = None
    if pbrt:
        for path in args.csv:
            note = os.path.join(os.path.dirname(path), "pbrt-build.txt")
            if os.path.exists(note):
                with open(note) as f:
                    pbrt_note = [line.rstrip("\n") for line in f]

    # The bars: the sweep figures' sides that have rows, in their order.
    groups = []  # (group name, [(schedule, label, colour, least seconds)])
    for name, sides in (("CPU", CPU_SIDES), ("GPU", GPU_SIDES)):
        bars = [(side, label, colour, min(compiler[side]))
                for side, label, colour in sides if side in compiler]
        if bars:
            groups.append((name, bars))
    drawn = {side for _, bars in groups for side, _, _, _ in bars}
    unplaced = sorted(set(compiler) - drawn)
    if not groups:
        sys.exit("no schedule of the sweeps in the rows")

    paper_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    gap = 1.0  # one bar's width between the groups
    x = 0.0
    positions, labels, centres = [], [], []
    for name, bars in groups:
        start = x
        for side, label, colour, seconds in bars:
            ax.bar(x, seconds, 0.8, color=colour, edgecolor=EDGE, linewidth=0.3, zorder=2)
            positions.append(x)
            labels.append(label)
            x += 1.0
        centres.append((name, (start + x - 1.0) / 2.0))
        x += gap
    top = max(seconds for _, bars in groups for _, _, _, seconds in bars)
    # pbrt-v4's own build, the dashed line (the user, 11-03: not an exact
    # comparison -- a parallel build of the whole renderer against one
    # thread of ours on one schedule -- and the .txt says so).
    if pbrt:
        reference = min(pbrt)
        ax.axhline(reference, color=PBRT_COLOUR, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
        ax.text(x - gap - 0.4, reference, " pbrt-v4", ha="right", va="bottom",
                fontsize=TICK_FONT, color=PBRT_COLOUR)
        top = max(top, reference)
    step = 60.0 if top > 150 else 30.0
    ax.set_ylim(0, top * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FuncFormatter(minutes_label))
    ax.set_ylabel("Compile time")
    ax.set_xticks(positions)
    xmin, xmax = -0.6, x - gap - 0.4
    ax.set_xlim(xmin, xmax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The labels lean as the figure's width needs (figure_style.lean_labels
    # wants the inches between ticks), and the group names sit below them.
    fig.tight_layout(pad=0.3)
    width_in = ax.get_window_extent().width / fig.dpi
    angle = lean_labels(ax, labels, width_in / (xmax - xmin))
    below_in = lean_height(labels, angle)
    for name, centre in centres:
        ax.annotate(name, (centre, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -(below_in * 72 + 4)), textcoords="offset points",
                    ha="center", va="top")
    fig.subplots_adjust(bottom=min(0.6, (below_in + 0.22) / args.height))
    fig.savefig(args.out, dpi=200)

    # The summary beside the figure, in write_summary's layout but in
    # seconds: figure_style.describe speaks in speedups.
    def builds(seconds):
        least = min(range(len(seconds)), key=lambda i: seconds[i])
        return (f"least {seconds[least]:.1f} s (build {least + 1})  "
                f"builds {', '.join(f'{s:.1f}' for s in seconds)} s")
    files = {side: f"schedules/{side}.bonsai" for side in compiler}
    sections = []
    for name, bars in groups:
        sections.append((f"{name} schedules: the compiler's seconds over the builds (the least is drawn)",
                         [(f"{label} ({files[side]})", builds(compiler[side])) for side, label, _, _ in bars]))
    sections.append(("the driver's seconds (clang on apps/pbrt's driver, the same code for every schedule)",
                     [(label, builds(driver[side])) for _, bars in groups for side, label, _, _ in bars]))
    if unplaced:
        sections.append(("rows of schedules the sweep figures do not draw (not in the figure)",
                         [(side, builds(compiler[side])) for side in unplaced]))
    builds = sorted({len(v) for v in compiler.values()})
    slowest = max(((s, min(v)) for s, v in compiler.items() if s in drawn), key=lambda p: p[1])
    fastest = min(((s, min(v)) for s, v in compiler.items() if s in drawn), key=lambda p: p[1])
    how_many = ("one build" if builds == [1]
                else f"the least of {' or '.join(str(b) for b in builds)} builds")
    notes = [
        f"a bar is {how_many} of the schedule's renderer, exactly as a comparison builds it "
        f"(compare.sh --build-renderer), on {', '.join(sorted(hosts))}, the compiler at "
        f"{', '.join(sorted(commits))}; the compiler runs on one thread;",
        "the compiler's seconds cover parsing, scheduling, the SSA passes, LLVM's optimization and the "
        "object, and for a GPU schedule its PTX and OptiX modules; the driver's clang build is the same "
        "for every schedule and is listed, not drawn;",
        f"slowest {slowest[0]} at {slowest[1]:.0f} s, fastest {fastest[0]} at {fastest[1]:.0f} s "
        f"({slowest[1] / fastest[1]:.1f}x);",
    ]
    if pbrt:
        notes.append(f"the dashed line is pbrt-v4's own build, {min(pbrt):.0f} s of wall time"
                     f"{f' at {pbrt_commit}' if pbrt_commit else ''}: the whole renderer, its CPU integrators, "
                     "its GPU wavefront and its bundled dependencies, from a clean build directory configured "
                     "as the reference build is, with a job per core -- not an exact comparison with one "
                     "thread of our compiler on one schedule, and marked so by the dashing;")
        if pbrt_note:
            notes.extend(f"  {line}" for line in pbrt_note)
    noisy = [s for s in drawn if disturbed[s]]
    if noisy:
        notes.append(f"builds delayed past ten minutes by another process and run anyway: {', '.join(sorted(noisy))}.")
    write_seconds_summary(args.out, "compile time of each schedule of the sweeps: the compiler's wall "
                          "seconds, the least over the builds", sections, notes)


def write_seconds_summary(figure_path, title, sections, notes):
    """figure_style.write_summary's layout -- the title, each section's
    heading and its aligned rows, then `reading the figure` -- with each
    row's text given, since describe() states speedups and these are
    seconds."""
    path = os.path.splitext(figure_path)[0] + ".txt"
    with open(path, "w") as f:
        f.write(f"{title}\n")
        for heading, rows in sections:
            f.write(f"\n{heading}\n")
            width = max((len(name) for name, _ in rows), default=0)
            for name, text in rows:
                f.write(f"  {name:<{width}}  {text}\n")
        if notes:
            f.write("\nreading the figure\n")
            for line in notes:
                f.write(f"  {line}\n")
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
