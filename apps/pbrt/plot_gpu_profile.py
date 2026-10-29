#!/usr/bin/env python3
"""The profile figure: why the megakernel loses to the wavefront on one
scene and beats it on another, read off Nsight Compute, one row of three
panels per scene (the user, 10-28: "a plot that shows the relevant parts
of this information for PLDI ... compare the Bonsai wf trace kernel to the
Bonsai mk", on ganesha and on a killeroo scene where the megakernel wins).

    python3 apps/pbrt/plot_gpu_profile.py --cell ganesha:16 --cell killeroo-gold:16 \
        [--sides gpu-optix,gpu-optix-mega-flat,gpu-optix-mega-ser-flat] \
        [--profiles apps/pbrt/profile-out] -o apps/pbrt/plots/pbrt-gpu-profile.pdf

(plots.sh draws it under that name, the figure's fourth fixed name since
10-28.)

Each side of each cell is one profiled OptiX launch, written by
profile_kernels.sh as PROFILES/SCENE-SPP-SIDE-L<n>.raw.csv (every metric
of the full section set, one row) and .source.csv (the launch's SASS, one
line per instruction with its executed-instruction and thread counts). For
the wavefront (gpu-optix, drawn `wf trace`) the launch is its first
closest-hit trace, the raygen over the ray queue -- the kernel that does
in the wavefront what the megakernel's raygen does in one piece with the
shading; for the megakernels (`mk-flat`, `mk-ser-flat`) it is the paths'
raygen, the frame. A side without a profile is left out of its row with a
note on stderr.

The three panels, per scene:

  Kernel size      the launch's SASS instructions over every function in
                   it (raygen, its continuations, hit and miss programs),
                   log scale: how much code the warps fetch through.
  Lanes active     threads executing per warp instruction, of 32, over the
                   launch: the width at which every instruction is paid.
  Warp cycles      where a warp's cycles between issues go, as shares of the
                   warp-cycles per issued instruction (the total written at
                   the bar's end): the instruction fetch (Nsight's
                   `no_instruction`: nothing fetched to issue), the wait on
                   the traversal hardware (`barrier`: our raygens have no
                   barrier of their own, the wait is OptiX's for the
                   trace's answer), memory (`long_scoreboard`, a load's
                   latency, with the `lg_throttle` and `mio_throttle` queue
                   stalls), and the rest (the issue itself, fixed-latency
                   waits, short scoreboards, branch resolution, ...).

Beside the figure, `<figure>.txt`: for every launch the duration, the
instructions issued, registers, achieved warps per SM, issue-slot use, the
four stall components in cycles, the lanes, the SASS size and the lanes of
each function in the launch (OptiX splits a raygen at each optixTrace into
continuations, `_ss_0` before the first trace and `_ss_1`, `_ss_2` after
one, so the lanes before and after the first bounce are told apart), and
where the reading is written up in PLAN.md. Each launch's bars in the size
and lanes panels take the colour its schedule has on the sweep figures
(figure_style.SCHEDULE_COLOURS: the wavefront's trace kernel `wf`'s pale
ramp class, the two megakernels their greys; the user, 10-29, one scheme
for every figure); the four stall shares are classes of the ramp, darkest
for the fetch, as before the schedules had colours of their own.
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from figure_style import FONT, TEXT_WIDTH, TICK_FONT, YLGNBU, paper_style
from plot_gpu_wavefront import SIDES

DEFAULT_SIDES = "gpu-optix,gpu-optix-mega-flat,gpu-optix-mega-ser-flat"
LABELS = {side: label for side, label, _ in SIDES}
COLOURS = {side: colour for side, _, colour in SIDES}   # the schedule's colour on every figure
LABELS["gpu-optix"] = "wf trace"      # the wavefront's profiled launch is its trace kernel

# The stall components, in stacking order (left to right), each with the
# Nsight metrics summed into it and its class of the ramp, darkest for
# the fetch (the greys are the megakernel schedules', in the panels
# beside this one).
STALLS = [
    ("instruction fetch", ["no_instruction"], YLGNBU[7]),
    ("traversal wait", ["barrier"], YLGNBU[5]),
    ("memory", ["long_scoreboard", "lg_throttle", "mio_throttle"], YLGNBU[3]),
    ("other", None, YLGNBU[1]),   # the remainder of the total
]
STALL_METRIC = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
TOTAL_METRIC = "smsp__average_warp_latency_per_inst_issued.ratio"


def number(text):
    text = text.replace(",", "").strip()
    if text in ("", "n/a") or text.startswith("(!)"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_raw(path):
    """The one launch's metrics of a .raw.csv as {metric: (value, unit)}."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, units, data = rows[0], rows[1], rows[2:]
    if not data:
        raise ValueError(f"{path}: no launch in it")
    # Nsight's metrics as numbers; the profiler's own bonsai__ columns as text.
    return {h: ((v.strip() or None) if h.startswith("bonsai__") else number(v), u)
            for h, u, v in zip(header, units, data[0])}


def read_source(path):
    """The functions of a .source.csv: [(name, instructions, executed warp
    instructions, executed thread instructions)] -- the static size of the
    kernel and the lanes active per function. Trailing NOP padding is not
    counted in the size."""
    functions = []
    name, header, lines = None, None, []
    def close():
        if name is not None:
            size = sum(1 for l in lines if not re.match(r"^\s*(@!?U?P\w+\s+)?NOP\b", l["Source"]))
            inst = sum(number(l.get("Instructions Executed", "")) or 0 for l in lines)
            threads = sum(number(l.get("Thread Instructions Executed", "")) or 0 for l in lines)
            functions.append((name, size, inst, threads))
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "Kernel Name":
                close()
                name, header, lines = row[1], None, []
            elif row[0] == "Address":
                header = row
            elif header is not None:
                lines.append(dict(zip(header, row)))
    close()
    return functions


def find_profile(profiles, scene, spp, side):
    """The stem of the profile of (scene, spp, side) under `profiles`, the
    lowest launch ordinal when several were profiled, or None."""
    # SCENE-SPP-SIDE-L<n>.raw.csv only: not the .multipass.raw.csv the
    # single-pass timing keeps beside it.
    paths = [p for p in glob.glob(os.path.join(profiles, f"{scene}-{spp}-{side}-L*.raw.csv"))
             if re.search(r"-L(\d+)\.raw\.csv$", p)]
    stems = sorted(paths, key=lambda p: int(re.search(r"-L(\d+)\.raw\.csv$", p).group(1)))
    return stems[0][: -len(".raw.csv")] if stems else None


def load(profiles, cells, sides):
    """{(scene, spp): [launch, ...]} in `sides` order; a launch is a dict of
    the figure's quantities with `label`, `side` and `functions`."""
    loaded = collections.OrderedDict()
    for scene, spp in cells:
        row = []
        for side in sides:
            stem = find_profile(profiles, scene, spp, side)
            if stem is None:
                print(f"plot_gpu_profile.py: no profile of {side} on {scene} at {spp} spp under {profiles}; left out",
                      file=sys.stderr)
                continue
            raw = read_raw(stem + ".raw.csv")
            value = lambda m: raw.get(m, (None, ""))[0]
            total = value(TOTAL_METRIC)
            parts = []
            for name, metrics, colour in STALLS:
                if metrics is None:
                    parts.append((name, max(0.0, total - sum(p[1] for p in parts)), colour))
                else:
                    parts.append((name, sum(value(STALL_METRIC.format(m)) or 0.0 for m in metrics), colour))
            functions = read_source(stem + ".source.csv") if os.path.exists(stem + ".source.csv") else []
            row.append({
                "side": side, "label": LABELS.get(side, side), "stem": stem,
                "colour": COLOURS.get(side, "#636363"),
                "duration_ms": (value("gpu__time_duration.sum") or 0.0)
                               / {"ns": 1e6, "nsecond": 1e6, "us": 1e3, "usecond": 1e3}.get(raw.get("gpu__time_duration.sum", (0, "ms"))[1], 1.0),
                "duration_unit": raw.get("gpu__time_duration.sum", (0, ""))[1],
                "instructions": value("smsp__inst_executed.sum"),
                "registers": value("launch__registers_per_thread"),
                "threads": value("launch__thread_count"),
                "occupancy_pct": value("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "issue_pct": value("smsp__issue_active.avg.pct_of_peak_sustained_active"),
                "lanes": value("smsp__thread_inst_executed_per_inst_executed.ratio"),
                "provenance": value("bonsai__timing_provenance"),
                "cycles": total, "stalls": parts,
                "size": sum(f[1] for f in functions) or None,
                "functions": functions,
            })
        loaded[(scene, spp)] = row
    return loaded


def draw(loaded, out):
    paper_style()
    rows = len(loaded)
    height = 0.42 + 0.78 * rows + 0.25
    fig, axes = plt.subplots(rows, 3, figsize=(TEXT_WIDTH, height), squeeze=False,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.35], "wspace": 0.12,
                                          "hspace": 0.55})
    for r, ((scene, spp), launches) in enumerate(loaded.items()):
        size_ax, lanes_ax, stall_ax = axes[r]
        labels = [l["label"] for l in launches]
        y = list(range(len(launches)))[::-1]
        bar_h = 0.62
        # Kernel size, log scale.
        sizes = [l["size"] or 0 for l in launches]
        colours = [l["colour"] for l in launches]
        size_ax.barh(y, sizes, height=bar_h, color=colours, edgecolor="none")
        size_ax.set_xscale("log")
        lo = min((s for s in sizes if s), default=1000) / 3.0
        size_ax.set_xlim(lo, max(sizes + [1000]) * 9.0)
        size_ax.set_yticks(y)
        size_ax.set_yticklabels(labels)
        size_ax.set_ylabel(f"{scene}, {spp} spp", fontsize=FONT)
        for yy, s in zip(y, sizes):
            if s:
                size_ax.text(s * 1.12, yy, f"{s / 1000:.0f}k" if s >= 10000 else f"{s / 1000:.1f}k",
                             va="center", ha="left", fontsize=TICK_FONT)
        ticks = [t for t in (1e2, 1e3, 1e4, 1e5, 1e6) if lo <= t <= max(sizes + [1000]) * 9.0]
        size_ax.set_xticks(ticks)
        size_ax.set_xticklabels([{1e2: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M"}[t] for t in ticks])
        size_ax.xaxis.set_minor_formatter(NullFormatter())
        size_ax.tick_params(axis="x", which="minor", length=0)
        # Lanes active, of 32.
        lanes = [l["lanes"] or 0 for l in launches]
        lanes_ax.barh(y, lanes, height=bar_h, color=colours, edgecolor="none")
        lanes_ax.set_xlim(0, 32)
        lanes_ax.set_xticks([0, 8, 16, 24, 32])
        lanes_ax.set_yticks(y)
        lanes_ax.set_yticklabels([])
        for yy, v in zip(y, lanes):
            # A full-lane bar's label sits inside the bar, in white: past 32
            # the stall panel's axes would draw over it.
            if v > 26:
                lanes_ax.text(v - 0.6, yy, f"{v:.1f}", va="center", ha="right", fontsize=TICK_FONT,
                              color="white")
            else:
                lanes_ax.text(v + 0.6, yy, f"{v:.1f}", va="center", ha="left", fontsize=TICK_FONT)
        # Warp cycles by stall, shares of the total.
        for yy, l in zip(y, launches):
            total = l["cycles"] or 0.0
            left = 0.0
            for name, cycles, colour in l["stalls"]:
                share = 100.0 * cycles / total if total else 0.0
                stall_ax.barh(yy, share, left=left, height=bar_h, color=colour,
                              edgecolor="white", linewidth=0.4)
                left += share
            stall_ax.text(101.5, yy, f"{total:.0f} cyc/inst", va="center", ha="left",
                          fontsize=TICK_FONT, clip_on=False)
        stall_ax.set_xlim(0, 100)
        stall_ax.set_xticks([0, 25, 50, 75, 100])
        stall_ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        stall_ax.set_yticks(y)
        stall_ax.set_yticklabels([])
        for ax in axes[r]:
            ax.set_ylim(-0.6, len(launches) - 0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", linewidth=0.4, color="#DDDDDD", zorder=0)
            ax.set_axisbelow(True)
        if r == 0:
            size_ax.set_title("Kernel size (SASS instr.)", fontsize=FONT, pad=4)
            lanes_ax.set_title("Lanes active (of 32)", fontsize=FONT, pad=4)
            stall_ax.set_title("Where a warp's cycles go", fontsize=FONT, pad=4)
    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.4, label=n) for n, _, c in STALLS]
    fig.legend(handles=handles, loc="lower center", ncol=len(STALLS), frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=FONT, handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.13, right=0.895, top=1 - 0.30 / height, bottom=0.42 / height)
    fig.savefig(out)
    plt.close(fig)


def write_summary(loaded, out):
    path = os.path.splitext(out)[0] + ".txt"
    with open(path, "w") as f:
        f.write("pbrt-gpu-profile: the wavefront's trace kernel against the megakernels under Nsight Compute,\n"
                "one OptiX launch each (profile_kernels.sh; the full section set under kernel replay; a launch whose\n"
                "timing the replay's memory restore distorted carries its timing from single-pass runs, noted below).\n")
        for (scene, spp), launches in loaded.items():
            f.write(f"\n{scene}, {spp} spp\n")
            for l in launches:
                f.write(f"  {l['label']} ({l['side']}, {os.path.basename(l['stem'])})\n")
                f.write(f"    duration {l['duration_ms']:.2f} ms  threads {int(l['threads'] or 0):,}  "
                        f"warp instructions {int(l['instructions'] or 0):,}  registers {int(l['registers'] or 0)}\n")
                f.write(f"    warps per SM achieved {(l['occupancy_pct'] or 0) * 0.48:.1f} of 48 ({l['occupancy_pct'] or 0:.1f}%)  "
                        f"issue slots busy {l['issue_pct'] or 0:.1f}%  lanes active {l['lanes'] or 0:.2f} of 32\n")
                if l["provenance"]:
                    f.write(f"    timing metrics from {l['provenance']} (profile_kernels.sh --single-pass-timing)\n")
                total = l["cycles"] or 0.0
                parts = "  ".join(f"{n} {c:.1f} ({100 * c / total if total else 0:.0f}%)" for n, c, _ in l["stalls"])
                f.write(f"    warp-cycles per issued instruction {total:.1f}: {parts}\n")
                f.write(f"    SASS instructions {l['size'] or 0:,} over {len(l['functions'])} functions:\n")
                for name, size, inst, threads in sorted(l["functions"], key=lambda x: -x[2]):
                    lanes = threads / inst if inst else 0.0
                    share = 100 * inst / l["instructions"] if l["instructions"] else 0.0
                    f.write(f"      {size:>7,} instr  {inst:>14,.0f} executed ({share:4.1f}%)  lanes {lanes:5.2f}  {name}\n")
        f.write("\nreading the figure\n")
        f.write("  the stall components are Nsight Compute's smsp__average_warps_issue_stalled_*_per_issue_active:\n"
                "    instruction fetch = no_instruction; traversal wait = barrier (OptiX's wait for the trace's answer;\n"
                "    our raygens have no barrier of their own); memory = long_scoreboard + lg_throttle + mio_throttle;\n"
                "    other = the remainder of smsp__average_warp_latency_per_inst_issued (the issue itself, wait,\n"
                "    short_scoreboard, branch_resolving, not_selected, ...).\n")
        f.write("  the lanes and the size are over the launch's every function; the per-function lines above tell\n"
                "    the raygen's continuations apart (_ss_0 up to the first optixTrace, _ss_1/_ss_2 after one).\n")
        f.write("  causes written up in apps/pbrt/PLAN.md: '10-28, the night' -> 'pbrt's trace kernel profiled, and read\n"
                "    against the megakernel's (10-28)' and the morning's ganesha reading before it; 'The megakernel\n"
                "    against Dr.Jit's (10-27)' for the megakernel's shapes.\n")
    return path


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, metavar="SCENE:SPP",
                    help="a scene and sample count to draw a row for (repeat for several rows)")
    ap.add_argument("--sides", default=DEFAULT_SIDES,
                    help=f"the schedules to draw, in order (default {DEFAULT_SIDES})")
    ap.add_argument("--profiles", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile-out"),
                    help="the directory profile_kernels.sh wrote to")
    ap.add_argument("-o", "--out", default="pbrt-gpu-profile.pdf")
    args = ap.parse_args(argv[1:])
    cells = []
    for c in args.cell:
        scene, _, spp = c.partition(":")
        if not spp.isdigit():
            print(f"plot_gpu_profile.py: --cell wants SCENE:SPP, got {c!r}", file=sys.stderr)
            return 2
        cells.append((scene, int(spp)))
    sides = [s for s in args.sides.split(",") if s]
    loaded = load(args.profiles, cells, sides)
    if not any(loaded.values()):
        print("plot_gpu_profile.py: nothing to draw", file=sys.stderr)
        return 1
    loaded = collections.OrderedDict((k, v) for k, v in loaded.items() if v)
    draw(loaded, args.out)
    summary = write_summary(loaded, args.out)
    print(f"wrote {args.out} and {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
