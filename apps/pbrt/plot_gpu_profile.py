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

  SASS instructions        the launch's instructions over every function
                           in it (raygen, its continuations, hit and miss
                           programs), log scale: the code the warps fetch
                           through.
  Active threads per warp  threads executing per warp instruction, of 32,
                           over the launch: the width every instruction
                           is paid at.
  Warp stall reasons       Nsight's stall reasons as shares of the
                           warp-cycles per issued instruction (the total at
                           the bar's end), under Nsight's names: `no
                           instruction` (nothing fetched to issue),
                           `barrier` (our raygens have no barrier of their
                           own; the wait is OptiX's for the traversal
                           hardware's answer), `long scoreboard` (a load's
                           latency; Nsight's LG and MIO throttles, the
                           load/store and MIO queues full, are summed into
                           it), `other` (the remainder: the
                           issue itself, fixed-latency waits, short
                           scoreboards, branch resolution, ...; the .txt
                           lists each launch's). The row labels are the
                           scenes; the sample count is the caption's (the
                           user, 10-29).

Beside the figure, `<figure>.txt`: for every launch the duration, the
instructions issued, registers, achieved warps per SM, issue-slot use, the
four stall components in cycles, the lanes, the SASS size and the lanes of
each function in the launch (OptiX splits a raygen at each optixTrace into
continuations, `_ss_0` before the first trace and `_ss_1`, `_ss_2` after
one, so the lanes before and after the first bounce are told apart), and
where the reading is written up in PLAN.md. Each launch's bars in the size
and lanes panels take the colour its schedule has on the sweep figures
(the user, 10-29) -- here one class of the paper's ramp for every bar of
the single-series panels, the rows being named by their schedule, and
four classes of the ramp for the stall shares, darkest for the fetch.
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
BAR_COLOUR = YLGNBU[6]                                 # the single-series panels' bars
COLOURS = {side: BAR_COLOUR for side, _, _ in SIDES}
LABELS["gpu-optix"] = "wf trace"      # the wavefront's profiled launch is its trace kernel

# The stall components, in stacking order (left to right), each with the
# Nsight metrics summed into it and its class of the ramp, darkest for
# the fetch.
# The names are Nsight's own stall reasons (the user, 10-29: plain
# labels; the caption says what each means for an OptiX launch -- the
# barrier is the warp's wait for the traversal hardware's answer, our
# raygens having no barrier of their own).
STALLS = [
    ("no instruction", ["no_instruction"], YLGNBU[7]),
    ("barrier", ["barrier"], YLGNBU[5]),
    # Named for its largest part (the user, 10-30: the legend reads "long
    # scoreboard"); the LG and MIO throttles, the memory queues full, are
    # summed into it and the .txt says so.
    ("long scoreboard", ["long_scoreboard", "lg_throttle", "mio_throttle"], YLGNBU[3]),
    ("other", None, YLGNBU[1]),   # the remainder of the total
]
STALL_METRIC = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
STALL_METRIC_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
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


def read_singlepass(stem, reason):
    """The stall reason's ratio from the launch's single-pass run of it,
    `<stem>.singlepass-<reason>.csv` (profile_kernels.sh --single-pass-
    timing: Nsight's CSV, the header row the one starting `ID`, then the
    units, then the launch); None when there is no such file or value."""
    path = f"{stem}.singlepass-{reason}.csv"
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    start = next((k for k, row in enumerate(rows) if row and row[0] == "ID"), None)
    if start is None or len(rows) <= start + 2:
        return None
    header, data = rows[start], rows[start + 2]
    metric = STALL_METRIC.format(reason)
    return number(data[header.index(metric)]) if metric in header else None


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
            # What "other" is made of: every stall reason Nsight reports
            # that no named group holds, largest first (the .txt lists
            # them). A launch timed in single passes (profile_kernels.sh
            # --single-pass-timing) has each reason's single-pass value
            # beside it, .singlepass-<reason>.csv; the merge writes only
            # the drawn reasons into .raw.csv, so the rest are read from
            # those files, the replay's distorted counters standing in
            # only where a file is missing.
            named = {m for _, metrics, _ in STALLS if metrics for m in metrics}
            other_parts = []
            for key, (v, _) in raw.items():
                m = STALL_METRIC_RE.match(key)
                if not m or m.group(1) in named:
                    continue
                single = read_singlepass(stem, m.group(1))
                other_parts.append((m.group(1), single if single is not None else v))
            other_parts = sorted(((r, v) for r, v in other_parts if v), key=lambda kv: -kv[1])
            functions = read_source(stem + ".source.csv") if os.path.exists(stem + ".source.csv") else []
            row.append({
                "side": side, "label": LABELS.get(side, side), "stem": stem,
                "colour": COLOURS.get(side, BAR_COLOUR),
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
                "cycles": total, "stalls": parts, "other_parts": other_parts,
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
        # The scene alone; the sample count is the caption's (the user, 10-29).
        size_ax.set_ylabel(scene, fontsize=FONT)
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
            size_ax.set_title("SASS instructions", fontsize=FONT, pad=4)
            lanes_ax.set_title("Active threads per warp", fontsize=FONT, pad=4)
            stall_ax.set_title("Warp stall reasons", fontsize=FONT, pad=4)
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
                if l["other_parts"]:
                    f.write("      other is the remainder after the three named groups; Nsight's stall reasons outside them: "
                            + ", ".join(f"{r} {v:.1f}" for r, v in l["other_parts"] if v >= 0.05) + "\n")
                    all_reasons = sum(c for n, c, _ in l["stalls"] if n != "other") + sum(v for _, v in l["other_parts"])
                    if total and all_reasons > 1.02 * total:
                        f.write(f"      (the reasons sum to {all_reasons:.1f} against the total {total:.1f}: kernel replay's "
                                "distortion of this launch's counters; the figure's other is the remainder)\n")
                f.write(f"    SASS instructions {l['size'] or 0:,} over {len(l['functions'])} functions:\n")
                for name, size, inst, threads in sorted(l["functions"], key=lambda x: -x[2]):
                    lanes = threads / inst if inst else 0.0
                    share = 100 * inst / l["instructions"] if l["instructions"] else 0.0
                    f.write(f"      {size:>7,} instr  {inst:>14,.0f} executed ({share:4.1f}%)  lanes {lanes:5.2f}  {name}\n")
        f.write("\nreading the figure\n")
        f.write("  the stall groups are Nsight Compute's smsp__average_warps_issue_stalled_*_per_issue_active, named\n"
                "    as Nsight names them: no instruction (the warp waits to fetch, an instruction-cache miss or the\n"
                "    fetch of a program this size); barrier (OptiX's wait for the traversal hardware's answer -- our\n"
                "    raygens have no barrier of their own); long scoreboard (global and local memory: a load's\n"
                "    result awaited, with Nsight's LG and MIO throttles -- the load/store or MIO queue full -- summed\n"
                "    into it, the bar named for its largest part); other = the remainder of\n"
                "    smsp__average_warp_latency_per_inst_issued, which on these launches is mostly `wait` (a fixed-\n"
                "    latency dependency on an arithmetic result, ~2 cycles), `selected` (the one cycle the instruction\n"
                "    issues), `short_scoreboard` (a shared-memory or MIO result awaited) and, on the trace kernel,\n"
                "    `branch_resolving` and `drain`; the line under each launch lists its own.\n")
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
