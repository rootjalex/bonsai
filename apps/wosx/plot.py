#!/usr/bin/env python3
"""The wosx comparison as plots, from the results CSV benchmark.sh folds
its logs into (one row per measured cell: schedule, mesh, both rates in
steps per second, the speedup).

    python apps/wosx/plot.py apps/wosx/results/wosx-results.csv -o apps/wosx/plots/
    python apps/wosx/plot.py apps/wosx/results/wosx-gpu-results.csv -o apps/wosx/plots/ --gpu

Two CPU figures, one per tree, the meshes along the x axis in order of
triangle count and three bars a mesh; a bar is bonsai's rate over WoSX's
on that cell, the y axis reads Speedup, and the dashed line at 1, in the
legend, is WoSX:

  wosx-cpu-fcpw4w16-speedup  FCPW's four-wide tree, both sides vectorized
                             (`fcpw4w16*`)
  wosx-cpu-fcpwb-speedup     WoSX's default tree, FCPW's scalar binary BVH,
                             both sides as WoSX ships (`fcpwb*`)

The three bars, on either tree: one thread against one thread (`fcpwb`,
`fcpw4w16`); the threads against the threads (`-threads`: WoSX's
tbb::parallel_for over the points against the program's parfor bound to the
same pinned threads, the same TBB); and sixteen persistent workers against
those same threads (`-persistent`: the workers claim points from a counter;
WoSX has no persistent CPU solver, so its side is the threaded solve again).
Walk on spheres has one schedule per thread count and no parameter the
speedup is swept over: a step is one distance query, and the epsilon shell
and the walk count set how many steps a point takes, not what a step costs
(benchmark.sh fixes them at WoSX's demo defaults). The landscape is left out
by default (--exclude names the meshes left out): its normalized box is a
thousandth thick, nearly every walk escapes on its first step, and its bar
measures the cost of an escaping walk, not the traversal.

--gpu draws the GPU figure, wosx-gpu-speedup (benchmark.sh --gpu): every
bar is a GPU schedule of the program over the one reference, WoSX's GPU
solver -- its persistent-thread kernel on FCPW's binary tree, 131072
resident threads each claiming one walk at a time from a counter, a warp's
consecutive claims neighbouring points. The matching schedule (`fcpwb-gpu`,
"persistent") does the same; the variants each change the schedule and
nothing else (--gpu-schedules names them; README.md's schedule list says
what each is): one launch with a thread per walk ("parallel"), a warp's
claims one point's walks ("persistent-reorder"), a point a claim with its
walks in sequence ("persistent-point"), twice the threads so the card is
full ("persistent-256k", "persistent-reorder-256k"), and one launch with
the walk fastest in blocks of 64 ("parallel-reorder-blk64").

Beside each figure (the user's rule, 10-24) go the statistics the paper
quotes, `<figure>.txt`: per schedule the geomean over the plotted meshes,
the least and greatest cell and which mesh each is, what each change of
schedule bought cell by cell, and notes reading the figure; and the mesh
facts go in one table rather than on every plot, `wosx-meshes.txt` to read
and `wosx-meshes.tex` to \\input (a booktabs tabular: triangles, closed or
not, sample points, walks, WoSX's mean walk length), so the tick labels
are the mesh names alone.

The colours are apps/pbrt's figures' (the user's choice, 10-24):
ColorBrewer's nine-class YlGnBu ramp, a colour a schedule, no hatches, the
edge and the line at 1 as there. No titles: the file's name says which tree
and which side, the legend what each bar compares. The figures are drawn
at the size they are printed (apps/pbrt/figure_style.py has the format):
5.478 in wide, acmsmall's text block, what \\includegraphics[width=
\\textwidth] scales a figure to in the paper, 8 pt labels and 7 pt ticks in
Liberation Sans embedded as TrueType (pdf.fonttype 42, which ACM's
production accepts where matplotlib's default Type 3 fonts are refused),
the mesh names leaning only as far as eighteen of them must to clear each
other (figure_style.lean_labels).
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# The paper's figure style, apps/pbrt's module: acmsmall's text width, the
# font sizes, the YlGnBu ramp, Liberation Sans embedded as TrueType, and
# the summary written beside a figure.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrt"))
from figure_style import (TEXT_WIDTH, YLGNBU, describe, lean_labels, paper_style,  # noqa: E402
                          ratios, write_summary)

EDGE = "#333333"  # every bar's thin edge, as apps/pbrt's figures

# The CPU figures' bars, on either tree: (suffix of the tree's schedule,
# legend label, colour), in drawing order. Three bars take classes 1, 4 and
# 7 of the ramp, far enough apart to tell at a glance.
CPU_BARS = [
    ("", "single-threaded", YLGNBU[1]),
    ("-threads", "parallel", YLGNBU[4]),
    ("-persistent", "persistent", YLGNBU[7]),
]
# (schedule base, figure name, what the tree is): the two trees.
CPU_TREES = [
    ("fcpw4w16", "wosx-cpu-fcpw4w16-speedup",
     "FCPW's four-wide tree, both sides vectorized"),
    ("fcpwb", "wosx-cpu-fcpwb-speedup",
     "WoSX's default tree, FCPW's scalar binary BVH, both sides as WoSX ships"),
]
# The GPU figure's bars, every one over WoSX's persistent-thread kernel:
# (schedule, legend label, colour), the labels apps/pbrt's figures' kind --
# the schedule's shape in a word or two; README.md's schedule list says
# what each is. Schedules only: the other layout and the other tree
# (fcpwb2-gpu, fcpw4w16-gpu) are a sweep of their own, not this figure's.
GPU_BARS = [
    ("fcpwb-gpu", "persistent", YLGNBU[1]),
    ("fcpwb-gpu-parallel", "parallel", YLGNBU[2]),
    ("fcpwb-gpu-reorder", "persistent-reorder", YLGNBU[3]),
    ("fcpwb-gpu-point", "persistent-point", YLGNBU[4]),
    ("fcpwb-gpu-256k", "persistent-256k", YLGNBU[5]),
    ("fcpwb-gpu-reorder-256k", "persistent-reorder-256k", YLGNBU[6]),
    ("fcpwb-gpu-parallel-reorder-blk64", "parallel-reorder-blk64", YLGNBU[7]),
]
# What each GPU schedule changed against another, for the summary: (label
# of the schedule, label of the one it differs from in one thing, the thing).
GPU_STEPS = [
    ("parallel", "persistent", "one launch, a thread per walk, instead of the persistent claim"),
    ("persistent-reorder", "persistent", "a warp's claims one point's walks instead of neighbouring points"),
    ("persistent-256k", "persistent", "twice the threads, the card full"),
    ("persistent-reorder-256k", "persistent-reorder", "twice the threads with the walk-fastest claims"),
    ("parallel-reorder-blk64", "parallel", "the walk fastest and blocks of 64 in the one launch"),
    ("persistent-point", "persistent", "a point a claim, its walks in sequence"),
]
REFERENCE = "WoSX"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meshes_by_triangles(rows, exclude):
    tris = {}
    for r in rows:
        if r["mesh"] not in exclude:
            tris[r["mesh"]] = int(r["triangles"])
    return sorted(tris, key=lambda m: tris[m]), tris


def cells(rows, schedule):
    return {r["mesh"]: float(r["speedup"]) for r in rows if r["schedule"] == schedule}


def points(data, meshes):
    """The (mesh, speedup) pairs of one schedule in mesh order."""
    return [(m, data[m]) for m in meshes if m in data]


def geomeans(rows, schedules, exclude):
    """The geometric mean of each schedule's bars over the plotted meshes, the
    least and the greatest named (apps/rtq's rule: every geomean a win, no
    cell more than 3% under the reference; the cells under 0.97 are named)."""
    meshes, _ = meshes_by_triangles(rows, exclude)
    print(f"geomeans, bonsai over WoSX, over {len(meshes)} meshes ({', '.join(meshes)}):")
    print(f"  {'schedule':<20} {'geomean':>8} {'min':>6} {'max':>6}  below 0.97")
    for schedule in schedules:
        data = cells(rows, schedule)
        values = {m: data[m] for m in meshes if m in data}
        if not values:
            continue
        g = math.exp(sum(math.log(v) for v in values.values()) / len(values))
        low = " ".join(f"{m} {v:.2f}" for m, v in values.items() if v < 0.97)
        print(f"  {schedule:<20} {g:8.3f} {min(values.values()):6.2f} {max(values.values()):6.2f}  {low}")


def figure(rows, meshes, bars, ylabel, height):
    """One figure: a bar per (schedule, label, colour) of `bars` per mesh, the
    dashed line at 1 as WoSX in the legend above the axes, the meshes by
    name. Returns the figure and the schedules drawn, or (None, [])."""
    data = {s: cells(rows, s) for s, _, _ in bars}
    drawn = [(s, label, colour) for s, label, colour in bars if data[s]]
    if not drawn:
        return None, []
    paper_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, height))
    width = 0.8 / len(drawn)
    top = 1.0
    for i, mesh in enumerate(meshes):
        for k, (s, _, colour) in enumerate(drawn):
            v = data[s].get(mesh)
            if v is None:
                continue
            x = i - 0.4 + width * (k + 0.5)
            ax.bar(x, v, width, color=colour, edgecolor=EDGE, linewidth=0.4, zorder=2)
            top = max(top, v)
    line = ax.axhline(1.0, color="black", linestyle=(0, (4, 2)), linewidth=0.8, zorder=3,
                      label=REFERENCE)
    ax.set_xticks(range(len(meshes)))
    # The names lean only as far as they must to clear their neighbours
    # (the user, 10-24: no needless whitespace under the axes); the pitch
    # is the axes' width less what the y label takes, over the meshes.
    lean_labels(ax, meshes, (TEXT_WIDTH - 0.6) / len(meshes))
    ax.set_xlim(-0.6, len(meshes) - 0.4)
    ax.set_ylim(0, top * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The reference line leads the legend, then a patch per schedule, as
    # apps/pbrt's figures have it: one row up to five entries, else two rows.
    handles = [Line2D([], [], color=line.get_color(), linestyle=line.get_linestyle(),
                      linewidth=line.get_linewidth(), label=REFERENCE)]
    handles += [Patch(facecolor=colour, edgecolor=EDGE, linewidth=0.4, label=label)
                for _, label, colour in drawn]
    ncol = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    ax.legend(handles=handles, ncol=ncol, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0, columnspacing=1.2)
    return fig, [s for s, _, _ in drawn]


def save(fig, out_dir, name, formats):
    """The page exactly the figure's size (no cropping to the content, which
    would leave LaTeX to rescale it): the layout fits the labels and the
    legend inside it. Returns the path of the first format written."""
    fig.tight_layout(pad=0.3)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=200)
        print(path)
        paths.append(path)
    plt.close(fig)
    return paths[0]


def tex_number(n):
    return f"{n:,}".replace(",", "\\,")


def mesh_table(rows, meshes, out_dir, exclude):
    """The mesh facts in one table rather than on every plot (the user,
    10-24): wosx-meshes.txt to read and wosx-meshes.tex to \\input, a
    booktabs tabular. Triangles, whether the surface is closed, the sample
    points on the slice and their walks, and WoSX's mean walk length, read
    off the first row of each mesh (the same on every schedule)."""
    first = {}
    for r in rows:
        first.setdefault(r["mesh"], r)
    columns = [("mesh", lambda r: r["mesh"]), ("triangles", lambda r: int(r["triangles"])),
               ("closed", lambda r: "yes" if r["closed"] == "1" else "no"),
               ("points", lambda r: int(r["points"])), ("walks", lambda r: int(r["walks"])),
               ("walk length", lambda r: f"{float(r['wosx_walk_length']):.1f}")]
    table = [[f(first[m]) for _, f in columns] for m in meshes]

    def text(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    txt = os.path.join(out_dir, "wosx-meshes.txt")
    with open(txt, "w") as f:
        widths = [max(len(h), *(len(text(row[i])) for row in table))
                  for i, (h, _) in enumerate(columns)]
        f.write("  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                          for i, ((h, _), w) in enumerate(zip(columns, widths))) + "\n")
        for row in table:
            f.write("  ".join(text(v).ljust(w) if i == 0 else text(v).rjust(w)
                              for i, (v, w) in enumerate(zip(row, widths))) + "\n")
        f.write("\nwalk length: WoSX's mean number of steps of a walk that ended, 64 walks a "
                "point, epsilon 0.001\n")
        if exclude:
            f.write(f"left out of the figures: {', '.join(sorted(exclude))}\n")
    print(txt)
    tex = os.path.join(out_dir, "wosx-meshes.tex")
    with open(tex, "w") as f:
        f.write("% apps/wosx/plot.py: the meshes of the wosx figures, in order of triangle\n"
                "% count; walk length is WoSX's mean number of steps of a walk that ended.\n"
                "% \\usepackage{booktabs}.\n")
        f.write("\\begin{tabular}{l" + "r" * (len(columns) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(h for h, _ in columns) + " \\\\\n\\midrule\n")
        for row in table:
            f.write(" & ".join(tex_number(v) if isinstance(v, int) else str(v) for v in row)
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(tex)


def cpu_notes(base, exclude):
    notes = [
        "single-threaded: one thread against one thread, both sides' traversal over the same "
        "tree; parallel: WoSX's tbb::parallel_for over the points against the parfor bound to "
        "the same pinned threads; persistent: sixteen workers claiming points from a counter, "
        "against WoSX's threaded solve (it has no persistent CPU solver)",
    ]
    if base == "fcpwb":
        notes += [
            "single-threaded under WoSX on the small trees (comb, pavilion, head, crown) and "
            "level or ahead from a million triangles up: the generated two-child visit executes "
            "about 130 instructions to FCPW's 70 (the children's boxes transposed into two "
            "lanes, the entered lanes picked through the stack, the keys ordered as integers), "
            "which the node loads' latency hides on the big trees and not on one that sits in "
            "L1; the traversal itself visits 17% fewer nodes than FCPW's on the comb (a strict "
            "improvement where FCPW's test is <=). apps/wosx/PLAN.md, 'The binary CPU walk's "
            "gap, read from the machine'",
            "the threaded and persistent forms are ahead on every mesh but the comb",
        ]
    else:
        notes += [
            "FCPW's vectorized build (enoki, FCPW_USE_ENOKI) on both sides; the tree WoSX's "
            "boundary handler builds when asked to vectorize",
        ]
    if exclude:
        notes.append(f"left out: {', '.join(sorted(exclude))} (nearly every walk escapes on its "
                     "first step; the bar measures an escaping walk, not the traversal)")
    return notes


def draw_cpu(rows, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    for base, name, what in CPU_TREES:
        bars = [(base + suffix, label, colour) for suffix, label, colour in CPU_BARS]
        fig, drawn = figure(rows, meshes, bars, "Speedup", height=1.7)
        if fig is None:
            print(f"{name}: no rows for {base}'s schedules; skipped", file=sys.stderr)
            continue
        path = save(fig, out_dir, name, formats)
        data = {s: cells(rows, s) for s in drawn}
        label = {base + suffix: lab for suffix, lab, _ in CPU_BARS}
        sections = [("every mesh, over WoSX", [(label[s], points(data[s], meshes)) for s in drawn])]
        notes = cpu_notes(base, exclude)
        threads, persistent = base + "-threads", base + "-persistent"
        if threads in data and persistent in data:
            notes.append("the persistent workers over TBB's partition, the same reference: "
                         + describe(ratios(points(data[persistent], meshes),
                                           points(data[threads], meshes))))
        print(write_summary(path, f"Speedup over WoSX on the CPU, {what}, {len(meshes)} meshes",
                            sections, notes))


def draw_gpu(rows, schedules, out_dir, formats, exclude):
    meshes, tris = meshes_by_triangles(rows, exclude)
    known = {s: (label, colour) for s, label, colour in GPU_BARS}
    bars = [(s, *known.get(s, (s, YLGNBU[8]))) for s in schedules]
    # A little taller: seven bars a mesh and a legend of two rows above them.
    fig, drawn = figure(rows, meshes, bars, "Speedup", height=2.0)
    if fig is None:
        print(f"no rows for {schedules}; nothing drawn", file=sys.stderr)
        return
    path = save(fig, out_dir, "wosx-gpu-speedup", formats)
    data = {s: cells(rows, s) for s in drawn}
    label = {s: known.get(s, (s, None))[0] for s in drawn}
    by_label = {label[s]: points(data[s], meshes) for s in drawn}
    sections = [("every mesh, over WoSX's persistent kernel", [(label[s], by_label[label[s]]) for s in drawn])]
    notes = [
        "persistent is the matching schedule (131072 threads claiming a walk at a time from a "
        "counter, a warp's claims neighbouring points); the others change one thing of it each",
    ]
    for over, base, thing in GPU_STEPS:
        if over in by_label and base in by_label:
            notes.append(f"{thing} ({over} over {base}): "
                         + describe(ratios(by_label[over], by_label[base])))
    best = {}
    for m in meshes:
        here = [(lab, dict(pts).get(m)) for lab, pts in by_label.items() if m in dict(pts)]
        if here:
            best[m] = max(here, key=lambda p: p[1])[0]
    counts = {}
    for m, lab in best.items():
        counts.setdefault(lab, []).append(m)
    for lab, ms in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        notes.append(f"{lab} is the best of the seven on {len(ms)} of {len(meshes)} meshes"
                     + (f" ({', '.join(ms)})" if len(ms) <= 4 else ""))
    notes += [
        "crown and beast have fewer walks (29440, 86016) than the kernel has threads, so a "
        "thread claims one walk and the card runs nearly empty (15-25% achieved occupancy "
        "against 32-49% on a full mesh); the kernel's time is the longest walks' latency on "
        "both sides, and every schedule sits at 1; with 1024 walks a point they are 1.16x and "
        "1.24x. apps/wosx/PLAN.md, 'The crown and the beast'",
        "on every mesh both kernels keep about three of a warp's 32 lanes active per "
        "instruction (ncu, head and dragon: WoSX 3.1 and 2.4, ours 2.8 and 2.1): a warp holds "
        "its 32 walks until the longest ends, and the lanes' traversals diverge within a step; "
        "our lead is in the registers (64 against 82, 49% occupancy against 32%), not the "
        "lanes. apps/wosx/PLAN.md, 'The lanes of a warp'",
        "persistent-point: a point a claim leaves 460 to 16348 threads busy of 131072, an empty "
        "card",
    ]
    if exclude:
        notes.append(f"left out: {', '.join(sorted(exclude))} (nearly every walk escapes on its "
                     "first step)")
    print(write_summary(path, f"Speedup over WoSX's GPU solver, {len(meshes)} meshes, seven "
                              "schedules of the one program", sections, notes))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=".", help="directory for the figures")
    p.add_argument("--gpu", action="store_true",
                   help="draw the GPU figure: the GPU schedules over WoSX's GPU solver")
    p.add_argument("--gpu-schedules", default=",".join(s for s, _, _ in GPU_BARS),
                   help="the GPU schedules, in the order of their bars (default: the matching "
                        "schedule and its variants, as benchmark.sh --gpu measures them)")
    p.add_argument("--format", default="pdf,png", help="comma-separated formats (default pdf,png)")
    p.add_argument("--exclude", default="landscape",
                   help="comma-separated meshes left out (default landscape, whose walks escape "
                        "on their first step)")
    a = p.parse_args()
    exclude = {m for m in a.exclude.split(",") if m}
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.csv)
    if a.gpu:
        schedules = [s for s in a.gpu_schedules.split(",") if s]
        draw_gpu(rows, schedules, a.out, a.format.split(","), exclude)
    else:
        schedules = [base + suffix for base, _, _ in CPU_TREES for suffix, _, _ in CPU_BARS]
        draw_cpu(rows, a.out, a.format.split(","), exclude)
    meshes, _ = meshes_by_triangles(rows, exclude)
    mesh_table(rows, meshes, a.out, exclude)
    geomeans(rows, schedules, exclude)


if __name__ == "__main__":
    main()
