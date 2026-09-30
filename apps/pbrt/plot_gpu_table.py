#!/usr/bin/env python3
"""Plots of gpu_compare.sh's tables: the speedup of every schedule over
`pbrt --gpu` and the kernel time of every side, one group of bars per scene,
one row of panels per table given.

    python3 apps/pbrt/plot_gpu_table.py OUT.png TABLE.tsv [TABLE.tsv ...]

Each table is a `table.tsv` gpu_compare.sh wrote (one sample count each). A
scene whose row lacks a number -- a schedule that failed on it -- gets no bar.
A bar whose image did not match pbrt's is hatched, so that a speedup on a
wrong picture is not read as a speedup.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_table(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])
    sides = []
    for r in rows:
        if r["side"] not in sides:
            sides.append(r["side"])
    spp = rows[0]["spp"] if rows else "?"
    return spp, scenes, sides, rows


def number(text):
    try:
        return float(text.rstrip("x"))
    except (ValueError, AttributeError):
        return None


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    out = argv[1]
    tables = [read_table(p) for p in argv[2:]]
    fig, axes = plt.subplots(len(tables), 2, figsize=(14, 4.2 * len(tables)),
                             squeeze=False)
    for row, (spp, scenes, sides, rows) in enumerate(tables):
        ours = [s for s in sides if not s.startswith("pbrt")]
        width = 0.8 / max(1, len(ours))
        ax = axes[row][0]
        for k, side in enumerate(ours):
            for i, scene in enumerate(scenes):
                cell = next((r for r in rows if r["scene"] == scene and r["side"] == side), None)
                v = number(cell["speedup"]) if cell else None
                if v is None:
                    continue
                ok = cell["verdict"].startswith("ok")
                ax.bar(i + (k - (len(ours) - 1) / 2) * width, v, width,
                       color=f"C{k}", label=side if i == 0 else None,
                       hatch=None if ok else "//", edgecolor="black" if not ok else None)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, rotation=20, ha="right")
        ax.set_ylabel("speedup over pbrt --gpu (wall)")
        ax.set_title(f"{spp} spp: wall-time speedup (above 1 is faster than pbrt)")
        ax.legend(fontsize=8)

        ax = axes[row][1]
        width = 0.8 / max(1, len(sides))
        for k, side in enumerate(sides):
            for i, scene in enumerate(scenes):
                cell = next((r for r in rows if r["scene"] == scene and r["side"] == side), None)
                v = number(cell["kernel_ms"]) if cell else None
                if v is None:
                    continue
                ok = side.startswith("pbrt") or cell["verdict"].startswith("ok")
                ax.bar(i + (k - (len(sides) - 1) / 2) * width, v, width,
                       color="gray" if side.startswith("pbrt") else f"C{k - 1}",
                       label=side if i == 0 else None,
                       hatch=None if ok else "//", edgecolor="black" if not ok else None)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, rotation=20, ha="right")
        ax.set_ylabel("kernel time (ms), log scale")
        ax.set_yscale("log")
        ax.set_title(f"{spp} spp: GPU kernel time (lower is better)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
