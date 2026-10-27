#!/usr/bin/env python3
"""Two runs of gpu_compare.sh (or heatmap.sh) cell by cell: what changed.

    compare_runs.py OLD_TABLE NEW_TABLE [--images] [--threshold 0.03]

Each table is a run's table.tsv (scene, spp, side, wall_s, kernel_ms,
speedup, agree, verdict, note, image, path). For every cell the new run has
-- a scene at a sample count on one side -- the old run's row is looked up
by (scene, spp, side) and the two compared:

- the kernel time (kernel_ms) and the wall time, as new/old ratios;
- the verdict: a cell that matched pbrt before and does not now is a
  regression whatever its time, and is listed first;
- with --images, the rendered image itself: the new .pfm against the old one
  byte for byte (the .pfm beside the table's .png), since a compiler change
  that is not meant to change what is computed should leave every image
  identical, and a differing image is the first thing to look at.

The summary: how many cells are slower than the threshold (3% by default),
faster than it, within it; the geometric mean of the kernel-time ratio per
side; the cells missing on either side. Rows for `pbrt --gpu` are compared
too (its time drifts with the machine; a drift there says the night was not
quiet). Exit status 1 when any verdict regressed or any image differs, so a
wrapper can tell.
"""
import argparse
import csv
import math
import os
import sys


def read_table(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["scene"], row["spp"], row["side"])
            rows[key] = row
    return rows


def number(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def pfm_of(row, table_path):
    image = row.get("image", "")
    if not image or image == "-":
        return None
    stem = image[:-4] if image.endswith(".png") else image
    pfm = stem + ".pfm"
    if not os.path.isabs(pfm):
        pfm = os.path.join(os.path.dirname(os.path.abspath(table_path)), pfm)
    return pfm


def files_identical(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return None
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x = fa.read(1 << 20)
            y = fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--images", action="store_true",
                    help="compare the rendered .pfm images byte for byte")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="the kernel-time ratio beyond which a cell is slower or faster")
    args = ap.parse_args()

    old = read_table(args.old)
    new = read_table(args.new)

    verdict_regressions = []
    image_differences = []
    slower, faster, within = [], [], []
    ratios = {}  # side -> [ratios]
    lines = []
    for key in sorted(new, key=lambda k: (k[2], k[0], int(k[1]) if k[1].isdigit() else 0)):
        scene, spp, side = key
        n = new[key]
        o = old.get(key)
        if o is None:
            lines.append(f"  {side:22s} {scene:28s} {spp:>4s}  new cell, no old row")
            continue
        ok_old = o["verdict"].startswith("ok")
        ok_new = n["verdict"].startswith("ok")
        kn, ko = number(n["kernel_ms"]), number(o["kernel_ms"])
        wn, wo = number(n["wall_s"]), number(o["wall_s"])
        ratio = kn / ko if kn and ko else None
        wratio = wn / wo if wn and wo else None
        if ratio is not None:
            ratios.setdefault(side, []).append(ratio)
            if ratio > 1 + args.threshold:
                slower.append((ratio, key))
            elif ratio < 1 - args.threshold:
                faster.append((ratio, key))
            else:
                within.append((ratio, key))
        flag = ""
        if ok_old and not ok_new:
            flag = "  VERDICT REGRESSED"
            verdict_regressions.append(key)
        elif not ok_old and ok_new:
            flag = "  verdict now ok"
        image_note = ""
        if args.images and side != "pbrt --gpu":
            a, b = pfm_of(o, args.old), pfm_of(n, args.new)
            same = files_identical(a, b) if a and b else None
            if same is None:
                image_note = "  image: not compared"
            elif same:
                image_note = "  image identical"
            else:
                image_note = "  IMAGE DIFFERS"
                image_differences.append(key)
        k = f"{ko:9.2f} -> {kn:9.2f} ms ({ratio:.3f}x)" if ratio is not None else "kernel: -"
        w = f"wall {wratio:.3f}x" if wratio is not None else "wall -"
        lines.append(f"  {side:22s} {scene:28s} {spp:>4s}  {k}  {w}{flag}{image_note}")
        if n["verdict"] != o["verdict"] and not (ok_old and ok_new):
            lines.append(f"      verdict: {o['verdict']}  ->  {n['verdict']}")

    missing = sorted(set(old) - set(new))

    print(f"old: {args.old}\nnew: {args.new}\n")
    print("cells (kernel_ms old -> new, wall ratio):")
    print("\n".join(lines))
    print()
    if verdict_regressions:
        print("VERDICT REGRESSIONS:")
        for scene, spp, side in verdict_regressions:
            print(f"  {side} {scene} {spp} spp")
    if image_differences:
        print("IMAGES THAT DIFFER:")
        for scene, spp, side in image_differences:
            print(f"  {side} {scene} {spp} spp")
    if missing:
        print("cells of the old run missing from the new:")
        for scene, spp, side in missing:
            print(f"  {side} {scene} {spp} spp")
    print(f"\nkernel time, new over old, beyond {args.threshold:.0%}: "
          f"{len(slower)} slower, {len(faster)} faster, {len(within)} within")
    for r, (scene, spp, side) in sorted(slower, reverse=True)[:12]:
        print(f"  slower {r:.3f}x  {side} {scene} {spp} spp")
    for r, (scene, spp, side) in sorted(faster)[:12]:
        print(f"  faster {r:.3f}x  {side} {scene} {spp} spp")
    for side, rs in sorted(ratios.items()):
        g = math.exp(sum(math.log(r) for r in rs) / len(rs))
        print(f"geomean kernel ratio {side}: {g:.4f}x over {len(rs)} cells")
    sys.exit(1 if (verdict_regressions or image_differences) else 0)


if __name__ == "__main__":
    main()
