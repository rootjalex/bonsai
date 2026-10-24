#!/usr/bin/env python3
"""The verdicts of a gpu_compare.sh table re-derived from the images it kept,
when the comparison's rules change and the renders need not be redone.

    python3 apps/pbrt/recompare.py OUT_DIR

For every row of OUT_DIR/table.tsv whose radiance images are still beside
it -- pbrt's `<cell>.pbrt-radiance.pfm` and ours `<cell>.<side>-radiance.pfm`
-- compare_gbuffer.py is run again exactly as gpu_compare.sh ran it
(--radiance-only, with `--pbrt-spread <cell>.pbrt-cpu-radiance.pfm` when the
run rendered pbrt's CPU integrator for that cell), the `.compare` file is
rewritten, and the row's `agree` and `verdict` columns are replaced from
the new output; the times, speedups and every other column stay as
measured. A row whose images are gone keeps its verdict and is listed. The
table is rewritten in place, with the old one kept as table.tsv.before-
recompare once (a second run keeps the first backup). Written for the rule
of 10-24 -- a pixel of ours that is not a number is a lit-pixel
disagreement, not a failure of its own -- so that the heatmap of
compare-out-heatmap7 could be read under it without re-rendering thirty
scenes at four counts. Nothing typed: every verdict is compare_gbuffer.py's.
"""
import csv
import os
import re
import shutil
import subprocess
import sys

PREFIX = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = argv[1]
    table = os.path.join(out, "table.tsv")
    with open(table, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
        fields = list(rows[0].keys()) if rows else None
    if not rows:
        print(f"{table}: no rows", file=sys.stderr)
        return 1
    backup = table + ".before-recompare"
    if not os.path.exists(backup):
        shutil.copy2(table, backup)
    redone, kept, changed = 0, [], []
    for r in rows:
        if r["side"].startswith("pbrt"):
            continue
        # The cell's file stem is the image path less its `.<side>.png`.
        stem = re.sub(r"\.[^./]+\.png$", "", r["image"])
        ours = f"{stem}.{r['side']}-radiance.pfm"
        ref = f"{stem}.pbrt-radiance.pfm"
        if not (os.path.isfile(ours) and os.path.isfile(ref)):
            kept.append(f"{r['scene']} {r['spp']} {r['side']}")
            continue
        cmd = [sys.executable, os.path.join(PREFIX, "compare_gbuffer.py"),
               "--radiance-only", "--radiance", ref, ours]
        spread = f"{stem}.pbrt-cpu-radiance.pfm"
        if os.path.isfile(spread):
            cmd += ["--pbrt-spread", spread]
        text = subprocess.run(cmd, capture_output=True, text=True).stdout
        with open(f"{stem}.{r['side']}.compare", "w") as f:
            f.write(text)
        agree = re.search(r"agree to 1e-03 relative \(([0-9.]+%)\)", text)
        verdict = next((line[:200] for line in text.splitlines()
                        if re.match(r"^ok|^FAILED|mismatch|differ", line)), "?")
        old = r["verdict"]
        r["agree"] = agree.group(1) if agree else r["agree"]
        r["verdict"] = verdict
        redone += 1
        if old != verdict:
            changed.append(f"{r['scene']} {r['spp']} {r['side']}: {old} -> {verdict}")
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{redone} rows re-compared, {len(changed)} verdicts changed, "
          f"{len(kept)} rows kept (no images); the old table is {backup}")
    for line in changed:
        print("  " + line)
    for line in kept:
        print("  kept: " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
