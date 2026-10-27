#!/usr/bin/env python3
"""Per-kernel reading of an Nsight Systems trace of apps/rtq's fcpwb-gpu
comparison (see README.md): kernels in launch order, runs of the same
kernel collapsed, each with its count, mean and least GPU duration,
registers per thread, local memory per thread, grid and block; then, where
GPU metrics were sampled, the mean of every sampled metric inside each
kernel's spans.

    python3 kernels.py landscape.sqlite

Beside the printed reading it writes two CSVs next to the sqlite, for
plots later: `<stem>-launches.csv`, one row per kernel launch in order
(index, kernel, start_ms from the first launch, duration_ms, registers,
local bytes, grid, block), and `<stem>-metrics.csv`, one row per kernel
name and sampled metric (kernel, metric, mean, samples).
"""
import bisect
import csv
import os
import sqlite3
import sys
from collections import defaultdict


def table_exists(db, name):
    return db.execute("select count(*) from sqlite_master where type='table' and name=?",
                      (name,)).fetchone()[0] > 0


def columns(db, table):
    return [r[1] for r in db.execute(f"pragma table_info({table})")]


def main(path):
    db = sqlite3.connect(path)
    cols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    want = ["start", "end", "demangledName", "registersPerThread", "localMemoryPerThread",
            "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "staticSharedMemory"]
    have = [c for c in want if c in cols]
    rows = db.execute(f"select {', '.join(have)} from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    names = dict(db.execute("select id, value from StringIds").fetchall())
    kernels = []
    for r in rows:
        d = dict(zip(have, r))
        d["name"] = names.get(d.get("demangledName"), str(d.get("demangledName")))
        kernels.append(d)
    if not kernels:
        print("no kernels in the trace")
        return

    def short(name):
        # The kernel's own name, without its parameter list or template noise.
        n = name.split("(")[0]
        return n[-48:]

    stem = os.path.splitext(path)[0]
    t0 = kernels[0]["start"]
    with open(stem + "-launches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "kernel", "start_ms", "duration_ms", "registers", "local_bytes",
                    "grid", "block"])
        for i, k in enumerate(kernels):
            w.writerow([i, short(k["name"]), f"{(k['start'] - t0) / 1e6:.3f}",
                        f"{(k['end'] - k['start']) / 1e6:.4f}", k.get("registersPerThread", ""),
                        k.get("localMemoryPerThread", ""),
                        k.get("gridX", 0) * k.get("gridY", 1) * k.get("gridZ", 1),
                        k.get("blockX", 0) * k.get("blockY", 1) * k.get("blockZ", 1)])

    # Runs of the same kernel in launch order.
    print(f"{len(kernels)} kernel launches, GPU time {sum(k['end'] - k['start'] for k in kernels) / 1e6:.1f} ms\n")
    # `period` is the mean start-to-start gap between a run's launches: what
    # the driver's clock sees per repeat (dispatch, kernel, wait), against
    # the kernel's own duration -- the difference is the launch and
    # synchronization cost each side's path pays.
    print(f"{'#':>3} {'kernel':48} {'n':>4} {'mean ms':>9} {'least ms':>9} {'period ms':>10} "
          f"{'regs':>5} {'local B':>8} {'grid':>8} {'block':>6}")
    runs = []
    for k in kernels:
        if runs and runs[-1][0]["name"] == k["name"]:
            runs[-1].append(k)
        else:
            runs.append([k])
    for i, run in enumerate(runs):
        durs = [(k["end"] - k["start"]) / 1e6 for k in run]
        k = run[0]
        grid = k.get("gridX", 0) * k.get("gridY", 1) * k.get("gridZ", 1)
        block = k.get("blockX", 0) * k.get("blockY", 1) * k.get("blockZ", 1)
        period = ((run[-1]["start"] - run[0]["start"]) / 1e6 / (len(run) - 1)) if len(run) > 1 else float("nan")
        print(f"{i:>3} {short(k['name']):48} {len(run):>4} {sum(durs) / len(durs):9.3f} {min(durs):9.3f} "
              f"{period:10.3f} {k.get('registersPerThread', '?'):>5} {k.get('localMemoryPerThread', '?'):>8} "
              f"{grid:>8} {block:>6}")

    # GPU metrics inside each kernel's spans, by kernel name.
    if not (table_exists(db, "GPU_METRICS") and table_exists(db, "TARGET_INFO_GPU_METRICS")):
        print("\nno GPU metrics in the trace (sampling needs the driver's profiling permission)")
        return
    info_cols = columns(db, "TARGET_INFO_GPU_METRICS")
    id_col = "metricId" if "metricId" in info_cols else info_cols[0]
    metric_names = {}
    for r in db.execute(f"select {id_col}, metricName from TARGET_INFO_GPU_METRICS"):
        metric_names[r[0]] = r[1]
    samples = defaultdict(list)  # metricId -> [(timestamp, value)]
    mcols = columns(db, "GPU_METRICS")
    key_col = "metricId" if "metricId" in mcols else ("typeId" if "typeId" in mcols else mcols[1])
    for ts, mid, value in db.execute(f"select timestamp, {key_col}, value from GPU_METRICS order by timestamp"):
        samples[mid].append((ts, value))
    by_name = defaultdict(list)
    for k in kernels:
        by_name[short(k["name"])].append((k["start"], k["end"]))
    print("\nGPU metrics, mean over the samples inside each kernel's spans (percent of peak unless the name says otherwise):")
    header = f"{'metric':52}" + "".join(f" {n[:14]:>14}" for n in by_name)
    print(header)
    out = open(stem + "-metrics.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["kernel", "metric", "mean", "samples"])
    for mid, series in samples.items():
        if not series:
            continue
        stamps = [s[0] for s in series]
        metric = metric_names.get(mid, str(mid))
        line = f"{metric[:52]:52}"
        any_value = False
        for name, spans in by_name.items():
            total, count = 0.0, 0
            for s, e in spans:
                lo = bisect.bisect_left(stamps, s)
                hi = bisect.bisect_right(stamps, e)
                for j in range(lo, hi):
                    total += series[j][1]
                    count += 1
            if count:
                any_value = True
                line += f" {total / count:14.1f}"
                w.writerow([name, metric, f"{total / count:.3f}", count])
            else:
                line += f" {'-':>14}"
        if any_value:
            print(line)
    out.close()


if __name__ == "__main__":
    main(sys.argv[1])
