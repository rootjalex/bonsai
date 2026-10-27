#!/usr/bin/env python3
"""Two kernels of one Nsight Compute report side by side: the reference's
and ours, from the raw page exported as CSV (`ncu --import x.ncu-rep --csv
--page raw > x.raw.csv`), the last profiled launch of each kernel name (the
warm one). Prints the metrics that say where a traversal kernel's time
goes -- duration, throughputs, occupancy, issue rate, instructions and
active threads per instruction, the load counts and their cache hit rates,
local-memory traffic (the stack), and the warp stall reasons -- and writes
`<stem>-tidy.csv` beside the raw page, every metric of every profiled
launch as (launch, kernel, metric, value), for plots.

    python3 ncu_kernels.py results/ncu/landscape-firsthit.raw.csv
"""
import csv
import os
import re
import sys


# (label, regular expression over the raw metric names, how to print)
WANTED = [
    ("duration (the raw page's unit, usually us)", r"^gpu__time_duration\.sum$", "f"),
    ("SM throughput % of peak", r"^sm__throughput\.avg\.pct_of_peak_sustained_elapsed$", "f"),
    ("memory throughput % of peak", r"^gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed$", "f"),
    ("L1/tex throughput %", r"^l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed$", "f"),
    ("L2 throughput %", r"^lts__throughput\.avg\.pct_of_peak_sustained_elapsed$", "f"),
    ("DRAM throughput %", r"^dram__throughput\.avg\.pct_of_peak_sustained_elapsed$", "f"),
    ("DRAM bytes read", r"^dram__bytes_read\.sum$", "i"),
    ("registers per thread", r"^launch__registers_per_thread$", "i"),
    ("occupancy limit, registers (warps)", r"^launch__occupancy_limit_registers$", "i"),
    ("achieved occupancy % (warps active)", r"^sm__warps_active\.avg\.pct_of_peak_sustained_active$", "f"),
    ("eligible warps per cycle", r"^smsp__warps_eligible\.avg\.per_cycle_active$", "f"),
    ("issue slot utilization %", r"^smsp__issue_active\.avg\.pct_of_peak_sustained_active$", "f"),
    ("cycles between a warp's issues", r"^smsp__average_warp_latency_per_inst_issued\.ratio$", "f"),
    ("warp instructions executed", r"^smsp__inst_executed\.sum$", "i"),
    ("threads active per instruction (of 32)", r"^smsp__thread_inst_executed_per_inst_executed\.ratio$", "f"),
    ("branch targets uniform %", r"^smsp__sass_average_branch_targets_threads_uniform\.pct$", "f"),
    ("global load instructions", r"^smsp__sass_inst_executed_op_global_ld\.sum$", "i"),
    ("global load requests", r"^l1tex__t_requests_pipe_lsu_mem_global_op_ld\.sum$", "i"),
    ("global load sectors", r"^l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum$", "i"),
    ("L1/tex sector hit rate %", r"^l1tex__t_sector_hit_rate\.pct$", "f"),
    ("L2 sector hit rate %", r"^lts__t_sector_hit_rate\.pct$", "f"),
    ("L2 sectors read from L1/tex", r"^lts__t_sectors_srcunit_tex_op_read\.sum$", "i"),
    ("local load instructions (stack)", r"^smsp__sass_inst_executed_op_local_ld\.sum$", "i"),
    ("local store instructions (stack)", r"^smsp__sass_inst_executed_op_local_st\.sum$", "i"),
    ("local load sectors", r"^l1tex__t_sectors_pipe_lsu_mem_local_op_ld\.sum$", "i"),
    ("local store sectors", r"^l1tex__t_sectors_pipe_lsu_mem_local_op_st\.sum$", "i"),
    ("local load sectors hit in L1", r"^l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit\.sum$", "i"),
]
STALL = re.compile(r"^smsp__average_warps?_(?:latency_)?issue_stalled_(\w+)_per_issue_active\.ratio$")


def number(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def fmt(v, how):
    if v is None:
        return "-"
    if how == "us":
        return f"{v / 1000:.1f}"
    if how == "i":
        return f"{v:,.0f}"
    return f"{v:.2f}"


def main(path):
    rows = list(csv.reader(open(path)))
    header, body = rows[0], [r for r in rows[1:] if r and r[0].strip()]
    col = {name: i for i, name in enumerate(header)}
    kname = col["Kernel Name"]
    # The last launch of each kernel name, in first-seen order.
    last = {}
    order = []
    for r in body:
        k = r[kname]
        if k not in last:
            order.append(k)
        last[k] = r
    launches = [last[k] for k in order]
    short = [k.split("(")[0][-24:] for k in order]

    def get(row, pattern):
        for name, i in col.items():
            if re.search(pattern, name):
                return number(row[i])
        return None

    print(os.path.basename(path), "-- the last profiled launch of each kernel")
    print(f"{'metric':42}" + "".join(f" {s:>24}" for s in short) + ("   ratio" if len(launches) == 2 else ""))
    for label, pattern, how in WANTED:
        values = [get(r, pattern) for r in launches]
        line = f"{label:42}" + "".join(f" {fmt(v, how):>24}" for v in values)
        if len(values) == 2 and values[0] and values[1]:
            line += f" {values[0] / values[1]:7.2f}"
        print(line)
    # Stall reasons: cycles a warp spends stalled on each reason per issued
    # instruction (the Warp State section's breakdown), largest first.
    stalls = []
    for name, i in col.items():
        m = STALL.match(name)
        if m:
            stalls.append((m.group(1), [number(r[i]) for r in launches]))
    if stalls:
        stalls.sort(key=lambda s: -max(v or 0 for v in s[1]))
        print(f"\n{'stall reason (cycles per issue)':42}" + "".join(f" {s:>24}" for s in short))
        for reason, values in stalls:
            if max(v or 0 for v in values) < 0.05:
                continue
            print(f"{reason:42}" + "".join(f" {fmt(v, 'f'):>24}" for v in values))

    stem = os.path.splitext(path)[0]
    if stem.endswith(".raw"):
        stem = stem[:-4]
    with open(stem + "-tidy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["launch", "kernel", "metric", "value"])
        for n, r in enumerate(body):
            for name, i in col.items():
                v = number(r[i])
                if v is not None and name not in ("ID", "Process ID", "Context", "Stream", "CC"):
                    w.writerow([n, r[kname].split("(")[0], name, r[i].replace(",", "")])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
