#!/usr/bin/env bash
# One OptiX launch of one of our renderers under Nsight Compute, for the
# profile figure (plot_gpu_profile.py; the user, 10-28: the profile of the
# wavefront's trace kernel against the megakernel's, for the paper).
#
#   bash apps/pbrt/profile_kernels.sh [--out DIR] [--renderers DIR] \
#        [--single-pass-timing] SCHEDULE SCENE SPP LAUNCH
#   bash apps/pbrt/profile_kernels.sh [--renderers DIR] --list N SCHEDULE SCENE SPP
#
# --single-pass-timing re-measures the timing metrics the figure draws
# (duration, warp cycles per issued instruction, the five stall ratios,
# warps active, issue slots busy) in renderer runs of one Nsight pass each,
# and merges them into the .raw.csv over the full set's values (kept as
# .multipass.raw.csv). Kernel replay saves and restores the launch's memory
# between passes, and on ganesha that made the wavefront's trace launch
# read 250 MB from system memory and run 47x slow (10-28); a single pass
# restores nothing. A stall ratio Nsight cannot collect in one pass (the
# load latency, long_scoreboard, on this card) is the remainder of the
# cycles per issued instruction over every other stall reason, each
# measured in a one-pass run of its own; the .raw.csv's
# bonsai__timing_provenance column says so. The SASS and the
# per-instruction counts are counts, not cycles, and stand from the full
# set. --merge-only redoes the merge from the .singlepass-*.csv files
# already beside the report, without the card.
#
# SCHEDULE names a renderer gpu_compare.sh compiled, RENDERERS/SCHEDULE/
# render.out (--renderers, default apps/pbrt/gpu-compare-out-e, the GPU
# figure's run); SCENE is a converted scene as gpu_compare.sh names it,
# <dir>/<name> under apps/pbrt/scenes/converted/ (killeroos/killeroo-gold;
# a bare name is <name>/<name>, as ganesha), the files named by <name>;
# SPP the sample count; LAUNCH the ordinal of the
# `optixLaunch` to profile, counted from 0 over the process's OptiX launches
# in order. Nsight Compute names every OptiX launch `optixLaunch`, whatever
# its raygen, so a launch is chosen by ordinal: `--list N` runs the renderer
# once with the cheapest metrics and prints the first N launches (duration,
# grid, threads, lanes active) to tell them apart -- the wavefront's first
# launch is its first band's closest-hit trace, the flat megakernels' first
# is the per-pixel prologue and their second the paths' raygen.
#
# The profile is the full section set under kernel replay (the renderer
# runs once, the launch is replayed some forty times with its memory saved
# and restored; BONSAI_REPEATS=1 so the frame renders once). Written to
# OUT (--out, default apps/pbrt/profile-out-e, the profile figure's run;
# profile-out/ is 10-28 daytime's under 39c67ab9), four files per launch, named
# SCENE-SPP-SCHEDULE-L<LAUNCH>: .ncu-rep (the report), .txt (its details
# page), .raw.csv (every metric, one row), .source.csv (the SASS of every
# function in the launch with its executed-instruction, thread and stall
# sample counts -- the kernel's size and the lanes active per function
# come from this), and .log (the run's output). The plotter reads the
# .raw.csv and .source.csv.
#
# Nsight Compute on an OptiX process is run only with the user's say-so
# (the rule of 10-20, after a driver crash; given 10-27 for ganesha, 10-28
# for the figure's scenes). About two to three minutes a launch for a
# frame of 300 ms, under a minute for a small one. The benchmark lock is
# the caller's: run this through a wrapper that takes and removes it.
set -euo pipefail
PREFIX=$(cd "$(dirname "$0")" && pwd)
OUT="$PREFIX/profile-out-e"
RENDERERS="$PREFIX/gpu-compare-out-e"
LIST=0
SINGLE_PASS=0
MERGE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --renderers) RENDERERS="$2"; shift 2 ;;
    --list) LIST="$2"; shift 2 ;;
    --single-pass-timing) SINGLE_PASS=1; shift ;;
    --merge-only) SINGLE_PASS=1; MERGE_ONLY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --*) echo "profile_kernels.sh: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
if (( LIST > 0 )); then
  [[ $# -eq 3 ]] || { echo "usage: profile_kernels.sh --list N SCHEDULE SCENE SPP" >&2; exit 2; }
else
  [[ $# -eq 4 ]] || { echo "usage: profile_kernels.sh [--out DIR] [--renderers DIR] SCHEDULE SCENE SPP LAUNCH" >&2; exit 2; }
fi
SCHEDULE=$1; SCENE=$2; SPP=$3; LAUNCH=${4:-}
RENDER="$RENDERERS/$SCHEDULE/render.out"
# The scene as gpu_compare.sh names it, <dir>/<name> under scenes/converted/
# (killeroos/killeroo-gold); a bare name is a scene in a directory of its own
# name (ganesha). The files are named by the scene's basename.
[[ "$SCENE" == */* ]] || SCENE="$SCENE/$SCENE"
SCENE_FILE="$PREFIX/scenes/converted/$SCENE.txt"
SCENE=${SCENE##*/}
[[ -x "$RENDER" ]] || { echo "profile_kernels.sh: no renderer $RENDER (gpu_compare.sh compiles it)" >&2; exit 1; }
[[ -f "$SCENE_FILE" ]] || { echo "profile_kernels.sh: no scene $SCENE_FILE" >&2; exit 1; }
command -v ncu > /dev/null || { echo "profile_kernels.sh: no ncu on PATH" >&2; exit 1; }
mkdir -p "$OUT"
IMAGE="$OUT/.$SCENE-$SPP-$SCHEDULE.pfm"

if (( LIST > 0 )); then
  # The cheapest look at the first N launches: one pass each.
  BONSAI_REPEATS=1 ncu --replay-mode kernel -k optixLaunch --launch-count "$LIST" \
    --metrics gpu__time_duration.sum,launch__grid_size,launch__thread_count,launch__registers_per_thread,smsp__thread_inst_executed_per_inst_executed.ratio \
    --target-processes all --csv --page raw \
    "$RENDER" --spp "$SPP" --no-implicit-copies "$SCENE_FILE" "$IMAGE" 2> /dev/null \
    | python3 -c '
import csv, sys
rows = list(csv.reader(sys.stdin))
start = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
hdr, data = rows[start], rows[start + 2:]
cols = ["gpu__time_duration.sum", "launch__grid_size", "launch__thread_count",
        "launch__registers_per_thread", "smsp__thread_inst_executed_per_inst_executed.ratio"]
units = dict(zip(hdr, rows[start + 1]))
print("launch  " + "  ".join(c.split("__")[-1] + " [" + units.get(c, "") + "]" for c in cols))
for r in data:
    d = dict(zip(hdr, r))
    print(d["ID"].rjust(6) + "  " + "  ".join(d.get(c, "?") for c in cols))
'
  rm -f "$IMAGE"
  exit 0
fi

STEM="$OUT/$SCENE-$SPP-$SCHEDULE-L$LAUNCH"
if (( ! MERGE_ONLY )); then
echo "== profile_kernels.sh: $SCHEDULE on $SCENE at $SPP spp, optixLaunch #$LAUNCH -> $STEM.* ($(date +%T))"
BONSAI_REPEATS=1 ncu --set full --replay-mode kernel -k optixLaunch --launch-skip "$LAUNCH" --launch-count 1 \
  --target-processes all -f -o "$STEM" \
  "$RENDER" --spp "$SPP" --no-implicit-copies "$SCENE_FILE" "$IMAGE" > "$STEM.log" 2>&1 \
  || { echo "profile_kernels.sh: ncu failed; the end of $STEM.log:" >&2; tail -5 "$STEM.log" >&2; exit 1; }
rm -f "$IMAGE"
[[ -f "$STEM.ncu-rep" ]] || { echo "profile_kernels.sh: no report written; the end of $STEM.log:" >&2; tail -5 "$STEM.log" >&2; exit 1; }
ncu --import "$STEM.ncu-rep" --page details > "$STEM.txt" 2>&1
ncu --import "$STEM.ncu-rep" --page raw --csv > "$STEM.raw.csv" 2> /dev/null
ncu --import "$STEM.ncu-rep" --page source --csv > "$STEM.source.csv" 2> /dev/null
fi

if (( SINGLE_PASS )); then
  # The timing metrics again, each group in a renderer run of its own that
  # Nsight completes in ONE pass, so no memory is saved and restored between
  # passes. Under multi-pass kernel replay the wavefront's trace launch on
  # ganesha read 250 MB from system memory and ran 47x slow (10-28; the
  # megakernels and the smaller scenes were untouched; application replay
  # is refused for OptiX launches by this driver). The first group's
  # duration is the launch's true duration; a later group whose duration is
  # off by more than a fifth needed more than one pass and is reported, not
  # merged. The merged columns replace the full set's in $STEM.raw.csv (the
  # full set's own values kept beside it as $STEM.multipass.raw.csv); the
  # SASS and the per-instruction counts of the full set are not affected by
  # the distortion (counts, not cycles) and stand.
  # A stall ratio that Nsight cannot collect in one pass (long_scoreboard on
  # this card) is taken as the remainder: the cycles per issued instruction
  # less every other stall reason, each measured in a one-pass run of its
  # own -- so if one of the drawn ratios fails its check, the other reasons
  # are measured too. --merge-only redoes the merge from the .singlepass-*
  # files already there, without the card.
  [[ -f "$STEM.multipass.raw.csv" ]] || cp "$STEM.raw.csv" "$STEM.multipass.raw.csv"
  cp "$STEM.multipass.raw.csv" "$STEM.raw.csv"
  DRAWN=(no_instruction barrier long_scoreboard lg_throttle mio_throttle)
  OTHERS=(short_scoreboard wait selected not_selected branch_resolving dispatch_stall drain math_pipe_throttle membar misc sleeping tex_throttle imc_miss)
  measure() {  # NAME METRICS: one renderer run, one pass, to $STEM.singlepass-NAME.csv
    (( MERGE_ONLY )) && return 0
    echo "== single-pass $1: $2 ($(date +%T))" >> "$STEM.log"
    BONSAI_REPEATS=1 ncu --replay-mode kernel -k optixLaunch --launch-skip "$LAUNCH" --launch-count 1 --metrics "$2" \
      --target-processes all --csv --page raw \
      "$RENDER" --spp "$SPP" --no-implicit-copies "$SCENE_FILE" "$IMAGE" > "$STEM.singlepass-$1.csv" 2>> "$STEM.log" \
      || echo "profile_kernels.sh: single-pass run $1 failed; see $STEM.log" >&2
  }
  ratio() { echo "smsp__average_warps_issue_stalled_$1_per_issue_active.ratio"; }
  measure timing "gpu__time_duration.sum,smsp__average_warp_latency_per_inst_issued.ratio,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active"
  for s in "${DRAWN[@]}"; do measure "$s" "gpu__time_duration.sum,$(ratio "$s")"; done
  # The merge, in Python; with "check" it only names the drawn ratios whose
  # run took more than one pass (their duration off the timing run's by more
  # than a fifth).
  merge() {
  python3 - "$STEM" "$1" "${DRAWN[*]}" "${OTHERS[*]}" <<'PY'
import csv, os, sys
stem, mode, drawn, others = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
RATIO = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
TOTAL = "smsp__average_warp_latency_per_inst_issued.ratio"
def table(path):
    if not os.path.exists(path): return None
    r = list(csv.reader(open(path)))
    s = next((k for k, row in enumerate(r) if row and row[0] == "ID"), None)
    return (r[s], r[s + 1], r[s + 2]) if s is not None and len(r) > s + 2 else None
def num(v):
    try: return float(v.replace(",", ""))
    except Exception: return None
def run(name):
    """(metrics {k: (value, unit)}, duration ns, one pass?) of a single-pass run."""
    t = table(f"{stem}.singlepass-{name}.csv")
    if t is None: return None
    h, u, d = t
    g = {k: (v, uu) for k, uu, v in zip(h, u, d)}
    return g, num(g.get("gpu__time_duration.sum", ("", ""))[0])
timing = run("timing")
if timing is None:
    print("  single-pass: no timing run, nothing merged"); sys.exit(0)
truth = timing[1]
def true_run(name):
    r = run(name)
    if r is None or r[1] is None or truth is None or abs(r[1] - truth) > 0.2 * truth: return None
    return r[0]
failed = [s for s in drawn if true_run(s) is None]
if mode == "check":
    print(" ".join(failed)); sys.exit(0)
rows = list(csv.reader(open(stem + ".raw.csv")))
hdr, units, data = rows[0], rows[1], rows[2]
col = {k: i for i, k in enumerate(hdr)}
def put(k, v, u):
    if k in col: data[col[k]] = v; units[col[k]] = u
    else: hdr.append(k); units.append(u); data.append(v); col[k] = len(hdr) - 1
merged, notes = [], []
for k, (v, u) in timing[0].items():
    if k.startswith(("gpu__", "smsp__", "sm__")) and k != "smsp__maximum_warps_avg_per_active_cycle":
        put(k, v, u); merged.append(k)
print(f"  single-pass timing: duration {truth:,.0f} ns; merged " + ", ".join(merged))
known = {}
for s in drawn + others:
    g = true_run(s)
    if g is None:
        if s in drawn: print(f"  single-pass {s}: more than one pass or missing, not merged")
        continue
    v = g.get(RATIO.format(s))
    if v is None or num(v[0]) is None:
        if s in drawn: print(f"  single-pass {s}: no value")
        continue
    known[s] = num(v[0])
    if s in drawn: put(RATIO.format(s), v[0], v[1]); print(f"  single-pass {s}: {v[0]} {v[1]}")
total = num(timing[0].get(TOTAL, ("", ""))[0])
if len(failed) == 1 and total is not None:
    f = failed[0]
    missing = [s for s in drawn + others if s != f and s not in known and s != "imc_miss"]
    if missing:
        print(f"  {f}: no remainder, the other reasons not all measured (missing {', '.join(missing)})")
    else:
        rem = total - sum(known.values())
        put(RATIO.format(f), f"{rem:.2f}", "inst")
        notes.append(f"{f} by remainder ({total:.2f} less the {len(known)} other reasons, each in one pass)")
        print(f"  single-pass {f}: {rem:.2f} by remainder of {total:.2f} over {len(known)} measured reasons")
elif len(failed) > 1:
    print(f"  {', '.join(failed)}: more than one drawn ratio failed, no remainder possible; their full-set values stand (distorted if the launch was)")
put("bonsai__timing_provenance", "single-pass runs" + ("; " + "; ".join(notes) if notes else ""), "")
with open(stem + ".raw.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerow(units); w.writerow(data)
    for row in rows[3:]: w.writerow(row)
print(f"  {stem}.raw.csv: timing columns from the single-pass runs, the rest from the full set ({stem}.multipass.raw.csv)")
PY
  }
  FAILED=$(merge check)
  if [[ -n "$FAILED" ]]; then
    echo "== single-pass: $FAILED took more than one pass; measuring the other stall reasons for the remainder ($(date +%T))" >> "$STEM.log"
    for s in "${OTHERS[@]}"; do measure "$s" "gpu__time_duration.sum,$(ratio "$s")"; done
  fi
  merge merge | tee -a "$STEM.log"
fi
rm -f "$IMAGE"
echo "== done ($(date +%T)): $(grep -m1 -E '^\s+Duration' "$STEM.txt" | tr -s ' ') ; $(wc -l < "$STEM.source.csv") SASS lines$( (( SINGLE_PASS )) && echo " ; timing from single-pass runs" )"
