#!/bin/bash
# Fold compare.sh logs into the results CSV: one row per measured cell.
#
#     apps/rtq/tocsv.sh [--date D] [--commit C] [--machine M] [--repeats N] \
#         [--keep old.csv] -o results.csv  <log>...
#
# A log is one compare.sh run over one mesh, with one or more schedules
# (`=== schedule S (...)` sections). Its name says the mesh and, optionally,
# the side: `<mesh>.s<side>[.<anything>].log` (sweep.sh writes them so); a
# name without `s<side>` takes the side off the primary count (side^2), so a
# `--query closest` log needs it in the name. The triangle count comes off
# the driver's `  N triangles, ...` line, the node and leaf counts off its
# `tree: ...` line (Embree's `N nodes ..., M leaves`; FCPW's `N node rows of
# B ..., M of them leaves`), the width off the schedule's name (embree/tuned
# 8, embree4/tuned4 4, fcpw<b>w<w> b, packet<K> 8 -- the tree's width, not
# the packet's, which the schedule's name keeps: a packet<K> row's reference
# rate is Embree's rtcIntersect<K> or rtcOccluded<K> over packets of K rays),
# the cpu off compare.sh's `pinned to cpu` line. Columns, as plot.py reads
# them:
#
#   date,commit,machine,cpu,repeats,reference,schedule,width,mesh,triangles,
#   nodes,leaves,query,rays,side,count,reference_rate,bonsai_rate,speedup,
#   same,ties,differ,reference_found,bonsai_found,callbacks_per_query,units,
#   file
#
# file is the mesh's path under the pbrt-v4-scenes checkout, off the
# driver's `mesh: /.../pbrt-v4-scenes/<scene>/geometry/<file>` line, so a
# row names the exact file it was measured on (the mesh table prints it).
# query is nearest|any|closest|silhouette (the driver's intersect|occluded|
# closest|silhouette); rays is primary|ao|diffuse|near|volume; reference is
# embree, fcpw or fcpw-gpu (the schedules named -gpu, against FCPW's
# GPUScene). The CSV is written whole from the
# logs given; with --keep, the rows of an earlier CSV whose cell (reference,
# schedule, width, mesh, query, rays) was not measured this time are carried
# over with their own date and commit, so one file holds every cell once.
set -euo pipefail
DATE="$(date +%F)"
COMMIT="$(git rev-parse --short HEAD)"
MACHINE="$(hostname -s)"
REPEATS=5
OUT=""
KEEP=""
LOGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date) DATE="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --machine) MACHINE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    *) LOGS+=("$1"); shift ;;
  esac
done
[[ -n "$OUT" && ${#LOGS[@]} -gt 0 ]] || { echo "usage: tocsv.sh [options] [--keep old.csv] -o out.csv log..." >&2; exit 1; }

HEADER="date,commit,machine,cpu,repeats,reference,schedule,width,mesh,triangles,nodes,leaves,query,rays,side,count,reference_rate,bonsai_rate,speedup,same,ties,differ,reference_found,bonsai_found,callbacks_per_query,units,file"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

for log in "${LOGS[@]}"; do
  name="$(basename "$log" .log)"
  mesh="${name%%.*}"
  side=""
  if [[ "$name" =~ \.s([0-9]+)(\.|$) ]]; then side="${BASH_REMATCH[1]}"; fi
  awk -v date="$DATE" -v commit="$COMMIT" -v machine="$MACHINE" -v repeats="$REPEATS" \
      -v mesh="$mesh" -v side_given="$side" '
    # The width of the tree (no apostrophes in these comments: the awk
    # program is a single-quoted string): the binary trees of FCPW (fcpwb,
    # fcpws, and their -gpu forms) 2, its Mbvh its branching, Embree eight
    # or four.
    function width_of(s) {
      if (s ~ /^fcpw[bs](-gpu)?$/) return 2
      if (s ~ /^fcpw/) { sub(/^fcpw/, "", s); sub(/w.*$/, "", s); return s }
      if (s ~ /^packet/) return 8
      if (s ~ /4$/) return 4
      return 8
    }
    # The reference: the GPU of FCPW for a schedule named -gpu, FCPW for
    # the other fcpw schedules, Embree otherwise.
    function reference_of(s) {
      if (s ~ /-gpu$/) return "fcpw-gpu"
      if (s ~ /^fcpw/) return "fcpw"
      return "embree"
    }
    /^pinned to cpu / { cpu = $4; sub(/,$/, "", cpu) }
    /^mesh: / { file = $2; sub(/^.*pbrt-v4-scenes\//, "", file); next }
    /^=== schedule / { schedule = $3; reference = reference_of(schedule); width = width_of(schedule); pside = ""; next }
    /^ +[0-9]+ triangles,/ { triangles = $1; next }
    /^tree: / {
      # The node count: the first number before "nodes" or "node rows"
      # ("tree: 12345 nodes ...", Embree; "tree: FCPW scalar BVH, 12345
      # binary nodes ..."; "tree: 12345 node rows of 4 ..."); the leaf
      # count the number before "leaves" or "of them leaves".
      nodes = ""
      for (i = 1; i <= NF; i++) {
        if (nodes == "" && $i ~ /^[0-9]+$/ && ($(i+1) ~ /^node/ || ($(i+1) == "binary" && $(i+2) ~ /^node/))) nodes = $i
        if ($(i+1) == "leaves" && $i ~ /^[0-9]+$/) leaves = $i
        if ($(i+1) == "of" && $(i+2) == "them" && $(i+3) == "leaves") leaves = $i
      }
      next
    }
    /^(primary|ao|diffuse|near|volume) +(intersect|occluded|closest|silhouette) / {
      rays = $1; q = $2; count = $3; ref = $4; ours = $5; sp = $6; sub(/x$/, "", sp)
      query = (q == "intersect") ? "nearest" : (q == "occluded") ? "any" : (q == "silhouette") ? "silhouette" : "closest"
      same = ""; ties = ""; differ = ""; rfound = ""; bfound = ""; cbq = ""
      rest = $0; sub(/^[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+x +/, "", rest)
      n = split(rest, parts, /[ ,;()]+/)
      for (i = 1; i <= n; i++) {
        # the first of each: "ties agree to ..." follows the counts
        if (parts[i+1] == "same" && same == "") same = parts[i]
        if (parts[i+1] == "ties" && ties == "") ties = parts[i]
        if (parts[i+1] == "differ" && differ == "") differ = parts[i]
        if ((parts[i] == "hits" || parts[i] == "blocked" || parts[i] == "found") && parts[i-1] != "bonsai") rfound = parts[i+1]
        if (parts[i] == "bonsai" && parts[i+1] ~ /^[0-9]+$/) bfound = parts[i+1]
        if (parts[i] == "ran" && parts[i+2] == "times") cbq = parts[i+1]
      }
      if (q == "occluded") ties = 0
      if (rays == "primary") pside = int(sqrt(count) + 0.5)
      side = (side_given != "") ? side_given : pside
      units = (q == "closest" || q == "silhouette") ? "Mqueries/s" : "Mrays/s"
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        date, commit, machine, cpu, repeats, reference, schedule, width, mesh, triangles, nodes, leaves,
        query, rays, side, count, ref, ours, sp, same, ties, differ, rfound, bfound, cbq, units, file
    }
  ' "$log"
done > "$TMP"

if [[ -n "$KEEP" ]]; then
  # Rows of the earlier CSV whose cell was not measured this time.
  awk -F, 'NR == FNR { seen[$6 "," $7 "," $8 "," $9 "," $13 "," $14] = 1; next }
           FNR > 1 && !(($6 "," $7 "," $8 "," $9 "," $13 "," $14) in seen)' "$TMP" "$KEEP" >> "$TMP"
fi

{
  echo "$HEADER"
  # reference, schedule, mesh by triangle count, query, ray set
  sort -t, -k6,6 -k7,7 -k10,10n -k13,13 -k14,14 "$TMP"
} > "$OUT"
echo "$(( $(wc -l < "$OUT") - 1 )) rows -> $OUT"
