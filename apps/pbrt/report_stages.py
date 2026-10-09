#!/usr/bin/env python3
"""The stage-by-stage tables of REPORT.md, from a heatmap run's `.stats` files.

    python3 apps/pbrt/report_stages.py apps/pbrt/compare-out-heatmap3 \
        [--spp 16] [--report apps/pbrt/REPORT.md]

A heatmap run (heatmap.sh -> gpu_compare.sh) leaves, for every scene and
sample count, our renderer's `BONSAI_KERNEL_STATS` profile
(`<scene>-s<spp>.gpu-optix.stats`) and pbrt's wavefront kernel profile
(`<scene>-s<spp>.pbrt.stats`), plus table.tsv with each cell's wall-clock
speedup and image verdict. This script pairs the kernels of the two profiles
into stages -- the same work under the two renderers' names, listed in
STAGES below -- and writes, in Markdown:

  * one line per scene: image verdict, kernel totals, wall speedups;
  * one line per stage: on how many scenes ours is behind, the worst, the
    sums over all scenes;
  * one table per stage: every scene that has it, ours, pbrt's, the ratio,
    worst first.

With --report the Markdown replaces what stands between the lines
`<!-- stages:begin -->` and `<!-- stages:end -->` of that file (the lines
themselves stay); without it, it goes to stdout. Times are milliseconds of
kernel time, the least of the harness's repeats; a ratio is ours over
pbrt's, below 1 a win. Only the chosen sample count's kernel times are
tabulated (16 by default: the 64 and 128 columns of the overview carry the
wall speedups only), since the stage split barely moves with spp.
"""
import argparse
import os
import re
import sys

# Each stage: its name, the pattern our kernel names match (the trailing
# `_<n>` of a block kernel already stripped), the pbrt kernel names that are
# the same work (a name ending in `*` matches by prefix), and a note.
# Material kernels are added per material (MATERIAL_*) and whatever is
# left on either side goes into "film + the rest".
STAGES = [
    ("closest-hit trace + routing", r"^__raygen__rays$",
     ["Trace closest hit rays"], ""),
    ("shadow traces (every pass)", r"^__raygen__(exit_)?shadow\$",
     ["Trace shadow rays", "Tracing shadow Tr rays"],
     "ours: `__raygen__shadow$<alpha>` and `__raygen__exit_shadow$<alpha>`; "
     "pbrt: the shadow and shadow-Tr trace kernels"),
    ("subsurface probe trace", r"^__raygen__probes$",
     ["Tracing subsurface scattering probe rays"], ""),
    ("subsurface exit", r"^_kernel_exits_blk$",
     ["Handle out-scattering after SSS"], ""),
    ("medium sampling", r"^_kernel_medium_samples_blk$",
     ["Sample medium interaction"], ""),
    ("medium scattering", r"^_kernel_medium_scatters_blk$",
     ["Sample direct/indirect - Henyey-Greenstein"], ""),
    ("sample draws (the bounce's dimensions, per ray)", r"^_kernel_rays_blk$",
     ["Generate ray samples - *"],
     "ours: the `rays` drain of the sample stage (10-09); a run before it has "
     "no kernel here, its draws having run inside the material and medium "
     "kernels, so pbrt's stands alone there"),
    ("camera rays", r"^_kernel_p_blk\$VolPath(\$prologue)?$",
     ["Generate camera rays"],
     "both sides draw the camera's own dimensions here"),
    ("escaped rays", r"^_kernel_escaped_blk$", ["Handle escaped rays"], ""),
    ("emitters hit", r"^_kernel_emissive_blk$",
     ["Handle emitters hit by indirect rays"], ""),
]
# `_kernel_hits_blk$<Material>` against pbrt's `<Material>Material + BxDF
# eval (...)` kernels; two of ours stand for more of pbrt's.
MATERIAL_EXTRA = {
    "Subsurface": ["Get BSSRDF and enqueue probe ray"],
}
MATERIAL_PBRT_NAME = {"Mixed": "Mix"}
MATERIAL_NOTE = {
    "Subsurface": "one kernel of ours against two of pbrt's: the material "
                  "evaluation and the BSSRDF sampling that enqueues the probe",
    "Mixed": "pbrt resolves a mix material inside its closest-hit program "
             "and has no kernel for it; ours is the choice step",
}
PBRT_ONLY_NOTE = ("pbrt launches a kernel for every material of the scene at "
                  "every depth, reached by a ray or not; ours launches only "
                  "for a non-empty queue, so a material no ray of ours reached "
                  "has no kernel here. Each row says how many launches pbrt "
                  "made and what each cost: at pbrt's launch floor of about "
                  "0.015 ms they were empty")
REST = "film + the rest"
REST_NOTE = ("ours: `rays_rest_blk` and `q_blk`; pbrt: Update film, the queue "
             "resets, the stats updates, Evaluate Displacement, Other")

BEGIN, END = "<!-- stages:begin -->", "<!-- stages:end -->"


def read_ours(path):
    """Our profile: {kernel name without its block number: ms}, total."""
    kernels, total = {}, None
    with open(path) as f:
        for line in f:
            m = re.match(r"\s+(\S+)\s+(\d+) launches\s+([\d.]+) ms /", line)
            if m:
                name = re.sub(r"_\d+$", "", m.group(1))
                kernels[name] = kernels.get(name, 0.0) + float(m.group(3))
                continue
            m = re.match(r"\s+total kernel time\s+([\d.]+) ms", line)
            if m:
                total = float(m.group(1))
    return kernels, total


def read_pbrt(path):
    """pbrt's wavefront profile: {kernel name: ms}, total, {name: launches}."""
    kernels, launches, total = {}, {}, None
    with open(path) as f:
        for line in f:
            m = re.match(r"\s+(.+?)\s+(\d+) launches\s+([\d.]+) ms /", line)
            if m:
                kernels[m.group(1)] = kernels.get(m.group(1), 0.0) + float(m.group(3))
                launches[m.group(1)] = launches.get(m.group(1), 0) + int(m.group(2))
                continue
            m = re.match(r"Total rendering time:\s+([\d.]+) ms", line)
            if m:
                total = float(m.group(1))
    return kernels, total, launches


def read_table(path):
    """table.tsv -> {scene path: {spp: {side: row dict}}}."""
    table = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            table.setdefault(row["path"], {}).setdefault(int(row["spp"]), {})[row["side"]] = row
    return table


def take(kernels, patterns):
    """Sum and remove the kernels matching the pbrt-side name patterns."""
    total = 0.0
    for name in list(kernels):
        for p in patterns:
            if (p.endswith("*") and name.startswith(p[:-1])) or name == p:
                total += kernels.pop(name)
                break
    return total


def take_re(kernels, pattern):
    total = 0.0
    for name in list(kernels):
        if re.match(pattern, name):
            total += kernels.pop(name)
    return total


VIEW = "materials with their sample draws (a view, not a stage)"
VIEW_NOTE = ("the material kernels above summed with the sample draws on each "
             "side -- pbrt's `Generate ray samples`, and our sample kernel where "
             "the run has one (until the sample stage of 10-09 the bounce's "
             "draws ran inside our material kernels) -- the like-for-like sum "
             "either way; a second view of rows already counted, left out of "
             "the totals")


def split_stages(ours, pbrt, launches=None):
    """Pair the two profiles' kernels into stages.

    Returns [(stage, ours ms, pbrt ms, note on the stage, detail on this scene's row)].
    """
    ours, pbrt, launches = dict(ours), dict(pbrt), launches or {}
    rows = []
    pbrt_samples = sum(v for k, v in pbrt.items() if k.startswith("Generate ray samples"))
    ours_samples = sum(v for k, v in ours.items() if k == "_kernel_rays_blk")
    for stage, ours_re, pbrt_names, note in STAGES:
        a, b = take_re(ours, ours_re), take(pbrt, pbrt_names)
        if a or b:
            rows.append((stage, a, b, note, ""))
    materials = sorted(m[len("_kernel_hits_blk$"):] for m in ours if m.startswith("_kernel_hits_blk$"))
    for mat in materials:
        a = ours.pop("_kernel_hits_blk$" + mat)
        pbrt_mat = MATERIAL_PBRT_NAME.get(mat, mat)
        b = take(pbrt, [pbrt_mat + "Material + BxDF eval*"] + MATERIAL_EXTRA.get(mat, []))
        rows.append(("material: " + mat, a, b, MATERIAL_NOTE.get(mat, ""), ""))
    # pbrt material kernels with no kernel of ours: kept visible, with what
    # each launch cost, rather than hidden in the rest.
    for name in sorted(pbrt):
        if "Material + BxDF eval" in name:
            n = launches.get(name, 0)
            b = pbrt.pop(name)
            detail = f" ({n} launches, {b / n:.3f} ms each)" if n else ""
            rows.append(("material: " + name.split("Material")[0] + " (pbrt only)", 0.0, b,
                         PBRT_ONLY_NOTE, detail))
    rows.append((REST, sum(ours.values()), sum(pbrt.values()), REST_NOTE, ""))
    ours_materials = sum(a for stage, a, _, _, _ in rows if stage.startswith("material: "))
    pbrt_materials = sum(b for stage, _, b, _, _ in rows if stage.startswith("material: "))
    if ours_materials or pbrt_materials:
        rows.append((VIEW, ours_materials + ours_samples, pbrt_materials + pbrt_samples,
                     VIEW_NOTE, ""))
    return rows


def ms(x):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 1000 else f"{x:.0f}"


def ratio(a, b):
    if not b:
        return "-" if not a else "ours only"
    if not a:
        return "pbrt only"
    return f"{a / b:.2f}"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", help="a heatmap.sh output directory")
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--report", help="REPORT.md to splice the tables into")
    args = ap.parse_args(argv[1:])

    table = read_table(os.path.join(args.out_dir, "table.tsv"))
    scenes = []
    for path in table:  # table order is the sweep's order
        cell = table[path].get(args.spp, {})
        if "gpu-optix" not in cell or "pbrt --gpu" not in cell:
            continue
        base = os.path.join(args.out_dir, path.replace("/", "-") + f"-s{args.spp}")
        if not (os.path.exists(base + ".gpu-optix.stats") and os.path.exists(base + ".pbrt.stats")):
            print(f"no stats for {path} at {args.spp} spp", file=sys.stderr)
            continue
        ours, ours_total = read_ours(base + ".gpu-optix.stats")
        pbrt, pbrt_total, pbrt_launches = read_pbrt(base + ".pbrt.stats")
        verdict = cell["gpu-optix"]["verdict"]
        speedups = {spp: table[path][spp]["gpu-optix"]["speedup"]
                    for spp in sorted(table[path]) if "gpu-optix" in table[path][spp]}
        scenes.append(dict(path=path, ours_total=ours_total, pbrt_total=pbrt_total,
                           verdict=verdict, speedups=speedups,
                           stages=split_stages(ours, pbrt, pbrt_launches)))

    out = []
    w = out.append
    spps = sorted({spp for s in scenes for spp in s["speedups"]})
    w(f"Generated by `report_stages.py` from `{os.path.basename(os.path.normpath(args.out_dir))}` "
      f"({len(scenes)} scenes; kernel times at {args.spp} spp).")
    w("")
    w("#### Every scene")
    w("")
    w("| scene | image | kernel ours | kernel pbrt | ratio | " +
      " | ".join(f"wall {spp} spp" for spp in spps) + " |")
    w("|---|---|---|---|---|" + "---|" * len(spps))
    for s in scenes:
        image = "ok" if s["verdict"].startswith("ok") else "**FAILED**"
        w(f"| {s['path']} | {image} | {ms(s['ours_total'])} | {ms(s['pbrt_total'])} | "
          f"{ratio(s['ours_total'], s['pbrt_total'])} | " +
          " | ".join(s["speedups"].get(spp, "-") for spp in spps) + " |")
    ok = [s for s in scenes if s["verdict"].startswith("ok")]
    w("")
    w(f"The wall speedup is pbrt's process over ours, both warm, from table.tsv. "
      f"Images: {len(ok)} of {len(scenes)} scenes match pbrt at {args.spp} spp; a FAILED row's "
      f"times are reported but do not count as beaten.")
    w("")

    # Per stage, over the scenes.
    by_stage = {}
    order = []
    for s in scenes:
        for stage, a, b, note, detail in s["stages"]:
            if stage not in by_stage:
                by_stage[stage] = dict(rows=[], note=note)
                order.append(stage)
            by_stage[stage]["rows"].append((s["path"], a, b, s["verdict"].startswith("ok"), detail))
    w("#### Every stage")
    w("")
    w("| stage | scenes | behind on | worst | sum ours | sum pbrt | ratio of sums |")
    w("|---|---|---|---|---|---|---|")
    for stage in order:
        rows = by_stage[stage]["rows"]
        comparable = [(p, a, b) for p, a, b, _, _ in rows if a and b]
        behind = [(a / b, p) for p, a, b in comparable if a > b]
        worst = max(((a / b, p) for p, a, b in comparable), default=None)
        sa, sb = sum(a for _, a, _, _, _ in rows), sum(b for _, _, b, _, _ in rows)
        w(f"| {stage} | {len(rows)} | {len(behind)} | "
          f"{(f'{worst[0]:.2f} ({worst[1]})' if worst else '-')} | "
          f"{ms(sa)} | {ms(sb)} | {ratio(sa, sb)} |")
    w("")
    w("A stage is beaten when ours is below pbrt's on every scene that has it. "
      "\"behind on\" counts the scenes where it is not; the sums are over all the "
      "scenes that have the stage on either side.")
    w("")
    for stage in order:
        rows = by_stage[stage]["rows"]
        note = by_stage[stage]["note"]
        w(f"##### {stage}")
        w("")
        if note:
            w(note + ".")
            w("")
        w("| scene | ours | pbrt | ratio |")
        w("|---|---|---|---|")

        def key(r):
            _, a, b, _, _ = r
            return -(a / b) if a and b else (1 if b else 2)

        for path, a, b, matched, detail in sorted(rows, key=key):
            flag = "" if matched else " (image FAILED)"
            r = ratio(a, b)
            if a and b and a > b:
                r = f"**{r}**"
            w(f"| {path}{flag}{detail} | {ms(a)} | {ms(b)} | {r} |")
        w("")

    text = "\n".join(out)
    if not args.report:
        sys.stdout.write(text)
        return 0
    with open(args.report) as f:
        report = f.read()
    i, j = report.find(BEGIN), report.find(END)
    if i < 0 or j < 0 or j < i:
        print(f"{args.report}: needs the lines {BEGIN} and {END}", file=sys.stderr)
        return 1
    report = report[:i + len(BEGIN)] + "\n" + text + "\n" + report[j:]
    with open(args.report, "w") as f:
        f.write(report)
    print(f"wrote {len(scenes)} scenes, {len(order)} stages into {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
