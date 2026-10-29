#!/usr/bin/env python3
"""The scenes of the figures described, one line each, into
plots/scene_statistics.txt (and a .tex twin the paper can \\input): what
each scene is made of and what it costs pbrt, so that a figure's tick
labels carry names alone and the facts live in one table (the user,
10-24 and 10-29: "a scene_statistics.txt file in the plots directory that
gives information on each scene, like triangle count, material count,
instance count, and any other useful statistics").

    python3 apps/pbrt/scene_statistics.py TABLE.tsv [TABLE.tsv ...] \\
        -o apps/pbrt/plots/scene_statistics.txt [--sweep TABLE.tsv ...]

The tables are gpu_compare.sh's or heatmap.sh's (`table.tsv`: scene, spp,
side, wall_s, kernel_ms, speedup, agree, verdict, note, image, path); every
scene any of them holds is described, keyed by its `path` under the scene
set (kroken/camera-1 and watercolor/camera-1 are two scenes of one file
name). `--sweep` names the tables whose scenes are the sweep figures' (the
paper's eight): those come first, in the tables' order, marked `*`; the
rest (the heatmap's) follow in order of the geometry pbrt built.

Nothing here is typed in. Each number is read from a file a run wrote:

- pbrt's `.stats` beside its image at the scene's highest sample count in
  the tables (`--stats` output of `pbrt --gpu`): the Geometry section's
  shape counts -- `Triangles per mesh N / M` is N triangles in M meshes,
  the triangles displacement mapping added, spheres, disks, cylinders,
  curves, bilinear patches; the Scene section's materials, textures,
  lights and area lights, and `Object instances created` (the object
  definitions, not their uses); the Memory section's acceleration
  structures and image maps; the "Wavefront integrator statistics" ray
  counts (camera, indirect by depth, shadow), reported as rays per camera
  ray and the deepest bounce reached; the "Wavefront Kernel Profile"
  lines, whose names say which material kinds the scene shades, whether
  it has participating media and subsurface scattering, and the sampler;
- the converter's log beside the converted scene (apps/pbrt/scenes/
  converted/<path>.txt.log, convert_scene.sh): the resolution, the
  top-level meshes and triangles, and the instances placed -- `I instances
  of O objects holding M meshes of T triangles` -- the instance count pbrt
  does not print (pbrt counts definitions);
- the converted scene's header (<path>.txt): the integrator's maximum
  depth;
- the tables themselves: pbrt --gpu's wall at the highest sample count
  and the gpu-optix wavefront's speedup at the lowest and the highest.

A scene lacking a file prints `-` in that column. The .tex twin holds
the columns a paper table wants (triangles, meshes, instances, materials,
textures, lights, resolution, depth) as a booktabs tabular, the scene
labels as the figures print them (figure_style.scene_label).
"""
import argparse
import csv
import os
import re
import sys

from figure_style import scene_label

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTED = os.path.join(HERE, "scenes", "converted")


def read_tables(paths):
    """{path key: {"name", "pbrt_stats": {spp: stats path}, "pbrt_wall":
    {spp: s}, "wf": {spp: speedup}}} and the keys in table order."""
    scenes = {}
    order = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = r.get("path") or r["scene"]
                try:
                    spp = int(r["spp"])
                except ValueError:
                    continue
                if key not in scenes:
                    scenes[key] = {"name": r["scene"], "pbrt_stats": {}, "pbrt_wall": {},
                                   "wf": {}}
                    order.append(key)
                s = scenes[key]
                if r["side"].startswith("pbrt"):
                    image = r.get("image") or ""
                    if image.endswith(".png"):
                        s["pbrt_stats"][spp] = image[:-len(".png")] + ".stats"
                    try:
                        s["pbrt_wall"][spp] = float(r["wall_s"])
                    except ValueError:
                        pass
                elif r["side"] == "gpu-optix":
                    try:
                        s["wf"][spp] = float(r["speedup"].rstrip("x"))
                    except ValueError:
                        pass
    return scenes, order


def pbrt_stats(path):
    """What pbrt's --stats output says of the scene (see the module
    docstring); {} when the file is missing."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return {}
    out = {"materials_shaded": set(), "media": False, "subsurface": False, "sampler": None}
    # The kernel profile: material kernels, media, subsurface, the sampler.
    for m in re.finditer(r"^  (\w+)Material \+ BxDF eval", text, re.M):
        out["materials_shaded"].add(m.group(1))
    if re.search(r"^  Sample medium interaction", text, re.M):
        out["media"] = True
    if re.search(r"^  Tracing subsurface scattering probe rays", text, re.M):
        out["subsurface"] = True
    m = re.search(r"^  Generate ray samples - (\w+)Sampler", text, re.M)
    if m:
        out["sampler"] = m.group(1)
    # The integrator's ray counts.
    camera = indirect = shadow = 0
    deepest = 0
    for m in re.finditer(r"^\s+(Camera rays|Indirect rays, depth (\d+)|Shadow rays, depth \d+)"
                         r"\s+(\d+)$", text, re.M):
        n = int(m.group(3))
        if m.group(1) == "Camera rays":
            camera += n
        elif m.group(1).startswith("Indirect"):
            indirect += n
            deepest = max(deepest, int(m.group(2)))
        else:
            shadow += n
    out["camera_rays"] = camera
    out["rays_per_camera_ray"] = (camera + indirect + shadow) / camera if camera else None
    out["deepest_bounce"] = deepest
    # The Statistics sections.
    section = None
    counts = {}
    for line in text.splitlines():
        if line.startswith("Statistics:"):
            section = ""
            continue
        if section is None:
            continue
        if line.startswith("  ") and not line.startswith("    "):
            section = line.strip()
            continue
        m = re.match(r"\s+(.+?)\s{2,}(\S.*)$", line)
        if not m:
            continue
        counts[(section, m.group(1).strip())] = m.group(2).strip()

    def count(section, name):
        v = counts.get((section, name))
        if v is None:
            return 0
        m = re.match(r"(\d+)", v)
        return int(m.group(1)) if m else 0

    def pair(section, name):
        v = counts.get((section, name), "")
        m = re.match(r"(\d+)\s*/\s*(\d+)", v)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    def memory(section, name):
        v = counts.get((section, name))
        if v is None:
            return None
        m = re.match(r"([\d.]+)\s*(kB|MiB|GiB|B)", v)
        if not m:
            return None
        scale = {"B": 1 / 2**20, "kB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}[m.group(2)]
        return float(m.group(1)) * scale

    tris, meshes = pair("Geometry", "Triangles per mesh")
    patches, patch_meshes = pair("Geometry", "Bilinear patches per mesh")
    out.update({
        "triangles": tris, "meshes": meshes,
        "displaced": count("Geometry", "Triangles added from displacement mapping"),
        "patches": patches, "patch_meshes": patch_meshes,
        "spheres": count("Geometry", "Spheres"), "disks": count("Geometry", "Disks"),
        "cylinders": count("Geometry", "Cylinders"), "curves": count("Geometry", "Curves"),
        "materials": count("Scene", "Materials"), "textures": count("Scene", "Textures"),
        "lights": count("Scene", "Lights"), "area_lights": count("Scene", "AreaLights"),
        "objects": count("Scene", "Object instances created"),
        "accel_mib": memory("Memory", "Acceleration structures"),
        "image_maps_mib": memory("Memory", "Image maps"),
        "volume_grids": count("Memory", "Volume grids") or counts.get(("Memory", "Volume grids")),
    })
    return out


def converter_facts(key):
    """The converter's summary line and the conversion's header: resolution,
    top-level meshes and triangles, instances placed, maximum depth; {}
    when the scene was not converted here."""
    out = {}
    log = os.path.join(CONVERTED, key + ".txt.log")
    try:
        with open(log, errors="replace") as f:
            text = f.read()
    except OSError:
        return out
    m = re.search(r"\((\d+)x(\d+), (\d+) meshes of (\d+) triangles and (\d+) other shapes"
                  r"(?:, (\d+) instances of (\d+) objects holding (\d+) meshes of (\d+) triangles"
                  r" and (\d+) other shapes)?\)", text)
    if m:
        out["resolution"] = f"{m.group(1)}x{m.group(2)}"
        out["top_meshes"] = int(m.group(3))
        out["top_triangles"] = int(m.group(4))
        out["top_other"] = int(m.group(5))
        if m.group(6) is not None:
            out["instances"] = int(m.group(6))
            out["instance_objects"] = int(m.group(7))
            out["instance_meshes"] = int(m.group(8))
            out["instance_triangles"] = int(m.group(9))
        else:
            out["instances"] = 0
            out["instance_objects"] = 0
            out["instance_meshes"] = 0
            out["instance_triangles"] = 0
    header = os.path.join(CONVERTED, key + ".txt")
    try:
        with open(header, errors="replace") as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                m = re.match(r"integrator (\w+) maxdepth (\d+)", line)
                if m:
                    out["integrator"] = m.group(1)
                    out["maxdepth"] = int(m.group(2))
                m = re.match(r"resolution (\d+) (\d+)", line)
                if m and "resolution" not in out:
                    out["resolution"] = f"{m.group(1)}x{m.group(2)}"
    except OSError:
        pass
    return out


def thousands(n):
    return "-" if n is None else f"{n:,}"


def compact(n):
    """A count to three digits with a letter: 66.5k, 2.50M, 288M."""
    if n is None:
        return "-"
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            v = n / scale
            return f"{v:.0f}{unit}" if v >= 100 else f"{v:.1f}{unit}" if v >= 10 else f"{v:.2f}{unit}"
    return str(n)


def mib(v):
    if v is None:
        return "-"
    return f"{v / 1024:.2f} GiB" if v >= 1024 else f"{v:.0f} MiB" if v >= 10 else f"{v:.2f} MiB"


def describe(key, scene, sweep):
    """One scene's facts as a dict of column -> text."""
    spps = sorted(scene["pbrt_stats"])
    stats = pbrt_stats(scene["pbrt_stats"][spps[-1]]) if spps else {}
    conv = converter_facts(key)
    other = sum(stats.get(k, 0) for k in ("spheres", "disks", "cylinders", "curves", "patches"))
    other_parts = [f"{stats[k]} {k}" for k in ("spheres", "disks", "cylinders", "curves")
                   if stats.get(k)]
    if stats.get("patches"):
        other_parts.append(f"{stats['patches']} bilinear patches")
    kinds = sorted(stats.get("materials_shaded", ()))
    notes = []
    if stats.get("media"):
        notes.append("participating media")
    if stats.get("subsurface"):
        notes.append("subsurface scattering")
    if stats.get("displaced"):
        notes.append(f"{compact(stats['displaced'])} triangles from displacement")
    if conv.get("instances"):
        notes.append(f"{conv['instances']:,} instances of {conv['instance_objects']} objects "
                     f"({compact(conv['instance_triangles'])} triangles instanced)")
    walls = scene["pbrt_wall"]
    wf = scene["wf"]
    wall_spp = max(walls) if walls else None
    return {
        "scene": ("* " if sweep else "  ") + scene_label(key),
        "path": key,
        "resolution": conv.get("resolution", "-"),
        "depth": str(conv["maxdepth"]) if "maxdepth" in conv else "-",
        "sampler": stats.get("sampler") or "-",
        "triangles": thousands(stats.get("triangles")) if stats else "-",
        "meshes": thousands(stats.get("meshes")) if stats else "-",
        "other shapes": (", ".join(other_parts) if other_parts else ("0" if stats else "-")),
        "instances": (f"{conv['instances']:,} of {conv['instance_objects']}"
                      if conv.get("instances") else ("0" if conv else "-")),
        "materials": thousands(stats.get("materials")) if stats else "-",
        "material kinds": ", ".join(kinds) if kinds else "-",
        "textures": thousands(stats.get("textures")) if stats else "-",
        "lights": (f"{stats['lights']:,}" + (f" ({stats['area_lights']:,} area)"
                                            if stats.get("area_lights") else "")) if stats else "-",
        "accel": mib(stats.get("accel_mib")) if stats else "-",
        "image maps": mib(stats.get("image_maps_mib")) if stats and stats.get("image_maps_mib")
        else ("0" if stats else "-"),
        "rays/camera ray": (f"{stats['rays_per_camera_ray']:.2f} (to bounce {stats['deepest_bounce']})"
                            if stats.get("rays_per_camera_ray") else "-"),
        "pbrt --gpu": (f"{walls[wall_spp]:.2f} s at {wall_spp} spp" if wall_spp else "-"),
        "wf speedup": (f"{wf[min(wf)]:.2f}x at {min(wf)}, {wf[max(wf)]:.2f}x at {max(wf)} spp"
                       if wf else "-"),
        "notes": "; ".join(notes) if notes else "",
        # For sorting and the .tex.
        "_triangles": stats.get("triangles", 0) + other if stats else 0,
        "_label": scene_label(key),
        "_conv": conv, "_stats": stats,
    }


COLUMNS = ["scene", "path", "resolution", "depth", "sampler", "triangles", "meshes",
           "other shapes", "instances", "materials", "material kinds", "textures", "lights",
           "accel", "image maps", "rays/camera ray", "pbrt --gpu", "wf speedup", "notes"]
HEADINGS = {
    "scene": "scene (* in the sweep figures; the label the figures print)",
    "path": "path under the scene set",
    "resolution": "resolution",
    "depth": "max depth (the scene's integrator)",
    "sampler": "sampler",
    "triangles": "triangles (pbrt's count, instances not expanded, displacement included)",
    "meshes": "triangle meshes",
    "other shapes": "other shapes",
    "instances": "instances placed, of how many object definitions (the converter's count)",
    "materials": "materials",
    "material kinds": "material kinds shaded (pbrt's kernel names)",
    "textures": "textures",
    "lights": "lights (area lights among them)",
    "accel": "acceleration structures (pbrt --gpu)",
    "image maps": "image maps in memory",
    "rays/camera ray": "rays per camera ray at the highest spp (camera + indirect + shadow), and the deepest bounce reached",
    "pbrt --gpu": "pbrt --gpu wall time, least of the run's repeats",
    "wf speedup": "the gpu-optix wavefront's speedup over pbrt --gpu at the lowest and highest spp in the tables",
    "notes": "notes",
}


def write_text(path, rows, sources, sweep_sources):
    here = HERE
    with open(path, "w") as f:
        f.write("the scenes of the pbrt figures: what each is made of and what it costs pbrt\n")
        f.write("drawn from apps/pbrt/{" + ", ".join(os.path.relpath(p, here) for p in sources)
                + "}, the pbrt .stats beside each table's pbrt image at the highest spp, the\n"
                "converter's log and header under apps/pbrt/scenes/converted/<path>.txt\n")
        if sweep_sources:
            f.write("* marks the sweep figures' scenes (apps/pbrt/{"
                    + ", ".join(os.path.relpath(p, here) for p in sweep_sources)
                    + "}), first and in their order; the rest by the geometry pbrt built\n")
        f.write("\ncolumns\n")
        for c in COLUMNS:
            f.write(f"  {c}: {HEADINGS[c]}\n")
        f.write("\n")
        # One block per scene: the columns down, since the row is long.
        width = max(len(c) for c in COLUMNS)
        for r in rows:
            f.write(f"{r['scene']}  ({r['path']})\n")
            for c in COLUMNS:
                if c in ("scene", "path"):
                    continue
                if c == "notes" and not r[c]:
                    continue
                f.write(f"    {c:<{width}}  {r[c]}\n")
            f.write("\n")
        # And the one-line table of the counts, for a glance.
        cols = ["scene", "triangles", "meshes", "instances", "materials", "textures", "lights",
                "resolution", "depth", "pbrt --gpu", "wf speedup"]
        widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
        f.write("at a glance\n")
        f.write("  " + "  ".join(f"{c:<{widths[c]}}" for c in cols).rstrip() + "\n")
        for r in rows:
            f.write("  " + "  ".join(f"{r[c]:<{widths[c]}}" for c in cols).rstrip() + "\n")


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(path, rows):
    """A booktabs tabular of the counts: the paper's scene table."""
    with open(path, "w") as f:
        f.write("% generated by apps/pbrt/scene_statistics.py; do not edit\n")
        f.write("\\begin{tabular}{@{}lrrrrrrlr@{}}\n\\toprule\n")
        f.write("scene & triangles & meshes & instances & materials & textures & lights & "
                "resolution & depth \\\\\n\\midrule\n")
        for r in rows:
            conv = r["_conv"]
            instances = f"{conv['instances']:,}" if conv.get("instances") else "--"
            cells = [tex_escape(r["_label"]) + ("$^*$" if r["scene"].startswith("*") else ""),
                     r["triangles"], r["meshes"], instances, r["materials"], r["textures"],
                     r["lights"].split(" ")[0], r["resolution"].replace("x", "$\\times$"),
                     r["depth"]]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+", help="gpu_compare.sh / heatmap.sh tables")
    ap.add_argument("--sweep", nargs="*", default=[],
                    help="the tables whose scenes are the sweep figures' (listed first, marked *)")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "plots", "scene_statistics.txt"))
    args = ap.parse_args(argv[1:])

    sweep_scenes, sweep_order = read_tables(args.sweep) if args.sweep else ({}, [])
    scenes, order = read_tables(list(args.sweep) + list(args.tables))
    if not scenes:
        print("no scenes in the tables given", file=sys.stderr)
        return 1
    rows = [describe(k, scenes[k], True) for k in sweep_order]
    rest = [describe(k, scenes[k], False) for k in order if k not in sweep_scenes]
    rest.sort(key=lambda r: (r["_triangles"], r["_label"]))
    rows += rest
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_text(args.out, rows, args.tables, args.sweep)
    tex = os.path.splitext(args.out)[0] + ".tex"
    write_tex(tex, rows)
    print(f"wrote {args.out} and {tex} ({len(rows)} scenes, {len(sweep_order)} in the sweep)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
