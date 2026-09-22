#!/usr/bin/env python3
"""Renders a scene at every (depth, spp) of a grid under each of apps/pbrt's
schedules and with pbrt, checks every image against pbrt's, and draws the
speedup heatmaps and, if asked, a grid of the images.

    python3 eval/render_matrix.py SCENE.pbrt --depths 1 2 3 4 5 --spps 16 64 256

Plots go to eval/plots/, everything else -- the converted scenes, the images,
the timings -- to eval/out/<scene>/. A cell that has been rendered is not
rendered again (--rerun to insist), so the plots can be redrawn without
waiting for the renders.

What is measured. The scene is converted once (scene_dump, on pbrt's own BVH
unless --own-tree); pbrt renders every (depth, spp) it has not rendered, and
then each schedule's renderer runs once over every cell it is missing from,
the scene loaded once and rendered at each cell's depth and count (the
driver's --cells), best of --repeats runs on each side, pbrt timed by its own
render timer and this renderer by the driver's. The speedup is pbrt's time
over the schedule's. Every render is checked against pbrt's the way
apps/pbrt/compare.sh checks one: the set of pixels that received light, the
mean over the image, and the fraction of lit pixels agreeing closely, with
compare_gbuffer.py's tolerances -- a stochastic estimate can promise no more
per pixel -- and the schedules' images are compared with each other bit for
bit. A cell that fails is marked in the heatmap and listed.

The images. --images says which renders to write out as PNGs: none, all, or
cells given as depth:spp or depth:spp:renderer (renderer one of pbrt or a
schedule). --grid draws a figure of them, one renderer per row and one value
of --grid-along per column, the other axis fixed at --grid-at: plain images
edge to edge, a label per row and column, and the time and speedup under
each, sized for a two-column page. --channel normals shows the geometric
normals instead, for a scene whose film is a gbuffer. --exposure applies a
viewing exposure (image_io's, Reinhard after a gain of 2**stops) to every
PNG and grid image; off, the encoding is pbrt's own quantisation.

Needs numpy and matplotlib beside the compiler in the bonsai environment, a
built pbrt (--pbrt, or PBRT in the environment), and the clang++ the app's
header needs (BONSAI_CXX, or clang++ on PATH).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = os.path.join(ROOT, "apps", "pbrt")
sys.path.insert(0, PREFIX)
import compare_gbuffer  # noqa: E402  the tolerances, so there is one set

# The queue schedules are labelled "compact (wf)", not "wavefront": what they
# have of a wavefront renderer is the queue of paths between bounces and its
# compaction; a true wavefront has a queue per stage (trace, shade, shadow),
# and "wavefront" is kept for that (the user's naming, 2026-09-21).
#
# "pbrt-gpu" is `pbrt --gpu`, which is not a schedule but a second reference:
# pbrt's wavefront renderer on the GPU. It runs pbrt's *volpath* integrator
# whatever the scene names -- pbrt has no other GPU integrator -- so on a
# scene without media it converges to the same image as pbrt's path
# integrator and is checked against it like the schedules are, but its time
# is the time of a different integrator until this renderer has volpath too.
# The label says so.
RENDERER_LABELS = {"pbrt": "pbrt", "pbrt-gpu": "pbrt --gpu (volpath)",
                   "scalar": "scalar",
                   "perlane": "per-lane", "packet": "packet",
                   "wavefront-perlane": "compact (wf) per-lane",
                   "wavefront": "compact (wf) packet",
                   "gpu": "GPU megakernel"}

# The schedules in apps/pbrt/schedules/, in the order the figures show them:
# the sample loop run three ways, then the queue run two ways.
DEFAULT_SCHEDULES = ["scalar", "perlane", "packet", "wavefront-perlane",
                     "wavefront"]
DEFAULT_SPPS = [16, 32, 64, 128, 256, 512, 1024]


def say(*parts):
    print(*parts, flush=True)


def run(cmd, **kw):
    """A subprocess that has to succeed, its output shown when it does not."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        say(f"failed: {' '.join(map(str, cmd))}")
        say(result.stdout)
        say(result.stderr)
        raise SystemExit(1)
    return result.stdout


#===------------------------------------------------------------------------===#
# Images
#===------------------------------------------------------------------------===#

def read_pfm(path):
    """An (height, width, 3) float32 array, row 0 at the top."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"PF":
            raise SystemExit(f"{path}: not a three-channel PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(width * height * 12),
                             dtype="<f4" if scale < 0 else ">f4")
    return data.reshape(height, width, 3)[::-1].astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def encode_radiance(image, stops=None):
    """Linear radiance to displayable sRGB in [0, 1]; see image_io.py."""
    if stops is None:
        return linear_to_srgb(image)
    x = np.maximum(image, 0.0) * 2.0 ** stops
    return linear_to_srgb(x / (1.0 + x))


def encode_normals(image):
    """[-1, 1] to [0, 1], a miss left black; see image_io.py."""
    hit = np.any(image != 0.0, axis=2, keepdims=True)
    return np.where(hit, np.clip(0.5 * (image + 1.0), 0.0, 1.0), 0.0)


def write_png(path, rgb):
    import matplotlib.image
    matplotlib.image.imsave(path, np.clip(rgb, 0.0, 1.0))


def check_radiance(pbrt, ours):
    """compare_gbuffer.compare_radiance, over arrays: the lit pixels on each
    side, the ratio of the means, and the fraction of pixels lit on either
    side that agree to RADIANCE_TOLERANCE relative to the brighter."""
    lit_pbrt = np.any(pbrt > 0, axis=2)
    lit_ours = np.any(ours > 0, axis=2)
    either = lit_pbrt | lit_ours
    pbrt_mean = float(pbrt.mean())
    ours_mean = float(ours.mean())
    if pbrt_mean == 0.0 and ours_mean == 0.0:
        ratio = 1.0
    else:
        ratio = ours_mean / pbrt_mean if pbrt_mean else float("inf")
    scale = np.maximum(pbrt.max(axis=2), ours.max(axis=2))
    close = np.abs(pbrt - ours).max(axis=2) <= \
        compare_gbuffer.RADIANCE_TOLERANCE * scale
    touched = int(either.sum())
    agree = float((close & either).sum()) / touched if touched else 1.0
    n_pbrt, n_ours = int(lit_pbrt.sum()), int(lit_ours.sum())
    failed = []
    if abs(n_pbrt - n_ours) > compare_gbuffer.RADIANCE_LIT_TOLERANCE * max(n_pbrt, 1):
        failed.append(f"{n_ours} lit pixels against pbrt's {n_pbrt}")
    if abs(ratio - 1.0) > compare_gbuffer.RADIANCE_MEAN_TOLERANCE:
        failed.append(f"mean {ratio:.5f}x pbrt's")
    return {"lit_pbrt": n_pbrt, "lit_ours": n_ours, "mean_ratio": ratio,
            "agree": agree, "failed": failed}


#===------------------------------------------------------------------------===#
# Building
#===------------------------------------------------------------------------===#

def cxx():
    compiler = os.environ.get("BONSAI_CXX") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("no clang++: set BONSAI_CXX (the generated header "
                         "needs clang's ext_vector_type)")
    return compiler


def tbb_flags(compiler):
    """TBB lives beside the compiler in a conda environment; see compare.sh."""
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(compiler)))
    if os.path.isfile(os.path.join(prefix, "include", "tbb", "parallel_for.h")):
        return [f"-I{prefix}/include", f"-L{prefix}/lib",
                f"-Wl,-rpath,{prefix}/lib", "-ltbb"]
    say("no TBB beside the C++ compiler; the renders will balance their loops "
        "with std::thread, which is slower")
    return []


def build(out, schedules):
    """The compiler, scene_dump, and a renderer per schedule (its compile
    timed, being part of what a schedule costs). Returns the compile times."""
    # The compiler's build directory: `build` unless BONSAI_BUILD_DIR names
    # another, as compare.sh reads it too.
    build_dir = os.environ.get("BONSAI_BUILD_DIR", "build")
    run(["cmake", "--build", build_dir, "-j"], cwd=ROOT)
    run(["bash", f"{PREFIX}/build_scene_dump.sh", f"{out}/scene_dump"], cwd=ROOT)
    run([f"{out}/scene_dump", "--check-tables"], cwd=ROOT)
    compiler = cxx()
    flags = tbb_flags(compiler)
    compile_seconds = {}
    for schedule in schedules:
        file = f"{PREFIX}/schedules/{schedule}.bonsai"
        if not os.path.isfile(file):
            raise SystemExit(f"no schedule {file}")
        say(f"compiling {schedule}")
        started = time.perf_counter()
        run([f"./{build_dir}/compiler", "-p", "ssa", "--no-heap", "--ffp-contract",
             "-i", f"{PREFIX}/render.bonsai", "-i", file, "-b", "cpp",
             "-o", f"{out}/render_{schedule}"], cwd=ROOT)
        compile_seconds[schedule] = time.perf_counter() - started
        # render_hook.cpp includes "render.h": each schedule's in a directory
        # of its own, with the driver copied beside it. Copied, not found
        # through -I: a quoted include looks in the including file's own
        # directory first, so a driver compiled from apps/pbrt/ takes the
        # render.h that render.sh last left there -- another schedule's --
        # whatever -I says. Every CPU schedule's header is the same, which is
        # why this went unnoticed until the GPU schedule's, whose buffers are
        # staged to the device rather than the host, was built against the
        # scalar one and found nothing on the device at its first launch.
        include = f"{out}/inc_{schedule}"
        os.makedirs(include, exist_ok=True)
        shutil.copy(f"{out}/render_{schedule}.h", f"{include}/render.h")
        shutil.copy(f"{PREFIX}/render_hook.cpp", f"{include}/render_hook.cpp")
        run([compiler, "-g", "-std=c++20", "-O3", "-I.", f"-I{PREFIX}",
             f"{include}/render_hook.cpp",
             f"{out}/render_{schedule}.o", *flags,
             "-o", f"{out}/render_{schedule}.out"], cwd=ROOT)
    return compile_seconds


#===------------------------------------------------------------------------===#
# Rendering
#===------------------------------------------------------------------------===#

def scene_film_and_integrator(scene):
    text = open(scene).read()
    film = re.search(r'^\s*Film\s*"([a-z]+)"', text, re.M)
    integrator = re.search(r'^\s*Integrator\s*"([a-z]+)"', text, re.M)
    return (film.group(1) if film else None,
            integrator.group(1) if integrator else None)


def pbrt_scene_text(scene, depth):
    """The scene as pbrt is to read it, at `depth`: its Integrator directive
    rewritten with the maxdepth, or one put in front of a scene that names
    none -- `path`, which is what scene_dump resolves such a scene to. pbrt
    has no flag for the depth; see compare.sh."""
    lines = open(scene).read().split("\n")
    out = []
    in_integrator = False
    named = False
    for line in lines:
        if re.match(r"^\s*Integrator\s", line):
            named = True
            in_integrator = True
            line = re.sub(r'"integer maxdepth"\s*\[[^\]]*\]', "", line)
            line = re.sub(r'^(\s*Integrator\s+"[a-z]+")',
                          rf'\1 "integer maxdepth" [ {depth} ]', line)
            out.append(line)
            continue
        if in_integrator and re.match(r'^\s*"', line):
            if '"integer maxdepth"' not in line:
                out.append(line)
            continue
        in_integrator = False
        out.append(line)
    text = "\n".join(out)
    if not named:
        text = f'Integrator "path" "integer maxdepth" [ {depth} ]\n' + text
    return text


def imgtool(pbrt):
    tool = os.environ.get("IMGTOOL") or os.path.join(os.path.dirname(pbrt), "imgtool")
    if not os.path.isfile(tool):
        raise SystemExit(f"no imgtool beside {pbrt}")
    return tool


def pbrt_render_seconds(tool, exr):
    info = run([tool, "info", exr])
    found = re.search(r"\(total ([0-9.]+)s\)", info)
    if not found:
        raise SystemExit(f"no render time in {exr}'s metadata")
    return float(found.group(1))


def repeats_at(args, spp):
    """How many runs a cell at `spp` is the best of: --repeats, or
    --long-repeats from --long-spp up, where a render is long enough for one
    run's noise to be small next to it and the repeats would cost hours."""
    if args.long_spp is not None and spp >= args.long_spp:
        return args.long_repeats
    return args.repeats


def render_pbrt(args, out, scene, tag, depth, spp, gbuffer, gpu=False):
    """pbrt on the scene at this depth and count: the best time of the cell's
    repeats (see repeats_at), the radiance as a PFM, and the normals when the
    film has them. With `gpu`, `pbrt --gpu` -- pbrt's wavefront volpath on
    the GPU (see RENDERER_LABELS), timed by the same render timer, which
    pbrt starts after the scene is on the device; its files are the
    `<tag>-pbrt-gpu` ones, and it writes no normals."""
    name = "pbrt-gpu" if gpu else "pbrt"
    exr = f"{out}/{tag}-{name}.exr"
    best = None
    for _ in range(repeats_at(args, spp)):
        if os.path.exists(exr):
            os.remove(exr)
        result = subprocess.run(
            [args.pbrt, *(["--gpu"] if gpu else []), "--outfile", exr,
             "--spp", str(spp)],
            input=pbrt_scene_text(scene, depth), capture_output=True,
            text=True, cwd=os.path.dirname(scene))
        if not os.path.exists(exr) or os.path.getsize(exr) == 0:
            say(f"pbrt wrote nothing for {tag}:\n{result.stdout}\n{result.stderr}")
            raise SystemExit(1)
        seconds = pbrt_render_seconds(imgtool(args.pbrt), exr)
        best = seconds if best is None else min(best, seconds)
    run([imgtool(args.pbrt), "convert", "--channels", "R,G,B",
         "--outfile", f"{out}/{tag}-{name}-radiance.pfm", exr])
    if gbuffer and not gpu:
        run([imgtool(args.pbrt), "convert", "--channels", "N.X,N.Y,N.Z",
             "--outfile", f"{out}/{tag}-pbrt.pfm", exr])
    return best


def convert_scene(args, out, scene):
    """The scene converted once, at its own sample count and depth: the
    renders set theirs (the driver's --cells), so the conversion -- pbrt's
    parse, its BVH, and a dump that is gigabytes for a large scene -- is not
    repeated per cell. Converted afresh each run, since the converter may have
    changed; the dumps an earlier version of this tool made per cell are
    removed, being the same scene thirty-five times over."""
    for stale in os.listdir(out):
        if re.fullmatch(r"d\d+-s\d+\.txt", stale):
            os.remove(os.path.join(out, stale))
    flags = [] if args.own_tree else ["--pbrt-tree"]
    path = f"{out}/scene.txt"
    say("converting the scene")
    run([f"{out}/scene_dump", *flags, scene, path], cwd=ROOT)
    return path


def render_cells(out, schedule, cells, repeats, on_cell):
    """A schedule's renderer over `cells` (tags), from the one converted
    scene loaded once: the best of `repeats` runs of each, which the driver
    takes itself, and the images it writes per cell -- the normals at
    <tag>-<schedule>.pfm, the radiance at <tag>-<schedule>-radiance.pfm.
    `on_cell(tag, seconds)` is called as each cell's time arrives, so that a
    long run reports as it goes. Returns {tag: seconds}."""
    # --no-implicit-copies: the driver stages every buffer before its timer
    # starts, and a copy the compiled render would otherwise make inside the
    # timed region is an error rather than a number.
    cmd = [f"{out}/render_{schedule}.out", "--no-implicit-copies",
           "--cells", ",".join(cells),
           f"{out}/scene.txt", f"{out}/{{cell}}-{schedule}.pfm"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env={**os.environ,
                                               "BONSAI_REPEATS": str(repeats)})
    seconds = {}
    for line in process.stdout:
        found = re.match(r"^render seconds (d\d+-s\d+): ([0-9.eE+-]+)", line)
        if found:
            seconds[found.group(1)] = float(found.group(2))
            on_cell(found.group(1), seconds[found.group(1)])
    process.wait()
    if process.returncode != 0 or set(seconds) != set(cells):
        say(f"failed: {' '.join(cmd)}")
        say(process.stderr.read())
        raise SystemExit(1)
    return seconds


def measure(args, out, scene, results):
    """Every cell of the grid, for every renderer that has not rendered it
    yet: pbrt first over the cells it is missing from, then each schedule over
    the cells it is missing from, in one run of its renderer over the scene
    loaded once. A schedule added later -- a new file in schedules/ -- is
    rendered into the cached grid without rendering the others again."""
    film, integrator = scene_film_and_integrator(scene)
    gbuffer = film == "gbuffer" and (integrator or "path") in ("path", "volpath")
    results["gbuffer"] = gbuffer
    cells = results.setdefault("cells", {})
    grid = [(d, p, f"d{d}-s{p}") for d in args.depths for p in args.spps]

    def save():
        with open(f"{out}/results.json", "w") as f:
            json.dump(results, f, indent=1)

    # pbrt, where it has not rendered: its time, and the image every schedule
    # is checked against.
    for depth, spp, tag in grid:
        cell = cells.setdefault(tag, {})
        if "pbrt_seconds" in cell and \
                os.path.isfile(f"{out}/{tag}-pbrt-radiance.pfm"):
            continue
        say(f"== pbrt: depth {depth}, {spp} spp (best of {repeats_at(args, spp)})")
        cell["pbrt_seconds"] = render_pbrt(args, out, scene, tag, depth, spp,
                                           gbuffer)
        # What the cell's times are the best of, for the figures to say.
        cell["repeats"] = repeats_at(args, spp)
        say(f"   pbrt: {cell['pbrt_seconds']:.3f} s")
        save()

    # pbrt on the GPU, where asked and not rendered: a second reference,
    # recorded like a schedule -- its time, its speedup over pbrt's CPU
    # render, and its image checked against that render's -- so that the
    # table and the figures carry it as a column. It is a different
    # integrator (see RENDERER_LABELS), which the label says.
    if args.pbrt_gpu:
        for depth, spp, tag in grid:
            cell = cells[tag]
            if "pbrt-gpu" in cell and \
                    os.path.isfile(f"{out}/{tag}-pbrt-gpu-radiance.pfm"):
                continue
            say(f"== pbrt --gpu: depth {depth}, {spp} spp "
                f"(best of {repeats_at(args, spp)})")
            seconds = render_pbrt(args, out, scene, tag, depth, spp,
                                  gbuffer=False, gpu=True)
            check = check_radiance(read_pfm(f"{out}/{tag}-pbrt-radiance.pfm"),
                                   read_pfm(f"{out}/{tag}-pbrt-gpu-radiance.pfm"))
            check["seconds"] = seconds
            check["speedup"] = cell["pbrt_seconds"] / seconds
            check["same_image_as"] = None
            cell["pbrt-gpu"] = check
            verdict = "FAILED: " + "; ".join(check["failed"]) if check["failed"] else "ok"
            say(f"   pbrt --gpu: {seconds:.3f} s ({check['speedup']:.2f}x pbrt), "
                f"mean {check['mean_ratio']:.5f}x, {100 * check['agree']:.1f}% close, "
                f"{verdict}")
            save()

    # Each schedule over the cells it is missing from, grouped by how many
    # runs each is the best of (one setting per run of the renderer).
    first = args.schedules[0]
    for schedule in args.schedules:
        missing = [(d, p, tag) for d, p, tag in grid if schedule not in cells[tag]]
        if not missing:
            continue
        say(f"== {schedule}: {len(missing)} cells")
        by_repeats = {}
        for depth, spp, tag in missing:
            by_repeats.setdefault(repeats_at(args, spp), []).append(tag)

        def checked(tag, seconds):
            cell = cells[tag]
            pbrt_radiance = read_pfm(f"{out}/{tag}-pbrt-radiance.pfm")
            ours = read_pfm(f"{out}/{tag}-{schedule}-radiance.pfm")
            check = check_radiance(pbrt_radiance, ours)
            check["seconds"] = seconds
            check["speedup"] = cell["pbrt_seconds"] / seconds
            # Bit for bit the same image as the first schedule's, where that
            # has been rendered (in this run or a cached one).
            reference = f"{out}/{tag}-{first}-radiance.pfm"
            if schedule == first or not os.path.isfile(reference):
                check["same_image_as"] = None
            else:
                check["same_image_as"] = \
                    first if np.array_equal(read_pfm(reference), ours) else "differs"
            cell[schedule] = check
            verdict = "FAILED: " + "; ".join(check["failed"]) if check["failed"] else "ok"
            say(f"   {tag}: {seconds:.3f} s ({check['speedup']:.2f}x pbrt), "
                f"mean {check['mean_ratio']:.5f}x, {100 * check['agree']:.1f}% close, "
                f"{verdict}")
            save()

        for repeats, tags in sorted(by_repeats.items(), reverse=True):
            render_cells(out, schedule, tags, repeats, checked)
    return results


#===------------------------------------------------------------------------===#
# Reporting
#===------------------------------------------------------------------------===#

def table(args, results, path):
    """results.tsv: one row per (depth, spp, schedule)."""
    rows = ["depth\tspp\tschedule\tseconds\tpbrt_seconds\tspeedup\tmean_ratio"
            "\tagree\tverdict\tsame_image_as"]
    for depth in args.depths:
        for spp in args.spps:
            cell = results["cells"][f"d{depth}-s{spp}"]
            for schedule in args.renderers:
                c = cell[schedule]
                rows.append("\t".join(map(str, [
                    depth, spp, schedule, f"{c['seconds']:.4f}",
                    f"{cell['pbrt_seconds']:.3f}", f"{c['speedup']:.2f}",
                    f"{c['mean_ratio']:.5f}", f"{c['agree']:.3f}",
                    "ok" if not c["failed"] else "FAILED",
                    c["same_image_as"] or "-"])))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    say("\n".join(rows))


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })
    return plt


def repeats_caption(args, results):
    """One line saying how many runs the cells are the best of: the same
    count everywhere, or the two counts a --long-spp run used, or that the
    cache predates the record."""
    counts = set()
    for tag, cell in results["cells"].items():
        if "repeats" in cell:
            counts.add(cell["repeats"])
        else:
            counts.add(None)
    if counts == {None}:
        return "timings: best of an unrecorded number of runs (older cache)"
    if None in counts:
        return "timings: best of a number of runs not recorded for every cell"
    if len(counts) == 1:
        return f"timings: best of {counts.pop()} runs, both sides"
    spp = [p for p in args.spps
           if results["cells"][f"d{args.depths[0]}-s{p}"]["repeats"] == min(counts)]
    return (f"timings: best of {max(counts)} runs up to {min(spp) // 2} spp, "
            f"best of {min(counts)} from {min(spp)} spp up (a first look, not "
            f"a benchmark)")


def heatmaps(args, results, scene_name):
    """One panel per schedule, depth down and spp across, the cell its speedup
    over pbrt; one colour scale for all of them so that they read against
    each other. A cell whose image fails the check is marked with a dagger."""
    plt = plot_style()
    speed = {s: np.array([[results["cells"][f"d{d}-s{p}"][s]["speedup"]
                           for p in args.spps] for d in args.depths])
             for s in args.renderers}
    failed = {s: np.array([[bool(results["cells"][f"d{d}-s{p}"][s]["failed"])
                            for p in args.spps] for d in args.depths])
              for s in args.renderers}
    vmax = max(float(v.max()) for v in speed.values())
    vmin = min(1.0, min(float(v.min()) for v in speed.values()))
    n = len(args.renderers)
    # Room for a "1.2×" in every cell: near half an inch per column, which
    # runs past a two-column page once the sample counts are seven and the
    # schedules five -- a figure to read, or to crop, rather than to print
    # whole. The tick labels lean over for the same reason.
    many = len(args.spps) > 4
    width = min(16.0, max(2.1 * n + 0.6, 0.46 * len(args.spps) * n + 0.8))
    fig, axes = plt.subplots(1, n, figsize=(width, 0.42 * len(args.depths) + 0.9),
                             squeeze=False)
    cmap = plt.get_cmap("Blues")
    for ax, schedule in zip(axes[0], args.renderers):
        image = ax.imshow(speed[schedule], cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect="auto")
        ax.set_title(RENDERER_LABELS.get(schedule, schedule), pad=4)
        ax.set_xticks(range(len(args.spps)), [str(p) for p in args.spps],
                      rotation=45 if many else 0)
        ax.set_yticks(range(len(args.depths)), [str(d) for d in args.depths])
        ax.set_xlabel("samples per pixel")
        if schedule == args.renderers[0]:
            ax.set_ylabel("path depth")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(args.depths)):
            for j in range(len(args.spps)):
                v = speed[schedule][i, j]
                dark = (v - vmin) / max(vmax - vmin, 1e-9) > 0.55
                ax.text(j, i, f"{v:.1f}×" + ("†" if failed[schedule][i, j] else ""),
                        ha="center", va="center", fontsize=6 if many else 7,
                        color="white" if dark else "black")
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    bar.set_label("speedup over pbrt")
    # How many runs each number is the best of, so a first look is not read
    # as a benchmark. Recorded per cell by measure(); an older cache says
    # nothing, and the caption says so.
    # Below the panels' own labels, which hang under the figure's edge since
    # the layout is the default one; the tight bounding box takes it in.
    fig.text(0.0, -0.22, repeats_caption(args, results), fontsize=6,
             ha="left", va="top", transform=fig.transFigure)
    bar.outline.set_linewidth(0.5)
    fig.savefig(f"{args.plots}/{scene_name}-speedup.pdf", bbox_inches="tight")
    fig.savefig(f"{args.plots}/{scene_name}-speedup.png", bbox_inches="tight")
    plt.close(fig)
    say(f"wrote {args.plots}/{scene_name}-speedup.pdf")


def image_for(args, out, tag, renderer):
    """The displayable image of one render: radiance, or the normals."""
    if args.channel == "normals":
        path = f"{out}/{tag}-{renderer}.pfm"
        if not os.path.isfile(path):
            raise SystemExit(f"no normals for {renderer} at {tag}: {path} -- a "
                             "scene needs a gbuffer film and the path "
                             "integrator for pbrt to write any")
        return encode_normals(read_pfm(path))
    return encode_radiance(read_pfm(f"{out}/{tag}-{renderer}-radiance.pfm"),
                           args.exposure)


def crop(image, box):
    if box is None:
        return image
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def images_wanted(args):
    """The (depth, spp, renderer) cells --images asks for, checked before any
    rendering so that a mistyped list fails in a second rather than after the
    run: `none`, `all`, or cells as depth:spp (every renderer) or
    depth:spp:renderer."""
    renderers = ["pbrt", *args.renderers]
    every = [(d, p, r) for d in args.depths for p in args.spps for r in renderers]
    if not args.images or args.images == ["none"]:
        return []
    if args.images == ["all"]:
        return every
    chosen = set()
    for spec in args.images:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts[:2]):
            raise SystemExit(
                f"--images takes none, all, or cells as depth:spp or "
                f"depth:spp:renderer (say 5:64 or 5:64:packet), not {spec!r}")
        depth, spp = int(parts[0]), int(parts[1])
        if depth not in args.depths or spp not in args.spps:
            raise SystemExit(f"--images {spec}: depth {depth} at {spp} spp is "
                             f"not a cell of this run (depths {args.depths}, "
                             f"spps {args.spps})")
        if len(parts) == 3 and parts[2] not in renderers:
            raise SystemExit(f"--images {spec}: renderer {parts[2]!r} is not "
                             f"one of {renderers}")
        for renderer in [parts[2]] if len(parts) == 3 else renderers:
            chosen.add((depth, spp, renderer))
    return [w for w in every if w in chosen]


def write_images(args, out, scene_name, wanted):
    """The renders --images asked for, as PNGs in the plots directory."""
    for depth, spp, renderer in wanted:
        tag = f"d{depth}-s{spp}"
        path = f"{args.plots}/{scene_name}-{tag}-{renderer}-{args.channel}.png"
        write_png(path, crop(image_for(args, out, tag, renderer), args.grid_crop))
        say(f"wrote {path}")


def image_grid(args, out, scene_name, results):
    """A grid of the images, a renderer per row and a depth (or a count) per
    column, at the other axis' --grid-at. Edge to edge, a label per row and
    column, the time and the speedup under each image; nothing else."""
    plt = plot_style()
    along = args.depths if args.grid_along == "depth" else args.spps
    at = args.grid_at
    if at is None:
        at = max(args.spps) if args.grid_along == "depth" else max(args.depths)
    rows = args.grid_rows
    tags = [f"d{v}-s{at}" if args.grid_along == "depth" else f"d{at}-s{v}"
            for v in along]
    first = crop(image_for(args, out, tags[0], rows[0]), args.grid_crop)
    aspect = first.shape[0] / first.shape[1]
    # Inches: a margin for the row labels, a gutter between images, a line
    # under each image for its caption, a line above the top row for the
    # column labels.
    width = args.grid_width
    margin, gutter, caption, header = 0.7, 0.04, 0.16, 0.16
    cell_w = (width - margin - gutter * (len(along) - 1)) / len(along)
    cell_h = cell_w * aspect
    fig_h = header + len(rows) * (cell_h + caption) + gutter * (len(rows) - 1)
    fig, axes = plt.subplots(len(rows), len(along), figsize=(width, fig_h),
                             squeeze=False,
                             gridspec_kw={"wspace": gutter / cell_w,
                                          "hspace": gutter / (cell_h + caption),
                                          "left": margin / width, "right": 1.0,
                                          "top": 1 - header / fig_h, "bottom": 0})
    for i, renderer in enumerate(rows):
        for j, (value, tag) in enumerate(zip(along, tags)):
            ax = axes[i][j]
            ax.imshow(crop(image_for(args, out, tag, renderer), args.grid_crop),
                      interpolation="antialiased")
            # The image at the top of its cell, the caption in the line below.
            ax.set_anchor("N")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                unit = "depth" if args.grid_along == "depth" else "spp"
                ax.set_title(f"{unit} {value}", pad=3)
            if j == 0:
                ax.set_ylabel(RENDERER_LABELS.get(renderer, renderer),
                              rotation=0, ha="right", va="center", labelpad=6)
            cell = results["cells"][tag]
            if renderer == "pbrt":
                caption = f"{cell['pbrt_seconds']:.2f} s"
            else:
                c = cell[renderer]
                caption = f"{c['seconds']:.2f} s, {c['speedup']:.1f}×"
            ax.text(0.5, -0.03, caption, transform=ax.transAxes, ha="center",
                    va="top", fontsize=7)
    stem = f"{args.plots}/{scene_name}-grid-{args.channel}"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    say(f"wrote {stem}.pdf")


#===------------------------------------------------------------------------===#

def main(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="a .pbrt scene")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--spps", type=int, nargs="+", default=DEFAULT_SPPS)
    parser.add_argument("--schedules", nargs="+", default=DEFAULT_SCHEDULES)
    parser.add_argument("--repeats", type=int, default=3,
                        help="best of this many runs, both sides")
    parser.add_argument("--long-spp", type=int, default=0,
                        help="from this sample count up a cell is the best of "
                             "--long-repeats runs instead -- a shortcut for a "
                             "first look, since a render that long varies "
                             "little between runs and repeating it costs "
                             "hours; the default, 0, keeps --repeats "
                             "everywhere, which a benchmark needs")
    parser.add_argument("--long-repeats", type=int, default=1,
                        help="best of this many runs from --long-spp up")
    parser.add_argument("--out", default=os.path.join(ROOT, "eval", "out"))
    parser.add_argument("--plots", default=os.path.join(ROOT, "eval", "plots"))
    parser.add_argument("--pbrt", default=os.environ.get(
        "PBRT", os.path.expanduser("~/projects/pbrt-v4/build/pbrt")))
    parser.add_argument("--own-tree", action="store_true",
                        help="build this renderer's own BVH rather than take "
                             "pbrt's")
    parser.add_argument("--pbrt-gpu", action="store_true",
                        help="render every cell with `pbrt --gpu` too, as a "
                             "second reference column: pbrt's wavefront "
                             "renderer on the GPU, which is its volpath "
                             "integrator whatever the scene names")
    parser.add_argument("--rerun", nargs="*", default=None, metavar="RENDERER",
                        help="render again, cached or not: every renderer at "
                             "every cell of this run when given alone, or "
                             "only the renderers named (pbrt, pbrt-gpu, or a "
                             "schedule) -- what to do after a change to the "
                             "compiler, which touches the schedules and not "
                             "pbrt, or to bring a cell measured once up to "
                             "--repeats runs")
    parser.add_argument("--images", nargs="*", default=["none"],
                        help="which renders to write as PNGs: none, all, or "
                             "cells as depth:spp or depth:spp:renderer")
    parser.add_argument("--channel", choices=["radiance", "normals"],
                        default="radiance", help="what the images show")
    parser.add_argument("--exposure", type=float, default=None,
                        help="viewing exposure for the images, in stops")
    parser.add_argument("--grid", action="store_true",
                        help="draw the image grid")
    parser.add_argument("--grid-rows", nargs="+", default=["pbrt", "packet"],
                        help="the renderers, one per row")
    parser.add_argument("--grid-along", choices=["depth", "spp"], default="depth",
                        help="what varies along the columns")
    parser.add_argument("--grid-at", type=int, default=None,
                        help="the other axis' value (default: its largest)")
    parser.add_argument("--grid-crop", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"), help="a crop, in pixels")
    parser.add_argument("--grid-width", type=float, default=7.0,
                        help="figure width in inches (7 is a two-column page)")
    args = parser.parse_args(argv[1:])
    if args.long_spp is not None and args.long_spp <= 0:
        args.long_spp = None
    if args.repeats < 1 or args.long_repeats < 1:
        raise SystemExit("--repeats and --long-repeats are at least 1")
    # What the table and the figures have a column for: the schedules, and
    # pbrt on the GPU when asked. The schedules alone are what is built.
    args.renderers = [*args.schedules, *(["pbrt-gpu"] if args.pbrt_gpu else [])]

    scene = os.path.abspath(args.scene)
    if not os.path.isfile(scene):
        raise SystemExit(f"no scene at {scene}")
    if not os.access(args.pbrt, os.X_OK):
        raise SystemExit(f"no pbrt at {args.pbrt}: set PBRT or pass --pbrt")
    # Everything the options can get wrong, before an hour of rendering.
    images = images_wanted(args)
    # The rows are checked only when a grid is drawn: the default rows name
    # the packet schedule, which a run over other schedules does not have and
    # does not need unless it draws the grid.
    for renderer in args.grid_rows if args.grid else []:
        if renderer not in ["pbrt", *args.renderers]:
            raise SystemExit(f"--grid-rows {renderer}: not pbrt or a renderer "
                             f"of this run ({args.renderers})")
    if args.grid_at is not None:
        axis = args.spps if args.grid_along == "depth" else args.depths
        if args.grid_at not in axis:
            raise SystemExit(f"--grid-at {args.grid_at}: not one of the "
                             f"{'spps' if args.grid_along == 'depth' else 'depths'} "
                             f"of this run ({axis})")
    scene_name = os.path.splitext(os.path.basename(scene))[0]
    out = os.path.join(os.path.abspath(args.out), scene_name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    results_path = f"{out}/results.json"
    results = json.load(open(results_path)) if os.path.isfile(results_path) else {}
    if args.rerun == []:
        results = {}
    elif args.rerun:
        # Only the renderers named, and only at this run's cells: a cached
        # cell of a wider grid is left as it was. pbrt's time is the one
        # every speedup is over, so rendering it again means the schedules
        # cached in the cell are stale too; they are kept, since a run that
        # does not name them does not print them, and their speedup is
        # recomputed from the new time where it is.
        for renderer in args.rerun:
            if renderer not in ["pbrt", *args.renderers]:
                raise SystemExit(f"--rerun {renderer}: not pbrt or a renderer "
                                 f"of this run ({args.renderers})")
        for d in args.depths:
            for p in args.spps:
                cell = results.get("cells", {}).get(f"d{d}-s{p}")
                if cell is None:
                    continue
                for renderer in args.rerun:
                    if renderer == "pbrt":
                        cell.pop("pbrt_seconds", None)
                        cell.pop("repeats", None)
                    else:
                        cell.pop(renderer, None)
    # A cell is done when pbrt and every schedule asked for have rendered it;
    # a schedule missing from a rendered cell is rendered into it.
    def done(tag):
        cell = results.get("cells", {}).get(tag)
        return cell is not None and "pbrt_seconds" in cell and \
            all(s in cell for s in args.renderers)
    missing = [f"d{d}-s{p}" for d in args.depths for p in args.spps
               if not done(f"d{d}-s{p}")]
    if missing:
        results["compile_seconds"] = build(out, args.schedules)
        say("compile seconds: " + ", ".join(
            f"{s} {t:.2f}" for s, t in results["compile_seconds"].items()))
        convert_scene(args, out, scene)
        measure(args, out, scene, results)
    else:
        say("every cell is rendered already (--rerun to render again)")

    table(args, results, f"{out}/results.tsv")
    heatmaps(args, results, scene_name)
    write_images(args, out, scene_name, images)
    if args.grid:
        image_grid(args, out, scene_name, results)

    # Over this run's grid only: the cache may hold cells of an earlier,
    # wider run that this run's schedules never rendered.
    failures = [(f"d{d}-s{p}", s) for d in args.depths for p in args.spps
                for s in args.renderers
                if results["cells"][f"d{d}-s{p}"][s]["failed"]]
    if failures:
        say("FAILED against pbrt: " + ", ".join(f"{t} {s}" for t, s in failures))
        return 1
    say("every image matches pbrt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
