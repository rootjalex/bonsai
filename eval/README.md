# eval: the schedules against pbrt, over depth and sample count

`render_matrix.py` renders one scene at every (path depth, samples per pixel)
of a grid under each of the pbrt app's schedules and with pbrt itself, checks
every image against pbrt's, and draws the speedup heatmaps and, if asked, a
grid of the images for a paper. `summary.py` then puts several scenes' results
on one figure (see "Several scenes on one figure").

    python3 eval/render_matrix.py ~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt
    python3 eval/render_matrix.py ~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt \
        --depths 1 2 3 4 5 --spps 16 64 256 --schedules scalar packet
    python3 eval/render_matrix.py ~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt \
        ~/projects/pbrt-v4-scenes/book/book.pbrt ~/projects/pbrt-v4-scenes/pavilion/pavilion-day.pbrt

Run from the repository root, inside the `bonsai` conda environment (its
`bin` first on `PATH`; `BONSAI_CXX` pointing at its `clang++` if the shell has
another). Several scenes may be named: the renderers are built once and the
scenes measured in turn, each with its own results and figures. Everything it
writes goes under `eval/`:

    eval/plots/<scene>-speedup.pdf, .png      the heatmaps
    eval/plots/<scene>-grid-<channel>.pdf     the image grid (--grid)
    eval/plots/<scene>-d<depth>-s<spp>-<renderer>-<channel>.png   images asked for (--images)
    eval/out/<scene>/results.json, results.tsv                    every number
    eval/out/<scene>/                         the converted scene and the renders
    eval/out/_build/                          the built converter and renderers

Both directories are generated and ignored by git.

## What is measured

The scene is converted once by `scene_dump`, on pbrt's own BVH (`--own-tree`
to build this renderer's instead), so that what is timed is how each schedule
traverses one tree rather than whose builder found the better one. The
conversion is kept between runs while nothing it depends on has changed: the
dump is used again if it is newer than every file under the scene's
directory, than the converter's sources in `apps/pbrt/`, and than the pbrt
binary, and was made with the same tree flag (the pavilion's conversion is
four minutes, the zero-day frame's five; remove `scene.txt` to force one).
pbrt then
renders every cell it has not rendered -- one process per render, since pbrt
has no other way -- and each schedule's renderer runs once over every cell it
is missing from, loading the converted scene once and rendering it at each
cell's depth and sample count (the driver's `--cells d1-s16,d5-s1024,...`;
`--spp` and `--maxdepth` do the same for one render). The pavilion's
converted scene is 1.7 GB and takes twenty seconds to read, so converting and
reading it per cell and per schedule, as the first version of this tool did,
was an hour and forty gigabytes of disk per scene. Both sides take the best of
`--repeats` runs (default 3) at every cell, which is what a benchmark needs;
`--long-spp N` is the shortcut for a first look, taking the best of
`--long-repeats` runs (default 1) from N samples per pixel up, since a render
of hundreds of samples per pixel varies little between runs and three of each
cost hours (the 2026-09-21 sweep used `--long-spp 128`, and its figures say
so). pbrt is timed by its own render timer read out of the EXR it writes and this
renderer by its driver, around the render call alone, with the scene and the
tree built beforehand on both sides. The speedup is pbrt's time over the
schedule's. Every render uses all cores; pbrt's depth reaches it by rewriting
the scene's `Integrator` directive, since pbrt has no flag for it (the same
rewrite `apps/pbrt/compare.sh --maxdepth` does).

The schedules are the files in `apps/pbrt/schedules/`, five by default and
`--schedules` for a subset: the sample loop run three ways -- `scalar`,
`perlane` (a gang of samples, each lane walking the tree on its own) and
`packet` (the gang walking the tree together) -- and the queue run two ways,
`wavefront-perlane` and `wavefront` (packet), whose gangs are drawn from a
per-pixel queue of paths and compacted between bounces. `gpu` is the sixth,
not in the default set: the pixel loop on the GPU's blocks and the sample
loop on their threads, one kernel per render, its buffers staged to the
device before the timer and its film fetched after (it needs the GPU and
CUDA's libdevice, as `-b ptx` does). Each is compiled beside
`apps/pbrt/render.bonsai` as a second input; the compile is timed too and
printed, being part of what a schedule costs. A GPU schedule is compiled
with the compiler's `--fast-math` (nvcc's `--use_fast_math`: flushed
denormals, approximate division, square root and transcendentals) and a cap
of 128 registers per thread, because that is how pbrt's GPU build is
compiled (`--use_fast_math -maxrregcount 128`) and `pbrt --gpu` is what it
is measured against; a CPU schedule is compiled exact, as pbrt's CPU build
is. `--fast-math on|off` overrides the first for every schedule,
`--gpu-max-registers N` sets the cap (0 for none), and each schedule's
flags are recorded beside its numbers in `results.json`. The renderers do not depend on
the scene, so they are built once per run, into `eval/out/_build/`, however
many scenes the run names (the compact packet's compile is two minutes). The
compiler is rebuilt first (`cmake --build build`, or the directory
`BONSAI_BUILD_DIR` names). The
default grid is depths 1 to 5 by 16, 32, 64, ..., 1024 samples per pixel: 35
cells, each rendered by pbrt and the five schedules, which is an hour for a
small scene and an afternoon for a large one, so start it and leave the
machine alone.

`--pbrt-gpu` adds a second reference: `pbrt --gpu`, pbrt's wavefront renderer
on the GPU, rendered at every cell the same way (one process per render,
best of the repeats, timed by pbrt's render timer, which starts after the
scene is on the device) and carried in the table and the heatmaps as a
column with its speedup over pbrt's CPU render. It is pbrt's *volpath*
integrator whatever the scene names, since pbrt has no other GPU integrator:
on a scene without media that converges to the same image as `path`, so it
is checked against pbrt's CPU image like the schedules are, but its time is
the time of a different integrator, and a comparison of the `gpu` schedule
against it means something only once this renderer has volpath too. The
column's label says so.

Run it with nothing else on the machine. Every source of noise adds time, and
the minimum of three runs removes only some of it.

## Correctness

Every render is checked against pbrt's the way `apps/pbrt/compare.sh` checks
one, with `compare_gbuffer.py`'s tolerances so that there is one definition:
the pixels that received any light must be the same set (to 0.1%), the mean
over the image must agree to 2%, and the fraction of lit pixels agreeing to
1e-3 relative is reported. Two random walks cannot promise more per pixel --
a last-bit difference reseeds a path and the pixel values are unrelated,
though the estimate they converge to is the same -- so the per-pixel figure
falls with depth and is a reading, not a test. The schedules' images are also
compared with each other bit for bit and the result recorded (per-lane and
packet differ from scalar in the last bit where vectorized arithmetic
contracts differently; the two vectorized schedules should agree exactly). A
cell that fails is marked with a dagger in the heatmap, listed at the end,
and makes the exit code non-zero.

## Caching

A cell that has been rendered is not rendered again: the numbers live in
`eval/out/<scene>/results.json` and the plots are redrawn from them, so the
figures can be reworked -- another crop, another row order, the normals
instead of the radiance -- without waiting for the renders. The cache is per
renderer within a cell: a schedule named that a cached cell lacks -- a new
file in `schedules/` -- is rendered into the grid on its own, against the
cached pbrt time and image, without rendering the others again, which is how
a new schedule joins figures that took an afternoon. `--rerun` alone throws
the cache away; `--rerun wavefront gpu` renders only the renderers named
again, at this run's cells, and keeps the rest -- which is what to do after
a change to the compiler or the renderer, since a cached schedule cell says
nothing about the current build while pbrt's cells say what they always
did (and `--rerun pbrt` brings a cell measured once up to `--repeats`).

## Options

    SCENE ...                   one or more .pbrt scenes (positional)
    --depths D ...              path depths, default 1 2 3 4 5
    --spps N ...                samples per pixel, default 16 32 64 128 256 512 1024
    --schedules S ...           default scalar perlane packet wavefront-perlane wavefront
    --repeats N                 best of N runs, both sides; default 3
    --long-spp N                from N samples per pixel up, best of --long-repeats
                                runs instead; default 0, --repeats throughout
    --long-repeats N            default 1
    --own-tree                  build this renderer's BVH rather than take pbrt's
    --fast-math auto|on|off     the compiler's --fast-math: auto (default) for the
                                GPU schedules alone, since pbrt's GPU build is
                                nvcc's --use_fast_math and its CPU build is exact;
                                recorded per schedule in results.json
    --gpu-max-registers N       cap a GPU schedule's registers per thread; default
                                128, as pbrt's GPU build is built; 0 for ptxas's
                                own choice
    --pbrt-gpu                  render every cell with `pbrt --gpu` too (see above)
    --rerun [RENDERER ...]      render again: every renderer at every cell, or
                                only the ones named (pbrt, pbrt-gpu, a schedule)
    --pbrt PATH                 the pbrt binary (default $PBRT or ~/projects/pbrt-v4/build/pbrt;
                                imgtool is expected beside it, or $IMGTOOL)
    --out DIR, --plots DIR      where things go (default eval/out, eval/plots)

    --images none | all | 5:64 3:16:packet ...
                                which renders to write as PNGs: none (default),
                                all, or cells spelled depth:spp (every renderer
                                at that cell) or depth:spp:renderer, the
                                renderer pbrt or a schedule. Checked before
                                anything is rendered.
    --channel radiance|normals  what the images show; normals need a gbuffer
                                film and the path integrator, since pbrt writes
                                none otherwise
    --exposure STOPS            a viewing exposure for every image: a gain of
                                2**stops and a Reinhard roll-off, image_io.py's;
                                off, the encoding is pbrt's own quantisation.
                                Never compare two images shown this way

    --grid                      draw the image grid
    --grid-rows R ...           the renderers, one per row; default pbrt packet
    --grid-along depth|spp      what varies along the columns; default depth
    --grid-at N                 the other axis' value; default its largest
    --grid-crop X Y W H         show a crop, in pixels of the render
    --grid-width INCHES         figure width; default 7, a two-column page

## Figures

The heatmap is one panel per schedule, depth down and sample count across,
each cell its speedup over pbrt, all panels on one colour scale so they read
against each other. The grid is images edge to edge with a label per row and
column and the render time and speedup under each, and nothing else: white,
no frames, 8-point sans-serif. Both are written as PDF (Type 42 fonts, for a
paper) and PNG. For a figure of the scene at its own depth with the renderers
down the side and the sample counts across:

    python3 eval/render_matrix.py scene.pbrt --depths 5 --spps 16 64 256 \
        --grid --grid-rows pbrt scalar perlane packet --grid-along spp --grid-at 5

and for the gbuffer scenes' normals, `--channel normals`.

## Several scenes on one figure

    python3 eval/summary.py killeroo-simple book pavilion-day
    python3 eval/summary.py killeroo-simple book pavilion-day --depths 5

`summary.py` reads the `results.json` each named scene's `render_matrix.py`
run left under `eval/out/` -- it renders nothing -- and draws one row per
scene, in the order given (meant to be increasing complexity), and one panel
per depth, each panel the speedup over pbrt against the sample count with a
line per schedule and pbrt the dotted line at 1x; a cell that failed the check
against pbrt carries the same dagger as the heatmaps. It writes
`eval/plots/summary-speedup.pdf`, `.png` and `.tsv` (the numbers, one line per
scene, depth and count), or `summary-d<depths>-speedup` for a subset of the
depths, and `--name` for another stem. `--spps` and `--schedules` narrow it
the same way as `render_matrix.py`; every asked-for cell has to have been
rendered, since a plot with a hole would read as a measurement.

## What it needs

The `bonsai` conda environment with Python, numpy and matplotlib in it, which
were added as

    conda install -n bonsai --freeze-installed -c conda-forge python numpy matplotlib-base

(`matplotlib-base` rather than `matplotlib`, whose Qt front end would have
pulled a second LLVM into the environment beside the compiler's); a built
pbrt-v4 with `imgtool`; and the app's own requirements, which are the
compiler's (`apps/pbrt/PLAN.md`).
