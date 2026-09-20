# eval: the schedules against pbrt, over depth and sample count

`render_matrix.py` renders one scene at every (path depth, samples per pixel)
of a grid under each of the pbrt app's schedules and with pbrt itself, checks
every image against pbrt's, and draws the speedup heatmaps and, if asked, a
grid of the images for a paper.

    python3 eval/render_matrix.py ~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt \
        --depths 1 2 3 4 5 --spps 16 64 256

Run from the repository root, inside the `bonsai` conda environment (its
`bin` first on `PATH`; `BONSAI_CXX` pointing at its `clang++` if the shell has
another). Everything it writes goes under `eval/`:

    eval/plots/<scene>-speedup.pdf, .png      the heatmaps
    eval/plots/<scene>-grid-<channel>.pdf     the image grid (--grid)
    eval/plots/<scene>-d<depth>-s<spp>-<renderer>-<channel>.png   images asked for (--images)
    eval/out/<scene>/results.json, results.tsv                    every number
    eval/out/<scene>/                         the converted scenes, the renders, the built renderers

Both directories are generated and ignored by git.

## What is measured

For every cell the scene is converted once by `scene_dump` at that depth and
sample count, on pbrt's own BVH (`--own-tree` to build this renderer's
instead), so that what is timed is how each schedule traverses one tree
rather than whose builder found the better one. pbrt renders it, then each
schedule's renderer; both sides take the best of `--repeats` runs (default 3),
pbrt timed by its own render timer read out of the EXR it writes and this
renderer by its driver, around the render call alone, with the scene and the
tree built beforehand on both sides. The speedup is pbrt's time over the
schedule's. Every render uses all cores; pbrt's depth reaches it by rewriting
the scene's `Integrator` directive, since pbrt has no flag for it (the same
rewrite `apps/pbrt/compare.sh --maxdepth` does).

The schedules are the files in `apps/pbrt/schedules/` -- `scalar`, `perlane`
and `packet` by default, `--schedules` for a subset -- each compiled beside
`apps/pbrt/render.bonsai` as a second input; the compile is timed too and
printed, being part of what a schedule costs. The compiler is rebuilt first
(`cmake --build build`).

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
instead of the radiance -- without waiting for the renders. `--rerun` throws
the cache away, which is what to do after a change to the compiler or the
renderer; a cached cell says nothing about the current build.

## Options

    SCENE                       a .pbrt scene (positional)
    --depths D ...              path depths, default 1 2 3 4 5
    --spps N ...                samples per pixel, default 16 64 256
    --schedules S ...           default scalar perlane packet
    --repeats N                 best of N runs, both sides; default 3
    --own-tree                  build this renderer's BVH rather than take pbrt's
    --rerun                     render every cell again
    --pbrt PATH                 the pbrt binary (default $PBRT or ~/projects/pbrt-v4/build/pbrt;
                                imgtool is expected beside it, or $IMGTOOL)
    --out DIR, --plots DIR      where things go (default eval/out, eval/plots)

    --images none | all | CELL ...
                                which renders to write as PNGs: none (default),
                                all, or cells as depth:spp (every renderer) or
                                depth:spp:renderer, renderer one of pbrt or a
                                schedule
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

## What it needs

The `bonsai` conda environment with Python, numpy and matplotlib in it, which
were added as

    conda install -n bonsai --freeze-installed -c conda-forge python numpy matplotlib-base

(`matplotlib-base` rather than `matplotlib`, whose Qt front end would have
pulled a second LLVM into the environment beside the compiler's); a built
pbrt-v4 with `imgtool`; and the app's own requirements, which are the
compiler's (`apps/pbrt/PLAN.md`).
