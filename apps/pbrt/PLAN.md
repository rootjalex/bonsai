# apps/pbrt: where this is and what is next

The goal is a faithful PBRT-v4 clone written in bonsai, where the algorithm
says what to compute and the `schedule` block says how — and where "faithful"
means checked against pbrt rather than asserted.

Run `apps/pbrt/compare.sh <scene.pbrt>` to see where things stand. It renders
the scene both ways and reports disagreeing pixels, albedo difference, and
relative speed. `apps/pbrt/render.sh <scene.pbrt>` renders and writes PNGs to
look at.

Both scripts need clang — the generated header uses `ext_vector_type`, which
gcc has no equivalent of. They default to `clang++` on PATH; `BONSAI_CXX`
overrides. Do **not** use `CXX`: conda's compiler packages export it as their
gcc wrapper.

## Where it is

The renderer answers one question — what does the camera ray hit, and what
colour is it — and writes pbrt's gbuffer normals and albedo. There is no light
transport yet.

Seven scenes in `scenes/` all match pbrt at **0 disagreeing pixels**, with the
albedo agreeing to ~6e-5 mean. bonsai runs 1.1–1.65x faster than pbrt on them,
though that is not the same work: pbrt integrates a path where this returns the
nearest hit.

Verified against pbrt directly, not by inspection:

- the sampler streams (independent, stratified, halton) value for value,
- the spectral tables, by `scene_dump --check-tables`,
- the BVH, which is pbrt's SAH build over AABBs in pbrt's 32-byte node layout,
- the reference gbuffer itself, against the pbrt binary's EXR.

## What is next, in order

### 1. `coateddiffuse`, and materials generally

This is the only thing between here and `killeroo-simple.pbrt`, the smallest
real scene:

```
$ scene_dump ~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt out.txt
scene_dump: only the diffuse material is supported, scene asks for "coateddiffuse"
```

Everything else that scene needs now works — its 66,533 triangles load through
pbrt's Loop subdivider, its halton sampler is reproduced, and its `Film "rgb"`
no longer stops the comparison.

`coateddiffuse` is `LayeredBxDF<DielectricBxDF, DiffuseBxDF>`: a dielectric
coating over a diffuse base, whose reflectance is estimated by a stochastic
random walk *between* the layers, with Trowbridge–Reitz microfacets, dielectric
Fresnel and its own RNG. It is a real piece of work, not an afternoon.

Two things make it less daunting than it sounds. The gbuffer only needs `rho`,
not the full `Sample_f` used for transport. And the reference side already has
it — `scene_dump --reference` calls pbrt's own `bsdf.rho`, so the target values
are available for any material before any of it is written.

Note that normals compare exactly regardless of material, so a scene can be
brought up in two stages if that helps.

The refusal lives in `material_reflectance` in `scene_dump.cpp`; the reflectance
travels to the renderer as an RGB in `scene_io.h` and is fitted to a sigmoid by
the driver, which is where a layered BSDF would have to fit differently.

### 2. Pixel jitter

`render.bonsai` hardcodes the camera sample at the pixel centre and never calls
`get_pixel_2d`, and `compare.sh` passes `--disable-pixel-jitter` to match. So
the whole pixel-sampling half of every sampler is written, verified against
pbrt, and unexercised by any render.

Turning it on is small and buys a lot: silhouettes get antialiased, stratified
and halton start to differ from independent visibly, and the comparison gets
much stronger — right now no render depends on `get_pixel_2d` being right.

One thing to fix when doing it: pbrt *normalizes* the summed normal
(`GBufferFilm::GetImage`) where `render.bonsai` divides by the sample count.
Identical while every sample of a pixel traces the same ray; different as soon
as they do not.

### 3. Light transport

The point of the whole thing, and what turns a gbuffer into an image:

- lights in `scene_io.h` and `scene_dump.cpp` — `UniformInfiniteLight` first,
  since it needs no light sampling and so isolates the transport;
- `DiffuseBxDF::f` = R/π, with cosine-hemisphere sampling and its pdf;
- `SpawnRay` / `OffsetRayOrigin`. This is what `interval.bonsai` was written
  for and has never been used: the error bounds on a hit point decide how far
  to push the next ray off the surface, and getting it wrong looks like shadow
  acne rather than like a bug in the integrator;
- the random walk itself. pbrt's `LiRandomWalk` recurses but is bounded by
  `maxdepth`, so it is a `for depth in 0:maxdepth` carrying throughput and
  radiance — a sequential loop, which is what `for` is for.

### 4. Where the schedule stops paying

`many-shapes` runs at 1.14x while the others reach 1.65x. More primitives means
more traversal and less of everything else, so that scene is measuring the tree
the schedule built against pbrt's. That is the question this whole project
exists to ask, so it is worth pulling on whenever the renderer side is not the
more interesting thread.

`PBRT_TREE=1 compare.sh ...` takes pbrt's own BVH instead of building one, which
separates "whose builder found a better tree" from "whose traversal is faster".

## Known-open, smaller

- `cie_tables.h` and `rgb2spec_tables.h` are generated by
  `make_spectrum_tables.py` and are both committed, which is against the rule
  that generated files stay out of git. The catch is that the generator fetches
  pbrt's source over HTTP; there is a local checkout at `~/projects/pbrt-v4`, so
  teaching it to read from there and having `render.sh` run it is the fix.
- Two CUDA `bind` tests have been failing since before this work
  (`backends/cuda/parallel`, `backends/cuda/rtiow-primer`).
- `scenes/stratified.pbrt` and `scenes/halton.pbrt` share three-spheres'
  geometry on purpose — they isolate the sampler — but that does mean their
  normal images are identical to `three-spheres`, which is confusing until you
  know why.
- Sobol and zsobol samplers are refused. Three scenes in the collection use
  them; 81 of 87 use halton, which is implemented.
- PLY files holding quads are refused: pbrt makes those bilinear patches rather
  than pairs of triangles, and the renderer has no bilinear patch.
