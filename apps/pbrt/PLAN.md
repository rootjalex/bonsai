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

The seven scenes in `scenes/` and `killeroo-simple.pbrt` from the pbrt-v4-scenes
collection all match pbrt at **0 disagreeing pixels** on the normals. The albedo
agrees to ~6e-5 mean on every diffuse scene. On killeroo-simple, which is the
first scene with a `coateddiffuse` material, 97.3% of pixels are inside the
comparison's tolerance and the other 2.7% differ by Monte Carlo noise — see
"the last 2.7%" below, which is the open question.

Verified against pbrt directly, not by inspection:

- the sampler streams (independent, stratified, halton) value for value,
- the spectral tables, by `scene_dump --check-tables`,
- the BVH, which is pbrt's SAH build over AABBs in pbrt's 32-byte node layout,
- the `coateddiffuse` BSDF, by `scene_dump --print-bsdf` against the golden in
  `tests/bonsai/correctness/llvm/coated-diffuse.bonsai`,
- the shading geometry and the frame a BSDF is evaluated in, by
  `scene_dump --print-shading` against
  `tests/bonsai/correctness/cpp/shading-frame.bonsai`,
- the reference gbuffer itself, against the pbrt binary's EXR.

### What the last round added

`coateddiffuse` works. That is pbrt's `LayeredBxDF<DielectricBxDF, DiffuseBxDF>`
in `bxdf.bonsai`: a dielectric coating over a diffuse base whose reflectance has
no closed form and is estimated by a random walk between the two interfaces,
driven by an RNG seeded from a Murmur hash of the directions it was asked about.
Every piece of it — Trowbridge–Reitz, the Fresnel terms, `Refract`, the
cosine-hemisphere and visible-normal samplers, `FastExp`, the walk itself — is
checked value for value against pbrt.

Getting there needed the shading geometry, which nothing had needed before: the
gbuffer's normal is turned to face the camera, so its sign and the surface
tangent both drop out of an image comparison, and a diffuse material's
reflectance does not depend on the frame at all. A layered one depends on all of
it. So `shapes.bonsai` now computes pbrt's `SurfaceInteraction`: the
barycentrics, the partial derivatives from the texture coordinates, the
interpolated shading normal, and the tangent orthogonalized against it — for
triangles, and pbrt's full sphere parameterization for spheres.

Along the way, six things in the compiler:

- `sqr` had no lowering in the direct-to-LLVM backend, only in the SSA one;
- there was no `acos`, which a sphere's parameterization needs;
- a variant arm whose payload had a default field value produced C++ the driver
  could not declare a variable of, because the union's default constructor came
  out deleted;
- an extern array with no size aborted the compiler. The free-variable walk that
  decides which functions an extern reaches looks inside an array's size --
  `array[Float, n]` names `n`, and `n` may itself be free -- without checking
  that there was one. A parameter of the same type never reached that code,
  because a function's own parameters are bound before its body is walked, so
  only an extern could get there. `tests/bonsai/lower/externs-unsized.bonsai`;
- `LowerExterns` ran before `LowerGeometrics` in all four pipelines. A geometric
  op is not yet a call, so the free-variable walk cannot see through one, and
  `LowerGeometrics` builds its call from the implementation's declared
  parameters -- so an implementation that reads an extern got extra parameters
  first and was then called with too few. That is exactly what a triangle
  fetching its vertices from a shared mesh does, so the order had to swap;
- the address of an array element did not survive a call. `f(materials[i],
  g(x))` splits the block at `g`, so the address the outer call is passed has to
  live across it, and the SSA builder threads it into the continuation as a
  block argument. Both places that emit a jump bind nothing when a value is
  passed under its own name -- the name is already there -- and an address has
  no name: `SSA/CodeGen_Stmt.cpp` skips GEP and FieldPtr when it emits a block's
  instructions and writes them out at each use instead. So the continuation
  named something nothing defined, and code generation failed on an undefined
  variable rather than on anything a reader could see.
  `tests/bonsai/ssa/address-across-call.bonsai`.

## What is next, in order

### 1. The last 2.7%, and what it says about matching a build

On killeroo-simple, 2.7% of pixels have an albedo that differs from pbrt's by
more than the comparison allows. They are all on the two `coateddiffuse`
killeroos, and the difference is not bias: over those pixels the ratio of the
two renders has a median of 0.98, a tenth percentile of 0.69 and a ninetieth of
1.45. It is noise. Two random walks that start from directions differing in the
last bit take completely different paths, and sixteen samples is nowhere near
enough for the two to converge to the same number.

So the whole question is whether the direction handed to the BSDF is bit for bit
pbrt's, and the obstacle is **fused multiply-adds**. pbrt is built with gcc and
contraction on, so `a*b + c` there rounds once where the same expression rounds
twice here — bonsai's SSA form has already split it into two instructions by the
time a C++ compiler sees it, so nothing can fuse it. Five places had to be
written as an explicit `fma` before 97.3% of the pixels agreed:

- the shear in the watertight triangle test (`shapes.bonsai`),
- the interpolation of the vertex normals,
- the dot product a frame changes coordinates with (`dot_` in
  `stdlib/numerics.bonsai`),
- the length a vector is normalized by (`sqlen_`, `unit_vector_`),
- the rows of the camera's matrix–vector product (`camera.bonsai`).

This is not fast maths and it is not gcc doing something it should not. pbrt's
build has no `-ffast-math` and no `-ffp-contract` flag, so it gets gcc's default
of `-ffp-contract=fast`: multiply-adds are fused, and nothing else is relaxed —
no reassociation, no assumptions about NaN or signed zero. Fusing is also the
*more* accurate of the two, one rounding instead of two, so pbrt's answer is the
better one and the difference is ours to close rather than theirs.

Where pbrt *is* explicit, this app already is too. `FMA` appears about forty
times in pbrt, all of it inside `util/`: `DifferenceOfProducts`,
`SumOfProducts`, `EvaluatePolynomial`, `FastExp`, and `Dot(Vector3, Normal3)`.
Those are `prod_diff`, `cross_` and `fast_exp` here, written out. None of the
five above is one of them — each is plain source that gcc contracts. The
sharpest illustration is that `Dot(Vector3, Normal3)` is an explicit
`FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z))` while
`Dot(Vector3, Vector3)` is `v.x*w.x + v.y*w.y + v.z*w.z`: the same operation,
two different roundings, chosen by overload resolution.

What cannot be read off pbrt's source is which multiply pairs with which add.
`(a*b + c*d) + e*f` has two legal contractions of the inner add — fuse `a*b` and
keep `c*d` as the addend, or the other way round — and they differ in the last
bit. gcc does not choose the same one every time: `Sqr(x)+Sqr(y)+Sqr(z)` fuses
the first product and `b0*n0 + b1*n1 + b2*n2` fuses the second. Each of the five
above was found by printing pbrt's answer and testing the candidates against it,
not by reasoning.

Three ways forward, in increasing order of how much they change:

- **Keep going by hand.** The harness works and each round is cheap: print the
  intermediates from pbrt, compare, find the one that differs. The remaining
  divergence is somewhere in the walk itself or in the sphere, and the same
  method will find it. It is also unbounded — there is no list of the places
  gcc fuses.
- **Contract in the compiler.** A pass that fuses every multiply-add would
  match gcc's `-ffp-contract=fast` closely, and would make the whole app
  faithful at once rather than one expression at a time. It changes every
  bonsai program's arithmetic, so it wants to be opt-in, and the goldens that
  already match pbrt would have to be re-checked rather than re-blessed.
- **Decide the comparison is asking the wrong question.** A layered material's
  albedo is a sixteen-sample estimate. Two correct implementations agree on its
  expectation and not on its value, and `compare_gbuffer.py` currently has no
  way to say that. Turning pixel jitter on (below) would average the estimate
  over the pixel's samples and shrink the difference, but not to zero: at 256
  samples the two means would still differ by around a per cent.

The first is what the last round did and it got from 11.6% to 2.7%. The second
is the one that would finish it.

### 2. Where the time went

Best of three on each side, against a pbrt that is integrating a whole path
where this returns the nearest hit and its reflectance:

    three-spheres      1.82x faster      halton             1.57x faster
    nested-transforms  1.66x faster      killeroo-simple    1.30x faster
    wide-fov           1.65x faster      many-shapes        1.26x faster
    stratified         1.61x faster      coloured           1.26x faster

Two things got it there, and both were bugs rather than tuning.

**Three mallocs per camera ray, never freed.** The sixteen fixed sample points
`rho` is estimated with were written where they are used -- three local arrays
in `visible_surface`. A local array is *allocated* where it is declared, on the
heap, so every camera ray that hit anything allocated 192 bytes and leaked them.
Three-spheres peaked at 4.18GB of resident memory and spent a fifth of the
render on it, on a scene with no layered material in it at all. They are externs
now, built once by the driver, and there is not a single `malloc` in the
generated code. 4.18GB became 182MB and 265ms became 210ms.

Worth knowing for next time, because the symptom pointed the wrong way: removing
the call to `coated_rho` made a scene that never executes it 20% faster, which
reads exactly like a function being inlined into a hot loop. It was not --
`material_rho`, `coated_rho` and `coated_sample_f` are all separate functions in
the generated LLVM and none of them is inlined. What removing the call did was
make the sample arrays dead, and with them the allocation.

**The generated code was built for a generic x86-64.** `make_target_machine`
took the host's triple and then passed an empty CPU and an empty feature string,
so LLVM targeted SSE2 and nothing after it: no AVX, no 256-bit vectors, and an
`fma` lowered to a call into libm rather than a `vfmadd` instruction. pbrt is
built with `-march=native`. Naming the host CPU and its features when neither
`--triple` nor `--mcpu` was given -- the goldens that diff generated code pin
both, so they are unaffected -- took three-spheres from 210ms to 155ms and put
193 `vfmadd`s in the object where there had been none.

That change also surfaced a **miscompile**: a dense vector load or store was
given the *vector's* ABI alignment, so `a[i:i+8]` inside an array of floats
claimed to be 32-byte aligned when nothing had aligned it. A generic target split
every such access into 16-byte pieces and got away with it; with AVX enabled
LLVM emitted one aligned 32-byte move and five `vectorize` tests segfaulted. The
element's alignment is what an array actually guarantees, and it is what the
gather and scatter paths beside it had always used.

The honest remaining cost: the renderer now does pbrt's work. `normal_at` used
to answer a sphere's normal as `(p - c) / r`, where pbrt computes the
parameterization, its two partial derivatives, an `acos` and a `sin`, and takes
the normal from their cross product. Measured on three-spheres, about 9ms of
265. That one we should keep paying.

A local array being a heap allocation is a compiler question rather than an app
one, and this app cannot be the only thing it bites. `Allocate::make` defaults
to `Heap`, nothing promotes a heap allocation to the stack, and
`CodeGen_LLVM.h`'s "TODO: support deallocation" means nothing is ever freed --
so any local aggregate in a hot path is an unbounded leak.

So the renderer is now built with **`--no-heap`**, which refuses to emit one:
every heap allocation the backend makes goes through `create_malloc`, and that
is where the refusal lives. It costs nothing to hold to -- the generated code
has no `malloc` in it, and the only storage it takes is fixed-size stack slots,
including the `[64 x i32]` that `trace.loopify(64)` makes and that is pbrt's
`nodesToVisit[64]` under another name -- and it turns "the inner loop does not
allocate" from something to hope for into something the compiler checks. It
rejects the three-constant-tables program outright.

What the flag costs elsewhere is the two things that genuinely need a heap: a
`dyn_array`, and a `map` or reduction whose result is *returned* rather than
written into storage the caller owns (`Lower/Maps.cpp`: "On the heap because it
is returned"). Neither is used here. Turning the second into destination-passing
is what would let the flag be the default rather than an opt-in, and
`ReturnToOutParameter` already does that transformation for exported functions.

The obvious fix for the third case was tried and backed out, and why is worth
writing down. Making
a `Build` of an array of constants into a private read-only global -- which is
what it is -- removes the allocation, and does miscompile `backends/llvm/
mut-basic`: `e[0] = {1, i32x2{2, 3}}` stores that array into a *mutable* field,
and `write_y` then writes through it, so every `Example` would have shared one
read-only array. The reason is that an array inside a struct is stored as a
pointer, so an array value assigned into a struct has reference semantics, and
whether it is later written is not visible at code generation. The fix has to
be an IR pass, where mutability still is; the two candidates are a constant
array becoming a module constant when nothing writes it, and a heap allocation
whose pointer does not escape becoming a stack one -- `promote_allocas` already
has the escape walk, and `create_alloca_at_entry` already hoists, so the second
is mostly assembly of parts that exist.

**A primitive was 208 bytes where pbrt's is 8. It is now 64.** Measured:

    bonsai (was)  Primitive 208 = Shape 144 + Material 64
                  Shape     144 = tag + max(Sphere 32, Triangle 128)
    bonsai (now)  Primitive  64 = Shape  48 + a material index
                  Shape      48 = tag + max(Sphere 32, Triangle 8)
    pbrt          Primitive   8   (TaggedPointer, what the BVH array holds)
                  GeometricPrimitive 48, Triangle 8, Sphere 48, apart

Three things made up the 208. A triangle carried its own copy of three
positions, three normals and three texture coordinates. The material was
*copied* into every primitive rather than referred to — 64 bytes across 66,533
triangles for the four distinct materials killeroo-simple has. And the ADT
lowering is a tag beside a union sized to the largest variant
(`Lower/ADTLayout.cpp`, `default_adt_layout`), so every sphere paid for the
triangle beside it in the variant.

The first two are fixed, by doing what pbrt does. Its `Triangle` is
`{int meshIndex, triIndex}` and `Triangle::allMeshes` is a global list of
`TriangleMesh` it indexes into; here the meshes' arrays are laid end to end in
`mesh_positions`, `mesh_normals`, `mesh_uvs` and `mesh_indices`, and a
`TriangleMesh` says where its own run of each begins. Its material is a pointer
because a scene has a handful of materials and thousands of shapes; here it is
an index into `materials`. All of them are externs, and none of them had to be
threaded by hand: `LowerExterns` puts a declaration into exactly the functions
that read it, which is how the CIE tables already reach `spectrum_to_rgb`.

What it bought: many-shapes went from 331ms to about 308ms, and three-spheres
did not move, which is right — it has five primitives, so the size of one cannot
matter. Killeroo-simple is unchanged within noise. Every scene produces the same
image it did before, to the last digit of the albedo comparison, which is what
says this was a change of layout and not of content.

It also stops paying for the vector ABI three times over. A `vec3f` is sixteen
bytes rather than twelve — the deliberate cost of matching LLVM's `<3 x float>`,
see the note in `src/CodeGen/CPP.cpp` — and that used to be paid per corner of
every triangle. It is now paid once per vertex, which two triangles share.

**What is left of the 64 is the tagged pointer.** `Shape`, `Material`,
`Primitive` and `BxDF` are all `TaggedPointer` in pbrt: a single `uint64_t` with
the pointer in the low 57 bits and a seven-bit type tag above it
(`util/taggedptr.h`), and the variant allocated elsewhere. A tagged union is a
different thing — the tag sits beside the payload and the whole value is as
large as the largest arm — so a `Shape` here is 48 bytes where pbrt's is 8, and
40 of those are the sphere the triangle no longer needs.

`ADTLayout`'s header already said choosing a layout and applying one are
separate "so that a schedule can eventually choose differently", and a schedule
can now say which:

    schedule {
        layout Shape = tagged_index;
    }

`inline` is what every ADT has always got and still gets by default. It lives in
the schedule rather than beside the `element` for the same reason a tree's node
layout does -- it changes how much memory a traversal streams and nothing a
program can observe -- and being per-target matters here: an index travels to a
device and a pointer does not, so the same program can want different answers on
a CPU and a GPU.

`tagged_index` is built. A value of the type is one `u64` with the variant's tag
in the top byte and, in the rest, an index into a pool per variant:

    Shape = u64
    match  ->  (s >> 56) == 0, then Shape_Sph_pool[s & 0x00ffffffffffffff]
    Sph(..) ->  i = atomic_add(&Shape_Sph_fill[0], 1)
                Shape_Sph_pool[i] = Sph{..}; return (0 << 56) | i

The pools are externs nobody declares -- `Lower/ADTs.cpp` declares them -- and
they arrive as parameters of exactly the functions that reach them, which is
what the extern machinery already does for everything else. That is the part
that makes this work without an allocator: the memory and the capacity are the
caller's, so building a variant is a store into a buffer the driver sized. A
function that only reads handles takes the pools and not the counters.

The eight bytes are pbrt's eight bytes, with a byte of tag where pbrt has seven
bits at bit 57 -- pbrt needs the low 57 to stay a usable pointer and an index is
under no such obligation. What it costs is the indirection, which pbrt pays too:
its eight bytes cost three dependent loads to reach a triangle's vertices. The
`tagged_ptr` is parsed, reaches `adt_layout()`, and stops there with an
unimplemented error naming what is missing, rather than quietly falling back to
the default and leaving a schedule that changed nothing and said so nowhere.

### `Shape` uses it, and it bought nothing measurable

`render.bonsai` asks for `layout Shape = tagged_index` and `render_hook.cpp`
owns the pools. `Primitive` is not a variant type and so has no layout of its
own, but it is a `Shape` and a `u32`, so it shrank with the Shape: **64 bytes to
16**. Every rendered pixel is identical -- 0 disagreeing normals on all eight
scenes, worst normal difference unchanged at 1.10e-05 on the same pixel.

The render time did not move:

    killeroo-simple  700x700, 66,533 tris   10629.7 ms -> 10612.1 ms
    many-shapes      3200x2400, 56 shapes     218.5 ms ->   218.3 ms

Both differences are under a quarter of a per cent, on best-of-5 and best-of-7
respectively. Permuting the pools into leaf order as well (`compact_pools`)
measured 10638.4 ms, which is the same number again.

**Neither scene can measure this, and that is the finding.** killeroo-simple is
not traversal-bound. It is 256 samples per pixel, so 10.6 seconds buys 125.4
million camera rays at about 85 nanoseconds each -- and a descent through a BVH
over 66,533 triangles is a fraction of that. The rest is `material_rho`, a
layered BSDF walked at sixteen sample points for every hit, evaluated once per
sample. How big a primitive is barely matters to a render that spends its time
shading. (That the estimate is identical for all 256 samples of a pixel, because
the jitter is off, is item 3 below.) And many-shapes is traversal-bound but
far too small: 56 primitives is 3.6 KB at 64 bytes each and 900 bytes at 16, and
both fit in L1. The seven per cent the earlier 208-to-64 change won here was a
cache boundary being crossed -- 11.6 KB did not fit alongside the nodes -- and
there is no second boundary below it to cross.

What would measure it is a scene that is both traversal-bound and large enough
that the primitives miss cache: a few hundred thousand triangles with cheap
shading. Until one is in the comparison, the case for this layout is that it is
what pbrt stores and that it is four times smaller, not that it is faster.

`compact_pools` is kept on the same terms. The BVH build reorders primitives so
that a leaf names a contiguous run; under the default layout that moves the
shapes themselves, and under this one it moves only the handles, leaving a
leaf's four triangles scattered across the pool wherever the scene file put
them. Permuting the pools to match restores what the reorder was for. It is
unmeasured -- nothing here misses cache either way -- but without it the layout
carries a scattered-access problem that the scene which finally measures this
would run straight into.

`Material` still gets the default. It is held by index already
(`materials[prim.material]`), so there is much less to win.

And a boxed variant has to be allocated. pbrt carries a per-thread
`ScratchBuffer` through its integrator and resets it every sample, and the
reason is exactly this: `BSDF` holds a `TaggedPointer` to a `BxDF`, so
`GetBSDF` has to `scratchBuffer.Alloc<CoatedDiffuseBxDF>` at every
intersection. A bonsai BxDF is a value, so this renderer needs no arena at
all -- the generated code contains no `malloc`, and the only storage it takes
is fixed-size stack slots, including the `[64 x i32]` that `trace.loopify(64)`
makes and that is pbrt's `nodesToVisit[64]` under another name. Boxing by
default would import the allocator that pbrt needs and this does not.

That is why `layout X = tagged_ptr` under `--no-heap` is refused rather than
merely warned about: it asks for a representation and forbids the only way to
build one. `compare.sh` and `render.sh` both pass `--no-heap`, so the two would
meet the first time this renderer asked for the layout it is imitating.

### 3. Pixel jitter

`render.bonsai` hardcodes the camera sample at the pixel centre and never calls
`get_pixel_2d`, and `compare.sh` passes `--disable-pixel-jitter` to match. So
the whole pixel-sampling half of every sampler is written, verified against
pbrt, and unexercised by any render.

Turning it on is small and buys a lot: silhouettes get antialiased, stratified
and halton start to differ from independent visibly, the comparison gets much
stronger — right now no render depends on `get_pixel_2d` being right — and a
layered material's albedo starts to average over the pixel's samples instead of
evaluating the same estimate 256 times.

One thing to fix when doing it: pbrt *normalizes* the summed normal
(`GBufferFilm::GetImage`) where `render.bonsai` divides by the sample count.
Identical while every sample of a pixel traces the same ray; different as soon
as they do not.

### 4. Light transport

The point of the whole thing, and what turns a gbuffer into an image:

- lights in `scene_io.h` and `scene_dump.cpp` — `UniformInfiniteLight` first,
  since it needs no light sampling and so isolates the transport;
- the `f` and `PDF` halves of every BxDF in `bxdf.bonsai`, which are deliberately
  absent: `rho` only needs `Sample_f`, and writing the others now would mean
  shipping code no comparison against pbrt could check. `LayeredBxDF::f` is the
  large one, and it needs `TransportMode` threaded through — `rho` is only ever
  asked in radiance mode, so the dielectric's non-symmetry factor is currently
  spelled out rather than switched on;
- `SpawnRay` / `OffsetRayOrigin`. This is what `interval.bonsai` was written
  for and has never been used: the error bounds on a hit point decide how far
  to push the next ray off the surface, and getting it wrong looks like shadow
  acne rather than like a bug in the integrator;
- the random walk itself. pbrt's `LiRandomWalk` recurses but is bounded by
  `maxdepth`, so it is a `for depth in 0:maxdepth` carrying throughput and
  radiance — a sequential loop, which is what `for` is for.

## Known-open, smaller

- `cie_tables.h` and `rgb2spec_tables.h` are generated by
  `make_spectrum_tables.py` and are both committed, which is against the rule
  that generated files stay out of git. The catch is that the generator fetches
  pbrt's source over HTTP; there is a local checkout at `~/projects/pbrt-v4`, so
  teaching it to read from there and having `render.sh` run it is the fix.
- Two CUDA `bind` tests have been failing since before this work
  (`backends/cuda/parallel`, `backends/cuda/rtiow-primer`).
- A `Ramp` threaded into a block argument would hit the same hole the address of
  an array element did (see the sixth compiler fix above) and cannot be closed
  the same way: binding a Ramp to a name is what makes a dense access look like
  a gather. Nothing reaches it today, so it is a note rather than a bug.
- `scenes/stratified.pbrt` and `scenes/halton.pbrt` share three-spheres'
  geometry on purpose — they isolate the sampler — but that does mean their
  normal images are identical to `three-spheres`, which is confusing until you
  know why.
- Sobol and zsobol samplers are refused. Three scenes in the collection use
  them; 81 of 87 use halton, which is implemented.
- PLY files holding quads are refused: pbrt makes those bilinear patches rather
  than pairs of triangles, and the renderer has no bilinear patch.
- A mesh with per-vertex tangents (`S`) is refused: its shading tangent is not
  the one the texture coordinates give. Nothing produces one yet — LoopSubdivide
  does not and the PLY reader is not asked for them — so this is a guard rather
  than a gap.
- `coateddiffuse` with a spectral `eta` is refused, because a spectral index
  terminates the secondary wavelengths and nothing here does that. A displacement
  or normal map is refused for the same kind of reason: it replaces the shading
  frame, which here comes from the geometry alone.
