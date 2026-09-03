# apps/pbrt: where this is and what is next

The goal is a faithful PBRT-v4 clone written in bonsai, where the algorithm
says what to compute and the `schedule` block says how — and where "faithful"
means checked against pbrt rather than asserted.

Run `apps/pbrt/compare.sh <scene.pbrt>` to see where things stand. It renders
the scene both ways and reports disagreeing pixels, albedo difference, and
relative speed. `apps/pbrt/render.sh <scene.pbrt>` renders and writes PNGs to
look at.

Both take `--spp <n>` and `--disable-pixel-jitter`, which are pbrt's own flags
reaching both sides by pbrt's own route: `scene_dump` sets them on `PBRTOptions`
before parsing, so every sampler's `Create` and pbrt's own `GetCameraSample`
read them and the scene handed to this renderer is converted under the same
overrides. The scenes in `scenes/` are sized for a comparison that has to
finish, so a picture worth looking at usually wants more samples than they ask
for:

    apps/pbrt/render.sh --spp 1024 apps/pbrt/scenes/area-light-mis.pbrt

Five of the twelve scenes have an emitter (`area-light`, `area-light-path`,
`area-light-mis` and the two `infinite-uniform` ones); the other six were
written when the comparison was a gbuffer test, and their radiance is correctly
black. For those, the images to open are the normals and the albedo.

Both scripts need clang — the generated header uses `ext_vector_type`, which
gcc has no equivalent of. They default to `clang++` on PATH; `BONSAI_CXX`
overrides. Do **not** use `CXX`: conda's compiler packages export it as their
gcc wrapper.

One thing to know before reading a number off a run at a different `--spp`. The
"pixels agreeing to 1e-3 relative" figure **gets worse as samples go up**, and
that is not a regression: a pixel agrees exactly when *none* of its samples took
a divergent branch in the layered walk, and the chance of that decays with the
sample count. The mean converges while the percentage falls. Compare that figure
only against runs at the same count. The albedo, which is an average rather than
a hit test, converges the way an estimate should — see item 1.

## Where it is

The renderer traces light. `randomwalk`, `simplepath` and `path` are all
implemented, area lights and uniform infinite lights are sampled, camera samples
are jittered within pbrt's Gaussian reconstruction filter, and the film records
pbrt's gbuffer normals and albedo beside the radiance — so a comparison can ask
three questions of the same render rather than one.

**killeroo-simple renders.** It names no integrator, so it gets `path`; it is
lit by a sphere of radius 3 seen from four hundred units away, which a random
walk found by chance in 5,038 of its 490,000 pixels and which light sampling
finds in 489,999 of them. Against pbrt running the same integrator over the same
samples, the two means agree to 0.99997x.

**Every scene in `scenes/` is compared, and every one of them matches pbrt on
the normals**: nine at 0 disagreeing pixels and many-shapes at one, which is on
a silhouette and is what the comparison tolerates by construction. That was not
true two rounds ago: six of the nine asked for `Integrator "path"`, which did
not exist, so `compare.sh` refused them and what was actually checked was three
scenes. It is ten now.

The albedo agrees to ~6e-5 mean on every diffuse scene. On the three scenes with
a `coateddiffuse` material it is 4.4e-04 at the 64 samples they ask for and it
**converges** — 2.9e-04 at 1024, where 5 pixels of 82,183 are outside the
comparison's tolerance and the check passes. That is item 1 answered; see below.

Radiance, against pbrt running the integrator the scene names over the same
samples of the same pixels:

    killeroo-simple          path        489,999 of 490,000 / 489,998  0.99997x
    area-light               randomwalk  58,476 both sides             1.00011x
    area-light-path          simplepath  82,108 / 82,105 here          1.00012x
    area-light-mis           path        81,993 both sides             1.00012x
    infinite-uniform         path        120,000 both sides            1.00012x
    infinite-uniform-simple  simplepath  120,000 both sides            1.00034x

It is also faster than pbrt on that same work: **1.43x on killeroo-simple**,
1.67x on area-light-mis, 1.50x on area-light-path, 1.68x on area-light, and
between 1.47x and 1.82x on the six gbuffer scenes. Killeroo's figure fell from
1.56x when it stopped being a random walk: `path` spends its time in shadow
rays, in `LayeredBxDF::PDF` and in MIS, and those are a different mix of work
from the one the schedule was tuned against. Both sides do the same work, so the
ratio is still the ratio. Item 5 below won 6.6% of it back by asking `any` for a
shadow ray instead of the nearest hit; item 4 is what is left to look at.

Verified against pbrt directly, not by inspection:

- the sampler streams (independent, stratified, halton) value for value,
- the spectral tables, by `scene_dump --check-tables`,
- the BVH, which is pbrt's SAH build over AABBs in pbrt's 32-byte node layout,
- the `coateddiffuse` BSDF — its `Sample_f`, its `f`, its `PDF` and its
  `Flags` — by `scene_dump --print-bsdf` against the golden in
  `tests/bonsai/correctness/llvm/coated-diffuse.bonsai`,
- the shading geometry and the frame a BSDF is evaluated in, by
  `scene_dump --print-shading` against
  `tests/bonsai/correctness/cpp/shading-frame.bonsai`,
- the reference gbuffer itself, against the pbrt binary's EXR.

### What the last round added

**Uniform infinite lights, and the machinery a light that is not a surface
needs.** `LightSource "infinite"` with no image behind it is pbrt's
`UniformInfiniteLight`: the same radiance from every direction. It unblocks no
real scene on its own — every infinite light in `pbrt-v4-scenes` has an image —
but it is the whole of the integrator side of that feature, and the image is
then a second arm rather than a second design.

Four things came with it, and the third is the one that would have been got
wrong:

- **A ray that hits nothing now carries radiance.** All three integrators had a
  branch that returned early there; each grows a loop over the scene's infinite
  lights, and each weighs it differently — the random walk sums with no weight,
  `simplepath` only counts it after a specular bounce, and `path` weighs it
  against the light's own sampling density with the power heuristic.
- **A light with no geometry to sample.** `SampleLi` draws over the whole sphere
  and the shadow ray it aims has nowhere to stop, so pbrt puts the sampled point
  two scene radii out. That means `Light::Preprocess` and a scene bounding
  sphere, computed in `scene_dump` from the same per-shape bounds a BVH build
  uses.
- **`allowIncompletePDF`**, which had been ignorable and stops being so here. It
  says the caller will find this light another way as well, so the light may
  decline to sample the parts that other way covers — and a uniform infinite
  light declines *entirely*, because a BSDF sample that escapes finds it for
  free. `path` passes true and `simplepath` passes false. Getting it wrong is
  not visible as an error: sampling the light *and* picking it up on escape
  counts the sky twice and looks like a stray scale factor. Both branches are
  checked against pbrt, by two scenes that differ only in the integrator line.
- **A light-list order that matches pbrt's.** pbrt builds area lights first, one
  per emissive shape, then everything else in declaration order. A uniform light
  sampler picks `lights[u * n]`, so two lists ordered differently hand the same
  random number to different lights.

`infinite-uniform.pbrt` and `infinite-uniform-simple.pbrt` are the same geometry
lit only by the sky, under `path` and under `simplepath`. Both light 120,000
pixels of 120,000 on each side and agree with pbrt to 1.00012x and 1.00034x on
the mean; the random walk was checked the same way and agrees to 1.00012x.

**Pixel jitter, which is pbrt's Gaussian reconstruction filter.** A camera
sample now lands where the filter puts it instead of at the pixel's centre, and
the film weighs it by what the filter says it is worth.

A Gaussian has no closed-form inverse, so pbrt does not sample it directly: it
tabulates the filter over a grid — 32 cells per unit of radius, so 48x48 at the
default 1.5 — and samples that as a piecewise-constant 2D distribution, a
marginal in y and a conditional row in x. `filter.bonsai` is that, and
`sampling.bonsai` gained the `PiecewiseConstant1D` and the `FindInterval` it
stands on. The table is the driver's, built once before the render as the Halton
digit permutations and the ADT pools are, and filled by an exported
`build_filter_table` so the Gaussian it tabulates is the renderer's own
`FastExp` polynomial rather than a second copy in C++.

Three things came with it:

- **The film stopped dividing by the sample count.** pbrt divides the colours by
  the *sum of the weights* and normalizes the summed normal rather than
  averaging it. Those were the same thing while every sample of a pixel traced
  one ray; they are not now, and `weight_out` is a film channel for the same
  reason pbrt's `Pixel::weightSum` is a member.
- **The camera sample carries the drawn lens and time**, which were fixed
  constants here while their draws were thrown away. Nothing reads them yet — a
  pinhole camera has no lens and no shutter — but pbrt hands them over and now
  so does this.
- **`--disable-pixel-jitter`** is pbrt's flag under pbrt's name, and it is worth
  keeping: with every sample of a pixel on one ray, a difference between the two
  images cannot be noise, so the gbuffer comparison becomes a question about the
  geometry alone. That is how the normals got to zero disagreeing pixels.

The result that mattered is under item 1: the albedo disagreement that had been
stuck at 1,287 pixels no matter how many samples were thrown at it now falls to
5 at 1024 samples.

**`path` became the fallback for a scene that names no integrator**, where the
random walk used to be. pbrt's default is `volpath`, which is `path` plus
participating media; a scene that named no integrator named no media either, so
`path` is the closer of the two by a long way. It is one scene in the collection
and it is killeroo-simple, which is the difference between 5,038 lit pixels and
a photograph. Both sides resolve the fallback from the same value now — the
reference render is handed the tag `load` decided rather than the name the scene
wrote, because deciding it twice is how the two would come to run different
algorithms and report it as a disagreement about transport.

### What the round before that added

**`path`, which is pbrt's own workhorse and what every scene here names.** It is
`simplepath` plus four ways of not wasting a sample, and all four are in:
multiple importance sampling between the light's estimator and the BSDF's,
Russian roulette on a throughput that has fallen away, the `etaScale` that stops
roulette reading a refraction's radiance compression as a path going dark, and
regularization — off by default, and a widening of a near-delta lobe after the
first non-specular bounce.

What it needed that nothing before it did:

- **`LayeredBxDF::PDF`**, a third random walk beside `Sample_f`'s and `f`'s. A
  layered BSDF reports a density only *proportional* to the one it sampled from
  — the entrance interface's rather than the whole stack's — which is fine for
  its own estimator and useless as an MIS weight, so a path tracer has to come
  back and ask. Two things about it cannot be read off pbrt's source: it seeds
  its RNG from `wi` and then `wo` where `f` seeds from `wo` and then `wi`, and it
  draws its entry samples as `Sample_f(w, r(), {r(), r()})` — a braced list,
  evaluated left to right, inside an argument list gcc evaluates right to left,
  so the pair comes out of the RNG *before* the single, which is the opposite of
  every other walk in the file. Both were measured rather than reasoned about,
  the second by compiling the three spellings and looking. All twelve of its
  values match pbrt to the six decimals the golden prints.
- **`BSDF::Flags`**, for the two questions a path integrator asks before it
  spends anything: whether the surface has a non-specular lobe at all (a delta
  gets no shadow ray and, importantly, no draws), and which side of the surface
  to nudge the light-sampling point to.
- **`SampleLd`'s nudge**, which is easy to overlook: a reflective surface samples
  the light as seen from just *above* itself, and the offset point carries no
  error interval of its own because it is already the outward-rounded one.

**A faithfulness fix found along the way, worth 267 pixels.** `walk_step` and
`path_step` both normalized the outgoing direction — `wo = unit_vector_(-ray.d)`
— and pbrt's integrators do not. pbrt writes `Vector3f wo = -ray.d;` and uses
`isect.wo`, which *is* normalized, only in `SampleLd` and in `rho`. A ray
direction is a unit vector put through a rigid motion, so the two differ in the
last bit or two; that is invisible in a gbuffer and is not invisible at all in a
layered BSDF, whose walk is seeded by hashing the bytes of this vector expressed
in the shading frame. On killeroo-simple the pixels agreeing with pbrt to 1e-3
relative went from 3,973 of 5,038 to **4,240 of 5,038**, measured both ways.
Same class as the five `fma` placements under item 1 below, and found the same
way — by reading pbrt closely rather than by the numbers pointing at it.

**A new scene, `area-light-mis.pbrt`.** The six scenes that say `path` have no
emitter, so none of them reaches MIS, roulette, or `LayeredBxDF::PDF` — they
would have let the whole integrator go unexercised while reporting that six more
scenes now compare. It is `area-light-path` with one line changed, so a
disagreement between the three is a disagreement about transport and nothing
else.

**And the comparison stopped failing scenes that agree.** Six of the ten renders
are black on both sides, and the ratio of two zero means is 0/0;
`compare_gbuffer.py` was reporting that as "the image is infx pbrt's".

**`--spp`**, which is pbrt's flag under pbrt's name and reaching both sides by
pbrt's route — set on `PBRTOptions` before parsing, so every sampler's `Create`
reads it. Each of the three samplers is overridden the way its own `Create`
overrides it, which is not one rule: independent and halton take the number,
and stratified factors it into a grid by walking down from its square root, so
`--spp 12` is 4x3 and `--spp 13` is 13x1. Checked against pbrt on a lit
stratified scene at 4, 12 and 13, where a transposed grid would put the samplers
in different strata and pull the streams apart. It is mostly for looking at
pictures, and it immediately paid for itself as the measurement under item 2.

### What the rounds before those added

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

## The mission: real scenes, and what stands between here and them

`scenes/` is ten scenes written for this app. The point of the exercise is
scenes nobody wrote for it, and the measurement is `pbrt-v4-scenes`: 98 real
scenes across 29 collections. **Two of them convert today** —
`killeroo-simple` and `killeroo-moving`.

That is measured rather than estimated. `scene_dump` was run over all 98 and its
first refusal recorded; then again with the integrator and sampler checks
relaxed, since those fire first and hide everything behind them. The second
number is the one that says what to build, because it is what is left when the
things that are only *named* by a scene are set aside:

    what refuses                                        scenes
    ------------------------------------------------------------
    a light that is not a shape's emission                  13
    a material parameter that is a texture or a spectrum    27
    a material other than diffuse/coateddiffuse             24
    a shape other than sphere/trianglemesh                  23
    an area light on a mesh rather than on a sphere          7
    a camera other than perspective                          1
    a PLY file holding quads                                 1

and, hiding in front of those, `Integrator "volpath"` (51 scenes) and the
Sobol-family samplers (17). Broken down:

    shapes      bilinearmesh 20, disk 2, curve 1
    materials   hair 10, dielectric 7, subsurface 4, interface 2, conductor 1
    parameters  an area light's L 20, a material's reflectance 7
    samplers    zsobol 12, sobol 3, pmj02bn 2

Two of those lines have moved since the survey was taken: `infinite` lights
without an image are implemented, and the light-list ordering bug below was
fixed with them. The counts are otherwise as measured.

The survey found one outright bug and it is fixed: **a `LightSource` that was
not an area light was silently dropped.** Thirteen of the fifteen scenes that
got as far as converting were lit by one -- almost all `infinite`, an
environment map -- so they converted without complaint and would have rendered
black. That is precisely the failure this app exists to refuse, and it went
unnoticed because no scene in `scenes/` has a light that is not a sphere. It is
a refusal now, which is why the headline number above is two rather than
fifteen.

Read in dependency order rather than by count, the road is:

1. **`infinite` lights.** The half of this that is not an image is **done** --
   `UniformInfiniteLight`, and with it everything an integrator needs to cope
   with a light that is not a surface. Written up below. What is left is
   `ImageInfiniteLight`, which every real scene actually asks for and which is
   really *image textures*: reading an EXR, a bilinear lookup, an equal-area
   octahedral mapping, and a 2D distribution to importance-sample it -- the last
   of which `sampling.bonsai` already has, since the pixel filter needed it.
2. **Textures**, the same machinery pointed at materials. 27 scenes ask for a
   material parameter that is not a plain RGB, and a fifth of a scene's look is
   in them.
3. **Materials**: `dielectric` first -- 7 scenes, and every piece of it is
   already written, since `coateddiffuse` is a dielectric over a diffuse and the
   coating *is* `DielectricBxDF`. `conductor` is one more BxDF. `hair` and
   `subsurface` are their own projects.
4. **Shapes**: `bilinearmesh` alone is 20 scenes and is also what the PLY quad
   refusal is about, so the two close together. `disk` is an afternoon.
5. **Triangle area lights**, which is `Triangle::Sample` towards a point and its
   PDF -- 7 scenes, and the thing that makes an emissive quad work, which is how
   most scenes are lit indoors.
6. **`volpath`**, which is 51 scenes by name and rather fewer by need: it is
   pbrt's default, so a scene names it whether or not it has any medium in it.
   The ones that genuinely need media are the cloud and smoke scenes. Every
   other scene naming it wants `path` and would agree with `path`; making that
   substitution *honestly* means implementing volpath, since a scene with no
   medium is a case volpath handles rather than a case that is not volpath.
7. **Samplers**, 17 scenes, and the cheapest item on this list per scene
   unblocked -- the Sobol family shares one matrix-based construction and
   `zsobol` alone is 12.

## What is next, in order

Numbered by how long they have been open rather than by what to do first. Items
1, 2 and 5 are answered.

These are the items that came out of comparing against pbrt on the scenes that
already work. They are not the same list as the section above, which is about
scenes that do not work yet, and that list is the more important of the two —
what to do first is an `infinite` light. The items below are what the comparison
itself still owes: item 3 is the light samplers, item 4 is a profile of a path
rather than of a walk.

### 1. The last 2.7% — answered: it was noise, and it converges

**Resolved.** It was Monte Carlo noise in the layered material's reflectance
estimate, and pixel jitter is what made that demonstrable. The albedo
disagreement on `area-light-mis`, as the sample count goes up:

     64 spp   mean 4.38e-04, 1325 pixels over 5e-03 (1.61%)
    256 spp   mean 3.34e-04,  324 pixels over 5e-03 (0.39%)
    1024 spp  mean 2.89e-04,    5 pixels over 5e-03 (0.006%)

Five pixels of 82,183, which is inside the 0.01% the comparison allows for
rounding a wavelength — so at 1024 samples the albedo check passes. Before
jitter the same sweep read 1287, 1284: an estimate that would not converge,
because with every sample of a pixel on one ray `rho` was handed the identical
`wo` and returned the identical sixteen-sample number over and over. The
comparison had no way to distinguish noise from a mistake and now it does.

killeroo-simple went the same way: 13,290 pixels over the threshold before,
1,007 after, at the 256 samples the scene asks for.

What is left of this item is the history below, which is worth keeping because
the method is what found five real bugs. The residual — 2.9e-04 mean at 1024
samples, against ~6e-5 on a diffuse scene — is still a real gap and still
consistent with a last-bit difference reseeding a walk. The way to close it
further is unchanged: print pbrt's intermediates at a pixel and find the first
one that differs.

The original writing-up follows.

On killeroo-simple, 2.7% of pixels had an albedo that differed from pbrt's by
more than the comparison allowed. They were all on the two `coateddiffuse`
killeroos, and the difference was not bias: over those pixels the ratio of the
two renders had a median of 0.98, a tenth percentile of 0.69 and a ninetieth of
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

A sixth divergence of the same *kind* turned up this round and it was not a
rounding one at all: `walk_step` and `path_step` were normalizing `wo` where
pbrt does not. It is written up under "What the last round added"; it belongs
here because it says something about method. It was found by reading pbrt line
by line rather than by printing intermediates, and it moved 267 killeroo pixels
across the agreement threshold where turning contraction on had moved none. The
question this section asks — is the direction handed to the BSDF bit for bit
pbrt's? — has answers that are not about rounding, and the search should not
assume they are.

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

The first is what the last round did and it got from 11.6% to 2.7%.

**The second is now built, and it did not finish it.** `--ffp-contract` is an
SSA rewrite (`include/SSA/Contract.h`) that fuses a float multiply into the add
that consumes it, and `compare.sh` passes it. It is doing real work: the
renderer goes from 199 `llvm.fma` calls to 344, and the worst normal difference
against pbrt improves from 1.10e-05 to 9.92e-06. It even rederives one of the
five placements found by hand -- given `ax*bx + ay*by + az*bz` it emits
`fma(az,bz, fma(ax,bx, ay*by))`, which is `dot_` in `stdlib/numerics.bonsai`
character for character.

And the albedo comparison does not move at all: 2.7122%, 13290 pixels, the same
worst pixel with the same values. Not because nothing changed --- 49.2% of
albedo pixels changed --- but because every change is tiny:

    normals   2,776 of 490,000 pixels changed, largest 7.21e-06
    albedo  241,121 of 490,000 pixels changed, largest 1.72e-05
            0 pixels crossed the comparison's 5e-3 threshold, either way

That last line is the finding, and it argues against the theory above. If the
divergence were a walk reseeded by a direction differing in its last bit, then
perturbing the arithmetic at half the pixels would reseed some of those walks
and their albedos would move by *tenths*, not by 1e-05. None did. The walk is
not being reseeded by these changes, so whatever makes those 13,290 pixels
disagree with pbrt is not something contraction reaches.

So the next step on this is a measurement rather than a change: take one of the
13,290 and ask `scene_dump --print-bsdf` for pbrt's `wo`, shading frame and
per-sample values at that pixel, and find the first one that differs. The
harness for that already exists and is how the previous five were found. Until
that says otherwise, the third option below -- that a sixteen-sample estimate is
not a thing two implementations can be expected to agree on pixel by pixel -- is
the more likely explanation of what is left.

Contraction stays on regardless, because it is what pbrt is built with and
because it made the normals closer. Two things it does not do: it does not match
gcc's *choice* when both operands of an add are products (gcc has no consistent
rule there), and it does not contract into a subtraction, because the SSA has no
float negate -- `UnOp::Neg` becomes `0 - x`, which is not exact for a signed
zero, so `fma(a, b, -c)` built on it would be wrong. A real `Neg` instruction is
the prerequisite, and it is the same shape of work as the `AtomicAdd` that went
in for `tagged_index`.

### 2. Pixel jitter — done

**Resolved**, and written up under "What the last round added". It is pbrt's
Gaussian in `filter.bonsai`, sampled through the piecewise-constant 2D
distribution in `sampling.bonsai`, over a table the driver builds once. The film
divides by the summed weight and normalizes the summed normal, as
`GBufferFilm::GetImage` does. `--disable-pixel-jitter` asks for the old
behaviour and the gbuffer comparison still wants it.

The claims made for it here all held. The pixel-sampling half of every sampler
is exercised now — nothing depended on `get_pixel_2d` being right before, and
the normals still land on pbrt's exactly, which says the filter table is bit for
bit pbrt's. Item 1 became answerable and was answered. And the one thing that
had to be fixed along with it, the summed normal, was.

What it cost: silhouette pixels can now disagree, because a pixel straddling an
edge averages normals from two surfaces and a last-bit difference in one
sample's jitter changes which side that sample landed on. `many-shapes` has one
such pixel and killeroo-simple has eight. The comparison has always treated a
silhouette disagreement as tolerable and reported them separately, which is
exactly the distinction that now earns its keep.

### 3. The light samplers `path` actually defaults to

`path` is in and every scene compares, but it runs with the uniform light
sampler and pbrt's `path` defaults to `bvh`. So this is item 3 renamed rather
than item 3 closed.

The two are the same function on a scene with a single non-infinite light, and
that is an argument rather than a hope: `BVHLightSampler` over one light has a
one-node tree, so its PMF is 1 — the number the uniform sampler returns — and
the only way its `Sample` can differ is by declining to return the light where
the light's bounds say no energy can reach the shading point, which contributes
nothing either way. Both of its draws happen before either sampler is consulted,
so the stream does not move. The comparison checks the argument rather than
restating it: `render_reference` builds pbrt's `PathIntegrator` with the light
sampler the *scene* named, so on `area-light-mis` pbrt's real BVH sampler runs
against the uniform one here, and they agree to 1.0001x on the mean.

With more than one light they genuinely differ, so `scene_dump` refuses the
scene and says so. That is what has to be lifted, and lifting it is real work
rather than a switch: `BVHLightSampler` needs `LightBounds`, `DirectionCone`,
`Importance`, a SAH-ish build over 12 buckets, and the bit trails `PMF` walks
back down. `PowerLightSampler` is much smaller — an alias table over each
light's `Phi` — and is the sensible first of the two, since it removes the
refusal for every scene where importance does not vary over the frame.

Until then, no scene with two lights can be compared, which is the shape the
`path` gap had before this round: the renderer refuses rather than substitutes,
and the refusal names what is missing.

### 4. Where the time goes in a path

The schedule was tuned against a render whose cost was a traced ray and a
sixteen-sample reflectance. It is not that any more, and the number moved:
killeroo-simple was 1.56x faster than pbrt as a random walk and is 1.43x faster
as a path. Both sides do the same work, so the ratio is still honest — but the
mix of work is new.

Shadow rays are done and are written up below. What is left, from a `perf` of
the path render as it now stands (fractions of its own cycles):

    _traverse_tree0     15.0   the camera and path rays
    dielectric_sample_f 14.7   the layered walk's coating
    sphere_roots        13.1   ray-sphere, in interval arithmetic
    halton_dimension     8.3
    triangle_hit         7.3
    _traverse_tree1      7.1   the shadow rays
    coated_sample_f      4.9

Two things in that list are worth a second look:

- **`sphere_roots` at 13%, in a scene with one sphere in it.** It is the light,
  and every shadow ray is aimed at it — so every shadow ray reaches the light's
  own leaf and solves the quadratic there, in the outward-rounded interval
  arithmetic `interval.bonsai` exists for. pbrt pays this too (its `IntersectP`
  also has the light's primitive in the aggregate) so it is not a divergence,
  but a scene where the emitter is not also the thing every ray points at would
  measure very differently. Worth knowing before reading this profile as
  typical.
- **A layered BSDF is now over 20%** across `dielectric_sample_f` and
  `coated_sample_f`, and that is before counting `LayeredBxDF::PDF`, which MIS
  calls once per light sample at every `coateddiffuse` vertex.

### 5. Shadow rays ask `any` now — and the lowering had a gap

**Done.** `unoccluded` called `trace`, which answers with the *nearest* hit
where an occlusion test only needs *any* hit. It is `trace_any` now, written as
`any(pred, primitives)` — the same source as `trace` with the `argmin` taken
off, which is the whole point: the algorithm says which question is being asked
and the schedule decides what that costs. `contains(Ray, AABB)` had to be
written to lower it, and it is `false`: a one-dimensional ray never contains a
three-dimensional box, so the sufficient condition an `any` could settle on is
never satisfiable and the saving has to come from stopping early instead.

On killeroo-simple, best of three, every pixel bit-identical throughout:

    trace, nearest hit                      14173.6 ms
    any, as it was lowered                  13745.5 ms   1.031x
    any, with the leaf prune fixed          13299.1 ms   1.066x

The middle line is the finding. A 3% win was less than it should have been, and
a `perf` said why: `_traverse_tree1` was cheaper than the traversal it replaced,
but `triangle_hit` went *up* by 20% — 156 to 187 gigacycles. The lowered IR
explained it. `build_filter` guards each element of a leaf with the predicate's
bound over that leaf's volume, so `argmin(f, filter(p, tree))` tests a leaf's
bounding box before any of its primitives; `RewriteQuantifier` did not, so a
bare `any(p, tree)` ran the full primitive test on every element of every leaf
it reached — no box test, and no check that the answer was already settled. A
BVH leaf holds about four primitives, so that is one box test traded against
four triangle intersections, every time.

`Lower/Trees.cpp` puts the same two guards on a quantifier's yields that it
already puts on its scans. `triangle_hit` fell to 139 gigacycles, below where it
started, and the win went from 3.1% to 6.6%. `correctness/cpp/setops` checks
`any` and `all` against a linear scan over the same data and passed throughout,
which is what says this was a pruning change and not a semantic one; the two
`lower/setops` goldens moved and were re-blessed.

The general lesson is worth keeping: a query written the more specific way is
not automatically lowered the better way, and the place to look when it is not
is what the *other* lowerings of the same shape already do.

## What has been built, and what it cost

### Where the time went

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
the jitter is off, is item 2 above.) And many-shapes is traversal-bound but
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

### Once the render was a whole path, killeroo was 1.36x slower

The numbers above are from the era when this app answered what a camera ray hit.
With the random walk in, the same scene went the other way: 15.7 seconds against
pbrt's 12.2. Two causes, neither in the traced ray. What a `perf` of each side
said, as fractions of that side's own cycles:

    sampler   bonsai  permuted_digit 27.3, halton_dimension 8.5,
                      get_pixel_2d 2.9, start_pixel_sample 0.7   39.4%
              pbrt    HaltonSampler::SampleDimension              5.6%
    traverse  bonsai  _traverse_tree0 15.2, triangle_hit 7.7     22.9%
              pbrt    BVHAggregate 11.6, Triangle::Intersect 7.7,
                      IntersectTriangle 6.8, SimplePrimitive 4.6,
                      Primitive 0.7                              31.4%

Which is worth reading twice: in absolute cycles the traversal and the BSDF were
already *faster* than pbrt's -- 336 against 562 gigacycles on the first and 240
against 335 on the second. Nothing about the ray was wrong.

**The loop was split into contiguous blocks.** `bonsai_parallel_for` gave each
thread a run of `n / threads` iterations, on the assumption stated in the header
that they cost about the same. A pixel of a render is one iteration and the
killeroo is in the middle of the frame, so the threads holding the middle did
most of the work and the rest finished early and idled: **18.8 of 32 threads
busy** on average, against pbrt's 26.9, which its own `ParallelFor2D` gets by
handing out tiles on demand.

The header was written so that this choice lives in it and nowhere else, and it
now asks TBB: `tbb::parallel_for` with the default partitioner, which splits the
range further while threads are idle and steals between them. std::thread
remains for a machine with no TBB, doing the crude version -- a fixed chunk
taken off an atomic, no stealing. Which one is used is `__has_include`, so a
build that has TBB gets it and a build that does not still compiles; `compare.sh`
and `render.sh` look for it beside `$BONSAI_CXX`, which is where a conda
environment puts both, and say so when they do not find it. 15.7 seconds became
10.9, and 1881% CPU became 3063%.

**The Halton sampler recomputed what pbrt looks up.** `permuted_digit` ran
`hash3` and then `permutation_element` -- a rejection loop -- for every digit of
every draw, where pbrt's `DigitPermutation` is a table its constructor fills
once. The old comment said computing it "buys back a table that is tens of
megabytes ... if the sampler ever shows up in a profile, the table is what to do
about it, and it answers the same". It showed up at 39%.

So the table is built, and it is pbrt's: 26.2 MB over all thousand primes, the
same `nDigits * base` uint16 per base, laid end to end with a 1000-entry offset
array because an array of arrays is a pointer to chase on the hot path. What
fills it is the *renderer's* `permutation_element` and `hash3`, through two
exported functions -- `digit_permutation_extent` sizes it and
`build_digit_permutations` fills it -- rather than a C++ copy of both in the
driver, because a second implementation of a thing this app checks against pbrt
is one more than it can afford. The driver owns the storage, as it owns the ADT
pools, which is what lets a table this large exist under `--no-heap`. 10.9
seconds became 7.9, and the 26 MB costs 0.2 seconds to build before the timer
starts, which is where pbrt builds its own.

The rule the two share: `digit_permutation_digits` and the loop in
`scrambled_radical_inverse` must stay the same expression, since the table
having one fewer digit than the loop asks for is a read past the end of it.

Where that leaves the comparison:

    killeroo-simple  1.36x slower  ->  1.57x faster
    area-light-path  1.01x faster  ->  1.58x faster
    area-light       (unmeasured)  ->  1.88x faster

Every rendered pixel is unchanged: killeroo-simple is still 0 disagreeing
normals and the radiance PFM is byte-for-byte what it was before the sampler
changed. The profile now reads like pbrt's -- `dielectric_sample_f` 23.6%,
`_traverse_tree0` 22.1%, `triangle_hit` 10.7%, and the whole sampler down to
13%.

### Light transport — killeroo-simple renders

Historical: killeroo-simple gets `path` now, since it names no integrator, and
the numbers below are from when a scene naming none got the random walk. The
scene has not changed and neither has the random walk, so `area-light.pbrt` —
which names `randomwalk` — is where this is still checked.

`RandomWalkIntegrator` is implemented and killeroo-simple renders through it.
Against pbrt's own randomwalk on the same scene at the same `maxdepth`:

    pbrt    5038 lit pixels of 490,000   mean 2.26936   max 2032
    bonsai  5038 lit pixels of 490,000   mean 2.26933   max 2032.23

The same pixels are lit, every unlit pixel is bit-identical, and the means agree
to a thousandth of a per cent. Of the 5,038 lit pixels 3,750 agreed to better
than 1e-3 relative when this was written; the rest diverge sharply, which is
what a last-bit difference does to a stochastic walk -- it flips a branch and the
path goes somewhere else entirely. Same class as the albedo divergence in item 1,
and not something contraction reached there either.

That figure is **4,240 of 5,038** now, and the whole of the gain was the `wo`
normalization written up under "What the last round added". The two were
measured against each other on this scene: normalized 3,973, raw 4,240.

What was built, and what each piece cost:

- **Area lights.** `DiffuseAreaLight` rather than the `UniformInfiniteLight`
  this list used to name first, because killeroo-simple has no infinite light --
  it is lit by an emissive sphere. For a random walk that is no harder: the
  integrator never samples a light, it finds one by hitting it, so this needed
  no `SampleLi`, no PDF and no shadow ray. `scene_dump` folds pbrt's
  `scale /= SpectrumToPhotometric(L)` in, since it is a property of the scene.
- **`f` and `PDF` for every BxDF**, with `TransportMode` threaded through as one
  `radiance : bool`. `LayeredBxDF::f` needed all of it: it samples a virtual
  light back through the exit interface in the *reversed* mode, and combines two
  estimators of the same quantity with the power heuristic, which is what the
  PDFs are for.
- **`SpawnRay` / `OffsetRayOrigin`**, which is what `interval.bonsai` was written
  for and had never been used. `SurfaceGeometry` now carries the hit point and
  its error bound -- `gamma(5)` for a sphere, `gamma(7)` for a barycentric
  interpolation -- and the next ray starts exactly that far off the surface.
- **The walk**, as a loop carrying throughput and radiance rather than pbrt's
  recursion, which is the same thing for a bounded depth.

Two bugs worth remembering, because neither looked like what it was.

**The film normalized a radiance like a reflectance.** Every lit pixel came out
106.86 times too dim -- a picture that looks like a weak light rather than like
an error. `SampledSpectrum::ToXYZ` divides by the CIE Y integral and
`PixelSensor::ToSensorRGB` does not: a reflectance is a ratio and has to be one
for a perfect reflector, a radiance is a quantity. `colour.bonsai` now has
`spectrum_to_rgb` for the first and `radiance_to_rgb` for the second.

**The camera sample was not drawing.** pbrt's `GetCameraSample` calls
`GetPixel2D`, `Get1D` and `Get2D` and *then* overwrites all three when
`--disable-pixel-jitter` is set; this app skipped the draws instead. That is
invisible in a gbuffer, where nothing else touches the sampler, and it is the
whole difference in a render: every scattering direction of the walk comes from
the stream, so the two renderers were integrating the same thing with different
numbers. Before the fix, 5,136 pixels were lit here against pbrt's 5,038 and
their values were unrelated; after it, the same 5,038.

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
- `SSA/CodeGen_Stmt.cpp` names `FieldPtr` as having no binding but has no case
  to rebuild one in `codegen_value`, so a FieldPtr read as a value would name
  something that does not exist. Unreachable today -- every FieldPtr is built by
  `walk_accesses` for a store's destination and consumed by `codegen_gep`, which
  asks for a WriteLoc and never for a value. The fix is the one GEP already has
  there. No failing test is possible without first making something produce a
  FieldPtr in value position, which is why it is a TODO in the file rather than
  a change: an untested rebuild of an address miscompiles quietly, which is the
  failure mode this whole app exists to catch.
- `scenes/stratified.pbrt` and `scenes/halton.pbrt` share three-spheres'
  geometry on purpose — they isolate the sampler — but that does mean their
  normal images are identical to `three-spheres`, which is confusing until you
  know why.
- `scenes/area-light-mis.pbrt` is `scenes/area-light-path.pbrt` with one line
  changed, for the same kind of reason and with the same kind of confusion
  available: the two render almost the same picture, because `simplepath` and
  `path` differ in how they weigh two estimators and not in what they estimate.
  `scenes/infinite-uniform-simple.pbrt` is the same trick again.
- **The area lights are in the wrong order, and it is measured.** The list has
  pbrt's *shape* — area lights first, then everything else — but not pbrt's
  order within the first part. pbrt walks the scene's shape entities as
  declared; the driver walks `shapes` after `build_bvh` has permuted them so
  that a leaf names a contiguous run, so light *i* is a different light on the
  two sides. Nothing about lights is sorted or built into a tree; this is the
  BVH over geometry, reordering the primitives the lights hang off.

  On a scene with two area lights of different colours under `simplepath`:
  the means agree, 1.00038x, because both sides are unbiased estimators of the
  same integral — and per-pixel agreement collapses to **44.2%**, where a
  one-light scene of the same kind gives 85-90%. Making the two lights the same
  colour, so the permutation cannot be observed, takes it to **100.0%**
  (79,124 of 79,127). That is the whole diagnosis: same lights, same count,
  same emission, different index.

  It bites `simplepath` and `path` and not `randomwalk`, which samples no light.
  `path` refuses a multi-light scene already for the light-sampler reason under
  item 3, so `simplepath` is what is exposed today. No scene in `scenes/` has
  more than one area light, which is why this survived.

  The fix is not a one-liner: `prim.light` indexes `loaded.lights`, which has
  one entry per `AreaLightSource` *directive*, while pbrt makes one Light per
  emissive *shape* — so a mesh's thousand triangles share one entry and their
  original order is not recoverable from what the scene file carries. Either a
  Shape gains its pre-reorder index, or the driver builds the light list before
  handing the primitives to the BVH. It belongs with item 3, which is where
  multiple lights start to matter anyway.
- pbrt's own reference render for `area-light` moved by 8e-6 in its mean at some
  point during this round — 0.759337 to 0.759329, which shifted about 1,100
  pixels across the comparison's 1e-3 relative tolerance and so read as a
  change of 2 percentage points in the agreement figure. It is stable at the
  second value over four runs and the obvious suspect was checked and cleared
  (constructing pbrt's three integrators rather than only the one the scene
  names does not change it). Nothing in the renderer moved; our side's mean is
  the same digit for digit either way. Recorded rather than chased, because it
  is a wobble in the reference and an order of magnitude below every threshold
  the comparison uses — but it does mean the agreement percentages here should
  be read as approximate, and it would be worth knowing what pbrt is doing that
  is not reproducible.
- Sobol, zsobol and pmj02bn samplers are refused. Measured over
  `pbrt-v4-scenes`: 17 of 98 scenes ask for one, zsobol being 12 of those.
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
