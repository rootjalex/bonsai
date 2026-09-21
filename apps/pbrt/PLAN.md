# apps/pbrt: where this is and what is next

The goal is a faithful PBRT-v4 clone written in bonsai, where the algorithm
says what to compute and the `schedule` block says how — and where "faithful"
means checked against pbrt rather than asserted.

Run `apps/pbrt/compare.sh <scene.pbrt>` to see where things stand. It renders
the scene both ways and reports disagreeing pixels, albedo difference, and
relative speed. `apps/pbrt/render.sh <scene.pbrt>` renders and writes PNGs to
look at.

**The reference is the `pbrt` binary and nothing else.** compare.sh runs it,
and compares against the image it writes. There used to be a second thing here
-- `render_reference` in scene_dump.cpp, which assembled pbrt's camera,
aggregate, BSDFs and integrator and drove them from a loop of our own. Every
value it ever produced agreed with the binary to a part in a million, and it
was still wrong to have: it differed from pbrt's `RenderCPU` in two ways found
by reading it rather than by any check failing (it traced the camera ray twice,
and it never called `ScaleDifferentials`), and a reference that can drift
without saying so is not a reference. It is deleted.

Two things follow from that, and both are honest limits rather than gaps in the
harness:

- **pbrt only writes normals and albedo for `path` and `volpath`.**
  `RandomWalkIntegrator` and `SimplePathIntegrator` take the `VisibleSurface`
  parameter and ignore it, so a gbuffer of one of those renders is all zeros.
  A scene under those integrators is compared on its radiance, and its geometry
  is checked by whichever scene shares it and names `path`.
- **A scene that names no integrator gets `volpath` from pbrt and `path` from
  us**, so the two run different algorithms over different random numbers.
  killeroo-simple is the one that does this: the means agree to 0.99997x,
  because both are unbiased estimates of the same integral, and 1.1% of pixels
  agree individually rather than the 32.7% the old harness reported when it was
  running `path` on both sides. Nothing regressed; the old number was measuring
  something easier.

Our own scenes therefore say `Film "gbuffer"` with `"string coordinatesystem"
"world"` and `"bool savefp16" false`, which is what makes pbrt write normals,
in the space this renderer works in, as floats rather than halves.

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
pbrt's gbuffer normals -- the shape's and the shading one a bump map tilts --
and albedo beside the radiance — so a comparison can ask four questions of the
same render rather than one.

**barcelona-pavilion renders, and matches.** `pavilion-day.pbrt` -- 133,575
shapes, 43 instances holding 5.2 million triangles, a lens camera, a sky, bump
maps on the pool and its walls -- agrees with pbrt to **0.99967x** in its mean
at 16 samples per pixel (1.00017x at 2), with every one of 48 image blocks
within 1% and no block systematically off. That took three fixes described
under "what the round before that added": a light leak of this renderer's own,
the last bits of a lens camera's rays, and the texture chain made bit-exact.
**It renders 1.13x faster than pbrt there, on the same integrator** (4.90 s
against pbrt's 5.55 s running `path`, at 16 samples per pixel, best of three
each side, on pbrt's own tree), where two rounds ago it was 1.59x slower and
one round ago 1.43x. The ratio is the figure to hold onto; the seconds move
with the machine's load, so only a pbrt and a bonsai run taken back to back
compare, which is what `compare.sh` does. The scene names no integrator;
pbrt's default for that is `volpath` and this renderer's is `path`, and the
1.27x the last round first reported was against volpath, whose closest-hit
shadow rays cost pbrt more than `path`'s any-hit ones -- a comparison of two
integrators, not two
renderers. `compare.sh` now tells pbrt the integrator this renderer resolves
the scene to, and every figure here is like for like. The last round is
written up below: four compiler changes took it to 1.22x slower with the image
unchanged to the bit, and one renderer change -- the visible surface filled
where pbrt fills it, and only for a film that records it -- took it past pbrt.

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

It is also faster than pbrt on that same work: **1.35x on killeroo-simple**,
1.35x on area-light-mis and area-light-path, 1.43x on area-light, 1.96x on
infinite-uniform, and between 1.30x and 1.50x on the six gbuffer scenes.

**Those numbers are lower than the ones this file used to quote, and the old
ones were wrong.** Not measured wrongly — earned wrongly. The renderer was
skipping work pbrt does. See "the same work, no more and no less" below; the
short version is that a diffuse surface's reflectance was being written down
instead of estimated, which is exactly the same number and sixteen BSDF samples
cheaper, on every camera ray that hits anything diffuse. On three-spheres that
alone was 17% of the render.

Killeroo's figure also fell when it stopped being a random walk: `path` spends
its time in shadow rays, in `LayeredBxDF::PDF` and in MIS, which is a different
mix of work from the one the schedule was tuned against. Item 5 below won 6.6%
back by asking `any` for a shadow ray instead of the nearest hit; item 4 is what
is left to look at.

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

### The same work, no more and no less

The rule this app is built on is that the algorithm is pbrt's and only the
*schedule* differs — where "schedule" includes memory layout and parallelism.
That is a claim about the work done and not only about the numbers produced, and
it had drifted in both directions. Two were found and fixed; the audit that
found them is worth repeating whenever a number looks good.

**A surface's BSDF was built up to five times per vertex.** pbrt calls
`GetBSDF` once at an intersection and then asks `Flags`, `f`, `PDF` and
`Sample_f` of the value it returned. Here each of those took a `Material` and
rebuilt the BxDF inside itself, so one vertex of the path integrator remapped
two roughnesses, evaluated two sigmoid polynomials over four wavelengths and
clamped them, five times over, to arrive at the same value each time. `BxDF` is
a variant now and `material_bxdf` is `GetBSDF`; everything after it is a method
on the value. Nothing about the image changed.

**A diffuse `rho` was written down rather than estimated.** pbrt's `BxDF::rho`
spends sixteen `Sample_f` calls whatever the BxDF is. A Lambertian BRDF's
hemispherical reflectance is exactly its reflectance -- the cosine and the
density cancel -- so this returned the reflectance and skipped the sixteen. The
value is identical, bit for bit, which is why it went unnoticed: three-spheres
and area-light-mis report the same albedo difference to every digit before and
after. The *cost* is not identical, and it is not small. On three-spheres,
which is one camera ray per pixel into diffuse spheres, spending the sixteen
samples pbrt spends is **17% of the render**: 1.57x faster than pbrt became
1.43x, and then 1.38x once the machine settled.

That is the shape of the failure to watch for. A shortcut that provably cannot
change the answer is invisible to every check this app has -- the images match,
the albedo matches, the sampler stream is untouched -- and it shows up only in
the clock, where it looks like the schedule doing well.

Two more asymmetries, both checked and both fine:

- **The traversal evaluates `intersects` once and `distmin` three times per
  candidate primitive** in the lowered IR, where pbrt intersects once. The
  backend common-subexpression-eliminates them: the generated
  `_traverse_tree0` contains exactly one call to `triangle_hit` and one to
  `sphere_roots`. Checked in the LLVM rather than assumed.
- **The camera ray is traced twice** -- once by `visible_surface` for the
  gbuffer and once by the integrator. That is the comparison harness rather
  than the renderer: pbrt fills a `VisibleSurface` from inside `Li` at depth
  zero, but only its *path* integrator does, so a gbuffer of a random-walk
  render would be empty. `render_reference` traces twice for the same reason,
  so both sides do the same work and the ratio is honest -- but a pbrt render
  of these scenes into an `rgb` film would trace once.

### What the last round added

**The comparison runs both sides on one integrator.** The pavilion names no
integrator; pbrt's default for that is `volpath` and this renderer's is
`path`, and every pavilion figure before this round compared the two. volpath
traces a shadow ray as a closest-hit transmittance loop where `path` asks an
any-hit question, so the 1.27x the previous round reported was partly pbrt
running the heavier integrator. `compare.sh` reads the integrator scene_dump
resolves the scene to and, when the scene named none, puts `Integrator
"path"` in front of it and hands the result to pbrt on standard input from the
scene's directory (pbrt has no flag for it, and reads standard input when
given no file). Like for like: pbrt 5.55 s, this renderer 4.90 s, **1.13x**,
mean 1.00009x, lit pixels within 0.03%. The rule is standing: a number against
a different integrator is not a number.

A second harness bug hid behind the first and is fixed in the same place. The
`cd` into the scene's directory was paired with a relative `--outfile`, so
after the change of directory pbrt could not create the file and -- exiting
zero on a failed write, having rendered first -- left the previous run's image
in place. Every no-integrator scene, the pavilion included, was compared
against whatever pbrt wrote last: the pixels still matched, since the stale
image was a `path` render of the same scene, but pbrt's *time* was read from
that stale image's metadata rather than measured. The output path is absolute
now and a render that writes nothing stops the run. The figures above are from
a pbrt actually re-run this round, which is why pbrt's second is 5.55 and not
the 4.66 the stale metadata reported.

**The layout language names an extern array's storage** -- `layout
mesh_positions { tight(root); };` -- described under "Where it is" above.

**The inliner runs one level a round, callers first, with a merge between
rounds** (`Opt/Inline.h`). The two fixed rounds of the previous round -- bodies
without mutable locals, a merge, bodies with -- were a heuristic cut to the
depth the app happened to have; a chain one level deeper would have left two
copies of the same slab test for good. A round now replaces every inlinable
call with the callee's body as the round found it, calls inside intact, and
is followed by simplification and CSE over the result; rounds run until one
copies nothing, which is after as many as the call graph is deep -- the
inlinable functions form a DAG once the strongly connected components of the
call graph (Tarjan, SIAM J. Comput. 1972) are excluded. The `allocates`
distinction is gone: the merge sees two calls side by side whatever their
bodies hold, and the next round copies the one call left. The app compiles
in 7.6 s where it took 5.2 (`opt/inline-rounds.bonsai`,
`correctness/llvm/inline-rounds.bonsai`).

**Jump threading is bounded by structure, not by tunables.** The first version
capped a copied run at 256 statements and threadings at three deep, numbers
chosen so the pavilion's leaf would thread. Now (`Opt/JumpThreading.h`): a
copy is final -- the statements copied into an arm are never threaded again,
so two independent conditions tested in a row cannot multiply and the
exponential code replication is capable of cannot happen; a `match`'s chain
of branches is followed link by link, so the run is copied once per case
rather than once per test; and a threading may add at most 15 statements,
counted as what survives in each copy once its arm's facts are applied,
summed over the copies, less the run as it stood -- GCC's bound for the same
quantity (`max-jump-thread-duplication-stmts`), where LLVM's is 6
instructions. What the copies decide is no longer the pass's business
(`opt/jump-threading.bonsai`: `chain`, `independent`, `wide`).

**The simplifier learns what a branch decides** (`Opt/Simplify.h`). Inside
`if (c) { A } else { B }` the condition is true throughout A and false
throughout B; after `if (c) { ...; return }` it is false for the rest of the
sequence; a branch or select on a known condition folds to the arm taken, a
select only when the arm not taken has no effect. A condition is learned only
when it is a pure value over names the function cannot assign. This is the
strict-reduction half of what jump threading used to do inside its copies,
and it now applies everywhere the simplifier runs, including between
inlining rounds -- which is where the nested `match` on a primitive's tag
folds, and where a traversal's `if (false) { holds = true }` from a pruning
condition predicate analysis decided goes, before SSA ever sees it
(`opt/simplify-facts.bonsai`, `correctness/llvm/simplify-facts.bonsai`).

Running CSE between rounds found two defects that had been dormant. CSE
counts an expression once per place it is read and a `let` variable's reads
as reads of what it binds, so a select bound to a variable read twice had
every subexpression given a temporary; the pass's last step put a temporary
used once back where it was read and *kept the `let`*, dead, with the call
still in it -- the next round inlined that call, the next CSE made the `let`
again, and the rounds never ended. The `let` goes now
(`opt/cse-dead-temporary.bonsai`). And the expression ordering CSE keeps its
map by had no case for `Construct` or `UnionOf`: it fell off the end of its
switch and answered whatever was in the register, which corrupted the map
and crashed the compiler on the first program that compared two variant
constructions (`opt/cse-adt.bonsai`).

**The image.** With `--ffp-contract`, 20,188 of the pavilion's 1,360,000
pixels differ from the previous round's render in the last bit (max relative
2.3e-05, sums equal to the digit): CSE now shares multiplies that were fused
into an fma before and are not when they have two uses. Without contraction,
the previous compiler and this one render the pavilion **bit for bit the
same**, which is the check that nothing about what is computed moved. The
speed is the same within noise as before the round -- the pass changes were
about principle and generality, not this scene.

**The running best, measured in the assembly.** The question left open was
whether the eleven scalars LLVM keeps for the `(metric, (Primitive,
Geometric))` accumulator cost the traversal anything. Retired instructions,
sampled on the same tree, `path` on both sides: this renderer's interior-node
loop (the slab test, the child order, the push) retires about 135 G against
pbrt's `BVHAggregate::Intersect` loop at about 210 G; a node is about 45
instructions here -- the slab test is three lanes wide, `vmulps` on the box
minus the origin times the inverse direction, then the per-axis compares --
against about 55 in pbrt's scalar loop with its per-axis early exits. Of the
accumulator, only the metric is in the loop at all, a register compared once
per node (`vucomiss %xmm0, %xmm17`); the other ten scalars are touched at a
leaf's store and at the exit and never spill into the loop. What the loop
does reload from the stack is three loop invariants -- the ray origin, the
inverse direction and the bound -- three loads a node, about 7% of its
instructions; pbrt's loop reloads its `invDir` from the stack five or six
times a node, so it is behind on the same count. The one thing worth
noting is that `dirIsNeg[axis]` re-stores the mask to the stack every node
(`vpmovm2d`, `vmovdqa`, 0.4% of `full_path_step`): a dynamic
`extractelement` is lowered through memory by LLVM at the use, inside the
loop, where pbrt writes its `int dirIsNeg[3]` once before it. About 1 G in
850; an SSA rewrite that binds a loop-invariant vector once to memory when
it is only ever indexed dynamically would remove it, and is not worth its
own round.

So the two-copies-of-`Geometric` question has its answer: it is the type of
what `flatten` yields, the outer element and the inner one it was reached
through, and in the `Geom` arm the inner *is* the outer's payload, so those
twenty bytes were there twice; in the `Inst` arm they are not. The type is
right. What was wrong was copying the element into the accumulator at all.

**An argmin keeps a reference to its best element, not a copy.** Every
improving hit used to copy the whole element into the accumulator -- the
41-byte pair here, and a 36-byte `Triangle` in a plain single-tree query --
where pbrt keeps `tMax` in a register and a pointer to where the hit goes.
`build_arg_extremum` (`Lower/Trees.cpp`) now holds a pointer to each
component of the element that is a place in a tree's storage, and the value
of any other; `stored_components` decides from the shape of the set
expression, before the traversal is built, since the accumulator's type is
threaded through it: a tree's elements are places, a tree held in a field is,
a `match` arm's set literal is when what it lists are fields of the matched
element, a `filter` is what its set is, and a `map` is not. The element is
read back through the pointers once, where the option is built, behind a
branch rather than the strict `select` that would have read through the null
the accumulator starts with. The pavilion's accumulator is a float and two
pointers -- three loop-carried values where there were eleven -- and the
render is unchanged to the bit, contraction on, still 1.13x faster than
pbrt on `path`.

The addresses it needs did not exist yet. `PtrTo` of a chain of accesses
rooted at an array -- `vs[i].w`, `prims[k].payload.Geom.g` -- was lowered by
the SSA conversion as the address of a *copy* of the loaded value, and that
was already a bug: `bump(vs[i].w)` with `bump` taking its argument by `mut`
compiled to `ret void`, the increment written into a temporary nothing read.
`address_of_place` (`SSA/Convert.cpp`) walks the chain to its root, a
dereferenced pointer or an array, and composes the offsets, GEP for an index
and FieldPtr for a field; a union's member is at offset zero and its address
is the union's, read at the member's type, which is what the per-arm layout
of a mixed tree stores an inline arm's fields behind. The relooper and
`CodeGen_LLVM`'s `PtrTo` know the same chains. CSE leaves the operand of an
address-of alone, since naming `vs[i]` would make it a copy again. Along the
way a second dormant bug: `ws : mut array[f32, 2] = {1.0, 5.0}` stored the
built array's *pointer* over the elements of `ws`, so `ws[1]` read back the
upper half of a heap address; the golden that documented it as a known
limitation is gone, and a whole-array assignment to a name copies the
elements (`correctness/llvm/mut-element-field-argument.bonsai`,
`correctness/llvm/mut-array-literal.bonsai`, and the traversal goldens).

### What the previous round added

**The traversal's code is pbrt's now, and the pavilion is faster than pbrt.**
The round began where the previous one ended: the per-node instruction count
of the traversal loop, measured against pbrt's `BVHAggregate::Intersect`, with
the query itself already saying exactly what pbrt's does. Each step below was
measured on the pavilion at 16 samples per pixel on pbrt's own tree, and the
image did not move by a bit through any of them -- checked by comparing the
radiance image against the previous commit's, byte for byte, after every one.

**CSE before the inliner, and the inliner in two rounds.** The sort key
`(1 / r.d)[axis] < 0` was its own expression: `aabb_span` stayed a call at the
bonsai level because it has mutable locals, so nothing shared its `inv_dir`,
and by the time LLVM had inlined it InstCombine had already scalarized the
single-use `extractelement(fdiv)` into a divide per node that depends on
`axis` and cannot be hoisted. The `ssa` pipeline runs CSE before inlining now
-- which also merges the two `distmin(r, g)` calls a candidate makes before
the statement inliner can copy them apart -- and inlines in two rounds, bodies
without mutable locals first, a merge, then the rest (`Inline(with_locals)`,
named `inline-locals` because the pass manager keeps one pass per name). The
simplifier rewrites a lane compared with a constant as that lane of the vector
compare, `v[i] < c` to `(v < c)[i]`, so the key's test becomes the same
`dir_is_neg` the slab test selects on; with two uses the compare stays a
vector, LLVM's own LICM carries the divide and the compare to the entry block,
and the node reads one lane of a mask -- pbrt's `dirIsNeg[node->axis]`. 1.39x
(`opt/inline-locals.bonsai`, `correctness/llvm/inline-locals.bonsai`).

**The sort network folds to `!c`.** The network compared `cast<f32>(c)` with
`cast<f32>(1 - 1)`, and the second key never folded because the simplifier had
no float constant folding at all; the `lt` that `sort_recursion` builds is made
in SSA, where nothing had ever looked at it. Floats fold now, in their own
precision. A comparison of two constants folds in the *operands'* type rather
than the result's -- it had been comparing the operands' bit patterns as
unsigned integers, which answered `-1 < 1` false. And a small SSA peephole
(`SSA/Simplify.h`) runs after the network: `cast(a) < cast(b)` for bools is
`!a & b`, `!!x` is `x`, `x & x` is `x`, and what that leaves unread goes. The
interior node's IR became an `extractelement` of the hoisted mask, two adds,
two selects and the pushes; 1.35x (`opt/constant-fold.bonsai`,
`correctness/llvm/compare-fold.bonsai`).

Turning the second inlining round on found two things. Alloca promotion placed
a block argument at every join in the iterated dominance frontier of a local's
stores, including joins the allocation's own block does not dominate -- a local
declared in one arm of an `if`, where a path through the other arm has no value
to hand the join -- and only dominated joins take one now. And the direct
SSA-to-LLVM path had no `load_field`, `make_struct`, `eps` or `inf`, which an
inlined slab test in a gang needs (`ssa/vectorize-arm-local.bonsai`,
`correctness/cpp/vectorize_arm_local.bonsai`).

**`loopify` continues with the first child.** It wrote every recursive call to
the stack and took the next node off it at the top of the loop, so an interior
node cost two pushes, a pop and two counter updates where pbrt pushes the far
child once and carries on into the near one. The loop has pbrt's shape: the
current node and a `live` flag are the header's arguments; a run of calls that
is the last thing a node does -- a continuation that only returns, followed
through the chain of empty merge blocks a match leaves behind -- pushes all but
its first and makes that one the current node; a node with nothing left to do
takes the top of the stack or, when there is none, clears the flag. The flag
stands in for the `break` the statement form cannot write, and LLVM's jump
threading turns its constant edges into direct branches. The early exit an
`any` derives moves with the pop, and the direct edge to the first child asks
the accumulator only if something on the way there could have moved it, which
a leaf's store never is on an interior node's path. 1.27x
(`ssa/loopify-queue.bonsai`, `ssa/ray-any-early-exit.bonsai`).

**A branch is threaded through the code that tests it again.** `intersects`,
`alpha_accepts` and `distmin` on a `Shape` are three `match`es over one tag,
and behind two of them the same ray-triangle test. `Opt/JumpThreading` takes
the statements that follow a branch and test its condition again into both of
the branch's arms, where the second test is decided and folds; the CSE after it
finds `triangle_hit` twice in one arm and makes it one, which is pbrt's single
`Shape::Intersect`. This is jump threading in its code-replication form
(Mueller and Whalley, PLDI 1995; Bodik, Gupta and Soffa, PLDI 1997) and, on a
variant's tag, the splitting of the SELF compiler (Chambers and Ungar, PLDI
1990); the header cites all three. A condition is threaded only if it is pure
and nothing in the function can assign what it reads, the copied run has no
loop in it (a schedule names loops), and the run's bindings are renamed per
copy. A `select` on a known condition folds only when the arm not taken does
nothing, because a select evaluates both arms and the arm not taken may be a
draw from a sampler. 1.22x (`opt/jump-threading.bonsai`,
`correctness/llvm/jump-threading.bonsai`).

That pass found two defects that had been waiting for a shape to expose them.
`Unswitch` merged two consecutive `if (c)` statements even when the first's
arm always returns, leaving statements behind a return. And `has_side_effects`
judged an `Intrinsic` or a `Call` without looking at its arguments, so
`min(cast<f32>(next_uint(rng)) * k, 1)` passed for pure: once threading had
decided the branch around a clamped sampler draw, the draw's binding was
unread, DCE deleted it, and the random stream of everything after it moved by
one. 661,200 of the pavilion's 1,360,000 pixels changed while the mean did not
-- the signature of a shifted stream rather than a wrong value, and what the
byte-for-byte check after every step was there to catch. The analysis descends
into arguments now (`correctness/llvm/dce-effectful-argument.bonsai`).

**The visible surface is filled where pbrt fills it, and only for a film that
records it.** With the traversal at pbrt's instruction count, the profile
moved: 1324 G instructions retired against pbrt's 1041 G at equal IPC, and the
largest excess was `dielectric_sample_f` at 222 G against pbrt's
`DielectricBxDF::Sample_f` and its Trowbridge-Reitz helpers at about 110 G --
while `coated_f`, the same layered walk evaluated rather than sampled, matched
pbrt's `LayeredBxDF::f` to 3%. That is a call count, not a code shape, and the
calls were the albedo: this renderer computed `visible_surface` -- a trace of
the camera ray, the surface geometry, the differentials, the material, and
`bsdf.rho` with its sixteen BSDF samples, each a layered walk -- for every
camera sample, whatever the film, and then the path integrator traced the same
ray and built the same BSDF again. pbrt does neither. Its `Li` fills the
`VisibleSurface` at the path's first vertex, from the intersection and the BSDF
the path itself goes on to use, and only when `Film::UsesVisibleSurface()`,
which is true of `gbuffer` and false of `rgb` -- the pavilion's film. The
renderer does the same now: `full_path_step` takes a `want_visible` flag and a
`visible : mut VisibleSurface` with pbrt's `set` member, fills it after
`GetBSDF` and before the depth test as pbrt does, the random walk and simple
path ignore it as pbrt's do, and `render` reads it after `Li` returns. The
scene carries `Film::UsesVisibleSurface()` from `scene_dump`, which now
records the film's type. The BSDF is also built at every vertex the path
reaches, the last one included, which is where pbrt builds it. 4.01 s against
5.09 s: **1.27x faster than pbrt**, radiance unchanged to the bit, and the
gbuffer scenes unchanged against pbrt -- three-spheres and area-light-mis at 0
disagreeing normals, albedo at 6.1e-05 and 4.3e-04 mean, radiance at 1.00012x.

**The layout language names an extern array's storage.** `layout
mesh_positions { tight(root); };` in the schedule stores the array's element
as exactly the bytes its fields state: a `vec3f` as three floats, twelve
bytes, where the vector the program computes with is the machine's sixteen.
`root` is the element of the topmost array -- the cursor the rewrite rules to
come (`split`, `interleave`, `deinterleave`) will refine -- and `tight` is the
first rule. The lowering (`Lower/Layouts.cpp`, `apply_array_layouts`) retypes
the extern, every parameter that carries it and every call that names such a
parameter, and widens each element read where it is read, the way a tree
layout's vector field already was; an extern is not assignable, so there is no
store to convert, and a `tight` array of structs is refused as unimplemented
rather than stored as the compute type behind a layout that says otherwise.
The driver hands over `std::array<float, 3>` and the generated header declares
it so. Positions and normals are twelve bytes apart now, pbrt's `Point3f` and
`Normal3f` stride, and the radiance is unchanged to the bit
(`lower/tight-array.bonsai`, `backends/llvm/tight-array.bonsai`,
`correctness/cpp/tight_array.bonsai`, `error/array-layout-*.bonsai`).

**What is left of the traversal, and what is next.** Of the six items the
previous round listed against the traversal, none is open: the last, the
running best carried as a copy, is closed under "What the last round added"
above -- the accumulator holds a reference now. The `--stats` counters,
deferred until the code generation matched pbrt, are the right next
measurement now that it does, and the dielectric should be re-profiled with
the albedo gone before anything is concluded about its code. The pre-existing
failure of the two `backends/cuda` goldens (`parallel`, `rtiow-primer`) is
LoopTransforms refusing a `bind` under the default pipeline, dates from before
this branch, and is untouched.

### What the round before that added

**The compiler, for the pavilion's speed.** Nothing in the renderer changed
this round. The gap to pbrt on the pavilion was measured with `perf stat` and
`perf record` and the compiler taken where the counts pointed, from 1.59x
slower to 1.41x: 7.42 s against pbrt's 5.27 s at 16 samples per pixel, best of
three on each side, with the image unchanged. Four changes, each measured on
its own, below in the order they were made.

**Aggregates come back the way the platform returns them.** The LLVM backend
returned every struct as a first-class aggregate, and LLVM's own convention for
one wider than two SSE registers is not the x86-64 ABI's: the third and fourth
float leaves come back on the x87 stack. An `Interval`, a `SampledSpectrum`, a
`PhaseSample` -- every such return was a store to memory and a load back
through a unit nothing else in the program touches, and the counters showed 4.1
billion x87 instructions in a render that should have had none. A struct of
more than two floating-point leaves or three integer ones is now written
through a hidden `sret` argument, on both the structured and the SSA paths, and
a tail call passes its own slot straight on rather than copying
(`CodeGen_LLVM::indirect_return_type`; `backends/llvm/sret.bonsai`).

**Common subexpressions are eliminated after inlining.** `opt::CSE` existed
and ran in no pipeline. It runs in `ssa` now, once `Inline` has put a tree
query's tests in one function: the traversal asked `distmin(r, g)` twice per
candidate, against the running best and again as the key, and for a triangle
each is a whole `triangle_hit`. LLVM could not merge the two because the
triangle is fetched from its pool between them and the traversal's own stack
stores stand between the fetches. One remains. Turning the pass on found that
its renaming visited a loop's condition before the loop's body, which shares
values with it, and stopped on meeting the body's copy (`opt/cse-loop.bonsai`).

**The relooper knows what is in scope.** CSE leaves a value under two names --
a temporary and the name the source gave it -- and the SSA builder makes that
one instruction and threads it into the blocks that want it under whichever
name each asked for. The structured-code generator assumed a block argument
always arrived under its own name: a branch into an arm bound nothing, a call
continuation bound only addresses, and where the two names met at a merge the
mutability analysis, which compares names, made the parameter storage that the
arm carrying the name along never wrote. Three miscompiles -- in `bump_map`,
`coated_f` and `full_path_step` -- and the last of them silent, a stack slot
read before anything had stored to it. The rule now is the one a structured
language has: a phi is a variable assigned at every edge into its block; any
other argument is bound once, at the block's entry, and only if nothing
enclosing has bound its name -- the relooper carries the set of names in scope
through the regions it builds, and follows an argument back through the blocks
that passed it on to the instruction it stands for, which is what a merge
after two arms has to name. The LLVM backend scopes the arms of an `if` as the
C++ one always did (`ssa/alias-edges.bonsai`,
`correctness/llvm/alias-edges.bonsai`). The stricter edges also caught
`collapse()` building a step block whose arguments its loop never passed, and
a block's recorded predecessors going stale under `split()`.
`BONSAI_DUMP_AFTER=<pass,...|all>` prints the program after any pass named,
which is how each of these was found.

**`max` and `min` are `std::max` and `std::min`.** They lowered to
`llvm.maxnum` and `llvm.minnum`, which are libm's `fmax` and `fmin`: a NaN in
either position is dropped. pbrt means `std::max` -- `a < b ? b : a`, a NaN in
front kept, one behind passed over -- in `SafeSqrt`, in every clamp, and in the
slab test its BVH runs at every node. And x86 has no maxnum: LLVM emitted a
compare for the NaN and a blend around every one, four to a node test, where
`std::max` is the one `maxss` instruction. 8.06 s became 7.51 s and no pixel
moved (`correctness/llvm/max-min-nan.bonsai`, which also found that -0.0 and
0.0 were one constant to the IR's equality and CSE folded them together).

**Small functions with statements are inlined, and CSE reaches through
them.** The inliner copied only functions that were a single expression, so
`intersects(r, box)` and `distmin(r, box)` stayed calls, and the traversal
computed the same slab test twice for every node the ray hit: once for the
filter, once for the key, with LLVM unable to merge the two inlined copies
across the early exits inside them. A function of a few statements whose
returns nest -- a `return` ending the body, or ending an arm of an `if` whose
other arm never returns, in which case the rest of the body moves into that
arm -- is copied to its call sites now, its returns becoming assignments to a
result variable; a call in a loop's condition, or behind a select or a
short-circuit, stays a call, since hoisting it would change when it runs. A
body with mutable locals of its own is left as a call too: copied, its state
is variables CSE cannot see through, while as a call it is one value two calls
share -- which is exactly `aabb_span`, kept as the call that `intersects` and
`distmin` both make and CSE then makes once. `[[noinline]]` says a function is
never copied, for the tests that are about calls and for programs that want
one copy. The pavilion went from 7.51 s to 7.42 s; the two `triangle_t` tests
a hit candidate still pays are behind separate `match` arms, and CSE does not
reach across those (`opt/inline-statements.bonsai`,
`correctness/llvm/inline-statements.bonsai`).

Turning that on found `loopify` mistaking an inlined bool for an `any`
accumulator: it looked at stores value by value, and a variable is one
instruction where it is made and a block argument everywhere else, so a `=
false` seen alone in one block passed for `all`, and the loop tested a slot
allocated inside its own body. It looks by name now, at variables that outlive
an iteration, and sees the load behind a block argument, which `_holds0 ||
hit` had always hidden from it (`ssa/ray-any-early-exit.bonsai` keeps its
early exit).

**A `vec3f` in a layout is twelve bytes.** A layout says how many bytes each
field is and lowers to exactly that -- no padding the compiler finds
convenient. It did not, for vectors: a `vec3f` field became LLVM's
`<3 x float>`, whose allocation size is sixteen, so the node this app's layout
spells out as 32 bytes -- `low`, `high`, `nPrims`, `axis`, a byte of padding
and the `u32` the switch stores, pbrt's `LinearBVHNode` exactly -- came out at
40, and three nodes in eight straddled a cache line where pbrt's never do. A
vector in a layout is a different type from the vector the program computes
with (`Vector_t::packed`): exactly its elements, aligned as one element is,
an `[N x T]` to the LLVM backend and a `std::array` to the C++ one, converted
to the compute vector where the field is read. The switch payload, which the
backends had special-cased to its exact width, is the same kind now, and
`Layout::bits()` counts every lane of a vector field, which is what the
padding in the layout had always assumed it did. The drivers write three
floats into a node. 7.42 s became 7.27 s against pbrt's 5.10.

**The traversal, measured.** Retired instructions, sampled on the same tree
as pbrt's: the closest-hit loop `_traverse_tree0` retires about 408 G and the
`any` loop about 81 G, against pbrt's `BVHAggregate::Intersect` at about
303 G plus 39 G of leaf dispatch -- and pbrt's `Intersect` also carries its
shadow rays, which `volpath` traces as closest-hit queries where this
renderer's are `any`. The triangle tests come out even: about 111 G here
(`triangle_hit`, twice for a hit candidate) against 108 G there
(`IntersectTriangle` and the interaction pbrt builds for every hit). In time
the loop is about 95 thread-seconds against 53: more instructions, and fewer
of them per cycle.

The query is not where the extra work is written. It says pbrt's traversal
and nothing more: the same tree (pbrt's own, under `PBRT_TREE=1`), the same
near-child-first order (the sort key *is* pbrt's `dirIsNeg[axis]` test), the
same pruning (`span.lo < best` is the `tMax` pbrt hands `IntersectP`), and
shadow rays that stop at the first hit where pbrt's do not. The extra
instructions are in what the lowering makes of it, node by node beside pbrt's
loop. `loopify` pushes both children and pops the near one on the next trip
-- two stores, two counter updates and a load per interior node -- where
pbrt pushes the far child once and continues with the near one. `sort()`
lowers to two float keys and a compare-and-swap of selects per node, where
pbrt reads a per-ray byte and branches; and the key's `(1/r.d)[axis]` is a
variable-lane extract at every node. A hit candidate is tested twice because
`intersects` and `distmin` are two `match`es on `Shape` and their common test
sits in separate arms. And the cycles per instruction: the node was 40 bytes
(fixed above), `mesh_positions : array[vec3f]` still has a 16-byte stride
against pbrt's twelve-byte `Point3f` -- the compute language's array of
vectors is storage too -- and the running best is four vector registers of
loop state where pbrt carries `tMax` and writes the interaction to memory on
a hit.

What is not yet known is the count: how many nodes this traversal visits and
how many candidates it tests, against pbrt's 5.83 G and 307 M. That is the
first thing to build -- a `--stats` the compiler emits for a traversal,
printed at exit, pbrt's own counters -- so that "no more work than pbrt" is
a number and not an inference. After the traversal, the dielectric path at
about 1.6x pbrt's time is the open question.

A note on `check_differentials.sh` on this scene: 982 of its 1083 rows are
exact and the rest are the `texrgb` and `texsr` rows of two textures declared
`float` over RGBA images, where pbrt's row filters the image's colour channels
and the shipped texture holds the alpha channel three times; the same rows
disagreed with the renderer built before this round, so they are the harness
comparing two different things and not a regression.

### What the round before that one added

**The pavilion matches, and what was in the way.** With every material and
instancing in, the whole scene rendered 1.036x too bright, with 37,000 more
lit pixels than pbrt, and the excess was one strip: the wall along the pool's
far edge, several times brighter than pbrt's. Cutting the pool out into a scene
of its own (`scenes/bump-lens.pbrt`, the four meshes and the sky under the
pavilion's own camera) and bisecting its features found three separate
things, none of them in the material:

- **A light leak.** `trace` and `trace_any` carried a `distmin > 0.001` guard
  from the first BVH commit, before hit points had error bounds. `t` is the
  parameter of the ray's own direction, and a shadow ray's direction is the
  whole way to the light, so a thousandth of it was six centimetres on this
  scene -- and the pool's walls are one-centimetre slabs. The wall's
  displacement is an 8-bit stone texture with no scale, whose finite
  differences tilt the shading normal nearly into the surface, so half its
  light samples point into the slab; pbrt's shadow rays hit the far face and
  stop, ours passed through and lit the wall from behind. pbrt has no such
  guard -- the spawn offset keeps a ray off the surface it left, and a shape's
  test takes any t in (0, tMax) -- and now neither does this. The wall went
  from 3.8x pbrt to 1.004x, and the pavilion from 1.036x to 1.0002x.
- **A lens camera's last bits.** The shading normals differed on 8,278
  pixels of the pool with the lens and on none with a pinhole. Three causes,
  each an ulp: pbrt puts a ray's origin through its interval point transform
  and nudges it by the error bound (`Transform::operator()(Ray)`), which is
  the exact zero for a pinhole in camera-world space and a point on the lens
  otherwise; gcc fuses `Transform::operator()(Vector3f)` with the *first*
  product in the fused add and `operator()(Point3f)` with the second, which
  agree whenever a row has a zero among its first two entries -- every
  axis-aligned camera, which is how it stayed hidden; and `ApplyInverse` on a
  point associates `(m0 x + m1 y) + (m2 z + m3)`. A bump map differences a
  texture over the footprint those rays give, on a wall seen at a grazing
  angle, so an ulp there is a different normal. `check_differentials.sh`
  checks all of it bit for bit now, through an off-centre lens sample.
- **`--disable-pixel-jitter` with a lens.** pbrt's `GetCameraSample` fixes
  the lens sample at the lens's centre under that option, after drawing it;
  this left it as drawn, and every single-sample comparison of a lens scene
  had each pixel's one ray start somewhere else on the lens than pbrt's.

**The texture chain is bit-exact, and the check says so.** `check_differentials.sh`
grew rows that put every texture the scene converted through pbrt's own
MIPMap and texture objects at three points and three footprints, and an RGB
through pbrt's own `RGBAlbedoSpectrum`, against this renderer's lookups. Four
things came out of the rows, each of which had been a plausible-looking
number before:

- `MIPMap::Bilerp<Float>` on a three-channel image averages the three
  *filtered* channels. The converter had averaged the texels and let the
  renderer filter the average, which is the same number only in exact
  arithmetic; the channels ship now and `texture_float` averages after the
  filter. `Texel<Float>` at the top of the pyramid reads channel 0 whatever
  the image has, and does now.
- a constant `scale` texture over an image texture **folds into the image's
  own scale**: `SpectrumScaledTexture::Create` copies the image texture and
  `MultiplyScale`s it rather than wrapping it, so the constant reaches the
  colour *before* the invert and the spectrum fit. This file's first reading
  of pbrt had it the other way, and the `texsr` row -- pbrt's spectrum
  recomposed from its own filtered colour -- is what caught it: it agreed with
  the renderer, and neither agreed with pbrt's texture.
- the table lookup's remapped coordinate is `rgb * (res - 1) / z`, product
  then divide; scaling by `(res - 1) / z` rounds in the other order, and a
  near-grey colour sits where the table's coefficients change fastest, so the
  last bit of it was a 13% difference in the spectrum at 830 nm.
- `Lerp` fuses its first product, standalone and nested three deep -- measured
  by compiling the expression with pbrt's compiler and flags.

**The gbuffer compares the shading normal too.** pbrt writes `N` and `Ns`;
only `N` was being read, and a displacement moves nothing `N` can see, so the
bump map had never been checked against pbrt at all. `compare_gbuffer.py`
takes `--shading` now and excuses a disagreement only where the *geometric*
image has an edge, since a bumped surface is discontinuous at every pixel by
design. A pixel in 130,000 still flips at eight samples per pixel: the
pyramid level is the floor of a log of the footprint, and one sample landing
on the other side of that floor is a different normal. The check tolerates one
in ten thousand, four orders below what the lens bug did.

What is left on the pavilion, measured at 16 samples per pixel: 14.5% of
touched pixels agree to 1e-3 and the lit-pixel counts differ by 0.47%, both of
which are per-path divergence and not bias -- in the pool, 48,000 pixels are
lit only in pbrt and 44,000 only here, over a mean that agrees to 1.0007x. The
paths part on the instanced leaves (the `ApplyInverse` nudge this renderer
deliberately does not do) and on the water's bump. A `coateddiffuse` under a
texture or a bump still shows per-pixel albedo noise against pbrt (0.2% of
pixels over 5e-3 without a bump, 6% with) for the reason the layered BSDF
always has: its walk is seeded by hashing the local direction, and the mean
converges. And it was 1.59x slower than pbrt on this scene when this round
ended, which is what the round after it, above, took up.

### What the rounds before those added

**Named materials, and `dielectric`.** A shape under `NamedMaterial` names its
material by string, and pbrt leaves `materialIndex` at -1 -- which is also what
a shape with no `Material` directive gets. The two are told apart now;
`MakeNamedMaterial` is captured with its `"string type"` lifted into the same
field a `Material` directive fills, and converted on first use as the indexed
ones are.

`dielectric` is glass, and it is cheap because `DielectricBxDF` has been in
`bxdf.bonsai` and checked against pbrt since `coateddiffuse` went in -- it is
the coating of a layered material, on its own. What is new is the *path* it
takes through an integrator, and `scenes/dielectric.pbrt` is there to exercise
it: it is the first **specular** surface here, so `IsNonSpecular` is false and
`path` spends no shadow ray and no draws on it and finds its light by
scattering; and the first **transmissive** one, so `etaScale` stops being one
and `SampleLd` takes the other arm of its nudge. Two spheres, one smooth and one
rough, take opposite branches of every `EffectivelySmooth` test.

It is the best-agreeing lit scene in the collection: **0 albedo pixels** outside
the comparison's tolerance -- the first lit scene where that check passes at all
-- and 94.3% of pixels within 1e-3 on the radiance, against 45-78% for the
others. That is what a material with no stochastic BSDF walk in it looks like.

**And a third silent substitution, in the same place as the first two.** The
scene file's material writer was `tag == CoatedDiffuse ? "coateddiffuse" :
"diffuse"`, so the first material added after it serialized as a diffuse and
rendered as pbrt's default grey -- with the converter, the renderer and the BxDF
all correct and only the file between them lying. It is a `switch` with no
default now, so a tag with no case is an error rather than a guess. Three of
these have now been found in three consecutive rounds, all of the same shape:
somewhere in the chain, an unknown case falls through to a plausible one.

**Sampling an environment map, which is what turns it from a background into a
light.** `SampleLi` draws a direction from a `PiecewiseConstant2D` built across
the map's texels -- so a bright patch of sky is found often and an even one
rarely -- and `PDF_Li` answers the density at a direction, which is what MIS
weighs the BSDF sample against. The density comes back in the square's measure
and is divided by `4pi` to become one in solid angle; that single constant is
the whole Jacobian, and it is why the map has to be equal-area.

There are **two** distributions, not one, and the second is the interesting
part. Under `allowIncompletePDF` pbrt samples a *compensated* table built over
`max(0, texel - average)`: the part of the sky that is brighter than average.
The reasoning is that a BSDF sample finds an even sky perfectly well on its own,
so the light sampler should spend its effort on what the BSDF sample is bad at.
`path` takes that table and `simplepath` takes the plain one, so the two scenes
below reach the same sky through different tables:

    envmap.pbrt       path        1.00006x   45.4% of pixels within 1e-3
    envmap-walk.pbrt  randomwalk  1.00009x   24.8%
    (simplepath)                  1.00031x   64.6%

The agreement percentages are the point of the feature rather than a caveat:
the random walk finds the sky only by flying into it, so every escaped ray lands
on a different texel of a high-frequency map. Sampling it is what makes the
picture converge.

`sampling.bonsai` gained `EqualAreaSquareToSphere` and a general
`PiecewiseConstant2D` -- general because two tables over one image want it, and
because the pixel filter already had the 1D half.

**Environment maps, as far as being able to see one.** `LightSource "infinite"`
with an image is pbrt's `ImageInfiniteLight`, and it is what every real infinite
light in `pbrt-v4-scenes` turns out to be. A ray that leaves the scene now reads
the map: the direction goes back into the light's own frame, the equal-area
octahedral mapping unfolds the sphere onto the square, and a nearest lookup with
pbrt's octahedral wrap picks the texel. Against pbrt on a real 2048x2048 map
placed by a rotation, the means agree to **1.00009x**.

Three things worth writing down:

- **The texels are fitted spectra, not RGB.** pbrt builds an
  `RGBIlluminantSpectrum` inside `ImageLe`, once per lookup. That is a
  deterministic function of the texel, so the driver does it once per texel
  instead -- four million times at load rather than once per escaped ray -- and
  gets the same numbers. It is the same division of labour as every other
  spectrum here, and it is why the rgb2spec tables are in the driver.
- **The transform had to be followed rather than asked for.** pbrt's
  `renderFromLight` is protected on the light and `RenderFromObject` is private
  on the builder, so `scene_dump` now mirrors the CTM: every transform directive
  is a `ParserTarget` virtual, so it sees the same calls in the same order and
  applies them with pbrt's own `Transform` arithmetic. Only the bookkeeping is
  reimplemented, only the world-space half is needed, and a wrong rotation puts
  the sky visibly in the wrong place -- so this is not a quantity that can be
  subtly wrong.
- **The map is a 24 MB sidecar**, written beside the scene file rather than into
  it. Twelve million floats do not belong in a text format.

**A second silent substitution, found the same way as the light one.** A shape
under `NamedMaterial` gets `materialIndex = -1` from pbrt, which is also what a
shape declared outside any `Material` directive gets -- so a
`MakeNamedMaterial` of any type at all quietly became pbrt's default fifty-per
cent grey diffuse. It surfaced while trying to use lte-orb as a test scene: its
`measured` BSDF rendered as a grey ball five times too bright, with the geometry
and the normals matching perfectly throughout. Refused now, and it means the
survey's material counts are understated.

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

**The survey was re-run after infinite lights went in**, and the shape of it
changed more than the headline did. Still two scenes convert -- but thirteen of
the fifteen that used to were converting with their light silently dropped, and
an unknown number with a named material silently turned to grey. What the
collection refuses on now:

    a portal on an infinite light                           25
    `Integrator "volpath"`                                  24
    a named material                                        11
    an environment map in a colour space other than sRGB    11
    a material parameter that is a texture or a spectrum     7
    a `distant` light                                        3
    a sampler in the Sobol family                            3
    a material other than diffuse/coateddiffuse              3
    `bdpt`, `sppm`                                           1 each

Two of those are new and both were previously *silent*: the named material, and
the colour space. The second is worth a note -- eleven scenes' `sky.exr`
declares **ACES**, and `RGBIlluminantSpectrum` over ACES uses that space's
illuminant and its own fit table, so reading one as sRGB would be a sky of the
wrong colour. Checked with `imgtool info` rather than assumed.

The scenes closest to working are `villa-daylight`, `sanmiguel-*` and `clouds`,
which want only the ACES colour space.

### What `barcelona-pavilion/pavilion-day.pbrt` needs, exactly

Worth writing down because it is the scene that looks closest and is not.
Named materials went in and it moved to the next wall; the wall behind that one
is five features deep. Its 28 `NamedMaterial` uses resolve to:

    metal                 21 uses   conductor, `spectrum eta`/`spectrum k`
    pavet                 19        coateddiffuse + reflectance and *displacement* textures
    concrete_Mies...png   15        coateddiffuse + a reflectance texture
    wood                  12        coateddiffuse + a reflectance texture
    leather_white         12        measured, from a .bsdf file
    wax                    6        coateddiffuse            -- works
    glass_architectural    6        dielectric               -- works now
    concrete               3        coateddiffuse + a texture
    white_mat, None,
    black_glossy, Material 7        diffuse / coateddiffuse  -- work
    marmol_verde, marble,
    pebbles.ground, grass  5        coateddiffuse + textures
    water                  1        coateddiffuse + a displacement texture
    xref_*                14        leaf materials, declared in the geometry

Walked forward by actually running the converter after each one. What it stops
on now, in the order it stopped:

1. ~~a material parameter that is a texture~~ — **done**
2. ~~`lensradius`~~ — **done**, both branches of `GenerateRayDifferential`
3. ~~`displacement`~~ — **done**, PBRT's `BumpMap`
4. ~~`conductor`~~ — **done**, with pbrt's named metal spectra
5. ~~`measured`~~ — **done**, PiecewiseLinear2D and all
6. ~~`iso` and `maxcomponentvalue`~~ — **done**; both were silently ignored
7. ~~`ObjectInstance`~~ — **done**, as pbrt's TransformedPrimitive beside its
   GeometricPrimitives in one tree; see below
8. ~~`Material "diffusetransmission"`~~ — **done**, pbrt's
   DiffuseTransmissionBxDF; `scenes/diffuse-transmission.pbrt` matches

**All five material walls are down**, and so are instancing and the leaves'
material. `pavilion-day.pbrt` converts end to end: 133,575 top-level shapes and
43 instances of 2 objects holding 5,178,144 triangles, in an 823 MB scene file
(the format is text, and 5 million triangles in text is what that costs). It
renders to 0.99967x of pbrt's mean; how it got there from 1.036x is under "what
the last round added".

pbrt keeps instanced geometry out of `BasicScene::shapes` entirely -- it is in
`instanceDefinitions`, and `instances` names one with a transform. This
converter used to read only the first list, so every tree in pavilion was simply
not in the scene, which rendered 1.49x too bright with ninety thousand pixels
lit that pbrt leaves dark. It reads both now.

#### What pbrt does with the second list

`BasicScene::CreateAggregate` (scene.cpp:1521-1580), and there is no flattening
anywhere in it:

- each **definition** becomes one `Primitive`. Its shapes are built into a
  `BVHAggregate` of their own when there is more than one, and used directly
  when there is exactly one; an empty definition becomes null and every
  instance of it is skipped.
- each **instance** becomes a `TransformedPrimitive` holding *that same*
  definition and a `renderFromInstance`. Forty-three placements of a tree share
  one tree; the geometry is stored once.
- those go into the top-level `primitives` list beside the non-instanced ones,
  and the scene's accelerator is built over the mixture. The top-level BVH does
  not know a `TransformedPrimitive` is special -- it is a `Primitive` with
  `Bounds()`, `Intersect()` and `IntersectP()` like any other, and the fact
  that answering those runs a second traversal is entirely inside it.

`TransformedPrimitive::Bounds()` is `renderFromPrimitive(primitive.Bounds())`,
the eight-corner box transform. `Intersect(r, tMax)` (primitive.cpp:112-125)
pulls the ray back with `renderFromPrimitive->ApplyInverse(r, &tMax)`,
intersects the definition with it, and pushes the interaction it gets back
forward again.

The `tMax` argument is the part worth being careful about, because everything
above it is only a container and this is where the pruning lives.
`ApplyInverse` does *not* renormalise the direction (transform.h:416), so a hit
at parametric `t` in instance space is at the same `t` in render space: `tMax`
means the same thing on both sides of the frame change, and an instance
therefore prunes against a hit already found in a *different* instance. Passing
a fresh `Infinity` per instance would give the same image and lose most of what
the top-level tree is for.

#### What that is here

A tree whose elements hold trees, which needed nothing new in the query
language and two things in the ADT language -- a `set`-typed field, and reading
a nested tree's root augmentation so an instance's extent is derived rather
than asserted beside it:

```
element Instance {
    render_from_instance : Transform;
    blas : set[Triangle];
} with extent = transform(render_from_instance, blas.AABB);
```

The searched set is every triangle of every instance -- `flatten(|i| i.blas,
instances)`, the triangles as each instance's tree holds them -- and the tests
place them: `intersects(r, transform(i.render_from_instance, tri))`, the
triangle where the instance puts it, against the world-space ray. Note the
direction: pbrt says `intersects(untransform(m, r), tri)`, the ray moved back.
Said our way there is one uniform ray and one varying geometric object, and the
ordinary bounding rule applies to it; said pbrt's way the frame change hides
inside a function predicate analysis cannot see into, and the triangle's box
and the node's box are in different spaces with nothing relating them.

Getting from here to pbrt's arithmetic is `pull-queries` (Opt/PullQueries.h).
`transform` and `untransform` are inverse motions, so a relation against a
moved extent is the same relation of the counter-moved query; once the motion
is on the ray it has one value per instance, and it is hoisted there -- pbrt's
`ApplyInverse` at the top of `TransformedPrimitive::Intersect`, with every test
inside the instance's tree then against stored geometry and the sort keys
carried into the same frame. The result is the pair `(instance, triangle)` in
the instance's frame, which is what pbrt's inner `Intersect` hands back, and
the one transform of the result is the caller's -- as it is pbrt's, in
`TransformedPrimitive`. Writing the set as the *placed* triangles instead,
`map(|tri| transform(m, tri), i.blas)`, asks for a placed triangle back, and a
reduction has to store the element of the set it ranges over: a transform per
recorded hit, for a result pbrt never computes. So the set is the raw
triangles, and the motion lives in the tests.

Storage is Scion's: every instance's tree is rows of one shared `indirect group`
in the enclosing layout, so two instances naming the same row share a subtree
and the object is stored once however often it appears.

    Instance.blas : BLAS from BlasNodes;

**The compiler side of this is done**, with the two-level structure lowering,
laying out, loopifying and sorting at both levels and the ray pulled back once
per instance: `tests/bonsai/lower/nested-tree-layout.bonsai`,
`tests/bonsai/lower/pull-the-ray-back.bonsai`,
`tests/bonsai/ssa/{loopify,sort}-nested.bonsai`,
`tests/bonsai/backends/llvm/tree-traversal-{nested,instanced}.bonsai` and
`tests/bonsai/correctness/cpp/blas-tlas{,-loopified,-sorted}.bonsai`.

#### The app has it

pbrt's top-level tree holds GeometricPrimitives *beside* TransformedPrimitives,
and that needed one more thing from the language: a tree whose elements are of
two kinds. `render.bonsai` now says exactly pbrt's structure:

```
element Geometric { shape; light; material; alpha }          // GeometricPrimitive
element Primitive =                                          // Primitive
    | Geom(g : Geometric)
    | Inst(render_from_instance : Transform, instance_from_render : Transform,
           blas : set[Geometric]);                           // TransformedPrimitive
func geometry(p : Primitive) -> set[Geometric] =             // what Intersect dispatches to
    match p { Geom(g) => set[Geometric]{g}, Inst(m, mi, blas) => blas };
element Primitive with extent = |p| transform(p, geometry(p).AABB);  // Primitive::Bounds()
```

The primitive itself is the motion -- `transform(p, ..)` and `untransform(p,
..)` are a match on `p` that moves for an instance and holds still for plain
geometry, pbrt's `renderFromPrimitive->...` lines and GeometricPrimitive's
nothing -- and `trace` is the query over `flatten(|p| geometry(p),
primitives)` with the tests on `transform(p, g)`. The compiler opens the match
into the leaf's element loop, so what runs is pbrt's dispatch: a tag test per
primitive, then the shape test against the ray as it arrived, or `ApplyInverse`
once and the walk of the instance's tree against the pulled ray. A hit is the
pair as stored, and `transform(via, surface_geometry(prim.shape,
untransform(via, ray)))` at each call site is TransformedPrimitive's transform
of the interaction, pbrt's `Transform::operator()(SurfaceInteraction)` field
for field. docs/trees-of-mixed-primitives.md is the compiler side of it;
`tests/bonsai/correctness/cpp/mixed-tlas{,-loopified,-sorted}.bonsai` run it.

`scene_dump` emits `instanceDefinitions` as runs of one `instance_shapes` list
and `instances` naming a definition with both of pbrt's matrices; with
`--pbrt-tree` it builds a BVHAggregate per definition and the top-level one over
the mixture, exactly as `CreateAggregate` does, and writes the leaf order out
as `prims`. `render_hook.cpp` builds or adopts both kinds of tree and fills the
two `tagged_index` pools. `apps/pbrt/scenes/instances.pbrt` -- one object placed
by a translation, a rotation and a scale, in front of and behind plain geometry
-- matches pbrt on **0 pixels** (worst normal difference 9.8e-6), with the
driver's trees and with pbrt's.

Three places where this and pbrt differ, all small and all deliberate:

- `ApplyInverse(Ray, tMax)` nudges the pulled ray's origin forward by its
  rounding error and takes the same `dt` off `tMax`. Neither is done here:
  `tMax` is the traversal's accumulator, which nothing in a query may alter,
  and without the nudge the identity the rewrite rests on -- a distance along
  the pulled ray is the distance along the original -- holds exactly. The hit
  point differs from pbrt's by `dt`, a few units in the last place.
- pbrt builds no tree over a definition of one shape; the shape is the
  primitive. The layout wants a row to start an instance's walk at, so such a
  definition gets a tree of one leaf: one box test more than pbrt per instance
  of it.
- AnimatedPrimitive -- an instance whose transform moves over the frame -- is
  refused, as are animated shapes inside a definition.

**What it cost on a scene with no instances, and what paid it back.** The
first build of this was +7.5% on `many-shapes` against the build before it
(0.326 s to 0.351 s, best of five, same scene file: +10.8% instructions, +36% L1
misses). Two things, found by reading the LLVM IR of the leaf loop rather than
by guessing:

- `Primitive` was stored `tagged_index`, the layout pbrt's TaggedPointer is: a
  word per primitive in the leaf and each arm's fields in a pool. Right for an
  instance, two matrices and a tree; but it put pbrt's own indirection --
  handle, `Primitive_Geom_pool[i]`, shape -- in front of every *plain*
  primitive, where the inline struct had one hop fewer. The layout language
  now says per arm what becomes of it: `layout Primitive { Geom = inline; Inst
  = tagged_index; }`, a plain primitive its own twenty bytes in the leaf and an
  instance a row number. pbrt cannot do this; its TaggedPointer is one shape
  for every arm.
- The bigger half was not the indirection. The query's tests are each written
  over `transform(p, g)`, and the pull rewrites each into
  `intersects(untransform(p, r), g)`, `distmin(untransform(p, r), g)`, and so
  on: four copies of the same term in the Geom arm. Inside an instance the
  term is hoisted once ahead of the walk; the Geom arm has no walk to hoist
  ahead of, so the four copies stayed, LLVM inlined each -- correctly folding
  each to `r` -- and then could not prove the four `r`s it had made were one
  value. `triangle_hit` ran three times per plain primitive instead of twice.
  `pull-queries` now binds each distinct pulled term once at the top of the
  match arm that uses it, which is what `Primitive::Intersect` has in each of
  its arms: one ray on entry, every test over it.

With both: 0.319 s and 0.319 s, the same scene, alternating runs. The generated
Geom arm is one tag test, then the shape test once on the ray as it arrived.

Behind it, on the leaves: `Material "diffusetransmission"` -- **done** -- and
the shape `alpha` cutouts, which were already implemented (pbrt's stochastic
test, which for a triangle reduces to a condition on the set because its retry
can only miss) and now have five million triangles to act on.

The original list, in the order they block it:

1. **Image textures — done.** Eight of its materials, and 27 scenes in the
   survey. The chain below is built: ray differentials, the camera
   approximation, the MIP pyramid built by PBRT and shipped, the level pick and
   the bilerp, the RGB-to-spectrum fit after filtering, `scale` textures folded
   into the image's own scale, and the `uscale/vscale/udelta/vdelta` mapping.
   Checked against pbrt on a textured floor at a glancing angle -- 99.9% of lit
   pixels within 1e-3, which is where an untextured scene sits at the same
   sample count.
2. **`conductor` — done.** 21 uses, keyed on `metal-Al-eta` and `metal-Al-k`,
   which are pbrt's *named spectra*. Checked against pbrt on a smooth aluminium
   sphere and a rough copper one, so both branches of `Sample_f` run: 0
   disagreeing pixels.

   This item used to say a spectral index drags in `TerminateSecondary`. It
   does not, and reading `ConductorMaterial::GetBxDF` is what settled it: only
   a *dielectric* terminates, and only for a non-constant eta. A dielectric
   bends the ray by an amount that differs per wavelength, so the four can no
   longer share a path; a conductor only absorbs differently, and they can.
3. **`measured`**, 12 uses. A reader for pbrt's tabulated BSDF format and the
   interpolation over it. Its own project.
4. **Displacement textures — done.** PBRT's `BumpMap` on `pavet` and `water`:
   the displacement is sampled at the hit and one step along each of `u` and
   `v`, the step being the footprint the differentials give, and the shading
   frame is rebuilt from the two perturbed derivatives. A *normal* map is still
   refused, since it replaces the shading normal rather than tilting it.
5. The `xref_*` leaf materials, which are declared in `geometry.pbrt` rather
   than `materials.pbrt` and have not been looked at.

None of that is a reason not to do it, but it is four rounds and not one, and
the first of them is the one that unblocks everything else.

### Image textures: the whole chain, mapped

Worth writing down in full because the research is the expensive part and it
has been done. A texture lookup is one line of pbrt --
`mipmap->Filter(st, {dsdx, dtdx}, {dsdy, dtdy})` -- and every argument of it is
a feature.

**Done.** `uv`, `dpdu` and `dpdv` are on the hit and agree with pbrt exactly,
checked by `--print-shading` against the `shading-frame` golden. `uv` is where
a textured material is asked about; `dpdu` and `dpdv` are what turn a
screen-space footprint into a texture-space one. Note they are the *geometric*
pair, not `dpdus`, which has been orthogonalized against the shading normal --
pbrt keeps both and filters in the geometric one.

**Ray differentials -- done.** `RayDifferential` is a companion to `Ray` and not
fields on it, so the 48 bytes stay off what the BVH copies; the camera generates
the two offset rays (`PerspectiveCamera::GenerateRayDifferential`, the
`lensRadius == 0` branch, and a lens is now refused rather than silently
rendered sharp); `RenderCPU`'s `ScaleDifferentials` is applied before tracing;
`FindMinimumDifferentials` runs in scene_dump against pbrt's own camera, since
the four vectors it produces are protected on `CameraBase`; and
`ComputeDifferentials` has both of its branches -- the ray's own differentials,
and `Approximate_dp_dxy` for every hit after a non-specular bounce.

Checked by `apps/pbrt/check_differentials.sh`, which prints the same four camera
rays and the same synthetic hit from both sides. All four camera rays and their
scaled differentials agree bit for bit. The derivatives agree to about four
digits, and no further because they cannot: `Approximate_dp_dxy` ends in
`px - pDownZ`, a difference of two vectors of magnitude `|p_camera|` that agree
to 1e-4, which cancels 4.6 of float32's 7.2 digits. pbrt's own answer has no
more digits than ours does. The script asserts the camera rays exactly and the
derivatives to 5e-3, which is that floor.

**Differentials through a specular bounce -- also done.**
`SurfaceInteraction::SpawnRay` carries them across `SpecularReflection` and
`SpecularTransmission` and drops them across everything else, which needed
`shading.dndu` and `shading.dndv` on the hit. Those are now computed where pbrt
computes them: from the sphere's first and second fundamental forms, and for a
triangle from the same 2x2 inversion that gives `dpdu` and `dpdv`, with the
vertex-normal deltas in place of the position ones -- including pbrt's
degenerate-UV branch, which it takes the trouble to answer rather than zero
precisely so that a reflection off such a triangle still has a sensible
footprint. A mesh with no vertex normals gets zero, since its shading normal
does not turn within a face.

The check covers it: three lobes at each hit, the two specular ones and a
glossy one. The specular origins come out bit-exact and the directions within
two or three ulps; the glossy one drops the differentials, and so does every
lobe when nothing came in, which is as much of the answer as the vectors are.

Without this a texture seen *through* glass -- pavilion's
`glass_architectural`, six uses -- would have been filtered at the camera's
minimum footprint rather than the one the refraction actually produces.

**What it was, for the record: the gate.** `Filter` needs a width, the width
comes from `dudx, dvdx, dudy, dvdy`, and those come from
`SurfaceInteraction::ComputeDifferentials`, which has two branches:

- With a differential ray, it intersects `rxOrigin + t * rxDirection` with the
  tangent plane at the hit and takes `dpdx = px - p`. So `Ray` needs a
  companion carrying `rxOrigin/rxDirection/ryOrigin/ryDirection` and a flag.
  It must be a *companion* and not fields on `Ray`: pbrt separates
  `RayDifferential` from `Ray` for the same reason, and putting 48 more bytes
  on the ray the BVH traversal copies would be a real cost for something the
  traversal never reads.
- Without one -- which is every vertex after a non-specular bounce, because
  `SurfaceInteraction::SpawnRay` only propagates differentials through specular
  reflection and transmission -- it calls `Camera::Approximate_dp_dxy`. That
  wants `RotateFromTo`, the camera's `CameraFromRender`, and four vectors
  `minPosDifferentialX/Y` and `minDirDifferentialX/Y` that pbrt finds by
  sampling its own camera at 512 points in `FindMinimumDifferentials`. Those
  four are setup and belong in the driver, computed with pbrt's own
  `GenerateRayDifferential` -- the same division of labour as the spectral fits
  and the BVH.

Then `dpdx, dpdy` become `dudx …` by solving the least-squares fit against
`dpdu, dpdv`, which is where the two derivatives above earn their keep.

Two smaller things go with it. The camera has to generate differentials at all
(`rxDirection = Normalize(pCamera + dxCamera)`, with `dxCamera` a constant of
the camera), and `RenderCPU` scales them by `max(0.125, 1/sqrt(spp))` before
tracing. That second one used to be written here as something the harness owed,
back when `render_reference` was the thing being compared against; the harness
is the `pbrt` binary now and does it itself, so it is simply a step this
renderer has to have, in the same place pbrt has it -- between generating the
camera ray and tracing it.

**Then the texture itself**, and the good news is that most of it is setup. The
default filter is `bilinear` over a MIP pyramid, not EWA, so a lookup is: pick
a level from the width, and bilerp at that level with the wrap mode. Building
the pyramid is pbrt's `Image::GenerateMIPMap` with its own resampling filter,
and that belongs in `scene_dump`, which links pbrt and already reads the image
-- the levels ship in the sidecar beside the environment maps. The renderer
then does the lookup and nothing else, which is the part that has to be
transcribed and the part that is small.

What is left after that is plumbing: a material parameter becomes either a
constant or a texture reference, `scale` textures compose over image ones, the
`uscale/vscale/udelta/vdelta` mapping applies, and PNG files are sRGB-encoded
where EXR files are linear.

The survey found one outright bug and it is fixed: **a `LightSource` that was
not an area light was silently dropped.** Thirteen of the fifteen scenes that
got as far as converting were lit by one -- almost all `infinite`, an
environment map -- so they converted without complaint and would have rendered
black. That is precisely the failure this app exists to refuse, and it went
unnoticed because no scene in `scenes/` has a light that is not a sphere. It is
a refusal now, which is why the headline number above is two rather than
fifteen.

Read in dependency order rather than by count, the road is:

1. **`infinite` lights — done.** `UniformInfiniteLight` and
   `ImageInfiniteLight`, seen *and* sampled: an escaped ray reads the right
   texel through the right rotation, and `SampleLi` draws a direction from a
   `PiecewiseConstant2D` over the map, including pbrt's MIS-compensated second
   table. Against pbrt on a real 2048x2048 map, 1.00006x under `path` and
   1.00031x under `simplepath`. What is left of the family is
   `PortalImageInfiniteLight`, which is now the single most common refusal in
   the collection at 25 scenes -- it was hidden behind the dropped-light bug
   until this round.
2. **Textures**, the same machinery pointed at materials. 27 scenes ask for a
   material parameter that is not a plain RGB, and a fifth of a scene's look is
   in them.
3. **Materials**: `dielectric` first -- 7 scenes, and every piece of it is
   already written, since `coateddiffuse` is a dielectric over a diffuse and the
   coating *is* `DielectricBxDF`. `conductor` is one more BxDF. `hair` and
   `subsurface` are their own projects.
4. **Shapes**: `bilinearmesh` alone is 20 scenes and is also what the PLY quad
   refusal is about, so the two close together. `disk` is an afternoon.
5. **Triangle area lights** -- *done*. `Triangle::Sample` towards a point and
   its PDF are Arvo's spherical-triangle sampling with a cosine warp
   (shapes.bonsai), and each triangle of an emissive mesh is one
   DiffuseAreaLight as it is in pbrt. `scenes/mesh-light.pbrt`, a one-triangle
   emitter, matches pbrt's `path` to 1.00002x of its mean. It took an `asin`
   intrinsic, added beside `acos`. A mesh light with more than one emitting
   triangle needs the BVH light sampler, which is now in (item 3): pbrt's `path`
   samples several lights by importance, and `ganesha` renders and matches.
6. **`volpath`**, which is 51 scenes by name and rather fewer by need: it is
   pbrt's default, so a scene names it whether or not it has any medium in it.
   The ones that genuinely need media are the cloud and smoke scenes. Every
   other scene naming it wants `path` and would agree with `path`; making that
   substitution *honestly* means implementing volpath, since a scene with no
   medium is a case volpath handles rather than a case that is not volpath.
7. **Samplers**, 17 scenes, and the cheapest item on this list per scene
   unblocked -- the Sobol family shares one matrix-based construction and
   `zsobol` alone is 12.
8. **A named camera sensor** -- *done*. A scene that names a real camera's
   sensor rather than the default `cie1931` -- `canon_eos_100d` is the common
   one, 16 scenes across the collection -- records a radiance through the
   camera's own spectral response curves and a sensor-to-XYZ matrix pbrt fits by
   least squares over the Macbeth chart, not the CIE curves and `RGBFromXYZ`.
   `scene_dump` builds the sensor with pbrt's own `PixelSensor` and ships its
   three curves and its `RGBFromXYZ * XYZFromSensorRGB`, so the fit is pbrt's;
   the renderer applies them in `radiance_to_rgb` (the reflectance path is
   untouched, since pbrt does not run albedo through the sensor). The default
   sensor takes the same path, its curves being X/Y/Z and its matrix pbrt's
   exact `RGBFromXYZ` -- which replaced a hand-rounded copy and moved
   killeroo-simple's mean from 1.0000x to 0.9999x of pbrt's. This unblocked the
   `pbrt-book` scene, which also needed one more thing: a `scale` texture whose
   factor is a `constant` texture rather than a literal, which folds the same
   way. `scenes/sensor.pbrt`, a named sensor on a simple scene, matches pbrt to
   99.9% of its pixels at a thousandth relative; `book.pbrt` matches to 1.00007x
   of its mean.

## What is next, in order

Numbered by how long they have been open rather than by what to do first. Items
1, 2, 5 and 6 are answered.

These are the items that came out of comparing against pbrt on the scenes that
already work. They are not the same list as the section above, which is about
scenes that do not work yet, and that list is the more important of the two —
what to do first is an `infinite` light. The items below are what the comparison
itself still owes: item 3, the light samplers, is answered — `path` runs pbrt's
BVH sampler now — and item 4 is a profile of a path rather than of a walk.

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

### 3. The light samplers `path` actually defaults to — done

`path` now runs pbrt's `BVHLightSampler`, the one pbrt's `path` defaults to, so
a scene with several lights compares rather than being refused. `bvh-lights`
(three separated triangle emitters) and `ganesha` (four mesh emitters and a sky)
both match pbrt on their radiance; `bvh-lights` matches on its normals and
albedo exactly and is 1.7x faster than pbrt.

The tree is pbrt's own, not a copy of how pbrt builds one. `scene_dump`
constructs a real `pbrt::DiffuseAreaLight` for each emitter and hands the set to
pbrt's `BVHLightSampler` constructor, then serializes the tree it built — its
nodes read back through pbrt's dequantizing accessors, and a bit trail per light
read from its `lightToBitTrail`. So the surface-area-heuristic split, the twelve
buckets, the octahedral and sixteen-bit quantization and the topology are all
pbrt's code running on pbrt's lights, and a future project on light-tree
construction replaces exactly the constructor call and nothing else. Reading
those private members takes a `#define private public` over pbrt's
`lightsamplers.h` alone, its dependencies included normally first so only that
header's own definitions are opened.

The renderer walks the serialized tree: `light_node_importance` is
`CompactLightBounds::Importance` on plain floats, `bvh_descend` is the
`SampleDiscrete` walk down to a leaf, and `bvh_pmf_bounded` recovers a light's
density from its bit trail. The infinite lights split off with pbrt's fixed
`pInfinite` before the walk. The two samplers still coincide on a single bounded
light — a one-node tree has PMF 1 and the pInfinite split is uniform — so
`scene_dump` keeps such a scene on the cheaper uniform arm, which is why a
single mesh or sphere emitter stays exactly as it compared before.

A light's slot in the renderer's list is its *ordinal*, the stable per-emitter
number `scene_dump` assigns in scene order and carries on the shape through the
driver's BVH reorder, so a tree leaf and the `lights[]` it names line up. That
ordinal is also the order pbrt builds its own area-light list in, so the uniform
sampler picks the same light too.

What is not yet in: `PowerLightSampler` (pbrt's other non-uniform sampler, an
alias table over each light's `Phi`) and area lights with an emission image or a
`power`, which `scene_dump` still refuses.

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

### 6. The traversal is a packet — done, by the order of two directives

**Done.** Profiling book at depth 1 -- camera rays only, sixteen lanes -- had
put two thirds of the vectorized time inside gather instructions at an IPC of
0.66: a gang at depth 1 is sixteen samples of one pixel, its lanes are at the
same BVH node nearly every step, and each gather fetched one address sixteen
times. The plan here was a `specialize(uniform(...))` directive with the
per-lane traversal as its fallback. What was built instead is simpler and is
the classic answer: the whole traversal is a packet, and it falls out of the
order the schedule already had the words for.

    render.split(s, s_gang, s_lane, 16, false).vectorize(s_lane);
    trace.sort(primitives.Interior, ...).sort(Inst.blas.Interior, ...).loopify(64);
    trace_any.loopify(64);

Loopified first and then vectorized, a traversal is a loop each lane walks at
its own pace over a stack of its own -- the per-lane form, still available by
writing it that way. The three ways of running the program are three files,
`apps/pbrt/schedules/{packet,perlane,scalar}.bonsai`, each compiled beside
`render.bonsai` as a second input (`-i render.bonsai -i schedules/packet.bonsai`;
`compare.sh --schedule perlane`), and `eval/render_matrix.py` renders a scene
over a grid of depths and sample counts under all three and pbrt, checks
every image against pbrt's, and plots the speedups. Vectorized
first, the recursion is made by the gang as a
whole: the callee is specialized for one node the lanes share and a mask of
the lanes whose ray reached it, and `loopify()` then puts that recursion on a
stack -- one of scalar node indices, `u32[64]`, beside one of masks,
`boolx16[64]`, with one count. Each node's bounds, children and primitive
range are one scalar load broadcast into the lanes' arithmetic, and only the
ray tests are per lane. Both levels of an instanced scene come out this way,
the instance's tree entered for the instance the gang is at with the lanes
that reached it (Wald, Slusallek, Benthin & Wagner, "Interactive Rendering
with Coherent Ray Tracing", Eurographics 2001).

Three things had to be true of the compiler for the order to mean this, and
each is a rule rather than a special case. The sort's order is one decision for
the gang: every compare-and-swap of the sorting network is a `vote`
(Instruction::Op::Vote), which the vectorizer settles as the majority of the
lanes that are on -- `2 * popcount(cmp & mask) > popcount(mask)`, two `ctpop`
and a compare -- when it decides between children the lanes share, and drops
when the children are already per lane. A child is descended into or pushed
only if some lane wants it -- without that a packet no ray reached the node
with pushed children read off a bypassed, zeroed node and descended without
end -- and the test is made once: the arm a run sits in is behind the
linearizer's own `any(mask)` guard (the BOSCC gadget), and `loopify()` puts a
test in front of its pushes only where no test above already covers the mask
(`known_nonempty`, SSA/Analysis.h), which in this render is nowhere; the same
rule decides the test in front of every masked call, and the backend makes no
tests of its own. And the divergence analysis had to stop calling a
loop header divergent for running inside a divergent branch's arm -- the join
rule now finds the joins of a divergent branch by propagating labels along
forward edges, as LLVM's sync dependence analysis does, and a header's lanes
are at different iterations only when a latch leaves through a branch inside
the loop -- since that misreading made the TLAS leaf's loop over its
primitives varying, and with it the instance and the BLAS root, and would have
left the instance's tree per lane. Schedules are applied in the order written,
across functions, which is what makes the order sayable (ir::TransformOrder).

Measured, depth 1 and 64 spp, sixteen lanes, best of three, each render alone
on the machine; the same compiler for all three, the scalar schedule from
commit 864c2c1a, the per-lane one HEAD's before this change:

    scene            scalar    per-lane    packet    packet/scalar  packet/per-lane
    bvh-lights       0.226 s   0.087 s     0.069 s   3.27x          1.26x
    killeroo-simple  1.74 s    1.30 s      0.70 s    2.49x          1.86x
    book             12.96 s   15.11 s     5.65 s    2.29x          2.68x
    ganesha          10.01 s   9.31 s      5.69 s    1.76x          1.64x
    instances        1.82 s    0.78 s      0.59 s    3.06x          1.31x

Book, the divergent scene the per-lane traversal made slower than scalar, is
2.3x faster than scalar as a packet. At the scenes' own depth, 16 spp, one run
each: killeroo 0.565 s scalar, 0.599 s per-lane, 0.343 s packet; instances
0.447 / 0.193 / 0.149 s. Against pbrt on the same integrator (`compare.sh
--spp 16`, pbrt's own render timer): killeroo pbrt 870 ms, packet 348 ms,
2.50x faster, where the per-lane schedule had been 1.42x (655 vs 930 ms);
book pbrt 4930 ms, packet 1876 ms, 2.63x faster; both images match pbrt. The
images against each other: packet and per-lane are identical to the bit on
every scene, radiance, normals and albedo alike, and both differ from scalar
only where vectorized arithmetic contracts differently (a few ulp).

What the plan's `specialize(uniform(...))` would still add is the *divergent*
gang: a packet whose lanes are at different nodes is not a packet, and here a
gang enters an instance only for the lanes that reached it, so the case does
not arise inside a traversal; it would for a gang of paths at their second
bounce entering the top-level tree, where the lanes agree on the root and
diverge below it -- exactly what a packet handles by masking, at the cost of
visiting the union of the lanes' nodes. Whether a per-lane fallback there is
worth having is a measurement not yet taken.

### 7. Where the gang skips an arm is the schedule's to say

**Done, with a list of what it leaves open.** A vectorized branch is
linearized: every arm is computed under a mask of the lanes in it, and an arm
no lane is in is paid for unless a uniform `any(mask)` test -- the BOSCC gadget
(Shin 2005; Moll & Hack 2018 §6; ispc's `emitMaskMixed`) -- is put in front of
it. The linearizer used to put one in front of every arm that touched memory or
made a call, which it must, and every arm of six operations or more, which was
ispc's line (`PREDICATE_SAFE_IF_STATEMENT_COST`). Two things were wrong with
that. Reading a field of a variant's payload is an `extract_idx` at a constant
index, and `extract_idx` counted as memory, so every arm of every `match` on an
inline ADT got a gadget whatever its size -- which is why the six-operation line
measured as making no difference: the arms that mattered were all "memory".
And whether an arm of arithmetic is worth a test is a fact about the data, not
the code: it pays when gangs often find no lane in it, and costs a `vptest` and
a branch every time when they do.

So the wager is the schedule's. A constant-index `extract_idx` is not memory;
the linearizer installs a gadget only where it must, and where a `skip` cursor
names the arm:

    intersects.skip(Shape);         // every arm of a match on a Shape in intersects
    surface_geometry.skip(Shape.Triangle);  // one arm
    path_step.skip();               // every arm the function has

A named arm is found by the *provenance* lowering leaves on it (`ir::Provenance`:
the function it was written in, the ADT and the variant), which Lower/ADTs.cpp
puts on the SwitchStmt, SSA/Convert.cpp on the arm's first block, and every copy
-- inlining, specialization -- carries along; so a match written in an
`[[inline]]` helper is named by the helper wherever it ends up, and a geometric
intrinsic by its program name (`intersects`, meaning every overload that takes
the ADT). The SSA dump prints it beside the block (`// Shape.Sph of
intersects_Ray_Shape`), and a cursor that names nothing is an error that lists
what the function does have. The bare form covers unlabelled arms too -- an
`if`'s -- since those carry no provenance. Tests: ssa/vectorize-skip-cursor and
its LLVM and execution twins; ssa/vectorize-skip-arms, now under `weigh.skip()`.

The measurement: the packet schedule, 64 spp, depth 1 and 5, best of three,
each render alone on the machine, seconds. `old` is the previous rule (memory,
calls, and any arm of six operations or more, with payload reads counted as
memory) from the earlier threshold sweep, so it is a different run; the four
cursor sets are one run. `none` is no cursors -- gadgets only where they must
be, 2052 `!any` mentions in the render's SSA; `shape` adds cursors on the Shape
dispatches (`intersects`, `distmin`, `distmax`, `surface_geometry`), which adds
nothing because those arms load a `tagged_index` variant and were mandatory
already; `shading` adds the Material, BxDF, Light and LightSampler dispatches
(2143); `all` adds `skip()` on every function of the path loop and the
traversals (2472).

    scene            d    old      none     shape    shading  all      none/old
    book             1    4.880    4.626    4.678    4.743    4.663    1.055x
    book             5    6.523    6.232    6.293    6.457    6.271    1.047x
    bvh-lights       1    0.0524   0.0514   0.0527   0.0522   0.0510   1.021x
    bvh-lights       5    0.0662   0.0656   0.0664   0.0678   0.0646   1.008x
    ganesha          1    5.328    5.221    5.233    5.254    5.195    1.020x
    ganesha          5    7.462    7.453    7.460    7.484    7.399    1.001x
    instances        1    0.432    0.424    0.426    0.424    0.417    1.018x
    instances        5    0.485    0.478    0.478    0.480    0.462    1.015x
    killeroo-simple  1    0.630    0.622    0.629    0.637    0.624    1.012x
    killeroo-simple  5    1.244    1.247    1.252    1.298    1.241    0.998x

Taking the tests off the arithmetic-only arms is worth 1-5% (book 5%, at both
depths); the shading cursors cost 3-4% on killeroo and book at depth 5 -- one
material per scene, so the tests bypass nothing -- and `all` is within the
run-to-run noise either way (instances d5 +3%, book d1 -1%). So the packet
schedule says nothing, and the comment in it says why.

What this leaves open, in the order they are likely to matter:

1. **`coherent` -- the all-on side.** ispc's `cif` (§6.5, `emitMaskAllOn`)
   tests, when the enclosing mask is all on, whether *every* lane takes an arm
   and runs a copy compiled for that: no masking inside, unmasked callees, no
   blends at the join. A masked variant of the same rule -- `all(tag == k ||
   !mask)` sending the gang to a single-arm body still under `mask` -- is not
   worth having: that body is the linearized arm with `m_k == mask`, and it
   buys back only the other arms' `any` tests and the join's phis at the cost of
   a copy per arm, which is why ispc does not do it either (its mixed path is
   the skip gadgets alone). Where it would pay is a region whose mask is all on
   -- primary rays, an unmasked variant, or a masked variant behind one
   `all(mask)` test at its entry that jumps to the unmasked variant (a tail
   call, no copy). Plan: `f.coherent(Adt)` cursors alongside `skip`; a
   versioning rewrite before linearization that clones the arm regions behind
   `all`/`none` tests with a first-active-lane broadcast of the tag; the
   linearizer narrows the tests to the block mask the way it seeds loops.
   Needs compaction (item 4) to matter past the first bounce.
2. **Counters, not guesses.** An instrumented build that counts, per gadget,
   how often it is taken and how many lanes were on, so that `skip` cursors are
   placed from occupancy data rather than by argument. `BONSAI_NO_BOSCC` and the
   A/B script are the manual version.
3. **Region-level gadgets and landing chains.** A gadget per arm means a chain
   of tests across a `match`; ispc tests once per arm too, but Moll & Hack put
   one gadget around a whole region. With cursors the count is the schedule's,
   but the landing blocks a bypass threads its values through are still one per
   arm and could be elided when consecutive.
4. **A gang that refills.** After the first bounce the mask is rarely full, so
   `coherent` and every uniform-when-all-on optimization are moot without
   wavefront compaction: paths that terminate leave the gang, new paths join.
   The `walk_step.loopify()` tail recursion is the place; the queue directives
   are the words.
5. **The divergent gang** (item 6 above): a per-lane traversal fallback where
   the lanes' nodes disagree, `foreach_unique`-style, is still unmeasured.
6. **§6.4 coherent memory access.** Uniform loads and `Ramp` dense loads are
   done; ispc's `ImproveMemoryOps` also proves gathers of `base + lane*stride`
   from the addresses' algebra. Not yet.
7. **SoA layouts** for per-lane aggregates, and inlining masked callees into
   their one call site; both are code-size against speed and neither has a
   measurement.

## The scheduling language's mission: MoonRay as a schedule

The mission of bonsai's scheduling language is to represent the transformations
a production vectorized renderer is made of, concisely, as a schedule of an
algorithm that says only what is computed. The target for the CPU is the
architecture of DreamWorks' MoonRay -- Lee, Green, Xie and Tabellion,
"Vectorized Production Path Tracing", HPG 2017,
https://www.tabellion.org/et/paper17/MoonRay.pdf -- which is a scalar path
tracer and a vectorized wavefront one that compute the same images from the same
shaders, the second built out of exactly the pieces below. `render.bonsai` is
to stay the scalar algorithm; `schedules/moonray.bonsai`, or whatever it is
called, is to be a page.

### What MoonRay does, read as a schedule

MoonRay's vectorized rendering phase (their figure 3) is breadth-first: paths
are not traced to completion one at a time but advanced a stage at a time
through **queues**, each with a **handler** that runs when the queue is
flushed. The queues, and what they are in the renderer here:

- **Primary ray queue** (thread-local): camera rays, filled from screen-space
  tiles, so its entries are coherent by construction. Its handler intersects
  them (Embree) and puts the hits on the shade queue. Here: `render`'s call of
  `trace` on the camera ray.
- **Incoherent ray queue** (thread-local): continuation rays spawned by the
  integrator; separate from the primary queue so as not to dilute its
  coherence. Its handler intersects and feeds the shade queue; the shade
  queue's handler feeds it back, and that cycle is the recursion. Here: the
  `trace` inside `walk_step` / `path_step`.
- **Occlusion ray queue** (thread-local): shadow rays, on their own because
  Embree's occlusion path is a different code path. Here: `trace_any`.
- **Shade queue** (one per shader instance, *shared* across threads): a hit
  waiting to be shaded. One queue per material instance is what makes a flush
  coherent: every entry in it runs the same material, so the shader graph runs
  once over a full gang with no divergence on the material dispatch. Shared
  rather than per thread because per-shader-per-thread queues would cost too
  much memory; they accept the lock contention. Its handler is all of shading,
  texturing and integration: sort, transpose to AOSOA, run the ISPC material
  to build a BSDF (up to 8 lobes, a lane mask per lobe), sample lights and the
  BSDF, MIS, path splitting, Russian roulette, then spawn continuation and
  occlusion rays onto their queues and radiance onto the radiance queue. Here:
  `material_bxdf`, the `bxdf_*` and `light_*` dispatches, and the body of the
  integrator step.
- **Radiance queue** (thread-local): film writes, queued so that the atomics
  on the shared frame buffer are batched. Here: the film accumulation at the
  end of a path.

A queue is a fixed-size buffer, not a FIFO. Entries are appended in batches;
**whoever fills a queue past its size flushes it, there and then**, which may
fill and flush another -- "autonomous scheduling", no central dispatcher. A
flush copies the entries out to a thread-local arena (a multiple of the lane
width, the rest stay), **sorts** them by a 32-bit key with a radix sort (11-,
22- or 32-bit variants; 1.5% of the frame), and hands the sorted batch to the
handler. The frame ends with a drain: each thread flushes its own queues, then
the shared shade queues, until all are empty. Each entry is 64 bits: the sort
key and a 32-bit **reference** into a pool of `RayState` records (ray
differential, throughput, framebuffer destination, whatever must persist while
a ray waits), allocated from a lock-free pool so a shade queue flushed on one
thread can free another's states. Rays of different generations mix freely in a
queue; nothing requires a lane to keep its identity, so there is no
regeneration or compaction pass -- the queues are the compaction.

The shade queue's sort key (their table 1): light-set index (7 bits), UDIM
texture tile (7), mip level (4), morton-coded uv (14). Sorting by light set
first is what keeps the light loop uniform; the rest is for the texture cache.
Ray sorting by direction for the intersector was tried and was not a win
(their figure 8); the queues' coherence by shader was enough.

Before the handler runs, the sorted references' AOS payloads are prefetched and
**transposed in place to AOSOA** with a stride of the SIMD width, so the ISPC
kernel reads a lane's field with an aligned vector load; outputs go back
through the reverse transposition. Pointers are split into two 32-bit halves
for the transpose. Aligned loads alone were 7-8%.

Threading: TBB for the preparation phase; in rendering each thread runs its own
loop pulling batches of primary rays from one shared work queue, with
thread-local storage for everything and no heap allocation, so the only lock
is on the shared shade queues. Scalar functions can be called per lane from
the vectorized framework, which is how features are prototyped before being
vectorized.

Their results (AVX2, 8 lanes, dual Xeon E5-2697 v3): shading 4.2-6.2x,
texturing 1.7-4.2x, integration 2.7-3.0x, ray intersection 1.0-1.2x (Embree
was already vectorized), overall 1.3-2.3x; vectorization overhead -- queuing,
sorting, AOSOA -- 7-19% of the frame; light-loop SIMD utilization 51-69%, BSDF
70-90%.

### What that is in bonsai's words

Every one of those is a transformation of the same algorithm, and most of the
words exist:

| MoonRay | bonsai directive | status |
|---|---|---|
| a call becomes an entry on a queue, the loop becomes a drain | `q = f.queue(loop)`, `g.defer(callee, q)` | **built** as an SSA rewrite (SSA/Defer.h) for a linear, tail-position deferral whose continuation needs nothing of the producer's frame; see "defer, phase 1" below for what that excludes |
| one queue per kind of ray: primary, incoherent, occlusion | `defer` per call site: `render`'s `trace`, `walk_step`'s `trace`, `trace_any` | needs call-site cursors, not just callee names; and a non-tail deferral |
| one shade queue per material instance | a queue sharded by the value's variant (and instance) | missing: `defer` keyed by an ADT tag |
| a flush sorts by a key before the handler | `sort(queue, key)` | `sort` is for tree children only (Lower/Sorts.cpp); "TODO: queue sorting" in Schedule.h |
| the handler runs a full gang over the sorted batch | `vectorize` over the drain loop | the gang exists; it does not yet draw from a queue |
| thread-local queues, one shared | `bind(queue, CPUThread)`; a `shared` queue | missing |
| flush when full, whoever filled it; drain at the end | queue size + flush policy on `make_queue` | missing |
| `RayState` pool, 32-bit references, AOS to AOSOA | the layout language: the queue entry's layout, a reference as an index (never a pointer), a struct-of-vectors layout at gang stride | layouts exist for trees and arrays; not yet for queues |
| radiance queue in front of the frame buffer | `defer` on the film write | same mechanism as the ray queues |
| a shader runs once per gang because the queue is per material | `coherent` (item 7's list, 1) on the dispatch, made worthwhile by the sort | designed, not built |
| scalar functions from vectorized code | a per-lane scalar call from a gang (`foreach_unique`-style; item 7's list, 5) | missing |

Everything in the left column is "how", and none of it may touch
`render.bonsai`: the algorithm already says the walk is a tail recursion and the
render a `parfor`, which is all the schedule needs. The reference for this,
too, is that the scalar and vectorized MoonRay compute identical images; here
the check is `compare.sh` against pbrt and against the scalar schedule.

### defer, phase 1: what was built (2026-09-19/20)

`SSA/Defer.h` has the design in full; the decisions, all the user's:

- **The generated code is to read as pbrt's wavefront integrator, or do
  less.** For a self-recursion the drain is two queues indexed by round parity,
  pbrt's `rayQueues[depth & 1]`: round r reads queue `r & 1`, its successors
  go to the other. Double buffering is needed exactly when the code a drain
  runs can push onto the queue being drained; a queue whose handler pushes
  only onto other queues -- the trace-to-shade-to-trace pipeline -- gets one
  buffer and one pass, decided from the queue graph.
- **A queue is one object: a count and its entry storage.** `Queue_paths {
  count : u32; data : Entry_paths[] }` beside `paths_entries : Entry_paths[N]`,
  both `mut` locals of the owner named after the queue, sized by the producer
  count (spp for a per-pixel queue), no allocator and no bound checks.
  Logically an array of continuations; its layout (AoS today, SoA or AOSOA
  later) is the layout language's to say.
- **The queue owns the continuation, by value.** An entry is the deferred
  call's varying arguments plus the producer frame's own live values, a struct
  named by the parameters and values it holds. Nothing that is a value on the
  stack is ever saved as an index into a pool ("Anything that is a value on
  the stack cannot be saved as an index somewhere"). A pointer argument that
  points at a mutable local of the producer's iteration -- the sampler state,
  the visible surface -- is stored as its contents, and the drain gives the
  entry a local of its own to run with.
- **The callee pushes; the frame adds its part.** The deferred call site
  becomes a `Push` of the entry onto the queue the callee is handed, plus a
  return of `Saved_f { saved, slot, value }`. Every chain function returns
  that; the owner acts on it: "if the return says we saved state, save this
  data in the queue location, otherwise act as normal" -- the producer writes
  the weight and the wavelengths into `entries[slot]` and ends its iteration,
  and the drain does the same for an entry that saves again.
- **What is left out of the entry** is decided without a cost model: a
  parameter is invariant when every call down the chain passes it one value
  of the owner's (a fixed point over the chain's call sites), and an invariant
  value that is not in scope before the producer loop is *moved* there when it
  is recomputable from things that are -- a load through a read-only
  parameter, arithmetic, a field unwrapped from the integrator inside
  `integrator_li` -- which is loop-invariant code motion for the values the
  deferral asks about, computed once per owner iteration rather than once per
  entry, so nothing is traded off. `BONSAI_EXPLAIN_DEFER=1` prints the plan.
- **Refused, with a message saying why**: a deferred call not in tail
  position; a frame between it and the owner that does more than return the
  value; a branching recursion (linear deferral only, the user's call); a
  stored pointer that is not a mutable local of the iteration; a chain
  function with live callers outside the chain; a deferral inside a
  vectorized gang; a capacity that cannot be checked.

Three compiler bugs this surfaced and fixed, none of them the pass's: the
generic statement mutator rebuilt a parfor without its binding, so any pass
that changed a bound loop's body after the schedule ran it sequentially; the
relooper declared a nested loop's carried variables at function scope, one
variable shared by every iteration of the enclosing parfor; and the LLVM
backend hoisted a run-time-sized stack allocation to the function entry ahead
of the value that sizes it.

Tests: `ssa/defer-tail`, `defer-lone-call`, `defer-loop-owned`, `defer-frame`
(goldens), `backends/llvm/defer` (the IR), `correctness/llvm/defer-sum`,
`defer-parfor`, `defer-loop-owned`, `defer-frame`, `defer-mut-local`,
`defer-off-chain` (run), and `error/defer-*` for each refusal.
`Lower/Defers.cpp` is gone; `defer` is SSA-only, like `loopify`.

**`schedules/wavefront.bonsai`** is the scalar schedule with
`full_path_step.loopify()` replaced by `paths = render.queue(p);
full_path_step.defer(full_path_step, paths);`. Its entry holds the ray, its
differential, `beta`, `l`, the depth and the four scalars and flags of a path
step, `prev_ctx`, `lambda`, the sampler state, the visible surface and the
filter weight; the camera, sampler, scene, light sampler and depth limit are
in scope at the drain. Measured 2026-09-20 with `compare.sh --spp 16
--maxdepth 5`, every image matching pbrt (mean radiance within 2e-5):

    scene              pbrt      scalar    wavefront (scalar drain)
    area-light-path    100.0 ms   71.4 ms   68.0 ms   (REPEATS=1)
    killeroo-simple    820.0 ms  580.6 ms  630.5 ms   (REPEATS=3)

The scalar wavefront costs 9% on killeroo: a path step's state -- a few
hundred bytes -- is written to the queue and read back once per bounce, where
the loop kept it in registers, and nothing yet is bought with it. What buys it
back is the gang drain: sixteen entries of one pixel's queue run as one gang,
which is the packet schedule's gang made from a queue instead of a loop, and
then a tile-owned queue that keeps the gang full as paths die.

**The gang drain** (built 2026-09-20, uncommitted at the time of writing):
`render.split(paths, paths_gang, paths_lane, 16, true).vectorize(paths_lane)`
after the deferral, which is what `schedules/wavefront.bonsai` now says.
Three compiler pieces made it possible. `split(.., true)` generates a tail:
Halide's GuardWithIf, the body behind `outer + inner < end` with the end
threaded into the new blocks as an argument, so a batch of any size is
drained by whole gangs and the last runs partly full (`SSA/Rewrite.cpp`; the
flag was parsed and ignored before, `false` meaning "assert the range
divides"). A gang's `Push` compacts: the vectorizer keeps the push one
instruction with the block's mask as a third operand, marks its value -- the
slot -- varying whatever its operands, and `lower_pushes` then emits one
`atomic_add` of the popcount of the lanes that push, each lane's rank among
them by a Hillis-Steele prefix scan of the mask (nine instructions of
shuffles and adds; ispc's `exclusive_scan_add`, CUDA's `thread_rank()`), and a
masked scatter of each field of the entry to the ranked slots, so the next
round's queue is dense (`SSA/Defer.cpp`). And a location indexed per lane
then accessed by field -- `entries[slot].weight` -- is a scatter the LLVM
backend now lowers (`create_scatter_at`; a `WriteLoc`'s type after a per-lane
index is `widen(element)`, the value a gang writes there). Two things came
out of making it run on the renderer: a vectorized body needs one yield for
linearization to end at, as a specialized callee needs one return
(`unify_yields`, beside `unify_returns`); and `promote_allocas` tied a local
to its threading by name only, so the drain handing its copy of a relocated
local to the continuation's parameter under the producer's name deleted the
alloca from under the copy's loads -- the pass now refuses a pointer handed
to an argument of another name, and `defer` renames the copy's parameter
after the drain's storage so the local is promoted after all.

Measured 2026-09-20, `compare.sh --spp 16 --maxdepth 5`, every image matching
pbrt (killeroo mean radiance 0.99999x, area-light-path 1.00015x):

    scene              pbrt      scalar    packet    wavefront   wavefront
                                                     (scalar)    (gang drain)
    area-light-path    100.0 ms   65.0 ms   22.0 ms   68.0 ms     67.1 ms
    killeroo-simple    830.0 ms  576.3 ms  318.5 ms  579.6 ms    670.9 ms

The gang drain is correct and *slower* than the scalar drain, and `perf` on
killeroo says why, in two parts. First, half the render is still scalar:
`full_path_step` 19.9%, `scrambled_radical_inverse` 8.8%, `sphere_roots`
8.0%, `triangle_hit` 4.9%, `dielectric_sample_f` 2.7%, `get_pixel_2d` 2.2%,
against 13.1% for the gang copy of `full_path_step` and 5.1% for the
sampler's. The producer loop -- the sample loop, where every path takes its
*first* step: the camera ray, the first and most expensive traversal, the
first shade -- is not vectorized by this schedule; only the steps after it
reach the gang, because `defer` turns the calls *inside* `full_path_step`
into pushes and leaves the chain's first call to it a call. pbrt's wavefront
generates camera rays into the ray queue and runs depth zero in the same
kernels as every other depth. Second, the queue round trip: an entry is 450
bytes in some ninety scalar fields (`Ray`, `RayDifferential`, `beta`, `l`,
`lambda`, the sampler state, `prev_ctx`, the visible surface, the frame's
weight and wavelengths), read with a masked gather per field and pushed with
a masked scatter per field, at a stride of the entry -- 341 gathers and
scatters in the generated IR -- and a gather moves an element a cycle where
the loop kept the state in registers. The pixel kernel's own share is 10.3%,
and the scatters are inside the gang copy's 13.1%.

So the two next steps for this schedule, in order of what they are worth:

1. **Defer the chain's first call too**, so that the producer only makes the
   camera ray and pushes, and the first step runs in the gang with the rest:
   `integrator_li.defer(full_path_step, paths)` beside the self-deferral, a
   deferral of a call in a function other than the recursion's own -- the
   call-site deferral of item 2 below, which is also what the trace, shadow
   and escaped-ray stages need. Worth about a third of the render by the
   profile above.
2. **A struct-of-arrays layout for the queue's storage**, the layout
   language's job (item 5 below): the entry read becomes a dense vector load
   per field and the compacting push a `compressstore` per field
   (`llvm.masked.compressstore`, `vpcompressd`), which is the form ispc's
   `packed_store_active` and pbrt's SoA queues have. With it the entry
   should also shed what a bounce does not need: the visible surface is set
   at the first hit and carried through every round because it is a `mut`
   local of the iteration; pbrt keeps it in the pixel sample state.

**Struct-of-arrays queues, and the producer in the gang** (2026-09-21). Both
steps above are done, the second first and by a shorter road than the layout
language: the user asked for struct-of-arrays as the queue's default
("Make the queue default to struct-of-arrays always. We will later add an
extension for rewriting queues, but for now, do a SoA default"). A queue is
now `Queue_<name> { count; <scalar> : T[]; ... }`, one array per scalar of
the entry -- every aggregate taken down to its leaves, `ray.o` to
`ray_o_x`, `ray_o_y`, `ray_o_z`, as pbrt's `SOA<Ray>` holds `SOA<Point3f>
o` holds `float *x, *y, *z` -- each sized by the producer count and named
after the queue and the scalar (`paths_ray_o_x_0`), 91 arrays per queue for
this renderer's entry (`Leaf`, `leaves_of`, `take_apart`, `rebuild` in
`SSA/Defer.cpp`; the header comment in `SSA/Defer.h`). The drain reads a
gang's entries with a masked dense load per scalar
(`masked_extract(paths_beta_x[ramp(paths_gang, 1u, 16)], mask)` →
`llvm.masked.load`) and the compacting push writes each scalar with one
`compress_store` (`llvm.masked.compressstore`, `vpcompressd`; a new `compact`
flag on the SSA `Store` and `ir::Store`, lowered by
`create_compress_store_at`), the prefix scan now built only when a frame
reads the slot. Two lowering pieces made the loads dense: the simplifier folds
`bc(a) + ramp(b, s)` into `ramp(a + b, s)` and runs once more after the
schedule (`SSA/Simplify.cpp`, `SSA/Convert.cpp`), and `as_dense_ramp` in the
LLVM backend accepts a `1u` stride, which an unsigned loop's ramp has and
which had been a gather all along.

Two bugs came out of running it on the renderer, neither in the queue. A
run-time-sized stack array -- the queue's arrays are `T[spp]` -- is carved out
by LLVM with its size rounded to sixteen bytes and no realignment, while the
kernel's frame is realigned to the register and the gang call below passes
forty vector arguments on the stack at the register's alignment: 91 small
arrays left the stack pointer 32 bytes off and the first `vmovdqa64` faulted
(the old two `Entry_paths[spp]` arrays happened to be multiples of 256 bytes).
Such an allocation is now aligned to the register, which makes LLVM realign
after it (`create_alloca_at_entry`; `correctness/llvm/defer-gang-runtime-size`
faults without it). And vectorizing the *producer* -- `render.split(s, ...)
.vectorize(s_lane)` beside the drain's, so that a path's first step runs in a
gang of sixteen samples and its push compacts into the first round -- found a
linearizer gap: a uniform branch the fold keeps as a branch (`depth > 1`,
uniform in a gang all at depth zero) around a divergent early return
(Russian roulette) leaves the block after it control dependent on the
survivors' edge, whose mask is computed inside the region; read after the
join it was a name bound on one path only. Now a mask read past its
definition's dominance becomes a block argument at the joins between
(Cytron's placement over the iterated dominance frontier of the graph as
linearized, pruned, not carried around a loop), handed false along any other
way in; and the predicate of a kept uniform branch's edge, where the block
can also be reached around it, is the source's mask conjoined with the
branch condition, as the paper defines it, rather than the source's mask
(`SSA/Linearize.cpp`, "The masks"). That also made
`correctness/cpp/vectorize_uniform_branch` -- pbrt's `AreaLight::L` shape,
whose golden had recorded this very compiler error -- run and match its scalar
run; `correctness/llvm/vectorize-uniform-around-return` is the roulette shape.

Measured 2026-09-21, killeroo-simple, `compare.sh --spp 16 --maxdepth 5`,
best of five, every image matching pbrt (mean radiance 0.99999x):

    pbrt                                        850 ms
    scalar                                      576 ms
    packet                                      318-344 ms
    wavefront, gang drain, AoS, scalar producer  671 ms   (2026-09-20)
    wavefront, gang drain, SoA, scalar producer  623 ms
    wavefront-perlane, SoA, scalar producer      805 ms
    wavefront, SoA, producer in the gang         353 ms   (2.41x pbrt)

`perf` on the 623 ms build said what the 671 ms one had: half the time in
scalar functions (`full_path_step` 19%, `scrambled_radical_inverse` 10%,
`sphere_roots` 9%, `triangle_hit` 5%), the first bounce of every sample run
scalar by the producer loop, against the gang copies (`full_path_step$gang`
11%, `_pfkernel4`, the drain itself, 7%) -- so the schedule now vectorizes
the producer too, which is what took the wavefront from 623 to 353 ms. The
queue's own cost is what is left between 353 and packet's 318-344: the drain
kernel's per-scalar loads and stores and a call passing forty vector
arguments by value, some 5-10% at this sample count. With a per-pixel queue
of 16 paths, compaction cannot make a gang fuller than the packet schedule's
-- both hold one pixel's 16 samples -- so this is the point at which the two
should be equal, and they nearly are; the sweep below measures what happens
as the sample count grows and a pixel's queue holds several gangs.

On the tail loop: `split(.., true)` makes no separate tail loop. The guard
`outer + inner < end` is one body, and when the inner loop is vectorized the
guard becomes the gang's mask, a ramp compare -- `ramp(paths_gang, 1u, 16) <
count` in the IR above -- so the last gang runs partly full under that mask
(`correctness/llvm/split-tail`'s `gang`).

Two more schedules for the comparison: `schedules/wavefront-perlane.bonsai`,
the queue's drain over a per-lane traversal (the traversals loopified before
the drain is vectorized), so that the two wavefront schedules differ from
`perlane` and `packet` in exactly the queue. `eval/render_matrix.py` runs all
five by default over depths 1-5 and 16, 32, ..., 1024 spp (best of one run
from 128 spp up, `--long-spp`), and `eval/summary.py` draws several scenes'
results on one figure (`eval/README.md`). The sweep asked for -- scenes of
increasing complexity, the five schedules, depths 1-5, 16-1024 spp -- is
`eval/plots/<scene>-speedup.{pdf,png}` per scene and
`eval/plots/summary-speedup.{pdf,png,tsv}` across them; its outcome is in the
section below when it has run.

### The next scheduling commands, in order

1. **Measure the scalar wavefront** (done, above) and the **gang drain**
   (done, above: `split` with the GuardWithIf tail and the compacting `Push`;
   the user: "It's probably best to have vectorized queue writes do a
   compaction step"). The first step is in the gang (the producer loop
   vectorized) and the queue's storage is laid out a scalar at a time
   (struct of arrays, the default; 2026-09-21, above). A per-pixel queue
   holds at most spp paths, so at 16 spp its gangs are exactly as full as the
   packet schedule's; the utilization win needs more paths per queue --
   more samples per pixel (the sweep), or a tile-owned queue,
   `render.split(p, p_tile, p_pix, 64)` then `queue(p_tile)`, and then the
   pixel index varies across the drain's lanes in the film write.
2. **Queue kinds by call site.** `defer` addressed to a call inside a function
   (a cursor like `skip`'s, by provenance of the call) so primary, secondary
   and occlusion rays get their own queues. Deferring `trace` is a *non-tail*
   deferral: the function is split at the call, as LLVM's coroutine splitting
   does, with the values live across the call in the entry and the resume
   point fixed per queue. Tree recursion itself (deferring the traversal's own
   recursive calls) stays out of scope: linear deferral only, the user's call.
3. **`sort(queue, key)`**: sort a queue's batch by a lambda over the entry
   before its handler runs -- radix on a 32-bit key as MoonRay does -- and
   **sharding a queue by variant**: `defer(..., queue)` where `queue` is
   indexed by the entry's material tag, one queue per material, so the shade
   handler's material dispatch is uniform. Then `coherent` on that dispatch is
   the all-on copy, which is where the shading speedup comes from.
4. **Queue placement and policy**: `bind(queue, CPUThread)` for thread-local
   queues, a shared queue for shading, `make_queue`'s size, flush-when-full by
   the filler, and the end-of-frame drain. This is the threading model; the
   `parfor` over pixels stays bound to `CPUThread` and each thread's queues are
   its own.
5. **The radiance queue**, and AOSOA layouts for queue entries and per-lane
   aggregates -- the layout language saying "struct of vectors at gang stride"
   for a queue's storage.

The measurement that says whether each step earned its place is the same as
always: `eval/render_matrix.py` over depth and sample count, packet against the
new schedule against pbrt, images checked.

### Resuming from here

Committed on `ajr/ssa` as of 2026-09-19: `d105110e` the eval tool
(`eval/render_matrix.py`, README in `eval/`); `51f09fd0` the undef value;
`926ad9f2` the eval tool checking its options up front; `a5b4769b` provenance
on match arms and the `skip` cursor, the mandatory-gadget rule corrected
(constant-index payload reads are not memory), and a linearizer fix for a
branch on a join's own argument. The test suite is green but for two CUDA
goldens (`backends/cuda/parallel`, `rtiow-primer`) that have failed since the
`map`-to-`parfor` change of 2026-08-28 and are not this work's. Build with
`make -C build bonsai_compiler bonsai_test_runner` in the `bonsai` conda
environment; `ctest -j 12` in `build/`. The schedule files are
`apps/pbrt/schedules/{packet,perlane,scalar}.bonsai`; `packet.bonsai` documents
`skip` and says why it uses none. Item 7 above has the gadget measurement and
the deferred list; this section has the plan for what comes next.

`defer` phases 1 and 1b (above) are committed as `ea016bc1`. As of the end of
2026-09-20 the gang drain is in the working tree and not committed: the
`split` tail (`src/SSA/Rewrite.cpp`, `include/SSA/Rewrite.h`, the flag's
comment in `include/IR/Schedule.h`), the masked and compacting `Push`
(`src/SSA/Linearize.cpp` predicates it, `src/SSA/AnalyzeDivergence.cpp` marks
its slot varying, `src/SSA/Vectorize.cpp` widens its entry, `lower_pushes` in
`src/SSA/Defer.cpp` emits the compaction; `include/SSA/SSA.h` and
`src/SSA/SSA.cpp` for the operand and its printing), the scatter in
`src/CodeGen/CodeGen_LLVM.cpp` and `include/CodeGen/CodeGen_LLVM.h` with the
`WriteLoc` typing in `src/IR/WriteLoc.cpp`, `AtomicAdd` in the SSA-to-LLVM
path (`src/CodeGen/CodeGen_LLVM_SSA.cpp`), `unify_yields` in
`src/SSA/CloneFunction.cpp` and its header, called from `vectorize()`, the
`promote_allocas` rule in `src/SSA/PromoteAllocas.cpp`, the parameter
renaming in `defer`, and `schedules/wavefront.bonsai`. Tests: `ssa/split-tail`
and `ssa/defer-gang` (goldens), `backends/llvm/defer-gang` (the IR),
`correctness/llvm/split-tail`, `defer-gang`, `defer-gang-frame`,
`defer-gang-mut-local` (run); the `error/split-not-divisible` golden moved
with the assertion. The suite is green but for the two CUDA goldens.

Committed 2026-09-21, in this order: `96d67c38` the split tail and
`unify_yields`; `dd620d2f` the compacting push; `c439a0d1` the schedules'
tails; `5db12cf3` the scene features (blackbody, distant light, disk, coated
conductor, textured roughness, named coordinate systems); `00a82358` the
plan; `296982c2` struct-of-arrays queues and the compress-store push;
`fe9d06ab` run-time-sized allocas aligned to the register; `cc462d2e` the
linearizer's edge predicates and mask joins around a kept uniform branch;
`d95c6b39` the wavefront schedules vectorizing the producer and
`wavefront-perlane`; `d638e8b5` the eval tool's five schedules and
`eval/summary.py`. Tests added: `correctness/llvm/defer-gang-runtime-size`,
`vectorize-uniform-around-return`; `correctness/cpp/vectorize_uniform_branch`
now runs. Next on this schedule: queues by call site (the trace and shadow
stages) and by material; a tile-owned queue for the compaction to have paths
to compact.

## What the scenes need, and what volpath needs

Two inventories, taken 2026-09-20 from `~/projects/pbrt-v4-scenes` and
`~/projects/pbrt-v4/src/pbrt`, of what this renderer lacks. The converter,
`scene_dump.cpp`, refuses every gap loudly except one: media are silently
dropped, because `CapturingBuilder` overrides neither `MakeNamedMedium` nor
`MediumInterface`, so nine scenes would convert with their fog and glass
interiors missing rather than error. That is the first thing to fix, since a
render that quietly leaves out a medium looks like a renderer that works.

### Missing features, by the scenes they unlock

| # | feature | scenes | n |
|---|---|---|---|
| 1 | `Integrator "volpath"` | bmw-m6, bunny-cloud, bunny-fur, clouds, crown, dambreak, disney-cloud, explosion, hair, head, kroken, lte-orb, sanmiguel, smoke-plume, sportscar, transparent-machines, villa, watercolor | 18 |
| 2 | participating media: `MakeNamedMedium` + `MediumInterface` (`homogeneous` x4, `nanovdb` x3, `cloud` x2, `uniformgrid` x1) | bunny-cloud, clouds, crown, dambreak, disney-cloud, explosion, kroken, smoke-plume, watercolor | 9 |
| 3 | env map in a non-sRGB colour space (EXR chromaticities, ACES) | bistro, bunny-cloud, clouds, explosion, sanmiguel, sportscar, villa | 7 |
| 4 | samplers `zsobol` (pbrt's default when a scene names none), `sobol`, `pmj02bn` | bistro, clouds, disney-cloud, explosion, kroken, lte-orb, sanmiguel | 7 |
| 5 | `blackbody L` on an area light (and an infinite light, in villa) -- **done 2026-09-20**, see below | barcelona-pavilion night, contemporary-bathroom, crown, kroken, villa, watercolor, zero-day | 7 |
| 6 | `Shape "disk"` -- **done 2026-09-20**, see below | bunny-cloud, disney-cloud, explosion, killeroos gold, villa | 5 |
| 7 | `Material "interface"` / `Material ""` (a medium boundary with no BSDF) | bunny-cloud, clouds, disney-cloud, explosion, smoke-plume | 5 |
| 8 | `Material "coatedconductor"` -- **done 2026-09-20**, see below | bistro, bmw-m6, killeroos coated-gold, kroken, watercolor | 5 |
| 9 | `conductor` given `reflectance` (rgb, or a texture) instead of `eta`/`k` -- **done 2026-09-20**, and with it `eta`/`k` given as `rgb`, inline pairs or a `.spd` file (item 21) | bmw-m6, crown, villa, watercolor, zero-day | 5 |
| 10 | `Material "mix"` | bmw-m6, crown, kroken, watercolor | 4 |
| 11 | `Shape "bilinearmesh"` (and PLY holding quads) | bunny-fur, sportscar, watercolor | 3 |
| 12 | `mix` textures (and `directionmix`) | kroken, villa, watercolor | 3 |
| 13 | `imagemap` with `mapping "planar"/"cylindrical"/"spherical"` | kroken, villa, watercolor | 3 |
| 14 | `normalmap` on a material | bistro (131), kroken, watercolor | 3 |
| 15 | spectral `eta` on `dielectric` (`glass-BK7`, `glass-BAF10`, `glass-F11`) | dambreak, transparent-machines, crown | 3 |
| 16 | `Shape "curve"` (millions of them) | bunny-fur, hair | 2 |
| 17 | `Material "hair"` | bunny-fur, hair | 2 |
| 18 | `Material "subsurface"` | head, sssdragon | 2 |
| 19 | infinite light with `portal` | kroken, watercolor | 2 |
| 20 | `LightSource "distant"` -- **done 2026-09-20**, see below | disney-cloud, killeroos gold | 2 |
| 21 | inline `spectrum` and `.spd` files for a conductor's `eta`/`k` -- **done 2026-09-20** with item 9 | crown, killeroos | 2 |
| 22-28 | `spot`/`point` lights, `thindielectric`, `windy`/`wrinkled` textures, partial `cylinder`, `realistic` camera, `bdpt`, `sppm` | villa, villa, villa, bunny-fur, sanmiguel (1 of 9), pavilion night, bathroom | 1 each |

Nearest to converting today, in order: `ganesha`, `landscape`, `pbrt-book`,
`lte-orb-simple-ball` (path, halton, only supported shapes, materials and
textures); `sssdragon` only `subsurface`; the `sanmiguel-*` files on
halton only `volpath` and the ACES colour space. `barcelona-pavilion` day,
`killeroo-simple` and `-moving` and, since 2026-09-20, `zero-day` and
`killeroo-gold` render and match.

**Done 2026-09-20, by the scenes they unlocked.** Media are refused loudly:
`CapturingBuilder` overrides `MakeNamedMedium` and `MediumInterface` and
fails naming the medium's kind and the file position (an interface naming no
medium on either side is what a scene writes to leave one, and passes). A
light's emission is an `Emission` variant in render.bonsai -- pbrt's
`Spectrum` as a light emits one: `RGBIlluminant(fit)`, `Blackbody(temperature,
normalization)`, or `Illuminant` (the colour space's own, for an infinite
light with no `L`) -- with `blackbody()` translated from pbrt's `Blackbody`,
`FastExp` included, and the normalization computed by the converter with
pbrt's own function so the two agree to the bit; the converter divides the
light's scale by the photometric integral of whichever spectrum the light
emits, as pbrt does, and builds pbrt's light tree from `BlackbodySpectrum`s
where the scene wrote them. A conductor's index is a `ConductorIndex`
variant: `Tabulated(spectra)` for `eta`/`k`, or `FromReflectance(albedo)`,
pbrt's other branch of `ConductorMaterial::GetBxDF`, eta one and `k = 2 sqrt(r)
/ sqrt(1 - r)` per wavelength at the hit; and `eta`/`k` themselves may now be
anything pbrt reads as an unbounded spectrum -- `rgb` (an
RGBUnboundedSpectrum), a `blackbody`, inline wavelength/value pairs, a named
spectrum or a `.spd` file beside the scene -- each built by pbrt's own
constructor in the converter and tabulated on the same tenth-of-a-nanometre
grid the named metals use (the rgb fit's residual on that grid is 5e-7).
Roughness is a `FloatParam` -- a constant or a float texture, pbrt's
`FloatTexture` -- on the coated diffuse, conductor and dielectric materials,
with pbrt's fallback of `uroughness`/`vroughness` to `roughness`. Scene:
`scenes/blackbody.pbrt` (a blackbody area light and sky, a reflectance
conductor beside a measured copper one), matching pbrt in radiance, albedo
and normals.

Two bugs the zero-day comparison found, neither in the new features. The
packet and per-lane schedules split the sample loop by sixteen *without* a
tail (`split(s, s_gang, s_lane, 16, false)`), so a scene rendered at a sample
count that is not a multiple of sixteen silently ran whole gangs: at four
samples a pixel this renderer took sixteen, with the right mean, a quarter of
pbrt's noise, sixteen percent more lit pixels, and a render that took twice
pbrt's time where it is usually two to three times faster. Both schedules now
say `true`; the guard costs nothing measurable at sixteen (killeroo 324 ms
before and after). And `rgb_to_sigmoid`, pbrt's fit used for textures at the
hit, guarded the grey case's division by zero and answered a coefficient of
zero for black and for white -- a sigmoid of one half -- where pbrt's is minus
and plus infinity, which `sigmoid` already read as nothing and everything. A
black texel reflected half the light: a textured floor pbrt left black in a
third of its pixels was lit in all of them, twelve percent brighter on
average. No guard now, as pbrt has none. Constants never reached it because
the driver fits those with the Gauss-Newton solver.

**zero-day**, `frame25.pbrt` at 1920x840, 4.26 million shapes, 151 instances,
283 blackbody emitters, at 16 spp and depth 5 with the packet schedule: pbrt
8.81 s, this renderer 4.39 s (2.0x), mean radiance 0.99866x, 1585903 lit
pixels against pbrt's 1585401. Per-pixel agreement is 0.8%, which is the
long-path effect described at the top of this file, not a defect: with every
material a plain diffuse the same render agrees on 95.4% of pixels, and each
of the scene's materials agrees in isolation on the small lit scene (a dark
coated diffuse at roughness 1e-4, 90.7%; a smooth coated diffuse, 84.8%; a
smooth silver mirror, 99.8%; an rgb-index conductor, 99.9%; a textured
roughness, 84.4%; a spectrum imagemap reflectance, 87.2%; halton at 400x300,
83.7%). The bisection that found the two bugs above went through a copy of
the scene with every material replaced, then with the textures dropped, then
with only the reflectance textures kept; `PBRT_TREE=1` produced a bit-identical
image to our own tree.

**Later the same day: the distant light and the disk, for killeroo-gold.**
`DistantLight` is a fourth arm of `Light` -- pbrt's delta light: `SampleLi`
answers the one direction with a density of one whatever
`allow_incomplete_pdf` says, `PDF_Li` is zero, `Le` is nothing, and
`light_is_delta` is true, which is what makes `path_sample_ld` take the
estimate alone; the driver keeps it in the infinite run after the area lights
as pbrt's sampler does, and the converter forms its direction from `from`,
`to` and the light's transform as `DistantLight::Create` does, with the scale
divided by the photometric integral and multiplied by an `illuminance`.
`Disk` is a third arm of `Shape`, and the first shape here that lives in an
object space its transform places: `disk_hit` is `Disk::BasicIntersect` on
the ray pulled into object space, `disk_geometry` is
`InteractionFromIntersection` pushed out by `push_surface` -- the same two
functions an instance's tree is entered and left by, now in shapes.bonsai --
and `disk_sample`/`disk_pdf` are `Sample`/`PDF` with the area density
converted to solid angle. It carries pbrt's two orientation flags separately,
since pbrt turns a hit's normal by reverseOrientation ^ swapsHandedness and a
sampled point's by reverseOrientation alone. The `Transform` element and its
point, vector and normal applications moved to `transform.bonsai`, which the
camera, the differentials and the shapes all import. Scenes: `distant.pbrt`
(91.7% agreement, mean 1.00002x) and `disk.pbrt` (94.4%, normals exact to
1.2e-7, a mirrored light transform telling the two flags apart).

Three more fixes fell out of killeroo-gold. The converter now follows pbrt's
named coordinate systems -- `CoordinateSystem` records the CTM, `"camera"` is
the inverse of the CTM at the Camera directive, `"world"` the identity at
WorldBegin, and `CoordSysTransform` installs one -- where it used to declare
a light placed by one untrackable; and its mirror of `LookAt` applied the
*inverse* of what pbrt applies (pbrt::LookAt's forward matrix is
camera-from-world), which nothing had exercised. The third is the one that
mattered: the converter put every scene with fewer than two bounded lights on
the *uniform* light sampler even when it named `bvh`, arguing the two are the
same function there. Their probabilities are; their draws are not, once an
infinite or distant light is present. pbrt's BVH sampler gives the infinite
lights the low end of `u` and the tree the rest, where the uniform sampler's
`min(u n, n - 1)` gives the low end to light zero, the bounded one -- so every
sample picked the other light, and killeroo-gold rendered the right mean with
six of 1.4 million pixels agreeing; the same swap had held `blackbody.pbrt`
and every area-plus-infinite scene at 37% agreement. The BVH arm is now taken
whenever the scene names it and has a light, and `bvh_descend` gained pbrt's
root-leaf importance test. After it: killeroo-gold 99.1% of pixels agreeing,
mean 1.00000x, pbrt 2250 ms against 992 ms (2.27x); blackbody.pbrt 97.3%; the
two-light rgb scene 96.4%; killeroo-simple and infinite-uniform-simple
unchanged at 67.3% and 87.5%.

**The coated conductor, for killeroo-coated-gold.** pbrt's
CoatedConductorBxDF is its LayeredBxDF over a ConductorBxDF where the coated
diffuse's is over a DiffuseBxDF, so the layered element in bxdf.bonsai is now
`LayeredBxDF` with a `LayerBase` variant for the bottom -- `DiffuseBase` or
`ConductorBase` -- that the four `interface_*` wrappers the walk asks the
bottom through match on; the walk itself did not change, and the
coated-diffuse golden test still passes. `CoatedConductorMaterial` is
`CoatedConductorMaterial::GetBxDF` translated: the interface's roughness
and scalar index, the metal's roughness and its index as `Tabulated` or
`FromReflectance` -- divided by the interface's index, since the metal sits
under a medium of that index -- the medium's albedo and `g`, and both
distributions regularized where the path integrator asks, which is what
pbrt's `LayeredBxDF::Regularize` does with a conductor below. The converter
reads pbrt's `interface.`/`conductor.` prefixed parameters with the same
fallbacks as the bare materials. killeroo-coated-gold: mean 1.00057x, 57.5%
of pixels agreeing (the layered walk's amplification of last-bit differences,
as killeroo-simple's coated diffuse sits at 67%), pbrt 2630 ms against
1181 ms (2.23x).

With that, every `killeroo` scene, `zero-day` and `barcelona-pavilion` day
render and match. What bistro still needs is `zsobol`, `normalmap` and an
ACES environment map; everything else in the inventory needs `volpath`.

Per scene, what is missing (see the converter for what is supported):

| scene | missing |
|---|---|
| barcelona-pavilion | day: nothing. night: `bdpt`, `blackbody L` |
| bistro | `zsobol`, `normalmap`, ACES env map (`coatedconductor` done 2026-09-20) |
| bmw-m6 | `volpath`, `mix`, `coatedconductor`, conductor `reflectance` |
| bunny-cloud | `volpath`, `disk`, `interface`, `nanovdb` medium, ACES env map |
| bunny-fur | `volpath`, `curve`, `bilinearmesh`, partial `cylinder`, `hair` |
| clouds | `volpath`, default `zsobol`, `Material ""`, `cloud` medium, ACES |
| contemporary-bathroom | `sppm`, `blackbody L` |
| crown | `volpath`, `mix`, conductor `reflectance`, inline `spectrum eta`, `blackbody L`, homogeneous media |
| dambreak | `volpath`, spectral `eta` on `dielectric`, homogeneous media |
| disney-cloud | `volpath`, `sobol`, `disk`, `interface`, `distant`, `nanovdb` |
| explosion | `volpath`, default `zsobol`, `disk`, `interface`, emissive `nanovdb`, ACES |
| ganesha, landscape, pbrt-book, lte-orb-simple-ball | nothing obvious |
| hair | `volpath`, `curve`, `hair` |
| head | `volpath`, `subsurface` |
| killeroos | simple, moving, and since 2026-09-20 gold and coated-gold: nothing, all four render and match |
| kroken | `volpath`, default `zsobol`, `mix`/`coatedconductor`, `mix`/`directionmix` textures, non-uv mapping, `normalmap`, `portal`, `blackbody L`, homogeneous media |
| lte-orb | `pmj02bn`, `sobol`, `volpath` (rough glass only) |
| sanmiguel | `volpath`, `sobol` (1 file), `realistic` camera (1 file), ACES env map |
| smoke-plume | `volpath`, `interface`, `uniformgrid` medium |
| sportscar | `volpath`, `bilinearmesh`, ACES env map |
| sssdragon | `subsurface` |
| transparent-machines | `volpath`, spectral `eta` on `dielectric` |
| villa | `volpath`, `disk`, `thindielectric`, conductor `reflectance`, `mix`/`windy`/`wrinkled` textures, non-uv mapping, ACES, `blackbody L`, `spot`/`point` |
| watercolor | `volpath`, `bilinearmesh`, `mix`/`coatedconductor`, conductor `texture reflectance`, `mix` textures, non-uv mapping, `normalmap`, `portal`, `blackbody L`, homogeneous medium |
| zero-day | nothing since 2026-09-20: `blackbody L`, conductor `reflectance` and `rgb eta`/`k`, textured roughness. Renders and matches (above) |

### What `volpath` is, over `path`

pbrt's wavefront renderer implements `volpath`, so this is the integrator the
wavefront schedule should end up scheduling. `VolPathIntegrator::Li`
(`cpu/integrators.cpp:953-1271`) differs from `PathIntegrator::Li` in:

- **Path state.** Two more spectra beside `beta`: `r_u` and `r_l`, the
  rescaled path probabilities of section 14.2.2, with which every MIS weight
  is `1 / r_u.Average()` or `1 / (r_u + r_l).Average()` in place of the power
  heuristic and the carried `p_b`; Russian roulette uses
  `beta * etaScale / r_u.Average()`. Rays carry a `medium`.
- **Medium sampling at every step**, when the ray is in a medium: a delta
  tracking loop over the majorant segments (`SampleT_maj`, `media.h:724`),
  with its own RNG seeded from the sampler, and per sampled point one of
  absorb (terminate), real scatter (a `MediumInteraction` with a phase
  function: sample lights from it, sample the phase function, continue), or
  null scatter (update `beta`, `r_u`, `r_l`, keep marching); emissive media
  add `sigma_a * Le` weighted by `r_e`. After the walk the residual `T_maj`
  scales `beta`, `r_u`, `r_l`.
- **Medium transitions at surfaces.** A hit with no BSDF (`Material
  "interface"`) skips the intersection and changes the ray's medium from the
  primitive's `MediumInterface`; `SpawnRay` picks the medium by the side of the
  surface. Shapes, cameras and lights carry a `MediumInterface`.
- **Transmittance in direct lighting.** `SampleLd` replaces the one
  `Unoccluded` test with a loop: intersect toward the light, an opaque hit
  returns zero, otherwise ratio-track the transmittance through the medium
  (`T_ray *= T_maj * sigma_n / pdf`, `r_l`, `r_u` updated, Russian roulette
  at `T_ray / (r_l + r_u).Average() < 0.05`) and spawn on through the
  interface.
- **BSSRDF**, subsurface scattering with a probe segment, in the same loop.
- Dispersive dielectrics terminate the secondary wavelengths.

To implement: `Medium` as an ADT of `HomogeneousMedium`, `GridMedium`,
`RGBGridMedium`, `CloudMedium`, `NanoVDBMedium` with `SamplePoint`,
`SampleRay` and `IsEmissive`; `MediumProperties`; `RayMajorantSegment` and the
majorant iterators (homogeneous, and the 3D DDA over a `MajorantGrid` of 16^3
maxima for the grids, `media.h:105-214`); `SampleT_maj`, `SampleExponential`,
`SampleDiscrete`; `HGPhaseFunction`; `MediumInterface` on primitives, the
camera and lights, `Interaction::GetMedium`, `SpawnRay`/`SpawnRayTo` setting
the medium, `SkipIntersection`; the `interface` material; `SampledGrid`;
blackbody emission from a temperature grid; and NanoVDB reading. Scenes by
medium: `homogeneous` in dambreak, crown, watercolor, kroken (dielectric
boundaries, absorbing-only gems); `uniformgrid` in smoke-plume; `nanovdb` in
bunny-cloud, disney-cloud and explosion (the one emissive medium); `cloud`
only in clouds. Homogeneous media plus the interface material and the
ratio-tracked `SampleLd` unlock four scenes; the grid and its DDA add
smoke-plume; NanoVDB adds three more.

### pbrt's wavefront queues, which a schedule has to reproduce

`wavefront/workitems.h`, `wavefront/integrator.cpp:240-432`. Per depth:
reset queues, `GenerateRaySamples`, `IntersectClosest`,
`SampleMediumInteraction`, `HandleEscapedRays`, `HandleEmissiveIntersection`,
stop at `maxdepth`, `EvaluateMaterialsAndBSDFs`, `TraceShadowRays`,
`SampleSubsurface`.

| queue | item carries | kernel |
|---|---|---|
| `RayQueue` (two, by depth parity) | ray, depth, lambda, pixelIndex, beta, r_u, r_l, prevIntrCtx, etaScale, specularBounce, anyNonSpecularBounces | intersect closest |
| `MediumSampleQueue` | the above plus tMax and the whole deferred hit record | sample the medium; then re-triage escaped / interface / emissive / material |
| `MediumScatterQueue`, one per phase function type | p, depth, lambda, beta, r_u, phase, wo, time, etaScale, medium, pixelIndex | sample the phase function, push a shadow ray and a ray |
| `ShadowRayQueue` | ray, tMax, lambda, Ld, r_u, r_l, pixelIndex | occlusion, or transmittance when there are media |
| `EscapedRayQueue` | ray origin and direction, depth, lambda, pixelIndex, beta, specularBounce, r_u, r_l, prevIntrCtx | infinite lights |
| `HitAreaLightQueue` | areaLight, p, n, uv, wo, lambda, depth, beta, r_u, r_l, prevIntrCtx, specularBounce, pixelIndex | emission with MIS |
| `MaterialEvalQueue`, one per material type | geometry, lambda, pixelIndex, anyNonSpecularBounces, wo, beta, r_u, etaScale, mediumInterface | BSDF, light sample, BSDF sample; pushes the next ray and a shadow ray |
| BSSRDF probe and subsurface scatter queues | beta, r_u, mediumInterface, etaScale, ... | subsurface |

Two things about it worth having in mind for the schedule. A hit in a medium
is not sent to material evaluation; it goes whole to the medium queue, whose
kernel repeats the triage. And the per-material queues are exactly item 3 of
the MoonRay list: a queue sharded by the variant's tag, so that the shading
kernel runs one material over a full batch.

### The order of work this suggests

1. **Loud refusal of media** in `scene_dump.cpp` -- done 2026-09-20.
2. **Wavefront on `path`** first, since every queue above but the medium ones
   exists for `path` too: `defer` on `full_path_step` (built), then the gang
   drain, then `defer` by call site for the trace, shadow and escaped-ray
   stages and a queue per material tag. The pbrt comparison is
   `--wavefront` against `path` on the scenes that render today.
3. **`volpath`** as a fourth integrator in `render.bonsai`, scalar first:
   `r_u`/`r_l` in the path state, media on rays and shapes, homogeneous media
   and the interface material, ratio-tracked `SampleLd`, then grids, NanoVDB,
   emission. Each step is checked against pbrt on the scene that needs it
   (dambreak, then smoke-plume, then bunny-cloud, then explosion).
4. **The scene features that are not integrator work**, by scenes unlocked:
   ACES env maps, the `zsobol`/`sobol`/`pmj02bn` samplers, `blackbody L`,
   `disk`, `coatedconductor`, conductor `reflectance`, `mix` materials and
   textures, the other uv mappings, `normalmap`, spectral `eta` on
   `dielectric`, `bilinearmesh`. Curves and hair, subsurface, portals,
   `distant`/`spot`/`point` lights and the realistic camera after those.

## What has been built, and what it cost

### Where the time went

Historical, and flattered: every figure in this section was measured while a
diffuse `rho` skipped the sixteen samples pbrt spends on it. See "the same work,
no more and no less" above for what that was worth -- on three-spheres, 17%.
The current numbers are in "Where it is".

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

Where that leaves the comparison, as measured at the time -- and, like the
table further up, flattered by the diffuse `rho` shortcut that was not found
until later:

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
