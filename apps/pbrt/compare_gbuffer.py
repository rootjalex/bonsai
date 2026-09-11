#!/usr/bin/env python3
"""Compare the channels of pbrt's gbuffer against the ones this renderer writes.

    python3 apps/pbrt/compare_gbuffer.py <pbrt.pfm> <bonsai.pfm>
        [--shading <pbrt-ns.pfm> <bonsai-ns.pfm>]
        [--albedo <pbrt-albedo.pfm> <bonsai-albedo.pfm>]
        [--pbrt-seconds <s>] [--bonsai-seconds <s>] [--repeats <n>]

Four things are reported, and they fail for different reasons:

  the hit mask -- which pixels have a normal at all. A disagreement here is a
  disagreement about geometry: the camera is pointing somewhere slightly
  different, or a ray missed something it should have hit. These are counted
  exactly, because one stray pixel along a silhouette is a real difference and
  an average would hide it.

  the normals themselves, where both renderers hit something. pbrt writes its
  gbuffer as half floats, so agreement is bounded by that: around 5e-4 near
  0.5, and there is no point asking for better.

  the shading normals, when --shading names the pair. The geometric normal is
  the shape's alone; the shading one is interpolated across a mesh's vertex
  normals and tilted by the material's displacement, and it is the normal the
  BSDF was built about. A bump map moves nothing the first channel can see, so
  this is the channel that checks one. Compared exactly as the geometric
  normals are, and a pixel is excused only where the *geometric* reference has
  an edge: a bumped surface is discontinuous at every pixel by design, and
  excusing its own edges would excuse everything.

  the albedo, when --albedo names the pair. This is the spectral pipeline end
  to end -- an RGB reflectance fitted to a sigmoid, evaluated at four sampled
  wavelengths, multiplied by the illuminant and integrated back to RGB -- and
  none of that round trip is the identity. It is a colour rather than a
  direction, so it is reported as a difference rather than as a hit mask; the
  geometry is already covered above. Two numbers, because it disagrees in two
  ways: a mean, which is what a wrong fit or a wrong illuminant moves, and a
  count of the pixels where a wavelength rounded to a different nanometre and
  so read the adjacent entry of a table. See compare_albedo.

Also writes pbrt.png, bonsai.png and diff.png beside the inputs, because a
number saying the images match is not the same as being able to look at them.

Exits non-zero if either check fails. Render times, if given, are reported but
never failed on: they are a measurement, not a claim about correctness.
"""

import os
import sys

from image_io import encode_normals, read_pfm, write_png

# Half floats hold about three decimal digits, so this is pbrt's storage
# precision rather than a tolerance chosen to make the test pass.
NORMAL_TOLERANCE = 2e-3

# How much the difference image is brightened. The differences worth seeing are
# far below one pixel level, so shown honestly the image would be black.
DIFF_GAIN = 255.0

# See compare_albedo. A pixel further apart than this is counted rather than
# tolerated; what it means is that the two rounded a wavelength to different
# nanometres, which is a discontinuity rather than an error.
ALBEDO_TOLERANCE = 5e-3

# What the albedo is actually held to. Half floats hold about three decimal
# digits, so a pixel pair drawn from the same wavelengths cannot agree better
# than this, and the mean over the image should sit at that level.
ALBEDO_MEAN_TOLERANCE = 5e-4

# How many pixels may round a wavelength differently, as a fraction of those
# with an albedo at all. An order of magnitude above what is observed, which
# leaves room for the discontinuity and none for a wrong sampler: sampling
# wavelengths uniformly rather than by visible response -- the mistake this was
# written to catch -- moves every pixel, not one in seventy thousand.
ALBEDO_OUTLIER_FRACTION = 1e-4

# How many pixels' shading normals may disagree away from any edge, as a
# fraction of those both renderers hit. Zero would be the right number for a
# mesh's interpolated normals, and it is what every scene without a bump map
# gets. A bump map is different in kind: pbrt differences the displacement
# texture over a footprint, at a pyramid level that is the *floor* of a log of
# that footprint, and one sample of one pixel landing on the other side of
# that floor reads a different level and gets a different normal -- a
# discontinuity in the inputs, not an error in the arithmetic, which
# check_differentials.sh checks bit for bit. Observed: one pixel in 130,000 on
# the pool scene, at eight samples per pixel. An order of magnitude above
# that, and four orders below what a wrong bump map does (8,278 of 128,798 on
# the same scene, when the lens camera's differentials were an ulp off).
SHADING_OUTLIER_FRACTION = 1e-4

# See compare_radiance. What counts as two lit pixels agreeing, which most of
# them do: the paths that took the same branches accumulate only last-bit
# differences, and the ones that took a different branch are not close at all.
RADIANCE_TOLERANCE = 1e-3

# How far the image's mean may be from pbrt's. This is the Monte Carlo estimate
# of the whole integral, so it converges where a pixel does not, and a
# systematic error -- a light of the wrong brightness, a missing cosine, a
# radiance normalized like a reflectance -- moves it and nothing else does.
# The 106.86x the film was once out by would show here as 0.009.
RADIANCE_MEAN_TOLERANCE = 0.02

# How far the *count* of lit pixels may differ, as a fraction of them.
#
# Zero would be right for a random walk: it never samples a light, so whether a
# pixel receives anything is decided by the directions it scattered in, and two
# renderers drawing the same numbers scatter identically. That is what caught
# the missing camera-sample draws.
#
# An integrator that samples lights has one more way to differ. Whether a shadow
# ray is blocked is a comparison against a surface, and a direction that differs
# in the last bit can fall on either side of it -- so a handful of pixels flip
# between lit and not without anything being wrong. What would still be wrong is
# a *lot* of them, which is what this bounds. Observed: six pixels in eighty-one
# thousand on area-light-path.
RADIANCE_LIT_TOLERANCE = 1e-3


def diff_png(path, width, height, ref, got):
    """Where the two disagree, brightened so that anything at all shows up.

    Magenta marks a pixel where only one of them hit something, which is a
    different kind of wrong from a normal being slightly off and should not be
    lost among it.
    """
    out = bytearray(width * height * 3)
    for i in range(width * height):
        a = ref[i * 3:i * 3 + 3]
        b = got[i * 3:i * 3 + 3]
        if any(a) != any(b):
            out[i * 3 + 0] = 255
            out[i * 3 + 2] = 255
            continue
        for k in range(3):
            out[i * 3 + k] = max(0, min(255, int(DIFF_GAIN * abs(a[k] - b[k]))))
    write_png(path, width, height, bytes(out))


def on_a_silhouette(ref, width, height, x, y):
    """Is the reference image discontinuous at this pixel?

    Where a ray grazes a surface, whether it hits at all comes down to
    rounding, and pbrt decides it differently: its sphere intersection carries
    error bounds through interval arithmetic where the stdlib here solves the
    quadratic directly. The two then disagree along the outline of a sphere, by
    at most the one pixel where the discriminant is near zero.

    Excusing those without excusing anything else means asking whether the
    reference itself has an edge here. A disagreement in the middle of a smooth
    surface is a real difference and is not covered by this.
    """
    i = y * width + x
    centre = ref[i * 3:i * 3 + 3]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        j = ny * width + nx
        neighbour = ref[j * 3:j * 3 + 3]
        if any(centre) != any(neighbour):
            return True
        if max(abs(centre[k] - neighbour[k]) for k in range(3)) > NORMAL_TOLERANCE:
            return True
    return False


def compare_normals(ref, got, edges, width, height):
    """Where two normal images disagree, and which of those pixels are excused.

    `edges` is the image whose discontinuities excuse a disagreement -- the
    geometric normals, whichever pair is being compared, since a silhouette is
    a fact about the geometry and not about the channel.

    Returns the count of pixels both hit, the worst difference and where it
    was, every disagreeing pixel, the excused ones and the rest.
    """
    both = 0
    worst = 0.0
    worst_at = None
    disagreements = []
    for i in range(width * height):
        a = ref[i * 3:i * 3 + 3]
        b = got[i * 3:i * 3 + 3]
        # A miss leaves the normal at zero in both renderers.
        hit_a = any(a)
        hit_b = any(b)
        if hit_a and hit_b:
            both += 1
        error = (max(abs(a[k] - b[k]) for k in range(3))
                 if hit_a and hit_b else (1.0 if hit_a != hit_b else 0.0))
        if hit_a != hit_b or error > NORMAL_TOLERANCE:
            disagreements.append((i % width, i // width))
        if hit_a and hit_b and error > worst:
            worst = error
            worst_at = (i % width, i // width, a, b)

    on_edge = [p for p in disagreements
               if on_a_silhouette(edges, width, height, p[0], p[1])]
    edge_set = set(on_edge)
    interior = [p for p in disagreements if p not in edge_set]
    return both, worst, worst_at, disagreements, on_edge, interior


def take_option(args, name):
    if name not in args:
        return None
    i = args.index(name)
    if i + 1 >= len(args):
        raise SystemExit(f"{name} needs a value")
    value = float(args[i + 1])
    del args[i:i + 2]
    return value


def take_pair(args, name):
    if name not in args:
        return None
    i = args.index(name)
    if i + 2 >= len(args):
        raise SystemExit(f"{name} needs two filenames")
    pair = (args[i + 1], args[i + 2])
    del args[i:i + 3]
    return pair


def compare_albedo(pbrt_path, bonsai_path, width, height):
    """The spectral round trip, reported as a mean and a count of outliers.

    Two numbers rather than one, because the albedo disagrees in two quite
    different ways.

    Almost everywhere the two agree to what pbrt's half-float gbuffer can
    express, which near one is about 5e-4. That is the mean, and it is what a
    wrong sigmoid fit or a wrong illuminant would move.

    A handful of pixels are much further apart, and those are not a matter of
    degree. A sampled spectrum is looked up per nanometre, so a wavelength is
    rounded to the nearest one before it indexes anything -- and the two
    renderers do not compute that wavelength with bit-identical arithmetic.
    They cannot: pbrt is built by a compiler that fuses its multiplies and adds
    where it likes, and its libm is not this one. Where a wavelength lands
    within an ulp of a half-nanometre the two round it to adjacent entries, and
    the answers are a table step apart rather than an ulp apart. About one pixel
    in seventy thousand does this, and no amount of care upstream removes them
    -- only making the spectrum continuous would, and pbrt's is not.

    So the count is what is bounded, the way the silhouette pixels are counted
    separately from the normals: a few is the discontinuity, and a lot is a
    sampler that draws the wrong wavelengths altogether.
    """
    pw, ph, pbrt = read_pfm(pbrt_path)
    bw, bh, bonsai = read_pfm(bonsai_path)
    if (pw, ph) != (width, height) or (bw, bh) != (width, height):
        raise SystemExit("albedo images are not the size of the normals")

    worst = 0.0
    worst_at = None
    total = 0.0
    counted = 0
    outliers = 0
    for i in range(0, len(pbrt), 3):
        a = pbrt[i:i + 3]
        b = bonsai[i:i + 3]
        # Where neither found a surface there is nothing to have an albedo, and
        # counting the agreement would be counting the background -- which is
        # two fifths of this scene and would drag the mean down by that much.
        # Only where one of them found something is there a claim to check.
        if not any(a) and not any(b):
            continue
        difference = max(abs(a[c] - b[c]) for c in range(3))
        total += difference
        counted += 1
        if difference > ALBEDO_TOLERANCE:
            outliers += 1
        if difference > worst:
            worst = difference
            worst_at = (i // 3 % width, i // 3 // width, a, b)

    if counted == 0:
        raise SystemExit("no pixels with an albedo to compare")
    mean = total / counted
    fraction = outliers / counted
    print(f"albedo: {counted} pixels, mean difference {mean:.2e}, "
          f"{outliers} over {ALBEDO_TOLERANCE:.0e} ({100 * fraction:.4f}%)")
    if worst_at:
        x, y, a, b = worst_at
        print(f"  worst {worst:.2e} at ({x}, {y}) "
              f"pbrt ({a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f}) "
              f"bonsai ({b[0]:+.4f} {b[1]:+.4f} {b[2]:+.4f})")
    return mean, fraction


def compare_radiance(pbrt_path, bonsai_path, width, height):
    """The render itself, held to what a stochastic estimate can promise.

    Not a per-pixel comparison, and it cannot be one. A random walk finds a
    light by scattering into it, so a pixel's value is a sixteen- or
    two-hundred-sample estimate whose variance is enormous: two implementations
    that agree exactly on the integral disagree wildly on any single pixel as
    soon as one path takes a different branch. That is what the albedo taught
    already -- a last-bit difference reseeds a walk and the answers are
    unrelated rather than close.

    What can be held is what the estimate converges to. Three things are
    checked, in increasing order of how much they say:

    - The set of pixels that received any light at all. Two walks consuming the
      same random numbers scatter the same way, so this should match exactly;
      when it does not, the sampler streams have come apart, which is a real
      bug and not noise. That is how the missing camera-sample draws were found.
    - The mean over the image, which is the Monte Carlo estimate of the whole
      integral and converges far faster than any pixel.
    - The fraction of lit pixels that agree closely anyway, which is most of
      them and which drops if the transport is wrong rather than merely noisy.
    """
    pw, ph, pbrt = read_pfm(pbrt_path)
    bw, bh, bonsai = read_pfm(bonsai_path)
    if (pw, ph) != (width, height) or (bw, bh) != (width, height):
        raise SystemExit("radiance images are not the size of the normals")

    lit_pbrt = lit_bonsai = lit_both = 0
    close = 0
    pbrt_total = bonsai_total = 0.0
    for i in range(0, len(pbrt), 3):
        a = pbrt[i:i + 3]
        b = bonsai[i:i + 3]
        pbrt_total += sum(a)
        bonsai_total += sum(b)
        a_lit = any(v > 0 for v in a)
        b_lit = any(v > 0 for v in b)
        lit_pbrt += a_lit
        lit_bonsai += b_lit
        if not a_lit and not b_lit:
            continue
        lit_both += 1
        scale = max(max(a), max(b))
        if scale > 0 and max(abs(a[c] - b[c]) for c in range(3)) <= \
                RADIANCE_TOLERANCE * scale:
            close += 1

    n = width * height * 3
    pbrt_mean = pbrt_total / n
    bonsai_mean = bonsai_total / n
    # Six of the nine scenes here have no emitter at all -- they were written
    # when the comparison was a gbuffer -- so both sides render an image that is
    # exactly black. Two black images agree perfectly, and the ratio of their
    # means is 0/0; reporting that as a failure said the transport was wrong on
    # the scenes where there is no transport to get wrong. A ratio of one is the
    # honest reading, and the two means printed beside it show why.
    if pbrt_mean == 0.0 and bonsai_mean == 0.0:
        ratio = 1.0
    else:
        ratio = bonsai_mean / pbrt_mean if pbrt_mean else float("inf")
    agree = close / lit_both if lit_both else 1.0
    print(f"radiance: {lit_pbrt} lit pixels in pbrt, {lit_bonsai} here; "
          f"mean {pbrt_mean:.6g} vs {bonsai_mean:.6g} ({ratio:.5f}x)")
    if lit_both:
        print(f"  {close} of {lit_both} touched pixels agree to "
              f"{RADIANCE_TOLERANCE:.0e} relative ({100 * agree:.1f}%)")
    else:
        print("  no pixel received light on either side, which is what a scene "
              "with no emitter renders")
    return lit_pbrt, lit_bonsai, ratio


def report_radiance_only(radiance_pair, pbrt_seconds, bonsai_seconds, repeats):
    """The image and nothing else, for a scene whose film is `rgb`.

    Which is every scene nobody wrote for this comparison. pbrt's `rgb` film
    computes no VisibleSurface and no reflectance, so there are no normals and
    no albedo for it to write -- and there is no way to ask for them without
    editing somebody else's scene, which would make the comparison about our
    copy of it.

    What is left is the thing the scene was written to produce, checked the way
    a stochastic estimate can be checked: the set of pixels that received light,
    the mean over the image, and how many pixels agree closely anyway.
    """
    pw, ph, _ = read_pfm(radiance_pair[0])
    lit_pbrt, lit_bonsai, ratio = compare_radiance(
        radiance_pair[0], radiance_pair[1], pw, ph)
    failed = False
    lit_gap = abs(lit_pbrt - lit_bonsai)
    if lit_gap > RADIANCE_LIT_TOLERANCE * max(lit_pbrt, 1):
        failed = True
        print(f"FAILED: {lit_bonsai} pixels received light here against "
              f"pbrt's {lit_pbrt}, a gap of {lit_gap}")
    if abs(ratio - 1.0) > RADIANCE_MEAN_TOLERANCE:
        failed = True
        print(f"FAILED: the image is {ratio:.5f}x pbrt's on average, over the "
              f"{RADIANCE_MEAN_TOLERANCE:.0%} two estimates of one integral "
              f"should agree to")
    if pbrt_seconds is not None and bonsai_seconds is not None:
        print(f"render time: pbrt {pbrt_seconds * 1e3:.1f} ms, "
              f"bonsai {bonsai_seconds * 1e3:.1f} ms", end="")
        if bonsai_seconds > 0:
            speedup = pbrt_seconds / bonsai_seconds
            if speedup >= 1.0:
                print(f" ({speedup:.2f}x faster than pbrt)", end="")
            else:
                print(f" ({1.0 / speedup:.2f}x slower than pbrt)", end="")
        print()
        if repeats is not None and repeats > 1:
            print(f"  (best of {int(repeats)} runs on each side)")
        print("  (pbrt's own render timer, from the image it wrote; ours wraps "
              "the render call)")
    if not failed:
        print("ok: the image matches pbrt")
    return 1 if failed else 0


def main(argv):
    args = argv[1:]
    shading_pair = take_pair(args, "--shading")
    albedo_pair = take_pair(args, "--albedo")
    radiance_pair = take_pair(args, "--radiance")
    pbrt_seconds = take_option(args, "--pbrt-seconds")
    bonsai_seconds = take_option(args, "--bonsai-seconds")
    repeats = take_option(args, "--repeats")
    # A scene whose film is `rgb` -- which is every scene nobody wrote for this
    # comparison -- gives pbrt no normals and no albedo to write, so there is
    # nothing to compare but the image. Said explicitly rather than inferred
    # from a missing file, so that a normals comparison cannot be skipped by
    # accident.
    radiance_only = "--radiance-only" in args
    if radiance_only:
        args.remove("--radiance-only")
    if radiance_only:
        if radiance_pair is None:
            raise SystemExit("--radiance-only needs --radiance")
        return report_radiance_only(radiance_pair, pbrt_seconds,
                                    bonsai_seconds, repeats)
    if len(args) != 2:
        raise SystemExit(
            f"usage: {argv[0]} <pbrt.pfm> <bonsai.pfm> "
            f"[--shading <a.pfm> <b.pfm>] [--albedo <a.pfm> <b.pfm>] "
            f"[--pbrt-seconds <s>] [--bonsai-seconds <s>] [--repeats <n>]\n"
            f"       {argv[0]} --radiance-only --radiance <a.pfm> <b.pfm>")

    ref_w, ref_h, ref = read_pfm(args[0])
    got_w, got_h, got = read_pfm(args[1])
    if (ref_w, ref_h) != (got_w, got_h):
        raise SystemExit(
            f"resolutions differ: {ref_w}x{ref_h} vs {got_w}x{got_h}")

    both, worst, worst_at, disagreements, on_edge, interior = compare_normals(
        ref, got, ref, ref_w, ref_h)

    out_dir = os.path.dirname(os.path.abspath(args[1]))
    write_png(os.path.join(out_dir, "pbrt.png"), ref_w, ref_h,
              encode_normals(ref_w, ref_h, ref))
    write_png(os.path.join(out_dir, "bonsai.png"), ref_w, ref_h,
              encode_normals(ref_w, ref_h, got))
    diff_png(os.path.join(out_dir, "diff.png"), ref_w, ref_h, ref, got)

    total = ref_w * ref_h
    print(f"resolution {ref_w}x{ref_h} ({total} pixels), hit by both: {both}")
    print(f"disagreeing pixels: {len(disagreements)} "
          f"({100.0 * len(disagreements) / total:.4f}%) -- "
          f"{len(on_edge)} on a silhouette, {len(interior)} elsewhere")
    if worst_at:
        x, y, a, b = worst_at
        print(f"worst normal difference where both hit: {worst:.2e} at ({x}, {y})"
              f" pbrt ({a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f})"
              f" bonsai ({b[0]:+.4f} {b[1]:+.4f} {b[2]:+.4f})")
    if interior:
        print(f"  first away from an edge: {interior[:5]}")

    # The shading normals, excused only where the geometric image has an edge.
    shading_interior = []
    if shading_pair is not None:
        s_w, s_h, s_ref = read_pfm(shading_pair[0])
        g_w, g_h, s_got = read_pfm(shading_pair[1])
        if (s_w, s_h) != (ref_w, ref_h) or (g_w, g_h) != (ref_w, ref_h):
            raise SystemExit("shading normal images are not the size of the "
                             "normals")
        (s_both, s_worst, s_worst_at, s_disagreements, s_on_edge,
         shading_interior) = compare_normals(s_ref, s_got, ref, ref_w, ref_h)
        write_png(os.path.join(out_dir, "pbrt-ns.png"), ref_w, ref_h,
                  encode_normals(ref_w, ref_h, s_ref))
        write_png(os.path.join(out_dir, "bonsai-ns.png"), ref_w, ref_h,
                  encode_normals(ref_w, ref_h, s_got))
        diff_png(os.path.join(out_dir, "diff-ns.png"), ref_w, ref_h, s_ref,
                 s_got)
        print(f"shading normals: hit by both: {s_both}, disagreeing pixels: "
              f"{len(s_disagreements)} -- {len(s_on_edge)} on a silhouette, "
              f"{len(shading_interior)} elsewhere")
        if s_worst_at:
            x, y, a, b = s_worst_at
            print(f"  worst shading normal difference where both hit: "
                  f"{s_worst:.2e} at ({x}, {y})"
                  f" pbrt ({a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f})"
                  f" bonsai ({b[0]:+.4f} {b[1]:+.4f} {b[2]:+.4f})")
        if shading_interior:
            print(f"  first away from an edge: {shading_interior[:5]}")

    if pbrt_seconds is not None and bonsai_seconds is not None:
        # Render only, on both sides: pbrt's number comes from a timer started
        # after its scene and BVH are built, and bonsai's wraps the render call
        # alone. Neither counts building a tree or generating a table.
        print(f"render time: pbrt {pbrt_seconds * 1e3:.1f} ms, "
              f"bonsai {bonsai_seconds * 1e3:.1f} ms", end="")
        # pbrt's time over bonsai's, so the number is a speedup and reads the
        # way a speedup should: above one is bonsai ahead. The ratio the other
        # way up is a ratio of durations, which says the same thing while
        # looking like its opposite.
        if bonsai_seconds > 0:
            speedup = pbrt_seconds / bonsai_seconds
            if speedup >= 1.0:
                print(f" ({speedup:.2f}x faster than pbrt)", end="")
            else:
                print(f" ({1.0 / speedup:.2f}x slower than pbrt)", end="")
        print()
        if repeats is not None and repeats > 1:
            print(f"  (best of {int(repeats)} runs on each side)")
        # Both sides render the same three channels over the same samples of
        # the same pixels, pbrt's through pbrt's own code inside scene_dump.
        # That was not always true, and the note that used to sit here said so:
        # the pbrt binary ran a path integrator into an rgb film while this
        # filled in a gbuffer as well, and the ratio compared two different
        # workloads plus a difference of algorithm.
        print("  (the same work on both sides: normals, a sixteen-sample "
              "reflectance, and the integrator the scene names)")

    # Only disagreements away from an edge are failures. A silhouette pixel is
    # a grazing ray, and which surface it lands on is decided by how carefully
    # the intersection rounds -- a genuine difference between the two
    # renderers, but not one that means anything is wrong here.
    albedo_failed = False
    if albedo_pair is not None:
        mean, fraction = compare_albedo(albedo_pair[0], albedo_pair[1], ref_w,
                                        ref_h)
        if mean > ALBEDO_MEAN_TOLERANCE:
            albedo_failed = True
            print(f"FAILED: albedo differs by {mean:.2e} on average, over the "
                  f"{ALBEDO_MEAN_TOLERANCE:.0e} a half-float gbuffer allows for")
        if fraction > ALBEDO_OUTLIER_FRACTION:
            albedo_failed = True
            print(f"FAILED: {100 * fraction:.4f}% of pixels are further apart "
                  f"than {ALBEDO_TOLERANCE:.0e}, over the "
                  f"{100 * ALBEDO_OUTLIER_FRACTION:.4f}% that rounding a "
                  f"wavelength to the nearest nanometre accounts for")

    radiance_failed = False
    if radiance_pair is not None:
        lit_pbrt, lit_bonsai, ratio = compare_radiance(
            radiance_pair[0], radiance_pair[1], ref_w, ref_h)
        lit_gap = abs(lit_pbrt - lit_bonsai)
        if lit_gap > RADIANCE_LIT_TOLERANCE * max(lit_pbrt, 1):
            radiance_failed = True
            print(f"FAILED: {lit_bonsai} pixels received light here against "
                  f"pbrt's {lit_pbrt}, a gap of {lit_gap}. Two paths drawing "
                  f"the same numbers go the same way, so a gap this size is a "
                  f"sampler stream that has come apart rather than noise")
        if abs(ratio - 1.0) > RADIANCE_MEAN_TOLERANCE:
            radiance_failed = True
            print(f"FAILED: the image is {ratio:.5f}x pbrt's on average, over "
                  f"the {RADIANCE_MEAN_TOLERANCE:.0%} two estimates of one "
                  f"integral should agree to")

    shading_failed = False
    if shading_pair is not None:
        allowed = SHADING_OUTLIER_FRACTION * max(s_both, 1)
        if len(shading_interior) > allowed:
            shading_failed = True
        elif shading_interior:
            print(f"  ({len(shading_interior)} shading normals disagree away "
                  f"from any edge, within the {SHADING_OUTLIER_FRACTION:.0e} "
                  f"a bump map's level choice accounts for)")
    if interior:
        print(f"FAILED: {len(interior)} pixels disagree away from any edge")
    if shading_failed:
        print(f"FAILED: the shading normal disagrees on "
              f"{len(shading_interior)} pixels away from any edge, over the "
              f"{SHADING_OUTLIER_FRACTION:.0e} of {s_both} a bump map's level "
              f"choice accounts for")
    if not interior and not shading_failed:
        print(f"ok: matches pbrt except on {len(on_edge)} silhouette pixels")
    print(f"images: {out_dir}/pbrt.png, {out_dir}/bonsai.png, "
          f"{out_dir}/diff.png (difference brightened {DIFF_GAIN:.0f}x)")
    if shading_pair is not None:
        print(f"  the shading normals: {out_dir}/pbrt-ns.png, "
              f"{out_dir}/bonsai-ns.png, {out_dir}/diff-ns.png")
    if radiance_pair is not None:
        print(f"  the render itself: {out_dir}/pbrt-radiance.png, "
              f"{out_dir}/bonsai-radiance.png")
    return 1 if (interior or shading_failed or albedo_failed or
                 radiance_failed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
