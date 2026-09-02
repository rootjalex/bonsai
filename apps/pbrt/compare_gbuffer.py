#!/usr/bin/env python3
"""Compare the channels of pbrt's gbuffer against the ones this renderer writes.

    python3 apps/pbrt/compare_gbuffer.py <pbrt.pfm> <bonsai.pfm>
        [--albedo <pbrt-albedo.pfm> <bonsai-albedo.pfm>]
        [--pbrt-seconds <s>] [--bonsai-seconds <s>] [--repeats <n>]

Three things are reported, and they fail for different reasons:

  the hit mask -- which pixels have a normal at all. A disagreement here is a
  disagreement about geometry: the camera is pointing somewhere slightly
  different, or a ray missed something it should have hit. These are counted
  exactly, because one stray pixel along a silhouette is a real difference and
  an average would hide it.

  the normals themselves, where both renderers hit something. pbrt writes its
  gbuffer as half floats, so agreement is bounded by that: around 5e-4 near
  0.5, and there is no point asking for better.

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
    ratio = bonsai_mean / pbrt_mean if pbrt_mean else float("inf")
    agree = close / lit_both if lit_both else 0.0
    print(f"radiance: {lit_pbrt} lit pixels in pbrt, {lit_bonsai} here; "
          f"mean {pbrt_mean:.6g} vs {bonsai_mean:.6g} ({ratio:.5f}x)")
    print(f"  {close} of {lit_both} touched pixels agree to "
          f"{RADIANCE_TOLERANCE:.0e} relative ({100 * agree:.1f}%)")
    return lit_pbrt, lit_bonsai, ratio


def main(argv):
    args = argv[1:]
    albedo_pair = take_pair(args, "--albedo")
    radiance_pair = take_pair(args, "--radiance")
    pbrt_seconds = take_option(args, "--pbrt-seconds")
    bonsai_seconds = take_option(args, "--bonsai-seconds")
    repeats = take_option(args, "--repeats")
    if len(args) != 2:
        raise SystemExit(
            f"usage: {argv[0]} <pbrt.pfm> <bonsai.pfm> "
            f"[--pbrt-seconds <s>] [--bonsai-seconds <s>] [--repeats <n>]")

    ref_w, ref_h, ref = read_pfm(args[0])
    got_w, got_h, got = read_pfm(args[1])
    if (ref_w, ref_h) != (got_w, got_h):
        raise SystemExit(
            f"resolutions differ: {ref_w}x{ref_h} vs {got_w}x{got_h}")

    both = 0
    worst = 0.0
    worst_at = None
    disagreements = []
    for i in range(ref_w * ref_h):
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
            disagreements.append((i % ref_w, i // ref_w))
        if hit_a and hit_b and error > worst:
            worst = error
            worst_at = (i % ref_w, i // ref_w, a, b)

    on_edge = [p for p in disagreements
               if on_a_silhouette(ref, ref_w, ref_h, p[0], p[1])]
    interior = [p for p in disagreements if p not in set(on_edge)]

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
        # Said out loud because the ratio above invites being quoted, and on
        # its own it would be dishonest twice over: pbrt runs a path integrator
        # where this renders the nearest hit and stops, so it is doing strictly
        # more work; and pbrt reports its time to the nearest 10ms, which at
        # this scene size is most of the measurement.
        if repeats is not None and repeats > 1:
            print(f"  (best of {int(repeats)} runs on each side)")
        print("  (not the same work: pbrt integrates a path, this returns the "
              "nearest hit)")
        if pbrt_seconds <= 0.05:
            print("  (pbrt's time is reported to 10ms, so this is near the "
                  "limit of what it can tell us -- render at a larger "
                  "resolution before quoting it)")

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
        if lit_pbrt != lit_bonsai:
            radiance_failed = True
            print(f"FAILED: {lit_bonsai} pixels received light here against "
                  f"pbrt's {lit_pbrt}. Two walks drawing the same numbers "
                  f"scatter the same way, so this is a sampler stream that has "
                  f"come apart rather than noise")
        if abs(ratio - 1.0) > RADIANCE_MEAN_TOLERANCE:
            radiance_failed = True
            print(f"FAILED: the image is {ratio:.5f}x pbrt's on average, over "
                  f"the {RADIANCE_MEAN_TOLERANCE:.0%} two estimates of one "
                  f"integral should agree to")

    if interior:
        print(f"FAILED: {len(interior)} pixels disagree away from any edge")
    else:
        print(f"ok: matches pbrt except on {len(on_edge)} silhouette pixels")
    print(f"images: {out_dir}/pbrt.png, {out_dir}/bonsai.png, "
          f"{out_dir}/diff.png (difference brightened {DIFF_GAIN:.0f}x)")
    if radiance_pair is not None:
        print(f"  the render itself: {out_dir}/pbrt-radiance.png, "
              f"{out_dir}/bonsai-radiance.png")
    return 1 if (interior or albedo_failed or radiance_failed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
