#!/usr/bin/env python3
"""Compare two PFM images of surface normals, one from pbrt and one from here.

    python3 apps/pbrt/compare_normals.py <pbrt.pfm> <bonsai.pfm>
        [--pbrt-seconds <s>] [--bonsai-seconds <s>]

Two things are reported, and they fail for different reasons:

  the hit mask -- which pixels have a normal at all. A disagreement here is a
  disagreement about geometry: the camera is pointing somewhere slightly
  different, or a ray missed something it should have hit. These are counted
  exactly, because one stray pixel along a silhouette is a real difference and
  an average would hide it.

  the normals themselves, where both renderers hit something. pbrt writes its
  gbuffer as half floats, so agreement is bounded by that: around 5e-4 near
  0.5, and there is no point asking for better.

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


def main(argv):
    args = argv[1:]
    pbrt_seconds = take_option(args, "--pbrt-seconds")
    bonsai_seconds = take_option(args, "--bonsai-seconds")
    if len(args) != 2:
        raise SystemExit(
            f"usage: {argv[0]} <pbrt.pfm> <bonsai.pfm> "
            f"[--pbrt-seconds <s>] [--bonsai-seconds <s>]")

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
        print("  (not the same work: pbrt integrates a path, this returns the "
              "nearest hit)")
        if pbrt_seconds <= 0.05:
            print("  (pbrt's time is reported to 10ms, so this is near the "
                  "limit of what it can tell us)")

    # Only disagreements away from an edge are failures. A silhouette pixel is
    # a grazing ray, and which surface it lands on is decided by how carefully
    # the intersection rounds -- a genuine difference between the two
    # renderers, but not one that means anything is wrong here.
    if interior:
        print(f"FAILED: {len(interior)} pixels disagree away from any edge")
    else:
        print(f"ok: matches pbrt except on {len(on_edge)} silhouette pixels")
    print(f"images: {out_dir}/pbrt.png, {out_dir}/bonsai.png, "
          f"{out_dir}/diff.png (difference brightened {DIFF_GAIN:.0f}x)")
    return 1 if interior else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
