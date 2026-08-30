#!/usr/bin/env python3
"""Encode a linear float image for viewing.

    python3 apps/pbrt/to_png.py <in.pfm> <out.png> [--normals]

The renderer emits linear float, the way pbrt does; this is the step pbrt does
inside Image::Write when the filename ends in .png. By default that is pbrt's
own sRGB quantisation, so a PNG made here and a PNG made by pbrt from the same
values hold the same bytes.

--normals instead remaps [-1,1] to [0,1] and leaves a miss black, for when the
image holds surface normals rather than radiance. pbrt would sRGB-encode those
too and clamp every negative component to zero, which throws away half of every
normal; that is the right thing for radiance and useless for a direction.
"""

import sys

from image_io import encode_normals, encode_radiance, read_pfm, write_png


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if len(args) != 2 or flags - {"--normals"}:
        raise SystemExit(f"usage: {argv[0]} <in.pfm> <out.png> [--normals]")

    width, height, values = read_pfm(args[0])
    encode = encode_normals if "--normals" in flags else encode_radiance
    write_png(args[1], width, height, encode(width, height, values))
    print(f"wrote {args[1]} ({width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
