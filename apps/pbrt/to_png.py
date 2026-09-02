#!/usr/bin/env python3
"""Encode a linear float image for viewing.

    python3 apps/pbrt/to_png.py <in.pfm> <out.png> [--normals] [--exposure <n>]

The renderer emits linear float, the way pbrt does; this is the step pbrt does
inside Image::Write when the filename ends in .png. By default that is pbrt's
own sRGB quantisation, so a PNG made here and a PNG made by pbrt from the same
values hold the same bytes.

--normals instead remaps [-1,1] to [0,1] and leaves a miss black, for when the
image holds surface normals rather than radiance. pbrt would sRGB-encode those
too and clamp every negative component to zero, which throws away half of every
normal; that is the right thing for radiance and useless for a direction.

--exposure <n> scales by 2**n and rolls the result off before encoding, for
looking at a radiance image whose bright pixels are far above the display white
point -- which a random walk's are. It is a viewing aid and deliberately not the
default: it is not invertible, so two images encoded with it cannot be compared,
and the default has to stay byte-identical to what pbrt writes.
"""

import sys

from image_io import encode_normals, encode_radiance, read_pfm, write_png


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = [a for a in argv[1:] if a.startswith("--")]

    stops = None
    if "--exposure" in flags:
        at = argv.index("--exposure")
        if at + 1 >= len(argv):
            raise SystemExit("--exposure needs a number of stops")
        stops = float(argv[at + 1])
        args = [a for a in args if a != argv[at + 1]]
        flags.remove("--exposure")

    if len(args) != 2 or set(flags) - {"--normals"}:
        raise SystemExit(
            f"usage: {argv[0]} <in.pfm> <out.png> [--normals] "
            f"[--exposure <stops>]")
    if "--normals" in flags and stops is not None:
        raise SystemExit("--exposure means nothing for normals")

    width, height, values = read_pfm(args[0])
    if "--normals" in flags:
        encoded = encode_normals(width, height, values)
    else:
        encoded = encode_radiance(width, height, values, stops)
    write_png(args[1], width, height, encoded)
    print(f"wrote {args[1]} ({width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
