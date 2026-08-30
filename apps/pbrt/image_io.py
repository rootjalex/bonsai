"""Reading the float images this app writes, and encoding them for viewing.

pbrt writes the film's linear values to .exr, .pfm and .hdr, and only quantizes
when asked for .png or .qoi. apps/pbrt does the same: the renderer emits linear
float and nothing else, and turning that into something to look at happens
afterwards, here.

PNG is written by hand from the standard library. This is tooling for a
renderer comparison, and having it depend on an image library that may not be
installed would make it the thing that breaks.
"""

import struct
import zlib


def read_pfm(path):
    """(width, height, floats) with pixel (0,0) at the top left."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"PF":
            raise SystemExit(f"{path}: not a three-channel PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        count = width * height * 3
        data = struct.unpack(
            ("<" if scale < 0 else ">") + str(count) + "f", f.read(count * 4)
        )
    # PFM rows run bottom to top.
    rows = [data[j * width * 3:(j + 1) * width * 3] for j in range(height)]
    rows.reverse()
    return width, height, [v for row in rows for v in row]


def write_pfm(path, width, height, values):
    with open(path, "wb") as f:
        # A negative scale says little-endian.
        f.write(b"PF\n%d %d\n-1.000000\n" % (width, height))
        for j in range(height - 1, -1, -1):
            row = values[j * width * 3:(j + 1) * width * 3]
            f.write(struct.pack("<" + str(len(row)) + "f", *row))


def linear_to_srgb(value):
    """pbrt's LinearToSRGB, from src/pbrt/util/color.h.

    The sRGB transfer function below the linear knee, and above it the minimax
    rational approximation pbrt uses rather than a pow. Reproduced rather than
    replaced with the exact analytic curve so that a PNG from here and a PNG
    from pbrt hold the same bytes.
    """
    if value <= 0.0031308:
        return 12.92 * value

    def poly(t, coefficients):
        # pbrt's EvaluatePolynomial takes the constant term first.
        result = 0.0
        for c in reversed(coefficients):
            result = result * t + c
        return result

    s = value ** 0.5 if value > 0 else 0.0
    p = poly(s, [-0.0016829072605308378, 0.03453868659826638,
                 0.7642611304733891, 2.0041169284241644,
                 0.7551545191665577, -0.016202083165206348])
    q = poly(s, [4.178892964897981e-7, -0.00004375359692957097,
                 0.03467195408529984, 0.6085338522168684,
                 1.8970238036421054, 1.0])
    return p / q * value


def linear_to_srgb8(value):
    """pbrt's LinearToSRGB8: the encoded value rounded to a byte."""
    if value <= 0:
        return 0
    if value >= 1:
        return 255
    return max(0, min(255, int(round(255.0 * linear_to_srgb(value)))))


def write_png(path, width, height, pixels):
    """An 8-bit RGB PNG. `pixels` is width*height*3 bytes, top row first."""

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    # Each row is prefixed with its filter type, and 0 means "no filter".
    raw = b"".join(
        b"\x00" + pixels[y * width * 3:(y + 1) * width * 3]
        for y in range(height)
    )
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        # Bit depth 8, colour type 2 (truecolour), no interlacing.
        f.write(chunk(b"IHDR",
                      struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def encode_radiance(width, height, values):
    """Linear radiance to 8-bit sRGB, which is what pbrt writes to a PNG."""
    out = bytearray(width * height * 3)
    for i in range(width * height * 3):
        out[i] = linear_to_srgb8(values[i])
    return bytes(out)


def encode_normals(width, height, values):
    """Normals as colour: [-1,1] remapped to [0,1], a miss left black.

    Not what pbrt would write for these floats -- it would sRGB-encode them and
    clamp everything negative to zero, which throws away half of every normal.
    This is a visualisation, and it is only ever compared against the same
    visualisation of pbrt's own normals.
    """
    out = bytearray(width * height * 3)
    for i in range(width * height):
        n = values[i * 3:i * 3 + 3]
        if any(n):
            for k in range(3):
                v = 0.5 * (n[k] + 1.0)
                out[i * 3 + k] = max(0, min(255, int(256.0 * v)))
    return bytes(out)
