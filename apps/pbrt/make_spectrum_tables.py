#!/usr/bin/env python3
"""Regenerate apps/pbrt's spectral data headers from pbrt-v4's own source.

    python3 apps/pbrt/make_spectrum_tables.py

Two headers, from two places in pbrt, because pbrt itself keeps two copies of
the CIE curves at different resolutions and uses each for a different job:

  cie_tables.h        the 471-sample curves from src/pbrt/util/spectrum.cpp,
                      which the film integrates a SampledSpectrum against.

  rgb2spec_tables.h   the 95-sample curves, the D65 illuminant and the sRGB
                      matrices from src/pbrt/cmd/rgb2spec_opt.cpp, which is
                      what pbrt fits RGB albedos against. apps/pbrt runs that
                      same fit (see rgb2spec.h) rather than shipping the 9.4MB
                      table pbrt generates with it, so it has to fit against
                      the same numbers to get the same answers.

These are tabulated measurements, not something to approximate or type out: the
point of the app is that it renders with pbrt's numbers. The generated headers
are checked in so that building needs no network; run this again to pick up a
change upstream.
"""

import re
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/mmp/pbrt-v4/master/src/pbrt/"
SOURCES = {
    "spectrum.cpp": BASE + "util/spectrum.cpp",
    "spectrum.h": BASE + "util/spectrum.h",
    "rgb2spec_opt.cpp": BASE + "cmd/rgb2spec_opt.cpp",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def as_float_literal(text):
    """`360` on its own would spell `360f`, which is not a C++ literal."""
    if text.endswith("f"):
        return text
    has_point = "." in text or "e" in text or "E" in text
    return (text if has_point else text + ".0") + "f"


def numbers_in(body, what, divisor=None):
    # pbrt labels some of these tables with a comment inside the braces, which
    # would otherwise ride along on the value it precedes.
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", " ", body)
    body = body.replace("{", " ").replace("}", " ")
    values = [v.strip() for v in body.split(",") if v.strip()]
    if not values:
        raise SystemExit(f"{what} came out empty")

    resolved = []
    for v in values:
        # pbrt writes its illuminants through a normalising macro, as N(46.6383).
        wrapped = re.fullmatch(r"N\(\s*([0-9.eE+-]+)\s*\)", v)
        if wrapped:
            if divisor is None:
                raise SystemExit(f"{what} uses N() but no divisor was found")
            # Kept as the division rather than worked out here, so the header
            # holds the same expression pbrt compiles.
            resolved.append(f"({wrapped.group(1)} / {divisor})")
            continue
        if not re.fullmatch(r"[0-9.eE+-]+f?", v):
            raise SystemExit(f"{what} has a value that is not a number: {v!r}")
        resolved.append(v)
    return resolved


def extract_array(source, name, extent):
    match = re.search(
        r"\b" + re.escape(name) + r"\s*\[\s*" + extent + r"\s*\]\s*=\s*\{(.*?)\};",
        source,
        re.DOTALL,
    )
    if not match:
        raise SystemExit(f"could not find {name}")
    # The N() macro is redefined before each illuminant, so the one that
    # applies is the last defined before this table.
    divisors = re.findall(
        r"#define\s+N\(x\)\s*\(\s*x\s*/\s*([0-9.eE+-]+)\s*\)",
        source[: match.start()],
    )
    return numbers_in(match.group(1), name, divisors[-1] if divisors else None)


def extract_matrix(source, name):
    match = re.search(
        r"\b" + re.escape(name) + r"\s*\[3\]\[3\]\s*=\s*(\{.*?\});", source, re.DOTALL
    )
    if not match:
        raise SystemExit(f"could not find {name}")
    values = numbers_in(match.group(1), name)
    if len(values) != 9:
        raise SystemExit(f"{name} has {len(values)} entries, expected 9")
    return values


def extract_constant(source, name):
    # The terminator may be a comma: pbrt declares Lambda_min and Lambda_max
    # on one line.
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*([0-9.eE+-]+)\s*[;,f]", source)
    if not match:
        raise SystemExit(f"could not find {name}")
    return match.group(1)


def extract_define(source, name):
    """rgb2spec_opt.cpp spells its constants as `#define NAME value`."""
    match = re.search(
        r"^\s*#define\s+" + re.escape(name) + r"\s+([0-9.eE+-]+)\s*$",
        source,
        re.MULTILINE,
    )
    if not match:
        raise SystemExit(f"could not find #define {name}")
    return match.group(1)


def emit_array(out, ctype, name, extent, values, per_row=8):
    out.append(f"inline constexpr {ctype} {name}[{extent}] = {{")
    suffix = as_float_literal if ctype == "float" else (lambda v: v)
    for i in range(0, len(values), per_row):
        out.append("    " + ", ".join(suffix(v) for v in values[i:i + per_row]) + ",")
    out.append("};")
    out.append("")


def emit_matrix(out, name, values):
    out.append(f"inline constexpr double {name}[3][3] = {{")
    for r in range(3):
        out.append("    {" + ", ".join(values[r * 3:r * 3 + 3]) + "},")
    out.append("};")
    out.append("")


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def make_cie_tables(cpp, header):
    tables = {
        name: extract_array(cpp, name, "nCIESamples")
        for name in ("CIE_lambda", "CIE_X", "CIE_Y", "CIE_Z")
    }
    count = len(tables["CIE_X"])
    for name, values in tables.items():
        if len(values) != count:
            raise SystemExit(f"{name} has {len(values)} entries, CIE_X has {count}")

    out = [
        "// Generated by apps/pbrt/make_spectrum_tables.py. Do not edit.",
        "//",
        "// The CIE 1931 standard observer's matching curves, taken verbatim from",
        "// pbrt-v4's src/pbrt/util/spectrum.cpp so that apps/pbrt and pbrt answer",
        "// with the same numbers. The driver hands these to bonsai as extern",
        "// arrays: bonsai has no file I/O, and tabulated constants arrive the",
        "// same way scene data does.",
        "#pragma once",
        "",
        f"inline constexpr int CIE_SAMPLES = {count};",
        f"inline constexpr float CIE_Y_INTEGRAL = "
        f"{as_float_literal(extract_constant(header, 'CIE_Y_integral'))};",
        f"inline constexpr float CIE_LAMBDA_MIN = "
        f"{as_float_literal(extract_constant(header, 'Lambda_min'))};",
        f"inline constexpr float CIE_LAMBDA_MAX = "
        f"{as_float_literal(extract_constant(header, 'Lambda_max'))};",
        "",
    ]
    for name in ("CIE_lambda", "CIE_X", "CIE_Y", "CIE_Z"):
        emitted = "CIE_LAMBDA" if name == "CIE_lambda" else name.upper()
        emit_array(out, "float", emitted, "CIE_SAMPLES", tables[name])
    write("apps/pbrt/cie_tables.h", out)
    return count


def make_rgb2spec_tables(opt):
    count = int(extract_define(opt, "CIE_SAMPLES"))
    tables = {
        name: extract_array(opt, name, "CIE_SAMPLES")
        for name in ("cie_x", "cie_y", "cie_z", "cie_d65")
    }
    for name, values in tables.items():
        if len(values) != count:
            raise SystemExit(f"{name} has {len(values)} entries, expected {count}")

    out = [
        "// Generated by apps/pbrt/make_spectrum_tables.py. Do not edit.",
        "//",
        "// What pbrt fits an RGB albedo against, taken verbatim from pbrt-v4's",
        "// src/pbrt/cmd/rgb2spec_opt.cpp. These are the same CIE curves as in",
        "// cie_tables.h at a coarser spacing, plus the D65 illuminant and the",
        "// sRGB matrices; pbrt's fitting tool uses these rather than the dense",
        "// ones, and apps/pbrt runs the same fit, so it uses them too.",
        "#pragma once",
        "",
        f"inline constexpr int RGB2SPEC_CIE_SAMPLES = {count};",
        "inline constexpr double RGB2SPEC_CIE_LAMBDA_MIN = "
        f"{extract_define(opt, 'CIE_LAMBDA_MIN')};",
        "inline constexpr double RGB2SPEC_CIE_LAMBDA_MAX = "
        f"{extract_define(opt, 'CIE_LAMBDA_MAX')};",
        "",
    ]
    # Prefixed, because cie_tables.h already has a CIE_X -- the same curve at
    # a different resolution, which is exactly the confusion worth preventing.
    for name in ("cie_x", "cie_y", "cie_z", "cie_d65"):
        emit_array(out, "double", "RGB2SPEC_" + name.upper(),
                   "RGB2SPEC_CIE_SAMPLES", tables[name], per_row=5)
    for name in ("xyz_to_srgb", "srgb_to_xyz"):
        emit_matrix(out, name.upper(), extract_matrix(opt, name))
    write("apps/pbrt/rgb2spec_tables.h", out)
    return count


def main():
    sources = {name: fetch(url) for name, url in SOURCES.items()}
    dense = make_cie_tables(sources["spectrum.cpp"], sources["spectrum.h"])
    coarse = make_rgb2spec_tables(sources["rgb2spec_opt.cpp"])
    print(f"  film curves: {dense} samples; fitting curves: {coarse} samples")


if __name__ == "__main__":
    sys.exit(main())
