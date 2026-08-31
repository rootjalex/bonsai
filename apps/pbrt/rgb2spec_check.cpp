// Does the RGB-to-spectrum fit reproduce the colour it was given?
//
// rgb2spec.h is a port of pbrt's fitting tool, and a port is worth only as
// much as its round trip: fit an RGB to sigmoid coefficients, integrate the
// resulting reflectance against the CIE curves under D65, convert back to
// sRGB, and compare. A fit that had the polynomial's domain wrong, or the
// coefficient rescaling backwards, still produces plausible-looking numbers --
// it just produces the wrong colour.
//
//     clang++ -std=c++20 -O2 -I. apps/pbrt/rgb2spec_check.cpp -o rgb2spec_check
//
// Run by apps/pbrt/render.sh.
#include "rgb2spec.h"

#include <cmath>
#include <cstdio>

namespace {

// The colour a fitted reflectance actually is, integrated the same way
// init_tables integrates, so that this measures the fit rather than the
// difference between two quadrature schemes.
void integrate(const rgb2spec::Tables &t, const rgb2spec::Coefficients &k,
               double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    for (int i = 0; i < rgb2spec::FineSamples; ++i) {
        const double lambda = t.lambda[i];
        const double s =
            rgb2spec::sigmoid((k.c0 * lambda + k.c1) * lambda + k.c2);
        for (int j = 0; j < 3; ++j) {
            out[j] += t.rgb[j][i] * s;
        }
    }
}

double max_error(const double out[3], float r, float g, float b) {
    return std::fmax(std::fabs(out[0] - r),
                     std::fmax(std::fabs(out[1] - g), std::fabs(out[2] - b)));
}

} // namespace

int main() {
    const rgb2spec::Tables t = rgb2spec::init_tables();

    struct Case {
        const char *name;
        float r, g, b;
    };
    // Saturated colours are the hard ones: a reflectance that is high in one
    // part of the spectrum and low elsewhere is what the sigmoid has to bend
    // to fit.
    const Case cases[] = {
        {"black", 0.0f, 0.0f, 0.0f},    {"dark", 0.05f, 0.05f, 0.05f},
        {"mid grey", 0.5f, 0.5f, 0.5f}, {"near white", 0.99f, 0.99f, 0.99f},
        {"red", 0.8f, 0.1f, 0.1f},      {"green", 0.1f, 0.7f, 0.2f},
        {"blue", 0.15f, 0.2f, 0.75f},   {"orange", 0.9f, 0.5f, 0.1f},
        {"cyan", 0.1f, 0.75f, 0.8f},    {"magenta", 0.75f, 0.1f, 0.7f},
    };

    double worst = 0.0;
    for (const Case &c : cases) {
        const rgb2spec::Coefficients k = rgb2spec::fit(t, c.r, c.g, c.b);
        double out[3];
        integrate(t, k, out);
        const double err = max_error(out, c.r, c.g, c.b);
        worst = std::fmax(worst, err);
        printf("%-11s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err %.1e\n", c.name,
               c.r, c.g, c.b, out[0], out[1], out[2], err);
    }

    // Pure white is the one colour this representation cannot reach, and it
    // is not a fault in the fit: it asks for a reflectance of exactly 1 at
    // every wavelength, and a sigmoid only approaches 1. pbrt's shipped table
    // is built by this same fit and has the same limit. Checked rather than
    // skipped, so that it stays a known bound instead of a surprise.
    const rgb2spec::Coefficients white = rgb2spec::fit(t, 1.0f, 1.0f, 1.0f);
    double out[3];
    integrate(t, white, out);
    const double white_err = max_error(out, 1.0f, 1.0f, 1.0f);
    printf("%-11s (1.00 1.00 1.00) -> (%.4f %.4f %.4f)  err %.1e (expected)\n",
           "pure white", out[0], out[1], out[2], white_err);

    const double tolerance = 1e-4;
    printf("worst error away from pure white: %.1e (tolerance %.0e)\n", worst,
           tolerance);
    if (worst > tolerance) {
        printf("FAILED\n");
        return 1;
    }
    if (white_err > 0.05) {
        printf("FAILED: pure white is worse than the representation's limit\n");
        return 1;
    }
    printf("ok\n");
    return 0;
}
