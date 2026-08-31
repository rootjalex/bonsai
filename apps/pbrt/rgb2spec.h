// Fitting an RGB albedo to a reflectance spectrum, the way pbrt does it.
//
// This is a port of pbrt-v4's src/pbrt/cmd/rgb2spec_opt.cpp, which implements
// Jakob and Hanika's "A Low-Dimensional Function Space for Efficient Spectral
// Upsampling" (2019): a reflectance is represented by three coefficients of a
// quadratic in wavelength put through a sigmoid, and the coefficients are
// found by Gauss-Newton against the residual in CIE Lab.
//
// pbrt runs this offline over a 64x64x64 grid and ships the result as a 9.4MB
// table so that any RGB can be converted at load time. That table is generated
// rather than checked in -- it is not in pbrt's repository -- and a scene with
// a handful of albedos does not need it: the same fit run on those few colours
// gives the same coefficients the table would have been interpolated to. So
// this is pbrt's algorithm against pbrt's numbers, applied per colour.
//
// The coefficients are what crosses into bonsai. Evaluating them is
// RGBSigmoidPolynomial, which lives in render.bonsai because it runs per
// wavelength per hit.
#pragma once

#include "rgb2spec_tables.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rgb2spec {

// pbrt: the quadrature is finer than the tables, three segments per 5nm step.
inline constexpr int FineSamples = (RGB2SPEC_CIE_SAMPLES - 1) * 3 + 1;
inline constexpr double Epsilon = 1e-4;

struct Tables {
    double lambda[FineSamples];
    double rgb[3][FineSamples];
    double xyz_whitepoint[3];
};

inline double sigmoid(double x) {
    return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5;
}

inline double cie_interp(const double *data, double x) {
    x -= RGB2SPEC_CIE_LAMBDA_MIN;
    x *= (RGB2SPEC_CIE_SAMPLES - 1) /
         (RGB2SPEC_CIE_LAMBDA_MAX - RGB2SPEC_CIE_LAMBDA_MIN);
    int offset = (int)x;
    if (offset < 0) {
        offset = 0;
    }
    if (offset > RGB2SPEC_CIE_SAMPLES - 2) {
        offset = RGB2SPEC_CIE_SAMPLES - 2;
    }
    const double weight = x - offset;
    return (1.0 - weight) * data[offset] + weight * data[offset + 1];
}

// pbrt: a composite quadrature integrating the CIE curves, the reflectance and
// the illuminant over each 5nm segment with Simpson's 3/8 rule. The CIE curves
// and the illuminant are linear over a segment; the reflectance need not be,
// which is why the rule is fourth-order rather than trapezoidal.
inline Tables init_tables() {
    Tables t{};
    const double h =
        (RGB2SPEC_CIE_LAMBDA_MAX - RGB2SPEC_CIE_LAMBDA_MIN) / (FineSamples - 1);

    for (int i = 0; i < FineSamples; ++i) {
        const double lambda = RGB2SPEC_CIE_LAMBDA_MIN + i * h;
        const double xyz[3] = {cie_interp(RGB2SPEC_CIE_X, lambda),
                               cie_interp(RGB2SPEC_CIE_Y, lambda),
                               cie_interp(RGB2SPEC_CIE_Z, lambda)};
        const double I = cie_interp(RGB2SPEC_CIE_D65, lambda);

        double weight = 3.0 / 8.0 * h;
        if (i == 0 || i == FineSamples - 1) {
            // The end points keep the plain weight.
        } else if ((i - 1) % 3 == 2) {
            weight *= 2.0;
        } else {
            weight *= 3.0;
        }

        t.lambda[i] = lambda;
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                t.rgb[k][i] += XYZ_TO_SRGB[k][j] * xyz[j] * I * weight;
            }
        }
        for (int j = 0; j < 3; ++j) {
            t.xyz_whitepoint[j] += xyz[j] * I * weight;
        }
    }
    return t;
}

// pbrt: the residual is measured in CIE Lab, so that the fit is even in
// perceived error rather than in linear RGB.
inline void cie_lab(const Tables &t, double *p) {
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int j = 0; j < 3; ++j) {
        X += p[j] * SRGB_TO_XYZ[0][j];
        Y += p[j] * SRGB_TO_XYZ[1][j];
        Z += p[j] * SRGB_TO_XYZ[2][j];
    }
    const double Xw = t.xyz_whitepoint[0];
    const double Yw = t.xyz_whitepoint[1];
    const double Zw = t.xyz_whitepoint[2];

    auto f = [](double v) -> double {
        const double delta = 6.0 / 29.0;
        if (v > delta * delta * delta) {
            return std::cbrt(v);
        }
        return v / (delta * delta * 3.0) + (4.0 / 29.0);
    };

    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

inline void eval_residual(const Tables &t, const double *coeffs,
                          const double *rgb, double *residual) {
    double out[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < FineSamples; ++i) {
        // Wavelength scaled into 0..1, which is what the polynomial is in.
        const double lambda =
            (t.lambda[i] - RGB2SPEC_CIE_LAMBDA_MIN) /
            (RGB2SPEC_CIE_LAMBDA_MAX - RGB2SPEC_CIE_LAMBDA_MIN);

        double x = 0.0;
        for (int k = 0; k < 3; ++k) {
            x = x * lambda + coeffs[k];
        }
        const double s = sigmoid(x);

        for (int j = 0; j < 3; ++j) {
            out[j] += t.rgb[j][i] * s;
        }
    }
    cie_lab(t, out);
    std::memcpy(residual, rgb, sizeof(double) * 3);
    cie_lab(t, residual);

    for (int j = 0; j < 3; ++j) {
        residual[j] -= out[j];
    }
}

inline void eval_jacobian(const Tables &t, const double *coeffs,
                          const double *rgb, double **jac) {
    double r0[3], r1[3], tmp[3];
    for (int i = 0; i < 3; ++i) {
        std::memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] -= Epsilon;
        eval_residual(t, tmp, rgb, r0);

        std::memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] += Epsilon;
        eval_residual(t, tmp, rgb, r1);

        for (int j = 0; j < 3; ++j) {
            jac[j][i] = (r1[j] - r0[j]) / (2 * Epsilon);
        }
    }
}

// LU decomposition and triangular solve, as in pbrt (which took them from
// Wikipedia). Only ever 3x3 here.
inline bool lup_decompose(double **A, int N, double tol, int *P) {
    for (int i = 0; i <= N; ++i) {
        P[i] = i;
    }
    for (int i = 0; i < N; ++i) {
        double maxA = 0.0;
        int imax = i;
        for (int k = i; k < N; ++k) {
            const double absA = std::fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }
        if (maxA < tol) {
            return false;
        }
        if (imax != i) {
            std::swap(P[i], P[imax]);
            std::swap(A[i], A[imax]);
            P[N]++;
        }
        for (int j = i + 1; j < N; ++j) {
            A[j][i] /= A[i][i];
            for (int k = i + 1; k < N; ++k) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }
    return true;
}

inline void lup_solve(double **A, const int *P, const double *b, int N,
                      double *x) {
    for (int i = 0; i < N; ++i) {
        x[i] = b[P[i]];
        for (int k = 0; k < i; ++k) {
            x[i] -= A[i][k] * x[k];
        }
    }
    for (int i = N - 1; i >= 0; --i) {
        for (int k = i + 1; k < N; ++k) {
            x[i] -= A[i][k] * x[k];
        }
        x[i] /= A[i][i];
    }
}

// pbrt: gauss_newton, fifteen iterations, clamped so the coefficients cannot
// run away and stopping once the residual is small.
inline void gauss_newton(const Tables &t, const double rgb[3], double coeffs[3],
                         int iterations = 15) {
    for (int i = 0; i < iterations; ++i) {
        double J0[3], J1[3], J2[3];
        double *J[3] = {J0, J1, J2};
        double residual[3];

        eval_residual(t, coeffs, rgb, residual);
        eval_jacobian(t, coeffs, rgb, J);

        int P[4];
        if (!lup_decompose(J, 3, 1e-15, P)) {
            throw std::runtime_error("rgb2spec: LU decomposition failed");
        }

        double x[3];
        lup_solve(J, P, residual, 3, x);

        double r = 0.0;
        for (int j = 0; j < 3; ++j) {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }

        const double max = std::max(std::max(coeffs[0], coeffs[1]), coeffs[2]);
        if (max > 200) {
            for (int j = 0; j < 3; ++j) {
                coeffs[j] *= 200 / max;
            }
        }

        if (r < 1e-6) {
            break;
        }
    }
}

// The three coefficients for one albedo: c0*l^2 + c1*l + c2, with l a
// wavelength in nanometres. pbrt's RGBSigmoidPolynomial takes them this way.
struct Coefficients {
    float c0, c1, c2;
};

inline Coefficients fit(const Tables &t, float r, float g, float b) {
    const double rgb[3] = {r, g, b};
    double coeffs[3] = {0.0, 0.0, 0.0};
    gauss_newton(t, rgb, coeffs);

    // The fit works in wavelength scaled to 0..1, because that is the range
    // the polynomial is conditioned for; what comes out has to be in
    // nanometres, because that is what a wavelength is everywhere else. This
    // is the substitution l -> (l - 360) / (830 - 360) multiplied out, and is
    // pbrt's own rescaling from the end of rgb2spec_opt.cpp.
    const double k0 = RGB2SPEC_CIE_LAMBDA_MIN;
    const double k1 = 1.0 / (RGB2SPEC_CIE_LAMBDA_MAX - RGB2SPEC_CIE_LAMBDA_MIN);
    const double A = coeffs[0], B = coeffs[1], C = coeffs[2];
    return Coefficients{
        float(A * k1 * k1),
        float(B * k1 - 2 * A * k0 * k1 * k1),
        float(C - B * k0 * k1 + A * (k0 * k1) * (k0 * k1)),
    };
}

} // namespace rgb2spec
