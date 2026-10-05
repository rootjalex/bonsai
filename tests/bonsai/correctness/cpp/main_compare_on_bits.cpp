// Used in compare_on_bits.bonsai.
//
// The program's four functions, run over the values on which a comparison
// of floats and a comparison of their bits could part -- a NaN of either
// sign on either side, a negative zero, the least denormal, the greatest
// finite float, the infinities -- against the same functions written in
// C++'s float arithmetic, which is what the program means. The simplifier
// compares `known`, `joined` and `gang` on the bits (the right side a
// non-negative number, the left side's sign clear) and leaves `unknown` on
// the floats (its right side may be a NaN); every answer has to agree.
#include "compare_on_bits.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool ref_unknown(float a, float b) { return std::fabs(a) < std::fabs(b); }

bool ref_known(float a, float b) {
    if (!(0.0f < b)) {
        return false;
    }
    return std::fabs(a) < b;
}

bool ref_joined(float a, uint32_t n, float z) {
    if (!(0.0f < z)) {
        return false;
    }
    const float x = std::fabs(a);
    return x <= float(n) && x < z;
}

std::string show(float v) {
    if (std::isnan(v)) {
        return std::signbit(v) ? "-nan" : "nan";
    }
    if (v == 0.0f) {
        return std::signbit(v) ? "-0" : "0";
    }
    if (std::isinf(v)) {
        return std::signbit(v) ? "-inf" : "inf";
    }
    if (v == std::numeric_limits<float>::denorm_min()) {
        return "denorm_min";
    }
    if (v == std::numeric_limits<float>::max()) {
        return "max";
    }
    return std::to_string(v);
}

int failures = 0;

void check(const std::string &what, bool got, bool expected) {
    if (got != expected) {
        std::cout << what << ": got " << got << ", expected " << expected << " -- WRONG\n";
        failures++;
    }
}

} // namespace

int main() {
    const float nan = std::bit_cast<float>(0x7fc00000u);
    const float negative_nan = std::bit_cast<float>(0xffc00000u);
    const float inf = std::numeric_limits<float>::infinity();
    const float denorm = std::numeric_limits<float>::denorm_min();
    const float largest = std::numeric_limits<float>::max();
    const std::vector<float> as = {0.0f, -0.0f,  1.0f, -1.0f, denorm, -denorm,
                                   3.5f, largest, inf,  -inf,  nan,    negative_nan};
    const std::vector<float> bs = {nan,  negative_nan, -0.0f, 0.0f, denorm,
                                   1.0f, 3.5f,         inf,   -1.0f};
    const std::vector<uint32_t> ns = {0u, 1u, 3u, 4000000000u};

    // A few by name, so that the output says what the edge cases do.
    std::cout << "known(nan, 2) = " << known(nan, 2.0f) << "\n";
    std::cout << "known(-nan, 2) = " << known(negative_nan, 2.0f) << "\n";
    std::cout << "known(1, nan) = " << known(1.0f, nan) << "\n";
    std::cout << "known(-0, denorm_min) = " << known(-0.0f, denorm) << "\n";
    std::cout << "known(-1, 1) = " << known(-1.0f, 1.0f) << "\n";
    std::cout << "known(max, inf) = " << known(largest, inf) << "\n";
    std::cout << "joined(-0, 0, 1) = " << joined(-0.0f, 0u, 1.0f) << "\n";
    std::cout << "joined(1, 1, 1) = " << joined(1.0f, 1u, 1.0f) << "\n";
    std::cout << "joined(nan, 4000000000, inf) = " << joined(nan, 4000000000u, inf) << "\n";
    std::cout << "unknown(1, nan) = " << unknown(1.0f, nan) << "\n";
    std::cout << "unknown(nan, 1) = " << unknown(nan, 1.0f) << "\n";

    size_t cases = 0;
    for (float a : as) {
        for (float b : bs) {
            check("unknown(" + show(a) + ", " + show(b) + ")", unknown(a, b),
                  ref_unknown(a, b));
            check("known(" + show(a) + ", " + show(b) + ")", known(a, b), ref_known(a, b));
            cases += 2;
            for (uint32_t n : ns) {
                check("joined(" + show(a) + ", " + std::to_string(n) + ", " + show(b) + ")",
                      joined(a, n, b), ref_joined(a, n, b));
                cases++;
            }
        }
    }
    // The gang: eight lanes of `a` against eight `n`, one `z` for all.
    for (float z : bs) {
        for (size_t start = 0; start + 8 <= as.size(); start += 4) {
            std::array<float, 8> lanes_a{};
            std::array<uint32_t, 8> lanes_n{};
            for (size_t k = 0; k < 8; k++) {
                lanes_a[k] = as[start + k];
                lanes_n[k] = ns[k % ns.size()];
            }
            std::array<bool, 8> out{};
            gang(lanes_a, lanes_n, z, out);
            for (size_t k = 0; k < 8; k++) {
                check("gang lane " + std::to_string(k) + " (" + show(lanes_a[k]) + ", " +
                          std::to_string(lanes_n[k]) + ", " + show(z) + ")",
                      out[k], ref_joined(lanes_a[k], lanes_n[k], z));
                cases++;
            }
        }
    }
    std::cout << cases << " cases, " << failures << " wrong\n";
    return failures == 0 ? 0 : 1;
}
