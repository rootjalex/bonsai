#include "vectorize_skip_cursor.h"

#include <array>
#include <cmath>
#include <iostream>

// Used in vectorize_skip_cursor.bonsai.
//
// Three gangs. In the first every lane is a Ring, so the Ring arms run with
// every lane on and the Disc and Box arms of the skipped match are bypassed;
// in the second no lane is a Ring, so the Ring arms are bypassed while the
// Disc and Box arms run under partial masks; the third has lanes in every
// arm. Each lane is compared with the same input weighed alone, and the mixed
// gang is printed.
namespace {

bool same_as_scalar(const std::array<float, 8> &a) {
    std::array<float, 8> out{};
    weigh_all(a, out);
    bool same = true;
    for (size_t i = 0; i < a.size(); i++) {
        same = same && std::fabs(weigh_one(a[i]) - out[i]) <= 1e-4f;
    }
    return same;
}

} // namespace

int main() {
    const std::array<float, 8> rings = {5.0f, 6.0f,  7.0f,  8.0f,
                                        9.0f, 10.0f, 11.0f, 12.0f};
    const std::array<float, 8> none = {-1.0f, -2.0f, 0.0f, 3.5f,
                                       1.0f,  -0.5f, 2.0f, -8.0f};
    const std::array<float, 8> some = {-1.0f, 3.0f, 0.0f, 4.5f,
                                       -5.0f, 2.5f, 8.0f, 1.0f};
    std::array<float, 8> out{};
    weigh_all(some, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n'
              << (same_as_scalar(rings) && same_as_scalar(none) &&
                          same_as_scalar(some)
                      ? "same as scalar"
                      : "DIFFERS from scalar")
              << '\n';
}
