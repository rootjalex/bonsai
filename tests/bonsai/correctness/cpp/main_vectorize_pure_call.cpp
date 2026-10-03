#include "vectorize_pure_call.h"

#include <array>
#include <cmath>
#include <iostream>

// Used in vectorize_pure_call.bonsai.
//
// Three gangs. In the first every lane passes `x > 2.0`, so the pure call
// is made for all of them; in the second no lane does, so the call is made
// for nobody and its answer read by nobody; the third is mixed, and lanes
// above 4.3 fail the pure call's own test (x * x + 1 < 20). Each lane is
// compared with the same input picked alone, and the mixed gang is printed.
namespace {

const std::array<float, 4> table = {0.25f, 0.5f, 0.75f, 1.0f};

bool same_as_scalar(const std::array<float, 8> &a) {
    std::array<float, 8> out{};
    pick_all(a, table, out);
    bool same = true;
    for (size_t i = 0; i < a.size(); i++) {
        same = same && std::fabs(pick_one(a[i], table) - out[i]) <= 1e-4f;
    }
    return same;
}

} // namespace

int main() {
    const std::array<float, 8> all = {2.5f, 3.0f, 3.5f, 4.0f,
                                      4.5f, 5.0f, 6.0f, 7.0f};
    const std::array<float, 8> none = {-1.0f, -2.0f, 0.0f, 2.0f,
                                       1.0f,  -0.5f, 1.5f, -8.0f};
    const std::array<float, 8> some = {-1.0f, 3.0f, 0.0f, 4.5f,
                                       9.0f,  2.5f, 8.0f, 1.0f};
    std::array<float, 8> out{};
    pick_all(some, table, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n'
              << (same_as_scalar(all) && same_as_scalar(none) &&
                          same_as_scalar(some)
                      ? "same as scalar"
                      : "DIFFERS from scalar")
              << '\n';
}
