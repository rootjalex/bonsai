#include "vectorize_skip.h"

#include <array>
#include <cmath>
#include <iostream>

// Used in vectorize_skip.bonsai.
//
// Three gangs: one where every lane takes the arm and loops, one where no
// lane does either (so every guarded block is skipped), and one mixed. Each
// lane is compared with the same input weighed alone, and the mixed gang is
// printed.
namespace {

bool same_as_scalar(const std::array<float, 8> &a) {
    std::array<float, 8> out{};
    weigh_all(a, out);
    bool same = true;
    for (size_t i = 0; i < a.size(); i++) {
        same = same && std::fabs(weigh_one(a[i]) - out[i]) <= 1e-5f;
    }
    return same;
}

} // namespace

int main() {
    const std::array<float, 8> all = {1.0f, 2.0f, 3.0f, 4.0f,
                                      5.0f, 6.0f, 7.0f, 8.0f};
    const std::array<float, 8> none = {-1.0f, -2.0f, 0.0f,  -4.0f,
                                       -5.0f, -0.5f, -7.0f, -8.0f};
    const std::array<float, 8> some = {-1.0f, 2.0f, 0.0f, 4.5f,
                                       -5.0f, 6.0f, -7.0f, 8.0f};
    std::array<float, 8> out{};
    weigh_all(some, out);
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
