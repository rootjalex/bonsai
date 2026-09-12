#include "vectorize_arm_local.h"

#include <iostream>

// Used in vectorize_arm_local.bonsai
namespace {

// The scalar function the gang has to agree with, written out by hand.
float clip_scalar(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    float y = x * 2.0f;
    if (y > hi) {
        y = hi;
    }
    return y + 1.0f;
}

} // namespace

int main() {
    // Lanes below the floor, between, and above the ceiling, with the
    // boundaries on both sides of every test.
    std::array<float, 8> a = {-4, -1, 0, 1, 2, 20, 5, -2};
    const Box b{0.0f, 10.0f};

    std::array<float, 8> out{};
    clip_all(a, b, out);

    bool same = true;
    for (size_t i = 0; i < a.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
        same = same && out[i] == clip_scalar(a[i], b.lo, b.hi);
    }
    std::cout << '\n' << (same ? "matches scalar" : "DIFFERS from scalar")
              << '\n';
}
