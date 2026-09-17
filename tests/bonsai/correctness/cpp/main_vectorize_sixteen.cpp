#include "vectorize_sixteen.h"

#include <array>
#include <iostream>

// Used in vectorize_sixteen.bonsai: the same computation in scalar C++, and
// the two printed side by side.
int main() {
    std::array<float, 64> a{};
    std::array<int32_t, 64> which{};
    for (int i = 0; i < 64; i++) {
        a[i] = float(i % 17);
        // A permutation, so every lane gathers from somewhere other than
        // its own slot.
        which[i] = (i * 37 + 11) % 64;
    }

    std::array<float, 4> out{};
    std::array<float, 64> per_lane{};
    work(a, which, out, per_lane);

    std::array<float, 4> expect{};
    std::array<float, 64> expect_lane{};
    for (int p = 0; p < 4; p++) {
        for (int s = 0; s < 16; s++) {
            const int i = p * 16 + s;
            const float v = a[which[i]];
            const float w = v > 8.0f ? v * 2.0f : v + 1.0f;
            expect_lane[i] = w;
            expect[p] += w;
        }
    }

    int mismatches = 0;
    for (int i = 0; i < 64; i++) {
        mismatches += per_lane[i] != expect_lane[i];
    }
    for (int p = 0; p < 4; p++) {
        if (p != 0) {
            std::cout << ' ';
        }
        std::cout << out[p] << '/' << expect[p];
    }
    std::cout << '\n' << "lane mismatches: " << mismatches << '\n';
}
