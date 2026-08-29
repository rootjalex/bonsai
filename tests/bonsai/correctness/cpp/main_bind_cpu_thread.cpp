#include "bind_cpu_thread.h"

#include <iostream>

// Used in bind_cpu_thread.bonsai
//
// Checks every element rather than printing them, so that a loop which ran
// only part of its range -- the way a bad launch count would -- fails loudly
// instead of producing a long line someone has to read.
int main() {
    std::array<int32_t, 64> a{};
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = static_cast<int32_t>(i);
    }

    std::array<int32_t, 64> doubled{};
    scale(a, doubled);

    std::array<int32_t, 64> shifted{};
    offset(1000, a, shifted);

    size_t wrong = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (doubled[i] != static_cast<int32_t>(i) * 2) {
            wrong++;
        }
        if (shifted[i] != static_cast<int32_t>(i) + 1000) {
            wrong++;
        }
    }

    std::cout << "wrong: " << wrong << '\n';
    std::cout << "spot: " << doubled[0] << ' ' << doubled[63] << ' '
              << shifted[0] << ' ' << shifted[63] << '\n';
}
