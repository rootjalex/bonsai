#include "persistent.h"

#include <iostream>

// Used in persistent.bonsai
//
// Checks every element rather than printing them, so that an iteration
// claimed twice or never -- the way a counter the threads did not share
// would hand them out -- fails loudly instead of producing a long line
// someone has to read.
int main() {
    std::array<int32_t, 1000> a{};
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = static_cast<int32_t>(i);
    }

    std::array<int32_t, 1000> doubled{};
    scale(a, doubled);

    std::array<int32_t, 1000> shifted{};
    chunked(1000, a, shifted);

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
    std::cout << "spot: " << doubled[0] << ' ' << doubled[999] << ' '
              << shifted[0] << ' ' << shifted[999] << '\n';
}
