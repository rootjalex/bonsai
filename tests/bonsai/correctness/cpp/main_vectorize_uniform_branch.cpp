#include "vectorize_uniform_branch.h"

#include <iostream>

// Used in vectorize_uniform_branch.bonsai.
//
// Both settings of the uniform flag, over lanes on either side of the
// divergent test; every lane must match the scalar function.
int main() {
    const std::array<float, 8> a = {-1.0f, 2.0f, -0.5f, 3.0f,
                                    0.0f, -4.0f, 1.5f, -2.5f};
    for (const bool two_sided : {false, true}) {
        std::array<float, 8> out{};
        shade_all(two_sided, a, out);
        bool same = true;
        for (size_t i = 0; i < a.size(); i++) {
            same = same && shade_one(two_sided, a[i]) == out[i];
            if (i != 0) {
                std::cout << ' ';
            }
            std::cout << out[i];
        }
        std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
                  << '\n';
    }
}
