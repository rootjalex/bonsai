#include "accumulate_vector_scalar.h"

#include <array>
#include <iostream>

// Used in accumulate_vector_scalar.bonsai.
//
// With s = 2 every component must read v * 2 + 2 - 1 = 2v + 1. The x-only
// lowering gave 2x + 1 for x and left y and z as they came in.
int main() {
    const std::array<float3, 4> a = {float3{1.0f, 2.0f, 3.0f},
                                     float3{4.0f, 5.0f, 6.0f},
                                     float3{-1.0f, 0.0f, 1.0f},
                                     float3{0.5f, 0.25f, 0.125f}};
    std::array<float3, 4> out;
    for (size_t i = 0; i < 4; i++) {
        out[i] = float3{0.0f, 0.0f, 0.0f};
    }
    run(a, 2.0f, out);
    bool right = true;
    for (size_t i = 0; i < 4; i++) {
        for (int c = 0; c < 3; c++) {
            const float want = a[i][c] * 2.0f + 2.0f - 1.0f;
            right = right && out[i][c] == want;
            std::cout << out[i][c] << (c == 2 ? "\n" : " ");
        }
    }
    std::cout << (right ? "every component scaled" : "WRONG components") << '\n';
}
