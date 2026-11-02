#include "vectorize_vector_accumulate.h"

#include <array>
#include <iostream>

// Used in vectorize_vector_accumulate.bonsai.
//
// Lanes 0, 2 and 7 are negative and keep the slot's (1, 1, 1); the others
// read (1 + a, 1 + 2a, 1 + 3a) / 2 -- lane 1 (a = 1) is (1, 1.5, 2), lane 6
// (a = 5) is (3, 5.5, 8). A lowering that accumulated the struct of gang
// vectors whole, or that broadcast the scalar over the lanes alone and not
// the components, has no well-typed form to reach these; the gang's answer
// is checked against the loop that runs one lane at a time.
int main() {
    const std::array<float, 8> a = {-1.0f, 1.0f, -2.0f, 2.0f,
                                    3.0f,  4.0f, 5.0f,  -3.0f};
    std::array<float3, 8> gang;
    std::array<float3, 8> one;
    for (size_t i = 0; i < 8; i++) {
        gang[i] = float3{0.0f, 0.0f, 0.0f};
        one[i] = gang[i];
    }
    run(a, gang);
    run_one(a, one);
    bool same = true;
    for (size_t i = 0; i < 8; i++) {
        for (int c = 0; c < 3; c++) {
            same = same && gang[i][c] == one[i][c];
            std::cout << gang[i][c] << (c == 2 ? "\n" : " ");
        }
    }
    std::cout << (same ? "same as scalar" : "DIFFERS from scalar") << '\n';
}
