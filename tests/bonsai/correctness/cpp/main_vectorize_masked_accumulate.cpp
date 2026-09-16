#include "vectorize_masked_accumulate.h"

#include <array>
#include <iostream>

// Used in vectorize_masked_accumulate.bonsai.
//
// Lanes 0 and 2 are negative and so left out; the others add (a, 2a, 3a).
// With the positive inputs summing to 15, both slots must read (15, 30, 45),
// and the same as the loop that adds one lane at a time. A lowering that
// gated the components by the first three lanes' mask bits would drop x and
// z and keep only y.
int main() {
    const std::array<float, 8> a = {-1.0f, 1.0f, -2.0f, 2.0f,
                                    3.0f,  4.0f, 5.0f,  -3.0f};
    std::array<float3, 2> gang = {float3{0.0f, 0.0f, 0.0f},
                                  float3{0.0f, 0.0f, 0.0f}};
    std::array<float3, 2> one = gang;
    sum_some(a, gang);
    sum_some_one(a, one);
    bool same = true;
    for (size_t s = 0; s < 2; s++) {
        for (int c = 0; c < 3; c++) {
            same = same && gang[s][c] == one[s][c];
            std::cout << gang[s][c] << (c == 2 ? (s == 1 ? "\n" : " | ") : " ");
        }
    }
    std::cout << (same ? "same as scalar" : "DIFFERS from scalar") << '\n';
}
