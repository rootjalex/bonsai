#include "vectorize_vector_store.h"

#include <array>
#include <iostream>

// Used in vectorize_vector_store.bonsai.
//
// Element i must read (a, 2a, 3a) for its own a, the same as the loop that
// writes one element at a time, and the gang's gather over the result must
// give 6a per lane. The whole-struct store put lanes 0..7's x values in
// elements 0 and 1, their y values in 2 and 3, and so on, and left elements
// 6 and 7 as they were.
int main() {
    const std::array<float, 8> a = {1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float3, 8> gang;
    std::array<float3, 8> one;
    for (size_t i = 0; i < 8; i++) {
        gang[i] = float3{-1.0f, -1.0f, -1.0f};
        one[i] = gang[i];
    }
    scatter(a, gang);
    scatter_one(a, one);
    bool same = true;
    for (size_t i = 0; i < 8; i++) {
        for (int c = 0; c < 3; c++) {
            same = same && gang[i][c] == one[i][c];
            std::cout << gang[i][c] << (c == 2 ? "\n" : " ");
        }
    }
    std::cout << (same ? "same as scalar" : "DIFFERS from scalar") << '\n';
    std::array<float, 8> sums;
    gather(gang, sums);
    for (size_t i = 0; i < 8; i++) {
        std::cout << sums[i] << (i == 7 ? "\n" : " ");
    }
}
