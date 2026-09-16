#include "vectorize_loop_early_return.h"

#include <iostream>

// Used in vectorize_loop_early_return.bonsai.
//
// Starting values and trip counts chosen so that some lanes run out of
// iterations, some leave early on different iterations, and one never enters
// the loop; each must match the scalar walk.
int main() {
    const std::array<float, 8> xs = {1.0f, 12.0f, 0.5f, 25.0f,
                                     3.0f, 9.0f, 30.0f, 2.0f};
    const std::array<int32_t, 8> ns = {3, 5, 8, 2, 6, 4, 1, 0};
    std::array<float, 8> out{};
    walk_all(xs, ns, out);
    bool same = true;
    for (size_t i = 0; i < xs.size(); i++) {
        same = same && walk_one(xs[i], ns[i]) == out[i];
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
