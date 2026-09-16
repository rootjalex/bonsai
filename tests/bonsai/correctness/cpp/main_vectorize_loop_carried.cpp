#include "vectorize_loop_carried.h"

#include <iostream>

// Used in vectorize_loop_carried.bonsai.
//
// `steps` is a small tree written as an array: a node's entry is the index of
// its right child, or -1 for a leaf, and its left child is the next entry.
// Every path from node 0 ends at a leaf within a few steps. The eight lanes
// take different paths, and each lane's answer must be the one the scalar
// descent gives for it alone.
int main() {
    const std::array<int32_t, 16> steps = {3, 7, -1, 5, -1, 6, -1, 9, -1,
                                           -1, -1, -1, -1, -1, -1, -1};
    const std::array<float, 8> us = {0.1f, 0.9f, 0.3f, 0.6f,
                                     0.45f, 0.75f, 0.2f, 0.55f};
    std::array<float, 8> out{};
    descend_all(us, steps, out);
    bool same = true;
    for (size_t i = 0; i < us.size(); i++) {
        const float one = descend_one(us[i], steps);
        same = same && one == out[i];
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
