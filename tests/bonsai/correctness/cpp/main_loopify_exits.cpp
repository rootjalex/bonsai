#include "loopify_exits.h"

#include <iostream>

// Used in loopify_exits.bonsai
int main() {
    // The eight walks of tests/bonsai/correctness/gpu/loopify-exits: thread
    // t walks t + 1 steps at most from a[t], and each ends by its own exit
    // -- out of steps (0 and 2), past half the limit (3, 4, 6, 7), past the
    // limit (1 and 5).
    std::array<float, 8> a = {1.0f, 10.0f, 0.5f, 2.0f, 1.0f, 3.0f, 0.25f, 1.0f};
    std::array<float, 8> out{};
    walks(a, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
