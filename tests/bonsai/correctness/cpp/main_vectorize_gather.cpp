#include "vectorize_gather.h"

#include <iostream>

// Used in vectorize_gather.bonsai
int main() {
    std::array<float, 8> a = {0, 10, 20, 30, 40, 50, 60, 70};
    // Deliberately not a permutation: index 3 is read by two lanes and some
    // elements by none, which a shuffle-based lowering would get wrong.
    std::array<int32_t, 8> idx = {7, 3, 0, 5, 3, 1, 6, 2};
    std::array<float, 8> out{};
    reorder(a, idx, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
