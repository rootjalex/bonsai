#include "vectorize_stride.h"

#include <iostream>

// Used in vectorize_stride.bonsai
int main() {
    std::array<float, 16> a = {0, 1, 2,  3,  4,  5,  6,  7,
                               8, 9, 10, 11, 12, 13, 14, 15};
    // Filled with a sentinel: the odd lanes must still hold it afterwards.
    std::array<float, 16> out;
    out.fill(-1);
    double_evens(a, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
