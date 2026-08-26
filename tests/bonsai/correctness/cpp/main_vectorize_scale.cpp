#include "vectorize_scale.h"

#include <iostream>

// Used in vectorize_scale.bonsai
int main() {
    std::array<float, 8> a = {0, 1, 2, 3, 4, 5, 6, 7};
    std::array<float, 8> out{};
    scale_all(a, 2.0f, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
