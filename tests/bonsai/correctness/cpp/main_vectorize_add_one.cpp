#include "vectorize_add_one.h"

#include <iostream>

// Used in vectorize_add_one.bonsai
int main() {
    std::array<float, 8> a = {0, 1, 2, 3, 4, 5, 6, 7};
    std::array<float, 8> out{};
    add_one(a, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
