#include "vectorize_nested_loops.h"

#include <iostream>

// Used in vectorize_nested_loops.bonsai
namespace {

void show(const std::array<int32_t, 8> &v) {
    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << v[i];
    }
    std::cout << '\n';
}

} // namespace

int main() {
    // Every trip count from none to seven, so that lanes leave the outer loop
    // on every iteration and the inner one at every length.
    std::array<int32_t, 8> n = {0, 1, 2, 3, 4, 5, 6, 7};

    std::array<int32_t, 8> out{};
    nested_sum(n, out);
    show(out);

    find_pair(n, 6, out);
    show(out);
}
