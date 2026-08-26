#include "vectorize_calls.h"

#include <iostream>

// Used in vectorize_calls.bonsai
namespace {

void show(const std::array<float, 8> &v) {
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
    std::array<float, 8> a = {-4, -1, 0, 1, 2, -3, 5, -2};

    std::array<float, 8> out{};
    unconditional(a, 3, out);
    show(out);

    with_branch(a, 0, out);
    show(out);

    // Sentinel-filled: the lanes that do not call must keep it, which is what
    // the callee's mask is for.
    std::array<float, 8> stored;
    stored.fill(-99);
    conditional_store(a, stored);
    show(stored);

    both_ways(a, out);
    show(out);
}
