#include "vectorize_divergent.h"

#include <iostream>

// Used in vectorize_divergent.bonsai
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
    // Deliberately mixed signs and magnitudes, so no single mask value is
    // right for the whole gang and an inverted mask cannot pass.
    std::array<float, 8> a = {-8, -3, -1, 0, 1, 2, -6, 4};

    std::array<float, 8> out{};
    clamp_negative(a, out);
    show(out);

    // Pre-filled with a sentinel: the lanes that fail the test must keep it.
    std::array<float, 8> kept;
    kept.fill(-99);
    keep_positive(a, kept);
    show(kept);

    std::array<float, 8> classified{};
    classify(a, classified);
    show(classified);
}
