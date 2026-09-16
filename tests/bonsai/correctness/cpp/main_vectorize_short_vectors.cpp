#include "vectorize_short_vectors.h"

#include <iostream>

// Used in vectorize_short_vectors.bonsai
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
    // Both signs and zero, so that the branch in flip_negative and the clamp
    // in clamp_spectrum split the gang.
    std::array<float, 8> a = {-4, -1, 0, 1, 2, 20, 5, -2};
    std::array<int32_t, 8> axis = {0, 1, 2, 2, 1, 0, 1, 2};

    std::array<float, 8> out{};
    march(a, out);
    show(out);

    pick_axis(a, axis, out);
    show(out);

    flip_negative(a, out);
    show(out);

    clamp_spectrum(a, out);
    show(out);
}
