#include "vectorize_locals.h"

#include <iostream>

// Used in vectorize_locals.bonsai
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
    // Spans both sides of every test's condition, including the boundary
    // values, so no single-mask answer can pass.
    std::array<float, 8> a = {-4, -1, 0, 1, 2, 20, 5, -2};

    std::array<float, 8> out{};
    accumulate(a, out);
    show(out);

    absolute(a, out);
    show(out);

    pick(a, out);
    show(out);

    staircase(a, out);
    show(out);
}
