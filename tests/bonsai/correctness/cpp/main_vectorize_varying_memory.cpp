#include "vectorize_varying_memory.h"

#include <iostream>

// Used in vectorize_varying_memory.bonsai
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
    // Both signs and zero, so that the branch in tally_some splits the gang.
    std::array<float, 8> a = {-4, -1, 0, 1, 2, 20, 5, -2};

    std::array<float, 8> out{};
    tally(a, out);
    show(out);

    tally_some(a, out);
    show(out);

    stepping(a, out);
    show(out);
}
