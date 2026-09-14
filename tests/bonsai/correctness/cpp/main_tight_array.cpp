#include "tight_array.h"

#include <array>
#include <iostream>
#include <vector>

// Used in tight_array.bonsai
//
// The header declares the positions parameter as the twelve-byte element the
// layout stores -- `std::array<float, 3> *` -- not as the sixteen-byte float3
// the program computes with. The calls below hand it exactly that, so a header
// that said float3 would not compile.

namespace {

void show(const float3 &v) {
    std::cout << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
}

} // namespace

int main() {
    static_assert(sizeof(std::array<float, 3>) == 12);
    // Four vertices, twelve bytes apart. The last is the one a
    // sixteen-byte stride would read past the end for.
    std::vector<std::array<float, 3>> positions = {
        {0.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}, {0.0f, 6.0f, 9.0f},
        {-1.0f, -2.0f, -3.0f}};
    std::vector<float> weights = {1.0f, 2.0f, 0.5f, 4.0f};

    // (0 + 3 + 0, 0 + 0 + 6, 0 + 0 + 9) / 3 = (1, 2, 3). Each exported
    // function takes the externs it reads and no others.
    show(centroid(0, 1, 2, positions.data()));
    // y of the third vertex times its weight: 6 * 0.5.
    std::cout << weighted(2, positions.data(), weights.data()) << '\n';
    // The last vertex, whole.
    show(vertex(3, positions.data()));
}
