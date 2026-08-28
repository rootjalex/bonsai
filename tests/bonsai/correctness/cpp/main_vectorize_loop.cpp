#include "vectorize_loop.h"

#include <iostream>

// Used in vectorize_loop.bonsai
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
    // Trip counts no two of which agree, including zero -- a lane that never
    // goes round at all -- and a longest one that the whole gang has to wait
    // for.
    std::array<int32_t, 8> n = {0, 1, 2, 3, 7, 4, 1, 5};

    std::array<int32_t, 8> out{};
    triangles(n, out);
    // n(n+1)/2 for each lane.
    show(out);

    // Lanes that do not enter the loop keep what was in the array: the mask
    // the call is made under has to survive every iteration of the loop
    // inside it, not just the first.
    std::array<int32_t, 8> guarded = {-1, -1, -1, -1, -1, -1, -1, -1};
    triangles_if_positive(n, guarded);
    show(guarded);

    // Same sums, except that a lane whose running total passes 20 leaves the
    // loop the other way and reports -1. Only the lane with the longest trip
    // count gets there, so the two exits are both taken and by different
    // lanes -- which is what tells the per-destination masks apart.
    std::array<int32_t, 8> capped{};
    capped_triangles(n, capped);
    show(capped);
}
