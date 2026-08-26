#include "vectorize_shapes.h"

#include <iostream>

// Used in vectorize_shapes.bonsai
namespace {

template <typename T, size_t N>
void show(const std::array<T, N> &v) {
    for (size_t i = 0; i < N; i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << v[i];
    }
    std::cout << '\n';
}

} // namespace

int main() {
    std::array<int32_t, 4> small = {1, -2, 3, -4};
    std::array<int32_t, 4> small_out{};
    triple(small, small_out);
    show(small_out);

    std::array<float, 16> wide;
    for (size_t i = 0; i < wide.size(); i++) {
        wide[i] = float(i);
    }
    std::array<float, 16> wide_out{};
    add_one(wide, wide_out);
    show(wide_out);

    std::array<float, 8> a = {-4, -1, 0, 1, 2, 20, 5, -2};

    // Every byte must be written, not just the first: a bit-packed store
    // would leave most of these untouched.
    std::array<bool, 8> flags;
    flags.fill(true);
    positive(a, flags);
    show(flags);

    // Only the lanes with a positive value are written; the rest keep false.
    std::array<bool, 8> big;
    big.fill(false);
    flag_big(a, big);
    show(big);

    std::array<int32_t, 8> ints = {-4, -1, 0, 1, 2, 20, 5, -2};
    std::array<int32_t, 8> ints_out{};
    clamp_int(ints, ints_out);
    show(ints_out);
}
