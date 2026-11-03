#include "vectorize_join_one_name_two_values.h"

#include <array>
#include <iostream>

// Used in vectorize_join_one_name_two_values.bonsai.
//
// `walk`, flag on: a lane counts up by one until it passes 10, and from then
// on each iteration doubles-and-adds-one and steps: 1 reaches 11, then 23,
// 24, 49, 50, 101, 102 -> 102; 30 gives 61, 62, 125, 126 -> 126. Flag off:
// every lane counts to 100. `walk_flat`: the first flag triples the input,
// the second doubles-and-adds-one whatever is above 10, and one is added.
// The gang must agree with the walk one lane at a time on every lane, under
// every setting of the flags.
static const std::array<float, 8> inputs = {1.0f, 5.0f, 11.0f, 30.0f, 49.0f, 60.0f, 99.5f, -3.0f};

static bool same(const std::array<float, 8> &gang, const std::array<float, 8> &one) {
    bool agree = true;
    for (size_t i = 0; i < 8; i++) {
        agree = agree && gang[i] == one[i];
        std::cout << gang[i] << (i == 7 ? "\n" : " ");
    }
    for (size_t i = 0; i < 8; i++) {
        std::cout << one[i] << (i == 7 ? "\n" : " ");
    }
    return agree;
}

static bool check_walk(bool flag) {
    std::array<float, 8> gang{}, one{};
    run(inputs, flag, gang);
    run_one(inputs, flag, one);
    std::cout << "walk, flag " << flag << "\n";
    return same(gang, one);
}

static bool check_flat(bool flag1, bool flag2) {
    std::array<float, 8> gang{}, one{};
    run_flat(inputs, flag1, flag2, gang);
    run_flat_one(inputs, flag1, flag2, one);
    std::cout << "walk_flat, flags " << flag1 << " " << flag2 << "\n";
    return same(gang, one);
}

int main() {
    bool agree = true;
    for (int flag = 0; flag < 2; flag++) {
        agree = check_walk(flag) && agree;
    }
    for (int flag1 = 0; flag1 < 2; flag1++) {
        for (int flag2 = 0; flag2 < 2; flag2++) {
            agree = check_flat(flag1, flag2) && agree;
        }
    }
    std::cout << (agree ? "same as scalar" : "DIFFERS from scalar") << '\n';
}
