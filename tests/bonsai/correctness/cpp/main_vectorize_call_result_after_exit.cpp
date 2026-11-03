#include "vectorize_call_result_after_exit.h"

#include <array>
#include <iostream>

// Used in vectorize_call_result_after_exit.bonsai.
//
// With the flag on, a lane counts up by one until it passes 10, and from
// then on each iteration doubles-and-adds-one, returning at once if that
// passes 50, and steps otherwise: 1 reaches 11, then 23, 24, 49, 50, 101 ->
// 101; 30 gives 61 -> 61. With the flag off every lane counts to 100. The
// gang must agree with the loop one lane at a time on every lane, for both
// flags.
static bool check(bool flag) {
    const std::array<float, 8> a = {1.0f, 5.0f, 11.0f, 30.0f, 49.0f, 60.0f, 99.5f, -3.0f};
    std::array<float, 8> gang, one;
    for (size_t i = 0; i < 8; i++) {
        gang[i] = 0.0f;
        one[i] = 0.0f;
    }
    run(a, flag, gang);
    run_one(a, flag, one);
    bool same = true;
    for (size_t i = 0; i < 8; i++) {
        same = same && gang[i] == one[i];
        std::cout << gang[i] << (i == 7 ? "\n" : " ");
    }
    for (size_t i = 0; i < 8; i++) {
        std::cout << one[i] << (i == 7 ? "\n" : " ");
    }
    return same;
}

int main() {
    const bool on = check(true);
    const bool off = check(false);
    std::cout << (on && off ? "same as scalar" : "DIFFERS from scalar") << '\n';
}
