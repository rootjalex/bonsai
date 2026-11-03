#include "vectorize_latch_through_uniform_join.h"

#include <array>
#include <iostream>

// Used in vectorize_latch_through_uniform_join.bonsai.
//
// A lane's walk: y = (2x + 1) / 4 each round; a lane with y past 20 is done
// (49, 60 and 99.5 in the first round), a lane with y past 0.5 takes the
// guarded arm -- with the flag on and z past 13, the inner arm's recursion
// or early return -- and recurses, the others recurse with half a point
// more. The gang must agree with the walk one lane at a time on every lane,
// for every round limit and both settings of the flag.
static const std::array<float, 8> inputs = {1.0f, 5.0f, 11.0f, 30.0f, 49.0f, 60.0f, 99.5f, -3.0f};
static const std::array<float, 8> table = {0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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

int main() {
    bool ok = true;
    for (int max_wave : {1, 2, 3, 5}) {
        for (bool flag : {false, true}) {
            std::array<float, 8> gang{}, one{};
            std::cout << "max_wave " << max_wave << ", flag " << flag << "\n";
            run(inputs, max_wave, flag, table, gang);
            run_one(inputs, max_wave, flag, table, one);
            const bool agree = same(gang, one);
            std::cout << (agree ? "  same as scalar\n" : "  DIFFERENT\n");
            ok = ok && agree;
        }
    }
    return ok ? 0 : 1;
}
