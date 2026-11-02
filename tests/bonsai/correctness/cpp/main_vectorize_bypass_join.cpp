#include "vectorize_bypass_join.h"

#include <array>
#include <iostream>

// Used in vectorize_bypass_join.bonsai.
//
// Four iterations, flags 0 1 0 1: a halving, then the else arm (early
// return past 10, else the store and the increment), twice over. First
// with inputs from 1 to 30: the small ones never pass 10 and run the whole
// loop; the large ones leave with -1 on the second or fourth iteration,
// after one store or two. Then with every input past 10: every lane leaves
// on the second iteration, so the rest of that arm is skipped by the
// bypass -- the join is entered along it, with the loop's counter and the
// gang's flags threaded and nothing for the merged value -- and the loop
// runs out with no lane on. Printed for both the gang and the scalar loop,
// then whether they agree.
static bool check(const std::array<float, 8> &a) {
    const std::array<uint32_t, 4> flags = {0u, 1u, 0u, 1u};
    std::array<float, 8> gang_trace, one_trace, gang_out, one_out;
    for (size_t i = 0; i < 8; i++) {
        gang_trace[i] = -7.0f;
        one_trace[i] = -7.0f;
        gang_out[i] = 0.0f;
        one_out[i] = 0.0f;
    }
    run(a, 4u, flags, gang_trace, gang_out);
    run_one(a, 4u, flags, one_trace, one_out);
    bool same = true;
    for (size_t i = 0; i < 8; i++) {
        same = same && gang_out[i] == one_out[i] &&
               gang_trace[i] == one_trace[i];
        std::cout << gang_out[i] << '/' << gang_trace[i] << ' ';
    }
    std::cout << '\n';
    for (size_t i = 0; i < 8; i++) {
        std::cout << one_out[i] << '/' << one_trace[i] << ' ';
    }
    std::cout << '\n';
    return same;
}

int main() {
    const bool mixed = check({1.0f, 2.0f, 4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 30.0f});
    const bool all_leave =
        check({12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 24.0f, 28.0f, 30.0f});
    std::cout << (mixed && all_leave ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
