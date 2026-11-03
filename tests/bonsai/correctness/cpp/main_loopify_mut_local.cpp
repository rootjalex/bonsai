#include "loopify_mut_local.h"

#include <iostream>

// Used in loopify_mut_local.bonsai.
//
// Four halvings of (1, 2, 3), summed: 3 + 1.5 + 0.75 + 0.375 = 5.625, from
// the loop and from the recursion alike.
int main() {
    const float3 v = {1.0f, 2.0f, 3.0f};
    const float looped = run(4, v);
    const float recursed = run_rec(4, v);
    std::cout << looped << ' ' << recursed << '\n';
    std::cout << (looped == recursed && looped == 5.625f ? "same as recursion"
                                                          : "DIFFERS")
              << '\n';
}
