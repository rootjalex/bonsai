#include "mutual_recursion_extern.h"

#include <cstdint>
#include <iostream>
#include <vector>

// Used in mutual_recursion_extern.bonsai.
//
// `weighted_sum` takes the extern as its last parameter, as every exported function
// that reaches an extern does, and passes it through even_step to odd_step.

int main() {
    std::vector<int32_t> weights = {10, 1, 20, 3, 40, 5, 60, 7};
    // even_step(6): odd_step reads weights at 5, 3, 1: 5 + 3 + 1.
    std::cout << weighted_sum(6, weights.data()) << '\n';
    // even_step(7): weights at 6, 4, 2, 0: 60 + 40 + 20 + 10.
    std::cout << weighted_sum(7, weights.data()) << '\n';
    // Nothing read.
    std::cout << weighted_sum(0, weights.data()) << '\n';
}
