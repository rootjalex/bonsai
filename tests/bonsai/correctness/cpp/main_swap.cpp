#include "swap.h"

#include <iostream>

// Used in swap.bonsai
int main() {
    int32_t i = 1;
    int32_t j = 2;
    std::cout << "i: " << i << ", j: " << j << std::endl;
    swap(i, j);
    std::cout << "i: " << i << ", j: " << j << std::endl;
}
