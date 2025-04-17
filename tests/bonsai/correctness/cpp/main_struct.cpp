#include "struct.h"

#include <iostream>

int main() {
    Position i{.x = 1, .y = 2};
    Position j{.x = 10, .y = 20};
    Position r{.x = 0, .y = 0};
    add(r, i, j);
    std::cout << "x: " << r.x << ", y: " << r.y << '\n';
}