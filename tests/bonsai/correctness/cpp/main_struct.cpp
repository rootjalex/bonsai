#include "struct.h"

#include <iostream>

// Used in struct.bonsai
int main() {
    Position i;
    i.x = 1;
    i.y = 2;
    Position j;
    j.x = 10;
    j.y = 20;
    Position r;
    r.x = 0;
    r.y = 0;
    add(r, i, j);
    std::cout << "x: " << r.x << ", y: " << r.y << '\n';
}
