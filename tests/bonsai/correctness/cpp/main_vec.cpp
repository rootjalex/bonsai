#include "vec.h"

#include <iostream>

// Used in struct.bonsai
int main() {
    vec3_float v;
    v[0] = -1;
    v[1] = -2;
    v[2] = 5;

    std::cout << v[1] << " and " << v[2] << std::endl;

    print_vec(v);

    Sphere s;
    s.c[0] = 4;
    s.c[1] = -8;
    s.c[2] = 16;
    s.r = 100;

    print_sphere(s);
}
