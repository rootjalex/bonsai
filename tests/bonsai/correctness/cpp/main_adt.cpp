#include "adt.h"

#include <cstdio>
#include <cstring>

// Used in adt.bonsai.
//
// Both directions across the boundary. Filling a Shape in as C++ and asking
// bonsai for its area only agrees if C++ puts each field in the word bonsai's
// lowering reads it from; asking bonsai to build one and reading its tag and
// fields back only agrees if they also number the variants the same way.
//
// The payload is the variant's fields as 32-bit words, at the offsets C
// gives them (see Lower/WordStorage.h): a vec3f takes four words, the fourth
// its padding, as the C++ vector type does, so a Sph is its centre in words 0
// to 2 and its radius in word 4, and a Tri its three points at words 0, 4
// and 8.
namespace {

void put(Shape &shape, size_t k, float f) {
    std::memcpy(&shape.payload[k], &f, sizeof f);
}

void put_point(Shape &shape, size_t k, float x, float y, float z) {
    put(shape, k, x);
    put(shape, k + 1, y);
    put(shape, k + 2, z);
}

float word(const Shape &shape, size_t k) {
    float f;
    std::memcpy(&f, &shape.payload[k], sizeof f);
    return f;
}

} // namespace

int main() {
    // Radius 2, so 4*pi*4. The centre is nonzero so that reading the radius
    // out of the wrong word would not quietly give the same answer.
    Shape sphere{};
    sphere.tag = 0;
    put_point(sphere, 0, 1.0f, 2.0f, 3.0f);
    put(sphere, 4, 2.0f);
    printf("%f\n", area(sphere));

    // A right triangle with legs 3 and 4: area 6. Its p1 sits where a sphere
    // keeps its radius, so a field read at the wrong offset shows up here.
    Shape triangle{};
    triangle.tag = 1;
    put_point(triangle, 0, 0.0f, 0.0f, 0.0f);
    put_point(triangle, 4, 3.0f, 0.0f, 0.0f);
    put_point(triangle, 8, 0.0f, 4.0f, 0.0f);
    printf("%f\n", area(triangle));

    Shape built;
    unit_sphere(built);
    printf("%d %f\n", (int)built.tag, word(built, 4));

    Shape built_triangle;
    right_triangle(built_triangle);
    printf("%d %f %f\n", (int)built_triangle.tag, word(built_triangle, 4),
           word(built_triangle, 9));

    // And the round trip: what bonsai built, handed back to bonsai.
    printf("%f\n", area(built_triangle));
}
