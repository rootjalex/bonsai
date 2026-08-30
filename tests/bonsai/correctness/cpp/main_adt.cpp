#include "adt.h"

#include <cstdio>

// Used in adt.bonsai.
//
// Both directions across the boundary. Filling a Shape in as C++ and asking
// bonsai for its area only agrees if C++'s union and bonsai's lowering put the
// payload in the same place; asking bonsai to build one and reading its tag
// and fields back only agrees if they also number the variants the same way.
int main() {
    // Radius 2, so 4*pi*4. The centre is nonzero so that reading the radius
    // out of the wrong field would not quietly give the same answer.
    Shape sphere;
    sphere.tag = 0;
    sphere.payload.Sph = Sph{{1.0f, 2.0f, 3.0f}, 2.0f};
    printf("%f\n", area(sphere));

    // A right triangle with legs 3 and 4: area 6. Its p1 sits where a sphere
    // keeps padding, so a payload that lost those bytes shows up here.
    Shape triangle;
    triangle.tag = 1;
    triangle.payload.Tri =
        Tri{{0.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}, {0.0f, 4.0f, 0.0f}};
    printf("%f\n", area(triangle));

    Shape built;
    unit_sphere(built);
    printf("%d %f\n", (int)built.tag, built.payload.Sph.radius);

    Shape built_triangle;
    right_triangle(built_triangle);
    printf("%d %f %f\n", (int)built_triangle.tag,
           built_triangle.payload.Tri.p1.x, built_triangle.payload.Tri.p2.y);

    // And the round trip: what bonsai built, handed back to bonsai.
    printf("%f\n", area(built_triangle));
}
