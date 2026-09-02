#include "adt-tagged-index.h"

#include <cstdio>

// Used in adt-tagged-index.bonsai.
//
// main_adt.cpp with the other layout, and the differences are the point.
//
// A `Shape` is a `uint64_t` here rather than a struct, so there is nothing to
// fill in: a handle is made by calling the constructor, which puts the fields
// in the pool and hands back the tag and the index naming them. The pools are
// this file's, sized here, and so is the counter that says how much of each is
// used -- that is what building a variant without an allocator means, and it
// is why every function that can build one takes both.
//
// Reading a handle back apart is the one thing that needs the layout written
// down, and it is written down in Lower/ADTLayout.h: the tag is the top byte
// and the index is the rest.
namespace {

constexpr uint64_t kTagShift = 56;

// Room for more than the four built below, so that a pool running exactly to
// its end is not what makes this pass.
Sph spheres[8];
Tri triangles[8];
uint64_t sphere_fill = 0;
uint64_t triangle_fill = 0;

uint64_t tag_of(uint64_t handle) { return handle >> kTagShift; }
uint64_t index_of(uint64_t handle) {
    return handle & ((uint64_t{1} << kTagShift) - 1);
}

} // namespace

int main() {
    // Radius 2, so 4*pi*4. The centre is nonzero so that reading the radius
    // out of the wrong field would not quietly give the same answer.
    const uint64_t sphere =
        Shape_Sph(float3{1.0f, 2.0f, 3.0f}, 2.0f, spheres, &sphere_fill);
    printf("%f\n", area(sphere, spheres, triangles));

    // A right triangle with legs 3 and 4: area 6.
    const uint64_t triangle =
        Shape_Tri(float3{0.0f, 0.0f, 0.0f}, float3{3.0f, 0.0f, 0.0f},
                  float3{0.0f, 4.0f, 0.0f}, triangles, &triangle_fill);
    printf("%f\n", area(triangle, spheres, triangles));

    // The tag numbers the variants in declaration order, as it does under the
    // other layout, so a sphere is 0 and a triangle is 1.
    printf("%d %d\n", (int)tag_of(sphere), (int)tag_of(triangle));

    // Two variants, two pools: the first of each is at index 0. A single
    // shared pool would put the triangle at 1 and this is what would say so.
    printf("%d %d\n", (int)index_of(sphere), (int)index_of(triangle));

    // Built in bonsai, read in C++ -- through the pool, which is the only
    // place the fields are.
    const uint64_t built = unit_sphere(spheres, &sphere_fill);
    printf("%d %f\n", (int)tag_of(built), spheres[index_of(built)].radius);

    const uint64_t built_triangle = right_triangle(triangles, &triangle_fill);
    printf("%d %f %f\n", (int)tag_of(built_triangle),
           triangles[index_of(built_triangle)].p1.x,
           triangles[index_of(built_triangle)].p2.y);

    // And the round trip: what bonsai built, handed back to bonsai.
    printf("%f\n", area(built_triangle, spheres, triangles));

    // Four built, two of each, and the counters say so. This is the part a
    // driver has to get right that the other layout has no equivalent of: the
    // pool is only as full as the constructors made it.
    printf("%d %d\n", (int)sphere_fill, (int)triangle_fill);
}
