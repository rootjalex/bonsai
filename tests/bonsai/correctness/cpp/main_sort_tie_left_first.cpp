// Used in sort_tie_left_first.bonsai: a root over two leaves, one triangle
// each, both triangles in the plane x = 10 across the ray's path so that
// the ray meets both at t = 10 exactly; the children's boxes are set by
// hand so that their slab entries tie, or one is strictly nearer. The
// first hit keeps the first triangle found, so the triangle that comes
// back says which child the traversal entered first. The layout and the
// builder are main_unroll_let_names.cpp's.
#include "sort_tie_left_first.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct Box {
    float3 lo, hi;
};

// Two triangles in the plane x = 10, both across the line y = 0, z = 0 the
// ray runs along, told apart by their first vertex.
Triangle left_triangle() {
    Triangle t;
    t.p0 = float3{10.0f, -2.0f, -2.0f};
    t.p1 = float3{10.0f, 2.0f, -2.0f};
    t.p2 = float3{10.0f, 0.0f, 2.0f};
    return t;
}

Triangle right_triangle() {
    Triangle t;
    t.p0 = float3{10.0f, -1.0f, -1.0f};
    t.p1 = float3{10.0f, 1.0f, -1.0f};
    t.p2 = float3{10.0f, 0.0f, 1.0f};
    return t;
}

// The tree in the layout the schedule declared (see main_unroll_let_names.cpp
// for the reference encoding): the root, row 0, over two leaves of one
// triangle each -- prim 0 on the left, prim 1 on the right -- with the boxes
// given, which need not be the triangles' own: they are what the node tests.
struct Tree {
    std::vector<Triangle> prims;
    std::vector<_tree_layout1> rows;
    bonsai_buffer prims_buffer, rows_buffer;
    _tree_layout0 layout;

    Tree(const Box &left, const Box &right) {
        prims = {left_triangle(), right_triangle()};
        rows.emplace_back();
        _tree_layout1 &root = rows[0];
        const uint32_t children[2] = {(0u << 4) | 9u, (1u << 4) | 9u};
        const Box boxes[2] = {left, right};
        for (int i = 0; i < 2; i++) {
            root.children[i] = children[i];
            root.lo.x[i] = boxes[i].lo[0];
            root.lo.y[i] = boxes[i].lo[1];
            root.lo.z[i] = boxes[i].lo[2];
            root.hi.x[i] = boxes[i].hi[0];
            root.hi.y[i] = boxes[i].hi[1];
            root.hi.z[i] = boxes[i].hi[2];
        }
        prims_buffer = bonsai_buffer_wrap(prims.data(), prims.size() * sizeof(Triangle));
        rows_buffer = bonsai_buffer_wrap(rows.data(), rows.size() * sizeof(_tree_layout1));
        layout.pCount = uint32_t(prims.size());
        layout.prims = &prims_buffer;
        layout.nCount = uint32_t(rows.size());
        layout.group0_row = &rows_buffer;
    }
};

int failures = 0;

void check(const char *what, const Box &left, const Box &right, const char *expected) {
    Tree tree(left, right);
    const Ray r{float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 0.0f, 0.0f}};
    _option0 hit;
    trace(hit, r, tree.layout);
    const char *got = !hit.set ? "miss" : hit.value.p0[1] == -2.0f ? "left" : "right";
    std::cout << what << ": " << got << " (expected " << expected << ")\n";
    if (std::string_view(got) != expected) {
        failures++;
    }
}

} // namespace

int main() {
    // Both boxes hold the origin: the slab entry is zero for both, a tie,
    // and the left child is entered first.
    check("tie", Box{float3{-5.0f, -5.0f, -5.0f}, float3{15.0f, 5.0f, 5.0f}},
          Box{float3{-3.0f, -3.0f, -3.0f}, float3{12.0f, 3.0f, 3.0f}}, "left");
    // The right box begins at x = 5, the left holds the origin: the left is
    // strictly nearer and entered first.
    check("left nearer", Box{float3{-5.0f, -5.0f, -5.0f}, float3{15.0f, 5.0f, 5.0f}},
          Box{float3{5.0f, -3.0f, -3.0f}, float3{12.0f, 3.0f, 3.0f}}, "left");
    // The left box begins at x = 5, the right holds the origin: the right is
    // strictly nearer and entered first, and its triangle is the one kept.
    check("right nearer", Box{float3{5.0f, -5.0f, -5.0f}, float3{15.0f, 5.0f, 5.0f}},
          Box{float3{-3.0f, -3.0f, -3.0f}, float3{12.0f, 3.0f, 3.0f}}, "right");
    if (failures != 0) {
        std::cout << failures << " wrong\n";
        return 1;
    }
    std::cout << "sort_tie_left_first: ok\n";
    return 0;
}
