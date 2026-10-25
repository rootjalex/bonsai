// Used in bvh2_unrolled.bonsai.
//
// The scene and the checks of main_bvh4_child_volumes.cpp -- walls facing
// the x axis in lanes along z, a ray down a lane finding the nearest wall
// in its direction -- over a binary tree whose nodes hold their two
// children's boxes, with a reference whose low four bits say whether it
// names a node or a leaf. The tree splits the walls in x first and the
// lanes in z after, so that a ray along x crosses both children's boxes of
// every node it enters, in order: the two-child visit the schedule unrolls
// and sorts is exercised on every ray, and the answer is known from how the
// scene was laid out.
#include "bvh2_unrolled.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// Walls at x = 10k, for eight values of k, in four lanes along z. A wall is
// a triangle in the plane x = const spanning y in [-1, 1] and z in [z-1,
// z+1], so a ray down the lane's centre (y = 0) hits it, and a ray between
// lanes misses every wall.
constexpr int kWalls = 8;
constexpr int kLanes = 4;
constexpr float kWallSpacing = 10.0f;
constexpr float kLaneSpacing = 5.0f;

Triangle wall(int k, int lane) {
    const float x = kWallSpacing * float(k);
    const float z = kLaneSpacing * float(lane);
    Triangle t;
    t.p0 = float3{x, -1.0f, z - 1.0f};
    t.p1 = float3{x, 1.0f, z - 1.0f};
    t.p2 = float3{x, 0.0f, z + 1.0f};
    return t;
}

struct Box {
    float3 lo, hi;
};

Box bounds_of(const Triangle &t) {
    Box b;
    for (int c = 0; c < 3; c++) {
        b.lo[c] = std::min({t.p0[c], t.p1[c], t.p2[c]});
        b.hi[c] = std::max({t.p0[c], t.p1[c], t.p2[c]});
    }
    return b;
}

Box merge(const Box &a, const Box &b) {
    Box out;
    for (int c = 0; c < 3; c++) {
        out.lo[c] = std::min(a.lo[c], b.lo[c]);
        out.hi[c] = std::max(a.hi[c], b.hi[c]);
    }
    return out;
}

// The tree in the layout the schedule declared. A reference is `row << 4`
// for an interior node and `(first << 4) | (8 + count)` for a leaf of
// `count` triangles starting at `first`. The root is row 0, where the
// traversal starts: a node's row is taken before its children are built,
// so the first node made, the root, takes the first row.
struct Builder {
    std::vector<Triangle> prims;     // in leaf order
    std::vector<_tree_layout1> rows; // the interior nodes

    uint32_t leaf(const std::vector<Triangle> &tris, Box &box) {
        const uint32_t first = uint32_t(prims.size());
        box = bounds_of(tris.front());
        for (const Triangle &t : tris) {
            prims.push_back(t);
            box = merge(box, bounds_of(t));
        }
        return (first << 4) | (8u + uint32_t(tris.size()));
    }

    // A node's row: its two children and their boxes, the boxes stored by
    // component -- `lo : vector[vec3f, 2]` is the two children's lower
    // corners, held as one array per component.
    static void fill(_tree_layout1 &n, const uint32_t children[2],
                     const Box boxes[2]) {
        for (int i = 0; i < 2; i++) {
            n.children[i] = children[i];
            n.lo.x[i] = boxes[i].lo[0];
            n.lo.y[i] = boxes[i].lo[1];
            n.lo.z[i] = boxes[i].lo[2];
            n.hi.x[i] = boxes[i].hi[0];
            n.hi.y[i] = boxes[i].hi[1];
            n.hi.z[i] = boxes[i].hi[2];
        }
    }

    // The subtree over walls [k0, k1) in lanes [l0, l1): a leaf of one
    // wall's two triangles in two lanes, else a node splitting the walls
    // where there are two or more and the lanes otherwise.
    uint32_t build(int k0, int k1, int l0, int l1, Box &box) {
        if (k1 - k0 == 1 && l1 - l0 == 2) {
            return leaf({wall(k0, l0), wall(k0, l0 + 1)}, box);
        }
        const uint32_t row = uint32_t(rows.size());
        rows.emplace_back();
        uint32_t children[2];
        Box boxes[2];
        if (k1 - k0 > 1) {
            const int mid = (k0 + k1) / 2;
            children[0] = build(k0, mid, l0, l1, boxes[0]);
            children[1] = build(mid, k1, l0, l1, boxes[1]);
        } else {
            const int mid = (l0 + l1) / 2;
            children[0] = build(k0, k1, l0, mid, boxes[0]);
            children[1] = build(k0, k1, mid, l1, boxes[1]);
        }
        fill(rows[row], children, boxes);
        box = merge(boxes[0], boxes[1]);
        return row << 4;
    }
};

_tree_layout0 build_tree(Builder &b) {
    Box box;
    const uint32_t root = b.build(0, kWalls, 0, kLanes, box);
    if (root != 0) {
        std::cerr << "the root is not row 0\n";
        std::exit(1);
    }

    // The layout's arrays reach the program as buffer descriptors
    // (runtime/bonsai_buffer.h), which have to outlive the tree.
    static bonsai_buffer prims_buffer, rows_buffer;
    prims_buffer =
        bonsai_buffer_wrap(b.prims.data(), b.prims.size() * sizeof(Triangle));
    rows_buffer = bonsai_buffer_wrap(b.rows.data(),
                                     b.rows.size() * sizeof(_tree_layout1));
    _tree_layout0 tree;
    tree.pCount = uint32_t(b.prims.size());
    tree.prims = &prims_buffer;
    tree.nCount = uint32_t(b.rows.size());
    tree.group0_row = &rows_buffer;
    return tree;
}

void report(const char *what, const Ray &r, const _tree_layout0 &tree,
            const Triangle *expected) {
    _option0 hit;
    trace(hit, r, tree);
    const bool blocked = occluded(r, tree);
    std::cout << what << ": ";
    if (!hit.set) {
        std::cout << "miss";
    } else {
        std::cout << "wall at x=" << hit.value.p0[0] << " z=" << hit.value.p0[2] + 1.0f;
    }
    std::cout << (blocked ? ", occluded" : ", clear");
    const bool right =
        expected == nullptr
            ? (!hit.set && !blocked)
            : (hit.set && blocked && hit.value.p0[0] == expected->p0[0] &&
               hit.value.p0[2] == expected->p0[2]);
    std::cout << (right ? " -- ok" : " -- WRONG") << '\n';
    if (!right) {
        std::exit(1);
    }
}

} // namespace

int main() {
    Builder b;
    _tree_layout0 tree = build_tree(b);
    std::cout << b.rows.size() << " nodes over " << b.prims.size() << " walls\n";

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const float3 backward = float3{-1.0f, 0.0f, 0.0f};
    for (int lane = 0; lane < kLanes; lane++) {
        const float z = kLaneSpacing * float(lane);
        // Down the lane towards +x: the nearest wall is the first.
        const Triangle first = wall(0, lane);
        report("+x", Ray{float3{-1.0f, 0.0f, z}, forward}, tree, &first);
        // And back towards -x from beyond the last: the farthest wall.
        const Triangle last = wall(kWalls - 1, lane);
        report("-x", Ray{float3{100.0f, 0.0f, z}, backward}, tree, &last);
        // From between two walls, towards +x: the one ahead, not the one
        // behind.
        const Triangle ahead = wall(4, lane);
        report("+x from 35", Ray{float3{35.0f, 0.0f, z}, forward}, tree, &ahead);
        // A ray that ends before the first wall.
        report("+x short", Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f}, tree,
               nullptr);
    }
    // Between lanes: through every box the walls' boxes span, hitting none.
    report("+x between lanes",
           Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward}, tree, nullptr);
    // Climbing across lanes: hits the first wall where it crosses lane 0.
    const Triangle first0 = wall(0, 0);
    report("diagonal", Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}},
           tree, &first0);
    // Along z at a wall's x, past the walls' y: nothing.
    report("+z beside", Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}},
           tree, nullptr);
    return 0;
}
