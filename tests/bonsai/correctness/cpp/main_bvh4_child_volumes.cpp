// Used in bvh4_child_volumes.bonsai.
//
// A scene of walls facing the x axis, and a BVH over them whose nodes hold
// their children's boxes (Embree's AABBNode, four wide), with a reference
// whose low four bits say whether it names a node or a leaf (Embree's
// NodeRef). The tree is built so that a ray along x crosses every child's
// box of every node it enters, in order: the nearest-child-first sort and
// the pruning against the best hit so far are both exercised on every ray,
// and the answer -- the nearest wall in the ray's direction down its lane --
// is known from how the scene was laid out.
#include "bvh4_child_volumes.h"

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
// `count` triangles starting at `first` -- Embree's encoding, with an index
// where Embree has a pointer. The root is row 0, which is where the
// traversal starts.
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

    // A node's row: its four children and their boxes, the boxes stored by
    // component -- `lo : vector[vec3f, 4]` is the four children's lower
    // corners, held as one array per component, as Embree's AABBNode holds
    // them.
    static void fill(_tree_layout1 &n, const uint32_t children[4],
                     const Box boxes[4]) {
        for (int i = 0; i < 4; i++) {
            n.children[i] = children[i];
            n.lo.x[i] = boxes[i].lo[0];
            n.lo.y[i] = boxes[i].lo[1];
            n.lo.z[i] = boxes[i].lo[2];
            n.hi.x[i] = boxes[i].hi[0];
            n.hi.y[i] = boxes[i].hi[1];
            n.hi.z[i] = boxes[i].hi[2];
        }
    }

    // An interior node over four children, each already built, with its box.
    uint32_t interior(const uint32_t children[4], const Box boxes[4],
                      Box &box) {
        const uint32_t row = uint32_t(rows.size());
        rows.emplace_back();
        fill(rows.back(), children, boxes);
        box = boxes[0];
        for (int i = 1; i < 4; i++) {
            box = merge(box, boxes[i]);
        }
        return row << 4;
    }
};

// Root: four children by pairs of wall positions (x-slabs, so a ray along x
// crosses them all). Each child: four leaves, one per (wall, pair of lanes),
// of two triangles each.
_tree_layout0 build_tree(Builder &b) {
    // The root's row has to be row 0, and rows are appended as nodes are
    // made, so the root is made first and its children filled in after.
    b.rows.emplace_back();
    uint32_t root_children[4];
    Box root_boxes[4];
    for (int quarter = 0; quarter < 4; quarter++) {
        uint32_t leaves[4];
        Box leaf_boxes[4];
        int l = 0;
        for (int k = 2 * quarter; k < 2 * quarter + 2; k++) {
            for (int lanes = 0; lanes < 2; lanes++) {
                std::vector<Triangle> tris = {wall(k, 2 * lanes),
                                              wall(k, 2 * lanes + 1)};
                leaves[l] = b.leaf(tris, leaf_boxes[l]);
                l++;
            }
        }
        root_children[quarter] =
            b.interior(leaves, leaf_boxes, root_boxes[quarter]);
    }
    Builder::fill(b.rows[0], root_children, root_boxes);

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
