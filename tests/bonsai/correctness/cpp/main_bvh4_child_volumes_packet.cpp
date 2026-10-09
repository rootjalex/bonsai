// Used in bvh4_child_volumes_packet.bonsai.
//
// main_bvh4_child_volumes.cpp's scene and tree -- walls facing the x axis in
// four lanes along z, and a four-wide BVH over them whose nodes hold their
// children's boxes, split in x so that a ray along x crosses every child's
// box of every node it enters -- and its rays, in gangs of eight. Under the
// packet schedule a gang shares one node, visits a child when any lane's ray
// meets its box, and decides the order of the children by a vote; a gang
// here holds rays towards +x and -x, rays that end before any wall and rays
// between the lanes that meet no wall, so the lanes want different children
// first and some want none. That may cost a lane a subtree visited early,
// and it must not change what any of them hits: every lane's answer has to
// match the same ray traced alone by the scalar `hit_x_one` and
// `occluded_one`.
#include "bvh4_child_volumes_packet.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

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

// The tree in the layout the schedule declared (see
// main_bvh4_child_volumes.cpp): a reference is `row << 4` for an interior
// node and `(first << 4) | (8 + count)` for a leaf; the root is row 0.
struct Builder {
    std::vector<Triangle> prims;
    std::vector<_tree_layout1> rows;

    uint32_t leaf(const std::vector<Triangle> &tris, Box &box) {
        const uint32_t first = uint32_t(prims.size());
        box = bounds_of(tris.front());
        for (const Triangle &t : tris) {
            prims.push_back(t);
            box = merge(box, bounds_of(t));
        }
        return (first << 4) | (8u + uint32_t(tris.size()));
    }

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

_tree_layout0 build_tree(Builder &b) {
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

} // namespace

int main() {
    Builder b;
    _tree_layout0 tree = build_tree(b);

    // main_bvh4_child_volumes.cpp's rays, four per lane and three more, in
    // the order it traces them; the last gang is filled out with the ray
    // between the lanes, which meets nothing.
    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const float3 backward = float3{-1.0f, 0.0f, 0.0f};
    std::vector<Ray> rays;
    for (int lane = 0; lane < kLanes; lane++) {
        const float z = kLaneSpacing * float(lane);
        rays.push_back(Ray{float3{-1.0f, 0.0f, z}, forward});
        rays.push_back(Ray{float3{100.0f, 0.0f, z}, backward});
        rays.push_back(Ray{float3{35.0f, 0.0f, z}, forward});
        rays.push_back(Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f});
    }
    rays.push_back(Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward});
    rays.push_back(Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}});
    rays.push_back(Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}});
    while (rays.size() % 8 != 0) {
        rays.push_back(Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward});
    }
    (void)kWalls;

    bool same = true;
    for (size_t gang = 0; gang < rays.size(); gang += 8) {
        std::array<Ray, 8> rs;
        std::copy(rays.begin() + gang, rays.begin() + gang + 8, rs.begin());
        std::array<float, 8> hit{};
        std::array<int32_t, 8> blocked{};
        hit_x_all(rs, hit, tree);
        occluded_all(rs, blocked, tree);
        for (size_t i = 0; i < 8; i++) {
            const float one = hit_x_one(rs[i], tree);
            const int32_t one_blocked = occluded_one(rs[i], tree);
            const bool agree = one == hit[i] && one_blocked == blocked[i];
            same = same && agree;
            std::cout << (hit[i] < 0.0f ? "miss" : "wall at x=")
                      << (hit[i] < 0.0f ? "" : std::to_string(int(hit[i])))
                      << (blocked[i] ? ", occluded" : ", clear")
                      << (agree ? "" : " -- DIFFERS from scalar") << '\n';
        }
    }
    std::cout << (same ? "same as scalar" : "DIFFERS from scalar") << '\n';
    return same ? 0 : 1;
}
