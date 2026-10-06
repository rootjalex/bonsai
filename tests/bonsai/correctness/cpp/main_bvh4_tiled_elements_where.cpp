// Used in bvh4_tiled_elements_where.bonsai.
//
// The scene, the tree and the rays of main_bvh4_tiled_elements.cpp -- walls
// facing the x axis in lanes along z, a tree split in x -- with every tile
// carrying an id per lane and the lanes a leaf does not fill holding DECOYS
// rather than the zero triangle: a wall half a unit in front of the leaf's
// first real wall, with an id of -1. A ray down a lane would hit the decoy
// first, and a point just behind a real wall would be nearer to it; the
// layout's `where id != 4294967295u` says those lanes are not triangles, and
// every answer here is checked to be the real wall's. The closest-point
// queries also exercise the lowering's tightening of the best by each
// child's farthest distance (lower/argmin-tighten-children.bonsai).
#include "bvh4_tiled_elements_where.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kWalls = 8;
constexpr int kLanes = 4;
constexpr float kWallSpacing = 10.0f;
constexpr float kLaneSpacing = 5.0f;
constexpr uint32_t kNoId = 0xffffffffu;

Triangle wall_at(float x, int lane) {
    const float z = kLaneSpacing * float(lane);
    Triangle t;
    t.p0 = float3{x, -1.0f, z - 1.0f};
    t.p1 = float3{x, 1.0f, z - 1.0f};
    t.p2 = float3{x, 0.0f, z + 1.0f};
    return t;
}

Triangle wall(int k, int lane) { return wall_at(kWallSpacing * float(k), lane); }

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

using NodeRow = _tree_layout2;
using Tile = _tree_layout1;

static_assert(sizeof(NodeRow) == 16 + 6 * 16, "a node row is Embree's AABBNode<4>");
static_assert(sizeof(Tile) == 3 * 3 * 16 + 16,
              "a tile is three vertices by component and an id, four wide");

struct Builder {
    std::vector<Tile> tiles;
    std::vector<NodeRow> rows;

    static void put(Tile &tile, unsigned lane, const Triangle &t, uint32_t id) {
        tile.p0.x[lane] = t.p0[0];
        tile.p0.y[lane] = t.p0[1];
        tile.p0.z[lane] = t.p0[2];
        tile.p1.x[lane] = t.p1[0];
        tile.p1.y[lane] = t.p1[1];
        tile.p1.z[lane] = t.p1[2];
        tile.p2.x[lane] = t.p2[0];
        tile.p2.y[lane] = t.p2[1];
        tile.p2.z[lane] = t.p2[2];
        tile.id[lane] = id;
    }

    // A leaf of `tris`, its box theirs alone: the decoy in each unfilled
    // lane sits half a unit before the first wall, outside nothing a ray
    // or a point would otherwise ask of this leaf, and inside its box.
    uint32_t leaf(const std::vector<Triangle> &tris, Box &box, uint32_t first_id) {
        const uint32_t first = uint32_t(tiles.size());
        const uint32_t count = uint32_t((tris.size() + 3) / 4);
        tiles.resize(tiles.size() + count);
        box = bounds_of(tris.front());
        const float x_first = tris.front().p0[0];
        const float z_first = tris.front().p0[2] + 1.0f;
        const Triangle decoy = wall_at(x_first - 0.5f, int(z_first / kLaneSpacing));
        for (size_t s = 0; s < 4 * size_t(count); s++) {
            Tile &tile = tiles[first + s / 4];
            if (s < tris.size()) {
                put(tile, unsigned(s % 4), tris[s], first_id + uint32_t(s));
                box = merge(box, bounds_of(tris[s]));
            } else {
                put(tile, unsigned(s % 4), decoy, kNoId);
                box = merge(box, bounds_of(decoy));
            }
        }
        return (first << 4) | (8u + count);
    }

    static void fill(NodeRow &n, const uint32_t children[4], const Box boxes[4]) {
        for (int i = 0; i < 4; i++) {
            n.children[i] = children[i];
            n.lower_x[i] = boxes[i].lo[0];
            n.lower_y[i] = boxes[i].lo[1];
            n.lower_z[i] = boxes[i].lo[2];
            n.upper_x[i] = boxes[i].hi[0];
            n.upper_y[i] = boxes[i].hi[1];
            n.upper_z[i] = boxes[i].hi[2];
        }
    }

    uint32_t interior(const uint32_t children[4], const Box boxes[4], Box &box) {
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
    uint32_t next_id = 0;
    for (int quarter = 0; quarter < 4; quarter++) {
        uint32_t leaves[4];
        Box leaf_boxes[4];
        int l = 0;
        for (int k = 2 * quarter; k < 2 * quarter + 2; k++) {
            for (int lanes = 0; lanes < 2; lanes++) {
                std::vector<Triangle> tris = {wall(k, 2 * lanes), wall(k, 2 * lanes + 1)};
                leaves[l] = b.leaf(tris, leaf_boxes[l], next_id);
                next_id += 2;
                l++;
            }
        }
        root_children[quarter] = b.interior(leaves, leaf_boxes, root_boxes[quarter]);
    }
    Builder::fill(b.rows[0], root_children, root_boxes);

    static bonsai_buffer tiles_buffer, rows_buffer;
    tiles_buffer = bonsai_buffer_wrap(b.tiles.data(), b.tiles.size() * sizeof(Tile));
    rows_buffer = bonsai_buffer_wrap(b.rows.data(), b.rows.size() * sizeof(NodeRow));
    _tree_layout0 tree;
    tree.pCount = uint32_t(4 * b.tiles.size());
    tree.group0_prims = &tiles_buffer;
    tree.nCount = uint32_t(b.rows.size());
    tree.group1_row = &rows_buffer;
    return tree;
}

bool same_wall(const Triangle &a, const Triangle &b) {
    return a.p0[0] == b.p0[0] && a.p0[2] == b.p0[2];
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
    const bool right = expected == nullptr
                           ? (!hit.set && !blocked)
                           : (hit.set && blocked && same_wall(hit.value, *expected));
    std::cout << (right ? " -- ok" : " -- WRONG") << '\n';
    if (!right) {
        std::exit(1);
    }
}

void report_closest(const char *what, const Point &p, const _tree_layout0 &tree,
                    const Triangle &expected) {
    _option0 found;
    closest(found, p, tree);
    std::cout << what << ": ";
    if (!found.set) {
        std::cout << "none";
    } else {
        std::cout << "wall at x=" << found.value.p0[0] << " z=" << found.value.p0[2] + 1.0f;
    }
    const bool right = found.set && same_wall(found.value, expected);
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
        // The decoy of the first leaf sits at x = -0.5, between this ray's
        // origin and the first wall: the answer is still the wall.
        const Triangle first = wall(0, lane);
        report("+x", Ray{float3{-1.0f, 0.0f, z}, forward}, tree, &first);
        const Triangle last = wall(kWalls - 1, lane);
        report("-x", Ray{float3{100.0f, 0.0f, z}, backward}, tree, &last);
        // From 35 towards wall 4 at 40, past its leaf's decoy at 39.5.
        const Triangle ahead = wall(4, lane);
        report("+x from 35", Ray{float3{35.0f, 0.0f, z}, forward}, tree, &ahead);
        report("+x short", Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f}, tree, nullptr);
        // Closest points: a point one unit before wall k is nearest to it,
        // and nearer still to its leaf's decoy half a unit before it, which
        // is not a triangle of the set.
        for (int k = 0; k < kWalls; k += 3) {
            const Triangle w = wall(k, lane);
            const Point p{float3{kWallSpacing * float(k) - 1.0f, 0.0f, z}};
            report_closest("closest before wall", p, tree, w);
        }
        // A point two units past wall k, eight before the next: wall k.
        const Triangle w2 = wall(2, lane);
        const Point past{float3{kWallSpacing * 2.0f + 2.0f, 0.0f, z}};
        report_closest("closest past wall 2", past, tree, w2);
    }
    report("+x between lanes", Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward},
           tree, nullptr);
    const Triangle first0 = wall(0, 0);
    report("diagonal", Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}}, tree,
           &first0);
    report("+z beside", Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}}, tree,
           nullptr);
    // Far off along -x: the first wall of lane 0, through every decoy's box
    // without counting one.
    const Triangle far_first = wall(0, 0);
    report_closest("closest from far", Point{float3{-100.0f, 0.0f, 0.0f}}, tree, far_first);
    return 0;
}
