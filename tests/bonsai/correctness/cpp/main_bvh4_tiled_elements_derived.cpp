// Used in bvh4_tiled_elements_derived.bonsai.
//
// The scene, the tree and the rays of main_bvh4_tiled_elements.cpp -- walls
// facing the x axis in lanes along z, a tree split in x so that a ray along
// x crosses every child's box of every node it enters -- with the leaf's
// tiles written as a vertex and two edges per lane (`a`, `e1 = p1 - p0`,
// `e2 = p2 - p0`) where the element has three vertices: the layout derives
// the vertices back (`p1 = a + e1`), exactly, the coordinates here being
// small integers. Each answer is checked against the one the scene was built
// to give, by the vertex the program derived.
#include "bvh4_tiled_elements_derived.h"

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

// The storage, as the generated header spells it: a row is Embree's
// AABBNode<4>; a tile holds the stored fields alone -- a vertex and two
// edges, each a vector over the four lanes -- and none of the derived ones.
using NodeRow = _tree_layout2;
using Tile = _tree_layout1;

static_assert(sizeof(NodeRow) == 16 + 6 * 16, "a node row is Embree's AABBNode<4>");
static_assert(sizeof(Tile) == 3 * 3 * 16,
              "a tile is a vertex and two edges by component, four wide, and nothing "
              "for the derived vertices");

struct Builder {
    std::vector<Tile> tiles;
    std::vector<NodeRow> rows;

    // A triangle into lane `lane` of a tile: its first vertex and the two
    // edges from it, by component.
    static void put(Tile &tile, unsigned lane, const Triangle &t) {
        tile.a.x[lane] = t.p0[0];
        tile.a.y[lane] = t.p0[1];
        tile.a.z[lane] = t.p0[2];
        tile.e1.x[lane] = t.p1[0] - t.p0[0];
        tile.e1.y[lane] = t.p1[1] - t.p0[1];
        tile.e1.z[lane] = t.p1[2] - t.p0[2];
        tile.e2.x[lane] = t.p2[0] - t.p0[0];
        tile.e2.y[lane] = t.p2[1] - t.p0[1];
        tile.e2.z[lane] = t.p2[2] - t.p0[2];
    }

    uint32_t leaf(const std::vector<Triangle> &tris, Box &box) {
        const uint32_t first = uint32_t(tiles.size());
        const uint32_t count = uint32_t((tris.size() + 3) / 4);
        tiles.resize(tiles.size() + count);
        box = bounds_of(tris.front());
        const Triangle zero{};
        for (size_t s = 0; s < 4 * size_t(count); s++) {
            Tile &tile = tiles[first + s / 4];
            if (s < tris.size()) {
                put(tile, unsigned(s % 4), tris[s]);
                box = merge(box, bounds_of(tris[s]));
            } else {
                put(tile, unsigned(s % 4), zero);
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
    for (int quarter = 0; quarter < 4; quarter++) {
        uint32_t leaves[4];
        Box leaf_boxes[4];
        int l = 0;
        for (int k = 2 * quarter; k < 2 * quarter + 2; k++) {
            for (int lanes = 0; lanes < 2; lanes++) {
                std::vector<Triangle> tris = {wall(k, 2 * lanes), wall(k, 2 * lanes + 1)};
                leaves[l] = b.leaf(tris, leaf_boxes[l]);
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

// The hit's triangle is the one the program derived from the tile: all three
// vertices are checked against the wall's, not the first alone, since the
// derived ones are what is new here.
void report(const char *what, const Ray &r, const _tree_layout0 &tree,
            const Triangle *expected) {
    _option0 hit;
    trace(hit, r, tree);
    const bool blocked = occluded(r, tree);
    std::cout << what << ": ";
    if (!hit.set) {
        std::cout << "miss";
    } else {
        std::cout << "wall at x=" << hit.value.p0[0] << " z=" << hit.value.p0[2] + 1.0f
                  << " (p1 " << hit.value.p1[0] << "," << hit.value.p1[1] << ","
                  << hit.value.p1[2] << "; p2 " << hit.value.p2[0] << ","
                  << hit.value.p2[1] << "," << hit.value.p2[2] << ")";
    }
    std::cout << (blocked ? ", occluded" : ", clear");
    bool right;
    if (expected == nullptr) {
        right = !hit.set && !blocked;
    } else {
        right = hit.set && blocked;
        for (int c = 0; c < 3 && right; c++) {
            right = hit.value.p0[c] == expected->p0[c] &&
                    hit.value.p1[c] == expected->p1[c] &&
                    hit.value.p2[c] == expected->p2[c];
        }
    }
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
        const Triangle first = wall(0, lane);
        report("+x", Ray{float3{-1.0f, 0.0f, z}, forward}, tree, &first);
        const Triangle last = wall(kWalls - 1, lane);
        report("-x", Ray{float3{100.0f, 0.0f, z}, backward}, tree, &last);
        const Triangle ahead = wall(4, lane);
        report("+x from 35", Ray{float3{35.0f, 0.0f, z}, forward}, tree, &ahead);
        report("+x short", Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f}, tree, nullptr);
    }
    report("+x between lanes", Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward},
           tree, nullptr);
    const Triangle first0 = wall(0, 0);
    report("diagonal", Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}}, tree,
           &first0);
    report("+z beside", Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}}, tree,
           nullptr);
    return 0;
}
