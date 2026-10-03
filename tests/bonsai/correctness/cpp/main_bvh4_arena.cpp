// Used in bvh4_arena.bonsai: the driver of main_bvh4_tiled_elements.cpp, the
// same scene and the same checks, against the program whose storage is one
// arena of bytes with references that are byte offsets (ir::Lookup::shape,
// Embree's NodeRef as an offset), its node and leaf tests vectorized and its
// hit children prefetched. The tree is written here as the rtq driver writes
// Embree's: node rows first, the root at offset 0, then the leaves' tiles,
// each reference the offset of what it names with the kind in its low bits.
//
// The scene and the rays of main_bvh4_child_volumes.cpp -- walls facing the
// x axis in lanes along z, a tree split in x so that a ray along x crosses
// every child's box of every node it enters -- written into Embree's bytes:
// a node row holding the children's bounds as six component vectors (lower
// x, upper x, lower y, ...), and leaves of tiles of four triangles with each
// field a vector over the four, the lanes a leaf does not fill left as the
// zero triangle. Each answer is checked against the one the scene was built
// to give.
#include "bvh4_arena.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// The storage, as the generated header spells it: a row is Embree's
// AABBNode<4>, a tile Embree's Triangle4 over this test's triangle; both
// are shapes read at an offset of the arena, which is all the layout struct
// holds.
using NodeRow = _tree_layout3;
using Tile = _tree_layout5;

// The bytes are Embree's: the bounds at the offsets AABBNode puts them, the
// tile's fields each a vector over its four lanes.
static_assert(sizeof(NodeRow) == 16 + 6 * 16, "a node row is Embree's AABBNode<4>");
static_assert(offsetof(NodeRow, lower_x) == 16 && offsetof(NodeRow, upper_x) == 32 &&
                  offsetof(NodeRow, lower_y) == 48 && offsetof(NodeRow, upper_y) == 64 &&
                  offsetof(NodeRow, lower_z) == 80 && offsetof(NodeRow, upper_z) == 96,
              "the bounds sit where Embree's AABBNode puts them");
static_assert(sizeof(Tile) == 3 * 3 * 16, "a tile is three vertices by component, four wide");

// The tree in the layout the schedule declared. The rows and tiles are
// gathered here as they are made, by index, and written into one arena
// afterwards: rows first, then tiles, each reference the byte offset of
// what it names with the kind in its low four bits -- `offset` for an
// interior node, `offset | (8 + count)` for a leaf of `count` tiles --
// Embree's encoding, with an offset where Embree has a pointer. The root is
// row 0, at offset 0, which is where the traversal starts.
struct Builder {
    std::vector<Tile> tiles;   // in leaf order
    std::vector<NodeRow> rows; // the interior nodes
    std::vector<uint8_t> arena;

    // A triangle into lane `lane` of a tile, by component.
    static void put(Tile &tile, unsigned lane, const Triangle &t) {
        tile.p0.x[lane] = t.p0[0];
        tile.p0.y[lane] = t.p0[1];
        tile.p0.z[lane] = t.p0[2];
        tile.p1.x[lane] = t.p1[0];
        tile.p1.y[lane] = t.p1[1];
        tile.p1.z[lane] = t.p1[2];
        tile.p2.x[lane] = t.p2[0];
        tile.p2.y[lane] = t.p2[1];
        tile.p2.z[lane] = t.p2[2];
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
                // A lane the leaf does not fill: Embree's zero triangle,
                // which no ray hits.
                put(tile, unsigned(s % 4), zero);
            }
        }
        return (first << 4) | (8u + count);
    }

    // A node's row: its four children and their boxes, into lane i of each of
    // the six bound vectors, as Embree's AABBNode::setBounds puts them.
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

    // An interior node over four children, each already built, with its box.
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

// Root: four children by pairs of wall positions (x-slabs, so a ray along x
// crosses them all). Each child: four leaves, one per (wall, pair of lanes),
// of two triangles each -- one tile, half full.
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
                std::vector<Triangle> tris = {wall(k, 2 * lanes), wall(k, 2 * lanes + 1)};
                leaves[l] = b.leaf(tris, leaf_boxes[l]);
                l++;
            }
        }
        root_children[quarter] = b.interior(leaves, leaf_boxes, root_boxes[quarter]);
    }
    Builder::fill(b.rows[0], root_children, root_boxes);

    // The arena: the rows, then the tiles, the references rewritten from
    // indices to byte offsets as the rows are copied in. It reaches the
    // program as a buffer descriptor (runtime/bonsai_buffer.h), which has to
    // outlive the tree.
    const uint32_t rows_bytes = uint32_t(b.rows.size() * sizeof(NodeRow));
    b.arena.assign(rows_bytes + b.tiles.size() * sizeof(Tile), 0);
    for (size_t row = 0; row < b.rows.size(); row++) {
        NodeRow n = b.rows[row];
        for (int i = 0; i < 4; i++) {
            const uint32_t child = n.children[i];
            n.children[i] = (child & 15) == 0
                                ? uint32_t((child >> 4) * sizeof(NodeRow))
                                : uint32_t(rows_bytes + (child >> 4) * sizeof(Tile)) |
                                      (child & 15);
        }
        std::memcpy(b.arena.data() + row * sizeof(NodeRow), &n, sizeof(NodeRow));
    }
    for (size_t tile = 0; tile < b.tiles.size(); tile++) {
        std::memcpy(b.arena.data() + rows_bytes + tile * sizeof(Tile), &b.tiles[tile],
                    sizeof(Tile));
    }
    static bonsai_buffer arena_buffer;
    arena_buffer = bonsai_buffer_wrap(b.arena.data(), b.arena.size());
    _tree_layout0 tree;
    tree.bytes = uint32_t(b.arena.size());
    tree.group0_arena = &arena_buffer;
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
        report("+x short", Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f}, tree, nullptr);
    }
    // Between lanes: through every box the walls' boxes span, hitting none.
    report("+x between lanes", Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward},
           tree, nullptr);
    // Climbing across lanes: hits the first wall where it crosses lane 0.
    const Triangle first0 = wall(0, 0);
    report("diagonal", Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}}, tree,
           &first0);
    // Along z at a wall's x, past the walls' y: nothing.
    report("+z beside", Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}}, tree,
           nullptr);
    return 0;
}
