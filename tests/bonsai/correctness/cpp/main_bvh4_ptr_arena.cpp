// Used in bvh4_ptr_arena.bonsai: the driver of main_bvh4_arena.cpp, the same
// scene and the same checks, against the program whose references are
// pointers -- Embree's NodeRef as it is, the address of what it names with
// the kind in its low four bits, bit 3 a leaf and the three below it the
// leaf's tile count -- looked up in a `ptr group`. The tree is written here
// as Embree's builder leaves it: node rows and then the leaves' tiles in one
// allocation, every reference an address into it, and the root's address in
// the layout's `root` field.
//
// The scene and the rays of main_bvh4_child_volumes.cpp -- walls facing the
// x axis in lanes along z, a tree split in x so that a ray along x crosses
// every child's box of every node it enters -- written into Embree's bytes:
// a node row holding the children's bounds as six component vectors (lower
// x, upper x, lower y, ...), and leaves of tiles of four triangles with each
// field a vector over the four, the lanes a leaf does not fill left as the
// zero triangle. Each answer is checked against the one the scene was built
// to give.
#include "bvh4_ptr_arena.h"

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
// are shapes read at the address a reference carries. The layout struct
// holds the root's address and the one allocation the addresses point into.
using NodeRow = _tree_layout3;
using Tile = _tree_layout5;

// The bytes are Embree's: four 64-bit children, then the bounds at the
// offsets AABBNode<4> puts them, the tile's fields each a vector over its
// four lanes.
static_assert(sizeof(NodeRow) == 32 + 6 * 16, "a node row is Embree's AABBNode<4>");
static_assert(offsetof(NodeRow, lower_x) == 32 && offsetof(NodeRow, upper_x) == 48 &&
                  offsetof(NodeRow, lower_y) == 64 && offsetof(NodeRow, upper_y) == 80 &&
                  offsetof(NodeRow, lower_z) == 96 && offsetof(NodeRow, upper_z) == 112,
              "the bounds sit where Embree's AABBNode puts them");
static_assert(sizeof(Tile) == 3 * 3 * 16, "a tile is three vertices by component, four wide");

// The tree in the layout the schedule declared. The rows and tiles are
// gathered here as they are made, by index, and written into one allocation
// afterwards: rows first, then tiles, each reference the address of what it
// names with the kind in its low four bits -- the address for an interior
// node, `address | (8 + count)` for a leaf of `count` tiles -- Embree's
// NodeRef exactly. The root is row 0, and its address goes in the layout's
// `root` field, where the traversal starts.
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

    uint64_t leaf(const std::vector<Triangle> &tris, Box &box) {
        const uint64_t first = uint64_t(tiles.size());
        const uint64_t count = uint64_t((tris.size() + 3) / 4);
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
    static void fill(NodeRow &n, const uint64_t children[4], const Box boxes[4]) {
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
    uint64_t interior(const uint64_t children[4], const Box boxes[4], Box &box) {
        const uint64_t row = uint64_t(rows.size());
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
    uint64_t root_children[4];
    Box root_boxes[4];
    for (int quarter = 0; quarter < 4; quarter++) {
        uint64_t leaves[4];
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

    // The allocation: the rows, then the tiles, the references rewritten
    // from indices to the addresses of what they name as the rows are copied
    // in -- an interior node's address with its kind bits zero, a leaf's
    // first tile's address with 8 + its tile count. The bytes reach the
    // program as a buffer descriptor (runtime/bonsai_buffer.h) that only
    // owns them; what the traversal follows are the addresses, so the
    // allocation is sized once and never moves.
    const uint64_t rows_bytes = uint64_t(b.rows.size() * sizeof(NodeRow));
    b.arena.assign(rows_bytes + b.tiles.size() * sizeof(Tile), 0);
    const uint64_t base = reinterpret_cast<uint64_t>(b.arena.data());
    for (size_t row = 0; row < b.rows.size(); row++) {
        NodeRow n = b.rows[row];
        for (int i = 0; i < 4; i++) {
            const uint64_t child = n.children[i];
            n.children[i] = (child & 8) == 0
                                ? base + (child >> 4) * sizeof(NodeRow)
                                : (base + rows_bytes + (child >> 4) * sizeof(Tile)) |
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
    tree.root = base; // row 0, an interior node: its address is its reference
    tree.bytes = uint64_t(b.arena.size());
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
