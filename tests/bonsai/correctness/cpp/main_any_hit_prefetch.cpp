// Used in any_hit_prefetch.bonsai.
//
// main_any_hit_arms.cpp's scene -- one eight-wide node over eight leaves of
// a triangle each, the children's boxes stepped in height so that a ray
// straight down at a given y crosses a chosen set of them, one box, two,
// three, four, five or all eight, the sets not contiguous in lane order --
// written in Embree's bytes with references that are pointers: one
// allocation holding the node's row (AABBNode<8>: eight 64-bit children,
// then the bounds by component) and then eight tiles of four triangles, the
// leaf k's triangle in lane 0 of tile k and the zero triangle, which no ray
// hits, in the other three; a child is the address of its tile with the
// kind bits 8 + 1 (a leaf of one tile), and the layout's `root` is the
// row's address. The same rays and the same expected answers as
// any_hit_arms: each arm of the switch on the hit count has to push every
// lane it does not descend into, and its prefetches, addressed by the
// children it loads, have to touch memory that exists.
#include "any_hit_prefetch.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kLanes = 8;
constexpr float kTriangleZ = 0.5f;
// Box k's top, and the apex of its triangle 0.02 below it. The rays' y
// values sit at least 0.02 from every one of these.
constexpr float kTop[kLanes] = {0.25f, 1.2f, 0.4f, 0.25f, 0.6f, 0.25f, 0.9f, 0.8f};

// A triangle as three vertices, to put into a tile by component. The
// generated header has no Triangle of its own here: the query's interface
// carries none (its storage is the tiles, its answer a bool), and the header
// spells out only what the interface needs.
struct Tri {
    float3 p0, p1, p2;
};

Tri triangle(int k) {
    const float x = float(k) + 0.5f;
    Tri t;
    t.p0 = float3{x - 0.25f, -0.5f, kTriangleZ};
    t.p1 = float3{x + 0.25f, -0.5f, kTriangleZ};
    t.p2 = float3{x, kTop[k] - 0.02f, kTriangleZ};
    return t;
}

// The storage, as the generated header spells it: the node's row is
// Embree's AABBNode<8>, a tile Embree's Triangle4 over this test's
// triangle; both are shapes read at the address a reference carries.
using NodeRow = _tree_layout3;
using Tile = _tree_layout5;

static_assert(sizeof(NodeRow) == 64 + 6 * 32, "a node row is Embree's AABBNode<8>");
static_assert(offsetof(NodeRow, lower_x) == 64 && offsetof(NodeRow, upper_x) == 96 &&
                  offsetof(NodeRow, lower_y) == 128 && offsetof(NodeRow, upper_y) == 160 &&
                  offsetof(NodeRow, lower_z) == 192 && offsetof(NodeRow, upper_z) == 224,
              "the bounds sit where Embree's AABBNode puts them");
static_assert(sizeof(Tile) == 3 * 3 * 16, "a tile is three vertices by component, four wide");

// A triangle into lane `lane` of a tile, by component.
void put(Tile &tile, unsigned lane, const Tri &t) {
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

// The tree: the row first, then tile k for leaf k, every reference the
// address of what it names. The allocation is sized once and never moves,
// and reaches the program as a buffer descriptor that only owns the bytes.
_tree_layout0 build_tree() {
    static std::vector<uint8_t> arena(sizeof(NodeRow) + kLanes * sizeof(Tile), 0);
    static bonsai_buffer arena_buffer;
    const uint64_t base = reinterpret_cast<uint64_t>(arena.data());
    NodeRow row{};
    const Tri zero{};
    for (int k = 0; k < kLanes; k++) {
        Tile tile{};
        put(tile, 0, triangle(k));
        for (unsigned lane = 1; lane < 4; lane++) {
            put(tile, lane, zero);
        }
        const uint64_t at = sizeof(NodeRow) + uint64_t(k) * sizeof(Tile);
        std::memcpy(arena.data() + at, &tile, sizeof(Tile));
        row.children[k] = (base + at) | 9u; // a leaf of one tile
        row.lower_x[k] = 0.0f;
        row.lower_y[k] = -1.0f;
        row.lower_z[k] = 0.0f;
        row.upper_x[k] = float(kLanes);
        row.upper_y[k] = kTop[k];
        row.upper_z[k] = 1.0f;
    }
    std::memcpy(arena.data(), &row, sizeof(NodeRow));
    arena_buffer = bonsai_buffer_wrap(arena.data(), arena.size());
    _tree_layout0 tree;
    tree.root = base; // the row, an interior node: its address is its reference
    tree.bytes = uint64_t(arena.size());
    tree.group0_arena = &arena_buffer;
    return tree;
}

// The boxes a ray straight down at `y` crosses, as the lanes they are in.
std::string boxes_at(float y) {
    std::string lanes;
    for (int k = 0; k < kLanes; k++) {
        if (y <= kTop[k]) {
            lanes += (lanes.empty() ? "" : ",") + std::to_string(k);
        }
    }
    return lanes.empty() ? "none" : lanes;
}

int failures = 0;

// Straight down from z = 2 over lane `lane` at height `y`, the triangles at
// t = 1.5 and the boxes entered at t = 1.
void down(int lane, float y, bool expected, float tmax = std::numeric_limits<float>::infinity()) {
    const Ray r{float3{float(lane) + 0.5f, y, 2.0f}, float3{0.0f, 0.0f, -1.0f}, tmax};
    const bool blocked = occluded(r, build_tree());
    std::cout << "down over lane " << lane << " at y=" << y << " (boxes " << boxes_at(y)
              << (tmax < 2.0f ? ", short" : "") << "): " << (blocked ? "occluded" : "clear")
              << (blocked == expected ? " -- ok" : " -- WRONG") << '\n';
    if (blocked != expected) {
        failures++;
    }
}

} // namespace

int main() {
    // One box, lane 1: the one hit descended into.
    down(1, 1.0f, true);
    down(6, 1.0f, false);
    // Two, lanes 1 and 6: the lower pushed, the higher descended into.
    down(1, 0.85f, true);
    down(6, 0.85f, true);
    down(3, 0.85f, false);
    // Three, lanes 1, 6 and 7: two pushed, lowest first.
    down(1, 0.75f, true);
    down(6, 0.75f, true);
    down(7, 0.75f, true);
    down(0, 0.75f, false);
    // Four, lanes 1, 4, 6 and 7: three pushed.
    down(1, 0.55f, true);
    down(4, 0.55f, true);
    down(6, 0.55f, true);
    down(7, 0.55f, true);
    down(2, 0.55f, false);
    // Five, lanes 1, 2, 4, 6 and 7: the compacting store.
    down(2, 0.35f, true);
    down(4, 0.35f, true);
    down(7, 0.35f, true);
    down(0, 0.35f, false);
    // All eight.
    down(0, 0.2f, true);
    down(3, 0.2f, true);
    down(5, 0.2f, true);
    down(7, 0.2f, true);
    // All eight boxes entered within the segment, every triangle past it.
    down(4, 0.2f, false, 1.2f);
    // Above every box.
    down(2, 1.5f, false);
    if (failures != 0) {
        std::cout << failures << " wrong\n";
        return 1;
    }
    return 0;
}
