// Used in any_hit_arms.bonsai.
//
// One eight-wide node over eight leaves of a triangle each, the children's
// boxes stepped in height so that a ray straight down at a given y crosses
// a chosen set of them: one box, two, three, four, five or all eight, the
// sets not contiguous in lane order. Every triangle k stands in box k with
// its apex just under the box's top, at x = k + 0.5, so the same ray hits
// triangle k exactly when box k is among those crossed and k is the lane
// the ray is over -- the lowest lane of the set, a middle one, or the
// highest -- which is what checks that each arm of the switch on the count
// pushes every lane it does not descend into, in both directions of the
// stack. Rays over a lane whose box the ray misses, or past every triangle
// (a short segment), have to come out clear after everything pushed has
// been visited.
#include "any_hit_arms.h"

#include <cstdint>
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

Triangle triangle(int k) {
    const float x = float(k) + 0.5f;
    Triangle t;
    t.p0 = float3{x - 0.25f, -0.5f, kTriangleZ};
    t.p1 = float3{x + 0.25f, -0.5f, kTriangleZ};
    t.p2 = float3{x, kTop[k] - 0.02f, kTriangleZ};
    return t;
}

// The tree in the layout the schedule declared: a reference is `row << 4`
// for an interior node and `(first << 4) | (8 + count)` for a leaf. One
// interior row, the root, whose child k is a leaf of triangle k in a box
// spanning every x and z the scene has, up to the lane's own top.
_tree_layout0 build_tree() {
    static std::vector<Triangle> prims;
    static std::vector<_tree_layout1> rows(1);
    for (int k = 0; k < kLanes; k++) {
        prims.push_back(triangle(k));
        _tree_layout1 &root = rows[0];
        root.children[k] = (uint32_t(k) << 4) | 9u;
        root.lo.x[k] = 0.0f;
        root.lo.y[k] = -1.0f;
        root.lo.z[k] = 0.0f;
        root.hi.x[k] = float(kLanes);
        root.hi.y[k] = kTop[k];
        root.hi.z[k] = 1.0f;
    }
    static bonsai_buffer prims_buffer, rows_buffer;
    prims_buffer = bonsai_buffer_wrap(prims.data(), prims.size() * sizeof(Triangle));
    rows_buffer = bonsai_buffer_wrap(rows.data(), rows.size() * sizeof(_tree_layout1));
    _tree_layout0 tree;
    tree.pCount = uint32_t(prims.size());
    tree.prims = &prims_buffer;
    tree.nCount = uint32_t(rows.size());
    tree.group0_row = &rows_buffer;
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
