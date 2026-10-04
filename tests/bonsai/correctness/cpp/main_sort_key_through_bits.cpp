// Used in sort_key_through_bits.bonsai: main_sort_key_nonnegative.cpp's
// scene, rays and checks against the program whose slab test is on the
// floats' bits, Embree's AVX-512 form. The same answers are expected of it:
// the walls scene answered as the scene says by both traversals, and the
// raw query accepting a wall behind the origin on a negative `tnear` where
// the clamped one sees the wall ahead.
#include "sort_key_through_bits.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int kWalls = 8;
constexpr int kLanes = 4;
constexpr float kWallSpacing = 10.0f;
constexpr float kLaneSpacing = 5.0f;
constexpr float kInf = std::numeric_limits<float>::infinity();

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

// The tree in the layout the schedule declared: a reference is `row << 4`
// for an interior node and `(first << 4) | (8 + count)` for a leaf.
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

Ray ray(float3 o, float3 d, float tnear = 0.0f, float tfar = kInf) {
    Ray r;
    r.o = o;
    r.tnear = tnear;
    r.d = d;
    r.tfar = tfar;
    return r;
}

void print_hit(const _option0 &hit) {
    if (!hit.set) {
        std::cout << "miss";
    } else {
        std::cout << "wall at x=" << hit.value.p0[0]
                  << " z=" << hit.value.p0[2] + 1.0f;
    }
}

bool matches(const _option0 &hit, const Triangle *expected) {
    if (expected == nullptr) {
        return !hit.set;
    }
    return hit.set && hit.value.p0[0] == expected->p0[0] &&
           hit.value.p0[2] == expected->p0[2];
}

// `clamped` is what `trace` and `occluded` should find, `raw` what
// `trace_raw` should -- where its answer is specified: the slab test on the
// floats' bits orders negative distances backwards (the bits of a negative
// float are a negative integer, and more negative the farther from zero),
// so on a segment that begins behind the origin the raw query, which does
// not clamp, is outside the test's precondition, as it would be outside
// Embree's (`assert(ray.valid())`, and `tnear = max(tnear, 0)` before any
// node is tested). Its answer is printed and not judged there.
void report(const char *what, const Ray &r, const _tree_layout0 &tree,
            const Triangle *clamped, const Triangle *raw,
            bool raw_specified = true) {
    _option0 hit, hit_raw;
    trace(hit, r, tree);
    trace_raw(hit_raw, r, tree);
    const bool blocked = occluded(r, tree);
    std::cout << what << ": clamped ";
    print_hit(hit);
    std::cout << ", raw ";
    print_hit(hit_raw);
    std::cout << (blocked ? ", occluded" : ", clear");
    const bool right = matches(hit, clamped) &&
                       (!raw_specified || matches(hit_raw, raw)) &&
                       blocked == (clamped != nullptr);
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
        const Triangle second = wall(1, lane);
        const Triangle last = wall(kWalls - 1, lane);
        const Triangle ahead = wall(4, lane);
        // Down the lane towards +x: the nearest wall is the first.
        report("+x", ray(float3{-1.0f, 0.0f, z}, forward), tree, &first, &first);
        // Back from beyond the last: the farthest wall.
        report("-x", ray(float3{100.0f, 0.0f, z}, backward), tree, &last, &last);
        // From between two walls: the one ahead.
        report("+x from 35", ray(float3{35.0f, 0.0f, z}, forward), tree, &ahead,
               &ahead);
        // A segment that ends before the first wall.
        report("+x short", ray(float3{-1.0f, 0.0f, z}, forward, 0.0f, 0.5f),
               tree, nullptr, nullptr);
        // From x = 2 with the segment starting at -5: the clamped query
        // sees the wall ahead at x = 10. The raw query's segment begins
        // behind the origin, which the slab test on bits does not order
        // (see report); its answer is shown, not checked.
        report("+x behind", ray(float3{2.0f, 0.0f, z}, forward, -5.0f), tree,
               &second, nullptr, /*raw_specified=*/false);
        // The same segment from before the first wall: the clamped query
        // sees it; the raw one is again outside the precondition.
        report("+x negative tnear", ray(float3{-1.0f, 0.0f, z}, forward, -5.0f),
               tree, &first, nullptr, /*raw_specified=*/false);
    }
    // Between lanes: through every box the walls' boxes span, hitting none.
    report("+x between lanes",
           ray(float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward), tree, nullptr,
           nullptr);
    // Climbing across lanes: the first wall where it crosses lane 0.
    const Triangle first0 = wall(0, 0);
    report("diagonal", ray(float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}),
           tree, &first0, &first0);
    // Along z at a wall's x, past the walls' y: nothing.
    report("+z beside", ray(float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}),
           tree, nullptr, nullptr);
    return 0;
}
