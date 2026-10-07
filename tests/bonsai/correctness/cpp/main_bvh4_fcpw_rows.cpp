// Used in bvh4_fcpw_rows.bonsai.
//
// The scene, tree and rays of main_bvh4_fcpw_slabs.cpp -- walls facing the
// x axis in lanes along z, a tree split in x so that a ray along x crosses
// every child's box of every node it enters -- written into FCPW's bytes:
// one array of node rows, interior and leaf alike, a leaf row naming its
// packets of four triangles (fcpw/aggregates/mbvh.inl, MbvhNode and
// MbvhLeafNode at a branching factor of 4 and a width of 4). Each answer is
// checked against the one the scene was built to give; the prefetch of the
// hit children's rows and packets is the thing under test, and a prefetch
// of a bad address would fault here.
#include "bvh4_fcpw_rows.h"

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

// The storage, as the generated header spells it: a packet is FCPW's
// MbvhLeafNode<4, 3>, a row its MbvhNode<3> over four children.
using Packet = _tree_layout1;
using NodeRow = _tree_layout2;

static_assert(sizeof(Packet) == 3 * 3 * 16 + 16,
              "a packet is three vertices by component, four wide, and four indices");
static_assert(sizeof(NodeRow) == 6 * 16 + 4 * 4,
              "a node row is six bounds by component, four wide, and four slots");
static_assert(offsetof(NodeRow, child0) == 96, "the slots follow the bounds");

// The tree in FCPW's layout. A child slot is a row index; a leaf row's first
// slot is `-(first packet + 1)`, its second the packet count, its third the
// offset into the primitive array, its fourth the triangle count. The root
// is row 0, where the traversal starts.
struct Builder {
    std::vector<Packet> packets;
    std::vector<NodeRow> rows;
    int32_t references = 0;

    static void put(Packet &packet, unsigned lane, const Triangle &t, int32_t index) {
        packet.pa.x[lane] = t.p0[0];
        packet.pa.y[lane] = t.p0[1];
        packet.pa.z[lane] = t.p0[2];
        packet.pb.x[lane] = t.p1[0];
        packet.pb.y[lane] = t.p1[1];
        packet.pb.z[lane] = t.p1[2];
        packet.pc.x[lane] = t.p2[0];
        packet.pc.y[lane] = t.p2[1];
        packet.pc.z[lane] = t.p2[2];
        packet.primitiveIndex[lane] = index;
    }

    // A leaf of `tris`: its packets, and a leaf row that names them.
    int32_t leaf(const std::vector<Triangle> &tris, Box &box) {
        const int32_t first = int32_t(packets.size());
        const int32_t count = int32_t((tris.size() + 3) / 4);
        packets.resize(packets.size() + size_t(count));
        box = bounds_of(tris.front());
        for (size_t s = 0; s < tris.size(); s++) {
            put(packets[size_t(first) + s / 4], unsigned(s % 4), tris[s], references + int32_t(s));
            box = merge(box, bounds_of(tris[s]));
        }
        // The lanes past the count stay zero; the count masks them.
        const int32_t row = int32_t(rows.size());
        rows.emplace_back();
        rows.back().child0 = -(first + 1);
        rows.back().child1 = count;
        rows.back().child2 = references;
        rows.back().child3 = int32_t(tris.size());
        references += int32_t(tris.size());
        return row;
    }

    static void fill(NodeRow &n, const int32_t children[4], const Box boxes[4]) {
        int32_t *slots[4] = {&n.child0, &n.child1, &n.child2, &n.child3};
        for (int i = 0; i < 4; i++) {
            *slots[i] = children[i];
            n.boxMin_x[i] = boxes[i].lo[0];
            n.boxMin_y[i] = boxes[i].lo[1];
            n.boxMin_z[i] = boxes[i].lo[2];
            n.boxMax_x[i] = boxes[i].hi[0];
            n.boxMax_y[i] = boxes[i].hi[1];
            n.boxMax_z[i] = boxes[i].hi[2];
        }
    }

    int32_t interior(const int32_t children[4], const Box boxes[4], Box &box) {
        const int32_t row = int32_t(rows.size());
        rows.emplace_back();
        fill(rows.back(), children, boxes);
        box = boxes[0];
        for (int i = 1; i < 4; i++) {
            box = merge(box, boxes[i]);
        }
        return row;
    }
};

_tree_layout0 build_tree(Builder &b) {
    b.rows.emplace_back(); // the root, row 0, filled in after its children
    int32_t root_children[4];
    Box root_boxes[4];
    for (int quarter = 0; quarter < 4; quarter++) {
        int32_t leaves[4];
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

    static bonsai_buffer packets_buffer, rows_buffer;
    packets_buffer = bonsai_buffer_wrap(b.packets.data(), b.packets.size() * sizeof(Packet));
    rows_buffer = bonsai_buffer_wrap(b.rows.data(), b.rows.size() * sizeof(NodeRow));
    _tree_layout0 tree;
    // The layout counts elements: slots, four to a packet.
    tree.nPackets = uint32_t(4 * b.packets.size());
    tree.group0_packets = &packets_buffer;
    tree.nNodes = uint32_t(b.rows.size());
    tree.group1_index = &rows_buffer;
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
