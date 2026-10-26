// Used in layout_tile_group_reads.bonsai.
//
// The scene, the tree and the rays of main_bvh4_tiled_elements_derived.cpp
// -- walls facing the x axis in lanes along z, a tree split in x so that a
// ray along x crosses every child's box of every node it enters -- with the
// triangles stored as an indexed mesh: a vertex table, face objects of three
// indices into it, and the leaves' tiles each holding the address of a
// face. The vertex table is filled in the reverse of the faces' order, so
// that a corner read by the wrong index, or by the index taken for a
// position, is some other wall's corner and the check below catches it.
// The faces live in a vector of their own and are reached by address alone
// (the layout's `ptr group`): the layout is handed the vector only so the
// buffer exists, and nothing indexes it. Each answer is checked against the
// wall the scene was built to give, by all three vertices the program
// gathered.
#include "layout_tile_group_reads.h"

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

// The storage, as the generated header spells it: a vertex is twelve bytes,
// a face sixteen (three indices and a tag), a tile the address of a face, a
// row Embree's AABBNode<4>.
using Vertex = _tree_layout1;
using Face = _tree_layout2;
using Tile = _tree_layout3;
using NodeRow = _tree_layout4;

static_assert(sizeof(Vertex) == 12, "a vertex is three floats");
static_assert(sizeof(Face) == 16, "a face is three indices and a tag");
static_assert(sizeof(Tile) == 8, "a tile is one face's address");
static_assert(sizeof(NodeRow) == 16 + 6 * 16, "a node row is Embree's AABBNode<4>");

struct Builder {
    std::vector<Vertex> vertices; // filled in reverse, see mesh()
    std::vector<Face> faces;      // one per triangle, never moved once built
    std::vector<Tile> tiles;
    std::vector<NodeRow> rows;

    // The mesh: every wall's three corners, with the vertex table laid out
    // backwards -- the last triangle's corners first -- so that a face's
    // indices are not its own position in the table.
    void mesh() {
        const int count = kWalls * kLanes;
        vertices.resize(3 * size_t(count));
        faces.reserve(size_t(count));
        for (int k = 0; k < kWalls; k++) {
            for (int lane = 0; lane < kLanes; lane++) {
                const Triangle t = wall(k, lane);
                const int f = k * kLanes + lane;
                const int base = 3 * (count - 1 - f);
                for (int c = 0; c < 3; c++) {
                    vertices[size_t(base)].p[c] = t.p0[c];
                    vertices[size_t(base + 1)].p[c] = t.p1[c];
                    vertices[size_t(base + 2)].p[c] = t.p2[c];
                }
                Face face;
                face.i0 = base;
                face.i1 = base + 1;
                face.i2 = base + 2;
                face.tag = f;
                faces.push_back(face);
            }
        }
    }

    // A leaf of the given walls: one tile a triangle, holding the address of
    // its face; the reference is the first tile's index with 8 + the count
    // in its low bits.
    uint32_t leaf(const std::vector<int> &which, Box &box) {
        const uint32_t first = uint32_t(tiles.size());
        box = bounds_of(wall(which.front() / kLanes, which.front() % kLanes));
        for (const int f : which) {
            Tile tile;
            tile.face[0] = reinterpret_cast<uint64_t>(&faces[size_t(f)]);
            tiles.push_back(tile);
            box = merge(box, bounds_of(wall(f / kLanes, f % kLanes)));
        }
        return (first << 4) | (8u + uint32_t(which.size()));
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

// Root: four children by pairs of wall positions. Each child: four leaves,
// one per (wall, pair of lanes), of two triangles each.
_tree_layout0 build_tree(Builder &b) {
    b.mesh();
    b.rows.emplace_back();
    uint32_t root_children[4];
    Box root_boxes[4];
    for (int quarter = 0; quarter < 4; quarter++) {
        uint32_t leaves[4];
        Box leaf_boxes[4];
        int l = 0;
        for (int k = 2 * quarter; k < 2 * quarter + 2; k++) {
            for (int lanes = 0; lanes < 2; lanes++) {
                const std::vector<int> which = {k * kLanes + 2 * lanes,
                                                k * kLanes + 2 * lanes + 1};
                leaves[l] = b.leaf(which, leaf_boxes[l]);
                l++;
            }
        }
        root_children[quarter] = b.interior(leaves, leaf_boxes, root_boxes[quarter]);
    }
    Builder::fill(b.rows[0], root_children, root_boxes);

    static bonsai_buffer vertices_buffer, faces_buffer, tiles_buffer, rows_buffer;
    vertices_buffer =
        bonsai_buffer_wrap(b.vertices.data(), b.vertices.size() * sizeof(Vertex));
    faces_buffer = bonsai_buffer_wrap(b.faces.data(), b.faces.size() * sizeof(Face));
    tiles_buffer = bonsai_buffer_wrap(b.tiles.data(), b.tiles.size() * sizeof(Tile));
    rows_buffer = bonsai_buffer_wrap(b.rows.data(), b.rows.size() * sizeof(NodeRow));
    _tree_layout0 tree;
    tree.vCount = uint32_t(b.vertices.size());
    tree.group0_vertices = &vertices_buffer;
    tree.fCount = uint32_t(b.faces.size());
    tree.group1_faces = &faces_buffer;
    tree.pCount = uint32_t(b.tiles.size());
    tree.group2_prims = &tiles_buffer;
    tree.nCount = uint32_t(b.rows.size());
    tree.group3_row = &rows_buffer;
    return tree;
}

// The hit's triangle is the one the program gathered: all three vertices are
// checked against the wall's.
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
