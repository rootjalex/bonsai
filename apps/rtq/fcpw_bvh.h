// FCPW's scalar BVH -- the tree WoSX builds by default, FCPW_USE_ENOKI off,
// and the one tree FCPW puts on a GPU -- in the layout that is its bytes, for
// the drivers that compare a program against FCPW on that tree: apps/rtq's
// (rtq_hook.cpp built with RTQ_FCPW_BINARY, which compare.sh does for the
// schedules named fcpwb*) and apps/wosx's (wosx_hook.cpp built with
// WOSX_BINARY_BVH, likewise for its fcpwb* schedules). fcpw_tree.h is the
// same for the vectorized Mbvh, apps/wosx/fcpw_bvh2.h for the layout storing
// the children's boxes in the parent.
//
// Two layouts, two sections, chosen by RTQ_FCPW_CPU_BYTES:
//
// - Without it, the tree in FCPW's GPU bytes (apps/rtq/schedules/layouts/
//   fcpwb_gpu.bonsai): the node rows copied byte for byte -- FCPW's node is
//   BvhNode<3>, the same bytes, which the copy relies on and the asserts
//   check -- and the triangles written as the rows FCPW's own GPU transfer
//   makes (fcpw/gpu/bvh_interop_structures.h, GPUTriangle, extractTriangles:
//   the three corners out of the polygon soup, then pIndex; 40 bytes), in
//   the order FCPW's build left its primitives (Bvh::primitives, which a
//   leaf's referenceOffset indexes), checked by size and offset. What FCPW's
//   GPU traverses, and so what the GPU comparison (RTQ_GPU) traverses.
//
// - With it, the tree in the bytes FCPW's CPU reads (layouts/fcpwb.bonsai):
//   nothing copied -- FCPW's own node array, its primitive pointer array,
//   its Triangle objects and its polygon soup's positions, each wrapped as a
//   buffer where it lies, so that the program's leaf reads the triangle as
//   FCPW's does: the pointer, the three indices through it, the three
//   corners gathered by index. The scene has to outlive the tree. What the
//   CPU comparison traverses, since 2026-10-25 (before, it traversed the GPU
//   rows, an indirection and a gather fewer a triangle than FCPW's CPU
//   pays; apps/rtq/PLAN.md).
//
// Included after the program's generated header and <fcpw/fcpw.h>: the
// layout's structs are the generated header's, as the compiler spells them
// for the layout in use -- for fcpwb_gpu.bonsai `_tree_layout0` the layout,
// `_tree_layout1` a triangle row, `_tree_layout2` a node row; for
// fcpwb.bonsai `_tree_layout1` a position, `_tree_layout2` a Triangle
// object, `_tree_layout3` a reference (one pointer), `_tree_layout4` a node
// row.
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace rtq_bvh {

using FcpwBvh = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle>;
using FcpwNode = fcpw::BvhNode<3>;

// The tree's rows are a protected member of FCPW's Bvh, read through a
// derived class that names them -- the one way to read them without editing
// FCPW (as apps/rtq/fcpw_tree.h reads the Mbvh's). The primitives the leaves
// index are the scene's own vector of triangle pointers, which the build
// sorted in place (SceneData::triangleObjectPtrs, the vector the Bvh holds a
// reference to).
struct Peek : FcpwBvh {
    static const std::vector<FcpwNode> &tree(const FcpwBvh &b) {
        return b.*(&Peek::flatTree);
    }
};

// The scalar BVH of a built FCPW scene, or an exit: the comparison is on
// this tree and no other. (The scene is not const because FCPW's
// getSceneData is not.)
inline const FcpwBvh &scalar_bvh(fcpw::Scene<3> &scene) {
    const auto *bvh = dynamic_cast<const FcpwBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's scene did not build the scalar BVH this driver was compiled "
                     "for (Bvh<3, BvhNode<3>, Triangle>)\n";
        std::exit(1);
    }
    return *bvh;
}

// The one object's triangle pointers, in the order the build left them,
// which a leaf's offset indexes.
inline const std::vector<fcpw::Triangle *> &scene_prims(fcpw::Scene<3> &scene) {
    const auto &objects = scene.getSceneData()->triangleObjectPtrs;
    if (objects.size() != 1) {
        std::cerr << "FCPW's scene holds " << objects.size()
                  << " triangle objects; the driver gives it one\n";
        std::exit(1);
    }
    return objects[0];
}

// Every leaf's range within the primitives, or an exit.
inline void check_leaves(const std::vector<FcpwNode> &rows, size_t prim_count) {
    for (const FcpwNode &node : rows) {
        if (node.nReferences > 0 &&
            uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prim_count) {
            std::cerr << "a leaf of FCPW's BVH references past its primitives\n";
            std::exit(1);
        }
    }
}

#ifdef RTQ_FCPW_CPU_BYTES

using PositionRow = _tree_layout1;
using ObjectRow = _tree_layout2;
using PrimRow = _tree_layout3;
using LayoutNode = _tree_layout4;

static_assert(sizeof(PositionRow) == sizeof(fcpw::Vector3) && sizeof(PositionRow) == 12,
              "a position row is FCPW's Vector3: three floats");
static_assert(sizeof(ObjectRow) == sizeof(fcpw::Triangle) && sizeof(ObjectRow) == 32,
              "an object row is FCPW's Triangle: 32 bytes");
static_assert(offsetof(ObjectRow, i0) == offsetof(fcpw::Triangle, indices) &&
                  offsetof(ObjectRow, i1) == offsetof(fcpw::Triangle, indices) + 4 &&
                  offsetof(ObjectRow, i2) == offsetof(fcpw::Triangle, indices) + 8 &&
                  offsetof(ObjectRow, pIndex) == offsetof(fcpw::Triangle, pIndex) &&
                  offsetof(ObjectRow, soup) == offsetof(fcpw::Triangle, soup),
              "the indices, the index and the soup sit where FCPW's Triangle puts them");
static_assert(sizeof(PrimRow) == sizeof(fcpw::Triangle *), "a reference row is one pointer");
static_assert(sizeof(LayoutNode) == sizeof(FcpwNode) && sizeof(LayoutNode) == 32,
              "a node row is FCPW's BvhNode<3>: 32 bytes");
static_assert(offsetof(LayoutNode, low) == offsetof(FcpwNode, box) &&
                  offsetof(LayoutNode, high) == offsetof(FcpwNode, box) + 12 &&
                  offsetof(LayoutNode, offset) == offsetof(FcpwNode, secondChildOffset) &&
                  offsetof(LayoutNode, nReferences) == offsetof(FcpwNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");

// FCPW's tree in the layout: four buffers over FCPW's own memory, which the
// scene owns and keeps where it is for as long as the tree is used.
struct Tree {
    _tree_layout0 layout{};
    bonsai_buffer positions_buffer{}, objects_buffer{}, prims_buffer{}, nodes_buffer{};
    uint64_t node_count = 0, leaf_count = 0, prim_count = 0, max_leaf = 0, object_count = 0,
             position_count = 0;
    Tree() = default;
    Tree(const Tree &) = delete;
    Tree &operator=(const Tree &) = delete;
    Tree(Tree &&other) noexcept { *this = std::move(other); }
    Tree &operator=(Tree &&other) noexcept {
        layout = other.layout;
        positions_buffer = other.positions_buffer;
        objects_buffer = other.objects_buffer;
        prims_buffer = other.prims_buffer;
        nodes_buffer = other.nodes_buffer;
        node_count = other.node_count;
        leaf_count = other.leaf_count;
        prim_count = other.prim_count;
        max_leaf = other.max_leaf;
        object_count = other.object_count;
        position_count = other.position_count;
        point();
        return *this;
    }
    // The layout's pointers, at the buffers where they sit now.
    void point() {
        layout.group0_positions = &positions_buffer;
        layout.group1_objects = &objects_buffer;
        layout.group2_prims = &prims_buffer;
        layout.group3_index = &nodes_buffer;
    }
};

// A buffer over memory the program only reads: the layout's storage is
// read-only to it, and FCPW's is const here.
template <typename T>
inline bonsai_buffer wrap_const(const std::vector<T> &v) {
    return bonsai_buffer_wrap(const_cast<void *>(static_cast<const void *>(v.data())),
                              uint64_t(v.size()) * sizeof(T));
}

// FCPW's tree as the program reads it on the CPU: the scene's own arrays,
// wrapped. The references are checked to point into the one vector of
// Triangle objects, and the objects to index the one soup's positions, so
// that what the program follows is what the layout declares.
inline Tree wrap_fcpw_bvh(fcpw::Scene<3> &scene) {
    const FcpwBvh &bvh = scalar_bvh(scene);
    const std::vector<FcpwNode> &rows = Peek::tree(bvh);
    const std::vector<fcpw::Triangle *> &prims = scene_prims(scene);
    const auto &data = *scene.getSceneData();
    if (data.triangleObjects.size() != 1 || data.soups.size() != 1) {
        std::cerr << "FCPW's scene holds " << data.triangleObjects.size()
                  << " triangle object vectors and " << data.soups.size()
                  << " soups; the driver gives it one of each\n";
        std::exit(1);
    }
    const std::vector<fcpw::Triangle> &objects = *data.triangleObjects[0];
    const std::vector<fcpw::Vector3> &positions = data.soups[0].positions;
    check_leaves(rows, prims.size());
    for (const fcpw::Triangle *t : prims) {
        if (t < objects.data() || t >= objects.data() + objects.size()) {
            std::cerr << "a reference of FCPW's BVH points outside its triangle objects\n";
            std::exit(1);
        }
        if (t->soup != &data.soups[0]) {
            std::cerr << "a triangle of FCPW's BVH belongs to another soup\n";
            std::exit(1);
        }
        for (int k = 0; k < 3; k++) {
            if (t->indices[k] < 0 || size_t(t->indices[k]) >= positions.size()) {
                std::cerr << "a triangle of FCPW's BVH indexes past the soup's positions\n";
                std::exit(1);
            }
        }
    }
    Tree tree;
    tree.node_count = rows.size();
    tree.prim_count = prims.size();
    tree.object_count = objects.size();
    tree.position_count = positions.size();
    for (const FcpwNode &node : rows) {
        if (node.nReferences > 0) {
            tree.leaf_count++;
            tree.max_leaf = std::max<uint64_t>(tree.max_leaf, uint64_t(node.nReferences));
        }
    }
    tree.positions_buffer = wrap_const(positions);
    tree.objects_buffer = wrap_const(objects);
    tree.prims_buffer = wrap_const(prims);
    tree.nodes_buffer = wrap_const(rows);
    tree.layout.vCount = uint32_t(positions.size());
    tree.layout.oCount = uint32_t(objects.size());
    tree.layout.pCount = uint32_t(prims.size());
    tree.layout.nNodes = uint32_t(rows.size());
    tree.point();
    return tree;
}

inline void describe(const Tree &tree) {
    std::cout << "tree: FCPW's scalar BVH, its own memory wrapped: " << tree.node_count
              << " binary nodes (" << tree.node_count * sizeof(LayoutNode) << " bytes), "
              << tree.leaf_count << " of them leaves of at most " << tree.max_leaf
              << " triangles, " << tree.prim_count << " triangle references ("
              << tree.prim_count * sizeof(PrimRow) << " bytes) to " << tree.object_count
              << " Triangle objects (" << tree.object_count * sizeof(ObjectRow)
              << " bytes) over " << tree.position_count << " positions ("
              << tree.position_count * sizeof(PositionRow) << " bytes)\n";
}

#else // FCPW's GPU bytes

using TriangleRow = _tree_layout1;
using LayoutNode = _tree_layout2;

static_assert(sizeof(LayoutNode) == sizeof(FcpwNode), "a node row is FCPW's BvhNode");
static_assert(sizeof(LayoutNode) == 32, "FCPW's BvhNode<3> is 32 bytes");
static_assert(offsetof(LayoutNode, low) == offsetof(FcpwNode, box) &&
                  offsetof(LayoutNode, high) == offsetof(FcpwNode, box) + 12 &&
                  offsetof(LayoutNode, offset) == offsetof(FcpwNode, secondChildOffset) &&
                  offsetof(LayoutNode, nReferences) == offsetof(FcpwNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");
static_assert(sizeof(TriangleRow) == 40, "a triangle row is FCPW's GPUTriangle: 40 bytes");
static_assert(offsetof(TriangleRow, pa) == 0 && offsetof(TriangleRow, pb) == 12 &&
                  offsetof(TriangleRow, pc) == 24 && offsetof(TriangleRow, primitiveIndex) == 36,
              "the corners and the index sit where FCPW's GPUTriangle puts them");

// FCPW's tree in the layout: the rows copied byte for byte into an allocation
// of the same alignment as FCPW's vector's, the triangle rows made once.
struct Tree {
    _tree_layout0 layout{};
    void *nodes = nullptr;
    void *prims = nullptr;
    bonsai_buffer nodes_buffer{}, prims_buffer{};
    uint64_t node_count = 0, leaf_count = 0, prim_count = 0, max_leaf = 0;
    Tree() = default;
    Tree(const Tree &) = delete;
    Tree &operator=(const Tree &) = delete;
    Tree(Tree &&other) noexcept { *this = std::move(other); }
    Tree &operator=(Tree &&other) noexcept {
        layout = other.layout;
        nodes = std::exchange(other.nodes, nullptr);
        prims = std::exchange(other.prims, nullptr);
        nodes_buffer = other.nodes_buffer;
        prims_buffer = other.prims_buffer;
        node_count = other.node_count;
        leaf_count = other.leaf_count;
        prim_count = other.prim_count;
        max_leaf = other.max_leaf;
        layout.group1_index = &nodes_buffer;
        layout.group0_prims = &prims_buffer;
        return *this;
    }
    ~Tree() {
        std::free(nodes);
        std::free(prims);
    }
};

// FCPW's GPUTriangle row for one of its triangles: the corners out of the
// soup through the triangle's indices, then its pIndex -- extractTriangles'
// bytes. The row's fields are one-lane vectors (a tile of one: a corner is
// three one-float arrays x, y, z; the index a one-element array), written
// component by component, which is the same twelve bytes a float3 is.
inline void write_triangle_row(TriangleRow &row, const fcpw::Triangle &t) {
    const fcpw::Vector3 &pa = t.soup->positions[t.indices[0]];
    const fcpw::Vector3 &pb = t.soup->positions[t.indices[1]];
    const fcpw::Vector3 &pc = t.soup->positions[t.indices[2]];
    row.pa.x[0] = pa.x();
    row.pa.y[0] = pa.y();
    row.pa.z[0] = pa.z();
    row.pb.x[0] = pb.x();
    row.pb.y[0] = pb.y();
    row.pb.z[0] = pb.z();
    row.pc.x[0] = pc.x();
    row.pc.y[0] = pc.y();
    row.pc.z[0] = pc.z();
    row.primitiveIndex[0] = uint32_t(t.pIndex);
}

// The tree of a built FCPW scene, which has to be the scalar one.
inline Tree copy_fcpw_bvh(fcpw::Scene<3> &scene) {
    const FcpwBvh &bvh = scalar_bvh(scene);
    const std::vector<FcpwNode> &rows = Peek::tree(bvh);
    const std::vector<fcpw::Triangle *> &prims = scene_prims(scene);
    check_leaves(rows, prims.size());
    Tree tree;
    tree.node_count = rows.size();
    tree.prim_count = prims.size();
    for (const FcpwNode &node : rows) {
        if (node.nReferences > 0) {
            tree.leaf_count++;
            tree.max_leaf = std::max<uint64_t>(tree.max_leaf, uint64_t(node.nReferences));
        }
    }
    const size_t nodes_bytes = rows.size() * sizeof(LayoutNode);
    const size_t prims_bytes = prims.size() * sizeof(TriangleRow);
    tree.nodes = std::aligned_alloc(alignof(FcpwNode) > 16 ? alignof(FcpwNode) : 16,
                                    (nodes_bytes + 63) / 64 * 64);
    tree.prims = std::aligned_alloc(64, (prims_bytes + 63) / 64 * 64);
    if (tree.nodes == nullptr || tree.prims == nullptr) {
        std::cerr << "cannot allocate the tree's storage\n";
        std::exit(1);
    }
    std::memcpy(tree.nodes, rows.data(), nodes_bytes);
    TriangleRow *out = static_cast<TriangleRow *>(tree.prims);
    for (size_t k = 0; k < prims.size(); k++) {
        write_triangle_row(out[k], *prims[k]);
    }
    tree.nodes_buffer = bonsai_buffer_wrap(tree.nodes, nodes_bytes);
    tree.prims_buffer = bonsai_buffer_wrap(tree.prims, prims_bytes);
    tree.layout.nNodes = uint32_t(rows.size());
    tree.layout.group1_index = &tree.nodes_buffer;
    tree.layout.pCount = uint32_t(prims.size());
    tree.layout.group0_prims = &tree.prims_buffer;
    return tree;
}

inline void describe(const Tree &tree) {
    std::cout << "tree: FCPW's scalar BVH, " << tree.node_count << " binary nodes ("
              << tree.node_count * sizeof(LayoutNode) << " bytes), " << tree.leaf_count
              << " of them leaves of at most " << tree.max_leaf << " triangles, "
              << tree.prim_count << " triangles as FCPW's GPUTriangle rows ("
              << tree.prim_count * sizeof(TriangleRow) << " bytes)\n";
}

#endif // RTQ_FCPW_CPU_BYTES

} // namespace rtq_bvh
