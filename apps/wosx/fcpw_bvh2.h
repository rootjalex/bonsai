// FCPW's binary BVH -- the tree WoSX walks on the GPU, and the tree its CPU
// solver walks as shipped -- copied into the layout of
// apps/wosx/schedules/layouts/fcpwb2.bonsai: each interior node holding its
// two children's boxes and rows, the triangles as the program's elements.
// For the driver (wosx_hook.cpp built with WOSX_BINARY_BVH2; compare.sh
// does so for the schedules named fcpwb2*). fcpw_bvh.h is the same for the
// byte-for-byte node layout, apps/rtq/fcpw_tree.h for the vectorized Mbvh.
//
// Included after the program's generated header and <fcpw/fcpw.h>: the
// layout's structs are the generated header's, as the compiler spells them
// for that layout -- `_tree_layout0` the layout, `_tree_layout1` a node row
// -- and FCPW's node is BvhNode<3>. The node rows are not FCPW's bytes (the
// layout says why): they are built here from FCPW's, the children's boxes
// read out of the children's rows once, at copy time, where FCPW's
// traversal reads them at every visit. The triangles are copied out of
// FCPW's polygon soup into the layout's array of Triangle elements as
// fcpw_bvh.h copies them: the order FCPW's build left its primitives in
// (Bvh::primitives, which a leaf's offset indexes), the fields as
// apps/rtq's four-wide layout derives them from FCPW's packets (v0 = pa,
// e1 = pa - pb, e2 = pc - pa, primID = pIndex).
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace wosx_bvh2 {

using FcpwBvh = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle>;
using FcpwNode = fcpw::BvhNode<3>;
using NodeRow = _tree_layout1;

// The tree's rows are a protected member of FCPW's Bvh, read through a
// derived class that names them (as fcpw_bvh.h and apps/rtq/fcpw_tree.h
// read theirs).
struct Peek : FcpwBvh {
    static const std::vector<FcpwNode> &tree(const FcpwBvh &b) {
        return b.*(&Peek::flatTree);
    }
};

static_assert(sizeof(NodeRow) == 56, "a node row is two boxes by component and two child slots");
static_assert(offsetof(NodeRow, boxMin_x) == 0 && offsetof(NodeRow, boxMax_x) == 24 &&
                  offsetof(NodeRow, child0) == 48 && offsetof(NodeRow, child1) == 52,
              "the boxes then the children, as the layout declares them");

// FCPW's tree in the layout: the node rows built from FCPW's, the triangle
// rows made once, both in allocations the kernel copies to the device
// whole (bonsai_buffer).
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
        layout.group0_index = &nodes_buffer;
        layout.prims = &prims_buffer;
        return *this;
    }
    ~Tree() {
        std::free(nodes);
        std::free(prims);
    }
};

// The tree of a built FCPW scene, which has to be the scalar binary one --
// what WoSX's GPU boundary handler builds (buildAccelerationStructure with
// vectorization off, the only tree FCPW transfers to the GPU). (The scene is
// not const because FCPW's getSceneData is not.)
inline Tree copy_fcpw_bvh2(fcpw::Scene<3> &scene) {
    const auto *bvh = dynamic_cast<const FcpwBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's scene did not build the scalar BVH this driver was compiled "
                     "for (Bvh<3, BvhNode<3>, Triangle>)\n";
        std::exit(1);
    }
    const std::vector<FcpwNode> &rows = Peek::tree(*bvh);
    const auto &objects = scene.getSceneData()->triangleObjectPtrs;
    if (objects.size() != 1) {
        std::cerr << "FCPW's scene holds " << objects.size()
                  << " triangle objects; the driver gives it one\n";
        std::exit(1);
    }
    const std::vector<fcpw::Triangle *> &prims = objects[0];
    Tree tree;
    tree.node_count = rows.size();
    tree.prim_count = prims.size();
    for (size_t i = 0; i < rows.size(); i++) {
        const FcpwNode &node = rows[i];
        if (node.nReferences > 0) {
            tree.leaf_count++;
            tree.max_leaf = std::max<uint64_t>(tree.max_leaf, uint64_t(node.nReferences));
            if (uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size()) {
                std::cerr << "a leaf of FCPW's BVH references past its primitives\n";
                std::exit(1);
            }
        } else if (i + 1 >= rows.size() || i + size_t(node.secondChildOffset) >= rows.size()) {
            std::cerr << "an interior node of FCPW's BVH names a child past its rows\n";
            std::exit(1);
        }
    }
    const size_t nodes_bytes = rows.size() * sizeof(NodeRow);
    const size_t prims_bytes = prims.size() * sizeof(Triangle);
    tree.nodes = std::aligned_alloc(64, (nodes_bytes + 63) / 64 * 64);
    tree.prims = std::aligned_alloc(64, (prims_bytes + 63) / 64 * 64);
    if (tree.nodes == nullptr || tree.prims == nullptr) {
        std::cerr << "cannot allocate the tree's storage\n";
        std::exit(1);
    }
    std::memset(tree.nodes, 0, nodes_bytes);
    NodeRow *out = static_cast<NodeRow *>(tree.nodes);
    for (size_t i = 0; i < rows.size(); i++) {
        const FcpwNode &node = rows[i];
        NodeRow &row = out[i];
        if (node.nReferences > 0) {
            // A leaf: FCPW's Mbvh convention for a leaf row, the first slot
            // negative, -(first + 1), the second the count.
            row.child0 = -(node.referenceOffset + 1);
            row.child1 = node.nReferences;
            continue;
        }
        const int32_t left = int32_t(i) + 1;
        const int32_t right = int32_t(i) + node.secondChildOffset;
        const FcpwNode *children[2] = {&rows[size_t(left)], &rows[size_t(right)]};
        for (int c = 0; c < 2; c++) {
            const auto &box = children[c]->box;
            row.boxMin_x[c] = box.pMin.x();
            row.boxMin_y[c] = box.pMin.y();
            row.boxMin_z[c] = box.pMin.z();
            row.boxMax_x[c] = box.pMax.x();
            row.boxMax_y[c] = box.pMax.y();
            row.boxMax_z[c] = box.pMax.z();
        }
        row.child0 = left;
        row.child1 = right;
    }
    Triangle *out_tri = static_cast<Triangle *>(tree.prims);
    for (size_t k = 0; k < prims.size(); k++) {
        const fcpw::Triangle &t = *prims[k];
        const fcpw::Vector3 &pa = t.soup->positions[t.indices[0]];
        const fcpw::Vector3 &pb = t.soup->positions[t.indices[1]];
        const fcpw::Vector3 &pc = t.soup->positions[t.indices[2]];
        out_tri[k].v0 = float3{pa.x(), pa.y(), pa.z()};
        out_tri[k].e1 = float3{pa.x() - pb.x(), pa.y() - pb.y(), pa.z() - pb.z()};
        out_tri[k].e2 = float3{pc.x() - pa.x(), pc.y() - pa.y(), pc.z() - pa.z()};
        out_tri[k].geomID = 0;
        out_tri[k].primID = uint32_t(t.pIndex);
    }
    tree.nodes_buffer = bonsai_buffer_wrap(tree.nodes, nodes_bytes);
    tree.prims_buffer = bonsai_buffer_wrap(tree.prims, prims_bytes);
    tree.layout.nNodes = uint32_t(rows.size());
    tree.layout.group0_index = &tree.nodes_buffer;
    tree.layout.pCount = uint32_t(prims.size());
    tree.layout.prims = &tree.prims_buffer;
    return tree;
}

inline void describe(const Tree &tree) {
    std::cout << "tree: FCPW's binary BVH, " << tree.node_count
              << " nodes each holding its children's boxes (" << tree.node_count * sizeof(NodeRow)
              << " bytes), " << tree.leaf_count << " of them leaves of at most " << tree.max_leaf
              << " triangles, " << tree.prim_count << " triangles copied out of the soup ("
              << tree.prim_count * sizeof(Triangle) << " bytes)\n";
}

} // namespace wosx_bvh2
