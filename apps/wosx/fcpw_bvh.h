// FCPW's scalar BVH -- the tree WoSX builds by default, FCPW_USE_ENOKI off --
// copied into the layout that is its nodes' bytes, for the driver that
// compares the program against WoSX on WoSX's own default tree
// (wosx_hook.cpp built with WOSX_BINARY_BVH; apps/rtq/fcpw_tree.h is the
// same for the vectorized Mbvh).
//
// Included after the program's generated header and <fcpw/fcpw.h>: the
// layout's structs are the generated header's, as the compiler spells them
// for apps/wosx/schedules/layouts/fcpwb.bonsai -- `_tree_layout0` the
// layout, `_tree_layout1` a node row -- and FCPW's node is BvhNode<3>. The
// rows are the same bytes, which the copy relies on and the asserts check.
// The primitives are not: FCPW's scalar leaf reads a triangle's corners
// through its indices into the polygon soup, so the corners are copied out
// once into the layout's array of Triangle elements, in the order FCPW's
// build left its primitives (Bvh::primitives, which a leaf's referenceOffset
// indexes) and with the element's fields as apps/rtq's four-wide layout
// derives them from FCPW's packets (v0 = pa, e1 = pa - pb, e2 = pc - pa,
// primID = pIndex).
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace wosx_bvh {

using FcpwBvh = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle>;
using FcpwNode = fcpw::BvhNode<3>;
using LayoutNode = _tree_layout1;

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

static_assert(sizeof(LayoutNode) == sizeof(FcpwNode), "a node row is FCPW's BvhNode");
static_assert(sizeof(LayoutNode) == 32, "FCPW's BvhNode<3> is 32 bytes");
static_assert(offsetof(LayoutNode, low) == offsetof(FcpwNode, box) &&
                  offsetof(LayoutNode, high) == offsetof(FcpwNode, box) + 12 &&
                  offsetof(LayoutNode, offset) == offsetof(FcpwNode, secondChildOffset) &&
                  offsetof(LayoutNode, nReferences) == offsetof(FcpwNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");

// FCPW's tree in the layout: the rows copied byte for byte into an allocation
// of the same alignment as FCPW's vector's, the triangles made once.
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

// The tree of a built FCPW scene, which has to be the scalar one. (The scene
// is not const because FCPW's getSceneData is not.)
inline Tree copy_fcpw_bvh(fcpw::Scene<3> &scene) {
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
    for (const FcpwNode &node : rows) {
        if (node.nReferences > 0 &&
            uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size()) {
            std::cerr << "a leaf of FCPW's BVH references past its primitives\n";
            std::exit(1);
        }
    }
    tree.node_count = rows.size();
    tree.prim_count = prims.size();
    for (const FcpwNode &node : rows) {
        if (node.nReferences > 0) {
            tree.leaf_count++;
            tree.max_leaf = std::max<uint64_t>(tree.max_leaf, uint64_t(node.nReferences));
        }
    }
    const size_t nodes_bytes = rows.size() * sizeof(LayoutNode);
    const size_t prims_bytes = prims.size() * sizeof(Triangle);
    tree.nodes = std::aligned_alloc(alignof(FcpwNode) > 16 ? alignof(FcpwNode) : 16,
                                    (nodes_bytes + 63) / 64 * 64);
    tree.prims = std::aligned_alloc(64, (prims_bytes + 63) / 64 * 64);
    if (tree.nodes == nullptr || tree.prims == nullptr) {
        std::cerr << "cannot allocate the tree's storage\n";
        std::exit(1);
    }
    std::memcpy(tree.nodes, rows.data(), nodes_bytes);
    Triangle *out = static_cast<Triangle *>(tree.prims);
    for (size_t k = 0; k < prims.size(); k++) {
        const fcpw::Triangle &t = *prims[k];
        const fcpw::Vector3 &pa = t.soup->positions[t.indices[0]];
        const fcpw::Vector3 &pb = t.soup->positions[t.indices[1]];
        const fcpw::Vector3 &pc = t.soup->positions[t.indices[2]];
        out[k].v0 = float3{pa.x(), pa.y(), pa.z()};
        out[k].e1 = float3{pa.x() - pb.x(), pa.y() - pb.y(), pa.z() - pb.z()};
        out[k].e2 = float3{pc.x() - pa.x(), pc.y() - pa.y(), pc.z() - pa.z()};
        out[k].geomID = 0;
        out[k].primID = uint32_t(t.pIndex);
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
    std::cout << "tree: FCPW's scalar BVH, " << tree.node_count << " binary nodes ("
              << tree.node_count * sizeof(LayoutNode) << " bytes), " << tree.leaf_count
              << " of them leaves of at most " << tree.max_leaf << " triangles, "
              << tree.prim_count << " triangles copied out of the soup ("
              << tree.prim_count * sizeof(Triangle) << " bytes)\n";
}

} // namespace wosx_bvh
