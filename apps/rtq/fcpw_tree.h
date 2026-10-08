// FCPW's vectorized BVH, copied into the layout that is its bytes, for the
// drivers that compare a program against FCPW on FCPW's own tree: apps/rtq's
// (rtq_hook.cpp, built with RTQ_FCPW) and apps/wosx's (wosx_hook.cpp).
//
// Included after the program's generated header and <fcpw/fcpw.h>: the
// layout's structs are the generated header's, as the compiler spells them
// for apps/rtq/schedules/layouts/fcpw*.bonsai -- `_tree_layout0` the
// layout, `_tree_layout1` a packet, `_tree_layout2` a node row -- and FCPW's
// are Mbvh<FCPW_SIMD_WIDTH, 3, Triangle>'s at the width and branching the
// including file was compiled for (FCPW_SIMD_WIDTH, FCPW_USE_EIGHT_WIDE_
// BRANCHING, the flags FCPW's own CMake sets). The two are the same structs
// byte for byte, which the copy relies on, so the sizes and offsets are
// checked here against FCPW's.
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

namespace rtq {

// FCPW's Mbvh over triangles; its two arrays, flatTree and leafNodes, are
// protected members, read here through a derived class that names them --
// the one way to read them without editing FCPW.
using FcpwMbvh = fcpw::Mbvh<FCPW_SIMD_WIDTH, 3, fcpw::Triangle>;
struct FcpwPeek : FcpwMbvh {
    static const auto &tree(const FcpwMbvh &m) { return m.*(&FcpwPeek::flatTree); }
    static const auto &leaves(const FcpwMbvh &m) { return m.*(&FcpwPeek::leafNodes); }
};
using FcpwNode = fcpw::MbvhNode<3>;
using FcpwPacket = fcpw::MbvhLeafNode<FCPW_SIMD_WIDTH, 3>;

using FcpwNodeRow = _tree_layout2;
using FcpwLayoutPacket = _tree_layout1;
constexpr unsigned kFcpwWidth = sizeof(FcpwLayoutPacket::primitiveIndex) / sizeof(int32_t);
constexpr unsigned kFcpwBranching = sizeof(FcpwNodeRow::boxMin_x) / sizeof(float);
static_assert(kFcpwWidth == FCPW_SIMD_WIDTH,
              "the layout's packet width is FCPW's leaf width (FCPW_SIMD_WIDTH)");
static_assert(kFcpwBranching == FCPW_MBVH_BRANCHING_FACTOR,
              "the layout's node width is FCPW's branching factor");
static_assert(sizeof(FcpwNodeRow) == sizeof(FcpwNode), "a node row is FCPW's MbvhNode");
static_assert(offsetof(FcpwNodeRow, boxMin_x) == offsetof(FcpwNode, boxMin) &&
                  offsetof(FcpwNodeRow, boxMax_x) == offsetof(FcpwNode, boxMax) &&
                  offsetof(FcpwNodeRow, child0) == offsetof(FcpwNode, child),
              "the boxes and the child slots sit where FCPW's MbvhNode puts them");
static_assert(sizeof(FcpwLayoutPacket) == sizeof(FcpwPacket), "a packet is FCPW's MbvhLeafNode");
static_assert(offsetof(FcpwLayoutPacket, pa) == offsetof(FcpwPacket, positions) &&
                  offsetof(FcpwLayoutPacket, pb) ==
                      offsetof(FcpwPacket, positions) + sizeof(FcpwLayoutPacket::pa) &&
                  offsetof(FcpwLayoutPacket, pc) ==
                      offsetof(FcpwPacket, positions) + 2 * sizeof(FcpwLayoutPacket::pa) &&
                  offsetof(FcpwLayoutPacket, primitiveIndex) == offsetof(FcpwPacket, primitiveIndex),
              "the vertices and the ids sit where FCPW's MbvhLeafNode puts them");

// FCPW's tree in the layout: its two arrays copied byte for byte into two
// allocations of the same alignment as FCPW's own (std::vector's, on the
// heap), so that the two traversals read the same bytes from the same kind
// of memory.
struct FcpwTree {
    _tree_layout0 layout{};
    void *nodes = nullptr;
    void *packets = nullptr;
    bonsai_buffer nodes_buffer{}, packets_buffer{};
    uint64_t node_rows = 0, leaf_nodes = 0, packet_count = 0, leaf_prims = 0;
    FcpwTree() = default;
    FcpwTree(const FcpwTree &) = delete;
    FcpwTree &operator=(const FcpwTree &) = delete;
    FcpwTree(FcpwTree &&other) noexcept { *this = std::move(other); }
    FcpwTree &operator=(FcpwTree &&other) noexcept {
        layout = other.layout;
        nodes = std::exchange(other.nodes, nullptr);
        packets = std::exchange(other.packets, nullptr);
        nodes_buffer = other.nodes_buffer;
        packets_buffer = other.packets_buffer;
        node_rows = other.node_rows;
        leaf_nodes = other.leaf_nodes;
        packet_count = other.packet_count;
        leaf_prims = other.leaf_prims;
        layout.group0_packets = &packets_buffer;
        layout.group1_index = &nodes_buffer;
        return *this;
    }
    ~FcpwTree() {
        std::free(nodes);
        std::free(packets);
    }
};

// The tree of a built FCPW scene, which has to be the vectorized one the
// including file was compiled for. (The scene is not const because FCPW's
// getSceneData is not.)
inline FcpwTree copy_fcpw_tree(fcpw::Scene<3> &scene) {
    const auto *mbvh = dynamic_cast<const FcpwMbvh *>(scene.getSceneData()->aggregate.get());
    if (mbvh == nullptr) {
        std::cerr << "FCPW's scene did not build the vectorized BVH this driver was "
                     "compiled for (Mbvh<" << FCPW_SIMD_WIDTH << ", 3, Triangle>)\n";
        std::exit(1);
    }
    const auto &rows = FcpwPeek::tree(*mbvh);
    const auto &packets = FcpwPeek::leaves(*mbvh);
    FcpwTree tree;
    tree.node_rows = rows.size();
    tree.packet_count = packets.size();
    for (const FcpwNode &node : rows) {
        if (node.child[0] < 0) {
            tree.leaf_nodes++;
            tree.leaf_prims += uint64_t(node.child[3]);
        }
    }
    const size_t nodes_bytes = rows.size() * sizeof(FcpwNodeRow);
    const size_t packets_bytes = packets.size() * sizeof(FcpwLayoutPacket);
    // Aligned as FCPW's vectors are: a packet to its vector's alignment, a
    // row to its own.
    tree.nodes = std::aligned_alloc(alignof(FcpwNode), (nodes_bytes + 63) / 64 * 64);
    tree.packets = std::aligned_alloc(alignof(FcpwPacket), (packets_bytes + 63) / 64 * 64);
    if (tree.nodes == nullptr || tree.packets == nullptr) {
        std::cerr << "cannot allocate the tree's storage\n";
        std::exit(1);
    }
    std::memcpy(tree.nodes, rows.data(), nodes_bytes);
    std::memcpy(tree.packets, packets.data(), packets_bytes);
    tree.nodes_buffer = bonsai_buffer_wrap(tree.nodes, nodes_bytes);
    tree.packets_buffer = bonsai_buffer_wrap(tree.packets, packets_bytes);
    tree.layout.nNodes = uint32_t(rows.size());
    tree.layout.group1_index = &tree.nodes_buffer;
    tree.layout.nLeafs = uint32_t(packets.size());
    tree.layout.group0_packets = &tree.packets_buffer;
    return tree;
}

inline void describe(const FcpwTree &tree) {
    std::cout << "tree: " << tree.node_rows << " node rows of " << kFcpwBranching << " ("
              << tree.node_rows * sizeof(FcpwNodeRow) << " bytes), " << tree.leaf_nodes
              << " of them leaves holding " << tree.leaf_prims << " triangles in "
              << tree.packet_count << " packets of " << kFcpwWidth << " ("
              << tree.packet_count * sizeof(FcpwLayoutPacket) << " bytes)\n";
}

} // namespace rtq
