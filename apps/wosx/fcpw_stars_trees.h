// The walk on stars' trees -- FCPW's scalar BVH over the absorbing boundary
// and its normal cone hierarchy over the reflecting boundary, the trees
// WoSX builds by default (FCPW_USE_ENOKI off) and the ones FCPW puts on a
// GPU -- copied into the layouts that are their bytes
// (apps/wosx/schedules/layouts/fcpw_stars.bonsai), for the driver that
// compares wost.bonsai against WoSX's WalkOnStars (wosx_hook.cpp built with
// WOSX_STARS; compare.sh does so for the schedules named wost-*).
//
// Included after the program's generated header and WoSX's: the layout
// structs are the generated header's -- `_tree_layout0` the absorbing
// layout with `_tree_layout1` a triangle row and `_tree_layout2` a node
// row, as apps/rtq/fcpw_bvh.h has them for the walk on spheres;
// `_tree_layout5` the reflecting triangles' layout (`_tree_layout6` a
// triangle row, `_tree_layout7` a node row) and `_tree_layout10` the
// silhouette edges' (`_tree_layout11` an edge row, `_tree_layout12` the
// same node row). The absorbing tree's node rows are FCPW's own BvhNode<3>,
// copied byte for byte as fcpw_bvh.h copies them; the reflecting tree's
// rows are the bytes FCPW's GPU transfer makes of its tree, written by
// apps/rtq/fcpw_snch.h as it writes them for apps/rtq's own comparison on
// that tree, into one allocation of node rows the two reflecting layouts
// both point at: FCPW's one tree holds the triangles and the silhouette
// edges, each leaf referencing its range of both, and the program reads it
// as two trees.
#pragma once

#include <fcpw/fcpw.h>

#include "apps/rtq/fcpw_snch.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace wosx_stars {

using AbsorbingBvh = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle, fcpw::SilhouettePrimitive<3>>;
using AbsorbingNode = fcpw::BvhNode<3>;

using AbsorbingTriangleRow = _tree_layout1;
using AbsorbingNodeRow = _tree_layout2;
using ReflectingTriangleRow = _tree_layout6;
using ReflectingNodeRow = _tree_layout7;
using EdgeRow = _tree_layout11;
using SilhouetteNodeRow = _tree_layout12;
using rtq_snch::Rows;

// The absorbing tree's rows are a protected member of FCPW's Bvh, read
// through a derived class that names them (as fcpw_bvh.h reads the walk on
// spheres' tree; fcpw_snch.h's Peek reads the reflecting tree's).
struct PeekAbsorbing : AbsorbingBvh {
    static const std::vector<AbsorbingNode> &tree(const AbsorbingBvh &b) {
        return b.*(&PeekAbsorbing::flatTree);
    }
};

static_assert(sizeof(AbsorbingNodeRow) == sizeof(AbsorbingNode) && sizeof(AbsorbingNodeRow) == 32,
              "an absorbing node row is FCPW's BvhNode<3>: 32 bytes");
static_assert(offsetof(AbsorbingNodeRow, low) == offsetof(AbsorbingNode, box) &&
                  offsetof(AbsorbingNodeRow, high) == offsetof(AbsorbingNode, box) + 12 &&
                  offsetof(AbsorbingNodeRow, offset) == offsetof(AbsorbingNode, secondChildOffset) &&
                  offsetof(AbsorbingNodeRow, nReferences) == offsetof(AbsorbingNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");
static_assert(sizeof(AbsorbingTriangleRow) == 40, "a triangle row is FCPW's GPUTriangle: 40 bytes");
static_assert(offsetof(AbsorbingTriangleRow, pa) == 0 && offsetof(AbsorbingTriangleRow, pb) == 12 &&
                  offsetof(AbsorbingTriangleRow, pc) == 24 &&
                  offsetof(AbsorbingTriangleRow, primitiveIndex) == 36,
              "the corners and the index sit where FCPW's GPUTriangle puts them");
static_assert(rtq_snch::rows_are_fcpws<ReflectingNodeRow, ReflectingTriangleRow, EdgeRow>());
static_assert(sizeof(SilhouetteNodeRow) == sizeof(ReflectingNodeRow) &&
                  offsetof(SilhouetteNodeRow, silhouetteOffset) ==
                      offsetof(ReflectingNodeRow, silhouetteOffset),
              "the two reflecting layouts read the same node row");

struct Trees {
    _tree_layout0 absorbing{};
    _tree_layout5 reflecting{};
    _tree_layout10 silhouettes{};
    Rows absorbing_nodes, absorbing_triangles, reflecting_nodes, reflecting_triangles, edges;
    uint64_t absorbing_node_count = 0, absorbing_triangle_count = 0;
    rtq_snch::Counts reflecting_counts;
    Trees() = default;
    Trees(const Trees &) = delete;
    Trees &operator=(const Trees &) = delete;
    Trees(Trees &&) = delete;
    Trees &operator=(Trees &&) = delete;
    // The layouts' pointers, set once the rows sit where they will stay.
    void point() {
        absorbing.aCount = uint32_t(absorbing_triangle_count);
        absorbing.group0_aprims = &absorbing_triangles.buffer;
        absorbing.aNodes = uint32_t(absorbing_node_count);
        absorbing.group1_anode = &absorbing_nodes.buffer;
        reflecting.rCount = uint32_t(reflecting_counts.triangles);
        reflecting.group0_rprims = &reflecting_triangles.buffer;
        reflecting.rNodes = uint32_t(reflecting_counts.nodes);
        reflecting.group1_rnode = &reflecting_nodes.buffer;
        silhouettes.eCount = uint32_t(reflecting_counts.edges);
        silhouettes.group0_edges = &edges.buffer;
        silhouettes.sNodes = uint32_t(reflecting_counts.nodes);
        silhouettes.group1_snode = &reflecting_nodes.buffer;
    }
};

// The absorbing boundary's tree, out of a built FCPW scene that holds the
// scalar BVH over triangles (the Dirichlet handler's). (The scene is not
// const because FCPW's getSceneData is not.)
inline void copy_absorbing(fcpw::Scene<3> &scene, Trees &trees) {
    const auto *bvh = dynamic_cast<const AbsorbingBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's absorbing scene did not build the scalar BVH this driver was "
                     "compiled for (Bvh<3, BvhNode<3>, Triangle>)\n";
        std::exit(1);
    }
    const std::vector<AbsorbingNode> &nodes = PeekAbsorbing::tree(*bvh);
    const std::vector<fcpw::Triangle *> &prims = rtq_snch::scene_triangles(scene, "absorbing");
    for (const AbsorbingNode &node : nodes) {
        if (node.nReferences > 0 &&
            uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size()) {
            std::cerr << "a leaf of FCPW's absorbing BVH references past its primitives\n";
            std::exit(1);
        }
    }
    trees.absorbing_node_count = nodes.size();
    trees.absorbing_triangle_count = prims.size();
    AbsorbingNodeRow *node_rows = trees.absorbing_nodes.make<AbsorbingNodeRow>(nodes.size());
    std::memcpy(node_rows, nodes.data(), nodes.size() * sizeof(AbsorbingNode));
    AbsorbingTriangleRow *triangle_rows =
        trees.absorbing_triangles.make<AbsorbingTriangleRow>(prims.size());
    for (size_t i = 0; i < prims.size(); i++) {
        rtq_snch::write_triangle_row(triangle_rows[i], *prims[i]);
    }
}

// The reflecting boundary's tree, out of a built FCPW scene that holds the
// normal cone hierarchy over triangles with silhouette edges (the Neumann
// handler's, built after computeSilhouettes): apps/rtq/fcpw_snch.h's copy,
// into the rows both reflecting layouts read.
inline void copy_reflecting(fcpw::Scene<3> &scene, Trees &trees) {
    trees.reflecting_counts =
        rtq_snch::copy_snch<ReflectingNodeRow, ReflectingTriangleRow, EdgeRow>(
            scene, trees.reflecting_nodes, trees.reflecting_triangles, trees.edges, "reflecting");
}

inline void describe(const Trees &trees) {
    std::cout << "trees: the absorbing boundary FCPW's scalar BVH, " << trees.absorbing_node_count
              << " binary nodes (" << trees.absorbing_node_count * sizeof(AbsorbingNodeRow)
              << " bytes) over " << trees.absorbing_triangle_count
              << " triangles as FCPW's GPUTriangle rows; the reflecting boundary FCPW's normal "
                 "cone hierarchy, "
              << trees.reflecting_counts.nodes << " nodes with cones ("
              << trees.reflecting_counts.nodes * sizeof(ReflectingNodeRow) << " bytes) over "
              << trees.reflecting_counts.triangles << " triangles and "
              << trees.reflecting_counts.edges << " silhouette edges as FCPW's GPUEdge rows ("
              << trees.reflecting_counts.edges * sizeof(EdgeRow) << " bytes)\n";
}

} // namespace wosx_stars
