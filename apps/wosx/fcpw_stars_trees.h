// The walk on stars' trees -- FCPW's scalar BVH over the absorbing boundary
// and its normal cone hierarchy over the reflecting boundary, the trees
// WoSX builds by default (FCPW_USE_ENOKI off) and the ones FCPW puts on a
// GPU -- in the layouts that are their bytes, for the driver that compares
// wost.bonsai against WoSX's WalkOnStars (wosx_hook.cpp built with
// WOSX_STARS; compare.sh does so for the schedules named wost-*).
//
// Two layouts, two sections, chosen by WOSX_GPU as apps/rtq/fcpw_bvh.h's
// are by RTQ_FCPW_CPU_BYTES:
//
// - Without it, the CPU comparison: the bytes FCPW's CPU reads
//   (schedules/layouts/fcpw_stars.bonsai), nothing copied. FCPW's own node
//   arrays, its primitive and silhouette pointer arrays, its Triangle and
//   SilhouetteEdge objects and the two soups' positions, each wrapped as a
//   buffer where it lies (the absorbing tree here, the reflecting one by
//   apps/rtq/fcpw_snch.h's wrap_snch), so that a leaf reads a triangle or
//   an edge as FCPW's leaf does: the pointer, the indices through it, the
//   corners gathered by index. The scenes have to outlive the trees.
//
// - With it, the GPU comparison: the rows FCPW's own GPU transfer makes of
//   its trees (schedules/layouts/fcpw_stars_gpu.bonsai; fcpw/gpu/
//   bvh_interop_structures.h), the absorbing tree's node rows FCPW's own
//   BvhNode<3> copied byte for byte and its triangles written as GPUTriangle
//   rows, the reflecting tree's rows written by apps/rtq/fcpw_snch.h as it
//   writes them for apps/rtq's own comparison, into one allocation of node
//   rows the two reflecting layouts both point at: FCPW's one tree holds
//   the triangles and the silhouette edges, each leaf referencing its range
//   of both, and the program reads it as two trees.
//
// Included after the program's generated header and WoSX's: the layout
// structs are the generated header's, as the compiler spells them for the
// layout in use. For fcpw_stars.bonsai: `_tree_layout0` the absorbing
// layout (`_tree_layout1` a position, `_tree_layout2` a Triangle object,
// `_tree_layout3` a reference, `_tree_layout4` a node row), `_tree_layout7`
// the reflecting triangles' (`_tree_layout8` a reference, `_tree_layout9` a
// node row, the position and object rows the same structs as the
// absorbing tree's) and `_tree_layout12` the silhouette edges' (`_tree_
// layout13` a SilhouetteEdge object, `_tree_layout14` a reference, `_tree_
// layout15` the same node row). For fcpw_stars_gpu.bonsai: `_tree_layout0`
// the absorbing layout (`_tree_layout1` a triangle row, `_tree_layout2` a
// node row), `_tree_layout5` the reflecting triangles' (`_tree_layout6`,
// `_tree_layout7`) and `_tree_layout10` the silhouette edges'
// (`_tree_layout11` an edge row, `_tree_layout12` the same node row).
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

// The absorbing tree's rows are a protected member of FCPW's Bvh, read
// through a derived class that names them (as apps/rtq/fcpw_bvh.h reads the
// walk on spheres' tree; fcpw_snch.h's Peek reads the reflecting tree's).
struct PeekAbsorbing : AbsorbingBvh {
    static const std::vector<AbsorbingNode> &tree(const AbsorbingBvh &b) {
        return b.*(&PeekAbsorbing::flatTree);
    }
};

// The absorbing boundary's tree, out of a built FCPW scene that holds the
// scalar BVH over triangles (the Dirichlet handler's), or an exit. (The
// scene is not const because FCPW's getSceneData is not.)
inline const AbsorbingBvh &absorbing_bvh(fcpw::Scene<3> &scene) {
    const auto *bvh = dynamic_cast<const AbsorbingBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's absorbing scene did not build the scalar BVH this driver was "
                     "compiled for (Bvh<3, BvhNode<3>, Triangle>)\n";
        std::exit(1);
    }
    return *bvh;
}

inline void check_absorbing_leaves(const std::vector<AbsorbingNode> &nodes, size_t prim_count) {
    for (const AbsorbingNode &node : nodes) {
        if (node.nReferences > 0 &&
            uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prim_count) {
            std::cerr << "a leaf of FCPW's absorbing BVH references past its primitives\n";
            std::exit(1);
        }
    }
}

#if !defined(WOSX_GPU)

using AbsorbingPositionRow = _tree_layout1;
using AbsorbingObjectRow = _tree_layout2;
using AbsorbingPrimRow = _tree_layout3;
using AbsorbingNodeRow = _tree_layout4;
using ReflectingPrimRow = _tree_layout8;
using ReflectingNodeRow = _tree_layout9;
using EdgeObjectRow = _tree_layout13;
using EdgeRefRow = _tree_layout14;
using SilhouetteNodeRow = _tree_layout15;

static_assert(sizeof(AbsorbingPositionRow) == sizeof(fcpw::Vector3) && sizeof(AbsorbingPositionRow) == 12,
              "a position row is FCPW's Vector3: three floats");
static_assert(sizeof(AbsorbingObjectRow) == sizeof(fcpw::Triangle) && sizeof(AbsorbingObjectRow) == 32,
              "an object row is FCPW's Triangle: 32 bytes");
static_assert(offsetof(AbsorbingObjectRow, i0) == offsetof(fcpw::Triangle, indices) &&
                  offsetof(AbsorbingObjectRow, pIndex) == offsetof(fcpw::Triangle, pIndex) &&
                  offsetof(AbsorbingObjectRow, soup) == offsetof(fcpw::Triangle, soup),
              "the indices, the index and the soup sit where FCPW's Triangle puts them");
static_assert(sizeof(AbsorbingPrimRow) == sizeof(fcpw::Triangle *), "a reference row is one pointer");
static_assert(sizeof(AbsorbingNodeRow) == sizeof(AbsorbingNode) && sizeof(AbsorbingNodeRow) == 32,
              "an absorbing node row is FCPW's BvhNode<3>: 32 bytes");
static_assert(offsetof(AbsorbingNodeRow, low) == offsetof(AbsorbingNode, box) &&
                  offsetof(AbsorbingNodeRow, high) == offsetof(AbsorbingNode, box) + 12 &&
                  offsetof(AbsorbingNodeRow, offset) == offsetof(AbsorbingNode, secondChildOffset) &&
                  offsetof(AbsorbingNodeRow, nReferences) == offsetof(AbsorbingNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");
static_assert(rtq_snch::rows_are_fcpws_cpu<ReflectingNodeRow, AbsorbingObjectRow, ReflectingPrimRow,
                                           EdgeObjectRow, EdgeRefRow, AbsorbingPositionRow>());
static_assert(sizeof(SilhouetteNodeRow) == sizeof(ReflectingNodeRow) &&
                  offsetof(SilhouetteNodeRow, silhouetteOffset) ==
                      offsetof(ReflectingNodeRow, silhouetteOffset) &&
                  offsetof(SilhouetteNodeRow, nSilhouettes) == offsetof(ReflectingNodeRow, nSilhouettes),
              "the two reflecting layouts read the same node row");

// The three layouts over FCPW's own memory: the absorbing tree's four
// buffers, and the reflecting tree's six (rtq_snch::Wrapped), which its two
// layouts share. The scenes keep the memory where it is.
struct Trees {
    _tree_layout0 absorbing{};
    _tree_layout7 reflecting{};
    _tree_layout12 silhouettes{};
    bonsai_buffer absorbing_positions{}, absorbing_objects{}, absorbing_prims{}, absorbing_nodes{};
    uint64_t absorbing_node_count = 0, absorbing_triangle_count = 0, absorbing_object_count = 0,
             absorbing_position_count = 0;
    rtq_snch::Wrapped reflecting_arrays;
    Trees() = default;
    Trees(const Trees &) = delete;
    Trees &operator=(const Trees &) = delete;
    Trees(Trees &&) = delete;
    Trees &operator=(Trees &&) = delete;
    // The layouts' pointers, at the buffers where they sit.
    void point() {
        absorbing.avCount = uint32_t(absorbing_position_count);
        absorbing.group0_apositions = &absorbing_positions;
        absorbing.aoCount = uint32_t(absorbing_object_count);
        absorbing.group1_aobjects = &absorbing_objects;
        absorbing.aCount = uint32_t(absorbing_triangle_count);
        absorbing.group2_aprims = &absorbing_prims;
        absorbing.aNodes = uint32_t(absorbing_node_count);
        absorbing.group3_anode = &absorbing_nodes;
        const rtq_snch::Wrapped &w = reflecting_arrays;
        reflecting.rvCount = uint32_t(w.position_count);
        reflecting.group0_rpositions = &reflecting_arrays.positions;
        reflecting.roCount = uint32_t(w.object_count);
        reflecting.group1_robjects = &reflecting_arrays.objects;
        reflecting.rCount = uint32_t(w.counts.triangles);
        reflecting.group2_rprims = &reflecting_arrays.prims;
        reflecting.rNodes = uint32_t(w.counts.nodes);
        reflecting.group3_rnode = &reflecting_arrays.nodes;
        silhouettes.evCount = uint32_t(w.position_count);
        silhouettes.group0_epositions = &reflecting_arrays.positions;
        silhouettes.eoCount = uint32_t(w.edge_object_count);
        silhouettes.group1_edge_objects = &reflecting_arrays.edge_objects;
        silhouettes.eCount = uint32_t(w.counts.edges);
        silhouettes.group2_edges = &reflecting_arrays.edge_refs;
        silhouettes.sNodes = uint32_t(w.counts.nodes);
        silhouettes.group3_snode = &reflecting_arrays.nodes;
    }
    // The twelve tree buffers in the order the generated header takes the
    // sets and their groups: the silhouette edges' layout (apps/rtq/
    // silhouette.bonsai declares the set, imported before the walk's own
    // two), then the absorbing and the reflecting triangles'.
    std::vector<bonsai_buffer *> buffers() {
        return {&reflecting_arrays.positions, &reflecting_arrays.edge_objects,
                &reflecting_arrays.edge_refs, &reflecting_arrays.nodes,
                &absorbing_positions,         &absorbing_objects,
                &absorbing_prims,             &absorbing_nodes,
                &reflecting_arrays.positions, &reflecting_arrays.objects,
                &reflecting_arrays.prims,     &reflecting_arrays.nodes};
    }
};

// The absorbing boundary's tree as the program reads it on the CPU: the
// scene's own arrays, wrapped and checked as rtq_bvh::wrap_fcpw_bvh checks
// the walk on spheres' (every reference into the one vector of Triangle
// objects, every object indexing the one soup's positions).
inline void wrap_absorbing(fcpw::Scene<3> &scene, Trees &trees) {
    const AbsorbingBvh &bvh = absorbing_bvh(scene);
    const std::vector<AbsorbingNode> &nodes = PeekAbsorbing::tree(bvh);
    const std::vector<fcpw::Triangle *> &prims = rtq_snch::scene_triangles(scene, "absorbing");
    const auto &data = *scene.getSceneData();
    if (data.triangleObjects.size() != 1 || data.soups.size() != 1) {
        std::cerr << "FCPW's absorbing scene holds " << data.triangleObjects.size()
                  << " triangle object vectors and " << data.soups.size()
                  << " soups; the driver gives it one of each\n";
        std::exit(1);
    }
    const std::vector<fcpw::Triangle> &objects = *data.triangleObjects[0];
    const std::vector<fcpw::Vector3> &positions = data.soups[0].positions;
    check_absorbing_leaves(nodes, prims.size());
    for (const fcpw::Triangle *t : prims) {
        if (t < objects.data() || t >= objects.data() + objects.size() || t->soup != &data.soups[0]) {
            std::cerr << "a reference of FCPW's absorbing BVH points outside its triangle objects, "
                         "or to another soup\n";
            std::exit(1);
        }
        for (int k = 0; k < 3; k++) {
            if (t->indices[k] < 0 || size_t(t->indices[k]) >= positions.size()) {
                std::cerr << "a triangle of FCPW's absorbing BVH indexes past the soup's positions\n";
                std::exit(1);
            }
        }
    }
    trees.absorbing_node_count = nodes.size();
    trees.absorbing_triangle_count = prims.size();
    trees.absorbing_object_count = objects.size();
    trees.absorbing_position_count = positions.size();
    trees.absorbing_positions = rtq_snch::wrap_const(positions);
    trees.absorbing_objects = rtq_snch::wrap_const(objects);
    trees.absorbing_prims = rtq_snch::wrap_const(prims);
    trees.absorbing_nodes = rtq_snch::wrap_const(nodes);
}

// The reflecting boundary's tree (the Neumann handler's, built after
// computeSilhouettes): FCPW's arrays wrapped by apps/rtq/fcpw_snch.h.
inline void wrap_reflecting(fcpw::Scene<3> &scene, Trees &trees) {
    trees.reflecting_arrays = rtq_snch::wrap_snch(scene, "reflecting");
}

inline void describe(const Trees &trees) {
    const rtq_snch::Wrapped &w = trees.reflecting_arrays;
    std::cout << "trees: FCPW's own memory wrapped; the absorbing boundary its scalar BVH, "
              << trees.absorbing_node_count << " binary nodes ("
              << trees.absorbing_node_count * sizeof(AbsorbingNodeRow) << " bytes) over "
              << trees.absorbing_triangle_count << " triangle references to "
              << trees.absorbing_object_count << " Triangle objects over "
              << trees.absorbing_position_count
              << " positions; the reflecting boundary its normal cone hierarchy, " << w.counts.nodes
              << " nodes with cones (" << w.counts.nodes * sizeof(ReflectingNodeRow) << " bytes) over "
              << w.counts.triangles << " triangle references to " << w.object_count
              << " Triangle objects and " << w.counts.edges << " silhouette edge references to "
              << w.edge_object_count << " SilhouetteEdge objects over " << w.position_count
              << " positions\n";
}

#else // WOSX_GPU: FCPW's GPU rows

using AbsorbingTriangleRow = _tree_layout1;
using AbsorbingNodeRow = _tree_layout2;
using ReflectingTriangleRow = _tree_layout6;
using ReflectingNodeRow = _tree_layout7;
using EdgeRow = _tree_layout11;
using SilhouetteNodeRow = _tree_layout12;
using rtq_snch::Rows;

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
    // The six tree buffers in the order the generated header takes the sets
    // and their groups: the silhouette edges' layout (apps/rtq/silhouette.
    // bonsai declares the set, imported before the walk's own two), then
    // the absorbing and the reflecting triangles'.
    std::vector<bonsai_buffer *> buffers() {
        return {&edges.buffer,               &reflecting_nodes.buffer,     &absorbing_triangles.buffer,
                &absorbing_nodes.buffer,     &reflecting_triangles.buffer, &reflecting_nodes.buffer};
    }
};

// The absorbing boundary's tree, out of a built FCPW scene that holds the
// scalar BVH over triangles (the Dirichlet handler's), copied into FCPW's
// GPU rows.
inline void copy_absorbing(fcpw::Scene<3> &scene, Trees &trees) {
    const AbsorbingBvh &bvh = absorbing_bvh(scene);
    const std::vector<AbsorbingNode> &nodes = PeekAbsorbing::tree(bvh);
    const std::vector<fcpw::Triangle *> &prims = rtq_snch::scene_triangles(scene, "absorbing");
    check_absorbing_leaves(nodes, prims.size());
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

#endif // WOSX_GPU

} // namespace wosx_stars
