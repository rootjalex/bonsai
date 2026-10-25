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
// row, as apps/wosx/fcpw_bvh.h has them for the walk on spheres;
// `_tree_layout5` the reflecting triangles' layout (`_tree_layout6` a
// triangle row, `_tree_layout7` a node row) and `_tree_layout10` the
// silhouette edges' (`_tree_layout11` an edge row, `_tree_layout12` the
// same node row). The absorbing tree's node rows are FCPW's own BvhNode<3>,
// copied byte for byte as fcpw_bvh.h copies them; the other rows are the
// bytes FCPW's GPU transfer makes of its tree (fcpw/gpu/
// bvh_interop_structures.h: GPUSnchNode, GPUTriangle and GPUEdge, written
// by extractSnchNodes, extractTriangles and extractSilhouetteEdges), written
// here field by field as those functions write them -- that header is not
// included, since it reaches for slang-rhi, which only a GPU build has --
// and the rows' sizes and offsets are checked against FCPW's. The
// reflecting tree's node rows are one allocation the two reflecting layouts
// both point at: FCPW's one tree holds the triangles and the silhouette
// edges, each leaf referencing its range of both, and the program reads it
// as two trees.
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace wosx_stars {

using AbsorbingBvh = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle, fcpw::SilhouettePrimitive<3>>;
using ReflectingBvh = fcpw::Bvh<3, fcpw::SnchNode<3>, fcpw::Triangle, fcpw::SilhouetteEdge>;
using AbsorbingNode = fcpw::BvhNode<3>;
using ReflectingNode = fcpw::SnchNode<3>;

using AbsorbingTriangleRow = _tree_layout1;
using AbsorbingNodeRow = _tree_layout2;
using ReflectingTriangleRow = _tree_layout6;
using ReflectingNodeRow = _tree_layout7;
using EdgeRow = _tree_layout11;
using SilhouetteNodeRow = _tree_layout12;

// The trees' rows are protected members of FCPW's Bvh, read through derived
// classes that name them -- the one way to read them without editing FCPW
// (as fcpw_bvh.h reads the walk on spheres' tree). The node rows and, for
// the normal cone hierarchy, the silhouette references the leaves index
// (silhouetteRefs: the Bvh's own vector, which its build orders) are the
// Bvh's; the triangles the leaves index are the scene's own vector of
// triangle pointers, which the build sorted in place
// (SceneData::triangleObjectPtrs).
struct PeekAbsorbing : AbsorbingBvh {
    static const std::vector<AbsorbingNode> &tree(const AbsorbingBvh &b) {
        return b.*(&PeekAbsorbing::flatTree);
    }
};
struct PeekReflecting : ReflectingBvh {
    static const std::vector<ReflectingNode> &tree(const ReflectingBvh &b) {
        return b.*(&PeekReflecting::flatTree);
    }
    static const std::vector<fcpw::SilhouetteEdge *> &silhouettes(const ReflectingBvh &b) {
        return b.*(&PeekReflecting::silhouetteRefs);
    }
};

static_assert(sizeof(AbsorbingNodeRow) == sizeof(AbsorbingNode) && sizeof(AbsorbingNodeRow) == 32,
              "an absorbing node row is FCPW's BvhNode<3>: 32 bytes");
static_assert(offsetof(AbsorbingNodeRow, low) == offsetof(AbsorbingNode, box) &&
                  offsetof(AbsorbingNodeRow, high) == offsetof(AbsorbingNode, box) + 12 &&
                  offsetof(AbsorbingNodeRow, offset) == offsetof(AbsorbingNode, secondChildOffset) &&
                  offsetof(AbsorbingNodeRow, nReferences) == offsetof(AbsorbingNode, nReferences),
              "the box, the offset and the count sit where FCPW's BvhNode puts them");
static_assert(sizeof(AbsorbingTriangleRow) == 40 && sizeof(ReflectingTriangleRow) == 40,
              "a triangle row is FCPW's GPUTriangle: 40 bytes");
static_assert(offsetof(AbsorbingTriangleRow, pa) == 0 && offsetof(AbsorbingTriangleRow, pb) == 12 &&
                  offsetof(AbsorbingTriangleRow, pc) == 24 &&
                  offsetof(AbsorbingTriangleRow, primitiveIndex) == 36 &&
                  offsetof(ReflectingTriangleRow, pa) == 0 && offsetof(ReflectingTriangleRow, pb) == 12 &&
                  offsetof(ReflectingTriangleRow, pc) == 24 &&
                  offsetof(ReflectingTriangleRow, primitiveIndex) == 36,
              "the corners and the index sit where FCPW's GPUTriangle puts them");
static_assert(sizeof(ReflectingNodeRow) == 60 && sizeof(SilhouetteNodeRow) == 60,
              "a reflecting node row is FCPW's GPUSnchNode: 60 bytes");
static_assert(offsetof(ReflectingNodeRow, low) == 0 && offsetof(ReflectingNodeRow, high) == 12 &&
                  offsetof(ReflectingNodeRow, cone_axis) == 24 &&
                  offsetof(ReflectingNodeRow, cone_half_angle) == 36 &&
                  offsetof(ReflectingNodeRow, cone_radius) == 40 &&
                  offsetof(ReflectingNodeRow, nPrimitives) == 44 &&
                  offsetof(ReflectingNodeRow, offset) == 48 &&
                  offsetof(ReflectingNodeRow, nSilhouettes) == 52 &&
                  offsetof(ReflectingNodeRow, silhouetteOffset) == 56,
              "the box, the cone, the counts and the offsets sit where FCPW's GPUSnchNode puts them");
static_assert(sizeof(EdgeRow) == 56, "an edge row is FCPW's GPUEdge: 56 bytes");
static_assert(offsetof(EdgeRow, a) == 0 && offsetof(EdgeRow, b) == 12 && offsetof(EdgeRow, na) == 24 &&
                  offsetof(EdgeRow, nb) == 36 && offsetof(EdgeRow, edgeIndex) == 48 &&
                  offsetof(EdgeRow, hasOneAdjacentFace) == 52,
              "the corners, the normals, the index and the flag sit where FCPW's GPUEdge puts them");

// An allocation of rows, sixty-four-byte aligned, wrapped as a buffer.
struct Rows {
    void *data = nullptr;
    size_t bytes = 0;
    bonsai_buffer buffer{};
    Rows() = default;
    Rows(const Rows &) = delete;
    Rows &operator=(const Rows &) = delete;
    ~Rows() { std::free(data); }
    template <typename Row>
    Row *make(size_t count) {
        bytes = count * sizeof(Row);
        const size_t rounded = (bytes + 63) / 64 * 64;
        data = std::aligned_alloc(64, rounded == 0 ? 64 : rounded);
        if (data == nullptr) {
            std::cerr << "cannot allocate the tree's storage\n";
            std::exit(1);
        }
        std::memset(data, 0, rounded == 0 ? 64 : rounded);
        buffer = bonsai_buffer_wrap(data, bytes);
        return static_cast<Row *>(data);
    }
};

struct Trees {
    _tree_layout0 absorbing{};
    _tree_layout5 reflecting{};
    _tree_layout10 silhouettes{};
    Rows absorbing_nodes, absorbing_triangles, reflecting_nodes, reflecting_triangles, edges;
    uint64_t absorbing_node_count = 0, absorbing_triangle_count = 0;
    uint64_t reflecting_node_count = 0, reflecting_triangle_count = 0, edge_count = 0;
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
        reflecting.rCount = uint32_t(reflecting_triangle_count);
        reflecting.group0_rprims = &reflecting_triangles.buffer;
        reflecting.rNodes = uint32_t(reflecting_node_count);
        reflecting.group1_rnode = &reflecting_nodes.buffer;
        silhouettes.eCount = uint32_t(edge_count);
        silhouettes.group0_edges = &edges.buffer;
        silhouettes.sNodes = uint32_t(reflecting_node_count);
        silhouettes.group1_snode = &reflecting_nodes.buffer;
    }
};

// FCPW's GPUTriangle row for one of its triangles: the corners out of the
// soup through the triangle's indices, then its pIndex -- extractTriangles'
// bytes. The row's fields are one-lane vectors (a tile of one: a corner is
// three one-float arrays x, y, z; the index a one-element array), written
// component by component, which is the same twelve bytes a float3 is.
template <typename Row>
inline void write_triangle_row(Row &row, const fcpw::Triangle &t) {
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

// FCPW's GPUSnchNode row for a node of its normal cone hierarchy --
// extractSnchNodes' bytes: the box, the cone, the leaf's triangle count
// and its offset (the first triangle's for a leaf, the second child's for
// an inner node: FCPW's one union, read as the GPU reads it), then the
// leaf's silhouette count and offset.
inline void write_snch_node_row(ReflectingNodeRow &row, const ReflectingNode &node) {
    for (int k = 0; k < 3; k++) {
        row.low[k] = node.box.pMin[k];
        row.high[k] = node.box.pMax[k];
        row.cone_axis[k] = node.cone.axis[k];
    }
    row.cone_half_angle = node.cone.halfAngle;
    row.cone_radius = node.cone.radius;
    row.nPrimitives = node.nReferences;
    row.offset = node.nReferences > 0 ? node.referenceOffset : node.secondChildOffset;
    row.nSilhouettes = node.nSilhouetteReferences;
    row.silhouetteOffset = node.silhouetteReferenceOffset;
}

// FCPW's GPUEdge row for one of its silhouette edges -- extractSilhouetteEdges'
// bytes: the edge's two ends (its indices[1] and indices[2]; indices[0] and
// [3] are the opposite corners of its two faces), the normal of each
// adjacent face or zero where there is none, the edge's index, and whether
// it has one adjacent face (a rim edge, a silhouette from everywhere).
inline void write_edge_row(EdgeRow &row, const fcpw::SilhouetteEdge &e) {
    const fcpw::Vector3 &pa = e.soup->positions[e.indices[1]];
    const fcpw::Vector3 &pb = e.soup->positions[e.indices[2]];
    const fcpw::Vector3 n0 = e.hasFace(0) ? e.normal(0) : fcpw::Vector3::Zero();
    const fcpw::Vector3 n1 = e.hasFace(1) ? e.normal(1) : fcpw::Vector3::Zero();
    row.a.x[0] = pa.x();
    row.a.y[0] = pa.y();
    row.a.z[0] = pa.z();
    row.b.x[0] = pb.x();
    row.b.y[0] = pb.y();
    row.b.z[0] = pb.z();
    row.na.x[0] = n0.x();
    row.na.y[0] = n0.y();
    row.na.z[0] = n0.z();
    row.nb.x[0] = n1.x();
    row.nb.y[0] = n1.y();
    row.nb.z[0] = n1.z();
    row.edgeIndex[0] = uint32_t(e.pIndex);
    row.hasOneAdjacentFace[0] = (e.hasFace(0) && e.hasFace(1)) ? 0u : 1u;
}

inline const std::vector<fcpw::Triangle *> &scene_triangles(fcpw::Scene<3> &scene, const char *which) {
    const auto &objects = scene.getSceneData()->triangleObjectPtrs;
    if (objects.size() != 1) {
        std::cerr << "FCPW's " << which << " scene holds " << objects.size()
                  << " triangle objects; the driver gives it one\n";
        std::exit(1);
    }
    return objects[0];
}

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
    const std::vector<fcpw::Triangle *> &prims = scene_triangles(scene, "absorbing");
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
        write_triangle_row(triangle_rows[i], *prims[i]);
    }
}

// The reflecting boundary's tree, out of a built FCPW scene that holds the
// normal cone hierarchy over triangles with silhouette edges (the Neumann
// handler's, built after computeSilhouettes).
inline void copy_reflecting(fcpw::Scene<3> &scene, Trees &trees) {
    const auto *bvh = dynamic_cast<const ReflectingBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's reflecting scene did not build the normal cone hierarchy this "
                     "driver was compiled for (Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge>)\n";
        std::exit(1);
    }
    const std::vector<ReflectingNode> &nodes = PeekReflecting::tree(*bvh);
    const std::vector<fcpw::Triangle *> &prims = scene_triangles(scene, "reflecting");
    const std::vector<fcpw::SilhouetteEdge *> &silhouettes = PeekReflecting::silhouettes(*bvh);
    for (const ReflectingNode &node : nodes) {
        if (node.nReferences > 0 &&
            (uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size() ||
             uint64_t(node.silhouetteReferenceOffset) + uint64_t(node.nSilhouetteReferences) >
                 silhouettes.size())) {
            std::cerr << "a leaf of FCPW's normal cone hierarchy references past its "
                         "primitives or silhouettes\n";
            std::exit(1);
        }
    }
    trees.reflecting_node_count = nodes.size();
    trees.reflecting_triangle_count = prims.size();
    trees.edge_count = silhouettes.size();
    ReflectingNodeRow *node_rows = trees.reflecting_nodes.make<ReflectingNodeRow>(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        write_snch_node_row(node_rows[i], nodes[i]);
    }
    ReflectingTriangleRow *triangle_rows =
        trees.reflecting_triangles.make<ReflectingTriangleRow>(prims.size());
    for (size_t i = 0; i < prims.size(); i++) {
        write_triangle_row(triangle_rows[i], *prims[i]);
    }
    EdgeRow *edge_rows = trees.edges.make<EdgeRow>(silhouettes.size());
    for (size_t i = 0; i < silhouettes.size(); i++) {
        write_edge_row(edge_rows[i], *silhouettes[i]);
    }
}

inline void describe(const Trees &trees) {
    std::cout << "trees: the absorbing boundary FCPW's scalar BVH, " << trees.absorbing_node_count
              << " binary nodes (" << trees.absorbing_node_count * sizeof(AbsorbingNodeRow)
              << " bytes) over " << trees.absorbing_triangle_count
              << " triangles as FCPW's GPUTriangle rows; the reflecting boundary FCPW's normal "
                 "cone hierarchy, "
              << trees.reflecting_node_count << " nodes with cones ("
              << trees.reflecting_node_count * sizeof(ReflectingNodeRow) << " bytes) over "
              << trees.reflecting_triangle_count << " triangles and " << trees.edge_count
              << " silhouette edges as FCPW's GPUEdge rows ("
              << trees.edge_count * sizeof(EdgeRow) << " bytes)\n";
}

} // namespace wosx_stars
