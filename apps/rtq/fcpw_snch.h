// FCPW's normal cone hierarchy (SNCH) -- fcpw::Bvh<3, SnchNode<3>, Triangle,
// SilhouetteEdge>, the tree FCPW builds once silhouettes are computed, the
// one WoSX builds for a reflecting boundary and the one FCPW puts on a GPU
// for it -- written into the rows of the layouts that are its bytes
// (apps/rtq/schedules/layouts/fcpws.bonsai; apps/wosx/schedules/layouts/
// fcpw_stars.bonsai's reflecting pair), for the drivers that compare a
// program against FCPW on that tree: apps/rtq's (rtq_hook.cpp built with
// RTQ_FCPW_SNCH, which compare.sh does for the schedules named fcpws*) and
// apps/wosx's (fcpw_stars_trees.h, for the walk on stars).
//
// The rows are the bytes FCPW's own GPU transfer makes of its tree
// (fcpw/gpu/bvh_interop_structures.h: GPUSnchNode, GPUTriangle and GPUEdge,
// written by extractSnchNodes, extractTriangles and extractSilhouetteEdges),
// written here field by field as those functions write them -- that header
// is not included, since it reaches for slang-rhi, which only a GPU build
// has -- and their sizes and offsets are checked against FCPW's
// (rows_are_fcpws). The writers are templates over the row structs, since
// the generated header spells them differently per program (`_tree_layout7`
// a node row in one, `_tree_layout12` in another); each including driver
// names its own. One node array serves two layouts: FCPW's one tree holds
// the triangles and the silhouette edges, each leaf referencing its range
// of both, and the program reads it as two trees, so the driver hands the
// same node rows to both.
#pragma once

#include <fcpw/fcpw.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace rtq_snch {

using SnchBvh = fcpw::Bvh<3, fcpw::SnchNode<3>, fcpw::Triangle, fcpw::SilhouetteEdge>;
using SnchNode = fcpw::SnchNode<3>;

// The tree's rows and the silhouette references its leaves index
// (silhouetteRefs: the Bvh's own vector, which its build orders, an edge
// referenced once per leaf whose triangles share it) are protected members
// of FCPW's Bvh, read through a derived class that names them -- the one
// way to read them without editing FCPW (as fcpw_tree.h reads the Mbvh's).
// The triangles the leaves index are the scene's own vector of triangle
// pointers, which the build sorted in place (SceneData::triangleObjectPtrs).
struct Peek : SnchBvh {
    static const std::vector<SnchNode> &tree(const SnchBvh &b) { return b.*(&Peek::flatTree); }
    static const std::vector<fcpw::SilhouetteEdge *> &silhouettes(const SnchBvh &b) {
        return b.*(&Peek::silhouetteRefs);
    }
};

// The rows are FCPW's GPU rows: checked by size and offset against
// bvh_interop_structures.h's GPUSnchNode (60 bytes), GPUTriangle (40) and
// GPUEdge (56). A function so that a driver's row types are checked where
// they are named.
template <typename NodeRow, typename TriangleRow, typename EdgeRow>
constexpr bool rows_are_fcpws() {
    static_assert(sizeof(NodeRow) == 60, "a node row is FCPW's GPUSnchNode: 60 bytes");
    static_assert(offsetof(NodeRow, low) == 0 && offsetof(NodeRow, high) == 12 &&
                      offsetof(NodeRow, cone_axis) == 24 && offsetof(NodeRow, cone_half_angle) == 36 &&
                      offsetof(NodeRow, cone_radius) == 40 && offsetof(NodeRow, nPrimitives) == 44 &&
                      offsetof(NodeRow, offset) == 48 && offsetof(NodeRow, nSilhouettes) == 52 &&
                      offsetof(NodeRow, silhouetteOffset) == 56,
                  "the box, the cone, the counts and the offsets sit where FCPW's GPUSnchNode puts them");
    static_assert(sizeof(TriangleRow) == 40, "a triangle row is FCPW's GPUTriangle: 40 bytes");
    static_assert(offsetof(TriangleRow, pa) == 0 && offsetof(TriangleRow, pb) == 12 &&
                      offsetof(TriangleRow, pc) == 24 && offsetof(TriangleRow, primitiveIndex) == 36,
                  "the corners and the index sit where FCPW's GPUTriangle puts them");
    static_assert(sizeof(EdgeRow) == 56, "an edge row is FCPW's GPUEdge: 56 bytes");
    static_assert(offsetof(EdgeRow, a) == 0 && offsetof(EdgeRow, b) == 12 && offsetof(EdgeRow, na) == 24 &&
                      offsetof(EdgeRow, nb) == 36 && offsetof(EdgeRow, edgeIndex) == 48 &&
                      offsetof(EdgeRow, hasOneAdjacentFace) == 52,
                  "the corners, the normals, the index and the flag sit where FCPW's GPUEdge puts them");
    return true;
}

// An allocation of rows, sixty-four-byte aligned, wrapped as a buffer. The
// data stays where it was allocated when the Rows move, so the buffer the
// layout points at keeps its address; the owner re-points at the buffer.
struct Rows {
    void *data = nullptr;
    size_t bytes = 0;
    bonsai_buffer buffer{};
    Rows() = default;
    Rows(const Rows &) = delete;
    Rows &operator=(const Rows &) = delete;
    Rows(Rows &&other) noexcept { *this = std::move(other); }
    Rows &operator=(Rows &&other) noexcept {
        std::free(data);
        data = std::exchange(other.data, nullptr);
        bytes = std::exchange(other.bytes, 0);
        buffer = other.buffer;
        other.buffer = bonsai_buffer{};
        return *this;
    }
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
template <typename Row>
inline void write_snch_node_row(Row &row, const SnchNode &node) {
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
// [3] are the opposite corners of its two faces), the unit normal of each
// adjacent face (SilhouetteEdge::normal, which FCPW's CPU traversal
// recomputes at every test and the row holds) or zero where there is none,
// the edge's index, and whether it has one adjacent face (a rim edge, a
// silhouette from everywhere).
template <typename Row>
inline void write_edge_row(Row &row, const fcpw::SilhouetteEdge &e) {
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

// The one object's triangles of a built FCPW scene, in the order the build
// left them, which a leaf's offset indexes. (The scene is not const because
// FCPW's getSceneData is not.)
inline const std::vector<fcpw::Triangle *> &scene_triangles(fcpw::Scene<3> &scene, const char *which) {
    const auto &objects = scene.getSceneData()->triangleObjectPtrs;
    if (objects.size() != 1) {
        std::cerr << "FCPW's " << which << " scene holds " << objects.size()
                  << " triangle objects; the driver gives it one\n";
        std::exit(1);
    }
    return objects[0];
}

// What was copied, for the driver's description of the tree.
struct Counts {
    uint64_t nodes = 0, leaves = 0, triangles = 0, edges = 0, max_leaf_triangles = 0,
             max_leaf_edges = 0;
};

// The normal cone hierarchy of a built FCPW scene (silhouettes computed, the
// scalar build) written into three allocations of rows: the nodes, which
// both of the program's layouts point at, the triangles and the edges.
template <typename NodeRow, typename TriangleRow, typename EdgeRow>
inline Counts copy_snch(fcpw::Scene<3> &scene, Rows &nodes, Rows &triangles, Rows &edges,
                        const char *which) {
    static_assert(rows_are_fcpws<NodeRow, TriangleRow, EdgeRow>());
    const auto *bvh = dynamic_cast<const SnchBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's " << which
                  << " scene did not build the normal cone hierarchy this driver was compiled "
                     "for (Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge>: computeSilhouettes, then "
                     "the scalar build)\n";
        std::exit(1);
    }
    const std::vector<SnchNode> &rows = Peek::tree(*bvh);
    const std::vector<fcpw::Triangle *> &prims = scene_triangles(scene, which);
    const std::vector<fcpw::SilhouetteEdge *> &silhouettes = Peek::silhouettes(*bvh);
    Counts counts;
    for (const SnchNode &node : rows) {
        if (node.nReferences > 0) {
            if (uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size() ||
                uint64_t(node.silhouetteReferenceOffset) + uint64_t(node.nSilhouetteReferences) >
                    silhouettes.size()) {
                std::cerr << "a leaf of FCPW's normal cone hierarchy references past its "
                             "primitives or silhouettes\n";
                std::exit(1);
            }
            counts.leaves++;
            counts.max_leaf_triangles =
                std::max<uint64_t>(counts.max_leaf_triangles, uint64_t(node.nReferences));
            counts.max_leaf_edges =
                std::max<uint64_t>(counts.max_leaf_edges, uint64_t(node.nSilhouetteReferences));
        }
    }
    counts.nodes = rows.size();
    counts.triangles = prims.size();
    counts.edges = silhouettes.size();
    NodeRow *node_rows = nodes.make<NodeRow>(rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
        write_snch_node_row(node_rows[i], rows[i]);
    }
    TriangleRow *triangle_rows = triangles.make<TriangleRow>(prims.size());
    for (size_t i = 0; i < prims.size(); i++) {
        write_triangle_row(triangle_rows[i], *prims[i]);
    }
    EdgeRow *edge_rows = edges.make<EdgeRow>(silhouettes.size());
    for (size_t i = 0; i < silhouettes.size(); i++) {
        write_edge_row(edge_rows[i], *silhouettes[i]);
    }
    return counts;
}

//===----------------------------------------------------------------------===//
// The tree in the bytes FCPW's CPU reads
//===----------------------------------------------------------------------===//
//
// For the CPU comparison (rtq_hook.cpp built with RTQ_FCPW_SNCH and without
// RTQ_GPU; apps/rtq/schedules/layouts/fcpws.bonsai): nothing copied. FCPW's
// own node array (SnchNode rows, in FCPW's CPU field order), its primitive
// pointer array and Triangle objects, its silhouette reference array
// (Bvh::silhouetteRefs) and SilhouetteEdge objects, and its polygon soup's
// positions, each wrapped as a buffer where it lies, so that the program's
// leaves read a triangle and an edge as FCPW's do: the pointer, the indices
// through it, the corners and the ends gathered by index, the edge's two
// face normals recomputed from the positions. The scene has to outlive the
// tree. The row structs are the generated header's for that layout, named
// by the driver; checked here against FCPW's own types.

template <typename NodeRow, typename ObjectRow, typename PrimRow, typename EdgeObjectRow,
          typename EdgeRefRow, typename PositionRow>
constexpr bool rows_are_fcpws_cpu() {
    static_assert(sizeof(NodeRow) == sizeof(SnchNode) && sizeof(NodeRow) == 60,
                  "a node row is FCPW's SnchNode<3>: 60 bytes");
    static_assert(offsetof(NodeRow, low) == offsetof(SnchNode, box) &&
                      offsetof(NodeRow, high) == offsetof(SnchNode, box) + 12 &&
                      offsetof(NodeRow, cone_axis) == offsetof(SnchNode, cone) &&
                      offsetof(NodeRow, cone_half_angle) == offsetof(SnchNode, cone) + 12 &&
                      offsetof(NodeRow, cone_radius) == offsetof(SnchNode, cone) + 16 &&
                      offsetof(NodeRow, offset) == offsetof(SnchNode, secondChildOffset) &&
                      offsetof(NodeRow, silhouetteOffset) ==
                          offsetof(SnchNode, silhouetteReferenceOffset) &&
                      offsetof(NodeRow, nReferences) == offsetof(SnchNode, nReferences) &&
                      offsetof(NodeRow, nSilhouettes) == offsetof(SnchNode, nSilhouetteReferences),
                  "the box, the cone, the offsets and the counts sit where FCPW's SnchNode puts them");
    static_assert(sizeof(ObjectRow) == sizeof(fcpw::Triangle) && sizeof(ObjectRow) == 32,
                  "an object row is FCPW's Triangle: 32 bytes");
    static_assert(offsetof(ObjectRow, i0) == offsetof(fcpw::Triangle, indices) &&
                      offsetof(ObjectRow, i1) == offsetof(fcpw::Triangle, indices) + 4 &&
                      offsetof(ObjectRow, i2) == offsetof(fcpw::Triangle, indices) + 8 &&
                      offsetof(ObjectRow, pIndex) == offsetof(fcpw::Triangle, pIndex) &&
                      offsetof(ObjectRow, soup) == offsetof(fcpw::Triangle, soup),
                  "the indices, the index and the soup sit where FCPW's Triangle puts them");
    static_assert(sizeof(PrimRow) == sizeof(fcpw::Triangle *), "a reference row is one pointer");
    static_assert(sizeof(EdgeObjectRow) == sizeof(fcpw::SilhouetteEdge) && sizeof(EdgeObjectRow) == 40,
                  "an edge object row is FCPW's SilhouetteEdge: 40 bytes");
    static_assert(offsetof(EdgeObjectRow, i0) == offsetof(fcpw::SilhouetteEdge, indices) &&
                      offsetof(EdgeObjectRow, i3) == offsetof(fcpw::SilhouetteEdge, indices) + 12 &&
                      offsetof(EdgeObjectRow, pIndex) == offsetof(fcpw::SilhouetteEdge, pIndex) &&
                      offsetof(EdgeObjectRow, soup) == offsetof(fcpw::SilhouetteEdge, soup),
                  "the indices, the index and the soup sit where FCPW's SilhouetteEdge puts them");
    static_assert(sizeof(EdgeRefRow) == sizeof(fcpw::SilhouetteEdge *),
                  "an edge reference row is one pointer");
    static_assert(sizeof(PositionRow) == sizeof(fcpw::Vector3) && sizeof(PositionRow) == 12,
                  "a position row is FCPW's Vector3: three floats");
    return true;
}

// A buffer over memory the program only reads: the layout's storage is
// read-only to it, and FCPW's is const here.
template <typename T>
inline bonsai_buffer wrap_const(const std::vector<T> &v) {
    return bonsai_buffer_wrap(const_cast<void *>(static_cast<const void *>(v.data())),
                              uint64_t(v.size()) * sizeof(T));
}

// The buffers over FCPW's own arrays, and their counts, for the driver to
// point its two layouts at: the nodes (both layouts), the triangle
// references and objects (the triangles' layout), the edge references and
// objects (the silhouettes' layout), and the positions (both).
struct Wrapped {
    bonsai_buffer nodes{}, prims{}, objects{}, edge_refs{}, edge_objects{}, positions{};
    uint64_t object_count = 0, edge_object_count = 0, position_count = 0;
    Counts counts;
};

// The normal cone hierarchy of a built FCPW scene (silhouettes computed, the
// scalar build), FCPW's arrays wrapped and checked: every reference points
// into the one vector of objects of its kind, and every object indexes the
// one soup's positions, so that what the program follows is what the
// layout declares.
inline Wrapped wrap_snch(fcpw::Scene<3> &scene, const char *which) {
    const auto *bvh = dynamic_cast<const SnchBvh *>(scene.getSceneData()->aggregate.get());
    if (bvh == nullptr) {
        std::cerr << "FCPW's " << which
                  << " scene did not build the normal cone hierarchy this driver was compiled "
                     "for (Bvh<3, SnchNode<3>, Triangle, SilhouetteEdge>: computeSilhouettes, then "
                     "the scalar build)\n";
        std::exit(1);
    }
    const auto &data = *scene.getSceneData();
    if (data.triangleObjects.size() != 1 || data.silhouetteEdgeObjects.size() != 1 ||
        data.soups.size() != 1) {
        std::cerr << "FCPW's " << which << " scene holds " << data.triangleObjects.size()
                  << " triangle object vectors, " << data.silhouetteEdgeObjects.size()
                  << " silhouette edge object vectors and " << data.soups.size()
                  << " soups; the driver gives it one of each\n";
        std::exit(1);
    }
    const std::vector<SnchNode> &rows = Peek::tree(*bvh);
    const std::vector<fcpw::Triangle *> &prims = scene_triangles(scene, which);
    const std::vector<fcpw::SilhouetteEdge *> &silhouettes = Peek::silhouettes(*bvh);
    const std::vector<fcpw::Triangle> &objects = *data.triangleObjects[0];
    const std::vector<fcpw::SilhouetteEdge> &edge_objects = *data.silhouetteEdgeObjects[0];
    const std::vector<fcpw::Vector3> &positions = data.soups[0].positions;
    Wrapped w;
    for (const SnchNode &node : rows) {
        if (node.nReferences > 0) {
            if (uint64_t(node.referenceOffset) + uint64_t(node.nReferences) > prims.size() ||
                uint64_t(node.silhouetteReferenceOffset) + uint64_t(node.nSilhouetteReferences) >
                    silhouettes.size()) {
                std::cerr << "a leaf of FCPW's normal cone hierarchy references past its "
                             "primitives or silhouettes\n";
                std::exit(1);
            }
            w.counts.leaves++;
            w.counts.max_leaf_triangles =
                std::max<uint64_t>(w.counts.max_leaf_triangles, uint64_t(node.nReferences));
            w.counts.max_leaf_edges =
                std::max<uint64_t>(w.counts.max_leaf_edges, uint64_t(node.nSilhouetteReferences));
        }
    }
    for (const fcpw::Triangle *t : prims) {
        if (t < objects.data() || t >= objects.data() + objects.size() || t->soup != &data.soups[0]) {
            std::cerr << "a reference of FCPW's normal cone hierarchy points outside its triangle "
                         "objects, or to another soup\n";
            std::exit(1);
        }
        for (int k = 0; k < 3; k++) {
            if (t->indices[k] < 0 || size_t(t->indices[k]) >= positions.size()) {
                std::cerr << "a triangle of FCPW's normal cone hierarchy indexes past the soup's "
                             "positions\n";
                std::exit(1);
            }
        }
    }
    for (const fcpw::SilhouetteEdge *e : silhouettes) {
        if (e < edge_objects.data() || e >= edge_objects.data() + edge_objects.size() ||
            e->soup != &data.soups[0]) {
            std::cerr << "a silhouette reference of FCPW's normal cone hierarchy points outside "
                         "its edge objects, or to another soup\n";
            std::exit(1);
        }
        // The ends are always there; the opposite corners are -1 where the
        // edge has one face, and the layout reads them only where it has two.
        for (int k = 0; k < 4; k++) {
            const bool absent = (k == 0 || k == 3) && e->indices[k] == -1;
            if (!absent && (e->indices[k] < 0 || size_t(e->indices[k]) >= positions.size())) {
                std::cerr << "a silhouette edge of FCPW's normal cone hierarchy indexes past the "
                             "soup's positions\n";
                std::exit(1);
            }
        }
    }
    w.counts.nodes = rows.size();
    w.counts.triangles = prims.size();
    w.counts.edges = silhouettes.size();
    w.object_count = objects.size();
    w.edge_object_count = edge_objects.size();
    w.position_count = positions.size();
    w.nodes = wrap_const(rows);
    w.prims = wrap_const(prims);
    w.objects = wrap_const(objects);
    w.edge_refs = wrap_const(silhouettes);
    w.edge_objects = wrap_const(edge_objects);
    w.positions = wrap_const(positions);
    return w;
}

} // namespace rtq_snch
