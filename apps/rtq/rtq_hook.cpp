// The driver for apps/rtq: the same triangles and the same rays to Embree
// and to the compiled program, on the same tree.
//
//   ./apps/rtq/rtq.out [options] <mesh.ply | mesh.ply.gz>
//
// What it does, in order. It reads a mesh (PLY, binary of either byte order,
// plain or gzipped). It gives the triangles to Embree twice: as a scene
// Embree builds its own BVH8Triangle4 over and answers rtcIntersect1 and
// rtcOccluded1 on, and to Embree's builder (rtcBuildBVH) with the settings
// Embree's internal BVH8Triangle4 builder uses, whose callbacks write the
// tree into the layout the schedule declared (schedules/trees/bvh8.bonsai),
// so that both sides traverse one tree: the same splits, the same leaves,
// the same boxes in the same places. It makes three batches of rays --
// primary rays from a camera, and from their hits short ambient-occlusion
// rays and long diffuse bounces, as Aila and Laine's ray sets are made --
// traces each with both sides, checks that they agree, and times them, each
// the least of several runs after a warm-up, both sides on one thread unless
// --threads says otherwise (and then both over one TBB pool of that many).
//
// Embree's traversal is the reference for what the schedule should cost;
// its answers are the reference for what the program should compute.
// The program's parallel loop is supplied by this file (for_blocks, below) so
// that both sides run their rays through one and the same loop.
#define BONSAI_PARALLEL_EXTERNAL
#include "rtq.h"

#include <embree4/rtcore.h>
#include <embree4/rtcore_builder.h>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <sched.h>
#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// How many threads each side runs on. 1, the default, runs a batch as one
// plain loop on the calling thread -- no pool, no splitting -- on both sides;
// more spreads the batch in blocks over a TBB pool of that many, on both
// sides. One loop serves Embree's rtcIntersect1 and the program's exported
// parfor alike: the compiled code hands its parfor to bonsai_parallel_for,
// which is defined here (BONSAI_PARALLEL_EXTERNAL, above).
int g_threads = 1;

template <typename F>
void for_blocks(int64_t n, F &&f) {
    if (g_threads == 1) {
        f(int64_t(0), n);
        return;
    }
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, n, 64),
                      [&](const tbb::blocked_range<int64_t> &r) { f(r.begin(), r.end()); });
}

extern "C" void bonsai_parallel_for(int64_t n, void *context,
                                    void (*body)(void *, int64_t, int64_t)) {
    for_blocks(n, [&](int64_t begin, int64_t end) { body(context, begin, end); });
}

namespace {

constexpr uint32_t kNoHit = 0xffffffffu;
constexpr float kInf = std::numeric_limits<float>::infinity();

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float length(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 0 ? a * (1.0f / l) : a;
}
Vec3 vmin(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
Vec3 vmax(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
float3 to_float3(Vec3 v) { return float3{v.x, v.y, v.z}; }

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<uint32_t> indices; // three per triangle
    size_t triangles() const { return indices.size() / 3; }
    Vec3 lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
};

//===----------------------------------------------------------------------===//
// PLY
//===----------------------------------------------------------------------===//

// A PLY file, plain or gzipped (zlib reads both), binary in either byte
// order: the vertex positions and the faces, triangulated as fans. pbrt's
// scenes store their meshes this way, which is what this reads.
struct PlyReader {
    gzFile file = nullptr;
    bool big_endian = false;

    explicit PlyReader(const std::string &path) {
        file = gzopen(path.c_str(), "rb");
        if (file == nullptr) {
            std::cerr << "cannot open " << path << '\n';
            std::exit(1);
        }
    }
    ~PlyReader() {
        if (file != nullptr) {
            gzclose(file);
        }
    }

    std::string line() {
        char buffer[4096];
        if (gzgets(file, buffer, sizeof buffer) == nullptr) {
            std::cerr << "unexpected end of PLY header\n";
            std::exit(1);
        }
        std::string s(buffer);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        return s;
    }

    void read_bytes(void *out, size_t n) {
        if (gzread(file, out, unsigned(n)) != int(n)) {
            std::cerr << "unexpected end of PLY data\n";
            std::exit(1);
        }
    }

    template <typename T>
    T read() {
        T v;
        read_bytes(&v, sizeof v);
        if (big_endian) {
            char *p = reinterpret_cast<char *>(&v);
            std::reverse(p, p + sizeof v);
        }
        return v;
    }

    // One scalar of the named PLY type, as a double.
    double read_scalar(const std::string &type) {
        if (type == "float" || type == "float32") {
            return read<float>();
        } else if (type == "double" || type == "float64") {
            return read<double>();
        } else if (type == "uchar" || type == "uint8") {
            return read<uint8_t>();
        } else if (type == "char" || type == "int8") {
            return read<int8_t>();
        } else if (type == "ushort" || type == "uint16") {
            return read<uint16_t>();
        } else if (type == "short" || type == "int16") {
            return read<int16_t>();
        } else if (type == "uint" || type == "uint32") {
            return read<uint32_t>();
        } else if (type == "int" || type == "int32") {
            return read<int32_t>();
        }
        std::cerr << "PLY property of unknown type " << type << '\n';
        std::exit(1);
    }
};

Mesh load_ply(const std::string &path) {
    PlyReader in(path);
    if (in.line() != "ply") {
        std::cerr << path << " is not a PLY file\n";
        std::exit(1);
    }
    struct Property {
        std::string type;      // or the list's count type
        std::string item_type; // for a list
        std::string name;
        bool list = false;
    };
    struct Element {
        std::string name;
        size_t count = 0;
        std::vector<Property> properties;
    };
    std::vector<Element> elements;
    for (std::string l = in.line(); l != "end_header"; l = in.line()) {
        if (l.rfind("format ", 0) == 0) {
            if (l.find("binary_big_endian") != std::string::npos) {
                in.big_endian = true;
            } else if (l.find("binary_little_endian") == std::string::npos) {
                std::cerr << "only binary PLY is read: " << l << '\n';
                std::exit(1);
            }
        } else if (l.rfind("element ", 0) == 0) {
            Element e;
            char name[256];
            unsigned long long count = 0;
            if (std::sscanf(l.c_str(), "element %255s %llu", name, &count) != 2) {
                std::cerr << "bad PLY element line: " << l << '\n';
                std::exit(1);
            }
            e.name = name;
            e.count = size_t(count);
            elements.push_back(e);
        } else if (l.rfind("property ", 0) == 0) {
            if (elements.empty()) {
                std::cerr << "PLY property before any element\n";
                std::exit(1);
            }
            Property p;
            char a[64], b[64], c[64];
            if (std::sscanf(l.c_str(), "property list %63s %63s %63s", a, b, c) == 3) {
                p.list = true;
                p.type = a;
                p.item_type = b;
                p.name = c;
            } else if (std::sscanf(l.c_str(), "property %63s %63s", a, b) == 2) {
                p.type = a;
                p.name = b;
            } else {
                std::cerr << "bad PLY property line: " << l << '\n';
                std::exit(1);
            }
            elements.back().properties.push_back(p);
        }
    }

    Mesh mesh;
    for (const Element &e : elements) {
        if (e.name == "vertex") {
            int ix = -1, iy = -1, iz = -1;
            for (size_t k = 0; k < e.properties.size(); k++) {
                if (e.properties[k].name == "x") ix = int(k);
                if (e.properties[k].name == "y") iy = int(k);
                if (e.properties[k].name == "z") iz = int(k);
            }
            if (ix < 0 || iy < 0 || iz < 0) {
                std::cerr << "PLY vertices without x, y, z\n";
                std::exit(1);
            }
            mesh.vertices.reserve(e.count);
            for (size_t i = 0; i < e.count; i++) {
                Vec3 v;
                for (size_t k = 0; k < e.properties.size(); k++) {
                    const Property &p = e.properties[k];
                    if (p.list) {
                        const size_t n = size_t(in.read_scalar(p.type));
                        for (size_t j = 0; j < n; j++) {
                            in.read_scalar(p.item_type);
                        }
                        continue;
                    }
                    const double value = in.read_scalar(p.type);
                    if (int(k) == ix) v.x = float(value);
                    if (int(k) == iy) v.y = float(value);
                    if (int(k) == iz) v.z = float(value);
                }
                mesh.vertices.push_back(v);
                mesh.lo = vmin(mesh.lo, v);
                mesh.hi = vmax(mesh.hi, v);
            }
        } else if (e.name == "face") {
            for (size_t i = 0; i < e.count; i++) {
                for (const Property &p : e.properties) {
                    if (!p.list) {
                        in.read_scalar(p.type);
                        continue;
                    }
                    const size_t n = size_t(in.read_scalar(p.type));
                    std::vector<uint32_t> corners(n);
                    for (size_t j = 0; j < n; j++) {
                        corners[j] = uint32_t(in.read_scalar(p.item_type));
                    }
                    if (p.name != "vertex_indices" && p.name != "vertex_index") {
                        continue;
                    }
                    // A polygon as a fan, which is how pbrt triangulates.
                    for (size_t j = 1; j + 1 < n; j++) {
                        mesh.indices.push_back(corners[0]);
                        mesh.indices.push_back(corners[j]);
                        mesh.indices.push_back(corners[j + 1]);
                    }
                }
            }
        } else {
            // Something else (edges, say): read past it.
            for (size_t i = 0; i < e.count; i++) {
                for (const Property &p : e.properties) {
                    if (!p.list) {
                        in.read_scalar(p.type);
                        continue;
                    }
                    const size_t n = size_t(in.read_scalar(p.type));
                    for (size_t j = 0; j < n; j++) {
                        in.read_scalar(p.item_type);
                    }
                }
            }
        }
    }
    for (uint32_t index : mesh.indices) {
        if (index >= mesh.vertices.size()) {
            std::cerr << "PLY face names vertex " << index << " of "
                      << mesh.vertices.size() << '\n';
            std::exit(1);
        }
    }
    return mesh;
}

//===----------------------------------------------------------------------===//
// Embree
//===----------------------------------------------------------------------===//

void embree_error(void *, RTCError code, const char *message) {
    std::cerr << "Embree error " << int(code) << ": " << message << '\n';
}

// Embree's own scene over the mesh: one triangle geometry, built and
// traversed by Embree as shipped (a BVH8 over Triangle4 leaves on a machine
// with AVX, see scene.cpp's createTriangleAccel).
RTCScene make_embree_scene(RTCDevice device, const Mesh &mesh) {
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float *vertices = static_cast<float *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh.vertices.size()));
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        vertices[3 * i + 0] = mesh.vertices[i].x;
        vertices[3 * i + 1] = mesh.vertices[i].y;
        vertices[3 * i + 2] = mesh.vertices[i].z;
    }
    uint32_t *indices = static_cast<uint32_t *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(uint32_t), mesh.triangles()));
    std::copy(mesh.indices.begin(), mesh.indices.end(), indices);
    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene);
    return scene;
}

//===----------------------------------------------------------------------===//
// The tree, built by Embree into the layout the schedule declared
//===----------------------------------------------------------------------===//

// Embree: NodeRefPtr<8>. The low four bits say what a reference is -- 0 an
// AABB node, tyLeaf (8) + n a leaf of n blocks -- and the rest where. Embree
// puts a byte address there; the layout (schedules/trees/bvh8.bonsai) puts
// the node's row or the leaf's first block. An empty child slot is
// `emptyNode`, a leaf of no blocks, with bounds no ray meets.
constexpr uint64_t kTyLeaf = 8;
constexpr uint64_t kEmptyNode = kTyLeaf;

uint64_t node_ref(uint64_t row) { return row << 4; }
uint64_t leaf_ref(uint64_t first_block, uint64_t blocks) {
    return (first_block << 4) | (kTyLeaf + blocks);
}

// The layout's storage, as the generated header spells it: a node row is
// Embree's AABBNode -- the eight children, then the bounds as six vectors of
// eight floats in Embree's order -- and a leaf's block is Embree's Triangle4,
// four triangles with every field a vector over the four (see
// schedules/trees/bvh8.bonsai).
using NodeRow = _tree_layout2;
using TriangleBlock = _tree_layout1;

// What rtcBuildBVH's callbacks fill: the rows and the triangle blocks,
// claimed by atomic counters since Embree builds in parallel.
struct Building {
    const Mesh *mesh = nullptr;
    std::vector<NodeRow> rows;
    std::atomic<uint64_t> rows_used{0};
    std::vector<TriangleBlock> blocks;
    std::atomic<uint64_t> blocks_used{0};
    std::atomic<uint64_t> leaves{0};
    std::atomic<uint64_t> leaf_prims{0};

    static Building *of(void *user) { return static_cast<Building *>(user); }

    static void *create_node(RTCThreadLocalAllocator, unsigned, void *user) {
        Building &b = *of(user);
        const uint64_t row = b.rows_used.fetch_add(1);
        if (row >= b.rows.size()) {
            std::cerr << "more nodes than rows were made for\n";
            std::abort();
        }
        // Embree: AABBNode::clear -- every child empty, every box empty.
        NodeRow &n = b.rows[row];
        for (int i = 0; i < 8; i++) {
            n.children[i] = kEmptyNode;
            n.lower_x[i] = n.lower_y[i] = n.lower_z[i] = kInf;
            n.upper_x[i] = n.upper_y[i] = n.upper_z[i] = -kInf;
        }
        return reinterpret_cast<void *>(node_ref(row));
    }

    static void set_children(void *node, void **children, unsigned count,
                             void *user) {
        Building &b = *of(user);
        NodeRow &n = b.rows[reinterpret_cast<uint64_t>(node) >> 4];
        for (unsigned i = 0; i < count; i++) {
            n.children[i] = reinterpret_cast<uint64_t>(children[i]);
        }
    }

    // Embree: AABBNode::setBounds -- the child's box into lane i of the six
    // bound vectors.
    static void set_bounds(void *node, const RTCBounds **bounds, unsigned count,
                           void *user) {
        Building &b = *of(user);
        NodeRow &n = b.rows[reinterpret_cast<uint64_t>(node) >> 4];
        for (unsigned i = 0; i < count; i++) {
            n.lower_x[i] = bounds[i]->lower_x;
            n.lower_y[i] = bounds[i]->lower_y;
            n.lower_z[i] = bounds[i]->lower_z;
            n.upper_x[i] = bounds[i]->upper_x;
            n.upper_y[i] = bounds[i]->upper_y;
            n.upper_z[i] = bounds[i]->upper_z;
        }
    }

    // A triangle into lane `lane` of a block: TriangleM::fill, with the
    // vertex and the two edges by component.
    static void fill(TriangleBlock &block, unsigned lane, const Vec3 &v0,
                     const Vec3 &e1, const Vec3 &e2, uint32_t geom_id,
                     uint32_t prim_id) {
        block.v0.x[lane] = v0.x;
        block.v0.y[lane] = v0.y;
        block.v0.z[lane] = v0.z;
        block.e1.x[lane] = e1.x;
        block.e1.y[lane] = e1.y;
        block.e1.z[lane] = e1.z;
        block.e2.x[lane] = e2.x;
        block.e2.y[lane] = e2.y;
        block.e2.z[lane] = e2.z;
        block.geomID[lane] = geom_id;
        block.primID[lane] = prim_id;
    }

    // Embree: CreateLeaf<8, Triangle4> -- blocks(n) = (n + 3) / 4 blocks,
    // each filled from the primitives in order (TriangleM::fill), a lane a
    // block does not fill left as the zero triangle with ids of -1.
    static void *create_leaf(RTCThreadLocalAllocator,
                             const RTCBuildPrimitive *prims, size_t count,
                             void *user) {
        Building &b = *of(user);
        const uint64_t blocks = (count + 3) / 4;
        const uint64_t first = b.blocks_used.fetch_add(blocks);
        if (first + blocks > b.blocks.size()) {
            std::cerr << "more leaf blocks than were made for\n";
            std::abort();
        }
        b.leaves.fetch_add(1);
        b.leaf_prims.fetch_add(count);
        const Mesh &mesh = *b.mesh;
        for (uint64_t s = 0; s < 4 * blocks; s++) {
            TriangleBlock &block = b.blocks[first + s / 4];
            const unsigned lane = unsigned(s % 4);
            if (s >= count) {
                fill(block, lane, Vec3{}, Vec3{}, Vec3{}, kNoHit, kNoHit);
                continue;
            }
            const uint32_t prim = prims[s].primID;
            const Vec3 p0 = mesh.vertices[mesh.indices[3 * prim + 0]];
            const Vec3 p1 = mesh.vertices[mesh.indices[3 * prim + 1]];
            const Vec3 p2 = mesh.vertices[mesh.indices[3 * prim + 2]];
            fill(block, lane, p0, p0 - p1, p2 - p0, prims[s].geomID, prim);
        }
        return reinterpret_cast<void *>(leaf_ref(first, blocks));
    }
};

// Embree: BVHN::layoutLargeNodes (kernels/bvh/bvh.cpp), run by the builder
// after the build over the `num` largest nodes. From the root, the node of
// greatest surface area is taken off a heap and its children put on, until
// `num` nodes are waiting; the ones taken -- the top of the tree by area --
// are then copied into fresh memory in depth-first order, root first, so
// that they sit together, while the ones waiting (the frontier) and
// everything below them stay where the build put them. Here the rows are
// permuted to the same effect: the copied nodes take the first rows in
// depth-first order, the root staying row 0, and the rest follow in their
// build order -- Embree leaves the copied nodes' old slots as holes, which
// a permutation closes up, and that is the one difference.
void relocate_large_nodes(Building &b, const Mesh &mesh, size_t num) {
    const uint64_t rows = b.rows_used.load();
    if (num == 0 || rows == 0) {
        return;
    }
    // Embree: area(bounds) = 2 * halfArea, halfArea(d) = d.x * (d.y + d.z) +
    // d.y * d.z; a leaf counts as -inf, so it is never expanded.
    const auto area = [](float lx, float ly, float lz, float ux, float uy,
                         float uz) {
        const float dx = ux - lx, dy = uy - ly, dz = uz - lz;
        return 2.0f * (dx * (dy + dz) + dy * dz);
    };
    struct Item {
        uint64_t ref;
        float area;
        bool operator<(const Item &o) const { return area < o.area; }
    };
    std::vector<Item> waiting;
    waiting.reserve(num + 8);
    waiting.push_back({node_ref(0), area(mesh.lo.x, mesh.lo.y, mesh.lo.z,
                                         mesh.hi.x, mesh.hi.y, mesh.hi.z)});
    std::vector<bool> copied(rows, false);
    while (waiting.size() < num) {
        std::pop_heap(waiting.begin(), waiting.end());
        const Item top = waiting.back();
        waiting.pop_back();
        if ((top.ref & 15) != 0) {
            break; // a leaf: nothing larger is left
        }
        const uint64_t row = top.ref >> 4;
        copied[row] = true;
        const NodeRow &n = b.rows[row];
        for (int i = 0; i < 8; i++) {
            if (n.children[i] == kEmptyNode) {
                continue;
            }
            const float a = (n.children[i] & 15) != 0
                                ? -kInf
                                : area(n.lower_x[i], n.lower_y[i], n.lower_z[i],
                                       n.upper_x[i], n.upper_y[i], n.upper_z[i]);
            waiting.push_back({n.children[i], a});
            std::push_heap(waiting.begin(), waiting.end());
        }
    }
    // The copied nodes in depth-first order from the root, then the rest in
    // their build order.
    std::vector<uint64_t> new_row(rows, UINT64_MAX);
    uint64_t next = 0;
    const std::function<void(uint64_t)> visit = [&](uint64_t row) {
        new_row[row] = next++;
        const NodeRow &n = b.rows[row];
        for (int i = 0; i < 8; i++) {
            const uint64_t child = n.children[i];
            if ((child & 15) == 0 && copied[child >> 4]) {
                visit(child >> 4);
            }
        }
    };
    if (copied[0]) {
        visit(0);
    }
    for (uint64_t row = 0; row < rows; row++) {
        if (new_row[row] == UINT64_MAX) {
            new_row[row] = next++;
        }
    }
    std::vector<NodeRow> moved(rows);
    for (uint64_t row = 0; row < rows; row++) {
        NodeRow &n = moved[new_row[row]];
        n = b.rows[row];
        for (int i = 0; i < 8; i++) {
            if ((n.children[i] & 15) == 0) {
                n.children[i] = node_ref(new_row[n.children[i] >> 4]);
            }
        }
    }
    b.rows.swap(moved);
}

struct Tree {
    std::unique_ptr<Building> building;
    _tree_layout0 layout{};
    bonsai_buffer blocks_buffer{}, rows_buffer{};
    uint64_t nodes = 0, leaves = 0, blocks = 0, leaf_prims = 0;
};

// Embree's tree over the mesh, in the layout. The settings are those of
// Embree's internal BVH8Triangle4 builder (bvh_builder_sah.cpp:
// BVHNBuilderSAH<8, Triangle4>(.., sahBlockSize 4, intCost 1, minLeafSize
// 4, maxLeafSize inf -> 4 * maxLeafBlocks = 28); bvh_builder.cpp:
// branchingFactor N, maxDepth maxBuildDepthLeaf = 40; medium quality is the
// binned SAH), so rtcBuildBVH runs the same builder over the same primitive
// references and makes the same tree, up to the order of the primitives
// within a leaf's blocks where the build partitions in parallel.
Tree build_tree(RTCDevice device, const Mesh &mesh) {
    Tree tree;
    tree.building = std::make_unique<Building>();
    Building &b = *tree.building;
    b.mesh = &mesh;
    const size_t count = mesh.triangles();
    // Every node has at least two children and every leaf at least one
    // primitive (bar one), so there are fewer nodes than primitives; a leaf
    // of n primitives takes (n + 3) / 4 blocks and holds at least 4 unless
    // it is the whole tree, so there are at most about half as many blocks
    // as primitives.
    b.rows.resize(count + 2);
    b.blocks.resize(count / 2 + 2);

    std::vector<RTCBuildPrimitive> prims(count);
    for (size_t i = 0; i < count; i++) {
        const Vec3 p0 = mesh.vertices[mesh.indices[3 * i + 0]];
        const Vec3 p1 = mesh.vertices[mesh.indices[3 * i + 1]];
        const Vec3 p2 = mesh.vertices[mesh.indices[3 * i + 2]];
        const Vec3 lo = vmin(vmin(p0, p1), p2), hi = vmax(vmax(p0, p1), p2);
        prims[i].lower_x = lo.x;
        prims[i].lower_y = lo.y;
        prims[i].lower_z = lo.z;
        prims[i].upper_x = hi.x;
        prims[i].upper_y = hi.y;
        prims[i].upper_z = hi.z;
        prims[i].geomID = 0;
        prims[i].primID = uint32_t(i);
    }

    RTCBVH bvh = rtcNewBVH(device);
    RTCBuildArguments args = rtcDefaultBuildArguments();
    args.byteSize = sizeof(args);
    args.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
    args.buildFlags = RTC_BUILD_FLAG_NONE;
    args.maxBranchingFactor = 8;
    args.maxDepth = 40;
    args.sahBlockSize = 4;
    args.minLeafSize = 4;
    args.maxLeafSize = 28;
    args.traversalCost = 1.0f;
    args.intersectionCost = 1.0f;
    args.bvh = bvh;
    args.primitives = prims.data();
    args.primitiveCount = count;
    args.primitiveArrayCapacity = count;
    args.createNode = Building::create_node;
    args.setNodeChildren = Building::set_children;
    args.setNodeBounds = Building::set_bounds;
    args.createLeaf = Building::create_leaf;
    args.userPtr = &b;
    void *root = rtcBuildBVH(&args);
    rtcReleaseBVH(bvh);

    // The traversal starts at reference 0, row 0, which the root is: Embree
    // creates the root node before any other. A mesh small enough to be one
    // leaf gets a node over it.
    const uint64_t root_ref = reinterpret_cast<uint64_t>(root);
    if (root_ref != node_ref(0)) {
        if ((root_ref & 15) == 0 || b.rows_used.load() != 0) {
            std::cerr << "the root is not the first row: " << root_ref << '\n';
            std::exit(1);
        }
        void *node = Building::create_node(nullptr, 1, &b);
        void *children[1] = {root};
        const RTCBounds bounds{mesh.lo.x, mesh.lo.y, mesh.lo.z, 0,
                               mesh.hi.x, mesh.hi.y, mesh.hi.z, 0};
        const RTCBounds *bounds_of[1] = {&bounds};
        Building::set_bounds(node, bounds_of, 1, &b);
        Building::set_children(node, children, 1, &b);
    }

    // Embree: bvh_builder_sah.cpp -- layoutLargeNodes(pinfo.size() * 0.005f)
    // after the build, over half a percent of the primitive count.
    relocate_large_nodes(b, mesh, size_t(float(count) * 0.005f));

    tree.nodes = b.rows_used.load();
    tree.blocks = b.blocks_used.load();
    tree.leaves = b.leaves.load();
    tree.leaf_prims = b.leaf_prims.load();
    b.rows.resize(tree.nodes);
    b.blocks.resize(tree.blocks);

    // The bytes, as Embree's: a 256-byte row and a 176-byte block.
    static_assert(sizeof(NodeRow) == 256, "a node row is Embree's AABBNode");
    static_assert(offsetof(NodeRow, lower_x) == 64 &&
                      offsetof(NodeRow, upper_x) == 96 &&
                      offsetof(NodeRow, lower_y) == 128 &&
                      offsetof(NodeRow, upper_y) == 160 &&
                      offsetof(NodeRow, lower_z) == 192 &&
                      offsetof(NodeRow, upper_z) == 224,
                  "the bounds sit where Embree's AABBNode puts them");
    static_assert(sizeof(TriangleBlock) == 176,
                  "a block is Embree's Triangle4");
    static_assert(offsetof(TriangleBlock, e1) == 48 &&
                      offsetof(TriangleBlock, e2) == 96 &&
                      offsetof(TriangleBlock, geomID) == 144 &&
                      offsetof(TriangleBlock, primID) == 160,
                  "the fields sit where Embree's TriangleM<4> puts them");

    tree.blocks_buffer = bonsai_buffer_wrap(
        b.blocks.data(), b.blocks.size() * sizeof(TriangleBlock));
    tree.rows_buffer =
        bonsai_buffer_wrap(b.rows.data(), b.rows.size() * sizeof(NodeRow));
    // The layout counts elements -- slots, four to a block.
    tree.layout.pCount = uint32_t(4 * tree.blocks);
    tree.layout.group0_prims = &tree.blocks_buffer;
    tree.layout.nCount = uint32_t(tree.nodes);
    tree.layout.group1_row = &tree.rows_buffer;
    return tree;
}

//===----------------------------------------------------------------------===//
// Rays
//===----------------------------------------------------------------------===//

// PCG32 (O'Neill), so the ray sets are the same on every run.
struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;
    uint32_t next() {
        const uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = uint32_t(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    float uniform() { return float(next() >> 8) * (1.0f / 16777216.0f); }
};

Ray make_ray(Vec3 o, Vec3 d, float tfar) {
    Ray r;
    r.o = to_float3(o);
    r.tnear = 0.0f;
    r.d = to_float3(d);
    r.tfar = tfar;
    return r;
}

// Primary rays: a pinhole camera a scene diagonal and a half from the
// mesh's centre, looking at it from above and to the right, over a square
// image of `side` by `side` pixels.
std::vector<Ray> primary_rays(const Mesh &mesh, uint32_t side) {
    const Vec3 centre = (mesh.lo + mesh.hi) * 0.5f;
    const float diagonal = length(mesh.hi - mesh.lo);
    const Vec3 eye = centre + normalize(Vec3{0.5f, 0.35f, 1.0f}) * (1.5f * diagonal);
    const Vec3 forward = normalize(centre - eye);
    const Vec3 right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
    const Vec3 up = cross(right, forward);
    const float tan_half = std::tan(0.5f * 40.0f * 3.14159265f / 180.0f);
    std::vector<Ray> rays;
    rays.reserve(size_t(side) * side);
    for (uint32_t y = 0; y < side; y++) {
        for (uint32_t x = 0; x < side; x++) {
            const float sx = (2.0f * (float(x) + 0.5f) / float(side) - 1.0f) * tan_half;
            const float sy = (1.0f - 2.0f * (float(y) + 0.5f) / float(side)) * tan_half;
            const Vec3 d = normalize(forward + right * sx + up * sy);
            rays.push_back(make_ray(eye, d, kInf));
        }
    }
    return rays;
}

// A direction in the hemisphere about `n`, cosine distributed.
Vec3 cosine_hemisphere(Vec3 n, Rng &rng) {
    const float u1 = rng.uniform(), u2 = rng.uniform();
    const float r = std::sqrt(u1);
    const float phi = 2.0f * 3.14159265f * u2;
    const Vec3 local{r * std::cos(phi), r * std::sin(phi),
                     std::sqrt(std::max(0.0f, 1.0f - u1))};
    const Vec3 a = std::fabs(n.x) > 0.9f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const Vec3 t = normalize(cross(a, n));
    const Vec3 b = cross(n, t);
    return normalize(t * local.x + b * local.y + n * local.z);
}

// Secondary rays from Embree's primary hits: one ray per hit, leaving the
// hit point along the hemisphere about the geometric normal facing the
// camera, `reach` scene diagonals long -- a tenth for ambient occlusion,
// unbounded for a diffuse bounce.
std::vector<Ray> secondary_rays(const std::vector<Ray> &primary,
                                const std::vector<RTCRayHit> &hits,
                                const Mesh &mesh, float reach, uint64_t seed) {
    const float diagonal = length(mesh.hi - mesh.lo);
    Rng rng;
    rng.state += seed;
    std::vector<Ray> rays;
    rays.reserve(primary.size());
    for (size_t i = 0; i < primary.size(); i++) {
        const RTCRayHit &h = hits[i];
        if (h.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            continue;
        }
        const Vec3 o{primary[i].o.x, primary[i].o.y, primary[i].o.z};
        const Vec3 d{primary[i].d.x, primary[i].d.y, primary[i].d.z};
        Vec3 n = normalize(Vec3{h.hit.Ng_x, h.hit.Ng_y, h.hit.Ng_z});
        if (dot(n, d) > 0) {
            n = n * -1.0f;
        }
        const Vec3 p = o + d * h.ray.tfar + n * (1e-4f * diagonal);
        const float tfar = reach > 0 ? reach * diagonal : kInf;
        rays.push_back(make_ray(p, cosine_hemisphere(n, rng), tfar));
    }
    return rays;
}

//===----------------------------------------------------------------------===//
// Tracing, both sides
//===----------------------------------------------------------------------===//

RTCRayHit to_rayhit(const Ray &r) {
    RTCRayHit rh;
    rh.ray.org_x = r.o.x;
    rh.ray.org_y = r.o.y;
    rh.ray.org_z = r.o.z;
    rh.ray.tnear = r.tnear;
    rh.ray.dir_x = r.d.x;
    rh.ray.dir_y = r.d.y;
    rh.ray.dir_z = r.d.z;
    rh.ray.time = 0.0f;
    rh.ray.tfar = r.tfar;
    rh.ray.mask = ~0u;
    rh.ray.id = 0;
    rh.ray.flags = 0;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    return rh;
}

// Embree over a batch: rtcIntersect1 per ray, through the same loop the
// program's parfor runs through.
void embree_intersect(RTCScene scene, const std::vector<Ray> &rays,
                      std::vector<RTCRayHit> &hits) {
    hits.resize(rays.size());
    for_blocks(int64_t(rays.size()), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            hits[i] = to_rayhit(rays[i]);
            rtcIntersect1(scene, &hits[i]);
        }
    });
}

void embree_occluded(RTCScene scene, const std::vector<Ray> &rays,
                     std::vector<uint32_t> &blocked) {
    blocked.resize(rays.size());
    for_blocks(int64_t(rays.size()), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            RTCRay ray = to_rayhit(rays[i]).ray;
            rtcOccluded1(scene, &ray);
            // Embree: an occluded ray has its tfar set to -inf.
            blocked[i] = ray.tfar < 0.0f ? 1u : 0u;
        }
    });
}

// The program over a batch: the exported parfor, bound to the threads.
void bonsai_intersect(const Tree &tree, const std::vector<Ray> &rays,
                      std::vector<uint32_t> &hits) {
    hits.resize(rays.size());
    trace_all(uint32_t(rays.size()), rays.data(), hits.data(), tree.layout);
}

void bonsai_occluded(const Tree &tree, const std::vector<Ray> &rays,
                     std::vector<uint32_t> &blocked) {
    blocked.resize(rays.size());
    occluded_all(uint32_t(rays.size()), rays.data(), blocked.data(),
                 tree.layout);
}

// The CPUs this process may run on -- what numactl left it -- for the record
// at the top of the output.
std::string allowed_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return "?";
    }
    std::string out;
    int run = -1;
    for (int cpu = 0; cpu <= CPU_SETSIZE; cpu++) {
        const bool in = cpu < CPU_SETSIZE && CPU_ISSET(cpu, &set);
        if (in && run < 0) {
            run = cpu;
        } else if (!in && run >= 0) {
            out += (out.empty() ? "" : ",") + std::to_string(run);
            if (cpu - 1 > run) {
                out += "-" + std::to_string(cpu - 1);
            }
            run = -1;
        }
    }
    return out;
}

// Seconds for `f`, the least of `repeats` runs after one unmeasured warm-up
// (bonsai-benchmark-repeats: never one run).
template <typename F>
double timed(int repeats, F &&f) {
    using clock = std::chrono::steady_clock;
    f();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
        const auto start = clock::now();
        f();
        const auto end = clock::now();
        best = std::min(best, std::chrono::duration<double>(end - start).count());
    }
    return best;
}

// The distance along the ray to a triangle, in double precision, for
// telling a disagreement about which triangle is nearest from a tie: two
// triangles at the same distance, Embree's and ours, both right.
double distance_to(const Mesh &mesh, const Ray &r, uint32_t prim) {
    if (prim == kNoHit) {
        return std::numeric_limits<double>::infinity();
    }
    const Vec3 p0 = mesh.vertices[mesh.indices[3 * prim + 0]];
    const Vec3 p1 = mesh.vertices[mesh.indices[3 * prim + 1]];
    const Vec3 p2 = mesh.vertices[mesh.indices[3 * prim + 2]];
    const double e1[3] = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
    const double e2[3] = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
    const double d[3] = {r.d.x, r.d.y, r.d.z};
    const double pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2],
                          d[0] * e2[1] - d[1] * e2[0]};
    const double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
    if (det == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double tv[3] = {r.o.x - p0.x, r.o.y - p0.y, r.o.z - p0.z};
    const double qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2],
                          tv[0] * e1[1] - tv[1] * e1[0]};
    return (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) / det;
}

struct Agreement {
    size_t rays = 0, same = 0, ties = 0, differ = 0;
    size_t embree_hits = 0, bonsai_hits = 0;
};

Agreement compare_hits(const Mesh &mesh, const std::vector<Ray> &rays,
                       const std::vector<RTCRayHit> &embree,
                       const std::vector<uint32_t> &ours) {
    Agreement a;
    a.rays = rays.size();
    for (size_t i = 0; i < rays.size(); i++) {
        const uint32_t e = embree[i].hit.geomID == RTC_INVALID_GEOMETRY_ID
                               ? kNoHit
                               : embree[i].hit.primID;
        const uint32_t b = ours[i];
        a.embree_hits += e != kNoHit;
        a.bonsai_hits += b != kNoHit;
        if (e == b) {
            a.same++;
            continue;
        }
        // Two different triangles at (nearly) the same distance are a tie,
        // which either side may answer either way: the arithmetic of the
        // two tests differs in its last bits (PLAN.md).
        const double te = distance_to(mesh, rays[i], e);
        const double tb = distance_to(mesh, rays[i], b);
        const double scale = std::max({1.0, std::fabs(te), std::fabs(tb)});
        if (std::isfinite(te) && std::isfinite(tb) &&
            std::fabs(te - tb) <= 1e-4 * scale) {
            a.ties++;
        } else {
            a.differ++;
        }
    }
    return a;
}

Agreement compare_occluded(const std::vector<uint32_t> &embree,
                           const std::vector<uint32_t> &ours) {
    Agreement a;
    a.rays = embree.size();
    for (size_t i = 0; i < embree.size(); i++) {
        a.embree_hits += embree[i];
        a.bonsai_hits += ours[i];
        if (embree[i] == ours[i]) {
            a.same++;
        } else {
            a.differ++;
        }
    }
    return a;
}

void usage() {
    std::cerr
        << "usage: rtq.out [options] <mesh.ply | mesh.ply.gz>\n"
           "  --side N       primary rays are an N x N image (default 1024)\n"
           "  --repeats N    timed runs per measurement, the least kept "
           "(default 5)\n"
           "  --threads N    threads for both sides (default 1: one plain loop; "
           "0: every core)\n"
           "  --embree-stats ask Embree to print its own tree's statistics\n";
    std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
    uint32_t side = 1024;
    int repeats = 5;
    int threads = 1;
    bool embree_stats = false;
    std::string path;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--side" && i + 1 < argc) {
            side = uint32_t(std::atoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--embree-stats") {
            embree_stats = true;
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
        } else {
            path = arg;
        }
    }
    if (path.empty() || side == 0 || repeats < 1) {
        usage();
    }
    // One thread needs no pool at all (for_blocks runs the loop in place), and
    // leaving TBB alone then keeps Embree's build, which is not timed, on
    // every core.
    std::unique_ptr<tbb::global_control> control;
    if (threads > 1) {
        control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, threads);
    }
    g_threads = threads;
    std::cout << "threads: " << (threads > 0 ? std::to_string(threads) : "all")
              << " on both sides; cpus allowed: " << allowed_cpus() << "\n";

    const Mesh mesh = load_ply(path);
    std::cout << "mesh: " << path << "\n  " << mesh.triangles() << " triangles, "
              << mesh.vertices.size() << " vertices\n";

    // Embree's defaults (an unconfigured device): the ISA the machine has,
    // the builder quality medium, the BVH8 over Triangle4 leaves.
    RTCDevice device = rtcNewDevice(embree_stats ? "verbose=2" : nullptr);
    if (device == nullptr) {
        std::cerr << "no Embree device\n";
        return 1;
    }
    rtcSetDeviceErrorFunction(device, embree_error, nullptr);
    RTCScene scene = make_embree_scene(device, mesh);
    const Tree tree = build_tree(device, mesh);
    std::cout << "tree: " << tree.nodes << " nodes (" << tree.nodes * sizeof(NodeRow)
              << " bytes), " << tree.leaves << " leaves holding " << tree.leaf_prims
              << " triangles in " << tree.blocks << " blocks of 4 ("
              << tree.blocks * sizeof(TriangleBlock) << " bytes)\n";

    // The ray sets: the camera's, then from what it saw.
    const std::vector<Ray> primary = primary_rays(mesh, side);
    std::vector<RTCRayHit> primary_hits;
    embree_intersect(scene, primary, primary_hits);
    const std::vector<Ray> ao = secondary_rays(primary, primary_hits, mesh, 0.1f, 1);
    const std::vector<Ray> diffuse = secondary_rays(primary, primary_hits, mesh, 0.0f, 2);

    struct Batch {
        const char *name;
        const std::vector<Ray> *rays;
    };
    const Batch batches[3] = {{"primary", &primary}, {"ao", &ao}, {"diffuse", &diffuse}};

    std::printf("\n%-10s %-10s %10s %12s %12s %10s  %s\n", "rays", "query", "count",
                "embree Mr/s", "bonsai Mr/s", "speedup", "agreement");
    bool all_agree = true;
    for (const Batch &batch : batches) {
        const std::vector<Ray> &rays = *batch.rays;
        if (rays.empty()) {
            continue;
        }
        std::vector<RTCRayHit> embree_hits;
        std::vector<uint32_t> embree_blocked, our_hits, our_blocked;

        const double te = timed(repeats, [&] { embree_intersect(scene, rays, embree_hits); });
        const double tb = timed(repeats, [&] { bonsai_intersect(tree, rays, our_hits); });
        const Agreement hit = compare_hits(mesh, rays, embree_hits, our_hits);
        std::printf("%-10s %-10s %10zu %12.2f %12.2f %9.2fx  %zu same, %zu ties, %zu differ "
                    "(embree hits %zu, bonsai %zu)\n",
                    batch.name, "intersect", rays.size(), rays.size() / te * 1e-6,
                    rays.size() / tb * 1e-6, te / tb, hit.same, hit.ties, hit.differ,
                    hit.embree_hits, hit.bonsai_hits);
        all_agree = all_agree && hit.differ == 0;

        const double oe = timed(repeats, [&] { embree_occluded(scene, rays, embree_blocked); });
        const double ob = timed(repeats, [&] { bonsai_occluded(tree, rays, our_blocked); });
        const Agreement occ = compare_occluded(embree_blocked, our_blocked);
        std::printf("%-10s %-10s %10zu %12.2f %12.2f %9.2fx  %zu same, %zu differ "
                    "(embree blocked %zu, bonsai %zu)\n",
                    batch.name, "occluded", rays.size(), rays.size() / oe * 1e-6,
                    rays.size() / ob * 1e-6, oe / ob, occ.same, occ.differ, occ.embree_hits,
                    occ.bonsai_hits);
        all_agree = all_agree && occ.differ == 0;
    }
    std::printf("\n%s\n", all_agree ? "every ray agrees with Embree (up to ties)"
                                    : "DISAGREEMENTS with Embree; see above");

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return all_agree ? 0 : 2;
}
