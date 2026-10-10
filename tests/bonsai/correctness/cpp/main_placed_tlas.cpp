// main_mesh_tlas.cpp's scene with object A made of two MESHES of one
// triangle each and placed twice as PLACED meshes (placed-tlas.bonsai: an
// instance's placement of one mesh of its object, the mesh's tree shared by
// both placements), object B kept as an instance over a tree, the
// standalone mesh (marker 7) and triangle as they were. The same eight
// rays: in lane z = +1 object A's second mesh at x = 0 has to win over the
// standalone mesh's triangle at x = 8, in lane z = -1 the standalone mesh's
// triangle at x = -3 wins over both placements, and a hit through a placed
// mesh reports its mesh's marker.
#include "placed-tlas.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t MAX_TREE_DEPTH = 64;

float3 min3(const float3 &a, const float3 &b) {
    return float3{std::fminf(a[0], b[0]), std::fminf(a[1], b[1]),
                  std::fminf(a[2], b[2])};
}
float3 max3(const float3 &a, const float3 &b) {
    return float3{std::fmaxf(a[0], b[0]), std::fmaxf(a[1], b[1]),
                  std::fmaxf(a[2], b[2])};
}

struct Box {
    float3 low;
    float3 high;
};

Box merge(const Box &a, const Box &b) {
    return Box{min3(a.low, b.low), max3(a.high, b.high)};
}

float3 centre(const Box &b) {
    return float3{0.5f * (b.low[0] + b.high[0]), 0.5f * (b.low[1] + b.high[1]),
                  0.5f * (b.low[2] + b.high[2])};
}

Box bounds_of(const Triangle &t) {
    return Box{min3(min3(t.p0, t.p1), t.p2), max3(max3(t.p0, t.p1), t.p2)};
}

float3 apply(const Transform &t, const float3 &p) {
    return t.m0 * p[0] + t.m1 * p[1] + t.m2 * p[2] + t.m3;
}

Transform rigid(const float3 &c0, const float3 &c1, const float3 &c2,
                const float3 &t) {
    Transform out;
    out.m0 = c0;
    out.m1 = c1;
    out.m2 = c2;
    out.m3 = t;
    out.i0 = float3{c0[0], c1[0], c2[0]};
    out.i1 = float3{c0[1], c1[1], c2[1]};
    out.i2 = float3{c0[2], c1[2], c2[2]};
    out.i3 = float3{-(c0[0] * t[0] + c0[1] * t[1] + c0[2] * t[2]),
                    -(c1[0] * t[0] + c1[1] * t[1] + c1[2] * t[2]),
                    -(c2[0] * t[0] + c2[1] * t[1] + c2[2] * t[2])};
    return out;
}

Transform translate(const float3 &t) {
    return rigid(float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f},
                 float3{0.0f, 0.0f, 1.0f}, t);
}

Transform rotate_x_quarter(const float3 &t) {
    return rigid(float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f},
                 float3{0.0f, -1.0f, 0.0f}, t);
}

template <typename Node>
void set_payload(Node &node, uint32_t value) {
    static_assert(sizeof(node.split0on_nPrims) == sizeof(uint32_t));
    std::memcpy(&node.split0on_nPrims, &value, sizeof(uint32_t));
}

// A median-split tree over `[low, high)` of `items`, appended to `nodes` and
// rooted at the row this returns; see main_blas_tlas.cpp.
template <typename Node, typename Item, typename BoundsOf>
uint32_t build_tree(std::vector<Node> &nodes, std::vector<Item> &items,
                    uint32_t low, uint32_t high, BoundsOf bounds_of_item,
                    uint32_t depth = 0) {
    assert(depth < MAX_TREE_DEPTH);
    const uint32_t self = uint32_t(nodes.size());
    nodes.emplace_back();
    Box box = bounds_of_item(items[low]);
    for (uint32_t i = low + 1; i < high; i++) {
        box = merge(box, bounds_of_item(items[i]));
    }
    nodes[self].low = {box.low[0], box.low[1], box.low[2]};
    nodes[self].high = {box.high[0], box.high[1], box.high[2]};
    if (high - low == 1) {
        nodes[self].nPrims = 1;
        nodes[self].axis = 0;
        set_payload(nodes[self], low);
        return self;
    }
    float3 lo = centre(bounds_of_item(items[low]));
    float3 hi = lo;
    for (uint32_t i = low + 1; i < high; i++) {
        const float3 c = centre(bounds_of_item(items[i]));
        lo = min3(lo, c);
        hi = max3(hi, c);
    }
    const float3 extent = hi - lo;
    int axis = 0;
    if (extent[1] > extent[0]) {
        axis = 1;
    }
    if (extent[2] > extent[axis]) {
        axis = 2;
    }
    const uint32_t mid = low + (high - low) / 2;
    std::nth_element(items.begin() + low, items.begin() + mid,
                     items.begin() + high,
                     [&](const Item &a, const Item &b) {
                         return centre(bounds_of_item(a))[axis] <
                                centre(bounds_of_item(b))[axis];
                     });
    nodes[self].nPrims = 0;
    nodes[self].axis = uint8_t(axis);
    build_tree(nodes, items, low, mid, bounds_of_item, depth + 1);
    const uint32_t right =
        build_tree(nodes, items, mid, high, bounds_of_item, depth + 1);
    set_payload(nodes[self], right - self);
    return self;
}

// A primitive as the generated constructor made it, and its world box, which
// is what the top-level tree is built from.
struct TopLevel {
    Prim prim;
    Box world;
};

void report(const char *name, const Ray &r, const _tree_layout0 &scene,
            const std::vector<MeshInfo> &infos, const std::vector<Placed> &placed_pool,
            const std::vector<Inst> &inst_pool) {
    std::cout << std::fixed << std::setprecision(2) << name << ": ";
    _option0 hit;
    trace(hit, r, scene, placed_pool.data(), inst_pool.data());
    if (hit.set) {
        const Prim &p = hit.value._field0;
        Triangle placed;
        transform_Prim_Triangle(placed, p, hit.value._field1, placed_pool.data(),
                                inst_pool.data());
        const int32_t kind = kind_of(p, placed_pool.data(), inst_pool.data());
        const char *kinds[4] = {"solo", "mesh", "instance", "placed"};
        std::cout << "hit " << kinds[kind] << " tri=(" << placed.p0[0] << ", "
                  << placed.p0[1] << ", " << placed.p0[2] << ")";
        if (kind == 1 || kind == 3) {
            std::cout << " marker="
                      << marker_of(p, infos.data(), placed_pool.data(), inst_pool.data());
        }
    } else {
        std::cout << "miss";
    }
    std::cout << " any="
              << (trace_any(r, scene, placed_pool.data(), inst_pool.data()) ? "yes"
                                                                              : "no")
              << '\n';
}

} // namespace

int main() {
    // Object B: one small triangle around the object-space origin, held by
    // an instance over a tree of it (`tris`, the BLAS's run).
    std::vector<Triangle> tris = {
        Triangle{float3{0.0f, -0.25f, -0.25f}, float3{0.0f, 0.25f, -0.25f},
                 float3{0.0f, 0.0f, 0.25f}},
    };
    std::vector<_tree_layout1> blas_nodes;
    const uint32_t object_b = build_tree(blas_nodes, tris, 0, 1, bounds_of);
    // The meshes' triangles: object A's two meshes, one triangle each in
    // the x = 0 plane, one per z lane (meshes 0 and 1), then the standalone
    // mesh's two (mesh 2: x = -3 in lane z = -1, x = 8 in lane z = +1).
    std::vector<Triangle> mesh_tris = {
        Triangle{float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f},
                 float3{0.0f, 0.0f, -0.5f}},
        Triangle{float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f},
                 float3{0.0f, 0.0f, 1.5f}},
        Triangle{float3{-3.0f, -1.0f, -1.5f}, float3{-3.0f, 1.0f, -1.5f},
                 float3{-3.0f, 0.0f, -0.5f}},
        Triangle{float3{8.0f, -1.0f, 0.5f}, float3{8.0f, 1.0f, 0.5f},
                 float3{8.0f, 0.0f, 1.5f}},
    };
    std::vector<_tree_layout4> mesh_nodes;
    const uint32_t mesh_a0 = build_tree(mesh_nodes, mesh_tris, 0, 1, bounds_of);
    const uint32_t mesh_a1 = build_tree(mesh_nodes, mesh_tris, 1, 2, bounds_of);
    const uint32_t mesh_standalone = build_tree(mesh_nodes, mesh_tris, 2, 4, bounds_of);
    std::vector<MeshInfo> infos = {MeshInfo{1}, MeshInfo{2}, MeshInfo{7}};
    const auto placed_box = [&](const Transform &t, uint32_t first, uint32_t count) {
        Box box{apply(t, mesh_tris[first].p0), apply(t, mesh_tris[first].p0)};
        for (uint32_t i = first; i < first + count; i++) {
            for (const float3 &p : {mesh_tris[i].p0, mesh_tris[i].p1, mesh_tris[i].p2}) {
                const float3 w = apply(t, p);
                box = merge(box, Box{w, w});
            }
        }
        return box;
    };
    std::vector<Placed> placed_pool(4);
    uint64_t placed_fill = 0;
    std::vector<Inst> inst_pool(1);
    uint64_t inst_fill = 0;
    std::vector<TopLevel> top;
    // Two placements of object A, one behind the other in both lanes: a
    // Placed primitive per mesh, the mesh's tree shared.
    for (const float3 &at : {float3{0.0f, 0.0f, 0.0f}, float3{5.0f, 0.0f, 0.0f}}) {
        const Transform t = translate(at);
        Prim p0;
        Prim_Placed(p0, t, 0, mesh_a0, placed_pool.data(), &placed_fill);
        top.push_back(TopLevel{p0, placed_box(t, 0, 1)});
        Prim p1;
        Prim_Placed(p1, t, 1, mesh_a1, placed_pool.data(), &placed_fill);
        top.push_back(TopLevel{p1, placed_box(t, 1, 1)});
    }
    {
        // Object B turned a quarter turn about x.
        const Transform t = rotate_x_quarter(float3{0.0f, 0.0f, 20.0f});
        Prim p;
        Prim_Inst(p, t, object_b, inst_pool.data(), &inst_fill);
        Box box{apply(t, tris[0].p0), apply(t, tris[0].p0)};
        for (const float3 &q : {tris[0].p0, tris[0].p1, tris[0].p2}) {
            const float3 w = apply(t, q);
            box = merge(box, Box{w, w});
        }
        top.push_back(TopLevel{p, box});
    }
    {
        Prim p;
        Prim_Mesh(p, 2, mesh_standalone);
        top.push_back(TopLevel{p, merge(bounds_of(mesh_tris[2]), bounds_of(mesh_tris[3]))});
    }
    {
        const Triangle t{float3{2.0f, -1.0f, 29.5f}, float3{2.0f, 1.0f, 29.5f},
                         float3{2.0f, 0.0f, 30.5f}};
        Prim p;
        Prim_Solo(p, t);
        top.push_back(TopLevel{p, bounds_of(t)});
    }
    assert(placed_fill == placed_pool.size());
    assert(inst_fill == inst_pool.size());
    std::vector<_tree_layout7> tlas_nodes;
    build_tree(tlas_nodes, top, 0, uint32_t(top.size()),
               [](const TopLevel &p) { return p.world; });
    std::vector<Prim> elems;
    elems.reserve(top.size());
    for (const TopLevel &p : top) {
        elems.push_back(p.prim);
    }
    bonsai_buffer tris_buffer =
        bonsai_buffer_wrap(tris.data(), tris.size() * sizeof(tris[0]));
    bonsai_buffer blas_buffer = bonsai_buffer_wrap(
        blas_nodes.data(), blas_nodes.size() * sizeof(blas_nodes[0]));
    bonsai_buffer mtris_buffer = bonsai_buffer_wrap(
        mesh_tris.data(), mesh_tris.size() * sizeof(mesh_tris[0]));
    bonsai_buffer mesh_buffer = bonsai_buffer_wrap(
        mesh_nodes.data(), mesh_nodes.size() * sizeof(mesh_nodes[0]));
    bonsai_buffer elems_buffer =
        bonsai_buffer_wrap(elems.data(), elems.size() * sizeof(elems[0]));
    bonsai_buffer tlas_buffer = bonsai_buffer_wrap(
        tlas_nodes.data(), tlas_nodes.size() * sizeof(tlas_nodes[0]));
    _tree_layout0 scene;
    scene.tCount = uint32_t(tris.size());
    scene.tris = &tris_buffer;
    scene.bCount = uint32_t(blas_nodes.size());
    scene.group0_bnode = &blas_buffer;
    scene.mtCount = uint32_t(mesh_tris.size());
    scene.mtris = &mtris_buffer;
    scene.mCount = uint32_t(mesh_nodes.size());
    scene.group1_mnode = &mesh_buffer;
    scene.pCount = uint32_t(elems.size());
    scene.elems = &elems_buffer;
    scene.nCount = uint32_t(tlas_nodes.size());
    scene.group2_index = &tlas_buffer;

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };
    // Lane z = -1: the standalone mesh's triangle at x = -3 is in front of
    // object A's first mesh at x = 0 and x = 5.
    report("mesh in front ", ray(0.0f, -1.0f), scene, infos, placed_pool, inst_pool);
    // Lane z = +1: object A's second mesh at x = 0 is in front of its
    // second placement and of the standalone mesh's triangle at x = 8: the
    // Placed arm wins over both.
    report("placed in front", ray(0.0f, 1.0f), scene, infos, placed_pool, inst_pool);
    report("solo alone    ", ray(0.0f, 30.0f), scene, infos, placed_pool, inst_pool);
    report("turned        ", ray(0.2f, 20.0f), scene, infos, placed_pool, inst_pool);
    report("unturned      ", ray(0.0f, 20.2f), scene, infos, placed_pool, inst_pool);
    report("wrong object  ", ray(0.8f, 20.0f), scene, infos, placed_pool, inst_pool);
    report("empty lane    ", ray(0.0f, 40.0f), scene, infos, placed_pool, inst_pool);
    report("past the top  ", ray(5.0f, 0.0f), scene, infos, placed_pool, inst_pool);
}
