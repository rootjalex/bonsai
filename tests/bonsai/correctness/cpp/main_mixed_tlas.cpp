// One driver, three schedules, one set of answers. mixed-tlas-loopified.bonsai
// is mixed-tlas.bonsai with a loopify() on each traversal and
// mixed-tlas-sorted.bonsai adds a front-to-back sort() to each; both change
// how the set is visited and nothing about what is in it, so all three have to
// print the same thing.
//
// The scene mixes the two kinds of primitive pbrt's top-level tree holds:
// triangles standing on their own (GeometricPrimitive) and instances holding a
// tree of them (TransformedPrimitive). What is worth running is the running
// best crossing between the two arms: a standalone triangle in front of an
// instance's triangle in the same lane, and an instance's triangle in front of
// a standalone one in another, each of which the other arm has to lose to.
#ifdef PER_ARM
#include "mixed-tlas-per-arm.h"
#elif defined(VECTORIZED)
#include "vectorize_tree_traversal_mixed.h"
#elif defined(SORTED)
#include "mixed-tlas-sorted.h"
#elif defined(LOOPIFIED)
#include "mixed-tlas-loopified.h"
#else
#include "mixed-tlas.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

// Used in mixed-tlas.bonsai.
//
// Two objects, three instances, three standalone triangles, one pool of tree
// nodes. Object A is two triangles in the x = 0 plane, in lanes at z = -1 and
// z = +1; object B is a small triangle around the origin. Two instances place
// object A, at x = 0 and at x = 5, and one places object B rotated a quarter
// turn about x at z = 20. The standalone triangles sit at x = -3 in lane
// z = -1 (in front of both placements of A), at x = 8 in lane z = +1 (behind
// both), and at x = 2 in lane z = 30 (alone).
//
// Every ray is fired along +x from x = -10.
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

// A rotation given by the images of the three basis vectors, and a
// translation, with the inverse stored beside it the way pbrt's Transform
// does.
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

// A quarter turn about x: y goes to z and z goes to -y.
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

// What a `Prim` is to the driver depends on the layout the schedule chose.
// Under `tagged_index` it is a handle -- a `uint64_t` naming a pool and a row
// -- and both arms have a pool. Under the per-arm layout it is a struct, the
// Solo arm's fields in the value and only the Inst arm behind a pool. The
// generated constructors are the same two calls either way, with the pools
// they need, which is the point of using them rather than assembling values
// here.
#ifdef PER_ARM
using PrimValue = Prim;
#define PRIM_POOLS(pools) (pools).inst.data()
#else
using PrimValue = uint64_t;
#define PRIM_POOLS(pools) (pools).solo.data(), (pools).inst.data()
#endif

struct Pools {
    std::vector<Solo> solo;
    std::vector<Inst> inst;
    uint64_t solo_fill = 0;
    uint64_t inst_fill = 0;
};

PrimValue make_solo(const Triangle &tri, Pools &pools) {
#ifdef PER_ARM
    Prim out;
    Prim_Solo(out, tri);
    return out;
#else
    return Prim_Solo(tri, pools.solo.data(), &pools.solo_fill);
#endif
}

PrimValue make_inst(const Transform &t, uint32_t blas, Pools &pools) {
#ifdef PER_ARM
    Prim out;
    Prim_Inst(out, t, blas, pools.inst.data(), &pools.inst_fill);
    return out;
#else
    return Prim_Inst(t, blas, pools.inst.data(), &pools.inst_fill);
#endif
}

// A primitive -- as the generated constructor returned it -- and its
// world-space box, which is what the top-level tree is built from. `with
// extent` says the same thing in the schedule; nothing evaluates it at
// traversal time, so the driver is what has to agree with it.
struct Placed {
    PrimValue prim;
    Box world;
};

void report(const char *name, const Ray &r, const _tree_layout0 &scene,
            const Pools &pools) {
    std::cout << std::fixed << std::setprecision(2) << name << ": ";
    _option0 hit;
    trace(hit, r, scene, PRIM_POOLS(pools));
    if (hit.set) {
        const PrimValue &p = hit.value._field0;
        // The query hands back the hit as stored: in the instance's frame for
        // one that came through an instance, which is what pbrt's inner
        // Intersect returns to TransformedPrimitive. `transform(p, tri)` is
        // TransformedPrimitive's one transform of the result -- and
        // GeometricPrimitive's none -- dispatched by the program.
        Triangle placed;
        transform_Prim_Triangle(placed, p, hit.value._field1, PRIM_POOLS(pools));
        const bool inst = is_instance(p, PRIM_POOLS(pools));
        std::cout << "hit " << (inst ? "instance" : "solo") << " tri=("
                  << placed.p0[0] << ", " << placed.p0[1] << ", "
                  << placed.p0[2] << ")";
    } else {
        std::cout << "miss";
    }
    std::cout << " any="
              << (trace_any(r, scene, PRIM_POOLS(pools)) ? "yes" : "no")
              << '\n';
}

} // namespace

int main() {
    // Object A: two triangles in the x = 0 plane, one in each z lane.
    // Object B: one small triangle around the object-space origin.
    std::vector<Triangle> tris = {
        Triangle{float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f},
                 float3{0.0f, 0.0f, -0.5f}},
        Triangle{float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f},
                 float3{0.0f, 0.0f, 1.5f}},
        Triangle{float3{0.0f, -0.25f, -0.25f}, float3{0.0f, 0.25f, -0.25f},
                 float3{0.0f, 0.0f, 0.25f}},
    };

    std::vector<_tree_layout1> blas_nodes;
    const uint32_t object_a = build_tree(blas_nodes, tris, 0, 2, bounds_of);
    const uint32_t object_b = build_tree(blas_nodes, tris, 2, 3, bounds_of);

    // The world box of an instance: the triangles its tree reaches, each where
    // the instance puts them.
    auto place = [&](const Transform &t, uint32_t first, uint32_t count) {
        Box box{apply(t, tris[first].p0), apply(t, tris[first].p0)};
        for (uint32_t i = first; i < first + count; i++) {
            for (const float3 &p : {tris[i].p0, tris[i].p1, tris[i].p2}) {
                const float3 w = apply(t, p);
                box = merge(box, Box{w, w});
            }
        }
        return box;
    };

    // The pools a `tagged_index` variant stores its arms in are the driver's,
    // sized for what it is about to build: three instances and three
    // standalone triangles.
    Pools pools;
    pools.solo.resize(3);
    pools.inst.resize(3);

    std::vector<Placed> placed;
    // Two placements of object A, both naming the row its tree is rooted at,
    // one behind the other in both lanes.
    for (const float3 &at : {float3{0.0f, 0.0f, 0.0f}, float3{5.0f, 0.0f, 0.0f}}) {
        const Transform t = translate(at);
        placed.push_back(Placed{make_inst(t, object_a, pools), place(t, 0, 2)});
    }
    // Object B, turned a quarter turn about x.
    {
        const Transform t = rotate_x_quarter(float3{0.0f, 0.0f, 20.0f});
        placed.push_back(Placed{make_inst(t, object_b, pools), place(t, 2, 1)});
    }
    // The standalone triangles, in world space already. The first sits in
    // lane z = -1 ahead of both placements of A; the second in lane z = +1
    // behind both; the third alone in lane z = 30.
    for (const Triangle &t :
         {Triangle{float3{-3.0f, -1.0f, -1.5f}, float3{-3.0f, 1.0f, -1.5f},
                   float3{-3.0f, 0.0f, -0.5f}},
          Triangle{float3{8.0f, -1.0f, 0.5f}, float3{8.0f, 1.0f, 0.5f},
                   float3{8.0f, 0.0f, 1.5f}},
          Triangle{float3{2.0f, -1.0f, 29.5f}, float3{2.0f, 1.0f, 29.5f},
                   float3{2.0f, 0.0f, 30.5f}}}) {
        placed.push_back(Placed{make_solo(t, pools), bounds_of(t)});
    }
#ifndef PER_ARM
    assert(pools.solo_fill == pools.solo.size());
#endif
    assert(pools.inst_fill == pools.inst.size());

    std::vector<_tree_layout4> tlas_nodes;
    build_tree(tlas_nodes, placed, 0, uint32_t(placed.size()),
               [](const Placed &p) { return p.world; });

    std::vector<PrimValue> elems;
    elems.reserve(placed.size());
    for (const Placed &p : placed) {
        elems.push_back(p.prim);
    }

    _tree_layout0 scene;
    scene.tCount = uint32_t(tris.size());
    scene.tris = tris.data();
    scene.bCount = uint32_t(blas_nodes.size());
    scene.group0_bnode = blas_nodes.data();
    scene.pCount = uint32_t(elems.size());
    scene.elems = elems.data();
    scene.nCount = uint32_t(tlas_nodes.size());
    scene.group1_index = tlas_nodes.data();

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };

    // Lane z = -1: the standalone triangle at x = -3 is in front of object A
    // at x = 0 and at x = 5. The Solo arm has to win over the Inst arm.
    report("solo in front ", ray(0.0f, -1.0f), scene, pools);
    // Lane z = +1: object A at x = 0 is in front of its second placement at
    // x = 5 and of the standalone triangle at x = 8. The Inst arm has to win
    // over both the other instance and the Solo arm.
    report("inst in front ", ray(0.0f, 1.0f), scene, pools);
    // Nothing but a standalone triangle in this lane.
    report("solo alone    ", ray(0.0f, 30.0f), scene, pools);
    // Object B, hit only because it is rotated...
    report("turned        ", ray(0.2f, 20.0f), scene, pools);
    // ...and missed for the same reason.
    report("unturned      ", ray(0.0f, 20.2f), scene, pools);
    // Well outside object B, but inside object A: reached only by a walk that
    // began at the wrong root.
    report("wrong object  ", ray(0.8f, 20.0f), scene, pools);
    // Nothing in this lane at all.
    report("empty lane    ", ray(0.0f, 40.0f), scene, pools);
    report("past the top  ", ray(5.0f, 0.0f), scene, pools);

#ifdef VECTORIZED
    // The same eight rays traced together, one per lane, each checked against
    // the same ray traced alone (see vectorize_tree_traversal_mixed.bonsai).
    const std::array<Ray, 8> rays = {
        ray(0.0f, -1.0f), ray(0.0f, 1.0f),  ray(0.0f, 30.0f), ray(0.2f, 20.0f),
        ray(0.0f, 20.2f), ray(0.8f, 20.0f), ray(0.0f, 40.0f), ray(5.0f, 0.0f)};
    std::array<float, 8> out{};
    hit_x_all(rays, out, scene, PRIM_POOLS(pools));
    bool same = true;
    for (size_t i = 0; i < rays.size(); i++) {
        same = same && hit_x_one(rays[i], scene, PRIM_POOLS(pools)) == out[i];
        std::cout << (i == 0 ? "" : " ") << out[i];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
#endif
}
