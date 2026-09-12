// One driver, three schedules, one set of answers. blas-tlas-loopified.bonsai
// is blas-tlas.bonsai with a loopify() on each traversal and
// blas-tlas-sorted.bonsai adds a front-to-back sort() to each; both change how
// the set is visited and nothing about what is in it, so all three have to
// print the same thing.
//
// A tree inside a tree is what makes them worth running. There are two
// recursions, so two stacks and two orderings, and mixing up either pair is
// the mistake this catches -- a walk of an instance's triangles whose node
// references went onto the stack of the walk over instances comes back with
// somebody else's geometry rather than with an error.
#ifdef SORTED
#include "blas-tlas-sorted.h"
#elif defined(LOOPIFIED)
#include "blas-tlas-loopified.h"
#else
#include "blas-tlas.h"
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

// Used in blas-tlas.bonsai.
//
// Two objects, four instances, one pool. Object A is two triangles in the
// x = 0 plane, in lanes at z = -1 and z = +1; object B is a small triangle
// around the origin. Three of the instances place object A -- and so name the
// same row of the node pool, which is the whole point of storing a tree
// separately from the elements that reach it -- and the fourth places object B
// rotated a quarter turn about x.
//
// Every ray is fired along +x from x = -10, so each one has one obvious
// answer, and the interesting ones are the last few: a ray aimed just off B's
// centre hits only if the rotation was applied, one aimed at where B would be
// unrotated hits only if it was not, and one aimed a long way out hits only if
// the walk of B's instance began at object A's root instead of B's.
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
// translation. The inverse is stored beside it the way pbrt's Transform does,
// which for an orthonormal rotation is the transpose and the translation
// pulled back through it.
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

// Where a node's arm keeps its one extra field. Both arms are four bytes wide,
// so which one this is depends on nPrims, exactly as the layout says.
template <typename Node>
void set_payload(Node &node, uint32_t value) {
    static_assert(sizeof(node.split0on_nPrims) == sizeof(uint32_t));
    std::memcpy(&node.split0on_nPrims, &value, sizeof(uint32_t));
}

// A median-split tree over `[low, high)` of `items`, appended to `nodes` and
// rooted at the row this returns. One item per leaf, so the tree is as deep as
// the traversal will ever have to go.
//
// `nodes` is shared: every object's tree is built into the same pool, and a
// child reference is an offset from the parent rather than an absolute row, so
// a subtree does not care where in the pool it landed.
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

// An instance and the world-space box of everything it places, which is what
// the tree over instances is built from. `with extent` says the same thing in
// the schedule; nothing evaluates it at traversal time, so the driver is what
// has to agree with it.
struct Placed {
    Instance instance;
    Box world;
};

void report(const char *name, const Ray &r, const _tree_layout0 &scene) {
    std::cout << std::fixed << std::setprecision(2) << name << ": ";
    _option0 hit;
    trace(hit, r, scene);
    if (hit.set) {
        const Instance &i = hit.value._field0;
        const Triangle &t = hit.value._field1;
        // The query hands back the hit as the instance's tree holds it -- in
        // the instance's frame -- which is what pbrt's inner Intersect returns
        // to TransformedPrimitive. This is TransformedPrimitive's one
        // transform of the result, done here so that the line printed is
        // where the triangle actually is.
        const float3 p0 = apply(i.render_from_instance, t.p0);
        std::cout << "hit blas=" << i.blas << " at (" << i.render_from_instance.m3[0]
                  << ", " << i.render_from_instance.m3[1] << ", "
                  << i.render_from_instance.m3[2] << ")"
                  << " tri=(" << p0[0] << ", " << p0[1] << ", " << p0[2]
                  << ")";
    } else {
        std::cout << "miss";
    }
    std::cout << " any=" << (trace_any(r, scene) ? "yes" : "no") << '\n';
}

} // namespace

int main() {
    // Object A: two triangles in the x = 0 plane, one in each z lane. Two of
    // them so its tree has an interior node, and disjoint so no ray lands on
    // the seam between them.
    //
    // Object B: one small triangle around the object-space origin.
    std::vector<Triangle> tris = {
        Triangle{float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f},
                 float3{0.0f, 0.0f, -0.5f}},
        Triangle{float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f},
                 float3{0.0f, 0.0f, 1.5f}},
        Triangle{float3{0.0f, -0.25f, -0.25f}, float3{0.0f, 0.25f, -0.25f},
                 float3{0.0f, 0.0f, 0.25f}},
    };

    // The layout says three floats twice, a u16, a u8, a byte of padding and
    // a u32: 32 bytes, and it lowers to exactly that. A vec3f in storage is
    // twelve bytes, not the sixteen a float3 register is.
    static_assert(sizeof(_tree_layout1) == 32,
                  "a BLAS node is the 32 bytes its layout says");
    std::vector<_tree_layout1> blas_nodes;
    const uint32_t object_a = build_tree(blas_nodes, tris, 0, 2, bounds_of);
    const uint32_t object_b = build_tree(blas_nodes, tris, 2, 3, bounds_of);

    // The world box of an instance, which is what its `with extent` says: the
    // triangles its tree reaches, each where the instance puts them.
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

    std::vector<Placed> placed;
    // Three placements of object A, all naming the one row its tree is rooted
    // at. Two of them share a lane, one behind the other, so the nearer has to
    // win.
    for (const float3 &at : {float3{0.0f, 0.0f, 0.0f}, float3{5.0f, 0.0f, 0.0f},
                             float3{0.0f, 0.0f, 10.0f}}) {
        const Transform t = translate(at);
        placed.push_back(Placed{Instance{t, object_a}, place(t, 0, 2)});
    }
    // And object B, turned a quarter turn about x.
    {
        const Transform t = rotate_x_quarter(float3{0.0f, 0.0f, 20.0f});
        placed.push_back(Placed{Instance{t, object_b}, place(t, 2, 1)});
    }

    std::vector<_tree_layout4> tlas_nodes;
    build_tree(tlas_nodes, placed, 0, uint32_t(placed.size()),
               [](const Placed &p) { return p.world; });

    std::vector<Instance> insts;
    insts.reserve(placed.size());
    for (const Placed &p : placed) {
        insts.push_back(p.instance);
    }

    _tree_layout0 scene;
    scene.tCount = uint32_t(tris.size());
    scene.tris = tris.data();
    scene.bCount = uint32_t(blas_nodes.size());
    scene.group0_bnode = blas_nodes.data();
    scene.iCount = uint32_t(insts.size());
    scene.insts = insts.data();
    scene.nCount = uint32_t(tlas_nodes.size());
    scene.group1_index = tlas_nodes.data();

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };

    // Both lanes of the instance at the origin. The instance at x = 5 places
    // the same two triangles in the same lanes and must lose to it.
    report("near lane -1", ray(0.0f, -1.0f), scene);
    report("near lane +1", ray(0.0f, 1.0f), scene);
    // The third placement of object A.
    report("far lane -1 ", ray(0.0f, 9.0f), scene);
    report("far lane +1 ", ray(0.0f, 11.0f), scene);
    // Object B, hit only because it is rotated...
    report("turned      ", ray(0.2f, 20.0f), scene);
    // ...and missed for the same reason.
    report("unturned    ", ray(0.0f, 20.2f), scene);
    // Well outside object B, but inside object A: reached only by a walk that
    // began at the wrong root.
    report("wrong object", ray(0.8f, 20.0f), scene);
    // Nothing in this lane at all.
    report("empty lane  ", ray(0.0f, 40.0f), scene);
    report("past the top", ray(5.0f, 0.0f), scene);
}
