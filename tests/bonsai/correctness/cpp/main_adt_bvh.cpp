#include "adt-bvh.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// Used in adt-bvh.bonsai.
//
// The scene is arranged in lanes along z so that every ray has one obvious
// answer: a ray fired along +x at a given z either hits the one shape sitting
// at that z or nothing at all. A traversal that descended the wrong child, or
// a variant read at the wrong type, gives a different shape rather than a
// slightly different number.
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

float length3(const float3 &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// The bounding volume the schedule asked for: `with Sphere(center, radius)`.
Sphere bounds_of(const Shape &shape) {
    if (shape.tag == 0) {
        return shape.payload.Sph.s;
    }
    const Triangle &t = shape.payload.Tri.t;
    const float3 centre = float3{(t.p0[0] + t.p1[0] + t.p2[0]) / 3.0f,
                                 (t.p0[1] + t.p1[1] + t.p2[1]) / 3.0f,
                                 (t.p0[2] + t.p1[2] + t.p2[2]) / 3.0f};
    const float radius =
        std::max({length3(t.p0 - centre), length3(t.p1 - centre),
                  length3(t.p2 - centre)});
    return Sphere{centre, radius};
}

Sphere merge(const Sphere &a, const Sphere &b) {
    const float3 d = b.center - a.center;
    const float dist = length3(d);
    if (a.radius >= dist + b.radius) {
        return a;
    }
    if (b.radius >= dist + a.radius) {
        return b;
    }
    const float radius = 0.5f * (dist + a.radius + b.radius);
    const float3 dir = (dist > 0.0f)
                           ? float3{d[0] / dist, d[1] / dist, d[2] / dist}
                           : float3{1.0f, 0.0f, 0.0f};
    const float scale = radius - a.radius;
    return Sphere{a.center +
                      float3{dir[0] * scale, dir[1] * scale, dir[2] * scale},
                  radius};
}

// A median-split BVH in the layout the schedule declared, built the same way
// apps/rtiow does it. One primitive per leaf, so the tree is as deep as the
// traversal will ever have to go for this many shapes.
_tree_layout0 build_tree(std::vector<Shape> &shapes) {
    _tree_layout0 tree;
    tree.pCount = uint32_t(shapes.size());
    tree.prims = shapes.data();

    const size_t leaf_count = tree.pCount;
    tree.nCount = uint32_t(leaf_count + (leaf_count - 1));
    tree.group0_index =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.nCount);

    uint32_t next_node = 0;
    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        const uint32_t count = high - low;
        const uint32_t self = next_node++;

        if (count == 1) {
            tree.group0_index[self].nPrims = 1;
            *reinterpret_cast<uint16_t *>(
                &tree.group0_index[self].split0on_nPrims) = uint16_t(low);
            const Sphere b = bounds_of(shapes[low]);
            tree.group0_index[self].center = b.center;
            tree.group0_index[self].radius = b.radius;
            return self;
        }

        tree.group0_index[self].nPrims = 0;

        float3 lo = bounds_of(shapes[low]).center;
        float3 hi = lo;
        for (uint32_t i = low + 1; i < high; i++) {
            const float3 c = bounds_of(shapes[i]).center;
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
        tree.group0_index[self].axis = uint8_t(axis);

        const uint32_t mid = low + count / 2;
        std::nth_element(
            shapes.begin() + low, shapes.begin() + mid, shapes.begin() + high,
            [axis](const Shape &a, const Shape &b) {
                return bounds_of(a).center[axis] < bounds_of(b).center[axis];
            });

        const uint32_t left = handle_range(low, mid, depth + 1);
        const uint32_t right = handle_range(mid, high, depth + 1);
        *reinterpret_cast<uint16_t *>(
            &tree.group0_index[self].split0on_nPrims) = uint16_t(right - self);

        const Sphere merged = merge(Sphere{tree.group0_index[left].center,
                                           tree.group0_index[left].radius},
                                    Sphere{tree.group0_index[right].center,
                                           tree.group0_index[right].radius});
        tree.group0_index[self].center = merged.center;
        tree.group0_index[self].radius = merged.radius;
        return self;
    };

    handle_range(0, tree.pCount, 0);
    return tree;
}

// What was hit, in a form that is the same however the scene got sorted.
void report(const Ray &r, const _tree_layout0 &tree) {
    _option0 hit;
    trace(hit, r, tree);
    if (!hit.set) {
        std::cout << "miss\n";
        return;
    }
    if (hit.value.tag == 0) {
        std::cout << "sphere at z=" << hit.value.payload.Sph.s.center[2]
                  << '\n';
    } else {
        std::cout << "triangle at z=" << hit.value.payload.Tri.t.p2[2] << '\n';
    }
}

} // namespace

int main() {
    std::vector<Shape> shapes;

    // Built through the generated constructors rather than by setting the tag
    // by hand: which number a variant is belongs to the compiler, and a driver
    // that spelled it out would go quietly wrong the day a schedule chose a
    // different layout.
    Shape a;
    Shape_Sph(a, Sphere{float3{0.0f, 0.0f, 0.0f}, 1.0f});
    shapes.push_back(a);

    Shape b;
    Shape_Sph(b, Sphere{float3{0.0f, 0.0f, 10.0f}, 1.0f});
    shapes.push_back(b);

    // Facing the rays, centred on the lane at z=20.
    Shape c;
    Shape_Tri(c, Triangle{float3{0.0f, -1.0f, 19.0f}, float3{0.0f, 1.0f, 19.0f},
                          float3{0.0f, 0.0f, 21.0f}});
    shapes.push_back(c);

    Shape d;
    Shape_Sph(d, Sphere{float3{0.0f, 0.0f, 30.0f}, 1.0f});
    shapes.push_back(d);

    _tree_layout0 tree = build_tree(shapes);

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    report(Ray{float3{-10.0f, 0.0f, 0.0f}, forward}, tree);
    report(Ray{float3{-10.0f, 0.0f, 10.0f}, forward}, tree);
    report(Ray{float3{-10.0f, 0.0f, 20.0f}, forward}, tree);
    report(Ray{float3{-10.0f, 0.0f, 30.0f}, forward}, tree);
    // No lane here.
    report(Ray{float3{-10.0f, 0.0f, 50.0f}, forward}, tree);

    free(tree.group0_index);
}
