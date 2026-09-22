#include "vectorize_tree_traversal.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

// Used in vectorize_tree_traversal.bonsai.
//
// The scene and the tree builder are main_adt_bvh.cpp's: shapes in lanes
// along z, one per leaf, so that a ray fired along +x at a given z has one
// obvious answer. Eight rays, at heights that hit each kind of shape, miss,
// and repeat, are traced together and each compared with the same ray traced
// alone.
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

// The payload is the variant's fields as 32-bit words, at the offsets C
// gives them (see Lower/WordStorage.h): a float3 takes four words, the
// fourth its padding, as the C++ vector type does, so a Sphere is its centre
// in words 0 to 2 and its radius in word 4, and a Triangle its three points
// at words 0, 4 and 8.
float word(const Shape &shape, size_t k) {
    float f;
    std::memcpy(&f, &shape.payload[k], sizeof f);
    return f;
}

float3 point(const Shape &shape, size_t k) {
    return float3{word(shape, k), word(shape, k + 1), word(shape, k + 2)};
}

Sphere sphere_of(const Shape &shape) {
    return Sphere{point(shape, 0), word(shape, 4)};
}

Triangle triangle_of(const Shape &shape) {
    return Triangle{point(shape, 0), point(shape, 4), point(shape, 8)};
}

Sphere bounds_of(const Shape &shape) {
    if (shape.tag == 0) {
        return sphere_of(shape);
    }
    const Triangle t = triangle_of(shape);
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

static float3 widen(const std::array<float, 3> &a) {
    return float3{a[0], a[1], a[2]};
}

static std::array<float, 3> pack(const float3 &v) {
    return {v[0], v[1], v[2]};
}

_tree_layout0 build_tree(std::vector<Shape> &shapes) {
    // The layout's arrays reach the program as buffer descriptors
    // (runtime/bonsai_buffer.h), which have to outlive the tree.
    _tree_layout0 tree;
    tree.pCount = uint32_t(shapes.size());
    static bonsai_buffer prims_buffer;
    prims_buffer = bonsai_buffer_wrap(shapes.data(), shapes.size() * sizeof(Shape));
    tree.prims = &prims_buffer;

    const size_t leaf_count = tree.pCount;
    tree.nCount = uint32_t(leaf_count + (leaf_count - 1));
    _tree_layout1 *nodes =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.nCount);
    static bonsai_buffer nodes_buffer;
    nodes_buffer = bonsai_buffer_wrap(nodes, sizeof(_tree_layout1) * tree.nCount);
    tree.group0_index = &nodes_buffer;

    uint32_t next_node = 0;
    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        const uint32_t count = high - low;
        const uint32_t self = next_node++;

        // Leaves of one or two primitives, unlike main_adt_bvh.cpp's one: the
        // loop over a leaf's primitives then runs a different number of times
        // in different lanes, which is how pbrt's leaves are and what a
        // gang's masked walk has to get right.
        if (count <= 2) {
            nodes[self].nPrims = uint8_t(count);
            *reinterpret_cast<uint16_t *>(
                &nodes[self].split0on_nPrims) = uint16_t(low);
            Sphere b = bounds_of(shapes[low]);
            for (uint32_t i = low + 1; i < high; i++) {
                b = merge(b, bounds_of(shapes[i]));
            }
            nodes[self].center = pack(b.center);
            nodes[self].radius = b.radius;
            return self;
        }

        nodes[self].nPrims = 0;

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
        nodes[self].axis = uint8_t(axis);

        const uint32_t mid = low + count / 2;
        std::nth_element(
            shapes.begin() + low, shapes.begin() + mid, shapes.begin() + high,
            [axis](const Shape &a, const Shape &b) {
                return bounds_of(a).center[axis] < bounds_of(b).center[axis];
            });

        const uint32_t left = handle_range(low, mid, depth + 1);
        const uint32_t right = handle_range(mid, high, depth + 1);
        *reinterpret_cast<uint16_t *>(
            &nodes[self].split0on_nPrims) = uint16_t(right - self);

        const Sphere merged =
            merge(Sphere{widen(nodes[left].center),
                         nodes[left].radius},
                  Sphere{widen(nodes[right].center),
                         nodes[right].radius});
        nodes[self].center = pack(merged.center);
        nodes[self].radius = merged.radius;
        return self;
    };

    handle_range(0, tree.pCount, 0);
    return tree;
}

} // namespace

int main() {
    std::vector<Shape> shapes;
    Shape a;
    Shape_Sph(a, Sphere{float3{0.0f, 0.0f, 0.0f}, 1.0f});
    shapes.push_back(a);
    Shape b;
    Shape_Sph(b, Sphere{float3{0.0f, 0.0f, 10.0f}, 1.0f});
    shapes.push_back(b);
    Shape c;
    Shape_Tri(c, Triangle{float3{0.0f, -1.0f, 19.0f}, float3{0.0f, 1.0f, 19.0f},
                          float3{0.0f, 0.0f, 21.0f}});
    shapes.push_back(c);
    Shape d;
    Shape_Sph(d, Sphere{float3{0.0f, 0.0f, 30.0f}, 1.0f});
    shapes.push_back(d);
    // Two more lanes, so the tree has leaves of two shapes beside leaves of
    // one, and one lane's leaf loop runs twice where another's runs once.
    Shape e;
    Shape_Sph(e, Sphere{float3{0.0f, 0.0f, 40.0f}, 1.0f});
    shapes.push_back(e);
    Shape f;
    Shape_Tri(f, Triangle{float3{0.0f, -1.0f, 49.0f}, float3{0.0f, 1.0f, 49.0f},
                          float3{0.0f, 0.0f, 51.0f}});
    shapes.push_back(f);

    _tree_layout0 tree = build_tree(shapes);

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const std::array<float, 8> heights = {0.0f, 10.0f, 20.0f, 30.0f,
                                          50.0f, 40.0f, 70.0f, 10.0f};
    std::array<Ray, 8> rays;
    for (size_t i = 0; i < rays.size(); i++) {
        rays[i] = Ray{float3{-10.0f, 0.0f, heights[i]}, forward};
    }
    std::array<float, 8> out{};
    hit_z_all(rays, out, tree);
    bool same = true;
    for (size_t i = 0; i < rays.size(); i++) {
        same = same && hit_z_one(rays[i], tree) == out[i];
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';

    free(tree.group0_index->host);
}
