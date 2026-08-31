// Driver for apps/pbrt/render.bonsai, milestone 0.
//
//   ./build/compiler -p ssa -b cpp -i apps/pbrt/render.bonsai -o
//   apps/pbrt/render
//   clang++ -std=c++20 -O3 apps/pbrt/render_hook.cpp apps/pbrt/render.o \
//       -o apps/pbrt/render_runner
//   ./apps/pbrt/render_runner out.ppm
//
// Everything PBRT does in a camera constructor happens here: the field of
// view, the screen window and the resolution become two matrices, and the
// bonsai side only ever applies them. That split is the app's rule, not a
// shortcut -- bonsai has no file I/O to parse a .pbrt scene with, and PBRT's
// hot path does not build transforms either.

#include "render.h"
#include "scene_io.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t MaxTreeDepth = 64;

// The camera transforms are no longer derived here. Perspective, LookAt, the
// screen window and the matrix inverse that composes them all used to live in
// this file, kept in step with PBRT's by hand; scene_dump.cpp now asks PBRT
// for them, using PBRT's own Transform arithmetic, and they arrive as two
// matrices this file only has to hand across.

// pbrt: Bounds3f, with the operations BVHAggregate's build asks of it.
//
// The empty box is pMin at +max and pMax at lowest, so that Union over nothing
// is the identity and a surface area computed from it comes out meaningless
// rather than zero -- which matters, because zero area is the signal the build
// uses to give up on splitting.
struct Bounds3f {
    float3 pMin;
    float3 pMax;

    Bounds3f()
        : pMin{std::numeric_limits<float>::max(),
               std::numeric_limits<float>::max(),
               std::numeric_limits<float>::max()},
          pMax{std::numeric_limits<float>::lowest(),
               std::numeric_limits<float>::lowest(),
               std::numeric_limits<float>::lowest()} {}

    Bounds3f(const float3 &a, const float3 &b)
        : pMin{std::fminf(a[0], b[0]), std::fminf(a[1], b[1]),
               std::fminf(a[2], b[2])},
          pMax{std::fmaxf(a[0], b[0]), std::fmaxf(a[1], b[1]),
               std::fmaxf(a[2], b[2])} {}

    float3 diagonal() const { return pMax - pMin; }

    float surface_area() const {
        const float3 d = diagonal();
        return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
    }

    int max_dimension() const {
        const float3 d = diagonal();
        if (d[0] > d[1] && d[0] > d[2]) {
            return 0;
        }
        return (d[1] > d[2]) ? 1 : 2;
    }

    // Where a point sits in the box, as a fraction along each axis.
    float3 offset(const float3 &p) const {
        float3 o = p - pMin;
        for (int i = 0; i < 3; i++) {
            if (pMax[i] > pMin[i]) {
                o[i] /= pMax[i] - pMin[i];
            }
        }
        return o;
    }

    float3 centroid() const { return 0.5f * pMin + 0.5f * pMax; }
};

Bounds3f merge(const Bounds3f &a, const Bounds3f &b) {
    Bounds3f r;
    r.pMin = float3{std::fminf(a.pMin[0], b.pMin[0]),
                    std::fminf(a.pMin[1], b.pMin[1]),
                    std::fminf(a.pMin[2], b.pMin[2])};
    r.pMax = float3{std::fmaxf(a.pMax[0], b.pMax[0]),
                    std::fmaxf(a.pMax[1], b.pMax[1]),
                    std::fmaxf(a.pMax[2], b.pMax[2])};
    return r;
}

Bounds3f merge(const Bounds3f &a, const float3 &p) {
    return merge(a, Bounds3f{p, p});
}

// pbrt: Sphere::Bounds and Triangle::Bounds. A sphere placed by a translation
// bounds to its centre plus and minus the radius on each axis; a triangle to
// the box around its three vertices.
Bounds3f bounds_of(const Shape &shape) {
    if (shape.tag == 0) {
        const Sphere &s = shape.payload.Sph.s;
        return Bounds3f{s.center - s.radius, s.center + s.radius};
    }
    const Triangle &t = shape.payload.Tri.t;
    return merge(Bounds3f{t.p0, t.p1}, t.p2);
}

// pbrt: BVHPrimitive, a primitive reduced to what the build sorts on.
struct BVHPrimitive {
    uint32_t index;
    Bounds3f bounds;
    float3 centroid() const { return bounds.centroid(); }
};

// pbrt: BVHSplitBucket.
struct BVHSplitBucket {
    int count = 0;
    Bounds3f bounds;
};

// pbrt: "integer maxnodeprims", whose default is 4. A node holding no more
// than this may stay a leaf when the SAH says splitting is not worth it.
constexpr int MaxPrimsInNode = 4;

// pbrt: BVHAggregate::buildRecursive with SplitMethod::SAH, followed by
// flattenBVH. This is here rather than in bonsai because building a tree is
// not yet expressible in the language -- only traversing one is, and that is
// the part that runs per ray. The node layout is not chosen here either: the
// `layout` block in render.bonsai decides it, and this fills in what that
// block named.
//
// The two passes are fused: pbrt builds a pointer tree and then flattens it
// depth first, and writing the nodes out depth first in the first place lands
// them in the same order. What that order buys is the layout's `left = index +
// 1` -- a child that needs no offset because it is always the next node.
//
// pbrt splits the build across threads above 128K primitives, which reorders
// the primitive array and so builds a different (equally valid) tree. This is
// the serial path, which is what pbrt itself takes at these sizes.
_tree_layout0 build_bvh(std::vector<Shape> &shapes) {
    std::vector<BVHPrimitive> prims;
    prims.reserve(shapes.size());
    for (uint32_t i = 0; i < shapes.size(); i++) {
        prims.push_back(BVHPrimitive{i, bounds_of(shapes[i])});
    }

    // pbrt: orderedPrims. A leaf names a run of primitives, so the primitives
    // it names have to be contiguous, which means the scene is rewritten into
    // the order the build discovered.
    std::vector<Shape> ordered;
    ordered.reserve(shapes.size());
    std::vector<_tree_layout1> nodes;

    std::function<uint32_t(BVHPrimitive *, size_t, uint32_t)> build =
        [&](BVHPrimitive *span, size_t n, uint32_t depth) -> uint32_t {
        assert(depth < MaxTreeDepth);
        const uint32_t self = uint32_t(nodes.size());
        nodes.emplace_back();

        Bounds3f bounds;
        for (size_t i = 0; i < n; i++) {
            bounds = merge(bounds, span[i].bounds);
        }

        // Note the deliberate re-index of `nodes[self]` after any recursion:
        // the vector reallocates, so a reference taken across a child build
        // would dangle.
        const auto make_leaf = [&]() {
            const uint32_t first = uint32_t(ordered.size());
            for (size_t i = 0; i < n; i++) {
                ordered.push_back(shapes[span[i].index]);
            }
            nodes[self].low = bounds.pMin;
            nodes[self].high = bounds.pMax;
            nodes[self].nPrims = uint16_t(n);
            const uint32_t offset = first;
            std::memcpy(nodes[self].split0on_nPrims.data(), &offset,
                        sizeof(offset));
        };

        if (bounds.surface_area() == 0.0f || n == 1) {
            make_leaf();
            return self;
        }

        Bounds3f centroidBounds;
        for (size_t i = 0; i < n; i++) {
            centroidBounds = merge(centroidBounds, span[i].centroid());
        }
        const int dim = centroidBounds.max_dimension();

        // Every centroid at the same place on the widest axis: no split can
        // separate them, so splitting would only add a level.
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            make_leaf();
            return self;
        }

        size_t mid = n / 2;
        if (n <= 2) {
            std::nth_element(span, span + mid, span + n,
                             [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                 return a.centroid()[dim] < b.centroid()[dim];
                             });
        } else {
            // The surface area heuristic, over twelve buckets along the
            // widest axis of the centroids.
            constexpr int nBuckets = 12;
            BVHSplitBucket buckets[nBuckets];
            const auto bucket_of = [&](const BVHPrimitive &p) {
                int b = int(nBuckets * centroidBounds.offset(p.centroid())[dim]);
                return (b == nBuckets) ? nBuckets - 1 : b;
            };
            for (size_t i = 0; i < n; i++) {
                const int b = bucket_of(span[i]);
                buckets[b].count++;
                buckets[b].bounds = merge(buckets[b].bounds, span[i].bounds);
            }

            // The cost of splitting after bucket i is the area of each side
            // weighted by how many primitives land there. Two scans, so that
            // each side's running union is computed once rather than per
            // candidate split.
            constexpr int nSplits = nBuckets - 1;
            float costs[nSplits] = {};
            int countBelow = 0;
            Bounds3f boundBelow;
            for (int i = 0; i < nSplits; i++) {
                boundBelow = merge(boundBelow, buckets[i].bounds);
                countBelow += buckets[i].count;
                costs[i] += float(countBelow) * boundBelow.surface_area();
            }
            int countAbove = 0;
            Bounds3f boundAbove;
            for (int i = nSplits; i >= 1; i--) {
                boundAbove = merge(boundAbove, buckets[i].bounds);
                countAbove += buckets[i].count;
                costs[i - 1] += float(countAbove) * boundAbove.surface_area();
            }

            int minCostSplitBucket = -1;
            float minCost = std::numeric_limits<float>::infinity();
            for (int i = 0; i < nSplits; i++) {
                if (costs[i] < minCost) {
                    minCost = costs[i];
                    minCostSplitBucket = i;
                }
            }

            // pbrt's half is the cost of the node traversal itself, against a
            // leaf costing one intersection per primitive.
            const float leafCost = float(n);
            minCost = 0.5f + minCost / bounds.surface_area();
            if (n > size_t(MaxPrimsInNode) || minCost < leafCost) {
                BVHPrimitive *midIter =
                    std::partition(span, span + n, [&](const BVHPrimitive &p) {
                        return bucket_of(p) <= minCostSplitBucket;
                    });
                mid = size_t(midIter - span);
            } else {
                make_leaf();
                return self;
            }
        }

        nodes[self].low = bounds.pMin;
        nodes[self].high = bounds.pMax;
        nodes[self].nPrims = 0;
        nodes[self].axis = uint8_t(dim);
        build(span, mid, depth + 1);
        const uint32_t right = build(span + mid, n - mid, depth + 1);
        const uint32_t offset = right - self;
        std::memcpy(nodes[self].split0on_nPrims.data(), &offset,
                    sizeof(offset));
        return self;
    };

    build(prims.data(), prims.size(), 0);

    shapes = std::move(ordered);

    _tree_layout0 tree;
    tree.pCount = uint32_t(shapes.size());
    tree.prims = shapes.data();
    tree.nCount = uint32_t(nodes.size());
    tree.group0_index =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.nCount);
    std::memcpy(tree.group0_index, nodes.data(),
                sizeof(_tree_layout1) * tree.nCount);
    return tree;
}

// Sixteen floats in row order, as scene_dump wrote them.
Transform to_bonsai(const float *m) {
    Transform t;
    t.r0 = float4{m[0], m[1], m[2], m[3]};
    t.r1 = float4{m[4], m[5], m[6], m[7]};
    t.r2 = float4{m[8], m[9], m[10], m[11]};
    t.r3 = float4{m[12], m[13], m[14], m[15]};
    return t;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: render <scene.bin> [out.pfm]\n";
        return 1;
    }
    const char *scene_path = argv[1];
    const char *output = (argc > 2) ? argv[2] : "pbrt.pfm";

    // The scene came from a .pbrt file through PBRT's own parser; see
    // scene_dump.cpp. Nothing about the scene is written down here, which is
    // the point -- there is one description of it and PBRT and this renderer
    // both read it.
    bonsai_scene::Header header;
    std::vector<float> matrices;
    std::vector<bonsai_scene::Shape> loaded;
    if (!bonsai_scene::read(scene_path, header, matrices, loaded)) {
        std::cerr << "cannot read scene " << scene_path
                  << " (run scene_dump on a .pbrt first)\n";
        return 1;
    }

    const int width = int(header.width);
    const int height = int(header.height);

    PerspectiveCamera camera;
    camera.camera_from_raster = to_bonsai(&matrices[0]);
    camera.render_from_camera = to_bonsai(&matrices[16]);

    std::vector<Shape> shapes;
    shapes.reserve(loaded.size());
    for (const bonsai_scene::Shape &s : loaded) {
        // Through the generated constructors rather than by setting the tag:
        // which number a variant is belongs to the compiler.
        Shape shape;
        if (s.tag == bonsai_scene::ShapeTag::Sphere) {
            Shape_Sph(shape, Sphere{float3{s.p0[0], s.p0[1], s.p0[2]},
                                    s.radius});
        } else {
            Shape_Tri(shape, Triangle{float3{s.p0[0], s.p0[1], s.p0[2]},
                                      float3{s.p1[0], s.p1[1], s.p1[2]},
                                      float3{s.p2[0], s.p2[1], s.p2[2]}});
        }
        shapes.push_back(shape);
    }

    _tree_layout0 tree = build_bvh(shapes);

    const uint32_t npixels = uint32_t(width) * uint32_t(height);
    float3 *out = (float3 *)malloc(sizeof(float3) * npixels);

    // Only the render is timed. Building the scene and the BVH is the work
    // pbrt does before its own timer starts (its renderTimeSeconds comes from
    // a progress reporter created after the scene is built), so counting it
    // here would be comparing two different things.
    const auto started = std::chrono::steady_clock::now();
    render(camera, uint32_t(width), uint32_t(height), out, tree);
    const auto finished = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(finished - started).count();

    // What pbrt writes: the film's linear values, unencoded. pbrt quantizes
    // only when asked for a .png or a .qoi, and applies its sRGB transfer
    // function when it does; that is post-processing, and it happens in
    // to_png.py rather than here. PFM is one of the formats pbrt itself
    // writes, so this file and one from pbrt are directly comparable.
    std::ofstream pfm(output, std::ios::binary);
    if (!pfm) {
        std::cerr << "cannot open " << output << " for writing\n";
        free(out);
        free(tree.group0_index);
        return 1;
    }
    // PFM rows run bottom to top, and a negative scale says little-endian.
    pfm << "PF\n" << width << ' ' << height << "\n-1.000000\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            const float3 &v = out[uint32_t(j) * uint32_t(width) + i];
            const float rgb[3] = {v[0], v[1], v[2]};
            pfm.write(reinterpret_cast<const char *>(rgb), sizeof(rgb));
        }
    }

    std::cout << "wrote " << output << " (" << width << 'x' << height << ", "
              << shapes.size() << " shapes)\n";
    // Parsed by compare.sh. Kept to a line of its own so that it stays easy
    // to find without the script having to understand anything else here.
    std::cout << "render seconds: " << seconds << '\n';
    free(out);
    free(tree.group0_index);
    return 0;
}
