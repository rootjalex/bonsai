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

#include "cie_tables.h"
#include "rgb2spec.h"
#include "scene_io.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
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

// Where a triangle's three vertices are, which is now a question for the mesh
// rather than for the triangle. pbrt's `&mesh->vertexIndices[3 * triIndex]`,
// with the mesh's own offset into the shared pools added.
struct Meshes {
    const TriangleMesh *meshes = nullptr;
    const uint32_t *indices = nullptr;
    const float3 *positions = nullptr;

    void corners(const Triangle &t, uint32_t out[3]) const {
        const TriangleMesh &m = meshes[t.mesh];
        for (uint32_t k = 0; k < 3; k++) {
            out[k] = m.first_vertex + indices[m.first_index + 3 * t.tri + k];
        }
    }
};

// Where a `Shape` keeps what it is, which under `layout Shape = tagged_index`
// is not in the Shape at all.
//
// A Shape is a `uint64_t`: the variant in the top byte and, in the rest, an
// index into the pool for that variant. These pools are this file's memory --
// that is the whole point of the layout, and why nothing in the renderer has to
// allocate to build a shape -- so taking one apart is this file's job too. It
// is the only place the encoding is written down outside the compiler; see
// `ADTLayout::tag_shift` in include/Lower/ADTLayout.h.
//
// pbrt does the same thing with the same handful of bits and calls it
// TaggedPointer. The difference is that its low bits are an address and these
// are an index, which is why nothing here has to be allocated or freed.
struct Shapes {
    static constexpr uint64_t kTagShift = 56;
    static constexpr uint64_t kSphere = 0;

    const Sph *spheres = nullptr;
    const Tri *triangles = nullptr;

    static uint64_t tag_of(uint64_t shape) { return shape >> kTagShift; }
    static uint64_t index_of(uint64_t shape) {
        return shape & ((uint64_t{1} << kTagShift) - 1);
    }

    static uint64_t handle(uint64_t tag, uint64_t index) {
        return (tag << kTagShift) | index;
    }

    bool is_sphere(uint64_t shape) const { return tag_of(shape) == kSphere; }
    const Sphere &sphere(uint64_t shape) const {
        return spheres[index_of(shape)].s;
    }
    const Triangle &triangle(uint64_t shape) const {
        return triangles[index_of(shape)].t;
    }
};

// pbrt: Sphere::Bounds and Triangle::Bounds. A sphere placed by a translation
// bounds to its centre plus and minus the radius on each axis; a triangle to
// the box around its three vertices.
Bounds3f bounds_of(const Geometric &prim, const Meshes &pool,
                   const Shapes &shapes) {
    if (shapes.is_sphere(prim.shape)) {
        const Sphere &s = shapes.sphere(prim.shape);
        return Bounds3f{s.center - s.radius, s.center + s.radius};
    }
    uint32_t c[3];
    pool.corners(shapes.triangle(prim.shape), c);
    return merge(Bounds3f{pool.positions[c[0]], pool.positions[c[1]]},
                 pool.positions[c[2]]);
}

// pbrt: Transform::operator()(Point3f) and Transform::operator()(const
// Bounds3f &) -- the eight corners moved and the box around them, which is
// TransformedPrimitive::Bounds(): what an instance occupies in render space,
// and so what the top-level tree is built from for it.
float3 apply_point(const Transform &t, const float3 &p) {
    const auto row = [&](const float4 &r) {
        return r.x * p.x + r.y * p.y + r.z * p.z + r.w;
    };
    const float w = row(t.r3);
    const float3 v{row(t.r0), row(t.r1), row(t.r2)};
    return w == 1.0f ? v : v / w;
}

Bounds3f transform_bounds(const Transform &t, const Bounds3f &b) {
    Bounds3f out;
    for (int corner = 0; corner < 8; corner++) {
        const float3 p{(corner & 1) ? b.pMax.x : b.pMin.x,
                       (corner & 2) ? b.pMax.y : b.pMin.y,
                       (corner & 4) ? b.pMax.z : b.pMin.z};
        out = merge(out, apply_point(t, p));
    }
    return out;
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
//
// One builder for every tree the scene has: the top-level one over its
// primitives, and one per instance definition over that definition's shapes.
// It appends to `nodes` -- an instance's tree is rows of a pool shared with
// every other instance's -- and returns the row its root landed on. `items`
// come back in the order the leaves name them (pbrt's orderedPrims), and a
// leaf's first item is `base` plus its place in `items`, `base` being where
// `items` sits in whatever array the layout's leaves index.
template <typename Node, typename Item, typename BoundsOf>
uint32_t build_bvh(Item *items, size_t count, uint32_t base,
                   BoundsOf bounds_of_item, std::vector<Node> &nodes) {
    std::vector<BVHPrimitive> prims;
    prims.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        prims.push_back(BVHPrimitive{i, bounds_of_item(items[i])});
    }

    // pbrt: orderedPrims. A leaf names a run of primitives, so the primitives
    // it names have to be contiguous, which means the scene is rewritten into
    // the order the build discovered.
    std::vector<Item> ordered;
    ordered.reserve(count);

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
            const uint32_t first = base + uint32_t(ordered.size());
            for (size_t i = 0; i < n; i++) {
                ordered.push_back(items[span[i].index]);
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

    const uint32_t root = build(prims.data(), prims.size(), 0);

    std::copy(ordered.begin(), ordered.end(), items);
    return root;
}

// Put the pools in the order the leaves read them, and rewrite the handles.
//
// This is the price of the indirection, and it has to be paid back here. The
// BVH build reorders the primitives so that a leaf names a contiguous run of
// them; under a tagged union that moves the shapes themselves, and a leaf's
// four triangles arrive on one or two cache lines. Under `tagged_index` the
// reorder moves only the eight-byte handles, and the fields they name stay
// wherever the scene file happened to put them -- so the same leaf reaches
// four scattered pool entries, and the smaller primitive buys nothing.
//
// Permuting the pools to match restores what the reorder was for. It is the
// driver's job rather than the compiler's for the same reason building the
// tree is: the layout says where a shape's fields live, and which order is a
// good one to put them in is a question about the tree above them.
//
// pbrt does not do this. Its TaggedPointers point at objects allocated while
// the scene was parsed, and its `orderedPrims` moves the pointers and not the
// objects, so a pbrt leaf chases the same scattered addresses.
//
// Two lists of primitives, each already in its leaves' order: the top-level
// tree's and, after them, every instance tree's.
void compact_pools(std::vector<Geometric> &shapes,
                   std::vector<Geometric> &instanced, std::vector<Sph> &spheres,
                   std::vector<Tri> &triangles) {
    std::vector<Sph> ordered_spheres;
    std::vector<Tri> ordered_triangles;
    ordered_spheres.reserve(spheres.size());
    ordered_triangles.reserve(triangles.size());

    for (std::vector<Geometric> *list : {&shapes, &instanced}) {
        for (Geometric &prim : *list) {
            const uint64_t tag = Shapes::tag_of(prim.shape);
            const uint64_t index = Shapes::index_of(prim.shape);
            if (tag == Shapes::kSphere) {
                prim.shape = Shapes::handle(tag, ordered_spheres.size());
                ordered_spheres.push_back(spheres[index]);
            } else {
                prim.shape = Shapes::handle(tag, ordered_triangles.size());
                ordered_triangles.push_back(triangles[index]);
            }
        }
    }

    spheres = std::move(ordered_spheres);
    triangles = std::move(ordered_triangles);
}

// The nodes of a tree PBRT built, packed into the layout the schedule
// declared.
//
// No building happens here: the nodes arrive already flattened depth first,
// with the primitives in the order its leaves expect, so this is only a change
// of representation. That the two layouts line up field for field is not luck
// -- the `layout` block in render.bonsai was written to be LinearBVHNode.
template <typename Node>
std::vector<Node> adopt_nodes(const std::vector<bonsai_scene::Node> &nodes) {
    std::vector<Node> out(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        const bonsai_scene::Node &n = nodes[i];
        out[i].low = float3{n.low[0], n.low[1], n.low[2]};
        out[i].high = float3{n.high[0], n.high[1], n.high[2]};
        out[i].nPrims = n.n_prims;
        out[i].axis = uint8_t(n.axis);
        std::memcpy(out[i].split0on_nPrims.data(), &n.offset,
                    sizeof(n.offset));
    }
    return out;
}

// pbrt: a Primitive of the top-level tree before it is one -- a
// GeometricPrimitive by its index into the scene's shapes, or a
// TransformedPrimitive by its index into the instances -- with the bounds the
// tree is built from, which for an instance are its object's tree's root
// bounds placed by its transform.
struct TopLevel {
    bool instance = false;
    uint32_t index = 0;
    Bounds3f bounds;
};

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

// Wall-clock for one stage of the run, reported as it finishes.
//
// Only the render used to be timed, and everything before it -- reading the
// scene, unpacking the pools, building the BVH -- happened in silence. On a
// scene with a hundred thousand shapes that silence was four and a half
// minutes against a render of one second, which reads as a hang and reads as a
// slow renderer afterwards, and neither is true.
struct Stage {
    const char *name;
    std::chrono::steady_clock::time_point begun;

    explicit Stage(const char *name)
        : name(name), begun(std::chrono::steady_clock::now()) {}

    ~Stage() {
        const double s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          begun)
                .count();
        fprintf(stderr, "  %-22s %7.3f s\n", name, s);
    }
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: render <scene.bin> [out.pfm]\n"
                  << "       render --print-differentials <scene.bin>\n";
        return 1;
    }
    // The same four pixels and the same synthetic hit
    // `scene_dump --print-differentials` uses, printed in the same format so a
    // plain diff is the comparison.
    bool print_differentials = false;
    int arg = 1;
    if (std::string(argv[arg]) == "--print-differentials") {
        print_differentials = true;
        arg++;
        if (arg >= argc) {
            std::cerr << "--print-differentials needs a scene\n";
            return 1;
        }
    }
    const char *scene_path = argv[arg++];
    const char *output = (arg < argc) ? argv[arg] : "pbrt.pfm";

    // The scene came from a .pbrt file through PBRT's own parser; see
    // scene_dump.cpp. Nothing about the scene is written down here, which is
    // the point -- there is one description of it and PBRT and this renderer
    // both read it.
    bonsai_scene::Scene loaded;
    {
        Stage stage("read scene");
        if (!bonsai_scene::read(scene_path, loaded)) {
            std::cerr << "cannot read scene " << scene_path
                      << " (run scene_dump on a .pbrt first)\n";
            return 1;
        }
    }

    // Everything from here to the tree: the camera, the sampler, the spectral
    // tables, the textures, the materials and the geometry pools. One timer
    // over the lot, since no part of it has ever been the slow one.
    std::unique_ptr<Stage> setup(new Stage("unpack scene"));

    const int width = int(loaded.width);
    const int height = int(loaded.height);

    const auto to_vec3 = [](const float *v) {
        return float3{v[0], v[1], v[2]};
    };

    PerspectiveCamera camera;
    camera.camera_from_raster = to_bonsai(&loaded.matrices[0]);
    camera.render_from_camera = to_bonsai(&loaded.matrices[16]);
    camera.camera_from_render = to_bonsai(&loaded.matrices[32]);
    camera.dx_camera = to_vec3(&loaded.d_camera[0]);
    camera.dy_camera = to_vec3(&loaded.d_camera[3]);
    camera.min_pos_differential_x = to_vec3(&loaded.min_differentials[0]);
    camera.min_pos_differential_y = to_vec3(&loaded.min_differentials[3]);
    camera.min_dir_differential_x = to_vec3(&loaded.min_differentials[6]);
    camera.min_dir_differential_y = to_vec3(&loaded.min_differentials[9]);
    camera.lens_radius = loaded.lens_radius;
    camera.focal_distance = loaded.focal_distance;

    if (print_differentials) {
        // Chosen to match scene_dump's; a parameterization that is neither
        // orthogonal nor unit-length, so the 2x2 solve has something to do.
        const float3 hit_p{-1.25f, 0.75f, -3.5f};
        const float3 dpdu{1.7f, 0.3f, -0.4f};
        const float3 dpdv{-0.2f, 1.1f, 0.9f};
        // The same lobes scene_dump asks about: the two exactly-specular ones,
        // which carry differentials across the bounce, and a glossy one, which
        // does not. Flags as apps/pbrt/bxdf.bonsai numbers them.
        const float3 wi = [] {
            float3 v{0.31f, -0.82f, 0.48f};
            const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            return float3{v.x / len, v.y / len, v.z / len};
        }();
        struct Lobe {
            const char *label;
            int32_t flags;
            float eta;
        };
        const Lobe lobes[] = {{"reflect", 16 | 1, 1.f},
                              {"transmit", 16 | 2, 1.5f},
                              {"rough", 8 | 1, 1.f}};

        const int pixels[][2] = {{0, 0}, {17, 42}, {640, 360}, {1279, 719}};
        for (const auto &px : pixels) {
            float out[35] = {};
            differentials_at(camera, float(px[0]), float(px[1]), true, 16,
                             hit_p, dpdu, dpdv, wi, lobes[0].flags,
                             lobes[0].eta, out);
            printf("camdiff %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                   "%.9g %.9g %.9g | %.9g %.9g %.9g\n",
                   px[0], px[1], double(out[0]), double(out[1]), double(out[2]),
                   double(out[3]), double(out[4]), double(out[5]),
                   double(out[6]), double(out[7]), double(out[8]),
                   double(out[9]), double(out[10]), double(out[11]));
            for (int has = 1; has >= 0; has--) {
                differentials_at(camera, float(px[0]), float(px[1]), has != 0,
                                 16, hit_p, dpdu, dpdv, wi, lobes[0].flags,
                                 lobes[0].eta, out);
                printf("dudxy %d %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                       "%.9g %.9g %.9g %.9g\n",
                       px[0], px[1], has, double(out[12]), double(out[13]),
                       double(out[14]), double(out[15]), double(out[16]),
                       double(out[17]), double(out[18]), double(out[19]),
                       double(out[20]), double(out[21]));
                for (const Lobe &lobe : lobes) {
                    differentials_at(camera, float(px[0]), float(px[1]),
                                     has != 0, 16, hit_p, dpdu, dpdv, wi,
                                     lobe.flags, lobe.eta, out);
                    printf("spawn %d %d %d %s: %d | %.9g %.9g %.9g | "
                           "%.9g %.9g %.9g | %.9g %.9g %.9g | %.9g %.9g %.9g\n",
                           px[0], px[1], has, lobe.label, int(out[22] != 0.f),
                           double(out[23]), double(out[24]), double(out[25]),
                           double(out[26]), double(out[27]), double(out[28]),
                           double(out[29]), double(out[30]), double(out[31]),
                           double(out[32]), double(out[33]), double(out[34]));
                }
            }
        }
        return 0;
    }

    // Which sampler the scene asked for, built through the generated
    // constructors rather than by setting the tag -- how a variant is laid out
    // is the compiler's business, and writing it by hand here would be a second
    // place that has to agree with it.
    Sampler sampler;
    if (loaded.sampler.tag == bonsai_scene::SamplerTag::Stratified) {
        Sampler_Stratified(sampler, loaded.sampler.x_samples,
                           loaded.sampler.y_samples, loaded.sampler.seed,
                           loaded.sampler.jitter != 0);
    } else if (loaded.sampler.tag == bonsai_scene::SamplerTag::Halton) {
        Sampler_Halton(sampler, loaded.sampler.samples_per_pixel,
                       loaded.sampler.seed, int32_t(loaded.sampler.randomize),
                       loaded.sampler.base_scales[0],
                       loaded.sampler.base_scales[1],
                       loaded.sampler.base_exponents[0],
                       loaded.sampler.base_exponents[1],
                       loaded.sampler.mult_inverse[0],
                       loaded.sampler.mult_inverse[1]);
    } else {
        Sampler_Independent(sampler, loaded.sampler.samples_per_pixel,
                            loaded.sampler.seed);
    }

    // pbrt: the first thousand primes, which are the Halton sequence's bases,
    // one per dimension. Sieved rather than tabulated -- a table of a thousand
    // numbers is a thousand chances to mistype one, and this is checked against
    // pbrt's own table by `scene_dump --print-sampler`.
    std::array<int32_t, 1000> primes;
    {
        size_t found = 0;
        for (int32_t n = 2; found < primes.size(); n++) {
            bool prime = true;
            for (int32_t d = 2; d * d <= n && prime; d++) {
                prime = n % d != 0;
            }
            if (prime) {
                primes[found++] = n;
            }
        }
    }

    // pbrt: ComputeRadicalInversePermutations, which the HaltonSampler's
    // constructor runs -- and only when the scene asked for the default
    // `permutedigits`, which is the only randomization that reads the table.
    //
    // The driver owns the storage for the same reason it owns the ADT pools:
    // the renderer is built with --no-heap and cannot allocate, and 26 MB is
    // not a stack slot. What fills it is the renderer's own
    // `permutation_element` and `hash3`, reached through the two exported
    // functions, so there is no second implementation of either here to drift
    // from the one that is checked against pbrt.
    //
    // Built before the timer starts, which is where pbrt builds it too: its
    // sampler is constructed while the scene is.
    std::array<int32_t, 1000> digit_permutation_offsets{};
    std::vector<uint16_t> digit_permutations;
    if (loaded.sampler.tag == bonsai_scene::SamplerTag::Halton &&
        loaded.sampler.randomize ==
            bonsai_scene::RandomizeTag::RandomizePermuteDigits) {
        const int32_t extent =
            digit_permutation_extent(digit_permutation_offsets.data(), primes);
        digit_permutations.resize(size_t(extent));
        build_digit_permutations(loaded.sampler.seed,
                                 digit_permutation_offsets.data(),
                                 digit_permutations.data(), primes);
    }

    // pbrt: the FilterSampler a GaussianFilter builds in its constructor -- the
    // filter tabulated over a grid and turned into a piecewise-constant 2D
    // distribution, because a Gaussian has no closed-form inverse.
    //
    // The driver owns it for the same reasons it owns the digit permutations:
    // the renderer is built with --no-heap, and this is a constant of the film
    // that has no business being rebuilt per sample. What fills it is the
    // renderer's own `build_filter_table`, so the Gaussian and the CDFs here
    // are the ones the render samples rather than a second copy in C++.
    //
    // `int(32 * radius)` per axis is pbrt's own sizing, from the FilterSampler
    // constructor: 48x48 at the default radius of 1.5.
    Filter pixel_filter;
    make_filter(pixel_filter,
                float2{loaded.filter_radius[0], loaded.filter_radius[1]},
                loaded.filter_sigma);
    if (pixel_filter.nx < 1 || pixel_filter.ny < 1) {
        fprintf(stderr, "a filter radius of %g x %g tabulates to nothing\n",
                double(loaded.filter_radius[0]),
                double(loaded.filter_radius[1]));
        return 1;
    }
    std::vector<float> filter_f(size_t(pixel_filter.nx) * pixel_filter.ny);
    std::vector<float> filter_cond_cdf(size_t(pixel_filter.nx + 1) *
                                       pixel_filter.ny);
    std::vector<float> filter_marg_func(size_t(pixel_filter.ny));
    std::vector<float> filter_marg_cdf(size_t(pixel_filter.ny) + 1);
    pixel_filter.marg_int =
        build_filter_table(pixel_filter, filter_f.data(),
                           filter_cond_cdf.data(), filter_marg_func.data(),
                           filter_marg_cdf.data());

    // pbrt: the path integrator's fixed sample points for a reflectance
    // estimate, which are file-scope constants there and reach the renderer as
    // externs here. Built once, because they are constants; they used to be
    // declared where they are used, which allocated them per camera ray.
    static constexpr int kRhoSamples = 16;
    static const float rho_uc[kRhoSamples] = {
        0.75741637f, 0.37870818f, 0.7083487f,  0.18935409f,
        0.9149363f,  0.35417435f, 0.5990858f,  0.09467703f,
        0.8578725f,  0.45746812f, 0.686759f,   0.17708716f,
        0.9674518f,  0.2995429f,  0.5083201f,  0.047338516f};
    static const float rho_ux[kRhoSamples] = {
        0.855985f, 0.381823f, 0.285328f, 0.733380f, 0.542663f, 0.127274f,
        0.964700f, 0.594089f, 0.095109f, 0.825444f, 0.429467f, 0.244460f,
        0.756135f, 0.516165f, 0.180888f, 0.898579f};
    static const float rho_uy[kRhoSamples] = {
        0.570367f, 0.851844f, 0.764262f, 0.114073f, 0.344465f, 0.414848f,
        0.947162f, 0.643463f, 0.170369f, 0.263359f, 0.454469f, 0.816459f,
        0.731258f, 0.152852f, 0.214174f, 0.503897f};

    // pbrt fits every RGB albedo to three sigmoid coefficients once, offline,
    // into a table it looks up while building the scene. This runs the same
    // fit here for the same reason: a Gauss-Newton solve has no business
    // anywhere near a ray. Cached by colour, since a scene reuses materials.
    const rgb2spec::Tables fit_tables = rgb2spec::init_tables();
    std::map<std::array<float, 3>, SigmoidPolynomial> fitted;
    const auto albedo_of = [&](const float *rgb) {
        const std::array<float, 3> key = {rgb[0], rgb[1], rgb[2]};
        auto it = fitted.find(key);
        if (it == fitted.end()) {
            const rgb2spec::Coefficients c =
                rgb2spec::fit(fit_tables, rgb[0], rgb[1], rgb[2]);
            it = fitted.emplace(key, SigmoidPolynomial{c.c0, c.c1, c.c2}).first;
        }
        return it->second;
    };

    // The scene's materials, with every RGB fitted. Through the generated
    // The image textures, their levels, and the texels the levels index into.
    // Laid out exactly as the scene wrote them; the pyramid was built by PBRT.
    std::vector<ImageTexture> textures;
    textures.reserve(loaded.textures.size());
    for (const bonsai_scene::ImageTexture &t : loaded.textures) {
        ImageTexture tex;
        tex.su = t.su;
        tex.sv = t.sv;
        tex.du = t.du;
        tex.dv = t.dv;
        tex.scale = t.scale;
        tex.invert = t.invert != 0;
        tex.wrap = t.wrap;
        tex.first_level = t.first_level;
        tex.n_levels = t.n_levels;
        textures.push_back(tex);
    }
    std::vector<TextureLevel> texture_levels;
    texture_levels.reserve(loaded.texture_levels.size());
    for (const bonsai_scene::TextureLevel &l : loaded.texture_levels) {
        TextureLevel level;
        level.width = l.width;
        level.height = l.height;
        level.first_texel = l.first_texel;
        texture_levels.push_back(level);
    }
    std::vector<float3> texture_texels;
    texture_texels.reserve(loaded.texture_texels.size() / 3);
    for (size_t i = 0; i + 2 < loaded.texture_texels.size(); i += 3) {
        texture_texels.push_back(float3{loaded.texture_texels[i],
                                        loaded.texture_texels[i + 1],
                                        loaded.texture_texels[i + 2]});
    }

    // The measured BRDFs and their interpolants. The pools they index into are
    // read straight out of the scene; only the headers are rebuilt, because
    // the generated struct is the compiler's layout and not the file's.
    std::vector<PL2D> pl2d;
    pl2d.reserve(loaded.pl2d.size());
    for (const bonsai_scene::PL2DHeader &h : loaded.pl2d) {
        PL2D p;
        p.size_x = h.size_x;
        p.size_y = h.size_y;
        p.dim = h.dim;
        p.param_size =
            uint32_t3{h.param_size[0], h.param_size[1], h.param_size[2]};
        p.param_stride =
            uint32_t3{h.param_stride[0], h.param_stride[1], h.param_stride[2]};
        p.first_param =
            uint32_t3{h.first_param[0], h.first_param[1], h.first_param[2]};
        p.first_data = h.first_data;
        p.first_marginal = h.first_marginal;
        p.first_conditional = h.first_conditional;
        p.has_cdf = h.has_cdf != 0;
        pl2d.push_back(p);
    }
    std::vector<MeasuredBRDF> measured_brdfs;
    measured_brdfs.reserve(loaded.measured_brdfs.size());
    for (const bonsai_scene::MeasuredBRDF &b : loaded.measured_brdfs) {
        MeasuredBRDF m;
        m.ndf = b.ndf;
        m.sigma = b.sigma;
        m.vndf = b.vndf;
        m.luminance = b.luminance;
        m.spectra = b.spectra;
        m.isotropic = b.isotropic != 0;
        measured_brdfs.push_back(m);
    }

    // constructors rather than by setting the tag: which number a variant is
    // belongs to the compiler.
    std::vector<Material> materials;
    // Beside the materials rather than inside them, exactly as PBRT keeps it:
    // a displacement tilts the shading frame before any material is asked
    // anything, so it belongs to every kind of material and to none of them.
    std::vector<int32_t> material_displacement;
    materials.reserve(loaded.materials.size());
    material_displacement.reserve(loaded.materials.size());
    for (const bonsai_scene::Material &m : loaded.materials) {
        material_displacement.push_back(m.displacement_texture);
        Material material;
        Reflectance reflectance;
        reflectance.albedo = albedo_of(m.reflectance);
        reflectance.texture = m.reflectance_texture;
        if (m.tag == bonsai_scene::MaterialTag::CoatedDiffuse) {
            CoatedDiffuseMaterial coated;
            coated.reflectance = reflectance;
            coated.u_roughness = m.u_roughness;
            coated.v_roughness = m.v_roughness;
            coated.remap = m.remap != 0;
            coated.thickness = m.thickness;
            coated.eta = m.eta;
            coated.medium_albedo = albedo_of(m.medium_albedo);
            coated.has_medium = m.has_medium != 0;
            coated.g = m.g;
            coated.max_depth = m.max_depth;
            coated.n_samples = m.n_samples;
            Material_CoatedDiffuse(material, coated);
        } else if (m.tag == bonsai_scene::MaterialTag::Measured) {
            Material_Measured(material, uint32_t(m.measured));
        } else if (m.tag == bonsai_scene::MaterialTag::Conductor) {
            ConductorMaterial metal;
            metal.spectra = m.conductor_spectra;
            metal.u_roughness = m.u_roughness;
            metal.v_roughness = m.v_roughness;
            metal.remap = m.remap != 0;
            Material_Conductor(material, metal);
        } else if (m.tag == bonsai_scene::MaterialTag::Dielectric) {
            DielectricMaterial glass;
            glass.u_roughness = m.u_roughness;
            glass.v_roughness = m.v_roughness;
            glass.remap = m.remap != 0;
            glass.eta = m.eta;
            Material_Dielectric(material, glass);
        } else {
            Material_Diffuse(material, reflectance);
        }
        materials.push_back(material);
    }

    // The meshes and the pools they index into, exactly as the scene wrote
    // them. Nothing is rearranged here: a triangle names a mesh and a triangle
    // in it, and the vertices are read out on a hit, which is what keeps a
    // primitive down to the size of a sphere.
    std::vector<TriangleMesh> meshes;
    meshes.reserve(loaded.meshes.size());
    for (const bonsai_scene::Mesh &m : loaded.meshes) {
        TriangleMesh out_mesh;
        out_mesh.first_index = m.first_index;
        out_mesh.first_vertex = m.first_vertex;
        out_mesh.first_normal = m.first_normal;
        out_mesh.first_uv = m.first_uv;
        out_mesh.has_normals = m.has_normals != 0;
        out_mesh.has_uv = m.has_uv != 0;
        out_mesh.flip = m.flip != 0;
        meshes.push_back(out_mesh);
    }
    const auto to_float3 = [](const std::vector<float> &v) {
        std::vector<float3> out(v.size() / 3);
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = float3{v[3 * i + 0], v[3 * i + 1], v[3 * i + 2]};
        }
        return out;
    };
    const std::vector<float3> positions = to_float3(loaded.positions);
    const std::vector<float3> normals = to_float3(loaded.normals);
    std::vector<float2> uvs(loaded.uvs.size() / 2);
    for (size_t i = 0; i < uvs.size(); i++) {
        uvs[i] = float2{loaded.uvs[2 * i + 0], loaded.uvs[2 * i + 1]};
    }
    const Meshes pool{meshes.data(), loaded.indices.data(), positions.data()};

    // The pools a `tagged_index` Shape indexes into, sized to what the scene
    // holds before anything is built. Sized exactly rather than generously:
    // there is no growing them, because the renderer's constructors write into
    // the memory handed to them and nothing tells this file when they have.
    // Both lists of shapes, since an instanced shape is a Shape like any other.
    size_t nspheres = 0;
    size_t nshapes = 0;
    for (const std::vector<bonsai_scene::Shape> *list :
         {&loaded.shapes, &loaded.instance_shapes}) {
        for (const bonsai_scene::Shape &s : *list) {
            nspheres += s.tag == bonsai_scene::ShapeTag::Sphere;
            nshapes++;
        }
    }
    std::vector<Sph> sphere_pool(nspheres);
    std::vector<Tri> triangle_pool(nshapes - nspheres);
    // What the constructors bump. Each ends up equal to its pool's size, which
    // is the check that the two passes counted the same thing.
    uint64_t sphere_fill = 0;
    uint64_t triangle_fill = 0;

    // pbrt: CreatePrimitivesForShapes, run over the top-level shapes and over
    // each instance definition's -- the same conversion, into two lists.
    // `shapes` is what the top-level tree holds directly; `instanced` is every
    // instance definition's geometry, laid end to end in definition order, and
    // each instance's tree names a run of it.
    const auto convert = [&](const std::vector<bonsai_scene::Shape> &in) {
        std::vector<Geometric> out;
        out.reserve(in.size());
        for (const bonsai_scene::Shape &s : in) {
            // The generated constructors rather than a handle assembled here:
            // they put the fields in the pool and return the tag and index
            // naming them, and they are generated from the same layout the
            // renderer reads with, so the numbering cannot drift.
            uint64_t shape;
            if (s.tag == bonsai_scene::ShapeTag::Sphere) {
                Sphere sphere;
                sphere.center = float3{s.center[0], s.center[1], s.center[2]};
                sphere.radius = s.radius;
                sphere.flip = s.flip != 0;
                shape = Shape_Sph(sphere, sphere_pool.data(), &sphere_fill);
            } else {
                shape = Shape_Tri(Triangle{s.mesh, s.tri}, triangle_pool.data(),
                                  &triangle_fill);
            }
            out.push_back(Geometric{shape, s.light, s.material, s.alpha});
        }
        return out;
    };
    std::vector<Geometric> shapes = convert(loaded.shapes);
    std::vector<Geometric> instanced = convert(loaded.instance_shapes);
    if (sphere_fill != sphere_pool.size() ||
        triangle_fill != triangle_pool.size()) {
        fprintf(stderr, "pool fill disagrees with the count: %zu/%zu spheres, "
                        "%zu/%zu triangles\n",
                size_t(sphere_fill), sphere_pool.size(),
                size_t(triangle_fill), triangle_pool.size());
        return 1;
    }
    const Shapes shape_pools{sphere_pool.data(), triangle_pool.data()};

    // A tree in the scene file is PBRT's own, and using it is what makes a
    // timing comparison about the traversal rather than about whose builder
    // found the better tree. Without one this builds its own -- which is not a
    // fallback but the general case: PBRT can only hand over a tree of the
    // shape PBRT builds, so any schedule asking for something else (a wider
    // arity, a different bounding volume) has to build it here.
    //
    // Two kinds of tree, as pbrt's CreateAggregate builds two: one per instance
    // definition over its shapes, all of them rows of one pool, and the
    // top-level one over the scene's own shapes and its instances together.
    setup.reset();

    const bool pbrt_trees = !loaded.nodes.empty();
    std::vector<_tree_layout1> instance_nodes;
    // The row each definition's tree is rooted at.
    std::vector<uint32_t> roots(loaded.definitions.size(), 0);
    {
        Stage stage(pbrt_trees ? "adopt pbrt's instance trees"
                               : "build instance trees");
        if (pbrt_trees) {
            instance_nodes = adopt_nodes<_tree_layout1>(loaded.instance_nodes);
            for (size_t d = 0; d < roots.size(); d++) {
                roots[d] = loaded.definitions[d].root_node;
            }
        } else {
            for (size_t d = 0; d < roots.size(); d++) {
                const bonsai_scene::Definition &def = loaded.definitions[d];
                if (def.shape_count == 0) {
                    continue; // pbrt: a null primitive, never instanced.
                }
                roots[d] = build_bvh(
                    instanced.data() + def.first_shape, def.shape_count,
                    def.first_shape,
                    [&](const Geometric &g) {
                        return bounds_of(g, pool, shape_pools);
                    },
                    instance_nodes);
            }
        }
    }

    // The top-level tree's primitives, in the order its leaves name them.
    std::vector<TopLevel> top;
    top.reserve(shapes.size() + loaded.instances.size());
    const auto instance_bounds = [&](const bonsai_scene::Instance &inst) {
        const _tree_layout1 &root = instance_nodes[roots[inst.definition]];
        return transform_bounds(to_bonsai(inst.render_from_instance),
                                Bounds3f{root.low, root.high});
    };
    std::vector<_tree_layout4> nodes;
    {
        Stage stage(pbrt_trees ? "adopt pbrt's bvh" : "build bvh");
        if (pbrt_trees) {
            nodes = adopt_nodes<_tree_layout4>(loaded.nodes);
            for (const bonsai_scene::Prim &p : loaded.prims) {
                top.push_back(TopLevel{p.kind == bonsai_scene::PrimInstance,
                                       p.index, Bounds3f{}});
            }
        } else {
            for (uint32_t i = 0; i < shapes.size(); i++) {
                top.push_back(
                    TopLevel{false, i, bounds_of(shapes[i], pool, shape_pools)});
            }
            for (uint32_t i = 0; i < loaded.instances.size(); i++) {
                top.push_back(
                    TopLevel{true, i, instance_bounds(loaded.instances[i])});
            }
            build_bvh(
                top.data(), top.size(), 0,
                [](const TopLevel &t) { return t.bounds; }, nodes);
        }
    }

    // The top-level shapes in the order the tree's leaves reach them, so that
    // the pools below can be laid out to match. The instanced ones are already
    // in their leaves' order: a definition's tree was built over its run in
    // place.
    std::vector<Geometric> ordered_shapes;
    ordered_shapes.reserve(shapes.size());
    for (const TopLevel &t : top) {
        if (!t.instance) {
            ordered_shapes.push_back(shapes[t.index]);
        }
    }
    if (ordered_shapes.size() != shapes.size()) {
        fprintf(stderr, "the top-level tree names %zu shapes of %zu\n",
                ordered_shapes.size(), shapes.size());
        return 1;
    }
    shapes = std::move(ordered_shapes);

    // After the trees, because it is the trees that decide the order.
    {
        Stage stage("compact pools");
        compact_pools(shapes, instanced, sphere_pool, triangle_pool);
    }

    std::unique_ptr<Stage> lights_stage(new Stage("lights and film"));

    const uint32_t npixels = uint32_t(width) * uint32_t(height);
    float3 *out = (float3 *)malloc(sizeof(float3) * npixels);
    float3 *albedo = (float3 *)malloc(sizeof(float3) * npixels);
    float3 *radiance = (float3 *)malloc(sizeof(float3) * npixels);
    // pbrt: Pixel::weightSum. It is a film channel like the others -- the sum
    // of the filter weights the pixel's samples carried, which is what its
    // colours are divided by.
    float *weights = (float *)malloc(sizeof(float) * npixels);

    // An emitted spectrum, fitted the way every other spectrum here is -- and
    // as an illuminant rather than as an albedo, which is what the division by
    // twice the largest component is for: pbrt's RGBIlluminantSpectrum scales
    // an RGB down until the sigmoid can represent it, fits that, and multiplies
    // the colour space's own illuminant back in. The factor comes back through
    // `scale`, which the renderer applies alongside pbrt's own.
    //
    // One function because two kinds of light need it: an area light's L and a
    // uniform infinite light's.
    const auto fit_emission = [&](const float rgb[3], float scale,
                                  SigmoidPolynomial *fit) {
        const float m = std::max({rgb[0], rgb[1], rgb[2]});
        const float rsp_scale = 2.f * m;
        const float inv = rsp_scale == 0.f ? 0.f : 1.f / rsp_scale;
        const rgb2spec::Coefficients c =
            rgb2spec::fit(fit_tables, rgb[0] * inv, rgb[1] * inv, rgb[2] * inv);
        *fit = SigmoidPolynomial{c.c0, c.c1, c.c2};
        return scale * rsp_scale;
    };

    // The emission of each area light the scene declared.
    std::vector<AreaLight> emission;
    emission.reserve(loaded.lights.size());
    for (const bonsai_scene::Light &l : loaded.lights) {
        AreaLight out_light;
        out_light.scale = fit_emission(l.l, l.scale, &out_light.l);
        out_light.two_sided = l.two_sided != 0;
        emission.push_back(out_light);
    }

    // One `Light` per emissive *shape*, which is what pbrt builds: a light has
    // to know the geometry it sits on, because sampling one is sampling that
    // geometry towards a point. A scene's `AreaLightSource` directive can cover
    // many shapes -- every triangle of a mesh -- and each becomes its own light.
    //
    // Built here rather than beside the emission above because the shape handle
    // is what the Light holds, and the handles are made in the loop below.
    // The index is read off the Primitive rather than off `loaded.shapes`,
    // because by now the BVH build has reordered them and the two no longer
    // correspond. And it happens after `compact_pools`, which rewrites every
    // shape handle: a Light holds one, and a stale one would name whatever
    // moved into that slot.
    //
    // Only the top-level shapes: pbrt gives a shape inside an instance
    // definition no area light, and scene_dump wrote -1 for every one of them.
    std::vector<Light> lights;
    for (Geometric &prim : shapes) {
        if (prim.light < 0) {
            continue;
        }
        Light light;
        Light_DiffuseArea(light, emission[size_t(prim.light)], prim.shape);
        prim.light = int32_t(lights.size());
        lights.push_back(light);
    }

    // pbrt: the Primitives the top-level BVHAggregate is built over, made
    // last: a GeometricPrimitive copies its fields into the `Geom` pool, and the
    // light index above had to be settled first. Made in the leaves' order, so
    // the pool entries a leaf reaches are consecutive, as `compact_pools` did
    // for the shapes. An instance is its two matrices and the row its object's
    // tree starts at.
    std::vector<Geom> geom_pool(shapes.size());
    std::vector<Inst> inst_pool(loaded.instances.size());
    uint64_t geom_fill = 0;
    uint64_t inst_fill = 0;
    std::vector<uint64_t> prims;
    prims.reserve(top.size());
    {
        size_t next_shape = 0;
        for (const TopLevel &t : top) {
            if (t.instance) {
                const bonsai_scene::Instance &inst = loaded.instances[t.index];
                prims.push_back(Primitive_Inst(
                    to_bonsai(inst.render_from_instance),
                    to_bonsai(inst.instance_from_render),
                    roots[inst.definition], inst_pool.data(), &inst_fill));
            } else {
                prims.push_back(Primitive_Geom(shapes[next_shape++],
                                               geom_pool.data(), &geom_fill));
            }
        }
    }
    if (geom_fill != geom_pool.size() || inst_fill != inst_pool.size()) {
        fprintf(stderr, "primitive pool fill disagrees with the count: "
                        "%zu/%zu shapes, %zu/%zu instances\n",
                size_t(geom_fill), geom_pool.size(), size_t(inst_fill),
                inst_pool.size());
        return 1;
    }

    _tree_layout0 tree;
    tree.gCount = uint32_t(instanced.size());
    tree.geoms = instanced.data();
    tree.bCount = uint32_t(instance_nodes.size());
    tree.group0_bnode = instance_nodes.data();
    tree.pCount = uint32_t(prims.size());
    tree.prims = prims.data();
    tree.nCount = uint32_t(nodes.size());
    tree.group1_index = nodes.data();

    // And the lights that are not shapes, appended after the area ones --
    // which is the order pbrt builds its own list in, area lights from
    // `CreateLights`'s loop over shapes and then the rest from its light jobs.
    // The order is not cosmetic: a uniform light sampler picks `lights[u * n]`,
    // so two lists in different orders hand the same random number to different
    // lights and render different images.
    //
    // The renderer finds them as a subrange rather than as a second array,
    // which is what `LightSet` says: everything from `first_infinite` to
    // `count` is a light a ray can fly off into.
    // Every environment map's texels, already fitted: four floats each, the
    // three sigmoid coefficients and the scale of the RGBIlluminantSpectrum
    // `ImageLe` would have built.
    //
    // Fitted by scene_dump, through PBRT's own RGB-to-spectrum table. It used
    // to be fitted here instead, with the Gauss-Newton solve in rgb2spec.h --
    // which is a different function from the one PBRT evaluates, and which on a
    // scene with a large sky took four and a half minutes against a render of
    // one second, in silence, and read as a hang.
    std::vector<float4> env_texels(loaded.env_texels.size() / 4);
    for (size_t i = 0; i < env_texels.size(); i++) {
        env_texels[i] =
            float4{loaded.env_texels[4 * i + 0], loaded.env_texels[4 * i + 1],
                   loaded.env_texels[4 * i + 2], loaded.env_texels[4 * i + 3]};
    }

    // pbrt: the ImageInfiniteLight constructor, which builds two
    // `PiecewiseConstant2D`s over the map.
    //
    // The first samples the map itself. The second is pbrt's MIS-compensated
    // one: the map with its own average subtracted and clamped at zero, so that
    // it samples only the part of the sky that is brighter than average. That
    // is what `allowIncompletePDF` selects, and the reasoning is that a BSDF
    // sample finds an even sky perfectly well by itself, so the light sampler
    // should spend its effort on what the BSDF sample is bad at.
    //
    // Built here rather than in scene_dump for the same reason the filter's
    // table is built here: it is derived from data the scene file already
    // carries, so writing it out as well would be writing a thing that can be
    // recomputed -- and it is thirty-four megabytes of it.
    //
    // The function being sampled is `Image::GetSamplingDistribution`: the
    // *average of the texel's channels*, taken from the raw image values and
    // not from the fitted spectrum above. scene_dump ships it as
    // `env_sampling`, because the raw image is not in `env_texels` to be
    // averaged -- what is there is the sigmoid the texel was fitted to, and a
    // black texel's fit is minus infinity. Averaging that built a CDF of
    // infinities, which sampled a NaN direction, which is how a fifth of a
    // 2048x2048 sky came back as an unwritable pixel.
    std::vector<float> env_dist_values, env_dist_cond_cdf, env_dist_marg_func,
        env_dist_marg_cdf;
    std::vector<Dist2D> env_dists; // Two per light: plain, then compensated.
    {
        // One PiecewiseConstant1D over `n` values, appended to the pools. The
        // CDF is the running integral over a unit domain, normalized by its own
        // total -- and where that total is zero the CDF is the uniform one, so
        // that a row the map never reaches is still sampleable.
        const auto build_1d = [](const float *values, size_t n,
                                 std::vector<float> *cdf) {
            const size_t base = cdf->size();
            cdf->push_back(0.f);
            for (size_t i = 0; i < n; i++) {
                cdf->push_back((*cdf)[base + i] +
                               std::abs(values[i]) / float(n));
            }
            const float total = (*cdf)[base + n];
            for (size_t i = 1; i <= n; i++) {
                (*cdf)[base + i] =
                    total == 0.f ? float(i) / float(n) : (*cdf)[base + i] / total;
            }
            return total;
        };
        const auto build_2d = [&](const std::vector<float> &values, size_t first,
                                  uint32_t res) {
            Dist2D d;
            d.first_value = int32_t(first);
            d.first_cond_cdf = int32_t(env_dist_cond_cdf.size());
            d.first_marg_func = int32_t(env_dist_marg_func.size());
            for (uint32_t y = 0; y < res; y++) {
                env_dist_marg_func.push_back(
                    build_1d(values.data() + first + size_t(y) * res, res,
                             &env_dist_cond_cdf));
            }
            d.first_marg_cdf = int32_t(env_dist_marg_cdf.size());
            d.integral = build_1d(env_dist_marg_func.data() + d.first_marg_func,
                                  res, &env_dist_marg_cdf);
            return d;
        };

        for (const bonsai_scene::InfiniteLight &l : loaded.infinite_lights) {
            if (l.resolution == 0) {
                continue;
            }
            const size_t n = size_t(l.resolution) * l.resolution;
            const size_t plain = env_dist_values.size();
            double total = 0.;
            for (size_t i = 0; i < n; i++) {
                const size_t t = size_t(l.first_texel) + i;
                const float avg = loaded.env_sampling[t];
                env_dist_values.push_back(avg);
                total += avg;
            }
            // pbrt: `average = accumulate(d) / d.size()`, then
            // `v = max(v - average, 0)`, and a uniform fallback if that leaves
            // nothing -- a map with no above-average texel at all.
            const float average = float(total / double(n));
            const size_t compensated = env_dist_values.size();
            bool any = false;
            for (size_t i = 0; i < n; i++) {
                const float v =
                    std::max(0.f, env_dist_values[plain + i] - average);
                any = any || v != 0.f;
                env_dist_values.push_back(v);
            }
            if (!any) {
                std::fill(env_dist_values.begin() + std::ptrdiff_t(compensated),
                          env_dist_values.end(), 1.f);
            }
            env_dists.push_back(build_2d(env_dist_values, plain, l.resolution));
            env_dists.push_back(
                build_2d(env_dist_values, compensated, l.resolution));
        }
    }

    const int32_t first_infinite = int32_t(lights.size());
    size_t next_dist = 0;
    for (const bonsai_scene::InfiniteLight &l : loaded.infinite_lights) {
        if (l.resolution != 0) {
            const auto rows = [](const float m[16]) {
                return Transform{float4{m[0], m[1], m[2], m[3]},
                                 float4{m[4], m[5], m[6], m[7]},
                                 float4{m[8], m[9], m[10], m[11]},
                                 float4{m[12], m[13], m[14], m[15]}};
            };
            ImageInfiniteLight img;
            img.light_from_render = rows(l.light_from_render);
            img.render_from_light = rows(l.render_from_light);
            img.scale = l.scale;
            img.resolution = int32_t(l.resolution);
            img.first_texel = int32_t(l.first_texel);
            img.scene_radius = loaded.scene_radius;
            img.dist = env_dists[next_dist++];
            img.compensated = env_dists[next_dist++];
            Light light;
            Light_ImageInfinite(light, img);
            lights.push_back(light);
            continue;
        }
        UniformInfiniteLight sky;
        // Fitted exactly as an area light's L is. The fit is meaningless when
        // the scene wrote no L, and `has_l` is what says so; pbrt emits the
        // colour space's illuminant itself in that case, which
        // `uniform_infinite_le` reaches directly and which is why the scale
        // must not pick up the fit's factor either.
        if (l.has_l) {
            sky.scale = fit_emission(l.l, l.scale, &sky.l);
        } else {
            sky.scale = l.scale;
            sky.l = SigmoidPolynomial{0.f, 0.f, 0.f};
        }
        sky.has_l = l.has_l != 0;
        sky.scene_radius = loaded.scene_radius;
        Light light;
        Light_UniformInfinite(light, sky);
        lights.push_back(light);
    }

    LightSet light_set;
    light_set.count = int32_t(lights.size());
    light_set.first_infinite = first_infinite;

    // The integrator the scene named, built through the constructor the
    // Integrator variant generated. This is the whole of what a vtable does
    // here: the driver picks a variant, hands it across once, and the bonsai
    // side matches on it -- so adding SimplePath means adding an arm and a case
    // here, not another exported entry point.
    Integrator integrator;
    switch (loaded.integrator) {
    case bonsai_scene::IntegratorTag::RandomWalk:
        Integrator_RandomWalk(integrator, loaded.max_depth, light_set);
        break;
    case bonsai_scene::IntegratorTag::SimplePath: {
        // The light sampler is the integrator's, as it is in pbrt: which light
        // to try is a decision about sampling and not about the scene.
        LightSampler light_sampler;
        LightSampler_UniformLights(light_sampler, int32_t(lights.size()));
        Integrator_SimplePath(integrator, loaded.max_depth, light_sampler,
                              light_set);
        break;
    }
    case bonsai_scene::IntegratorTag::Path: {
        // Uniform here too. pbrt's `path` defaults to its BVH light sampler,
        // and scene_dump refuses any scene where the two could disagree -- see
        // the note there.
        LightSampler light_sampler;
        LightSampler_UniformLights(light_sampler, int32_t(lights.size()));
        Integrator_Path(integrator, loaded.max_depth, light_sampler, light_set,
                        loaded.regularize != 0);
        break;
    }
    default:
        fprintf(stderr, "unknown integrator tag %u\n", loaded.integrator);
        return 1;
    }

    lights_stage.reset();

    // The tables the spectral conversion reads, as the generated header wants
    // them. Checked against a running pbrt by `scene_dump --check-tables`.
    std::array<float, CIE_SAMPLES> x, y, z, d65;
    std::copy(CIE_X, CIE_X + CIE_SAMPLES, x.begin());
    std::copy(CIE_Y, CIE_Y + CIE_SAMPLES, y.begin());
    std::copy(CIE_Z, CIE_Z + CIE_SAMPLES, z.begin());
    std::copy(CIE_D65_FILM, CIE_D65_FILM + CIE_SAMPLES, d65.begin());

    // Only the render is timed. Building the scene and the BVH is the work
    // pbrt does before its own timer starts (its renderTimeSeconds comes from
    // a progress reporter created after the scene is built), so counting it
    // here would be comparing two different things.
    //
    // The best of several runs rather than one, because the thing being
    // measured is how long the work takes and every source of noise here only
    // ever adds: a scheduler taking the core away, another process evicting
    // the cache, the clock still ramping. None of them can make a render
    // finish sooner than it can, so the minimum is the closest estimate of it,
    // where a mean is an estimate of the machine's mood. The render is a pure
    // function of its inputs, so repeating it is free of consequences.
    int repeats = 5;
    if (const char *r = getenv("BONSAI_REPEATS")) {
        repeats = std::max(1, atoi(r));
    }
    double seconds = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
        const auto started = std::chrono::steady_clock::now();
        render(camera, uint32_t(width), uint32_t(height), sampler, integrator,
               pixel_filter, loaded.seed, loaded.disable_pixel_jitter != 0,
               loaded.imaging_ratio, loaded.max_component_value, out,
               albedo, radiance, weights, textures.data(),
               texture_levels.data(), texture_texels.data(),
               loaded.rgb_table.data(), pl2d.data(), loaded.pl_data.data(),
               loaded.pl_marginal.data(), loaded.pl_conditional.data(),
               loaded.pl_params.data(), measured_brdfs.data(),
               loaded.conductor_eta.data(), loaded.conductor_k.data(),
               meshes.data(),
               loaded.indices.data(), positions.data(), normals.data(),
               uvs.data(), x, y, z, d65, filter_f.data(),
               filter_cond_cdf.data(), filter_marg_func.data(),
               filter_marg_cdf.data(), primes, digit_permutations.data(),
               digit_permutation_offsets, env_texels.data(),
               env_dist_values.data(), env_dist_cond_cdf.data(),
               env_dist_marg_func.data(), env_dist_marg_cdf.data(),
               lights.data(),
               materials.data(), material_displacement.data(), rho_uc, rho_ux,
               rho_uy, tree, geom_pool.data(), inst_pool.data(),
               sphere_pool.data(), triangle_pool.data());
        const auto finished = std::chrono::steady_clock::now();
        seconds = std::min(
            seconds, std::chrono::duration<double>(finished - started).count());
    }

    // What pbrt writes: the film's linear values, unencoded. pbrt quantizes
    // only when asked for a .png or a .qoi, and applies its sRGB transfer
    // function when it does; that is post-processing, and it happens in
    // to_png.py rather than here. PFM is one of the formats pbrt itself
    // writes, so this file and one from pbrt are directly comparable.
    // Both channels the gbuffer holds, as two images. pbrt keeps them in one
    // EXR; PFM has no way to say more than three channels, so they are written
    // side by side and compared in pairs.
    const auto write_pfm = [&](const std::string &path, const float3 *pixels) {
        std::ofstream pfm(path, std::ios::binary);
        if (!pfm) {
            std::cerr << "cannot open " << path << " for writing\n";
            return false;
        }
        // PFM rows run bottom to top, and a negative scale says little-endian.
        pfm << "PF\n" << width << ' ' << height << "\n-1.000000\n";
        for (int j = height - 1; j >= 0; j--) {
            for (int i = 0; i < width; i++) {
                const float3 &v = pixels[uint32_t(j) * uint32_t(width) + i];
                const float rgb[3] = {v[0], v[1], v[2]};
                pfm.write(reinterpret_cast<const char *>(rgb), sizeof(rgb));
            }
        }
        return bool(pfm);
    };

    const std::string stem =
        std::string(output).substr(0, std::string(output).rfind('.'));
    const std::string albedo_output = stem + "-albedo.pfm";
    const std::string radiance_output = stem + "-radiance.pfm";
    const bool wrote = write_pfm(output, out) &&
                       write_pfm(albedo_output, albedo) &&
                       write_pfm(radiance_output, radiance);
    free(out);
    free(albedo);
    free(radiance);
    if (!wrote) {
        return 1;
    }

    std::cout << "wrote " << output << ", " << albedo_output << " and "
              << radiance_output << " (" << width << 'x' << height << ", "
              << shapes.size() << " shapes)\n";
    // Parsed by compare.sh. Kept to a line of its own so that it stays easy
    // to find without the script having to understand anything else here.
    std::cout << "render seconds: " << seconds << '\n';
    return 0;
}
