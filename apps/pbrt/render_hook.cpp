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
Bounds3f bounds_of(const Primitive &prim, const Meshes &pool,
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
_tree_layout0 build_bvh(std::vector<Primitive> &shapes, const Meshes &pool,
                        const Shapes &shape_pools) {
    std::vector<BVHPrimitive> prims;
    prims.reserve(shapes.size());
    for (uint32_t i = 0; i < shapes.size(); i++) {
        prims.push_back(BVHPrimitive{i, bounds_of(shapes[i], pool, shape_pools)});
    }

    // pbrt: orderedPrims. A leaf names a run of primitives, so the primitives
    // it names have to be contiguous, which means the scene is rewritten into
    // the order the build discovered.
    std::vector<Primitive> ordered;
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
void compact_pools(std::vector<Primitive> &shapes, std::vector<Sph> &spheres,
                   std::vector<Tri> &triangles) {
    std::vector<Sph> ordered_spheres;
    std::vector<Tri> ordered_triangles;
    ordered_spheres.reserve(spheres.size());
    ordered_triangles.reserve(triangles.size());

    for (Primitive &prim : shapes) {
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

    spheres = std::move(ordered_spheres);
    triangles = std::move(ordered_triangles);
}

// The tree PBRT built, packed into the layout the schedule declared.
//
// No building happens here: the nodes arrive already flattened depth first,
// with the shapes in the order its leaves expect, so this is only a change of
// representation. That the two layouts line up field for field is not luck --
// the `layout` block in render.bonsai was written to be LinearBVHNode.
_tree_layout0 adopt_bvh(std::vector<Primitive> &shapes,
                        const std::vector<bonsai_scene::Node> &nodes) {
    _tree_layout0 tree;
    tree.pCount = uint32_t(shapes.size());
    tree.prims = shapes.data();
    tree.nCount = uint32_t(nodes.size());
    tree.group0_index =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.nCount);

    for (size_t i = 0; i < nodes.size(); i++) {
        const bonsai_scene::Node &n = nodes[i];
        tree.group0_index[i].low = float3{n.low[0], n.low[1], n.low[2]};
        tree.group0_index[i].high = float3{n.high[0], n.high[1], n.high[2]};
        tree.group0_index[i].nPrims = n.n_prims;
        tree.group0_index[i].axis = uint8_t(n.axis);
        std::memcpy(tree.group0_index[i].split0on_nPrims.data(), &n.offset,
                    sizeof(n.offset));
    }
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
    bonsai_scene::Scene loaded;
    if (!bonsai_scene::read(scene_path, loaded)) {
        std::cerr << "cannot read scene " << scene_path
                  << " (run scene_dump on a .pbrt first)\n";
        return 1;
    }

    const int width = int(loaded.width);
    const int height = int(loaded.height);

    PerspectiveCamera camera;
    camera.camera_from_raster = to_bonsai(&loaded.matrices[0]);
    camera.render_from_camera = to_bonsai(&loaded.matrices[16]);

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
    // constructors rather than by setting the tag: which number a variant is
    // belongs to the compiler.
    std::vector<Material> materials;
    materials.reserve(loaded.materials.size());
    for (const bonsai_scene::Material &m : loaded.materials) {
        Material material;
        if (m.tag == bonsai_scene::MaterialTag::CoatedDiffuse) {
            CoatedDiffuseMaterial coated;
            coated.reflectance = albedo_of(m.reflectance);
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
        } else {
            Material_Diffuse(material, albedo_of(m.reflectance));
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
    size_t nspheres = 0;
    for (const bonsai_scene::Shape &s : loaded.shapes) {
        nspheres += s.tag == bonsai_scene::ShapeTag::Sphere;
    }
    std::vector<Sph> sphere_pool(nspheres);
    std::vector<Tri> triangle_pool(loaded.shapes.size() - nspheres);
    // What the constructors bump. Each ends up equal to its pool's size, which
    // is the check that the two passes counted the same thing.
    uint64_t sphere_fill = 0;
    uint64_t triangle_fill = 0;

    std::vector<Primitive> shapes;
    shapes.reserve(loaded.shapes.size());
    for (const bonsai_scene::Shape &s : loaded.shapes) {
        // The generated constructors rather than a handle assembled here: they
        // put the fields in the pool and return the tag and index naming them,
        // and they are generated from the same layout the renderer reads with,
        // so the numbering cannot drift.
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
        shapes.push_back(Primitive{shape, s.light, s.material});
    }
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
    _tree_layout0 tree = loaded.nodes.empty()
                             ? build_bvh(shapes, pool, shape_pools)
                             : adopt_bvh(shapes, loaded.nodes);

    // After the tree, because it is the tree that decides the order. `shapes`
    // is rewritten in place, so the `prims` the layout above points at is
    // still the same array.
    compact_pools(shapes, sphere_pool, triangle_pool);

    const uint32_t npixels = uint32_t(width) * uint32_t(height);
    float3 *out = (float3 *)malloc(sizeof(float3) * npixels);
    float3 *albedo = (float3 *)malloc(sizeof(float3) * npixels);
    float3 *radiance = (float3 *)malloc(sizeof(float3) * npixels);

    // The lights, with L fitted the same way every other spectrum is. An
    // illuminant rather than an albedo, which is why the fit is of L divided by
    // twice its largest component and the scale carries that factor back:
    // pbrt's RGBIlluminantSpectrum is a fit scaled to fit inside the sigmoid,
    // multiplied by the colour space's own illuminant.
    // The emission of each light the scene declared, with L fitted the same way
    // every other spectrum is. An illuminant rather than an albedo, which is
    // why the fit is of L divided by twice its largest component and the scale
    // carries that factor back: pbrt's RGBIlluminantSpectrum is a fit scaled to
    // fit inside the sigmoid, multiplied by the colour space's own illuminant.
    std::vector<AreaLight> emission;
    emission.reserve(loaded.lights.size());
    for (const bonsai_scene::Light &l : loaded.lights) {
        const float m = std::max({l.l[0], l.l[1], l.l[2]});
        const float rsp_scale = 2.f * m;
        const float inv = rsp_scale == 0.f ? 0.f : 1.f / rsp_scale;
        const rgb2spec::Coefficients c = rgb2spec::fit(
            fit_tables, l.l[0] * inv, l.l[1] * inv, l.l[2] * inv);
        AreaLight out_light;
        out_light.l = SigmoidPolynomial{c.c0, c.c1, c.c2};
        out_light.scale = l.scale * rsp_scale;
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
    std::vector<Light> lights;
    for (Primitive &prim : shapes) {
        if (prim.light < 0) {
            continue;
        }
        Light light;
        Light_DiffuseArea(light, emission[size_t(prim.light)], prim.shape);
        prim.light = int32_t(lights.size());
        lights.push_back(light);
    }

    // The integrator the scene named, built through the constructor the
    // Integrator variant generated. This is the whole of what a vtable does
    // here: the driver picks a variant, hands it across once, and the bonsai
    // side matches on it -- so adding SimplePath means adding an arm and a case
    // here, not another exported entry point.
    Integrator integrator;
    switch (loaded.integrator) {
    case bonsai_scene::IntegratorTag::RandomWalk:
        Integrator_RandomWalk(integrator, loaded.max_depth);
        break;
    case bonsai_scene::IntegratorTag::SimplePath: {
        // The light sampler is the integrator's, as it is in pbrt: which light
        // to try is a decision about sampling and not about the scene.
        LightSampler light_sampler;
        LightSampler_UniformLights(light_sampler, int32_t(lights.size()));
        Integrator_SimplePath(integrator, loaded.max_depth, light_sampler);
        break;
    }
    default:
        fprintf(stderr, "unknown integrator tag %u\n", loaded.integrator);
        return 1;
    }

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
               loaded.seed, out, albedo, radiance, meshes.data(),
               loaded.indices.data(), positions.data(), normals.data(),
               uvs.data(), x, y, z, d65, primes, lights.data(),
               materials.data(), rho_uc, rho_ux, rho_uy, tree,
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
    free(tree.group0_index);
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
