#pragma once

// The flat scene that scene_dump.cpp writes and render_hook.cpp reads.
//
// This is a transport format, not a scene description. It exists because PBRT's
// parser and the bonsai renderer cannot be linked into one program (see the
// comment at the top of scene_dump.cpp), so the two halves hand off through a
// file. Nobody writes one by hand -- the .pbrt file remains the only authored
// description of a scene, and this is derived from it every run.
//
// Text, so that a scene which renders wrong can be read rather than hex-dumped,
// and so that the file needs no magic number or version to guard it: a stale or
// truncated one fails when a field does not parse. Parsing is slower than
// blitting structs would be, which does not matter -- loading a scene is
// preprocessing, and is outside what either renderer counts as render time.
//
// Floats are written with nine significant digits, which is exactly what it
// takes to read a float back unchanged. Anything less would quietly move
// geometry, and the whole point of this app is that it does not.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace bonsai_scene {

enum ShapeTag : uint32_t {
    Sphere = 0,
    Triangle = 1,
};

enum MaterialTag : uint32_t {
    Diffuse = 0,
    CoatedDiffuse = 1,
};

// One material, with every texture already evaluated to a constant.
//
// The spectra travel as the RGB the scene wrote rather than as PBRT's fitted
// coefficients: PBRT turns an RGB into a spectrum with a sigmoid fit, and the
// renderer runs the same fit, so handing over the fit's output would be handing
// over the answer to a question the renderer is supposed to be answering.
//
// The defaults are PBRT's own, from CoatedDiffuseMaterial::Create, so a
// material that names nothing arrives as the one PBRT would have built.
struct Material {
    uint32_t tag = MaterialTag::Diffuse;
    float reflectance[3] = {0.5f, 0.5f, 0.5f};
    // CoatedDiffuse only. The roughness as authored, not as remapped: PBRT
    // remaps per intersection and `remaproughness` says whether it does at all.
    float u_roughness = 0.f;
    float v_roughness = 0.f;
    uint32_t remap = 1;
    float thickness = 0.01f;
    float eta = 1.5f;
    // The medium between the two interfaces. `has_medium` is not the same
    // question as whether the albedo is zero: PBRT's default is a spectrum that
    // is exactly zero, where an RGB of (0, 0, 0) put through the fit is small
    // and is not, and the layered walk branches on which it has.
    float medium_albedo[3] = {0.f, 0.f, 0.f};
    uint32_t has_medium = 0;
    float g = 0.f;
    int32_t max_depth = 10;
    int32_t n_samples = 1;
};

// One triangle mesh, as a run of each of the shared pools below.
//
// PBRT keeps a TriangleMesh per shape and a global list of them, and a Triangle
// is a pair of indices into that list and into its own triangles. This is the
// same arrangement with the meshes' arrays laid end to end, so a mesh is where
// its own run of each begins. `indices` are mesh-local, as PBRT's are, which is
// what `first_vertex` adds back.
struct Mesh {
    uint32_t first_index = 0;
    uint32_t first_vertex = 0;
    uint32_t first_normal = 0;
    uint32_t first_uv = 0;
    uint32_t has_normals = 0;
    uint32_t has_uv = 0;
    // PBRT: reverseOrientation ^ transformSwapsHandedness, which decides which
    // way the surface normal points. A mesh's, because PBRT keeps it there.
    uint32_t flip = 0;
};

// One shape, in render space.
//
// A sphere is its centre and radius. A triangle is a mesh and a triangle in it,
// which is exactly what PBRT's `Triangle` holds -- the vertex data is fetched
// from the mesh on a hit rather than copied per triangle. The renderer's own
// Shape is a variant type and stays its business: these tags are this file's,
// and the driver maps across by calling the generated constructors.
// A DiffuseAreaLight: a shape that emits, and emits the same radiance in every
// direction it faces.
//
// The only kind of light there is here so far, and the one killeroo-simple
// uses. It is also the kind that costs an integrator the least: a random walk
// never samples a light at all, it finds one by hitting it, so this needs no
// sampling routine, no PDF and no shadow ray -- only the radiance to return
// when a ray lands on it.
//
// `l` is the RGB the scene wrote, per the note on Material above: PBRT turns an
// RGB into a spectrum with a fit and the renderer runs the same fit, so handing
// over its output would be handing over the answer. `scale` is not that fit --
// it is PBRT's own `scale` parameter after the division by
// SpectrumToPhotometric that makes a radiance of one mean one nit, which is a
// property of the scene rather than of the conversion.
struct Light {
    float l[3] = {1.f, 1.f, 1.f};
    float scale = 1.f;
    // PBRT's `twosided`. A one-sided light emits only where its normal points,
    // which for a sphere is outwards.
    uint32_t two_sided = 0;
};

struct Shape {
    uint32_t tag;
    // Sphere.
    float radius = 0.f;
    float center[3] = {0.f, 0.f, 0.f};
    uint32_t flip = 0;
    // Triangle.
    uint32_t mesh = 0;
    uint32_t tri = 0;
    // Which of the scene's materials this shape was declared under.
    uint32_t material = 0;
    // Which of the scene's lights this shape emits as, or -1 for a shape that
    // does not emit. PBRT's ShapeSceneEntity::lightIndex, which is per shape
    // rather than per material because `AreaLightSource` is a graphics-state
    // directive like `Material` and the two are set independently.
    int32_t light = -1;
};

// One BVH node, in PBRT's LinearBVHNode shape.
//
// Present only when the scene was dumped with --pbrt-tree, in which case it is
// PBRT's own tree -- the nodes its BVHAggregate built and flattened -- and the
// shapes are in the order its leaves expect, so a leaf's `offset` indexes them
// directly. Without a tree the driver builds its own, which is the general
// case: PBRT can only hand over a tree of the shape PBRT builds, so a schedule
// wanting a wider arity or a different bounding volume has to build its own.
struct Node {
    float low[3];
    float high[3];
    // Leaf: the first of n_prims shapes. Interior: the second child, relative
    // to this node, the first being the next node along.
    uint32_t offset;
    uint16_t n_prims; // 0 for an interior node.
    uint16_t axis;
};

enum SamplerTag : uint32_t {
    Independent = 0,
    Stratified = 1,
    Halton = 2,
};

// pbrt: RandomizeStrategy, which is how a Halton sampler breaks up the
// correlation between its dimensions. `permutedigits` is what a scene gets
// when it does not say.
enum RandomizeTag : uint32_t {
    RandomizeNone = 0,
    RandomizePermuteDigits = 1,
    RandomizeOwen = 2,
};

// Which sampler the scene asked for, and what it was given.
//
// A scene names its sampler, so this travels with the scene rather than being
// a switch on the renderer. Reproducing pbrt's noise means drawing from the
// same stream, and which stream that is depends on the kind of sampler as much
// as on the pixel and the seed.
struct Sampler {
    uint32_t tag = SamplerTag::Independent;
    // Independent: `integer pixelsamples`. Stratified: the product of the two
    // grid dimensions, which is what pbrt reports as its sample count.
    uint32_t samples_per_pixel = 1;
    int32_t seed = 0;
    // Stratified only. Its grid, and whether a sample is jittered inside its
    // cell or sits at the centre.
    uint32_t x_samples = 1;
    uint32_t y_samples = 1;
    uint32_t jitter = 1;
    // Halton only. The randomization, and what pbrt's constructor derives from
    // the film resolution: how far the first two dimensions of the sequence
    // tile before repeating, as a scale and its exponent, and the
    // multiplicative inverse of each scale modulo the other. Those last are
    // what combine a pixel's two radical-inverse offsets into one index, and
    // they are derived rather than authored -- see scene_dump.cpp.
    uint32_t randomize = RandomizeTag::RandomizePermuteDigits;
    int32_t base_scales[2] = {1, 1};
    int32_t base_exponents[2] = {0, 0};
    int32_t mult_inverse[2] = {0, 0};
};

struct Scene {
    uint32_t width = 0;
    uint32_t height = 0;
    Sampler sampler;
    // PBRT's `--seed` option, which is not the sampler's seed: it is a global
    // that a layered BSDF hashes together with the direction it was asked
    // about, to seed the random walk that estimates its reflectance.
    int32_t seed = 0;
    // The integrator's `maxdepth`, which is how many times a path may scatter.
    // Not the same number as a layered material's `maxdepth`, which bounds the
    // walk *inside* a BSDF; PBRT's default for both happens to differ, so they
    // are carried separately rather than shared.
    int32_t max_depth = 5;
    // camera_from_raster then render_from_camera, each 4x4 in row order.
    float matrices[32] = {};
    std::vector<Material> materials;
    // The meshes, and the pools their runs live in. Three floats per position
    // and normal, two per texture coordinate.
    std::vector<Mesh> meshes;
    std::vector<uint32_t> indices;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<Light> lights;
    std::vector<Shape> shapes;
    std::vector<Node> nodes;

    // The three vertices of a triangle, as PBRT's
    // `&mesh->vertexIndices[3 * triIndex]` reads them, with the mesh's own
    // offset added. Here rather than in each of the three places that wants
    // them -- the tree build, the bounds, the PBRT-tree path.
    void corners(const Shape &s, uint32_t out[3]) const {
        const Mesh &m = meshes[s.mesh];
        for (uint32_t k = 0; k < 3; k++) {
            out[k] = m.first_vertex + indices[m.first_index + 3 * s.tri + k];
        }
    }
};

namespace detail {

inline void put(std::ofstream &out, const float *v, int n) {
    char buf[32];
    for (int i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), " %.9g", double(v[i]));
        out << buf;
    }
}

} // namespace detail

inline bool write(const char *path, const Scene &scene) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "resolution " << scene.width << ' ' << scene.height << '\n';
    if (scene.sampler.tag == SamplerTag::Stratified) {
        out << "sampler stratified " << scene.sampler.x_samples << ' '
            << scene.sampler.y_samples << ' ' << scene.sampler.seed << ' '
            << scene.sampler.jitter << '\n';
    } else if (scene.sampler.tag == SamplerTag::Halton) {
        out << "sampler halton " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << ' ' << scene.sampler.randomize << ' '
            << scene.sampler.base_scales[0] << ' '
            << scene.sampler.base_scales[1] << ' '
            << scene.sampler.base_exponents[0] << ' '
            << scene.sampler.base_exponents[1] << ' '
            << scene.sampler.mult_inverse[0] << ' '
            << scene.sampler.mult_inverse[1] << '\n';
    } else {
        out << "sampler independent " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << '\n';
    }
    out << "seed " << scene.seed << '\n';
    out << "maxdepth " << scene.max_depth << '\n';
    out << "camera_from_raster";
    detail::put(out, scene.matrices, 16);
    out << "\nrender_from_camera";
    detail::put(out, scene.matrices + 16, 16);
    out << '\n';

    out << "materials " << scene.materials.size() << '\n';
    for (const Material &m : scene.materials) {
        out << (m.tag == MaterialTag::CoatedDiffuse ? "  coateddiffuse"
                                                    : "  diffuse");
        out << " reflectance";
        detail::put(out, m.reflectance, 3);
        if (m.tag == MaterialTag::CoatedDiffuse) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " remap " << m.remap;
            out << " thickness";
            detail::put(out, &m.thickness, 1);
            out << " eta";
            detail::put(out, &m.eta, 1);
            out << " albedo";
            detail::put(out, m.medium_albedo, 3);
            out << " hasalbedo " << m.has_medium;
            out << " g";
            detail::put(out, &m.g, 1);
            out << " maxdepth " << m.max_depth;
            out << " nsamples " << m.n_samples;
        }
        out << '\n';
    }

    out << "meshes " << scene.meshes.size() << '\n';
    for (const Mesh &m : scene.meshes) {
        out << "  mesh " << m.first_index << ' ' << m.first_vertex << ' '
            << m.first_normal << ' ' << m.first_uv << " normals "
            << m.has_normals << " uv " << m.has_uv << " flip " << m.flip
            << '\n';
    }
    out << "indices " << scene.indices.size() << '\n';
    for (const uint32_t i : scene.indices) {
        out << ' ' << i;
    }
    out << '\n';
    const auto pool = [&](const char *name, const std::vector<float> &values,
                          int per) {
        out << name << ' ' << values.size() / per << '\n';
        for (size_t i = 0; i < values.size(); i += per) {
            detail::put(out, values.data() + i, per);
            out << '\n';
        }
    };
    pool("positions", scene.positions, 3);
    pool("normals", scene.normals, 3);
    pool("uvs", scene.uvs, 2);

    out << "lights " << scene.lights.size() << '\n';
    for (const Light &l : scene.lights) {
        out << "  diffuse";
        detail::put(out, l.l, 3);
        detail::put(out, &l.scale, 1);
        out << " twosided " << l.two_sided << '\n';
    }

    out << "shapes " << scene.shapes.size() << '\n';
    for (const Shape &s : scene.shapes) {
        if (s.tag == ShapeTag::Sphere) {
            out << "  sphere";
            detail::put(out, s.center, 3);
            detail::put(out, &s.radius, 1);
            out << " flip " << s.flip;
        } else {
            out << "  tri " << s.mesh << ' ' << s.tri;
        }
        out << " material " << s.material << " light " << s.light << '\n';
    }

    out << "nodes " << scene.nodes.size() << '\n';
    for (const Node &n : scene.nodes) {
        out << (n.n_prims == 0 ? "  interior" : "  leaf");
        detail::put(out, n.low, 3);
        detail::put(out, n.high, 3);
        if (n.n_prims == 0) {
            out << " axis " << n.axis << " right " << n.offset;
        } else {
            out << " first " << n.offset << " count " << n.n_prims;
        }
        out << '\n';
    }
    return bool(out);
}

inline bool read(const char *path, Scene &scene) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string word;

    auto floats = [&](float *v, int n) {
        for (int i = 0; i < n; i++) {
            in >> v[i];
        }
    };

    if (!(in >> word) || word != "resolution") {
        return false;
    }
    in >> scene.width >> scene.height;

    if (!(in >> word) || word != "sampler") {
        return false;
    }
    if (!(in >> word)) {
        return false;
    }
    if (word == "stratified") {
        scene.sampler.tag = SamplerTag::Stratified;
        in >> scene.sampler.x_samples >> scene.sampler.y_samples >>
            scene.sampler.seed >> scene.sampler.jitter;
        scene.sampler.samples_per_pixel =
            scene.sampler.x_samples * scene.sampler.y_samples;
    } else if (word == "halton") {
        scene.sampler.tag = SamplerTag::Halton;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed >>
            scene.sampler.randomize >> scene.sampler.base_scales[0] >>
            scene.sampler.base_scales[1] >> scene.sampler.base_exponents[0] >>
            scene.sampler.base_exponents[1] >> scene.sampler.mult_inverse[0] >>
            scene.sampler.mult_inverse[1];
    } else if (word == "independent") {
        scene.sampler.tag = SamplerTag::Independent;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed;
    } else {
        return false;
    }

    if (!(in >> word) || word != "seed") {
        return false;
    }
    in >> scene.seed;

    if (!(in >> word) || word != "maxdepth") {
        return false;
    }
    in >> scene.max_depth;

    if (!(in >> word) || word != "camera_from_raster") {
        return false;
    }
    floats(scene.matrices, 16);
    if (!(in >> word) || word != "render_from_camera") {
        return false;
    }
    floats(scene.matrices + 16, 16);

    // A labelled field, checked as it is read. The labels are what makes a
    // stale file fail here rather than three fields later with plausible
    // numbers in the wrong places.
    const auto tagged = [&](const char *name) {
        return bool(in >> word) && word == name;
    };

    size_t count = 0;
    if (!(in >> word) || word != "materials") {
        return false;
    }
    in >> count;
    scene.materials.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Material m;
        if (word == "diffuse") {
            m.tag = MaterialTag::Diffuse;
        } else if (word == "coateddiffuse") {
            m.tag = MaterialTag::CoatedDiffuse;
        } else {
            return false;
        }
        if (!tagged("reflectance")) {
            return false;
        }
        floats(m.reflectance, 3);
        if (m.tag == MaterialTag::CoatedDiffuse) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (!tagged("thickness")) {
                return false;
            }
            floats(&m.thickness, 1);
            if (!tagged("eta")) {
                return false;
            }
            floats(&m.eta, 1);
            if (!tagged("albedo")) {
                return false;
            }
            floats(m.medium_albedo, 3);
            if (!tagged("hasalbedo")) {
                return false;
            }
            in >> m.has_medium;
            if (!tagged("g")) {
                return false;
            }
            floats(&m.g, 1);
            if (!tagged("maxdepth")) {
                return false;
            }
            in >> m.max_depth;
            if (!tagged("nsamples")) {
                return false;
            }
            in >> m.n_samples;
        }
        scene.materials.push_back(m);
    }

    if (!(in >> word) || word != "meshes") {
        return false;
    }
    in >> count;
    scene.meshes.clear();
    for (size_t i = 0; i < count; i++) {
        if (!tagged("mesh")) {
            return false;
        }
        Mesh m;
        in >> m.first_index >> m.first_vertex >> m.first_normal >> m.first_uv;
        if (!tagged("normals")) {
            return false;
        }
        in >> m.has_normals;
        if (!tagged("uv")) {
            return false;
        }
        in >> m.has_uv;
        if (!tagged("flip")) {
            return false;
        }
        in >> m.flip;
        scene.meshes.push_back(m);
    }

    if (!(in >> word) || word != "indices") {
        return false;
    }
    in >> count;
    scene.indices.assign(count, 0);
    for (size_t i = 0; i < count; i++) {
        in >> scene.indices[i];
    }
    const auto pool = [&](const char *name, std::vector<float> &values,
                          int per) {
        if (!(in >> word) || word != name) {
            return false;
        }
        size_t n = 0;
        in >> n;
        values.assign(n * per, 0.f);
        for (size_t i = 0; i < n * size_t(per); i++) {
            in >> values[i];
        }
        return true;
    };
    if (!pool("positions", scene.positions, 3) ||
        !pool("normals", scene.normals, 3) || !pool("uvs", scene.uvs, 2)) {
        return false;
    }

    if (!(in >> word) || word != "lights") {
        return false;
    }
    in >> count;
    scene.lights.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word) || word != "diffuse") {
            return false;
        }
        Light l;
        floats(l.l, 3);
        floats(&l.scale, 1);
        if (!tagged("twosided")) {
            return false;
        }
        in >> l.two_sided;
        scene.lights.push_back(l);
    }

    if (!(in >> word) || word != "shapes") {
        return false;
    }
    in >> count;
    scene.shapes.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Shape s;
        if (word == "sphere") {
            s.tag = ShapeTag::Sphere;
            floats(s.center, 3);
            floats(&s.radius, 1);
            if (!tagged("flip")) {
                return false;
            }
            in >> s.flip;
        } else if (word == "tri") {
            s.tag = ShapeTag::Triangle;
            in >> s.mesh >> s.tri;
            if (s.mesh >= scene.meshes.size()) {
                return false;
            }
        } else {
            return false;
        }
        if (!tagged("material")) {
            return false;
        }
        in >> s.material;
        if (s.material >= scene.materials.size()) {
            return false;
        }
        if (!tagged("light")) {
            return false;
        }
        in >> s.light;
        if (s.light >= int32_t(scene.lights.size())) {
            return false;
        }
        scene.shapes.push_back(s);
    }

    if (!(in >> word) || word != "nodes") {
        return false;
    }
    in >> count;
    scene.nodes.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Node n = {};
        const bool interior = word == "interior";
        if (!interior && word != "leaf") {
            return false;
        }
        floats(n.low, 3);
        floats(n.high, 3);
        std::string a, b;
        uint32_t x = 0, y = 0;
        in >> a >> x >> b >> y;
        if (interior) {
            if (a != "axis" || b != "right") {
                return false;
            }
            n.axis = uint16_t(x);
            n.offset = y;
            n.n_prims = 0;
        } else {
            if (a != "first" || b != "count") {
                return false;
            }
            n.offset = x;
            n.n_prims = uint16_t(y);
        }
        scene.nodes.push_back(n);
    }

    return bool(in);
}

} // namespace bonsai_scene
