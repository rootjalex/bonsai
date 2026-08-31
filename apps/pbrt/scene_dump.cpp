// Reads a .pbrt scene with PBRT's own parser and writes what apps/pbrt needs.
//
// The scene file is the single source of truth. Nothing here re-implements
// PBRT's grammar, its defaults, its transform stack or its camera derivation:
// it links PBRT and asks. What comes out the other end is a flat description
// the renderer's driver can load without knowing anything about .pbrt at all.
//
// This is a separate program rather than part of render_hook.cpp for a reason
// that is not aesthetic. PBRT's headers only compile with the toolchain PBRT
// was built with, and mixing that toolchain's objects with the ones the bonsai
// backend emits means mixing two standard libraries in one link. A pipe
// between two programs has no ABI. It also keeps the driver small and free of
// PBRT headers, which is what the app's rule asks for.
//
// Scene loading is preprocessing and is not part of what gets timed, on either
// side, so the cost of the extra step does not enter the comparison.
//
//     scene_dump scene.pbrt scene.bin
//
// See scene_io.h for the format.

#include <pbrt/pbrt.h>

#include <pbrt/base/material.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "scene_io.h"

namespace {

// PBRT hands the camera and the film to BasicScene through methods that keep
// them to itself, and the renderer needs the resolution and the field of view
// rather than a constructed Camera. So this listens in: the parser's calls go
// to the base as usual, and what they carried is kept alongside.
class CapturingBuilder : public pbrt::BasicSceneBuilder {
  public:
    explicit CapturingBuilder(pbrt::BasicScene *scene)
        : pbrt::BasicSceneBuilder(scene) {
        // BasicSceneBuilder's own constructor registers a default `diffuse`
        // before any directive is seen, which is what a shape declared outside
        // every Material block ends up with. Recording it here keeps the
        // indices below lined up with PBRT's.
        MaterialInfo implicit;
        implicit.name = "diffuse";
        materials.push_back(std::move(implicit));
    }

    void Camera(const std::string &name, pbrt::ParsedParameterVector params,
                pbrt::FileLoc loc) override {
        camera_name = name;
        // The vector holds pointers, so a copy still refers to the parameters
        // the base is about to take; reading them here does not consume them.
        camera_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Camera(name, std::move(params), loc);
    }

    void Film(const std::string &type, pbrt::ParsedParameterVector params,
              pbrt::FileLoc loc) override {
        film_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Film(type, std::move(params), loc);
    }

    // What M1 needs of a material: its kind, and the reflectance if it names
    // one as RGB. The parameters are read here in their parsed form rather than
    // through a ParameterDictionary, because an `rgb reflectance` reaches a
    // dictionary already turned into an RGBAlbedoSpectrum -- PBRT's own sigmoid
    // fit -- and that fit is the thing the renderer is supposed to be doing.
    // Taking the three numbers as written leaves it that way.
    struct MaterialInfo {
        std::string name;
        bool has_reflectance = false;
        bool reflectance_is_rgb = false;
        float reflectance[3] = {0.f, 0.f, 0.f};
    };

    // A shape names its material by an index, and the materials themselves are
    // BasicScene's to keep. BasicScene::AddMaterial appends, and the base's
    // Material() calls it once per directive, so recording them here in the
    // same order gives a list the shapes' materialIndex indexes into.
    void Material(const std::string &name, pbrt::ParsedParameterVector params,
                  pbrt::FileLoc loc) override {
        MaterialInfo info;
        info.name = name;
        for (const pbrt::ParsedParameter *p : params) {
            if (p->name != "reflectance") {
                continue;
            }
            info.has_reflectance = true;
            if (p->type == "rgb" && p->floats.size() == 3) {
                info.reflectance_is_rgb = true;
                for (int i = 0; i < 3; i++) {
                    info.reflectance[i] = float(p->floats[i]);
                }
            }
        }
        materials.push_back(std::move(info));
        pbrt::BasicSceneBuilder::Material(name, std::move(params), loc);
    }

    std::string camera_name;
    pbrt::ParameterDictionary camera_params;
    pbrt::ParameterDictionary film_params;
    std::vector<MaterialInfo> materials;
};

[[noreturn]] void fail(const std::string &message) {
    fprintf(stderr, "scene_dump: %s\n", message.c_str());
    exit(1);
}

// The reflectance a shape's material carries, as RGB.
//
// This is all M1 needs. The albedo PBRT's gbuffer records for a diffuse
// surface is exactly the material's reflectance, because a Lambertian BRDF's
// hemispherical reflectance is its reflectance -- PBRT estimates it with
// sixteen cosine-weighted samples and every one of them returns the same
// number. Anything else -- textures, conductors, dielectrics -- is a later
// milestone, and saying so is better than substituting a grey and calling the
// result a match.
void material_reflectance(const std::vector<CapturingBuilder::MaterialInfo> &materials,
                          int index, float *rgb) {
    // PBRT's default reflectance for `diffuse`, and what a shape declared
    // outside any Material directive gets.
    rgb[0] = rgb[1] = rgb[2] = 0.5f;
    if (index < 0) {
        return;
    }
    if (index >= int(materials.size())) {
        fail("a shape names a material that was never declared");
    }
    const CapturingBuilder::MaterialInfo &m = materials[index];
    if (m.name.empty() || m.name == "none") {
        return;
    }
    if (m.name != "diffuse") {
        fail("only the diffuse material is supported, scene asks for \"" +
             m.name + "\"");
    }
    if (m.has_reflectance) {
        if (!m.reflectance_is_rgb) {
            fail("only an `rgb reflectance` is supported, not a texture or a "
                 "named spectrum");
        }
        for (int i = 0; i < 3; i++) {
            rgb[i] = m.reflectance[i];
        }
    }
}

// PBRT's LinearBVHNode, which is declared in aggregates.h but defined inside
// aggregates.cpp, so it cannot be named from out here with a body. This is the
// same layout, and the only thing it is used for is reading the array PBRT
// built.
//
// A mirrored layout is a thing that can silently rot, so it is checked twice
// over: the size is asserted at compile time, and after extraction the root
// node's bounds are compared against `BVHAggregate::Bounds()`, which is public
// and does not go through this struct. If the layout ever stops matching,
// those bounds come out as nonsense and the run stops.
struct alignas(32) MirroredNode {
    pbrt::Bounds3f bounds;
    union {
        int primitives_offset; // Leaf.
        int second_child_offset; // Interior.
    };
    uint16_t n_primitives;
    uint8_t axis;
};
static_assert(sizeof(MirroredNode) == 32,
              "LinearBVHNode is 32 bytes in pbrt; this mirror has drifted");

// Reaching BVHAggregate's `nodes` and `primitives`, which are private.
//
// PBRT is not modified and not rebuilt: the tree below is the one PBRT's own
// BVHAggregate constructor produced, and this only reads it. Explicit template
// instantiation is not subject to access checking -- [temp.spec] is clear that
// it may name private members -- so this is a legal way to obtain a pointer to
// a member that the class does not expose, and it is confined to these few
// lines. The alternative would be a patched PBRT, which would make "the same
// tree PBRT uses" a claim about a fork rather than about PBRT.
template <typename Tag, typename Tag::type Member>
struct Rob {
    friend typename Tag::type get(Tag) { return Member; }
};

struct NodesTag {
    using type = pbrt::LinearBVHNode *pbrt::BVHAggregate::*;
    friend type get(NodesTag);
};
template struct Rob<NodesTag, &pbrt::BVHAggregate::nodes>;

struct PrimitivesTag {
    using type = std::vector<pbrt::Primitive> pbrt::BVHAggregate::*;
    friend type get(PrimitivesTag);
};
template struct Rob<PrimitivesTag, &pbrt::BVHAggregate::primitives>;

// PBRT: ProjectiveCamera's constructor and PerspectiveCamera::Create, in
// PBRT's own Transform arithmetic rather than an equivalent of it. The
// renderer applies these two matrices and nothing else, so they are the whole
// of what the camera means to it.
pbrt::Transform camera_from_raster(const pbrt::ParameterDictionary &params,
                                   int x_resolution, int y_resolution) {
    const pbrt::Float frame = params.GetOneFloat(
        "frameaspectratio", pbrt::Float(x_resolution) / pbrt::Float(y_resolution));
    pbrt::Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    const std::vector<pbrt::Float> sw = params.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() != 4) {
            fail("\"screenwindow\" should have four values");
        }
        screen.pMin.x = sw[0];
        screen.pMax.x = sw[1];
        screen.pMin.y = sw[2];
        screen.pMax.y = sw[3];
    }

    const pbrt::Float fov = params.GetOneFloat("fov", 90.);
    const pbrt::Transform screen_from_camera =
        pbrt::Perspective(fov, 1e-2f, 1000.f);

    const pbrt::Transform ndc_from_screen =
        pbrt::Scale(1 / (screen.pMax.x - screen.pMin.x),
                    1 / (screen.pMax.y - screen.pMin.y), 1) *
        pbrt::Translate(pbrt::Vector3f(-screen.pMin.x, -screen.pMax.y, 0));
    const pbrt::Transform raster_from_ndc =
        pbrt::Scale(x_resolution, -y_resolution, 1);
    const pbrt::Transform raster_from_screen = raster_from_ndc * ndc_from_screen;

    return pbrt::Inverse(screen_from_camera) * pbrt::Inverse(raster_from_screen);
}

void write_matrix(std::vector<float> &out, const pbrt::Transform &t) {
    const pbrt::SquareMatrix<4> &m = t.GetMatrix();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out.push_back(float(m[i][j]));
        }
    }
}

// A shape's transform is what places it, and the renderer holds geometry in
// render space with no transform of its own. A sphere survives that only if
// the transform is a translation: anything else makes it an ellipsoid, which
// is a different shape than the one the renderer knows how to intersect. Say
// so rather than render something subtly wrong.
bool translation_only(const pbrt::Transform &t, pbrt::Vector3f *offset) {
    const pbrt::SquareMatrix<4> &m = t.GetMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const pbrt::Float expected = (i == j) ? 1.f : 0.f;
            if (m[i][j] != expected) {
                return false;
            }
        }
    }
    if (m[3][0] != 0.f || m[3][1] != 0.f || m[3][2] != 0.f || m[3][3] != 1.f) {
        return false;
    }
    *offset = pbrt::Vector3f(m[0][3], m[1][3], m[2][3]);
    return true;
}

// Parse and convert. Everything PBRT owns is local to this function, so all of
// it is destroyed on the way out -- before CleanupPBRT takes the arenas it was
// allocated from out from under it. Doing this inline in main instead crashes
// on the way out, because the locals outlive the cleanup call.
void load(const char *filename, bonsai_scene::Scene &out) {
    std::vector<float> matrices;
    std::vector<bonsai_scene::Shape> &shapes = out.shapes;

    pbrt::BasicScene scene;
    CapturingBuilder builder(&scene);
    const std::vector<std::string> filenames = {filename};
    pbrt::ParseFiles(&builder, filenames);

    if (builder.camera_name != "perspective") {
        fail("only the perspective camera is supported, scene asks for \"" +
             builder.camera_name + "\"");
    }

    const int x_resolution = builder.film_params.GetOneInt("xresolution", 1280);
    const int y_resolution = builder.film_params.GetOneInt("yresolution", 720);

    out.width = uint32_t(x_resolution);
    out.height = uint32_t(y_resolution);

    write_matrix(matrices,
                 camera_from_raster(builder.camera_params, x_resolution,
                                    y_resolution));
    // PBRT renders in a space of its own choosing -- by default the world
    // translated to the camera's origin -- and the camera transform is what
    // knows which. Asking it, rather than assuming world space, is what keeps
    // this correct for any `Option "rendercoordsys"`.
    write_matrix(matrices,
                 pbrt::Inverse(
                     scene.GetCamera().GetCameraTransform().CameraFromRender(0.f)));

    for (const pbrt::ShapeSceneEntity &entity : scene.shapes) {
        const std::string name(entity.name);
        const pbrt::Transform &render_from_object = *entity.renderFromObject;
        float reflectance[3];
        material_reflectance(builder.materials, entity.materialIndex,
                             reflectance);

        if (name == "sphere") {
            pbrt::Vector3f centre;
            if (!translation_only(render_from_object, &centre)) {
                fail("a sphere is placed by something other than a "
                     "translation, which this renderer cannot represent");
            }
            const pbrt::Float radius =
                entity.parameters.GetOneFloat("radius", 1.f);
            // A partial sphere is a different shape; the renderer has no zmin,
            // zmax or phimax.
            if (entity.parameters.GetOneFloat("zmin", -radius) != -radius ||
                entity.parameters.GetOneFloat("zmax", radius) != radius ||
                entity.parameters.GetOneFloat("phimax", 360.f) != 360.f) {
                fail("partial spheres (zmin/zmax/phimax) are not supported");
            }
            bonsai_scene::Shape shape = {};
            shape.tag = bonsai_scene::ShapeTag::Sphere;
            shape.p0[0] = float(centre.x);
            shape.p0[1] = float(centre.y);
            shape.p0[2] = float(centre.z);
            shape.radius = float(radius);
            memcpy(shape.reflectance, reflectance, sizeof(reflectance));
            shapes.push_back(shape);

        } else if (name == "trianglemesh") {
            const std::vector<int> indices =
                entity.parameters.GetIntArray("indices");
            const std::vector<pbrt::Point3f> P =
                entity.parameters.GetPoint3fArray("P");
            if (indices.size() % 3 != 0) {
                fail("a trianglemesh has an index count that is not a multiple "
                     "of three");
            }
            for (size_t i = 0; i < indices.size(); i += 3) {
                // Object space to render space, which for a mesh is a matter of
                // moving the vertices; PBRT does the same in TriangleMesh's
                // constructor rather than keeping the transform.
                const pbrt::Point3f p0 = render_from_object(P[indices[i + 0]]);
                const pbrt::Point3f p1 = render_from_object(P[indices[i + 1]]);
                const pbrt::Point3f p2 = render_from_object(P[indices[i + 2]]);
                bonsai_scene::Shape shape = {};
                shape.tag = bonsai_scene::ShapeTag::Triangle;
                shape.p0[0] = float(p0.x);
                shape.p0[1] = float(p0.y);
                shape.p0[2] = float(p0.z);
                shape.p1[0] = float(p1.x);
                shape.p1[1] = float(p1.y);
                shape.p1[2] = float(p1.z);
                shape.p2[0] = float(p2.x);
                shape.p2[1] = float(p2.y);
                shape.p2[2] = float(p2.z);
                memcpy(shape.reflectance, reflectance, sizeof(reflectance));
                shapes.push_back(shape);
            }

        } else {
            fail("unsupported shape \"" + name + "\"");
        }
    }

    if (shapes.empty()) {
        fail("the scene has no shapes this renderer understands");
    }
    for (size_t i = 0; i < matrices.size(); i++) {
        out.matrices[i] = matrices[i];
    }
}

// Build the tree with PBRT's own BVHAggregate, over the shapes this scene
// produced, and read back what it built.
//
// The primitives go in in our order and each one's address is remembered, so
// the permutation the build settled on can be recovered by identity rather
// than by matching geometry. `shapes` comes back reordered to match, which is
// what lets a leaf name a contiguous run.
void build_pbrt_tree(std::vector<bonsai_scene::Shape> &shapes,
                     std::vector<bonsai_scene::Node> &nodes) {
    pbrt::Allocator alloc;

    // Every triangle in the scene as one mesh, because that is what PBRT's
    // Triangle refers into. The vertices are already in render space, so the
    // mesh's transform is the identity.
    std::vector<int> indices;
    std::vector<pbrt::Point3f> points;
    for (const bonsai_scene::Shape &s : shapes) {
        if (s.tag == bonsai_scene::ShapeTag::Triangle) {
            for (const float *p : {s.p0, s.p1, s.p2}) {
                indices.push_back(int(points.size()));
                points.push_back(pbrt::Point3f(p[0], p[1], p[2]));
            }
        }
    }
    pstd::vector<pbrt::Shape> triangles;
    if (!indices.empty()) {
        pbrt::TriangleMesh *mesh = alloc.new_object<pbrt::TriangleMesh>(
            pbrt::Transform(), /*reverseOrientation=*/false, indices, points,
            std::vector<pbrt::Vector3f>(), std::vector<pbrt::Normal3f>(),
            std::vector<pbrt::Point2f>(), std::vector<int>(), alloc);
        triangles = pbrt::Triangle::CreateTriangles(mesh, alloc);
    }

    std::vector<pbrt::Primitive> primitives;
    std::map<const void *, uint32_t> index_of;
    size_t next_triangle = 0;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        const bonsai_scene::Shape &s = shapes[i];
        pbrt::Shape shape;
        if (s.tag == bonsai_scene::ShapeTag::Sphere) {
            const pbrt::Transform *render_from_object =
                alloc.new_object<pbrt::Transform>(pbrt::Translate(
                    pbrt::Vector3f(s.p0[0], s.p0[1], s.p0[2])));
            const pbrt::Transform *object_from_render =
                alloc.new_object<pbrt::Transform>(
                    pbrt::Inverse(*render_from_object));
            shape = alloc.new_object<pbrt::Sphere>(
                render_from_object, object_from_render,
                /*reverseOrientation=*/false, s.radius, -s.radius, s.radius,
                360.f);
        } else {
            shape = triangles[next_triangle++];
        }
        pbrt::Primitive prim =
            alloc.new_object<pbrt::SimplePrimitive>(shape, pbrt::Material());
        index_of[prim.ptr()] = i;
        primitives.push_back(prim);
    }

    // PBRT's default maxnodeprims, and the split method its `bvh` accelerator
    // uses unless a scene says otherwise.
    pbrt::BVHAggregate aggregate(primitives, 4,
                                 pbrt::BVHAggregate::SplitMethod::SAH);

    pbrt::LinearBVHNode *raw = aggregate.*get(NodesTag());
    const std::vector<pbrt::Primitive> &ordered =
        aggregate.*get(PrimitivesTag());
    const MirroredNode *built = reinterpret_cast<const MirroredNode *>(raw);
    if (!built || ordered.size() != shapes.size()) {
        fail("pbrt's bvh came back with the wrong number of primitives");
    }

    // The check that the mirrored node layout above is still right. Bounds()
    // is public and reads the root through PBRT's own definition, so if this
    // agrees, the struct agrees.
    const pbrt::Bounds3f expected = aggregate.Bounds();
    if (built[0].bounds.pMin != expected.pMin ||
        built[0].bounds.pMax != expected.pMax) {
        fail("pbrt's bvh root does not match Bounds(), so the node layout "
             "mirrored in this file no longer matches pbrt's");
    }

    // The shapes, in the order the tree wants them.
    std::vector<bonsai_scene::Shape> reordered;
    reordered.reserve(ordered.size());
    for (const pbrt::Primitive &prim : ordered) {
        auto it = index_of.find(prim.ptr());
        if (it == index_of.end()) {
            fail("pbrt's bvh holds a primitive this scene did not put in it");
        }
        reordered.push_back(shapes[it->second]);
    }

    // Walk the flattened array to find its length, and convert as we go. The
    // node count is not something PBRT hands over, but the tree is laid out
    // depth first, so the last node reachable from the root is the end of it.
    size_t count = 0;
    uint32_t total_prims = 0;
    std::function<void(uint32_t)> walk = [&](uint32_t at) {
        count = std::max(count, size_t(at) + 1);
        const MirroredNode &n = built[at];
        if (n.n_primitives > 0) {
            total_prims += n.n_primitives;
            return;
        }
        walk(at + 1);
        walk(uint32_t(n.second_child_offset));
    };
    walk(0);
    if (total_prims != shapes.size()) {
        fail("pbrt's bvh does not reach every primitive");
    }

    nodes.clear();
    nodes.reserve(count);
    for (size_t i = 0; i < count; i++) {
        const MirroredNode &n = built[i];
        bonsai_scene::Node out;
        out.low[0] = float(n.bounds.pMin.x);
        out.low[1] = float(n.bounds.pMin.y);
        out.low[2] = float(n.bounds.pMin.z);
        out.high[0] = float(n.bounds.pMax.x);
        out.high[1] = float(n.bounds.pMax.y);
        out.high[2] = float(n.bounds.pMax.z);
        out.n_prims = n.n_primitives;
        out.axis = n.n_primitives > 0 ? 0 : n.axis;
        // PBRT stores the second child absolutely; the renderer's layout wants
        // it relative, because that is what its `right = index + offset` says.
        out.offset = n.n_primitives > 0
                         ? uint32_t(n.primitives_offset)
                         : uint32_t(n.second_child_offset) - uint32_t(i);
        nodes.push_back(out);
    }

    shapes = std::move(reordered);
}

} // namespace

int main(int argc, char **argv) {
    // --pbrt-tree dumps the BVH PBRT built alongside the geometry, and the
    // renderer then traverses that rather than building its own. It is how a
    // timing comparison is made to be about the traversal the schedule
    // produced rather than about whose builder found a better tree.
    bool pbrt_tree = false;
    std::vector<const char *> positional;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--pbrt-tree") {
            pbrt_tree = true;
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 2) {
        fail("usage: scene_dump [--pbrt-tree] <scene.pbrt> <out.bin>");
    }

    pbrt::PBRTOptions options;
    // The renderer takes one sample at the centre of each pixel, and the
    // reference is rendered the same way. This has to be set before parsing
    // because it is read while the scene is built.
    options.disablePixelJitter = true;
    pbrt::InitPBRT(options);

    bonsai_scene::Scene scene;
    load(positional[0], scene);
    if (pbrt_tree) {
        build_pbrt_tree(scene.shapes, scene.nodes);
    }

    pbrt::CleanupPBRT();

    if (!bonsai_scene::write(positional[1], scene)) {
        fail(std::string("cannot write ") + positional[1]);
    }

    printf("scene_dump: %s -> %s (%ux%u, %zu shapes", positional[0],
           positional[1], scene.width, scene.height, scene.shapes.size());
    if (pbrt_tree) {
        printf(", %zu nodes from pbrt's bvh", scene.nodes.size());
    }
    printf(")\n");
    return 0;
}
