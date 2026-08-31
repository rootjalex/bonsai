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

#include <pbrt/cameras.h>
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <cstdio>
#include <cstring>
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
        : pbrt::BasicSceneBuilder(scene) {}

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

    std::string camera_name;
    pbrt::ParameterDictionary camera_params;
    pbrt::ParameterDictionary film_params;
};

[[noreturn]] void fail(const std::string &message) {
    fprintf(stderr, "scene_dump: %s\n", message.c_str());
    exit(1);
}

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
void load(const char *filename, bonsai_scene::Header &header,
          std::vector<float> &matrices,
          std::vector<bonsai_scene::Shape> &shapes) {
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

    memcpy(header.magic, bonsai_scene::Magic, sizeof(header.magic));
    header.version = bonsai_scene::Version;
    header.width = uint32_t(x_resolution);
    header.height = uint32_t(y_resolution);

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
            bonsai_scene::Shape shape;
            shape.tag = bonsai_scene::ShapeTag::Sphere;
            shape.p0[0] = float(centre.x);
            shape.p0[1] = float(centre.y);
            shape.p0[2] = float(centre.z);
            shape.radius = float(radius);
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
                bonsai_scene::Shape shape;
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
                shapes.push_back(shape);
            }

        } else {
            fail("unsupported shape \"" + name + "\"");
        }
    }

    if (shapes.empty()) {
        fail("the scene has no shapes this renderer understands");
    }
    header.shape_count = uint32_t(shapes.size());
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        fail("usage: scene_dump <scene.pbrt> <out.bin>");
    }

    pbrt::PBRTOptions options;
    // The renderer takes one sample at the centre of each pixel, and the
    // reference is rendered the same way. This has to be set before parsing
    // because it is read while the scene is built.
    options.disablePixelJitter = true;
    pbrt::InitPBRT(options);

    bonsai_scene::Header header;
    std::vector<float> matrices;
    std::vector<bonsai_scene::Shape> shapes;
    load(argv[1], header, matrices, shapes);

    pbrt::CleanupPBRT();

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fail(std::string("cannot write ") + argv[2]);
    }
    fwrite(&header, sizeof(header), 1, out);
    fwrite(matrices.data(), sizeof(float), matrices.size(), out);
    fwrite(shapes.data(), sizeof(bonsai_scene::Shape), shapes.size(), out);
    fclose(out);

    printf("scene_dump: %s -> %s (%ux%u, %u shapes)\n", argv[1], argv[2],
           header.width, header.height, header.shape_count);
    return 0;
}
