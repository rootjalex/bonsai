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

// The light tree is dumped by building PBRT's own BVHLightSampler and reading
// the tree it produced -- not by re-deriving the build here, which would risk a
// transcription mistake in the surface-area heuristic that the whole point is
// to avoid. That sampler keeps its nodes, its bit trails, and the octahedral
// `w` of its compact bounds private with no accessor, so `lightsamplers.h`
// alone is included with member access opened up. Every header it depends on is
// included normally just above, so their guards are already set and only
// `lightsamplers.h`'s own definitions are seen under the macro -- opening a
// standard-library header this way would break it, which is why the two are
// kept apart. Making an access specifier public changes no layout and no
// behaviour; it only lets the dump read what it has to serialize. It must
// precede anything that could pull `lightsamplers.h` in normally (an integrator
// header does), or the guard would win and the access would stay closed.
#include <pbrt/pbrt.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>
#define private public
#include <pbrt/lightsamplers.h>
#undef private

#include <pbrt/base/bxdf.h>
#include <pbrt/base/material.h>
#include <pbrt/bssrdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/integrators.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/lights.h>
#include <pbrt/media.h>
// NanoVDB's reader, for the `nanovdb` medium's grids (pbrt/media.h brings
// the rest of NanoVDB), with the zip codec the scene set's grids are
// written in -- a define in the including file, which is how PBRT's own
// media.cpp turns it on; the zlib it needs is in PBRT's link line already.
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
// ScratchBuffer, for probing PBRT's own medium (BONSAI_CHECK_MAJORANT).
#include <pbrt/util/memory.h>
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/samplers.h>
#include <pbrt/scene.h>
#include <pbrt/util/hash.h>
#include <pbrt/shapes.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/file.h>
#include <pbrt/util/loopsubdiv.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/image.h>
#include <pbrt/util/math.h>
#include <pbrt/util/mipmap.h>
#include <pbrt/util/noise.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include "cie_tables.h"
#include "sobol_tables.h"
#include "srgb_table.h"
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/pmj02tables.h>
#include <pbrt/util/sobolmatrices.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "measured_io.h"
#include "scene_io.h"

// PBRT's sRGB RGB-to-spectrum table, which lives in a translation unit of its
// own and is reached through a `RGBToSpectrumTable` that keeps it private.
//
// Declared here because the renderer needs the *table* and not the lookup: it
// has to fit a filtered texture colour to a spectrum per lookup, which is what
// PBRT does with these numbers.
namespace pbrt {
extern const float sRGBToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray sRGBToSpectrumTable_Data;
// And the other colour spaces' tables PBRT generates and links, for an
// environment map in one of them (a sky in ACES2065-1, as the bistro,
// sportscar and villa scenes have). Rec.2020's is not in the library this
// builds against, so a Rec.2020 image is refused.
extern const float DCI_P3ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray DCI_P3ToSpectrumTable_Data;
extern const float ACES2065_1ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray ACES2065_1ToSpectrumTable_Data;
} // namespace pbrt

namespace {

[[noreturn]] void fail(const std::string &message);

// Set by `--print-differentials`. Read at the end of `load`, which is where the
// parsed scene and PBRT's own camera are both in scope; see the block there.
bool g_print_differentials = false;

// Set by `--print-hits`. Read at the end of `load` too: PBRT's own camera ray
// for a grid of pixel samples, and the distance its aggregate finds along it.
bool g_print_hits = false;

// Set by `--maxdepth`: the path depth the renderer is given in place of the
// scene's, for timing the primary hits alone or a depth of one's choosing.
// Zero for "the scene decides". Only the dump sees it -- PBRT reads its depth
// from the scene file and has no flag for it -- so a comparison at this depth
// has to give PBRT the same one by rewriting the scene's `Integrator`
// directive, which is what compare.sh's `--maxdepth` does.
int g_max_depth_override = 0;

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
        // PBRT: `namedCoordinateSystems["camera"] = Inverse(cameraFromWorld)`,
        // the CTM at this directive being cameraFromWorld.
        named_systems["camera"] = pbrt::Inverse(ctm);
        camera_name = name;
        // PBRT: `CameraSceneEntity(..., graphicsState.currentOutsideMedium)`
        // -- the camera sits in whatever medium is current when it is
        // declared, which is what `CameraMedium` sets.
        camera_medium_name = current_outside_medium;
        // The vector holds pointers, so a copy still refers to the parameters
        // the base is about to take; reading them here does not consume them.
        camera_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Camera(name, std::move(params), loc);
    }

    void Film(const std::string &type, pbrt::ParsedParameterVector params,
              pbrt::FileLoc loc) override {
        film_type = type;
        film_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Film(type, std::move(params), loc);
    }

    void Sampler(const std::string &name, pbrt::ParsedParameterVector params,
                 pbrt::FileLoc loc) override {
        sampler_name = name;
        sampler_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Sampler(name, std::move(params), loc);
    }

    // The reconstruction filter, which decides where in a pixel a sample lands.
    // PBRT's default is `gaussian` and the scene may name another; the renderer
    // has only the Gaussian, so `load` refuses the rest rather than substituting
    // -- a box filter and a Gaussian are different images and both look right.
    void PixelFilter(const std::string &name, pbrt::ParsedParameterVector params,
                     pbrt::FileLoc loc) override {
        filter_name = name;
        filter_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::PixelFilter(name, std::move(params), loc);
    }

    // What a material directive said, kept in the form the scene wrote it.
    //
    // Deliberately not a ParameterDictionary. An `rgb reflectance` reaches a
    // dictionary already turned into an RGBAlbedoSpectrum -- PBRT's own sigmoid
    // fit -- and that fit is the thing the renderer is supposed to be doing.
    // Taking the numbers as written leaves it that way, at the cost of doing
    // PBRT's defaulting by hand below.
    //
    // The values are copied rather than pointed at: the vector holds pointers
    // the base is about to take ownership of, and these outlive parsing.
    struct MaterialInfo {
        struct Value {
            std::string type;
            std::vector<float> floats;
            std::vector<int> ints;
            std::vector<uint8_t> bools;
            // Only a light reads these, and only to notice that it was given a
            // filename -- which is what makes an infinite light an image one
            // rather than a uniform one, and so refused rather than converted.
            std::vector<std::string> strings;
        };
        std::string name;
        std::map<std::string, Value> params;
        // Where the directive was, for the ones that are placed. Only a light
        // reads it, and only an environment map needs it: the map is a sphere
        // and the transform says which way round.
        pbrt::Transform ctm;
        bool ctm_is_tracked = true;

        const Value *find(const std::string &key) const {
            const auto it = params.find(key);
            return it == params.end() ? nullptr : &it->second;
        }
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
            MaterialInfo::Value v;
            v.type = p->type;
            v.floats.assign(p->floats.begin(), p->floats.end());
            v.ints.assign(p->ints.begin(), p->ints.end());
            v.bools.assign(p->bools.begin(), p->bools.end());
            // The strings too. Without them a `"texture reflectance"` arrived
            // with its type intact and the name it referred to gone, which read
            // as a texture parameter naming nothing.
            v.strings.assign(p->strings.begin(), p->strings.end());
            info.params.emplace(p->name, std::move(v));
        }
        materials.push_back(std::move(info));
        pbrt::BasicSceneBuilder::Material(name, std::move(params), loc);
    }

    // The integrator, and the three things about it both sides need to agree
    // on: which one, how deep it goes, and -- for `path` -- which light sampler
    // and whether it regularizes.
    //
    // PBRT's own defaults, taken from PathIntegrator::Create: `maxdepth` 5,
    // `lightsampler` "bvh", `regularize` false. They are read here rather than
    // assumed downstream because a scene that names none of them still means
    // all three, and the reference render has to be built with the same ones.
    void Integrator(const std::string &name, pbrt::ParsedParameterVector params,
                    pbrt::FileLoc loc) override {
        integrator_name = name;
        for (const pbrt::ParsedParameter *p : params) {
            if (p->name == "maxdepth" && !p->ints.empty()) {
                integrator_max_depth = int(p->ints[0]);
            } else if (p->name == "lightsampler" && !p->strings.empty()) {
                light_sampler_name = p->strings[0];
            } else if (p->name == "regularize" && !p->bools.empty()) {
                integrator_regularize = p->bools[0];
            }
        }
        pbrt::BasicSceneBuilder::Integrator(name, std::move(params), loc);
    }

    // Every light that is not attached to a shape: `infinite`, `distant`,
    // `point`, `spot`, `projection`, `goniometric`. None of them is
    // implemented, and until this was here none of them was refused either --
    // the directive was simply not overridden, so a scene lit by an
    // environment map converted without complaint and rendered black.
    //
    // That is the failure this whole app is built to refuse, and it was the
    // most common thing wrong with a real scene: of the fifteen scenes in
    // pbrt-v4-scenes that got as far as converting, thirteen were lit by one
    // of these. Recorded rather than refused on the spot so that the message
    // can name the kind, which is what decides which one to implement next.
    // The current transformation matrix, mirrored.
    //
    // PBRT keeps one and hands it to every directive that needs placing, but
    // `BasicSceneBuilder::RenderFromObject` is private and a light's
    // `renderFromLight` is protected on the light it ends up in -- so a tool
    // that wants to know where a light was pointed has to follow along. Every
    // directive below is a `ParserTarget` virtual, so this is the same sequence
    // of calls PBRT sees, in the same order, applied with PBRT's own Transform
    // arithmetic. What is reimplemented is the bookkeeping and not the maths.
    //
    // Only the world-space half is tracked. `RenderFromObject` is
    // `renderFromWorld * ctm`, WorldBegin resets the CTM to the identity, and
    // `renderFromWorld` is available afterwards from the camera -- so what has
    // to be followed is only what happens between `WorldBegin` and the light.
    //
    // It is checked rather than trusted: an environment map placed by the wrong
    // rotation is a sky in the wrong place, and the comparison against PBRT
    // sees that immediately. This is not a quantity that can be subtly wrong.
    void WorldBegin(pbrt::FileLoc loc) override {
        ctm = pbrt::Transform();
        ctm_stack.clear();
        named_systems["world"] = ctm;
        pbrt::BasicSceneBuilder::WorldBegin(loc);
    }
    void AttributeBegin(pbrt::FileLoc loc) override {
        ctm_stack.push_back(ctm);
        medium_stack.push_back(current_outside_medium);
        pbrt::BasicSceneBuilder::AttributeBegin(loc);
    }
    void AttributeEnd(pbrt::FileLoc loc) override {
        if (!ctm_stack.empty()) {
            ctm = ctm_stack.back();
            ctm_stack.pop_back();
        }
        if (!medium_stack.empty()) {
            current_outside_medium = medium_stack.back();
            medium_stack.pop_back();
        }
        pbrt::BasicSceneBuilder::AttributeEnd(loc);
    }
    void Identity(pbrt::FileLoc loc) override {
        ctm = pbrt::Transform();
        pbrt::BasicSceneBuilder::Identity(loc);
    }
    void Translate(pbrt::Float dx, pbrt::Float dy, pbrt::Float dz,
                   pbrt::FileLoc loc) override {
        ctm = ctm * pbrt::Translate(pbrt::Vector3f(dx, dy, dz));
        pbrt::BasicSceneBuilder::Translate(dx, dy, dz, loc);
    }
    void Scale(pbrt::Float sx, pbrt::Float sy, pbrt::Float sz,
               pbrt::FileLoc loc) override {
        ctm = ctm * pbrt::Scale(sx, sy, sz);
        pbrt::BasicSceneBuilder::Scale(sx, sy, sz, loc);
    }
    void Rotate(pbrt::Float angle, pbrt::Float ax, pbrt::Float ay,
                pbrt::Float az, pbrt::FileLoc loc) override {
        ctm = ctm * pbrt::Rotate(angle, pbrt::Vector3f(ax, ay, az));
        pbrt::BasicSceneBuilder::Rotate(angle, ax, ay, az, loc);
    }
    void LookAt(pbrt::Float ex, pbrt::Float ey, pbrt::Float ez, pbrt::Float lx,
                pbrt::Float ly, pbrt::Float lz, pbrt::Float ux, pbrt::Float uy,
                pbrt::Float uz, pbrt::FileLoc loc) override {
        // PBRT: `ctm * lookAt`, where pbrt::LookAt returns the transform whose
        // forward matrix is cameraFromWorld. Not its inverse, which this
        // applied until 2026-09-20 -- unnoticed, since nothing placed after
        // a LookAt inside the world block had been converted, and the camera's
        // own LookAt comes before WorldBegin, where only the named coordinate
        // system "camera" reads what it left behind.
        ctm = ctm * pbrt::LookAt(pbrt::Point3f(ex, ey, ez),
                                 pbrt::Point3f(lx, ly, lz),
                                 pbrt::Vector3f(ux, uy, uz));
        pbrt::BasicSceneBuilder::LookAt(ex, ey, ez, lx, ly, lz, ux, uy, uz,
                                        loc);
    }
    void Transform(pbrt::Float tr[16], pbrt::FileLoc loc) override {
        ctm = pbrt::Transpose(
            pbrt::Transform(pbrt::SquareMatrix<4>(pstd::MakeSpan(tr, 16))));
        pbrt::BasicSceneBuilder::Transform(tr, loc);
    }
    void ConcatTransform(pbrt::Float tr[16], pbrt::FileLoc loc) override {
        ctm = ctm * pbrt::Transpose(
            pbrt::Transform(pbrt::SquareMatrix<4>(pstd::MakeSpan(tr, 16))));
        pbrt::BasicSceneBuilder::ConcatTransform(tr, loc);
    }
    // The two that name a transform rather than compose one, mirroring PBRT's
    // named-coordinate-system table: `CoordinateSystem` records the CTM under
    // a name, `CoordSysTransform` makes a recorded one the CTM, and two names
    // PBRT records itself -- "camera" at the Camera directive, the inverse of
    // the CTM there (BasicSceneBuilder::Camera), and "world" at WorldBegin.
    // killeroo-gold places its distant light with `CoordSysTransform
    // "camera"`. A name nothing declared is refused: PBRT only warns and
    // leaves the CTM as it was, and a light placed by it would be somewhere
    // else.
    void CoordinateSystem(const std::string &n, pbrt::FileLoc loc) override {
        named_systems[n] = ctm;
        pbrt::BasicSceneBuilder::CoordinateSystem(n, loc);
    }
    void CoordSysTransform(const std::string &n, pbrt::FileLoc loc) override {
        const auto it = named_systems.find(n);
        if (it == named_systems.end()) {
            fail(where(loc) + "CoordSysTransform \"" + n +
                 "\": no coordinate system of that name has been declared");
        }
        ctm = it->second;
        pbrt::BasicSceneBuilder::CoordSysTransform(n, loc);
    }

    // Participating media. A `homogeneous` medium is recorded here -- its
    // parameters, for the conversion to read the way HomogeneousMedium::
    // Create does -- and the base is told too, so that PBRT's own checks run
    // (a name defined twice is PBRT's error). The other kinds are refused by
    // name, which is what decides which to implement next (PLAN.md,
    // "volpath"): a scene with fog in it must not convert with the fog
    // silently missing, since a render that quietly leaves out a medium
    // looks like a renderer that works.
    void MakeNamedMedium(const std::string &name,
                         pbrt::ParsedParameterVector params,
                         pbrt::FileLoc loc) override {
        std::string kind = "(no type)";
        for (const pbrt::ParsedParameter *p : params) {
            if (p->name == "type" && !p->strings.empty()) {
                kind = p->strings[0];
            }
        }
        if (kind != "homogeneous" && kind != "uniformgrid" && kind != "rgbgrid" &&
            kind != "cloud" && kind != "nanovdb") {
            fail(where(loc) + "MakeNamedMedium \"" + name + "\" of type \"" +
                 kind + "\": PBRT has no such medium");
        }
        MediumInfo info;
        info.name = name;
        info.ctm = ctm;
        info.ctm_is_tracked = ctm_is_tracked;
        // The vector holds pointers, so a copy still refers to the parameters
        // the base is about to take; reading them here does not consume them.
        info.params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        media.push_back(std::move(info));
        pbrt::BasicSceneBuilder::MakeNamedMedium(name, std::move(params), loc);
    }
    // PBRT: the two names become the graphics state's current media, which
    // every shape declared after takes (ShapeSceneEntity::insideMedium and
    // outsideMedium, which the base fills) and which the camera takes as its
    // own medium. The base holds that state privately, so the outside name is
    // mirrored here for the camera's sake, with the same pushing and popping
    // the CTM gets (AttributeBegin/End below).
    void MediumInterface(const std::string &inside, const std::string &outside,
                         pbrt::FileLoc loc) override {
        current_outside_medium = outside;
        pbrt::BasicSceneBuilder::MediumInterface(inside, outside, loc);
    }
    static std::string where(const pbrt::FileLoc &loc) {
        return std::string(loc.filename) + ":" + std::to_string(loc.line) +
               ": ";
    }

    void LightSource(const std::string &name, pbrt::ParsedParameterVector params,
                     pbrt::FileLoc loc) override {
        MaterialInfo info;
        info.name = name;
        info.ctm = ctm;
        info.ctm_is_tracked = ctm_is_tracked;
        for (const pbrt::ParsedParameter *p : params) {
            MaterialInfo::Value v;
            v.type = p->type;
            v.floats.assign(p->floats.begin(), p->floats.end());
            v.ints.assign(p->ints.begin(), p->ints.end());
            v.bools.assign(p->bools.begin(), p->bools.end());
            v.strings.assign(p->strings.begin(), p->strings.end());
            info.params.emplace(p->name, std::move(v));
        }
        lights.push_back(std::move(info));
        pbrt::BasicSceneBuilder::LightSource(name, std::move(params), loc);
    }

    // The same trick for area lights, and for the same reason: a shape names
    // one by index into BasicScene::areaLights, which is private. The base
    // appends one per directive, so recording them here in the same order
    // gives a list that index reaches.
    void AreaLightSource(const std::string &name,
                         pbrt::ParsedParameterVector params,
                         pbrt::FileLoc loc) override {
        MaterialInfo info;
        info.name = name;
        for (const pbrt::ParsedParameter *p : params) {
            MaterialInfo::Value v;
            v.type = p->type;
            v.floats.assign(p->floats.begin(), p->floats.end());
            v.ints.assign(p->ints.begin(), p->ints.end());
            v.bools.assign(p->bools.begin(), p->bools.end());
            info.params.emplace(p->name, std::move(v));
        }
        area_lights.push_back(std::move(info));
        pbrt::BasicSceneBuilder::AreaLightSource(name, std::move(params), loc);
    }

    std::string camera_name;
    pbrt::ParameterDictionary camera_params;
    // The medium the camera was declared in, by name; empty for none. See
    // MediumInterface above for how the name is followed.
    std::string camera_medium_name;
    std::string current_outside_medium;
    std::vector<std::string> medium_stack;
    // `MakeNamedMedium`, recorded for conversion on first use like a material:
    // only the media a shape's interface or the camera names are written.
    struct MediumInfo {
        std::string name;
        pbrt::ParameterDictionary params;
        // The transform at the directive: PBRT's `renderFromMedium` is
        // render-from-world times this, and a grid medium or a cloud is
        // placed by it (a homogeneous medium has no place).
        pbrt::Transform ctm;
        bool ctm_is_tracked = true;
    };
    std::vector<MediumInfo> media;
    // PBRT's default film, from BasicSceneBuilder's own initialization.
    std::string film_type = "rgb";
    pbrt::ParameterDictionary film_params;
    std::string sampler_name = "zsobol";
    pbrt::ParameterDictionary sampler_params;
    // PBRT's default filter, from BasicSceneBuilder's own initialization.
    std::string filter_name = "gaussian";
    pbrt::ParameterDictionary filter_params;
    std::vector<MaterialInfo> materials;
    std::vector<MaterialInfo> area_lights;
    // `MakeNamedMaterial`, which is the same thing as `Material` written down
    // for later: a shape reaches one through `NamedMaterial` and PBRT records
    // it on the shape by *name* rather than by index.
    //
    // The one structural difference is where the type lives. A `Material`
    // directive names its type as the directive's own argument -- `Material
    // "diffuse"` -- and this one carries it as a `"string type"` parameter, so
    // it is lifted out into the same field the other kind fills.
    void MakeNamedMaterial(const std::string &name,
                           pbrt::ParsedParameterVector params,
                           pbrt::FileLoc loc) override {
        MaterialInfo info;
        for (const pbrt::ParsedParameter *p : params) {
            if (p->name == "type" && !p->strings.empty()) {
                info.name = p->strings[0];
                continue;
            }
            MaterialInfo::Value v;
            v.type = p->type;
            v.floats.assign(p->floats.begin(), p->floats.end());
            v.ints.assign(p->ints.begin(), p->ints.end());
            v.bools.assign(p->bools.begin(), p->bools.end());
            v.strings.assign(p->strings.begin(), p->strings.end());
            info.params.emplace(p->name, std::move(v));
        }
        named_materials.emplace(name, std::move(info));
        pbrt::BasicSceneBuilder::MakeNamedMaterial(name, std::move(params), loc);
    }

    // Declared but not used is fine and common -- a scene's `materials.pbrt`
    // usually defines every material the model ever had. Nothing is converted
    // until a shape asks for it.
    std::map<std::string, MaterialInfo> named_materials;

    // `Texture "name" "float|spectrum" "class" ...`, recorded the same way and
    // for the same reason: a material names one, and which ones a scene
    // actually uses is not known until its materials are converted.
    //
    // `info.name` holds the texture's class -- `imagemap`, `scale`, ... -- so
    // that it reads the same way as a material's type. The declared type,
    // float or spectrum, is kept beside it because a `scale` texture over a
    // float image and one over a spectrum image are different lookups.
    struct TextureInfo {
        MaterialInfo params;
        std::string declared_type;
        // The transform in force at the declaration -- PBRT's
        // `renderFromTexture` less the camera's render-from-world, which is
        // composed in when the texture is converted. A geometric mapping and
        // a noise map the hit through its inverse.
        pbrt::Transform ctm;
    };
    std::map<std::string, TextureInfo> named_textures;
    // Declaration order, which is the order a `scale` texture's operand must
    // already have been declared in -- PBRT resolves texture names as it
    // parses, so a forward reference is not possible and this can be resolved
    // in one pass.
    std::vector<std::string> texture_order;

    void Texture(const std::string &name, const std::string &type,
                 const std::string &texname, pbrt::ParsedParameterVector params,
                 pbrt::FileLoc loc) override {
        TextureInfo info;
        info.declared_type = type;
        info.params.name = texname;
        info.ctm = ctm;
        for (const pbrt::ParsedParameter *p : params) {
            MaterialInfo::Value v;
            v.type = p->type;
            v.floats.assign(p->floats.begin(), p->floats.end());
            v.ints.assign(p->ints.begin(), p->ints.end());
            v.bools.assign(p->bools.begin(), p->bools.end());
            v.strings.assign(p->strings.begin(), p->strings.end());
            info.params.params.emplace(p->name, std::move(v));
        }
        if (named_textures.emplace(name, std::move(info)).second) {
            texture_order.push_back(name);
        }
        pbrt::BasicSceneBuilder::Texture(name, type, texname, std::move(params),
                                         loc);
    }

    // Every non-area LightSource the scene declared, in declaration order --
    // which is the order PBRT appends them to its own light list, after the
    // area lights. See above.
    std::vector<MaterialInfo> lights;
    // The mirrored CTM. See the transform overrides above.
    pbrt::Transform ctm;
    std::vector<pbrt::Transform> ctm_stack;
    // Always true now that the named coordinate systems are followed too;
    // kept as the record a light carries of how it was placed.
    bool ctm_is_tracked = true;
    std::map<std::string, pbrt::Transform> named_systems;
    // Empty when the scene named no integrator, which PBRT resolves to
    // `volpath`, and so does the conversion below.
    std::string integrator_name;
    int integrator_max_depth = 5;
    // PathIntegrator::Create's (and VolPathIntegrator::Create's) defaults for
    // the two parameters only they read.
    std::string light_sampler_name = "bvh";
    bool integrator_regularize = false;
};

[[noreturn]] void fail(const std::string &message) {
    fprintf(stderr, "scene_dump: %s\n", message.c_str());
    exit(1);
}

// A parameter the renderer can carry, or a refusal.
//
// Every one of these is a constant here where PBRT's is a texture. A texture
// evaluated at the wrong place is a picture that looks plausible and is not the
// scene's, so a material naming one is refused rather than flattened.
float material_float(const CapturingBuilder::MaterialInfo &m,
                     const std::string &key, float fallback) {
    const CapturingBuilder::MaterialInfo::Value *v = m.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (v->type != "float" || v->floats.size() != 1) {
        fail("the material parameter \"" + key +
             "\" has to be a single float here, not a texture or a curve");
    }
    return v->floats[0];
}

// An RGB spectrum parameter. Taken as the three numbers the scene wrote,
// because turning them into a spectrum is what the renderer does.
bool material_rgb(const CapturingBuilder::MaterialInfo &m,
                  const std::string &key, float *rgb) {
    const CapturingBuilder::MaterialInfo::Value *v = m.find(key);
    if (v == nullptr) {
        return false;
    }
    if (v->type != "rgb" || v->floats.size() != 3) {
        fail("the material parameter \"" + key +
             "\" has to be an `rgb` here, not a texture or a named spectrum");
    }
    for (int i = 0; i < 3; i++) {
        rgb[i] = v->floats[i];
    }
    return true;
}

// A `blackbody <key>` parameter -- one temperature in kelvin -- as PBRT's
// BlackbodySpectrum: the temperature, and the constructor's normalization,
// which makes the spectrum one at the wavelength Wien's law puts its peak at.
// The normalization is computed here with PBRT's own `Blackbody`, `FastExp`
// and all, and shipped, so the renderer's copy of the spectrum is PBRT's to
// the last bit rather than a recomputation of it. False when the parameter is
// absent or is not a blackbody, so the caller can read it as something else.
bool read_blackbody(const CapturingBuilder::MaterialInfo &m,
                    const std::string &key, uint32_t *blackbody,
                    float *temperature, float *normalization) {
    const CapturingBuilder::MaterialInfo::Value *v = m.find(key);
    if (v == nullptr || v->type != "blackbody") {
        return false;
    }
    if (v->floats.size() != 1) {
        fail("the parameter \"blackbody " + key +
             "\" takes one temperature, in kelvin");
    }
    const float t = v->floats[0];
    if (!(t > 0.f)) {
        fail("the parameter \"blackbody " + key +
             "\" needs a positive temperature, not " + std::to_string(t));
    }
    *blackbody = 1u;
    *temperature = t;
    // PBRT: BlackbodySpectrum::BlackbodySpectrum.
    const float lambda_max = 2.8977721e-3f / t;
    *normalization = 1.f / pbrt::Blackbody(lambda_max * 1e9f, t);
    return true;
}

// PBRT's RGBToSpectrumTable::operator(), over the table itself.
//
// PBRT returns an `RGBSigmoidPolynomial` and keeps its three coefficients
// private, so the answer cannot be read off the object -- and the coefficients
// are exactly what has to be shipped. So the lookup is repeated here, against
// PBRT's own table declared above, which makes it the same arithmetic on the
// same numbers rather than an equivalent of it.
struct Sigmoid {
    float c0, c1, c2;
};

// A colour space's RGB-to-spectrum table, by its two arrays.
struct SpectrumTable {
    const float *scale;
    const pbrt::RGBToSpectrumTable::CoefficientArray *data;
};

// The table of `space`, or a failure naming the space: sRGB, DCI-P3 and
// ACES2065-1 are linked; Rec.2020 is not (see the declarations above).
SpectrumTable table_of(const pbrt::RGBColorSpace *space, const std::string &what) {
    if (space == pbrt::RGBColorSpace::sRGB) {
        return {pbrt::sRGBToSpectrumTable_Scale, &pbrt::sRGBToSpectrumTable_Data};
    }
    if (space == pbrt::RGBColorSpace::DCI_P3) {
        return {pbrt::DCI_P3ToSpectrumTable_Scale, &pbrt::DCI_P3ToSpectrumTable_Data};
    }
    if (space == pbrt::RGBColorSpace::ACES2065_1) {
        return {pbrt::ACES2065_1ToSpectrumTable_Scale,
                &pbrt::ACES2065_1ToSpectrumTable_Data};
    }
    fail(what + ": its colour space is " +
         (space == pbrt::RGBColorSpace::Rec2020 ? std::string("Rec.2020")
                                                : std::string("not a named one")) +
         ", whose RGB-to-spectrum table this converter does not have; sRGB, "
         "DCI-P3 and ACES2065-1 are supported");
    return {nullptr, nullptr};
}

// Which illuminant an image light in `space` multiplies its fits by (PBRT's
// RGBIlluminantSpectrum against the image's colour space): zero for D65,
// which the renderer's driver has and which sRGB and DCI-P3 use; for another
// space, its illuminant densely sampled over 360 to 830 nm, appended to
// `out.env_illuminants` once per space and named by index plus one
// (bonsai_scene::InfiniteLight::illuminant).
uint32_t illuminant_index(const pbrt::RGBColorSpace *space, bonsai_scene::Scene &out) {
    if (space == pbrt::RGBColorSpace::sRGB || space == pbrt::RGBColorSpace::DCI_P3) {
        return 0;
    }
    static std::map<const pbrt::RGBColorSpace *, uint32_t> dumped;
    const auto found = dumped.find(space);
    if (found != dumped.end()) {
        return found->second;
    }
    constexpr int first = 360, last = 830;
    out.env_illuminants.reserve(out.env_illuminants.size() + (last - first + 1));
    for (int lambda = first; lambda <= last; lambda++) {
        out.env_illuminants.push_back(float(space->illuminant(float(lambda))));
    }
    const uint32_t index = uint32_t(out.env_illuminants.size() / (last - first + 1));
    dumped[space] = index;
    return index;
}

// PBRT: `RGBToSpectrumTable::operator()`, the trilinear lookup of the sigmoid
// that fits an RGB in the table's colour space.
Sigmoid fit_sigmoid(const SpectrumTable &table, float r, float g, float b) {
    constexpr int res = 64;
    r = std::min(std::max(r, 0.f), 1.f);
    g = std::min(std::max(g, 0.f), 1.f);
    b = std::min(std::max(b, 0.f), 1.f);

    // A grey is exact, and PBRT answers it without touching the table.
    if (r == g && g == b) {
        return Sigmoid{0.f, 0.f,
                       r * (1 - r) <= 0.f
                           ? (r < 0.5f ? -std::numeric_limits<float>::infinity()
                                       : std::numeric_limits<float>::infinity())
                           : (r - .5f) / std::sqrt(r * (1 - r))};
    }

    const float rgb[3] = {r, g, b};
    const int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
    const float z = rgb[maxc];
    const float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
    const float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

    const int xi = std::min(int(x), res - 2);
    const int yi = std::min(int(y), res - 2);
    int zi = 0;
    for (int i = 1; i < res - 1; i++) {
        if (table.scale[i] < z) {
            zi = i;
        }
    }
    const float dx = x - xi;
    const float dy = y - yi;
    const float z0 = table.scale[zi];
    const float z1 = table.scale[zi + 1];
    const float dz = (z - z0) / (z1 - z0);

    const auto co = [&](int ddx, int ddy, int ddz, int i) {
        return (*table.data)[maxc][zi + ddz][yi + ddy][xi + ddx][i];
    };
    const auto lerp = [](float t, float a, float bb) {
        return (1 - t) * a + t * bb;
    };

    Sigmoid out{};
    float *c[3] = {&out.c0, &out.c1, &out.c2};
    for (int i = 0; i < 3; i++) {
        *c[i] = lerp(dz,
                     lerp(dy, lerp(dx, co(0, 0, 0, i), co(1, 0, 0, i)),
                          lerp(dx, co(0, 1, 0, i), co(1, 1, 0, i))),
                     lerp(dy, lerp(dx, co(0, 0, 1, i), co(1, 0, 1, i)),
                          lerp(dx, co(0, 1, 1, i), co(1, 1, 1, i))));
    }
    return out;
}

// Where a texture's images are looked for, and what has already been
// converted.
//
// File-scope rather than arguments threaded through `convert_material` and its
// helpers: the conversion is a tree walk over parameters, and these tables are
// the only things it needs that are not in the parameter it is looking at.
const CapturingBuilder *g_builder = nullptr;
bonsai_scene::Scene *g_scene = nullptr;
std::string g_scene_dir;
std::map<std::string, int32_t> g_texture_index;
// The pyramid PBRT built for each converted texture, by the name the material
// asked for, kept for `--print-differentials` to filter through PBRT's own
// MIPMap beside this renderer's lookup.
std::map<std::string, pbrt::MIPMap *> g_texture_mip;
// The level each normal map image became, by filename: bistro's hundred and
// thirty materials name a hundred and twelve maps, several times each.
std::map<std::string, int32_t> g_normal_map_index;
// Keyed by the pair of spectrum names, since a scene usually names the same
// metal from several materials.
std::map<std::string, int32_t> g_conductor_index;
std::map<std::string, int32_t> g_measured_index;

// Defined below, once the texture table it fills is in scope.
int32_t convert_texture(const std::string &name);

// One PiecewiseLinear2D, appended to the pools.
uint32_t append_pl2d(const measured_io::PL2D &pl, bool has_cdf) {
    bonsai_scene::PL2DHeader h;
    h.size_x = uint32_t(pl.size_x);
    h.size_y = uint32_t(pl.size_y);
    h.dim = uint32_t(pl.param_size.size());
    for (size_t i = 0; i < pl.param_size.size() && i < 3; i++) {
        h.param_size[i] = uint32_t(pl.param_size[i]);
        h.param_stride[i] = pl.param_stride[i];
        h.first_param[i] = uint32_t(g_scene->pl_params.size());
        g_scene->pl_params.insert(g_scene->pl_params.end(),
                                  pl.param_values[i].begin(),
                                  pl.param_values[i].end());
    }
    h.first_data = uint32_t(g_scene->pl_data.size());
    g_scene->pl_data.insert(g_scene->pl_data.end(), pl.data.begin(),
                            pl.data.end());
    h.first_marginal = uint32_t(g_scene->pl_marginal.size());
    g_scene->pl_marginal.insert(g_scene->pl_marginal.end(),
                                pl.marginal_cdf.begin(), pl.marginal_cdf.end());
    h.first_conditional = uint32_t(g_scene->pl_conditional.size());
    g_scene->pl_conditional.insert(g_scene->pl_conditional.end(),
                                   pl.conditional_cdf.begin(),
                                   pl.conditional_cdf.end());
    h.has_cdf = has_cdf ? 1u : 0u;

    const uint32_t index = uint32_t(g_scene->pl2d.size());
    g_scene->pl2d.push_back(h);
    return index;
}

// One `.bsdf` file, read and precomputed.
//
// This is PBRT's MeasuredBxDFData::Create, transcribed rather than called: both
// `Tensor` and `MeasuredBxDFData` are file-local to PBRT's bxdfs.cpp, so
// linking PBRT gives access to neither. What the five interpolants are is
// PBRT's structure exactly -- the shapes are checked against each other the
// same way, because a file that disagrees with itself would otherwise be read
// as a valid one of different dimensions.
int32_t convert_measured(const std::string &filename) {
    const auto cached = g_measured_index.find(filename);
    if (cached != g_measured_index.end()) {
        return cached->second;
    }

    const measured_io::Tensor tf = measured_io::read_tensor(filename);
    if (!tf.ok()) {
        fail(tf.error);
    }
    const auto need = [&](const char *name) {
        const measured_io::Tensor::Field *f = tf.find(name);
        if (f == nullptr) {
            fail(filename + ": no `" + name + "` field");
        }
        return f;
    };
    const auto *theta_i = need("theta_i");
    const auto *phi_i = need("phi_i");
    const auto *ndf = need("ndf");
    const auto *sigma = need("sigma");
    const auto *vndf = need("vndf");
    const auto *spectra = need("spectra");
    const auto *luminance = need("luminance");
    const auto *wavelengths = need("wavelengths");

    const auto f32 = measured_io::Tensor::Float32;
    if (!(theta_i->shape.size() == 1 && theta_i->dtype == f32 &&
          phi_i->shape.size() == 1 && phi_i->dtype == f32 &&
          wavelengths->shape.size() == 1 && wavelengths->dtype == f32 &&
          ndf->shape.size() == 2 && ndf->dtype == f32 &&
          sigma->shape.size() == 2 && sigma->dtype == f32 &&
          vndf->shape.size() == 4 && vndf->dtype == f32 &&
          vndf->shape[0] == phi_i->shape[0] &&
          vndf->shape[1] == theta_i->shape[0] &&
          luminance->shape.size() == 4 && luminance->dtype == f32 &&
          luminance->shape[0] == phi_i->shape[0] &&
          luminance->shape[1] == theta_i->shape[0] &&
          luminance->shape[2] == luminance->shape[3] &&
          spectra->shape.size() == 5 && spectra->dtype == f32 &&
          spectra->shape[0] == phi_i->shape[0] &&
          spectra->shape[1] == theta_i->shape[0] &&
          spectra->shape[2] == wavelengths->shape[0] &&
          spectra->shape[3] == spectra->shape[4] &&
          luminance->shape[2] == spectra->shape[3] &&
          luminance->shape[3] == spectra->shape[4])) {
        fail(filename + ": the fields do not have the shapes a measured BRDF "
                        "has");
    }

    bonsai_scene::MeasuredBRDF brdf;
    brdf.isotropic = phi_i->shape[0] <= 2 ? 1u : 0u;
    if (!brdf.isotropic) {
        const float *p = phi_i->as_float();
        const int reduction = int(
            std::rint((2 * M_PI) / (p[phi_i->shape[0] - 1] - p[0])));
        if (reduction != 1) {
            fail(filename + ": a phi reduction other than 1 is not supported, "
                            "which is PBRT's own refusal");
        }
    }

    const std::vector<int> wo_res = {int(phi_i->shape[0]),
                                     int(theta_i->shape[0])};
    const std::vector<const float *> wo_vals = {phi_i->as_float(),
                                                theta_i->as_float()};

    // The NDF and the projected area are only ever evaluated, so PBRT builds
    // no CDF for them and does not normalize them either.
    brdf.ndf = append_pl2d(
        measured_io::build_pl2d(ndf->as_float(), int(ndf->shape[1]),
                                int(ndf->shape[0]), {}, {}, false, false),
        false);
    brdf.sigma = append_pl2d(
        measured_io::build_pl2d(sigma->as_float(), int(sigma->shape[1]),
                                int(sigma->shape[0]), {}, {}, false, false),
        false);
    brdf.vndf = append_pl2d(
        measured_io::build_pl2d(vndf->as_float(), int(vndf->shape[3]),
                                int(vndf->shape[2]), wo_res, wo_vals, true,
                                true),
        true);
    brdf.luminance = append_pl2d(
        measured_io::build_pl2d(luminance->as_float(),
                                int(luminance->shape[3]),
                                int(luminance->shape[2]), wo_res, wo_vals, true,
                                true),
        true);

    const std::vector<int> spec_res = {int(phi_i->shape[0]),
                                       int(theta_i->shape[0]),
                                       int(wavelengths->shape[0])};
    const std::vector<const float *> spec_vals = {phi_i->as_float(),
                                                  theta_i->as_float(),
                                                  wavelengths->as_float()};
    brdf.spectra = append_pl2d(
        measured_io::build_pl2d(spectra->as_float(), int(spectra->shape[4]),
                                int(spectra->shape[3]), spec_res, spec_vals,
                                false, false),
        false);

    const int32_t index = int32_t(g_scene->measured_brdfs.size());
    g_scene->measured_brdfs.push_back(brdf);
    g_measured_index.emplace(filename, index);
    fprintf(stderr, "scene_dump: read %s (%zu wavelengths, %s)\n",
            filename.c_str(), wavelengths->shape[0],
            brdf.isotropic ? "isotropic" : "anisotropic");
    return index;
}

// A spectrum parameter read as an unbounded spectrum, which is what a
// conductor's `eta` and `k` are -- PBRT: ParameterDictionary::
// extractSpectrumArray with SpectrumType::Unbounded, for one value. An RGB is
// an RGBUnboundedSpectrum, the sigmoid fit scaled by twice the largest
// component; a blackbody a BlackbodySpectrum; an inline list of increasing
// wavelength/value pairs a PiecewiseLinearSpectrum; and a string a named
// spectrum or, failing that, a file of such pairs found from the scene's
// directory (readSpectrumFromFile). The spectrum is PBRT's own object, built
// by PBRT's constructors, and is only ever *evaluated* here, into the tables
// the renderer interpolates; `key` describes the value for the cache of them.
pbrt::Spectrum unbounded_spectrum(const CapturingBuilder::MaterialInfo::Value &v,
                                  const std::string &param,
                                  pbrt::Allocator alloc, std::string *key) {
    const auto describe = [&](const std::string &kind) {
        std::string s = kind;
        for (const float f : v.floats) {
            s += " " + std::to_string(f);
        }
        for (const std::string &str : v.strings) {
            s += " " + str;
        }
        *key = s;
    };
    if (v.type == "rgb") {
        if (v.floats.size() != 3) {
            fail("the parameter \"rgb " + param + "\" takes three values");
        }
        const pbrt::RGB rgb(v.floats[0], v.floats[1], v.floats[2]);
        if (rgb.r < 0 || rgb.g < 0 || rgb.b < 0) {
            fail("the parameter \"rgb " + param +
                 "\" has a negative component, which PBRT refuses too");
        }
        describe("rgb");
        return alloc.new_object<pbrt::RGBUnboundedSpectrum>(
            *pbrt::RGBColorSpace::sRGB, rgb);
    }
    if (v.type == "blackbody") {
        if (v.floats.size() != 1) {
            fail("the parameter \"blackbody " + param +
                 "\" takes one temperature");
        }
        describe("blackbody");
        return alloc.new_object<pbrt::BlackbodySpectrum>(v.floats[0]);
    }
    if (v.type == "spectrum" && !v.floats.empty()) {
        if (v.floats.size() % 2 != 0) {
            fail("the parameter \"spectrum " + param +
                 "\" has an odd number of values; it takes wavelength/value "
                 "pairs");
        }
        const size_t n = v.floats.size() / 2;
        std::vector<pbrt::Float> lambda(n), value(n);
        for (size_t i = 0; i < n; i++) {
            if (i > 0 && v.floats[2 * i] <= lambda[i - 1]) {
                fail("the parameter \"spectrum " + param +
                     "\" has wavelengths that do not increase, at pair " +
                     std::to_string(i));
            }
            lambda[i] = v.floats[2 * i];
            value[i] = v.floats[2 * i + 1];
        }
        describe("spectrum");
        return alloc.new_object<pbrt::PiecewiseLinearSpectrum>(lambda, value,
                                                              alloc);
    }
    if (v.type == "spectrum" && v.strings.size() == 1) {
        describe("spectrum");
        if (const pbrt::Spectrum named = pbrt::GetNamedSpectrum(v.strings[0])) {
            return named;
        }
        const std::string resolved = pbrt::ResolveFilename(v.strings[0]);
        pstd::optional<pbrt::Spectrum> from_file =
            pbrt::PiecewiseLinearSpectrum::Read(resolved, alloc);
        if (!from_file) {
            fail(v.strings[0] + ": neither a spectrum PBRT knows by name nor "
                                "a file of wavelength/value pairs it can read");
        }
        return *from_file;
    }
    fail("the parameter \"" + param + "\" is a `" + v.type +
         "`, which is not a spectrum: an `rgb`, a `blackbody`, or a `spectrum` "
         "given as pairs, a name or a file");
}

// One pair of index-of-refraction curves, resampled at a tenth of a nanometre.
//
// PBRT keeps a named metal's as a PiecewiseLinearSpectrum over the published
// measurements and interpolates between them; this is the same function
// sampled on a grid, which the renderer interpolates the same way. That
// reproduces PBRT exactly wherever a grid step does not straddle one of the
// original knots -- they are four to six nanometres apart -- and the residual
// is measured below rather than assumed. A pair given some other way -- an
// RGB, a list of pairs, a file -- is PBRT's spectrum for it (see
// unbounded_spectrum), sampled onto the same grid. Neither given is PBRT's
// default, copper.
int32_t conductor_spectra(const CapturingBuilder::MaterialInfo::Value *eta_v,
                          const CapturingBuilder::MaterialInfo::Value *k_v) {
    pbrt::Allocator alloc;
    std::string eta_key = "metal-Cu-eta";
    std::string k_key = "metal-Cu-k";
    const pbrt::Spectrum eta =
        eta_v == nullptr ? pbrt::GetNamedSpectrum("metal-Cu-eta")
                         : unbounded_spectrum(*eta_v, "eta", alloc, &eta_key);
    const pbrt::Spectrum k =
        k_v == nullptr ? pbrt::GetNamedSpectrum("metal-Cu-k")
                       : unbounded_spectrum(*k_v, "k", alloc, &k_key);
    const std::string key = eta_key + "|" + k_key;
    const auto cached = g_conductor_index.find(key);
    if (cached != g_conductor_index.end()) {
        return cached->second;
    }

    const int32_t index =
        int32_t(g_scene->conductor_eta.size() / bonsai_scene::kConductorSamples);
    for (int i = 0; i < bonsai_scene::kConductorSamples; i++) {
        const pbrt::Float lambda =
            pbrt::Float(360.0 + double(i) / bonsai_scene::kConductorPerNm);
        g_scene->conductor_eta.push_back(float(eta(lambda)));
        g_scene->conductor_k.push_back(float(k(lambda)));
    }

    // What the resampling costs, measured against PBRT's own spectrum on a
    // grid ten times finer than the one shipped -- which is what lands inside
    // the intervals that straddle one of its knots, where the reconstruction is
    // a chord across a corner and everywhere else is exact.
    double worst = 0.0;
    double worst_at = 0.0;
    const int probes = (bonsai_scene::kConductorSamples - 1) * 10;
    for (int i = 0; i <= probes; i++) {
        const double lambda =
            360.0 + double(i) / (bonsai_scene::kConductorPerNm * 10.0);
        const double x = (lambda - 360.0) * bonsai_scene::kConductorPerNm;
        const int lo = int(x);
        if (lo < 0 || lo + 1 >= bonsai_scene::kConductorSamples) {
            continue;
        }
        const double t = x - double(lo);
        const size_t base =
            size_t(index) * bonsai_scene::kConductorSamples + size_t(lo);
        const double ours = (1 - t) * g_scene->conductor_eta[base] +
                            t * g_scene->conductor_eta[base + 1];
        const double theirs = double(eta(pbrt::Float(lambda)));
        const double scale = std::max(1e-6, std::abs(theirs));
        const double rel = std::abs(ours - theirs) / scale;
        if (rel > worst) {
            worst = rel;
            worst_at = lambda;
        }
    }
    fprintf(stderr,
            "scene_dump: %s resampled, worst relative error %.2e at %.2f nm\n",
            eta_key.c_str(), worst, worst_at);

    g_conductor_index.emplace(key, index);
    return index;
}

// A spectrum a material gave on its own, not as one of a conductor's pair -- a
// dielectric's index of refraction as a named glass (`glass-BK7`,
// `glass-BAF10`, `glass-F11`: PBRT's PiecewiseLinearSpectrum over the
// published dispersion) or written any other way, or a reflectance given as
// `"spectrum reflectance"` (disney-cloud's ground) -- sampled onto the same
// grid the conductors use, into the same tables, with a zero `k` beside it:
// the tables are the scene's sampled spectra whatever the material. The
// renderer reads a dielectric's at the path's first wavelength alone
// (bxdf.bonsai, eta_at) and a reflectance at all four (reflectance_at).
// Cached by what it was given, as a conductor's pair is.
int32_t sampled_spectrum_index(const pbrt::Spectrum spectrum,
                               const std::string &key) {
    const auto cached = g_conductor_index.find(key);
    if (cached != g_conductor_index.end()) {
        return cached->second;
    }
    const int32_t index =
        int32_t(g_scene->conductor_eta.size() / bonsai_scene::kConductorSamples);
    for (int i = 0; i < bonsai_scene::kConductorSamples; i++) {
        const pbrt::Float lambda =
            pbrt::Float(360.0 + double(i) / bonsai_scene::kConductorPerNm);
        g_scene->conductor_eta.push_back(float(spectrum(lambda)));
        g_scene->conductor_k.push_back(0.f);
    }
    g_conductor_index.emplace(key, index);
    return index;
}

int32_t spectrum_table_index(const CapturingBuilder::MaterialInfo::Value &v,
                             const std::string &param) {
    pbrt::Allocator alloc;
    std::string described = "?";
    const pbrt::Spectrum spectrum = unbounded_spectrum(v, param, alloc, &described);
    return sampled_spectrum_index(spectrum, described + "|alone");
}

// A subsurface material's BSSRDF table, PBRT's own: `BSSRDFTable(100, 64)`
// filled by ComputeBeamDiffusionBSSRDF for the material's `g` and `eta` --
// the photon beam diffusion profile, its effective albedos and its running
// integral, computed by PBRT's code and copied out as it is -- appended to
// the scene's tables as one run of kBSSRDFTableFloats, once per distinct
// (g, eta): the three sssdragon files name one material each, head one.
std::map<std::pair<float, float>, int32_t> g_bssrdf_index;

int32_t bssrdf_table_index(pbrt::Float g, pbrt::Float eta) {
    const std::pair<float, float> key{float(g), float(eta)};
    const auto cached = g_bssrdf_index.find(key);
    if (cached != g_bssrdf_index.end()) {
        return cached->second;
    }
    pbrt::Allocator alloc;
    pbrt::BSSRDFTable table(100, 64, alloc);
    pbrt::ComputeBeamDiffusionBSSRDF(g, eta, &table);
    const int32_t index =
        int32_t(g_scene->bssrdf_tables.size() / bonsai_scene::kBSSRDFTableFloats);
    const auto append = [&](const pstd::vector<pbrt::Float> &run) {
        for (const pbrt::Float f : run) {
            g_scene->bssrdf_tables.push_back(float(f));
        }
    };
    append(table.rhoSamples);
    append(table.radiusSamples);
    append(table.profile);
    append(table.rhoEff);
    append(table.profileCDF);
    g_bssrdf_index.emplace(key, index);
    return index;
}

// PBRT's `eta` on a material with a dielectric interface: a bare float is
// the constant, anything else a spectrum (above). `param` is the parameter's
// name, `eta` or `interface.eta`.
void dielectric_eta_of(const CapturingBuilder::MaterialInfo &m, const char *param,
                       bonsai_scene::Material &out) {
    const CapturingBuilder::MaterialInfo::Value *eta = m.find(param);
    if (eta == nullptr) {
        return;
    }
    if (eta->type == "float" && eta->floats.size() == 1) {
        out.eta = eta->floats[0];
        return;
    }
    out.eta_spectrum = spectrum_table_index(*eta, param);
}

// The same parameter, where it is allowed to be a texture.
//
// PBRT's material parameters are all textures and a constant is a
// `FloatConstantTexture`; here the constant stays a constant and the texture
// index is -1 unless there is one, because making every untextured surface do a
// lookup to find out it has none is a cost with nothing behind it.
//
// A `float` parameter used as a spectrum is PBRT's own widening -- a scalar
// reflectance is grey -- and is taken the same way.
//
// A `spectrum` (wavelength/value pairs, a named spectrum or a file) or a
// `blackbody` is PBRT's GetSpectrumTexture with SpectrumType::Albedo on one:
// GetOneSpectrum and a SpectrumConstantTexture over it, no check on its range
// -- the `> 1` error is the RGB's alone -- and the clamp to [0, 1] left to
// GetBxDF at the hit, which every material here does. It is sampled onto the
// index tables as a dielectric's eta is (spectrum_table_index) and its index
// written to `spectrum`, the RGB left as it was; -1 is returned, as for a
// constant.
int32_t material_rgb_or_texture(const CapturingBuilder::MaterialInfo &m,
                                const std::string &key, float *rgb,
                                int32_t *spectrum) {
    const CapturingBuilder::MaterialInfo::Value *v = m.find(key);
    if (v == nullptr) {
        return -1;
    }
    if (v->type == "texture") {
        if (v->strings.empty()) {
            fail("the material parameter \"" + key + "\" names no texture");
        }
        return convert_texture(v->strings[0]);
    }
    if (v->type == "float" && v->floats.size() == 1) {
        rgb[0] = rgb[1] = rgb[2] = v->floats[0];
        return -1;
    }
    if (v->type == "spectrum" || v->type == "blackbody") {
        *spectrum = spectrum_table_index(*v, key);
        return -1;
    }
    if (v->type != "rgb" || v->floats.size() != 3) {
        fail("the material parameter \"" + key +
             "\" has to be an `rgb`, a `float`, a `spectrum` or a texture "
             "here, and this one is a `" + v->type + "`");
    }
    for (int i = 0; i < 3; i++) {
        rgb[i] = v->floats[i];
    }
    return -1;
}

// A float parameter that may be a texture -- PBRT's FloatTexture, which every
// roughness is. The constant is written to `value` and -1 returned; a texture
// is converted and its index returned, `value` left alone. Absent, `fallback`
// is written and -1 returned.
int32_t material_float_or_texture(const CapturingBuilder::MaterialInfo &m,
                                  const std::string &key, float fallback,
                                  float *value) {
    const CapturingBuilder::MaterialInfo::Value *v = m.find(key);
    if (v == nullptr) {
        *value = fallback;
        return -1;
    }
    if (v->type == "texture") {
        if (v->strings.empty()) {
            fail("the material parameter \"" + key + "\" names no texture");
        }
        return convert_texture(v->strings[0]);
    }
    if (v->type != "float" || v->floats.size() != 1) {
        fail("the material parameter \"" + key +
             "\" has to be a `float` or a texture here, and this one is a `" +
             v->type + "`");
    }
    *value = v->floats[0];
    return -1;
}

// PBRT: the roughness of a coated diffuse, conductor or dielectric material,
// which every one of their `Create`s reads the same way --
//
//     FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness");
//     if (!uRoughness) uRoughness = parameters.GetFloatTexture("roughness", 0.f);
//
// and the same for v: an axis's own where given, `roughness` otherwise, each a
// float or a float texture, and zero -- a perfect mirror -- with none of them.
// A coated conductor reads two of these, under the prefixes `interface.` and
// `conductor.`.
void material_roughness_prefixed(const CapturingBuilder::MaterialInfo &m,
                                 const std::string &prefix, float *u, float *v,
                                 int32_t *u_texture, int32_t *v_texture) {
    const auto axis = [&](const std::string &own, float *value, int32_t *texture) {
        if (m.find(own) != nullptr) {
            *texture = material_float_or_texture(m, own, 0.f, value);
        } else {
            *texture = material_float_or_texture(m, prefix + "roughness", 0.f,
                                                 value);
        }
    };
    axis(prefix + "uroughness", u, u_texture);
    axis(prefix + "vroughness", v, v_texture);
}

void material_roughness(const CapturingBuilder::MaterialInfo &m,
                        bonsai_scene::Material &out) {
    material_roughness_prefixed(m, "", &out.u_roughness, &out.v_roughness,
                                &out.u_roughness_texture,
                                &out.v_roughness_texture);
}

// PBRT's WrapMode, by the names a scene writes.
uint32_t wrap_mode_from(const std::string &s) {
    if (s == "repeat") {
        return bonsai_scene::WrapMode::Repeat;
    }
    if (s == "clamp") {
        return bonsai_scene::WrapMode::Clamp;
    }
    if (s == "black") {
        return bonsai_scene::WrapMode::Black;
    }
    fail("unknown texture wrap mode \"" + s + "\"");
    return 0;
}

// One `Texture` declaration, converted -- the whole graph under it, since a
// `mix` or a `scale` names other textures. `convert_texture` is the entry,
// returning the index of the root; `convert_texture_node` is the recursion,
// which counts the graph's depth and refuses one deeper than the renderer
// reads (textures.bonsai evaluates three levels of composites); the kinds
// each have a converter below.
int32_t convert_texture(const std::string &name);

// The transform from world to render space, from the camera, kept here for
// the textures: a geometric mapping or a noise maps the hit through the
// inverse of PBRT's `renderFromTexture`, which is this times the transform
// in force at the texture's declaration (CapturingBuilder::TextureInfo::ctm).
pbrt::Transform g_render_from_world;

// The image at an Image record, and the constant scale folded into it, so a
// `scale` over a scaled image can fold again as PBRT's does (below).
struct ImageSource {
    std::string name;
    float folded_scale = 1.f;
};
std::map<int32_t, ImageSource> g_image_source;

void write_frame(float out[16], const pbrt::Transform &t) {
    const pbrt::SquareMatrix<4> &m = t.GetMatrix();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[4 * i + j] = float(m[i][j]);
        }
    }
}

// PBRT: `Inverse(renderFromTexture)`, which every mapping but uv takes.
pbrt::Transform texture_from_render_of(const pbrt::Transform &ctm) {
    return pbrt::Inverse(g_render_from_world * ctm);
}

// A constant texture made from an inline value -- PBRT's
// `TextureParameterDictionary::GetFloatTexture` wraps a `float tex1` in a
// FloatConstantTexture, and `GetSpectrumTexture` an `rgb tex1` in a
// SpectrumConstantTexture -- or from a `constant` declaration. One record per
// distinct value, by the value's text.
int32_t constant_float_texture(float value) {
    const std::string key = "float:" + std::to_string(value);
    const auto cached = g_texture_index.find(key);
    if (cached != g_texture_index.end()) {
        return cached->second;
    }
    bonsai_scene::Texture t;
    t.kind = bonsai_scene::TextureKind::Constant;
    t.value = value;
    t.rgb[0] = t.rgb[1] = t.rgb[2] = value;
    const int32_t index = int32_t(g_scene->textures.size());
    g_scene->textures.push_back(t);
    g_texture_index.emplace(key, index);
    return index;
}

int32_t constant_rgb_texture(const float rgb[3]) {
    const std::string key = "rgb:" + std::to_string(rgb[0]) + "," +
                            std::to_string(rgb[1]) + "," +
                            std::to_string(rgb[2]);
    const auto cached = g_texture_index.find(key);
    if (cached != g_texture_index.end()) {
        return cached->second;
    }
    bonsai_scene::Texture t;
    t.kind = bonsai_scene::TextureKind::Constant;
    t.value = rgb[0];
    t.rgb[0] = rgb[0];
    t.rgb[1] = rgb[1];
    t.rgb[2] = rgb[2];
    const int32_t index = int32_t(g_scene->textures.size());
    g_scene->textures.push_back(t);
    g_texture_index.emplace(key, index);
    return index;
}

int32_t convert_image_texture(const std::string &name, float folded_scale);

// The converted graph under a texture: the root's index and how many
// composites deep it is (a leaf is zero).
struct ConvertedTexture {
    int32_t index = -1;
    int depth = 0;
};

ConvertedTexture convert_texture_node(const std::string &name, int nesting);

// A texture-typed parameter of a composite -- `tex1`, `tex2`, `amount`,
// `tex`, `scale` -- which PBRT reads as a texture name, or as an inline
// value wrapped in a constant texture, or as its default when absent.
ConvertedTexture operand_texture(const CapturingBuilder::MaterialInfo &p,
                                 const char *key, bool spectrum,
                                 float default_value, int nesting,
                                 const std::string &owner) {
    const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
    if (v == nullptr) {
        const float grey[3] = {default_value, default_value, default_value};
        return ConvertedTexture{spectrum ? constant_rgb_texture(grey)
                                         : constant_float_texture(default_value),
                                0};
    }
    if (v->type == "texture") {
        if (v->strings.empty()) {
            fail("the texture \"" + owner + "\" names no texture for \"" +
                 key + "\"");
        }
        return convert_texture_node(v->strings[0], nesting);
    }
    if (v->type == "float") {
        if (v->floats.empty()) {
            fail("the texture \"" + owner + "\" has an empty \"" + key + "\"");
        }
        if (spectrum) {
            const float grey[3] = {v->floats[0], v->floats[0], v->floats[0]};
            return ConvertedTexture{constant_rgb_texture(grey), 0};
        }
        return ConvertedTexture{constant_float_texture(v->floats[0]), 0};
    }
    if (v->type == "rgb" && spectrum) {
        if (v->floats.size() != 3) {
            fail("the texture \"" + owner + "\": \"rgb " + key +
                 "\" takes three values");
        }
        const float rgb[3] = {v->floats[0], v->floats[1], v->floats[2]};
        return ConvertedTexture{constant_rgb_texture(rgb), 0};
    }
    fail("the texture \"" + owner + "\": \"" + key + "\" is a `" + v->type +
         "`, and this renderer reads a texture, a float" +
         (spectrum ? std::string(" or an rgb") : std::string("")) + " there");
    return ConvertedTexture{};
}

ConvertedTexture convert_texture_node(const std::string &name, int nesting) {
    if (nesting > 8) {
        fail("a texture graph more than eight deep under \"" + name +
             "\", which is a cycle");
    }
    const auto it = g_builder->named_textures.find(name);
    if (it == g_builder->named_textures.end()) {
        fail("a material names the texture \"" + name +
             "\", which the scene never declared");
    }
    const CapturingBuilder::TextureInfo &info = it->second;
    const CapturingBuilder::MaterialInfo &p = info.params;
    const bool spectrum = info.declared_type != "float";
    const auto float_of = [&](const char *key, float dflt) {
        const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
        return (v == nullptr || v->floats.empty()) ? dflt : v->floats[0];
    };
    const auto int_of = [&](const char *key, int dflt) {
        const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
        if (v == nullptr) {
            return dflt;
        }
        if (!v->ints.empty()) {
            return v->ints[0];
        }
        return v->floats.empty() ? dflt : int(v->floats[0]);
    };

    if (p.name == "imagemap") {
        return ConvertedTexture{convert_image_texture(name, 1.f), 0};
    }
    if (p.name == "constant") {
        // PBRT: FloatConstantTexture::Create, `value` default 1;
        // SpectrumConstantTexture::Create, `value` a spectrum default one,
        // read as an albedo here -- the type every spectrum texture a
        // material names is.
        const CapturingBuilder::MaterialInfo::Value *v = p.find("value");
        if (!spectrum) {
            return ConvertedTexture{constant_float_texture(float_of("value", 1.f)),
                                    0};
        }
        float rgb[3] = {1.f, 1.f, 1.f};
        if (v != nullptr) {
            if (v->type == "rgb" && v->floats.size() == 3) {
                rgb[0] = v->floats[0];
                rgb[1] = v->floats[1];
                rgb[2] = v->floats[2];
            } else if (v->type == "float" && !v->floats.empty()) {
                rgb[0] = rgb[1] = rgb[2] = v->floats[0];
            } else {
                fail("the `constant` texture \"" + name +
                     "\" has a `" + v->type +
                     "` value; this renderer reads an rgb or a float");
            }
        }
        return ConvertedTexture{constant_rgb_texture(rgb), 0};
    }
    if (p.name == "scale") {
        // PBRT: FloatScaledTexture::Create and its spectrum twin. The scale
        // is a float texture, default one; `tex` defaults to one as well. A
        // scale that is a constant folds: one leaves `tex` as it is, and over
        // an image it is multiplied into a copy of the image (`MultiplyScale`
        // on the copy), so the constant reaches the filtered colour before the
        // invert and the spectrum fit, which the fit being nonlinear makes a
        // different number from scaling after. Anything else is a Scale node,
        // the genuine product of two lookups.
        // A float operand that is a constant, and its value: given as a
        // number, or as the name of a `constant` texture; absent, the
        // default.
        const auto constant_operand = [&](const char *key, float dflt,
                                          float *value) {
            const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
            *value = dflt;
            if (v == nullptr) {
                return true;
            }
            if (v->type == "texture") {
                if (v->strings.empty()) {
                    fail("the `scale` texture \"" + name + "\" names no `" +
                         key + "` texture");
                }
                const auto it = g_builder->named_textures.find(v->strings[0]);
                if (it == g_builder->named_textures.end()) {
                    fail("the `scale` texture \"" + name + "\" names \"" +
                         v->strings[0] + "\" as its `" + key +
                         "`, which the scene never declared");
                }
                if (it->second.params.name != "constant") {
                    return false;
                }
                const CapturingBuilder::MaterialInfo::Value *val =
                    it->second.params.find("value");
                *value = (val != nullptr && !val->floats.empty())
                             ? val->floats[0]
                             : 1.f;
                return true;
            }
            if (v->floats.empty()) {
                fail("the `scale` texture \"" + name + "\" has a `" + key +
                     "` with no value");
            }
            *value = v->floats[0];
            return true;
        };
        const CapturingBuilder::MaterialInfo::Value *sc = p.find("scale");
        float cs = 1.f;
        const bool constant_scale = constant_operand("scale", 1.f, &cs);
        if (!constant_scale && !spectrum) {
            // PBRT's float version tries the other order as well -- the loop
            // over `std::swap(tex, scale)` in FloatScaledTexture::Create --
            // so a constant `tex` folds into an image `scale` the same way,
            // and a `tex` of one is the scale texture itself. The spectrum
            // version has no such loop: its `tex` is a spectrum and its
            // `scale` a float, and nothing swaps.
            float ct = 1.f;
            if (constant_operand("tex", 1.f, &ct)) {
                ConvertedTexture scale =
                    convert_texture_node(sc->strings[0], nesting + 1);
                if (ct == 1.f) {
                    return scale;
                }
                const auto source = g_image_source.find(scale.index);
                if (source != g_image_source.end()) {
                    return ConvertedTexture{
                        convert_image_texture(source->second.name,
                                              source->second.folded_scale * ct),
                        0};
                }
            }
        }
        ConvertedTexture tex = operand_texture(p, "tex", spectrum, 1.f, nesting + 1,
                                               name);
        if (constant_scale) {
            if (cs == 1.f) {
                return tex;
            }
            const auto source = g_image_source.find(tex.index);
            if (source != g_image_source.end()) {
                return ConvertedTexture{
                    convert_image_texture(source->second.name,
                                          source->second.folded_scale * cs),
                    0};
            }
            bonsai_scene::Texture t;
            t.kind = bonsai_scene::TextureKind::Scale;
            t.tex1 = tex.index;
            t.amount = constant_float_texture(cs);
            const int32_t index = int32_t(g_scene->textures.size());
            g_scene->textures.push_back(t);
            return ConvertedTexture{index, tex.depth + 1};
        }
        ConvertedTexture scale = convert_texture_node(sc->strings[0], nesting + 1);
        bonsai_scene::Texture t;
        t.kind = bonsai_scene::TextureKind::Scale;
        t.tex1 = tex.index;
        t.amount = scale.index;
        const int32_t index = int32_t(g_scene->textures.size());
        g_scene->textures.push_back(t);
        return ConvertedTexture{index, std::max(tex.depth, scale.depth) + 1};
    }
    if (p.name == "mix") {
        // PBRT: FloatMixTexture::Create -- tex1 default 0, tex2 default 1,
        // amount default 0.5 -- and SpectrumMixTexture::Create, the same with
        // the constant spectra zero and one.
        ConvertedTexture t1 = operand_texture(p, "tex1", spectrum, 0.f, nesting + 1, name);
        ConvertedTexture t2 = operand_texture(p, "tex2", spectrum, 1.f, nesting + 1, name);
        ConvertedTexture amount =
            operand_texture(p, "amount", false, 0.5f, nesting + 1, name);
        bonsai_scene::Texture t;
        t.kind = bonsai_scene::TextureKind::Mix;
        t.tex1 = t1.index;
        t.tex2 = t2.index;
        t.amount = amount.index;
        const int32_t index = int32_t(g_scene->textures.size());
        g_scene->textures.push_back(t);
        return ConvertedTexture{index,
                                std::max({t1.depth, t2.depth, amount.depth}) + 1};
    }
    if (p.name == "directionmix") {
        // PBRT: FloatDirectionMixTexture::Create -- `dir` default (0, 1, 0),
        // taken to render space and normalized.
        ConvertedTexture t1 = operand_texture(p, "tex1", spectrum, 0.f, nesting + 1, name);
        ConvertedTexture t2 = operand_texture(p, "tex2", spectrum, 1.f, nesting + 1, name);
        pbrt::Vector3f dir(0.f, 1.f, 0.f);
        const CapturingBuilder::MaterialInfo::Value *d = p.find("dir");
        if (d != nullptr && d->floats.size() == 3) {
            dir = pbrt::Vector3f(d->floats[0], d->floats[1], d->floats[2]);
        }
        dir = pbrt::Normalize((g_render_from_world * info.ctm)(dir));
        bonsai_scene::Texture t;
        t.kind = bonsai_scene::TextureKind::DirectionMix;
        t.tex1 = t1.index;
        t.tex2 = t2.index;
        t.dir[0] = float(dir.x);
        t.dir[1] = float(dir.y);
        t.dir[2] = float(dir.z);
        const int32_t index = int32_t(g_scene->textures.size());
        g_scene->textures.push_back(t);
        return ConvertedTexture{index, std::max(t1.depth, t2.depth) + 1};
    }
    if (p.name == "fbm" || p.name == "wrinkled" || p.name == "windy" ||
        p.name == "marble") {
        // PBRT: each Create takes `TextureMapping3D::Create`, the point
        // transform mapping by `Inverse(renderFromTexture)`; fbm and wrinkled
        // take `octaves` (8) and `roughness` (.5); marble those and `scale`
        // (1) and `variation` (.2). A noise is a number and marble a colour,
        // as PBRT's parser types them.
        if (p.name == "marble" && !spectrum) {
            fail("the `marble` texture \"" + name +
                 "\" is declared \"float\"; PBRT's marble is a spectrum texture");
        }
        if (p.name != "marble" && spectrum) {
            fail("the `" + p.name + "` texture \"" + name +
                 "\" is declared \"" + info.declared_type +
                 "\"; PBRT's is a float texture");
        }
        bonsai_scene::Texture t;
        t.kind = p.name == "fbm"        ? bonsai_scene::TextureKind::FBm
                 : p.name == "wrinkled" ? bonsai_scene::TextureKind::Wrinkled
                 : p.name == "windy"    ? bonsai_scene::TextureKind::Windy
                                        : bonsai_scene::TextureKind::Marble;
        write_frame(t.texture_from_render, texture_from_render_of(info.ctm));
        t.octaves = int_of("octaves", 8);
        t.omega = float_of("roughness", .5f);
        t.noise_scale = float_of("scale", 1.f);
        t.variation = float_of("variation", .2f);
        const int32_t index = int32_t(g_scene->textures.size());
        g_scene->textures.push_back(t);
        return ConvertedTexture{index, 0};
    }
    fail("this renderer implements the `imagemap`, `constant`, `scale`, `mix`, "
         "`directionmix`, `fbm`, `wrinkled`, `windy` and `marble` textures, "
         "and the scene asks for `" + p.name + "` in texture \"" + name + "\"");
    return ConvertedTexture{};
}

int32_t convert_texture(const std::string &name) {
    const ConvertedTexture root = convert_texture_node(name, 0);
    if (root.depth > 3) {
        fail("the texture \"" + name + "\" is a graph " +
             std::to_string(root.depth) +
             " composites deep; this renderer reads three (textures.bonsai, "
             "eval_float and eval_spectrum)");
    }
    return root.index;
}

// An `imagemap` declaration, converted -- its MIP pyramid built by PBRT and
// shipped level by level -- with `folded_scale`, the constant a `scale` over
// it multiplied in (PBRT copies the image texture and `MultiplyScale`s the
// copy; here a copy is a second record over the same levels). One record per
// image and scale.
int32_t convert_image_texture(const std::string &name, float folded_scale) {
    const std::string key = name + "@" + std::to_string(folded_scale);
    const auto cached = g_texture_index.find(key);
    if (cached != g_texture_index.end()) {
        return cached->second;
    }

    const auto it = g_builder->named_textures.find(name);
    const CapturingBuilder::MaterialInfo &p = it->second.params;

    const CapturingBuilder::MaterialInfo::Value *fn = p.find("filename");
    if (fn == nullptr || fn->strings.empty()) {
        fail("the `imagemap` texture \"" + name + "\" has no filename");
    }
    std::string filename = fn->strings[0];
    if (!filename.empty() && filename[0] != '/') {
        filename = g_scene_dir + "/" + filename;
    }

    // PBRT's ImageTexture defaults, from SpectrumImageTexture::Create.
    const auto float_of = [&](const char *key, float dflt) {
        const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
        if (v == nullptr) {
            return dflt;
        }
        if (v->type == "texture") {
            fail(std::string("the `imagemap` parameter \"") + key +
                 "\" is a texture, which PBRT does not allow either");
        }
        return v->floats.empty() ? dflt : v->floats[0];
    };
    const auto string_of = [&](const char *key, const char *dflt) {
        const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
        return (v == nullptr || v->strings.empty()) ? std::string(dflt)
                                                    : v->strings[0];
    };

    const std::string filter = string_of("filter", "bilinear");
    if (filter != "bilinear") {
        fail("the texture \"" + name + "\" asks for the `" + filter +
             "` filter; this renderer implements `bilinear`, which is PBRT's "
             "default");
    }
    // PBRT: TextureMapping2D::Create -- `uv` with its scales and offsets, or
    // one of the three geometric mappings through the inverse of the
    // texture's own transform; planar with its two axes and offsets.
    bonsai_scene::Texture t;
    t.kind = bonsai_scene::TextureKind::Image;
    const std::string mapping = string_of("mapping", "uv");
    const auto vector_of = [&](const char *key, float x, float y, float z,
                               float out[3]) {
        const CapturingBuilder::MaterialInfo::Value *v = p.find(key);
        out[0] = x;
        out[1] = y;
        out[2] = z;
        if (v != nullptr && v->floats.size() == 3) {
            out[0] = v->floats[0];
            out[1] = v->floats[1];
            out[2] = v->floats[2];
        }
    };
    if (mapping == "uv") {
        t.mapping = bonsai_scene::MappingKind::UV;
        t.su = float_of("uscale", 1.f);
        t.sv = float_of("vscale", 1.f);
        t.du = float_of("udelta", 0.f);
        t.dv = float_of("vdelta", 0.f);
    } else if (mapping == "spherical" || mapping == "cylindrical" ||
               mapping == "planar") {
        t.mapping = mapping == "spherical"     ? bonsai_scene::MappingKind::Spherical
                    : mapping == "cylindrical" ? bonsai_scene::MappingKind::Cylindrical
                                               : bonsai_scene::MappingKind::Planar;
        write_frame(t.texture_from_render, texture_from_render_of(it->second.ctm));
        vector_of("v1", 1.f, 0.f, 0.f, t.vs);
        vector_of("v2", 0.f, 1.f, 0.f, t.vt);
        t.ds = float_of("udelta", 0.f);
        t.dt = float_of("vdelta", 0.f);
    } else {
        fail("the texture \"" + name + "\" asks for the `" + mapping +
             "` mapping, which PBRT has not got either");
    }
    // The `scale` textures above the image, if any, folded into the image's
    // own scale -- which is what PBRT does: a constant scale over an image
    // texture copies the image texture and multiplies its scale, so the
    // constant reaches the filtered colour before the invert and the spectrum
    // fit. See ImageTexture::scale.
    t.scale = float_of("scale", 1.f) * folded_scale;
    const CapturingBuilder::MaterialInfo::Value *inv = p.find("invert");
    t.invert = (inv != nullptr && !inv->bools.empty() && inv->bools[0]) ? 1u
                                                                       : 0u;
    t.wrap = wrap_mode_from(string_of("wrap", "repeat"));

    // PBRT builds the pyramid, with PBRT's own resampling filter, and this
    // reads the levels off it. `CreateFromFile` is what
    // SpectrumImageTexture::Create calls, so the encoding -- sRGB for a PNG,
    // linear for an EXR -- is resolved the way PBRT resolves it.
    pbrt::MIPMapFilterOptions options;
    options.filter = pbrt::FilterFunction::Bilinear;
    pbrt::WrapMode wrap = pbrt::WrapMode::Repeat;
    if (t.wrap == bonsai_scene::WrapMode::Clamp) {
        wrap = pbrt::WrapMode::Clamp;
    } else if (t.wrap == bonsai_scene::WrapMode::Black) {
        wrap = pbrt::WrapMode::Black;
    }
    pbrt::MIPMap *mip = pbrt::MIPMap::CreateFromFile(
        filename, options, wrap, pbrt::ColorEncoding::sRGB, pbrt::Allocator());
    if (mip == nullptr) {
        fail("cannot read the texture image " + filename);
    }
    g_texture_mip[name] = mip;

    t.first_level = uint32_t(g_scene->texture_levels.size());
    t.n_levels = uint32_t(mip->Levels());
    for (int l = 0; l < mip->Levels(); l++) {
        const pbrt::Image &img = mip->GetLevel(l);
        const pbrt::Point2i res = img.Resolution();
        bonsai_scene::TextureLevel level;
        level.width = uint32_t(res.x);
        level.height = uint32_t(res.y);
        // The level is kept as PBRT keeps it: an 8-bit image stays bytes,
        // with its encoding, decoded when a texel is read (Image::GetChannel
        // on a U256 pixel); a float or half image, or a byte one with a
        // gamma encoding -- which no scene here has -- goes over decoded,
        // as floats. PBRT's pyramid has the image's format on every level
        // (Image::GeneratePyramid converts back to it), so one texture is
        // all bytes or all floats.
        level.format = bonsai_scene::TexelFormat::Float;
        if (img.Format() == pbrt::PixelFormat::U256) {
            if (img.Encoding().Is<pbrt::sRGBColorEncoding>()) {
                level.format = bonsai_scene::TexelFormat::SRGB8;
            } else if (img.Encoding().Is<pbrt::LinearColorEncoding>()) {
                level.format = bonsai_scene::TexelFormat::Linear8;
            }
        }
        const bool as_bytes = level.format != bonsai_scene::TexelFormat::Float;
        // Counted in texels, not in floats or bytes: the renderer reads each
        // pool three channels at a time.
        level.first_texel = uint32_t(
            (as_bytes ? g_scene->texture_bytes.size()
                      : g_scene->texture_texels.size()) /
            3);
        g_scene->texture_levels.push_back(level);
        // A U256 pixel's bytes, where PBRT's GetChannel would decode them.
        const auto raw = [&](const pbrt::Point2i &px, int c) -> uint8_t {
            return static_cast<const uint8_t *>(img.RawPointer(px))[c];
        };
        // Three channels, whatever the image brought, and whatever the
        // texture was declared as.
        //
        // A `spectrum` texture is PBRT's `Texel<RGB>`: the first three
        // channels, or a one-channel image widened to grey. Doing the widening
        // here means the renderer has one case and not two, and it is the same
        // number.
        //
        // A `float` texture is PBRT's `MIPMap::Bilerp<Float>`, which is not
        // the red channel: a one-channel image is that channel, a
        // three-channel image is the *average* of the three, and a
        // four-channel image is its *alpha* -- `CreateFromFile` keeps the
        // alpha channel only where it is not one everywhere, and this is what
        // an `alpha` cutout is: the same PNG named twice, once as the leaf's
        // colour and once, as a float, for where the leaf is. The one-channel
        // and alpha values are shipped in all three components, so that
        // `texture_float`'s `.x` is it after the same bilinear filter; the
        // three-channel case ships the three channels and is averaged *after*
        // the filter, as PBRT averages the three filtered channels, because
        // filtering the average is the same number only in exact arithmetic
        // and a bump map is a finite difference of two of these. Reading `.x`
        // of the colour instead tested every leaf texel against how red it
        // was, which stripped the trees.
        const int nc = img.NChannels();
        const bool as_float = it->second.declared_type == "float";
        if (as_float && nc != 1 && nc != 3 && nc != 4) {
            fail("the `imagemap` texture \"" + name + "\" has " +
                 std::to_string(nc) +
                 " channels, which PBRT does not read as a float texture");
        }
        t.average_channels = (as_float && nc == 3) ? 1u : 0u;
        for (int y = 0; y < res.y; y++) {
            for (int x = 0; x < res.x; x++) {
                const pbrt::Point2i px(x, y);
                if (as_bytes) {
                    // The same three rules, on the bytes as they are.
                    if (as_float && nc != 3) {
                        const uint8_t v = raw(px, nc == 1 ? 0 : 3);
                        g_scene->texture_bytes.push_back(v);
                        g_scene->texture_bytes.push_back(v);
                        g_scene->texture_bytes.push_back(v);
                    } else if (nc == 1) {
                        const uint8_t v = raw(px, 0);
                        g_scene->texture_bytes.push_back(v);
                        g_scene->texture_bytes.push_back(v);
                        g_scene->texture_bytes.push_back(v);
                    } else {
                        for (int c = 0; c < 3; c++) {
                            g_scene->texture_bytes.push_back(raw(px, c));
                        }
                    }
                } else if (as_float && nc != 3) {
                    const float v = float(img.GetChannel(px, nc == 1 ? 0 : 3));
                    g_scene->texture_texels.push_back(v);
                    g_scene->texture_texels.push_back(v);
                    g_scene->texture_texels.push_back(v);
                } else if (nc == 1) {
                    const float v = float(img.GetChannel(px, 0));
                    g_scene->texture_texels.push_back(v);
                    g_scene->texture_texels.push_back(v);
                    g_scene->texture_texels.push_back(v);
                } else {
                    for (int c = 0; c < 3; c++) {
                        g_scene->texture_texels.push_back(
                            float(img.GetChannel(px, c)));
                    }
                }
            }
        }
    }

    // The first texture pulls in the RGB-to-spectrum table, since it is what
    // makes a filtered RGB into a spectrum and nothing else needs it.
    if (g_scene->rgb_table.empty()) {
        g_scene->rgb_table.reserve(64 + 3 * 64 * 64 * 64 * 3);
        for (int i = 0; i < 64; i++) {
            g_scene->rgb_table.push_back(
                pbrt::sRGBToSpectrumTable_Scale[i]);
        }
        const float *data =
            reinterpret_cast<const float *>(pbrt::sRGBToSpectrumTable_Data);
        for (size_t i = 0; i < size_t(3) * 64 * 64 * 64 * 3; i++) {
            g_scene->rgb_table.push_back(data[i]);
        }
    }

    const int32_t index = int32_t(g_scene->textures.size());
    g_scene->textures.push_back(t);
    g_texture_index.emplace(key, index);
    g_image_source[index] = ImageSource{name, folded_scale};
    return index;
}

// A material's `normalmap`, read as PBRT reads it (scene.cpp): the image by
// filename, linear whatever it is -- a normal map's bytes are coordinates,
// not colours, so an 8-bit PNG is not sRGB-decoded -- its R, G and B
// channels, and no pyramid. It becomes one level of `texture_levels`, in the
// image's own format (bytes, for a PNG), that the renderer reads bilinearly
// with repeat wrap (differentials.bonsai, normal_map); the level's index is
// what the material carries. One image is read once however many materials
// name it.
int32_t convert_normal_map(const std::string &given) {
    std::string filename = given;
    if (!filename.empty() && filename[0] != '/') {
        filename = g_scene_dir + "/" + filename;
    }
    const auto cached = g_normal_map_index.find(filename);
    if (cached != g_normal_map_index.end()) {
        return cached->second;
    }
    pbrt::ImageAndMetadata im = pbrt::Image::Read(filename, pbrt::Allocator(),
                                                   pbrt::ColorEncoding::Linear);
    pbrt::ImageChannelDesc rgb = im.image.GetChannelDesc({"R", "G", "B"});
    if (!rgb) {
        fail("the normal map " + filename + " has no R, G and B channels");
    }
    const pbrt::Image img = im.image.SelectChannels(rgb);
    const pbrt::Point2i res = img.Resolution();

    bonsai_scene::TextureLevel level;
    level.width = uint32_t(res.x);
    level.height = uint32_t(res.y);
    level.format = bonsai_scene::TexelFormat::Float;
    if (img.Format() == pbrt::PixelFormat::U256) {
        // Read as linear, so this is the encoding it has; an sRGB one could
        // only come from a file that names its encoding, which a PNG does
        // not.
        level.format = img.Encoding().Is<pbrt::sRGBColorEncoding>()
                           ? bonsai_scene::TexelFormat::SRGB8
                           : bonsai_scene::TexelFormat::Linear8;
    }
    const bool as_bytes = level.format != bonsai_scene::TexelFormat::Float;
    level.first_texel = uint32_t(
        (as_bytes ? g_scene->texture_bytes.size() : g_scene->texture_texels.size()) /
        3);
    for (int y = 0; y < res.y; y++) {
        for (int x = 0; x < res.x; x++) {
            const pbrt::Point2i px(x, y);
            if (as_bytes) {
                const uint8_t *raw =
                    static_cast<const uint8_t *>(img.RawPointer(px));
                for (int c = 0; c < 3; c++) {
                    g_scene->texture_bytes.push_back(raw[c]);
                }
            } else {
                for (int c = 0; c < 3; c++) {
                    g_scene->texture_texels.push_back(
                        float(img.GetChannel(px, c)));
                }
            }
        }
    }
    const int32_t index = int32_t(g_scene->texture_levels.size());
    g_scene->texture_levels.push_back(level);
    g_normal_map_index.emplace(filename, index);
    return index;
}

// The scene's named materials, converted on first use and numbered in the
// file -- `material_for_named` in `load`, reached from here by a `mix`,
// whose operands are materials by name. Set for the duration of a load.
std::function<uint32_t(const std::string &)> g_material_for_named;

// The material a shape was declared under, in the form the renderer reads.
//
// The defaults are PBRT's own, from DiffuseMaterial::Create and
// CoatedDiffuseMaterial::Create, because a material that names nothing has to
// arrive as the one PBRT would have built rather than as a guess.
bonsai_scene::Material
convert_material(const CapturingBuilder::MaterialInfo &m) {
    bonsai_scene::Material out;
    // PBRT: Material::Create returns no material for `interface`, and for
    // `""` and `none` with a deprecation warning -- a surface with no BSDF,
    // which an integrator steps through (see MaterialTag::Interface).
    if (m.name.empty() || m.name == "none" || m.name == "interface") {
        out.tag = bonsai_scene::MaterialTag::Interface;
        return out;
    }

    // PBRT keeps `displacement` and `normalmap` on the base Material and
    // applies them in `GetBSDF` before the material is asked for anything, so
    // they belong to every kind and are read here rather than per material.
    //
    // A normal map is a different thing from a bump map -- it replaces the
    // shading normal outright rather than tilting it by a gradient -- and
    // PBRT reads it differently too: not through a Texture but as a plain
    // `Image`, linear, by filename (scene.cpp, the normal map futures).
    {
        const CapturingBuilder::MaterialInfo::Value *nm = m.find("normalmap");
        if (nm != nullptr) {
            if (nm->type != "string" || nm->strings.empty()) {
                fail("`normalmap` has to be a string, the image's filename");
            }
            out.normal_map = convert_normal_map(nm->strings[0]);
        }
    }
    {
        const CapturingBuilder::MaterialInfo::Value *d = m.find("displacement");
        if (d != nullptr) {
            if (d->type != "texture" || d->strings.empty()) {
                fail("`displacement` has to be a texture");
            }
            out.displacement_texture = convert_texture(d->strings[0]);
        }
    }

    if (m.name == "diffuse") {
        out.tag = bonsai_scene::MaterialTag::Diffuse;
        out.reflectance_texture = material_rgb_or_texture(
            m, "reflectance", out.reflectance, &out.reflectance_spectrum);
        return out;
    }

    if (m.name == "subsurface") {
        out.tag = bonsai_scene::MaterialTag::Subsurface;
        // PBRT: SubsurfaceMaterial::Create, whose "4, mutually-exclusive,
        // ways to specify the subsurface properties" are taken in its order:
        // a `name` (GetMediumScatteringProperties' table of measured media,
        // which gives reduced scattering coefficients, so `g` is forced to
        // zero), `sigma_a` and `sigma_s` (both or neither -- PBRT's error),
        // a `reflectance` with an `mfp` defaulting to a ConstantSpectrum of
        // one, or nothing: whole milk's RGBs as RGBUnboundedSpectra. Each
        // spectrum is PBRT's own object evaluated onto the sampled tables
        // (sampled_spectrum_index), and the renderer reads it at the hit's
        // four wavelengths. A texture for sigma_a, sigma_s or mfp is
        // refused: PBRT reads one as an unbounded spectrum texture, which
        // this renderer's textures do not evaluate as, and no scene gives
        // one.
        pbrt::Allocator alloc;
        pbrt::Float g = material_float(m, "g", 0.f);
        const CapturingBuilder::MaterialInfo::Value *preset = m.find("name");
        const CapturingBuilder::MaterialInfo::Value *sa = m.find("sigma_a");
        const CapturingBuilder::MaterialInfo::Value *ss = m.find("sigma_s");
        const CapturingBuilder::MaterialInfo::Value *refl = m.find("reflectance");
        const auto no_texture = [&](const CapturingBuilder::MaterialInfo::Value *v,
                                    const char *what) {
            if (v != nullptr && v->type == "texture") {
                fail(std::string("a subsurface material's `") + what +
                     "` given as a texture is not supported yet: PBRT reads it "
                     "as an unbounded spectrum texture, which this renderer's "
                     "textures do not evaluate as; give it as an rgb or a "
                     "spectrum");
            }
        };
        if (preset != nullptr) {
            if (preset->type != "string" || preset->strings.size() != 1) {
                fail("a subsurface material's `name` has to be one string");
            }
            pbrt::Spectrum sig_a, sig_s;
            if (!pbrt::GetMediumScatteringProperties(preset->strings[0], &sig_a,
                                                      &sig_s, alloc)) {
                fail(preset->strings[0] +
                     ": named medium not found -- which is PBRT's own error too");
            }
            g = 0;
            out.sigma_a_spectrum = sampled_spectrum_index(
                sig_a, "preset " + preset->strings[0] + " sigma_a");
            out.sigma_s_spectrum = sampled_spectrum_index(
                sig_s, "preset " + preset->strings[0] + " sigma_s");
        } else if (sa != nullptr || ss != nullptr) {
            if (sa == nullptr || ss == nullptr) {
                fail("a subsurface material given `sigma_a` or `sigma_s` needs "
                     "both -- which is PBRT's own error too");
            }
            no_texture(sa, "sigma_a");
            no_texture(ss, "sigma_s");
            out.sigma_a_spectrum = spectrum_table_index(*sa, "sigma_a");
            out.sigma_s_spectrum = spectrum_table_index(*ss, "sigma_s");
        } else if (refl != nullptr) {
            out.subsurface_from_reflectance = 1u;
            out.reflectance_texture = material_rgb_or_texture(
                m, "reflectance", out.reflectance, &out.reflectance_spectrum);
            const CapturingBuilder::MaterialInfo::Value *mfp = m.find("mfp");
            if (mfp == nullptr) {
                out.mfp_spectrum = sampled_spectrum_index(
                    alloc.new_object<pbrt::ConstantSpectrum>(1.f),
                    "constant 1|alone");
            } else {
                no_texture(mfp, "mfp");
                out.mfp_spectrum = spectrum_table_index(*mfp, "mfp");
            }
        } else {
            out.sigma_a_spectrum = sampled_spectrum_index(
                alloc.new_object<pbrt::RGBUnboundedSpectrum>(
                    *pbrt::RGBColorSpace::sRGB, pbrt::RGB(.0011f, .0024f, .014f)),
                "default subsurface sigma_a|alone");
            out.sigma_s_spectrum = sampled_spectrum_index(
                alloc.new_object<pbrt::RGBUnboundedSpectrum>(
                    *pbrt::RGBColorSpace::sRGB, pbrt::RGB(2.55f, 3.21f, 3.77f)),
                "default subsurface sigma_s|alone");
        }
        out.scale = material_float(m, "scale", 1.f);
        out.eta = material_float(m, "eta", 1.33f);
        // The roughness as a Dielectric's: `uroughness`/`vroughness` each
        // falling back to `roughness`, zero by default, and `remaproughness`
        // true by default.
        material_roughness(m, out);
        const CapturingBuilder::MaterialInfo::Value *sremap =
            m.find("remaproughness");
        if (sremap != nullptr) {
            if (sremap->type != "bool" || sremap->bools.size() != 1) {
                fail("`remaproughness` has to be a single bool");
            }
            out.remap = sremap->bools[0] ? 1u : 0u;
        }
        out.bssrdf_table = bssrdf_table_index(g, out.eta);
        return out;
    }

    if (m.name == "measured") {
        out.tag = bonsai_scene::MaterialTag::Measured;
        const CapturingBuilder::MaterialInfo::Value *fn = m.find("filename");
        if (fn == nullptr || fn->strings.empty()) {
            fail("a `measured` material needs a `filename`");
        }
        std::string path = fn->strings[0];
        if (!path.empty() && path[0] != '/') {
            path = g_scene_dir + "/" + path;
        }
        out.measured = convert_measured(path);
        return out;
    }

    if (m.name == "conductor") {
        out.tag = bonsai_scene::MaterialTag::Conductor;

        // PBRT: ConductorMaterial::Create. `reflectance` and `eta`/`k` are
        // alternatives -- PBRT errors if both are given -- and with neither it
        // falls back to copper.
        const CapturingBuilder::MaterialInfo::Value *refl =
            m.find("reflectance");
        const CapturingBuilder::MaterialInfo::Value *eta = m.find("eta");
        const CapturingBuilder::MaterialInfo::Value *k = m.find("k");
        if (refl != nullptr && (eta != nullptr || k != nullptr)) {
            fail("a conductor may name `reflectance` or `eta`/`k`, not both -- "
                 "which is PBRT's own error too");
        }
        if (refl != nullptr) {
            // The other branch of ConductorMaterial::GetBxDF: no index at
            // all, but a reflectance the renderer inverts at the hit -- eta
            // one, k the value that reflects it at normal incidence. An RGB, a
            // grey float or a texture, as a diffuse material's is.
            out.conductor_from_reflectance = 1u;
            out.reflectance_texture = material_rgb_or_texture(
                m, "reflectance", out.reflectance, &out.reflectance_spectrum);
        } else {
            out.conductor_spectra = conductor_spectra(eta, k);
        }

        // The roughness falls back exactly as CoatedDiffuse's does, and
        // defaults to zero -- which makes a perfect mirror.
        material_roughness(m, out);
        const CapturingBuilder::MaterialInfo::Value *cremap =
            m.find("remaproughness");
        if (cremap != nullptr) {
            if (cremap->type != "bool" || cremap->bools.size() != 1) {
                fail("`remaproughness` has to be a single bool");
            }
            out.remap = cremap->bools[0] ? 1u : 0u;
        }
        return out;
    }

    if (m.name == "dielectric") {
        out.tag = bonsai_scene::MaterialTag::Dielectric;
        // PBRT: DielectricMaterial::Create. The roughness falls back the same
        // way CoatedDiffuse's does, and defaults to zero -- which makes the
        // boundary a perfect one and its BSDF a pair of deltas.
        material_roughness(m, out);
        const CapturingBuilder::MaterialInfo::Value *dremap =
            m.find("remaproughness");
        if (dremap != nullptr) {
            if (dremap->type != "bool" || dremap->bools.size() != 1) {
                fail("`remaproughness` has to be a single bool");
            }
            out.remap = dremap->bools[0] ? 1u : 0u;
        }
        // As on coateddiffuse: a spectral index of refraction terminates the
        // secondary wavelengths, which nothing here does, so it is refused
        // rather than read as its value at the first wavelength.
        // PBRT: `eta` a spectrum, 1.5 by default; a named glass is sampled
        // onto the index tables and terminates the secondary wavelengths at
        // the hit, as PBRT's does (dielectric_eta_of).
        dielectric_eta_of(m, "eta", out);
        return out;
    }

    if (m.name == "coateddiffuse") {
        out.tag = bonsai_scene::MaterialTag::CoatedDiffuse;
        out.reflectance_texture = material_rgb_or_texture(
            m, "reflectance", out.reflectance, &out.reflectance_spectrum);
        // PBRT takes `uroughness` and `vroughness` where they are given and
        // falls back to `roughness` for each independently, which is not the
        // same as falling back to `roughness` only when neither is given.
        material_roughness(m, out);
        // PBRT: `thickness` and `g` are FloatTextures, 0.01 and 0 by default
        // (watercolor gives a thickness as an image).
        out.thickness_texture =
            material_float_or_texture(m, "thickness", 0.01f, &out.thickness);
        out.g_texture = material_float_or_texture(m, "g", 0.f, &out.g);
        // `eta` is a spectrum in PBRT unless the scene writes it as a bare
        // float, and a spectral one terminates the secondary wavelengths --
        // which changes what every later stage of the render integrates over.
        dielectric_eta_of(m, "eta", out);
        out.has_medium = material_rgb(m, "albedo", out.medium_albedo) ? 1u : 0u;

        const CapturingBuilder::MaterialInfo::Value *remap =
            m.find("remaproughness");
        if (remap != nullptr) {
            if (remap->type != "bool" || remap->bools.size() != 1) {
                fail("`remaproughness` has to be a single bool");
            }
            out.remap = remap->bools[0] ? 1u : 0u;
        }
        const CapturingBuilder::MaterialInfo::Value *depth = m.find("maxdepth");
        if (depth != nullptr) {
            if (depth->type != "integer" || depth->ints.size() != 1) {
                fail("`maxdepth` has to be a single integer");
            }
            out.max_depth = depth->ints[0];
        }
        const CapturingBuilder::MaterialInfo::Value *n = m.find("nsamples");
        if (n != nullptr) {
            if (n->type != "integer" || n->ints.size() != 1) {
                fail("`nsamples` has to be a single integer");
            }
            out.n_samples = n->ints[0];
        }
        return out;
    }

    if (m.name == "coatedconductor") {
        // PBRT: CoatedConductorMaterial::Create. The interface is the coated
        // diffuse's coating under the prefix `interface.` -- its roughness, a
        // scalar `interface.eta` defaulting to 1.5 -- and the metal is the
        // conductor material under `conductor.`: its roughness, and its index
        // as `conductor.eta`/`conductor.k` or a `reflectance`, copper with
        // none; `thickness`, `g`, `albedo`, `maxdepth`, `nsamples` and
        // `remaproughness` as the coated diffuse reads them.
        out.tag = bonsai_scene::MaterialTag::CoatedConductor;
        material_roughness_prefixed(m, "interface.", &out.u_roughness,
                                    &out.v_roughness, &out.u_roughness_texture,
                                    &out.v_roughness_texture);
        out.thickness_texture =
            material_float_or_texture(m, "thickness", 0.01f, &out.thickness);
        out.g_texture = material_float_or_texture(m, "g", 0.f, &out.g);
        const CapturingBuilder::MaterialInfo::Value *ieta =
            m.find("interface.eta");
        if (ieta != nullptr) {
            dielectric_eta_of(m, "interface.eta", out);
        }
        out.has_medium = material_rgb(m, "albedo", out.medium_albedo) ? 1u : 0u;
        material_roughness_prefixed(m, "conductor.", &out.conductor_u_roughness,
                                    &out.conductor_v_roughness,
                                    &out.conductor_u_roughness_texture,
                                    &out.conductor_v_roughness_texture);
        const CapturingBuilder::MaterialInfo::Value *refl =
            m.find("reflectance");
        const CapturingBuilder::MaterialInfo::Value *ceta =
            m.find("conductor.eta");
        const CapturingBuilder::MaterialInfo::Value *ck = m.find("conductor.k");
        if (refl != nullptr && (ceta != nullptr || ck != nullptr)) {
            fail("a coated conductor may name `reflectance` or "
                 "`conductor.eta`/`conductor.k`, not both -- which is PBRT's "
                 "own error too");
        }
        if (refl != nullptr) {
            out.conductor_from_reflectance = 1u;
            out.reflectance_texture = material_rgb_or_texture(
                m, "reflectance", out.reflectance, &out.reflectance_spectrum);
        } else {
            out.conductor_spectra = conductor_spectra(ceta, ck);
        }
        const CapturingBuilder::MaterialInfo::Value *remap =
            m.find("remaproughness");
        if (remap != nullptr) {
            if (remap->type != "bool" || remap->bools.size() != 1) {
                fail("`remaproughness` has to be a single bool");
            }
            out.remap = remap->bools[0] ? 1u : 0u;
        }
        const CapturingBuilder::MaterialInfo::Value *depth = m.find("maxdepth");
        if (depth != nullptr) {
            if (depth->type != "integer" || depth->ints.size() != 1) {
                fail("`maxdepth` has to be a single integer");
            }
            out.max_depth = depth->ints[0];
        }
        const CapturingBuilder::MaterialInfo::Value *n = m.find("nsamples");
        if (n != nullptr) {
            if (n->type != "integer" || n->ints.size() != 1) {
                fail("`nsamples` has to be a single integer");
            }
            out.n_samples = n->ints[0];
        }
        return out;
    }

    if (m.name == "diffusetransmission") {
        out.tag = bonsai_scene::MaterialTag::DiffuseTransmission;
        // PBRT: DiffuseTransmissionMaterial::Create. Both spectra default to
        // a constant 0.25 -- not the 0.5 a `diffuse` gets -- and `scale`
        // multiplies both before GetBxDF clamps them.
        for (int i = 0; i < 3; i++) {
            out.reflectance[i] = 0.25f;
            out.transmittance[i] = 0.25f;
        }
        out.reflectance_texture = material_rgb_or_texture(
            m, "reflectance", out.reflectance, &out.reflectance_spectrum);
        out.transmittance_texture = material_rgb_or_texture(
            m, "transmittance", out.transmittance, &out.transmittance_spectrum);
        out.scale = material_float(m, "scale", 1.f);
        return out;
    }

    if (m.name == "mix") {
        // PBRT: MixMaterial::Create -- `materials`, the names of two
        // materials, and `amount`, a float texture defaulting to 0.5 (a
        // number given for it is a constant texture, as GetFloatTexture
        // wraps one). The two are converted first, through the same path a
        // shape's named material takes, so a mix's operands sit below it in
        // the file and a mix of mixes resolves downward; an operand that is
        // an interface is kept as one, which the renderer resolves to -1.
        out.tag = bonsai_scene::MaterialTag::Mix;
        const CapturingBuilder::MaterialInfo::Value *names = m.find("materials");
        if (names == nullptr || names->type != "string" ||
            names->strings.size() != 2) {
            fail("a `mix` material needs `string materials` naming exactly two "
                 "materials, as PBRT's does");
        }
        if (!g_material_for_named) {
            fail("a `mix` material was converted outside a scene");
        }
        out.mix_first = int32_t(g_material_for_named(names->strings[0]));
        out.mix_second = int32_t(g_material_for_named(names->strings[1]));
        out.mix_amount_texture =
            material_float_or_texture(m, "amount", 0.5f, &out.mix_amount);
        return out;
    }

    if (m.name == "thindielectric") {
        // PBRT: ThinDielectricMaterial::Create -- the one parameter is `eta`,
        // 1.5 by default; a spectral one is refused as on dielectric.
        out.tag = bonsai_scene::MaterialTag::ThinDielectric;
        dielectric_eta_of(m, "eta", out);
        return out;
    }

    fail("only the diffuse, coateddiffuse, coatedconductor, dielectric, "
         "thindielectric, conductor, measured, diffusetransmission and mix "
         "materials are supported, scene asks for \"" +
         m.name + "\"");
}

// Check the generated spectral tables against the ones a running PBRT holds.
//
// make_spectrum_tables.py transcribes the CIE curves and reproduces the film's
// D65 by following what PBRT does to build it. Transcription can go stale and a
// reproduction can be subtly wrong -- picking the other D65 in the source, say,
// which is a table of the same name at a different resolution and would give
// colours that look plausible and are not. This asks PBRT instead of assuming.
//
// The tolerance is one ulp rather than zero: PBRT is built with contraction on,
// so some of its multiply-adds are fused and the last bit of a few samples
// cannot be reproduced from Python. Anything larger than that is a real
// difference and worth stopping for.
bool check_tables() {
    const pbrt::DenselySampledSpectrum &illuminant =
        pbrt::RGBColorSpace::sRGB->illuminant;
    struct Curve {
        const char *name;
        const float *ours;
        const pbrt::DenselySampledSpectrum &theirs;
    };
    const Curve curves[] = {
        {"CIE_X", CIE_X, pbrt::Spectra::X()},
        {"CIE_Y", CIE_Y, pbrt::Spectra::Y()},
        {"CIE_Z", CIE_Z, pbrt::Spectra::Z()},
    };

    bool ok = true;
    for (const Curve &c : curves) {
        for (int i = 0; i < CIE_SAMPLES; i++) {
            const float lambda = CIE_LAMBDA_MIN + float(i);
            const float theirs = c.theirs(lambda);
            if (theirs != c.ours[i]) {
                printf("scene_dump: %s differs at %g nm: pbrt %.9g, ours %.9g\n",
                       c.name, double(lambda), double(theirs), double(c.ours[i]));
                ok = false;
                break;
            }
        }
    }

    int exact = 0;
    double worst = 0;
    float worst_lambda = 0;
    for (int i = 0; i < CIE_SAMPLES; i++) {
        const float lambda = CIE_LAMBDA_MIN + float(i);
        const float theirs = illuminant(lambda);
        const float ours = CIE_D65_FILM[i];
        if (theirs == ours) {
            exact++;
        }
        // One ulp of a float, relative, with a floor for values near zero.
        const double tolerance =
            std::max(1e-30, double(std::fabs(theirs)) * 1.2e-7);
        const double difference = std::fabs(double(theirs) - double(ours));
        if (difference > worst) {
            worst = difference;
            worst_lambda = lambda;
        }
        if (difference > tolerance) {
            printf("scene_dump: the film illuminant differs at %g nm: "
                   "pbrt %.9g, ours %.9g\n",
                   double(lambda), double(theirs), double(ours));
            ok = false;
            break;
        }
    }
    printf("scene_dump: spectral tables match pbrt (film illuminant %d/%d exact, "
           "worst %.3g at %g nm)\n",
           exact, CIE_SAMPLES, worst, double(worst_lambda));

    // The Sobol' matrices the ZSobol sampler draws from, word for word
    // against pbrt's table -- a transcription, and the check is what makes a
    // transcription of a hundred and four hex words trustworthy.
    static_assert(SOBOL_MATRIX_SIZE == pbrt::SobolMatrixSize,
                  "sobol_tables.h disagrees with pbrt on a matrix's columns");
    for (int i = 0; i < SOBOL_DIMENSIONS_SHIPPED * SOBOL_MATRIX_SIZE; i++) {
        if (pbrt::SobolMatrices32[i] != SOBOL_MATRICES[i]) {
            printf("scene_dump: SOBOL_MATRICES differs from pbrt at dimension "
                   "%d column %d: pbrt 0x%08x, ours 0x%08x\n",
                   i / SOBOL_MATRIX_SIZE, i % SOBOL_MATRIX_SIZE,
                   pbrt::SobolMatrices32[i], SOBOL_MATRICES[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("scene_dump: Sobol' matrices match pbrt (%d dimensions)\n",
               SOBOL_DIMENSIONS_SHIPPED);
    }

    // The sRGB decode of a byte, which the software texture filter runs an
    // 8-bit texel through -- pbrt's SRGBToLinearLUT, entry for entry.
    bool srgb_ok = true;
    for (int i = 0; i < 256; i++) {
        const float theirs = float(pbrt::SRGB8ToLinear(uint8_t(i)));
        if (theirs != SRGB8_TO_LINEAR[i]) {
            printf("scene_dump: SRGB8_TO_LINEAR differs from pbrt at %d: pbrt "
                   "%.9g, ours %.9g\n",
                   i, double(theirs), double(SRGB8_TO_LINEAR[i]));
            srgb_ok = false;
            break;
        }
    }
    if (srgb_ok) {
        printf("scene_dump: the sRGB decode table matches pbrt (256 entries)\n");
    }
    return ok && srgb_ok;
}

// Print what PBRT's own samplers produce, so that the ones written in bonsai
// can be checked against them rather than against a reading of the source.
//
// The sampler is where matching PBRT stops being about arithmetic and starts
// being about reproducing a stream exactly: the same RNG, seeded by the same
// hash of the same pixel, advanced by the same amount, drawn from in the same
// order. Any of those wrong gives noise that looks perfectly good and is not
// PBRT's, and no image comparison at low sample counts would say which of the
// two was right. These numbers are what tests/bonsai/correctness/llvm's
// sampler golden holds.
void print_sampler() {
    // Pixels chosen to be unalike: the origin, a small one, and one far out,
    // so a hash that ignored part of its input would still differ here.
    const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {0, 1}, {37, 11},
                                    {1279, 719}};
    for (const pbrt::Point2i &p : pixels) {
        for (int sample = 0; sample < 2; sample++) {
            pbrt::IndependentSampler sampler(16, /*seed=*/0);
            sampler.StartPixelSample(p, sample, 0);
            printf("pixel %d %d sample %d:", p.x, p.y, sample);
            for (int i = 0; i < 4; i++) {
                printf(" %.9g", double(sampler.Get1D()));
            }
            printf("\n");
        }
    }
    // The hash on its own, which is the part most likely to be transcribed
    // wrongly and the hardest to see through the RNG.
    for (const pbrt::Point2i &p : pixels) {
        const uint64_t h = pbrt::Hash(p, 0);
        printf("hash %d %d: %llu\n", p.x, p.y,
               static_cast<unsigned long long>(h));
    }

    // The stratified sampler, which is a different construction rather than a
    // differently-seeded one: a draw is the sample's own cell of a grid, and
    // which cell that is comes from a permutation of the sample indices chosen
    // by the pixel, the dimension and the seed. Two more things to reproduce,
    // and both are printed on their own below for the same reason the hash is.
    for (const pbrt::Point2i &p : pixels) {
        for (int sample = 0; sample < 2; sample++) {
            // Not a square grid, so that x and y cannot be swapped unnoticed.
            pbrt::StratifiedSampler sampler(4, 2, /*jitter=*/true, /*seed=*/0);
            sampler.StartPixelSample(p, sample, 0);
            printf("strat %d %d sample %d:", p.x, p.y, sample);
            for (int i = 0; i < 2; i++) {
                printf(" %.9g", double(sampler.Get1D()));
            }
            const pbrt::Point2f uv = sampler.Get2D();
            printf(" | %.9g %.9g\n", double(uv.x), double(uv.y));
        }
    }
    // And without the jitter, where the sampler draws nothing at all from its
    // generator -- which is the part easiest to get wrong by evaluating both
    // sides of what PBRT writes as a conditional.
    for (const pbrt::Point2i &p : pixels) {
        pbrt::StratifiedSampler sampler(4, 2, /*jitter=*/false, /*seed=*/0);
        sampler.StartPixelSample(p, 1, 0);
        printf("strat-nojitter %d %d:", p.x, p.y);
        for (int i = 0; i < 2; i++) {
            printf(" %.9g", double(sampler.Get1D()));
        }
        const pbrt::Point2f uv = sampler.Get2D();
        printf(" | %.9g %.9g\n", double(uv.x), double(uv.y));
    }

    // The sixteen-byte hash the stratified sampler asks for, whose block loop
    // runs twice and has no tail where the twelve-byte one runs once and does.
    for (const pbrt::Point2i &p : pixels) {
        for (int dim = 0; dim < 2; dim++) {
            printf("hash3 %d %d %d: %llu\n", p.x, p.y, dim,
                   static_cast<unsigned long long>(pbrt::Hash(p, dim, 0)));
        }
    }
    // And the permutation on its own, over a range that is not a power of two
    // so that the rejection loop runs more than once for some of its inputs.
    for (uint32_t l : {8u, 13u}) {
        for (uint32_t seed : {0u, 0x9e3779b9u}) {
            printf("perm %u %u:", l, seed);
            for (uint32_t i = 0; i < l; i++) {
                printf(" %d", pbrt::PermutationElement(i, l, seed));
            }
            printf("\n");
        }
    }

    // The Halton sampler, which is what scenes in the wild actually ask for.
    // Not a stream of uniforms at all: a deterministic sequence shared by the
    // whole image, so a pixel's samples are the entries of it that land inside
    // that pixel. Three separate things have to be reproduced -- the radical
    // inverses, the digit permutations that randomize them, and the index
    // arithmetic that decides where a pixel starts -- and each is printed on
    // its own below before the sampler that combines them.
    for (int base_index : {0, 1, 4, 25}) {
        printf("radinv %d:", base_index);
        for (uint64_t a : {0ull, 1ull, 2ull, 7ull, 1000ull, 123456789ull}) {
            printf(" %.9g", double(pbrt::RadicalInverse(base_index, a)));
        }
        printf("\n");
    }
    for (int base : {2, 3, 7}) {
        for (int n : {1, 4}) {
            printf("invradinv %d %d:", base, n);
            for (uint64_t v : {0ull, 1ull, 5ull, 40ull}) {
                printf(" %llu", static_cast<unsigned long long>(
                                    pbrt::InverseRadicalInverse(v, base, n)));
            }
            printf("\n");
        }
    }
    // The permutations are built once for every dimension; asking for a few
    // checks both the table and the hash that seeds each digit of it.
    {
        pstd::vector<pbrt::DigitPermutation> *perms =
            pbrt::ComputeRadicalInversePermutations(0, pbrt::Allocator());
        for (int base_index : {0, 1, 4, 25}) {
            printf("scramrad %d:", base_index);
            for (uint64_t a : {0ull, 1ull, 2ull, 7ull, 1000ull, 123456789ull}) {
                printf(" %.9g", double(pbrt::ScrambledRadicalInverse(
                                    base_index, a, (*perms)[base_index])));
            }
            printf("\n");
        }
    }
    for (int base_index : {0, 1, 4}) {
        printf("owenrad %d:", base_index);
        for (uint64_t a : {0ull, 1ull, 7ull, 1000ull}) {
            const uint32_t h =
                pbrt::MixBits(1 + (uint64_t(base_index) << 4));
            printf(" %.9g",
                   double(pbrt::OwenScrambledRadicalInverse(base_index, a, h)));
        }
        printf("\n");
    }
    // And the sampler itself. A resolution that is not square and not a power
    // of two, so the two base scales differ and neither is trivial.
    {
        const pbrt::Point2i res(700, 700);
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {0, 1}, {37, 11},
                                        {699, 699}};
        for (const pbrt::Point2i &p : pixels) {
            for (int sample = 0; sample < 2; sample++) {
                pbrt::HaltonSampler sampler(
                    16, res, pbrt::RandomizeStrategy::PermuteDigits, 0);
                sampler.StartPixelSample(p, sample, 0);
                printf("halton %d %d sample %d:", p.x, p.y, sample);
                for (int i = 0; i < 2; i++) {
                    printf(" %.9g", double(sampler.Get1D()));
                }
                const pbrt::Point2f uv = sampler.Get2D();
                const pbrt::Point2f pix = sampler.GetPixel2D();
                printf(" | %.9g %.9g | %.9g %.9g\n", double(uv.x), double(uv.y),
                       double(pix.x), double(pix.y));
            }
        }
    }
    // The other two samplers, in the order the renderer draws from them: the
    // wavelength, then GetCameraSample's pixel, time and lens. Only Halton's
    // GetPixel2D was checked here before, which is why the independent
    // sampler's went wrong unnoticed -- every scene compared so far used
    // halton, and the camera-sample draws are recent.
    {
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {37, 11}};
        for (const pbrt::Point2i &p : pixels) {
            for (int sample = 0; sample < 2; sample++) {
                pbrt::IndependentSampler sampler(16, 0);
                sampler.StartPixelSample(p, sample, 0);
                const pbrt::Float lu = sampler.Get1D();
                const pbrt::Point2f pix = sampler.GetPixel2D();
                const pbrt::Float t = sampler.Get1D();
                const pbrt::Point2f lens = sampler.Get2D();
                printf("independent %d %d sample %d: %.9g | %.9g %.9g | %.9g "
                       "| %.9g %.9g\n",
                       p.x, p.y, sample, double(lu), double(pix.x),
                       double(pix.y), double(t), double(lens.x),
                       double(lens.y));
            }
        }
        for (const pbrt::Point2i &p : pixels) {
            for (int sample = 0; sample < 2; sample++) {
                pbrt::StratifiedSampler sampler(4, 4, true, 0);
                sampler.StartPixelSample(p, sample, 0);
                const pbrt::Float lu = sampler.Get1D();
                const pbrt::Point2f pix = sampler.GetPixel2D();
                const pbrt::Float t = sampler.Get1D();
                const pbrt::Point2f lens = sampler.Get2D();
                printf("stratified %d %d sample %d: %.9g | %.9g %.9g | %.9g "
                       "| %.9g %.9g\n",
                       p.x, p.y, sample, double(lu), double(pix.x),
                       double(pix.y), double(t), double(lens.x),
                       double(lens.y));
            }
        }
    }

    // The ZSobol sampler, pbrt's default when a scene names none. Its pieces
    // first -- the sequence unscrambled, each scramble on its own, the
    // eight-byte hash that seeds a dimension's scramble, and the permuted
    // index -- then the sampler, in the order the renderer draws from it: the
    // wavelength, then GetCameraSample's pixel, time and lens. At a sample
    // count that is a power of four and at one that is not, since the two
    // take different paths through GetSampleIndex, and under each of the
    // four randomizations.
    for (int dim : {0, 1}) {
        printf("sobol %d:", dim);
        for (uint64_t a : {0ull, 1ull, 2ull, 7ull, 1000ull, 123456789ull,
                           (1ull << 40) + 5}) {
            printf(" %.9g", double(pbrt::SobolSample(int64_t(a), dim,
                                                     pbrt::NoRandomizer())));
        }
        printf("\n");
    }
    for (uint32_t v : {0u, 1u, 0x80000000u, 0x12345678u, 0xdeadbeefu}) {
        const uint32_t seed = 0x9e3779b9u;
        printf("scramble %u: %u %u %u\n", v,
               pbrt::BinaryPermuteScrambler(seed)(v),
               pbrt::FastOwenScrambler(seed)(v), pbrt::OwenScrambler(seed)(v));
    }
    for (int dim : {1, 2, 7, 13}) {
        printf("hash2 %d 0: %llu\n", dim,
               static_cast<unsigned long long>(pbrt::Hash(dim, 0)));
    }
    {
        const pbrt::Point2i res(700, 700);
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {0, 1}, {37, 11},
                                        {699, 699}};
        for (int spp : {16, 8}) {
            for (const pbrt::Point2i &p : pixels) {
                for (int sample = 0; sample < 2; sample++) {
                    pbrt::ZSobolSampler sampler(
                        spp, res, pbrt::RandomizeStrategy::FastOwen, 0);
                    sampler.StartPixelSample(p, sample, 0);
                    const uint64_t index = sampler.GetSampleIndex();
                    const pbrt::Float lu = sampler.Get1D();
                    const pbrt::Point2f pix = sampler.GetPixel2D();
                    const pbrt::Float t = sampler.Get1D();
                    const pbrt::Point2f lens = sampler.Get2D();
                    printf("zsobol %d %d %d sample %d: %llu | %.9g | %.9g %.9g "
                           "| %.9g | %.9g %.9g\n",
                           spp, p.x, p.y, sample,
                           static_cast<unsigned long long>(index), double(lu),
                           double(pix.x), double(pix.y), double(t),
                           double(lens.x), double(lens.y));
                }
            }
        }
        const struct {
            const char *name;
            pbrt::RandomizeStrategy strategy;
        } strategies[] = {
            {"none", pbrt::RandomizeStrategy::None},
            {"permutedigits", pbrt::RandomizeStrategy::PermuteDigits},
            {"owen", pbrt::RandomizeStrategy::Owen},
        };
        for (const auto &s : strategies) {
            pbrt::ZSobolSampler sampler(16, res, s.strategy, 0);
            sampler.StartPixelSample(pbrt::Point2i(37, 11), 1, 0);
            const pbrt::Float lu = sampler.Get1D();
            const pbrt::Point2f pix = sampler.GetPixel2D();
            printf("zsobol-%s 37 11 sample 1: %.9g | %.9g %.9g\n", s.name,
                   double(lu), double(pix.x), double(pix.y));
        }
    }

    // The sobol sampler. First SobolIntervalToIndex on its own, at a film
    // side of 8 (m = 3, whose two van der Corput rows a test can carry) and
    // of 1024 (m = 10); then dimensions two to five of the sequence
    // unscrambled, which a test carries the columns of; then the sampler on
    // an 8 by 8 film at sixteen samples a pixel in the renderer's draw order
    // -- the wavelength (Get1D), the pixel (GetPixel2D), the time (Get1D)
    // and the lens (Get2D) -- under the default `fastowen` and, at one
    // pixel, under the other three.
    {
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {5, 3}, {7, 7}};
        for (int m : {3, 10}) {
            for (uint64_t frame : {0ull, 1ull, 5ull}) {
                for (const pbrt::Point2i &p : pixels) {
                    printf("sobol-index %d %llu %d %d: %llu\n", m,
                           static_cast<unsigned long long>(frame), p.x, p.y,
                           static_cast<unsigned long long>(
                               pbrt::SobolIntervalToIndex(m, frame, p)));
                }
            }
        }
        for (int dim : {2, 3, 4, 5}) {
            printf("sobol %d:", dim);
            for (uint64_t a : {0ull, 1ull, 2ull, 7ull, 1000ull, 123456789ull,
                               (1ull << 40) + 5}) {
                printf(" %.9g", double(pbrt::SobolSample(int64_t(a), dim,
                                                         pbrt::NoRandomizer())));
            }
            printf("\n");
        }
        const pbrt::Point2i res(8, 8);
        const struct {
            const char *name;
            pbrt::RandomizeStrategy strategy;
        } strategies[] = {
            {"fastowen", pbrt::RandomizeStrategy::FastOwen},
            {"none", pbrt::RandomizeStrategy::None},
            {"permutedigits", pbrt::RandomizeStrategy::PermuteDigits},
            {"owen", pbrt::RandomizeStrategy::Owen},
        };
        for (const auto &s : strategies) {
            for (const pbrt::Point2i &p : pixels) {
                for (int sample = 0; sample < 2; sample++) {
                    if (s.strategy != pbrt::RandomizeStrategy::FastOwen &&
                        !(p.x == 5 && sample == 1)) {
                        continue;
                    }
                    pbrt::SobolSampler sampler(16, res, s.strategy, 0);
                    sampler.StartPixelSample(p, sample, 0);
                    const pbrt::Float lu = sampler.Get1D();
                    const pbrt::Point2f pix = sampler.GetPixel2D();
                    const pbrt::Float t = sampler.Get1D();
                    const pbrt::Point2f lens = sampler.Get2D();
                    printf("sobol-%s %d %d sample %d: %.9g | %.9g %.9g | %.9g "
                           "| %.9g %.9g\n",
                           s.name, p.x, p.y, sample, double(lu), double(pix.x),
                           double(pix.y), double(t), double(lens.x),
                           double(lens.y));
                }
            }
        }
    }

    // The paddedsobol sampler at sixteen samples a pixel, in the renderer's
    // draw order, under `fastowen` at three pixels and under the other three
    // randomizations at one. Its pieces are the stratified sampler's hash
    // and permuted index and zsobol's scrambles, all printed above.
    {
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {37, 11}};
        const struct {
            const char *name;
            pbrt::RandomizeStrategy strategy;
        } strategies[] = {
            {"fastowen", pbrt::RandomizeStrategy::FastOwen},
            {"none", pbrt::RandomizeStrategy::None},
            {"permutedigits", pbrt::RandomizeStrategy::PermuteDigits},
            {"owen", pbrt::RandomizeStrategy::Owen},
        };
        for (const auto &s : strategies) {
            for (const pbrt::Point2i &p : pixels) {
                for (int sample = 0; sample < 2; sample++) {
                    if (s.strategy != pbrt::RandomizeStrategy::FastOwen &&
                        !(p.x == 37 && sample == 1)) {
                        continue;
                    }
                    pbrt::PaddedSobolSampler sampler(16, s.strategy, 0);
                    sampler.StartPixelSample(p, sample, 0);
                    const pbrt::Float lu = sampler.Get1D();
                    const pbrt::Point2f pix = sampler.GetPixel2D();
                    const pbrt::Float t = sampler.Get1D();
                    const pbrt::Point2f lens = sampler.Get2D();
                    printf("paddedsobol-%s %d %d sample %d: %.9g | %.9g %.9g | "
                           "%.9g | %.9g %.9g\n",
                           s.name, p.x, p.y, sample, double(lu), double(pix.x),
                           double(pix.y), double(t), double(lens.x),
                           double(lens.y));
                }
            }
        }
    }

    // The pmj02bn sampler at sixteen samples a pixel. Its tables first, as
    // the integers they are stored as, for the entries a test carries: the
    // blue-noise textures at three pixels for dimensions two to thirteen,
    // and the first sixteen points of each of the five sets. Then the
    // sampler in the renderer's draw order and on past the fifth set --
    // dimensions ten and up permute the index where the first five sets
    // take the sample's own -- and the camera samples of two pixels, which
    // are the first set sorted into pixels by the constructor.
    {
        const pbrt::Point2i pixels[] = {{0, 0}, {1, 0}, {37, 11}};
        for (const pbrt::Point2i &p : pixels) {
            printf("bluenoise %d %d:", p.x, p.y);
            for (int dim = 2; dim < 14; dim++) {
                printf(" %u",
                       unsigned(pbrt::BlueNoiseTextures[dim][p.x % 128]
                                                        [p.y % 128]));
            }
            printf("\n");
        }
        for (int set = 0; set < pbrt::nPMJ02bnSets; set++) {
            printf("pmj02bn-set %d:", set);
            for (int i = 0; i < 16; i++) {
                printf(" %u %u", pbrt::pmj02bnSamples[set][i][0],
                       pbrt::pmj02bnSamples[set][i][1]);
            }
            printf("\n");
        }
        for (const pbrt::Point2i &p : pixels) {
            for (int sample = 0; sample < 2; sample++) {
                pbrt::PMJ02BNSampler sampler(16, 0);
                sampler.StartPixelSample(p, sample, 0);
                const pbrt::Float lu = sampler.Get1D();
                const pbrt::Point2f pix = sampler.GetPixel2D();
                const pbrt::Float t = sampler.Get1D();
                const pbrt::Point2f lens = sampler.Get2D();
                printf("pmj02bn %d %d sample %d: %.9g | %.9g %.9g | %.9g | "
                       "%.9g %.9g |",
                       p.x, p.y, sample, double(lu), double(pix.x),
                       double(pix.y), double(t), double(lens.x),
                       double(lens.y));
                // Dimensions six to thirteen: three more pairs within the
                // five sets, then one past them.
                for (int k = 0; k < 4; k++) {
                    const pbrt::Point2f u = sampler.Get2D();
                    printf(" %.9g %.9g", double(u.x), double(u.y));
                }
                printf("\n");
            }
        }
        for (const pbrt::Point2i &p : {pbrt::Point2i(0, 0),
                                       pbrt::Point2i(37, 11)}) {
            pbrt::PMJ02BNSampler sampler(16, 0);
            printf("pmj02bn-pixel %d %d:", p.x, p.y);
            for (int sample = 0; sample < 16; sample++) {
                sampler.StartPixelSample(p, sample, 0);
                const pbrt::Point2f pix = sampler.GetPixel2D();
                printf(" %.9g %.9g", double(pix.x), double(pix.y));
            }
            printf("\n");
        }
    }

    // The first ten primes, so a mistranscribed table is caught here rather
    // than as a wrong answer three functions later.
    printf("primes:");
    for (int i = 0; i < 10; i++) {
        printf(" %d", pbrt::Primes[i]);
    }
    printf(" ... %d\n", pbrt::Primes[pbrt::PrimeTableSize - 1]);
}

// pbrt's Perlin noise and the two sums over it (util/noise.cpp), as numbers
// the transliteration in textures.bonsai is checked against
// (tests/bonsai/correctness/llvm/noise.bonsai holds these lines as its
// expected output, so the two print bare numbers in the same order and to
// the same six places as bonsai's `print`).
//
// The points land in different lattice cells at different fractions, one of
// them with negative coordinates, where the floor and the `& 255` of the
// cell index are the things to get wrong. The footprints give the octave
// count its three shapes: `1/64` along one axis is `-1 - log2(1/4096) / 2`,
// five octaves exactly; the second is a fraction under four, so the last
// octave is faded in by the smoothstep; the third is wider than a cell, so
// the count clamps to none and Turbulence is its 0.2 per unresolved octave
// alone.
void print_noise() {
    const pbrt::Point3f points[] = {{0.5f, 0.5f, 0.5f},
                                    {1.25f, -3.75f, 7.5f},
                                    {-0.1f, 0.2f, -0.3f},
                                    {100.375f, 255.5f, -256.25f},
                                    {12.345f, 67.89f, 0.f}};
    for (const pbrt::Point3f &p : points) {
        printf("%.6f\n", double(pbrt::Noise(p)));
    }
    const pbrt::Vector3f footprints[] = {
        {1 / 64.f, 0, 0}, {0.02f, 0.015f, 0.01f}, {2, 0, 0}};
    for (const pbrt::Point3f &p : points) {
        for (const pbrt::Vector3f &dpdx : footprints) {
            // A second footprint vector that is shorter, so that the max of
            // the two is the first and a sum that took the other would show.
            const pbrt::Vector3f dpdy(0.5f * dpdx.z, 0.5f * dpdx.x,
                                      0.5f * dpdx.y);
            printf("%.6f\n", double(pbrt::FBm(p, dpdx, dpdy, 0.5f, 8)));
            printf("%.6f\n", double(pbrt::Turbulence(p, dpdx, dpdy, 0.6f, 6)));
        }
    }
}

// PBRT's fixed sample points for a reflectance estimate, from the path
// integrator. Fixed rather than drawn, so that the albedo of a pixel does not
// depend on where in the sampler's stream the estimate happens to fall.
constexpr int kRhoSamples = 16;
const pbrt::Float kRhoUC[kRhoSamples] = {
    0.75741637, 0.37870818, 0.7083487,  0.18935409, 0.9149363,  0.35417435,
    0.5990858,  0.09467703, 0.8578725,  0.45746812, 0.686759,   0.17708716,
    0.9674518,  0.2995429,  0.5083201,  0.047338516};
const pbrt::Point2f kRhoU[kRhoSamples] = {
    {0.855985f, 0.570367f}, {0.381823f, 0.851844f}, {0.285328f, 0.764262f},
    {0.733380f, 0.114073f}, {0.542663f, 0.344465f}, {0.127274f, 0.414848f},
    {0.964700f, 0.947162f}, {0.594089f, 0.643463f}, {0.095109f, 0.170369f},
    {0.825444f, 0.263359f}, {0.429467f, 0.454469f}, {0.244460f, 0.816459f},
    {0.756135f, 0.731258f}, {0.516165f, 0.152852f}, {0.180888f, 0.214174f},
    {0.898579f, 0.503897f}};

// Print what PBRT's own BSDFs answer, so that the ones written in bonsai can be
// checked against them rather than against a reading of the source.
//
// `coateddiffuse` is where matching PBRT stops being about arithmetic for the
// second time. A layered BSDF has no closed form: a sample of it is a random
// walk between the coating and the base, driven by an RNG seeded from a hash of
// the very directions it was asked about. Reproducing the walk means
// reproducing that stream, and a walk with different noise agrees with PBRT in
// the limit and nowhere before it -- so an image comparison at sixteen samples
// could not tell a correct implementation from a plausible one. These numbers
// are what tests/bonsai/correctness/llvm's coated-diffuse golden holds.
//
// The BxDF is constructed here rather than obtained from a material, so the
// spectrum the diffuse base carries is a number written down on both sides
// instead of the output of the sigmoid fit. That keeps this a test of the
// scattering and not of the colour conversion, which has a check of its own.
void print_bsdf() {
    struct Case {
        const char *name;
        pbrt::Float roughness; // As a scene writes it, before the remap.
        pbrt::Float eta;
        pbrt::Float thickness;
        pbrt::Float r[4]; // The base's reflectance at the four wavelengths.
    };
    // The two roughnesses killeroo-simple asks for, a smooth coating -- which
    // takes the dielectric's delta path instead of the microfacet one -- and a
    // case whose four wavelengths differ, so a spectrum collapsed to one number
    // would show up here.
    const Case cases[] = {
        {"rough", 0.025f, 1.5f, 0.01f, {0.4f, 0.4f, 0.4f, 0.4f}},
        {"rougher", 0.15f, 1.5f, 0.01f, {0.4f, 0.5f, 0.4f, 0.4f}},
        {"smooth", 0.f, 1.5f, 0.01f, {0.5f, 0.5f, 0.5f, 0.5f}},
        {"spectral", 0.05f, 1.33f, 0.5f, {0.1f, 0.35f, 0.62f, 0.9f}},
    };
    // Directions chosen to be unalike: straight on, oblique, grazing, and one
    // from below -- which is the side a two-sided layered BSDF has to mirror.
    const pbrt::Vector3f directions[] = {
        pbrt::Normalize(pbrt::Vector3f(0.f, 0.f, 1.f)),
        pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f)),
        pbrt::Normalize(pbrt::Vector3f(0.7f, -0.5f, 0.2f)),
        pbrt::Normalize(pbrt::Vector3f(0.6f, 0.1f, 0.05f)),
        pbrt::Normalize(pbrt::Vector3f(-0.2f, 0.4f, -0.85f)),
    };

    for (const Case &c : cases) {
        // pbrt: CoatedDiffuseMaterial::GetBxDF, with `remaproughness` at its
        // default of true and no medium between the interfaces.
        const pbrt::Float alpha =
            pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(c.roughness);
        const pbrt::TrowbridgeReitzDistribution distrib(alpha, alpha);
        pbrt::SampledSpectrum r;
        for (int i = 0; i < 4; i++) {
            r[i] = c.r[i];
        }
        pbrt::CoatedDiffuseBxDF coated(pbrt::DielectricBxDF(c.eta, distrib),
                                       pbrt::DiffuseBxDF(r), c.thickness,
                                       pbrt::SampledSpectrum(0.f), /*g=*/0.f,
                                       /*maxDepth=*/10, /*nSamples=*/1);
        pbrt::BxDF bxdf(&coated);
        for (const pbrt::Vector3f &wo : directions) {
            const pbrt::SampledSpectrum rho = bxdf.rho(wo, kRhoUC, kRhoU);
            printf("rho %s %.9g %.9g %.9g:", c.name, double(wo.x), double(wo.y),
                   double(wo.z));
            for (int i = 0; i < 4; i++) {
                printf(" %.9g", double(rho[i]));
            }
            printf("\n");
        }
    }

    // `f` and `PDF`, which is what light transport needs and `rho` never asked
    // for: `rho` is built from Sample_f alone.
    //
    // Both transport modes, because they differ -- transmission into a
    // different medium is not symmetric, and `LayeredBxDF::f` samples its
    // virtual light with the mode reversed, so a renderer that only ever
    // implemented radiance mode would be wrong in exactly one term of one
    // estimator and nowhere else.
    {
        const pbrt::Vector3f pairs[][2] = {
            // Reflection, both above: the ordinary case.
            {pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f)),
             pbrt::Normalize(pbrt::Vector3f(-0.1f, 0.35f, 0.8f))},
            // Grazing on both sides, where D and G are small and the
            // denominators are not.
            {pbrt::Normalize(pbrt::Vector3f(0.7f, -0.5f, 0.15f)),
             pbrt::Normalize(pbrt::Vector3f(-0.6f, 0.4f, 0.2f))},
            // Transmission: opposite hemispheres, so etap is not 1 and the
            // generalized half vector is not wi + wo.
            {pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f)),
             pbrt::Normalize(pbrt::Vector3f(0.1f, -0.2f, -0.95f))},
            // The same, entered from below, which swaps eta for 1/eta.
            {pbrt::Normalize(pbrt::Vector3f(0.2f, 0.1f, -0.9f)),
             pbrt::Normalize(pbrt::Vector3f(-0.3f, 0.25f, 0.88f))},
        };
        for (const pbrt::Float roughness : {0.05f, 0.3f}) {
            const pbrt::Float alpha =
                pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(roughness);
            const pbrt::TrowbridgeReitzDistribution distrib(alpha, alpha);
            const pbrt::DielectricBxDF dielectric(1.5f, distrib);
            for (const auto &p : pairs) {
                const pbrt::SampledSpectrum fr =
                    dielectric.f(p[0], p[1], pbrt::TransportMode::Radiance);
                const pbrt::SampledSpectrum fi =
                    dielectric.f(p[0], p[1], pbrt::TransportMode::Importance);
                const pbrt::Float pdf = dielectric.PDF(
                    p[0], p[1], pbrt::TransportMode::Radiance);
                printf("dielectricf %.9g %.9g %.9g %.9g | %.9g %.9g %.9g\n",
                       double(roughness), double(p[0].z), double(p[1].z),
                       double(p[1].x), double(fr[0]), double(fi[0]),
                       double(pdf));
            }
        }
        // The bottom interface, whose f and PDF have no cases at all -- which
        // is worth pinning precisely because there is nothing to get wrong
        // except the hemisphere test.
        pbrt::SampledSpectrum r;
        for (int i = 0; i < 4; i++) {
            r[i] = 0.2f + 0.2f * i;
        }
        const pbrt::DiffuseBxDF diffuse(r);
        for (const auto &p : pairs) {
            const pbrt::SampledSpectrum f =
                diffuse.f(p[0], p[1], pbrt::TransportMode::Radiance);
            const pbrt::Float pdf =
                diffuse.PDF(p[0], p[1], pbrt::TransportMode::Radiance);
            printf("diffusef %.9g %.9g | %.9g %.9g %.9g\n", double(p[0].z),
                   double(p[1].z), double(f[0]), double(f[3]), double(pdf));
        }
    }

    // The pieces the walk is built from, each on its own. A layered BSDF is a
    // composition of four or five separate reproductions, and a rho that is
    // merely close says nothing about which of them is wrong.
    {
        const pbrt::Vector3f wo =
            pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f));
        const pbrt::Float alpha =
            pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(0.05f);
        const pbrt::TrowbridgeReitzDistribution distrib(alpha, alpha);
        for (int i = 0; i < 4; i++) {
            const pbrt::Vector3f wm = distrib.Sample_wm(wo, kRhoU[i]);
            printf("samplewm %d: %.9g %.9g %.9g | %.9g %.9g\n", i, double(wm.x),
                   double(wm.y), double(wm.z), double(distrib.D(wm)),
                   double(distrib.G(wo, wm)));
        }
        const pbrt::DielectricBxDF dielectric(1.5f, distrib);
        for (int i = 0; i < 4; i++) {
            const pstd::optional<pbrt::BSDFSample> bs = dielectric.Sample_f(
                wo, kRhoUC[i], kRhoU[i], pbrt::TransportMode::Radiance);
            printf("dielectric %d:", i);
            if (!bs) {
                printf(" none\n");
                continue;
            }
            printf(" %.9g | %.9g %.9g %.9g | %.9g | %d\n", double(bs->f[0]),
                   double(bs->wi.x), double(bs->wi.y), double(bs->wi.z),
                   double(bs->pdf), int(bs->flags));
        }
        // Reflection only and transmission only. Asked for by name because
        // that is how the walk asks -- and because a Fresnel term of four per
        // cent means sixteen samples of the unrestricted call are sixteen
        // transmissions, so the reflection branch would otherwise go
        // unexercised until it appeared inside a walk.
        const struct {
            const char *label;
            pbrt::BxDFReflTransFlags flags;
        } restrictions[] = {
            {"dielectric-r", pbrt::BxDFReflTransFlags::Reflection},
            {"dielectric-t", pbrt::BxDFReflTransFlags::Transmission},
        };
        for (const auto &restriction : restrictions) {
            for (int i = 0; i < 4; i++) {
                const pstd::optional<pbrt::BSDFSample> bs = dielectric.Sample_f(
                    wo, kRhoUC[i], kRhoU[i], pbrt::TransportMode::Radiance,
                    restriction.flags);
                printf("%s %d:", restriction.label, i);
                if (!bs) {
                    printf(" none\n");
                    continue;
                }
                printf(" %.9g | %.9g %.9g %.9g | %.9g | %d\n", double(bs->f[0]),
                       double(bs->wi.x), double(bs->wi.y), double(bs->wi.z),
                       double(bs->pdf), int(bs->flags));
                // The masking term about the sampled direction rather than
                // about the microfacet normal, which is the only factor of the
                // reflection branch not printed on its own above.
                if (bs->wi.z != 0) {
                    printf("%s-g %d: %.9g\n", restriction.label, i,
                           double(distrib.G(wo, bs->wi)));
                }
            }
        }

        // And from the other side. The walk spends most of its steps down
        // between the two interfaces, so every direction it asks about after
        // the first has a negative z -- a hemisphere none of the calls above
        // reach.
        const pbrt::Vector3f below = -wo;
        for (int i = 0; i < 4; i++) {
            const pbrt::Vector3f wm = distrib.Sample_wm(below, kRhoU[i]);
            printf("samplewm-b %d: %.9g %.9g %.9g | %.9g %.9g\n", i,
                   double(wm.x), double(wm.y), double(wm.z),
                   double(distrib.D(wm)), double(distrib.G(below, wm)));
            const pstd::optional<pbrt::BSDFSample> bs = dielectric.Sample_f(
                below, kRhoUC[i], kRhoU[i], pbrt::TransportMode::Radiance);
            printf("dielectric-b %d:", i);
            if (!bs) {
                printf(" none\n");
                continue;
            }
            printf(" %.9g | %.9g %.9g %.9g | %.9g | %d\n", double(bs->f[0]),
                   double(bs->wi.x), double(bs->wi.y), double(bs->wi.z),
                   double(bs->pdf), int(bs->flags));
        }
        // The intermediates of the rough transmission branch, which is the
        // longest chain of arithmetic in the file and the one where a fused
        // multiply-add on pbrt's side would first show up.
        for (int i = 0; i < 4; i++) {
            const pbrt::Vector3f wm = distrib.Sample_wm(wo, kRhoU[i]);
            pbrt::Float etap = 0;
            pbrt::Vector3f wi;
            const bool ok =
                pbrt::Refract(wo, pbrt::Normal3f(wm), 1.5f, &etap, &wi);
            if (!ok) {
                printf("refract %d: none\n", i);
                continue;
            }
            const pbrt::Float denom =
                pbrt::Sqr(pbrt::Dot(wi, wm) + pbrt::Dot(wo, wm) / etap);
            printf("refract %d: %.9g %.9g %.9g | %.9g | %.9g | %.9g | %.9g\n", i,
                   double(wi.x), double(wi.y), double(wi.z), double(etap),
                   double(denom), double(distrib.G(wo, wi)),
                   double(pbrt::FrDielectric(pbrt::Dot(wo, wm), 1.5f)));
        }

        const pbrt::DiffuseBxDF diffuse(pbrt::SampledSpectrum(0.4f));
        for (int i = 0; i < 4; i++) {
            const pstd::optional<pbrt::BSDFSample> bs = diffuse.Sample_f(
                wo, kRhoUC[i], kRhoU[i], pbrt::TransportMode::Radiance);
            printf("diffuse %d: %.9g | %.9g %.9g %.9g | %.9g | %d\n", i,
                   double(bs->f[0]), double(bs->wi.x), double(bs->wi.y),
                   double(bs->wi.z), double(bs->pdf), int(bs->flags));
        }
        printf("fresnel:");
        for (pbrt::Float c : {1.f, 0.9f, 0.3f, 0.05f, -0.4f}) {
            printf(" %.9g", double(pbrt::FrDielectric(c, 1.5f)));
        }
        printf("\n");
        // FastExp is not std::exp: pbrt scales a cubic in the fractional part
        // of x/ln 2 by writing the integer part into the exponent field, and
        // the walk multiplies by one of these at every layer crossing.
        printf("fastexp:");
        for (pbrt::Float x : {0.f, -0.01f, -0.5f, -3.25f, -40.f}) {
            printf(" %.9g", double(pbrt::FastExp(x)));
        }
        printf("\n");

        // Which end of `Point2f(r(), r())` is drawn first.
        //
        // This is not a detail. The layered walk draws its two-dimensional
        // sample as two calls in a constructor's argument list, and the order
        // C++ evaluates those in is unspecified -- gcc goes right to left,
        // clang left to right. So which of the two components gets the earlier
        // number out of the RNG is a property of the compiler pbrt was built
        // with, and getting it backwards swaps every second and third draw of
        // every step of every walk. Printed rather than assumed, because it
        // cannot be read off the source.
        {
            pbrt::RNG rng(1, 2);
            const auto r = [&rng]() {
                return std::min<pbrt::Float>(rng.Uniform<pbrt::Float>(),
                                             pbrt::OneMinusEpsilon);
            };
            const pbrt::Point2f u(r(), r());
            pbrt::RNG plain(1, 2);
            const pbrt::Float first = plain.Uniform<pbrt::Float>();
            const pbrt::Float second = plain.Uniform<pbrt::Float>();
            printf("argorder: %s (%.9g %.9g of %.9g %.9g)\n",
                   u.x == first ? "x-first" : "y-first", double(u.x),
                   double(u.y), double(first), double(second));
        }
    }

    // A single sample of the walk as well as the average of sixteen, because a
    // mean can hide a step that is wrong for a few inputs, and because the
    // direction and pdf a sample reports are not visible in rho at all.
    {
        const pbrt::Float alpha =
            pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(0.05f);
        pbrt::CoatedDiffuseBxDF coated(
            pbrt::DielectricBxDF(
                1.5f, pbrt::TrowbridgeReitzDistribution(alpha, alpha)),
            pbrt::DiffuseBxDF(pbrt::SampledSpectrum(0.4f)), 0.01f,
            pbrt::SampledSpectrum(0.f), 0.f, 10, 1);
        const pbrt::Vector3f wo = pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f));
        for (int i = 0; i < kRhoSamples; i++) {
            const pstd::optional<pbrt::BSDFSample> bs =
                coated.Sample_f(wo, kRhoUC[i], kRhoU[i],
                                pbrt::TransportMode::Radiance);
            printf("sample %d:", i);
            if (!bs) {
                printf(" none\n");
                continue;
            }
            printf(" %.9g %.9g %.9g %.9g | %.9g %.9g %.9g | %.9g | %d\n",
                   double(bs->f[0]), double(bs->f[1]), double(bs->f[2]),
                   double(bs->f[3]), double(bs->wi.x), double(bs->wi.y),
                   double(bs->wi.z), double(bs->pdf), int(bs->flags));
        }
    }

    // `LayeredBxDF::PDF` and `LayeredBxDF::Flags`, which nothing needed until
    // the path integrator did.
    //
    // The PDF is a third random walk, distinct from Sample_f's and from f's:
    // seeded from `wi` and `wo` in that order where `f` seeds from `wo` and
    // `wi`, and drawing its two entry samples as `Sample_f(w, r(), {r(), r()})`
    // -- a braced list, evaluated left to right, sitting in an argument list
    // gcc evaluates right to left, so the pair comes out of the RNG before the
    // single does. Every one of those is a thing a careful reader would get
    // wrong, and none of them is visible in an image: a walk with different
    // noise is an unbiased estimate of the same density, so it converges to the
    // same picture and agrees with pbrt at no finite sample count. Hence
    // numbers.
    //
    // Both hemispheres, because the two branches of it are entirely different
    // estimators -- a direction on pbrt's side of the surface takes the TRT
    // branch and one through it takes TT.
    {
        const struct {
            const char *name;
            pbrt::Float roughness;
            pbrt::Float eta;
            pbrt::Float thickness;
            int n_samples;
        } cases[] = {
            {"rough", 0.05f, 1.5f, 0.01f, 1},
            {"smooth", 0.f, 1.5f, 0.01f, 1},
            {"thick", 0.15f, 1.33f, 0.5f, 4},
        };
        const pbrt::Vector3f pairs[][2] = {
            {pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f)),
             pbrt::Normalize(pbrt::Vector3f(-0.1f, 0.35f, 0.8f))},
            {pbrt::Normalize(pbrt::Vector3f(0.7f, -0.5f, 0.15f)),
             pbrt::Normalize(pbrt::Vector3f(-0.6f, 0.4f, 0.2f))},
            {pbrt::Normalize(pbrt::Vector3f(0.3f, 0.2f, 0.9f)),
             pbrt::Normalize(pbrt::Vector3f(0.1f, -0.2f, -0.95f))},
            {pbrt::Normalize(pbrt::Vector3f(0.2f, 0.1f, -0.9f)),
             pbrt::Normalize(pbrt::Vector3f(-0.3f, 0.25f, 0.88f))},
        };
        for (const auto &c : cases) {
            const pbrt::Float alpha =
                pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(
                    c.roughness);
            pbrt::CoatedDiffuseBxDF coated(
                pbrt::DielectricBxDF(
                    c.eta, pbrt::TrowbridgeReitzDistribution(alpha, alpha)),
                pbrt::DiffuseBxDF(pbrt::SampledSpectrum(0.4f)), c.thickness,
                pbrt::SampledSpectrum(0.f), 0.f, 10, c.n_samples);
            printf("coatedflags %s: %d\n", c.name, int(coated.Flags()));
            for (const auto &p : pairs) {
                printf("coatedpdf %s %.9g %.9g: %.9g\n", c.name,
                       double(p[0].z), double(p[1].z),
                       double(coated.PDF(p[0], p[1],
                                         pbrt::TransportMode::Radiance)));
            }
        }
    }

    // The two hashes the walk is seeded with, on their own. They are the part
    // most likely to be transcribed wrongly -- a Murmur over the bytes of a
    // float is not something a reader can check by eye -- and the hardest to
    // see through an RNG and a random walk.
    for (const pbrt::Vector3f &wo : directions) {
        printf("seedhash %.9g %.9g %.9g: %llu\n", double(wo.x), double(wo.y),
               double(wo.z),
               static_cast<unsigned long long>(
                   pbrt::Hash(pbrt::GetOptions().seed, wo)));
    }
    for (int i = 0; i < 4; i++) {
        printf("uhash %d: %llu\n", i,
               static_cast<unsigned long long>(
                   pbrt::Hash(kRhoUC[i], kRhoU[i])));
    }
}

// Print the shading geometry PBRT computes at a hit, and the frame a BSDF is
// evaluated in.
//
// This is the other half of what a layered material needs and the half that is
// invisible in the gbuffer. The normal a film records is turned to face the
// camera, so its sign and the whole tangent direction drop out of the
// comparison; a BSDF sees all of it, and a layered one hashes the outgoing
// direction expressed in that frame to seed its random walk. A tangent one bit
// out gives a walk with different noise, which converges to the same
// reflectance and agrees with PBRT at no finite sample count.
//
// The mesh is built here rather than taken from a scene so that the inputs are
// numbers written down on both sides -- the same reason print_bsdf constructs
// its BxDFs. What tests/bonsai/correctness/llvm's shading-frame golden holds is
// this output.
// Sphere::Sample(ctx, u) and Sphere::PDF(ctx, wi), asked of pbrt.
//
// This is what a light-sampling integrator stands on: a direction chosen inside
// the cone the sphere subtends, and the density it was chosen with. Getting it
// wrong gives a picture that is merely differently noisy, so it is pinned here
// rather than inferred from an image.
void print_shape_sample() {
    const pbrt::Transform identity;
    const pbrt::Transform *to_render = &identity;
    // Radius and centre are separated so that the object-to-render transform is
    // a translation, which is all this renderer represents.
    const struct {
        pbrt::Point3f centre;
        pbrt::Float radius;
    } spheres[] = {{{0.f, 7.f, 2.f}, 2.5f}, {{-246.7f, 65.2f, -10.f}, 3.f}};
    // Points to look at each sphere from: near, far, and off to one side.
    const pbrt::Point3f from[] = {
        {0.f, 0.f, 0.f}, {3.f, 1.f, 4.f}, {-100.f, 20.f, 60.f}};
    const pbrt::Point2f us[] = {
        {0.f, 0.f}, {0.5f, 0.25f}, {0.87f, 0.63f}, {0.999f, 0.001f}};

    for (const auto &sph : spheres) {
        const pbrt::Transform translate = pbrt::Translate(
            pbrt::Vector3f(sph.centre.x, sph.centre.y, sph.centre.z));
        const pbrt::Transform inverse = pbrt::Inverse(translate);
        pbrt::Sphere sphere(&translate, &inverse, /*reverseOrientation=*/false,
                            sph.radius, -sph.radius, sph.radius, 360.f);
        for (const pbrt::Point3f &p : from) {
            const pbrt::ShapeSampleContext ctx(pbrt::Point3fi(p),
                                               pbrt::Normal3f(0, 1, 0),
                                               pbrt::Normal3f(0, 1, 0), 0.f);
            for (const pbrt::Point2f &u : us) {
                const pstd::optional<pbrt::ShapeSample> ss =
                    sphere.Sample(ctx, u);
                if (!ss) {
                    printf("spheresample %g %g: none\n", double(u.x),
                           double(u.y));
                    continue;
                }
                const pbrt::Vector3f wi =
                    pbrt::Normalize(ss->intr.p() - ctx.p());
                printf("spheresample %g %g %g r %g from %g %g %g u %g %g:"
                       " p %.9g %.9g %.9g n %.9g %.9g %.9g pdf %.9g"
                       " pdfback %.9g\n",
                       double(sph.centre.x), double(sph.centre.y),
                       double(sph.centre.z), double(sph.radius), double(p.x),
                       double(p.y), double(p.z), double(u.x), double(u.y),
                       double(ss->intr.p().x), double(ss->intr.p().y),
                       double(ss->intr.p().z), double(ss->intr.n.x),
                       double(ss->intr.n.y), double(ss->intr.n.z),
                       double(ss->pdf), double(sphere.PDF(ctx, wi)));
            }
        }
    }
}

// What a DiffuseAreaLight actually emits, asked of pbrt.
//
// The scale a light carries is the one piece of a scene that cannot be checked
// by looking at the picture: it multiplies every lit pixel equally, so getting
// it wrong is a render that is uniformly too bright or too dim and otherwise
// perfectly correct. This prints pbrt's own answer for the light
// killeroo-simple declares, built through the same `Create` path pbrt uses.
void print_light() {
    // `AreaLightSource "diffuse" "rgb L" [2000 2000 2000]`, which is
    // killeroo-simple's, and a unit one for contrast.
    const pbrt::Float rgbs[][3] = {{2000.f, 2000.f, 2000.f},
                                   {1.f, 1.f, 1.f},
                                   {0.4f, 0.8f, 0.2f}};
    pbrt::SampledWavelengths lambda =
        pbrt::SampledWavelengths::SampleVisible(0.5f);
    printf("lambda");
    for (int i = 0; i < 4; i++) {
        printf(" %.9g", double(lambda[i]));
    }
    printf("\n");

    for (const auto &rgb : rgbs) {
        const pbrt::RGB c(rgb[0], rgb[1], rgb[2]);
        // What GetOneSpectrum(..., SpectrumType::Illuminant) builds.
        pbrt::RGBIlluminantSpectrum emitted(*pbrt::RGBColorSpace::sRGB, c);
        const pbrt::Float photometric =
            pbrt::SpectrumToPhotometric(&emitted);
        // DiffuseAreaLight::Create: `scale /= SpectrumToPhotometric(L)`, with
        // the scene's own scale of one.
        const pbrt::Float scale = 1.f / photometric;
        const pbrt::SampledSpectrum sampled = emitted.Sample(lambda);
        printf("light %g %g %g | photometric %.9g scale %.9g |", double(rgb[0]),
               double(rgb[1]), double(rgb[2]), double(photometric),
               double(scale));
        for (int i = 0; i < 4; i++) {
            printf(" %.9g", double(scale * sampled[i]));
        }
        printf("\n");
    }
}

void print_shading() {
    pbrt::Allocator alloc;
    // Two triangles sharing an edge, at a scale and an angle nothing about is
    // round: a vertex on an axis or a normal already perpendicular to the
    // tangent would take a branch that the general case does not.
    const std::vector<int> indices = {0, 1, 2, 2, 1, 3};
    const std::vector<pbrt::Point3f> p = {
        {-36.876f, 26.033f, -137.748f},
        {-37.039f, 21.507f, -137.128f},
        {-35.466f, -2.075f, -129.337f},
        {-30.348f, -8.431f, -132.687f}};
    // Not unit vectors, and not agreeing with the face they sit on: a
    // subdivision surface's limit normals are neither.
    const std::vector<pbrt::Normal3f> n = {{0.31f, 0.42f, 0.85f},
                                           {0.09f, 0.55f, 0.83f},
                                           {-0.22f, 0.61f, 0.76f},
                                           {0.40f, -0.13f, 0.91f}};
    const std::vector<pbrt::Point2f> uv = {
        {0.f, 0.f}, {0.7f, 0.1f}, {0.2f, 0.9f}, {1.f, 1.f}};

    struct Case {
        const char *label;
        bool with_normals;
        bool with_uv;
        bool reverse;
    };
    const Case cases[] = {
        // What a `loopsubdiv` shape produces: normals and no texture
        // coordinates, so the tangent comes from PBRT's default (0,0), (1,0),
        // (1,1) parameterization and is then made perpendicular to the
        // interpolated normal.
        {"subdiv", true, false, false},
        // What a plain `trianglemesh` with `uv` produces.
        {"uv", false, true, false},
        // Both, and the orientation reversed -- which flips the geometric
        // normal and so the frame's second axis.
        {"both", true, true, true},
    };

    const pbrt::Point3f origin(-20.f, 60.f, -60.f);
    const pbrt::Point3f targets[] = {{-36.f, 20.f, -135.f},
                                     {-34.f, 8.f, -132.f},
                                     {-33.f, 0.f, -131.f}};

    for (const Case &c : cases) {
        const pbrt::TriangleMesh *mesh = alloc.new_object<pbrt::TriangleMesh>(
            pbrt::Transform(), c.reverse, indices, p,
            std::vector<pbrt::Vector3f>(),
            c.with_normals ? n : std::vector<pbrt::Normal3f>(),
            c.with_uv ? uv : std::vector<pbrt::Point2f>(), std::vector<int>(),
            alloc);
        for (const pbrt::Point3f &target : targets) {
            const pbrt::Ray ray(origin, pbrt::Normalize(target - origin));
            for (int tri = 0; tri < 2; tri++) {
                const pstd::optional<pbrt::TriangleIntersection> ti =
                    pbrt::IntersectTriangle(ray, pbrt::Infinity,
                                            p[indices[3 * tri + 0]],
                                            p[indices[3 * tri + 1]],
                                            p[indices[3 * tri + 2]]);
                if (!ti) {
                    continue;
                }
                const pbrt::SurfaceInteraction isect =
                    pbrt::Triangle::InteractionFromIntersection(
                        mesh, tri, *ti, 0.f, -ray.d);
                const pbrt::Frame frame = pbrt::Frame::FromXZ(
                    pbrt::Normalize(isect.shading.dpdu),
                    pbrt::Vector3f(isect.shading.n));
                const pbrt::Vector3f local = frame.ToLocal(isect.wo);
                printf("bary %s %d %.9g: %.9g %.9g %.9g %.9g\n", c.label, tri,
                       double(target.y), double(ti->b0), double(ti->b1),
                       double(ti->b2), double(ti->t));
                printf("shading %s %d %.9g: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                       "%.9g %.9g %.9g | %.9g %.9g %.9g\n",
                       c.label, tri, double(target.y), double(isect.n.x),
                       double(isect.n.y), double(isect.n.z),
                       double(isect.shading.n.x), double(isect.shading.n.y),
                       double(isect.shading.n.z), double(isect.shading.dpdu.x),
                       double(isect.shading.dpdu.y), double(isect.shading.dpdu.z),
                       double(local.x), double(local.y), double(local.z));
                // The parametric coordinates and the two derivatives a texture
                // is filtered in. `uv` is where a textured material is asked
                // about, and `dpdu`/`dpdv` are the *geometric* pair rather than
                // the shading one printed above -- pbrt keeps both, and the
                // difference between them is the orthogonalization against the
                // shading normal.
                printf("uvdp %s %d %.9g: %.9g %.9g | %.9g %.9g %.9g | "
                       "%.9g %.9g %.9g\n",
                       c.label, tri, double(target.y), double(isect.uv.x),
                       double(isect.uv.y), double(isect.dpdu.x),
                       double(isect.dpdu.y), double(isect.dpdu.z),
                       double(isect.dpdv.x), double(isect.dpdv.y),
                       double(isect.dpdv.z));
                // The interpolated shading normal before it is normalized,
                // written as PBRT writes it and so compiled the way PBRT's is.
                // Three multiplies and two adds, which the compiler fuses, and
                // which of them it fuses is not something the source says.
                if (mesh->n != nullptr) {
                    const int *iv = &indices[3 * tri];
                    const pbrt::Normal3f raw = ti->b0 * mesh->n[iv[0]] +
                                               ti->b1 * mesh->n[iv[1]] +
                                               ti->b2 * mesh->n[iv[2]];
                    printf("rawns %s %d %.9g: %.9g %.9g %.9g\n", c.label, tri,
                           double(target.y), double(raw.x), double(raw.y),
                           double(raw.z));
                }
                // The length of the tangent before it is normalized. A dot
                // product is a chain of multiply-adds and so a chain of
                // roundings, and which of them the compiler fuses decides the
                // last bit of every axis built from it.
                printf("length %s %d %.9g: %.9g %.9g\n", c.label, tri,
                       double(target.y),
                       double(pbrt::LengthSquared(isect.shading.dpdu)),
                       double(pbrt::Length(isect.shading.dpdu)));
                // The frame itself and the direction expressed in it, so that a
                // disagreement about the answer can be traced to which axis.
                printf("frame %s %d %.9g: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                       "%.9g %.9g %.9g\n",
                       c.label, tri, double(target.y), double(frame.x.x),
                       double(frame.x.y), double(frame.x.z), double(frame.y.x),
                       double(frame.y.y), double(frame.y.z), double(isect.wo.x),
                       double(isect.wo.y), double(isect.wo.z));
            }
        }
    }
    // The rays themselves, so that the other side is answering about the same
    // ones rather than about its own idea of them.
    for (const pbrt::Point3f &target : targets) {
        const pbrt::Vector3f d = pbrt::Normalize(target - origin);
        printf("ray %.9g: %.9g %.9g %.9g\n", double(target.y), double(d.x),
               double(d.y), double(d.z));
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

// What PBRT's HaltonSampler constructor derives from the film resolution.
//
// The first two dimensions of the Halton sequence are base 2 and base 3, and
// their leading digits are what decide which pixel a sample lands in. So the
// scales are the smallest power of each base that covers the image -- capped at
// MaxHaltonResolution, above which the pattern repeats rather than growing --
// and the exponents say how many digits that took.
//
// The multiplicative inverses are the Chinese remainder theorem: given where a
// sample sits along each axis, they recover the one sequence index that puts it
// there. Powers of two and three are coprime, so each is invertible modulo the
// other.
//
// Here rather than in bonsai because PBRT computes it in a constructor, once per
// render, from numbers the scene already fixed -- the same reason the sigmoid
// fit and the spectral tables are computed out here.
void halton_scales(int x_resolution, int y_resolution,
                   bonsai_scene::Sampler &sampler) {
    static constexpr int kMaxHaltonResolution = 128;
    const int resolution[2] = {x_resolution, y_resolution};
    for (int i = 0; i < 2; i++) {
        const int base = (i == 0) ? 2 : 3;
        int scale = 1, exponent = 0;
        while (scale < std::min(resolution[i], kMaxHaltonResolution)) {
            scale *= base;
            ++exponent;
        }
        sampler.base_scales[i] = scale;
        sampler.base_exponents[i] = exponent;
    }

    // PBRT's extendedGCD, iteratively. `x` is the inverse of a modulo n once
    // the recursion unwinds, and Mod rather than % because it can come out
    // negative.
    const auto inverse = [](int64_t a, int64_t n) {
        int64_t old_r = a, r = n;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            const int64_t q = old_r / r;
            int64_t t = old_r - q * r;
            old_r = r;
            r = t;
            t = old_s - q * s;
            old_s = s;
            s = t;
        }
        const int64_t m = old_s % n;
        return int32_t(m < 0 ? m + n : m);
    };
    sampler.mult_inverse[0] =
        inverse(sampler.base_scales[1], sampler.base_scales[0]);
    sampler.mult_inverse[1] =
        inverse(sampler.base_scales[0], sampler.base_scales[1]);
}

// The triangles a shape entity comes to, or null if it is not a mesh.
//
// A .pbrt file spells its geometry many ways -- an explicit index buffer, a PLY
// file, a subdivision control cage -- and all but a handful of them are a
// triangle mesh by the time PBRT is finished with them. Only the last step of
// that is shared, so this dispatches on the name exactly as PBRT's own
// Shape::Create does, and each arm hands the work straight back to PBRT: the
// PLY reader, the Loop subdivider, the mesh constructor. Nothing here
// reimplements any of it, which matters most for `loopsubdiv` -- the tessellated
// vertex positions are a limit surface, not a copy of the control cage, and a
// second implementation of Loop subdivision would be a second answer.
//
// Returning the mesh rather than PBRT's `Shape`s is what keeps this to public
// API. A built Triangle knows its vertices only through private members, where
// TriangleMesh publishes an index buffer and a vertex array -- and the mesh is
// what every one of these factories produces anyway.
//
// Render space, not object space: TriangleMesh's constructor applies the
// A PLY read once: both the triangle mesh and the patch mesh of a `plymesh`
// come from the same file, and the big ones take a while.
const pbrt::TriQuadMesh &read_ply(const std::string &file) {
    static std::map<std::string, pbrt::TriQuadMesh> cache;
    auto it = cache.find(file);
    if (it == cache.end()) {
        it = cache.emplace(file, pbrt::TriQuadMesh::ReadPLY(file)).first;
    }
    return it->second;
}

// transform to the vertices rather than storing it, so a shape's placement is
// already baked in here the same way it is in PBRT.
const pbrt::TriangleMesh *triangulate(const pbrt::ShapeSceneEntity &entity) {
    const std::string name(entity.name);
    pbrt::Allocator alloc;

    if (name == "trianglemesh") {
        return pbrt::Triangle::CreateMesh(entity.renderFromObject,
                                          entity.reverseOrientation,
                                          entity.parameters, &entity.loc, alloc);
    }

    if (name == "loopsubdiv") {
        // PBRT's default is three levels, and the level count changes the
        // geometry rather than merely refining it, so the default has to be
        // PBRT's default and not a cheaper one.
        const int levels = entity.parameters.GetOneInt("levels", 3);
        const std::vector<int> indices =
            entity.parameters.GetIntArray("indices");
        const std::vector<pbrt::Point3f> P =
            entity.parameters.GetPoint3fArray("P");
        if (indices.empty() || P.empty()) {
            fail("a loopsubdiv shape is missing \"indices\" or \"P\"");
        }
        return pbrt::LoopSubdivide(entity.renderFromObject,
                                   entity.reverseOrientation, levels, indices,
                                   P, alloc);
    }

    if (name == "plymesh") {
        const std::string file =
            pbrt::ResolveFilename(entity.parameters.GetOneString("filename", ""));
        if (file.empty()) {
            fail("a plymesh has no \"filename\"");
        }
        const pbrt::TriQuadMesh &ply = read_ply(file);
        // A PLY may hold quads as well as triangles, and PBRT makes those a
        // bilinear patch mesh rather than splitting them -- a bilinear patch
        // is not two triangles unless it happens to be planar. The quads are
        // `patch_mesh_of`'s; the triangles, if there are any, are this mesh.
        if (ply.triIndices.empty()) {
            if (ply.quadIndices.empty()) {
                fail("the PLY file " + file + " has no faces");
            }
            return nullptr;
        }
        return alloc.new_object<pbrt::TriangleMesh>(
            *entity.renderFromObject, entity.reverseOrientation, ply.triIndices,
            ply.p, std::vector<pbrt::Vector3f>(), ply.n, ply.uv,
            ply.faceIndices, alloc);
    }

    return nullptr;
}

// PBRT: the bilinear patch mesh a shape is, if it is one -- `bilinearmesh`
// (BilinearPatch::CreateMesh, with PBRT's own checks) or the quads of a
// `plymesh` (shapes.cpp: a BilinearPatchMesh over `quadIndices`, beside the
// TriangleMesh over its triangles) -- in render space, as `triangulate`'s
// is. `emissionfilename`, an image distribution over an emitting patch, is
// refused: the renderer samples a patch by its geometry alone.
const pbrt::BilinearPatchMesh *
patch_mesh_of(const pbrt::ShapeSceneEntity &entity) {
    const std::string name(entity.name);
    pbrt::Allocator alloc;

    if (name == "bilinearmesh") {
        if (!entity.parameters.GetOneString("emissionfilename", "").empty()) {
            fail("a bilinearmesh with an `emissionfilename` is not supported; "
                 "this renderer samples a patch by its area or its solid angle");
        }
        return pbrt::BilinearPatch::CreateMesh(entity.renderFromObject,
                                               entity.reverseOrientation,
                                               entity.parameters, &entity.loc,
                                               alloc);
    }

    if (name == "plymesh") {
        const std::string file =
            pbrt::ResolveFilename(entity.parameters.GetOneString("filename", ""));
        const pbrt::TriQuadMesh &ply = read_ply(file);
        if (ply.quadIndices.empty()) {
            return nullptr;
        }
        return alloc.new_object<pbrt::BilinearPatchMesh>(
            *entity.renderFromObject, entity.reverseOrientation, ply.quadIndices,
            ply.p, ply.n, ply.uv, ply.faceIndices, nullptr, alloc);
    }

    return nullptr;
}

// PBRT: BilinearPatch's constructor's `area` -- exact for a rectangle, else
// the sum over a 3x3 subdivision of the patch into bilinear cells, each the
// half-cross-product of its diagonals. IsRectangle as PBRT writes it: four
// distinct coplanar corners the same distance from their centre.
float patch_area_of(const pbrt::BilinearPatchMesh *mesh, int patch) {
    const int *v = &mesh->vertexIndices[4 * patch];
    const pbrt::Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    const pbrt::Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];
    const auto is_rectangle = [&]() {
        if (p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00) {
            return false;
        }
        const pbrt::Normal3f n(pbrt::Normalize(pbrt::Cross(p10 - p00, p01 - p00)));
        if (pbrt::AbsDot(pbrt::Normalize(p11 - p00), n) > 1e-5f) {
            return false;
        }
        const pbrt::Point3f centre = (p00 + p01 + p10 + p11) / 4;
        const pbrt::Float d2[4] = {
            pbrt::DistanceSquared(p00, centre), pbrt::DistanceSquared(p01, centre),
            pbrt::DistanceSquared(p10, centre), pbrt::DistanceSquared(p11, centre)};
        for (int i = 1; i < 4; ++i) {
            if (std::abs(d2[i] - d2[0]) / d2[0] > 1e-4f) {
                return false;
            }
        }
        return true;
    };
    if (is_rectangle()) {
        return float(pbrt::Distance(p00, p01) * pbrt::Distance(p00, p10));
    }
    constexpr int na = 3;
    pbrt::Point3f p[na + 1][na + 1];
    for (int i = 0; i <= na; ++i) {
        const pbrt::Float u = pbrt::Float(i) / pbrt::Float(na);
        for (int j = 0; j <= na; ++j) {
            const pbrt::Float w = pbrt::Float(j) / pbrt::Float(na);
            p[i][j] = pbrt::Lerp(u, pbrt::Lerp(w, p00, p01), pbrt::Lerp(w, p10, p11));
        }
    }
    pbrt::Float area = 0;
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < na; ++j) {
            area += 0.5f * pbrt::Length(pbrt::Cross(p[i + 1][j + 1] - p[i][j],
                                                    p[i + 1][j] - p[i][j + 1]));
        }
    }
    return float(area);
}

// Serializes PBRT's light BVH: builds PBRT's own BVHLightSampler over `lights`
// and hands back the tree it produced, its nodes dequantized (see
// bonsai_scene::LightTreeNode) and a bit trail per light ordinal.
//
// This is PBRT's actual builder. `BVHLightSampler`'s constructor runs its
// surface-area heuristic, its octahedral and sixteen-bit quantization and all;
// what is read out here is that result, node for node, not a re-derivation of
// it. The tree, the bit trails and the compact bounds' `w` are private with no
// accessor, which is why the file opens member access for that one header at
// the top -- the seam a future light-tree project would replace is the
// `BVHLightSampler` constructor call, and nothing here.
//
// A leaf's `childOrLightIndex` is the light's position in the span passed in,
// which is the ordinal the emitting shapes carry, so the leaves line up with
// the renderer's per-shape light list; each ordinal's bit trail is read from
// `lightToBitTrail` by the light handle sitting at that ordinal. The compact
// bounds are read back through PBRT's own dequantizing accessors, so the floats
// stored are the ones PBRT's importance actually uses.
void dump_light_tree(const std::vector<pbrt::Light> &lights,
                     bonsai_scene::Scene &out) {
    pbrt::Allocator alloc;
    const pbrt::BVHLightSampler sampler(
        pstd::span<const pbrt::Light>(lights.data(), lights.size()), alloc);

    for (const pbrt::LightBVHNode &node : sampler.nodes) {
        const pbrt::CompactLightBounds &cb = node.lightBounds;
        const pbrt::Bounds3f b = cb.Bounds(sampler.allLightBounds);
        const pbrt::Vector3f w = pbrt::Vector3f(cb.w);
        bonsai_scene::LightTreeNode n;
        n.w[0] = float(w.x);
        n.w[1] = float(w.y);
        n.w[2] = float(w.z);
        n.phi = float(cb.phi);
        n.cos_theta_o = float(cb.CosTheta_o());
        n.cos_theta_e = float(cb.CosTheta_e());
        n.bounds_min[0] = float(b.pMin.x);
        n.bounds_min[1] = float(b.pMin.y);
        n.bounds_min[2] = float(b.pMin.z);
        n.bounds_max[0] = float(b.pMax.x);
        n.bounds_max[1] = float(b.pMax.y);
        n.bounds_max[2] = float(b.pMax.z);
        n.two_sided = cb.TwoSided() ? 1u : 0u;
        n.child_or_light = node.childOrLightIndex;
        n.is_leaf = node.isLeaf ? 1u : 0u;
        out.light_tree.push_back(n);
    }

    out.light_bit_trails.assign(lights.size(), 0u);
    for (size_t ord = 0; ord < lights.size(); ++ord) {
        out.light_bit_trails[ord] = sampler.lightToBitTrail[lights[ord]];
    }
}

// Parse and convert. Everything PBRT owns is local to this function, so all of
// it is destroyed on the way out -- before CleanupPBRT takes the arenas it was
// allocated from out from under it. Doing this inline in main instead crashes
// on the way out, because the locals outlive the cleanup call.
// PBRT: PortalImageInfiniteLight's constructor, the part of it that is a
// function of the scene alone -- the corners checked, the equal-area map
// resampled into the portal's own parameterization (each texel of the new
// image is the direction its (u, v) names, taken back into the light's frame
// and read bilinearly off the map with the octahedral wrap), and the sampling
// values, the texel average times the Jacobian d(u,v)/dw at the texel -- done
// here once with PBRT's own functions, and written as a `portal` light whose
// texels are fitted spectra exactly as the environment map's are. The frame,
// the window and the sampling at render time are the renderer's (render.bonsai,
// sampling.bonsai); the summed-area table over the values is the driver's.
// `portal` is in render space already, `out_light` carries the scale.
static void emit_portal_light(const pbrt::Image &equal_area,
                              const SpectrumTable &table,
                              const pbrt::Transform &render_from_light,
                              const std::vector<pbrt::Point3f> &portal,
                              bonsai_scene::InfiniteLight out_light,
                              bonsai_scene::Scene &out) {
    if (portal.size() != 4) {
        fail("an `infinite` light's portal takes exactly four points, and "
             "this one has " + std::to_string(portal.size()));
    }
    // PBRT: opposite edges the same direction, adjacent ones perpendicular
    // -- reported with `Error` and rendered anyway, so a warning here.
    const pbrt::Vector3f p01 = pbrt::Normalize(portal[1] - portal[0]);
    const pbrt::Vector3f p12 = pbrt::Normalize(portal[2] - portal[1]);
    const pbrt::Vector3f p32 = pbrt::Normalize(portal[2] - portal[3]);
    const pbrt::Vector3f p03 = pbrt::Normalize(portal[3] - portal[0]);
    if (std::abs(pbrt::Dot(p01, p32) - 1) > .001 ||
        std::abs(pbrt::Dot(p12, p03) - 1) > .001 ||
        std::abs(pbrt::Dot(p01, p12)) > .001 ||
        std::abs(pbrt::Dot(p12, p32)) > .001 ||
        std::abs(pbrt::Dot(p32, p03)) > .001 ||
        std::abs(pbrt::Dot(p03, p01)) > .001) {
        std::fprintf(stderr, "scene_dump: the infinite light's portal is not a "
                             "planar rectangle; PBRT reports that and renders "
                             "anyway, and so does this\n");
    }
    const pbrt::Frame frame = pbrt::Frame::FromXY(p03, p01);
    const int res = equal_area.Resolution().x;
    // PBRT: RenderFromImage(uv, &duv_dw).
    const auto render_from_image = [&](pbrt::Point2f uv, float *duv_dw) {
        const float alpha = -pbrt::Pi / 2 + uv[0] * pbrt::Pi;
        const float beta = -pbrt::Pi / 2 + uv[1] * pbrt::Pi;
        const float x = std::tan(alpha), y = std::tan(beta);
        const pbrt::Vector3f w = pbrt::Normalize(pbrt::Vector3f(x, y, 1));
        *duv_dw = pbrt::Sqr(pbrt::Pi) * (1 - pbrt::Sqr(w.x)) *
                  (1 - pbrt::Sqr(w.y)) / w.z;
        return frame.FromLocal(w);
    };
    out_light.portal = 1;
    for (int i = 0; i < 4; i++) {
        for (int c = 0; c < 3; c++) {
            out_light.portal_points[3 * i + c] = float(portal[i][c]);
        }
    }
    out_light.resolution = uint32_t(res);
    out_light.first_texel = uint32_t(out.env_sampling.size());
    out.env_texels.reserve(out.env_texels.size() + size_t(res) * res * 4);
    out.env_sampling.reserve(out.env_sampling.size() + size_t(res) * res);
    for (int y = 0; y < res; y++) {
        for (int x = 0; x < res; x++) {
            const pbrt::Point2f uv((x + 0.5f) / res, (y + 0.5f) / res);
            float duv_dw = 0.f;
            pbrt::Vector3f w = render_from_image(uv, &duv_dw);
            w = pbrt::Normalize(render_from_light.ApplyInverse(w));
            const pbrt::Point2f uv_equi = pbrt::EqualAreaSphereToSquare(w);
            float rgb[3];
            for (int c = 0; c < 3; c++) {
                rgb[c] = equal_area.BilerpChannel(
                    uv_equi, c, pbrt::WrapMode::OctahedralSphere);
            }
            // PBRT: GetSamplingDistribution(duv_dw): the channels' average
            // (unclamped, in float) times the Jacobian at the texel's centre.
            out.env_sampling.push_back(((rgb[0] + rgb[1] + rgb[2]) / 3.f) *
                                       duv_dw);
            // PBRT: ImageLookup's ClampZero and RGBIlluminantSpectrum, as for
            // the environment map's texels.
            const float r = std::max(0.f, rgb[0]);
            const float g = std::max(0.f, rgb[1]);
            const float b = std::max(0.f, rgb[2]);
            const float m = std::max({r, g, b});
            const float texel_scale = 2 * m;
            const Sigmoid rsp =
                texel_scale != 0.f
                    ? fit_sigmoid(table, r / texel_scale, g / texel_scale,
                                  b / texel_scale)
                    : fit_sigmoid(table, 0.f, 0.f, 0.f);
            out.env_texels.push_back(rsp.c0);
            out.env_texels.push_back(rsp.c1);
            out.env_texels.push_back(rsp.c2);
            out.env_texels.push_back(texel_scale);
        }
    }
    out.infinite_lights.push_back(out_light);
}

void load(const char *filename, bonsai_scene::Scene &out) {
    std::vector<float> matrices;
    std::vector<bonsai_scene::Shape> &shapes = out.shapes;

    pbrt::BasicScene scene;
    CapturingBuilder builder(&scene);
    const std::vector<std::string> filenames = {filename};
    pbrt::ParseFiles(&builder, filenames);

    // What `convert_texture` needs: where the images are, and where to put the
    // pyramids it builds. A texture's filename is relative to the scene file,
    // as every other path in a .pbrt is.
    g_builder = &builder;
    g_scene = &out;
    g_texture_index.clear();
    g_texture_mip.clear();
    g_normal_map_index.clear();
    g_image_source.clear();
    g_material_for_named = nullptr;
    {
        const std::string path(filename);
        const size_t slash = path.find_last_of('/');
        g_scene_dir = (slash == std::string::npos) ? std::string(".")
                                                   : path.substr(0, slash);
    }


    if (builder.camera_name != "perspective") {
        fail("only the perspective camera is supported, scene asks for \"" +
             builder.camera_name + "\"");
    }

    const int x_resolution = builder.film_params.GetOneInt("xresolution", 1280);
    const int y_resolution = builder.film_params.GetOneInt("yresolution", 720);

    out.width = uint32_t(x_resolution);
    out.height = uint32_t(y_resolution);

    // A light that is not a shape's emission. Until one of these is
    // implemented it has to be refused rather than dropped: a scene lit only by
    // an environment map that quietly rendered black would look like a scene
    // and not like an error, which is how this went unnoticed for as long as it
    // did.
    //
    // `infinite` with no image is PBRT's UniformInfiniteLight and is converted.
    // `infinite` *with* one is ImageInfiniteLight -- a different light, and the
    // one every real scene in pbrt-v4-scenes actually asks for -- so it is
    // named in the refusal rather than approximated by its average.
    //
    // PBRT's own PointLight and SpotLight for each of the scene's, for the
    // light tree: BVHLightSampler asks every light for its bounds, and these
    // are built by PBRT's constructors from what the conversion below
    // computes, so the bounds, the surface-area heuristic and the
    // quantization are PBRT's, as the area lights' are. They join the area
    // lights' at the tree build, after them and in declaration order, which
    // is where PBRT's own light list has them (BasicScene::CreateLights:
    // every area light first, then the `LightSource` lights as declared) and
    // where the driver puts them. Allocated to outlive the build.
    pbrt::Allocator positional_alloc;
    std::vector<pbrt::Light> positional_lights;
    for (const CapturingBuilder::MaterialInfo &light : builder.lights) {
        if (light.name == "point" || light.name == "spot") {
            // PBRT: PointLight::Create and SpotLight::Create. `I` defaults to
            // the colour space's illuminant and the scale is divided by its
            // photometric integral, as a distant light's L is below; a
            // `power` then multiplies the scale by itself over the solid
            // angle the light covers -- 4 pi for a point, the cone's for a
            // spot. The light sits at `from` under the scene's transform,
            // and a spot's frame has its z axis towards `to`
            // (Frame::FromZ), PBRT's renderFromLight being the scene's
            // times the translation times that frame's inverse.
            if (!light.ctm_is_tracked) {
                fail("a `" + light.name + "` light placed by a named "
                     "coordinate system is not supported: scene_dump follows "
                     "the transform stack itself and does not follow that "
                     "directive");
            }
            const bool spot = light.name == "spot";
            bonsai_scene::PointLight bulb;
            bulb.spot = spot ? 1u : 0u;
            const auto point = [&](const char *key, float fallback[3]) {
                const CapturingBuilder::MaterialInfo::Value *v =
                    light.find(key);
                if (v == nullptr) {
                    return pbrt::Point3f(fallback[0], fallback[1], fallback[2]);
                }
                if (v->type != "point3" || v->floats.size() != 3) {
                    fail("a `" + light.name + "` light's `" + key +
                         "` has to be one point3");
                }
                return pbrt::Point3f(v->floats[0], v->floats[1], v->floats[2]);
            };
            float origin[3] = {0.f, 0.f, 0.f};
            float ahead[3] = {0.f, 0.f, 1.f};
            const pbrt::Point3f from = point("from", origin);
            const pbrt::Transform render_from_light =
                scene.GetCamera().GetCameraTransform().RenderFromWorld() *
                light.ctm;
            pbrt::Transform final_render_from_light =
                render_from_light *
                pbrt::Translate(pbrt::Vector3f(from.x, from.y, from.z));
            pbrt::Float cone_angle = 30.f;
            pbrt::Float cone_delta = 5.f;
            if (spot) {
                cone_angle = material_float(light, "coneangle", 30.f);
                cone_delta = material_float(light, "conedeltaangle", 5.f);
                const pbrt::Point3f to = point("to", ahead);
                const pbrt::Transform dir_to_z =
                    pbrt::Transform(pbrt::Frame::FromZ(pbrt::Normalize(to - from)));
                final_render_from_light =
                    render_from_light *
                    pbrt::Translate(pbrt::Vector3f(from.x, from.y, from.z)) *
                    pbrt::Inverse(dir_to_z);
            }

            pbrt::Float scale = material_float(light, "scale", 1.f);
            pbrt::Spectrum intensity;
            if (read_blackbody(light, "I", &bulb.blackbody, &bulb.temperature,
                               &bulb.blackbody_normalization)) {
                intensity = positional_alloc.new_object<pbrt::BlackbodySpectrum>(
                    bulb.temperature);
            } else if (material_rgb(light, "I", bulb.l)) {
                bulb.has_l = 1u;
                intensity =
                    positional_alloc.new_object<pbrt::RGBIlluminantSpectrum>(
                        *pbrt::RGBColorSpace::sRGB,
                        pbrt::RGB(bulb.l[0], bulb.l[1], bulb.l[2]));
            } else {
                bulb.has_l = 0u;
                intensity = &pbrt::RGBColorSpace::sRGB->illuminant;
            }
            scale /= pbrt::SpectrumToPhotometric(intensity);
            const pbrt::Float power = material_float(light, "power", -1.f);
            if (power > 0) {
                if (spot) {
                    const pbrt::Float cos_end = std::cos(pbrt::Radians(cone_angle));
                    const pbrt::Float cos_start =
                        std::cos(pbrt::Radians(cone_angle - cone_delta));
                    const pbrt::Float k_e =
                        2 * pbrt::Pi *
                        ((1 - cos_start) + (cos_start - cos_end) / 2);
                    scale *= power / k_e;
                } else {
                    scale *= power / (4 * pbrt::Pi);
                }
            }
            bulb.scale = float(scale);
            const pbrt::Point3f p = final_render_from_light(pbrt::Point3f(0, 0, 0));
            bulb.position[0] = float(p.x);
            bulb.position[1] = float(p.y);
            bulb.position[2] = float(p.z);
            if (spot) {
                // PBRT: SpotLight::SpotLight's two cosines, of `totalWidth`
                // and `falloffStart` -- `coneangle` and `coneangle -
                // conedeltaangle` as Create hands them over.
                bulb.cos_falloff_end = float(std::cos(pbrt::Radians(cone_angle)));
                bulb.cos_falloff_start =
                    float(std::cos(pbrt::Radians(cone_angle - cone_delta)));
                const pbrt::SquareMatrix<4> inverse =
                    pbrt::Inverse(final_render_from_light).GetMatrix();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        bulb.light_from_render[i * 4 + j] = float(inverse[i][j]);
                    }
                }
                positional_lights.push_back(
                    pbrt::Light(positional_alloc.new_object<pbrt::SpotLight>(
                        final_render_from_light, pbrt::MediumInterface{},
                        intensity, scale, cone_angle, cone_angle - cone_delta)));
            } else {
                positional_lights.push_back(
                    pbrt::Light(positional_alloc.new_object<pbrt::PointLight>(
                        final_render_from_light, pbrt::MediumInterface{},
                        intensity, scale)));
            }
            out.point_lights.push_back(bulb);
            continue;
        }
        if (light.name == "distant") {
            // PBRT: DistantLight::Create. The direction is `from - to` under
            // the light's transform -- PBRT builds a frame whose z axis is
            // that direction and multiplies `renderFromLight` by it, and
            // `SampleLi` then reads back `renderFromLight(0, 0, 1)`, which is
            // the same vector -- and the scale is divided by the photometric
            // integral of L, then multiplied by an `illuminance` when one is
            // given, exactly as PBRT does it. L defaults to the colour
            // space's illuminant, as the uniform infinite light's does.
            if (!light.ctm_is_tracked) {
                fail("a `distant` light placed by a named coordinate system "
                     "is not supported: scene_dump follows the transform stack "
                     "itself and does not follow that directive");
            }
            bonsai_scene::InfiniteLight sun;
            sun.distant = 1u;
            const auto point = [&](const char *key, float fallback[3]) {
                const CapturingBuilder::MaterialInfo::Value *v =
                    light.find(key);
                if (v == nullptr) {
                    return pbrt::Point3f(fallback[0], fallback[1], fallback[2]);
                }
                if (v->type != "point3" || v->floats.size() != 3) {
                    fail(std::string("a `distant` light's `") + key +
                         "` has to be one point3");
                }
                return pbrt::Point3f(v->floats[0], v->floats[1], v->floats[2]);
            };
            float origin[3] = {0.f, 0.f, 0.f};
            float ahead[3] = {0.f, 0.f, 1.f};
            const pbrt::Point3f from = point("from", origin);
            const pbrt::Point3f to = point("to", ahead);
            const pbrt::Transform render_from_light =
                scene.GetCamera().GetCameraTransform().RenderFromWorld() *
                light.ctm;
            const pbrt::Vector3f w =
                pbrt::Normalize(render_from_light(pbrt::Normalize(from - to)));
            sun.direction[0] = float(w.x);
            sun.direction[1] = float(w.y);
            sun.direction[2] = float(w.z);

            float scale = material_float(light, "scale", 1.f);
            pbrt::Allocator alloc;
            if (read_blackbody(light, "L", &sun.blackbody, &sun.temperature,
                               &sun.blackbody_normalization)) {
                const pbrt::BlackbodySpectrum emitted(sun.temperature);
                scale /= float(pbrt::SpectrumToPhotometric(&emitted));
            } else if (material_rgb(light, "L", sun.l)) {
                sun.has_l = 1u;
                const pbrt::RGBIlluminantSpectrum emitted(
                    *pbrt::RGBColorSpace::sRGB,
                    pbrt::RGB(sun.l[0], sun.l[1], sun.l[2]));
                scale /= float(pbrt::SpectrumToPhotometric(&emitted));
            } else {
                sun.has_l = 0u;
                scale /= float(pbrt::SpectrumToPhotometric(
                    &pbrt::RGBColorSpace::sRGB->illuminant));
            }
            const float illuminance = material_float(light, "illuminance", -1.f);
            if (illuminance > 0.f) {
                scale *= illuminance;
            }
            sun.scale = scale;
            out.infinite_lights.push_back(sun);
            continue;
        }
        if (light.name != "infinite") {
            fail("this renderer has area lights, uniform and image infinite "
                 "lights, distant lights and point and spot lights, and the "
                 "scene declares a `" + light.name + "` light");
        }
        const CapturingBuilder::MaterialInfo::Value *filename =
            light.find("filename");
        if (filename == nullptr) {
            filename = light.find("mapname");
        }
        if (light.find("illuminance") != nullptr) {
            fail("an `infinite` light with an `illuminance` is not supported");
        }
        // PBRT: `portal`, four points in world space, which Light::Create
        // moves to render space with the camera transform -- not with the
        // light's own, which reaches the light only through its image.
        const CapturingBuilder::MaterialInfo::Value *portal_param =
            light.find("portal");
        std::vector<pbrt::Point3f> portal;
        if (portal_param != nullptr) {
            if (portal_param->floats.size() % 3 != 0) {
                fail("an `infinite` light's portal is a list of points");
            }
            const pbrt::Transform render_from_world =
                scene.GetCamera().GetCameraTransform().RenderFromWorld();
            for (size_t i = 0; i + 2 < portal_param->floats.size(); i += 3) {
                portal.push_back(render_from_world(pbrt::Point3f(
                    portal_param->floats[i], portal_param->floats[i + 1],
                    portal_param->floats[i + 2])));
            }
        }

        bonsai_scene::InfiniteLight out_light;
        // PBRT: `scale /= SpectrumToPhotometric(...)`, over whichever spectrum
        // it ends up emitting -- the colour space's illuminant when the scene
        // wrote no L, and the RGBIlluminantSpectrum built from it when it did.
        // Folded in here for the same reason the area light's is: it is a
        // property of the scene rather than of the conversion.
        pbrt::Allocator alloc;
        const float scale = material_float(light, "scale", 1.f);
        if (portal_param != nullptr && filename == nullptr) {
            // PBRT: an `L` with a portal is the third branch of Light::Create
            // too -- the spectrum becomes a one-texel sRGB map (`SpectrumToXYZ`,
            // then sRGB's `ToRGB`), and the portal light is built over that.
            if (!light.ctm_is_tracked) {
                fail("an `infinite` light placed by a named coordinate system "
                     "is not supported: scene_dump follows the transform stack "
                     "itself and does not follow that directive");
            }
            pbrt::RGB colour;
            uint32_t blackbody = 0;
            float temperature = 0.f, normalization = 1.f;
            float rgb[3];
            if (read_blackbody(light, "L", &blackbody, &temperature,
                               &normalization) &&
                blackbody != 0) {
                pbrt::BlackbodySpectrum body(temperature);
                colour = pbrt::RGBColorSpace::sRGB->ToRGB(
                    pbrt::SpectrumToXYZ(&body));
            } else if (material_rgb(light, "L", rgb)) {
                pbrt::RGBIlluminantSpectrum given(
                    *pbrt::RGBColorSpace::sRGB, pbrt::RGB(rgb[0], rgb[1], rgb[2]));
                colour = pbrt::RGBColorSpace::sRGB->ToRGB(
                    pbrt::SpectrumToXYZ(&given));
            } else {
                fail("an `infinite` light with a portal needs an `L` or a "
                     "`filename`");
            }
            pbrt::Image one(pbrt::PixelFormat::Float, {1, 1}, {"R", "G", "B"},
                            nullptr, alloc);
            for (int c = 0; c < 3; c++) {
                one.SetChannel({0, 0}, c, colour[c]);
            }
            out_light.scale =
                float(scale / pbrt::SpectrumToPhotometric(
                                  &pbrt::RGBColorSpace::sRGB->illuminant));
            emit_portal_light(one, table_of(pbrt::RGBColorSpace::sRGB, "a uniform portal light"),
                              scene.GetCamera().GetCameraTransform().RenderFromWorld() *
                                  light.ctm,
                              portal, out_light, out);
            continue;
        }
        if (filename != nullptr) {
            if (filename->strings.empty()) {
                fail("an `infinite` light's filename is not a string");
            }
            if (light.find("L") != nullptr) {
                // PBRT refuses this pairing too.
                fail("an `infinite` light cannot have both an `L` and a "
                     "`filename`");
            }
            if (!light.ctm_is_tracked) {
                fail("an `infinite` light placed by a named coordinate system "
                     "is not supported: scene_dump follows the transform stack "
                     "itself and does not follow that directive");
            }

            const std::string resolved =
                pbrt::ResolveFilename(filename->strings[0]);
            pbrt::ImageAndMetadata im = pbrt::Image::Read(resolved, alloc);
            if (im.image.HasAnyInfinitePixels() || im.image.HasAnyNaNPixels()) {
                fail(resolved + ": an environment map with infinite or "
                                "not-a-number pixels is refused, as PBRT "
                                "refuses it");
            }
            const pbrt::Point2i res = im.image.Resolution();
            // PBRT's own check, and the reason it exists is worth keeping: a
            // non-square image is almost certainly a latitude-longitude
            // environment map, and reading one as an equal-area octahedral map
            // produces a sky that is wrong everywhere and looks like a sky.
            if (res.x != res.y) {
                fail(resolved + ": an environment map must be square to be an "
                                "equal-area octahedral one, and this is " +
                     std::to_string(res.x) + "x" + std::to_string(res.y));
            }
            // The colour space the texels are fitted against -- the image's
            // own, as PBRT's RGBIlluminantSpectrum is built against it: its
            // RGB-to-spectrum table for the fit, its illuminant for the
            // renderer to multiply the fit by (`illuminant` on the light; the
            // bistro, sportscar and villa skies are ACES2065-1, whose
            // illuminant is D60 rather than D65). A space whose table is not
            // linked is refused by name.
            const pbrt::RGBColorSpace *image_space = im.metadata.GetColorSpace();
            const SpectrumTable table = table_of(image_space, resolved);
            out_light.illuminant = illuminant_index(image_space, out);
            const pbrt::ImageChannelDesc desc =
                im.image.GetChannelDesc({"R", "G", "B"});
            if (!desc) {
                fail(resolved + ": an environment map needs R, G and B "
                                "channels");
            }
            if (portal_param != nullptr) {
                // PBRT: PortalImageInfiniteLight over `image.SelectChannels(
                // channelDesc)`, the scale divided as for the image light.
                out_light.scale =
                    float(scale / pbrt::SpectrumToPhotometric(
                                      &image_space->illuminant));
                emit_portal_light(
                    im.image.SelectChannels(desc, alloc), table,
                    scene.GetCamera().GetCameraTransform().RenderFromWorld() *
                        light.ctm,
                    portal, out_light, out);
                continue;
            }

            out_light.resolution = uint32_t(res.x);
            out_light.first_texel = uint32_t(out.env_sampling.size());
            // PBRT: the RGBIlluminantSpectrum an `ImageLe` builds per lookup --
            //
            //     Float m = max(rgb.r, rgb.g, rgb.b);
            //     scale = 2 * m;
            //     rsp = cs.ToRGBCoeffs(scale ? rgb / scale : RGB(0, 0, 0));
            //
            // done once here, since it is a deterministic function of the
            // texel. Four floats out: the three sigmoid coefficients and the
            // scale.
            //
            // `ToRGBCoeffs` is a trilinear lookup in PBRT's own table, and that
            // is the point of doing it on this side. The driver used to fit
            // each texel with the Gauss-Newton solve in rgb2spec.h, which is a
            // *different function* -- close, but not the one PBRT evaluates --
            // and which took four minutes and thirty-eight seconds on this
            // scene's sky against a render of one second. It looked exactly
            // like a hang.
            out.env_texels.reserve(out.env_texels.size() +
                                   size_t(res.x) * res.y * 4);
            out.env_sampling.reserve(out.env_sampling.size() +
                                     size_t(res.x) * res.y);
            for (int y = 0; y < res.y; y++) {
                for (int x = 0; x < res.x; x++) {
                    const pbrt::ImageChannelValues v =
                        im.image.GetChannels({x, y}, desc);
                    // PBRT: `Image::GetSamplingDistribution`, whose value at a
                    // texel is `GetChannels({x, y}).Average()` -- every
                    // channel, and not clamped. The fit below cannot stand in
                    // for it: it is a *different function of the texel*, and
                    // where the texel is black it is minus infinity.
                    out.env_sampling.push_back(
                        float(im.image.GetChannels({x, y}).Average()));
                    // PBRT: ClampZero, applied where it applies it -- inside
                    // ImageLe, before the spectrum is built. A negative texel
                    // is not a colour and the fit has nothing to say about one.
                    const float r = std::max(0.f, float(v[0]));
                    const float g = std::max(0.f, float(v[1]));
                    const float b = std::max(0.f, float(v[2]));
                    const float m = std::max({r, g, b});
                    const float texel_scale = 2 * m;
                    const Sigmoid rsp =
                        texel_scale != 0.f
                            ? fit_sigmoid(table, r / texel_scale, g / texel_scale,
                                          b / texel_scale)
                            : fit_sigmoid(table, 0.f, 0.f, 0.f);
                    out.env_texels.push_back(rsp.c0);
                    out.env_texels.push_back(rsp.c1);
                    out.env_texels.push_back(rsp.c2);
                    out.env_texels.push_back(texel_scale);
                }
            }
            // PBRT: `scale /= SpectrumToPhotometric(&colorSpace->illuminant)`
            // for the image case -- over the image's colour space's illuminant
            // and not over the image, which is the same division the no-L
            // uniform case makes.
            out_light.scale =
                float(scale / pbrt::SpectrumToPhotometric(&image_space->illuminant));
            // PBRT: `renderFromLight`, inverted here because ApplyInverse is
            // the only use it has. `renderFromWorld * ctm` is what
            // BasicSceneBuilder would have built.
            const pbrt::Transform render_from_light =
                scene.GetCamera().GetCameraTransform().RenderFromWorld() *
                light.ctm;
            const pbrt::Transform light_from_render =
                pbrt::Inverse(render_from_light);
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    out_light.light_from_render[4 * r + c] =
                        float(light_from_render.GetMatrix()[r][c]);
                    out_light.render_from_light[4 * r + c] =
                        float(render_from_light.GetMatrix()[r][c]);
                }
            }
            out.infinite_lights.push_back(out_light);
            continue;
        }
        if (read_blackbody(light, "L", &out_light.blackbody,
                           &out_light.temperature,
                           &out_light.blackbody_normalization)) {
            // PBRT: UniformInfiniteLight::Create with a `blackbody L`, the
            // scale divided by the photometric integral of that blackbody.
            const pbrt::BlackbodySpectrum emitted(out_light.temperature);
            out_light.scale =
                float(scale / pbrt::SpectrumToPhotometric(&emitted));
        } else if (material_rgb(light, "L", out_light.l)) {
            out_light.has_l = 1u;
            const pbrt::RGBIlluminantSpectrum emitted(
                *pbrt::RGBColorSpace::sRGB,
                pbrt::RGB(out_light.l[0], out_light.l[1], out_light.l[2]));
            out_light.scale =
                float(scale / pbrt::SpectrumToPhotometric(&emitted));
        } else {
            out_light.has_l = 0u;
            out_light.scale =
                float(scale / pbrt::SpectrumToPhotometric(
                                  &pbrt::RGBColorSpace::sRGB->illuminant));
        }
        out.infinite_lights.push_back(out_light);
    }

    // The reconstruction filter. Only the Gaussian, which is PBRT's default and
    // what every scene here gets; the others differ in one function and a
    // radius, and go in as arms of a variant when one is asked for. The
    // defaults are GaussianFilter::Create's own.
    if (builder.filter_name != "gaussian") {
        fail("only the `gaussian` reconstruction filter is supported, scene "
             "asks for \"" + builder.filter_name + "\"");
    }
    out.filter_radius[0] = float(builder.filter_params.GetOneFloat("xradius",
                                                                   1.5f));
    out.filter_radius[1] = float(builder.filter_params.GetOneFloat("yradius",
                                                                   1.5f));
    out.filter_sigma = float(builder.filter_params.GetOneFloat("sigma", 0.5f));
    // Read from PBRT rather than written down, so that scene_dump's flag and
    // the reference render below cannot mean different things by it.
    out.disable_pixel_jitter = pbrt::Options->disablePixelJitter ? 1u : 0u;
    // PBRT: Film::UsesVisibleSurface(), which only GBufferFilm answers yes to.
    out.film_visible_surface = builder.film_type == "gbuffer" ? 1u : 0u;

    // The seven samplers pbrt has, every one of them reproduced.
    //
    // The defaults below are pbrt's own, from each sampler's Create; a scene
    // that names a sampler without naming its parameters has to get the same
    // ones pbrt would have given it. `zsobol` is pbrt's default sampler, the
    // one `sampler_name` starts as, so it is what a scene naming none gets.
    //
    // `--spp` overrides them, and each sampler is overridden the way its own
    // Create overrides it. That is not one rule: an independent, halton,
    // zsobol, sobol, paddedsobol or pmj02bn sampler takes the number as given
    // -- zsobol's constructor then floors it to a power of two, which the
    // driver does where pbrt does, and the two Sobol' ones and pmj02bn warn
    // about a count that is not a power of two or four and render it -- and
    // a stratified one has to factor it into a grid, because what it samples
    // is a grid. PBRT's own reference render below reads
    // `Options->pixelSamples` directly through `scene.GetSampler()`, so the
    // two sides cannot drift on what the flag means -- but this side does have
    // to transcribe the same rules.
    //
    // pbrt: RandomizeStrategy from the `randomization` string, with each
    // sampler's own default -- `permutedigits` for Halton, `fastowen` for the
    // three Sobol' samplers -- and Halton refusing `fastowen`, as pbrt's does.
    const auto randomization_of = [&](const char *fallback,
                                      bool fast_owen_allowed) {
        const std::string randomization =
            builder.sampler_params.GetOneString("randomization", fallback);
        if (randomization == "none") {
            return bonsai_scene::RandomizeTag::RandomizeNone;
        } else if (randomization == "permutedigits") {
            return bonsai_scene::RandomizeTag::RandomizePermuteDigits;
        } else if (randomization == "fastowen" && fast_owen_allowed) {
            return bonsai_scene::RandomizeTag::RandomizeFastOwen;
        } else if (randomization == "owen") {
            return bonsai_scene::RandomizeTag::RandomizeOwen;
        }
        fail("unknown " + builder.sampler_name + " randomization \"" +
             randomization + "\"");
        return bonsai_scene::RandomizeTag::RandomizeNone;
    };
    const pstd::optional<int> spp_override = pbrt::Options->pixelSamples;
    if (builder.sampler_name == "independent") {
        out.sampler.tag = bonsai_scene::SamplerTag::Independent;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 4));
    } else if (builder.sampler_name == "halton") {
        out.sampler.tag = bonsai_scene::SamplerTag::Halton;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 16));
        out.sampler.randomize = randomization_of("permutedigits", false);
        halton_scales(x_resolution, y_resolution, out.sampler);
    } else if (builder.sampler_name == "stratified") {
        out.sampler.tag = bonsai_scene::SamplerTag::Stratified;
        int x_samples = builder.sampler_params.GetOneInt("xsamples", 4);
        int y_samples = builder.sampler_params.GetOneInt("ysamples", 4);
        if (spp_override) {
            // PBRT's own factoring, from StratifiedSampler::Create. It walks
            // down from the square root until it finds a divisor, so `--spp 12`
            // is a 4x3 grid and `--spp 13` -- a prime -- is 13x1. Not the
            // rounding a reader would guess, and the grid is what decides which
            // stratum each sample falls in, so guessing would give a different
            // image rather than a differently-sized one.
            const int n = *spp_override;
            int div = int(std::sqrt(double(n)));
            while (n % div) {
                div--;
            }
            x_samples = n / div;
            y_samples = n / x_samples;
        }
        out.sampler.x_samples = uint32_t(x_samples);
        out.sampler.y_samples = uint32_t(y_samples);
        out.sampler.jitter =
            builder.sampler_params.GetOneBool("jitter", true) ? 1u : 0u;
        out.sampler.samples_per_pixel =
            out.sampler.x_samples * out.sampler.y_samples;
    } else if (builder.sampler_name == "zsobol") {
        out.sampler.tag = bonsai_scene::SamplerTag::ZSobol;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 16));
        out.sampler.randomize = randomization_of("fastowen", true);
        // PBRT: ZSobolSampler's constructor, the part of it that depends on
        // the film and not on the sample count -- which `--spp` may still
        // change at the driver, where the rest is derived. The film's full
        // resolution, as PBRT hands its sampler (scene.cpp), through PBRT's
        // own arithmetic.
        out.sampler.log2_resolution = pbrt::Log2Int(
            pbrt::RoundUpPow2(std::max(x_resolution, y_resolution)));
    } else if (builder.sampler_name == "sobol") {
        out.sampler.tag = bonsai_scene::SamplerTag::Sobol;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 16));
        out.sampler.randomize = randomization_of("fastowen", true);
        // PBRT: SobolSampler's constructor, `scale = RoundUpPow2(max(res))`
        // -- the power-of-two side the sequence's first two dimensions are
        // mapped onto -- kept as its log2, which is what
        // SobolIntervalToIndex takes.
        out.sampler.log2_resolution = pbrt::Log2Int(
            pbrt::RoundUpPow2(std::max(x_resolution, y_resolution)));
        // The whole of pbrt's generator matrices -- this sampler walks the
        // dimensions, where zsobol and paddedsobol read only the first two --
        // and the two van der Corput matrices SobolIntervalToIndex works
        // with, copied from pbrt's own arrays into the `.smp` sidecar
        // (scene_io.h). Twenty-five rows of the latter, m = 1 to 25, which
        // is as far as pbrt's table goes; a film side of 2^25 would be
        // beyond it.
        out.sobol_matrices.assign(
            pbrt::SobolMatrices32,
            pbrt::SobolMatrices32 + pbrt::NSobolDimensions * pbrt::SobolMatrixSize);
        for (int m = 0; m < 25; m++) {
            for (int c = 0; c < pbrt::SobolMatrixSize; c++) {
                out.vdc_matrices.push_back(pbrt::VdCSobolMatrices[m][c]);
                out.vdc_matrices_inv.push_back(pbrt::VdCSobolMatricesInv[m][c]);
            }
        }
    } else if (builder.sampler_name == "paddedsobol") {
        out.sampler.tag = bonsai_scene::SamplerTag::PaddedSobol;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 16));
        out.sampler.randomize = randomization_of("fastowen", true);
    } else if (builder.sampler_name == "pmj02bn") {
        out.sampler.tag = bonsai_scene::SamplerTag::PMJ02BN;
        out.sampler.samples_per_pixel = uint32_t(
            spp_override ? *spp_override
                         : builder.sampler_params.GetOneInt("pixelsamples", 16));
        // PBRT: the sampler refuses a count past its table.
        if (out.sampler.samples_per_pixel > uint32_t(pbrt::nPMJ02bnSamples)) {
            fail("PMJ02BNSampler only supports up to " +
                 std::to_string(pbrt::nPMJ02bnSamples) + " samples per pixel");
        }
        // pbrt's five blue-noise point sets and forty-eight blue-noise
        // textures, copied from its own arrays into the `.smp` sidecar. The
        // sorting of the first set into pixels is the driver's: it depends
        // on the sample count, which `--spp` may change.
        out.pmj02bn_samples.assign(
            &pbrt::pmj02bnSamples[0][0][0],
            &pbrt::pmj02bnSamples[0][0][0] +
                size_t(pbrt::nPMJ02bnSets) * pbrt::nPMJ02bnSamples * 2);
        out.blue_noise.assign(&pbrt::BlueNoiseTextures[0][0][0],
                              &pbrt::BlueNoiseTextures[0][0][0] +
                                  size_t(pbrt::NumBlueNoiseTextures) *
                                      pbrt::BlueNoiseResolution *
                                      pbrt::BlueNoiseResolution);
    } else {
        fail("unknown sampler \"" + builder.sampler_name +
             "\": pbrt has independent, stratified, halton, zsobol, sobol, "
             "paddedsobol and pmj02bn");
    }
    out.sampler.seed = builder.sampler_params.GetOneInt("seed", 0);
    // Not the sampler's seed: PBRT's global `--seed`, which a layered BSDF
    // hashes with the direction it was asked about. Read from PBRT rather than
    // written down, so that running scene_dump with one has the effect it has
    // on PBRT.
    out.seed = pbrt::GetOptions().seed;
    // The same depth the reference render uses, so that the two integrators are
    // asked to go equally far -- unless `--maxdepth` asked for another.
    out.max_depth = g_max_depth_override > 0 ? g_max_depth_override
                                             : builder.integrator_max_depth;
    // Which integrator, refused rather than substituted. A scene that names
    // `volpath` and gets a random walk is an image that answers a question
    // nobody asked, and it would look plausible -- which is worse than an
    // error.
    //
    // A scene naming none is the one exception, and it now falls back to `path`
    // rather than to the random walk. PBRT's default is `volpath`, which is
    // `path` plus participating media; a scene with no media in it -- and one
    // that named no integrator has named no media either -- is asking `path`
    // for an answer volpath would give the same way. The comparison renders
    // both sides with whatever this resolves to, so the two cannot disagree
    // about it.
    //
    // The random walk was the fallback while it was the only integrator here,
    // and the difference is not academic: killeroo-simple names no integrator
    // and is lit by a sphere of radius 3 seen from four hundred units away, so
    // a walk that finds a light only by scattering into one lit 5,038 of its
    // 490,000 pixels. The same scene through `path` is a photograph.
    //
    // With `volpath` here, a scene that names no integrator gets PBRT's own
    // default, as PBRT gives it (`pbrt::BasicScene`: `"volpath"` when the
    // directive is absent); the comparison tools render pbrt's side with the
    // same rewrite. The paragraph above is kept as the record of why the
    // random walk stopped being the fallback.
    if (builder.integrator_name.empty()) {
        out.integrator = bonsai_scene::IntegratorTag::VolPath;
    } else if (builder.integrator_name == "randomwalk") {
        out.integrator = bonsai_scene::IntegratorTag::RandomWalk;
    } else if (builder.integrator_name == "simplepath") {
        out.integrator = bonsai_scene::IntegratorTag::SimplePath;
    } else if (builder.integrator_name == "path") {
        out.integrator = bonsai_scene::IntegratorTag::Path;
    } else if (builder.integrator_name == "volpath") {
        out.integrator = bonsai_scene::IntegratorTag::VolPath;
    } else {
        fail("this renderer implements `randomwalk`, `simplepath`, `path` and "
             "`volpath`, and the scene asks for `" + builder.integrator_name +
             "`");
    }
    out.regularize = builder.integrator_regularize ? 1u : 0u;

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
    // And the same transform the other way, which `Approximate_dp_dxy` needs:
    // it takes a hit back to camera space, builds a tangent plane there, and
    // brings the answer forward again.
    write_matrix(matrices,
                 scene.GetCamera().GetCameraTransform().CameraFromRender(0.f));

    // PBRT: ProjectiveCamera's lensRadius and focalDistance, from
    // PerspectiveCamera::Create. Zero radius is a pinhole and is what almost
    // every test scene is; a real one usually has a lens.
    out.lens_radius = builder.camera_params.GetOneFloat("lensradius", 0.f);
    out.focal_distance = builder.camera_params.GetOneFloat("focaldistance", 1e6f);

    // PBRT: PixelSensor::Create's `imagingRatio = exposureTime * ISO / 100`.
    // The shutter defaults to one and this renderer has no moving geometry to
    // open it over, so a scene that sets `shutteropen`/`shutterclose` is
    // refused rather than silently exposed for a different length of time.
    {
        const float shutter_open =
            builder.camera_params.GetOneFloat("shutteropen", 0.f);
        const float shutter_close =
            builder.camera_params.GetOneFloat("shutterclose", 1.f);
        if (shutter_open != 0.f || shutter_close != 1.f) {
            fail("a shutter other than [0, 1] changes the exposure, and this "
                 "renderer has nothing that moves during it");
        }
        const float iso = builder.film_params.GetOneFloat("iso", 100.f);
        out.imaging_ratio = (shutter_close - shutter_open) * iso / 100.f;
        // PBRT: RGBFilm's own default is Infinity, i.e. no clamping.
        out.max_component_value = builder.film_params.GetOneFloat(
            "maxcomponentvalue", std::numeric_limits<float>::infinity());

        // The pixel sensor, built by PBRT's own PixelSensor::Create from the
        // film's parameters -- the sensor name, the ISO, any whitebalance -- so
        // the response curves and the sensor-to-XYZ matrix are PBRT's, whether
        // the sensor is the default cie1931 or a named camera like
        // canon_eos_100d. What is serialized is that result: the three response
        // curves resampled to the renderer's 360..830 nm grid, and the matrix
        // RGBFilm applies once per pixel. A named sensor was refused before
        // this, which is what kept the pbrt-book scenes out.
        pbrt::Allocator sensor_alloc;
        const pbrt::PixelSensor *sensor = pbrt::PixelSensor::Create(
            builder.film_params, pbrt::RGBColorSpace::sRGB,
            shutter_close - shutter_open, nullptr, sensor_alloc);
        // The sensor's own response curves. cie1931 uses the CIE matching
        // functions; a named sensor names its three as spectra PBRT interns
        // under `<sensor>_r/_g/_b`. This is the source PBRT's own densely
        // sampled sensor curves are built from, read on the same integer-nm grid
        // the renderer rounds a wavelength to, so a lookup there matches PBRT's.
        const std::string sensor_name =
            builder.film_params.GetOneString("sensor", "cie1931");
        pbrt::Spectrum r_bar, g_bar, b_bar;
        if (sensor_name == "cie1931") {
            r_bar = &pbrt::Spectra::X();
            g_bar = &pbrt::Spectra::Y();
            b_bar = &pbrt::Spectra::Z();
        } else {
            r_bar = pbrt::GetNamedSpectrum(sensor_name + "_r");
            g_bar = pbrt::GetNamedSpectrum(sensor_name + "_g");
            b_bar = pbrt::GetNamedSpectrum(sensor_name + "_b");
            if (!r_bar || !g_bar || !b_bar) {
                fail("the `" + sensor_name +
                     "` sensor is not one PBRT has response curves for");
            }
        }
        out.sensor_r.resize(471);
        out.sensor_g.resize(471);
        out.sensor_b.resize(471);
        for (int i = 0; i < 471; i++) {
            const float lambda = float(360 + i);
            out.sensor_r[size_t(i)] = float(r_bar(lambda));
            out.sensor_g[size_t(i)] = float(g_bar(lambda));
            out.sensor_b[size_t(i)] = float(b_bar(lambda));
        }
        // RGBFilm's outputRGBFromSensorRGB: RGBFromXYZ * XYZFromSensorRGB, the
        // second read straight off the sensor PBRT built -- it is public -- so
        // the least-squares fit a named sensor needs is PBRT's, not a copy.
        const pbrt::SquareMatrix<3> m =
            pbrt::RGBColorSpace::sRGB->RGBFromXYZ * sensor->XYZFromSensorRGB;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                out.output_rgb_from_sensor[size_t(3 * r + c)] = float(m[r][c]);
            }
        }
    }

    // PBRT: PerspectiveCamera's constructor, which derives dxCamera and
    // dyCamera from the same cameraFromRaster written above.
    {
        const pbrt::Transform cfr =
            camera_from_raster(builder.camera_params, x_resolution,
                               y_resolution);
        const pbrt::Point3f p0 = cfr(pbrt::Point3f(0, 0, 0));
        const pbrt::Vector3f dx = cfr(pbrt::Point3f(1, 0, 0)) - p0;
        const pbrt::Vector3f dy = cfr(pbrt::Point3f(0, 1, 0)) - p0;
        out.d_camera[0] = float(dx.x);
        out.d_camera[1] = float(dx.y);
        out.d_camera[2] = float(dx.z);
        out.d_camera[3] = float(dy.x);
        out.d_camera[4] = float(dy.y);
        out.d_camera[5] = float(dy.z);
    }

    // PBRT: CameraBase::FindMinimumDifferentials, run against PBRT's own
    // camera rather than reimplemented.
    //
    // The four vectors it produces are members of CameraBase and protected, so
    // they cannot simply be read off the camera -- but every input to the loop
    // is public, so the loop itself is what is repeated here. It walks 512
    // samples along the film's diagonal, generates a differential ray at each,
    // and keeps the shortest positional and directional offset it saw. That is
    // a property of the camera and the resolution alone, so it belongs on this
    // side with the spectral fits and the BVH.
    {
        const pbrt::Camera camera = scene.GetCamera();
        const pbrt::CameraTransform &ct = camera.GetCameraTransform();
        pbrt::Vector3f min_pos_x(pbrt::Infinity, pbrt::Infinity, pbrt::Infinity);
        pbrt::Vector3f min_pos_y = min_pos_x;
        pbrt::Vector3f min_dir_x = min_pos_x;
        pbrt::Vector3f min_dir_y = min_pos_x;

        pbrt::CameraSample sample;
        sample.pLens = pbrt::Point2f(0.5f, 0.5f);
        sample.time = 0.5f;
        pbrt::SampledWavelengths lambda =
            pbrt::SampledWavelengths::SampleVisible(0.5f);

        const int n = 512;
        for (int i = 0; i < n; ++i) {
            sample.pFilm.x = pbrt::Float(i) / (n - 1) * x_resolution;
            sample.pFilm.y = pbrt::Float(i) / (n - 1) * y_resolution;

            pstd::optional<pbrt::CameraRayDifferential> crd =
                camera.GenerateRayDifferential(sample, lambda);
            if (!crd) {
                continue;
            }
            pbrt::RayDifferential &ray = crd->ray;

            const pbrt::Vector3f dox =
                ct.CameraFromRender(ray.time)(ray.rxOrigin - ray.o);
            if (pbrt::Length(dox) < pbrt::Length(min_pos_x)) {
                min_pos_x = dox;
            }
            const pbrt::Vector3f doy =
                ct.CameraFromRender(ray.time)(ray.ryOrigin - ray.o);
            if (pbrt::Length(doy) < pbrt::Length(min_pos_y)) {
                min_pos_y = doy;
            }

            ray.d = pbrt::Normalize(ray.d);
            ray.rxDirection = pbrt::Normalize(ray.rxDirection);
            ray.ryDirection = pbrt::Normalize(ray.ryDirection);

            const pbrt::Frame f = pbrt::Frame::FromZ(ray.d);
            const pbrt::Vector3f df = f.ToLocal(ray.d);
            const pbrt::Vector3f dxf = pbrt::Normalize(f.ToLocal(ray.rxDirection));
            const pbrt::Vector3f dyf = pbrt::Normalize(f.ToLocal(ray.ryDirection));
            if (pbrt::Length(dxf - df) < pbrt::Length(min_dir_x)) {
                min_dir_x = dxf - df;
            }
            if (pbrt::Length(dyf - df) < pbrt::Length(min_dir_y)) {
                min_dir_y = dyf - df;
            }
        }

        const pbrt::Vector3f mins[4] = {min_pos_x, min_pos_y, min_dir_x,
                                        min_dir_y};
        for (int i = 0; i < 4; i++) {
            out.min_differentials[3 * i + 0] = float(mins[i].x);
            out.min_differentials[3 * i + 1] = float(mins[i].y);
            out.min_differentials[3 * i + 2] = float(mins[i].z);
        }
    }

    // `--print-differentials`: PBRT's own answers for the numbers a texture is
    // filtered by, printed so the renderer can be checked against them before
    // there is any texture to notice a difference in.
    //
    // A wrong footprint is invisible in an image. It does not move an edge or
    // change a colour, it only makes a texture slightly too blurry or too
    // sharp, which no pixel comparison against pbrt would fail on -- so this is
    // checked directly or not at all.
    //
    // The hits are synthetic rather than found by tracing: what is under test
    // is `ComputeDifferentials`, and giving it chosen inputs exercises both of
    // its branches, including the one a real first hit never takes. Everything
    // else here is PBRT's -- its camera, its ray differentials, its solve.
    if (g_print_differentials) {
        const pbrt::Camera camera = scene.GetCamera();
        pbrt::SampledWavelengths lambda =
            pbrt::SampledWavelengths::SampleVisible(0.5f);

        // A hit with a parameterization nothing about is round: dpdu and dpdv
        // are neither perpendicular nor the same length, which is what makes
        // the 2x2 solve do some work.
        const pbrt::Point3f hit_p(-1.25f, 0.75f, -3.5f);
        const pbrt::Vector3f dpdu(1.7f, 0.3f, -0.4f);
        const pbrt::Vector3f dpdv(-0.2f, 1.1f, 0.9f);
        const pbrt::Point2f hit_uv(0.375f, 0.625f);

        const int pixels[][2] = {{0, 0}, {17, 42}, {640, 360}, {1279, 719}};
        for (const auto &px : pixels) {
            pbrt::CameraSample sample;
            sample.pFilm = pbrt::Point2f(px[0] + 0.5f, px[1] + 0.5f);
            // Off centre, so a camera with a lens takes its lens branch in
            // earnest; the same point differentials_at uses.
            sample.pLens = pbrt::Point2f(0.9f, 0.3f);
            sample.time = 0.f;
            sample.filterWeight = 1.f;

            pstd::optional<pbrt::CameraRayDifferential> crd =
                camera.GenerateRayDifferential(sample, lambda);
            if (!crd) {
                continue;
            }
            // The ray itself, before the differentials are scaled -- scaling
            // leaves it alone.
            printf("camray %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g\n", px[0],
                   px[1], double(crd->ray.o.x), double(crd->ray.o.y),
                   double(crd->ray.o.z), double(crd->ray.d.x),
                   double(crd->ray.d.y), double(crd->ray.d.z));
            // The camera transform the other way, on a point and on a vector:
            // Transform::ApplyInverse, which this renderer does with a second
            // matrix and the forward code, so its fusion has to be checked
            // separately.
            {
                const pbrt::CameraTransform &ct = camera.GetCameraTransform();
                const pbrt::Point3f ip = ct.CameraFromRender(hit_p, 0.f);
                const pbrt::Vector3f iv = ct.CameraFromRender(crd->ray.d, 0.f);
                printf("invpoint %d %d: %.9g %.9g %.9g\n", px[0], px[1],
                       double(ip.x), double(ip.y), double(ip.z));
                printf("invvec %d %d: %.9g %.9g %.9g\n", px[0], px[1],
                       double(iv.x), double(iv.y), double(iv.z));
            }
            // RenderCPU scales before tracing; 16 samples per pixel, so the
            // scale is 1/4 and not the 0.125 floor.
            crd->ray.ScaleDifferentials(
                std::max<pbrt::Float>(.125f, 1 / std::sqrt((pbrt::Float)16)));
            const pbrt::RayDifferential &ray = crd->ray;
            printf("camdiff %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                   "%.9g %.9g %.9g | %.9g %.9g %.9g\n",
                   px[0], px[1], double(ray.rxOrigin.x), double(ray.rxOrigin.y),
                   double(ray.rxOrigin.z), double(ray.ryOrigin.x),
                   double(ray.ryOrigin.y), double(ray.ryOrigin.z),
                   double(ray.rxDirection.x), double(ray.rxDirection.y),
                   double(ray.rxDirection.z), double(ray.ryDirection.x),
                   double(ray.ryDirection.y), double(ray.ryDirection.z));

            // Both branches, at the same hit. `has` chooses whether the ray is
            // believed to carry differentials, which is the only difference
            // between "first hit from the camera" and "every hit after a
            // diffuse bounce".
            for (int has = 1; has >= 0; has--) {
                pbrt::SurfaceInteraction isect(
                    pbrt::Point3fi(hit_p), hit_uv, -ray.d, dpdu, dpdv,
                    pbrt::Normal3f(0, 0, 0), pbrt::Normal3f(0, 0, 0), 0.f,
                    false);
                pbrt::RayDifferential r = ray;
                r.hasDifferentials = has != 0;
                isect.ComputeDifferentials(r, camera, 16);
                printf("dudxy %d %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                       "%.9g %.9g %.9g %.9g\n",
                       px[0], px[1], has, double(isect.dpdx.x),
                       double(isect.dpdx.y), double(isect.dpdx.z),
                       double(isect.dpdy.x), double(isect.dpdy.y),
                       double(isect.dpdy.z), double(isect.dudx),
                       double(isect.dvdx), double(isect.dudy),
                       double(isect.dvdy));

                // And what the differentials become across a bounce. Only the
                // two exactly-specular lobes carry them; a rough one drops
                // them, and that `hasDifferentials` going to 0 is as much a
                // part of the answer as the vectors are.
                //
                // `bsdf` is declared by SpawnRay and never read by it, so a
                // default one is not a stand-in for anything.
                const pbrt::BSDF bsdf;
                const pbrt::Vector3f wi =
                    pbrt::Normalize(pbrt::Vector3f(0.31f, -0.82f, 0.48f));
                struct Lobe {
                    const char *label;
                    int flags;
                    float eta;
                };
                const Lobe lobes[] = {
                    {"reflect", int(pbrt::BxDFFlags::SpecularReflection), 1.f},
                    {"transmit", int(pbrt::BxDFFlags::SpecularTransmission),
                     1.5f},
                    {"rough", int(pbrt::BxDFFlags::GlossyReflection), 1.f}};
                for (const Lobe &lobe : lobes) {
                    const pbrt::RayDifferential sp =
                        isect.SpawnRay(r, bsdf, wi, lobe.flags, lobe.eta);
                    printf("spawn %d %d %d %s: %d | %.9g %.9g %.9g | "
                           "%.9g %.9g %.9g | %.9g %.9g %.9g | %.9g %.9g %.9g\n",
                           px[0], px[1], has, lobe.label,
                           int(sp.hasDifferentials), double(sp.rxOrigin.x),
                           double(sp.rxOrigin.y), double(sp.rxOrigin.z),
                           double(sp.ryOrigin.x), double(sp.ryOrigin.y),
                           double(sp.ryOrigin.z), double(sp.rxDirection.x),
                           double(sp.rxDirection.y), double(sp.rxDirection.z),
                           double(sp.ryDirection.x), double(sp.ryDirection.y),
                           double(sp.ryDirection.z));
                }
            }
        }
    }

    // A medium a shape's interface or the camera names, converted on first
    // use as the materials below are: PBRT's Medium::Create, by `type`.
    //
    // Every kind reads sigma_a and sigma_s the same way -- a `preset` for the
    // homogeneous medium alone, else the two spectra, each a constant one when
    // absent -- and `g`. The spectra are tabulated at the integer nanometres a
    // DenselySampledSpectrum holds, which is how PBRT's media keep them (see
    // Medium in scene_io.h): three runs, sigma_a and sigma_s scaled by `scale`
    // where the kind's constructor scales them, then Le -- scaled by `Lescale`
    // over its photometric integral for a homogeneous medium, as given for a
    // grid whose Lescale is a grid of its own, zeros for a cloud, which emits
    // nothing.
    //
    // A grid medium or a cloud is placed by PBRT's `renderFromMedium`:
    // render-from-world times the CTM at the MakeNamedMedium
    // (BasicSceneBuilder::RenderFromObject; the start transform of an animated
    // one). Both directions are written, the inverse being PBRT's own mInv, so
    // that the renderer's pull of a ray into the medium's space rounds as
    // PBRT's ApplyInverse does. The grids go into `out.medium_grid`, and the
    // majorant grid each constructor builds is built here with PBRT's own
    // SampledGrid, so that its maxima are PBRT's to the bit: a majorant is part
    // of the random walk and not only of its cost -- a different bound is a
    // different sequence of tentative collisions, and a different picture per
    // pixel.
    std::map<std::string, int32_t> medium_index;
    const auto medium_for = [&](const std::string &name) -> int32_t {
        if (name.empty()) {
            return -1;
        }
        const auto it = medium_index.find(name);
        if (it != medium_index.end()) {
            return it->second;
        }
        const CapturingBuilder::MediumInfo *info = nullptr;
        for (const CapturingBuilder::MediumInfo &m : builder.media) {
            if (m.name == name) {
                info = &m;
            }
        }
        if (info == nullptr) {
            fail("a shape or the camera names the medium \"" + name +
                 "\", which was never declared");
        }
        pbrt::Allocator alloc;
        const pbrt::ParameterDictionary &params = info->params;
        const std::string kind = params.GetOneString("type", "");
        const std::string what = "the " + kind + " medium \"" + name + "\"";

        // The two coefficient spectra of every kind but the rgbgrid, whose
        // sigma_a and sigma_s are arrays of RGB and are read below.
        pbrt::Spectrum sig_a = nullptr, sig_s = nullptr;
        if (kind != "rgbgrid") {
            const std::string preset = params.GetOneString("preset", "");
            if (kind == "homogeneous" && !preset.empty() &&
                !pbrt::GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc)) {
                fail(what + " names the preset \"" + preset +
                     "\", which PBRT does not have either");
            }
            if (!sig_a) {
                sig_a = params.GetOneSpectrum("sigma_a", nullptr,
                                              pbrt::SpectrumType::Unbounded, alloc);
                if (!sig_a) {
                    sig_a = alloc.new_object<pbrt::ConstantSpectrum>(1.f);
                }
            }
            if (!sig_s) {
                sig_s = params.GetOneSpectrum("sigma_s", nullptr,
                                              pbrt::SpectrumType::Unbounded, alloc);
                if (!sig_s) {
                    sig_s = alloc.new_object<pbrt::ConstantSpectrum>(1.f);
                }
            }
        }
        const pbrt::Float sigma_scale = params.GetOneFloat("scale", 1.f);

        bonsai_scene::Medium medium;
        medium.g = params.GetOneFloat("g", 0.f);

        // PBRT: DenselySampledSpectrum's constructor, `spec(lambda)` at each
        // integer wavelength from 360 to 830, then Scale. Returns the largest
        // value of Le's run, for `emissive`.
        const auto push_spectra = [&](pbrt::Spectrum a, pbrt::Spectrum s,
                                      pbrt::Spectrum e, pbrt::Float a_scale,
                                      pbrt::Float e_scale) -> float {
            medium.spectra = uint32_t(out.medium_spectra.size());
            float le_max = 0.f;
            for (const auto &[spec, scale] :
                 {std::pair<pbrt::Spectrum, pbrt::Float>{a, a_scale},
                  std::pair<pbrt::Spectrum, pbrt::Float>{s, a_scale},
                  std::pair<pbrt::Spectrum, pbrt::Float>{e, e_scale}}) {
                for (int lambda = 360;
                     lambda < 360 + bonsai_scene::kMediumSpectrumSamples; lambda++) {
                    out.medium_spectra.push_back(
                        float(spec(pbrt::Float(lambda)) * scale));
                }
            }
            for (int i = 0; i < bonsai_scene::kMediumSpectrumSamples; i++) {
                le_max = std::max(
                    le_max,
                    out.medium_spectra[medium.spectra +
                                       2 * bonsai_scene::kMediumSpectrumSamples +
                                       size_t(i)]);
            }
            return le_max;
        };
        // The box and the transform of a placed medium (every kind but the
        // homogeneous one).
        const auto place = [&]() {
            if (!info->ctm_is_tracked) {
                fail(what + " is placed by a named coordinate system, which is "
                            "not supported: scene_dump follows the transform "
                            "stack itself and does not follow that directive");
            }
            const pbrt::Point3f p0 =
                params.GetOnePoint3f("p0", pbrt::Point3f(0.f, 0.f, 0.f));
            const pbrt::Point3f p1 =
                params.GetOnePoint3f("p1", pbrt::Point3f(1.f, 1.f, 1.f));
            for (int c = 0; c < 3; c++) {
                medium.low[c] = float(p0[c]);
                medium.high[c] = float(p1[c]);
            }
            const pbrt::Transform render_from_medium =
                scene.GetCamera().GetCameraTransform().RenderFromWorld() * info->ctm;
            write_frame(medium.render_from_medium, render_from_medium);
            // PBRT's `Inverse(t)` is `Transform(t.mInv, t.m)`: the inverse PBRT
            // itself carries, not one computed again here.
            write_frame(medium.medium_from_render, pbrt::Inverse(render_from_medium));
            return pbrt::Bounds3f(p0, p1);
        };
        // A grid into the pool, `channels` floats per voxel.
        const auto push_grid = [&](const std::vector<float> &values, int nx, int ny,
                                   int nz) {
            bonsai_scene::GridRef g;
            g.nx = nx;
            g.ny = ny;
            g.nz = nz;
            g.at = uint32_t(out.medium_grid.size());
            out.medium_grid.insert(out.medium_grid.end(), values.begin(),
                                   values.end());
            return g;
        };
        // PBRT: MajorantGrid::VoxelBounds, the cell of a 16x16x16 grid over
        // the unit cube; the constructors take each cell's maximum over it.
        constexpr int kMajorantRes = 16;
        const auto cell_bounds = [](int x, int y, int z) {
            return pbrt::Bounds3f(
                pbrt::Point3f(pbrt::Float(x) / kMajorantRes,
                              pbrt::Float(y) / kMajorantRes,
                              pbrt::Float(z) / kMajorantRes),
                pbrt::Point3f(pbrt::Float(x + 1) / kMajorantRes,
                              pbrt::Float(y + 1) / kMajorantRes,
                              pbrt::Float(z + 1) / kMajorantRes));
        };
        const auto grid_dims = [&](size_t n, int *nx, int *ny, int *nz) {
            *nx = params.GetOneInt("nx", 1);
            *ny = params.GetOneInt("ny", 1);
            *nz = params.GetOneInt("nz", 1);
            if (*nx <= 0 || *ny <= 0 || *nz <= 0 ||
                n != size_t(*nx) * size_t(*ny) * size_t(*nz)) {
                fail(what + " has " + std::to_string(n) +
                     " values; expected nx*ny*nz = " +
                     std::to_string(*nx * *ny * *nz));
            }
        };

        if (kind == "homogeneous") {
            // PBRT: HomogeneousMedium::Create -- `Le` with `Lescale` divided by
            // its photometric integral, or nothing.
            pbrt::Spectrum le = params.GetOneSpectrum(
                "Le", nullptr, pbrt::SpectrumType::Illuminant, alloc);
            pbrt::Float le_scale = params.GetOneFloat("Lescale", 1.f);
            if (!le || le.MaxValue() == 0) {
                le = alloc.new_object<pbrt::ConstantSpectrum>(0.f);
            } else {
                le_scale /= pbrt::SpectrumToPhotometric(le);
            }
            medium.tag = bonsai_scene::MediumTag::Homogeneous;
            medium.emissive =
                push_spectra(sig_a, sig_s, le, sigma_scale, le_scale) > 0.f ? 1u : 0u;
        } else if (kind == "uniformgrid") {
            // PBRT: GridMedium::Create and the constructor.
            const std::vector<pbrt::Float> density = params.GetFloatArray("density");
            const std::vector<pbrt::Float> temperature =
                params.GetFloatArray("temperature");
            if (density.empty()) {
                fail(what + " has no \"density\"");
            }
            if (!temperature.empty() && temperature.size() != density.size()) {
                fail(what + " gives " + std::to_string(density.size()) +
                     " density values and " + std::to_string(temperature.size()) +
                     " temperatures");
            }
            int nx = 0, ny = 0, nz = 0;
            grid_dims(density.size(), &nx, &ny, &nz);
            pbrt::Spectrum le = params.GetOneSpectrum(
                "Le", nullptr, pbrt::SpectrumType::Illuminant, alloc);
            if (le && !temperature.empty()) {
                fail(what + " gives both \"Le\" and \"temperature\", which PBRT "
                            "refuses too");
            }
            pbrt::Float le_norm = 1.f;
            if (!le || le.MaxValue() == 0) {
                le = alloc.new_object<pbrt::ConstantSpectrum>(0.f);
            } else {
                le_norm = 1 / pbrt::SpectrumToPhotometric(le);
            }
            std::vector<pbrt::Float> le_scale = params.GetFloatArray("Lescale");
            int lx = 1, ly = 1, lz = 1;
            if (le_scale.empty()) {
                le_scale = {le_norm};
            } else {
                if (le_scale.size() != density.size()) {
                    fail(what + " gives " + std::to_string(le_scale.size()) +
                         " \"Lescale\" values for " +
                         std::to_string(density.size()) + " voxels");
                }
                for (pbrt::Float &v : le_scale) {
                    v *= le_norm;
                }
                lx = nx;
                ly = ny;
                lz = nz;
            }
            medium.tag = bonsai_scene::MediumTag::UniformGrid;
            place();
            medium.temperature_offset = params.GetOneFloat(
                "temperatureoffset", params.GetOneFloat("temperaturecutoff", 0.f));
            medium.temperature_scale = params.GetOneFloat("temperaturescale", 1.f);
            const float le_max = push_spectra(sig_a, sig_s, le, sigma_scale, 1.f);
            // PBRT: `isEmissive = temperatureGrid ? true : (Le_spec.MaxValue() > 0)`.
            medium.emissive = (!temperature.empty() || le_max > 0.f) ? 1u : 0u;
            medium.density = push_grid(density, nx, ny, nz);
            if (!temperature.empty()) {
                medium.temperature = push_grid(temperature, nx, ny, nz);
            }
            medium.le_scale = push_grid(le_scale, lx, ly, lz);
            // The majorant grid: `densityGrid.MaxValue(bounds)` per cell, by
            // PBRT's own SampledGrid.
            const pbrt::SampledGrid<pbrt::Float> density_grid(density, nx, ny, nz,
                                                              alloc);
            std::vector<float> majorant(size_t(kMajorantRes) * kMajorantRes *
                                        kMajorantRes);
            for (int z = 0; z < kMajorantRes; z++) {
                for (int y = 0; y < kMajorantRes; y++) {
                    for (int x = 0; x < kMajorantRes; x++) {
                        majorant[size_t(x) + kMajorantRes * (size_t(y) +
                                                             kMajorantRes * size_t(z))] =
                            float(density_grid.MaxValue(cell_bounds(x, y, z)));
                    }
                }
            }
            medium.majorant = push_grid(majorant, kMajorantRes, kMajorantRes,
                                        kMajorantRes);
        } else if (kind == "rgbgrid") {
            // PBRT: RGBGridMedium::Create and the constructor.
            const std::vector<pbrt::RGB> rgb_a = params.GetRGBArray("sigma_a");
            const std::vector<pbrt::RGB> rgb_s = params.GetRGBArray("sigma_s");
            const std::vector<pbrt::RGB> rgb_le = params.GetRGBArray("Le");
            if (rgb_a.empty() && rgb_s.empty()) {
                fail(what + " needs \"sigma_a\" and/or \"sigma_s\"");
            }
            const size_t n = rgb_a.empty() ? rgb_s.size() : rgb_a.size();
            if (!rgb_s.empty() && rgb_s.size() != n) {
                fail(what + " gives " + std::to_string(rgb_a.size()) +
                     " sigma_a values and " + std::to_string(rgb_s.size()) +
                     " sigma_s values");
            }
            if (!rgb_le.empty() && rgb_a.empty()) {
                fail(what + " gives \"Le\" without \"sigma_a\", which PBRT refuses too");
            }
            if (!rgb_le.empty() && rgb_le.size() != n) {
                fail(what + " gives " + std::to_string(rgb_le.size()) +
                     " \"Le\" values for " + std::to_string(n) + " voxels");
            }
            int nx = 0, ny = 0, nz = 0;
            grid_dims(n, &nx, &ny, &nz);
            const pbrt::RGBColorSpace *space = params.ColorSpace();
            const SpectrumTable table = table_of(space, what);
            // The renderer multiplies an emission voxel's fit by the D65
            // illuminant, which is sRGB's and DCI-P3's (illuminant_index).
            if (!rgb_le.empty() && illuminant_index(space, out) != 0) {
                fail(what + " emits in a colour space whose illuminant is not "
                            "D65, which the grid media do not carry yet");
            }
            // PBRT: RGBUnboundedSpectrum's constructor (RGBIlluminantSpectrum's
            // is the same, plus the illuminant): twice the largest component,
            // and the colour space's sigmoid coefficients for the colour
            // divided by it -- the scale and the three coefficients per voxel.
            const auto rgb_grid = [&](const std::vector<pbrt::RGB> &rgbs) {
                bonsai_scene::GridRef none;
                if (rgbs.empty()) {
                    return none;
                }
                std::vector<float> fitted;
                fitted.reserve(rgbs.size() * 4);
                for (const pbrt::RGB &c : rgbs) {
                    const pbrt::Float m = std::max({c.r, c.g, c.b});
                    const pbrt::Float scale = 2 * m;
                    const pbrt::RGB unit = scale ? c / scale : pbrt::RGB(0, 0, 0);
                    const Sigmoid fit = fit_sigmoid(table, float(unit.r),
                                                    float(unit.g), float(unit.b));
                    fitted.push_back(float(scale));
                    fitted.push_back(fit.c0);
                    fitted.push_back(fit.c1);
                    fitted.push_back(fit.c2);
                }
                return push_grid(fitted, nx, ny, nz);
            };
            medium.tag = bonsai_scene::MediumTag::RGBGrid;
            place();
            medium.sigma_scale = sigma_scale;
            medium.le_scale_value = params.GetOneFloat("Lescale", 1.f);
            medium.sigma_a = rgb_grid(rgb_a);
            medium.sigma_s = rgb_grid(rgb_s);
            medium.le = rgb_grid(rgb_le);
            // The majorant grid: `sigmaScale * (max sigma_a + max sigma_s)` per
            // cell, each maximum over PBRT's own grid of its own spectra, a
            // grid absent counting one.
            const auto spectra_of = [&](const std::vector<pbrt::RGB> &rgbs) {
                std::vector<pbrt::RGBUnboundedSpectrum> spectra;
                spectra.reserve(rgbs.size());
                for (const pbrt::RGB &c : rgbs) {
                    spectra.emplace_back(*space, c);
                }
                return spectra;
            };
            const std::vector<pbrt::RGBUnboundedSpectrum> spectra_a = spectra_of(rgb_a);
            const std::vector<pbrt::RGBUnboundedSpectrum> spectra_s = spectra_of(rgb_s);
            std::optional<pbrt::SampledGrid<pbrt::RGBUnboundedSpectrum>> grid_a, grid_s;
            if (!spectra_a.empty()) {
                grid_a.emplace(spectra_a, nx, ny, nz, alloc);
            }
            if (!spectra_s.empty()) {
                grid_s.emplace(spectra_s, nx, ny, nz, alloc);
            }
            const auto max_of = [](pbrt::RGBUnboundedSpectrum s) { return s.MaxValue(); };
            std::vector<float> majorant(size_t(kMajorantRes) * kMajorantRes *
                                        kMajorantRes);
            for (int z = 0; z < kMajorantRes; z++) {
                for (int y = 0; y < kMajorantRes; y++) {
                    for (int x = 0; x < kMajorantRes; x++) {
                        const pbrt::Bounds3f cell = cell_bounds(x, y, z);
                        const pbrt::Float max_sigma_t =
                            (grid_a ? grid_a->MaxValue(cell, max_of) : 1) +
                            (grid_s ? grid_s->MaxValue(cell, max_of) : 1);
                        majorant[size_t(x) + kMajorantRes * (size_t(y) +
                                                             kMajorantRes * size_t(z))] =
                            float(sigma_scale * max_sigma_t);
                    }
                }
            }
            medium.majorant = push_grid(majorant, kMajorantRes, kMajorantRes,
                                        kMajorantRes);
        } else if (kind == "nanovdb") {
            // PBRT: NanoVDBMedium::Create and the constructor. The grids are
            // read with NanoVDB's own reader and copied out as they are
            // (Scene::vdb_bytes): the renderer reads them through NanoVDB's
            // accessor in a shim (nanovdb_shim.cpp, docs/foreign-functions.md)
            // and never looks into the bytes itself, which is how PBRT reads
            // them too.
            const std::string filename =
                pbrt::ResolveFilename(params.GetOneString("filename", ""));
            if (filename.empty()) {
                fail(what + " names no \"filename\"");
            }
            // PBRT: readGrid -- a grid not in the file is an empty handle, a
            // grid that is not a fog volume an error.
            const auto read_grid = [&](const std::string &grid_name) {
                nanovdb::GridHandle<nanovdb::HostBuffer> handle;
                try {
                    handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(
                        filename, grid_name, 0 /* not verbose */);
                } catch (const std::exception &e) {
                    fail(what + ": " + filename + ": " + e.what());
                }
                if (handle && !handle.gridMetaData()->isFogVolume() &&
                    !handle.gridMetaData()->isUnknown()) {
                    fail(what + ": " + filename + ": \"" + grid_name +
                         "\" is not a fog volume grid");
                }
                return handle;
            };
            nanovdb::GridHandle<nanovdb::HostBuffer> density_handle =
                read_grid(params.GetOneString("gridname", "density"));
            if (!density_handle) {
                fail(what + ": " + filename + " has no \"density\" grid");
            }
            nanovdb::GridHandle<nanovdb::HostBuffer> temperature_handle =
                read_grid(params.GetOneString("temperaturename", "temperature"));
            const nanovdb::FloatGrid *density = density_handle.grid<float>();
            if (density == nullptr) {
                fail(what + ": " + filename + ": the density grid holds no floats");
            }
            const nanovdb::FloatGrid *temperature =
                temperature_handle ? temperature_handle.grid<float>() : nullptr;
            if (temperature_handle && temperature == nullptr) {
                fail(what + ": " + filename + ": the temperature grid holds no floats");
            }
            // Into the sidecar as they are, each grid beginning on NanoVDB's
            // 32-byte alignment; the renderer's handle is the address.
            const auto push_vdb = [&](const nanovdb::GridHandle<nanovdb::HostBuffer> &h) {
                while (out.vdb_bytes.size() % 32 != 0) {
                    out.vdb_bytes.push_back(0);
                }
                const uint32_t at = uint32_t(out.vdb_bytes.size());
                out.vdb_bytes.insert(out.vdb_bytes.end(), h.data(), h.data() + h.size());
                return at;
            };
            medium.tag = bonsai_scene::MediumTag::NanoVDB;
            place(); // the transform; the box is the grids' own, below
            medium.density_at = push_vdb(density_handle);
            if (temperature != nullptr) {
                medium.has_temperature = 1u;
                medium.temperature_at = push_vdb(temperature_handle);
            }
            // Each grid's map from world (the medium's space) to index space,
            // as NanoVDB's Map holds it: the 3x3 inverse, then the translation.
            const auto write_map = [](float *out12, const nanovdb::Map &map) {
                for (int i = 0; i < 9; i++) {
                    out12[i] = map.mInvMatF[i];
                }
                for (int i = 0; i < 3; i++) {
                    out12[9 + i] = map.mVecF[i];
                }
            };
            write_map(medium.density_map, density->map());
            if (temperature != nullptr) {
                write_map(medium.temperature_map, temperature->map());
            }
            // PBRT: `bounds`, the density grid's world box, unioned with the
            // temperature grid's.
            const auto box_of = [](const nanovdb::FloatGrid *g) {
                const nanovdb::BBox<nanovdb::Vec3R> b = g->worldBBox();
                return pbrt::Bounds3f(
                    pbrt::Point3f(pbrt::Float(b.min()[0]), pbrt::Float(b.min()[1]),
                                  pbrt::Float(b.min()[2])),
                    pbrt::Point3f(pbrt::Float(b.max()[0]), pbrt::Float(b.max()[1]),
                                  pbrt::Float(b.max()[2])));
            };
            pbrt::Bounds3f bounds = box_of(density);
            if (temperature != nullptr) {
                bounds = pbrt::Union(bounds, box_of(temperature));
            }
            for (int c = 0; c < 3; c++) {
                medium.low[c] = float(bounds.pMin[c]);
                medium.high[c] = float(bounds.pMax[c]);
            }
            medium.le_scale_value = params.GetOneFloat("Lescale", 1.f);
            medium.temperature_offset = params.GetOneFloat(
                "temperatureoffset", params.GetOneFloat("temperaturecutoff", 0.f));
            medium.temperature_scale = params.GetOneFloat("temperaturescale", 1.f);
            // PBRT: IsEmissive, `temperatureFloatGrid && LeScale > 0`.
            medium.emissive =
                (temperature != nullptr && medium.le_scale_value > 0.f) ? 1u : 0u;
            // The emission is a blackbody from the temperature grid, so the Le
            // spectrum is nothing.
            push_spectra(sig_a, sig_s, alloc.new_object<pbrt::ConstantSpectrum>(0.f),
                         sigma_scale, 1.f);
            // PBRT: the 64x64x64 majorant grid the constructor builds -- each
            // cell the maximum of the density over the index-space box its
            // world box reaches, widened by one voxel of filter slop and
            // clipped to the grid's own box, through NanoVDB's accessor --
            // the constructor's loop as it is written, in parallel as PBRT
            // runs it. The density grid alone: the emission has no majorant.
            constexpr int kVdbMajorantRes = 64;
            std::vector<float> majorant(size_t(kVdbMajorantRes) * kVdbMajorantRes *
                                        kVdbMajorantRes);
            pbrt::ParallelFor(0, int64_t(majorant.size()), [&](int64_t index) {
                const int x = int(index % kVdbMajorantRes);
                const int y = int((index / kVdbMajorantRes) % kVdbMajorantRes);
                const int z = int(index / (kVdbMajorantRes * kVdbMajorantRes));
                const pbrt::Bounds3f wb(
                    bounds.Lerp(pbrt::Point3f(pbrt::Float(x) / kVdbMajorantRes,
                                              pbrt::Float(y) / kVdbMajorantRes,
                                              pbrt::Float(z) / kVdbMajorantRes)),
                    bounds.Lerp(pbrt::Point3f(pbrt::Float(x + 1) / kVdbMajorantRes,
                                              pbrt::Float(y + 1) / kVdbMajorantRes,
                                              pbrt::Float(z + 1) / kVdbMajorantRes)));
                const nanovdb::Vec3R i0 = density->worldToIndexF(
                    nanovdb::Vec3R(wb.pMin.x, wb.pMin.y, wb.pMin.z));
                const nanovdb::Vec3R i1 = density->worldToIndexF(
                    nanovdb::Vec3R(wb.pMax.x, wb.pMax.y, wb.pMax.z));
                const auto bbox = density->indexBBox();
                const pbrt::Float delta = 1.f; // filter slop
                const int nx0 = std::max(int(i0[0] - delta), bbox.min()[0]);
                const int nx1 = std::min(int(i1[0] + delta), bbox.max()[0]);
                const int ny0 = std::max(int(i0[1] - delta), bbox.min()[1]);
                const int ny1 = std::min(int(i1[1] + delta), bbox.max()[1]);
                const int nz0 = std::max(int(i0[2] - delta), bbox.min()[2]);
                const int nz1 = std::min(int(i1[2] + delta), bbox.max()[2]);
                float max_value = 0;
                auto accessor = density->getAccessor();
                // NanoVDB's integer boxes are inclusive at the top.
                for (int nz = nz0; nz <= nz1; ++nz) {
                    for (int ny = ny0; ny <= ny1; ++ny) {
                        for (int nx = nx0; nx <= nx1; ++nx) {
                            max_value = std::max(max_value,
                                                 accessor.getValue({nx, ny, nz}));
                        }
                    }
                }
                majorant[size_t(index)] = max_value;
            });
            medium.majorant =
                push_grid(majorant, kVdbMajorantRes, kVdbMajorantRes, kVdbMajorantRes);
            // A check, when BONSAI_CHECK_MAJORANT is set: PBRT's own
            // NanoVDBMedium built from the same parameters under the
            // identity, its majorant read cell by cell through SampleRay --
            // a ray from the cell's centre along x, too short to leave the
            // cell, whose first segment carries sigma_t times the cell's
            // maximum -- against the grid just built. The majorant decides
            // how often a path's collisions are tentative, and with an
            // emissive medium each tentative collision adds light, so a
            // majorant that is not PBRT's shows as pixels lit here and not
            // there before it shows anywhere else.
            if (std::getenv("BONSAI_CHECK_MAJORANT") != nullptr) {
                pbrt::Medium theirs = pbrt::Medium::Create("nanovdb", params,
                                                           pbrt::Transform(), nullptr, alloc);
                pbrt::ScratchBuffer scratch;
                const pbrt::SampledWavelengths lambda =
                    pbrt::SampledWavelengths::SampleUniform(0.5f);
                const pbrt::Float sigma_t0 =
                    (sig_a.Sample(lambda)[0] + sig_s.Sample(lambda)[0]) * sigma_scale;
                const pbrt::Float cell = (bounds.pMax.x - bounds.pMin.x) / kVdbMajorantRes;
                int differing = 0;
                float worst = 0.f;
                for (size_t index = 0; index < majorant.size(); index++) {
                    const int x = int(index % kVdbMajorantRes);
                    const int y = int((index / kVdbMajorantRes) % kVdbMajorantRes);
                    const int z = int(index / (kVdbMajorantRes * kVdbMajorantRes));
                    const pbrt::Point3f centre = bounds.Lerp(pbrt::Point3f(
                        (pbrt::Float(x) + 0.5f) / kVdbMajorantRes,
                        (pbrt::Float(y) + 0.5f) / kVdbMajorantRes,
                        (pbrt::Float(z) + 0.5f) / kVdbMajorantRes));
                    pbrt::Ray ray(centre, pbrt::Vector3f(1.f, 0.f, 0.f));
                    pbrt::RayMajorantIterator iter =
                        theirs.SampleRay(ray, cell * 0.01f, lambda, scratch);
                    const pstd::optional<pbrt::RayMajorantSegment> seg = iter.Next();
                    const float pbrt_max = seg ? float(seg->sigma_maj[0] / sigma_t0) : -1.f;
                    const float ours = majorant[index];
                    if (std::fabs(pbrt_max - ours) > 1e-5f * std::max(1.f, ours)) {
                        if (differing < 5) {
                            printf("scene_dump: majorant cell (%d, %d, %d): ours %g, PBRT's %g\n",
                                   x, y, z, ours, pbrt_max);
                        }
                        differing++;
                        worst = std::max(worst, std::fabs(pbrt_max - ours));
                    }
                }
                printf("scene_dump: nanovdb majorant check for %s: %d of %zu cells differ "
                       "from PBRT's (worst %g)\n",
                       what.c_str(), differing, majorant.size(), worst);
            }
        } else if (kind == "cloud") {
            // PBRT: CloudMedium::Create -- no `scale`, no emission.
            medium.tag = bonsai_scene::MediumTag::Cloud;
            place();
            medium.cloud_density = params.GetOneFloat("density", 1.f);
            medium.wispiness = params.GetOneFloat("wispiness", 1.f);
            medium.frequency = params.GetOneFloat("frequency", 5.f);
            push_spectra(sig_a, sig_s, alloc.new_object<pbrt::ConstantSpectrum>(0.f),
                         1.f, 1.f);
            medium.emissive = 0u;
        } else {
            fail(what + " is of a kind this converter does not have");
        }
        const int32_t at = int32_t(out.media.size());
        out.media.push_back(medium);
        medium_index.emplace(name, at);
        return at;
    };

    // What the textures' geometric mappings and noises compose their frames
    // with (texture_from_render_of): PBRT's `renderFromTexture` is the
    // camera's render-from-world times the transform at the declaration.
    g_render_from_world = scene.GetCamera().GetCameraTransform().RenderFromWorld();

    // A shape names its material by index, and several shapes usually name the
    // same one, so the materials are written once and indexed rather than
    // copied per shape. The index a shape carries is this file's, not PBRT's:
    // only the materials some shape actually uses are written.
    std::map<int, uint32_t> material_index;
    const auto material_for = [&](int declared) {
        const auto it = material_index.find(declared);
        if (it != material_index.end()) {
            return it->second;
        }
        // What a shape declared outside any Material directive gets: PBRT's
        // own default, which is a fifty-per-cent grey diffuse.
        bonsai_scene::Material converted;
        if (declared >= 0) {
            if (declared >= int(builder.materials.size())) {
                fail("a shape names a material that was never declared");
            }
            converted = convert_material(builder.materials[size_t(declared)]);
        }
        // The index is taken after the conversion: a `mix` converts its two
        // operands on the way, and they take the indices below its own.
        const uint32_t at = uint32_t(out.materials.size());
        out.materials.push_back(converted);
        material_index.emplace(declared, at);
        return at;
    };

    // And the same for a material a shape reached by name. Separate maps
    // because PBRT indexes the two differently -- a `Material` directive gets
    // an index and a `MakeNamedMaterial` gets a name -- and a shape carries
    // whichever applied.
    std::map<std::string, uint32_t> named_material_index;
    const auto material_for_named = [&](const std::string &name) {
        const auto it = named_material_index.find(name);
        if (it != named_material_index.end()) {
            return it->second;
        }
        const auto declared = builder.named_materials.find(name);
        if (declared == builder.named_materials.end()) {
            fail("a shape uses the named material \"" + name +
                 "\", which was never declared");
        }
        // Converted before the index is taken, as above: a `mix` reaches
        // its operands through this same function while it is converted.
        bonsai_scene::Material converted = convert_material(declared->second);
        const uint32_t at = uint32_t(out.materials.size());
        out.materials.push_back(converted);
        named_material_index.emplace(name, at);
        return at;
    };
    g_material_for_named = material_for_named;

    // The area lights, converted on first use as the materials are, so that
    // the scene file carries only the ones a shape actually emits with.
    //
    // PBRT's own `scale` is folded with the division that makes a radiance of
    // one mean one nit -- `scale /= SpectrumToPhotometric(L)` in
    // DiffuseAreaLight::Create. Doing that here rather than in the renderer is
    // the same division of labour the rest of this file follows: it is a
    // property of the scene, decided once when it is read, and it needs PBRT's
    // photometric integral, which is a table lookup over the whole visible
    // range rather than anything a ray does.
    std::map<int, int32_t> light_index;
    const auto light_for = [&](int declared) -> int32_t {
        if (declared < 0) {
            return -1;
        }
        const auto it = light_index.find(declared);
        if (it != light_index.end()) {
            return it->second;
        }
        if (size_t(declared) >= builder.area_lights.size()) {
            fail("a shape names an area light the parser did not record");
        }
        const CapturingBuilder::MaterialInfo &info =
            builder.area_lights[size_t(declared)];
        if (info.name != "diffuse") {
            fail("only `diffuse` area lights are supported, not: " + info.name);
        }
        if (info.find("filename") != nullptr) {
            fail("an area light with an image is not supported");
        }
        if (info.find("power") != nullptr) {
            fail("an area light given a `power` is not supported");
        }

        bonsai_scene::Light light;
        // PBRT's own scale, divided by the photometric integral of L so that a
        // radiance of one means one nit. The spectrum that division is over is
        // whichever one PBRT would have built from the scene's `L`, so it is
        // built here the same way: a blackbody at the temperature written, or
        // the illuminant fitted from the RGB. The default when a scene names
        // no L is the colour space's own illuminant, which for sRGB is D65 --
        // and an RGB of one puts the fit through the same path, since that is
        // what the fit of a flat illuminant is.
        pbrt::Allocator alloc;
        const float scale = material_float(info, "scale", 1.f);
        if (read_blackbody(info, "L", &light.blackbody, &light.temperature,
                           &light.blackbody_normalization)) {
            const pbrt::BlackbodySpectrum emitted(light.temperature);
            light.scale = float(scale / pbrt::SpectrumToPhotometric(&emitted));
        } else {
            if (!material_rgb(info, "L", light.l)) {
                light.l[0] = light.l[1] = light.l[2] = 1.f;
            }
            const pbrt::RGBIlluminantSpectrum emitted(
                *pbrt::RGBColorSpace::sRGB,
                pbrt::RGB(light.l[0], light.l[1], light.l[2]));
            light.scale = float(scale / pbrt::SpectrumToPhotometric(&emitted));
        }
        const CapturingBuilder::MaterialInfo::Value *two =
            info.find("twosided");
        light.two_sided =
            (two != nullptr && !two->bools.empty() && two->bools[0]) ? 1u : 0u;

        const int32_t at = int32_t(out.lights.size());
        out.lights.push_back(light);
        light_index.emplace(declared, at);
        return at;
    };

    // The light BVH is built over the emitting shapes. Each is given a stable
    // ordinal here, in the order shapes are converted; that ordinal is the slot
    // the light takes in the renderer's per-shape light list, and carrying it on
    // the shape is what lets the tree survive the driver's later reorder.
    //
    // A real pbrt::DiffuseAreaLight is built for each emitter and the set of them
    // is handed to pbrt's own BVHLightSampler (dump_light_tree). The bounds, the
    // surface-area heuristic and the quantization are then all pbrt's, computed
    // by pbrt's code on pbrt's lights, not a copy of any of it. The lights, their
    // emission spectra and the sphere shapes and transforms they name are
    // allocated to outlive that build, which runs once every shape is converted;
    // the default pbrt::Allocator is new/delete-backed, so they live to the end.
    pbrt::Allocator light_alloc;
    std::vector<pbrt::Light> emitter_lights;
    int next_light_ordinal = 0;
    // pbrt's DiffuseAreaLight::Create for the cases scene_dump admits -- an RGB
    // or default L, a scale, twosided, with images and power refused above. The
    // illuminant spectrum and the scale already divided by its photometric
    // integral are what out.lights carries; the rest is pbrt's constructor.
    const auto make_area_light =
        [&](const pbrt::Transform &render_from_object, const pbrt::Shape shape,
            int emission) -> pbrt::Light {
        const bonsai_scene::Light &em = out.lights[size_t(emission)];
        const pbrt::Spectrum Lemit =
            em.blackbody != 0
                ? pbrt::Spectrum(light_alloc.new_object<pbrt::BlackbodySpectrum>(
                      em.temperature))
                : pbrt::Spectrum(
                      light_alloc.new_object<pbrt::RGBIlluminantSpectrum>(
                          *pbrt::RGBColorSpace::sRGB,
                          pbrt::RGB(em.l[0], em.l[1], em.l[2])));
        pbrt::DiffuseAreaLight *area =
            light_alloc.new_object<pbrt::DiffuseAreaLight>(
                render_from_object, pbrt::MediumInterface{}, Lemit, em.scale,
                shape, pbrt::FloatTexture(), pbrt::Image(light_alloc),
                pbrt::RGBColorSpace::sRGB, em.two_sided != 0);
        return pbrt::Light(area);
    };

    // One shape, converted into `into`. The same for a shape at the top level
    // and for one inside an `ObjectBegin` block: PBRT's CreateAggregate runs
    // both through one `CreatePrimitivesForShapes`, and the only difference it
    // makes is where the primitive then goes.
    const auto convert_shape = [&](const pbrt::ShapeSceneEntity &entity,
                                   std::vector<bonsai_scene::Shape> &into) {
        const std::string name(entity.name);
        const pbrt::Transform &render_from_object = *entity.renderFromObject;
        // A shape under `NamedMaterial` names its material by string and PBRT
        // leaves `materialIndex` at -1 -- which is also what a shape declared
        // outside any Material directive gets. Telling the two apart is the
        // whole of this line, and not telling them apart is how a
        // `MakeNamedMaterial` of any type at all silently became PBRT's default
        // fifty-per-cent grey diffuse for as long as it did.
        const uint32_t material =
            entity.materialName.empty()
                ? material_for(entity.materialIndex)
                : material_for_named(std::string(entity.materialName));
        // PBRT (scene.cpp, CreateLights): "Ignoring area light specification
        // for shape with "interface" material" -- a boundary emits nothing,
        // whatever the graphics state said when it was declared.
        const bool interface_material =
            out.materials[material].tag == bonsai_scene::MaterialTag::Interface;
        const int32_t light =
            interface_material ? -1 : light_for(entity.lightIndex);
        // PBRT: `MediumInterface mi(findMedium(sh.insideMedium),
        // findMedium(sh.outsideMedium))`, the two names the graphics state
        // held when the shape was declared.
        const int32_t medium_inside = medium_for(entity.insideMedium);
        const int32_t medium_outside = medium_for(entity.outsideMedium);

        // PBRT: the `alpha` a shape may carry, which puts it in a
        // GeometricPrimitive rather than a SimplePrimitive. Ignoring it made a
        // tree leaf a solid quad -- and quietly, since a scene with cutouts
        // still converted and still rendered.
        int32_t alpha = -1;
        {
            const auto &params = entity.parameters;
            for (const pbrt::ParsedParameter *p :
                 params.GetParameterVector()) {
                if (p->name != "alpha") {
                    continue;
                }
                if (p->type == "texture" && !p->strings.empty()) {
                    alpha = convert_texture(p->strings[0]);
                } else if (p->type == "float" && !p->floats.empty()) {
                    // PBRT: getAlphaTexture in scene.cpp -- a number below
                    // one is a FloatConstantTexture of it, which puts the
                    // shape in a GeometricPrimitive with the stochastic test;
                    // one or more is no alpha at all, as a shape without the
                    // parameter has. Zero stays a texture too: pbrt keeps the
                    // shape and ignores every hit on it (`u` is one when the
                    // alpha is not positive).
                    if (p->floats[0] < 1.f) {
                        alpha = constant_float_texture(p->floats[0]);
                    }
                } else {
                    fail("`alpha` on a shape has to be a texture or a float");
                }
            }
        }

        if (name == "sphere") {
            // PBRT: Sphere::Create and the constructor -- the radius; zmin and
            // zmax clamped to it, their arc cosines taken from the values as
            // given (the constructor's `std::acos(Clamp(std::min(zMin, zMax) /
            // radius, -1, 1))`), phimax in radians clamped to a turn; and the
            // transform whole, both matrices, as every quadric keeps it.
            // `flip` is reverseOrientation ^ the transform's handedness swap,
            // which turns a hit's normal; `reverse` the orientation alone,
            // which turns a sampled point's.
            const pbrt::Float radius =
                entity.parameters.GetOneFloat("radius", 1.f);
            const pbrt::Float zmin_given =
                entity.parameters.GetOneFloat("zmin", -radius);
            const pbrt::Float zmax_given =
                entity.parameters.GetOneFloat("zmax", radius);
            const pbrt::Float phimax_given =
                entity.parameters.GetOneFloat("phimax", 360.f);
            const pbrt::Float z_low = std::min(zmin_given, zmax_given);
            const pbrt::Float z_high = std::max(zmin_given, zmax_given);
            bonsai_scene::Shape shape;
            shape.tag = bonsai_scene::ShapeTag::Sphere;
            shape.radius = float(radius);
            shape.z_min = float(pbrt::Clamp(z_low, -radius, radius));
            shape.z_max = float(pbrt::Clamp(z_high, -radius, radius));
            shape.theta_z_min =
                float(std::acos(pbrt::Clamp(z_low / radius, -1, 1)));
            shape.theta_z_max =
                float(std::acos(pbrt::Clamp(z_high / radius, -1, 1)));
            shape.phi_max = float(pbrt::Radians(pbrt::Clamp(phimax_given, 0, 360)));
            write_frame(shape.render_from_object, render_from_object);
            write_frame(shape.object_from_render, pbrt::Inverse(render_from_object));
            shape.flip = (entity.reverseOrientation ^
                          render_from_object.SwapsHandedness())
                             ? 1u
                             : 0u;
            shape.reverse = entity.reverseOrientation ? 1u : 0u;
            shape.material = material;
            shape.light = light;
            shape.alpha = alpha;
            shape.medium_inside = medium_inside;
            shape.medium_outside = medium_outside;
            if (light >= 0) {
                shape.light_ordinal = next_light_ordinal++;
                // The Sphere and its two transforms are read by
                // DiffuseAreaLight::Bounds() when the tree is built at the end,
                // so they are allocated to persist rather than left on the stack.
                pbrt::Transform *r_from_o =
                    light_alloc.new_object<pbrt::Transform>(render_from_object);
                pbrt::Transform *o_from_r =
                    light_alloc.new_object<pbrt::Transform>(
                        pbrt::Inverse(render_from_object));
                pbrt::Sphere *sph = light_alloc.new_object<pbrt::Sphere>(
                    r_from_o, o_from_r, entity.reverseOrientation, float(radius),
                    float(zmin_given), float(zmax_given), float(phimax_given));
                emitter_lights.push_back(
                    make_area_light(render_from_object, pbrt::Shape(sph), light));
            }
            into.push_back(shape);

        } else if (name == "disk") {
            // PBRT: Disk::Create, and the constructor's clamp of phimax to a
            // turn and conversion to radians. Unlike the sphere above, any
            // transform is admitted: the disk stays in its object space and
            // both matrices go with it, which is how PBRT keeps every quadric.
            const pbrt::Float height =
                entity.parameters.GetOneFloat("height", 0.f);
            const pbrt::Float radius =
                entity.parameters.GetOneFloat("radius", 1.f);
            const pbrt::Float inner_radius =
                entity.parameters.GetOneFloat("innerradius", 0.f);
            const pbrt::Float phimax =
                entity.parameters.GetOneFloat("phimax", 360.f);
            bonsai_scene::Shape shape;
            shape.tag = bonsai_scene::ShapeTag::Disk;
            shape.height = float(height);
            shape.radius = float(radius);
            shape.inner_radius = float(inner_radius);
            shape.phi_max = float(pbrt::Radians(pbrt::Clamp(phimax, 0, 360)));
            const pbrt::Transform object_from_render =
                pbrt::Inverse(render_from_object);
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    shape.render_from_object[4 * r + c] =
                        float(render_from_object.GetMatrix()[r][c]);
                    shape.object_from_render[4 * r + c] =
                        float(object_from_render.GetMatrix()[r][c]);
                }
            }
            // PBRT: the two orientation flags a Disk keeps, and turns a hit's
            // and a sampled point's normal by respectively.
            shape.flip = (entity.reverseOrientation ^
                          render_from_object.SwapsHandedness())
                             ? 1u
                             : 0u;
            shape.reverse = entity.reverseOrientation ? 1u : 0u;
            shape.material = material;
            shape.light = light;
            shape.alpha = alpha;
            shape.medium_inside = medium_inside;
            shape.medium_outside = medium_outside;
            if (light >= 0) {
                shape.light_ordinal = next_light_ordinal++;
                pbrt::Transform *r_from_o =
                    light_alloc.new_object<pbrt::Transform>(render_from_object);
                pbrt::Transform *o_from_r =
                    light_alloc.new_object<pbrt::Transform>(object_from_render);
                pbrt::Disk *disk = light_alloc.new_object<pbrt::Disk>(
                    r_from_o, o_from_r, entity.reverseOrientation, height,
                    radius, inner_radius, phimax);
                emitter_lights.push_back(
                    make_area_light(render_from_object, pbrt::Shape(disk), light));
            }
            into.push_back(shape);

        } else {
            // Everything that is ultimately a mesh arrives here in render
            // space: a triangle mesh (`triangulate`), a bilinear patch mesh
            // (`patch_mesh_of`), or -- a PLY with both kinds of face -- one
            // of each, converted in turn.
            const pbrt::TriangleMesh *mesh = triangulate(entity);
            const pbrt::BilinearPatchMesh *patches = patch_mesh_of(entity);
            if (mesh == nullptr && patches == nullptr) {
                fail("unsupported shape \"" + name + "\"");
            }
            if (mesh != nullptr) {
            // Everything that is ultimately a mesh arrives here already
            // triangulated by PBRT, in render space, so there is one loop for
            // all of them rather than one per shape type. See `triangulate`.
            //
            // The mesh is appended to the shared pools and the triangles name
            // it, which is PBRT's own arrangement: a Triangle there is a mesh
            // index and a triangle index, and the vertices are read from the
            // mesh on a hit. Copying them per triangle instead made a primitive
            // four times larger than it needed to be, and made every sphere in
            // the scene pay for it too.
            //
            // PBRT's Triangle also reads a per-vertex tangent, which no shape
            // here produces: LoopSubdivide does not compute one and PLY meshes
            // are read without one. A mesh that had them would take a different
            // shading tangent, so refuse rather than ignore.
            if (mesh->s != nullptr) {
                fail("a mesh with per-vertex tangents (\"S\") is not "
                     "supported; its shading tangent is not the one the "
                     "texture coordinates give");
            }

            bonsai_scene::Mesh out_mesh;
            out_mesh.first_index = uint32_t(out.indices.size());
            out_mesh.first_vertex = uint32_t(out.positions.size() / 3);
            out_mesh.first_normal = uint32_t(out.normals.size() / 3);
            out_mesh.first_uv = uint32_t(out.uvs.size() / 2);
            // PBRT's TriangleMesh has already folded reverseOrientation and
            // transformSwapsHandedness together, and its Triangle reads the
            // pair back out of the mesh.
            out_mesh.flip =
                (mesh->reverseOrientation ^ mesh->transformSwapsHandedness) ? 1u
                                                                            : 0u;
            // The vertex normals a subdivision surface or a PLY brings with it.
            // Without them the shading normal is the geometric one, which is a
            // different path in PBRT and not the same answer, so the flag
            // travels rather than a pool of copies of the face normal.
            out_mesh.has_normals = mesh->n != nullptr ? 1u : 0u;
            // PBRT substitutes (0,0), (1,0), (1,1) for a mesh with no texture
            // coordinates. Those are per triangle rather than per vertex, so
            // they cannot go in the pool and the renderer substitutes them too.
            out_mesh.has_uv = mesh->uv != nullptr ? 1u : 0u;

            for (int i = 0; i < 3 * mesh->nTriangles; i++) {
                out.indices.push_back(uint32_t(mesh->vertexIndices[i]));
            }
            for (int i = 0; i < mesh->nVertices; i++) {
                out.positions.push_back(float(mesh->p[i].x));
                out.positions.push_back(float(mesh->p[i].y));
                out.positions.push_back(float(mesh->p[i].z));
                if (mesh->n != nullptr) {
                    out.normals.push_back(float(mesh->n[i].x));
                    out.normals.push_back(float(mesh->n[i].y));
                    out.normals.push_back(float(mesh->n[i].z));
                }
                if (mesh->uv != nullptr) {
                    out.uvs.push_back(float(mesh->uv[i].x));
                    out.uvs.push_back(float(mesh->uv[i].y));
                }
            }

            const uint32_t mesh_index = uint32_t(out.meshes.size());
            out.meshes.push_back(out_mesh);
            // A mesh that emits becomes one DiffuseAreaLight per triangle, as it
            // is in pbrt -- the renderer builds a Light per emissive shape
            // (render_hook.cpp) -- and each is sampled with Triangle::Sample,
            // pbrt's spherical-triangle sampling (shapes.bonsai). The Triangle
            // shapes come from pbrt's own CreateTriangles and feed pbrt's real
            // DiffuseAreaLight, which is what the light BVH is then built over.
            pstd::vector<pbrt::Shape> emit_tris;
            if (light >= 0) {
                emit_tris = pbrt::Triangle::CreateTriangles(mesh, light_alloc);
            }
            for (int i = 0; i < mesh->nTriangles; i++) {
                bonsai_scene::Shape shape;
                shape.tag = bonsai_scene::ShapeTag::Triangle;
                shape.mesh = mesh_index;
                shape.tri = uint32_t(i);
                shape.material = material;
                shape.light = light;
                shape.alpha = alpha;
                shape.medium_inside = medium_inside;
                shape.medium_outside = medium_outside;
                if (light >= 0) {
                    shape.light_ordinal = next_light_ordinal++;
                    emitter_lights.push_back(make_area_light(
                        render_from_object, emit_tris[size_t(i)], light));
                }
                into.push_back(shape);
            }
            }

            if (patches != nullptr) {
                // PBRT: BilinearPatch::CreatePatches -- one shape per four
                // indices of the mesh. The mesh goes into the same pools as a
                // triangle mesh (its record is the same, with the indices in
                // fours), and the patch names it and its place in it, as a
                // Triangle does. The area is PBRT's constructor's; the
                // renderer's sampling divides by it.
                if (patches->imageDistribution != nullptr) {
                    fail("a bilinear patch mesh with an image distribution is "
                         "not supported");
                }
                bonsai_scene::Mesh out_mesh;
                out_mesh.first_index = uint32_t(out.indices.size());
                out_mesh.first_vertex = uint32_t(out.positions.size() / 3);
                out_mesh.first_normal = uint32_t(out.normals.size() / 3);
                out_mesh.first_uv = uint32_t(out.uvs.size() / 2);
                out_mesh.flip = (patches->reverseOrientation ^
                                 patches->transformSwapsHandedness)
                                    ? 1u
                                    : 0u;
                out_mesh.has_normals = patches->n != nullptr ? 1u : 0u;
                out_mesh.has_uv = patches->uv != nullptr ? 1u : 0u;
                for (int i = 0; i < 4 * patches->nPatches; i++) {
                    out.indices.push_back(uint32_t(patches->vertexIndices[i]));
                }
                for (int i = 0; i < patches->nVertices; i++) {
                    out.positions.push_back(float(patches->p[i].x));
                    out.positions.push_back(float(patches->p[i].y));
                    out.positions.push_back(float(patches->p[i].z));
                    if (patches->n != nullptr) {
                        out.normals.push_back(float(patches->n[i].x));
                        out.normals.push_back(float(patches->n[i].y));
                        out.normals.push_back(float(patches->n[i].z));
                    }
                    if (patches->uv != nullptr) {
                        out.uvs.push_back(float(patches->uv[i].x));
                        out.uvs.push_back(float(patches->uv[i].y));
                    }
                }
                const uint32_t mesh_index = uint32_t(out.meshes.size());
                out.meshes.push_back(out_mesh);
                // An emitting patch mesh is a DiffuseAreaLight per patch, as a
                // triangle mesh's is per triangle, over PBRT's own patches so
                // that the light BVH is built over PBRT's bounds.
                pstd::vector<pbrt::Shape> emit_patches;
                if (light >= 0) {
                    emit_patches =
                        pbrt::BilinearPatch::CreatePatches(patches, light_alloc);
                }
                for (int i = 0; i < patches->nPatches; i++) {
                    bonsai_scene::Shape shape;
                    shape.tag = bonsai_scene::ShapeTag::Patch;
                    shape.mesh = mesh_index;
                    shape.tri = uint32_t(i);
                    shape.patch_area = patch_area_of(patches, i);
                    shape.material = material;
                    shape.light = light;
                    shape.alpha = alpha;
                    shape.medium_inside = medium_inside;
                    shape.medium_outside = medium_outside;
                    if (light >= 0) {
                        shape.light_ordinal = next_light_ordinal++;
                        emitter_lights.push_back(make_area_light(
                            render_from_object, emit_patches[size_t(i)], light));
                    }
                    into.push_back(shape);
                }
            }
        }
    };

    for (const pbrt::ShapeSceneEntity &entity : scene.shapes) {
        convert_shape(entity, shapes);
    }
    // PBRT: `CameraSceneEntity::medium`, resolved through the same map as
    // the shapes' -- a medium only the camera sits in is still a medium the
    // scene needs.
    out.camera_medium = medium_for(builder.camera_medium_name);

    // PBRT: the second list. `BasicScene::CreateAggregate` (scene.cpp) turns
    // each `instanceDefinitions` entry into one Primitive -- its shapes built
    // into a BVHAggregate of their own -- and each `instances` entry into a
    // TransformedPrimitive naming that Primitive and a `renderFromInstance`,
    // and puts those beside the top-level shapes in the list the scene's
    // accelerator is built over. Nothing is flattened: forty-three placements
    // of a tree share one tree.
    //
    // The definitions go in the order PBRT keeps them, which is the map's,
    // and their shapes go through the same conversion as everything else.
    // PBRT gives a shape inside a definition no area light -- `Shape` in the
    // builder warns and drops the light -- so `light` comes out -1 of its own
    // accord.
    std::map<pbrt::InternedString, uint32_t> definition_index;
    for (const auto &[name, definition] : scene.instanceDefinitions) {
        if (!definition->animatedShapes.empty()) {
            // PBRT: an AnimatedPrimitive inside the definition, whose
            // transform is interpolated per ray. A different kind of
            // primitive, not yet here.
            fail("instance definition \"" + std::string(name) +
                 "\" holds animated shapes, which are not supported");
        }
        bonsai_scene::Definition d;
        d.first_shape = uint32_t(out.instance_shapes.size());
        for (const pbrt::ShapeSceneEntity &entity : definition->shapes) {
            convert_shape(entity, out.instance_shapes);
        }
        d.shape_count = uint32_t(out.instance_shapes.size()) - d.first_shape;
        definition_index.emplace(name, uint32_t(out.definitions.size()));
        out.definitions.push_back(d);
    }
    for (const pbrt::InstanceSceneEntity &inst : scene.instances) {
        if (inst.renderFromInstanceAnim != nullptr) {
            // PBRT: AnimatedPrimitive, the instance placed by a transform that
            // moves over the frame. Its Intersect interpolates the transform
            // at the ray's time before pulling the ray back; a different
            // primitive kind again, refused rather than frozen at one time.
            fail("object instance \"" + std::string(inst.name) +
                 "\" has an animated transform, which is not supported");
        }
        const auto found = definition_index.find(inst.name);
        if (found == definition_index.cend()) {
            fail("object instance \"" + std::string(inst.name) +
                 "\" names no definition");
        }
        // PBRT: `if (!iter->second) continue;` -- an empty definition became
        // a null primitive, and an instance of it is skipped.
        if (out.definitions[found->second].shape_count == 0) {
            continue;
        }
        bonsai_scene::Instance placed;
        placed.definition = found->second;
        const pbrt::SquareMatrix<4> &m = inst.renderFromInstance->GetMatrix();
        const pbrt::SquareMatrix<4> &mi =
            inst.renderFromInstance->GetInverseMatrix();
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                placed.render_from_instance[4 * r + c] = float(m[r][c]);
                placed.instance_from_render[4 * r + c] = float(mi[r][c]);
            }
        }
        out.instances.push_back(placed);
    }

    if (shapes.empty() && out.instances.empty()) {
        fail("the scene has no shapes this renderer understands");
    }
    for (size_t i = 0; i < matrices.size(); i++) {
        out.matrices[i] = matrices[i];
    }

    // PBRT: the Integrator constructor's
    // `aggregate.Bounds().BoundingSphere(&sceneCenter, &sceneRadius)`, which is
    // what an infinite light is preprocessed with. The aggregate's bounds are
    // the union of its primitives', and a primitive's are Sphere::Bounds --
    // centre plus or minus the radius, since these are placed by a translation
    // -- or Triangle::Bounds, the union of its three vertices; an instance's
    // are `TransformedPrimitive::Bounds()`, its definition's bounds put
    // through the transform, which is the eight-corner union PBRT's
    // `Transform::operator()(const Bounds3f &)` takes. So the union is
    // computable here without building the tree first.
    if (!out.infinite_lights.empty()) {
        const auto shape_bounds = [&](const bonsai_scene::Shape &s) {
            if (s.tag == bonsai_scene::ShapeTag::Sphere) {
                // PBRT: Sphere::Bounds -- the clipped sphere's object-space
                // box, `(-r, -r, zMin)..(r, r, zMax)`, under its transform.
                pbrt::Float m[4][4];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        m[i][j] = s.render_from_object[4 * i + j];
                    }
                }
                const pbrt::Transform render_from_object{pbrt::SquareMatrix<4>(m)};
                return render_from_object(pbrt::Bounds3f(
                    pbrt::Point3f(-s.radius, -s.radius, s.z_min),
                    pbrt::Point3f(s.radius, s.radius, s.z_max)));
            }
            if (s.tag == bonsai_scene::ShapeTag::Disk) {
                // PBRT: Disk::Bounds -- the disk's object-space box, flat at
                // its height, under its transform. (Found by bunny-cloud, the
                // first scene with a disk under an infinite light: the
                // triangle case below read a disk's mesh index.)
                pbrt::Float m[4][4];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        m[i][j] = s.render_from_object[4 * i + j];
                    }
                }
                const pbrt::Transform render_from_object{pbrt::SquareMatrix<4>(m)};
                return render_from_object(pbrt::Bounds3f(
                    pbrt::Point3f(-s.radius, -s.radius, s.height),
                    pbrt::Point3f(s.radius, s.radius, s.height)));
            }
            if (s.tag == bonsai_scene::ShapeTag::Patch) {
                pbrt::Bounds3f b;
                uint32_t corner[4];
                out.patch_corners(s, corner);
                for (const uint32_t v : corner) {
                    b = pbrt::Union(b, pbrt::Point3f(out.positions[3 * v + 0],
                                                     out.positions[3 * v + 1],
                                                     out.positions[3 * v + 2]));
                }
                return b;
            }
            pbrt::Bounds3f b;
            uint32_t corner[3];
            out.corners(s, corner);
            for (const uint32_t v : corner) {
                b = pbrt::Union(b, pbrt::Point3f(out.positions[3 * v + 0],
                                                 out.positions[3 * v + 1],
                                                 out.positions[3 * v + 2]));
            }
            return b;
        };
        pbrt::Bounds3f scene_bounds;
        for (const bonsai_scene::Shape &s : shapes) {
            scene_bounds = pbrt::Union(scene_bounds, shape_bounds(s));
        }
        std::vector<pbrt::Bounds3f> definition_bounds(out.definitions.size());
        for (size_t d = 0; d < out.definitions.size(); d++) {
            const bonsai_scene::Definition &def = out.definitions[d];
            for (uint32_t i = 0; i < def.shape_count; i++) {
                definition_bounds[d] = pbrt::Union(
                    definition_bounds[d],
                    shape_bounds(out.instance_shapes[def.first_shape + i]));
            }
        }
        for (const bonsai_scene::Instance &inst : out.instances) {
            pbrt::SquareMatrix<4> m;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    m[r][c] = inst.render_from_instance[4 * r + c];
                }
            }
            const pbrt::Transform render_from_instance(m);
            scene_bounds = pbrt::Union(
                scene_bounds,
                render_from_instance(definition_bounds[inst.definition]));
        }
        pbrt::Point3f centre;
        pbrt::Float radius = 0;
        scene_bounds.BoundingSphere(&centre, &radius);
        out.scene_radius = float(radius);
    }

    // The light sampler, which only `path` reads and which PBRT defaults to
    // `bvh` there. Whichever the scene names is the one the renderer runs,
    // with PBRT's own tree serialized for the BVH one.
    //
    // Until 2026-09-20 a scene with fewer than two bounded lights was put on
    // the uniform sampler whatever it named, on the argument that the two are
    // the same function there: over one bounded light the tree is one node
    // and its PMF is one, as the uniform sampler's is. The PMF is; the *draw*
    // is not, as soon as an infinite or a distant light is in the scene too.
    // BVHLightSampler::Sample gives the infinite lights the low end of u --
    // `u < pInfinite` picks one of them -- and the tree the rest, where the
    // uniform sampler's `min(u * n, n - 1)` gives the low end to light zero,
    // which is the bounded one, since the area lights come first in the list.
    // Same probabilities, opposite assignment of every random number: every
    // sample picked the other light, and killeroo-gold -- one disk light and
    // one distant light -- rendered the right mean with not a pixel agreeing.
    // With one bounded light and nothing else the two do agree, except that
    // the BVH sampler declines to return the light where its bounds say no
    // energy can reach the point (the root-leaf importance test in
    // `bvh_descend`); that contributes nothing either way, and the sampler
    // stream does not move, but the arm is PBRT's, so it is taken.
    //
    // `volpath` too: VolPathIntegrator::Create reads `lightsampler` with the
    // same "bvh" default as PathIntegrator::Create. Until 2026-09-22 only
    // `path` took this branch, so a volpath scene sampled its lights
    // uniformly where pbrt walked the tree -- the same PMF over one light,
    // a different draw over two, and book (two sphere lights) rendered the
    // right mean with not a pixel agreeing.
    if (out.integrator == bonsai_scene::IntegratorTag::Path ||
        out.integrator == bonsai_scene::IntegratorTag::VolPath) {
        if (builder.light_sampler_name != "uniform" &&
            builder.light_sampler_name != "bvh") {
            fail("this renderer's `path` and `volpath` implement the `uniform` "
                 "and `bvh` light samplers, and the scene asks for `" +
                 builder.light_sampler_name + "`");
        }
        if (builder.light_sampler_name == "bvh" &&
            (!emitter_lights.empty() || !positional_lights.empty() ||
             !out.infinite_lights.empty())) {
            // The bounded lights in the renderer's order: the area lights by
            // ordinal, then the point and spot lights as declared.
            std::vector<pbrt::Light> bounded = emitter_lights;
            bounded.insert(bounded.end(), positional_lights.begin(),
                           positional_lights.end());
            if (!bounded.empty()) {
                dump_light_tree(bounded, out);
            }
            out.light_sampler = 1u;
        }
    }

    // Every texture the scene converted, evaluated by PBRT's own texture
    // objects at a few points and footprints: the mapping, the flip, the
    // pyramid level the footprint picks, the bilinear filter, the scale chain,
    // and -- for a spectrum texture -- the sigmoid fit at four wavelengths.
    // This renderer reproduces all of it from the levels scene_dump shipped,
    // and the check is that it reproduces it bit for bit: a bump map is a
    // finite difference of two of these lookups, and a layered BSDF hashes the
    // direction a bump map tilted, so the last bit of a texel is the first bit
    // of a different image.
    //
    // After the materials, because that is what converts the textures and
    // gives them their indices. `texf` rows are float textures and `texs` rows
    // spectrum ones; the renderer, which does not know which a texture was
    // declared as, prints both and the script keeps the one PBRT printed.
    if (g_print_differentials) {
        pbrt::NamedTextures textures = scene.CreateTextures();
        const pbrt::SampledWavelengths lambda =
            pbrt::SampledWavelengths::SampleVisible(0.5f);
        printf("lambda: %.9g %.9g %.9g %.9g\n", double(lambda[0]),
               double(lambda[1]), double(lambda[2]), double(lambda[3]));
        const pbrt::Point2f uvs[] = {{0.3f, 0.7f}, {0.51f, 0.49f},
                                     {1.7f, -0.2f}};
        // (dudx, dudy, dvdx, dvdy): no footprint, one that lands on a fine
        // level, and one that lands on a coarse one.
        const float footprints[][4] = {{0.f, 0.f, 0.f, 0.f},
                                       {0.0007f, 0.0002f, -0.0003f, 0.0009f},
                                       {0.02f, 0.01f, 0.015f, 0.03f}};
        for (const auto &[name, index] : g_texture_index) {
            const auto declared = builder.named_textures.find(name);
            if (declared == builder.named_textures.end()) {
                continue;
            }
            const bool is_float = declared->second.declared_type == "float";
            const bonsai_scene::Texture &shipped = out.textures[size_t(index)];
            if (shipped.kind != bonsai_scene::TextureKind::Image ||
                shipped.mapping != bonsai_scene::MappingKind::UV) {
                // The rows are the image filter's; a graph or a geometric
                // mapping has nothing for them to compare.
                continue;
            }
            int k = 0;
            for (const pbrt::Point2f &uv : uvs) {
                for (const float *fp : footprints) {
                    const pbrt::TextureEvalContext ctx(
                        pbrt::Point3f(0, 0, 0), pbrt::Vector3f(0, 0, 0),
                        pbrt::Vector3f(0, 0, 0), pbrt::Normal3f(0, 0, 1), uv,
                        fp[0], fp[1], fp[2], fp[3], 0);
                    // The image's filtered RGB through PBRT's own MIPMap,
                    // before it is read as a number or fitted to a spectrum:
                    // SpectrumImageTexture::Evaluate's first lines, with the
                    // UV mapping and the flip written out. Splits a texture
                    // row that differs into the filter and what follows it.
                    {
                        const auto mip = g_texture_mip.find(name);
                        if (mip == g_texture_mip.end()) {
                            fail("no pyramid was kept for texture \"" + name +
                                 "\"");
                        }
                        pbrt::Point2f st(shipped.su * uv[0] + shipped.du,
                                         shipped.sv * uv[1] + shipped.dv);
                        st[1] = 1 - st[1];
                        const pbrt::Vector2f dst0(shipped.su * fp[0],
                                                  shipped.sv * fp[2]);
                        const pbrt::Vector2f dst1(shipped.su * fp[1],
                                                  shipped.sv * fp[3]);
                        pbrt::RGB rgb = shipped.scale *
                                        mip->second->Filter<pbrt::RGB>(
                                            st, dst0, dst1);
                        rgb = pbrt::ClampZero(shipped.invert != 0
                                                  ? (pbrt::RGB(1, 1, 1) - rgb)
                                                  : rgb);
                        printf("texrgb %d %d: %.9g %.9g %.9g\n", index, k,
                               double(rgb.r), double(rgb.g), double(rgb.b));
                        // And that colour fitted and sampled by PBRT's
                        // RGBAlbedoSpectrum: what a spectrum texture's
                        // Evaluate is made of, taken apart. This is how the
                        // folding of a constant scale was settled -- with the
                        // scale applied to the spectrum instead, this row
                        // agreed with the renderer and neither agreed with
                        // PBRT's texture.
                        const pbrt::SampledSpectrum s =
                            pbrt::RGBAlbedoSpectrum(*pbrt::RGBColorSpace::sRGB,
                                                    pbrt::Clamp(rgb, 0, 1))
                                .Sample(lambda);
                        printf("texsr %d %d: %.9g %.9g %.9g %.9g\n", index, k,
                               double(s[0]), double(s[1]), double(s[2]),
                               double(s[3]));
                    }
                    if (is_float) {
                        const auto tex = textures.floatTextures.find(name);
                        if (tex == textures.floatTextures.end()) {
                            fail("PBRT has no float texture named \"" + name +
                                 "\"");
                        }
                        printf("texf %d %d: %.9g\n", index, k,
                               double(tex->second.Evaluate(ctx)));
                    } else {
                        const auto tex =
                            textures.albedoSpectrumTextures.find(name);
                        if (tex == textures.albedoSpectrumTextures.end()) {
                            fail("PBRT has no spectrum texture named \"" +
                                 name + "\"");
                        }
                        const pbrt::SampledSpectrum s =
                            tex->second.Evaluate(ctx, lambda);
                        printf("texs %d %d: %.9g %.9g %.9g %.9g\n", index, k,
                               double(s[0]), double(s[1]), double(s[2]),
                               double(s[3]));
                    }
                    k++;
                }
            }
        }
        // An RGB fitted to a spectrum and sampled at the four wavelengths, by
        // PBRT's own RGBAlbedoSpectrum -- the table lookup and the sigmoid
        // with nothing of this file's in between -- for the same colours the
        // renderer puts through its per-lookup version. Near-greys on purpose:
        // that is where the table's coefficients change fastest and the last
        // bit of the lookup shows.
        const float colours[][3] = {{0.3f, 0.5f, 0.7f},   {0.7f, 0.5f, 0.3f},
                                    {0.5f, 0.7f, 0.3f},   {0.41f, 0.40f, 0.39f},
                                    {0.2f, 0.2f, 0.21f},  {0.9f, 0.1f, 0.5f},
                                    {0.05f, 0.6f, 0.6f},  {0.33f, 0.33f, 0.34f}};
        int k = 0;
        for (const float *rgb : colours) {
            const pbrt::SampledSpectrum s =
                pbrt::RGBAlbedoSpectrum(*pbrt::RGBColorSpace::sRGB,
                                        pbrt::RGB(rgb[0], rgb[1], rgb[2]))
                    .Sample(lambda);
            printf("sig %d: %.9g %.9g %.9g %.9g\n", k++, double(s[0]),
                   double(s[1]), double(s[2]), double(s[3]));
        }
    }

    // `--print-hits`: for every eighth pixel each way and the first four
    // samples of each, the camera ray PBRT's own sampler and camera make and
    // the distance PBRT's own aggregate finds along it -- infinity for a
    // miss. These three are what PBRT's wavefront seeds a medium's RNG from
    // (`RNG rng(Hash(ray.o, tMax), Hash(ray.d))`, wavefront/media.cpp), so a
    // last bit's difference in any of them is a different scattering
    // decision for that ray, which no image comparison can attribute. The
    // renderer prints the same rows from `hit_at` (render.bonsai), and
    // check_hits.sh wants them identical.
    //
    // The aggregate is built the way pbrt's own render builds it
    // (cpu/render.cpp): media, textures, lights, materials, then the
    // primitives over them.
    if (g_print_hits) {
        std::map<std::string, pbrt::Medium> media = scene.CreateMedia();
        pbrt::NamedTextures textures = scene.CreateTextures();
        std::map<int, pstd::vector<pbrt::Light> *> shape_area_lights;
        std::vector<pbrt::Light> lights =
            scene.CreateLights(textures, &shape_area_lights);
        std::map<std::string, pbrt::Material> named_materials;
        std::vector<pbrt::Material> materials;
        scene.CreateMaterials(textures, &named_materials, &materials);
        const pbrt::Primitive aggregate = scene.CreateAggregate(
            textures, shape_area_lights, media, named_materials, materials);

        const pbrt::Camera camera = scene.GetCamera();
        pbrt::Sampler sampler = scene.GetSampler();
        const pbrt::Film film = camera.GetFilm();
        const pbrt::Filter filter = film.GetFilter();
        const pbrt::Point2i resolution = film.FullResolution();

        // The filter's sampling tables -- the filter tabulated over a `32 *
        // radius` grid, a PiecewiseConstant1D over each row, one over the
        // rows' integrals -- as PBRT's own FilterSampler holds them, through
        // `ToString`, which prints every float in its shortest round-trip
        // form (util/print.cpp, FloatToString). The tables are private, and
        // rebuilding them here from the same source is not the same thing:
        // `filter.Evaluate` inlined into a loop of this file contracts its
        // multiply-adds as gcc sees fit for *that* loop, and two builds of
        // this program disagreed with each other in a sixth of the cells.
        // check_hits.sh reads the rows out of this line; the renderer's
        // `build_filter_table` prints them as rows.
        printf("filtertables: %s\n", filter.ToString().c_str());

        for (int py = 0; py < resolution.y; py += 8) {
            for (int px = 0; px < resolution.x; px += 8) {
                for (int s = 0; s < 4; s++) {
                    // pbrt: RayIntegrator::EvaluatePixelSample, and the
                    // wavefront's GenerateCameraRays -- the wavelength's draw
                    // first, then GetCameraSample's three.
                    const pbrt::Point2i pixel(px, py);
                    sampler.StartPixelSample(pixel, s, 0);
                    const pbrt::Float lu = sampler.Get1D();
                    // Not const: GenerateRay may terminate the secondary
                    // wavelengths of a dispersive camera's ray.
                    pbrt::SampledWavelengths lambda = film.SampleWavelengths(lu);
                    // pbrt: GetCameraSample (samplers.h), written out so the
                    // filter's draw can be printed beside what it became.
                    const pbrt::Point2f u = sampler.GetPixel2D();
                    const pbrt::FilterSample fs = filter.Sample(u);
                    pbrt::CameraSample cs;
                    cs.pFilm = pbrt::Point2f(pixel) + fs.p +
                               pbrt::Vector2f(0.5f, 0.5f);
                    cs.time = sampler.Get1D();
                    cs.pLens = sampler.Get2D();
                    cs.filterWeight = fs.weight;
                    const pstd::optional<pbrt::CameraRay> cr =
                        camera.GenerateRay(cs, lambda);
                    if (!cr) {
                        printf("hit %d %d %d: none\n", px, py, s);
                        continue;
                    }
                    const pstd::optional<pbrt::ShapeIntersection> si =
                        aggregate.Intersect(cr->ray, pbrt::Infinity);
                    const pbrt::Float t = si ? si->tHit : pbrt::Infinity;
                    // The camera sample and the filter's draw after the ray,
                    // so that a direction that differs can be laid at the
                    // sampler's door, the filter's or the camera's.
                    printf("hit %d %d %d: %.9g %.9g %.9g | %.9g %.9g %.9g | "
                           "%.9g | %.9g %.9g | %.9g %.9g | %.9g | %.9g %.9g\n",
                           px, py, s, double(cr->ray.o.x), double(cr->ray.o.y),
                           double(cr->ray.o.z), double(cr->ray.d.x),
                           double(cr->ray.d.y), double(cr->ray.d.z), double(t),
                           double(cs.pFilm.x), double(cs.pFilm.y),
                           double(cs.pLens.x), double(cs.pLens.y),
                           double(cs.time), double(u.x), double(u.y));
                }
            }
        }
    }
}

// PBRT's shapes for a list of this file's, as `Shape::Create` would have made
// them: every triangle in the list as one mesh, because that is what PBRT's
// Triangle refers into, and a sphere placed by its translation. The vertices
// are already in render space, so the mesh's transform is the identity, and
// they are read back out of the scene's own pools rather than carried on the
// triangle.
std::vector<pbrt::Shape> pbrt_shapes(const bonsai_scene::Scene &scene,
                                     const bonsai_scene::Shape *shapes,
                                     size_t count, pbrt::Allocator alloc) {
    std::vector<int> indices;
    std::vector<pbrt::Point3f> points;
    // And the bilinear patches, as one patch mesh over their corners.
    std::vector<int> patch_indices;
    std::vector<pbrt::Point3f> patch_points;
    for (size_t i = 0; i < count; i++) {
        const bonsai_scene::Shape &s = shapes[i];
        if (s.tag == bonsai_scene::ShapeTag::Triangle) {
            uint32_t corner[3];
            scene.corners(s, corner);
            for (const uint32_t c : corner) {
                indices.push_back(int(points.size()));
                points.push_back(pbrt::Point3f(scene.positions[3 * c + 0],
                                               scene.positions[3 * c + 1],
                                               scene.positions[3 * c + 2]));
            }
        } else if (s.tag == bonsai_scene::ShapeTag::Patch) {
            uint32_t corner[4];
            scene.patch_corners(s, corner);
            for (const uint32_t c : corner) {
                patch_indices.push_back(int(patch_points.size()));
                patch_points.push_back(pbrt::Point3f(scene.positions[3 * c + 0],
                                                     scene.positions[3 * c + 1],
                                                     scene.positions[3 * c + 2]));
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
    pstd::vector<pbrt::Shape> patches;
    if (!patch_indices.empty()) {
        pbrt::BilinearPatchMesh *mesh = alloc.new_object<pbrt::BilinearPatchMesh>(
            pbrt::Transform(), /*reverseOrientation=*/false, patch_indices,
            patch_points, std::vector<pbrt::Normal3f>(),
            std::vector<pbrt::Point2f>(), std::vector<int>(), nullptr, alloc);
        patches = pbrt::BilinearPatch::CreatePatches(mesh, alloc);
    }

    std::vector<pbrt::Shape> out;
    out.reserve(count);
    size_t next_triangle = 0;
    size_t next_patch = 0;
    for (size_t i = 0; i < count; i++) {
        const bonsai_scene::Shape &s = shapes[i];
        if (s.tag == bonsai_scene::ShapeTag::Patch) {
            out.push_back(patches[next_patch++]);
        } else if (s.tag == bonsai_scene::ShapeTag::Sphere) {
            // The sphere's own transform, both matrices as PBRT keeps them,
            // and its clipping -- phimax back in degrees, as the constructor
            // takes it; the clamped heights give the constructor the same
            // thetas the given ones did.
            pbrt::Float m[4][4], minv[4][4];
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    m[r][c] = s.render_from_object[4 * r + c];
                    minv[r][c] = s.object_from_render[4 * r + c];
                }
            }
            const pbrt::Transform *render_from_object =
                alloc.new_object<pbrt::Transform>(pbrt::SquareMatrix<4>(m),
                                                  pbrt::SquareMatrix<4>(minv));
            const pbrt::Transform *object_from_render =
                alloc.new_object<pbrt::Transform>(pbrt::SquareMatrix<4>(minv),
                                                  pbrt::SquareMatrix<4>(m));
            out.push_back(alloc.new_object<pbrt::Sphere>(
                render_from_object, object_from_render, s.reverse != 0, s.radius,
                s.z_min, s.z_max, pbrt::Degrees(s.phi_max)));
        } else {
            out.push_back(triangles[next_triangle++]);
        }
    }
    return out;
}

// What a BVHAggregate PBRT built holds, read back through the mirror of its
// private members: the flattened nodes, converted, and the primitives in the
// order its leaves name them.
struct ReadBackTree {
    std::vector<bonsai_scene::Node> nodes;
    std::vector<pbrt::Primitive> ordered;
};

ReadBackTree read_back(const pbrt::BVHAggregate &aggregate,
                       size_t expected_primitives) {
    pbrt::LinearBVHNode *raw = aggregate.*get(NodesTag());
    const std::vector<pbrt::Primitive> &ordered =
        aggregate.*get(PrimitivesTag());
    const MirroredNode *built = reinterpret_cast<const MirroredNode *>(raw);
    if (!built || ordered.size() != expected_primitives) {
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
    if (total_prims != expected_primitives) {
        fail("pbrt's bvh does not reach every primitive");
    }

    ReadBackTree out;
    out.nodes.reserve(count);
    for (size_t i = 0; i < count; i++) {
        const MirroredNode &n = built[i];
        bonsai_scene::Node node;
        node.low[0] = float(n.bounds.pMin.x);
        node.low[1] = float(n.bounds.pMin.y);
        node.low[2] = float(n.bounds.pMin.z);
        node.high[0] = float(n.bounds.pMax.x);
        node.high[1] = float(n.bounds.pMax.y);
        node.high[2] = float(n.bounds.pMax.z);
        node.n_prims = n.n_primitives;
        node.axis = n.n_primitives > 0 ? 0 : n.axis;
        // PBRT stores the second child absolutely; the renderer's layout wants
        // it relative, because that is what its `right = index + offset` says.
        node.offset = n.n_primitives > 0
                          ? uint32_t(n.primitives_offset)
                          : uint32_t(n.second_child_offset) - uint32_t(i);
        out.nodes.push_back(node);
    }
    out.ordered = ordered;
    return out;
}

// Build the trees with PBRT's own BVHAggregate, over the shapes and instances
// this scene produced, and read back what it built. `BasicScene::
// CreateAggregate`'s arrangement exactly: a tree per instance definition over
// its shapes, then the top-level tree over the top-level shapes and a
// TransformedPrimitive per instance, mixed.
//
// The primitives go in in our order and each one's address is remembered, so
// the permutation the build settled on can be recovered by identity rather
// than by matching geometry. A definition's shapes come back reordered within
// their run, which is what lets its leaves name contiguous runs; the top-level
// order mixes shapes and instances and is written down as `prims` instead.
//
// PBRT builds no tree over a definition holding one shape -- the shape is the
// definition's primitive -- where this always builds one, of a single leaf.
// The renderer's layout wants a row to start an instance's walk at, and a
// one-leaf tree costs that instance one box test more than PBRT spends on it.
void build_pbrt_tree(bonsai_scene::Scene &scene) {
    pbrt::Allocator alloc;

    // Each definition's tree, into one pool, and the Primitive PBRT would hand
    // its instances.
    scene.instance_nodes.clear();
    std::vector<pbrt::Primitive> definition_prims;
    for (bonsai_scene::Definition &def : scene.definitions) {
        if (def.shape_count == 0) {
            definition_prims.push_back(pbrt::Primitive());
            continue;
        }
        bonsai_scene::Shape *run = scene.instance_shapes.data() + def.first_shape;
        const std::vector<pbrt::Shape> shapes =
            pbrt_shapes(scene, run, def.shape_count, alloc);
        std::vector<pbrt::Primitive> primitives;
        std::map<const void *, uint32_t> index_of;
        for (uint32_t i = 0; i < def.shape_count; i++) {
            pbrt::Primitive prim = alloc.new_object<pbrt::SimplePrimitive>(
                shapes[i], pbrt::Material());
            index_of[prim.ptr()] = i;
            primitives.push_back(prim);
        }
        pbrt::BVHAggregate *aggregate = alloc.new_object<pbrt::BVHAggregate>(
            primitives, 4, pbrt::BVHAggregate::SplitMethod::SAH);
        ReadBackTree tree = read_back(*aggregate, def.shape_count);

        std::vector<bonsai_scene::Shape> reordered;
        reordered.reserve(def.shape_count);
        for (const pbrt::Primitive &prim : tree.ordered) {
            const auto it = index_of.find(prim.ptr());
            if (it == index_of.end()) {
                fail("pbrt's bvh holds a primitive this scene did not put in "
                     "it");
            }
            reordered.push_back(run[it->second]);
        }
        std::copy(reordered.begin(), reordered.end(), run);

        // The definition's rows in the shared pool. A child offset is relative
        // to its parent, so it survives the move; a leaf's first shape is an
        // index into `instance_shapes`, so it gets the run's start added.
        def.root_node = uint32_t(scene.instance_nodes.size());
        for (bonsai_scene::Node node : tree.nodes) {
            if (node.n_prims > 0) {
                node.offset += def.first_shape;
            }
            scene.instance_nodes.push_back(node);
        }
        definition_prims.push_back(aggregate);
    }

    // The top-level tree, over the shapes and the instances together.
    std::vector<bonsai_scene::Shape> &shapes = scene.shapes;
    const std::vector<pbrt::Shape> top_shapes =
        pbrt_shapes(scene, shapes.data(), shapes.size(), alloc);
    std::vector<pbrt::Primitive> primitives;
    std::map<const void *, bonsai_scene::Prim> prim_of;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        pbrt::Primitive prim = alloc.new_object<pbrt::SimplePrimitive>(
            top_shapes[i], pbrt::Material());
        prim_of[prim.ptr()] = bonsai_scene::Prim{bonsai_scene::PrimShape, i};
        primitives.push_back(prim);
    }
    for (uint32_t i = 0; i < scene.instances.size(); i++) {
        const bonsai_scene::Instance &inst = scene.instances[i];
        pbrt::SquareMatrix<4> m, mi;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                m[r][c] = inst.render_from_instance[4 * r + c];
                mi[r][c] = inst.instance_from_render[4 * r + c];
            }
        }
        const pbrt::Transform *render_from_instance =
            alloc.new_object<pbrt::Transform>(m, mi);
        pbrt::Primitive prim = alloc.new_object<pbrt::TransformedPrimitive>(
            definition_prims[inst.definition], render_from_instance);
        prim_of[prim.ptr()] = bonsai_scene::Prim{bonsai_scene::PrimInstance, i};
        primitives.push_back(prim);
    }

    // PBRT's default maxnodeprims, and the split method its `bvh` accelerator
    // uses unless a scene says otherwise.
    pbrt::BVHAggregate aggregate(primitives, 4,
                                 pbrt::BVHAggregate::SplitMethod::SAH);
    ReadBackTree tree = read_back(aggregate, primitives.size());
    scene.nodes = std::move(tree.nodes);
    scene.prims.clear();
    scene.prims.reserve(tree.ordered.size());
    for (const pbrt::Primitive &prim : tree.ordered) {
        const auto it = prim_of.find(prim.ptr());
        if (it == prim_of.end()) {
            fail("pbrt's bvh holds a primitive this scene did not put in it");
        }
        scene.prims.push_back(it->second);
    }
}

} // namespace

int main(int argc, char **argv) {
    // --pbrt-tree dumps the BVH PBRT built alongside the geometry, and the
    // renderer then traverses that rather than building its own. It is how a
    // timing comparison is made to be about the traversal the schedule
    // produced rather than about whose builder found a better tree.
    bool pbrt_tree = false;
    bool tables_only = false;
    bool sampler_only = false;
    bool noise_only = false;
    bool bsdf_only = false;
    bool shading_only = false;
    bool light_only = false;
    bool shape_sample_only = false;
    // There was a `--reference` here, which rendered the gbuffer with PBRT's
    // own camera, aggregate and BSDFs driven from a loop written in this file.
    // It is gone, and what replaced it is the `pbrt` binary: compare.sh runs
    // it and compares against what it writes.
    //
    // The loop was faithful -- every value it produced agreed with the binary
    // to a part in a million -- and it was still the wrong thing to have. It
    // differed from PBRT's own `RenderCPU` in two ways that were found by
    // reading it rather than by any check failing: it traced the camera ray
    // twice, and it never called `ScaleDifferentials`. Neither showed, because
    // the first costs only time and the second had nothing to affect yet. A
    // reference that can drift without saying so is not a reference.
    //
    // What this program does now is convert a scene and print what PBRT thinks
    // about small things -- the tables, the sampler, a BSDF. It renders
    // nothing.

    // --spp is PBRT's own option under PBRT's own name, and it reaches both
    // sides the same way PBRT's does: by being set on PBRTOptions before the
    // scene is parsed, so that every sampler's Create reads it. A scene here
    // says how many samples it wants and that number is part of a comparison's
    // meaning, so this is for looking at pictures rather than for the numbers
    // -- 64 samples with no pixel jitter is a noisy image and the scenes are
    // sized for a comparison that has to finish.
    int spp_override = 0; // 0 for "the scene decides", as PBRT's unset does.
    bool disable_pixel_jitter = false;
    std::vector<const char *> positional;
    for (int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--pbrt-tree") {
            pbrt_tree = true;
        } else if (arg == "--check-tables") {
            tables_only = true;
        } else if (arg == "--print-sampler") {
            sampler_only = true;
        } else if (arg == "--print-noise") {
            noise_only = true;
        } else if (arg == "--print-bsdf") {
            bsdf_only = true;
        } else if (arg == "--print-shading") {
            shading_only = true;
        } else if (arg == "--print-light") {
            light_only = true;
        } else if (arg == "--print-shape-sample") {
            shape_sample_only = true;
        } else if (arg == "--print-differentials") {
            // Not a `*_only` mode: it needs a parsed scene, so it rides along
            // with a normal conversion and prints from inside `load`.
            g_print_differentials = true;
        } else if (arg == "--print-hits") {
            // Likewise: PBRT's aggregate is built from the parsed scene.
            g_print_hits = true;
        } else if (arg == "--disable-pixel-jitter") {
            disable_pixel_jitter = true;
        } else if (arg == "--spp") {
            if (i + 1 >= argc) {
                fail("--spp needs a sample count");
            }
            const int n = atoi(argv[++i]);
            if (n < 1) {
                fail("--spp needs a positive sample count");
            }
            spp_override = n;
        } else if (arg == "--maxdepth") {
            if (i + 1 >= argc) {
                fail("--maxdepth needs a depth");
            }
            const int n = atoi(argv[++i]);
            if (n < 1) {
                fail("--maxdepth needs a positive depth");
            }
            g_max_depth_override = n;
        } else if (arg == "--repeats") {
            // Accepted and ignored: it timed the reference render that used to
            // live here, and compare.sh still passes it. The repeats that
            // matter now are the binary's, which compare.sh runs itself.
            if (i + 1 >= argc) {
                fail("--repeats needs a count");
            }
            i++;
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (!tables_only && !sampler_only && !noise_only && !bsdf_only &&
        !shading_only && !light_only && !shape_sample_only &&
        positional.size() != 2) {
        fail("usage: scene_dump [--pbrt-tree] [--spp <n>] [--maxdepth <n>]"
             " [--disable-pixel-jitter] <scene.pbrt> <out.txt>\n"
             "       scene_dump --check-tables\n"
             "       scene_dump --print-sampler\n"
             "       scene_dump --print-noise\n"
             "       scene_dump --print-bsdf\n"
             "       scene_dump --print-shading");
    }

    pbrt::PBRTOptions options;
    // The pixel jitter is *on*, as PBRT has it. It used to be forced off here
    // and could not be anything else: the renderer had no reconstruction
    // filter, so it had no way to place a sample anywhere but a pixel's centre,
    // and the reference had to be crippled to match. Now both sides sample
    // PBRT's Gaussian, and `--disable-pixel-jitter` -- PBRT's own flag, under
    // PBRT's own name -- is what asks for the old behaviour.
    //
    // It is still worth asking for. With every sample of a pixel on the same
    // ray, a difference between the two images cannot be noise, so the gbuffer
    // comparison becomes a question about the geometry alone. That is how the
    // normals got to zero disagreeing pixels.
    //
    // Set before parsing because it is read while the scene is built.
    options.disablePixelJitter = disable_pixel_jitter;
    // PBRT's `--spp`, in the place PBRT puts it. Every sampler's Create reads
    // `Options->pixelSamples` and so does the conversion in `load`, so the
    // reference render and the scene the renderer is given cannot disagree
    // about how many samples were asked for.
    if (spp_override > 0) {
        options.pixelSamples = spp_override;
    }
    pbrt::InitPBRT(options);

    if (tables_only) {
        const bool ok = check_tables();
        pbrt::CleanupPBRT();
        return ok ? 0 : 1;
    }
    if (sampler_only) {
        print_sampler();
        pbrt::CleanupPBRT();
        return 0;
    }
    if (noise_only) {
        print_noise();
        pbrt::CleanupPBRT();
        return 0;
    }
    if (bsdf_only) {
        print_bsdf();
        pbrt::CleanupPBRT();
        return 0;
    }
    if (shading_only) {
        print_shading();
        pbrt::CleanupPBRT();
        return 0;
    }
    if (light_only) {
        print_light();
        pbrt::CleanupPBRT();
        return 0;
    }
    if (shape_sample_only) {
        print_shape_sample();
        pbrt::CleanupPBRT();
        return 0;
    }

    bonsai_scene::Scene scene;
    load(positional[0], scene);
    if (pbrt_tree) {
        build_pbrt_tree(scene);
    }

    pbrt::CleanupPBRT();

    if (!bonsai_scene::write(positional[1], scene)) {
        fail(std::string("cannot write ") + positional[1]);
    }

    printf("scene_dump: %s -> %s (%ux%u, %zu shapes", positional[0],
           positional[1], scene.width, scene.height, scene.shapes.size());
    if (!scene.instances.empty()) {
        printf(", %zu instances of %zu objects holding %zu shapes",
               scene.instances.size(), scene.definitions.size(),
               scene.instance_shapes.size());
    }
    if (pbrt_tree) {
        printf(", %zu nodes from pbrt's bvh", scene.nodes.size());
        if (!scene.instance_nodes.empty()) {
            printf(" and %zu in the instances' trees",
                   scene.instance_nodes.size());
        }
    }
    printf(")\n");
    return 0;
}
