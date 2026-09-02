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

#include <pbrt/base/bxdf.h>
#include <pbrt/base/material.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/integrators.h>
#include <pbrt/cpu/primitive.h>
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
#include <pbrt/util/math.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include "cie_tables.h"

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

    void Sampler(const std::string &name, pbrt::ParsedParameterVector params,
                 pbrt::FileLoc loc) override {
        sampler_name = name;
        sampler_params = pbrt::ParameterDictionary(
            pbrt::ParsedParameterVector(params), pbrt::RGBColorSpace::sRGB);
        pbrt::BasicSceneBuilder::Sampler(name, std::move(params), loc);
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
        };
        std::string name;
        std::map<std::string, Value> params;

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
            info.params.emplace(p->name, std::move(v));
        }
        materials.push_back(std::move(info));
        pbrt::BasicSceneBuilder::Material(name, std::move(params), loc);
    }

    // The integrator, for its `maxdepth` alone.
    //
    // Which integrator the scene names is deliberately ignored: this renderer
    // implements pbrt's RandomWalkIntegrator and the comparison runs pbrt's
    // too, so that both sides answer the same question. What the scene can
    // still say is how deep to go.
    void Integrator(const std::string &name, pbrt::ParsedParameterVector params,
                    pbrt::FileLoc loc) override {
        integrator_name = name;
        for (const pbrt::ParsedParameter *p : params) {
            if (p->name == "maxdepth" && !p->ints.empty()) {
                integrator_max_depth = int(p->ints[0]);
            }
        }
        pbrt::BasicSceneBuilder::Integrator(name, std::move(params), loc);
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
    pbrt::ParameterDictionary film_params;
    std::string sampler_name = "zsobol";
    pbrt::ParameterDictionary sampler_params;
    std::vector<MaterialInfo> materials;
    std::vector<MaterialInfo> area_lights;
    // PBRT's RandomWalkIntegrator default. The name is empty when the scene
    // named no integrator, which is not the same as naming the default: PBRT
    // would fall back to volpath, and this renderer has only the random walk,
    // so the two cases are told apart where the scene is converted.
    std::string integrator_name;
    int integrator_max_depth = 5;
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

// The material a shape was declared under, in the form the renderer reads.
//
// The defaults are PBRT's own, from DiffuseMaterial::Create and
// CoatedDiffuseMaterial::Create, because a material that names nothing has to
// arrive as the one PBRT would have built rather than as a guess.
bonsai_scene::Material
convert_material(const std::vector<CapturingBuilder::MaterialInfo> &materials,
                 int index) {
    bonsai_scene::Material out;
    // What a shape declared outside any Material directive gets.
    if (index < 0) {
        return out;
    }
    if (index >= int(materials.size())) {
        fail("a shape names a material that was never declared");
    }
    const CapturingBuilder::MaterialInfo &m = materials[index];
    if (m.name.empty() || m.name == "none") {
        return out;
    }

    if (m.name == "diffuse") {
        out.tag = bonsai_scene::MaterialTag::Diffuse;
        material_rgb(m, "reflectance", out.reflectance);
        return out;
    }

    if (m.name == "coateddiffuse") {
        out.tag = bonsai_scene::MaterialTag::CoatedDiffuse;
        material_rgb(m, "reflectance", out.reflectance);
        // PBRT takes `uroughness` and `vroughness` where they are given and
        // falls back to `roughness` for each independently, which is not the
        // same as falling back to `roughness` only when neither is given.
        const float roughness = material_float(m, "roughness", 0.f);
        out.u_roughness = material_float(m, "uroughness", roughness);
        out.v_roughness = material_float(m, "vroughness", roughness);
        out.thickness = material_float(m, "thickness", 0.01f);
        out.g = material_float(m, "g", 0.f);
        // `eta` is a spectrum in PBRT unless the scene writes it as a bare
        // float, and a spectral one terminates the secondary wavelengths --
        // which changes what every later stage of the render integrates over.
        const CapturingBuilder::MaterialInfo::Value *eta = m.find("eta");
        if (eta != nullptr) {
            if (eta->type != "float" || eta->floats.size() != 1) {
                fail("only a scalar `float eta` is supported on coateddiffuse, "
                     "not a named spectrum -- a spectral index terminates the "
                     "secondary wavelengths, which nothing here does");
            }
            out.eta = eta->floats[0];
        }
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
        if (m.find("displacement") != nullptr ||
            m.find("normalmap") != nullptr) {
            fail("a displacement or normal map changes the shading frame, "
                 "which this renderer takes from the geometry alone");
        }
        return out;
    }

    fail("only the diffuse and coateddiffuse materials are supported, scene "
         "asks for \"" +
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
    return ok;
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

    // The first ten primes, so a mistranscribed table is caught here rather
    // than as a wrong answer three functions later.
    printf("primes:");
    for (int i = 0; i < 10; i++) {
        printf(" %d", pbrt::Primes[i]);
    }
    printf(" ... %d\n", pbrt::Primes[pbrt::PrimeTableSize - 1]);
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
        const pbrt::TriQuadMesh ply = pbrt::TriQuadMesh::ReadPLY(file);
        // A PLY may hold quads as well as triangles, and PBRT makes those a
        // bilinear patch mesh rather than splitting them -- a bilinear patch is
        // not two triangles unless it happens to be planar. Refusing is the
        // honest answer until the renderer has the shape.
        if (!ply.quadIndices.empty()) {
            fail("the PLY file " + file +
                 " contains quads, which are bilinear patches in pbrt rather "
                 "than pairs of triangles");
        }
        if (ply.triIndices.empty()) {
            fail("the PLY file " + file + " has no triangles");
        }
        return alloc.new_object<pbrt::TriangleMesh>(
            *entity.renderFromObject, entity.reverseOrientation, ply.triIndices,
            ply.p, std::vector<pbrt::Vector3f>(), ply.n, ply.uv,
            ply.faceIndices, alloc);
    }

    return nullptr;
}

// The gbuffer PBRT would have written, rendered here with PBRT's own code.
//
// The comparison used to be against the `pbrt` binary's output, which meant the
// scene had to ask for `Film "gbuffer"` -- and no scene anyone else wrote does.
// They say `Film "rgb"`, and PBRT has a command-line override for the sample
// count but none for the film, so the comparison simply could not be run on
// them. Rewriting someone's scene to change one line is not a fix; asking PBRT
// directly is.
//
// Nothing here is a reimplementation. The camera, the sampler, the aggregate,
// the intersection and the BSDF are PBRT's, assembled the way RenderCPU
// assembles them and driven the way the path integrator drives them at depth
// zero. What is left out is everything past the first hit, which is the part
// the gbuffer does not record.
//
// A consequence worth having: `bsdf.rho` is PBRT's, so the albedo is right for
// every material PBRT has, whether or not this renderer can yet reproduce it --
// and the normals, which no material affects, compare exactly regardless.
//
// The normals are in render space, which is where the rays are. PBRT's gbuffer
// film writes them in whichever space `coordinatesystem` names, and matching
// that used to mean every scene here saying "world"; a scene someone else wrote
// says nothing and gets "camera". Reporting the space both renderers actually
// work in removes the question rather than answering it.
void render_reference(pbrt::BasicScene &scene,
                      const pbrt::RGBColorSpace *colour_space,
                      int integrator_max_depth, int repeats,
                      const std::string &prefix) {
    // Sampled at the pixel centre, which main() asked for before parsing began.
    // GetCameraSample below is PBRT's own and reads PBRT's own option, so there
    // is no second implementation of what that flag means.
    std::map<std::string, pbrt::Medium> media = scene.CreateMedia();
    pbrt::NamedTextures textures = scene.CreateTextures();
    std::map<int, pstd::vector<pbrt::Light> *> shape_index_to_area_lights;
    std::vector<pbrt::Light> lights =
        scene.CreateLights(textures, &shape_index_to_area_lights);
    std::map<std::string, pbrt::Material> named_materials;
    std::vector<pbrt::Material> materials;
    scene.CreateMaterials(textures, &named_materials, &materials);
    pbrt::Primitive accel = scene.CreateAggregate(
        textures, shape_index_to_area_lights, media, named_materials, materials);

    pbrt::Camera camera = scene.GetCamera();
    pbrt::Film film = camera.GetFilm();
    pbrt::Filter filter = film.GetFilter();
    pbrt::Sampler sampler = scene.GetSampler();
    const pbrt::Bounds2i bounds = film.PixelBounds();
    const pbrt::Vector2i extent = bounds.Diagonal();
    const int width = extent.x, height = extent.y;
    const int spp = sampler.SamplesPerPixel();

    std::vector<float> normals(size_t(width) * height * 3, 0.f);
    std::vector<float> albedos(size_t(width) * height * 3, 0.f);
    std::vector<float> radiances(size_t(width) * height * 3, 0.f);

    // PBRT's own RandomWalkIntegrator, which is what render.bonsai's
    // `li_random_walk` is a transcription of. Constructed here rather than
    // taken from the scene: the scene names whatever integrator it was written
    // for -- killeroo-simple names none, so PBRT would default to volpath --
    // and what this file is for is asking PBRT the same question the renderer
    // answers, not a harder one. The depth is the scene's if it named one and
    // PBRT's default of five otherwise, which is the same number `load` writes
    // into the scene file for the renderer to use.
    pbrt::RandomWalkIntegrator reference_integrator(integrator_max_depth,
                                                    camera, sampler, accel,
                                                    lights);

    pbrt::ThreadLocal<pbrt::ScratchBuffer> buffers(
        []() { return pbrt::ScratchBuffer(); });
    pbrt::ThreadLocal<pbrt::Sampler> samplers(
        [&sampler]() { return sampler.Clone({}); });

    // Timed, and this is the number worth comparing against. The pbrt binary
    // renders this scene with whatever integrator the scene names -- none, for
    // killeroo-simple, so PBRT's default of volpath -- into an rgb film, which
    // computes no VisibleSurface and no reflectance. That is three differences
    // at once from what the renderer does, and none of them is the schedule.
    //
    // What is below is the same work: PBRT's own intersection, PBRT's own rho,
    // PBRT's own RandomWalkIntegrator, over the same samples of the same
    // pixels. Best of `repeats`, as the other side is, and for the same reason:
    // every source of noise adds time and none removes it.
    double seconds = std::numeric_limits<double>::infinity();
    for (int run = 0; run < repeats; run++) {
        const auto started = std::chrono::steady_clock::now();
        std::fill(normals.begin(), normals.end(), 0.f);
        std::fill(albedos.begin(), albedos.end(), 0.f);
        std::fill(radiances.begin(), radiances.end(), 0.f);
    pbrt::ParallelFor2D(bounds, [&](pbrt::Bounds2i tile) {
        pbrt::ScratchBuffer &scratch = buffers.Get();
        pbrt::Sampler tile_sampler = samplers.Get();
        for (pbrt::Point2i p : tile) {
            // The accumulator widths are PBRT's, from GBufferFilm::Pixel: the
            // colour sums are double and the normal sum is not. A float sum
            // here would agree to about six digits and disagree in the last
            // bit of the half the film writes, which is a difference nobody
            // could explain from the source.
            pbrt::Normal3f n_sum(0, 0, 0);
            double albedo_sum[3] = {0., 0., 0.};
            double radiance_sum[3] = {0., 0., 0.};
            double weight_sum = 0.;

            for (int i = 0; i < spp; i++) {
                scratch.Reset();
                tile_sampler.StartPixelSample(p, i);

                // The order the path integrator draws in: the wavelength
                // first, then the camera sample. Getting it wrong would put
                // every later draw one value out.
                pbrt::Float lu = tile_sampler.Get1D();
                if (pbrt::GetOptions().disableWavelengthJitter) {
                    lu = 0.5f;
                }
                pbrt::SampledWavelengths lambda = film.SampleWavelengths(lu);
                pbrt::CameraSample cs =
                    pbrt::GetCameraSample(tile_sampler, p, filter);
                pstd::optional<pbrt::CameraRayDifferential> cr =
                    camera.GenerateRayDifferential(cs, lambda);
                weight_sum += cs.filterWeight;
                if (!cr) {
                    continue;
                }

                // The radiance, from PBRT's own integrator and drawing from
                // this sample's stream exactly where the renderer's walk draws
                // from it. Before the gbuffer work below, because that is where
                // an integrator runs: PBRT fills a VisibleSurface from inside
                // `Li`, and every draw after the camera sample belongs to the
                // walk.
                //
                // Converted the way PixelSensor does and not the way a
                // reflectance is: XYZ without the division by the CIE Y
                // integral that `ToRGB` applies, which is why the multiply
                // undoing it is here. See colour.bonsai.
                {
                    pbrt::RayDifferential ray = cr->ray;
                    const pbrt::SampledSpectrum L = reference_integrator.Li(
                        ray, lambda, tile_sampler, scratch, nullptr);
                    const pbrt::XYZ xyz =
                        L.ToXYZ(lambda) * pbrt::CIE_Y_integral;
                    const pbrt::RGB rgb = colour_space->ToRGB(xyz);
                    for (int c = 0; c < 3; c++) {
                        radiance_sum[c] += cs.filterWeight * rgb[c];
                    }
                }

                pstd::optional<pbrt::ShapeIntersection> si =
                    accel.Intersect(cr->ray, pbrt::Infinity);
                if (!si) {
                    continue;
                }
                pbrt::SurfaceInteraction &isect = si->intr;
                pbrt::BSDF bsdf = isect.GetBSDF(cr->ray, lambda, camera,
                                                scratch, tile_sampler);
                if (!bsdf) {
                    continue;
                }

                // pbrt: VisibleSurface's constructor, which turns the normal to
                // face the way the ray came from -- the film's convention
                // rather than the geometry's winding.
                const pbrt::Normal3f n =
                    pbrt::FaceForward(isect.n, isect.wo);
                n_sum += cs.filterWeight * n;

                // pbrt: GBufferFilm::AddSample, which lights the reflectance by
                // the colour space's illuminant before converting it, so that
                // the answer is a colour under a white point rather than a
                // reflectance under an equal-energy light.
                const pbrt::SampledSpectrum rho =
                    bsdf.rho(isect.wo, kRhoUC, kRhoU);
                const pbrt::SampledSpectrum lit =
                    rho * colour_space->illuminant.Sample(lambda);
                const pbrt::RGB rgb = lit.ToRGB(lambda, *colour_space);
                for (int c = 0; c < 3; c++) {
                    albedo_sum[c] += cs.filterWeight * rgb[c];
                }
            }

            const size_t at =
                (size_t(p.y - bounds.pMin.y) * width + (p.x - bounds.pMin.x)) * 3;
            // pbrt: GetImage. The normal is normalized rather than averaged --
            // a direction has no magnitude to average -- while the albedo is
            // divided by the whole weight, including the samples that hit
            // nothing, so a pixel on a silhouette comes out darker.
            if (pbrt::LengthSquared(n_sum) > 0) {
                const pbrt::Normal3f n = pbrt::Normalize(n_sum);
                normals[at + 0] = float(n.x);
                normals[at + 1] = float(n.y);
                normals[at + 2] = float(n.z);
            }
            if (weight_sum != 0) {
                for (int c = 0; c < 3; c++) {
                    albedos[at + c] = float(albedo_sum[c] / weight_sum);
                    radiances[at + c] = float(radiance_sum[c] / weight_sum);
                }
            }
        }
    });
        const auto finished = std::chrono::steady_clock::now();
        seconds = std::min(
            seconds, std::chrono::duration<double>(finished - started).count());
    }

    const auto write = [&](const std::string &path,
                           const std::vector<float> &pixels) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            fail("cannot open " + path + " for writing");
        }
        // PFM rows run bottom to top, and a negative scale says little-endian.
        out << "PF\n" << width << ' ' << height << "\n-1.000000\n";
        for (int y = height - 1; y >= 0; y--) {
            out.write(reinterpret_cast<const char *>(
                          pixels.data() + size_t(y) * width * 3),
                      std::streamsize(sizeof(float)) * width * 3);
        }
        if (!out) {
            fail("cannot write " + path);
        }
    };
    write(prefix + ".pfm", normals);
    write(prefix + "-albedo.pfm", albedos);
    write(prefix + "-radiance.pfm", radiances);
    printf("scene_dump: reference %s.pfm, %s-albedo.pfm and %s-radiance.pfm "
           "(%dx%d, %d spp, randomwalk maxdepth %d)\n",
           prefix.c_str(), prefix.c_str(), prefix.c_str(), width, height, spp,
           integrator_max_depth);
    // Parsed by compare.sh. On a line of its own, as the renderer's is.
    printf("scene_dump: reference seconds: %g\n", seconds);
    // What PBRT's own binary would write this scene to, so that a comparison
    // script can find the image it timed. A scene names it and no two need
    // agree.
    printf("scene_dump: film filename %s\n",
           film.GetFilename().c_str());
}

// Parse and convert. Everything PBRT owns is local to this function, so all of
// it is destroyed on the way out -- before CleanupPBRT takes the arenas it was
// allocated from out from under it. Doing this inline in main instead crashes
// on the way out, because the locals outlive the cleanup call.
void load(const char *filename, bonsai_scene::Scene &out,
          const std::string &reference_prefix, int repeats) {
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

    // The three samplers the renderer reproduces. Sobol and zsobol are a
    // different construction again and are refused rather than approximated: a
    // stand-in would give noise that is not pbrt's while looking perfectly
    // reasonable, which is the failure worth refusing.
    //
    // The defaults below are pbrt's own, from IndependentSampler::Create and
    // StratifiedSampler::Create; a scene that names a sampler without naming
    // its parameters has to get the same ones pbrt would have given it.
    if (builder.sampler_name == "independent") {
        out.sampler.tag = bonsai_scene::SamplerTag::Independent;
        out.sampler.samples_per_pixel =
            uint32_t(builder.sampler_params.GetOneInt("pixelsamples", 4));
    } else if (builder.sampler_name == "halton") {
        out.sampler.tag = bonsai_scene::SamplerTag::Halton;
        out.sampler.samples_per_pixel =
            uint32_t(builder.sampler_params.GetOneInt("pixelsamples", 16));
        const std::string randomization =
            builder.sampler_params.GetOneString("randomization",
                                                "permutedigits");
        if (randomization == "none") {
            out.sampler.randomize = bonsai_scene::RandomizeTag::RandomizeNone;
        } else if (randomization == "permutedigits") {
            out.sampler.randomize =
                bonsai_scene::RandomizeTag::RandomizePermuteDigits;
        } else if (randomization == "owen") {
            out.sampler.randomize = bonsai_scene::RandomizeTag::RandomizeOwen;
        } else {
            // "fastowen" is what PBRT itself refuses for this sampler.
            fail("unknown Halton randomization \"" + randomization + "\"");
        }
        halton_scales(x_resolution, y_resolution, out.sampler);
    } else if (builder.sampler_name == "stratified") {
        out.sampler.tag = bonsai_scene::SamplerTag::Stratified;
        out.sampler.x_samples =
            uint32_t(builder.sampler_params.GetOneInt("xsamples", 4));
        out.sampler.y_samples =
            uint32_t(builder.sampler_params.GetOneInt("ysamples", 4));
        out.sampler.jitter =
            builder.sampler_params.GetOneBool("jitter", true) ? 1u : 0u;
        out.sampler.samples_per_pixel =
            out.sampler.x_samples * out.sampler.y_samples;
    } else {
        fail("only the independent, stratified and halton samplers are "
             "supported, scene asks for \"" +
             builder.sampler_name + "\"");
    }
    out.sampler.seed = builder.sampler_params.GetOneInt("seed", 0);
    // Not the sampler's seed: PBRT's global `--seed`, which a layered BSDF
    // hashes with the direction it was asked about. Read from PBRT rather than
    // written down, so that running scene_dump with one has the effect it has
    // on PBRT.
    out.seed = pbrt::GetOptions().seed;
    // The same depth the reference render uses, so that the two integrators are
    // asked to go equally far.
    out.max_depth = builder.integrator_max_depth;
    // Which integrator, refused rather than substituted. A scene that names
    // `volpath` and gets a random walk is an image that answers a question
    // nobody asked, and it would look plausible -- which is worse than an
    // error. A scene naming none is the one exception: PBRT's default is
    // volpath, but the comparison renders both sides with the integrator this
    // renderer has, so there is nothing to disagree about.
    if (!builder.integrator_name.empty() &&
        builder.integrator_name != "randomwalk") {
        fail("this renderer implements `randomwalk`, and the scene asks for `" +
             builder.integrator_name + "`");
    }
    out.integrator = bonsai_scene::IntegratorTag::RandomWalk;

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
        const uint32_t at = uint32_t(out.materials.size());
        out.materials.push_back(convert_material(builder.materials, declared));
        material_index.emplace(declared, at);
        return at;
    };

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
        // The default when a scene names no L is the colour space's own
        // illuminant, which for sRGB is D65 -- and an RGB of one puts the fit
        // through the same path, since that is what the fit of a flat
        // illuminant is.
        if (!material_rgb(info, "L", light.l)) {
            light.l[0] = light.l[1] = light.l[2] = 1.f;
        }
        // PBRT's own scale, divided by the photometric integral of L so that a
        // radiance of one means one nit. The spectrum that division is over is
        // the illuminant PBRT would have built from this RGB, so it is built
        // here the same way.
        pbrt::Allocator alloc;
        const pbrt::RGBIlluminantSpectrum emitted(
            *pbrt::RGBColorSpace::sRGB,
            pbrt::RGB(light.l[0], light.l[1], light.l[2]));
        light.scale =
            float(material_float(info, "scale", 1.f) /
                  pbrt::SpectrumToPhotometric(&emitted));
        const CapturingBuilder::MaterialInfo::Value *two =
            info.find("twosided");
        light.two_sided =
            (two != nullptr && !two->bools.empty() && two->bools[0]) ? 1u : 0u;

        const int32_t at = int32_t(out.lights.size());
        out.lights.push_back(light);
        light_index.emplace(declared, at);
        return at;
    };

    for (const pbrt::ShapeSceneEntity &entity : scene.shapes) {
        const std::string name(entity.name);
        const pbrt::Transform &render_from_object = *entity.renderFromObject;
        const uint32_t material = material_for(entity.materialIndex);
        const int32_t light = light_for(entity.lightIndex);

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
            shape.center[0] = float(centre.x);
            shape.center[1] = float(centre.y);
            shape.center[2] = float(centre.z);
            shape.radius = float(radius);
            // A translation never swaps handedness, so this is the scene's
            // ReverseOrientation alone.
            shape.flip = entity.reverseOrientation ? 1u : 0u;
            shape.material = material;
            shape.light = light;
            shapes.push_back(shape);

        } else if (const pbrt::TriangleMesh *mesh = triangulate(entity)) {
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
            for (int i = 0; i < mesh->nTriangles; i++) {
                bonsai_scene::Shape shape;
                shape.tag = bonsai_scene::ShapeTag::Triangle;
                shape.mesh = mesh_index;
                shape.tri = uint32_t(i);
                shape.material = material;
                shape.light = light;
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

    // Last, because CreateAggregate empties `scene.shapes` on its way through
    // -- it hands the entities to the primitives it builds -- and the
    // conversion above reads them.
    if (!reference_prefix.empty()) {
        // The film's colour space rather than sRGB by assumption: it is what
        // GBufferFilm converts the albedo with, and a scene can name another.
        render_reference(scene, builder.film_params.ColorSpace(),
                         builder.integrator_max_depth, repeats,
                         reference_prefix);
    }
}

// Build the tree with PBRT's own BVHAggregate, over the shapes this scene
// produced, and read back what it built.
//
// The primitives go in in our order and each one's address is remembered, so
// the permutation the build settled on can be recovered by identity rather
// than by matching geometry. `shapes` comes back reordered to match, which is
// what lets a leaf name a contiguous run.
void build_pbrt_tree(bonsai_scene::Scene &scene) {
    std::vector<bonsai_scene::Shape> &shapes = scene.shapes;
    std::vector<bonsai_scene::Node> &nodes = scene.nodes;
    pbrt::Allocator alloc;

    // Every triangle in the scene as one mesh, because that is what PBRT's
    // Triangle refers into. The vertices are already in render space, so the
    // mesh's transform is the identity, and they are read back out of the
    // scene's own pools rather than carried on the triangle.
    std::vector<int> indices;
    std::vector<pbrt::Point3f> points;
    for (const bonsai_scene::Shape &s : shapes) {
        if (s.tag == bonsai_scene::ShapeTag::Triangle) {
            uint32_t corner[3];
            scene.corners(s, corner);
            for (const uint32_t c : corner) {
                indices.push_back(int(points.size()));
                points.push_back(pbrt::Point3f(scene.positions[3 * c + 0],
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

    std::vector<pbrt::Primitive> primitives;
    std::map<const void *, uint32_t> index_of;
    size_t next_triangle = 0;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        const bonsai_scene::Shape &s = shapes[i];
        pbrt::Shape shape;
        if (s.tag == bonsai_scene::ShapeTag::Sphere) {
            const pbrt::Transform *render_from_object =
                alloc.new_object<pbrt::Transform>(pbrt::Translate(
                    pbrt::Vector3f(s.center[0], s.center[1], s.center[2])));
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
    bool tables_only = false;
    bool sampler_only = false;
    bool bsdf_only = false;
    bool shading_only = false;
    bool light_only = false;
    int repeats = 1;
    // --reference writes the gbuffer PBRT would have written, rendered with
    // PBRT's own camera, aggregate and BSDFs. It is how the comparison runs on
    // a scene someone else wrote: those say `Film "rgb"`, PBRT has no
    // command-line override for the film, and rewriting a scene to change one
    // line would make the comparison about our copy of it.
    std::string reference_prefix;
    std::vector<const char *> positional;
    for (int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--pbrt-tree") {
            pbrt_tree = true;
        } else if (arg == "--check-tables") {
            tables_only = true;
        } else if (arg == "--print-sampler") {
            sampler_only = true;
        } else if (arg == "--print-bsdf") {
            bsdf_only = true;
        } else if (arg == "--print-shading") {
            shading_only = true;
        } else if (arg == "--print-light") {
            light_only = true;
        } else if (arg == "--reference") {
            if (i + 1 >= argc) {
                fail("--reference needs a path prefix to write to");
            }
            reference_prefix = argv[++i];
        } else if (arg == "--repeats") {
            if (i + 1 >= argc) {
                fail("--repeats needs a count");
            }
            repeats = std::max(1, atoi(argv[++i]));
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (!tables_only && !sampler_only && !bsdf_only && !shading_only &&
        !light_only && positional.size() != 2) {
        fail("usage: scene_dump [--pbrt-tree] [--reference <prefix>]"
             " <scene.pbrt> <out.txt>\n"
             "       scene_dump --check-tables\n"
             "       scene_dump --print-sampler\n"
             "       scene_dump --print-bsdf\n"
             "       scene_dump --print-shading");
    }

    pbrt::PBRTOptions options;
    // The renderer takes one sample at the centre of each pixel, and the
    // reference is rendered the same way. This has to be set before parsing
    // because it is read while the scene is built.
    options.disablePixelJitter = true;
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

    bonsai_scene::Scene scene;
    load(positional[0], scene, reference_prefix, repeats);
    if (pbrt_tree) {
        build_pbrt_tree(scene);
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
