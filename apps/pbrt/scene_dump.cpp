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
    std::string sampler_name = "zsobol";
    pbrt::ParameterDictionary sampler_params;
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
    // The first ten primes, so a mistranscribed table is caught here rather
    // than as a wrong answer three functions later.
    printf("primes:");
    for (int i = 0; i < 10; i++) {
        printf(" %d", pbrt::Primes[i]);
    }
    printf(" ... %d\n", pbrt::Primes[pbrt::PrimeTableSize - 1]);
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

    // PBRT's fixed sample arrays for the reflectance estimate, from the path
    // integrator. Fixed rather than drawn, so the albedo of a pixel does not
    // depend on where in the sampler's stream the estimate happens to fall.
    static constexpr int kRhoSamples = 16;
    const pbrt::Float uc_rho[kRhoSamples] = {
        0.75741637, 0.37870818, 0.7083487,  0.18935409, 0.9149363,  0.35417435,
        0.5990858,  0.09467703, 0.8578725,  0.45746812, 0.686759,   0.17708716,
        0.9674518,  0.2995429,  0.5083201,  0.047338516};
    const pbrt::Point2f u_rho[kRhoSamples] = {
        {0.855985f, 0.570367f}, {0.381823f, 0.851844f}, {0.285328f, 0.764262f},
        {0.733380f, 0.114073f}, {0.542663f, 0.344465f}, {0.127274f, 0.414848f},
        {0.964700f, 0.947162f}, {0.594089f, 0.643463f}, {0.095109f, 0.170369f},
        {0.825444f, 0.263359f}, {0.429467f, 0.454469f}, {0.244460f, 0.816459f},
        {0.756135f, 0.731258f}, {0.516165f, 0.152852f}, {0.180888f, 0.214174f},
        {0.898579f, 0.503897f}};

    std::vector<float> normals(size_t(width) * height * 3, 0.f);
    std::vector<float> albedos(size_t(width) * height * 3, 0.f);

    pbrt::ThreadLocal<pbrt::ScratchBuffer> buffers(
        []() { return pbrt::ScratchBuffer(); });
    pbrt::ThreadLocal<pbrt::Sampler> samplers(
        [&sampler]() { return sampler.Clone({}); });

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
                    bsdf.rho(isect.wo, uc_rho, u_rho);
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
                }
            }
        }
    });

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
    printf("scene_dump: reference %s.pfm and %s-albedo.pfm (%dx%d, %d spp)\n",
           prefix.c_str(), prefix.c_str(), width, height, spp);
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
          const std::string &reference_prefix) {
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

        } else if (const pbrt::TriangleMesh *mesh = triangulate(entity)) {
            // Everything that is ultimately a mesh arrives here already
            // triangulated by PBRT, in render space, so there is one loop for
            // all of them rather than one per shape type. See `triangulate`.
            const int *v = mesh->vertexIndices;
            for (int i = 0; i < mesh->nTriangles; i++) {
                const pbrt::Point3f p0 = mesh->p[v[3 * i + 0]];
                const pbrt::Point3f p1 = mesh->p[v[3 * i + 1]];
                const pbrt::Point3f p2 = mesh->p[v[3 * i + 2]];
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

    // Last, because CreateAggregate empties `scene.shapes` on its way through
    // -- it hands the entities to the primitives it builds -- and the
    // conversion above reads them.
    if (!reference_prefix.empty()) {
        // The film's colour space rather than sRGB by assumption: it is what
        // GBufferFilm converts the albedo with, and a scene can name another.
        render_reference(scene, builder.film_params.ColorSpace(),
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
    bool tables_only = false;
    bool sampler_only = false;
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
        } else if (arg == "--reference") {
            if (i + 1 >= argc) {
                fail("--reference needs a path prefix to write to");
            }
            reference_prefix = argv[++i];
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (!tables_only && !sampler_only && positional.size() != 2) {
        fail("usage: scene_dump [--pbrt-tree] [--reference <prefix>]"
             " <scene.pbrt> <out.txt>\n"
             "       scene_dump --check-tables\n"
             "       scene_dump --print-sampler");
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

    bonsai_scene::Scene scene;
    load(positional[0], scene, reference_prefix);
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
