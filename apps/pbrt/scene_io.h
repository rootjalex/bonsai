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
#include <limits>
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
    // A boundary between two dielectrics -- glass. It reuses the roughness and
    // `eta` fields below, which is what the coating of a CoatedDiffuse already
    // is: PBRT's DielectricBxDF appears in both.
    Dielectric = 2,
    // A metal: the same microfacet distribution over a Fresnel term with a
    // *complex* index, whose imaginary part is the absorption that makes a
    // metal a metal and whose variation with wavelength is its colour.
    Conductor = 3,
    // A measured BRDF: no model at all, just a table of what the surface was
    // observed to do, in Dupuy and Jakob's parameterization.
    Measured = 4,
};

// One of PBRT's PiecewiseLinear2D interpolants, as the renderer reads it.
//
// The arrays live in the shared pools below; this is where each one's run
// starts and how it is shaped. `dim` is how many parameters the distribution
// depends on besides its own two axes -- zero for the NDF and the projected
// area, two (incoming direction) for the VNDF and the luminance, three
// (incoming direction and wavelength) for the spectra.
struct PL2DHeader {
    uint32_t size_x = 0;
    uint32_t size_y = 0;
    uint32_t dim = 0;
    // Per parameter dimension, outermost first. Unused entries are zero.
    uint32_t param_size[3] = {0, 0, 0};
    uint32_t param_stride[3] = {0, 0, 0};
    uint32_t first_param[3] = {0, 0, 0};
    uint32_t first_data = 0;
    uint32_t first_marginal = 0;
    uint32_t first_conditional = 0;
    // Whether the CDFs were built. The NDF and the projected area are only
    // ever evaluated, never sampled, so PBRT does not build theirs.
    uint32_t has_cdf = 0;
};

// One measured BRDF: the five interpolants PBRT's MeasuredBxDFData holds.
struct MeasuredBRDF {
    uint32_t ndf = 0;
    uint32_t sigma = 0;
    uint32_t vndf = 0;
    uint32_t luminance = 0;
    uint32_t spectra = 0;
    uint32_t isotropic = 0;
};

// How finely a conductor's index of refraction is resampled, and how many
// entries that makes over 360 to 830 nm.
//
// PBRT keeps these curves as a PiecewiseLinearSpectrum over the published
// measurements, whose knots are four to six nanometres apart, and interpolates
// between them. Resampling on a uniform grid is exact everywhere except inside
// the one interval that straddles each knot, where a chord replaces a corner --
// so the error is proportional to the step, not to its square. At one
// nanometre that peaked at 2.5e-3 on copper, whose index has a knee at 590 nm
// and which is orange for that reason; at a tenth of a nanometre it is a tenth
// of that, for 37 kB a metal.
inline constexpr int kConductorPerNm = 10;
inline constexpr int kConductorSamples = 470 * kConductorPerNm + 1;

// Which of PBRT's integrators the scene asked for, of the ones this renderer
// has. PBRT dispatches these through a TaggedPointer and so does the renderer,
// through an `Integrator` variant; this is the tag that says which arm.
//
// A scene naming one this renderer does not have is refused rather than
// silently rendered with another, which would be an image that looks like an
// answer to a question nobody asked.
enum IntegratorTag : uint32_t {
    RandomWalk = 0,
    SimplePath = 1,
    Path = 2,
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
// PBRT's WrapMode, for what a texture lookup does off the edge of its image.
namespace WrapMode {
enum : uint32_t { Repeat = 0, Clamp = 1, Black = 2, OctahedralSphere = 3 };
}

// One level of one texture's MIP pyramid, as a run of `texture_texels`.
//
// Three floats per texel, linear, in the image's own colour space. Already
// decoded: a PNG is sRGB-encoded and an EXR is not, and the difference is
// resolved on the way in by PBRT's own image reader rather than carried here
// for the renderer to have an opinion about.
struct TextureLevel {
    uint32_t width = 0;
    uint32_t height = 0;
    // Counted in texels and not in floats, since the renderer reads
    // `texture_texels` as an array of three-vectors. Getting that wrong reads
    // three times past the end of the pool, which on a small texture is still
    // mapped memory and so shows up as an occasional segfault rather than as a
    // wrong picture.
    uint32_t first_texel = 0;
};

// PBRT's ImageTexture, as its MIP pyramid plus what the lookup needs.
//
// The pyramid is built by PBRT's own `MIPMap`, with PBRT's own resampling
// filter, and shipped level by level -- the same division of labour as the
// spectral fits and the BVH. What is left for the renderer is choosing a level
// from the footprint and bilerping in it, which is the part that runs per
// lookup and the part worth transcribing.
struct ImageTexture {
    // PBRT's UVMapping: `st = (su * u + du, sv * v + dv)`.
    float su = 1.f;
    float sv = 1.f;
    float du = 0.f;
    float dv = 0.f;
    // PBRT's `scale` and `invert`, applied to the filtered RGB. A `scale`
    // texture over an image one folds into this, since PBRT's ImageTexture
    // already has a scale of its own and multiplying them is the same thing --
    // but only when the scale is a constant, and a scale by another *texture*
    // is refused rather than flattened.
    float scale = 1.f;
    uint32_t invert = 0;
    uint32_t wrap = WrapMode::Repeat;
    // The pyramid, as a run of `texture_levels`, coarsest last.
    uint32_t first_level = 0;
    uint32_t n_levels = 0;
};

struct Material {
    uint32_t tag = MaterialTag::Diffuse;
    float reflectance[3] = {0.5f, 0.5f, 0.5f};
    // An index into `textures` when the reflectance is a texture rather than
    // the constant above, and -1 when it is not. PBRT's material parameters are
    // all textures and a constant is a `FloatConstantTexture`; here the
    // constant is the common case and stays a constant, because making every
    // diffuse surface do a texture lookup to find out it has none would cost
    // more than the uniformity is worth.
    int32_t reflectance_texture = -1;
    // PBRT's `displacement`, a float texture every material may carry. It does
    // not change what the material *is* -- it tilts the shading frame before
    // the BSDF is built, which is why PBRT keeps it on the base Material and
    // applies it in GetBSDF rather than in any one material's GetBxDF.
    int32_t displacement_texture = -1;
    // Conductor only: which pair of `conductor_eta` / `conductor_k` tables this
    // material's index of refraction is.
    int32_t conductor_spectra = -1;
    // Measured only: which entry of `measured_brdfs`.
    int32_t measured = -1;
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

// A UniformInfiniteLight: the same radiance from every direction, which is what
// `LightSource "infinite"` means with no image behind it.
//
// Not a shape, so it is not in `shapes` and no primitive points at it; it goes
// in its own list and the driver appends it to the renderer's lights after the
// area ones, which is the order PBRT builds them in.
//
// `has_l` is the difference between the two spectra PBRT can put here. With no
// `L` written it emits the colour space's illuminant itself; with one it emits
// an RGBIlluminantSpectrum, which is a fit of that RGB *times* the illuminant.
// The fit of a flat spectrum is not exactly flat, so which of the two this is
// cannot be recovered from the numbers and has to be said.
struct InfiniteLight {
    float l[3] = {1.f, 1.f, 1.f};
    float scale = 1.f;
    uint32_t has_l = 0;
    // An ImageInfiniteLight rather than a uniform one: the equal-area
    // octahedral environment map's square resolution, and where its texels
    // begin in the scene's shared texel pool. Zero resolution means there is no
    // image and this is the uniform light above.
    uint32_t resolution = 0;
    uint32_t first_texel = 0;
    // PBRT's `renderFromLight`, already inverted -- 4x4 in row order. The only
    // thing done with it is ApplyInverse, so the direction it is used in is the
    // one handed over. Identity when the scene wrapped the light in no
    // transform, which most do not.
    float light_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    // And the way round PBRT stores it. Both, because both are used: `Le` and
    // `PDF_Li` take a render-space direction into the light's frame, and
    // `SampleLi` sends a sampled one the other way.
    float render_from_light[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
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
    // PBRT's GeometricPrimitive alpha texture, or -1 where the shape has none.
    // A cutout: the texture says how much of the surface is really there, which
    // is what makes a tree leaf leaf-shaped rather than a quad.
    int32_t alpha = -1;
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

// PBRT: an `ObjectBegin`/`ObjectEnd` block -- an instance definition, the
// shapes an instanced object is made of. They are a run of `instance_shapes`,
// kept apart from `shapes` because PBRT keeps them apart: an instance's
// geometry is in no top-level list, and the top-level tree holds the instances
// rather than what they hold.
//
// `root_node` is the row of `instance_nodes` the object's own tree starts at,
// when the tree is PBRT's (--pbrt-tree). Otherwise the driver builds one.
struct Definition {
    uint32_t first_shape = 0;
    uint32_t shape_count = 0;
    uint32_t root_node = 0;
};

// PBRT: an `ObjectInstance` -- a TransformedPrimitive, naming a definition and
// placing it. Both matrices are PBRT's own, `renderFromInstance` and its
// inverse as `Transform` stores the pair, because the renderer needs both: the
// forward one places the object's bounds and the interaction it finds, and the
// inverse pulls a ray back into the object.
struct Instance {
    uint32_t definition = 0;
    float render_from_instance[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                      0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    float instance_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                      0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
};

// What the top-level tree's leaves name, in the order they name it, when the
// tree is PBRT's: PBRT's BVHAggregate is built over its shapes and its
// instances together and reorders the mixture, so the order is a list of
// (which kind, which one) rather than a permutation of either list.
enum PrimKind : uint32_t {
    PrimShape = 0,    // `index` into `shapes`
    PrimInstance = 1, // `index` into `instances`
};

struct Prim {
    uint32_t kind = PrimShape;
    uint32_t index = 0;
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
    uint32_t integrator = IntegratorTag::RandomWalk;
    // PBRT's `Integrator "path" "bool regularize"`, which widens a near-delta
    // lobe once a path has already scattered off something rough. Only the path
    // integrator has it, and its default is off.
    uint32_t regularize = 0;
    // The reconstruction filter a camera sample is jittered within. PBRT's
    // defaults, from GaussianFilter::Create -- which is also PBRT's default
    // filter, so a scene that names none gets these.
    float filter_radius[2] = {1.5f, 1.5f};
    float filter_sigma = 0.5f;
    // PBRT's `--disable-pixel-jitter`, which pins every sample of a pixel to
    // its centre. Not a property of the scene; carried here because it has to
    // reach the renderer and this is the channel that exists.
    uint32_t disable_pixel_jitter = 0;
    // camera_from_raster, render_from_camera then camera_from_render, each
    // 4x4 in row order. The third is not the second's inverse recomputed --
    // it is PBRT's own `CameraFromRender`, and the pair is carried in both
    // directions because `Approximate_dp_dxy` goes camera-ward and back and
    // nothing here inverts a matrix.
    float matrices[48] = {};
    // PBRT: PerspectiveCamera's `dxCamera` and `dyCamera`, three floats each.
    // The change in the camera-space sample position for a one-pixel step in
    // raster x and in raster y, which is what makes a camera ray's
    // differentials. Constants of the camera, so they are derived here from
    // the same `cameraFromRaster` rather than in the renderer.
    float d_camera[6] = {};
    // PBRT: `CameraBase::minPosDifferentialX/Y` and `minDirDifferentialX/Y`,
    // three floats each in that order.
    //
    // `FindMinimumDifferentials` finds them by generating 512 differential
    // rays across the film and keeping the shortest offset it saw, which is a
    // loop over the camera and not over the scene -- so it runs here, using
    // PBRT's own `GenerateRayDifferential`, and the four vectors ship as data.
    // They are what a hit falls back on once a path has scattered and its ray
    // no longer carries differentials of its own, which after the first
    // non-specular bounce is every hit.
    float min_differentials[12] = {};
    // PBRT: ProjectiveCamera's lensRadius and focalDistance. Zero radius is a
    // pinhole, which is what every scene in `scenes/` is and what most real
    // ones are not.
    float lens_radius = 0.f;
    float focal_distance = 1e6f;
    // PBRT: PixelSensor's `exposureTime * ISO / 100`, which scales the recorded
    // radiance. One at PBRT's defaults, which is why every scene here had it
    // until one set `"float iso"`.
    float imaging_ratio = 1.f;
    // PBRT: RGBFilm's `maxcomponentvalue`, which clamps each *sample*'s sensor
    // RGB before it is accumulated -- a firefly suppressor. Infinite unless the
    // scene names one.
    float max_component_value = std::numeric_limits<float>::infinity();
    std::vector<Material> materials;
    // The meshes, and the pools their runs live in. Three floats per position
    // and normal, two per texture coordinate.
    std::vector<Mesh> meshes;
    std::vector<uint32_t> indices;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<Light> lights;
    std::vector<InfiniteLight> infinite_lights;
    // Every environment map's texels, laid end to end -- *four* floats each:
    // the three coefficients of the sigmoid PBRT's RGBIlluminantSpectrum fits,
    // and its scale.
    //
    // Fitted here rather than in the renderer, through PBRT's own
    // RGB-to-spectrum table, because the fit is a deterministic function of the
    // texel and doing it per lookup would put a table under every escaped ray.
    // Written to a binary file beside the scene rather than into it: a
    // 2048x2048 map is sixteen million numbers, and the scene file is text.
    std::vector<float> env_texels;
    // What the *sampling distribution* over each map is built from: one float
    // per texel, in the same order as `env_texels`.
    //
    // PBRT: `Image::GetSamplingDistribution`, which is the average of the
    // texel's channels taken from the image itself. It has to be shipped
    // separately because `env_texels` no longer holds the image -- it holds the
    // sigmoid the image was fitted to, and a fit is not a thing you can average
    // to get a density. A black texel's fit is minus infinity, which is a
    // perfectly good answer to "what does this reflect" and a catastrophic one
    // to "how often should this be sampled".
    std::vector<float> env_sampling;
    // The image textures, their pyramid levels, and every level's texels laid
    // end to end. The texels go in the sidecar beside the environment maps and
    // for the same reason: a 2048x2048 pyramid is seventeen million numbers.
    std::vector<ImageTexture> textures;
    std::vector<TextureLevel> texture_levels;
    std::vector<float> texture_texels;
    // PBRT's `RGBToSpectrumTable` for the sRGB colour space: the 64 z nodes
    // followed by 3 * 64 * 64 * 64 * 3 coefficients, laid out as PBRT lays them
    // out. Empty when the scene has no textures.
    //
    // Carried rather than fitted. A constant reflectance is fitted once in
    // scene_dump by a Gauss-Newton solve, which is fine for a handful of
    // materials and hopeless per texture lookup -- and PBRT does not solve
    // there either, it interpolates this table. So the table is what ships, and
    // the renderer does the same trilinear lookup PBRT does.
    //
    // It has to happen per lookup, and not per texel in advance: PBRT filters
    // in RGB and fits the *filtered* colour, and fitting each texel and
    // interpolating the coefficients is a different, nonlinear thing.
    std::vector<float> rgb_table;
    // Every conductor's index of refraction: 471 entries each at one nanometre
    // from 360 nm, laid end to end, one run per distinct pair a material named.
    //
    // Resampled from PBRT's own named spectra rather than fitted. PBRT keeps
    // them as PiecewiseLinearSpectrum and interpolates between the published
    // measurements; this is that function at one nanometre, which reproduces it
    // exactly except inside the single nanometre containing each of its own
    // knots. scene_dump measures that residual and prints it.
    std::vector<float> conductor_eta;
    std::vector<float> conductor_k;
    // The measured BRDFs, their interpolants, and the four pools those index
    // into. The pools go in the `.tex` sidecar with the texture texels: one
    // `.bsdf` file is seven megabytes and the scene file is text.
    std::vector<MeasuredBRDF> measured_brdfs;
    std::vector<PL2DHeader> pl2d;
    std::vector<float> pl_data;
    std::vector<float> pl_marginal;
    std::vector<float> pl_conditional;
    std::vector<float> pl_params;
    // PBRT's `sceneRadius`, from `Light::Preprocess` -- the radius of the
    // scene's bounding sphere. An infinite light has no geometry, so a shadow
    // ray aimed at one needs somewhere to stop, and PBRT puts that two radii
    // out. Computed here because it is a property of the whole scene and the
    // renderer sees the scene one primitive at a time.
    float scene_radius = 0.f;
    std::vector<Shape> shapes;
    std::vector<Node> nodes;

    // PBRT's instancing: the definitions' shapes, the definitions as runs of
    // them, and the instances placing them. `instance_nodes` and `prims` are
    // filled only alongside `nodes`, when the trees are PBRT's own -- the
    // former one pool of every definition's tree, the latter the top-level
    // tree's leaf order over shapes and instances together.
    std::vector<Shape> instance_shapes;
    std::vector<Definition> definitions;
    std::vector<Instance> instances;
    std::vector<Node> instance_nodes;
    std::vector<Prim> prims;

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

// Where an environment map's texels live: beside the scene file rather than in
// it. A 2048x2048 map is twelve million floats, and the scene file is text --
// writing them there would make it eighty times the size of the geometry and
// slower to parse than to render.
inline std::string env_path(const char *scene_path) {
    return std::string(scene_path) + ".env";
}

inline std::string texel_path(const char *scene_path) {
    return std::string(scene_path) + ".tex";
}

inline std::string pl_path(const char *scene_path) {
    return std::string(scene_path) + ".pl";
}

inline bool write(const char *path, const Scene &scene) {
    if (!scene.env_texels.empty()) {
        std::ofstream env(env_path(path), std::ios::binary);
        if (!env) {
            return false;
        }
        env.write(reinterpret_cast<const char *>(scene.env_texels.data()),
                  std::streamsize(sizeof(float) * scene.env_texels.size()));
        env.write(reinterpret_cast<const char *>(scene.env_sampling.data()),
                  std::streamsize(sizeof(float) * scene.env_sampling.size()));
        if (!env) {
            return false;
        }
    }
    if (!scene.texture_texels.empty() || !scene.rgb_table.empty()) {
        std::ofstream tex(texel_path(path), std::ios::binary);
        if (!tex) {
            return false;
        }
        // The table first, so a reader can take it without knowing how many
        // texels follow.
        tex.write(reinterpret_cast<const char *>(scene.rgb_table.data()),
                  std::streamsize(sizeof(float) * scene.rgb_table.size()));
        tex.write(
            reinterpret_cast<const char *>(scene.texture_texels.data()),
            std::streamsize(sizeof(float) * scene.texture_texels.size()));
        if (!tex) {
            return false;
        }
    }
    if (!scene.pl_data.empty()) {
        std::ofstream pl(pl_path(path), std::ios::binary);
        if (!pl) {
            return false;
        }
        const auto put_pool = [&](const std::vector<float> &v) {
            pl.write(reinterpret_cast<const char *>(v.data()),
                     std::streamsize(sizeof(float) * v.size()));
        };
        put_pool(scene.pl_data);
        put_pool(scene.pl_marginal);
        put_pool(scene.pl_conditional);
        put_pool(scene.pl_params);
        if (!pl) {
            return false;
        }
    }

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
    out << "integrator "
        << (scene.integrator == IntegratorTag::Path
                ? "path"
                : (scene.integrator == IntegratorTag::SimplePath ? "simplepath"
                                                                 : "randomwalk"))
        << " maxdepth " << scene.max_depth << " regularize "
        << scene.regularize << '\n';
    out << "filter gaussian " << scene.filter_radius[0] << ' '
        << scene.filter_radius[1] << ' ' << scene.filter_sigma << " jitter "
        << (scene.disable_pixel_jitter ? 0 : 1) << '\n';
    out << "camera_from_raster";
    detail::put(out, scene.matrices, 16);
    out << "\nrender_from_camera";
    detail::put(out, scene.matrices + 16, 16);
    out << "\ncamera_from_render";
    detail::put(out, scene.matrices + 32, 16);
    out << "\nd_camera";
    detail::put(out, scene.d_camera, 6);
    out << "\nmin_differentials";
    detail::put(out, scene.min_differentials, 12);
    out << "\nlens";
    detail::put(out, &scene.lens_radius, 1);
    detail::put(out, &scene.focal_distance, 1);
    out << "\nimagingratio";
    detail::put(out, &scene.imaging_ratio, 1);
    // As a flag and a value, not as the number itself: `%.9g` writes infinity
    // as `inf`, and reading that back with `>>` leaves zero and a failed
    // stream -- which would clamp every sample to black rather than to nothing.
    {
        const bool finite = scene.max_component_value !=
                            std::numeric_limits<float>::infinity();
        out << "\nmaxcomponent " << (finite ? 1 : 0);
        const float v = finite ? scene.max_component_value : 0.f;
        detail::put(out, &v, 1);
    }
    out << '\n';

    // Before the materials, because a material names one by index.
    out << "textures " << scene.textures.size() << '\n';
    for (const ImageTexture &t : scene.textures) {
        out << "  uv";
        detail::put(out, &t.su, 1);
        detail::put(out, &t.sv, 1);
        detail::put(out, &t.du, 1);
        detail::put(out, &t.dv, 1);
        out << " scale";
        detail::put(out, &t.scale, 1);
        out << " invert " << t.invert << " wrap " << t.wrap << " levels "
            << t.first_level << ' ' << t.n_levels << '\n';
    }
    out << "texturelevels " << scene.texture_levels.size() << '\n';
    for (const TextureLevel &l : scene.texture_levels) {
        out << "  " << l.width << ' ' << l.height << ' ' << l.first_texel
            << '\n';
    }
    out << "measured " << scene.measured_brdfs.size() << '\n';
    for (const MeasuredBRDF &b : scene.measured_brdfs) {
        out << "  " << b.ndf << ' ' << b.sigma << ' ' << b.vndf << ' '
            << b.luminance << ' ' << b.spectra << ' ' << b.isotropic << '\n';
    }
    out << "pl2d " << scene.pl2d.size() << '\n';
    for (const PL2DHeader &h : scene.pl2d) {
        out << "  " << h.size_x << ' ' << h.size_y << ' ' << h.dim;
        for (int i = 0; i < 3; i++) {
            out << ' ' << h.param_size[i] << ' ' << h.param_stride[i] << ' '
                << h.first_param[i];
        }
        out << ' ' << h.first_data << ' ' << h.first_marginal << ' '
            << h.first_conditional << ' ' << h.has_cdf << '\n';
    }
    out << "plpools " << scene.pl_data.size() << ' ' << scene.pl_marginal.size()
        << ' ' << scene.pl_conditional.size() << ' ' << scene.pl_params.size()
        << '\n';

    out << "conductorspectra " << scene.conductor_eta.size() << '\n';
    for (size_t i = 0; i < scene.conductor_eta.size(); i++) {
        detail::put(out, &scene.conductor_eta[i], 1);
        detail::put(out, &scene.conductor_k[i], 1);
        if ((i + 1) % 8 == 0) {
            out << '\n';
        }
    }
    out << '\n';
    out << "rgbtable " << scene.rgb_table.size() << '\n';
    out << "texturetexels " << scene.texture_texels.size() << '\n';

    out << "materials " << scene.materials.size() << '\n';
    for (const Material &m : scene.materials) {
        // Written out rather than defaulted. This used to be
        // `tag == CoatedDiffuse ? "coateddiffuse" : "diffuse"`, so the first
        // material added after it -- `dielectric` -- serialized as a diffuse
        // and rendered as PBRT's default grey, with the converter, the
        // renderer and the BxDF all correct and only the file between them
        // lying. A tag with no case here is a bug in this function, so it says
        // so instead of picking one.
        switch (m.tag) {
        case MaterialTag::Diffuse:
            out << "  diffuse";
            break;
        case MaterialTag::CoatedDiffuse:
            out << "  coateddiffuse";
            break;
        case MaterialTag::Dielectric:
            out << "  dielectric";
            break;
        case MaterialTag::Conductor:
            out << "  conductor";
            break;
        case MaterialTag::Measured:
            out << "  measured";
            break;
        default:
            return false;
        }
        out << " reflectance";
        detail::put(out, m.reflectance, 3);
        out << " reflectancetex " << m.reflectance_texture << " displacement "
            << m.displacement_texture << " measured " << m.measured;
        if (m.tag == MaterialTag::Dielectric) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " remap " << m.remap;
            out << " eta";
            detail::put(out, &m.eta, 1);
        }
        if (m.tag == MaterialTag::Conductor) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " remap " << m.remap;
            out << " spectra " << m.conductor_spectra;
        }
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

    out << "infinite_lights " << scene.infinite_lights.size() << " radius "
        << scene.scene_radius << '\n';
    for (const InfiniteLight &l : scene.infinite_lights) {
        out << (l.resolution == 0 ? "  uniform" : "  image");
        detail::put(out, l.l, 3);
        detail::put(out, &l.scale, 1);
        out << " hasl " << l.has_l;
        if (l.resolution != 0) {
            out << " resolution " << l.resolution << " first " << l.first_texel
                << " light_from_render";
            detail::put(out, l.light_from_render, 16);
            out << " render_from_light";
            detail::put(out, l.render_from_light, 16);
        }
        out << '\n';
    }

    const auto put_shapes = [&](const char *name,
                                const std::vector<Shape> &shapes) {
        out << name << ' ' << shapes.size() << '\n';
        for (const Shape &s : shapes) {
            if (s.tag == ShapeTag::Sphere) {
                out << "  sphere";
                detail::put(out, s.center, 3);
                detail::put(out, &s.radius, 1);
                out << " flip " << s.flip;
            } else {
                out << "  tri " << s.mesh << ' ' << s.tri;
            }
            out << " material " << s.material << " light " << s.light
                << " alpha " << s.alpha << '\n';
        }
    };
    const auto put_nodes = [&](const char *name,
                               const std::vector<Node> &nodes) {
        out << name << ' ' << nodes.size() << '\n';
        for (const Node &n : nodes) {
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
    };

    put_shapes("shapes", scene.shapes);
    put_nodes("nodes", scene.nodes);

    put_shapes("instance_shapes", scene.instance_shapes);
    out << "definitions " << scene.definitions.size() << '\n';
    for (const Definition &d : scene.definitions) {
        out << "  first " << d.first_shape << " count " << d.shape_count
            << " root " << d.root_node << '\n';
    }
    out << "instances " << scene.instances.size() << '\n';
    for (const Instance &i : scene.instances) {
        out << "  definition " << i.definition << " render_from_instance";
        detail::put(out, i.render_from_instance, 16);
        out << " instance_from_render";
        detail::put(out, i.instance_from_render, 16);
        out << '\n';
    }
    put_nodes("instance_nodes", scene.instance_nodes);
    out << "prims " << scene.prims.size() << '\n';
    for (const Prim &p : scene.prims) {
        out << (p.kind == PrimShape ? "  shape " : "  instance ") << p.index
            << '\n';
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

    if (!(in >> word) || word != "integrator") {
        return false;
    }
    if (!(in >> word)) {
        return false;
    }
    if (word == "randomwalk") {
        scene.integrator = IntegratorTag::RandomWalk;
    } else if (word == "simplepath") {
        scene.integrator = IntegratorTag::SimplePath;
    } else if (word == "path") {
        scene.integrator = IntegratorTag::Path;
    } else {
        return false;
    }
    if (!(in >> word) || word != "maxdepth") {
        return false;
    }
    in >> scene.max_depth;
    if (!(in >> word) || word != "regularize") {
        return false;
    }
    in >> scene.regularize;

    if (!(in >> word) || word != "filter") {
        return false;
    }
    if (!(in >> word) || word != "gaussian") {
        return false;
    }
    in >> scene.filter_radius[0] >> scene.filter_radius[1] >>
        scene.filter_sigma;
    if (!(in >> word) || word != "jitter") {
        return false;
    }
    uint32_t jitter = 1;
    in >> jitter;
    scene.disable_pixel_jitter = jitter ? 0u : 1u;

    if (!(in >> word) || word != "camera_from_raster") {
        return false;
    }
    floats(scene.matrices, 16);
    if (!(in >> word) || word != "render_from_camera") {
        return false;
    }
    floats(scene.matrices + 16, 16);
    if (!(in >> word) || word != "camera_from_render") {
        return false;
    }
    floats(scene.matrices + 32, 16);
    if (!(in >> word) || word != "d_camera") {
        return false;
    }
    floats(scene.d_camera, 6);
    if (!(in >> word) || word != "min_differentials") {
        return false;
    }
    floats(scene.min_differentials, 12);
    if (!(in >> word) || word != "lens") {
        return false;
    }
    floats(&scene.lens_radius, 1);
    floats(&scene.focal_distance, 1);
    if (!(in >> word) || word != "imagingratio") {
        return false;
    }
    floats(&scene.imaging_ratio, 1);
    if (!(in >> word) || word != "maxcomponent") {
        return false;
    }
    {
        int finite = 0;
        in >> finite;
        float v = 0.f;
        floats(&v, 1);
        scene.max_component_value =
            finite ? v : std::numeric_limits<float>::infinity();
    }

    // A labelled field, checked as it is read. The labels are what makes a
    // stale file fail here rather than three fields later with plausible
    // numbers in the wrong places.
    const auto tagged = [&](const char *name) {
        return bool(in >> word) && word == name;
    };

    size_t count = 0;
    if (!(in >> word) || word != "textures") {
        return false;
    }
    in >> count;
    scene.textures.clear();
    for (size_t i = 0; i < count; i++) {
        ImageTexture t;
        if (!tagged("uv")) {
            return false;
        }
        floats(&t.su, 1);
        floats(&t.sv, 1);
        floats(&t.du, 1);
        floats(&t.dv, 1);
        if (!tagged("scale")) {
            return false;
        }
        floats(&t.scale, 1);
        if (!tagged("invert")) {
            return false;
        }
        in >> t.invert;
        if (!tagged("wrap")) {
            return false;
        }
        in >> t.wrap;
        if (!tagged("levels")) {
            return false;
        }
        in >> t.first_level >> t.n_levels;
        scene.textures.push_back(t);
    }

    if (!(in >> word) || word != "texturelevels") {
        return false;
    }
    in >> count;
    scene.texture_levels.clear();
    for (size_t i = 0; i < count; i++) {
        TextureLevel l;
        in >> l.width >> l.height >> l.first_texel;
        scene.texture_levels.push_back(l);
    }

    if (!(in >> word) || word != "measured") {
        return false;
    }
    in >> count;
    scene.measured_brdfs.clear();
    for (size_t i = 0; i < count; i++) {
        MeasuredBRDF b;
        in >> b.ndf >> b.sigma >> b.vndf >> b.luminance >> b.spectra >>
            b.isotropic;
        scene.measured_brdfs.push_back(b);
    }

    if (!(in >> word) || word != "pl2d") {
        return false;
    }
    in >> count;
    scene.pl2d.clear();
    for (size_t i = 0; i < count; i++) {
        PL2DHeader h;
        in >> h.size_x >> h.size_y >> h.dim;
        for (int j = 0; j < 3; j++) {
            in >> h.param_size[j] >> h.param_stride[j] >> h.first_param[j];
        }
        in >> h.first_data >> h.first_marginal >> h.first_conditional >>
            h.has_cdf;
        scene.pl2d.push_back(h);
    }

    if (!(in >> word) || word != "plpools") {
        return false;
    }
    {
        size_t n_data = 0, n_marg = 0, n_cond = 0, n_param = 0;
        in >> n_data >> n_marg >> n_cond >> n_param;
        scene.pl_data.clear();
        scene.pl_marginal.clear();
        scene.pl_conditional.clear();
        scene.pl_params.clear();
        if (n_data > 0) {
            std::ifstream pl(pl_path(path), std::ios::binary);
            if (!pl) {
                return false;
            }
            const auto get_pool = [&](std::vector<float> &v, size_t n) {
                v.resize(n);
                pl.read(reinterpret_cast<char *>(v.data()),
                        std::streamsize(sizeof(float) * n));
                return pl.gcount() == std::streamsize(sizeof(float) * n);
            };
            if (!get_pool(scene.pl_data, n_data) ||
                !get_pool(scene.pl_marginal, n_marg) ||
                !get_pool(scene.pl_conditional, n_cond) ||
                !get_pool(scene.pl_params, n_param)) {
                return false;
            }
        }
    }

    if (!(in >> word) || word != "conductorspectra") {
        return false;
    }
    in >> count;
    scene.conductor_eta.assign(count, 0.f);
    scene.conductor_k.assign(count, 0.f);
    for (size_t i = 0; i < count; i++) {
        floats(&scene.conductor_eta[i], 1);
        floats(&scene.conductor_k[i], 1);
    }

    if (!(in >> word) || word != "rgbtable") {
        return false;
    }
    size_t table_size = 0;
    in >> table_size;

    if (!(in >> word) || word != "texturetexels") {
        return false;
    }
    {
        size_t texels = 0;
        in >> texels;
        scene.rgb_table.clear();
        scene.texture_texels.clear();
        if (table_size > 0 || texels > 0) {
            std::ifstream tex(texel_path(path), std::ios::binary);
            if (!tex) {
                return false;
            }
            scene.rgb_table.resize(table_size);
            tex.read(reinterpret_cast<char *>(scene.rgb_table.data()),
                     std::streamsize(sizeof(float) * table_size));
            if (tex.gcount() != std::streamsize(sizeof(float) * table_size)) {
                return false;
            }
            scene.texture_texels.resize(texels);
            tex.read(reinterpret_cast<char *>(scene.texture_texels.data()),
                     std::streamsize(sizeof(float) * texels));
            if (tex.gcount() !=
                std::streamsize(sizeof(float) * texels)) {
                return false;
            }
        }
    }

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
        } else if (word == "dielectric") {
            m.tag = MaterialTag::Dielectric;
        } else if (word == "conductor") {
            m.tag = MaterialTag::Conductor;
        } else if (word == "measured") {
            m.tag = MaterialTag::Measured;
        } else {
            return false;
        }
        if (!tagged("reflectance")) {
            return false;
        }
        floats(m.reflectance, 3);
        if (!tagged("reflectancetex")) {
            return false;
        }
        in >> m.reflectance_texture;
        if (!tagged("displacement")) {
            return false;
        }
        in >> m.displacement_texture;
        if (!tagged("measured")) {
            return false;
        }
        in >> m.measured;
        if (m.tag == MaterialTag::Dielectric) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (!tagged("eta")) {
                return false;
            }
            floats(&m.eta, 1);
        }
        if (m.tag == MaterialTag::Conductor) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (!tagged("spectra")) {
                return false;
            }
            in >> m.conductor_spectra;
        }
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

    if (!(in >> word) || word != "infinite_lights") {
        return false;
    }
    in >> count;
    if (!tagged("radius")) {
        return false;
    }
    in >> scene.scene_radius;
    scene.infinite_lights.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word) || (word != "uniform" && word != "image")) {
            return false;
        }
        const bool is_image = word == "image";
        InfiniteLight l;
        floats(l.l, 3);
        floats(&l.scale, 1);
        if (!tagged("hasl")) {
            return false;
        }
        in >> l.has_l;
        if (is_image) {
            if (!tagged("resolution")) {
                return false;
            }
            in >> l.resolution;
            if (!tagged("first")) {
                return false;
            }
            in >> l.first_texel;
            if (!tagged("light_from_render")) {
                return false;
            }
            floats(l.light_from_render, 16);
            if (!tagged("render_from_light")) {
                return false;
            }
            floats(l.render_from_light, 16);
        }
        scene.infinite_lights.push_back(l);
    }

    const auto get_shapes = [&](const char *name,
                                std::vector<Shape> &shapes) {
        if (!tagged(name)) {
            return false;
        }
        in >> count;
        shapes.clear();
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
            if (!tagged("alpha")) {
                return false;
            }
            in >> s.alpha;
            if (s.alpha >= int32_t(scene.textures.size())) {
                return false;
            }
            shapes.push_back(s);
        }
        return bool(in);
    };
    const auto get_nodes = [&](const char *name, std::vector<Node> &nodes) {
        if (!tagged(name)) {
            return false;
        }
        in >> count;
        nodes.clear();
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
            nodes.push_back(n);
        }
        return bool(in);
    };

    if (!get_shapes("shapes", scene.shapes) ||
        !get_nodes("nodes", scene.nodes) ||
        !get_shapes("instance_shapes", scene.instance_shapes)) {
        return false;
    }

    if (!tagged("definitions")) {
        return false;
    }
    in >> count;
    scene.definitions.clear();
    for (size_t i = 0; i < count; i++) {
        Definition d;
        if (!tagged("first")) {
            return false;
        }
        in >> d.first_shape;
        if (!tagged("count")) {
            return false;
        }
        in >> d.shape_count;
        if (!tagged("root")) {
            return false;
        }
        in >> d.root_node;
        if (size_t(d.first_shape) + d.shape_count >
            scene.instance_shapes.size()) {
            return false;
        }
        scene.definitions.push_back(d);
    }

    if (!tagged("instances")) {
        return false;
    }
    in >> count;
    scene.instances.clear();
    for (size_t i = 0; i < count; i++) {
        Instance inst;
        if (!tagged("definition")) {
            return false;
        }
        in >> inst.definition;
        if (inst.definition >= scene.definitions.size()) {
            return false;
        }
        if (!tagged("render_from_instance")) {
            return false;
        }
        floats(inst.render_from_instance, 16);
        if (!tagged("instance_from_render")) {
            return false;
        }
        floats(inst.instance_from_render, 16);
        scene.instances.push_back(inst);
    }

    if (!get_nodes("instance_nodes", scene.instance_nodes)) {
        return false;
    }
    for (const Definition &d : scene.definitions) {
        if (!scene.instance_nodes.empty() &&
            d.root_node >= scene.instance_nodes.size()) {
            return false;
        }
    }

    if (!tagged("prims")) {
        return false;
    }
    in >> count;
    scene.prims.clear();
    for (size_t i = 0; i < count; i++) {
        Prim p;
        if (!(in >> word)) {
            return false;
        }
        if (word == "shape") {
            p.kind = PrimShape;
        } else if (word == "instance") {
            p.kind = PrimInstance;
        } else {
            return false;
        }
        in >> p.index;
        const size_t limit = p.kind == PrimShape ? scene.shapes.size()
                                                 : scene.instances.size();
        if (p.index >= limit) {
            return false;
        }
        scene.prims.push_back(p);
    }
    // A tree of PBRT's names every shape and every instance exactly once.
    if (!scene.nodes.empty() &&
        scene.prims.size() != scene.shapes.size() + scene.instances.size()) {
        return false;
    }

    if (!in) {
        return false;
    }

    // And the environment maps' texels, from the file beside this one. How many
    // there are is the sum of the squares of the resolutions, which the lights
    // have already said.
    size_t texels = 0;
    for (const InfiniteLight &l : scene.infinite_lights) {
        texels += size_t(l.resolution) * l.resolution;
    }
    if (texels != 0) {
        std::ifstream env(env_path(path), std::ios::binary);
        if (!env) {
            return false;
        }
        scene.env_texels.resize(texels * 4);
        env.read(reinterpret_cast<char *>(scene.env_texels.data()),
                 std::streamsize(sizeof(float) * scene.env_texels.size()));
        if (env.gcount() !=
            std::streamsize(sizeof(float) * scene.env_texels.size())) {
            return false;
        }
        scene.env_sampling.resize(texels);
        env.read(reinterpret_cast<char *>(scene.env_sampling.data()),
                 std::streamsize(sizeof(float) * scene.env_sampling.size()));
        if (env.gcount() !=
            std::streamsize(sizeof(float) * scene.env_sampling.size())) {
            return false;
        }
    }

    return true;
}

} // namespace bonsai_scene
