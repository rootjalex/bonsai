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
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// The geometry sidecar's record types, generated from scene_geometry.fbs by
// scene_schema.sh (flatc); the header is a build product, found beside this
// file, and brings FlatBuffers' own headers with it.
#include "scene_geometry_generated.h"

namespace bonsai_scene {

enum ShapeTag : uint32_t {
    Sphere = 0,
    Triangle = 1,
    Disk = 2,
    // PBRT's BilinearPatch: a mesh (`mesh`, its indices in fours) and a patch
    // in it (`tri`, reused as the patch index), with the area PBRT's
    // constructor computes.
    Patch = 3,
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
    // A Lambertian lobe on each side of the surface -- a leaf. It reuses
    // `reflectance` for the front and adds `transmittance` for what comes
    // through, both scaled by `scale`.
    DiffuseTransmission = 5,
    // A dielectric coating over a metal: the CoatedDiffuse fields for the
    // interface (its roughness, `thickness`, `eta`, the medium's albedo and
    // `g`, `max_depth`, `n_samples`) and the Conductor fields for the metal
    // (`conductor_spectra` or `reflectance`), plus the metal's own roughness
    // below.
    CoatedConductor = 6,
    // A thin sheet of dielectric (PBRT's ThinDielectricMaterial): `eta`
    // alone, no roughness; two specular lobes, the inter-reflections summed.
    ThinDielectric = 8,
    // PBRT's MixMaterial: two of this file's materials and the float texture
    // that chooses between them per hit. Resolved at the hit, never a BSDF.
    Mix = 9,
    // PBRT's SubsurfaceMaterial: a dielectric boundary (the Dielectric
    // fields: `eta`, the roughness, `remap`) over a tabulated BSSRDF -- its
    // table in `bssrdf_tables`, and its coefficients as sampled spectra or
    // as a reflectance and a mean free path (the fields below `mfp_spectrum`).
    Subsurface = 10,
    // PBRT's `Material "interface"` (and its deprecated spellings `""` and
    // `"none"`), for which `Material::Create` returns no material at all: a
    // surface that scatters nothing and exists only to bound a participating
    // medium. An integrator that hits one steps through it, keeping its
    // direction and taking the medium on the other side
    // (SurfaceInteraction::SkipIntersection), and the depth does not count
    // it. No field below applies.
    Interface = 7,
};

// A participating medium, of the kinds PBRT has: `homogeneous`, `uniformgrid`
// (PBRT's GridMedium), `rgbgrid`, `cloud` and `nanovdb` (a sparse voxel grid
// in NanoVDB's own format, read through NanoVDB's own library -- see
// Medium below and docs/foreign-functions.md).
enum MediumTag : uint32_t {
    Homogeneous = 0,
    UniformGrid = 1,
    RGBGrid = 2,
    Cloud = 3,
    NanoVDB = 4,
};

// A dense grid of a grid medium -- PBRT's SampledGrid, or the MajorantGrid --
// as a run of `medium_grid`: `nx * ny * nz` voxels from `at`, x fastest, one
// float each or four for an RGB spectrum (see Medium). `nx` zero is a grid
// the medium does not have.
struct GridRef {
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    uint32_t at = 0;
};

// The three spectra a medium is made of, each tabulated at the 471 integer
// nanometres from 360 to 830 that PBRT's DenselySampledSpectrum holds -- which
// is how HomogeneousMedium keeps them, so a lookup that rounds to the
// nearest nanometre (the renderer's `spectrum_at_dense`) reproduces PBRT's
// exactly. Laid end to end in `Scene::medium_spectra`, three runs per medium:
// sigma_a, sigma_s, Le, in that order, each already scaled the way
// HomogeneousMedium::Create scales it (`scale` on the two coefficients,
// `Lescale / SpectrumToPhotometric(Le)` on the emission).
inline constexpr int kMediumSpectrumSamples = 471;

struct Medium {
    uint32_t tag = MediumTag::Homogeneous;
    // Where this medium's three spectra begin in `medium_spectra`, in floats.
    uint32_t spectra = 0;
    // PBRT: the Henyey-Greenstein asymmetry of the phase function.
    float g = 0.f;
    // PBRT: Medium::IsEmissive, which is `Le_spec.MaxValue() > 0` -- decided
    // here so that the renderer need not scan the table. For a `uniformgrid`
    // it is PBRT's isEmissive: a temperature grid, or an Le with a value.
    uint32_t emissive = 0;
    // The grid media and the cloud: PBRT's `bounds`, p0 and p1 in the medium's
    // own space, and `renderFromMedium` both ways -- the inverse is PBRT's own
    // mInv, so that the renderer's pull of a ray into the medium's space rounds
    // as PBRT's ApplyInverse does.
    float low[3] = {0.f, 0.f, 0.f};
    float high[3] = {1.f, 1.f, 1.f};
    float render_from_medium[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    float medium_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    // `uniformgrid`: the density, the optional temperature (nx zero for none)
    // and the Lescale grid (one voxel when the scene gave a number, already
    // divided by Le's photometric integral), one float per voxel; and the
    // temperature's offset and scale.
    GridRef density;
    GridRef temperature;
    GridRef le_scale;
    float temperature_scale = 1.f;
    float temperature_offset = 0.f;
    // `rgbgrid`: sigma_a, sigma_s and Le as grids of four floats per voxel --
    // an RGBUnboundedSpectrum's scale and its three sigmoid coefficients, as
    // PBRT's constructor fits them -- any of the three absent; PBRT's `scale`
    // and `Lescale`.
    GridRef sigma_a;
    GridRef sigma_s;
    GridRef le;
    float sigma_scale = 1.f;
    float le_scale_value = 1.f;
    // Both grid kinds: the 16x16x16 majorant grid PBRT's constructor builds,
    // one float per cell -- the maximum of the density over the cell, or for
    // an `rgbgrid` the scaled maximum of sigma_a plus sigma_s.
    GridRef majorant;
    // `cloud`: PBRT's density, wispiness and frequency.
    float cloud_density = 1.f;
    float wispiness = 1.f;
    float frequency = 5.f;
    // `nanovdb`: PBRT's NanoVDBMedium. The grid buffers are copied out of the
    // `.nvdb` file as they are, into `Scene::vdb_bytes` (the `.vdb` sidecar),
    // each starting at a multiple of 32 bytes as NanoVDB's alignment wants;
    // these are where the density grid and the temperature grid begin
    // (`has_temperature` zero for none, and then `temperature_at` unused).
    // The renderer never looks into the bytes: it hands them to NanoVDB's own
    // accessor through a foreign function (media.bonsai, nanovdb_shim.cpp).
    // Each grid's map from the medium's space to its index space, as
    // NanoVDB's `Map` holds it and applies it in `worldToIndexF`: the 3x3
    // inverse matrix, row-major, then the translation taken off first. The
    // emission is `le_scale_value` times a blackbody at the temperature grid's
    // value under `temperature_offset` and `temperature_scale`; `emissive` is
    // PBRT's IsEmissive, a temperature grid and an Lescale above zero. The
    // majorant grid is 64x64x64 here, as PBRT's constructor builds it for this
    // medium, over the grids' world bounds (`low`..`high`).
    uint32_t density_at = 0;
    uint32_t temperature_at = 0;
    uint32_t has_temperature = 0;
    float density_map[12] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
    float temperature_map[12] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
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

// One of PBRT's BSSRDFTable(100, 64)s as `bssrdf_tables` holds it: the 100
// albedo nodes, the 64 radius nodes, the 100 x 64 profile, the 100 effective
// albedos and the 100 x 64 profile integral, end to end.
inline constexpr int kBSSRDFTableFloats = 100 + 64 + 100 * 64 + 100 + 100 * 64;

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
    // PBRT's VolPathIntegrator: `path` with participating media, and the one
    // integrator PBRT's GPU renderer runs. PBRT's own default when a scene
    // names none.
    VolPath = 3,
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

// How a texture level's texels are stored, which is how PBRT's `Image` stores
// them: an 8-bit image (a PNG) stays 8-bit, with the encoding it came with, and
// is decoded when a texel is read -- PBRT's `Image::GetChannel` runs a U256
// texel through `SRGBToLinearLUT`, or divides by 255 for a linear one -- and
// a float image (an EXR, a PFM) stays float. PBRT's MIP pyramid keeps every
// level in the image's format (`Image::GeneratePyramid` converts back to it),
// so a texture's levels all share one format. Keeping the bytes is not an
// optimization but what makes the large scenes possible at all: bistro's
// hundred-odd 2048x2048 PNGs are 1.4 GB as bytes and 5.6 GB as floats, and as
// many normal maps again.
namespace TexelFormat {
enum : uint32_t {
    Float = 0,   // three floats per texel in `texture_texels`, linear
    SRGB8 = 1,   // three bytes per texel in `texture_bytes`, sRGB-encoded
    Linear8 = 2, // three bytes per texel in `texture_bytes`, linear
};
}

// One level of one texture's MIP pyramid, as a run of `texture_texels` or of
// `texture_bytes`, by its format.
//
// Three channels per texel, in the image's own colour space; what the image
// brought and what the texture was declared as are resolved on the way in by
// PBRT's own image reader (see scene_dump.cpp), not here.
struct TextureLevel {
    uint32_t width = 0;
    uint32_t height = 0;
    // Counted in texels and not in floats or bytes, since the renderer reads
    // each pool three channels at a time. Getting that wrong reads three
    // times past the end of the pool, which on a small texture is still
    // mapped memory and so shows up as an occasional segfault rather than as a
    // wrong picture.
    uint32_t first_texel = 0;
    uint32_t format = TexelFormat::Float;
};

// PBRT's FloatTexture and SpectrumTexture, the kinds of texture a material
// parameter may be. One record with a `kind`, the way the Material record is
// one struct with a tag: the renderer's `Texture` variant is built from it
// by the driver. The composite kinds name their operands by index into the
// same table, so the table is a graph -- a `mix` of a `scale` of an image --
// which the converter keeps at most three deep, since the renderer writes
// the depth out rather than recursing (textures.bonsai).
namespace TextureKind {
enum : uint32_t {
    Image = 0,        // an image's MIP pyramid
    Constant = 1,     // a number, or an rgb read as an albedo
    Scale = 2,        // `tex` times the float texture `scale`
    Mix = 3,          // `tex1` and `tex2` lerped by the float texture `amount`
    DirectionMix = 4, // `tex1` and `tex2` lerped by |n . dir|
    FBm = 5,          // fractional Brownian motion of Perlin noise
    Wrinkled = 6,     // turbulence: the octaves' magnitudes
    Windy = 7,        // a fine fbm scaled by a coarse one
    Marble = 8,       // a colour spline through a noised height
};
}

// PBRT's TextureMapping2D, how an image texture finds its coordinate: the
// surface's uv scaled and offset, or one of three geometric mappings through
// the texture's own frame.
namespace MappingKind {
enum : uint32_t { UV = 0, Spherical = 1, Cylindrical = 2, Planar = 3 };
}

// PBRT's ImageTexture, as its MIP pyramid plus what the lookup needs.
//
// The pyramid is built by PBRT's own `MIPMap`, with PBRT's own resampling
// filter, and shipped level by level -- the same division of labour as the
// spectral fits and the BVH. What is left for the renderer is choosing a level
// from the footprint and bilerping in it, which is the part that runs per
// lookup and the part worth transcribing.
struct Texture {
    uint32_t kind = TextureKind::Image;

    // Every kind with a frame of its own: the inverse of the transform in
    // force when the texture was declared, composed into render space, row
    // major, as PBRT's `Inverse(renderFromTexture)`. The identity for a uv
    // mapping, which has no frame.
    float texture_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                     0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};

    // Image: the mapping. PBRT's UVMapping is `st = (su * u + du, sv * v +
    // dv)`; its PlanarMapping is `st = (ds + p . vs, dt + p . vt)` of the
    // point in the texture's frame.
    uint32_t mapping = MappingKind::UV;
    float su = 1.f;
    float sv = 1.f;
    float du = 0.f;
    float dv = 0.f;
    float vs[3] = {1.f, 0.f, 0.f};
    float vt[3] = {0.f, 1.f, 0.f};
    float ds = 0.f;
    float dt = 0.f;
    // PBRT's `scale` and `invert`, applied to the filtered RGB. A `scale`
    // texture over an image one, when its scale is a constant, folds into this:
    // that is what PBRT itself does -- `SpectrumScaledTexture::Create` and its
    // float twin copy the image texture and `MultiplyScale` it rather than
    // wrap it -- so the constant multiplies the filtered colour *before* it is
    // inverted, clamped and fitted to a spectrum, and not the spectrum after.
    // The fit is not linear, so the two orders are different numbers, and
    // check_differentials.sh's `texs` rows are what settles which PBRT takes.
    // A scale by another *texture* is refused rather than flattened.
    float scale = 1.f;
    uint32_t invert = 0;
    // Set for a float texture over a three-channel image, whose value PBRT's
    // `MIPMap::Bilerp<Float>` takes as the *average of the three filtered
    // channels*. The texels are shipped as the three channels and the renderer
    // averages after it filters, because averaging first and filtering the
    // average is the same number only in exact arithmetic. A one-channel image
    // or an RGBA one's alpha is shipped as that channel in all three places
    // and read from the first.
    uint32_t average_channels = 0;
    uint32_t wrap = WrapMode::Repeat;
    // The pyramid, as a run of `texture_levels`, coarsest last.
    uint32_t first_level = 0;
    uint32_t n_levels = 0;

    // Constant: PBRT's FloatConstantTexture's `value`, and its
    // SpectrumConstantTexture's `rgb value`, read as an albedo (the only
    // spectrum type a material's textures have).
    float value = 1.f;
    float rgb[3] = {1.f, 1.f, 1.f};

    // Scale, Mix and DirectionMix: the operands, as indices into this table.
    // Scale reads `tex1` as the texture and `amount` as its float scale;
    // Mix reads `tex1`, `tex2` and the float `amount`; DirectionMix reads
    // `tex1`, `tex2` and `dir`, a unit direction in render space.
    int32_t tex1 = -1;
    int32_t tex2 = -1;
    int32_t amount = -1;
    float dir[3] = {0.f, 1.f, 0.f};

    // The noises: PBRT's `octaves` and `roughness` (omega), and Marble's
    // `scale` and `variation`.
    int32_t octaves = 8;
    float omega = 0.5f;
    float noise_scale = 1.f;
    float variation = 0.2f;
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
    // An index into the sampled spectra (`conductor_eta`, with a zero `k`
    // beside it) when the reflectance was authored as a spectrum -- PBRT's
    // `"spectrum reflectance"`, wavelength/value pairs, a named spectrum or a
    // file, a PiecewiseLinearSpectrum there -- and -1 when it was not. The
    // constant above is then unused: the renderer reads the spectrum at the
    // hit's four wavelengths as it reads a conductor's index.
    // DiffuseTransmission's `transmittance` has the same beside it below.
    int32_t reflectance_spectrum = -1;
    // PBRT's `displacement`, a float texture every material may carry. It does
    // not change what the material *is* -- it tilts the shading frame before
    // the BSDF is built, which is why PBRT keeps it on the base Material and
    // applies it in GetBSDF rather than in any one material's GetBxDF.
    int32_t displacement_texture = -1;
    // PBRT's `normalmap`, the other thing every material may carry, and the
    // one that wins when both are given (`GetBSDF` asks for the normal map
    // first): an image whose texel is the shading normal in the tangent
    // frame, replacing the frame outright rather than tilting it. Not a
    // texture -- PBRT reads it as a plain `Image`, linear, and bilinearly
    // interpolates it with repeat wrap, no pyramid -- so this is an index
    // into `texture_levels`, one level, or -1 for none.
    int32_t normal_map = -1;
    // Conductor only: which pair of `conductor_eta` / `conductor_k` tables this
    // material's index of refraction is -- or, when the scene gave a
    // `reflectance` instead of `eta` and `k`, none: PBRT then takes eta as one
    // and k as the value that reflects `reflectance` at normal incidence, per
    // wavelength at the hit (ConductorMaterial::GetBxDF), and `reflectance` /
    // `reflectance_texture` above hold what it was given.
    int32_t conductor_spectra = -1;
    uint32_t conductor_from_reflectance = 0;
    // Measured only: which entry of `measured_brdfs`.
    int32_t measured = -1;
    // DiffuseTransmission only: what comes through the surface, as an RGB or
    // a texture like `reflectance` above, and PBRT's `scale` on both. PBRT's
    // defaults are 0.25 for each, not `reflectance`'s 0.5, so the converter
    // sets both before it reads the scene's.
    float transmittance[3] = {0.25f, 0.25f, 0.25f};
    int32_t transmittance_texture = -1;
    int32_t transmittance_spectrum = -1;
    float scale = 1.f;
    // CoatedDiffuse, Conductor and Dielectric. The roughness as authored, not
    // as remapped: PBRT remaps per intersection and `remaproughness` says
    // whether it does at all. Each is PBRT's FloatTexture: the constant here,
    // or an index into `textures` when the scene gave a texture, and -1 when
    // it did not, as `reflectance_texture` is for a spectrum.
    float u_roughness = 0.f;
    float v_roughness = 0.f;
    int32_t u_roughness_texture = -1;
    int32_t v_roughness_texture = -1;
    // CoatedConductor only: the metal's roughness, the fields above being
    // the coating's.
    float conductor_u_roughness = 0.f;
    float conductor_v_roughness = 0.f;
    int32_t conductor_u_roughness_texture = -1;
    int32_t conductor_v_roughness_texture = -1;
    uint32_t remap = 1;
    float thickness = 0.01f;
    float eta = 1.5f;
    // Dielectric, ThinDielectric, CoatedDiffuse and CoatedConductor: PBRT's
    // `eta` is a Spectrum. A number is `eta` above; a named glass or a
    // spectrum given some other way is sampled onto the index tables as a
    // conductor's is (conductor_eta, with a zero `k` beside it) and this is
    // its index, or -1 for a number. A spectral index terminates the path's
    // secondary wavelengths at the hit, as PBRT's does.
    int32_t eta_spectrum = -1;
    // Mix only: the two operands by this file's material index -- written
    // before the mix, so both are below it -- and PBRT's `amount`, a float
    // texture defaulting to 0.5, as a constant or an index into `textures`.
    int32_t mix_first = -1;
    int32_t mix_second = -1;
    float mix_amount = 0.5f;
    int32_t mix_amount_texture = -1;
    // The medium between the two interfaces. `has_medium` is not the same
    // question as whether the albedo is zero: PBRT's default is a spectrum that
    // is exactly zero, where an RGB of (0, 0, 0) put through the fit is small
    // and is not, and the layered walk branches on which it has.
    float medium_albedo[3] = {0.f, 0.f, 0.f};
    uint32_t has_medium = 0;
    float g = 0.f;
    // PBRT's `thickness` and `g` are FloatTextures, as the roughnesses are:
    // the constants above, or an index into `textures`, and -1 when the scene
    // gave a number.
    int32_t thickness_texture = -1;
    int32_t g_texture = -1;
    int32_t max_depth = 10;
    int32_t n_samples = 1;
    // Subsurface only. PBRT's SubsurfaceMaterial::Create admits the
    // scattering coefficients four ways: a named measurement (`"string name"
    // "Skin1"`, the table in media.cpp), `sigma_a` and `sigma_s` given, a
    // `reflectance` and a mean free path `mfp`, or nothing (whole milk). The
    // first, second and fourth give sigma_a and sigma_s, each a spectrum
    // sampled onto the index tables (`conductor_eta`, with a zero `k` beside
    // it; PBRT's own Spectrum objects evaluated onto the grid) and
    // `subsurface_from_reflectance` is zero; the third sets it and gives
    // `reflectance` (the fields above, texture and all) and `mfp_spectrum`
    // the same way, PBRT's ConstantSpectrum(1) when no `mfp` was written.
    // `scale` multiplies the coefficients or the mean free path, as PBRT's
    // does; `eta` and the roughness are the boundary's. `bssrdf_table` is
    // which 13,064-float table of `bssrdf_tables` the material's is -- one
    // per distinct (g, eta), computed with PBRT's own
    // ComputeBeamDiffusionBSSRDF.
    int32_t sigma_a_spectrum = -1;
    int32_t sigma_s_spectrum = -1;
    int32_t mfp_spectrum = -1;
    uint32_t subsurface_from_reflectance = 0;
    int32_t bssrdf_table = -1;
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
    // A `blackbody L` in place of the RGB: PBRT's BlackbodySpectrum, Planck's
    // law at `temperature` kelvin normalized to one at its peak, the
    // normalization being the constructor's, computed with PBRT's own
    // `Blackbody` so that the renderer's spectrum is PBRT's to the bit. `l`
    // is then unused, and `scale` was divided by the photometric integral of
    // *this* spectrum, as PBRT divides it.
    uint32_t blackbody = 0;
    float temperature = 0.f;
    float blackbody_normalization = 1.f;
};

// A PointLight, or a SpotLight when `spot` is set: PBRT's two lights at a
// point, with no geometry. The emission is `I` in place of `L`, read as the
// infinite light's below (`has_l` for an RGB illuminant, `blackbody` for a
// blackbody, neither for the colour space's illuminant itself), and `scale`
// is PBRT's: divided by the photometric integral of I, then multiplied by
// `power` over the solid angle the light covers when a power was given, as
// PointLight::Create and SpotLight::Create compute it. `position` is
// renderFromLight(0, 0, 0). A spot carries its cone -- the cosines of the
// angles where the falloff starts and where it ends, PBRT's cosFalloffStart
// and cosFalloffEnd -- and `light_from_render`, the inverse of PBRT's final
// renderFromLight (the scene's transform, the translation to `from` and the
// frame whose z axis points at `to`), 4x4 in row order, through which the
// renderer takes a direction back into the cone's frame as PBRT's
// `ApplyInverse` does.
struct PointLight {
    float l[3] = {1.f, 1.f, 1.f};
    float scale = 1.f;
    uint32_t has_l = 0;
    uint32_t blackbody = 0;
    float temperature = 0.f;
    float blackbody_normalization = 1.f;
    float position[3] = {0.f, 0.f, 0.f};
    uint32_t spot = 0;
    float cos_falloff_start = 0.f;
    float cos_falloff_end = 0.f;
    float light_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
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
    // A `blackbody L`, as on Light above; `has_l` is then zero and `l` unused.
    uint32_t blackbody = 0;
    float temperature = 0.f;
    float blackbody_normalization = 1.f;
    // A DistantLight rather than an infinite one: light from the one
    // direction `direction`, in render space -- PBRT's
    // Normalize(renderFromLight(0, 0, 1)), the direction it arrives *from*.
    // The emission fields above are read as for the uniform light, and
    // `resolution` is zero. It goes in this list because PBRT's light sampler
    // keeps it with the infinite lights, having no bounds.
    uint32_t distant = 0;
    float direction[3] = {0.f, 0.f, 1.f};
    // An ImageInfiniteLight rather than a uniform one: the equal-area
    // octahedral environment map's square resolution, and where its texels
    // begin in the scene's shared texel pool. Zero resolution means there is no
    // image and this is the uniform light above.
    uint32_t resolution = 0;
    uint32_t first_texel = 0;
    // Which illuminant the texels' fits are multiplied by: the image's colour
    // space's (PBRT builds an RGBIlluminantSpectrum against the image's
    // space). Zero is D65 -- sRGB's and DCI-P3's, which the renderer's driver
    // has -- and k > 0 is entry k - 1 of `env_illuminants` below, the dense
    // spectrum of another space's illuminant (ACES2065-1's D60).
    uint32_t illuminant = 0;
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
    // A PortalImageInfiniteLight rather than an ImageInfiniteLight: the sky
    // is seen only through a rectangle, PBRT's `portal`, whose four corners
    // are here in render space (PBRT moves them with the camera transform
    // and not with the light's). `resolution` and `first_texel` then name
    // the *rectified* image -- the environment map resampled into the
    // portal's own parameterization, which is what the light reads -- and
    // its sampling values are the texel average times PBRT's d(u,v)/dw
    // Jacobian at the texel; the two matrices above are unused, the light's
    // transform having been baked into the resampling.
    uint32_t portal = 0;
    float portal_points[12] = {0.f};
};

struct Shape {
    uint32_t tag;
    // Sphere and disk: the radius. The sphere's clipping -- PBRT's
    // constructor's zmin and zmax clamped to the radius, their arc cosines
    // (the theta the v coordinate runs between), and phimax in radians
    // clamped to a turn, all computed as the constructor computes them.
    float radius = 0.f;
    float z_min = 0.f;
    float z_max = 0.f;
    float theta_z_min = 0.f;
    float theta_z_max = 0.f;
    // PBRT's reverseOrientation ^ transformSwapsHandedness: which way the
    // surface normal points.
    uint32_t flip = 0;
    // Triangle.
    uint32_t mesh = 0;
    uint32_t tri = 0;
    // Sphere and disk. PBRT's quadrics are in an object space their transform
    // places, so both matrices come along, 4x4 in row order, as an instance's
    // do. `phi_max` is in radians, clamped to a turn, as PBRT's constructors
    // leave it; `reverse` is reverseOrientation alone, which PBRT turns a
    // *sampled* point's normal by where a hit's normal is turned by `flip`.
    float render_from_object[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    float object_from_render[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    float height = 0.f;
    float inner_radius = 0.f;
    float phi_max = 6.28318530717958647692f;
    uint32_t reverse = 0;
    // Patch. PBRT's BilinearPatch::area -- exact for a rectangle, else the
    // sum over a 3x3 subdivision -- computed by the converter as its
    // constructor computes it, since the renderer's sampling divides by it.
    float patch_area = 0.f;
    // Which of the scene's materials this shape was declared under.
    uint32_t material = 0;
    // Which of the scene's lights this shape emits as, or -1 for a shape that
    // does not emit. PBRT's ShapeSceneEntity::lightIndex, which is per shape
    // rather than per material because `AreaLightSource` is a graphics-state
    // directive like `Material` and the two are set independently.
    int32_t light = -1;
    // The slot this emitter takes in the renderer's per-shape light list, and
    // so the leaf index the light-tree below points at, or -1 for a shape that
    // does not emit. Assigned in scene_dump's enumeration order and carried on
    // the shape so that it survives the BVH reorder the driver does -- the
    // light order the tree was built over cannot otherwise be recovered once
    // the shapes have moved. See the note on the light tree.
    int32_t light_ordinal = -1;
    // PBRT's GeometricPrimitive alpha texture, or -1 where the shape has none.
    // A cutout: the texture says how much of the surface is really there, which
    // is what makes a tree leaf leaf-shaped rather than a quad.
    int32_t alpha = -1;
    // PBRT's MediumInterface on the shape: which of the scene's media is on
    // the side its normal points away from and which on the side it points to,
    // or -1 for none. A shape whose two sides name the same medium (both
    // none, usually) is not a medium transition -- `MediumInterface::
    // IsMediumTransition` -- and a ray crossing it keeps the medium it had; one
    // whose sides differ hands a ray leaving it the medium on the side it
    // leaves towards (`Interaction::GetMedium`).
    int32_t medium_inside = -1;
    int32_t medium_outside = -1;
};

// One node of PBRT's BVHLightSampler tree, dequantized.
//
// PBRT keeps a CompactLightBounds -- the box quantized to sixteen bits and the
// direction to an octahedral code -- because the tree is walked per shading
// point and cache footprint matters. The importance it computes reads those
// back out through accessors that dequantize, so what actually enters the
// arithmetic is the box and cosines at reduced precision. scene_dump builds
// the tree with PBRT's own code and stores exactly those dequantized values,
// so the renderer's importance runs on plain floats and matches PBRT bit for
// bit without carrying the quantizer.
struct LightTreeNode {
    // The bounds' average emission direction (CompactLightBounds' `w`), decoded
    // from the octahedral code so it is the vector PBRT's importance dots.
    float w[3] = {0.f, 0.f, 1.f};
    float phi = 0.f;
    float cos_theta_o = 1.f;
    float cos_theta_e = 0.f;
    float bounds_min[3] = {0.f, 0.f, 0.f};
    float bounds_max[3] = {0.f, 0.f, 0.f};
    uint32_t two_sided = 0;
    // An interior node's second child, an absolute node index; the first child
    // is the next node along. A leaf's light ordinal.
    uint32_t child_or_light = 0;
    uint32_t is_leaf = 0;
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
    ZSobol = 3,
    Sobol = 4,
    PaddedSobol = 5,
    PMJ02BN = 6,
};

// pbrt: RandomizeStrategy, in pbrt's order -- how a low-discrepancy sampler
// breaks up the correlation between its dimensions. `permutedigits` is what a
// Halton sampler gets when the scene does not say, `fastowen` what a zsobol
// one gets; pbrt's Halton has no `fastowen` and refuses it.
enum RandomizeTag : uint32_t {
    RandomizeNone = 0,
    RandomizePermuteDigits = 1,
    RandomizeFastOwen = 2,
    RandomizeOwen = 3,
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
    // ZSobol and Sobol, besides the randomization: log2 of the film's full
    // resolution rounded up to a power of two. For ZSobol it is, with log2 of
    // the sample count, how many base-4 digits of a sample's Morton index the
    // sampler permutes; for Sobol it is the constructor's `scale`, the
    // power-of-two side the first two dimensions of the sequence are mapped
    // onto. Derived where pbrt's constructors derive it, from the resolution
    // pbrt hands them (scene_dump.cpp); the sample count's part is the
    // driver's, since `--spp` can change the count after this is written.
    // PaddedSobol carries the randomization and the count and nothing
    // derived; PMJ02BN the count alone -- the side of the tile its camera
    // samples are sorted into follows from the count, in the driver.
    int32_t log2_resolution = 0;
};

struct Scene {
    // The film's full resolution, which the camera's raster transform and
    // the sampler's resolution are for -- and the pixel bounds within it
    // that are rendered, PBRT's Film::PixelBounds(): the whole frame unless
    // the scene gave a `cropwindow` or `pixelbounds` (scene_dump.cpp), and
    // then [x0, x1) x [y0, y1) in the frame's pixel coordinates. The
    // renderer walks the window's pixels under their frame coordinates and
    // the driver writes an image of the window's size, as PBRT's does.
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixel_x0 = 0;
    uint32_t pixel_y0 = 0;
    uint32_t pixel_x1 = 0;
    uint32_t pixel_y1 = 0;
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
    // PBRT: Film::UsesVisibleSurface(), true for `Film "gbuffer"` and false
    // for `rgb` and `spectral`. What decides whether the path integrator
    // fills a VisibleSurface at its first vertex -- the reflectance estimate
    // it costs is sixteen BSDF samples per camera ray, which pbrt spends only
    // for a film that records them.
    uint32_t film_visible_surface = 0;
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
    // PBRT: PixelSensor's three response curves and RGBFilm's
    // `outputRGBFromSensorRGB`. A camera's sensor records a radiance as three
    // numbers by integrating it against its own r/g/b spectral responses, and
    // the film then maps those to the output space through one 3x3. The default
    // `cie1931` sensor uses the CIE X/Y/Z curves and a white-balance matrix; a
    // named sensor like `canon_eos_100d` uses the camera's measured curves and a
    // matrix pbrt fits by least squares over the Macbeth chart. Both are built
    // by pbrt's own PixelSensor in scene_dump, so this carries the result: the
    // three curves resampled to the same 360..830 nm grid as `cie_x` and the
    // 3x3 `RGBFromXYZ * XYZFromSensorRGB`, row-major. Always present -- the
    // default sensor fills the curves with X/Y/Z -- so the renderer has one path.
    std::vector<float> sensor_r;
    std::vector<float> sensor_g;
    std::vector<float> sensor_b;
    float output_rgb_from_sensor[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
                                       0.f, 0.f, 1.f};
    // PBRT: RGBFilm's `maxcomponentvalue`, which clamps each *sample*'s sensor
    // RGB before it is accumulated -- a firefly suppressor. Infinite unless the
    // scene names one.
    float max_component_value = std::numeric_limits<float>::infinity();
    std::vector<Material> materials;
    // The participating media, and their spectra (see Medium). Only the media
    // some shape's interface or the camera names are written.
    std::vector<Medium> media;
    std::vector<float> medium_spectra;
    // The grid media's grids, laid end to end (see GridRef). Binary, in the
    // `.vol` sidecar: smoke-plume's density alone is nine million floats.
    std::vector<float> medium_grid;
    // The `nanovdb` media's grid buffers, copied out of their `.nvdb` files
    // as they are (see Medium::density_at). Binary, in the `.vdb` sidecar:
    // 76 MB for bunny-cloud's.
    std::vector<uint8_t> vdb_bytes;
    // The tables the scene's sampler reads, when it is one of the two that
    // read tables pbrt ships as source (sampler.bonsai's externs): the `sobol`
    // sampler's generator matrices for all 1024 dimensions and the two van
    // der Corput matrices SobolIntervalToIndex works with (25 rows of 52);
    // the `pmj02bn` sampler's five sets of 65536 blue-noise points (x then y,
    // fixed point) and its forty-eight 128x128 blue-noise textures. Binary,
    // in the `.smp` sidecar -- 233 KB for sobol, 4.1 MB for pmj02bn -- copied
    // from pbrt's own arrays by scene_dump, which links pbrt, so that no
    // transcription stands between pbrt's tables and the renderer's; the two
    // dimensions zsobol and paddedsobol read are small enough to be a header
    // (sobol_tables.h). Empty for every other sampler.
    std::vector<uint32_t> sobol_matrices;
    std::vector<uint64_t> vdc_matrices;
    std::vector<uint64_t> vdc_matrices_inv;
    std::vector<uint32_t> pmj02bn_samples;
    std::vector<uint16_t> blue_noise;
    // PBRT's `CameraMedium`: the medium the camera sits in, which every camera
    // ray starts in, or -1 for none.
    int32_t camera_medium = -1;
    // The meshes, and the pools their runs live in. Three floats per position
    // and normal, two per texture coordinate.
    std::vector<Mesh> meshes;
    std::vector<uint32_t> indices;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<Light> lights;
    // PBRT's point and spot lights, in declaration order. The driver puts
    // them after the area lights and before the infinite ones -- bounded, so
    // inside the run PBRT's light tree covers, at the ordinals the converter
    // gave PBRT's own PointLight and SpotLight objects when it built the tree
    // (which is PBRT's own order too: BasicScene::CreateLights lists every
    // area light first and then the `LightSource` lights as declared).
    std::vector<PointLight> point_lights;
    std::vector<InfiniteLight> infinite_lights;
    // Which light sampler the path integrator draws with: 0 for the uniform
    // one, 1 for PBRT's BVH. The uniform sampler needs nothing below it; the
    // BVH one is the tree and the per-ordinal bit trails that follow. A scene
    // with one non-infinite light resolves to uniform either way, since the
    // two are the same function there.
    uint32_t light_sampler = 0;
    // PBRT's BVHLightSampler, over the bounded (area) lights only -- infinite
    // lights are sampled uniformly beside the tree, as PBRT does. Node 0 is the
    // root; a leaf's `child_or_light` is a light ordinal.
    std::vector<LightTreeNode> light_tree;
    // Per light ordinal, the sequence of left/right turns from the root to that
    // light's leaf, low bit first -- what PBRT's PMF walks back down to recover
    // the probability the light was chosen with.
    std::vector<uint32_t> light_bit_trails;
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
    // The illuminants of the environment maps' colour spaces other than D65,
    // 471 values each over 360 to 830 nm, end to end: what an image light's
    // `illuminant - 1` indexes (ACES2065-1's D60 for a sky in ACES). Small,
    // so in the scene file itself.
    std::vector<float> env_illuminants;
    // The image textures, their pyramid levels, and every level's texels laid
    // end to end. The texels go in the sidecar beside the environment maps and
    // for the same reason: a 2048x2048 pyramid is seventeen million numbers.
    std::vector<Texture> textures;
    std::vector<TextureLevel> texture_levels;
    std::vector<float> texture_texels;
    // The 8-bit levels' texels, three bytes each, as the images stored them
    // (TexelFormat::SRGB8 and Linear8).
    std::vector<uint8_t> texture_bytes;
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
    // Every spectrum the scene's materials gave as one: a conductor's index of
    // refraction and extinction as a pair, a dielectric's index with a zero
    // extinction beside it, a reflectance given as a spectrum likewise.
    // kConductorSamples entries each, a tenth of a nanometre apart from
    // 360 nm, laid end to end, one run per distinct spectrum a material named.
    //
    // Resampled from PBRT's own spectra rather than fitted. PBRT keeps them
    // as PiecewiseLinearSpectrum and interpolates between the published
    // measurements or the scene's pairs; this is that function on the grid,
    // which reproduces it exactly except inside the single step containing
    // each of its own knots. scene_dump measures that residual and prints it.
    std::vector<float> conductor_eta;
    std::vector<float> conductor_k;
    // The subsurface materials' BSSRDF tables, PBRT's BSSRDFTable(100, 64)
    // laid end to end: rhoSamples (100), radiusSamples (64), profile (6,400),
    // rhoEff (100), profileCDF (6,400) -- 13,064 floats a table, one table
    // per distinct (g, eta) the scene's subsurface materials name, computed
    // by PBRT's own ComputeBeamDiffusionBSSRDF in the converter. A material's
    // `bssrdf_table` is which.
    std::vector<float> bssrdf_tables;
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
    // A bilinear patch's four, in PBRT's order: p00, p10, p01, p11.
    void patch_corners(const Shape &s, uint32_t out[4]) const {
        const Mesh &m = meshes[s.mesh];
        for (uint32_t k = 0; k < 4; k++) {
            out[k] = m.first_vertex + indices[m.first_index + 4 * s.tri + k];
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

// Where the grid media's voxels live (Scene::medium_grid).
inline std::string vol_path(const char *scene_path) {
    return std::string(scene_path) + ".vol";
}

// Where the sampler's tables live (Scene::sobol_matrices and the four after
// it), when the scene's sampler reads any.
inline std::string smp_path(const char *scene_path) {
    return std::string(scene_path) + ".smp";
}

// Where the NanoVDB grids live (Scene::vdb_bytes), when the scene has any.
inline std::string vdb_path(const char *scene_path) {
    return std::string(scene_path) + ".vdb";
}

// Where the geometry lives: the FlatBuffer beside the scene file
// (scene_geometry.fbs). The meshes, their vertices, every shape, the trees
// and the instances are arrays of fixed-layout records, and the text form
// of them parsed at 80-90 MB/s -- watercolor's 24 million shapes took 91.6 s
// to read before a render of 0.9 s. As a FlatBuffer they are read in place
// and copied into the vectors below at memory speed.
inline std::string geo_path(const char *scene_path) {
    return std::string(scene_path) + ".geo";
}

namespace detail {

// Bumped whenever a record in scene_geometry.fbs changes layout; a sidecar
// of another version is refused rather than misread, as a text file with a
// missing tag is.
constexpr uint32_t kGeometryVersion = 2;

inline geo::Placement placement_of(const Shape &s) {
    return geo::Placement(s.material, s.light, s.light_ordinal, s.alpha,
                          s.medium_inside, s.medium_outside);
}

inline void place(const geo::Placement &at, Shape &s) {
    s.material = at.material();
    s.light = at.light();
    s.light_ordinal = at.light_ordinal();
    s.alpha = at.alpha();
    s.medium_inside = at.medium_inside();
    s.medium_outside = at.medium_outside();
}

inline flatbuffers::span<const float, 3> span3(const float *v) {
    return flatbuffers::span<const float, 3>(v, 3);
}
inline flatbuffers::span<const float, 16> span16(const float *v) {
    return flatbuffers::span<const float, 16>(v, 16);
}

// A shape list as the sidecar holds it: the kind and the slot of every shape
// in the order the list had, and a vector per kind the slots index. One
// record per shape of its own size, where the text wrote a line per shape
// and `Shape` holds every kind's fields.
struct ShapeColumns {
    std::vector<uint8_t> kinds;
    std::vector<uint32_t> slots;
    std::vector<geo::Triangle> triangles;
    std::vector<geo::Patch> patches;
    std::vector<geo::Sphere> spheres;
    std::vector<geo::Disk> disks;
};

inline ShapeColumns columns_of(const std::vector<Shape> &shapes) {
    ShapeColumns c;
    c.kinds.reserve(shapes.size());
    c.slots.reserve(shapes.size());
    for (const Shape &s : shapes) {
        c.kinds.push_back(uint8_t(s.tag));
        const geo::Placement at = placement_of(s);
        switch (s.tag) {
        case ShapeTag::Sphere:
            c.slots.push_back(uint32_t(c.spheres.size()));
            c.spheres.emplace_back(s.radius, s.z_min, s.z_max, s.theta_z_min,
                                   s.theta_z_max, s.phi_max, s.flip, s.reverse,
                                   span16(s.render_from_object),
                                   span16(s.object_from_render), at);
            break;
        case ShapeTag::Disk:
            c.slots.push_back(uint32_t(c.disks.size()));
            c.disks.emplace_back(s.height, s.radius, s.inner_radius, s.phi_max,
                                 s.flip, s.reverse, span16(s.render_from_object),
                                 span16(s.object_from_render), at);
            break;
        case ShapeTag::Patch:
            c.slots.push_back(uint32_t(c.patches.size()));
            c.patches.emplace_back(s.mesh, s.tri, s.patch_area, at);
            break;
        default:
            c.slots.push_back(uint32_t(c.triangles.size()));
            c.triangles.emplace_back(s.mesh, s.tri, at);
            break;
        }
    }
    return c;
}

inline std::vector<geo::Node> nodes_of(const std::vector<Node> &nodes) {
    std::vector<geo::Node> out;
    out.reserve(nodes.size());
    for (const Node &n : nodes) {
        out.emplace_back(span3(n.low), span3(n.high), n.offset, n.n_prims, n.axis);
    }
    return out;
}

// The sidecar written: every 64-bit vector first -- FlatBuffers places the
// 64-bit-addressed data ahead of the 32-bit-addressed and its builder keeps
// that order -- then the small vectors and the table. `bytes` is the size
// the scene text records, which the reader checks the file against.
inline bool write_geometry(const char *path, const Scene &scene,
                           uint64_t *bytes) {
    flatbuffers::FlatBufferBuilder64 fbb;
    const auto indices = fbb.CreateVector64(scene.indices);
    const auto positions = fbb.CreateVector64(scene.positions);
    const auto normals = fbb.CreateVector64(scene.normals);
    const auto uvs = fbb.CreateVector64(scene.uvs);
    const ShapeColumns top = columns_of(scene.shapes);
    const auto shape_kinds = fbb.CreateVector64(top.kinds);
    const auto shape_slots = fbb.CreateVector64(top.slots);
    const auto triangles = fbb.CreateVectorOfStructs64(top.triangles);
    const auto patches = fbb.CreateVectorOfStructs64(top.patches);
    const auto spheres = fbb.CreateVectorOfStructs64(top.spheres);
    const auto disks = fbb.CreateVectorOfStructs64(top.disks);
    const auto nodes = fbb.CreateVectorOfStructs64(nodes_of(scene.nodes));
    const ShapeColumns inst = columns_of(scene.instance_shapes);
    const auto instance_shape_kinds = fbb.CreateVector64(inst.kinds);
    const auto instance_shape_slots = fbb.CreateVector64(inst.slots);
    const auto instance_triangles = fbb.CreateVectorOfStructs64(inst.triangles);
    const auto instance_patches = fbb.CreateVectorOfStructs64(inst.patches);
    const auto instance_spheres = fbb.CreateVectorOfStructs64(inst.spheres);
    const auto instance_disks = fbb.CreateVectorOfStructs64(inst.disks);
    const auto instance_nodes =
        fbb.CreateVectorOfStructs64(nodes_of(scene.instance_nodes));
    std::vector<geo::Prim> prim_records;
    prim_records.reserve(scene.prims.size());
    for (const Prim &p : scene.prims) {
        prim_records.emplace_back(p.kind, p.index);
    }
    const auto prims = fbb.CreateVectorOfStructs64(prim_records);

    std::vector<geo::Mesh> mesh_records;
    mesh_records.reserve(scene.meshes.size());
    for (const Mesh &m : scene.meshes) {
        mesh_records.emplace_back(m.first_index, m.first_vertex, m.first_normal,
                                  m.first_uv, m.has_normals, m.has_uv, m.flip);
    }
    const auto meshes = fbb.CreateVectorOfStructs(mesh_records);
    std::vector<geo::Definition> definition_records;
    definition_records.reserve(scene.definitions.size());
    for (const Definition &d : scene.definitions) {
        definition_records.emplace_back(d.first_shape, d.shape_count, d.root_node);
    }
    const auto definitions = fbb.CreateVectorOfStructs(definition_records);
    std::vector<geo::Instance> instance_records;
    instance_records.reserve(scene.instances.size());
    for (const Instance &i : scene.instances) {
        instance_records.emplace_back(i.definition, span16(i.render_from_instance),
                                      span16(i.instance_from_render));
    }
    const auto instances = fbb.CreateVectorOfStructs(instance_records);

    geo::GeometryBuilder gb(fbb);
    gb.add_version(kGeometryVersion);
    gb.add_meshes(meshes);
    gb.add_indices(indices);
    gb.add_positions(positions);
    gb.add_normals(normals);
    gb.add_uvs(uvs);
    gb.add_shape_kinds(shape_kinds);
    gb.add_shape_slots(shape_slots);
    gb.add_triangles(triangles);
    gb.add_patches(patches);
    gb.add_spheres(spheres);
    gb.add_disks(disks);
    gb.add_nodes(nodes);
    gb.add_instance_shape_kinds(instance_shape_kinds);
    gb.add_instance_shape_slots(instance_shape_slots);
    gb.add_instance_triangles(instance_triangles);
    gb.add_instance_patches(instance_patches);
    gb.add_instance_spheres(instance_spheres);
    gb.add_instance_disks(instance_disks);
    gb.add_instance_nodes(instance_nodes);
    gb.add_definitions(definitions);
    gb.add_instances(instances);
    gb.add_prims(prims);
    fbb.Finish(gb.Finish(), geo::GeometryIdentifier());

    std::ofstream out(geo_path(path), std::ios::binary);
    out.write(reinterpret_cast<const char *>(fbb.GetBufferPointer()),
              std::streamsize(fbb.GetSize()));
    *bytes = uint64_t(fbb.GetSize());
    return bool(out);
}

// A shape list back from its columns. False where a slot points past its
// vector, which no writer of this file produces.
template <typename Kinds, typename Slots, typename Tris, typename Patches,
          typename Spheres, typename Disks>
inline bool shapes_of(const Kinds *kinds, const Slots *slots, const Tris *tris,
                      const Patches *patches, const Spheres *spheres,
                      const Disks *disks, std::vector<Shape> &out) {
    out.clear();
    if (kinds == nullptr || slots == nullptr) {
        return true;
    }
    if (kinds->size() != slots->size()) {
        return false;
    }
    out.resize(kinds->size());
    for (size_t i = 0; i < out.size(); i++) {
        Shape &s = out[i];
        const uint32_t slot = slots->Get(i);
        switch (kinds->Get(i)) {
        case ShapeTag::Sphere: {
            if (spheres == nullptr || slot >= spheres->size()) {
                return false;
            }
            const geo::Sphere *r = spheres->Get(slot);
            s.tag = ShapeTag::Sphere;
            s.radius = r->radius();
            s.z_min = r->z_min();
            s.z_max = r->z_max();
            s.theta_z_min = r->theta_z_min();
            s.theta_z_max = r->theta_z_max();
            s.phi_max = r->phi_max();
            s.flip = r->flip();
            s.reverse = r->reverse();
            std::memcpy(s.render_from_object, r->render_from_object()->data(),
                        sizeof(s.render_from_object));
            std::memcpy(s.object_from_render, r->object_from_render()->data(),
                        sizeof(s.object_from_render));
            place(r->at(), s);
            break;
        }
        case ShapeTag::Disk: {
            if (disks == nullptr || slot >= disks->size()) {
                return false;
            }
            const geo::Disk *r = disks->Get(slot);
            s.tag = ShapeTag::Disk;
            s.height = r->height();
            s.radius = r->radius();
            s.inner_radius = r->inner_radius();
            s.phi_max = r->phi_max();
            s.flip = r->flip();
            s.reverse = r->reverse();
            std::memcpy(s.render_from_object, r->render_from_object()->data(),
                        sizeof(s.render_from_object));
            std::memcpy(s.object_from_render, r->object_from_render()->data(),
                        sizeof(s.object_from_render));
            place(r->at(), s);
            break;
        }
        case ShapeTag::Patch: {
            if (patches == nullptr || slot >= patches->size()) {
                return false;
            }
            const geo::Patch *r = patches->Get(slot);
            s.tag = ShapeTag::Patch;
            s.mesh = r->mesh();
            s.tri = r->patch();
            s.patch_area = r->area();
            place(r->at(), s);
            break;
        }
        case ShapeTag::Triangle: {
            if (tris == nullptr || slot >= tris->size()) {
                return false;
            }
            const geo::Triangle *r = tris->Get(slot);
            s.tag = ShapeTag::Triangle;
            s.mesh = r->mesh();
            s.tri = r->tri();
            place(r->at(), s);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

template <typename Nodes>
inline void nodes_from(const Nodes *in, std::vector<Node> &out) {
    out.clear();
    if (in == nullptr) {
        return;
    }
    out.resize(in->size());
    for (size_t i = 0; i < out.size(); i++) {
        const geo::Node *r = in->Get(i);
        std::memcpy(out[i].low, r->low()->data(), sizeof(out[i].low));
        std::memcpy(out[i].high, r->high()->data(), sizeof(out[i].high));
        out[i].offset = r->offset();
        out[i].n_prims = r->n_prims();
        out[i].axis = r->axis();
    }
}

template <typename T, typename V>
inline void copy_from(const V *in, std::vector<T> &out) {
    out.clear();
    if (in != nullptr) {
        out.assign(in->data(), in->data() + in->size());
    }
}

// The sidecar read: the whole file into memory, verified -- FlatBuffers'
// verifier walks the table and bounds every vector, which for vectors of
// structs is a size check each -- and copied out into the scene's vectors.
// `bytes` is what the scene text said the file is.
inline bool read_geometry(const char *path, uint64_t bytes, Scene &scene) {
    std::ifstream in(geo_path(path), std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const std::streamoff size = in.tellg();
    if (size < 0 || uint64_t(size) != bytes) {
        return false;
    }
    in.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(buf.data()), size);
    if (in.gcount() != size) {
        return false;
    }
    flatbuffers::Verifier::Options opts;
    // The default is FlatBuffers' 32-bit limit; the 64-bit vectors may pass it.
    opts.max_size = buf.size() + 1;
    flatbuffers::Verifier verifier(buf.data(), buf.size(), opts);
    if (!geo::VerifyGeometryBuffer(verifier)) {
        return false;
    }
    const geo::Geometry *g = geo::GetGeometry(buf.data());
    if (g->version() != kGeometryVersion) {
        return false;
    }

    scene.meshes.clear();
    if (g->meshes() != nullptr) {
        scene.meshes.reserve(g->meshes()->size());
        for (const geo::Mesh *m : *g->meshes()) {
            Mesh out;
            out.first_index = m->first_index();
            out.first_vertex = m->first_vertex();
            out.first_normal = m->first_normal();
            out.first_uv = m->first_uv();
            out.has_normals = m->has_normals();
            out.has_uv = m->has_uv();
            out.flip = m->flip();
            scene.meshes.push_back(out);
        }
    }
    copy_from(g->indices(), scene.indices);
    copy_from(g->positions(), scene.positions);
    copy_from(g->normals(), scene.normals);
    copy_from(g->uvs(), scene.uvs);
    if (!shapes_of(g->shape_kinds(), g->shape_slots(), g->triangles(),
                   g->patches(), g->spheres(), g->disks(), scene.shapes)) {
        return false;
    }
    nodes_from(g->nodes(), scene.nodes);
    if (!shapes_of(g->instance_shape_kinds(), g->instance_shape_slots(),
                   g->instance_triangles(), g->instance_patches(),
                   g->instance_spheres(), g->instance_disks(),
                   scene.instance_shapes)) {
        return false;
    }
    nodes_from(g->instance_nodes(), scene.instance_nodes);
    scene.definitions.clear();
    if (g->definitions() != nullptr) {
        for (const geo::Definition *d : *g->definitions()) {
            Definition out;
            out.first_shape = d->first_shape();
            out.shape_count = d->shape_count();
            out.root_node = d->root_node();
            scene.definitions.push_back(out);
        }
    }
    scene.instances.clear();
    if (g->instances() != nullptr) {
        for (const geo::Instance *i : *g->instances()) {
            Instance out;
            out.definition = i->definition();
            std::memcpy(out.render_from_instance, i->render_from_instance()->data(),
                        sizeof(out.render_from_instance));
            std::memcpy(out.instance_from_render, i->instance_from_render()->data(),
                        sizeof(out.instance_from_render));
            scene.instances.push_back(out);
        }
    }
    scene.prims.clear();
    if (g->prims() != nullptr) {
        scene.prims.resize(g->prims()->size());
        for (size_t i = 0; i < scene.prims.size(); i++) {
            const geo::Prim *p = g->prims()->Get(i);
            scene.prims[i].kind = p->kind();
            scene.prims[i].index = p->index();
        }
    }
    return true;
}

// What the text reader checked about the geometry as it read it, checked
// once it is all in: every index a shape, a definition, an instance or a
// primitive carries points at something that exists. The lights are read
// after the geometry, which is why this runs where the shapes' section used
// to be read.
inline bool validate_geometry(const Scene &scene) {
    const auto shapes_ok = [&](const std::vector<Shape> &shapes) {
        for (const Shape &s : shapes) {
            if ((s.tag == ShapeTag::Triangle || s.tag == ShapeTag::Patch) &&
                s.mesh >= scene.meshes.size()) {
                return false;
            }
            if (s.material >= scene.materials.size() ||
                s.light >= int32_t(scene.lights.size()) ||
                s.alpha >= int32_t(scene.textures.size()) ||
                s.medium_inside >= int32_t(scene.media.size()) ||
                s.medium_outside >= int32_t(scene.media.size())) {
                return false;
            }
        }
        return true;
    };
    if (!shapes_ok(scene.shapes) || !shapes_ok(scene.instance_shapes)) {
        return false;
    }
    for (const Definition &d : scene.definitions) {
        if (size_t(d.first_shape) + d.shape_count > scene.instance_shapes.size()) {
            return false;
        }
        if (!scene.instance_nodes.empty() &&
            d.root_node >= scene.instance_nodes.size()) {
            return false;
        }
    }
    for (const Instance &i : scene.instances) {
        if (i.definition >= scene.definitions.size()) {
            return false;
        }
    }
    for (const Prim &p : scene.prims) {
        if (p.kind != PrimShape && p.kind != PrimInstance) {
            return false;
        }
        const size_t limit = p.kind == PrimShape ? scene.shapes.size()
                                                 : scene.instances.size();
        if (p.index >= limit) {
            return false;
        }
    }
    // A tree of PBRT's names every shape and every instance exactly once.
    if (!scene.nodes.empty() &&
        scene.prims.size() != scene.shapes.size() + scene.instances.size()) {
        return false;
    }
    return true;
}

} // namespace detail

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
    if (!scene.texture_texels.empty() || !scene.rgb_table.empty() ||
        !scene.texture_bytes.empty()) {
        std::ofstream tex(texel_path(path), std::ios::binary);
        if (!tex) {
            return false;
        }
        // The table first, so a reader can take it without knowing how many
        // texels follow; then the float texels, then the bytes.
        tex.write(reinterpret_cast<const char *>(scene.rgb_table.data()),
                  std::streamsize(sizeof(float) * scene.rgb_table.size()));
        tex.write(
            reinterpret_cast<const char *>(scene.texture_texels.data()),
            std::streamsize(sizeof(float) * scene.texture_texels.size()));
        tex.write(reinterpret_cast<const char *>(scene.texture_bytes.data()),
                  std::streamsize(scene.texture_bytes.size()));
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
    if (!scene.medium_grid.empty()) {
        std::ofstream vol(vol_path(path), std::ios::binary);
        if (!vol) {
            return false;
        }
        vol.write(reinterpret_cast<const char *>(scene.medium_grid.data()),
                  std::streamsize(sizeof(float) * scene.medium_grid.size()));
        if (!vol) {
            return false;
        }
    }
    if (!scene.vdb_bytes.empty()) {
        std::ofstream vdb(vdb_path(path), std::ios::binary);
        if (!vdb) {
            return false;
        }
        vdb.write(reinterpret_cast<const char *>(scene.vdb_bytes.data()),
                  std::streamsize(scene.vdb_bytes.size()));
        if (!vdb) {
            return false;
        }
    }
    if (!scene.sobol_matrices.empty() || !scene.pmj02bn_samples.empty()) {
        // The five tables back to back, in the order the text line counts
        // them.
        std::ofstream smp(smp_path(path), std::ios::binary);
        if (!smp) {
            return false;
        }
        const auto put = [&](const auto &v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            smp.write(reinterpret_cast<const char *>(v.data()),
                      std::streamsize(sizeof(T) * v.size()));
        };
        put(scene.sobol_matrices);
        put(scene.vdc_matrices);
        put(scene.vdc_matrices_inv);
        put(scene.pmj02bn_samples);
        put(scene.blue_noise);
        if (!smp) {
            return false;
        }
    }

    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "resolution " << scene.width << ' ' << scene.height << '\n';
    out << "pixelbounds " << scene.pixel_x0 << ' ' << scene.pixel_y0 << ' '
        << scene.pixel_x1 << ' ' << scene.pixel_y1 << '\n';
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
    } else if (scene.sampler.tag == SamplerTag::ZSobol) {
        out << "sampler zsobol " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << ' ' << scene.sampler.randomize << ' '
            << scene.sampler.log2_resolution << '\n';
    } else if (scene.sampler.tag == SamplerTag::Sobol) {
        out << "sampler sobol " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << ' ' << scene.sampler.randomize << ' '
            << scene.sampler.log2_resolution << '\n';
    } else if (scene.sampler.tag == SamplerTag::PaddedSobol) {
        out << "sampler paddedsobol " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << ' ' << scene.sampler.randomize << '\n';
    } else if (scene.sampler.tag == SamplerTag::PMJ02BN) {
        out << "sampler pmj02bn " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << '\n';
    } else {
        out << "sampler independent " << scene.sampler.samples_per_pixel << ' '
            << scene.sampler.seed << '\n';
    }
    out << "seed " << scene.seed << '\n';
    out << "integrator "
        << (scene.integrator == IntegratorTag::VolPath      ? "volpath"
            : scene.integrator == IntegratorTag::Path       ? "path"
            : scene.integrator == IntegratorTag::SimplePath ? "simplepath"
                                                            : "randomwalk")
        << " maxdepth " << scene.max_depth << " regularize "
        << scene.regularize << " cameramedium " << scene.camera_medium << '\n';
    out << "filter gaussian " << scene.filter_radius[0] << ' '
        << scene.filter_radius[1] << ' ' << scene.filter_sigma << " jitter "
        << (scene.disable_pixel_jitter ? 0 : 1) << " gbuffer "
        << (scene.film_visible_surface ? 1 : 0) << '\n';
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

    // The pixel sensor: its output matrix and its three response curves. See
    // the fields on Scene. Always written, the default sensor's curves being
    // X/Y/Z, so the reader and the renderer have one shape to handle.
    out << "sensor";
    detail::put(out, scene.output_rgb_from_sensor, 9);
    out << "\nsensor_r " << scene.sensor_r.size();
    detail::put(out, scene.sensor_r.data(), scene.sensor_r.size());
    out << "\nsensor_g " << scene.sensor_g.size();
    detail::put(out, scene.sensor_g.data(), scene.sensor_g.size());
    out << "\nsensor_b " << scene.sensor_b.size();
    detail::put(out, scene.sensor_b.data(), scene.sensor_b.size());
    out << '\n';

    // Before the materials, because a material names one by index.
    // Every field of every kind, so the reader has one shape to read.
    out << "textures " << scene.textures.size() << '\n';
    for (const Texture &t : scene.textures) {
        out << "  kind " << t.kind << " frame";
        detail::put(out, t.texture_from_render, 16);
        out << " mapping " << t.mapping << " uv";
        detail::put(out, &t.su, 1);
        detail::put(out, &t.sv, 1);
        detail::put(out, &t.du, 1);
        detail::put(out, &t.dv, 1);
        out << " planar";
        detail::put(out, t.vs, 3);
        detail::put(out, t.vt, 3);
        detail::put(out, &t.ds, 1);
        detail::put(out, &t.dt, 1);
        out << " scale";
        detail::put(out, &t.scale, 1);
        out << " average " << t.average_channels << " invert " << t.invert
            << " wrap " << t.wrap << " levels " << t.first_level << ' '
            << t.n_levels << " value";
        detail::put(out, &t.value, 1);
        detail::put(out, t.rgb, 3);
        out << " operands " << t.tex1 << ' ' << t.tex2 << ' ' << t.amount
            << " dir";
        detail::put(out, t.dir, 3);
        out << " noise " << t.octaves;
        detail::put(out, &t.omega, 1);
        detail::put(out, &t.noise_scale, 1);
        detail::put(out, &t.variation, 1);
        out << '\n';
    }
    out << "texturelevels " << scene.texture_levels.size() << '\n';
    for (const TextureLevel &l : scene.texture_levels) {
        out << "  " << l.width << ' ' << l.height << ' ' << l.first_texel
            << ' ' << l.format << '\n';
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
    out << "bssrdftables " << scene.bssrdf_tables.size() << '\n';
    for (size_t i = 0; i < scene.bssrdf_tables.size(); i++) {
        detail::put(out, &scene.bssrdf_tables[i], 1);
        if ((i + 1) % 8 == 0) {
            out << '\n';
        }
    }
    out << '\n';
    out << "rgbtable " << scene.rgb_table.size() << '\n';
    out << "texturetexels " << scene.texture_texels.size() << '\n';
    out << "texturebytes " << scene.texture_bytes.size() << '\n';

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
        case MaterialTag::DiffuseTransmission:
            out << "  diffusetransmission";
            break;
        case MaterialTag::CoatedConductor:
            out << "  coatedconductor";
            break;
        case MaterialTag::ThinDielectric:
            out << "  thindielectric";
            detail::put(out, &m.eta, 1);
            out << " etaspectrum " << m.eta_spectrum;
            break;
        case MaterialTag::Subsurface:
            out << "  subsurface " << m.sigma_a_spectrum << ' '
                << m.sigma_s_spectrum << ' ' << m.mfp_spectrum
                << " fromreflectance " << m.subsurface_from_reflectance
                << " table " << m.bssrdf_table << " scale";
            detail::put(out, &m.scale, 1);
            out << " eta";
            detail::put(out, &m.eta, 1);
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " roughnesstex " << m.u_roughness_texture << ' '
                << m.v_roughness_texture << " remap " << m.remap;
            break;
        case MaterialTag::Interface:
            out << "  interface";
            break;
        case MaterialTag::Mix:
            // The two operands by this file's material index -- both below
            // this record, since the converter writes a mix's operands
            // before the mix -- and the amount, a constant or a texture.
            out << "  mix " << m.mix_first << ' ' << m.mix_second << " amount";
            detail::put(out, &m.mix_amount, 1);
            out << " amounttex " << m.mix_amount_texture;
            break;
        default:
            return false;
        }
        out << " reflectance";
        detail::put(out, m.reflectance, 3);
        out << " reflectancetex " << m.reflectance_texture << " reflectancespec "
            << m.reflectance_spectrum << " displacement "
            << m.displacement_texture << " normalmap " << m.normal_map
            << " measured " << m.measured;
        if (m.tag == MaterialTag::DiffuseTransmission) {
            out << " transmittance";
            detail::put(out, m.transmittance, 3);
            out << " transmittancetex " << m.transmittance_texture
                << " transmittancespec " << m.transmittance_spectrum << " scale";
            detail::put(out, &m.scale, 1);
        }
        if (m.tag == MaterialTag::Dielectric) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " roughnesstex " << m.u_roughness_texture << ' '
                << m.v_roughness_texture;
            out << " remap " << m.remap;
            out << " eta";
            detail::put(out, &m.eta, 1);
            out << " etaspectrum " << m.eta_spectrum;
        }
        if (m.tag == MaterialTag::Conductor) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " roughnesstex " << m.u_roughness_texture << ' '
                << m.v_roughness_texture;
            out << " remap " << m.remap;
            out << " spectra " << m.conductor_spectra;
            out << " fromreflectance " << m.conductor_from_reflectance;
        }
        if (m.tag == MaterialTag::CoatedDiffuse ||
            m.tag == MaterialTag::CoatedConductor) {
            out << " roughness";
            detail::put(out, &m.u_roughness, 1);
            detail::put(out, &m.v_roughness, 1);
            out << " roughnesstex " << m.u_roughness_texture << ' '
                << m.v_roughness_texture;
            out << " remap " << m.remap;
            out << " thickness";
            detail::put(out, &m.thickness, 1);
            out << " eta";
            detail::put(out, &m.eta, 1);
            out << " etaspectrum " << m.eta_spectrum;
            out << " albedo";
            detail::put(out, m.medium_albedo, 3);
            out << " hasalbedo " << m.has_medium;
            out << " g";
            detail::put(out, &m.g, 1);
            out << " thicknesstex " << m.thickness_texture << " gtex "
                << m.g_texture;
            out << " maxdepth " << m.max_depth;
            out << " nsamples " << m.n_samples;
        }
        if (m.tag == MaterialTag::CoatedConductor) {
            out << " conductorroughness";
            detail::put(out, &m.conductor_u_roughness, 1);
            detail::put(out, &m.conductor_v_roughness, 1);
            out << " conductorroughnesstex " << m.conductor_u_roughness_texture
                << ' ' << m.conductor_v_roughness_texture;
            out << " spectra " << m.conductor_spectra;
            out << " fromreflectance " << m.conductor_from_reflectance;
        }
        out << '\n';
    }

    // The media, and then their spectra as one table -- three runs of
    // kMediumSpectrumSamples per medium (see Medium).
    out << "media " << scene.media.size() << '\n';
    for (const Medium &m : scene.media) {
        const auto grid = [&](const char *key, const GridRef &g) {
            out << ' ' << key << ' ' << g.nx << ' ' << g.ny << ' ' << g.nz << ' '
                << g.at;
        };
        // The box and the two matrices every placed medium has.
        const auto frame = [&]() {
            out << " box";
            detail::put(out, m.low, 3);
            detail::put(out, m.high, 3);
            out << " renderfrommedium";
            detail::put(out, m.render_from_medium, 16);
            out << " mediumfromrender";
            detail::put(out, m.medium_from_render, 16);
        };
        switch (m.tag) {
        case MediumTag::Homogeneous:
            out << "  homogeneous spectra " << m.spectra << " g";
            detail::put(out, &m.g, 1);
            out << " emissive " << m.emissive << '\n';
            break;
        case MediumTag::UniformGrid:
            out << "  uniformgrid spectra " << m.spectra << " g";
            detail::put(out, &m.g, 1);
            out << " emissive " << m.emissive;
            frame();
            grid("density", m.density);
            grid("temperature", m.temperature);
            grid("lescale", m.le_scale);
            out << " temperaturescale";
            detail::put(out, &m.temperature_scale, 1);
            out << " temperatureoffset";
            detail::put(out, &m.temperature_offset, 1);
            grid("majorant", m.majorant);
            out << '\n';
            break;
        case MediumTag::RGBGrid:
            out << "  rgbgrid g";
            detail::put(out, &m.g, 1);
            frame();
            grid("sigmaa", m.sigma_a);
            grid("sigmas", m.sigma_s);
            grid("le", m.le);
            out << " sigmascale";
            detail::put(out, &m.sigma_scale, 1);
            out << " lescale";
            detail::put(out, &m.le_scale_value, 1);
            grid("majorant", m.majorant);
            out << '\n';
            break;
        case MediumTag::Cloud:
            out << "  cloud spectra " << m.spectra << " g";
            detail::put(out, &m.g, 1);
            frame();
            out << " density";
            detail::put(out, &m.cloud_density, 1);
            out << " wispiness";
            detail::put(out, &m.wispiness, 1);
            out << " frequency";
            detail::put(out, &m.frequency, 1);
            out << '\n';
            break;
        case MediumTag::NanoVDB:
            out << "  nanovdb spectra " << m.spectra << " g";
            detail::put(out, &m.g, 1);
            out << " emissive " << m.emissive;
            frame();
            out << " densityat " << m.density_at << " temperatureat "
                << m.temperature_at << " hastemperature " << m.has_temperature;
            out << " densitymap";
            detail::put(out, m.density_map, 12);
            out << " temperaturemap";
            detail::put(out, m.temperature_map, 12);
            out << " lescale";
            detail::put(out, &m.le_scale_value, 1);
            out << " temperaturescale";
            detail::put(out, &m.temperature_scale, 1);
            out << " temperatureoffset";
            detail::put(out, &m.temperature_offset, 1);
            grid("majorant", m.majorant);
            out << '\n';
            break;
        default:
            return false;
        }
    }
    out << "envilluminants " << scene.env_illuminants.size() << '\n';
    for (size_t i = 0; i < scene.env_illuminants.size(); i++) {
        detail::put(out, &scene.env_illuminants[i], 1);
        out << (i % 8 == 7 || i + 1 == scene.env_illuminants.size() ? '\n' : ' ');
    }
    out << "mediumspectra " << scene.medium_spectra.size() << '\n';
    for (size_t i = 0; i < scene.medium_spectra.size(); i++) {
        detail::put(out, &scene.medium_spectra[i], 1);
        if (i % 8 == 7 || i + 1 == scene.medium_spectra.size()) {
            out << '\n';
        }
    }
    // The grids themselves are in the `.vol` sidecar; this is how many floats.
    out << "mediumgrid " << scene.medium_grid.size() << '\n';
    // The NanoVDB grids are in the `.vdb` sidecar; this is how many bytes.
    out << "vdbbytes " << scene.vdb_bytes.size() << '\n';
    // The sampler's tables are in the `.smp` sidecar; this is how many words
    // of each, in the order they are written there.
    out << "samplertables " << scene.sobol_matrices.size() << ' '
        << scene.vdc_matrices.size() << ' ' << scene.vdc_matrices_inv.size()
        << ' ' << scene.pmj02bn_samples.size() << ' ' << scene.blue_noise.size()
        << '\n';

    // The geometry -- meshes, vertices, shapes, trees, instances, primitives
    // -- is the FlatBuffer beside this file (write_geometry); the text keeps
    // its size, which the reader checks the sidecar against.
    uint64_t geometry_bytes = 0;
    if (!detail::write_geometry(path, scene, &geometry_bytes)) {
        return false;
    }
    out << "geometry " << geometry_bytes << '\n';

    out << "lights " << scene.lights.size() << '\n';
    for (const Light &l : scene.lights) {
        out << "  diffuse";
        detail::put(out, l.l, 3);
        detail::put(out, &l.scale, 1);
        out << " twosided " << l.two_sided;
        out << " blackbody " << l.blackbody;
        detail::put(out, &l.temperature, 1);
        detail::put(out, &l.blackbody_normalization, 1);
        out << '\n';
    }

    out << "point_lights " << scene.point_lights.size() << '\n';
    for (const PointLight &l : scene.point_lights) {
        out << (l.spot != 0 ? "  spot" : "  point");
        detail::put(out, l.l, 3);
        detail::put(out, &l.scale, 1);
        out << " hasl " << l.has_l;
        out << " blackbody " << l.blackbody;
        detail::put(out, &l.temperature, 1);
        detail::put(out, &l.blackbody_normalization, 1);
        out << " position";
        detail::put(out, l.position, 3);
        if (l.spot != 0) {
            out << " cone";
            detail::put(out, &l.cos_falloff_start, 1);
            detail::put(out, &l.cos_falloff_end, 1);
            out << " light_from_render";
            detail::put(out, l.light_from_render, 16);
        }
        out << '\n';
    }

    out << "infinite_lights " << scene.infinite_lights.size() << " radius "
        << scene.scene_radius << '\n';
    for (const InfiniteLight &l : scene.infinite_lights) {
        out << (l.distant != 0     ? "  distant"
                : l.resolution == 0 ? "  uniform"
                : l.portal != 0     ? "  portal"
                                    : "  image");
        detail::put(out, l.l, 3);
        detail::put(out, &l.scale, 1);
        out << " hasl " << l.has_l;
        out << " blackbody " << l.blackbody;
        detail::put(out, &l.temperature, 1);
        detail::put(out, &l.blackbody_normalization, 1);
        if (l.distant != 0) {
            out << " direction";
            detail::put(out, l.direction, 3);
        }
        if (l.resolution != 0) {
            out << " resolution " << l.resolution << " first " << l.first_texel
                << " illuminant " << l.illuminant;
            if (l.portal != 0) {
                out << " points";
                detail::put(out, l.portal_points, 12);
            } else {
                out << " light_from_render";
                detail::put(out, l.light_from_render, 16);
                out << " render_from_light";
                detail::put(out, l.render_from_light, 16);
            }
        }
        out << '\n';
    }

    out << "light_tree " << scene.light_sampler << ' '
        << scene.light_tree.size() << '\n';
    for (const LightTreeNode &n : scene.light_tree) {
        out << (n.is_leaf ? "  leaf" : "  interior");
        detail::put(out, n.w, 3);
        detail::put(out, &n.phi, 1);
        detail::put(out, &n.cos_theta_o, 1);
        detail::put(out, &n.cos_theta_e, 1);
        detail::put(out, n.bounds_min, 3);
        detail::put(out, n.bounds_max, 3);
        out << " twosided " << n.two_sided
            << (n.is_leaf ? " light " : " right ") << n.child_or_light << '\n';
    }
    out << "light_bit_trails " << scene.light_bit_trails.size();
    for (uint32_t t : scene.light_bit_trails) {
        out << ' ' << t;
    }
    out << '\n';

    // The shapes, the trees, the instance definitions and placements and the
    // primitives are in the geometry sidecar, written above.
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
    if (!(in >> word) || word != "pixelbounds") {
        return false;
    }
    in >> scene.pixel_x0 >> scene.pixel_y0 >> scene.pixel_x1 >> scene.pixel_y1;
    if (scene.pixel_x0 >= scene.pixel_x1 || scene.pixel_y0 >= scene.pixel_y1 ||
        scene.pixel_x1 > scene.width || scene.pixel_y1 > scene.height) {
        return false;
    }

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
    } else if (word == "zsobol") {
        scene.sampler.tag = SamplerTag::ZSobol;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed >>
            scene.sampler.randomize >> scene.sampler.log2_resolution;
    } else if (word == "sobol") {
        scene.sampler.tag = SamplerTag::Sobol;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed >>
            scene.sampler.randomize >> scene.sampler.log2_resolution;
    } else if (word == "paddedsobol") {
        scene.sampler.tag = SamplerTag::PaddedSobol;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed >>
            scene.sampler.randomize;
    } else if (word == "pmj02bn") {
        scene.sampler.tag = SamplerTag::PMJ02BN;
        in >> scene.sampler.samples_per_pixel >> scene.sampler.seed;
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
    } else if (word == "volpath") {
        scene.integrator = IntegratorTag::VolPath;
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
    if (!(in >> word) || word != "cameramedium") {
        return false;
    }
    in >> scene.camera_medium;

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
    if (!(in >> word) || word != "gbuffer") {
        return false;
    }
    uint32_t gbuffer = 0;
    in >> gbuffer;
    scene.film_visible_surface = gbuffer ? 1u : 0u;

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

    // The pixel sensor: its output matrix, then its three response curves each
    // as a count and that many values.
    if (!(in >> word) || word != "sensor") {
        return false;
    }
    floats(scene.output_rgb_from_sensor, 9);
    {
        const auto curve = [&](const char *name, std::vector<float> &into) {
            size_t n = 0;
            if (!(in >> word) || word != name) {
                return false;
            }
            in >> n;
            into.resize(n);
            floats(into.data(), int(n));
            return true;
        };
        if (!curve("sensor_r", scene.sensor_r) ||
            !curve("sensor_g", scene.sensor_g) ||
            !curve("sensor_b", scene.sensor_b)) {
            return false;
        }
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
        Texture t;
        if (!tagged("kind")) {
            return false;
        }
        in >> t.kind;
        if (!tagged("frame")) {
            return false;
        }
        floats(t.texture_from_render, 16);
        if (!tagged("mapping")) {
            return false;
        }
        in >> t.mapping;
        if (!tagged("uv")) {
            return false;
        }
        floats(&t.su, 1);
        floats(&t.sv, 1);
        floats(&t.du, 1);
        floats(&t.dv, 1);
        if (!tagged("planar")) {
            return false;
        }
        floats(t.vs, 3);
        floats(t.vt, 3);
        floats(&t.ds, 1);
        floats(&t.dt, 1);
        if (!tagged("scale")) {
            return false;
        }
        floats(&t.scale, 1);
        if (!tagged("average")) {
            return false;
        }
        in >> t.average_channels;
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
        if (!tagged("value")) {
            return false;
        }
        floats(&t.value, 1);
        floats(t.rgb, 3);
        if (!tagged("operands")) {
            return false;
        }
        in >> t.tex1 >> t.tex2 >> t.amount;
        if (!tagged("dir")) {
            return false;
        }
        floats(t.dir, 3);
        if (!tagged("noise")) {
            return false;
        }
        in >> t.octaves;
        floats(&t.omega, 1);
        floats(&t.noise_scale, 1);
        floats(&t.variation, 1);
        scene.textures.push_back(t);
    }

    if (!(in >> word) || word != "texturelevels") {
        return false;
    }
    in >> count;
    scene.texture_levels.clear();
    for (size_t i = 0; i < count; i++) {
        TextureLevel l;
        in >> l.width >> l.height >> l.first_texel >> l.format;
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

    if (!(in >> word) || word != "bssrdftables") {
        return false;
    }
    in >> count;
    if (count % size_t(kBSSRDFTableFloats) != 0) {
        return false;
    }
    scene.bssrdf_tables.assign(count, 0.f);
    for (size_t i = 0; i < count; i++) {
        floats(&scene.bssrdf_tables[i], 1);
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
        if (!(in >> word) || word != "texturebytes") {
            return false;
        }
        size_t bytes = 0;
        in >> bytes;
        scene.texture_bytes.clear();
        if (bytes > 0) {
            std::ifstream tex(texel_path(path), std::ios::binary);
            if (!tex) {
                return false;
            }
            tex.seekg(std::streamoff(sizeof(float) * (table_size + texels)));
            scene.texture_bytes.resize(bytes);
            tex.read(reinterpret_cast<char *>(scene.texture_bytes.data()),
                     std::streamsize(bytes));
            if (tex.gcount() != std::streamsize(bytes)) {
                return false;
            }
        }
    }

    if (!(in >> word) || word != "materials") {
        return false;
    }
    in >> count;
    scene.materials.clear();
    // How many sampled spectra the tables above hold, for the indices below.
    const int32_t spectra =
        int32_t(scene.conductor_eta.size() / size_t(kConductorSamples));
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
        } else if (word == "diffusetransmission") {
            m.tag = MaterialTag::DiffuseTransmission;
        } else if (word == "coatedconductor") {
            m.tag = MaterialTag::CoatedConductor;
        } else if (word == "thindielectric") {
            m.tag = MaterialTag::ThinDielectric;
            floats(&m.eta, 1);
            if (!tagged("etaspectrum")) {
                return false;
            }
            in >> m.eta_spectrum;
        } else if (word == "subsurface") {
            m.tag = MaterialTag::Subsurface;
            in >> m.sigma_a_spectrum >> m.sigma_s_spectrum >> m.mfp_spectrum;
            if (!tagged("fromreflectance")) {
                return false;
            }
            in >> m.subsurface_from_reflectance;
            if (!tagged("table")) {
                return false;
            }
            in >> m.bssrdf_table;
            if (!tagged("scale")) {
                return false;
            }
            floats(&m.scale, 1);
            if (!tagged("eta")) {
                return false;
            }
            floats(&m.eta, 1);
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("roughnesstex")) {
                return false;
            }
            in >> m.u_roughness_texture >> m.v_roughness_texture;
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (m.sigma_a_spectrum >= spectra || m.sigma_s_spectrum >= spectra ||
                m.mfp_spectrum >= spectra) {
                return false;
            }
            if (m.bssrdf_table < 0 ||
                m.bssrdf_table >=
                    int32_t(scene.bssrdf_tables.size() / size_t(kBSSRDFTableFloats))) {
                return false;
            }
        } else if (word == "interface") {
            m.tag = MaterialTag::Interface;
        } else if (word == "mix") {
            m.tag = MaterialTag::Mix;
            in >> m.mix_first >> m.mix_second;
            // Operands precede the mix in the file, so a forward reference
            // is a corrupt file rather than a late binding.
            if (m.mix_first >= int32_t(i) || m.mix_second >= int32_t(i)) {
                return false;
            }
            if (!tagged("amount")) {
                return false;
            }
            floats(&m.mix_amount, 1);
            if (!tagged("amounttex")) {
                return false;
            }
            in >> m.mix_amount_texture;
            if (m.mix_amount_texture >= int32_t(scene.textures.size())) {
                return false;
            }
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
        if (!tagged("reflectancespec")) {
            return false;
        }
        in >> m.reflectance_spectrum;
        if (m.reflectance_spectrum >= spectra) {
            return false;
        }
        if (!tagged("displacement")) {
            return false;
        }
        in >> m.displacement_texture;
        if (!tagged("normalmap")) {
            return false;
        }
        in >> m.normal_map;
        if (!tagged("measured")) {
            return false;
        }
        in >> m.measured;
        if (m.tag == MaterialTag::DiffuseTransmission) {
            if (!tagged("transmittance")) {
                return false;
            }
            floats(m.transmittance, 3);
            if (!tagged("transmittancetex")) {
                return false;
            }
            in >> m.transmittance_texture;
            if (m.transmittance_texture >= int32_t(scene.textures.size())) {
                return false;
            }
            if (!tagged("transmittancespec")) {
                return false;
            }
            in >> m.transmittance_spectrum;
            if (m.transmittance_spectrum >= spectra) {
                return false;
            }
            if (!tagged("scale")) {
                return false;
            }
            floats(&m.scale, 1);
        }
        if (m.tag == MaterialTag::Dielectric) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("roughnesstex")) {
                return false;
            }
            in >> m.u_roughness_texture >> m.v_roughness_texture;
            if (m.u_roughness_texture >= int32_t(scene.textures.size()) ||
                m.v_roughness_texture >= int32_t(scene.textures.size())) {
                return false;
            }
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (!tagged("eta")) {
                return false;
            }
            floats(&m.eta, 1);
            if (!tagged("etaspectrum")) {
                return false;
            }
            in >> m.eta_spectrum;
        }
        if (m.tag == MaterialTag::Conductor) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("roughnesstex")) {
                return false;
            }
            in >> m.u_roughness_texture >> m.v_roughness_texture;
            if (m.u_roughness_texture >= int32_t(scene.textures.size()) ||
                m.v_roughness_texture >= int32_t(scene.textures.size())) {
                return false;
            }
            if (!tagged("remap")) {
                return false;
            }
            in >> m.remap;
            if (!tagged("spectra")) {
                return false;
            }
            in >> m.conductor_spectra;
            if (!tagged("fromreflectance")) {
                return false;
            }
            in >> m.conductor_from_reflectance;
        }
        if (m.tag == MaterialTag::CoatedDiffuse ||
            m.tag == MaterialTag::CoatedConductor) {
            if (!tagged("roughness")) {
                return false;
            }
            floats(&m.u_roughness, 1);
            floats(&m.v_roughness, 1);
            if (!tagged("roughnesstex")) {
                return false;
            }
            in >> m.u_roughness_texture >> m.v_roughness_texture;
            if (m.u_roughness_texture >= int32_t(scene.textures.size()) ||
                m.v_roughness_texture >= int32_t(scene.textures.size())) {
                return false;
            }
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
            if (!tagged("etaspectrum")) {
                return false;
            }
            in >> m.eta_spectrum;
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
            if (!tagged("thicknesstex")) {
                return false;
            }
            in >> m.thickness_texture;
            if (!tagged("gtex")) {
                return false;
            }
            in >> m.g_texture;
            if (m.thickness_texture >= int32_t(scene.textures.size()) ||
                m.g_texture >= int32_t(scene.textures.size())) {
                return false;
            }
            if (!tagged("maxdepth")) {
                return false;
            }
            in >> m.max_depth;
            if (!tagged("nsamples")) {
                return false;
            }
            in >> m.n_samples;
        }
        if (m.tag == MaterialTag::CoatedConductor) {
            if (!tagged("conductorroughness")) {
                return false;
            }
            floats(&m.conductor_u_roughness, 1);
            floats(&m.conductor_v_roughness, 1);
            if (!tagged("conductorroughnesstex")) {
                return false;
            }
            in >> m.conductor_u_roughness_texture >>
                m.conductor_v_roughness_texture;
            if (m.conductor_u_roughness_texture >=
                    int32_t(scene.textures.size()) ||
                m.conductor_v_roughness_texture >=
                    int32_t(scene.textures.size())) {
                return false;
            }
            if (!tagged("spectra")) {
                return false;
            }
            in >> m.conductor_spectra;
            if (!tagged("fromreflectance")) {
                return false;
            }
            in >> m.conductor_from_reflectance;
        }
        scene.materials.push_back(m);
    }

    if (!(in >> word) || word != "media") {
        return false;
    }
    in >> count;
    scene.media.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Medium m;
        const auto grid = [&](const char *key, GridRef &g) {
            if (!tagged(key)) {
                return false;
            }
            in >> g.nx >> g.ny >> g.nz >> g.at;
            return bool(in);
        };
        const auto frame = [&]() {
            if (!tagged("box")) {
                return false;
            }
            floats(m.low, 3);
            floats(m.high, 3);
            if (!tagged("renderfrommedium")) {
                return false;
            }
            floats(m.render_from_medium, 16);
            if (!tagged("mediumfromrender")) {
                return false;
            }
            floats(m.medium_from_render, 16);
            return true;
        };
        const auto one = [&](const char *key, float *value) {
            if (!tagged(key)) {
                return false;
            }
            floats(value, 1);
            return true;
        };
        if (word == "homogeneous") {
            m.tag = MediumTag::Homogeneous;
        } else if (word == "uniformgrid") {
            m.tag = MediumTag::UniformGrid;
        } else if (word == "rgbgrid") {
            m.tag = MediumTag::RGBGrid;
        } else if (word == "cloud") {
            m.tag = MediumTag::Cloud;
        } else if (word == "nanovdb") {
            m.tag = MediumTag::NanoVDB;
        } else {
            return false;
        }
        if (m.tag != MediumTag::RGBGrid) {
            if (!tagged("spectra")) {
                return false;
            }
            in >> m.spectra;
        }
        if (!one("g", &m.g)) {
            return false;
        }
        if (m.tag == MediumTag::Homogeneous || m.tag == MediumTag::UniformGrid ||
            m.tag == MediumTag::NanoVDB) {
            if (!tagged("emissive")) {
                return false;
            }
            in >> m.emissive;
        }
        if (m.tag != MediumTag::Homogeneous && !frame()) {
            return false;
        }
        if (m.tag == MediumTag::UniformGrid) {
            if (!grid("density", m.density) || !grid("temperature", m.temperature) ||
                !grid("lescale", m.le_scale) ||
                !one("temperaturescale", &m.temperature_scale) ||
                !one("temperatureoffset", &m.temperature_offset) ||
                !grid("majorant", m.majorant)) {
                return false;
            }
        } else if (m.tag == MediumTag::RGBGrid) {
            if (!grid("sigmaa", m.sigma_a) || !grid("sigmas", m.sigma_s) ||
                !grid("le", m.le) || !one("sigmascale", &m.sigma_scale) ||
                !one("lescale", &m.le_scale_value) ||
                !grid("majorant", m.majorant)) {
                return false;
            }
        } else if (m.tag == MediumTag::Cloud) {
            if (!one("density", &m.cloud_density) ||
                !one("wispiness", &m.wispiness) || !one("frequency", &m.frequency)) {
                return false;
            }
        } else if (m.tag == MediumTag::NanoVDB) {
            if (!tagged("densityat")) {
                return false;
            }
            in >> m.density_at;
            if (!tagged("temperatureat")) {
                return false;
            }
            in >> m.temperature_at;
            if (!tagged("hastemperature")) {
                return false;
            }
            in >> m.has_temperature;
            if (!tagged("densitymap")) {
                return false;
            }
            floats(m.density_map, 12);
            if (!tagged("temperaturemap")) {
                return false;
            }
            floats(m.temperature_map, 12);
            if (!one("lescale", &m.le_scale_value) ||
                !one("temperaturescale", &m.temperature_scale) ||
                !one("temperatureoffset", &m.temperature_offset) ||
                !grid("majorant", m.majorant)) {
                return false;
            }
        }
        scene.media.push_back(m);
    }
    if (!(in >> word) || word != "envilluminants") {
        return false;
    }
    in >> count;
    scene.env_illuminants.assign(count, 0.f);
    for (size_t i = 0; i < count; i++) {
        floats(&scene.env_illuminants[i], 1);
    }
    if (count % 471 != 0) {
        return false; // 471 values per illuminant, 360 to 830 nm
    }
    if (!(in >> word) || word != "mediumspectra") {
        return false;
    }
    in >> count;
    scene.medium_spectra.assign(count, 0.f);
    for (size_t i = 0; i < count; i++) {
        floats(&scene.medium_spectra[i], 1);
    }
    for (const Medium &m : scene.media) {
        // An rgbgrid has no spectra of its own: its coefficients are grids.
        if (m.tag != MediumTag::RGBGrid &&
            size_t(m.spectra) + 3 * kMediumSpectrumSamples > count) {
            return false;
        }
    }
    if (!(in >> word) || word != "mediumgrid") {
        return false;
    }
    in >> count;
    scene.medium_grid.clear();
    if (count > 0) {
        std::ifstream vol(vol_path(path), std::ios::binary);
        if (!vol) {
            return false;
        }
        scene.medium_grid.resize(count);
        vol.read(reinterpret_cast<char *>(scene.medium_grid.data()),
                 std::streamsize(sizeof(float) * count));
        if (vol.gcount() != std::streamsize(sizeof(float) * count)) {
            return false;
        }
    }
    if (!(in >> word) || word != "vdbbytes") {
        return false;
    }
    in >> count;
    scene.vdb_bytes.clear();
    if (count > 0) {
        std::ifstream vdb(vdb_path(path), std::ios::binary);
        if (!vdb) {
            return false;
        }
        scene.vdb_bytes.resize(count);
        vdb.read(reinterpret_cast<char *>(scene.vdb_bytes.data()),
                 std::streamsize(count));
        if (vdb.gcount() != std::streamsize(count)) {
            return false;
        }
    }
    if (!(in >> word) || word != "samplertables") {
        return false;
    }
    {
        size_t counts[5];
        for (size_t &c : counts) {
            in >> c;
        }
        scene.sobol_matrices.assign(counts[0], 0);
        scene.vdc_matrices.assign(counts[1], 0);
        scene.vdc_matrices_inv.assign(counts[2], 0);
        scene.pmj02bn_samples.assign(counts[3], 0);
        scene.blue_noise.assign(counts[4], 0);
        if (counts[0] + counts[1] + counts[2] + counts[3] + counts[4] > 0) {
            std::ifstream smp(smp_path(path), std::ios::binary);
            if (!smp) {
                return false;
            }
            const auto get = [&](auto &v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                const std::streamsize bytes =
                    std::streamsize(sizeof(T) * v.size());
                smp.read(reinterpret_cast<char *>(v.data()), bytes);
                return smp.gcount() == bytes;
            };
            if (!get(scene.sobol_matrices) || !get(scene.vdc_matrices) ||
                !get(scene.vdc_matrices_inv) || !get(scene.pmj02bn_samples) ||
                !get(scene.blue_noise)) {
                return false;
            }
        }
        // A sobol scene carries the whole of each of its three tables or
        // none; a pmj02bn scene the whole of its two.
        const bool sobol = scene.sampler.tag == SamplerTag::Sobol;
        const bool pmj = scene.sampler.tag == SamplerTag::PMJ02BN;
        if (scene.sobol_matrices.size() != (sobol ? 1024u * 52u : 0u) ||
            scene.vdc_matrices.size() != (sobol ? 25u * 52u : 0u) ||
            scene.vdc_matrices_inv.size() != (sobol ? 25u * 52u : 0u) ||
            scene.pmj02bn_samples.size() != (pmj ? 5u * 65536u * 2u : 0u) ||
            scene.blue_noise.size() != (pmj ? 48u * 128u * 128u : 0u)) {
            return false;
        }
    }
    // Every grid within the pool: a present grid has positive extents and its
    // voxels inside, an absent one is all zeros.
    const auto grid_ok = [&](const GridRef &g, size_t channels) {
        if (g.nx == 0) {
            return g.ny == 0 && g.nz == 0 && g.at == 0;
        }
        if (g.nx < 0 || g.ny <= 0 || g.nz <= 0) {
            return false;
        }
        return size_t(g.at) + size_t(g.nx) * size_t(g.ny) * size_t(g.nz) * channels <=
               scene.medium_grid.size();
    };
    for (const Medium &m : scene.media) {
        if (m.tag == MediumTag::UniformGrid) {
            if (m.density.nx == 0 || m.le_scale.nx == 0 || m.majorant.nx == 0 ||
                !grid_ok(m.density, 1) || !grid_ok(m.temperature, 1) ||
                !grid_ok(m.le_scale, 1) || !grid_ok(m.majorant, 1)) {
                return false;
            }
        } else if (m.tag == MediumTag::RGBGrid) {
            if (m.majorant.nx == 0 || !grid_ok(m.sigma_a, 4) ||
                !grid_ok(m.sigma_s, 4) || !grid_ok(m.le, 4) ||
                !grid_ok(m.majorant, 1)) {
                return false;
            }
        } else if (m.tag == MediumTag::NanoVDB) {
            // A grid begins within the bytes, on NanoVDB's 32-byte alignment,
            // with room for at least its header (672 bytes).
            const auto grid_at = [&](uint32_t at) {
                return at % 32 == 0 && size_t(at) + 672 <= scene.vdb_bytes.size();
            };
            if (m.majorant.nx == 0 || !grid_ok(m.majorant, 1) ||
                !grid_at(m.density_at) ||
                (m.has_temperature != 0 && !grid_at(m.temperature_at))) {
                return false;
            }
        }
    }

    // The geometry, from the FlatBuffer beside this file (read_geometry):
    // meshes, vertices, shapes, trees, instances and primitives at once. Its
    // indices into the lights are checked after the lights are read
    // (validate_geometry, below).
    if (!(in >> word) || word != "geometry") {
        return false;
    }
    uint64_t geometry_bytes = 0;
    in >> geometry_bytes;
    if (!detail::read_geometry(path, geometry_bytes, scene)) {
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
        if (!tagged("blackbody")) {
            return false;
        }
        in >> l.blackbody;
        floats(&l.temperature, 1);
        floats(&l.blackbody_normalization, 1);
        scene.lights.push_back(l);
    }

    if (!(in >> word) || word != "point_lights") {
        return false;
    }
    in >> count;
    scene.point_lights.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word) || (word != "point" && word != "spot")) {
            return false;
        }
        PointLight l;
        l.spot = word == "spot" ? 1u : 0u;
        floats(l.l, 3);
        floats(&l.scale, 1);
        if (!tagged("hasl")) {
            return false;
        }
        in >> l.has_l;
        if (!tagged("blackbody")) {
            return false;
        }
        in >> l.blackbody;
        floats(&l.temperature, 1);
        floats(&l.blackbody_normalization, 1);
        if (!tagged("position")) {
            return false;
        }
        floats(l.position, 3);
        if (l.spot != 0) {
            if (!tagged("cone")) {
                return false;
            }
            floats(&l.cos_falloff_start, 1);
            floats(&l.cos_falloff_end, 1);
            if (!tagged("light_from_render")) {
                return false;
            }
            floats(l.light_from_render, 16);
        }
        scene.point_lights.push_back(l);
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
        if (!(in >> word) || (word != "uniform" && word != "image" &&
                              word != "distant" && word != "portal")) {
            return false;
        }
        const bool is_image = word == "image" || word == "portal";
        InfiniteLight l;
        l.distant = word == "distant" ? 1u : 0u;
        l.portal = word == "portal" ? 1u : 0u;
        floats(l.l, 3);
        floats(&l.scale, 1);
        if (!tagged("hasl")) {
            return false;
        }
        in >> l.has_l;
        if (!tagged("blackbody")) {
            return false;
        }
        in >> l.blackbody;
        floats(&l.temperature, 1);
        floats(&l.blackbody_normalization, 1);
        if (l.distant != 0) {
            if (!tagged("direction")) {
                return false;
            }
            floats(l.direction, 3);
        }
        if (is_image) {
            if (!tagged("resolution")) {
                return false;
            }
            in >> l.resolution;
            if (!tagged("first")) {
                return false;
            }
            in >> l.first_texel;
            if (!tagged("illuminant")) {
                return false;
            }
            in >> l.illuminant;
            if (l.portal != 0) {
                if (!tagged("points")) {
                    return false;
                }
                floats(l.portal_points, 12);
            } else {
                if (!tagged("light_from_render")) {
                    return false;
                }
                floats(l.light_from_render, 16);
                if (!tagged("render_from_light")) {
                    return false;
                }
                floats(l.render_from_light, 16);
            }
        }
        scene.infinite_lights.push_back(l);
    }

    if (!(in >> word) || word != "light_tree") {
        return false;
    }
    in >> scene.light_sampler;
    size_t tree_count = 0;
    in >> tree_count;
    scene.light_tree.clear();
    for (size_t i = 0; i < tree_count; i++) {
        if (!(in >> word) || (word != "leaf" && word != "interior")) {
            return false;
        }
        LightTreeNode n;
        n.is_leaf = word == "leaf" ? 1u : 0u;
        floats(n.w, 3);
        floats(&n.phi, 1);
        floats(&n.cos_theta_o, 1);
        floats(&n.cos_theta_e, 1);
        floats(n.bounds_min, 3);
        floats(n.bounds_max, 3);
        if (!tagged("twosided")) {
            return false;
        }
        in >> n.two_sided;
        if (!tagged(n.is_leaf ? "light" : "right")) {
            return false;
        }
        in >> n.child_or_light;
        scene.light_tree.push_back(n);
    }
    if (!(in >> word) || word != "light_bit_trails") {
        return false;
    }
    size_t trail_count = 0;
    in >> trail_count;
    scene.light_bit_trails.clear();
    for (size_t i = 0; i < trail_count; i++) {
        uint32_t t = 0;
        in >> t;
        scene.light_bit_trails.push_back(t);
    }


    // The geometry was read with the meshes above (read_geometry); now that
    // the lights are in as well, every index it carries is checked.
    if (!detail::validate_geometry(scene)) {
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
