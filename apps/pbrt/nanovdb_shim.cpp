// The implementation of media.bonsai's two foreign functions over NanoVDB
// (docs/foreign-functions.md): what the renderer calls to read a voxel of a
// `nanovdb` medium, on the CPU and on the GPU, written once over NanoVDB's
// own header -- the library pbrt reads its grids with, and the one thing
// about such a grid the program does not describe itself.
//
//   nanovdb_grid(bytes, at)    the grid that begins `at` bytes into the
//                              buffer the converter copied out of the .nvdb
//                              file (Scene::vdb_bytes): NanoVDB's GridData
//                              is the first thing in a grid's buffer, so
//                              the handle is the address.
//   nanovdb_sample(g, x, y, z) the grid filtered at a medium-space point:
//                              pbrt's NanoVDBMedium::SamplePoint read,
//                              verbatim -- `worldToIndexF` and then
//                              `SampleFromVoxels<FloatGrid::TreeType, 1,
//                              false>(tree())(pIndex)`, eight getValue tree
//                              walks and the trilinear weights, all inside
//                              this one function. One function on purpose:
//                              the renderer once did the eight voxel reads
//                              and the lerps itself over a per-voxel
//                              primitive, the same arithmetic, and the
//                              inliner's copy of eight tree walks in the
//                              shadow march's step made that kernel twice
//                              pbrt's time (explosion, 2026-10-05; ncu put
//                              both at equal occupancy, so the step's size
//                              was the cost -- nvcc keeps pbrt's read
//                              outlined, and this call keeps ours so).
//
// Built three ways by build_nanovdb_shim.sh. Into the driver, where it
// includes the generated header (BONSAI_SHIM_CHECK) so that these
// definitions are checked against the prototypes the program declared and
// a mismatch is a compile error. To host bitcode with `clang++ -emit-llvm`,
// which `--link` folds into the generated module before it is optimized,
// so that a density lookup is eight inlined walks and not eight calls. And
// for a GPU schedule, to nvptx64 bitcode the same way (plain C++ against
// the nvptx64 target -- clang's CUDA mode refuses this machine's newer
// CUDA headers, and nothing here needs them), so the device and OptiX
// modules inline the lookup into the march loop exactly as nvcc inlines
// pbrt's; `nvcc -rdc=true` PTX remains as build_nanovdb_shim.sh's loud
// fallback, an outlined call per read.
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <cstdint>

#if defined(__CUDACC__)
#define BONSAI_SHIM extern "C" __device__
typedef const void *NanoVDBGrid;
#else
#define BONSAI_SHIM extern "C"
#if defined(BONSAI_SHIM_CHECK)
#include "render.h"
#else
typedef const void *NanoVDBGrid;
#endif
#endif

BONSAI_SHIM NanoVDBGrid nanovdb_grid(const uint8_t *bytes, uint32_t at) {
    return bytes + at;
}

BONSAI_SHIM float nanovdb_sample(NanoVDBGrid g, float x, float y, float z) {
    const auto *grid = static_cast<const nanovdb::FloatGrid *>(g);
    const nanovdb::Vec3<float> pIndex =
        grid->worldToIndexF(nanovdb::Vec3<float>(x, y, z));
    using Sampler =
        nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
    return Sampler(grid->tree())(pIndex);
}
