// The implementation of media.bonsai's two foreign functions over NanoVDB
// (docs/foreign-functions.md): what the renderer calls to read a voxel of a
// `nanovdb` medium, on the CPU and on the GPU, written once over NanoVDB's
// own header -- the library pbrt reads its grids with, and the one thing
// about such a grid the program does not describe itself.
//
//   nanovdb_grid(bytes, at)   the grid that begins `at` bytes into the
//                             buffer the converter copied out of the .nvdb
//                             file (Scene::vdb_bytes): NanoVDB's GridData is
//                             the first thing in a grid's buffer, so the
//                             handle is the address.
//   nanovdb_value(g, i, j, k) the voxel at (i, j, k) of a float grid, the
//                             background where none is stored -- NanoVDB's
//                             tree().getValue, the root-to-leaf walk pbrt's
//                             SampleFromVoxels<TreeType, 1, false> makes
//                             for each of its eight reads (it is given the
//                             tree, not an accessor, so nothing is cached
//                             between them; the same here).
//
// Built three ways by build_nanovdb_shim.sh. Into the driver, where it
// includes the generated header (BONSAI_SHIM_CHECK) so that these
// definitions are checked against the prototypes the program declared and
// a mismatch is a compile error. To host bitcode with `clang++ -emit-llvm`,
// which `--link` folds into the generated module before it is optimized,
// so that a density lookup is eight inlined walks and not eight calls. And
// to PTX with `nvcc -rdc=true` for a GPU schedule, spliced into the device
// module's PTX by the same flag; nvcc rather than clang because this
// machine's CUDA is newer than clang's device headers admit.
#include <nanovdb/NanoVDB.h>

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

BONSAI_SHIM float nanovdb_value(NanoVDBGrid g, int32_t i, int32_t j, int32_t k) {
    const auto *grid = static_cast<const nanovdb::FloatGrid *>(g);
    return grid->tree().getValue(nanovdb::Coord(i, j, k));
}
