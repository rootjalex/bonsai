#pragma once

#include <stdint.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// The CUDA runtime a `bind(i, GPUBlock)` or `bind(i, GPUThread)` schedule
// lowers to.
//
// Generated code makes exactly one call per bound loop, to
// `bonsai_cuda_launch`, meaning: run the kernel named `kernel` from the PTX
// module `ptx` over `grid_x` blocks of `block_x` threads, with `params` as
// its arguments, and return when it is done. Everything about *how* -- the
// driver API, contexts, module loading, memory -- is this file's, so that the
// generated code, and every golden of it, knows only the arguments.
//
// The driver is found at run time by `dlopen("libcuda.so.1")`, so a program
// that never launches links and runs on a machine without a GPU, and the
// compiler itself has no build-time dependency on the CUDA toolkit. A
// program that does launch on such a machine stops with a message saying so.
//
// Memory. A kernel argument that is an address is described by a
// `bonsai_cuda_buffer`: the host memory it names, how many bytes of it, and
// which parameter slot holds it. Before the launch each buffer is copied to
// device memory of its own and the slot is repointed at it; after the launch
// each is copied back and freed. This is the placement a loop-local array
// has when the schedule puts its loop on the GPU: the array lives with the
// function that owns it, on the host, and the schedule moved the loop, so it
// moves the data. An exported function's arrays are different: they arrive
// as `bonsai_buffer`s (runtime/bonsai_buffer.h) that know where they are
// resident, and the launch asks for them on the device rather than copying;
// the memory primitives at the end are what that runtime allocates and
// copies with.
//
// Header-only, like runtime/bonsai_parallel.h, so that a driver has the
// whole runtime by including the generated header; the compiler includes it
// once too (runtime/bonsai_cuda.cpp), for the programs it runs itself and
// to ask which GPU the machine has.

extern "C" {

struct bonsai_cuda_buffer {
    void *host;
    uint64_t bytes;
    // The index into `params` of the slot holding this buffer's address.
    uint64_t param;
};

// Runs `kernel` of `ptx` over grid_x x 1 x 1 blocks of block_x x 1 x 1
// threads. `params[i]` is the address of the i-th argument's value, as
// cuLaunchKernel takes them. The PTX module is loaded once per distinct
// `ptx` pointer and kept; the primary context of device 0 is used. Any
// failure -- no driver, no device, a kernel the PTX does not define, a
// launch the device rejects -- prints the driver's reason and aborts, since
// a render that silently skipped its kernel would be a wrong image rather
// than an error.
void bonsai_cuda_launch(const char *ptx, const char *kernel, int64_t grid_x,
                        int64_t block_x, void **params, int64_t nparams,
                        bonsai_cuda_buffer *buffers, int64_t nbuffers);

// Loads the PTX module ahead of its first launch, so that the load is part
// of a driver's setup rather than of its first timed call. What the
// generated `bonsai_gpu_prepare()` calls (see CodeGen_GPU_Host).
void bonsai_cuda_load(const char *ptx);

// The compute capability of device 0, as NVPTX names it -- "sm_120" -- or
// an empty string when there is no driver or no device. What the compiler
// follows when no `--gpu-arch` was given.
const char *bonsai_cuda_device_arch(void);

// Device memory, for runtime/bonsai_buffer.h: `bytes` of it (never zero),
// its release, and the two copies. Allocation and release are the
// stream-ordered `cuMemAllocAsync`/`cuMemFreeAsync` on the null stream,
// which take memory from and return it to the device's pool rather than
// synchronizing the device as the plain calls do. Each aborts with the
// driver's reason on failure, as the launch does.
void *bonsai_cuda_malloc(uint64_t bytes);
void bonsai_cuda_free(void *device);
void bonsai_cuda_copy_to_device(void *device, const void *host, uint64_t bytes);
void bonsai_cuda_copy_to_host(void *host, const void *device, uint64_t bytes);

// A 2D image texture as the GPU's texture units sample it, for a program
// whose schedule bound a texture lookup to `TextureUnit`
// (`tex_sample_grad_2d` in the compiler): a mipmapped CUDA array of `levels`
// levels, level `l` being `widths[l]` by `heights[l]` texels of four floats
// (RGBA, rows tightly packed) at `texels[l]`, and a texture object over it
// -- linear filtering within a level, the nearest level for the gradients
// given (point mipmap filtering, which is what pbrt's GPU build sets for its
// default `bilinear` filter), normalized coordinates, `wrap` 0 repeating, 1
// clamping, 2 a black border, and `max_anisotropy` as pbrt's `maxanisotropy`
// (8 by default). This is pbrt's GPUSpectrumImageTexture::Create, driver
// API for runtime API. Returns the object's handle, which the kernel
// samples with; `bonsai_cuda_texture_destroy` releases it and its array.
// Aborts with the driver's reason on failure.
uint64_t bonsai_cuda_texture_create(int64_t levels, const uint32_t *widths,
                                    const uint32_t *heights,
                                    const float *const *texels, int32_t wrap,
                                    int32_t max_anisotropy);
void bonsai_cuda_texture_destroy(uint64_t texture);
}

// The few entry points of the CUDA driver API this uses, declared here
// rather than by including cuda.h, so that neither the compiler nor a
// program it produced needs the CUDA toolkit to build -- only the driver, at
// run time, to launch. The types are the driver's: a device is an int,
// handles are opaque pointers, a device address is 64 bits. The `_v2` names
// are the 64-bit ABI the driver has exported since CUDA 3.2; the unsuffixed
// names in cuda.h are macros for them.
namespace bonsai_cuda_detail {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream = struct CUstream_st *;
using CUdeviceptr = unsigned long long;
using CUarray = struct CUarray_st *;
using CUmipmappedArray = struct CUmipmappedArray_st *;
using CUtexObject = unsigned long long;

// The texture objects' descriptors, laid out as cuda.h lays them out
// (CUDA_ARRAY3D_DESCRIPTOR_v2, CUDA_MEMCPY2D_v2, CUDA_RESOURCE_DESC_v1,
// CUDA_TEXTURE_DESC_v1): the driver reads these by offset, so every field
// and its padding is the header's. The enums are ints of the header's
// values.
constexpr int CU_AD_FORMAT_FLOAT = 0x20;
constexpr int CU_MEMORYTYPE_HOST = 1;
constexpr int CU_MEMORYTYPE_ARRAY = 3;
constexpr int CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1;
constexpr int CU_TR_ADDRESS_MODE_WRAP = 0;
constexpr int CU_TR_ADDRESS_MODE_CLAMP = 1;
constexpr int CU_TR_ADDRESS_MODE_BORDER = 3;
constexpr int CU_TR_FILTER_MODE_POINT = 0;
constexpr int CU_TR_FILTER_MODE_LINEAR = 1;
constexpr unsigned CU_TRSF_NORMALIZED_COORDINATES = 0x02;

struct CUDA_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    int Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};

struct CUDA_RESOURCE_DESC {
    int resType;
    union {
        struct {
            CUarray hArray;
        } array;
        struct {
            CUmipmappedArray hMipmappedArray;
        } mipmap;
        // The header's largest member is its `pitch2D` at 40 bytes; the
        // reserve is what fixes the union's size at 128.
        struct {
            CUdeviceptr devPtr;
            int reserved[30];
        } reserved;
    } res;
    unsigned int flags;
};

struct CUDA_TEXTURE_DESC {
    int addressMode[3];
    int filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};
static_assert(sizeof(CUDA_ARRAY3D_DESCRIPTOR) == 40, "cuda.h's layout");
static_assert(sizeof(CUDA_MEMCPY2D) == 128, "cuda.h's layout");
static_assert(sizeof(CUDA_RESOURCE_DESC) == 144, "cuda.h's layout");
static_assert(sizeof(CUDA_TEXTURE_DESC) == 104, "cuda.h's layout");

constexpr CUresult CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct Driver {
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream);
    CUresult (*cuMemFreeAsync)(CUdeviceptr, CUstream);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void **, void **);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuGetErrorName)(CUresult, const char **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
    // The texture objects (bonsai_cuda_texture_create).
    CUresult (*cuMipmappedArrayCreate)(CUmipmappedArray *,
                                       const CUDA_ARRAY3D_DESCRIPTOR *,
                                       unsigned int);
    CUresult (*cuMipmappedArrayGetLevel)(CUarray *, CUmipmappedArray,
                                         unsigned int);
    CUresult (*cuMipmappedArrayDestroy)(CUmipmappedArray);
    CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D *);
    CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                  const CUDA_TEXTURE_DESC *, const void *);
    CUresult (*cuTexObjectDestroy)(CUtexObject);
    // Each texture object's array, to release with it.
    std::unordered_map<CUtexObject, CUmipmappedArray> texture_arrays;

    // Loaded once. `ok` says whether every symbol was found and a device
    // exists; `why` is the reason when not.
    bool ok = false;
    std::string why;
    CUdevice device = 0;
    CUcontext context = nullptr;
    int max_threads_per_block = 0;
    std::string arch;
    std::unordered_map<const char *, CUmodule> modules;
    std::mutex mutex;
};

inline Driver &driver() {
    static Driver d;
    static std::once_flag once;
    std::call_once(once, [] {
        void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (lib == nullptr) {
            d.why = "libcuda.so.1 could not be loaded (is the NVIDIA driver "
                    "installed?)";
            return;
        }
        const auto load = [&](auto &fn, const char *name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
                dlsym(lib, name));
            if (fn == nullptr && d.why.empty()) {
                d.why = std::string("libcuda.so.1 has no ") + name;
            }
        };
        load(d.cuInit, "cuInit");
        load(d.cuDeviceGetCount, "cuDeviceGetCount");
        load(d.cuDeviceGet, "cuDeviceGet");
        load(d.cuDeviceGetAttribute, "cuDeviceGetAttribute");
        load(d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        load(d.cuCtxSetCurrent, "cuCtxSetCurrent");
        load(d.cuModuleLoadData, "cuModuleLoadData");
        load(d.cuModuleGetFunction, "cuModuleGetFunction");
        load(d.cuMemAllocAsync, "cuMemAllocAsync");
        load(d.cuMemFreeAsync, "cuMemFreeAsync");
        load(d.cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        load(d.cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        load(d.cuLaunchKernel, "cuLaunchKernel");
        load(d.cuCtxSynchronize, "cuCtxSynchronize");
        load(d.cuGetErrorName, "cuGetErrorName");
        load(d.cuGetErrorString, "cuGetErrorString");
        load(d.cuMipmappedArrayCreate, "cuMipmappedArrayCreate");
        load(d.cuMipmappedArrayGetLevel, "cuMipmappedArrayGetLevel");
        load(d.cuMipmappedArrayDestroy, "cuMipmappedArrayDestroy");
        load(d.cuMemcpy2D, "cuMemcpy2D_v2");
        load(d.cuTexObjectCreate, "cuTexObjectCreate");
        load(d.cuTexObjectDestroy, "cuTexObjectDestroy");
        if (!d.why.empty()) {
            return;
        }
        const auto failed = [&](CUresult r, const char *what) {
            if (r == CUDA_SUCCESS) {
                return false;
            }
            const char *name = "?";
            d.cuGetErrorName(r, &name);
            d.why = std::string(what) + " failed: " + name;
            return true;
        };
        int count = 0;
        if (failed(d.cuInit(0), "cuInit") ||
            failed(d.cuDeviceGetCount(&count), "cuDeviceGetCount")) {
            return;
        }
        if (count == 0) {
            d.why = "the driver reports no CUDA device";
            return;
        }
        int major = 0, minor = 0;
        if (failed(d.cuDeviceGet(&d.device, 0), "cuDeviceGet") ||
            failed(d.cuDeviceGetAttribute(
                       &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       d.device),
                   "cuDeviceGetAttribute(compute capability major)") ||
            failed(d.cuDeviceGetAttribute(
                       &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       d.device),
                   "cuDeviceGetAttribute(compute capability minor)") ||
            failed(d.cuDeviceGetAttribute(
                       &d.max_threads_per_block,
                       CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, d.device),
                   "cuDeviceGetAttribute(max threads per block)")) {
            return;
        }
        d.arch = "sm_" + std::to_string(major * 10 + minor);
        d.ok = true;
    });
    return d;
}

[[noreturn]] inline void fail(const std::string &message) {
    std::fprintf(stderr, "bonsai_cuda: %s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}

inline void check(Driver &d, CUresult r, const std::string &what) {
    if (r == CUDA_SUCCESS) {
        return;
    }
    const char *name = "?";
    const char *why = "";
    d.cuGetErrorName(r, &name);
    d.cuGetErrorString(r, &why);
    fail(what + " failed: " + name + " (" + why + ")");
}

// The driver, ready to be used, with the context current on this thread;
// the first use makes the context. Aborts with the reason when there is no
// usable driver or device.
inline Driver &ready(const char *what) {
    Driver &d = driver();
    if (!d.ok) {
        fail(std::string("cannot ") + what + ": " + d.why);
    }
    if (d.context == nullptr) {
        check(d, d.cuDevicePrimaryCtxRetain(&d.context, d.device),
              "cuDevicePrimaryCtxRetain");
    }
    check(d, d.cuCtxSetCurrent(d.context), "cuCtxSetCurrent");
    return d;
}

// The module for `ptx`, loaded on first use and kept. Keyed by the address
// of the text, which is a constant of the program that embeds it and so the
// same for every launch from that program. The caller holds the mutex.
inline CUmodule module_of(Driver &d, const char *ptx) {
    if (const auto it = d.modules.find(ptx); it != d.modules.end()) {
        return it->second;
    }
    CUmodule module = nullptr;
    check(d, d.cuModuleLoadData(&module, ptx), "cuModuleLoadData");
    d.modules[ptx] = module;
    return module;
}

// The null stream: every allocation, release and launch here is ordered on
// it, so a kernel sees the memory allocated before it and a release waits
// for the kernel that used the memory.
constexpr CUstream null_stream = nullptr;

} // namespace bonsai_cuda_detail

extern "C" {

__attribute__((used)) inline const char *bonsai_cuda_device_arch(void) {
    bonsai_cuda_detail::Driver &d = bonsai_cuda_detail::driver();
    return d.ok ? d.arch.c_str() : "";
}

__attribute__((used)) inline void bonsai_cuda_load(const char *ptx) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("load the PTX module");
    std::lock_guard<std::mutex> lock(d.mutex);
    module_of(d, ptx);
}

__attribute__((used)) inline void *bonsai_cuda_malloc(uint64_t bytes) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("allocate device memory");
    std::lock_guard<std::mutex> lock(d.mutex);
    CUdeviceptr device = 0;
    check(d,
          d.cuMemAllocAsync(&device, size_t(bytes == 0 ? 1 : bytes),
                            null_stream),
          "cuMemAllocAsync(" + std::to_string(bytes) + ")");
    return reinterpret_cast<void *>(device);
}

__attribute__((used)) inline void bonsai_cuda_free(void *device) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("free device memory");
    std::lock_guard<std::mutex> lock(d.mutex);
    check(d,
          d.cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(device), null_stream),
          "cuMemFreeAsync");
}

__attribute__((used)) inline void
bonsai_cuda_copy_to_device(void *device, const void *host, uint64_t bytes) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("copy to the device");
    std::lock_guard<std::mutex> lock(d.mutex);
    check(d,
          d.cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(device), host,
                         size_t(bytes)),
          "cuMemcpyHtoD(" + std::to_string(bytes) + ")");
}

__attribute__((used)) inline void
bonsai_cuda_copy_to_host(void *host, const void *device, uint64_t bytes) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("copy to the host");
    std::lock_guard<std::mutex> lock(d.mutex);
    check(d,
          d.cuMemcpyDtoH(host, reinterpret_cast<CUdeviceptr>(device),
                         size_t(bytes)),
          "cuMemcpyDtoH(" + std::to_string(bytes) + ")");
}

__attribute__((used)) inline uint64_t
bonsai_cuda_texture_create(int64_t levels, const uint32_t *widths,
                           const uint32_t *heights, const float *const *texels,
                           int32_t wrap, int32_t max_anisotropy) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("create a texture object");
    std::lock_guard<std::mutex> lock(d.mutex);
    if (levels < 1) {
        fail("a texture needs at least one level");
    }

    // The pyramid: one mipmapped array of `levels` levels, each filled from
    // its rows of RGBA floats.
    CUDA_ARRAY3D_DESCRIPTOR shape = {};
    shape.Width = widths[0];
    shape.Height = heights[0];
    shape.Depth = 0;
    shape.Format = CU_AD_FORMAT_FLOAT;
    shape.NumChannels = 4;
    shape.Flags = 0;
    CUmipmappedArray pyramid = nullptr;
    check(d, d.cuMipmappedArrayCreate(&pyramid, &shape, unsigned(levels)),
          "cuMipmappedArrayCreate(" + std::to_string(widths[0]) + "x" +
              std::to_string(heights[0]) + ", " + std::to_string(levels) +
              " levels)");
    for (int64_t l = 0; l < levels; l++) {
        CUarray level = nullptr;
        check(d, d.cuMipmappedArrayGetLevel(&level, pyramid, unsigned(l)),
              "cuMipmappedArrayGetLevel(" + std::to_string(l) + ")");
        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = texels[l];
        copy.srcPitch = size_t(widths[l]) * 4 * sizeof(float);
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = level;
        copy.WidthInBytes = copy.srcPitch;
        copy.Height = heights[l];
        check(d, d.cuMemcpy2D(&copy),
              "cuMemcpy2D(texture level " + std::to_string(l) + ")");
    }

    // The object over it: pbrt's texture descriptor for an image texture
    // (textures.cpp, GPUSpectrumImageTexture::Create) -- linear within a
    // level, the nearest level (its default `bilinear` filter), the levels
    // clamped to the pyramid, the wrap mode the scene named, a black border
    // for `black`, normalized coordinates.
    CUDA_RESOURCE_DESC resource = {};
    resource.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resource.res.mipmap.hMipmappedArray = pyramid;
    CUDA_TEXTURE_DESC texture = {};
    const int address = wrap == 0   ? CU_TR_ADDRESS_MODE_WRAP
                        : wrap == 1 ? CU_TR_ADDRESS_MODE_CLAMP
                                    : CU_TR_ADDRESS_MODE_BORDER;
    texture.addressMode[0] = address;
    texture.addressMode[1] = address;
    texture.addressMode[2] = address;
    texture.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texture.flags = CU_TRSF_NORMALIZED_COORDINATES;
    texture.maxAnisotropy =
        unsigned(max_anisotropy < 1 ? 1 : max_anisotropy > 16 ? 16 : max_anisotropy);
    texture.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    texture.mipmapLevelBias = 0.f;
    texture.minMipmapLevelClamp = 0.f;
    texture.maxMipmapLevelClamp = float(levels - 1);
    CUtexObject object = 0;
    check(d, d.cuTexObjectCreate(&object, &resource, &texture, nullptr),
          "cuTexObjectCreate");
    d.texture_arrays[object] = pyramid;
    return uint64_t(object);
}

__attribute__((used)) inline void bonsai_cuda_texture_destroy(uint64_t texture) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready("destroy a texture object");
    std::lock_guard<std::mutex> lock(d.mutex);
    const auto it = d.texture_arrays.find(CUtexObject(texture));
    if (it == d.texture_arrays.end()) {
        fail("bonsai_cuda_texture_destroy: not a texture this runtime made");
    }
    check(d, d.cuTexObjectDestroy(CUtexObject(texture)), "cuTexObjectDestroy");
    check(d, d.cuMipmappedArrayDestroy(it->second), "cuMipmappedArrayDestroy");
    d.texture_arrays.erase(it);
}

__attribute__((used)) inline void
bonsai_cuda_launch(const char *ptx, const char *kernel, int64_t grid_x,
                   int64_t block_x, void **params, int64_t nparams,
                   bonsai_cuda_buffer *buffers, int64_t nbuffers) {
    using namespace bonsai_cuda_detail;
    Driver &d = ready((std::string("launch `") + kernel + "`").c_str());
    // One launch at a time: the module cache and the context are shared,
    // and a program's bound loops are launched from whatever thread reaches
    // them.
    std::lock_guard<std::mutex> lock(d.mutex);

    CUmodule module = module_of(d, ptx);
    CUfunction function = nullptr;
    check(d, d.cuModuleGetFunction(&function, module, kernel),
          "cuModuleGetFunction(" + std::string(kernel) + ")");

    if (block_x < 1 || block_x > d.max_threads_per_block) {
        fail("`" + std::string(kernel) + "` asks for a block of " +
             std::to_string(block_x) + " threads; the device allows 1 to " +
             std::to_string(d.max_threads_per_block) +
             ". Split the thread loop so that its inner count fits.");
    }
    if (grid_x < 1 || grid_x > 2147483647LL) {
        fail("`" + std::string(kernel) + "` asks for a grid of " +
             std::to_string(grid_x) + " blocks; the device allows 1 to 2^31-1.");
    }

    // The buffers: to the device, and their slots repointed.
    std::vector<CUdeviceptr> device_memory(size_t(nbuffers), 0);
    for (int64_t b = 0; b < nbuffers; b++) {
        const bonsai_cuda_buffer &buffer = buffers[b];
        if (buffer.param >= uint64_t(nparams)) {
            fail("buffer " + std::to_string(b) + " of `" + kernel +
                 "` names parameter " + std::to_string(buffer.param) +
                 " of " + std::to_string(nparams));
        }
        // A zero-length array is a valid pointer with nothing behind it;
        // an allocation of zero bytes is an error, so it is a byte.
        const size_t bytes = buffer.bytes == 0 ? 1 : size_t(buffer.bytes);
        check(d, d.cuMemAllocAsync(&device_memory[b], bytes, null_stream),
              "cuMemAllocAsync(" + std::to_string(bytes) + ")");
        if (buffer.bytes != 0) {
            check(d,
                  d.cuMemcpyHtoD(device_memory[b], buffer.host,
                                 size_t(buffer.bytes)),
                  "cuMemcpyHtoD");
        }
        *static_cast<CUdeviceptr *>(params[buffer.param]) = device_memory[b];
    }

    check(d,
          d.cuLaunchKernel(function, unsigned(grid_x), 1, 1, unsigned(block_x),
                           1, 1, /*sharedMemBytes=*/0, /*stream=*/nullptr,
                           params, nullptr),
          "cuLaunchKernel(" + std::string(kernel) + ")");
    check(d, d.cuCtxSynchronize(),
          "cuCtxSynchronize after " + std::string(kernel));

    // Back, and the slots as they were: the host's own addresses.
    for (int64_t b = 0; b < nbuffers; b++) {
        const bonsai_cuda_buffer &buffer = buffers[b];
        if (buffer.bytes != 0) {
            check(d,
                  d.cuMemcpyDtoH(buffer.host, device_memory[b],
                                 size_t(buffer.bytes)),
                  "cuMemcpyDtoH");
        }
        check(d, d.cuMemFreeAsync(device_memory[b], null_stream),
              "cuMemFreeAsync");
        *static_cast<void **>(params[buffer.param]) = buffer.host;
    }
}

} // extern "C"
