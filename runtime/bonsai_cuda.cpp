#include "bonsai_cuda.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// The few entry points of the CUDA driver API this uses, declared here
// rather than by including cuda.h, so that neither the compiler nor a
// program it produced needs the CUDA toolkit to build -- only the driver, at
// run time, to launch. The types are the driver's: a device is an int,
// handles are opaque pointers, a device address is 64 bits. The `_v2` names
// are the 64-bit ABI the driver has exported since CUDA 3.2; the unsuffixed
// names in cuda.h are macros for them.
namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream = struct CUstream_st *;
using CUdeviceptr = unsigned long long;

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
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void **, void **);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuGetErrorName)(CUresult, const char **);
    CUresult (*cuGetErrorString)(CUresult, const char **);

    // Loaded once. `ok` says whether every symbol was found; `why` is the
    // reason when it was not.
    bool ok = false;
    std::string why;
    CUdevice device = 0;
    CUcontext context = nullptr;
    int max_threads_per_block = 0;
    std::string arch;
    std::unordered_map<const char *, CUmodule> modules;
    std::mutex mutex;
};

Driver &driver() {
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
        load(d.cuMemAlloc, "cuMemAlloc_v2");
        load(d.cuMemFree, "cuMemFree_v2");
        load(d.cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        load(d.cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        load(d.cuLaunchKernel, "cuLaunchKernel");
        load(d.cuCtxSynchronize, "cuCtxSynchronize");
        load(d.cuGetErrorName, "cuGetErrorName");
        load(d.cuGetErrorString, "cuGetErrorString");
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

[[noreturn]] void fail(const std::string &message) {
    std::fprintf(stderr, "bonsai_cuda: %s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}

void check(Driver &d, CUresult r, const std::string &what) {
    if (r == CUDA_SUCCESS) {
        return;
    }
    const char *name = "?";
    const char *why = "";
    d.cuGetErrorName(r, &name);
    d.cuGetErrorString(r, &why);
    fail(what + " failed: " + name + " (" + why + ")");
}

// The context, made current on this thread; the first launch makes it.
void require_context(Driver &d) {
    if (d.context == nullptr) {
        check(d, d.cuDevicePrimaryCtxRetain(&d.context, d.device),
              "cuDevicePrimaryCtxRetain");
    }
    check(d, d.cuCtxSetCurrent(d.context), "cuCtxSetCurrent");
}

} // namespace

extern "C" const char *bonsai_cuda_device_arch(void) {
    Driver &d = driver();
    return d.ok ? d.arch.c_str() : "";
}

extern "C" void bonsai_cuda_launch(const char *ptx, const char *kernel,
                                   int64_t grid_x, int64_t block_x,
                                   void **params, int64_t nparams,
                                   bonsai_cuda_buffer *buffers,
                                   int64_t nbuffers) {
    Driver &d = driver();
    if (!d.ok) {
        fail("cannot launch `" + std::string(kernel) + "`: " + d.why);
    }
    // One launch at a time: the module cache and the context are shared,
    // and a program's bound loops are launched from whatever thread reaches
    // them.
    std::lock_guard<std::mutex> lock(d.mutex);
    require_context(d);

    // The module, loaded on first use. Keyed by the address of the text,
    // which is a constant of the program that embeds it and so the same for
    // every launch from that program.
    CUmodule module = nullptr;
    if (const auto it = d.modules.find(ptx); it != d.modules.end()) {
        module = it->second;
    } else {
        check(d, d.cuModuleLoadData(&module, ptx), "cuModuleLoadData");
        d.modules[ptx] = module;
    }
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
        if (buffer.param < 0 || buffer.param >= uint64_t(nparams)) {
            fail("buffer " + std::to_string(b) + " of `" + kernel +
                 "` names parameter " + std::to_string(buffer.param) +
                 " of " + std::to_string(nparams));
        }
        // A zero-length array is a valid pointer with nothing behind it;
        // cuMemAlloc of zero bytes is an error, so it is a byte.
        const size_t bytes = buffer.bytes == 0 ? 1 : size_t(buffer.bytes);
        check(d, d.cuMemAlloc(&device_memory[b], bytes),
              "cuMemAlloc(" + std::to_string(bytes) + ")");
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
    check(d, d.cuCtxSynchronize(), "cuCtxSynchronize after " + std::string(kernel));

    // Back, and the slots as they were: the host's own addresses.
    for (int64_t b = 0; b < nbuffers; b++) {
        const bonsai_cuda_buffer &buffer = buffers[b];
        if (buffer.bytes != 0) {
            check(d,
                  d.cuMemcpyDtoH(buffer.host, device_memory[b],
                                 size_t(buffer.bytes)),
                  "cuMemcpyDtoH");
        }
        check(d, d.cuMemFree(device_memory[b]), "cuMemFree");
        *static_cast<void **>(params[buffer.param]) = buffer.host;
    }
}
