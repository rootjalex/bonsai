#pragma once

#include <stdint.h>

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
// moves the data. Buffers that persist across launches -- a scene, a film --
// are the buffer descriptors of apps/pbrt/PLAN.md, "Where the data lives",
// which will let a driver stage them once; that is built on top of this.

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

// The compute capability of device 0, as NVPTX names it -- "sm_120" -- or
// an empty string when there is no driver or no device. What the compiler
// follows when no `--gpu-arch` was given.
const char *bonsai_cuda_device_arch(void);
}
