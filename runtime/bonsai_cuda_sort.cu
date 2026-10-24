// The device sort of a queue's entry indices by their keys -- the GPU form
// of runtime/bonsai_parallel.h's bonsai_sort_indices, for a sorted queue
// (`f.sort(g, |args| key)` on a call that `defer` sends to a queue;
// SSA/Defer.cpp, ir::Intrinsic::sort_indices) whose drain runs on the GPU
// and whose arrays live in device memory. Same contract: fill perm[0..n)
// with the indices 0..n-1 ordered by ascending keys[index], ties in index
// order (a stable sort). Both arrays are device addresses; the call is
// issued on the null stream, where the generated code's launches are
// (bonsai_cuda.h), so it is ordered with them and the host neither waits
// for it nor reads anything back.
//
// The sort is CUB's radix sort of (key, value) pairs -- Merrill and
// Grimshaw, "High Performance and Scalable Radix Sorting" (2011); Adinets
// and Merrill, "Onesweep: A Faster Least Significant Digit Radix Sort for
// GPUs" (2022), which is the algorithm CUB's DeviceRadixSort runs on this
// hardware -- over the full 64 bits of the key (ray reordering keys are
// 64-bit Morton codes, Meister, Boksansky, Guthe, Bittner 2025), the values
// the indices 0..n-1 written by a kernel here. CUB sorts out of place, so
// the keys are written sorted to a scratch array nobody reads (the queue's
// own key array stays as the entries were pushed, read through `perm`),
// and the scratch -- the sorted keys, the index values, CUB's temporary
// storage -- is kept between calls and grown to the largest count seen,
// since a render's rounds sort the same queue a few hundred times and the
// first render is the warm-up. One scratch, since the host issues a
// program's sorts from one thread in the order of the rounds.
//
// This is a separate translation unit built by the CUDA toolkit's compiler
// (nvcc, CMakeLists.txt: the shared library bonsai_cuda_sort beside the
// compiler), because CUB is device code and the rest of the runtime is
// header-only C++ that loads the driver with dlopen and links no CUDA
// library. The runtime loads this library in the same way, at the first
// device sort (bonsai_cuda.h, bonsai_cuda_sort_indices), so a program
// without a sorted queue on the device never needs it.
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

__global__ void write_indices(uint32_t *out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = uint32_t(i);
    }
}

struct Scratch {
    uint64_t *keys_out = nullptr;
    uint32_t *indices = nullptr;
    void *temp = nullptr;
    size_t temp_bytes = 0;
    int64_t capacity = 0;
};

Scratch &scratch() {
    static Scratch s;
    return s;
}

void check(cudaError_t result, const char *what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "bonsai_cuda_sort_pairs: %s: %s\n", what,
                     cudaGetErrorString(result));
        std::abort();
    }
}

} // namespace

extern "C" void bonsai_cuda_sort_pairs(const uint64_t *keys, uint32_t *perm,
                                       int64_t n) {
    if (n <= 0) {
        return;
    }
    Scratch &s = scratch();
    if (n > s.capacity) {
        // Grown to the count, doubling so that a queue filling round by
        // round reallocates a few times and not once a round. cudaMalloc
        // waits for the stream, which is why the scratch is kept.
        const int64_t capacity = std::max<int64_t>(n, 2 * s.capacity);
        check(cudaFree(s.keys_out), "cudaFree(keys_out)");
        check(cudaFree(s.indices), "cudaFree(indices)");
        check(cudaFree(s.temp), "cudaFree(temp)");
        check(cudaMalloc(&s.keys_out, size_t(capacity) * sizeof(uint64_t)),
              "cudaMalloc(keys_out)");
        check(cudaMalloc(&s.indices, size_t(capacity) * sizeof(uint32_t)),
              "cudaMalloc(indices)");
        s.temp_bytes = 0;
        check(cub::DeviceRadixSort::SortPairs(nullptr, s.temp_bytes, keys, s.keys_out,
                                              s.indices, perm, capacity),
              "SortPairs(size)");
        check(cudaMalloc(&s.temp, s.temp_bytes), "cudaMalloc(temp)");
        s.capacity = capacity;
    }
    const int64_t blocks = (n + 255) / 256;
    write_indices<<<unsigned(blocks), 256>>>(s.indices, n);
    check(cudaGetLastError(), "write_indices");
    check(cub::DeviceRadixSort::SortPairs(s.temp, s.temp_bytes, keys, s.keys_out,
                                          s.indices, perm, n, 0, 64, cudaStream_t(0)),
          "SortPairs");
}
