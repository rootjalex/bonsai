#include "sort-indices-cpp.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Used in sort-indices-cpp.bonsai: the device sort against the standard
// library's stable sort.

// Sorts `keys` on the device and returns the permutation.
static std::vector<uint32_t> device_sort(const std::vector<uint64_t> &keys) {
    const int64_t n = int64_t(keys.size());
    auto *d_keys = static_cast<uint64_t *>(bonsai_cuda_malloc(uint64_t(n) * sizeof(uint64_t)));
    auto *d_perm = static_cast<uint32_t *>(bonsai_cuda_malloc(uint64_t(n) * sizeof(uint32_t)));
    bonsai_cuda_copy_to_device(d_keys, keys.data(), uint64_t(n) * sizeof(uint64_t));
    bonsai_cuda_sort_indices(d_keys, d_perm, n);
    bonsai_cuda_synchronize();
    std::vector<uint32_t> perm(static_cast<size_t>(n));
    bonsai_cuda_copy_to_host(perm.data(), d_perm, uint64_t(n) * sizeof(uint32_t));
    bonsai_cuda_free(d_keys);
    bonsai_cuda_free(d_perm);
    return perm;
}

// The host's answer: the indices by ascending key, ties in index order.
static std::vector<uint32_t> host_sort(const std::vector<uint64_t> &keys) {
    std::vector<uint32_t> perm(keys.size());
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    return perm;
}

static void check(const char *what, const std::vector<uint64_t> &keys) {
    const std::vector<uint32_t> got = device_sort(keys);
    const std::vector<uint32_t> want = host_sort(keys);
    for (size_t i = 0; i < keys.size(); i++) {
        if (got[i] != want[i]) {
            std::cout << what << ": differs at " << i << ": index " << got[i]
                      << " (key " << keys[got[i]] << ") where " << want[i] << " (key "
                      << keys[want[i]] << ") was expected\n";
            return;
        }
    }
    std::cout << what << ": ok\n";
}

int main() {
    // A million random 36-bit keys, every thousandth repeating the first,
    // so that the stable order of equal keys is tested across the array.
    std::vector<uint64_t> keys(1000003);
    std::mt19937_64 rng(7);
    for (uint64_t &k : keys) {
        k = rng() >> 28;
    }
    for (size_t i = 0; i < keys.size(); i += 1000) {
        keys[i] = keys[0];
    }
    check("sorted 1000003 keys", keys);
    // A second, smaller sort reuses the scratch of the first; all keys
    // equal, so the permutation is the identity.
    check("sorted 1000 equal keys", std::vector<uint64_t>(1000, 42));
    return 0;
}
