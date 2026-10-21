#include "defer_sorted_queue.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// Used in defer_sorted_queue.bonsai
//
// Two checks. First the program: the order log the drained function writes
// is printed as the `-e` twin prints it (correctness/llvm/defer-sorted-queue),
// so the two agree on what a sorted drain does, and the total too. Then the
// runtime's sort itself, `bonsai_sort_indices` of runtime/bonsai_parallel.h,
// which the generated code called above over forty entries and which picks
// a different sort by size: called here at a hundred entries, five
// thousand, and a million and five, each with runs of equal keys (a third
// of the keys are drawn from ninety-seven values, the rest are sixty bits
// wide), and its permutation compared with a stable sort of the same keys.
// Counting the disagreements rather than printing the permutation keeps the
// output the same whichever sort the environment supplies; the two sorts
// the header falls back on are then checked by name.

static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

using Sort = void (*)(const uint64_t *, uint32_t *, int64_t);

static int64_t check_sort(Sort sort, int64_t n) {
    std::vector<uint64_t> keys(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        const uint64_t h = mix(static_cast<uint64_t>(i) + 1);
        keys[static_cast<size_t>(i)] = (i % 3 == 0) ? h % 97 : h >> 4;
    }
    std::vector<uint32_t> expected(static_cast<size_t>(n));
    std::iota(expected.begin(), expected.end(), 0u);
    std::stable_sort(expected.begin(), expected.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    std::vector<uint32_t> perm(static_cast<size_t>(n), 0xffffffffu);
    sort(keys.data(), perm.data(), n);
    int64_t wrong = 0;
    for (int64_t i = 0; i < n; i++) {
        if (perm[static_cast<size_t>(i)] != expected[static_cast<size_t>(i)]) {
            wrong++;
        }
    }
    return wrong;
}

int main() {
    // The keys 0..39 scrambled: x * 4 is the key.
    const std::array<float, 40> values = {
        7.25f, 0.0f,  9.5f,  3.75f, 1.0f,  8.0f,  2.5f,  6.0f,  4.25f, 9.75f,
        0.5f,  5.75f, 1.5f,  7.0f,  3.0f,  8.75f, 2.0f,  6.25f, 4.5f,  9.0f,
        0.25f, 5.5f,  1.75f, 7.5f,  3.25f, 8.25f, 2.75f, 6.5f,  4.0f,  9.25f,
        0.75f, 5.0f,  1.25f, 7.75f, 3.5f,  8.5f,  2.25f, 6.75f, 4.75f, 5.25f};
    std::array<int32_t, 1> counter = {0};
    std::array<int32_t, 60> order{};
    std::array<float, 1> total = {0.0f};
    run(values, counter, order, total);

    std::cout << "order:";
    for (int32_t key : order) {
        std::cout << ' ' << key;
    }
    std::cout << '\n';
    std::cout << "total: " << total[0] << '\n';

    for (int64_t n : {int64_t{100}, int64_t{5000}, (int64_t{1} << 20) + 5}) {
        std::cout << "sort of " << n << ": " << check_sort(bonsai_sort_indices, n)
                  << " wrong\n";
    }
    // The two sorts the header falls back on, by name, so that they are
    // checked on a machine whose environment picks the vectorized one.
    std::cout << "std of 100: " << check_sort(bonsai_sort_detail::sort_std, 100)
              << " wrong\n";
    std::cout << "radix of 5000: " << check_sort(bonsai_sort_detail::sort_radix, 5000)
              << " wrong\n";
}
