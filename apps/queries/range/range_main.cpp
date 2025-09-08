#include "range_fast_gen.h"
#include "range_gen.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>

// Helper function to generate a random set of int32_t
set<int32_t> generate_random_set(size_t size, int32_t min_val = -1000,
                                 int32_t max_val = 1000) {
    std::vector<int32_t> result;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);

    while (result.size() < size) {
        result.push_back(dist(rng));
    }
    return set<int32_t>(std::move(result));
}

void print_set(const set<int32_t> &s) {
    std::cout << "{ ";
    s.for_each([&](const int32_t &i) { std::cout << i << " "; });
    std::cout << "}" << std::endl;
}

_tree_layout0 build_tree(const set<int32_t> &input) {
    _tree_layout0 tree;
    tree.pCount = input.size();
    tree.prims =
        static_cast<int32_t *>(std::malloc(sizeof(int32_t) * tree.pCount));
    if (!tree.prims) {
        throw std::bad_alloc();
    }

    std::copy(input.data.begin(), input.data.end(), tree.prims);
    std::sort(tree.prims, tree.prims + tree.pCount);

    constexpr size_t MAX_LEAF_COUNT = 8;

    size_t leaf_count = (tree.pCount + (MAX_LEAF_COUNT - 1)) / MAX_LEAF_COUNT;
    // size_t internal_count = leaf_count - 1;
    // tree.count = leaf_count + internal_count;
    // tree.count = 2 * leaf_count - 1;
    tree.count = 2 * tree.pCount - 1;
    tree.group0_index = static_cast<_tree_layout1 *>(
        std::malloc(sizeof(_tree_layout1) * tree.count));

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        // assert(depth < MAX_TREE_DEPTH);

        uint32_t count = high - low;
        uint32_t this_index = next_node++;
        assert(this_index < tree.count);

        tree.group0_index[this_index].low = tree.prims[low];
        tree.group0_index[this_index].high = tree.prims[high - 1];

        if (count <= MAX_LEAF_COUNT) {
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            *reinterpret_cast<uint24_t *>(
                &tree.group0_index[this_index].split0on_nPrims) = low;
        } else {
            tree.group0_index[this_index].nPrims = 0;
            uint32_t mid = low + count / 2;

            uint32_t left = handle_range(low, mid, depth + 1);
            uint32_t right = handle_range(mid, high, depth + 1);

            // Set split offset (offset from this node to right child)
            uint32_t offset = right - this_index;
            *reinterpret_cast<uint24_t *>(
                &tree.group0_index[this_index].split0on_nPrims) = offset;
        }
        return this_index;
    };

    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

int main() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int32_t> bound_dist(-1000, 1000);
    std::vector<size_t> test_sizes = {1,    5,     10,    100,     1000,
                                      5000, 10000, 65535, 1000000, 16777215};

    for (size_t size : test_sizes) {
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;

        // Generate input
        auto input_set = generate_random_set(size);

        // Random bounds
        int32_t low = -10;
        int32_t high = 10;
        if (low > high)
            std::swap(low, high);

        std::cout << "Low: " << low << ", High: " << high << std::endl;
        std::cout << "Input size: " << input_set.size() << std::endl;

        // Time query()
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = query(low, high, input_set);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Time tree build
        auto t3 = std::chrono::high_resolution_clock::now();
        auto input_tree = build_tree(input_set);
        auto t4 = std::chrono::high_resolution_clock::now();

        // Time query_fast()
        auto t5 = std::chrono::high_resolution_clock::now();
        auto fast_result = query_fast(low, high, input_tree);
        auto t6 = std::chrono::high_resolution_clock::now();

        auto duration_query =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count();
        auto tree_build =
            std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3)
                .count();
        auto duration_fast =
            std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5)
                .count();

        std::cout << "query() result size: " << result.size()
                  << ", time: " << duration_query << " us" << std::endl;
        std::cout << "build_tree() time:        " << tree_build << " us"
                  << std::endl;
        std::cout << "query_fast() result size: " << fast_result.size()
                  << ", time: " << duration_fast << " us" << std::endl;

        // Validate equality
        if (!sets_equal(result, fast_result)) {
            std::cerr << "ERROR: query() and query_fast() results differ!"
                      << std::endl;
            abort();
        } else {
            std::cout << "Results match." << std::endl;
            if (duration_fast > 0) {
                double speedup =
                    static_cast<double>(duration_query) / duration_fast;
                std::cout << "Speedup (query / query_fast): " << speedup
                          << "x\n";
            } else {
                std::cout
                    << "query_fast() was too fast to measure accurately.\n";
            }
        }

        std::free(input_tree.prims);
        std::free(input_tree.group0_index);
    }

    return 0;
}
