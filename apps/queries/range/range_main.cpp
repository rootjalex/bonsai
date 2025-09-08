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

#define PROFILE 1

template <typename Result, typename Func0, typename Func1, class... Args>
double benchmark_queries(const std::string &benchmark_name,
                         const set<int32_t> &input, const _tree_layout0 &tree,
                         const int k, const int m, Func0 &&f0, Func1 &&f1,
                         Args &&...args) {
    // Run and time query()
    std::vector<Result> query_results;
    int64_t avg_query_time = benchmark_function(
        [&]() {
            query_results.push_back(f0(std::forward<Args>(args)..., input));
        },
        k, m);

    // Run and time query_fast()
    std::vector<Result> fast_results;
    int64_t avg_fast_time = benchmark_function(
        [&]() {
            fast_results.push_back(f1(std::forward<Args>(args)..., tree));
        },
        k, m);

    // Verify all results match
    bool all_match = true;
    for (int i = 0; i < k; ++i) {
        if (!(query_results[i] == fast_results[i])) {
            all_match = false;
            break;
        }
    }
#ifndef PROFILE
    std::cout << benchmark_name << "() avg time: " << avg_query_time << " ns\n";
    std::cout << benchmark_name << "_fast() avg time: " << avg_fast_time
              << " ns\n";
#endif
    if (!all_match) {
        std::cerr << "ERROR: " << benchmark_name << " results differ! "
                  << input.size() << std::endl;
        std::abort();
    } else {
#ifndef PROFILE
        std::cout << "Results match.\n";
#endif
        if (avg_fast_time > 0) {
            double speedup =
                static_cast<double>(avg_query_time) / avg_fast_time;
#ifndef PROFILE
            std::cout << "Speedup: " << speedup << "x\n";
#else
            return speedup;
#endif
        } else {
            std::cout << benchmark_name
                      << " was too fast to measure accurately on input size: "
                      << input.size() << std::endl;
            return static_cast<double>(avg_query_time) / avg_fast_time; // inf
        }
    }
}

double benchmark_range_query(const set<int32_t> &input,
                             const _tree_layout0 &tree, const int k,
                             const int m) {
    int32_t low = -10;
    int32_t high = 10;
#ifndef PROFILE
    std::cout << "Range query, range = " << low << ", " << high << "]"
              << std::endl;
    std::cout << "Input size: " << input_set.size() << std::endl;
#endif
    // Example usage:
    return benchmark_queries<set<int32_t>>("range_query", input, tree, k, m,
                                           query, query_fast, low, high);
}

double benchmark_eq_query(const set<int32_t> &input, const _tree_layout0 &tree,
                          const int k, const int m) {
    // Random bounds
    int32_t value = 42;
#ifndef PROFILE
    std::cout << "Equality query, value = " << value << std::endl;
    std::cout << "Input size: " << input_set.size() << std::endl;
#endif
    // Example usage:
    return benchmark_queries<set<int32_t>>("range_query", input, tree, k, m,
                                           eq_query, eq_query_fast, value);
}

template <typename T>
void pretty_print_vector(const std::vector<T> &vec) {
    bool first = true;
    std::cout << "[";
    for (const auto &v : vec) {
        if (!first) {
            std::cout << ", ";
        }
        first = false;
        std::cout << v;
    }
    std::cout << "]";
}

int main() {
    const int k = 14; // total runs
    const int m = 2;  // number of fastest and slowest to drop

    // std::mt19937 rng(std::random_device{}());
    // For consistent results
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> bound_dist(-1000, 1000);
    // std::vector<size_t> test_sizes = {1,    5,     10,    100,     1000,
    //                                   5000, 10000, 65535, 1000000, 16777215};
    std::vector<size_t> test_sizes = {
        1 << 8,  1 << 9,  1 << 10, 1 << 11, 1 << 12,      1 << 13,
        1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18,      1 << 19,
        1 << 20, 1 << 21, 1 << 22, 1 << 23, (1 << 24) - 1};

#ifdef PROFILE
    pretty_print_vector(test_sizes);
    std::cout << std::endl;
    static constexpr int N_BENCHMARKS = 2;
    std::vector<std::pair<std::string, std::vector<double>>> results(
        N_BENCHMARKS);
    results[0].first = "range_query";
    results[0].second.reserve(test_sizes.size());
    results[1].first = "eq_query";
    results[1].second.reserve(test_sizes.size());
#endif
    for (size_t size : test_sizes) {
#ifndef PROFILE
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;
#endif
        // Generate input
        auto input_set = generate_random_set(size);

        // Build tree
        auto t_build_start = std::chrono::high_resolution_clock::now();
        auto input_tree = build_tree(input_set);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        int64_t build_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build_end -
                                                                 t_build_start)
                .count();
#ifndef PROFILE
        std::cout << "build_tree() time: " << build_time << " ns\n";
#endif

        results[0].second.push_back(
            benchmark_range_query(input_set, input_tree, k, m));

        results[1].second.push_back(
            benchmark_eq_query(input_set, input_tree, k, m));

        std::free(input_tree.prims);
        std::free(input_tree.group0_index);
    }
#ifdef PROFILE
    for (const auto &res : results) {
        std::cout << "(" << res.first << ", ";
        pretty_print_vector(res.second);
        std::cout << ")" << std::endl;
    }
#endif
    return 0;
}
