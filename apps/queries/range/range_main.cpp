#include "range_fast_gen.h"
#include "range_gen.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

// -------- Choose one by uncommenting or defining via -D flag --------
// #define USE_UNIFORM
// #define USE_NORMAL
// #define USE_EXPONENTIAL
// #define USE_LOGNORMAL
// #define USE_CAUCHY
// #define USE_WEIBULL

set<float> generate_random_set(std::mt19937 &rng, size_t size,
                               float min_val = -1000, float max_val = 1000) {
    set<float> result;
#if defined(USE_NORMAL)
    // Centered at 0, stddev so most values fall in [min, max]
    std::normal_distribution<float> dist(0.0f, (max_val - min_val) / 4.0f);

#elif defined(USE_EXPONENTIAL)
    // Shifted exponential: λ controls spread; result shifted by min_val
    std::exponential_distribution<float> dist(1.0f / (max_val - min_val));

#elif defined(USE_LOGNORMAL)
    // log-normal parameters — mean and stddev of the underlying normal
    std::lognormal_distribution<float> dist(0.0f, (max_val - min_val) / 4.0f);

#elif defined(USE_CAUCHY)
    // Heavy-tailed: median=0, scale controls width
    std::cauchy_distribution<float> dist(0.0f, (max_val - min_val) / 10.0f);

#elif defined(USE_WEIBULL)
    // Shape > 0, scale > 0; used in survival analysis, reliability
    std::weibull_distribution<float> dist(2.0f, (max_val - min_val) / 2.0f);

#else
    std::uniform_real_distribution<float> dist(min_val, max_val);
#endif

    while (result.size() < size) {
        float val = dist(rng);

#if defined(USE_EXPONENTIAL)
        val += min_val;
#elif defined(USE_LOGNORMAL) || defined(USE_CAUCHY) || defined(USE_WEIBULL)
        val = min_val + fmod(val, max_val - min_val); // wrap into range
#endif

        // Optional clamp for distributions that might go out of range
        if (val < min_val || val > max_val || !std::isfinite(val))
            continue;

        result.push_back(val);
    }

    return result;
}

template <typename T>
void print_set(const set<T> &s) {
    std::cout << "{ ";
    s.for_each([&](const T &i) { std::cout << i << " "; });
    std::cout << "}" << std::endl;
}

void export_to_csv(const set<float> &input_set, const std::string &filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename
                  << std::endl;
        return;
    }

    out << "value\n"; // CSV header
    input_set.for_each([&](const float &val) { out << val << "\n"; });

    out.close();
}

_tree_layout0 build_tree(const set<float> &input) {
    _tree_layout0 tree;
    tree.pCount = input.size();
    tree.prims = static_cast<float *>(std::malloc(sizeof(float) * tree.pCount));
    if (!tree.prims) {
        throw std::bad_alloc();
    }

    std::copy(input.data.begin(), input.data.end(), tree.prims);
    std::sort(tree.prims, tree.prims + tree.pCount);

    constexpr uint64_t MAX_LEAF_COUNT = 8;

    uint64_t leaf_count = (tree.pCount + (MAX_LEAF_COUNT - 1)) / MAX_LEAF_COUNT;
    // uint64_t internal_count = leaf_count - 1;
    // tree.count = leaf_count + internal_count;
    // tree.count = 2 * leaf_count - 1;
    tree.count = 2 * tree.pCount - 1;
    tree.group0_index = static_cast<_tree_layout1 *>(
        std::malloc(sizeof(_tree_layout1) * tree.count));

    uint64_t next_node = 0;

    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> handle_range =
        [&](uint64_t low, uint64_t high, uint64_t depth) -> uint64_t {
        // assert(depth < MAX_TREE_DEPTH);

        uint64_t count = high - low;
        uint64_t this_index = next_node++;
        assert(this_index < tree.count);

        tree.group0_index[this_index].low = tree.prims[low];
        tree.group0_index[this_index].high = tree.prims[high - 1];

        if (count <= MAX_LEAF_COUNT) {
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            reinterpret_cast<_tree_layout3 *>(
                &tree.group0_index[this_index].split0on_nPrims)
                ->pOffset = low;
        } else {
            tree.group0_index[this_index].nPrims = 0;
            uint64_t mid = low + count / 2;

            uint64_t left = handle_range(low, mid, depth + 1);
            uint64_t right = handle_range(mid, high, depth + 1);

            // Set split offset (offset from this node to right child)
            uint64_t offset = right - this_index;
            reinterpret_cast<_tree_layout2 *>(
                &tree.group0_index[this_index].split0on_nPrims)
                ->offset = offset;
        }
        return this_index;
    };

    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

void verify_IntervalTree(const uint64_t input_index,
                         const _tree_layout0 input) {
    if (input.group0_index[input_index].nPrims == 0u) {
        verify_IntervalTree(input_index + 1u, input);
        verify_IntervalTree(
            input_index +
                (uint64_t)(reinterpret<_tree_layout2>(
                               input.group0_index[input_index].split0on_nPrims)
                               .offset),
            input);
    } else {
        for (uint64_t _idx0 = 0u;
             _idx0 < (uint64_t)(input.group0_index[input_index].nPrims);
             _idx0 += 1u) {
            const float prim =
                input.prims[(uint64_t)(reinterpret<_tree_layout3>(
                                           input.group0_index[input_index]
                                               .split0on_nPrims)
                                           .pOffset) +
                            _idx0];
            if (input.group0_index[input_index].low > prim) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and lb = " << input.group0_index[input_index].low
                          << std::endl;
                abort();
            }
            if (input.group0_index[input_index].high < prim) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and ub = "
                          << input.group0_index[input_index].high << std::endl;
                abort();
            }
        }
    }
}

#define PROFILE 1

template <typename Result, typename Func0, typename Func1, class... Args>
double benchmark_queries(const std::string &benchmark_name,
                         const set<float> &input, const _tree_layout0 &tree,
                         const int k, const int m, Func0 &&f0, Func1 &&f1,
                         const Args &...args) {
    // Run and time query()
    std::vector<Result> query_results;
    int64_t avg_query_time = benchmark_function(
        [&]() { query_results.push_back(f0(args..., input)); }, k, m);

    // Run and time query_fast()
    std::vector<Result> fast_results;
    int64_t avg_fast_time = benchmark_function(
        [&]() { fast_results.push_back(f1(args..., tree)); }, k, m);

    // Verify all results match
    bool all_match = true;
    for (int i = 0; i < k; ++i) {
        if (!(query_results[i] == fast_results[i])) {
            std::cout << "Failed: " << i << std::endl;
            std::cout << "Linear: " << query_results[i].size() << std::endl;
            std::cout << "Tree  : " << fast_results[i].size() << std::endl;
            all_match = false;
            break;
        }
    }
    // std::cout << benchmark_name << " -- ";
    // std::cout << "input size: " << input.size()
    //           << " output size: " << fast_results[0].size() << std::endl;
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
#endif
            return speedup;
        } else {
            std::cout << benchmark_name
                      << " was too fast to measure accurately on input size: "
                      << input.size() << std::endl;
            return static_cast<double>(avg_query_time) / avg_fast_time; // inf
        }
    }
}

double benchmark_range_query(const set<float> &input, const _tree_layout0 &tree,
                             const int k, const int m) {
    float low = -10;
    float high = 10;
#ifndef PROFILE
    std::cout << "Range query, range = [" << low << ", " << high << "]"
              << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("range_query", input, tree, k, m,
                                         range_query, range_query_fast, low,
                                         high);
}

double benchmark_eq_query(const set<float> &input, const _tree_layout0 &tree,
                          const int k, const int m) {
    float value = 42;
#ifndef PROFILE
    std::cout << "Equality query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("eq_query", input, tree, k, m,
                                         eq_query, eq_query_fast, value);
}

double benchmark_abs_query(const set<float> &input, const _tree_layout0 &tree,
                           const int k, const int m) {
    float value = 10.0f;
#ifndef PROFILE
    std::cout << "Abs query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("abs_query", input, tree, k, m,
                                         abs_query, abs_query_fast, value);
}

double benchmark_sqr_query(const set<float> &input, const _tree_layout0 &tree,
                           const int k, const int m) {
    float value = 100.0f;
#ifndef PROFILE
    std::cout << "Sqr query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("sqr_query", input, tree, k, m,
                                         sqr_query, sqr_query_fast, value);
}

double benchmark_round_query(const set<float> &input, const _tree_layout0 &tree,
                             const int k, const int m) {
    float value = 10.0f;
#ifndef PROFILE
    std::cout << "Round query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("round_query", input, tree, k, m,
                                         round_query, round_query_fast, value);
}

double benchmark_poly_query(const set<float> &input, const _tree_layout0 &tree,
                            const int k, const int m) {
    float value = 0.0f;
#ifndef PROFILE
    std::cout << "Poly query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("poly_query", input, tree, k, m,
                                         poly_query, poly_query_fast, value);
}

double benchmark_sqrt_query(const set<float> &input, const _tree_layout0 &tree,
                            const int k, const int m) {
    float value = std::sqrt(10.0f);
#ifndef PROFILE
    std::cout << "Sqrt query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("sqrt_query", input, tree, k, m,
                                         sqrt_query, sqrt_query_fast, value);
}

std::pair<float, float> compute_mean_and_stdev(const std::vector<float> &data) {
    if (data.size() < 2)
        return {0.0f, 0.0f};

    float mean = 0.0f;
    float M2 = 0.0f;
    size_t n = 0;

    // Welford's algorithm
    for (float x : data) {
        ++n;
        float delta = x - mean;
        mean += delta / n;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    return {mean, std::sqrt(M2 / (n - 1))};
}

double benchmark_stddev_query(const set<float> &input,
                              const _tree_layout0 &tree, const int k,
                              const int m) {
    const auto [mean, stddev] = compute_mean_and_stdev(input.data);
    const float stddev3 = stddev * 3;
#ifndef PROFILE
    std::cout << "Stdev query, mean = " << value << " stddev * 3 = " << stddev3
              << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_queries<set<float>>("stddev_query", input, tree, k, m,
                                         stddev_query, stddev_query_fast, mean,
                                         stddev3);
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

// #define EXPORT 1

int main() {
    const int k = 14; // total runs
    const int m = 2;  // number of fastest and slowest to drop

    // std::mt19937 rng(std::random_device{}());
    // For consistent results
    std::mt19937 rng(42);

    std::vector<size_t> test_sizes = {
        1ull << 8,  1ull << 9,  1ull << 10, 1ull << 11, 1ull << 12,
        1ull << 13, 1ull << 14, 1ull << 15, 1ull << 16, 1ull << 17,
        1ull << 18, 1ull << 19, 1ull << 20, 1ull << 21, 1ull << 22,
        1ull << 23, 1ull << 24, 1ull << 25, 1ull << 26, 1ull << 27,
        // 1ull << 28, 1ull << 29, 1ull << 30, 1ull << 31, (1ull << 32) - 1
    };
#if defined(USE_NORMAL)
    std::cout << "normal distribution" << std::endl;
#elif defined(USE_EXPONENTIAL)
    std::cout << "exponential distribution" << std::endl;
#elif defined(USE_LOGNORMAL)
    std::cout << "lognormal distribution" << std::endl;
#elif defined(USE_CAUCHY)
    std::cout << "cauchy distribution" << std::endl;
#elif defined(USE_WEIBULL)
    std::cout << "weibull distribution" << std::endl;
#else
    std::cout << "uniform distribution" << std::endl;
#endif

#ifdef PROFILE
    pretty_print_vector(test_sizes);
    std::cout << std::endl;
    static constexpr int N_BENCHMARKS = 8;
    std::vector<std::pair<std::string, std::vector<double>>> results(
        N_BENCHMARKS);
    results[0].first = "range_query";
    results[0].second.reserve(test_sizes.size());
    results[1].first = "eq_query";
    results[1].second.reserve(test_sizes.size());
    results[2].first = "abs_query";
    results[2].second.reserve(test_sizes.size());
    results[3].first = "sqr_query";
    results[3].second.reserve(test_sizes.size());
    results[4].first = "round_query";
    results[4].second.reserve(test_sizes.size());
    results[5].first = "poly_query";
    results[5].second.reserve(test_sizes.size());
    results[6].first = "sqrt_query";
    results[6].second.reserve(test_sizes.size());
    results[7].first = "stddev_query";
    results[6].second.reserve(test_sizes.size());
#endif
    for (size_t size : test_sizes) {
        // std::cout << size << std::endl;
#ifndef PROFILE
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;
#endif
        // Generate input
        auto input_set = generate_random_set(rng, size);

#ifdef EXPORT
        export_to_csv(input_set,
                      "/Users/ajroot/projects/learn-sql/data/input_" +
                          std::to_string(size) + ".csv");
#endif

        // Build tree
        auto t_build_start = std::chrono::high_resolution_clock::now();
        const auto input_tree = build_tree(input_set);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        int64_t build_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build_end -
                                                                 t_build_start)
                .count();
#ifndef PROFILE
        std::cout << "build_tree() time: " << build_time << " ns\n";
        verify_IntervalTree(0, input_tree);
#endif

#ifdef PROFILE
        results[0].second.push_back(
#endif
            benchmark_range_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[1].second.push_back(
#endif
            benchmark_eq_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[2].second.push_back(
#endif
            benchmark_abs_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[3].second.push_back(
#endif
            benchmark_sqr_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[4].second.push_back(
#endif
            benchmark_round_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[5].second.push_back(
#endif
            benchmark_poly_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[6].second.push_back(
#endif
            benchmark_sqrt_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[7].second.push_back(
#endif
            benchmark_stddev_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;
        std::free(input_tree.prims);
        std::free(input_tree.group0_index);
    }
#ifdef PROFILE
    for (const auto &res : results) {
        std::cout << "(\"" << res.first << "\", ";
        pretty_print_vector(res.second);
        std::cout << ")," << std::endl;
    }
#endif
    return 0;
}
