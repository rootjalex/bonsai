#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>

#include "bonsai_set.h"


inline void flush_cache() {
    constexpr size_t CACHE_FLUSH_SIZE = 50 * 1024 * 1024; // ~50 MB
    static std::vector<uint8_t> buffer(CACHE_FLUSH_SIZE, 1);

    volatile uint8_t sink = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; ++i) {
        sink += buffer[i];  // read to evict other data
        buffer[i] = sink;   // write back
    }
}

template<typename Func>
// k is the number of runs, m is the number of low and high runs to drop.
int64_t benchmark_function(Func&& func, int k, int m) {
    if (2 * m >= k) {
        throw std::invalid_argument("Cannot drop more times than available runs (2 * m >= k)");
    }

    std::vector<int64_t> times;
    times.reserve(k);

    for (int i = 0; i < k; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(); // Run benchmarked function
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    auto begin = times.begin() + m;
    auto end = times.end() - m;

    int64_t sum = std::accumulate(begin, end, int64_t{0});
    return sum / std::distance(begin, end);  // average of middle runs
}

template<typename Func>
// k is the number of runs, m is the number of low and high runs to drop.
auto benchmark_function2(Func&& func, int k, int m) {
    if (2 * m >= k) {
        throw std::invalid_argument("Cannot drop more times than available runs (2 * m >= k)");
    }

    std::vector<int64_t> times;
    times.reserve(k);

    auto start0 = std::chrono::high_resolution_clock::now();
    auto result = func(); // Run benchmarked function
    auto end0 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - start0).count());

    for (int i = 1; i < k; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto temp_result = func(); // Run benchmarked function
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        // temp_result gets deallocated here
    }

    std::sort(times.begin(), times.end());
    auto begin = times.begin() + m;
    auto end = times.end() - m;

    int64_t sum = std::accumulate(begin, end, int64_t{0});
    // Average of the middle runs
    return std::make_tuple(sum / std::distance(begin, end), result);
}

template <bool verbose, typename Result, typename Input, typename Tree,
          typename Func0, typename Func1, class... Args>
double benchmark_1d_queries(const std::string &benchmark_name,
                            const Input &input, const Tree &tree, const int k,
                            const int m, Func0 &&f0, Func1 &&f1,
                            const Args &...args) {
    flush_cache();
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
            if constexpr (requires { query_results[i].size(); }) {
                std::cout << "Linear: " << query_results[i].size() << std::endl;
                std::cout << "Tree  : " << fast_results[i].size() << std::endl;
            } else {
                std::cout << "Linear (value): " << query_results[i]
                          << std::endl;
                std::cout << "Tree   (value): " << fast_results[i] << std::endl;
            }
            all_match = false;
            break;
        }
    }
    // std::cout << benchmark_name << " -- ";
    // std::cout << "input size: " << input.size()
    //           << " output size: " << fast_results[0].size() << std::endl;
    if constexpr (verbose) {
        std::cout << benchmark_name << "() avg time: " << avg_query_time
                  << " ns\n";
        std::cout << benchmark_name << "_fast() avg time: " << avg_fast_time
                  << " ns\n";
    }

    if (!all_match) {
        std::cerr << "ERROR: " << benchmark_name << " results differ! "
                  << input.size() << std::endl;
        std::abort();
    } else {
        if constexpr (verbose) {
            std::cout << "Results match.\n";
        }
        if (avg_fast_time > 0) {
            double speedup =
                static_cast<double>(avg_query_time) / avg_fast_time;
            if constexpr (verbose) {
                std::cout << "Speedup: " << speedup
                          << "x for input size = " << input.size()
                          << " and output size = " << query_results[0].size()
                          << std::endl;
            }
            return speedup;
        } else {
            std::cout << benchmark_name
                      << " was too fast to measure accurately on input size: "
                      << input.size() << std::endl;
            return static_cast<double>(avg_query_time) / avg_fast_time; // inf
        }
    }
}

template <bool verbose, typename Result, typename SingleResult, typename Input0, typename Input1,
          typename Tree0, typename Tree1, typename Func0, typename Func1,
          typename Func2, class... Args>
auto benchmark_join(const std::string &benchmark_name,
                            const Input0 &input0, const Input1 &input1,
                            const Tree0 &tree0, const Tree1 &tree1,
                            const int k, const int m,
                            Func0 &&nested, Func1 &&single,
                            Func2 &&dual, const Args &...args) {
    std::cout << "Benchmarking nested join" << std::endl;
    flush_cache();
    // Run and time nested join
    // std::vector<Result> nested_results;
    // nested_results.reserve(k);
    auto [avg_nested_time, nested_results] = benchmark_function2([&] { return nested(args..., input0, input1); }, k, m);

    std::cout << "Benchmarking single join" << std::endl;
    flush_cache();
    // Run and time single index join
    auto [avg_single_time, single_results] = benchmark_function2([&] { return single(args..., input0, tree1); }, k, m);

    // std::vector<SingleResult> single_results;
    // single_results.reserve(k);
    // int64_t avg_single_time = benchmark_function(
    //     [&]() { single_results.push_back(single(args..., input0, tree1)); }, k, m);

    std::cout << "Benchmarking dual join" << std::endl;
    flush_cache();
    // Run and time dual join
    auto [avg_dual_time, dual_results] = benchmark_function2([&] { return dual(args..., tree0, tree1); }, k, m);

    // std::vector<Result> dual_results;
    // dual_results.reserve(k);
    // int64_t avg_dual_time = benchmark_function(
    //     [&]() { dual_results.push_back(dual(args..., tree0, tree1)); }, k, m);


    // Verify results
    std::cout << "Checking dual tree results\n";
    if (!(nested_results == dual_results)) {
        std::cerr << "Dual join results do not match nested join results for size: "
                  << input0.size() << " " << input1.size();
        if constexpr (requires { dual_results.size(); }) {
            std::cerr << " nested: " << nested_results.size()
                      << " vs. dual: " << dual_results.size();
            // abort();
        } else {
            std::cerr << " nested (value): " << nested_results
                      << " vs. dual (value): " << dual_results;
            // abort();
        }
    }
    std::cout << "Checked dual tree results\n";

    std::cout << "Checking single tree results\n";
    if constexpr (requires { dual_results.size(); }) {
        if (!(nested_results == flatten(single_results))) {
            std::cerr << "Single join results do not match nested join results for size: "
                      << input0.size() << " " << input1.size()
                      << " nested: " << nested_results.size()
                      << " vs. single: " << single_results.size();
            // abort();
        }
    } else {
        if (nested_results != single_results) {
            std::cerr << "Single join results do not match nested join results for size: "
                      << input0.size() << " " << input1.size()
                      << " nested (value): " << nested_results
                      << " vs. single (value): " << single_results;
            // abort();
        }
    }
    std::cout << "Checked single tree results\n";

    if constexpr (verbose) {
        std::cout << "Results match.\n";
        std::cout << benchmark_name << " nested time: " << avg_nested_time
                  << " ns\n";
        std::cout << benchmark_name << " single time: " << avg_single_time
                  << " ns\n";
        std::cout << benchmark_name << " dual time: " << avg_dual_time
                  << " ns\n";
    }

    return std::make_tuple(avg_nested_time, avg_single_time, avg_dual_time);
}
