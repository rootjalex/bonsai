#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>

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

template <bool verbose, typename Result, typename Input, typename Tree,
          typename Func0, typename Func1, class... Args>
double benchmark_1d_queries(const std::string &benchmark_name,
                            const Input &input, const Tree &tree, const int k,
                            const int m, Func0 &&f0, Func1 &&f1,
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

template <bool verbose, typename Result, typename Input0, typename Input1,
          typename Tree0, typename Tree1, typename Func0, typename Func1,
          class... Args>
double benchmark_join(const std::string &benchmark_name,
                            const Input0 &input0, const Input1 &input1,
                            const Tree0 &tree0, const Tree1 &tree1,
                            const int k, const int m,
                            Func0 &&f0, Func1 &&f1,
                            const Args &...args) {
    // Run and time query()
    std::vector<Result> query_results;
    int64_t avg_query_time = benchmark_function(
        [&]() { query_results.push_back(f0(args..., input0, input1)); }, k, m);

    // Run and time query_fast()
    std::vector<Result> fast_results;
    int64_t avg_fast_time = benchmark_function(
        [&]() { fast_results.push_back(f1(args..., tree0, tree1)); }, k, m);

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
                  << input0.size() << " x " << input1.size() << std::endl;
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
                          << "x for input size = " << input0.size() << " x " << input1.size()
                          << " and output size = " << query_results[0].size()
                          << std::endl;
            }
            return speedup;
        } else {
            std::cout << benchmark_name
                      << " was too fast to measure accurately on input size: "
                      << input0.size() << " x " << input1.size() << std::endl;
            return static_cast<double>(avg_query_time) / avg_fast_time; // inf
        }
    }
}
