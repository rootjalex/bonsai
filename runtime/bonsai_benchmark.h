#pragma once

#include <algorithm>
#include <chrono>
#include <csignal>
#include <functional>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "bonsai_set.h"


inline void flush_cache() {
    constexpr size_t CACHE_FLUSH_SIZE = 50 * 1024 * 1024; // ~50 MB
    static std::vector<uint8_t> buffer(CACHE_FLUSH_SIZE, 1);

    volatile uint8_t sink = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; ++i) {
        sink = sink + buffer[i]; // read to evict other data
        buffer[i] = sink;   // write back
    }
}

static constexpr int64_t timeout_sec = 30; // timeout after 30s
static constexpr int64_t timeout_ns = timeout_sec * 1000000000;

template <typename Func>
auto run_with_timeout(Func &&func) -> std::pair<int64_t, bool> {
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        // Child process: run the function
        func();
        _exit(0); // ensure child exits
    } else {
        // Parent process: wait with timeout
        auto start = std::chrono::high_resolution_clock::now();
        int status;

        while (true) {
            pid_t ret = waitpid(pid, &status, WNOHANG);
            if (ret != 0)
                break; // child finished

            auto now = std::chrono::high_resolution_clock::now();
            int64_t elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(now -
                                                                     start)
                    .count();
            if (elapsed >= timeout_sec * 1'000'000'000LL) {
                // Timeout
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0); // clean up
                return {timeout_ns, false};
            }

            // Busy-wait for ~50ns
            auto busy_start = std::chrono::high_resolution_clock::now();
            while (std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::high_resolution_clock::now() - busy_start)
                       .count() < 50)
                ;
        }

        auto end = std::chrono::high_resolution_clock::now();
        int64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        return {elapsed_ns, true};
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

    uint64_t timeouts = 0;

    for (int i = 0; i < k; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // Run function asynchronously with timeout
        auto fut = std::async(std::launch::async, [&] { func(); });
        if (fut.wait_for(std::chrono::seconds(timeout_sec)) ==
            std::future_status::timeout) {
            if (++timeouts > k) {
                std::cout << "Timeout max reached.\n";
                return timeout_ns;
            }
        }
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

    using ResultT = std::invoke_result_t<Func>;
    ResultT result{}; // dummy, since we cannot return the actual result from a
                      // forked child

    for (int i = 0; i < (k - 1); ++i) {
        auto [elapsed, ok] = run_with_timeout(func);
        if (!ok) {
            std::cout << "Timeout on run " << i << "\n";
            return std::make_tuple(timeout_ns, result);
        }
        times.push_back(elapsed);
    }

    // Now get the result, assume we won't timeout if we haven't yet.
    auto t0 = std::chrono::high_resolution_clock::now();
    result = func();
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    std::sort(times.begin(), times.end());
    auto begin = times.begin() + m;
    auto end = times.end() - m;

    int64_t sum = std::accumulate(begin, end, int64_t{0});
    // Average of the middle runs
    return std::make_tuple(sum / std::distance(begin, end), result);
}

template <typename Func>
// k is the number of runs, m is the number of low and high runs to drop.
auto benchmark_function_notimeout(Func &&func, int k, int m) {
    if (2 * m >= k) {
        throw std::invalid_argument(
            "Cannot drop more times than available runs (2 * m >= k)");
    }

    std::vector<int64_t> times;
    times.reserve(k);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = func();
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    uint64_t timeouts = 0;

    for (int i = 1; i < k; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count());
    }

    std::sort(times.begin(), times.end());
    auto begin = times.begin() + m;
    auto end = times.end() - m;

    int64_t sum = std::accumulate(begin, end, int64_t{0});
    return std::make_tuple(sum / std::distance(begin, end),
                           result); // average of middle runs
}

template <bool verbose, typename Result, typename Input, typename Tree,
          typename Func0, typename Func1, class... Args>
auto benchmark_1d_queries(const std::string &benchmark_name, const Input &input,
                          const Tree &tree, const int k, const int m,
                          Func0 &&linear, Func1 &&indexed,
                          const Args &...args) {

    if (verbose)
        std::cout << "Benchmarking linear query" << std::endl;

    flush_cache();

    // Run and time query() (linear)
    auto [avg_linear_time, linear_results] = benchmark_function_notimeout(
        [&] { return linear(args..., input); }, k, m);

    if (verbose)
        std::cout << "Benchmarking indexed query" << std::endl;

    flush_cache();
    // Run and time query_fast() (indexed)
    auto [avg_indexed_time, indexed_results] = benchmark_function_notimeout(
        [&] { return indexed(args..., tree); }, k, m);

    if ((avg_linear_time != timeout_ns) && (avg_indexed_time != timeout_ns)) {
        if (!(linear_results == indexed_results)) {
            std::cerr << "ERROR: " << benchmark_name;
            if constexpr (requires { input.size(); }) {
                std::cerr << " results differ for input size: " << input.size();
            }
            std::cerr << std::endl;
            if constexpr (requires { linear_results.size(); }) {
                std::cerr << "Linear: " << linear_results.size()
                          << " vs. Indexed: " << indexed_results.size()
                          << std::endl;
            } else {
                std::cerr << "Linear (value): " << linear_results
                          << " vs. Indexed (value): " << indexed_results
                          << std::endl;
            }
            abort();
        }
        }

    if constexpr (verbose) {
        std::cout << "Results match.\n";
        std::cout << benchmark_name << " linear time: " << avg_linear_time
                  << " ns\n";
        std::cout << benchmark_name << " indexed time: " << avg_indexed_time
                  << " ns\n";
    }

    return std::make_tuple(avg_linear_time, avg_indexed_time);
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
    if ((avg_nested_time != timeout_ns) && (avg_dual_time != timeout_ns) &&
        !(nested_results == dual_results)) {
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
        if ((avg_nested_time != timeout_ns) &&
            (avg_single_time != timeout_ns) &&
            !(nested_results == flatten(single_results))) {
            std::cerr << "Single join results do not match nested join results for size: "
                      << input0.size() << " " << input1.size()
                      << " nested: " << nested_results.size()
                      << " vs. single: " << single_results.size();
            // abort();
        }
    } else {
        if ((avg_nested_time != timeout_ns) &&
            (avg_single_time != timeout_ns) &&
            (nested_results != single_results)) {
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
