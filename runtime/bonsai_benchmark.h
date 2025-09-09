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
