#pragma once

#include <algorithm>
#include <random>
#include <type_traits>

template <typename T>
T random_uniform(std::mt19937 &rng, T low, T high) {
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(low, high);
        return dist(rng);
    } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(low, high);
        return dist(rng);
    } else {
        static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
    }
}

template<typename T>
T random_normal(std::mt19937 &rng, const T mean, const T stddev) {
    std::normal_distribution<T> dist(mean, stddev);
    return dist(rng);
}

template<typename T>
T random_exponential(std::mt19937 &rng, const T lam) {
    std::exponential_distribution<T> dist(lam);
    return dist(rng);
}


template<typename T>
T random_lognormal(std::mt19937 &rng, const T mean, const T stddev) {
    std::lognormal_distribution<T> dist(mean, stddev);
    return dist(rng);
}

