#pragma once

#include <cassert>
#include <cfenv>
#include <cmath>
#include <functional>
#include <limits>
#include <math.h>
#include <tuple>

inline int frexpf(float f) {
    int x;
    frexpf(f, &x);
    return x;
}

// [M, N]
template <int M, int N, typename T>
inline T slice(T value) {
    static_assert(std::is_unsigned_v<T>);
    constexpr int bits = std::numeric_limits<T>::digits;
    static_assert(M >= 0 && N >= 0 && M < N && N < bits);
    constexpr int width = N - M + 1;
    constexpr T mask = (width == bits) ? ~T{0} : (T{1} << width) - 1;
    return (value >> M) & mask;
}

// [M, N]
template <typename T>
inline T slice(int M, int N, T value) {
    static_assert(std::is_unsigned_v<T>);
    const int bits = std::numeric_limits<T>::digits;
    int width = N - M + 1;
    T mask = (width == bits) ? ~T{0} : (T{1} << width) - 1;
    return (value >> M) & mask;
}

// Forward declaration for vector type.
template <typename T, std::size_t N>
struct vector;

// Type traits to detect if something is a vector type.
template <typename T>
struct is_vector : std::false_type {};

template <typename T, std::size_t N>
struct is_vector<vector<T, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

// Used for std::variant visitor.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Extract element type from vector or return T if scalar.
template <typename T>
struct element_type {
    using type = T;
};

template <typename T, std::size_t N>
struct element_type<vector<T, N>> {
    using type = T;
};

template <typename T>
using element_type_t = typename element_type<T>::type;

template <typename T, size_t N>
vector<T, N> next_after_element_wise(const vector<T, N> &from, T to) {
    vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = std::nextafter(from[i], to);
    }
    return result;
}

template <typename T>
T next_after(T from, T to) {
    return std::nextafterf(from, to);
}

// Template function to perform operations with specific rounding modes.
template <int RoundingMode, typename T, typename Op>
T directed_operation(T a, T b, Op &&op) {
    using E = element_type_t<T>;
    constexpr E MAX = std::numeric_limits<E>::max();
    T result = op(a, b);

    if constexpr (RoundingMode == FE_DOWNWARD) {
        if constexpr (is_vector_v<T>) {
            return next_after_element_wise(result, -MAX);
        } else {
            return std::nextafter(result, -MAX);
        }
    }
    if constexpr (is_vector_v<T>) {
        return next_after_element_wise(result, MAX);
    } else {
        return std::nextafter(result, MAX);
    }
}

template <typename T>
T fadd_rd(T a, T b) {
    return directed_operation<FE_DOWNWARD>(a, b, std::plus<T>{});
}

template <typename T>
T fsub_ru(T a, T b) {
    return directed_operation<FE_UPWARD>(a, b, std::minus<T>{});
}

template <typename T>
T fsub_rd(T a, T b) {
    return directed_operation<FE_DOWNWARD>(a, b, std::minus<T>{});
}

template <typename T>
T fmul_ru(T a, T b) {
    return directed_operation<FE_UPWARD>(a, b, std::multiplies<T>{});
}

template <typename T>
T fmul_rd(T a, T b) {
    return directed_operation<FE_DOWNWARD>(a, b, std::multiplies<T>{});
}

template <typename T>
T fdiv_ru(T a, T b) {
    return directed_operation<FE_UPWARD>(a, b, std::divides<T>{});
}

template <typename T>
T fdiv_rd(T a, T b) {
    return directed_operation<FE_DOWNWARD>(a, b, std::divides<T>{});
}

template <typename T>
T frcp_ru(T x) {
    return directed_operation<FE_UPWARD>(T{1}, x, std::divides<T>{});
}

template <typename T>
T frcp_rd(T x) {
    return directed_operation<FE_DOWNWARD>(T{1}, x, std::divides<T>{});
}
