#pragma once

#include <cfenv>
#include <cmath>
#include <functional>
#include <limits>
#include <math.h>
#include <tuple>

template <typename T1, typename T2>
std::tuple<T1, T2> argmin(const std::tuple<T1, T2> &a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(a) < std::get<0>(b)) {
        return a;
    }
    return b;
}

template <typename T1, typename T2>
std::tuple<T1, T2> argmax(const std::tuple<T1, T2> &a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(a) > std::get<0>(b)) {
        return a;
    }
    return b;
}

// Used for std::variant visitor.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

// TODO(cgyurgyik): support vector types as well.
template <typename T>
T next_after(T from, T to) {
    if constexpr (std::is_scalar_v<T>) {
        return std::nextafterf(from, to);
    }
    return nextafter(from, to);
}

// Template function to perform operations with specific rounding modes
template <int RoundingMode, typename T, typename Op>
T directed_operation(T a, T b, Op &&op) {
    if constexpr (RoundingMode == FE_DOWNWARD) {
        return next_after(op(a, b), -std::numeric_limits<T>::max());
    }
    return next_after(op(a, b), std::numeric_limits<T>::max());
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
