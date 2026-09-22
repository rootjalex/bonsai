#pragma once

#include "bonsai_benchmark.h"
#include "bonsai_buffer.h"
#include "bonsai_parallel.h"
#include "bonsai_set.h"
#include "bonsai_tree.h"
#include "bonsai_vector.h"
#include "u24.h"
#include "u56.h"
#include <cmath>

// For the std:: names pulled into the global namespace below, and the
// bit_cast/memcpy used by reinterpret(). libc++ provides these transitively;
// libstdc++ does not.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

template <typename T, typename U>
__attribute__((always_inline)) T reinterpret(const U &bits) {
    static_assert(sizeof(T) == sizeof(U), "Size mismatch in reinterpret");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<U>,
                  "U must be trivially copyable");

#if __cpp_lib_bit_cast >= 201806L // C++20
    return std::bit_cast<T>(bits);
#else
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
#endif
}

// A value of T that nothing reads (the compiler's `undef`; see ir::Undef).
// C++ has no way to say so, and reading an uninitialized object is undefined
// behavior rather than an unspecified value, so this is a value-initialized T.
template <typename T>
__attribute__((always_inline)) T undef() {
    return T{};
}

using std::abs;
using std::exp;
using std::log;
using std::max;
using std::min;
using std::round;
using std::sqrtf;

template <typename T>
__attribute__((always_inline)) T sqr(const T &v) {
    return v * v;
}

// The integer intrinsics a division by an invariant integer is rewritten
// into (see include/IR/Expr.h, Intrinsic::clz, mulhi and div_multiplier, and
// SSA/InvariantDivision.h). Scalars only: the C++ backend never sees a gang.

// Leading zero bits, and the width for zero.
template <typename T>
__attribute__((always_inline)) T clz(const T &v) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    constexpr int width = std::numeric_limits<U>::digits;
    if (v == 0) {
        return T(width);
    }
    if constexpr (width <= 32) {
        return T(__builtin_clz(U(v)) - (32 - width));
    } else {
        return T(__builtin_clzll(U(v)));
    }
}

// The top half of the full product.
template <typename T>
__attribute__((always_inline)) T mulhi(const T &a, const T &b) {
    static_assert(std::is_integral_v<T>);
    constexpr int width = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    if constexpr (width <= 32) {
        using W = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
        return T((W(a) * W(b)) >> width);
    } else {
        using W = std::conditional_t<std::is_signed_v<T>, __int128,
                                     unsigned __int128>;
        return T((W(a) * W(b)) >> width);
    }
}

// Granlund & Montgomery's multiplier for dividing by `d`: the round-up
// multiplier for an unsigned d, the signed one for |d| otherwise, as
// CodeGen_LLVM::division_multiplier computes it. Defined for every d.
template <typename T>
__attribute__((always_inline)) T div_multiplier(const T &d) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    using W = std::conditional_t<(sizeof(T) > 4), unsigned __int128, uint64_t>;
    constexpr int width = std::numeric_limits<U>::digits;
    if constexpr (std::is_signed_v<T>) {
        U ad = d < 0 ? U(0) - U(d) : U(d);
        if (ad == 0) {
            ad = 1;
        }
        int l = width - int(clz(U(ad - 1)));
        if (l < 1) {
            l = 1;
        }
        const W numerator = W(1) << (width + l - 1);
        return T(U(numerator / W(ad)) + 1);
    } else {
        U ds = d == 0 ? U(1) : U(d);
        const int l = width - int(clz(U(ds - 1)));
        const U two_l = l == width ? U(0) : U(U(1) << l);
        const W numerator = W(U(two_l - ds)) << width;
        return T(U(numerator / W(ds)) + 1);
    }
}

// Temp hack.
using bool3 = vector<bool, 3>;
