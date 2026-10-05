#pragma once

#include "bonsai_benchmark.h"
#include "bonsai_buffer.h"
#include "bonsai_parallel.h"
#include "bonsai_set.h"
#include "bonsai_tree.h"
#include "bonsai_vector.h"
#include "u24.h"
#include "u4.h"
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

// Trailing zero bits, and the width for zero.
template <typename T>
__attribute__((always_inline)) T ctz(const T &v) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    constexpr int width = std::numeric_limits<U>::digits;
    if (v == 0) {
        return T(width);
    }
    if constexpr (width <= 32) {
        return T(__builtin_ctz(U(v)));
    } else {
        return T(__builtin_ctzll(U(v)));
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

// The variable permute and the compress (include/IR/Expr.h,
// Intrinsic::permute and Intrinsic::compress), a lane at a time: C++ has no
// instruction to name for either, and the C++ backend never sees a gang.
template <typename T, size_t N, typename I, size_t M>
__attribute__((always_inline)) vector<T, M>
bonsai_permute(const vector<T, N> &v, const vector<I, M> &indices) {
    vector<T, M> out;
    for (size_t k = 0; k < M; k++) {
        out[k] = v[size_t(indices[k])];
    }
    return out;
}

// The lanes past the ones packed hold `fill`'s -- the vector's own where
// the program named no fill, as Embree's `compact` merges into itself.
template <typename T, size_t N>
__attribute__((always_inline)) vector<T, N>
bonsai_compress(const vector<T, N> &v, const vector<bool, N> &mask,
                const vector<T, N> &fill) {
    vector<T, N> out = fill;
    size_t next = 0;
    for (size_t k = 0; k < N; k++) {
        if (mask[k]) {
            out[next++] = v[k];
        }
    }
    return out;
}

template <typename T, size_t N>
__attribute__((always_inline)) vector<T, N>
bonsai_compress(const vector<T, N> &v, const vector<bool, N> &mask, T fill) {
    vector<T, N> filled;
    for (size_t k = 0; k < N; k++) {
        filled[k] = fill;
    }
    return bonsai_compress(v, mask, filled);
}

template <typename T, size_t N>
__attribute__((always_inline)) vector<T, N>
bonsai_compress(const vector<T, N> &v, const vector<bool, N> &mask) {
    return bonsai_compress(v, mask, v);
}

// The prefetch (Intrinsic::prefetch): the `bytes` at `address` into the
// first-level cache, a 64-byte line at a time, for one address or for each
// lane's address that the mask has on.
template <typename T>
__attribute__((always_inline)) void bonsai_prefetch(const T *address,
                                                    uint32_t bytes,
                                                    bool wanted = true) {
    if (!wanted) {
        return;
    }
    const char *at = reinterpret_cast<const char *>(address);
    for (uint32_t k = 0; k < bytes; k += 64) {
        __builtin_prefetch(at + k, 0, 3);
    }
}

template <typename T, size_t N>
__attribute__((always_inline)) void
bonsai_prefetch(const vector<T *, N> &addresses, uint32_t bytes,
                const vector<bool, N> &mask) {
    for (size_t k = 0; k < N; k++) {
        bonsai_prefetch(addresses[k], bytes, mask[k]);
    }
}

// The same with the addresses as integers: a reference that is a pointer
// with its kind bits in place (a `ptr group`'s), fetched as Embree's
// BVH::prefetch fetches a NodeRef.
__attribute__((always_inline)) inline void
bonsai_prefetch(uint64_t address, uint32_t bytes, bool wanted = true) {
    bonsai_prefetch(reinterpret_cast<const char *>(address), bytes, wanted);
}

template <size_t N>
__attribute__((always_inline)) void
bonsai_prefetch(const vector<uint64_t, N> &addresses, uint32_t bytes,
                const vector<bool, N> &mask) {
    for (size_t k = 0; k < N; k++) {
        bonsai_prefetch(addresses[k], bytes, mask[k]);
    }
}

// Temp hack.
using bool3 = vector<bool, 3>;
