#pragma once

#include "bonsai_benchmark.h"
#include "bonsai_set.h"
#include "bonsai_tree.h"
#include "bonsai_vector.h"
#include "u24.h"
#include "u56.h"

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

using std::abs;
using std::max;
using std::min;
using std::round;

template <typename T>
T sqr(const T &v) {
    return v * v;
}
