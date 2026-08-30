#pragma once

#include "bonsai_benchmark.h"
#include "bonsai_misc.h"
#include "bonsai_random.h"
#include "bonsai_set.h"
#include "bonsai_tree.h"
#include "bonsai_vector.h"
#include "u24.h"
#include "u56.h"

// For the std:: names pulled into the global namespace below, and the
// bit_cast/memcpy used by reinterpret(). libc++ provides these transitively;
// libstdc++ does not.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

// What generated code itself spells: std::array for a wide node's children,
// std::atomic for the counters a parallel build increments, std::optional for
// a partial intersection, std::variant for a tree's node, and iostream for
// print. Generated files include only this header, so it owes them these.
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

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
