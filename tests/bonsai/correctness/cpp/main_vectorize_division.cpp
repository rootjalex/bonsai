#include "vectorize_division.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

// Used in vectorize_division.bonsai.
//
// Each gang's lanes are compared with the same digit reversal done here by
// the C++ compiler's division, and the unsigned 64-bit gang is printed.
namespace {

template <typename T>
T reverse(T n, T b) {
    T v = n;
    T r = 0;
    for (int k = 0; k < 6; k++) {
        const T next = v / b;
        r = r * b + (v - next * b);
        v = next;
    }
    return r;
}

template <typename T, typename F>
bool same_as_scalar(const std::array<T, 8> &ns, const std::array<T, 8> &bs,
                    F &&all, std::array<T, 8> &out) {
    all(ns, bs, out);
    bool same = true;
    for (size_t i = 0; i < ns.size(); i++) {
        same = same && reverse(ns[i], bs[i]) == out[i];
    }
    return same;
}

} // namespace

int main() {
    const std::array<uint64_t, 8> u64_ns = {
        0u, 1000003u, 4294967295u, 4294967296u, 9223372036854775807u,
        18446744073709551615u, 7u, 123456789012345678u};
    const std::array<uint64_t, 8> u64_bs = {
        3u, 1u, 2u, 7u, 641u, 4294967297u, 9223372036854775809u,
        18446744073709551615u};
    std::array<uint64_t, 8> u64_out{};
    const bool u64_same =
        same_as_scalar(u64_ns, u64_bs, reverse_all_u64, u64_out);
    for (size_t i = 0; i < u64_out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << u64_out[i];
    }
    std::cout << '\n';

    const std::array<int64_t, 8> i64_ns = {
        0, -1000003, 2147483647, -4294967296, 9223372036854775807,
        -9223372036854775807, 7, -123456789012345678};
    const std::array<int64_t, 8> i64_bs = {
        3, -1, 2, -7, 641, -4294967297, std::numeric_limits<int64_t>::min(),
        1};
    std::array<int64_t, 8> i64_out{};
    const bool i64_same =
        same_as_scalar(i64_ns, i64_bs, reverse_all_i64, i64_out);

    const std::array<uint32_t, 8> u32_ns = {0u,          1000003u, 65535u,
                                            65536u,      2147483647u,
                                            4294967295u, 7u,       12345678u};
    const std::array<uint32_t, 8> u32_bs = {3u,   1u,          2u,
                                            7u,   641u,        65537u,
                                            2147483649u, 4294967295u};
    std::array<uint32_t, 8> u32_out{};
    const bool u32_same =
        same_as_scalar(u32_ns, u32_bs, reverse_all_u32, u32_out);

    const std::array<int32_t, 8> i32_ns = {0,          -1000003, 65535,
                                           -65536,     2147483647,
                                           -2147483647, 7,       -12345678};
    const std::array<int32_t, 8> i32_bs = {
        3, -1, 2, -7, 641, -65537, std::numeric_limits<int32_t>::min(), 1};
    std::array<int32_t, 8> i32_out{};
    const bool i32_same =
        same_as_scalar(i32_ns, i32_bs, reverse_all_i32, i32_out);

    std::cout << (u64_same ? "u64 same as scalar" : "u64 DIFFERS from scalar")
              << '\n'
              << (i64_same ? "i64 same as scalar" : "i64 DIFFERS from scalar")
              << '\n'
              << (u32_same ? "u32 same as scalar" : "u32 DIFFERS from scalar")
              << '\n'
              << (i32_same ? "i32 same as scalar" : "i32 DIFFERS from scalar")
              << '\n';
}
