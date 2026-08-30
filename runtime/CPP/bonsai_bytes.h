#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

template <int N>
struct bytes {
    static_assert(N > 0 && N <= 8, "bytes<N>: N must be in range 1..8");

    uint8_t data[N] = {}; // Zero-initialize

    // Indexing
    uint8_t &operator[](size_t i) {
        if (i >= N)
            throw std::out_of_range("bytes[] index out of range");
        return data[i];
    }

    const uint8_t &operator[](size_t i) const {
        if (i >= N)
            throw std::out_of_range("bytes[] index out of range");
        return data[i];
    }

    // Construct from integer (up to 64 bits)
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T> &&
                                                      sizeof(T) <= 8>>
    explicit bytes(T value) {
        static_assert(std::is_unsigned_v<T>, "Only unsigned types supported");
        for (int i = 0; i < N; ++i)
            data[i] = (value >> (8 * i)) & 0xFF;
    }

    // Convert to integer
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T> &&
                                                      sizeof(T) <= 8>>
    explicit operator T() const {
        static_assert(std::is_unsigned_v<T>, "Only unsigned types supported");
        T result = 0;
        for (int i = 0; i < N; ++i)
            result |= (T(data[i]) << (8 * i));
        return result;
    }

    // Assignment from integer
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T> &&
                                                      sizeof(T) <= 8>>
    bytes &operator=(T value) {
        static_assert(std::is_unsigned_v<T>, "Only unsigned types supported");
        for (int i = 0; i < N; ++i)
            data[i] = (value >> (8 * i)) & 0xFF;
        return *this;
    }

    // Equality
    bool operator==(const bytes &other) const {
        return std::memcmp(data, other.data, N) == 0;
    }

    bool operator!=(const bytes &other) const { return !(*this == other); }
} __attribute__((packed));
