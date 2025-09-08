#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

template <typename T, size_t N>
struct vector {
    static_assert(N > 0, "vector<N>: N must be > 0");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");

    T data[N] = {}; // Zero-initialize

    // Indexing (with bounds checking in debug builds)
    T &operator[](size_t i) {
        if (i >= N)
            throw std::out_of_range("vector[] index out of range");
        return data[i];
    }

    const T &operator[](size_t i) const {
        if (i >= N)
            throw std::out_of_range("vector[] index out of range");
        return data[i];
    }

    // Comparison operators
    bool operator==(const vector &other) const {
        return std::memcmp(data, other.data, sizeof(data)) == 0;
    }

    bool operator!=(const vector &other) const { return !(*this == other); }

    // Allow initialization from a single scalar value
    explicit vector(const T &value) {
        for (size_t i = 0; i < N; ++i)
            data[i] = value;
    }

    // Default constructor = zero
    vector() = default;

    // Support initializer list if needed
    vector(std::initializer_list<T> init) {
        size_t i = 0;
        for (T v : init) {
            if (i < N)
                data[i++] = v;
            else
                break;
        }
    }
} __attribute__((packed));
