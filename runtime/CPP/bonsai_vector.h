#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
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

    vector operator+(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    vector operator-(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    vector operator*(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    vector operator/(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    // Arithmetic operators - vector + scalar
    vector operator+(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] + scalar;
        }
        return result;
    }

    vector operator-(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] - scalar;
        }
        return result;
    }

    vector operator*(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    vector operator/(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    // Compound assignment operators
    vector &operator+=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    vector &operator-=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    vector &operator*=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    vector &operator/=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    // Compound assignment operators with scalars
    vector &operator+=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    vector &operator-=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            data[i] -= scalar;
        }
        return *this;
    }

    vector &operator*=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    vector &operator/=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            data[i] /= scalar;
        }
        return *this;
    }

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

template <typename T, size_t N>
T reduce_add(const vector<T, N> &v) {
    T t = v[0];
    for (int i = 1; i < N; ++i) {
        t += v[i];
    }
    return t;
}

template <typename T, size_t N>
T reduce_mul(const vector<T, N> &v) {
    T t = v[0];
    for (int i = 1; i < N; ++i) {
        t *= v[i];
    }
    return t;
}

template <typename T, size_t N>
T reduce_max(const vector<T, N> &v) {
    T t = v[0];
    for (int i = 1; i < N; ++i) {
        t = std::max<T>(t, v[i]);
    }
    return t;
}

template <typename T, size_t N>
T reduce_min(const vector<T, N> &v) {
    T t = v[0];
    for (int i = 1; i < N; ++i) {
        t = std::min<T>(t, v[i]);
    }
    return t;
}

template <typename T, size_t N>
size_t reduce_idxmax(const vector<T, N> &v) {
    size_t t = 0;
    for (size_t i = 1; i < N; ++i) {
        if (v[i] > v[t]) {
            t = i;
        }
    }
    return t;
}

template <typename T, size_t N>
size_t reduce_idxmin(const vector<T, N> &v) {
    size_t t = 0;
    for (size_t i = 1; i < N; ++i) {
        if (v[i] < v[t]) {
            t = i;
        }
    }
    return t;
}

template <size_t N>
bool reduce_and(const vector<bool, N> &v) {
    bool t = v[0];
    for (int i = 1; i < N; ++i) {
        t &= v[i];
    }
    return t;
}

template <size_t N>
bool reduce_or(const vector<bool, N> &v) {
    bool t = v[0];
    for (int i = v[i]; i < N; ++i) {
        t |= v[i];
    }
    return t;
}