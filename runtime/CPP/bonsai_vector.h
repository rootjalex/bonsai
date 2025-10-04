#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
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

    // Allow initialization from a single scalar value
    explicit vector(const T &value) {
        for (size_t i = 0; i < N; ++i)
            data[i] = value;
    }

    // Default constructor = zero
    vector() = default;

    // TODO(cgyurgyik): this leads to subtle bugs, e.g.,
    // `vec3_float{1}` will be `{1, 0, 0}`. Remove this.
    vector(std::initializer_list<T> list) {
        if (list.size() == 0) {
            for (int i = 0; i < N; ++i) {
                data[i] = T{};
            }
            return;
        }
        if (list.size() == 1) {
            const T v = *list.begin();
            for (int i = 0; i < N; ++i) {
                data[i] = v;
            }
            return;
        }
        size_t i = 0;
        for (T v : list) {
            if (i < N)
                data[i++] = v;
            else
                break;
        }
    }

    // cast
    template <typename U>
    explicit vector(const vector<U, N> &other) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<T>(other[i]);
        }
    }

    // Indexing (with bounds checking in debug builds)
    T &operator[](size_t i) { return *reinterpret_cast<T *>(&data[i]); }

    const T &operator[](size_t i) const {
        return *reinterpret_cast<const T *>(&data[i]);
    }

    vector operator-() const {
        vector<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = -data[i];
        }
        return result;
    }

    vector operator!() const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = !data[i];
        }
        return result;
    }

    // Comparison operators
    vector<bool, N> operator==(const vector &other) const {
        // return std::memcmp(data, other.data, sizeof(data)) == 0;
        vector<bool, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] == other.data[i];
        }
        return result;
    }

    vector<bool, N> operator<(const vector &other) const {
        vector<bool, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] < other.data[i];
        }
        return result;
    }

    vector<bool, N> operator!=(const vector &other) const {
        return !(*this == other);
    }

    vector<bool, N> operator>=(const vector &other) const {
        return !((*this) < other);
    }

    vector<bool, N> operator>(const vector &other) const {
        return other < (*this);
    }

    vector<bool, N> operator<=(const vector &other) const {
        return !(other < (*this));
    }

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

} __attribute__((packed));

template <typename T, size_t N>
vector<T, N> max(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = std::max(a[i], b[i]);
    }
    return r;
}

template <typename T, size_t N>
vector<T, N> min(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = std::min(a[i], b[i]);
    }
    return r;
}

template <typename T, size_t N>
vector<T, N> ceil(const vector<T, N> &in) {
    vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = std::ceil(in[i]);
    }
    return r;
}

template <typename T, size_t N>
vector<T, N> floor(const vector<T, N> &in) {
    vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = std::floor(in[i]);
    }
    return r;
}

template <typename T, size_t N>
vector<T, N> round(const vector<T, N> &in) {
    vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = std::round(in[i]);
    }
    return r;
}

template <typename T, size_t N>
size_t argmax(const vector<T, N> &a) {
    size_t p = 0;
    T r = a[0];
    for (int i = 1; i < N; ++i) {
        if (r <= a[i]) {
            continue;
        }
        p = i;
        r = a[i];
    }
    return p;
}

template <typename T, size_t N>
static vector<T, N> select(const vector<bool, N> &mask,
                           const vector<T, N> &if_true,
                           const vector<T, N> &if_false) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result.data[i] = mask.data[i] ? if_true.data[i] : if_false.data[i];
    }
    return result;
}

template <typename T, size_t N>
T dot(const vector<T, N> &a, const vector<T, N> &b) {
    T result = a[0] * b[0];
    for (size_t i = 1; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template <typename T, size_t N>
T norm(const vector<T, N> &v) {
    return std::sqrt(dot(v, v));
}

template <typename T, size_t N>
vector<T, N> normalize(const vector<T, N> &v) {
    return v / norm(v);
}

template <typename T>
vector<T, 3> cross(const vector<T, 3> &a, const vector<T, 3> &b) {
    return vector<T, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
}

template <typename T, size_t N>
vector<T, N> abs(const vector<T, N> &v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result.data[i] = std::abs(v[i]);
    }
    return result;
}

template <typename T, size_t N>
vector<T, N> shuffle(const vector<T, N> &v,
                     std::initializer_list<size_t> indices) {
    vector<T, N> result;
    size_t i = 0;
    for (size_t idx : indices) {
        result.data[i++] = v[idx];
    }
    return result;
}

template <size_t N>
vector<bool, N> operator&(const vector<bool, N> &a, const vector<bool, N> &b) {
    vector<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
        result.data[i] = a[i] & b[i];
    }
    return result;
}

template <size_t N>
vector<bool, N> operator|(const vector<bool, N> &a, const vector<bool, N> &b) {
    vector<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
        result.data[i] = a[i] | b[i];
    }
    return result;
}

// ################################################
// Reductions
// ################################################

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
    for (int i = 1; i < N; ++i) {
        t |= v[i];
    }
    return t;
}

static_assert(sizeof(vector<float, 3>) == sizeof(float) * 3);
