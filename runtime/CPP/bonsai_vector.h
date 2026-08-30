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

namespace bonsai_detail {

constexpr size_t round_up_pow2(size_t n) {
    size_t p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

// Two different roles share vector<T, N>, and they want different layouts.
//
// A vector of a multi-byte scalar is a value. The LLVM backend represents it
// as <N x T>, which rounds the lane count up to a power of two, and which the
// C ABI hands over in a single SSE register (SSE + SSEUP) rather than one
// register per eightbyte. Reproducing that is what makes such a vector survive
// a call boundary, and what puts a field following it inside an element at the
// offset the generated code reads it from. Only a bare native vector member
// gets that classification: pairing it with a T[lanes] in a union demotes the
// second eightbyte to SSE and splits the value across two registers.
template <typename T, size_t N,
          bool Native = (std::is_arithmetic_v<T> && sizeof(T) > 1)>
struct vector_storage {
    static constexpr size_t lanes = round_up_pow2(N);
    typedef T native_vector __attribute__((vector_size(sizeof(T) * lanes)));
    native_vector lanes_;
};

// A vector of bytes is instead an opaque blob standing in for a bit-packed
// field, and the generated layout arithmetic assumes it occupies exactly N
// bytes. Padding those would move every field after them. A vector of vectors
// takes this layout too, since vector_size only accepts a scalar element.
template <typename T, size_t N>
struct vector_storage<T, N, false> {
    static constexpr size_t lanes = N;
    T lanes_[N];
};

} // namespace bonsai_detail

template <typename T, size_t N>
struct vector {
    static_assert(N > 0, "vector<N>: N must be > 0");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");

    using storage = bonsai_detail::vector_storage<T, N>;
    static constexpr size_t lanes = storage::lanes;

    storage data;

    T *lane_data() { return reinterpret_cast<T *>(&data); }
    const T *lane_data() const { return reinterpret_cast<const T *>(&data); }

    vector() : data{} {}

    // Allow initialization from a single scalar value
    explicit vector(const T &value) : data{} {
        for (size_t i = 0; i < N; ++i) {
            lane_data()[i] = value;
        }
    }

    // A one-element list broadcasts; a longer one fills lane by lane.
    vector(std::initializer_list<T> init) : data{} {
        if (init.size() == 1) {
            const T v = *init.begin();
            for (size_t i = 0; i < N; ++i) {
                lane_data()[i] = v;
            }
            return;
        }
        size_t i = 0;
        for (T v : init) {
            if (i < N) {
                lane_data()[i++] = v;
            } else {
                break;
            }
        }
    }

    // cast
    template <typename U>
    explicit vector(const vector<U, N> &other) : data{} {
        for (size_t i = 0; i < N; ++i) {
            lane_data()[i] = static_cast<T>(other[i]);
        }
    }

    T &operator[](size_t i) {
        assert(i < N);
        return lane_data()[i];
    }

    const T &operator[](size_t i) const {
        assert(i < N);
        return lane_data()[i];
    }

    vector operator-() const {
        vector<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = -(*this)[i];
        }
        return result;
    }

    vector operator!() const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = !(*this)[i];
        }
        return result;
    }

    // Comparison operators
    vector<bool, N> operator==(const vector &other) const {
        // return std::memcmp(data, other.data, sizeof(data)) == 0;
        vector<bool, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] == other[i];
        }
        return result;
    }

    vector<bool, N> operator<(const vector &other) const {
        vector<bool, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] < other[i];
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
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    vector operator-(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }

    vector operator*(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] * other[i];
        }
        return result;
    }

    vector operator/(const vector &other) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] / other[i];
        }
        return result;
    }

    // Arithmetic operators - vector + scalar
    vector operator+(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] + scalar;
        }
        return result;
    }

    vector operator-(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] - scalar;
        }
        return result;
    }

    vector operator*(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] * scalar;
        }
        return result;
    }

    vector operator/(const T &scalar) const {
        vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] / scalar;
        }
        return result;
    }

    // Compound assignment operators
    vector &operator+=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    vector &operator-=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] -= other[i];
        }
        return *this;
    }

    vector &operator*=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] *= other[i];
        }
        return *this;
    }

    vector &operator/=(const vector &other) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] /= other[i];
        }
        return *this;
    }

    // Compound assignment operators with scalars
    vector &operator+=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] += scalar;
        }
        return *this;
    }

    vector &operator-=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] -= scalar;
        }
        return *this;
    }

    vector &operator*=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] *= scalar;
        }
        return *this;
    }

    vector &operator/=(const T &scalar) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] /= scalar;
        }
        return *this;
    }
};


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
        result[i] = mask[i] ? if_true[i] : if_false[i];
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
        result[i] = std::abs(v[i]);
    }
    return result;
}

template <typename T, size_t N>
vector<T, N> shuffle(const vector<T, N> &v,
                     std::initializer_list<size_t> indices) {
    vector<T, N> result;
    size_t i = 0;
    for (size_t idx : indices) {
        result[i++] = v[idx];
    }
    return result;
}

template <size_t N>
vector<bool, N> operator&(const vector<bool, N> &a, const vector<bool, N> &b) {
    vector<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] & b[i];
    }
    return result;
}

template <size_t N>
vector<bool, N> operator|(const vector<bool, N> &a, const vector<bool, N> &b) {
    vector<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] | b[i];
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

