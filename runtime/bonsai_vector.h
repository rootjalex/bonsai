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

    union {
        T data[N];
        struct {
            T x;
            T y;
            T z;
        };
    };

    // Allow initialization from a single scalar value
    explicit vector(const T &value) {
        for (size_t i = 0; i < N; ++i)
            data[i] = value;
    }

    // Default constructor = zero
    vector() {
        for (size_t i = 0; i < N; ++i)
            data[i] = T{};
    }

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
        if (a[i] > r) {
            r = a[i];
            p = i;
        }
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

//===--------------------------------------------------------------------===//
// Native vectors
//===--------------------------------------------------------------------===//
//
// Generated code spells a bonsai vector as a native clang vector --
// `T __attribute__((ext_vector_type(N)))` -- because that is the only C++
// type whose size, alignment and argument passing match the `<N x T>` the
// LLVM backend emits for the same value. (The class above is 12 bytes for a
// float3 and arrives in two registers; `<3 x float>` is 16 bytes and arrives
// in one.) So the helpers generated code calls have to exist for native
// vectors too. Arithmetic, comparison and indexing come from the compiler;
// what follows is everything else.
//
// A native vector has no members to hang a lane count off, hence the
// templates and `__builtin_vectorelements`.

template <typename V>
concept native_vector =
    !std::is_class_v<V> && requires { __builtin_vectorelements(V); };

template <native_vector V>
using vector_element_t = std::remove_cvref_t<decltype(std::declval<V>()[0])>;

template <native_vector V>
inline constexpr size_t vector_lanes_v = __builtin_vectorelements(V);

// Elementwise operations.

template <native_vector V>
V max(V a, V b) {
    return __builtin_elementwise_max(a, b);
}

template <native_vector V>
V min(V a, V b) {
    return __builtin_elementwise_min(a, b);
}

template <native_vector V>
V abs(V v) {
    return __builtin_elementwise_abs(v);
}

template <native_vector V>
V ceil(V v) {
    return __builtin_elementwise_ceil(v);
}

template <native_vector V>
V floor(V v) {
    return __builtin_elementwise_floor(v);
}

template <native_vector V>
V round(V v) {
    return __builtin_elementwise_round(v);
}

// Lane selection. A comparison of native vectors yields a vector of integers
// (all bits set where true), and the conditional operator on vectors selects
// per lane, so the mask type is whatever the comparison produced rather than
// a vector of bool.
template <native_vector M, native_vector V>
V select(M mask, V a, V b) {
    return mask ? a : b;
}

// Reductions. Written as loops rather than __builtin_reduce_*, which does not
// cover floating point addition and multiplication; clang turns these back
// into reduction intrinsics anyway.

template <native_vector V>
vector_element_t<V> reduce_add(V v) {
    vector_element_t<V> t = v[0];
    for (size_t i = 1; i < vector_lanes_v<V>; ++i) {
        t += v[i];
    }
    return t;
}

template <native_vector V>
vector_element_t<V> reduce_mul(V v) {
    vector_element_t<V> t = v[0];
    for (size_t i = 1; i < vector_lanes_v<V>; ++i) {
        t *= v[i];
    }
    return t;
}

template <native_vector V>
vector_element_t<V> reduce_max(V v) {
    return __builtin_reduce_max(v);
}

template <native_vector V>
vector_element_t<V> reduce_min(V v) {
    return __builtin_reduce_min(v);
}

template <native_vector V>
bool reduce_and(V v) {
    for (size_t i = 0; i < vector_lanes_v<V>; ++i) {
        if (!v[i]) {
            return false;
        }
    }
    return true;
}

template <native_vector V>
bool reduce_or(V v) {
    for (size_t i = 0; i < vector_lanes_v<V>; ++i) {
        if (v[i]) {
            return true;
        }
    }
    return false;
}

template <native_vector V>
size_t reduce_idxmax(V v) {
    size_t t = 0;
    for (size_t i = 1; i < vector_lanes_v<V>; ++i) {
        if (v[i] > v[t]) {
            t = i;
        }
    }
    return t;
}

template <native_vector V>
size_t reduce_idxmin(V v) {
    size_t t = 0;
    for (size_t i = 1; i < vector_lanes_v<V>; ++i) {
        if (v[i] < v[t]) {
            t = i;
        }
    }
    return t;
}

template <native_vector V>
size_t argmax(V v) {
    return reduce_idxmax(v);
}

// Geometric helpers, as used by the standard library's distance and
// intersection routines.

template <native_vector V>
vector_element_t<V> dot(V a, V b) {
    return reduce_add(a * b);
}

template <native_vector V>
vector_element_t<V> norm(V v) {
    return std::sqrt(dot(v, v));
}

template <native_vector V>
V normalize(V v) {
    return v / norm(v);
}

template <native_vector V>
    requires(vector_lanes_v<V> == 3)
V cross(V a, V b) {
    return V{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x};
}
