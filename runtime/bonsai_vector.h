#pragma once
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

// A vector rounds its lane count up to a power of two, whatever it holds.
//
// That is what the LLVM backend does for `<N x T>`, and what the C ABI then
// hands over in a single SSE register (SSE + SSEUP) rather than one register
// per eightbyte. Reproducing it is what makes a vector survive a call
// boundary, and what puts a field following one inside an element at the
// offset the generated code reads it from. Only a bare native vector member
// gets that classification: pairing it with a T[lanes] in a union demotes the
// second eightbyte to SSE and splits the value across two registers.
//
// Byte vectors are no exception. They stand in for bit-packed fields, and the
// rounding has to be what the layout arithmetic assumes too, rather than the
// two disagreeing about how wide a `u8x7` is.
template <typename T, size_t N>
struct vector_storage {
    static constexpr size_t lanes = round_up_pow2(N);
    typedef T native_vector __attribute__((vector_size(sizeof(T) * lanes)));
    native_vector lanes_;
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

    // Indexing (with bounds checking in debug builds)
    T &operator[](size_t i) {
        if (i >= N)
            throw std::out_of_range("vector[] index out of range");
        return lane_data()[i];
    }

    const T &operator[](size_t i) const {
        if (i >= N)
            throw std::out_of_range("vector[] index out of range");
        return lane_data()[i];
    }

    // Comparison operators. Only the N addressable lanes participate; any
    // padding lane is not part of the value.
    bool operator==(const vector &other) const {
        return std::memcmp(lane_data(), other.lane_data(), sizeof(T) * N) == 0;
    }

    bool operator!=(const vector &other) const { return !(*this == other); }

    // Default constructor = zero
    vector() : data{} {}

    // Allow initialization from a single scalar value
    explicit vector(const T &value) : data{} {
        for (size_t i = 0; i < N; ++i)
            lane_data()[i] = value;
    }

    // Support initializer list if needed
    vector(std::initializer_list<T> init) : data{} {
        size_t i = 0;
        for (T v : init) {
            if (i < N)
                lane_data()[i++] = v;
            else
                break;
        }
    }
};
