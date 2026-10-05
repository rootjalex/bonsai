#pragma once
// A 4-bit unsigned integer in a byte, as u24.h and u56.h give the C++
// backend the widths C++ has no type for: what a four-lane mask is
// reinterpreted as (its lanes as the low four bits of one byte, which is
// how clang stores a vector of four bools), so that a lane can be counted
// to with `ctz` and cleared with `x & (x - 1)` on an ordinary word once the
// four bits are widened to it. The conversions mask to four bits, so the
// byte's upper bits, which the vector of bools leaves unspecified, never
// show.
#include <cstdint>

struct uint4_t {
    uint8_t data;
    uint4_t() : data(0) {}
    uint4_t(uint32_t value) : data(uint8_t(value & 0xF)) {}
    operator uint32_t() const { return uint32_t(data & 0xF); }
    uint4_t &operator=(uint32_t value) {
        data = uint8_t(value & 0xF);
        return *this;
    }
    bool operator==(const uint4_t &other) const {
        return (data & 0xF) == (other.data & 0xF);
    }
    bool operator!=(const uint4_t &other) const { return !(*this == other); }
};
static_assert(sizeof(uint4_t) == 1, "a 4-bit integer is one byte");
