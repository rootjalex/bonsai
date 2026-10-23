#pragma once
// A 2-bit unsigned integer in a byte, as u4.h holds four bits: what a
// two-lane mask -- a binary tree's two children's tests -- is reinterpreted
// as (its lanes as the low two bits of one byte, which is how clang stores a
// vector of two bools), so that a lane can be counted to with `ctz` and
// cleared with `x & (x - 1)` on an ordinary word once the two bits are
// widened to it. The conversions mask to two bits, so the byte's upper bits,
// which the vector of bools leaves unspecified, never show.
#include <cstdint>

struct uint2_t {
    uint8_t data;
    uint2_t() : data(0) {}
    uint2_t(uint32_t value) : data(uint8_t(value & 0x3)) {}
    operator uint32_t() const { return uint32_t(data & 0x3); }
    uint2_t &operator=(uint32_t value) {
        data = uint8_t(value & 0x3);
        return *this;
    }
    bool operator==(const uint2_t &other) const {
        return (data & 0x3) == (other.data & 0x3);
    }
    bool operator!=(const uint2_t &other) const { return !(*this == other); }
};
static_assert(sizeof(uint2_t) == 1, "a 2-bit integer is one byte");
