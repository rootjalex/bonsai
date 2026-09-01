#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>

struct uint24_t {
    uint8_t data[3];

    // Default constructor
    uint24_t() : data{0, 0, 0} {}

    // Construct from uint32_t (truncate to 24 bits)
    uint24_t(uint32_t value) {
        data[0] = value & 0xFF;
        data[1] = (value >> 8) & 0xFF;
        data[2] = (value >> 16) & 0xFF;
    }

    // Convert to uint32_t
    operator uint32_t() const {
        return (uint32_t(data[2]) << 16) | (uint32_t(data[1]) << 8) |
               (uint32_t(data[0]));
    }

    // Assignment from uint32_t
    uint24_t &operator=(uint32_t value) {
        data[0] = value & 0xFF;
        data[1] = (value >> 8) & 0xFF;
        data[2] = (value >> 16) & 0xFF;
        return *this;
    }

    // Comparison operators
    bool operator==(const uint24_t &other) const {
        return std::memcmp(data, other.data, 3) == 0;
    }

    bool operator!=(const uint24_t &other) const { return !(*this == other); }
} __attribute__((packed));