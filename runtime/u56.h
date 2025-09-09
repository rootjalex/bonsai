#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>

struct uint56_t {
    uint8_t data[7] = {};

    // Default constructor
    uint56_t() {}

    // Construct from uint64_t (truncate to 56 bits)
    uint56_t(uint64_t value) {
        data[0] = value & 0xFF;
        data[1] = (value >> 8) & 0xFF;
        data[2] = (value >> 16) & 0xFF;
        data[3] = (value >> 24) & 0xFF;
        data[4] = (value >> 32) & 0xFF;
        data[5] = (value >> 40) & 0xFF;
        data[6] = (value >> 48) & 0xFF;
    }

    // Convert to uint64_t
    operator uint64_t() const {
        return (uint64_t(data[6]) << 48) |
               (uint64_t(data[5]) << 40) |
               (uint64_t(data[4]) << 32) |
               (uint64_t(data[3]) << 24) |
               (uint64_t(data[2]) << 16) |
               (uint64_t(data[1]) << 8)  |
               (uint64_t(data[0]));
    }

    // Assignment from uint64_t
    uint56_t& operator=(uint64_t value) {
        data[0] = value & 0xFF;
        data[1] = (value >> 8) & 0xFF;
        data[2] = (value >> 16) & 0xFF;
        data[3] = (value >> 24) & 0xFF;
        data[4] = (value >> 32) & 0xFF;
        data[5] = (value >> 40) & 0xFF;
        data[6] = (value >> 48) & 0xFF;
        return *this;
    }

    // Comparison operators
    bool operator==(const uint56_t& other) const {
        return std::memcmp(data, other.data, 7) == 0;
    }

    bool operator!=(const uint56_t& other) const {
        return !(*this == other);
    }
} __attribute__((packed));