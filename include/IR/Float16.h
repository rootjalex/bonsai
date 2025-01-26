#pragma once

#include <cstdint>
#include <cstring>

namespace bonsai {
namespace ir {

struct float16_t {

    static const int mantissa_bits = 10;
    static const uint16_t sign_mask = 0x8000;
    static const uint16_t exponent_mask = 0x7c00;
    static const uint16_t mantissa_mask = 0x03ff;

    // TODO: copy over Halide's implementation!
  private:
    // The raw bits.
    // uint16_t data = 0;
};

double cast_to_float16(const double value);

/** An aggressive form of reinterpret cast used for correct type-punning. */
template <typename DstType, typename SrcType>
DstType reinterpret_bits(const SrcType &src) {
    static_assert(sizeof(SrcType) == sizeof(DstType),
                  "Types must be same size");
    DstType dst;
    memcpy(&dst, &src, sizeof(SrcType));
    return dst;
}

} // namespace ir
} // namespace bonsai