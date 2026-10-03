// Used in permute.bonsai.
//
// Every lane of every shape against the definition: lane k of a permute is
// the lane of the vector its index names, and a compress holds the lanes
// the mask had on, in their order, then its fill -- or, with no fill
// named, nothing in particular, and only the packed lanes are checked. The
// indices are a fixed scramble with repeats, so that a lane taken twice and
// a lane taken never both occur, and the masks have hits in the first, the
// last and the middle lanes, so that the packing moves lanes by every
// distance.
#include "permute.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

// `got` has the indices' lanes, which may be fewer than `v` has.
template <typename V, typename I, typename G>
void check_permute(const char *what, const V &v, const I &i, const G &got,
                   int lanes) {
    for (int k = 0; k < lanes; k++) {
        const auto want = v[int(i[k])];
        if (got[k] != want) {
            std::cout << what << ": lane " << k << " is " << double(got[k])
                      << ", wanted lane " << int(i[k]) << " = "
                      << double(want) << " -- WRONG\n";
            failures++;
        }
    }
    std::cout << what << " ok\n";
}

// `zeroed` says the compress was given a fill of zero, so the lanes past the
// packed ones are checked for it; otherwise they may hold anything.
template <typename V, typename M>
void check_compress(const char *what, const V &v, const M &m, const V &got,
                    int lanes, bool zeroed = true) {
    int next = 0;
    for (int k = 0; k < lanes; k++) {
        if (m[k] != 0) {
            if (got[next] != v[k]) {
                std::cout << what << ": slot " << next << " is "
                          << double(got[next]) << ", wanted lane " << k
                          << " = " << double(v[k]) << " -- WRONG\n";
                failures++;
            }
            next++;
        }
    }
    for (int k = next; zeroed && k < lanes; k++) {
        if (got[k] != 0) {
            std::cout << what << ": slot " << k << " past the " << next
                      << " packed is " << double(got[k])
                      << ", wanted zero -- WRONG\n";
            failures++;
        }
    }
    std::cout << what << " ok (" << next << " packed)\n";
}

} // namespace

int main() {
    // A scramble of eight with a repeat (lane 5 twice, lane 2 never).
    const int scramble8[8] = {6, 1, 7, 5, 0, 3, 5, 4};
    const int scramble16[16] = {9, 0, 15, 3, 3, 12, 7, 1, 14, 2, 11, 6, 8, 5, 10, 4};
    const int scramble4[4] = {3, 0, 0, 2};
    const int scramble2[2] = {1, 0};
    const int mask8[8] = {1, 0, 0, 1, 1, 0, 0, 1};
    const int mask16[16] = {0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1};

    {
        float8 v;
        uint32_t8 i, m;
        for (int k = 0; k < 8; k++) {
            v[k] = 1.5f * float(k) - 2.0f;
            i[k] = uint32_t(scramble8[k]);
            m[k] = uint32_t(mask8[k]);
        }
        check_permute("permute f32x8", v, i, permute_f32x8(v, i), 8);
        check_compress("compress f32x8", v, m, compress_f32x8(v, m), 8);
        check_compress("compress f32x8, no fill", v, m,
                       compress_front_f32x8(v, m), 8, /*zeroed=*/false);
    }
    {
        uint32_t16 v, i, m;
        for (int k = 0; k < 16; k++) {
            v[k] = uint32_t(1000 + 7 * k);
            i[k] = uint32_t(scramble16[k]);
            m[k] = uint32_t(mask16[k]);
        }
        check_permute("permute u32x16", v, i, permute_u32x16(v, i), 16);
        check_compress("compress u32x16", v, m, compress_u32x16(v, m), 16);
    }
    {
        float4 v;
        uint32_t4 i;
        for (int k = 0; k < 4; k++) {
            v[k] = 0.25f * float(k + 1);
            i[k] = uint32_t(scramble4[k]);
        }
        check_permute("permute f32x4", v, i, permute_f32x4(v, i), 4);
    }
    {
        uint64_t8 v;
        uint32_t8 i, m;
        for (int k = 0; k < 8; k++) {
            // Distinct in both halves, so that a permute of 32-bit halves
            // that mixed them up would show.
            v[k] = (uint64_t(0xa0 + k) << 40) | uint64_t(0x1000 + k);
            i[k] = uint32_t(scramble8[k]);
            m[k] = uint32_t(mask8[k]);
        }
        check_permute("permute u64x8", v, i, permute_u64x8(v, i), 8);
        check_compress("compress u64x8", v, m, compress_u64x8(v, m), 8);
        check_compress("compress u64x8, no fill", v, m,
                       compress_front_u64x8(v, m), 8, /*zeroed=*/false);
    }
    {
        uint64_t4 v;
        uint32_t4 i;
        for (int k = 0; k < 4; k++) {
            v[k] = (uint64_t(0xb0 + k) << 40) | uint64_t(0x2000 + k);
            i[k] = uint32_t(scramble4[k]);
        }
        check_permute("permute u64x4", v, i, permute_u64x4(v, i), 4);
    }
    {
        uint64_t2 v;
        uint32_t2 i;
        for (int k = 0; k < 2; k++) {
            v[k] = (uint64_t(0xc0 + k) << 40) | uint64_t(0x3000 + k);
            i[k] = uint32_t(scramble2[k]);
        }
        check_permute("permute u64x2", v, i, permute_u64x2(v, i), 2);
    }
    {
        uint16_t16 v, i;
        for (int k = 0; k < 16; k++) {
            v[k] = uint16_t(300 + 3 * k);
            i[k] = uint16_t(scramble16[k]);
        }
        check_permute("permute u16x16", v, i, permute_u16x16(v, i), 16);
    }
    {
        float8 v;
        uint32_t4 i;
        for (int k = 0; k < 8; k++) {
            v[k] = float(k * k);
        }
        for (int k = 0; k < 4; k++) {
            i[k] = uint32_t(scramble8[k]);
        }
        check_permute("permute f32x8 to four", v, i, permute_narrow(v, i), 4);
    }
    if (failures != 0) {
        std::cout << failures << " WRONG\n";
        return EXIT_FAILURE;
    }
    std::cout << "all ok\n";
    return EXIT_SUCCESS;
}
