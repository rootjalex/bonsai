// Drives vectorize_gather_unsigned.bonsai over a mapping of 2 GB and a
// megabyte more: known values written on both sides of the 2^31 mark and
// read back through the three vectorized functions, every lane checked
// against the element its own index names. The mapping is reserved and
// never touched beyond those elements, so the pages written are all the
// memory the test costs. The indices within one call are distinct, so that
// a lane reading another lane's element is caught and not matched by
// chance.
//
// Before the lane offsets were read as unsigned, a lane at or past the mark
// read 2^32 units below its element: a fault, or a plausible value from
// whatever is mapped there.

#include "vectorize_gather_unsigned.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>

namespace {

constexpr uint64_t MARK = uint64_t(1) << 31;
constexpr uint64_t EXTRA = uint64_t(1) << 20;
constexpr uint64_t BYTES = MARK + EXTRA;

void *reserve() {
    void *p = mmap(nullptr, BYTES, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        std::cerr << "mmap of " << BYTES << " bytes failed\n";
        std::exit(1);
    }
    return p;
}

}  // namespace

int main() {
    size_t wrong = 0;

    // Bytes, the texture pool's case: offsets below the mark, at it, and up
    // to the mapping's last byte.
    {
        uint8_t *a = static_cast<uint8_t *>(reserve());
        const std::array<uint32_t, 16> idx = {
            0,
            7,
            4095,
            uint32_t(MARK - 1),
            uint32_t(MARK),
            uint32_t(MARK + 1),
            uint32_t(MARK + 5),
            uint32_t(MARK + 4096),
            uint32_t(MARK + 1000000),
            uint32_t(MARK + EXTRA - 1),
            uint32_t(MARK + 3),
            uint32_t(MARK + 65536),
            1,
            uint32_t(MARK - 4096),
            uint32_t(MARK + 2),
            uint32_t(MARK + 500000),
        };
        for (size_t i = 0; i < idx.size(); i++) {
            a[idx[i]] = uint8_t(17 * i + 3);
        }
        std::array<uint32_t, 16> out{};
        bytes_at(a, idx, out);
        for (size_t i = 0; i < idx.size(); i++) {
            const uint32_t want = uint8_t(17 * i + 3);
            if (out[i] != want) {
                wrong++;
                std::cout << "bytes_at lane " << i << " at " << idx[i] << ": "
                          << out[i] << ", not " << want << "\n";
            }
        }
        munmap(a, BYTES);
    }

    // A field of an eight-byte element: element indices whose byte offsets
    // straddle the mark.
    {
        Pair *p = static_cast<Pair *>(reserve());
        const uint32_t at_mark = uint32_t(MARK / sizeof(Pair));
        const std::array<uint32_t, 16> idx = {
            0,
            3,
            at_mark - 1,
            at_mark,
            at_mark + 1,
            at_mark + 2,
            at_mark + 100,
            at_mark + 1000,
            at_mark + 65535,
            uint32_t(BYTES / sizeof(Pair)) - 1,
            at_mark - 2,
            5,
            at_mark + 7,
            at_mark + 77777,
            at_mark + 9,
            at_mark + 123456,
        };
        for (size_t i = 0; i < idx.size(); i++) {
            p[idx[i]].a = uint32_t(1000 + i);
            p[idx[i]].b = 0xA0000000u + uint32_t(11 * i);
        }
        std::array<uint32_t, 16> out{};
        second_of(p, idx, out);
        for (size_t i = 0; i < idx.size(); i++) {
            const uint32_t want = 0xA0000000u + uint32_t(11 * i);
            if (out[i] != want) {
                wrong++;
                std::cout << "second_of lane " << i << " at " << idx[i] << ": "
                          << out[i] << ", not " << want << "\n";
            }
        }
        munmap(p, BYTES);
    }

    // A scatter of words to elements past the mark.
    {
        uint32_t *out = static_cast<uint32_t *>(reserve());
        const uint32_t at_mark = uint32_t(MARK / sizeof(uint32_t));
        const std::array<uint32_t, 16> idx = {
            0,
            9,
            at_mark - 1,
            at_mark,
            at_mark + 1,
            at_mark + 2,
            at_mark + 100,
            at_mark + 1000,
            at_mark + 65535,
            uint32_t(BYTES / sizeof(uint32_t)) - 1,
            at_mark - 2,
            5,
            at_mark + 7,
            at_mark + 77777,
            at_mark + 9,
            at_mark + 123456,
        };
        std::array<uint32_t, 16> v{};
        for (size_t i = 0; i < v.size(); i++) {
            v[i] = 0x5000000u + uint32_t(13 * i);
        }
        scatter_to(idx, v, out);
        for (size_t i = 0; i < idx.size(); i++) {
            if (out[idx[i]] != v[i]) {
                wrong++;
                std::cout << "scatter_to lane " << i << " at " << idx[i] << ": "
                          << out[idx[i]] << ", not " << v[i] << "\n";
            }
        }
        munmap(out, BYTES);
    }

    std::cout << (wrong == 0 ? "ok" : "wrong") << "\n";
    return wrong == 0 ? 0 : 1;
}
