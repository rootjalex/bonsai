// Used in mirror_derived_loads.bonsai.
//
// `count_within` and `count_near` as written, in C++, against the generated
// code, over sequences where the running best moves at every element (a
// descending run), never (an ascending run), and in the middle, with the
// bound above, within and below the elements.
#include "mirror_derived_loads.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace {

uint32_t ref_count_within(const std::array<float, 16> &xs, float bound) {
    float best = std::numeric_limits<float>::infinity();
    uint32_t n = 0;
    for (float x : xs) {
        const int32_t limit =
            std::min(std::bit_cast<int32_t>(bound), std::bit_cast<int32_t>(best) - 1);
        if (std::bit_cast<int32_t>(x) <= limit) {
            n++;
        }
        if (x < best) {
            best = x;
        }
    }
    return n;
}

uint32_t ref_count_near(const std::array<float, 16> &xs, float bound) {
    bool seen = false;
    uint32_t n = 0;
    for (float x : xs) {
        if (!seen) {
            n++;
        }
        if (x < bound) {
            seen = true;
        }
    }
    return n;
}

int failures = 0;

void check(const std::string &what, uint32_t got, uint32_t expected) {
    std::cout << what << " = " << got << (got == expected ? "" : " -- WRONG") << "\n";
    if (got != expected) {
        failures++;
    }
}

} // namespace

int main() {
    std::array<float, 16> descending{}, ascending{}, dipping{};
    for (size_t k = 0; k < 16; k++) {
        descending[k] = 16.0f - float(k);
        ascending[k] = 1.0f + float(k);
        dipping[k] = k < 8 ? 10.0f - float(k) : 3.0f + float(k);
    }
    const struct {
        const char *name;
        const std::array<float, 16> *xs;
    } sequences[] = {{"descending", &descending}, {"ascending", &ascending}, {"dipping", &dipping}};
    for (const auto &s : sequences) {
        for (float bound : {20.0f, 8.0f, 0.5f}) {
            const std::string at = std::string(s.name) + ", bound " + std::to_string(int(bound * 2)) + "/2";
            check("count_within(" + at + ")", count_within(*s.xs, bound),
                  ref_count_within(*s.xs, bound));
            check("count_near(" + at + ")", count_near(*s.xs, bound), ref_count_near(*s.xs, bound));
        }
    }
    if (failures != 0) {
        std::cout << failures << " wrong\n";
        return 1;
    }
    return 0;
}
