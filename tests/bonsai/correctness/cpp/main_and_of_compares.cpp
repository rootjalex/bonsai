// Used in and_of_compares.bonsai.
//
// Every call is checked against the expression the program wrote, evaluated
// here as C++ writes it, so that a rewrite that changed the meaning at a
// tie, on a NaN or in a lane shows up as a WRONG line.
#include "and_of_compares.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void check(const char *what, bool got, bool expected) {
    std::cout << what << ": " << (got ? "true" : "false")
              << (got == expected ? " -- ok" : " -- WRONG") << '\n';
    if (got != expected) {
        failures++;
    }
}

} // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // c && (select(c, x, y) < 10): under c the select is x.
    check("decided(true, 5, 50)", decided(true, 5, 50), true);
    check("decided(true, 50, 5)", decided(true, 50, 5), false);
    check("decided(false, 5, 5)", decided(false, 5, 5), false);
    // c || (select(c, x, y) < 10): where c fails the select is y.
    check("decided_or(false, 50, 5)", decided_or(false, 50, 5), true);
    check("decided_or(false, 5, 50)", decided_or(false, 5, 50), false);
    check("decided_or(true, 50, 50)", decided_or(true, 50, 50), true);
    // i <= n && i <= m, at the ties.
    check("at_most(3, 3, 7)", at_most(3, 3, 7), true);
    check("at_most(4, 3, 7)", at_most(4, 3, 7), false);
    check("at_most(3, 7, 3)", at_most(3, 7, 3), true);
    check("at_most(3, 7, 2)", at_most(3, 7, 2), false);
    // n < i || m < i.
    check("below(5, 3, 9)", below(5, 3, 9), true);
    check("below(5, 6, 9)", below(5, 6, 9), false);
    check("below(5, 9, 4)", below(5, 9, 4), true);
    check("below(5, 5, 5)", below(5, 5, 5), false);
    // t <= a && t <= b over floats: a NaN on either side is false.
    check("at_most_f(1, 2, 3)", at_most_f(1.0f, 2.0f, 3.0f), true);
    check("at_most_f(1, 0.5, 3)", at_most_f(1.0f, 0.5f, 3.0f), false);
    check("at_most_f(1, 2, nan)", at_most_f(1.0f, 2.0f, nan), false);
    check("at_most_f(1, nan, 2)", at_most_f(1.0f, nan, 2.0f), false);
    check("at_most_f(nan, 2, 3)", at_most_f(nan, 2.0f, 3.0f), false);
    // The lanes: 0..7 against n = 5 and m = 3 keep lanes 0..3.
    std::array<uint32_t, 8> is{}, out{};
    for (int k = 0; k < 8; k++) {
        is[k] = uint32_t(k);
        out[k] = 7u;
    }
    at_most_all(is, 5u, 3u, out);
    for (int k = 0; k < 8; k++) {
        const bool expected = uint32_t(k) <= 5u && uint32_t(k) <= 3u;
        std::cout << "at_most_all lane " << k << ": " << out[k]
                  << ((out[k] == (expected ? 1u : 0u)) ? " -- ok" : " -- WRONG")
                  << '\n';
        if (out[k] != (expected ? 1u : 0u)) {
            failures++;
        }
    }
    if (failures != 0) {
        std::cout << failures << " wrong\n";
        return 1;
    }
    return 0;
}
