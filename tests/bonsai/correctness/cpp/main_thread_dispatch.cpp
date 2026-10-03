#include "thread_dispatch.h"

#include <cstdint>
#include <iostream>

// Used in thread_dispatch.bonsai.
//
// `pick` as written: with c, y is 2x and the answer 2x + 1 whatever d is;
// without c, d decides between x + 1 and x - 1. Each combination is
// printed and checked.
namespace {

int32_t expected(bool c, bool d, int32_t x) {
    if (c) {
        return 2 * x + 1;
    }
    return d ? x + 1 : x - 1;
}

} // namespace

int main() {
    bool ok = true;
    for (int c = 0; c < 2; c++) {
        for (int d = 0; d < 2; d++) {
            const int32_t got = pick(c != 0, d != 0, 10);
            const int32_t want = expected(c != 0, d != 0, 10);
            std::cout << "c=" << c << " d=" << d << " -> " << got << '\n';
            ok = ok && got == want;
        }
    }
    std::cout << (ok ? "as written" : "DIFFERS from the function as written")
              << '\n';
    return ok ? 0 : 1;
}
