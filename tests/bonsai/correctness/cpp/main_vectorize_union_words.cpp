#include "vectorize_union_words.h"

#include <array>
#include <cstdio>

// Used in vectorize_union_words.bonsai.
//
// Lane i holds variant i % 3. Small (lanes 0, 3, 6): a + b/2 + 100 c with
// a = i + 200 as a byte, b = i even, c = 1000 i + 7. Big (lanes 1, 4, 7):
// 10^10 i + 3, which does not fit 32 bits. Dbl (lanes 2, 5): 2 (i/2 + 1/4)
// + 3 i. Each lane is checked against the scalar path.
int main() {
    std::array<double, 8> out{};
    weigh_all(out);
    bool same = true;
    for (int i = 0; i < 8; i++) {
        same = same && weigh_one(i) == out[size_t(i)];
        std::printf("%s%.2f", i == 0 ? "" : " ", out[size_t(i)]);
    }
    std::printf("\n%s\n", same ? "same as scalar" : "DIFFERS from scalar");
}
