#include "vectorize_switch_merge.h"

#include <array>
#include <iostream>

// Used in vectorize_switch_merge.bonsai.
//
// Lane i builds variant i % 3 of its input v, so the expected answers are, in
// turn: three times v plus twice v (Whole: v + 2v, then tripled), eleven
// times the lane index (Split, which returns from inside the match), and
// three times v (Wide, whose arm does nothing). A lane blended from the wrong
// arm, or one whose early return leaked into the merge, gives a different
// number.
int main() {
    const std::array<float, 8> a = {1.5f, -2.0f, 0.5f, 4.0f,
                                    -1.0f, 3.0f, -0.25f, 2.0f};
    std::array<float, 8> out{};
    weigh_all(a, out);
    bool same = true;
    for (size_t i = 0; i < out.size(); i++) {
        same = same && weigh_one(a[i], static_cast<int32_t>(i)) == out[i];
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
