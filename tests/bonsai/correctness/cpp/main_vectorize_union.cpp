#include "vectorize_union.h"

#include <iostream>

// Used in vectorize_union.bonsai.
//
// Lane i builds variant i % 3 of its input, so the expected answers are, in
// turn: twice the input (Whole), eleven times the lane index (Split), and six
// times the input's magnitude (Wide, which negates the sum of its vector when
// the input is negative). A lane that read its neighbour's bytes, or its own
// at the wrong member's type, gives a different number.
int main() {
    const std::array<float, 8> a = {1.5f, -2.0f, 0.5f, 4.0f,
                                    -1.0f, 3.0f, -0.25f, 2.0f};
    std::array<float, 8> out{};
    weigh_all(a, out);
    for (size_t i = 0; i < out.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
