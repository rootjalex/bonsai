// Used in vectorize_constant_argument.bonsai.
//
// `bias` is the program's foreign function, defined here and counting its
// calls: the clone of `shade` for `lit = true` calls it once per lane, the
// clone for `lit = false` has no call to make. Prints the eight outputs for
// each value of the flag and the count of calls after each.
#include "vectorize_constant_argument.h"

#include <cstdint>
#include <iostream>

static int calls = 0;

extern "C" float bias(int32_t i) {
    calls++;
    return 0.5f * static_cast<float>(i);
}

int main() {
    const float inp[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    bonsai_buffer in_buffer = bonsai_buffer_wrap(
        const_cast<void *>(static_cast<const void *>(inp)), sizeof(inp));
    float out[8] = {0};
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out, sizeof(out));
    for (const bool lit : {true, false}) {
        render(&in_buffer, &out_buffer, lit);
        std::cout << (lit ? "lit:" : "unlit:");
        for (float v : out) {
            std::cout << ' ' << v;
        }
        std::cout << " (bias called " << calls << " times)\n";
        calls = 0;
    }
}
