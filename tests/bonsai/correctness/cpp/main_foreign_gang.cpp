// Used in foreign_gang.bonsai.
//
// The program's two foreign functions, defined against the prototypes the
// generated header declares, as main_foreign.cpp defines them -- and
// `grid_value` keeps a record of every call it gets: how many, and which
// index each asked for, in the order they came. A gang of eight lanes calls
// it for the four even lanes only, one scalar call each, in lane order; an
// odd lane, whose branch did not take the call, must not reach it with an
// index nothing computed.
#include "foreign_gang.h"

#include <cstdint>
#include <iostream>
#include <vector>

static std::vector<int32_t> asked;

extern "C" Grid grid_of(const float *bytes, uint32_t at) { return bytes + at; }

extern "C" float grid_value(Grid g, int32_t i) {
    asked.push_back(i);
    return 2.0f * static_cast<const float *>(g)[i];
}

int main() {
    const float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    bonsai_buffer buffer = bonsai_buffer_wrap(
        const_cast<void *>(static_cast<const void *>(vals)), sizeof(vals));
    float out[8] = {0};
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out, sizeof(out));
    // From offset 1: lane i reads vals[1 + i] doubled on the even lanes,
    // -1 on the odd ones: 4 -1 8 -1 12 -1 16 -1.
    sample(&buffer, 1, &out_buffer);
    for (float v : out) {
        std::cout << v << ' ';
    }
    std::cout << '\n';
    std::cout << asked.size() << " calls:";
    for (int32_t i : asked) {
        std::cout << ' ' << i;
    }
    std::cout << '\n';
}
