#include "defer_thread_queue.h"

#include <array>
#include <cstdint>
#include <iostream>

// Used in defer_thread_queue.bonsai
//
// Four chunks, each a thread's iteration with its own queue of three
// entries sized at run time (counts[0]), drained inside the thread: `out[i]`
// sums the chunk's values weighted by how many times each step recurs
// (value j of the chunk, j + 1 times), `done[i]` counts the continuations
// the pass over the record ran. Run twice, so that the second run's
// iterations reuse the arena storage the first run's released.

int main() {
    const std::array<float, 12> values = {1.0f,   2.0f,   3.0f,    10.0f,   20.0f,   30.0f,
                                          100.0f, 200.0f, 300.0f, 1000.0f, 2000.0f, 3000.0f};
    const std::array<uint32_t, 1> counts = {3u};
    for (int pass = 0; pass < 2; pass++) {
        std::array<float, 4> out{};
        std::array<float, 4> done{};
        run(values, counts, out, done);
        std::cout << "out:";
        for (float v : out) {
            std::cout << ' ' << v;
        }
        std::cout << "\ndone:";
        for (float v : done) {
            std::cout << ' ' << v;
        }
        std::cout << '\n';
    }
}
