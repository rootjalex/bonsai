#include "buffers-cpp.h"

#include <array>
#include <iostream>
#include <vector>

// Used in buffers-cpp.bonsai.
int main() {
    const int32_t n = 8;
    std::vector<int32_t> a(n);
    for (int32_t i = 0; i < n; i++) {
        a[i] = i + 1;
    }
    std::vector<int32_t> out(n, -1);
    std::array<int32_t, 4> scale = {1, 10, 100, 1000};

    bonsai_buffer a_buffer = bonsai_buffer_wrap(a.data(), a.size() * sizeof(int32_t));
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), out.size() * sizeof(int32_t));
    bonsai_buffer scale_buffer = bonsai_buffer_wrap(scale.data(), sizeof(scale));
    bonsai_buffer *buffers[] = {&a_buffer, &out_buffer, &scale_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(run_sides));
    std::cout << "sides:";
    for (uint8_t side : run_sides) {
        std::cout << ' ' << unsigned(side);
    }
    std::cout << '\n';

    // Staged where the function needs them, then no copy may happen.
    bonsai_buffer_stage_all(buffers, run_sides, 3);
    bonsai_buffer_implicit_copies(0);
    run(n, &a_buffer, &out_buffer, &scale_buffer);
    run(n, &a_buffer, &out_buffer, &scale_buffer);
    std::cout << "out dirty on device: "
              << ((out_buffer.flags & BONSAI_BUFFER_DEVICE_DIRTY) != 0) << '\n';
    // The host copy is untouched until asked for.
    std::cout << "host before fetch: " << out[0] << '\n';
    bonsai_buffer_stage(&out_buffer, BONSAI_HOST);
    for (int32_t i = 0; i < n; i++) {
        std::cout << (i ? " " : "") << out[i];
    }
    std::cout << '\n';

    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
