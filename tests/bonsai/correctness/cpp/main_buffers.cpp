#include "buffers.h"

#include <array>
#include <iostream>
#include <vector>

// Used in buffers.bonsai.
//
// Two ways to hand an exported function its arrays. The descriptors, which
// a driver makes once, stages where the function's `_sides` table says, and
// keeps across calls -- with implicit copies forbidden, so that a copy the
// staging did not cover would stop the program rather than be timed. And the
// header's pointer form, which wraps each array for the one call, for a
// driver that does not stage.
int main() {
    const uint32_t n = 8;
    std::array<float, 4> weights = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> a(n);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = float(i + 1);
    }
    std::vector<float> out(n, -1.0f);

    // The descriptors: wrapped, staged, kept.
    bonsai_buffer weights_buffer = bonsai_buffer_wrap(weights.data(), sizeof(weights));
    bonsai_buffer a_buffer = bonsai_buffer_wrap(a.data(), a.size() * sizeof(float));
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), out.size() * sizeof(float));
    bonsai_buffer *combine_buffers[] = {&weights_buffer, &a_buffer, &out_buffer};
    static_assert(sizeof(combine_buffers) / sizeof(combine_buffers[0]) ==
                  sizeof(combine_sides));
    bonsai_buffer_stage_all(combine_buffers, combine_sides, 3);
    bonsai_buffer_implicit_copies(0);

    combine(n, &weights_buffer, &a_buffer, &out_buffer);
    // A second call finds every buffer where it left it.
    combine(n, &weights_buffer, &a_buffer, &out_buffer);
    bonsai_buffer_stage(&out_buffer, BONSAI_HOST);
    for (uint32_t i = 0; i < n; i++) {
        std::cout << (i ? " " : "") << out[i];
    }
    std::cout << '\n';
    // The function wrote `out` on the host, and said so.
    std::cout << "out dirty on host: "
              << ((out_buffer.flags & BONSAI_BUFFER_HOST_DIRTY) != 0) << '\n';
    std::cout << "total: " << total(n, &a_buffer) << '\n';

    // The pointer form, wrapping for the call.
    bonsai_buffer_implicit_copies(1);
    std::vector<float> out2(n, -1.0f);
    combine(n, weights, a.data(), out2.data());
    for (uint32_t i = 0; i < n; i++) {
        std::cout << (i ? " " : "") << out2[i];
    }
    std::cout << '\n';
    std::cout << "total: " << total(n, a.data()) << '\n';

    for (bonsai_buffer *b : combine_buffers) {
        bonsai_buffer_free(b);
    }
}
