#include "dirty-where-touched.h"

#include <iostream>
#include <vector>

// Used in dirty-where-touched.bonsai.
namespace {

void print(const std::vector<int32_t> &out) {
    for (size_t i = 0; i < out.size(); i++) {
        std::cout << (i ? " " : "") << out[i];
    }
    std::cout << '\n';
}

void print_flags(const char *when, const bonsai_buffer &b) {
    std::cout << when << ": dirty on host "
              << ((b.flags & BONSAI_BUFFER_HOST_DIRTY) != 0)
              << ", dirty on device "
              << ((b.flags & BONSAI_BUFFER_DEVICE_DIRTY) != 0) << '\n';
}

} // namespace

int main() {
    const int32_t n = 8;
    std::vector<int32_t> a(n);
    for (int32_t i = 0; i < n; i++) {
        a[i] = i + 1;
    }
    std::vector<int32_t> out(n, -1);

    bonsai_buffer a_buffer = bonsai_buffer_wrap(a.data(), a.size() * sizeof(int32_t));
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), out.size() * sizeof(int32_t));
    bonsai_buffer *buffers[] = {&a_buffer, &out_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(run_sides));
    std::cout << "sides:";
    for (uint8_t side : run_sides) {
        std::cout << ' ' << unsigned(side);
    }
    std::cout << '\n';

    bonsai_buffer_stage_all(buffers, run_sides, 2);
    bonsai_buffer_implicit_copies(0);

    // The GPU's arm: nothing may be copied, and `out` ends up on the device.
    run(false, n, &a_buffer, &out_buffer);
    print_flags("after the GPU arm", out_buffer);
    std::cout << "host before fetch: " << out[0] << '\n';
    bonsai_buffer_stage(&out_buffer, BONSAI_HOST);
    print(out);

    // The host's arm: `out` is written on the host, and the device's copy is
    // the stale one.
    run(true, n, &a_buffer, &out_buffer);
    print_flags("after the host arm", out_buffer);
    print(out);

    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
