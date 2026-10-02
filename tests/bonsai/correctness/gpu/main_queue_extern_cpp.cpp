#include "queue-extern-cpp.h"

#include <array>
#include <iostream>

// Used in queue-extern-cpp.bonsai: the driver owns the queue's storage.
int main() {
    std::array<int32_t, 8> out;
    out.fill(-1);
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), sizeof(out));

    // The queue's arrays, in the order the header declares them, each in
    // device memory of the size the header gives.
    struct Extern {
        const char *name;
        uint64_t bytes;
        bonsai_buffer buffer;
    };
    Extern externs[] = {
        {"sums_k_0", BONSAI_run_sums_k_0_BYTES, {}},
        {"sums_acc_0", BONSAI_run_sums_acc_0_BYTES, {}},
        {"sums__slot_0", BONSAI_run_sums__slot_0_BYTES, {}},
        {"sums_k_1", BONSAI_run_sums_k_1_BYTES, {}},
        {"sums_acc_1", BONSAI_run_sums_acc_1_BYTES, {}},
        {"sums__slot_1", BONSAI_run_sums__slot_1_BYTES, {}},
    };
    for (Extern &e : externs) {
        e.buffer = bonsai_buffer_wrap_device(bonsai_cuda_malloc(e.bytes), e.bytes);
    }
    bonsai_buffer *buffers[] = {&out_buffer,
                                &externs[0].buffer, &externs[1].buffer,
                                &externs[2].buffer, &externs[3].buffer,
                                &externs[4].buffer, &externs[5].buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(run_sides));
    std::cout << "sides:";
    for (uint8_t side : run_sides) {
        std::cout << ' ' << unsigned(side);
    }
    std::cout << '\n';
    std::cout << "bytes per array: " << externs[0].bytes << '\n';

    // Staged where the function needs them -- the queue's arrays are
    // already on the device, so nothing moves -- then no copy may happen.
    bonsai_buffer_stage_all(buffers, run_sides, 7);
    bonsai_buffer_implicit_copies(0);
    run(&out_buffer, &externs[0].buffer, &externs[1].buffer, &externs[2].buffer,
        &externs[3].buffer, &externs[4].buffer, &externs[5].buffer);
    run(&out_buffer, &externs[0].buffer, &externs[1].buffer, &externs[2].buffer,
        &externs[3].buffer, &externs[4].buffer, &externs[5].buffer);
    bonsai_buffer_stage(&out_buffer, BONSAI_HOST);
    for (int32_t i = 0; i < 8; i++) {
        std::cout << (i ? " " : "") << out[i];
    }
    std::cout << '\n';

    bonsai_buffer_free(&out_buffer);
    for (Extern &e : externs) {
        bonsai_cuda_free(e.buffer.device);
    }
}
