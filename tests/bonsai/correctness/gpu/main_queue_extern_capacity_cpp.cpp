#include "queue-extern-capacity-cpp.h"

#include <array>
#include <iostream>

// Used in queue-extern-capacity-cpp.bonsai: the driver owns the queue's
// storage and sizes it by the capacity the header states, an expression
// over the function's parameter `n`.
int main() {
    const int32_t n = 4;
    std::array<int32_t, 8> out;
    out.fill(-1);
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), sizeof(out));

    // The queue's arrays, in the order the header declares them, each in
    // device memory of the capacity the header gives -- `(n * 2)`, over the
    // parameter by name, which is in scope here as it is in the call --
    // times the element's bytes.
    struct Extern {
        const char *name;
        uint64_t capacity;
        uint64_t bytes;
        bonsai_buffer buffer;
    };
#define EXTERN(name)                                                           \
    {#name, uint64_t(BONSAI_run_##name##_CAPACITY),                            \
     uint64_t(BONSAI_run_##name##_CAPACITY) * BONSAI_run_##name##_ELEMENT_BYTES, \
     {}}
    Extern externs[] = {
        EXTERN(sums_k_0),     EXTERN(sums_acc_0), EXTERN(sums__slot_0),
        EXTERN(sums_k_1),     EXTERN(sums_acc_1), EXTERN(sums__slot_1),
    };
#undef EXTERN
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
    std::cout << "capacity: " << externs[0].capacity << '\n';
    std::cout << "bytes per array: " << externs[0].bytes << '\n';

    // Staged where the function needs them -- the queue's arrays are
    // already on the device, so nothing moves -- then no copy may happen.
    bonsai_buffer_stage_all(buffers, run_sides, 7);
    bonsai_buffer_implicit_copies(0);
    run(n, &out_buffer, &externs[0].buffer, &externs[1].buffer, &externs[2].buffer,
        &externs[3].buffer, &externs[4].buffer, &externs[5].buffer);
    run(n, &out_buffer, &externs[0].buffer, &externs[1].buffer, &externs[2].buffer,
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
