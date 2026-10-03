// Used in foreign.bonsai.
//
// The implementation of the program's two foreign functions, defined against
// the prototypes the generated header declares: `Grid` is the header's
// typedef (`const void *`), an array parameter is the address of its
// elements. `grid_of` is the address of the `at`th float; `grid_value` reads
// the `i`th float from there and doubles it, as foreign-impl.ll does for the
// LLVM backend's tests.
#include "foreign.h"

#include <cstdint>
#include <iostream>

extern "C" Grid grid_of(const float *bytes, uint32_t at) { return bytes + at; }

extern "C" float grid_value(Grid g, int32_t i) {
    return 2.0f * static_cast<const float *>(g)[i];
}

int main() {
    const float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    bonsai_buffer buffer = bonsai_buffer_wrap(
        const_cast<void *>(static_cast<const void *>(vals)), sizeof(vals));
    // (2 + 3) * 2
    std::cout << sample(&buffer, 1, 0) << '\n';
    // (4 + 5) * 2
    std::cout << sample(&buffer, 3, 0) << '\n';
    // (3 + 4) * 2: offset 1, index 1.
    std::cout << sample(&buffer, 1, 1) << '\n';
}
