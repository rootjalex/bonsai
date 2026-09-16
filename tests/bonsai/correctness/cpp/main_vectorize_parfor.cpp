// Sequential so the test needs neither TBB nor a thread library to link: the
// bind(p, CPUThread) in vectorize_parfor.bonsai lowers to one call to
// bonsai_parallel_for, whose definition this picks the run-them-in-order one of.
#define BONSAI_PARALLEL_SEQUENTIAL
#include "vectorize_parfor.h"

#include <array>
#include <iostream>

int main() {
    std::array<float, 16> a{};
    for (int i = 0; i < 16; i++) {
        a[i] = float(i);
    }

    std::array<float, 16> out{};
    work(a, out);

    // 32*a[p] + 28, with a[p] = p.
    for (int i = 0; i < 16; i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << out[i];
    }
    std::cout << '\n';
}
