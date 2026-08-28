#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

// The parallel loop a `cpu_thread` schedule lowers to.
//
// On Apple platforms the generated code calls libdispatch's `dispatch_apply_f`
// directly. Everywhere else it calls this, which has the same shape and the
// same meaning: run `body(context, i)` once for each i in [0, n).
//
// It is `inline` so that a program picks it up by including the generated
// header, and `used` so that the definition is emitted even though nothing in
// the including translation unit calls it -- the caller is the object file the
// compiler produced, which refers to it by name.
extern "C" __attribute__((used)) inline void
bonsai_parallel_for(int64_t n, void *context, void (*body)(void *, size_t)) {
    if (n <= 0) {
        return;
    }

    unsigned int threads = std::thread::hardware_concurrency();
    if (threads == 0) {
        threads = 1;
    }
    threads = std::min<unsigned int>(threads, static_cast<unsigned int>(n));

    if (threads == 1) {
        for (int64_t i = 0; i < n; i++) {
            body(context, static_cast<size_t>(i));
        }
        return;
    }

    // A contiguous block each, which is what dispatch_apply_f effectively
    // gives for a loop whose iterations cost about the same.
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const int64_t per = (n + threads - 1) / threads;
    for (unsigned int t = 0; t < threads; t++) {
        const int64_t begin = static_cast<int64_t>(t) * per;
        const int64_t end = std::min(begin + per, n);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([=]() {
            for (int64_t i = begin; i < end; i++) {
                body(context, static_cast<size_t>(i));
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
}
