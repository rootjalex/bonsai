#pragma once

#include <cstddef>
#include <cstdint>

// The parallel loop a `bind(i, CPUThread)` schedule lowers to.
//
// Generated code makes exactly one call, to `bonsai_parallel_for(n, context,
// body)`, meaning: run `body(context, i)` once for each i in [0, n), in any
// order, and return when all of them are done. Choosing *how* is this header's
// job rather than the compiler's -- a compiler that decided would make its own
// output, and every golden of it, depend on which machine it ran on.
//
// Which implementation is used is picked here, in this order:
//
//   1. BONSAI_PARALLEL_EXTERNAL -- the program supplies its own definition of
//      bonsai_parallel_for and this header declares it only. That is the hook
//      for a host that already has a thread pool it would rather bonsai used.
//   2. BONSAI_PARALLEL_SEQUENTIAL -- run the iterations one after another.
//      Useful for debugging, and for a target with no threads at all.
//   3. BONSAI_PARALLEL_GCD / BONSAI_PARALLEL_THREADS -- ask for one of the
//      two below by name.
//   4. Otherwise: libdispatch on Apple, std::thread everywhere else.
//
// Adding a backend -- OpenMP, TBB, a Win32 pool -- is a change here and
// nowhere else, because the generated code only knows the three arguments.

#if !defined(BONSAI_PARALLEL_EXTERNAL) &&                                      \
    !defined(BONSAI_PARALLEL_SEQUENTIAL) && !defined(BONSAI_PARALLEL_GCD) &&   \
    !defined(BONSAI_PARALLEL_THREADS)
#if defined(__APPLE__)
#define BONSAI_PARALLEL_GCD
#else
#define BONSAI_PARALLEL_THREADS
#endif
#endif

#if defined(BONSAI_PARALLEL_EXTERNAL)

// Defined by the program. Declared with the same linkage and shape so that the
// object file the compiler produced finds it.
extern "C" void bonsai_parallel_for(int64_t n, void *context,
                                    void (*body)(void *, size_t));

#else

// `inline` so that a program picks the definition up by including the
// generated header, and `used` so that it is emitted even though nothing in
// the including translation unit calls it -- the caller is the object file the
// compiler produced, which refers to it by name.
#define BONSAI_PARALLEL_DEFN                                                   \
    extern "C" __attribute__((used)) inline void bonsai_parallel_for(          \
        int64_t n, void *context, void (*body)(void *, size_t))

#if defined(BONSAI_PARALLEL_GCD)

#include <dispatch/dispatch.h>

BONSAI_PARALLEL_DEFN {
    if (n <= 0) {
        return;
    }
    dispatch_apply_f(static_cast<size_t>(n), dispatch_get_global_queue(0, 0),
                     context, reinterpret_cast<void (*)(void *, size_t)>(body));
}

#elif defined(BONSAI_PARALLEL_SEQUENTIAL)

BONSAI_PARALLEL_DEFN {
    for (int64_t i = 0; i < n; i++) {
        body(context, static_cast<size_t>(i));
    }
}

#else // BONSAI_PARALLEL_THREADS

#include <algorithm>
#include <thread>
#include <vector>

BONSAI_PARALLEL_DEFN {
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

#endif

#undef BONSAI_PARALLEL_DEFN

#endif // BONSAI_PARALLEL_EXTERNAL
