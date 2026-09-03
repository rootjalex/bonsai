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
//   3. BONSAI_PARALLEL_GCD / BONSAI_PARALLEL_TBB / BONSAI_PARALLEL_THREADS --
//      ask for one of the three below by name.
//   4. Otherwise: libdispatch on Apple, TBB where its header is installed, and
//      std::thread where it is not.
//
// Adding a backend -- OpenMP, a Win32 pool -- is a change here and nowhere
// else, because the generated code only knows the three arguments.
//
// The iterations are not assumed to cost the same, because the loops this
// lowers from are exactly the ones where they do not: a pixel of a render is
// one iteration, and a pixel that sees the subject costs many times one that
// sees the background. Handing each thread a contiguous block -- which is what
// this used to do -- gave the threads holding the middle of the frame most of
// the work and the rest nothing to do but finish early: on killeroo-simple,
// 18.8 of 32 threads busy on average against pbrt's 26.9, which was most of why
// that render was slower than pbrt's. Balancing the loop dynamically is a
// solved problem with a great deal of tuning behind it, so the default asks a
// library that has done the tuning rather than answering it here.

#if !defined(BONSAI_PARALLEL_EXTERNAL) &&                                      \
    !defined(BONSAI_PARALLEL_SEQUENTIAL) && !defined(BONSAI_PARALLEL_GCD) &&   \
    !defined(BONSAI_PARALLEL_TBB) && !defined(BONSAI_PARALLEL_THREADS)
#if defined(__APPLE__)
#define BONSAI_PARALLEL_GCD
#elif defined(__has_include)
#if __has_include(<tbb/parallel_for.h>)
#define BONSAI_PARALLEL_TBB
#else
#define BONSAI_PARALLEL_THREADS
#endif
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

#elif defined(BONSAI_PARALLEL_TBB)

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// The default partitioner splits the range further while threads are idle and
// steals between them, which is what makes an uneven loop finish when its work
// does rather than when its unluckiest thread does. A program built this way
// has to link the library: `-ltbb`.
BONSAI_PARALLEL_DEFN {
    if (n <= 0) {
        return;
    }
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, n),
                      [context, body](const tbb::blocked_range<int64_t> &r) {
                          for (int64_t i = r.begin(); i != r.end(); i++) {
                              body(context, static_cast<size_t>(i));
                          }
                      });
}

#elif defined(BONSAI_PARALLEL_SEQUENTIAL)

BONSAI_PARALLEL_DEFN {
    for (int64_t i = 0; i < n; i++) {
        body(context, static_cast<size_t>(i));
    }
}

#else // BONSAI_PARALLEL_THREADS

#include <algorithm>
#include <atomic>
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

    // Chunks handed out on demand rather than a contiguous block each, for the
    // reason at the top of this file. This is the backend for a machine with
    // no TBB, so it is the crude version of what that does: a fixed chunk and
    // no stealing, sized small enough that a straggler costs a chunk rather
    // than a thirty-second of the loop and large enough that the atomic is
    // paid once per few hundred iterations and a chunk's iterations are
    // neighbours.
    const int64_t chunk = std::max<int64_t>(
        1, n / (static_cast<int64_t>(threads) * 32));
    std::atomic<int64_t> next(0);

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned int t = 0; t < threads; t++) {
        workers.emplace_back([&next, chunk, n, context, body]() {
            for (;;) {
                const int64_t begin =
                    next.fetch_add(chunk, std::memory_order_relaxed);
                if (begin >= n) {
                    return;
                }
                const int64_t end = std::min(begin + chunk, n);
                for (int64_t i = begin; i < end; i++) {
                    body(context, static_cast<size_t>(i));
                }
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
