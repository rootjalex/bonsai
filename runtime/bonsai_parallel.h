#pragma once

#include <cstddef>
#include <cstdint>

// The parallel loop a `bind(i, CPUThread)` schedule lowers to.
//
// Generated code makes exactly one call, to `bonsai_parallel_for(n, context,
// body)`, meaning: run every iteration i in [0, n), in any order, and return
// when all of them are done, where `body(context, begin, end)` runs the
// iterations [begin, end) one after another. The body takes a range rather
// than one iteration so that a thread pays one call per range it is handed
// and not one per iteration: the generated body reads its captures once and
// loops, as a device kernel does over its grid stride. A wavefront's drains
// run a body per queue entry per round per pass, a hundred million calls a
// frame with sixty captures each, and the calls were most of the render.
// Choosing *how* the ranges are handed out is this header's job rather than
// the compiler's -- a compiler that decided would make its own output, and
// every golden of it, depend on which machine it ran on.
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
                                    void (*body)(void *, int64_t, int64_t));

#else

// `inline` so that a program picks the definition up by including the
// generated header, and `used` so that it is emitted even though nothing in
// the including translation unit calls it -- the caller is the object file the
// compiler produced, which refers to it by name.
#define BONSAI_PARALLEL_DEFN                                                   \
    extern "C" __attribute__((used)) inline void bonsai_parallel_for(          \
        int64_t n, void *context, void (*body)(void *, int64_t, int64_t))

#if defined(BONSAI_PARALLEL_GCD)

#include <algorithm>
#include <dispatch/dispatch.h>

// libdispatch hands out one iteration number per call, so the loop is cut
// into chunks first and each dispatched iteration runs a chunk's range: a few
// hundred iterations, as the thread backend below chooses, so that a chunk
// costs one call and a straggler costs a chunk.
BONSAI_PARALLEL_DEFN {
    if (n <= 0) {
        return;
    }
    struct Chunked {
        void *context;
        void (*body)(void *, int64_t, int64_t);
        int64_t n, chunk;
    };
    Chunked chunked{context, body, n, std::max<int64_t>(1, n / 256)};
    const size_t chunks =
        static_cast<size_t>((n + chunked.chunk - 1) / chunked.chunk);
    dispatch_apply_f(chunks, dispatch_get_global_queue(0, 0), &chunked,
                     [](void *c, size_t k) {
                         const Chunked &ch = *static_cast<const Chunked *>(c);
                         const int64_t begin = static_cast<int64_t>(k) * ch.chunk;
                         ch.body(ch.context, begin,
                                 std::min(begin + ch.chunk, ch.n));
                     });
}

#elif defined(BONSAI_PARALLEL_TBB)

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// The default partitioner splits the range further while threads are idle and
// steals between them, which is what makes an uneven loop finish when its work
// does rather than when its unluckiest thread does; each range it settles on
// is one call of the body. A program built this way has to link the library:
// `-ltbb`.
BONSAI_PARALLEL_DEFN {
    if (n <= 0) {
        return;
    }
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, n),
                      [context, body](const tbb::blocked_range<int64_t> &r) {
                          body(context, r.begin(), r.end());
                      });
}

#elif defined(BONSAI_PARALLEL_SEQUENTIAL)

BONSAI_PARALLEL_DEFN {
    if (n > 0) {
        body(context, 0, n);
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
        body(context, 0, n);
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
                body(context, begin, std::min(begin + chunk, n));
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

// The arena: a thread's storage for the iteration it is running.
//
// A loop bound to CPUThread runs its body as a kernel on the runtime's
// threads, and storage the body makes with a run-time size -- the queues of
// a wavefront schedule that puts them on the bound loop, sized by a chunk
// of pixels times the sample count, and the record slots beside them -- is
// the iteration's: made when it starts, dead when it ends (docs/schedule-
// semantics.md, "queue"). The stack was that storage, and is right for a
// pixel's queues of a few hundred kilobytes; a chunk's run to tens of
// megabytes an iteration, which no worker's stack holds, and a heap
// allocation per iteration is what --no-heap exists to refuse. So the
// generated kernel takes such storage from here: `bonsai_arena_mark` before
// the iteration, `bonsai_arena_alloc` for each array, `bonsai_arena_release`
// to the mark after it (CodeGen_LLVM_SSA.cpp, the parfor kernel; CodeGen_
// LLVM::in_cpu_kernel) -- a bump allocator per thread, whose blocks are
// kept for the thread's life, so that once a thread has run one iteration
// of a loop its later iterations allocate nothing. MoonRay's per-thread
// queues are this, allocated once per thread and reused; Halide's
// `halide_malloc` inside a parallel loop is the per-iteration heap
// allocation this avoids.
//
// Marks nest: a thread that waits inside a nested bound loop may be handed
// another iteration of the outer loop by the scheduler's work stealing, and
// that iteration marks above the suspended one's storage and releases back
// to its own mark, so the suspended iteration's storage is untouched. The
// blocks are sixty-four-byte aligned for the vector stores of a gang, and a
// block never moves once made, so a pointer handed out stays good until its
// mark is released.
#include <algorithm>
#include <cstdlib>
#include <new>
#include <vector>

namespace bonsai_arena_detail {

struct Arena {
    struct Block {
        char *base;
        uint64_t size;
    };
    std::vector<Block> blocks;
    size_t current = 0;  // the block allocations come from
    uint64_t offset = 0; // the bytes of it used
    ~Arena() {
        for (Block &b : blocks) {
            std::free(b.base);
        }
    }
};

inline Arena &arena() {
    static thread_local Arena a;
    return a;
}

constexpr uint64_t kArenaAlign = 64;
constexpr uint64_t kArenaFirstBlock = uint64_t{1} << 20;

}  // namespace bonsai_arena_detail

// The arena's position, to release to.
extern "C" __attribute__((used)) inline uint64_t bonsai_arena_mark() {
    const bonsai_arena_detail::Arena &a = bonsai_arena_detail::arena();
    return (static_cast<uint64_t>(a.current) << 48) | a.offset;
}

// `bytes` of storage, sixty-four-byte aligned, good until the mark taken
// before it is released.
extern "C" __attribute__((used)) inline void *bonsai_arena_alloc(uint64_t bytes) {
    using namespace bonsai_arena_detail;
    Arena &a = arena();
    bytes = std::max<uint64_t>((bytes + kArenaAlign - 1) & ~(kArenaAlign - 1), kArenaAlign);
    while (true) {
        if (a.current < a.blocks.size()) {
            Arena::Block &b = a.blocks[a.current];
            if (b.size - a.offset >= bytes) {
                void *p = b.base + a.offset;
                a.offset += bytes;
                return p;
            }
            if (a.current + 1 < a.blocks.size()) {
                // A block kept from before; what is left of this one waits
                // for the release.
                a.current++;
                a.offset = 0;
                continue;
            }
        }
        // A new block: room for this allocation, and at least twice the
        // last block, so that the blocks are few.
        const uint64_t last = a.blocks.empty() ? 0 : a.blocks.back().size;
        const uint64_t size = std::max<uint64_t>({bytes, 2 * last, kArenaFirstBlock});
        void *base = std::aligned_alloc(kArenaAlign, size);
        if (base == nullptr) {
            throw std::bad_alloc();
        }
        a.blocks.push_back(Arena::Block{static_cast<char *>(base), size});
        a.current = a.blocks.size() - 1;
        a.offset = 0;
    }
}

// Back to a mark: everything allocated since is given up.
extern "C" __attribute__((used)) inline void bonsai_arena_release(uint64_t mark) {
    bonsai_arena_detail::Arena &a = bonsai_arena_detail::arena();
    a.current = static_cast<size_t>(mark >> 48);
    a.offset = mark & ((uint64_t{1} << 48) - 1);
}

// The sort a queue's drain runs its entries in, for a schedule that sorts
// a deferred call by a key (`f.sort(g, |args| key)`, SSA/Defer.cpp): the
// generated code makes one call, `bonsai_sort_indices(keys, perm, n)`,
// meaning: fill perm[0..n) with the indices 0..n-1 ordered by ascending
// keys[index], ties in index order (a stable sort, so that a run of equal
// keys keeps the push order). Which sort runs is this header's business, as
// the loop's is. Ray reordering keys are 64-bit Morton codes (Meister,
// Boksansky, Guthe, Bittner 2025), hence the width.
//
// The call comes from wherever the drain is: once per pass over a queue the
// size of a band of the frame, or, when the queue is a thread's own, from
// every thread at once over a few thousand entries. A sort that forks tasks
// is right for the first and wrong for the second -- the threads are all
// busy already -- so the choice is by size. Measured on the (key, index)
// pairs of 60-bit random keys, one thread of a Zen 5 with AVX-512, the
// least of seven (microseconds; scratch bench of 2026-10-21):
//
//      n      std::sort   tbb::parallel_sort   radix (11-bit)   x86-simd-sort
//    1024          4.7            4.8                6.8              3.9
//    4096         91.1           30.2               20.5             17.5
//   16384        495.1          154.7              116.5             83.5
//   65536       2378.3          550.9              536.0            402.8
//  262144      10635.8         2271.7             2264.3           2002.0
// 1048576      46987.2         8649.6            16170.2           8873.4
//
// So: Intel's x86-simd-sort (a vectorized quicksort with bitonic networks
// at the leaves, the key-value form sorting the indices alongside the keys)
// where its header is installed and this translation unit is compiled for
// an AVX-512 or AVX2 target, the fastest at every size up to a million;
// else an LSD radix sort of the pairs, 11-bit digits, which beats
// std::sort from a few thousand entries up and is stable by construction;
// std::sort under a thousand entries, where the radix sort's histograms cost
// more than the sort. Above a million entries TBB's parallel sort, where it
// is linked -- a tie at a million on this machine and ahead beyond it.
//
// x86-simd-sort is header-only (BSD-3, github.com/intel/x86-simd-sort); it
// is found as <x86simdsort/x86simdsort-static-incl.h>, so installing it is
// copying its src/ to <prefix>/include/x86simdsort, beside TBB's headers in a
// conda environment. Its sort is not stable; the stable order is restored
// afterwards by sorting the indices within each run of equal keys, which
// costs one pass when, as for Morton keys, the runs are short.
//
// The pairs are gathered first so that the sort moves sixteen-byte records
// and the keys array, a queue's own storage, is read once and left as it was.
#include <algorithm>
#include <cstdint>
#include <vector>
#if defined(BONSAI_PARALLEL_TBB)
#include <tbb/parallel_sort.h>
#endif
#if defined(__has_include)
#if __has_include(<x86simdsort/x86simdsort-static-incl.h>) &&                 \
    ((defined(__AVX512DQ__) && defined(__AVX512VL__)) || defined(__AVX2__))
#define BONSAI_SORT_X86_SIMD_SORT
#include <x86simdsort/x86simdsort-static-incl.h>
#endif
#endif

namespace bonsai_sort_detail {

struct Pair {
    uint64_t key;
    uint32_t index;
};

inline void gather(const uint64_t *keys, int64_t n, std::vector<Pair> &pairs) {
    pairs.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        pairs[static_cast<size_t>(i)] = Pair{keys[i], static_cast<uint32_t>(i)};
    }
}

inline void scatter(const std::vector<Pair> &pairs, uint32_t *perm, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        perm[i] = pairs[static_cast<size_t>(i)].index;
    }
}

// The index breaks ties, so an unstable sort of (key, index) is a stable
// sort of the keys.
inline bool before(const Pair &a, const Pair &b) {
    return a.key < b.key || (a.key == b.key && a.index < b.index);
}

inline void sort_std(const uint64_t *keys, uint32_t *perm, int64_t n) {
    std::vector<Pair> pairs;
    gather(keys, n, pairs);
    std::sort(pairs.begin(), pairs.end(), before);
    scatter(pairs, perm, n);
}

// LSD radix sort: six passes of eleven bits cover the 64-bit key, every
// pass's histogram taken in the one read that gathers the pairs, and a pass
// whose digit is the same in every key -- the top passes of a Morton key
// that uses sixty bits, say -- skipped.
inline void sort_radix(const uint64_t *keys, uint32_t *perm, int64_t n) {
    constexpr int kBits = 11;
    constexpr int kPasses = 6;
    constexpr int kBuckets = 1 << kBits;
    std::vector<Pair> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    std::vector<uint32_t> hist(static_cast<size_t>(kPasses * kBuckets), 0);
    for (int64_t i = 0; i < n; i++) {
        const uint64_t k = keys[i];
        a[static_cast<size_t>(i)] = Pair{k, static_cast<uint32_t>(i)};
        for (int p = 0; p < kPasses; p++) {
            hist[static_cast<size_t>(p * kBuckets) +
                 ((k >> (p * kBits)) & (kBuckets - 1))]++;
        }
    }
    Pair *src = a.data();
    Pair *dst = b.data();
    for (int p = 0; p < kPasses; p++) {
        uint32_t *h = hist.data() + p * kBuckets;
        bool constant_digit = false;
        for (int bucket = 0; bucket < kBuckets; bucket++) {
            if (h[bucket] == static_cast<uint32_t>(n)) {
                constant_digit = true;
                break;
            }
        }
        if (constant_digit) {
            continue;
        }
        uint32_t sum = 0;
        for (int bucket = 0; bucket < kBuckets; bucket++) {
            const uint32_t count = h[bucket];
            h[bucket] = sum;
            sum += count;
        }
        const int shift = p * kBits;
        for (int64_t i = 0; i < n; i++) {
            const Pair &e = src[i];
            dst[h[(e.key >> shift) & (kBuckets - 1)]++] = e;
        }
        std::swap(src, dst);
    }
    for (int64_t i = 0; i < n; i++) {
        perm[i] = src[i].index;
    }
}

#if defined(BONSAI_SORT_X86_SIMD_SORT)
inline void sort_simd(const uint64_t *keys, uint32_t *perm, int64_t n) {
    std::vector<uint64_t> sorted(keys, keys + n);
    for (int64_t i = 0; i < n; i++) {
        perm[i] = static_cast<uint32_t>(i);
    }
    x86simdsortStatic::keyvalue_qsort(sorted.data(), perm, static_cast<size_t>(n));
    // Stability: within a run of equal keys, the indices in ascending order.
    for (int64_t i = 0; i < n;) {
        int64_t j = i + 1;
        while (j < n && sorted[static_cast<size_t>(j)] == sorted[static_cast<size_t>(i)]) {
            j++;
        }
        if (j - i > 1) {
            std::sort(perm + i, perm + j);
        }
        i = j;
    }
}
#endif

#if defined(BONSAI_PARALLEL_TBB)
inline void sort_parallel(const uint64_t *keys, uint32_t *perm, int64_t n) {
    std::vector<Pair> pairs;
    gather(keys, n, pairs);
    tbb::parallel_sort(pairs.begin(), pairs.end(), before);
    scatter(pairs, perm, n);
}
#endif

}  // namespace bonsai_sort_detail

extern "C" __attribute__((used)) inline void
bonsai_sort_indices(const uint64_t *keys, uint32_t *perm, int64_t n) {
    using namespace bonsai_sort_detail;
    if (n <= 0) {
        return;
    }
#if defined(BONSAI_PARALLEL_TBB)
    if (n >= (int64_t{1} << 20)) {
        sort_parallel(keys, perm, n);
        return;
    }
#endif
#if defined(BONSAI_SORT_X86_SIMD_SORT)
    sort_simd(keys, perm, n);
#else
    if (n < 1024) {
        sort_std(keys, perm, n);
    } else {
        sort_radix(keys, perm, n);
    }
#endif
}
