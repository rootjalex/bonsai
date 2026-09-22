#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bonsai_cuda.h"

// A buffer an exported function is handed: where its bytes are, and which
// copy is current.
//
// An exported function's array parameters arrive as one of these each rather
// than as a bare pointer, because a bare pointer says where the bytes are on
// one side only. A program whose schedule puts a loop on the GPU reads the
// same array on the device, and the question of when it is copied there --
// and whether a render's timed region contains that copy -- is the driver's
// to answer, not the compiler's to guess. This is Halide's `halide_buffer_t`
// with its host and device pointers and its two dirty bits, cut down to what
// bonsai needs:
//
//   - `host` and `device` are the two copies, either of which may not exist
//     yet (null). `bytes` is how long the buffer is; BONSAI_BUFFER_UNSIZED
//     when the driver did not say, which is enough for a host-only use and
//     an error for anything that has to move it.
//   - `host_dirty` says the host copy has writes the device copy lacks, and
//     `device_dirty` the reverse. At most one is set.
//
// The compiled function calls `bonsai_buffer_require(b, side)` before it
// reads or writes the buffer on a side, and `bonsai_buffer_mark_dirty(b,
// side)` after it may have written there: `require` returns the pointer for
// that side, allocating and copying only if that side's copy is missing or
// stale, so a buffer that is already where it is needed costs a flag test.
// A driver that wants a timed region free of transfers stages its buffers
// first (`bonsai_buffer_stage`, per the sides the generated header lists for
// each function) and then forbids implicit copies
// (`bonsai_buffer_implicit_copies(0)`): a `require` that would have to copy
// then stops the program and says which buffer and which side, rather than
// quietly timing a transfer. A driver that does not care gets the copies
// made for it, lazily, as Halide's pipelines do.
//
// Header-only, like runtime/bonsai_parallel.h, so that a driver has the
// whole runtime by including the generated header; the compiler includes it
// once too, for the programs it runs itself.

extern "C" {

struct bonsai_buffer {
    void *host;
    void *device;
    uint64_t bytes;
    uint32_t flags;
};

// The sides. As bits, so that one byte can say a buffer is needed on both:
// what the generated header's `<function>_sides` tables say per parameter.
enum {
    BONSAI_HOST = 1,
    BONSAI_DEVICE = 2,
};

enum {
    BONSAI_BUFFER_HOST_DIRTY = 1,
    BONSAI_BUFFER_DEVICE_DIRTY = 2,
    // The host copy was allocated here (for a buffer that began on the
    // device), so bonsai_buffer_free frees it.
    BONSAI_BUFFER_HOST_OWNED = 4,
};

#define BONSAI_BUFFER_UNSIZED UINT64_MAX

// Whether `require` may copy. On by default; a driver turns it off around a
// timed region, having staged its buffers.
__attribute__((used)) inline int *bonsai_buffer_implicit_copies_flag(void) {
    static int allowed = 1;
    return &allowed;
}

__attribute__((used)) inline void bonsai_buffer_implicit_copies(int allowed) {
    *bonsai_buffer_implicit_copies_flag() = allowed;
}

// A buffer over memory the driver owns, resident on the host and clean.
__attribute__((used)) inline bonsai_buffer bonsai_buffer_wrap(void *host,
                                                              uint64_t bytes) {
    bonsai_buffer b;
    b.host = host;
    b.device = NULL;
    b.bytes = bytes;
    b.flags = 0;
    return b;
}

__attribute__((used, noreturn)) inline void
bonsai_buffer_fail(const bonsai_buffer *b, const char *what) {
    fprintf(stderr,
            "bonsai_buffer: %s (buffer of %llu bytes, host %p, device %p, "
            "flags %u)\n",
            what, (unsigned long long)b->bytes, b->host, b->device,
            (unsigned)b->flags);
    fflush(stderr);
    abort();
}

// The pointer for `side`, with that side's copy made current: allocated if
// it does not exist, copied into if the other side has writes it lacks. The
// copy is refused when implicit copies are off, unless `explicit_stage`.
__attribute__((used)) inline void *
bonsai_buffer_require_impl(bonsai_buffer *b, int side, int explicit_stage) {
    const int may_copy = explicit_stage || *bonsai_buffer_implicit_copies_flag();
    if (side == BONSAI_HOST) {
        if (b->host != NULL && !(b->flags & BONSAI_BUFFER_DEVICE_DIRTY)) {
            return b->host;
        }
        if (!may_copy) {
            bonsai_buffer_fail(
                b, "needed on the host but current on the device, and "
                   "implicit copies are off: stage it with "
                   "bonsai_buffer_stage(b, BONSAI_HOST) first");
        }
        if (b->bytes == BONSAI_BUFFER_UNSIZED) {
            bonsai_buffer_fail(b, "needed on the host from the device, but its "
                                  "size was never given");
        }
        if (b->host == NULL) {
            b->host = malloc(b->bytes == 0 ? 1 : (size_t)b->bytes);
            if (b->host == NULL) {
                bonsai_buffer_fail(b, "malloc of the host copy failed");
            }
            b->flags |= BONSAI_BUFFER_HOST_OWNED;
        }
        if (b->device != NULL && (b->flags & BONSAI_BUFFER_DEVICE_DIRTY)) {
            bonsai_cuda_copy_to_host(b->host, b->device, b->bytes);
        }
        b->flags &= ~(uint32_t)BONSAI_BUFFER_DEVICE_DIRTY;
        return b->host;
    }
    if (side == BONSAI_DEVICE) {
        if (b->device != NULL && !(b->flags & BONSAI_BUFFER_HOST_DIRTY)) {
            return b->device;
        }
        if (!may_copy) {
            bonsai_buffer_fail(
                b, "needed on the device but current on the host, and "
                   "implicit copies are off: stage it with "
                   "bonsai_buffer_stage(b, BONSAI_DEVICE) first");
        }
        if (b->bytes == BONSAI_BUFFER_UNSIZED) {
            bonsai_buffer_fail(
                b, "needed on the device, but its size was never given: it "
                   "was passed as a bare pointer where the loop that reads "
                   "it runs on the GPU. Pass a bonsai_buffer_wrap(pointer, "
                   "bytes) instead");
        }
        if (b->device == NULL) {
            b->device = bonsai_cuda_malloc(b->bytes == 0 ? 1 : b->bytes);
            // A fresh device copy holds nothing yet, so it takes the host's
            // bytes whether or not the host was marked dirty.
            if (b->host != NULL && b->bytes != 0) {
                bonsai_cuda_copy_to_device(b->device, b->host, b->bytes);
            }
        } else if (b->host != NULL && b->bytes != 0) {
            bonsai_cuda_copy_to_device(b->device, b->host, b->bytes);
        }
        b->flags &= ~(uint32_t)BONSAI_BUFFER_HOST_DIRTY;
        return b->device;
    }
    bonsai_buffer_fail(b, "require of a side that is neither host nor device");
}

// What a compiled function calls before touching a buffer on a side.
__attribute__((used)) inline void *bonsai_buffer_require(bonsai_buffer *b,
                                                         int side) {
    return bonsai_buffer_require_impl(b, side, /*explicit_stage=*/0);
}

// What a compiled function calls after it may have written a buffer on a
// side: the other side's copy, if there is one, is stale from here on.
__attribute__((used)) inline void bonsai_buffer_mark_dirty(bonsai_buffer *b,
                                                           int side) {
    if (side == BONSAI_HOST) {
        if (b->flags & BONSAI_BUFFER_DEVICE_DIRTY) {
            bonsai_buffer_fail(b, "written on the host while the device copy "
                                  "held writes the host never received");
        }
        b->flags |= BONSAI_BUFFER_HOST_DIRTY;
    } else if (side == BONSAI_DEVICE) {
        if (b->flags & BONSAI_BUFFER_HOST_DIRTY) {
            bonsai_buffer_fail(b, "written on the device while the host copy "
                                  "held writes the device never received");
        }
        b->flags |= BONSAI_BUFFER_DEVICE_DIRTY;
    }
}

// The driver's own copy, made ahead of time: `require`, with the copy
// allowed whatever bonsai_buffer_implicit_copies says. `sides` may name
// both, in which case the buffer is made current on both.
__attribute__((used)) inline void bonsai_buffer_stage(bonsai_buffer *b,
                                                      int sides) {
    if (sides & BONSAI_DEVICE) {
        bonsai_buffer_require_impl(b, BONSAI_DEVICE, /*explicit_stage=*/1);
    }
    if (sides & BONSAI_HOST) {
        bonsai_buffer_require_impl(b, BONSAI_HOST, /*explicit_stage=*/1);
    }
}

// Every buffer of a call to its sides, as the generated header's
// `<function>_sides` table gives them, in parameter order.
__attribute__((used)) inline void
bonsai_buffer_stage_all(bonsai_buffer **buffers, const uint8_t *sides,
                        uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        bonsai_buffer_stage(buffers[i], sides[i]);
    }
}

// Frees what the runtime allocated: the device copy, and the host copy if
// the runtime made it. Memory the driver wrapped is the driver's.
__attribute__((used)) inline void bonsai_buffer_free(bonsai_buffer *b) {
    if (b->device != NULL) {
        bonsai_cuda_free(b->device);
        b->device = NULL;
    }
    if (b->host != NULL && (b->flags & BONSAI_BUFFER_HOST_OWNED)) {
        free(b->host);
        b->host = NULL;
    }
    b->flags = 0;
}

} // extern "C"
