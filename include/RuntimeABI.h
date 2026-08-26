#pragma once

#include <cstddef>

#include <pthread.h>

namespace bonsai {

// Layout constants shared between the compiler and the code it generates.
//
// Anything here ends up in generated IR, so it must not depend on the machine
// the compiler happens to run on: a value read from the host (a `sizeof`, an
// alignment) would make the IR differ between platforms, break
// cross-compilation, and make every IR golden test host-specific.

// Words reserved inside a dynamic array for the pthread mutex that guards
// growing its buffer. Deliberately larger than any platform needs -- 40 bytes
// on glibc, 64 on macOS -- so that the reservation is a constant rather than
// a property of the host.
constexpr size_t BONSAI_MUTEX_RESERVED_WORDS = 8;

static_assert(sizeof(pthread_mutex_t) <= BONSAI_MUTEX_RESERVED_WORDS * 8,
              "A pthread_mutex_t does not fit in the space dynamic arrays "
              "reserve for it; raise BONSAI_MUTEX_RESERVED_WORDS.");
static_assert(alignof(pthread_mutex_t) <= 8,
              "A pthread_mutex_t needs more alignment than the reservation "
              "guarantees.");

} // namespace bonsai
