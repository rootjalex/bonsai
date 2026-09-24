#pragma once

#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// A run-time-sized array that a function makes once per call lives on the
// heap, and the function frees it before it returns.
//
// The stack is where an `Alloca` goes, and for a queue family sized by a
// pixel's samples, made per pixel inside a thread, that is right: a few
// hundred kilobytes per iteration, gone when the iteration is. Hoisted to
// the function's top (SSA/HoistAllocations.h) the same family is sized by
// the frame -- pixels times every field of every entry, hundreds of
// megabytes for a scene of ordinary size -- and no stack holds that. pbrt's
// wavefront allocates its queues once, before rendering; Halide's pipelines
// allocate what a call needs on the heap through `halide_malloc` and free it
// before the call returns, keeping only small constant-sized buffers on the
// stack (src/CodeGen_LLVM.cpp, `create_alloca_at_entry` for those; src/
// Lower.cpp's `inject_host_dev_buffer_copies` and the runtime for the rest).
// This pass is Halide's rule: an array whose size is a run-time value, in a
// block of the function's top region that runs once per call, becomes an
// `Alloc`, and a `Free` of it goes before every `Return` the block
// dominates. What stays on the stack: every allocation inside a loop or a
// bound loop's body (the code generators place those in the frame that runs
// them), every constant-sized one, and every pointer-typed slot.
//
// The two conditions for the switch, checked per array: the allocation's
// block dominates every return of the function, so that each is freed on
// every path out; and the array itself is neither returned nor stored as a
// value anywhere, so that no reference to it survives the free. An address
// into it stored in another array of the same function -- a record slot's
// address in a queue entry -- is fine, both being freed at the same return.
//
// `--no-heap` (CompilerOptions::no_heap) exists because an allocation
// nothing frees is a leak the moment it is reached twice, and a renderer's
// loop reaches everything twice. An allocation this pass makes is freed on
// every return, so the LLVM backend admits it under the flag and goes on
// refusing every other heap allocation (see create_malloc). The cost that
// remains is the allocation itself, once per call, and the page faults of
// first touch inside the call -- pbrt pays the same faults in its first
// pass, having allocated before its timer.
//
// Runs after the hoist and before the parfor bodies are closed; returns how
// many arrays were moved to the heap.
size_t heap_arrays(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
