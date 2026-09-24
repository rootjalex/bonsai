#pragma once

#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// Loop-invariant code motion for storage: an allocation made inside a loop
// whose every iteration could reuse one allocation is made once, before the
// loop.
//
// A queue family a pass owns -- `render[VolPath].queue(s)`, the sample loop
// outside the pixel loop -- has its storage made where the deferral put it,
// at the head of the pass's iteration: pixels times entries, once per pass.
// Nothing in that storage outlives the pass, the queues being emptied before
// they are filled, so one allocation before the sample loop serves every
// pass, as pbrt allocates its queues once. Halide's `hoist_storage` is the
// same move written as a schedule directive ("moves an actual allocation to
// a given loop level", src/Func.h) and its `HoistedStorage` lowering
// (src/StorageFlattening.cpp) the same mechanics; here it is a pass, because
// what it needs to know -- that the storage is dead when an iteration starts
// -- the rewrite that made the storage already knows and says.
//
// WHAT IS HOISTED, AND HOW FAR
//
// An `Alloca` in a block inside a loop is moved outward, one loop at a time,
// while three things hold:
//
//   * Its size is available where it is going. An SSA allocation has no
//     operands; a run-time size is a value named in its type (`f32[n]`, see
//     make_alloca in SSA/Storage.h). Each such name is settled to its
//     definition (SSA/Definitions.h), which has to be in a block dominating
//     the target: a loop index, a loop-carried argument, a value computed
//     inside the loop stops the allocation at that loop. Halide's
//     `hoist_storage` bounds a size that varies over the crossed loops
//     instead (`bounds_of_expr_in_scope` over each loop variable); a size
//     that varies is not something a queue has, so the settled value or
//     nothing.
//   * Nothing reads what the storage held when an iteration began. Either
//     the allocation is marked `scratch` by the rewrite that made it -- a
//     queue is emptied before it is filled, a record slot written by its
//     producer before its pass reads it, an expansion array written by the
//     prologue loop before the nest reads it -- or, for a pointer-typed slot,
//     the whole slot is stored in a block that dominates every other use, so
//     each iteration writes before it reads whatever the address goes on to.
//     LLVM's `llvm.lifetime` markers are the same contract: the producer
//     states the lifetime and the analysis trusts it. An array nothing marked
//     is not hoisted; the SSA form has no whole-array store to prove it by,
//     and a user's `mut array` local is left where the program declared it.
//   * The loop is one whose iterations run one after another: a `while`, a
//     loopified recursion, or a parfor no bind put anywhere, which the code
//     generators run as a counted loop. A bound parfor is never crossed --
//     its iterations are threads, and one allocation shared between them is
//     Halide's "stored outside the parallel loop but computed within it,
//     a potential race" (src/ScheduleFunctions.cpp), refused there too. Per-
//     thread storage across a bound loop is a different construct: the
//     deferral's record, Halide's `ring_buffer`.
//
// The target for a natural loop is its header's immediate dominator, the
// block every path into the loop passes through last -- the preheader's
// position, where Halide's LICM wraps its lifted lets around the `For` (src/
// LICM.cpp) -- and for an unbound parfor the block that holds it. The
// allocation is appended to the target's instructions, after whatever there
// defines its size, and its type's names are rewritten to the values as the
// target refers to them. It is moved once, to the outermost block it may
// reach; the iterations inside then share it.
//
// WHERE IT RUNS
//
// After every directive and after the allocas are promoted (SSA/
// PromoteAllocas.h), before the parfor bodies are closed. After the
// directives because a `bind` written last decides which loops may be
// crossed and a deferral written last makes storage to hoist; after the
// promotion because a `mut` scalar first written inside its loop is kept in
// memory by the promotion's must-assign rule, and hoisting it first would
// have turned a register value into a slot. Storage class is not decided
// here: what becomes of a run-time-sized array that now sits at a function's
// top is SSA/HeapArrays.h.
//
// Returns how many allocations were moved.
//
//   J. Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing
//     Parallelism, Locality, and Recomputation in Image Processing
//     Pipelines", PLDI 2013 (Section 4.1: the callee's allocation "injected
//     at some containing loop level specified by the schedule"; 4.2: hoisting
//     bounds to the outermost level possible).
//   Halide, src/LICM.cpp (loop-invariant values lifted to the loop's
//     preheader, never across a GPU block/thread boundary), src/Func.h
//     `Func::hoist_storage`, src/StorageFlattening.cpp `HoistStorage`, src/
//     ScheduleFunctions.cpp (the parallel-loop refusal).
//   A. Aho, M. Lam, R. Sethi and J. Ullman, "Compilers: Principles,
//     Techniques, and Tools", 2nd ed., 2006, Section 9.5.1: loop-invariant
//     code motion to a preheader, and the conditions under which a statement
//     may be moved.
size_t hoist_invariant_allocations(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
