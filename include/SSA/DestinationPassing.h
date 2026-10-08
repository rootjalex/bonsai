#pragma once

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// Destination passing for a stack slot that is filled and then copied out:
// the slot is replaced by the place the copy goes to, and the copy is gone.
//
//   slot = alloca<(Hit*)>()
//   call traverse(..., slot)          -- writes slot when the key improves
//   ...
//   h = load(slot)
//   store gep(hits, i) h               -- the copy
//
// becomes
//
//   slot = gep(hits, i)
//   call traverse(..., slot)          -- writes hits[i] when the key improves
//
// WHY
//
// An argmin that carries a record beside its key (ir::Accumulate::alongside)
// writes the record into a slot, and the program then copies the slot into
// its answer. A slot that nothing but this function can see is promoted into
// registers by any optimizer worth the name -- LLVM's SROA after inlining,
// its LICM for the stores in a loop -- and the record becomes loop-carried
// values through the whole traversal, held across every node test and
// spilled around it. The answer's own memory is not promotable: a store into
// it that happens only when the key improves may not be hoisted into a
// store that always happens, since another thread may be looking (LLVM's
// LICM, promoteLoopAccessesToScalars: a conditional store is sunk out of a
// loop only to thread-local memory). Embree's traversal is written this way
// -- its leaf writes the caller's RayHit through a pointer -- which is why
// its record costs it no registers; this pass puts ours in the same place.
//
// WHAT IS REWRITTEN
//
// An Alloca S is a candidate when every use of it is a write of the memory
// it names or the one read that is copied out:
//
//   * written: the address of a Store, the place of an accumulate's
//     alongside pair, or an argument to a call whose callee uses the
//     parameter the same way (through further calls, recursively);
//   * read once: a single Load L, whose value flows -- through MakeStruct,
//     LoadField of the field it was put in, and block arguments -- into
//     nothing but Stores of it, whole, at one address D (the sinks).
//
// D is the destination, and the rewrite is sound when:
//
//   * D is computable where S is made: its address is a chain of GEP and
//     FieldPtr over values defined in blocks dominating S's (or earlier in
//     S's own block), rebuilt there;
//   * D's memory is otherwise untouched: nothing in the function reads
//     through a root of D (Load, an array element read, an accumulate's
//     place), every store through a root of D is to D itself, and no
//     pointer rooted at D is handed to a callee -- so the intermediate
//     writes the slot's producer now makes into D are observed by nobody,
//     and the final one is the value the copy would have stored;
//   * every path from S to the end of its region (the yield of the parfor
//     body it is in, or a return) passes a store to D, so D is never left
//     holding a partial record where the program would have left it alone;
//   * no write of S can follow a store to D, so a later write cannot undo
//     what the program's own store put there.
//
// The soundness of the aliasing conditions rests on the memory model the
// rest of this form uses (roots_of, roots_related, kernel_writes in
// SSA/Analysis.h): distinct captures and allocations name distinct memory,
// and a pointer whose root cannot be named -- one reinterpreted from an
// integer a tree layout stored -- is a reference into that layout, not into
// an export's output array. A pointer into D could only be one this function
// made, and the conditions above refuse every escape of one.
//
// WHERE IT RUNS
//
// After the allocas are promoted and the function simplified, when the
// copy is a Load and a Store with values between them rather than a chain
// of slots, and before storage is hoisted and placed. A slot the pattern
// does not fit is left alone and promoted as before.
//
// Returns how many slots were given a destination.
//
//   LLVM, lib/Transforms/Scalar/MemCpyOptimizer.cpp, performCallSlotOptzn:
//     "the call slot optimization", a memcpy out of a temporary a call
//     filled is removed and the call writes the destination -- the same
//     conditions (the destination dereferenceable at the call, not accessed
//     between the call and the copy, not captured by the callee).
//   A. Shaikhha, A. Fitzgibbon, S. Peyton Jones and D. Vytiniotis,
//     "Destination-Passing Style for Efficient Memory Management", FHPC
//     2017: a function that returns an aggregate takes the place to write
//     it instead, eliminating the copy.
//   Clang's named return value optimization and C++17's guaranteed copy
//     elision are the same move written into a language.
size_t pass_destinations(Function &func, const FuncMap &fmap);

} // namespace ssa
} // namespace ir
} // namespace bonsai
