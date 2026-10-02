#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

// The storage a schedule chose for a queue (`queue(l, cap, ExternDevice)`,
// IR/Storage.h), checked against the hardware and, for the Extern kinds,
// made the caller's.
//
// A queue's storage is the arrays and header a deferral made (SSA/Defer.h),
// each an allocation carrying the choice (Instruction::storage). The choice
// names a memory with a level in the hardware hierarchy -- the host's heap
// or stack, which only host code addresses; the device's global memory,
// which only kernels address and the host reaches by copies; managed memory,
// which both address -- and the check is that every access the function
// makes of the storage is from a level that can make it:
//
//   * Host memory (Heap, Stack, ExternHost) is never touched inside a loop
//     bound to the GPU, nor handed to one as a capture: a kernel cannot
//     dereference a host address.
//   * Device memory (DeviceGlobal, ExternDevice) carries no entry read or
//     written by host code. What the host may do to it is the header
//     protocol -- fill the header, read or zero its count, pick a slot --
//     which code generation turns into copies (CodeGen_LLVM::
//     load_from_device, store_to_device); a drain on the CPU or a push from
//     host code would read the entries themselves.
//   * Managed memory (Managed, ExternManaged) is addressed by both, so
//     anything goes.
//   * Storage placed inside a loop bound to CPU threads is that thread's:
//     an access from a different thread-bound loop nested inside it is a
//     second thread writing one thread's queue. (The per-thread queue is a
//     placement, `queue(<loop bound to CPUThread>)`, not a storage word.)
//   * DeviceShared is per block the same way, and is refused until the PTX
//     backend allocates shared memory for a queue (apps/pbrt/PLAN.md).
//   * Extern storage is the caller's, so there is one of it per call: the
//     allocation has to sit in the function's top region, outside every
//     loop (as a queue placed at a loop the hoisting pass could cross does
//     once hoisted, SSA/HoistAllocations.h), and the function has to be
//     exported, since the caller is the program's driver and no function of
//     the program passes queue storage to another.
//
// An Extern queue's arrays then become parameters of the function: an
// entry-block argument per array, of the array's type (its capacity a
// constant, so that the header the C++ backend prints can say its size in
// bytes, `BONSAI_<function>_<array>_BYTES`), mutating, carrying the storage
// so that the exported prologue binds the name to the device's pointer for
// device and managed memory (CodeGen_LLVM_SSA.cpp). The header stays the
// function's own allocation, in the memory the Extern kind names: it is a
// few words the function fills with the arrays' addresses and the count,
// and the caller has no business with it.
//
// Runs after the hoisting pass and before the heap pass, so that what is
// checked is where the storage finally lands and the Extern arrays are
// parameters before anything decides their memory. Every refusal is an
// error naming the queue, the storage word, the loop and its binding
// (tests/bonsai/error/queue-storage-*).
//
// Returns how many arrays became parameters.
size_t place_queue_storage(Function &func, bool exported);

} // namespace ssa
} // namespace ir
} // namespace bonsai
