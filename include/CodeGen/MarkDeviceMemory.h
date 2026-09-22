#pragma once

// What a kernel's memory is, said in the IR so that the NVPTX backend can
// use it. A kernel's pointer parameters, and the pointers inside the
// structs it takes by value (a layout struct of device arrays), are device
// memory: the launch put them there. Nothing in the IR says so -- they are
// generic pointers, and the backend's own inference reaches only what it
// can trace to a `byval` parameter -- so every load and store through them
// is the generic `ld`/`st`, an address-space check on every access and no
// use of the read-only path. And the buffers the kernel never writes are
// invariant for the launch, which is what selects `ld.global.nc`, the
// non-coherent load through the texture path that a scene's nodes,
// triangles, materials and lights should all be fetched with.
//
// So, per kernel, at the end of the optimization pipeline (after inlining
// and SROA have made the by-value structs' pointers `extractvalue`s of the
// parameter): every such root is cast to the global address space and back,
// which the backend's inference propagates to everything derived from it;
// and a load whose every underlying object is a root the kernel never
// stores to, never operates on atomically and never hands to a call is
// tagged `!invariant.load`. Runs on the device module only.
#include <llvm/IR/PassManager.h>

namespace bonsai {

struct MarkDeviceMemory : llvm::PassInfoMixin<MarkDeviceMemory> {
    llvm::PreservedAnalyses run(llvm::Function &function,
                                llvm::FunctionAnalysisManager &);
};

} // namespace bonsai
