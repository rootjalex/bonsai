#include "CodeGen/JIT.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include "bonsai_buffer.h"
#include "bonsai_cuda.h"
#include "bonsai_optix.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
// The sort of a queue's entry indices by their keys for a program run under
// the JIT (ir::Intrinsic::sort_indices): what runtime/bonsai_parallel.h's
// bonsai_sort_indices does, by the standard library, since that header's
// sort runs on TBB wherever TBB is installed and the compiler links no TBB.
// Same meaning: perm[0..n) holds 0..n-1 by ascending keys[index], ties in
// index order; the key's live bits (`bits`) are a radix sort's business and
// nothing here.
void jit_sort_indices(const uint64_t *keys, uint32_t *perm, int64_t n, uint32_t bits) {
    (void)bits;
    std::vector<uint32_t> order(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; i++) {
        order[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    for (int64_t i = 0; i < n; i++) {
        perm[i] = order[static_cast<size_t>(i)];
    }
}
} // namespace

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

namespace bonsai {
namespace codegen {

void jit(const ir::Program &program, const CompilerOptions &options) {
    // What the JIT cannot do is *supply* an extern: it calls `main()` with no
    // arguments and has nowhere to get an array from. So the question is
    // whether `main` ends up needing one, not whether the program declares any.
    //
    // Those are different questions, and the difference is a whole class of
    // program: LowerExterns gives an extern to the functions that read it and
    // to their callers, so a `main` that touches none of them takes no extra
    // arguments however many the file it imported declared. Asking the coarser
    // question meant that importing a library was enough to make a program
    // unrunnable under `-e` -- which is what happened when the BSDF library
    // gained a texture table that its BSDF tests never look at.
    const auto entry = program.funcs.find("main");
    if (entry != program.funcs.end()) {
        internal_assert(entry->second->args.empty())
            << "[unimplemented] JIT of a main() that takes arguments: it needs "
            << entry->second->args.size()
            << ", which are the externs it reads and which nothing here can "
               "supply. Compile to a backend and link a driver instead.";
    }
    std::unique_ptr<CodeGen_LLVM> codegen = make_llvm_codegen(program, options);
    std::unique_ptr<llvm::orc::LLJIT> JIT =
        llvm::cantFail(llvm::orc::LLJITBuilder().create());
    internal_assert(JIT != nullptr) << "Failed to generate JIT";

    // The runtime the generated code calls into is the one linked into this
    // process. libc's symbols the JIT finds on its own, since they are in the
    // dynamic symbol table; the runtime's are in a static library and are
    // not, so they are defined here by address.
    {
        llvm::orc::SymbolMap runtime;
        const auto define = [&](const char *name, auto *fn) {
            runtime[JIT->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(fn),
                llvm::JITSymbolFlags::Exported);
        };
        define("bonsai_cuda_launch", &bonsai_cuda_launch);
        define("bonsai_cuda_load", &bonsai_cuda_load);
        // Device memory for the allocations only kernels touch (see
        // CodeGen_LLVM::device_resident).
        define("bonsai_cuda_malloc", &bonsai_cuda_malloc);
        define("bonsai_cuda_malloc_managed", &bonsai_cuda_malloc_managed);
        define("bonsai_cuda_free", &bonsai_cuda_free);
        // Host reads and writes of device-resident memory (a queue's header
        // drained on the GPU): the count read between rounds, zeroed before
        // one, and the header filled at the start (CodeGen_LLVM::
        // load_from_device, store_to_device).
        define("bonsai_cuda_copy_to_host", &bonsai_cuda_copy_to_host);
        define("bonsai_cuda_copy_to_device", &bonsai_cuda_copy_to_device);
        define("bonsai_cuda_store_word_async", &bonsai_cuda_store_word_async);
        define("bonsai_cuda_store_async", &bonsai_cuda_store_async);
        // A byte fill of a device-resident array (the record's `ran` bytes
        // cleared as the owner's iteration starts; CodeGen_LLVM's fill_bytes).
        define("bonsai_cuda_fill_bytes_async", &bonsai_cuda_fill_bytes_async);
        // The round loop's pipelined guard (SSA/RoundGuard.h): the count
        // tested between rounds, snapshotted on the stream.
        define("bonsai_cuda_round_guard", &bonsai_cuda_round_guard);
        // The wait at the return of a function that launched: every launch
        // is asynchronous (CodeGen_LLVM::wait_for_device_if_launched).
        define("bonsai_cuda_synchronize", &bonsai_cuda_synchronize);
        // The ray tracing hardware (runtime/bonsai_optix.h): the launches a
        // loop bound to OptixThread makes, and the module load ahead of
        // them; a program's driver builds the acceleration structures.
        define("bonsai_optix_launch", &bonsai_optix_launch);
        define("bonsai_optix_load", &bonsai_optix_load);
        define("bonsai_optix_ray_types", &bonsai_optix_ray_types);
        define("bonsai_optix_geometry", &bonsai_optix_geometry);
        define("bonsai_optix_scene", &bonsai_optix_scene);
        // An exported function's prologue calls these; under the JIT nothing
        // calls the exported entry (main calls its internal twin), but the
        // entry is in the module and has to link.
        define("bonsai_buffer_require", &bonsai_buffer_require);
        define("bonsai_buffer_mark_dirty", &bonsai_buffer_mark_dirty);
        // A sorted queue's permutation (ir::Intrinsic::sort_indices): on
        // the host, and on the device for a queue whose drain runs there
        // (runtime/bonsai_cuda.h, which loads runtime/bonsai_cuda_sort.cu's
        // library at the first call).
        define("bonsai_sort_indices", &jit_sort_indices);
        define("bonsai_cuda_sort_indices", &bonsai_cuda_sort_indices);
#ifdef BONSAI_CUDA_SORT_LIB
        // The device sort's library, built beside this compiler
        // (CMakeLists.txt): where the runtime finds it under the JIT.
        bonsai_cuda_set_sort_library(BONSAI_CUDA_SORT_LIB);
#endif
        llvm::cantFail(JIT->getMainJITDylib().define(
            llvm::orc::absoluteSymbols(std::move(runtime))));
    }

    std::unique_ptr<llvm::Module> module =
        codegen->compile_program(program, options);
    module->setDataLayout(JIT->getDataLayout());
    std::unique_ptr<llvm::LLVMContext> context = codegen->steal_context();

    llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
    auto err = JIT->addIRModule(std::move(tsm));
    internal_assert(!err) << llvm::toString(std::move(err)) << "\n";

    auto main_function = JIT->lookup("main");
    if (!main_function) {
        internal_error << "No main() function found, with error: "
                       << llvm::toString(main_function.takeError());
    }
    internal_assert(!main_function->isNull());
    auto *main = main_function->toPtr<void (*)()>();
    main();
}

} // namespace codegen
} // namespace bonsai
