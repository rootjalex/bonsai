#include "CodeGen/JIT.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include "bonsai_buffer.h"
#include "bonsai_cuda.h"

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
        // An exported function's prologue calls these; under the JIT nothing
        // calls the exported entry (main calls its internal twin), but the
        // entry is in the module and has to link.
        define("bonsai_buffer_require", &bonsai_buffer_require);
        define("bonsai_buffer_mark_dirty", &bonsai_buffer_mark_dirty);
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
