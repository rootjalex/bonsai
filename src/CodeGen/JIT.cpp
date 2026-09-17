#include "CodeGen/JIT.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

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
    std::unique_ptr<CodeGen_LLVM> codegen = make_llvm_codegen(options);
    std::unique_ptr<llvm::orc::LLJIT> JIT =
        llvm::cantFail(llvm::orc::LLJITBuilder().create());
    internal_assert(JIT != nullptr) << "Failed to generate JIT";

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
