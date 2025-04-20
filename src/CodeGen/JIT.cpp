#include "CodeGen/JIT.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

namespace bonsai {
namespace codegen {

void jit(const ir::Program &program, const CompilerOptions &options) {
    internal_assert(program.externs.empty())
        << "[unimplemented] JIT with bonsai externs";
    CodeGen_LLVM codegen;
    std::unique_ptr<llvm::orc::LLJIT> JIT =
        llvm::cantFail(llvm::orc::LLJITBuilder().create());
    internal_assert(JIT != nullptr) << "Failed to generate JIT";

    std::unique_ptr<llvm::Module> module =
        codegen.compile_program(program, options);
    module->setDataLayout(JIT->getDataLayout());
    std::unique_ptr<llvm::LLVMContext> context = codegen.steal_context();

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
