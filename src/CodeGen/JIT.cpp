#include "CodeGen/JIT.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

namespace bonsai {
namespace codegen {

void jit(const ir::Program &program, const CompilerOptions &options) {
    CodeGen_LLVM codegen;
    std::unique_ptr<llvm::orc::LLJIT> JIT =
        llvm::cantFail(llvm::orc::LLJITBuilder().create());
    internal_assert(JIT != nullptr) << "Failed to generate JIT";

    // TODO: use JIT->getTargetTriple() in codegen?
    auto _module = codegen.compile_program(program, options);
    auto _context = codegen.steal_context();

    // TODO: optimize module for JIT->getTargetTriple() ?

    // internal_assert(JIT->getTargetTriple().str() ==
    // _module->getTargetTriple())
    //     << "JIT and Module have different target triples: " <<
    //     JIT->getTargetTriple().str() << " versus " <<
    //     _module->getTargetTriple();

    llvm::orc::ThreadSafeModule tsm(std::move(_module), std::move(_context));
    auto err = JIT->addIRModule(std::move(tsm));
    internal_assert(!err) << llvm::toString(std::move(err)) << "\n";

    auto mainFunc = JIT->lookup("main");
    if (!mainFunc) {
        internal_error << "No main() function found, with error: "
                       << llvm::toString(mainFunc.takeError());
    }

    internal_assert(!mainFunc->isNull());

    internal_assert(program.externs.empty())
        << "TODO: implement JIT with externs!";

    auto *Main = mainFunc->toPtr<void (*)()>();
    Main();
}

} // namespace codegen
} // namespace bonsai
