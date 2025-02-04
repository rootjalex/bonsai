#include "CodeGen/ASM.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
// #include <llvm/Passes/PassManager.h>
#include <llvm/IR/LegacyPassManager.h>

#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include "Error.h"

namespace bonsai {
namespace codegen {

namespace {

void emit_file(std::unique_ptr<llvm::Module> _module,
               std::unique_ptr<llvm::raw_fd_ostream> &out,
               llvm::CodeGenFileType file_type) {
    std::string target_triple = llvm::sys::getDefaultTargetTriple();
    _module->setTargetTriple(target_triple);

    std::string Error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(target_triple, Error);
    if (!target) {
        internal_error << "Error: " << Error;
    }

    // Create the target machine
    llvm::TargetOptions Opts;
    llvm::TargetMachine *target_machine =
        target->createTargetMachine(target_triple, "generic", "", Opts,
                                    std::optional<llvm::Reloc::Model>());

    llvm::DataLayout target_data_layout(target_machine->createDataLayout());
    _module->setDataLayout(target_data_layout); // this can't be right...
    if (!(target_data_layout == _module->getDataLayout())) {
        internal_error
            << "Warning: module's data layout does not match target machine's\n"
            << target_data_layout.getStringRepresentation() << "\n"
            << _module->getDataLayout().getStringRepresentation() << "\n";
    }

    // Build up all of the passes that we want to do to the module.

    // NOTE: use of the "legacy" PassManager here is still required; it is
    // deprecated for optimization, but is still the only complete API for
    // codegen as of work-in-progress LLVM14. At the time of this comment (Dec
    // 2021), there is no firm plan as to when codegen will be fully available
    // in the new PassManager, so don't worry about this 'legacy' tag until
    // there's any indication that the old APIs start breaking.
    //
    // See:
    // https://lists.llvm.org/pipermail/llvm-dev/2021-April/150100.html
    // https://releases.llvm.org/13.0.0/docs/ReleaseNotes.html#changes-to-the-llvm-ir
    // https://groups.google.com/g/llvm-dev/c/HoS07gXx0p8
    llvm::legacy::PassManager pass_manager;

    pass_manager.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(_module->getTargetTriple())));

    // Make sure things marked as always-inline get inlined
    pass_manager.add(llvm::createAlwaysInlinerLegacyPass());

    if (target_machine->isPositionIndependent()) {
        std::cout << "Target machine is Position Independent!\n";
    }

    // Override default to generate verbose assembly.
    target_machine->Options.MCOptions.AsmVerbose = true;

    // Ask the target to add backend passes as necessary.
    target_machine->addPassesToEmitFile(pass_manager, *out, nullptr, file_type);

    pass_manager.run(*_module);
}

} // namespace

void to_asm(const std::string &filename, const ir::Program &program,
            CodeGen_LLVM *gen) {
    auto _module = gen->compile_program(program);
    auto _context = gen->steal_context();

    auto out = make_raw_fd_ostream(filename);
    emit_file(std::move(_module), out, llvm::CodeGenFileType::AssemblyFile);
}

} // namespace codegen
} // namespace bonsai
