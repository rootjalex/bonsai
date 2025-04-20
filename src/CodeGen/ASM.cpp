#include "CodeGen/ASM.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include <string>

namespace bonsai {
namespace codegen {

namespace {

void emit_file(const std::string &filename,
               std::unique_ptr<llvm::Module> module,
               llvm::CodeGenFileType file_type) {
    // This is set LLVM compilation, see CodeGen_LLVM::make_target_machine.
    std::string target_triple = module->getTargetTriple();
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (target == nullptr) {
        internal_error << "error: " << error;
    }

    // Create the target machine for emitting assembly.
    llvm::TargetOptions target_options;
    llvm::TargetMachine *target_machine = target->createTargetMachine(
        target_triple, "generic", "", target_options,
        std::optional<llvm::Reloc::Model>());
    module->setDataLayout(target_machine->createDataLayout());

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
        llvm::Triple(module->getTargetTriple())));

    // Make sure things marked as always-inline get inlined
    pass_manager.add(llvm::createAlwaysInlinerLegacyPass());

    if (target_machine->isPositionIndependent()) {
        std::cout << "; target machine is Position Independent!\n";
    }

    // Override default to generate verbose assembly.
    target_machine->Options.MCOptions.AsmVerbose = true;

    // Ask the target to add backend passes as necessary.
    if (!filename.empty()) {
        auto os = make_raw_fd_ostream(filename);
        target_machine->addPassesToEmitFile(pass_manager, *os, nullptr,
                                            file_type);
    } else {
        // Print this to standard I/O.
        target_machine->addPassesToEmitFile(pass_manager, llvm::outs(), nullptr,
                                            file_type);
    }

    pass_manager.run(*module);
}

} // namespace

void to_asm(const ir::Program &program, const CompilerOptions &options) {
    CodeGen_LLVM codegen;
    std::unique_ptr<llvm::Module> result =
        codegen.compile_program(program, options);
    std::unique_ptr<llvm::LLVMContext> context = codegen.steal_context();
    emit_file(options.output_file, std::move(result),
              llvm::CodeGenFileType::AssemblyFile);
}

} // namespace codegen
} // namespace bonsai
