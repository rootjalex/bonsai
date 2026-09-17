#include "CodeGen/ASM.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Error.h"

#include <llvm/Analysis/TargetLibraryInfo.h>
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

// `target_machine` is the one the module was compiled for (see
// CodeGen_LLVM::make_target_machine): the host's CPU and features when
// nothing was named. It used to be a fresh "generic" machine here, which
// made the assembly SSE2 code for a module optimized for AVX-512 -- not the
// code the object file gets, and so useless for reading what the compiler
// emits. `library_info` is the same the optimizer worked from, so a vector
// math intrinsic still standing is lowered to the same libmvec call.
void emit_file(const std::string &filename,
               std::unique_ptr<llvm::Module> module,
               llvm::TargetMachine *target_machine,
               const llvm::TargetLibraryInfoImpl &library_info,
               llvm::CodeGenFileType file_type) {
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
    // The stream is declared before the pass manager so that it outlives it:
    // the printer the pass manager owns wraps the stream and flushes into it
    // when it is destroyed, which with the stream already gone was a write
    // through a dangling pointer after every byte of assembly was out.
    std::unique_ptr<llvm::raw_fd_ostream> os;
    llvm::legacy::PassManager pass_manager;

    pass_manager.add(new llvm::TargetLibraryInfoWrapperPass(library_info));

    // Make sure things marked as always-inline get inlined
    pass_manager.add(llvm::createAlwaysInlinerLegacyPass());

    if (target_machine->isPositionIndependent()) {
        std::cout << "; target machine is Position Independent!\n";
    }

    // Override default to generate verbose assembly.
    target_machine->Options.MCOptions.AsmVerbose = true;

    // Ask the target to add backend passes as necessary. The stream has to
    // outlive the run below, which is what writes to it: scoped to the
    // branch that opened it, it was destroyed before the first byte of
    // assembly was written into it.
    if (!filename.empty()) {
        os = make_raw_fd_ostream(filename);
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
    std::unique_ptr<llvm::TargetMachine> target_machine =
        codegen.make_target_machine(*result, options);
    const llvm::TargetLibraryInfoImpl library_info =
        codegen.target_library_info(llvm::Triple(result->getTargetTriple()));
    std::unique_ptr<llvm::LLVMContext> context = codegen.steal_context();
    emit_file(options.output_file, std::move(result), target_machine.get(),
              library_info, llvm::CodeGenFileType::AssemblyFile);
}

} // namespace codegen
} // namespace bonsai
