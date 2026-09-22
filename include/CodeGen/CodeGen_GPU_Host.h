#pragma once

#include "CodeGen/CodeGen_LLVM.h"
#include "CodeGen/CodeGen_PTX.h"

#include <memory>
#include <string>

namespace bonsai {

// What the backends that print the device side need of a GPU host, without
// knowing which CPU it is over: see codegen::to_ptx.
struct CodeGen_GPU_Host_Interface {
    virtual ~CodeGen_GPU_Host_Interface() = default;
    // The device code generator the program's GPU-bound loops were compiled
    // through, once compile_program has run; null when there were none.
    virtual CodeGen_PTX *device_codegen() const = 0;
};

// The host side of a program with loops on the GPU: Halide's
// CodeGen_GPU_Host<CodeGen_CPU>, for bonsai. A subclass of whichever code
// generator the host machine has -- CodeGen_X86, or the target-agnostic
// CodeGen_LLVM -- that adds what the GPU asks of a host and nothing else,
// so that a program with no loop on the GPU is compiled by the plain CPU
// class and never sees any of this (see make_llvm_codegen).
//
// A parfor a schedule bound to GPUBlock or GPUThread is handed to the device
// generator, CodeGen_PTX, which cuts its body out of the block graph as a
// kernel of one device module shared by every such loop of the program; here,
// where the loop was, its launch is emitted: one call into
// runtime/bonsai_cuda.h with the PTX text, the kernel's name, the grid and
// block sizes, and the kernel's arguments -- the loop's begin and stride,
// then the body's captures -- each in a slot of its own whose address the
// runtime hands the driver. A capture that is an address is a buffer the
// runtime moves to the device for the launch and back after it: the memory
// it names is the host's, a local of the function or an argument the driver
// passed, and the schedule, in putting the loop on the GPU, is what moves the
// data. Once every function of the host module is generated the device
// module is finished and its PTX embedded as a string the launches load.
template <typename CodeGen_CPU>
struct CodeGen_GPU_Host : public CodeGen_CPU,
                          public CodeGen_GPU_Host_Interface {
    CodeGen_GPU_Host() = default;

    std::unique_ptr<llvm::Module>
    compile_program(const ir::Program &program,
                    const CompilerOptions &options) override;

    CodeGen_PTX *device_codegen() const override { return device.get(); }

  protected:
    // The GPU bindings become a kernel and a launch; everything else is the
    // CPU's answer.
    void emit_bound_parfor(CodeGen_LLVM::BoundLoop &loop) override;
    // Finishes the device module and embeds its PTX where the launches look.
    void end_functions() override;

  private:
    using CodeGen_CPU::builder;
    using CodeGen_CPU::codegen_expr;
    using CodeGen_CPU::codegen_type;
    using CodeGen_CPU::context;
    using CodeGen_CPU::create_alloca_at_entry;
    using CodeGen_CPU::i64_t;
    using CodeGen_CPU::module;
    using CodeGen_CPU::trip_count;
    using CodeGen_CPU::void_t;

    void emit_gpu_launch(CodeGen_LLVM::BoundLoop &loop);
    // How many bytes the memory a kernel capture points at occupies: what
    // the launch hands the runtime to copy. An error for a pointee that
    // itself holds pointers, which is a structure the layout language owns
    // and will place on the device itself.
    llvm::Value *capture_bytes(const ir::Type &type, const std::string &name);

    // The program and options being compiled, for the device module, which
    // is opened on the first bound loop part-way through compile_program.
    const ir::Program *program = nullptr;
    const CompilerOptions *options = nullptr;
    std::unique_ptr<CodeGen_PTX> device;
    // `internal constant ptr`: the address of the PTX string, which does not
    // exist until the device module is finished, so a launch loads it from
    // here and the initializer is filled in at the end.
    llvm::GlobalVariable *ptx_source = nullptr;
};

} // namespace bonsai
