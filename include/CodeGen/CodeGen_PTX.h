#pragma once

#include "CodeGen/CodeGen_LLVM.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace codegen {

// `-b ptx`: the device side of a program. Compiles the program as `-b llvm`
// would and prints, instead of the host module, the LLVM IR of the device
// module its GPU-bound loops became and then the PTX LLVM made of it. An
// error for a program no loop of which is bound to the GPU.
void to_ptx(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen

// The LLVM code generator for NVIDIA GPUs, by way of PTX: Halide's
// CodeGen_PTX_Dev, for bonsai. It is the *device* side of a program. The host
// side (CodeGen_LLVM, or CodeGen_X86) compiles the program's functions for
// the CPU and, on meeting a parfor a schedule bound to GPUBlock or GPUThread,
// hands the loop to this generator, which cuts the loop's body out of the
// function's block graph as a kernel of the device module and compiles into
// that module every function the body reaches. Once the host's functions are
// all generated the module is finished -- CUDA's libdevice linked in for the
// maths, LLVM's optimizer run for the NVPTX machine, PTX text emitted -- and
// the text is embedded in the host module, where each launch names it and
// its kernel (see CodeGen_LLVM::emit_gpu_launch and runtime/bonsai_cuda.h).
//
// A kernel is the body of one bound loop, so its shape follows the binds. A
// loop bound to GPUBlock runs one *block* per iteration, and the loop bound
// to GPUThread inside it, if there is one, one thread per iteration of that:
// the grid is the outer count, the block is the inner, and both are exact,
// so no thread has to ask whether it is past the end. The block loop's body
// outside the thread loop is run by every thread of the block, because a
// block is nothing but its threads; a value computed there is computed by
// each of them alike, and a side effect there -- a store, an accumulate, a
// print -- is made by thread zero alone with a barrier on either side of the
// thread loop, which is Halide's rule for the same code. A loop bound to
// GPUThread with no block loop around it is one block of that many threads.
//
// This is the software path: the traversals are the loops the schedule made
// of them, on the SM. The RT cores come with CodeGen_OptiX, a subclass of
// this, whose kernels are OptiX programs.
struct CodeGen_PTX : public CodeGen_LLVM {
    CodeGen_PTX();

    // Opens the device module for `program`: the module, the NVPTX machine
    // for the GPU `options` name (or this machine's own), the struct types.
    // Once per host module, on its first GPU-bound loop.
    void begin(const ir::Program &program, const CompilerOptions &options);

    // What the host has to know about a kernel to launch it.
    struct Kernel {
        std::string name;
        // How many bytes each parameter occupies as the device declared it,
        // in order: begin, stride, then the captures. The host checks its
        // own layout of each against this before it hands the driver a slot.
        std::vector<uint64_t> param_bytes;
        // For a loop bound to GPUBlock: the loop bound to GPUThread inside
        // its body, whose iteration count is the block's size; null when
        // there is none, and the block is one thread.
        const ir::ssa::Terminator::ParFor *thread_loop = nullptr;
    };

    // Adds the body of `loop` -- a parfor of `host` bound to GPUBlock or
    // GPUThread -- to the module as a kernel, and every function the body
    // reaches as a device function. The kernel takes the loop's begin and
    // stride and then the body's captures, in the order the body block
    // declares them after its index (CodeGen_LLVM::launch_captures: the
    // random generator's state is not one, each thread seeding its own). A
    // capture the host marks in `by_value`, one flag per capture, is a
    // pointer to a struct the kernel never writes: it comes as the struct
    // itself, a kernel parameter rather than memory the launch would have to
    // copy over, and the kernel's prologue gives it an address in a local of
    // its own.
    Kernel add_kernel(const ir::ssa::Function &host,
                      const ir::ssa::Terminator::ParFor &loop,
                      const std::vector<bool> &by_value);

    // Links libdevice for whatever maths the module calls, optimizes, and
    // emits the PTX. Nothing may be added after.
    void finish();

    // After finish(): the PTX text, and the optimized LLVM IR it was made
    // from (printed before the PTX backend's own passes rewrite the module).
    const std::string &ptx() const { return ptx_text; }
    const std::string &optimized_ir() const { return ir_text; }

    // The thread loop directly inside `loop`'s body, or null when there is
    // none; an error for more than one, which one block size cannot serve.
    static const ir::ssa::Terminator::ParFor *
    nested_thread_loop(const ir::ssa::Function &host,
                       const ir::ssa::Terminator::ParFor &loop);

  protected:
    // The NVPTX machine: `nvptx64-nvidia-cuda`, the `sm_NN` asked for or the
    // GPU in this machine, and the PTX ISA version that SM needs.
    std::unique_ptr<llvm::TargetMachine>
    make_target_machine(llvm::Module &module,
                        const CompilerOptions &options) override;
    void init_module() override;
    // MarkDeviceMemory at the end of the pipeline (see that header).
    void register_backend_passes(llvm::PassBuilder &pb) override;
    // None: nvcc's --use_fast_math is not LLVM's flags (set_device_arithmetic
    // in CodeGen_PTX.cpp says what it is instead).
    llvm::FastMathFlags fast_math_flags() override;
    // `tex_sample_grad_2d` as `tex.grad.2d.v4.f32.f32` on a texture object.
    llvm::Value *codegen_texture_sample(const ir::Intrinsic *node) override;

    // A loop bound to GPUThread met inside a block loop's kernel: the
    // thread's iteration, between two barriers. Anything else bound is an
    // error here -- a block loop inside a kernel, or a CPU thread.
    void emit_bound_parfor(BoundLoop &loop) override;
    // `tid.x == 0` at the block level of a kernel; null elsewhere.
    llvm::Value *effects_once_guard() override;
    // Device scope: `atom.gpu`, what CUDA's atomicAdd is, rather than the
    // system-scope `atom.sys` LLVM's default names.
    llvm::SyncScope::ID atomic_scope() override;
    // The address cast to the global address space when it is rooted at a
    // kernel argument -- device memory, by construction -- so that a float
    // accumulate is `atom.add.f32` and not a compare-and-swap loop.
    llvm::Value *atomic_address(llvm::Value *loc) override;
    bool effects_run_once() const override { return block_level; }
    void check_block_level_call(const std::string &callee) override;

    // libdevice's `__nv_sinf`, `__nv_exp`, ...: what CUDA's own `sinf` is,
    // since NVPTX has no lowering of `llvm.sin.f32` by itself. A vector is
    // a call per lane, as a GPU thread computes it.
    llvm::Value *codegen_math_call(const std::string &name,
                                   const ir::Intrinsic *node) override;
    // `vprintf(format, args)`: the device's printf takes its arguments
    // packed in memory, each at its natural alignment.
    llvm::Value *emit_printf(llvm::Value *format,
                             const std::vector<llvm::Value *> &args) override;
    // A device has no pthreads; a dynamic array that would lock one is an
    // error.
    llvm::FunctionCallee get_pthread_lock() override;
    llvm::FunctionCallee get_pthread_unlock() override;
    llvm::FunctionCallee get_pthread_init() override;

  private:
    // Every function `roots` reach, declared and then compiled into the
    // module, once each; the kernel's body is compiled after them so that
    // its calls find their callees.
    void compile_reachable(const std::vector<std::string> &roots);
    // The functions the blocks of `region` call.
    static std::vector<std::string>
    callees(const std::vector<std::shared_ptr<ir::ssa::Block>> &region);
    // Whether `function` -- or anything it calls -- has a side effect: what
    // decides if a call at the block level of a kernel may be made by every
    // thread.
    bool has_effects(const std::string &function);

    // The special registers.
    llvm::Value *thread_index();
    llvm::Value *block_index();
    // `bar.sync 0`: every thread of the block, and their writes so far.
    void barrier();

    // The libdevice bitcode, wherever the CUDA toolkit put it.
    static std::string find_libdevice();
    void link_libdevice();

    const ir::Program *program = nullptr;
    const CompilerOptions *options = nullptr;
    std::string gpu_arch;
    // Functions already in the module, and what is known of their effects.
    std::set<std::string> declared;
    std::map<std::string, bool> effects;
    // True while emitting the body of a block loop outside its thread loop:
    // where every thread runs the same code and effects are made once. Set
    // by add_kernel, cleared for the thread loop's body.
    bool block_level = false;
    unsigned kernel_count = 0;
    std::string ptx_text, ir_text;
};

} // namespace bonsai
