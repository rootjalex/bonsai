#pragma once

#include "CodeGen/CodeGen_LLVM.h"
#include "SSA/ThreadPhases.h"

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
// loop bound to GPUBlock runs one *block* per iteration, and the loops bound
// to GPUThread inside its body, if there are any, run one thread per
// iteration of theirs: the grid is the outer count, the block is the largest
// of the inner counts, and each thread loop is a *phase* of the kernel that
// every thread of the block runs under `tid < extent` -- no test when the
// extent is the block's, which it is by construction for the one thread
// loop of a block or for loops whose bounds resolve to the same values
// (SSA/ThreadPhases.h) -- with a barrier on either side. The block loop's
// body outside the thread loops is run by every thread of the block,
// because a block is nothing but its threads: a value computed there is
// computed by each of them alike; an allocation made there is one per
// block, the block's shared memory (allocate_shared), since a copy per
// thread would be no use to the thread loop that reads it; and a side
// effect there -- a store, an accumulate, a print -- is made by thread zero
// alone and followed by a barrier, so that every thread reads what it
// wrote. This is Halide's FuseGPUThreadLoops (the shared allocations
// extracted, the thread loops fused to the largest extent with a predicate
// each, the barriers injected) as the device generator emits it, kernel by
// kernel. A loop bound to GPUThread with no block loop around it is one
// block of that many threads.
//
// This is the software path: the traversals are the loops the schedule made
// of them, on the SM. The RT cores come with CodeGen_OptiX, a subclass of
// this, whose kernels are OptiX programs.
struct CodeGen_PTX : public CodeGen_LLVM {
    CodeGen_PTX();

    // Opens the device module for `program`: the module, the NVPTX machine
    // for the GPU `options` name (or this machine's own), the struct types.
    // Once per host module, on its first GPU-bound loop. Virtual for
    // CodeGen_OptiX, which sets the backend's options around it.
    virtual void begin(const ir::Program &program, const CompilerOptions &options);

    // What the host has to know about a kernel to launch it.
    struct Kernel {
        std::string name;
        // How many bytes each parameter occupies as the device declared it,
        // in order: begin, stride, then the captures. The host checks its
        // own layout of each against this before it hands the driver a slot.
        std::vector<uint64_t> param_bytes;
        // For a raygen program, whose parameters cross as one struct (the
        // module's `params`) rather than one slot at a time: where the
        // device's layout of that struct puts each, and how long it is. The
        // host lays the same struct out with its own machine's rules and
        // checks that the two agree, field for field, before it copies its
        // bytes over.
        std::vector<uint64_t> param_offsets;
        uint64_t param_struct_bytes = 0;
        // For a loop bound to GPUBlock: the loops bound to GPUThread inside
        // its body, the kernel's thread phases (SSA/ThreadPhases.h), the
        // largest of whose iteration counts is the block's size; empty when
        // there is none, and the block is one thread.
        std::vector<const ir::ssa::Terminator::ParFor *> thread_loops;
        // The allocations of the block loop's body whose size is known only
        // at the launch -- a queue's arrays, sized by its capacity -- laid
        // out one after another in the block's dynamic shared memory
        // (allocate_shared), in this order, each at a sixteen-byte boundary.
        // The host adds their bytes up for the launch's shared memory from
        // the same sizes the kernel computes its offsets from: each an
        // expression over the body's captures (SSA/LaunchValues.h), which
        // the host maps to its own operands.
        struct SharedAllocation {
            std::string name;
            uint64_t element_bytes = 0;
            ir::Expr size;
        };
        std::vector<SharedAllocation> dynamic_shared;
        // For a raygen program: how many payload words its traces use --
        // two for a context by address, a promoted context's own count
        // otherwise -- which the launch hands the runtime so the pipeline
        // is compiled against it (OptixPipelineCompileOptions'
        // numPayloadValues).
        uint64_t payload_values = 2;
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

    // A kernel of one thread that stores `words` 32-bit words, each at an
    // address of its own: `(ptr p0, i32 v0, .., ptr p(n-1), i32 v(n-1))`,
    // every operand a parameter. The host's run of writes into device
    // memory -- the queue counts a round of the wavefront zeroes -- as one
    // launch (CodeGen_GPU_Host::flush_device_stores): pbrt's `GPUDo("Reset
    // queues ..", lambda)`, a lambda launched on one thread. One kernel per
    // word count, named `_bonsai_store_words_<n>`.
    Kernel add_store_kernel(size_t words);

    // Links libdevice for whatever maths the module calls, optimizes, and
    // emits the PTX. Nothing may be added after. Virtual for CodeGen_OptiX,
    // which sets the backend's options around it.
    virtual void finish();

    // After finish(): the PTX text, and the optimized LLVM IR it was made
    // from (printed before the PTX backend's own passes rewrite the module).
    // The text ends with whatever PTX `--link` named (append_linked_ptx).
    const std::string &ptx() const { return ptx_text; }
    void append_linked_ptx();
    const std::string &optimized_ir() const { return ir_text; }

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

    // A loop bound to GPUThread met inside a block loop's kernel: a phase
    // of the block -- the thread's iteration, under `tid < extent` unless
    // the phase is the whole block, between two barriers. Anything else
    // bound is an error here -- a block loop inside a kernel, or a CPU
    // thread.
    void emit_bound_parfor(BoundLoop &loop) override;
    // `tid.x == 0` at the block level of a kernel; null elsewhere.
    llvm::Value *effects_once_guard() override;
    // The barrier after an effect the leader made at the block level, so
    // that every thread of the block reads what it wrote.
    void after_once_effect() override;
    bool device_code() const override { return true; }
    // A block's shared memory for an allocation of the block loop's body
    // (Allocate::Memory::Shared): a `.shared` global of the module for a
    // size the compiler knows, as the block reduction's slots are; a run of
    // the kernel's dynamic shared memory, `extern .shared .b8
    // _bonsai_dynamic_shared[]`, for one the launch sets, at the offset the
    // allocations before it end at (Kernel::dynamic_shared). Either as a
    // generic pointer, which the NVPTX backend's address-space inference
    // takes back to `ld.shared` and `st.shared`.
    llvm::Value *allocate_shared(llvm::Type *elem_t, const ir::Expr &size,
                                 const std::string &name) override;
    // Device scope: `atom.gpu`, what CUDA's atomicAdd is, rather than the
    // system-scope `atom.sys` LLVM's default names.
    llvm::SyncScope::ID atomic_scope() override;
    // The address cast to the global address space when it is rooted at a
    // kernel argument -- device memory, by construction -- so that a float
    // accumulate is `atom.add.f32` and not a compare-and-swap loop.
    llvm::Value *atomic_address(llvm::Value *loc) override;
    // A short vector's lane at a run-time index as a chain of selects on the
    // index, since NVPTX lowers the extractelement through local memory.
    // PTX identifiers are letters, digits, `_` and `$`: a program name's
    // other characters (`!`, the mark of a copy) become `$`.
    std::string symbol_name(const std::string &name) const override;

    llvm::Value *extract_lane(llvm::Value *vec, llvm::Value *idx) override;
    // The fold of every thread's value over the block: a warp shuffle tree,
    // one shared-memory slot per warp, a barrier, the warps' totals folded
    // by every thread (see the definition).
    llvm::Value *block_reduce(ir::Intrinsic::OpType op, const ir::Type &type,
                              llvm::Value *v) override;
    // The block's thread count, `ntid.x`.
    llvm::Value *block_size();
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

  protected:
    // Protected rather than private: CodeGen_OptiX, the subclass whose
    // module OptiX runs, compiles its programs with the same machinery.
    // One of LLVM's own command-line options, set as if given on a tool's
    // command line: what the NVPTX backend reads some of its choices from
    // and nothing else. Process-wide.
    static void llvm_option(const char *name, const char *value);
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
    // True while emitting the body of a block loop outside its thread loops:
    // where every thread runs the same code and effects are made once. Set
    // by add_kernel, cleared for a thread loop's body.
    bool block_level = false;
    // The kernel being emitted (add_kernel), for the shared allocations its
    // body makes; and the thread phases of its block loop's body
    // (SSA/ThreadPhases.h), for which of them run under a predicate.
    Kernel *building = nullptr;
    std::vector<ir::ssa::ThreadPhase> phases;
    // `extern .shared .b8 _bonsai_dynamic_shared[]`, made on the first
    // allocation that wants it: the one array the launch's dynamic shared
    // memory is, which every such allocation is a run of.
    llvm::GlobalVariable *dynamic_shared = nullptr;
    unsigned kernel_count = 0;
    std::string ptx_text, ir_text;
};

} // namespace bonsai
