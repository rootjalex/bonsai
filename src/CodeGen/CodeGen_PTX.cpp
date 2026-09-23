#include "CodeGen/CodeGen_PTX.h"

#include "CodeGen/CodeGen_GPU_Host.h"
#include "CodeGen/MarkDeviceMemory.h"
#include "IR/Analysis.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "Lower/Random.h"
#include "SSA/Analysis.h"

#include "bonsai_cuda.h"

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdlib>
#include <set>

namespace bonsai {

using namespace ir;
using ir::ssa::Argument;
using ir::ssa::Block;
using ir::ssa::Constant;
using ir::ssa::Instruction;
using ir::ssa::Terminator;
using ir::ssa::Value;

namespace codegen {

void to_ptx(const Program &program, const CompilerOptions &options) {
    // The whole program is compiled, as `-b llvm` compiles it: the device
    // module is made by the host's generator as it meets the bound loops.
    std::unique_ptr<CodeGen_LLVM> codegen = make_llvm_codegen(program, options);
    std::unique_ptr<llvm::Module> host = codegen->compile_program(program, options);
    auto *gpu_host = dynamic_cast<CodeGen_GPU_Host_Interface *>(codegen.get());
    CodeGen_PTX *device = gpu_host ? gpu_host->device_codegen() : nullptr;
    internal_assert(device)
        << "-b ptx: no loop of this program is bound to the GPU, so there is "
        << "no device code. Bind one in the schedule with `f.bind(i, GPUBlock)` "
        << "or `f.bind(i, GPUThread)`.";
    std::unique_ptr<llvm::raw_fd_ostream> file;
    llvm::raw_ostream *os = &llvm::outs();
    if (!options.output_file.empty()) {
        file = make_raw_fd_ostream(options.output_file);
        os = file.get();
    }
    *os << "; ---- device module: LLVM IR ----\n"
        << device->optimized_ir() << "\n; ---- device module: PTX ----\n"
        << device->ptx();
    // llvm::outs() is buffered and flushes on exit, which is after the test
    // runner has stopped capturing it.
    os->flush();
}

} // namespace codegen

namespace {

// The PTX ISA version an SM needs, as the NVPTX feature that asks for it.
// The driver compiles any PTX at or above the version its release knows, so
// the lowest one the SM exists in is the one every driver that has the SM
// accepts: 8.7 introduced sm_120 (CUDA 12.8), 8.6 the sm_100 family (12.7),
// 7.8 sm_90 (11.8), 7.0 sm_80 (11.0), 6.3 sm_75 (10.0).
std::string ptx_feature(const std::string &sm) {
    internal_assert(sm.rfind("sm_", 0) == 0 && sm.size() > 3)
        << "--gpu-arch takes an SM name, `sm_120` say, not `" << sm << "`";
    const int n = std::atoi(sm.c_str() + 3);
    if (n >= 120) {
        return "+ptx87";
    }
    if (n >= 100) {
        return "+ptx86";
    }
    if (n >= 90) {
        return "+ptx78";
    }
    if (n >= 80) {
        return "+ptx70";
    }
    if (n >= 75) {
        return "+ptx63";
    }
    return "+ptx60";
}

// One of LLVM's own command-line options, set as if it had been given on the
// command line of a tool: the NVPTX backend reads the precision of its float
// division and square root from two of these and from nothing else -- no
// function attribute, no target option -- so this is the only way to choose
// them. Process-wide, which is fine: they are read by the NVPTX backend
// alone.
void set_llvm_option(const char *name, const char *value) {
    auto &options = llvm::cl::getRegisteredOptions();
    const auto it = options.find(llvm::StringRef(name));
    internal_assert(it != options.end())
        << "this LLVM has no option `" << name << "`";
    internal_assert(!it->second->addOccurrence(0, name, value))
        << "LLVM refused `-" << name << "=" << value << "`";
}

// The device arithmetic `--fast-math` chooses (CompilerOptions::
// fast_math). nvcc's `--use_fast_math`, which pbrt's GPU build compiles
// with, is four flags: `--ftz=true` (finish() gives every device function
// that denormal mode), `--fmad=true` (contraction, which the NVPTX backend
// does by default), `--prec-div=false` and `--prec-sqrt=false`. The last two
// make a float division `div.full.f32` (two ulp, the full range) and `1 / x`
// `rcp.approx.f32`, and a square root `sqrt.approx.f32`, where the defaults
// are the IEEE `div.rn.f32` and `sqrt.rn.f32` -- each a check and a slow-path
// call around the fast estimate, and the megakernel had 3,285 divisions and
// 1,133 square roots of that kind. Off, the arithmetic is IEEE as on the CPU.
// Set every time, since the options are process-wide and keep their last
// value.
//
// Whichever way, a float accumulate is the hardware's `atom.add.f32`. LLVM
// lowers `atomicrmw fadd` to it only in a function that flushes denormals,
// because the instruction flushes them on global memory whatever the
// function does, and otherwise expands it to a load and a compare-and-swap
// loop -- which every film accumulate of the megakernel was, at 5.7 attempts
// per add with a block's threads all adding into one pixel
// (NVPTXISelLowering.cpp, shouldExpandAtomicRMWInIR). With fast math off
// the atomic is allowed its own flush: a denormal partial sum of radiance
// flushed to zero is not a difference anything measures, and a CAS loop per
// sample is.
void set_device_arithmetic(bool fast_math) {
    set_llvm_option("nvptx-prec-divf32", fast_math ? "1" : "2");
    set_llvm_option("nvptx-prec-sqrtf32", fast_math ? "0" : "1");
    set_llvm_option("nvptx-allow-ftz-atomics", "1");
}

// An instruction with an effect on memory or the world, as the block level
// of a kernel sees it (see CodeGen_PTX::has_effects). An allocation counts:
// memory made by one thread is that thread's.
bool has_effect(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::Store:
    case Instruction::Op::Print:
    case Instruction::Op::Append:
    case Instruction::Op::Push:
    case Instruction::Op::AtomicAdd:
    case Instruction::Op::Alloc:
        return true;
    default:
        return false;
    }
}

} // namespace

CodeGen_PTX::CodeGen_PTX() = default;

void CodeGen_PTX::init_module() {
    CodeGen_LLVM::init_module();
    module->setModuleIdentifier("bonsai_ptx");
    // libdevice's functions ask `__nvvm_reflect("__CUDA_FTZ")` whether
    // denormals are to be flushed to zero, and the NVVMReflect pass answers
    // from this flag: yes under --fast-math, which is nvcc's
    // `--use_fast_math` and so `--ftz=true` (see set_device_arithmetic), and
    // finish() gives each device function the matching denormal mode; no
    // otherwise, CUDA's own default and pbrt's CPU build's.
    module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                          uint32_t(options->fast_math ? 1 : 0));
}

std::unique_ptr<llvm::TargetMachine>
CodeGen_PTX::make_target_machine(llvm::Module &m,
                                 const CompilerOptions &opts) {
    // The GPU named, or the one in this machine: following the host, as the
    // CPU generator does with the host's CPU. With neither there is nothing
    // to follow, and generating for some GPU the machine may not have would
    // be code that looks right until it is launched.
    gpu_arch = opts.gpu_arch.empty() ? std::string(bonsai_cuda_device_arch())
                                     : opts.gpu_arch;
    internal_assert(!gpu_arch.empty())
        << "no GPU to compile for: this machine has no CUDA device the "
        << "driver can see, and no --gpu-arch was given. Name the GPU the "
        << "code is for, e.g. `--gpu-arch sm_90`.";
    const std::string features = ptx_feature(gpu_arch);
    set_device_arithmetic(opts.fast_math);

    const llvm::Triple triple("nvptx64-nvidia-cuda");
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    internal_assert(target != nullptr)
        << "this LLVM was built without the NVPTX target: " << error;
    llvm::TargetOptions target_options;
    auto *tm = target->createTargetMachine(
        triple, gpu_arch, features, target_options, llvm::Reloc::Static,
        llvm::CodeModel::Small, llvm::CodeGenOptLevel::Aggressive);
    m.setDataLayout(tm->createDataLayout());
    m.setTargetTriple(triple);
    target_triple = triple.str();
    return std::unique_ptr<llvm::TargetMachine>(tm);
}

void CodeGen_PTX::begin(const Program &program_, const CompilerOptions &options_) {
    program = &program_;
    options = &options_;
    init_module();
    no_heap = options->no_heap;
    target_machine = make_target_machine(*module, *options);
    declare_struct_types(gather_struct_types(*program));
    frames.push_frame();
}

const Terminator::ParFor *
CodeGen_PTX::nested_thread_loop(const ir::ssa::Function &host,
                                const Terminator::ParFor &loop) {
    // The body's blocks end at its yield, so this is the body and whatever
    // loops it holds.
    const ir::ssa::Cfg body(host, loop.body.name);
    const Terminator::ParFor *found = nullptr;
    for (const auto &block : body.blocks()) {
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p == nullptr || !p->binding.has_value() ||
            *p->binding != Resource::GPUThread) {
            continue;
        }
        internal_assert(found == nullptr)
            << "[unimplemented] the loop over " << loop.index
            << ", bound to GPUBlock, holds two loops bound to GPUThread ("
            << found->index << " and " << p->index << "). A launch has one "
            << "block size, so two thread loops would have to be fused into "
            << "one, which is not built yet.";
        found = p;
    }
    return found;
}

std::vector<std::string>
CodeGen_PTX::callees(const std::vector<std::shared_ptr<Block>> &region) {
    std::vector<std::string> names;
    for (const auto &block : region) {
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                internal_assert(!operand->get_type().template is<Function_t>())
                    << "[unimplemented] `" << instr->name << "` takes a "
                    << "function as a value; device code reaches functions "
                    << "by call only";
            }
        }
        std::visit(ir::ssa::overloads{
                       [&](const Terminator::Call &c) {
                           names.push_back(c.call.name);
                       },
                       [&](const Terminator::MultiCall &c) {
                           names.push_back(c.call.name);
                       },
                       [](const auto &) {},
                   },
                   block->terminator.data);
    }
    return names;
}

bool CodeGen_PTX::has_effects(const std::string &function) {
    if (const auto it = effects.find(function); it != effects.end()) {
        return it->second;
    }
    // Provisionally none, so that a recursion through this function comes
    // back with an answer; whatever else it does is found below.
    effects[function] = false;
    const auto f = program->ssa_funcs.find(function);
    internal_assert(f != program->ssa_funcs.end())
        << "`" << function << "` has no SSA form";
    bool result = false;
    for (const auto &block : f->second->blocks) {
        for (const auto &instr : block->instrs) {
            result = result || has_effect(instr->op);
        }
        std::visit(ir::ssa::overloads{
                       [&](const Terminator::Call &c) {
                           result = result || has_effects(c.call.name);
                       },
                       [&](const Terminator::MultiCall &c) {
                           result = result || has_effects(c.call.name);
                       },
                       [&](const Terminator::ParFor &p) {
                           result = result || p.binding.has_value();
                       },
                       [](const auto &) {},
                   },
                   block->terminator.data);
    }
    effects[function] = result;
    return result;
}

void CodeGen_PTX::compile_reachable(const std::vector<std::string> &roots) {
    // Declared as they are found, compiled once all are declared: a body's
    // calls have to find their callees in the module.
    std::vector<std::pair<const ir::ssa::Function *, llvm::Function *>> found;
    std::vector<std::string> work(roots);
    while (!work.empty()) {
        const std::string name = work.back();
        work.pop_back();
        if (!declared.insert(name).second) {
            continue;
        }
        const auto f = program->funcs.find(name);
        internal_assert(f != program->funcs.end())
            << "a kernel calls `" << name << "`, which the program does not define";
        const auto s = program->ssa_funcs.find(name);
        internal_assert(s != program->ssa_funcs.end())
            << "`" << name << "` has no SSA form to compile for the device. A "
            << "program with a loop on the GPU keeps every function's (see "
            << "SSA/Convert.cpp).";
        found.emplace_back(s->second.get(), declare_function(*f->second));
        for (std::string &callee : callees(s->second->blocks)) {
            work.push_back(std::move(callee));
        }
    }
    for (auto &[ssa, fn] : found) {
        compile_function(*ssa, fn);
    }
}

CodeGen_PTX::Kernel CodeGen_PTX::add_kernel(const ir::ssa::Function &host,
                                            const Terminator::ParFor &loop,
                                            const std::vector<bool> &by_value) {
    internal_assert(loop.binding.has_value() &&
                    (*loop.binding == Resource::GPUBlock ||
                     *loop.binding == Resource::GPUThread))
        << "add_kernel of a loop not bound to the GPU";
    const bool blocks = *loop.binding == Resource::GPUBlock;
    Kernel kernel;
    kernel.name = symbol_name("_kernel_" + loop.index + "_" +
                              std::to_string(kernel_count++));
    kernel.thread_loop = blocks ? nested_thread_loop(host, loop) : nullptr;

    const Block *body_head = nullptr;
    for (const auto &block : host.blocks) {
        if (block->name == loop.body.name) {
            body_head = block.get();
        }
    }
    internal_assert(body_head != nullptr && !body_head->args.empty())
        << "the body of the loop over " << loop.index << " has no index";

    // Everything the body reaches, first, so that the body's calls resolve.
    compile_reachable(callees(ir::ssa::Cfg(host, loop.body.name).blocks()));

    // The kernel: void(begin, stride, captures...), every argument by value
    // -- a capture that is a pointer to an unwritten struct as the struct.
    // The random generator's state, if the body reads one, is not a
    // parameter: each thread seeds its own (see emit_rng_setup(index)).
    const std::vector<size_t> captures = launch_captures(*body_head);
    internal_assert(by_value.size() == captures.size())
        << "add_kernel: " << by_value.size() << " by-value flags for "
        << captures.size() << " captures";
    std::vector<llvm::Type *> params{codegen_type(loop.start->get_type()),
                                     codegen_type(loop.stride->get_type())};
    for (size_t k = 0; k < captures.size(); k++) {
        const Type &type = body_head->args[captures[k]].type;
        if (by_value[k]) {
            const Ptr_t *ptr_t = type.as<Ptr_t>();
            internal_assert(ptr_t != nullptr)
                << "a by-value capture that is not a pointer: " << type;
            params.push_back(codegen_type(ptr_t->etype));
        } else {
            params.push_back(codegen_type(type));
        }
    }
    const llvm::DataLayout &dl = module->getDataLayout();
    for (llvm::Type *t : params) {
        kernel.param_bytes.push_back(dl.getTypeAllocSize(t));
    }
    llvm::Function *fn = llvm::Function::Create(
        llvm::FunctionType::get(void_t, params, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, kernel.name, module.get());
    // What makes it an entry point the driver can launch rather than a
    // device function: the calling convention, which is how LLVM has marked
    // kernels since the nvvm.annotations metadata was retired.
    fn->setCallingConv(llvm::CallingConv::PTX_Kernel);
    fn->getArg(0)->setName("_begin");
    fn->getArg(1)->setName("_stride");
    // `--gpu-max-registers`: the `.maxnreg` directive on the kernel, which
    // the NVPTX printer takes from this attribute (NVVMAttributes.h).
    if (options->gpu_max_registers != 0) {
        fn->addFnAttr("nvvm.maxnreg", std::to_string(options->gpu_max_registers));
    }

    compile_kernel_body(host, loop, fn, [&](BoundLoop &bound) {
        // The index: the block's number for a block loop, the thread's for
        // a thread loop, as the loop counts -- begin + n * stride. Exact,
        // because the launch made exactly as many of each as the loop has
        // iterations, so nothing is past the end.
        llvm::Value *raw = blocks ? block_index() : thread_index();
        llvm::Value *begin = fn->getArg(0);
        llvm::Value *stride = fn->getArg(1);
        llvm::Value *n = builder->CreateIntCast(raw, begin->getType(), false);
        llvm::Value *index = builder->CreateAdd(
            begin, builder->CreateMul(n, stride), body_head->args[0].name);
        bound.bind(body_head->args[0].name, index);
        for (size_t k = 0; k < captures.size(); k++) {
            llvm::Argument *arg = fn->getArg(unsigned(k + 2));
            const std::string &name = body_head->args[captures[k]].name;
            if (by_value[k]) {
                // The struct came as a parameter; the body wants its
                // address, so it gets a local holding the value.
                arg->setName(name + "_value");
                llvm::Value *local =
                    create_alloca_at_entry(arg->getType(), name);
                builder->CreateStore(arg, local);
                bound.bind(name, local);
                continue;
            }
            arg->setName(name);
            bound.bind(name, arg);
        }
        if (seeds_rng(*body_head)) {
            emit_rng_setup(index, /*outer_state=*/nullptr);
        }
        // A block loop's body with a thread loop in it is run by every
        // thread of the block, and its effects are made once; see
        // effects_once_guard. Without a thread loop the block is one thread
        // and there is nothing to guard.
        block_level = kernel.thread_loop != nullptr;
    });
    block_level = false;
    return kernel;
}

llvm::Value *CodeGen_PTX::thread_index() {
    return builder->CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(
            module.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
        {}, "tid");
}

llvm::Value *CodeGen_PTX::block_index() {
    return builder->CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(
            module.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x),
        {}, "ctaid");
}

void CodeGen_PTX::barrier() {
    // `bar.sync 0`: every thread of the block arrives, and the writes each
    // made before are visible to all of them after.
    builder->CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(
            module.get(), llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all),
        {llvm::ConstantInt::get(i32_t, 0)});
}

void CodeGen_PTX::emit_bound_parfor(BoundLoop &loop) {
    const Terminator::ParFor &p = loop.loop;
    const Block &body_head = loop.body_head;
    switch (*p.binding) {
    case Resource::GPUThread: {
        // Reached from a block loop's kernel body, where every thread of the
        // block runs and this is where they part: each takes the iteration
        // its thread number names. A barrier on either side orders the
        // block-level effects (made by one thread, see effects_once_guard)
        // before and after the iterations. Anywhere else -- inside another
        // thread loop, or in a function a kernel calls -- there is no block
        // of threads waiting to be given iterations.
        internal_assert(block_level)
            << "bind(" << p.index << ", GPUThread): this loop is not directly "
            << "inside a loop bound to GPUBlock. A thread loop is the threads "
            << "of a block, so it sits inside the block loop, once; it cannot "
            << "be nested in another thread loop or reached from a function "
            << "a kernel calls.";
        // What the thread loop reads from the block body: values every
        // thread computed alike, or the kernel's own arguments -- never a
        // local of the block body, which each thread has a copy of its own
        // of and which the threads would have to share.
        for (const auto &capture : p.body.args) {
            const std::shared_ptr<Value> *base = &capture;
            while (true) {
                const auto *instr =
                    std::get_if<std::shared_ptr<Instruction>>(&(*base)->data);
                if (instr == nullptr) {
                    break;
                }
                const Instruction::Op op = (*instr)->op;
                if ((op == Instruction::Op::GEP ||
                     op == Instruction::Op::FieldPtr ||
                     op == Instruction::Op::AddressOf) &&
                    !(*instr)->operands.empty()) {
                    base = &(*instr)->operands[0];
                    continue;
                }
                internal_assert(op != Instruction::Op::Alloca &&
                                op != Instruction::Op::Alloc)
                    << "[unimplemented] the thread loop over " << p.index
                    << " reads `" << (*instr)->name << "`, a local of the "
                    << "block loop's body. A block's threads each have a "
                    << "copy of such a local, so one the thread loop reads "
                    << "or writes has to be one thing the block shares -- "
                    << "shared memory, which the PTX backend does not "
                    << "allocate yet. Move it into the thread loop, or out "
                    << "of the block loop.";
                break;
            }
        }
        barrier();
        const Expr begin_e = loop.operand(p.start),
                   stride_e = loop.operand(p.stride);
        frames.push_frame();
        llvm::Value *begin = codegen_expr(begin_e);
        llvm::Value *stride = codegen_expr(stride_e);
        llvm::Value *n =
            builder->CreateIntCast(thread_index(), begin->getType(), false);
        llvm::Value *index = builder->CreateAdd(
            begin, builder->CreateMul(n, stride), body_head.args[0].name);
        loop.bind(body_head.args[0].name, index);
        for (const size_t i : launch_captures(body_head)) {
            loop.bind(body_head.args[i].name,
                      codegen_expr(loop.operand(p.body.args[i - 1])));
        }
        if (seeds_rng(body_head)) {
            // The thread's own state, from the block's -- which the block
            // body has, seeded from the block's number, since the thread
            // loop reading the state is what made the block body capture it
            // -- and the thread's number.
            const auto block_state = frames.from_frames(lower::rng_state_name);
            internal_assert(block_state.has_value())
                << "the thread loop over " << p.index
                << " draws random numbers, but the block loop around it has "
                << "no generator state";
            emit_rng_setup(index, *block_state);
        }
        llvm::BasicBlock *body_bb =
            llvm::BasicBlock::Create(*context, p.body.name, current_function);
        llvm::BasicBlock *join = llvm::BasicBlock::Create(
            *context, p.index + "_join", current_function);
        builder->CreateBr(body_bb);
        builder->SetInsertPoint(body_bb);
        block_level = false;
        loop.emit_body(body_bb, join);
        block_level = true;
        builder->SetInsertPoint(join);
        barrier();
        frames.pop_frame();
        return;
    }
    case Resource::GPUBlock:
        internal_error << "bind(" << p.index << ", GPUBlock): this loop is "
                       << "inside code that already runs on the GPU. Blocks "
                       << "do not nest; a block loop is the outermost loop "
                       << "on the device.";
    case Resource::CPUThread:
        internal_error << "bind(" << p.index << ", CPUThread): this loop is "
                       << "inside code that runs on the GPU, where there are "
                       << "no CPU threads to bind to.";
    case Resource::RTCore:
    case Resource::OptixThread:
        internal_error << "bind(" << p.index << ", " << to_string(*p.binding)
                       << "): the OptiX backend is not built yet.";
    case Resource::TextureUnit:
        internal_error << "bind(" << p.index << ", TextureUnit): a texture "
                       << "unit runs a function, not a loop (see "
                       << "ir::Bind::lambda).";
    }
}

llvm::Value *CodeGen_PTX::codegen_texture_sample(const Intrinsic *node) {
    // `tex.grad.2d.v4.f32.f32`: CUDA's tex2DGrad on a texture object, which
    // is what pbrt's GPU build samples its image textures with
    // (GPUSpectrumImageTexture::Evaluate). The unit takes the level from
    // the two gradients, filters within it, wraps at the edges, and clamps
    // the level, as the object was created to (bonsai_cuda_texture_create);
    // the four floats come back as a struct, gathered into the vec4f.
    llvm::Value *handle = codegen_expr(node->args[0]);
    llvm::Value *st = codegen_expr(node->args[1]);
    llvm::Value *dx = codegen_expr(node->args[2]);
    llvm::Value *dy = codegen_expr(node->args[3]);
    const auto lane = [&](llvm::Value *v, uint64_t k) {
        return builder->CreateExtractElement(v, k);
    };
    llvm::Function *tex = llvm::Intrinsic::getOrInsertDeclaration(
        module.get(), llvm::Intrinsic::nvvm_tex_unified_2d_grad_v4f32_f32);
    llvm::Value *fetched = builder->CreateCall(
        tex,
        {handle, lane(st, 0), lane(st, 1), lane(dx, 0), lane(dx, 1),
         lane(dy, 0), lane(dy, 1)},
        "tex");
    llvm::Value *result = llvm::UndefValue::get(codegen_type(node->type));
    for (uint64_t k = 0; k < 4; k++) {
        result = builder->CreateInsertElement(
            result, builder->CreateExtractValue(fetched, unsigned(k)), k);
    }
    return result;
}

llvm::SyncScope::ID CodeGen_PTX::atomic_scope() {
    return context->getOrInsertSyncScopeID("device");
}

llvm::FastMathFlags CodeGen_PTX::fast_math_flags() {
    // nvcc's `--use_fast_math` assumes nothing about NaNs or infinities and
    // reassociates nothing; it is the four flags set_device_arithmetic and
    // finish() apply and the intrinsics codegen_math_call picks. LLVM's
    // `fast` would let the optimizer do more than pbrt's device code does.
    return llvm::FastMathFlags();
}

void CodeGen_PTX::register_backend_passes(llvm::PassBuilder &pb) {
    // At the end of the pipeline: after inlining has brought every device
    // function into its kernel and SROA has made the by-value structs'
    // pointers extractvalues of the parameter, which is what the pass roots
    // its analysis at.
    pb.registerOptimizerLastEPCallback(
        [](llvm::ModulePassManager &mpm, llvm::OptimizationLevel,
           llvm::ThinOrFullLTOPhase) {
            mpm.addPass(llvm::createModuleToFunctionPassAdaptor(
                MarkDeviceMemory()));
        });
}

llvm::Value *CodeGen_PTX::atomic_address(llvm::Value *loc) {
    // LLVM lowers `atomicrmw fadd` on a float to `atom.add.f32` only when
    // the address is in the global address space in the IR it sees -- the
    // instruction flushes denormals there, and the function's denormal mode
    // has to say so too (finish()) -- and expands every other case to a
    // compare-and-swap loop (NVPTXISelLowering.cpp,
    // shouldExpandAtomicRMWInIR). A kernel's pointer parameters are generic
    // pointers in the IR, and nothing before the expansion pass qualifies
    // them: the backend's own inference runs on loads and stores it can
    // trace to a `byval` parameter, not on a pointer parameter's atomics.
    // What every such pointer is, though, is device memory: the launch put
    // it there. So an address rooted at a kernel argument is cast to the
    // global space here, before the atomic is made on it.
    const llvm::Value *root = llvm::getUnderlyingObject(loc);
    const auto *arg = llvm::dyn_cast<llvm::Argument>(root);
    if (arg == nullptr ||
        arg->getParent()->getCallingConv() != llvm::CallingConv::PTX_Kernel) {
        return loc;
    }
    return builder->CreateAddrSpaceCast(
        loc, llvm::PointerType::get(*context, /*AddressSpace=*/1),
        loc->getName() + ".global");
}

std::string CodeGen_PTX::symbol_name(const std::string &name) const {
    std::string out = name;
    for (char &c : out) {
        const bool legal = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_' || c == '$';
        if (!legal) {
            c = '$';
        }
    }
    return out;
}

llvm::Value *CodeGen_PTX::extract_lane(llvm::Value *vec, llvm::Value *idx) {
    // NVPTX has no instruction for a lane at a variable index: it lowers the
    // extractelement by storing the vector to the thread's local memory and
    // loading the one lane back (NVPTXISelLowering, EXTRACT_VECTOR_ELT ->
    // Expand), so every `d[axis]` in a traversal put a frame under the
    // kernel -- 318 generic loads and 636 stores in the megakernel, and
    // local memory a third of its L1 traffic. A short vector's lane is
    // cheaper in registers: the lanes at constant indices, then for each
    // lane past the first a compare of the index and a `selp`. Long vectors
    // keep the instruction; the gangs a CPU schedule makes never reach the
    // device, so what arrives here is a vec2/3/4.
    const auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(vec->getType());
    if (llvm::isa<llvm::Constant>(idx) || vt == nullptr ||
        vt->getNumElements() > 8) {
        return CodeGen_LLVM::extract_lane(vec, idx);
    }
    llvm::Value *result = builder->CreateExtractElement(vec, uint64_t(0));
    for (unsigned k = 1; k < vt->getNumElements(); k++) {
        llvm::Value *is_k = builder->CreateICmpEQ(
            idx, llvm::ConstantInt::get(idx->getType(), k), "lane_is");
        result = builder->CreateSelect(
            is_k, builder->CreateExtractElement(vec, uint64_t(k)), result,
            "lane");
    }
    return result;
}

llvm::Value *CodeGen_PTX::block_size() {
    return builder->CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(
            module.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
        {}, "ntid");
}

llvm::Value *CodeGen_PTX::block_reduce(ir::Intrinsic::OpType op,
                                       const Type &type, llvm::Value *v) {
    // The fold of every thread's `v` over the block, in every thread, for
    // the one accumulate a block makes where each thread made one
    // (SSA/BlockAccumulates.h). Harris's reduction ("Optimizing Parallel
    // Reduction in CUDA", NVIDIA, 2007) with the tree's last five levels
    // as warp shuffles (Luitjens, "Faster Parallel Reductions on Kepler",
    // NVIDIA Developer Blog, 2014): (1) in each warp a shuffle-down tree of
    // five steps leaves the warp's total in lane 0 -- the shuffle's member
    // mask is the lanes the block has in this warp, and a lane whose
    // partner is past the block's end folds the identity, so a block that
    // ends in a partial warp reduces right; (2) lane 0 writes the warp's
    // total to a shared-memory slot of the warp's own; (3) a barrier, which
    // every thread reaches because the pass put the reduction where every
    // path of the body ends; (4) every thread folds the slots of the
    // block's warps. The lanes of a short vector go through the tree side by
    // side and take a slot each.
    llvm::Type *t = v->getType();
    auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t);
    const unsigned lanes = vt ? unsigned(vt->getNumElements()) : 1;
    llvm::Type *elem_t = vt ? vt->getElementType() : t;
    const bool is_float = elem_t->isFloatTy();
    internal_assert(is_float || elem_t->isIntegerTy(32))
        << "a block reduces 32-bit words, not " << type;
    const Type scalar_t = type.is_vector() ? type.element_of() : type;
    const bool is_signed = scalar_t.is_int();

    // The operation's identity, and the operation.
    llvm::Value *identity = nullptr;
    switch (op) {
    case Intrinsic::block_reduce_add:
        identity = is_float ? llvm::ConstantFP::get(elem_t, 0.0)
                            : llvm::ConstantInt::get(elem_t, 0);
        break;
    case Intrinsic::block_reduce_mul:
        identity = is_float ? llvm::ConstantFP::get(elem_t, 1.0)
                            : llvm::ConstantInt::get(elem_t, 1);
        break;
    case Intrinsic::block_reduce_min:
        identity = is_float ? llvm::ConstantFP::getInfinity(elem_t, false)
                   : is_signed
                       ? llvm::ConstantInt::get(elem_t, INT32_MAX, true)
                       : llvm::ConstantInt::get(elem_t, UINT32_MAX);
        break;
    case Intrinsic::block_reduce_max:
        identity = is_float ? llvm::ConstantFP::getInfinity(elem_t, true)
                   : is_signed
                       ? llvm::ConstantInt::get(elem_t, uint64_t(int64_t(INT32_MIN)), true)
                       : llvm::ConstantInt::get(elem_t, 0);
        break;
    default:
        internal_error << "not a block reduction: " << to_string(op);
    }
    const auto combine = [&](llvm::Value *a, llvm::Value *b) -> llvm::Value * {
        switch (op) {
        case Intrinsic::block_reduce_add:
            return is_float ? builder->CreateFAdd(a, b) : builder->CreateAdd(a, b);
        case Intrinsic::block_reduce_mul:
            return is_float ? builder->CreateFMul(a, b) : builder->CreateMul(a, b);
        case Intrinsic::block_reduce_min:
            return is_float ? builder->CreateMinNum(a, b)
                            : builder->CreateBinaryIntrinsic(
                                  is_signed ? llvm::Intrinsic::smin
                                            : llvm::Intrinsic::umin,
                                  a, b);
        default:
            return is_float ? builder->CreateMaxNum(a, b)
                            : builder->CreateBinaryIntrinsic(
                                  is_signed ? llvm::Intrinsic::smax
                                            : llvm::Intrinsic::umax,
                                  a, b);
        }
    };

    // The thread's place: its lane and warp, how many warps the block has,
    // and how many lanes of this warp exist -- 32, or the block's tail.
    llvm::Value *tid = thread_index();
    llvm::Value *ntid = block_size();
    llvm::Value *c0 = llvm::ConstantInt::get(i32_t, 0);
    llvm::Value *c1 = llvm::ConstantInt::get(i32_t, 1);
    llvm::Value *c32 = llvm::ConstantInt::get(i32_t, 32);
    llvm::Value *lane = builder->CreateAnd(tid, llvm::ConstantInt::get(i32_t, 31), "lane");
    llvm::Value *warp = builder->CreateLShr(tid, llvm::ConstantInt::get(i32_t, 5), "warp");
    llvm::Value *nwarps = builder->CreateLShr(
        builder->CreateAdd(ntid, llvm::ConstantInt::get(i32_t, 31)),
        llvm::ConstantInt::get(i32_t, 5), "nwarps");
    llvm::Value *have = builder->CreateSub(
        ntid, builder->CreateShl(warp, llvm::ConstantInt::get(i32_t, 5)), "have");
    llvm::Value *full = builder->CreateICmpUGE(have, c32, "full_warp");
    llvm::Value *active = builder->CreateSelect(full, c32, have, "active");
    // The tail's member mask is (1 << have) - 1; the shift is taken only
    // where have < 32, a full warp's mask being all ones.
    llvm::Value *tail = builder->CreateSub(
        builder->CreateShl(c1, builder->CreateSelect(full, c0, have)), c1);
    llvm::Value *member = builder->CreateSelect(
        full, llvm::ConstantInt::get(i32_t, -1, true), tail, "member");
    // shfl.sync's c operand: the clamp, the last lane that exists, in bits
    // 0-4 and a segment mask of zero; a source lane past it is invalid and
    // the shuffle says so.
    llvm::Value *clamp = builder->CreateSub(active, c1, "clamp");

    // The slots: a block has at most 32 warps (1024 threads), one slot per
    // warp per lane of the value, in shared memory (address space 3).
    auto *slots_t = llvm::ArrayType::get(elem_t, 32 * lanes);
    auto *slots = new llvm::GlobalVariable(
        *module, slots_t, /*isConstant=*/false,
        llvm::GlobalValue::InternalLinkage, llvm::UndefValue::get(slots_t),
        "block_reduce_slots", nullptr, llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/3);
    slots->setAlignment(llvm::Align(16));
    const auto slot_ptr = [&](llvm::Value *w, unsigned c) {
        llvm::Value *index = builder->CreateAdd(
            builder->CreateMul(w, llvm::ConstantInt::get(i32_t, lanes)),
            llvm::ConstantInt::get(i32_t, c));
        return builder->CreateInBoundsGEP(slots_t, slots, {c0, index});
    };

    // (1) The warp's total, in lane 0.
    llvm::Function *shfl = llvm::Intrinsic::getOrInsertDeclaration(
        module.get(), is_float ? llvm::Intrinsic::nvvm_shfl_sync_down_f32p
                               : llvm::Intrinsic::nvvm_shfl_sync_down_i32p);
    std::vector<llvm::Value *> parts(lanes);
    for (unsigned c = 0; c < lanes; c++) {
        llvm::Value *x = vt ? builder->CreateExtractElement(v, uint64_t(c)) : v;
        for (unsigned off = 16; off >= 1; off >>= 1) {
            llvm::Value *r = builder->CreateCall(
                shfl, {member, x, llvm::ConstantInt::get(i32_t, off), clamp});
            llvm::Value *other = builder->CreateExtractValue(r, 0);
            llvm::Value *valid = builder->CreateExtractValue(r, 1);
            other = builder->CreateSelect(valid, other, identity);
            x = combine(x, other);
        }
        parts[c] = x;
    }
    // (2) Lane 0 writes it.
    emit_if(builder->CreateICmpEQ(lane, c0), [&] {
        for (unsigned c = 0; c < lanes; c++) {
            builder->CreateStore(parts[c], slot_ptr(warp, c));
        }
    });
    // (3)
    barrier();
    // (4) Every thread folds the warps' totals, a loop over the warps the
    // block has (at least one).
    llvm::BasicBlock *entry_bb = builder->GetInsertBlock();
    llvm::BasicBlock *loop_bb =
        llvm::BasicBlock::Create(*context, "reduce_warps", current_function);
    llvm::BasicBlock *after_bb = llvm::BasicBlock::Create(
        *context, "reduce_warps_after", current_function);
    builder->CreateBr(loop_bb);
    builder->SetInsertPoint(loop_bb);
    llvm::PHINode *w = builder->CreatePHI(i32_t, 2, "w");
    w->addIncoming(c0, entry_bb);
    std::vector<llvm::PHINode *> totals(lanes);
    for (unsigned c = 0; c < lanes; c++) {
        totals[c] = builder->CreatePHI(elem_t, 2, "total");
        totals[c]->addIncoming(identity, entry_bb);
    }
    std::vector<llvm::Value *> folded(lanes);
    for (unsigned c = 0; c < lanes; c++) {
        llvm::Value *loaded = builder->CreateLoad(elem_t, slot_ptr(w, c));
        folded[c] = combine(totals[c], loaded);
        totals[c]->addIncoming(folded[c], loop_bb);
    }
    llvm::Value *w_next = builder->CreateAdd(w, c1, "w_next");
    w->addIncoming(w_next, loop_bb);
    builder->CreateCondBr(builder->CreateICmpULT(w_next, nwarps), loop_bb,
                          after_bb);
    builder->SetInsertPoint(after_bb);
    if (vt == nullptr) {
        return folded[0];
    }
    llvm::Value *result = llvm::UndefValue::get(t);
    for (unsigned c = 0; c < lanes; c++) {
        result = builder->CreateInsertElement(result, folded[c], uint64_t(c));
    }
    return result;
}

llvm::Value *CodeGen_PTX::effects_once_guard() {
    if (!block_level) {
        return nullptr;
    }
    return builder->CreateICmpEQ(thread_index(),
                                 llvm::ConstantInt::get(i32_t, 0), "leader");
}

void CodeGen_PTX::check_block_level_call(const std::string &callee) {
    if (!block_level || !has_effects(callee)) {
        return;
    }
    internal_error
        << "[unimplemented] `" << callee << "` is called in the body of a "
        << "loop bound to GPUBlock, outside its thread loop, and has side "
        << "effects. Every thread of the block runs that body, so a call "
        << "there is made once per thread unless it is pure; a call with "
        << "effects would have to be made by one thread and its result "
        << "shared through shared memory, which the PTX backend does not do "
        << "yet. Move the call into the thread loop, or out of the block "
        << "loop.";
}

llvm::Value *CodeGen_PTX::codegen_math_call(const std::string &name,
                                            const Intrinsic *node) {
    internal_assert(!node->args.empty())
        << "libdevice call " << name << " takes arguments: " << Expr(node);
    const Type arg_type = node->args[0].type();
    const bool is_vector = arg_type.is<Vector_t>();
    const Type scalar_type = is_vector ? arg_type.element_of() : arg_type;
    internal_assert(scalar_type.is_float())
        << "libdevice call " << name << " expects a float: " << Expr(node);
    const bool single = scalar_type.bits() == 32;
    internal_assert(single || scalar_type.bits() == 64)
        << "No libdevice entry point for " << scalar_type << " in "
        << Expr(node);
    llvm::Type *scalar_ll = single ? f32_t : f64_t;
    const std::vector<llvm::Type *> params(node->args.size(), scalar_ll);
    // libdevice names the float overload with an `f`, as libm does: what
    // CUDA's own `sinf` compiles to. Under --fast-math -- nvcc's
    // `--use_fast_math`, which pbrt's GPU build is (see
    // set_device_arithmetic) -- these ten single-precision functions are
    // their intrinsic forms instead, `__sinf` for `sinf`, `sin.approx` on the
    // hardware, which libdevice holds as `__nv_fast_sinf` and so on. The
    // rest (`atan2f`, `acosf`, `fabsf`, the double overloads) have no fast
    // form and stay as they are.
    static const std::set<std::string> nvcc_fast_math_intrinsics = {
        "sin", "cos", "tan", "sincos", "exp", "exp10", "log", "log2", "log10",
        "pow"};
    const bool fast = single && options->fast_math &&
                      nvcc_fast_math_intrinsics.count(name) != 0;
    llvm::FunctionCallee callee = module->getOrInsertFunction(
        std::string("__nv_") + (fast ? "fast_" : "") + name +
            (single ? "f" : ""),
        llvm::FunctionType::get(scalar_ll, params, /*isVarArg=*/false));

    std::vector<llvm::Value *> args;
    args.reserve(node->args.size());
    for (const Expr &arg : node->args) {
        args.push_back(codegen_expr(arg));
    }
    if (!is_vector) {
        return builder->CreateCall(callee, args, name);
    }
    // A vector is its components, each computed as a thread computes a
    // scalar: there is no gang here to compute them side by side.
    llvm::Value *result = llvm::UndefValue::get(codegen_type(node->type));
    for (uint32_t lane = 0; lane < arg_type.lanes(); lane++) {
        llvm::Value *index = llvm::ConstantInt::get(i32_t, lane);
        std::vector<llvm::Value *> lane_args;
        lane_args.reserve(args.size());
        for (llvm::Value *arg : args) {
            lane_args.push_back(builder->CreateExtractElement(arg, index));
        }
        llvm::Value *applied = builder->CreateCall(callee, lane_args, name);
        result = builder->CreateInsertElement(result, applied, index);
    }
    return result;
}

llvm::Value *CodeGen_PTX::emit_printf(llvm::Value *format,
                                      const std::vector<llvm::Value *> &args) {
    // int vprintf(const char *format, void *args): the arguments packed in
    // memory one after another, each at its natural alignment, which is
    // what a struct of them is. print_helper has already applied C's
    // promotions -- a float is a double -- so the specifiers read what is
    // there.
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    llvm::FunctionCallee vprintf = module->getOrInsertFunction(
        "vprintf",
        llvm::FunctionType::get(i32_t, {ptr_t, ptr_t}, /*isVarArg=*/false));
    llvm::Value *buffer = llvm::ConstantPointerNull::get(ptr_t);
    if (!args.empty()) {
        std::vector<llvm::Type *> types;
        types.reserve(args.size());
        for (llvm::Value *arg : args) {
            types.push_back(arg->getType());
        }
        auto *packed = llvm::StructType::get(*context, types);
        llvm::Value *slot = create_alloca_at_entry(packed, "_print_args");
        for (size_t i = 0; i < args.size(); i++) {
            builder->CreateStore(args[i],
                                 builder->CreateStructGEP(packed, slot, i));
        }
        buffer = slot;
    }
    return builder->CreateCall(vprintf, {format, buffer});
}

llvm::FunctionCallee CodeGen_PTX::get_pthread_lock() {
    internal_error << "[unimplemented] a dynamic array grown on the device: "
                   << "there is no mutex to grow it under. Size it on the host.";
}

llvm::FunctionCallee CodeGen_PTX::get_pthread_unlock() {
    return get_pthread_lock();
}

llvm::FunctionCallee CodeGen_PTX::get_pthread_init() {
    return get_pthread_lock();
}

std::string CodeGen_PTX::find_libdevice() {
    // Where the CUDA toolkit keeps it: named outright, or under the
    // toolkit's root, or where the toolkit installs itself by default.
    std::vector<std::string> tried;
    if (const char *named = std::getenv("BONSAI_LIBDEVICE")) {
        if (llvm::sys::fs::exists(named)) {
            return named;
        }
        tried.push_back(named);
    }
    std::vector<std::string> roots;
    for (const char *var : {"CUDA_HOME", "CUDA_PATH"}) {
        if (const char *root = std::getenv(var)) {
            roots.push_back(root);
        }
    }
    roots.push_back("/usr/local/cuda");
    roots.push_back("/opt/cuda");
    for (const std::string &root : roots) {
        const std::string path = root + "/nvvm/libdevice/libdevice.10.bc";
        if (llvm::sys::fs::exists(path)) {
            return path;
        }
        tried.push_back(path);
    }
    std::string message =
        "libdevice.10.bc, CUDA's device maths library, was not found; the "
        "device code calls into it. Set BONSAI_LIBDEVICE to the file or "
        "CUDA_HOME to the toolkit. Tried:";
    for (const std::string &path : tried) {
        message += "\n  " + path;
    }
    internal_error << message;
}

void CodeGen_PTX::link_libdevice() {
    // Only when something calls into it: a kernel with no transcendental
    // maths needs no library, and so compiles on a machine without the
    // toolkit.
    bool needed = false;
    for (const llvm::Function &f : *module) {
        if (f.isDeclaration() && f.getName().starts_with("__nv_")) {
            needed = true;
            break;
        }
    }
    if (!needed) {
        return;
    }
    const std::string path = find_libdevice();
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> lib =
        llvm::parseIRFile(path, diagnostic, *context);
    internal_assert(lib != nullptr)
        << "could not read " << path << ": " << diagnostic.getMessage().str();
    // libdevice is written for a generic nvptx64 machine; ours is this one,
    // and the linker refuses two layouts.
    lib->setDataLayout(module->getDataLayout());
    lib->setTargetTriple(module->getTargetTriple());
    const bool failed = llvm::Linker::linkModules(
        *module, std::move(lib), llvm::Linker::Flags::LinkOnlyNeeded);
    internal_assert(!failed) << "linking " << path << " failed";
    // Internal, as clang makes builtin bitcode it links: what was pulled in
    // is then inlined into its one caller and dropped, rather than kept as
    // an exported function of the module.
    for (llvm::Function &f : *module) {
        if (!f.isDeclaration() && f.getName().starts_with("__nv_")) {
            f.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
    }
}

void CodeGen_PTX::finish() {
    frames.pop_frame();
    // Every device function folded into the kernels that call it. ptxas
    // inlines them all regardless -- the megakernel's SASS is one function
    // -- but at the LLVM level twenty-one had stayed separate, parameters
    // marshalled through local memory, and MarkDeviceMemory (registered in
    // register_backend_passes) can only tag the loads it sees in the kernel:
    // a pointer handed to a call is one it has to assume is written. A
    // function the program marked noinline keeps its word; a recursive one
    // stays a call, since the inliner will not fold a function into itself.
    for (llvm::Function &fn : *module) {
        if (!fn.isDeclaration() &&
            fn.getCallingConv() != llvm::CallingConv::PTX_Kernel &&
            !fn.hasFnAttribute(llvm::Attribute::NoInline)) {
            fn.addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }
    link_libdevice();
    if (options->fast_math) {
        // Denormals flushed to zero in every device function, libdevice's
        // included: nvcc's `--ftz=true`, part of the `--use_fast_math` pbrt's
        // GPU build has (see set_device_arithmetic). The typed attribute,
        // `denormal_fpenv(...)`: the string forms of old
        // (`"denormal-fp-math-f32"`) are what the bitcode reader upgrades,
        // not what Function::getDenormalMode reads.
        const llvm::DenormalFPEnv flush(llvm::DenormalMode::getPreserveSign(),
                                        llvm::DenormalMode::getPreserveSign());
        for (llvm::Function &fn : *module) {
            llvm::AttrBuilder flushing(*context);
            flushing.addDenormalFPEnvAttr(flush);
            fn.addFnAttrs(flushing);
        }
    }
    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[pre-optimization] the device module is invalid";
    optimize_module(*target_machine, *options);
    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[post-optimization] the device module is invalid";
    {
        llvm::raw_string_ostream os(ir_text);
        module->print(os, nullptr);
    }
    // The PTX, as text: what the driver takes (cuModuleLoadData) and what
    // OptiX will. The legacy pass manager is still the only complete API for
    // the code generation passes.
    llvm::SmallString<0> buffer;
    llvm::raw_svector_ostream os(buffer);
    llvm::legacy::PassManager pm;
    pm.add(new llvm::TargetLibraryInfoWrapperPass(
        target_library_info(llvm::Triple(module->getTargetTriple()))));
    internal_assert(!target_machine->addPassesToEmitFile(
        pm, os, nullptr, llvm::CodeGenFileType::AssemblyFile))
        << "the NVPTX target cannot emit assembly";
    pm.run(*module);
    ptx_text = buffer.str().str();
}

} // namespace bonsai
