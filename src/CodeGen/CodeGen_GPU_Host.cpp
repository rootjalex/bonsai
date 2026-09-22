#include "CodeGen/CodeGen_GPU_Host.h"

#include "CodeGen/CodeGen_X86.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "SSA/Analysis.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>

#include <algorithm>

namespace bonsai {

using namespace ir;
using ir::ssa::Block;
using ir::ssa::Instruction;
using ir::ssa::Terminator;
using ir::ssa::Value;

template <typename CodeGen_CPU>
std::unique_ptr<llvm::Module>
CodeGen_GPU_Host<CodeGen_CPU>::compile_program(const Program &program,
                                               const CompilerOptions &options) {
    this->program = &program;
    this->options = &options;
    return CodeGen_CPU::compile_program(program, options);
}

template <typename CodeGen_CPU>
void CodeGen_GPU_Host<CodeGen_CPU>::emit_bound_parfor(
    CodeGen_LLVM::BoundLoop &loop) {
    switch (*loop.loop.binding) {
    case Resource::GPUBlock:
    case Resource::GPUThread:
        emit_gpu_launch(loop);
        return;
    default:
        CodeGen_CPU::emit_bound_parfor(loop);
        return;
    }
}

namespace {

// The parameter a chain of addresses -- GEPs, field pointers, an address-of
// -- bottoms out at, or null.
const ir::ssa::Argument *base_argument(const std::shared_ptr<Value> &v) {
    const std::shared_ptr<Value> *base = &v;
    while (const auto *instr =
               std::get_if<std::shared_ptr<Instruction>>(&(*base)->data)) {
        const Instruction::Op op = (*instr)->op;
        if ((op != Instruction::Op::GEP && op != Instruction::Op::FieldPtr &&
             op != Instruction::Op::AddressOf) ||
            (*instr)->operands.empty()) {
            return nullptr;
        }
        base = &(*instr)->operands[0];
    }
    return std::get_if<ir::ssa::Argument>(&(*base)->data);
}

// Whether the body of `loop` may write through the capture named `name`: a
// store, accumulate or fetch-and-add whose address is it, or a call that
// hands it (or an address inside it) to a parameter the callee declares
// mutating. A call to a function the program does not define counts as a
// write, since nothing is known of it.
bool capture_written(const Program &program, const ir::ssa::Function &host,
                     const Terminator::ParFor &loop, const std::string &name) {
    const auto is_it = [&](const std::shared_ptr<Value> &v) {
        const ir::ssa::Argument *a = base_argument(v);
        return a != nullptr && a->name == name;
    };
    const auto call_writes = [&](const std::string &callee,
                                 const std::vector<std::shared_ptr<Value>> &args) {
        const auto f = program.funcs.find(callee);
        for (size_t k = 0; k < args.size(); k++) {
            if (!is_it(args[k])) {
                continue;
            }
            if (f == program.funcs.end() || k >= f->second->args.size() ||
                f->second->args[k].mutating) {
                return true;
            }
        }
        return false;
    };
    const ir::ssa::Cfg body(host, loop.body.name);
    for (const auto &block : body.blocks()) {
        for (const auto &instr : block->instrs) {
            switch (instr->op) {
            case Instruction::Op::Store:
            case Instruction::Op::AccAdd:
            case Instruction::Op::AccMul:
            case Instruction::Op::AccSub:
            case Instruction::Op::AccMin:
            case Instruction::Op::AccMax:
            case Instruction::Op::AccArgmin:
            case Instruction::Op::AccArgmax:
            case Instruction::Op::AtomicAdd:
                if (!instr->operands.empty() && is_it(instr->operands[0])) {
                    return true;
                }
                break;
            default:
                break;
            }
        }
        bool written = false;
        std::visit(ir::ssa::overloads{
                       [&](const Terminator::Call &c) {
                           written = call_writes(c.call.name, c.call.args);
                       },
                       [&](const Terminator::MultiCall &c) {
                           written = call_writes(c.call.name, c.call.args);
                           for (const auto &one : c.varying) {
                               written = written ||
                                         call_writes(c.call.name, one);
                           }
                       },
                       [](const auto &) {},
                   },
                   block->terminator.data);
        if (written) {
            return true;
        }
    }
    return false;
}

// Whether a value of `type` holds an address: what a kernel cannot be handed
// a copy of, since the memory behind the address is not copied with it.
bool holds_pointers(const Type &type) {
    if (type.is<Array_t, Ptr_t, Ref_t, ElementRef_t, DynArray_t, Function_t>()) {
        return true;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        for (const auto &field : s->fields) {
            if (holds_pointers(field.type)) {
                return true;
            }
        }
        return false;
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        return holds_pointers(v->etype);
    }
    return false;
}

} // namespace

template <typename CodeGen_CPU>
llvm::Value *
CodeGen_GPU_Host<CodeGen_CPU>::capture_bytes(const Type &type,
                                             const std::string &name) {
    const llvm::DataLayout &dl = module->getDataLayout();
    Type pointee;
    llvm::Value *count = nullptr;
    if (const Array_t *a = type.as<Array_t>()) {
        // An array of no stated length -- `array[T]`, an exported function's
        // parameter whose length is the driver's business -- is a pointer
        // and nothing more here, and a pointer says nothing about how much
        // lies behind it. What such a parameter needs is to arrive as a
        // buffer that knows its own size and where it is resident, so that
        // the launch asks for it on the device rather than copying it: the
        // buffer descriptors of apps/pbrt/PLAN.md, "Where the data lives",
        // phase A0.
        internal_assert(a->size.defined())
            << "[unimplemented] the loop bound to the GPU reads `" << name
            << "` of type " << type << ", an array of no stated length, so "
            << "the launch cannot know how many bytes to move to the device. "
            << "An exported function's arrays have to come in as buffers "
            << "that know their size and residency (apps/pbrt/PLAN.md, phase "
            << "A0), which is not built yet.";
        pointee = a->etype;
        count = builder->CreateIntCast(codegen_expr(a->size), i64_t,
                                       a->size.type().is_int());
    } else if (const Ptr_t *p = type.as<Ptr_t>()) {
        pointee = p->etype;
        count = llvm::ConstantInt::get(i64_t, 1);
    } else {
        internal_error << "`" << name << "` of type " << type
                       << " is not something a kernel is handed by address";
    }
    // A pointee with addresses in it -- a tree's layout, whose node arrays
    // its struct points at; a queue's arrays -- is a structure the layout
    // language owns, and moving it to the device means moving what it points
    // at too. That placement is the layout's to make (see apps/pbrt/PLAN.md,
    // "Where the data lives"); a copy of the top-level struct would be a
    // device pointer to host memory.
    internal_assert(!holds_pointers(pointee))
        << "[unimplemented] the loop bound to the GPU reads `" << name
        << "` of type " << type << ", whose elements hold pointers of their "
        << "own. Only the top level of it would reach the device; placing "
        << "what it points at is the layout's job and is not built yet.";
    const uint64_t element_bytes = dl.getTypeAllocSize(codegen_type(pointee));
    return builder->CreateMul(count,
                              llvm::ConstantInt::get(i64_t, element_bytes),
                              name + "_bytes");
}

template <typename CodeGen_CPU>
void CodeGen_GPU_Host<CodeGen_CPU>::emit_gpu_launch(
    CodeGen_LLVM::BoundLoop &loop) {
    internal_assert(program && options) << "a GPU launch outside compile_program";
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    if (!device) {
        device = std::make_unique<CodeGen_PTX>();
        device->begin(*program, *options);
        ptx_source = new llvm::GlobalVariable(
            *module, ptr_t, /*isConstant=*/true,
            llvm::GlobalValue::InternalLinkage,
            llvm::ConstantPointerNull::get(ptr_t), "_bonsai_ptx");
    }
    const Terminator::ParFor &p = loop.loop;
    const Block &body_head = loop.body_head;
    const Expr begin_e = loop.operand(p.start), end_e = loop.operand(p.end),
               stride_e = loop.operand(p.stride);

    // The arguments: begin, stride, then the captures, in the order the
    // kernel takes them (CodeGen_LLVM::launch_captures: the random
    // generator's state stays behind, each thread seeding its own).
    std::vector<Expr> arg_exprs{begin_e, stride_e};
    std::vector<Type> arg_types{begin_e.type(), stride_e.type()};
    std::vector<std::string> arg_names{"_begin", "_stride"};
    // Which of the body's arguments each slot after the first two carries.
    std::vector<size_t> slot_arg{0, 0};
    for (const size_t i : CodeGen_LLVM::launch_captures(body_head)) {
        arg_exprs.push_back(loop.operand(p.body.args[i - 1]));
        arg_types.push_back(body_head.args[i].type);
        arg_names.push_back(body_head.args[i].name);
        slot_arg.push_back(i);
    }
    const size_t n = arg_exprs.size();

    // Which captures travel as values. A pointer to a struct the kernel
    // only reads -- the camera, a sampler, a layout struct of device
    // pointers -- is the struct itself as a kernel parameter, which costs
    // the launch nothing, where a pointer would be memory to allocate, copy
    // over and free around every launch. One the kernel writes has to be
    // memory, and is copied.
    const auto argument_of = [&](size_t i) -> const ir::ssa::Argument * {
        if (i < 2) {
            return nullptr;
        }
        return std::get_if<ir::ssa::Argument>(
            &p.body.args[slot_arg[i] - 1]->data);
    };
    std::vector<bool> by_value(n - 2, false);
    for (size_t i = 2; i < n; i++) {
        const Ptr_t *ptr_t = arg_types[i].as<Ptr_t>();
        if (ptr_t == nullptr || !ptr_t->etype.is<Struct_t>()) {
            continue;
        }
        const ir::ssa::Argument *argument = argument_of(i);
        if (argument != nullptr &&
            exported_buffers.count(argument->name) != 0 &&
            exported_buffers.at(argument->name).layout != nullptr) {
            by_value[i - 2] = true;
            continue;
        }
        by_value[i - 2] = argument != nullptr &&
                          !capture_written(*program, loop.func, p, argument->name);
    }
    const CodeGen_PTX::Kernel kernel =
        device->add_kernel(loop.func, loop.loop, by_value);

    std::vector<llvm::Type *> slot_types(n);
    for (size_t i = 0; i < n; i++) {
        slot_types[i] = i >= 2 && by_value[i - 2]
                            ? codegen_type(arg_types[i].as<Ptr_t>()->etype)
                            : codegen_type(arg_types[i]);
    }
    // Each argument's bytes have to mean the same thing on both sides: the
    // driver copies each slot as the device declared its parameter, so a
    // struct laid out differently by the two machines would be read wrong.
    // Both layouts are at hand, so this is checked rather than assumed.
    const llvm::DataLayout &dl = module->getDataLayout();
    internal_assert(kernel.param_bytes.size() == n);
    for (size_t i = 0; i < n; i++) {
        internal_assert(dl.getTypeAllocSize(slot_types[i]) ==
                        kernel.param_bytes[i])
            << "kernel argument `" << arg_names[i] << "` of type "
            << arg_types[i] << " occupies " << dl.getTypeAllocSize(slot_types[i])
            << " bytes on the host and " << kernel.param_bytes[i]
            << " on the device";
    }
    auto *args_ty = llvm::StructType::get(*context, slot_types);
    llvm::Value *args = create_alloca_at_entry(args_ty, "_launch_args");
    auto *params_ty = llvm::ArrayType::get(ptr_t, n);
    llvm::Value *params = create_alloca_at_entry(params_ty, "_launch_params");

    // The captures that are addresses come in three kinds. One that is an
    // exported function's array has a descriptor (see
    // CodeGen_LLVM::ExportedBuffer): the launch asks for it on the device,
    // which is a flag test when the driver staged it there and a copy when
    // not, and marks it dirty there after if the kernel may write it. One
    // that is an exported function's layout struct -- a tree, whose fields
    // point at its node and primitive arrays -- has a descriptor per array
    // field: the launch asks for each on the device and builds, in a stack
    // slot, the struct of device pointers the kernel dereferences, which
    // then travels like any other struct. Any other -- a local of the
    // function, a pointer to a struct -- is a `bonsai_cuda_buffer { host,
    // bytes, param }` (runtime/bonsai_cuda.h) the runtime copies over for
    // the launch and back after it, with how many bytes lie behind it.
    const auto exported_of =
        [&](size_t i) -> const CodeGen_LLVM::ExportedBuffer * {
        if (i < 2) {
            return nullptr;
        }
        const std::shared_ptr<Value> &v = p.body.args[slot_arg[i] - 1];
        if (const auto *a = std::get_if<ir::ssa::Argument>(&v->data)) {
            const auto found = exported_buffers.find(a->name);
            if (found != exported_buffers.end()) {
                return &found->second;
            }
        }
        // Part of such an array -- an address computed from it -- would
        // reach the device as a pointer into memory that is not there.
        const std::shared_ptr<Value> *base = &v;
        while (const auto *instr =
                   std::get_if<std::shared_ptr<Instruction>>(&(*base)->data)) {
            const Instruction::Op op = (*instr)->op;
            if ((op != Instruction::Op::GEP && op != Instruction::Op::FieldPtr &&
                 op != Instruction::Op::AddressOf) ||
                (*instr)->operands.empty()) {
                break;
            }
            base = &(*instr)->operands[0];
        }
        if (const auto *a = std::get_if<ir::ssa::Argument>(&(*base)->data)) {
            internal_assert(exported_buffers.find(a->name) ==
                            exported_buffers.end())
                << "[unimplemented] the loop bound to the GPU reads `"
                << arg_names[i] << "`, an address inside the buffer `"
                << a->name << "`. A kernel takes a buffer whole; index into "
                << "it inside the loop instead.";
        }
        return nullptr;
    };
    auto *buffer_ty = llvm::StructType::get(*context, {ptr_t, i64_t, i64_t});
    // What the runtime copies for the launch: the host address and the
    // byte count, per parameter slot.
    struct Copy {
        size_t param;
        llvm::Value *host;
        llvm::Value *bytes;
    };
    std::vector<Copy> copies;
    std::vector<std::pair<size_t, const CodeGen_LLVM::ExportedBuffer *>>
        descriptor_params;
    // The slot values: computed first, since building a layout's device
    // struct asks for its buffers on the device.
    std::vector<llvm::Value *> slot_values(n, nullptr);
    for (size_t i = 0; i < n; i++) {
        const CodeGen_LLVM::ExportedBuffer *buffer = exported_of(i);
        if (buffer != nullptr && buffer->layout != nullptr) {
            // The layout as the kernel sees it: its arrays' device pointers,
            // built here and passed as the struct.
            llvm::Value *device_struct = unwrap_layout(
                buffer->descriptor, buffer->layout, /*device=*/true,
                /*mark_dirty=*/false, arg_names[i] + "_device");
            slot_values[i] = builder->CreateLoad(slot_types[i], device_struct,
                                                 arg_names[i]);
            descriptor_params.emplace_back(i, buffer);
        } else if (buffer != nullptr) {
            slot_values[i] = buffer_require(buffer->descriptor, /*device=*/true);
            descriptor_params.emplace_back(i, buffer);
        } else if (i >= 2 && by_value[i - 2]) {
            slot_values[i] = builder->CreateLoad(
                slot_types[i], codegen_expr(arg_exprs[i]), arg_names[i]);
        } else {
            slot_values[i] = codegen_expr(arg_exprs[i]);
            if (i >= 2 && arg_types[i].template is<Array_t, Ptr_t>()) {
                copies.push_back({i, slot_values[i],
                                  capture_bytes(arg_types[i], arg_names[i])});
            }
        }
    }
    auto *buffers_ty = llvm::ArrayType::get(buffer_ty, copies.size());
    llvm::Value *buffers =
        copies.empty()
            ? static_cast<llvm::Value *>(llvm::ConstantPointerNull::get(ptr_t))
            : create_alloca_at_entry(buffers_ty, "_launch_buffers");

    for (size_t i = 0; i < n; i++) {
        llvm::Value *slot = builder->CreateStructGEP(args_ty, args, i);
        builder->CreateStore(slot_values[i], slot);
        builder->CreateStore(
            slot, builder->CreateConstInBoundsGEP2_64(params_ty, params, 0, i));
    }
    for (size_t b = 0; b < copies.size(); b++) {
        llvm::Value *entry =
            builder->CreateConstInBoundsGEP2_64(buffers_ty, buffers, 0, b);
        builder->CreateStore(copies[b].host,
                             builder->CreateStructGEP(buffer_ty, entry, 0));
        builder->CreateStore(copies[b].bytes,
                             builder->CreateStructGEP(buffer_ty, entry, 1));
        builder->CreateStore(llvm::ConstantInt::get(i64_t, copies[b].param),
                             builder->CreateStructGEP(buffer_ty, entry, 2));
    }

    // The grid and the block: as many blocks as the block loop has
    // iterations, as many threads as the thread loop has, and one of either
    // that there is no loop for. Exact, so the kernel needs no guard. The
    // thread loop's bounds are values of the block loop's body, so they are
    // read here only if they came into that body from outside it -- as
    // captures, or constants -- which is what makes them the same for every
    // block, as one block size has to be.
    const auto host_value = [&](const std::shared_ptr<Value> &v,
                                const char *what) -> Expr {
        if (std::holds_alternative<ir::ssa::Constant>(v->data)) {
            return loop.operand(v);
        }
        if (const auto *a = std::get_if<ir::ssa::Argument>(&v->data)) {
            for (size_t i = 1; i < body_head.args.size(); i++) {
                if (body_head.args[i].name == a->name) {
                    return loop.operand(p.body.args[i - 1]);
                }
            }
        }
        internal_error
            << "the " << what << " of the thread loop over "
            << kernel.thread_loop->index << " is computed inside the body of "
            << "the block loop over " << p.index
            << ". A launch has one block size, so the thread loop's bounds "
            << "have to be known before the block loop starts: compute them "
            << "outside it.";
        return Expr();
    };
    const Expr count_e = trip_count(begin_e, end_e, stride_e);
    llvm::Value *count = builder->CreateIntCast(codegen_expr(count_e), i64_t,
                                                begin_e.type().is_int());
    llvm::Value *one = llvm::ConstantInt::get(i64_t, 1);
    llvm::Value *grid = one, *block = one;
    if (*p.binding == Resource::GPUThread) {
        block = count;
    } else {
        grid = count;
        if (kernel.thread_loop != nullptr) {
            const Terminator::ParFor &t = *kernel.thread_loop;
            const Expr t_begin = host_value(t.start, "start"),
                       t_end = host_value(t.end, "end"),
                       t_stride = host_value(t.stride, "stride");
            block = builder->CreateIntCast(
                codegen_expr(trip_count(t_begin, t_end, t_stride)), i64_t,
                t_begin.type().is_int());
        }
    }

    // void bonsai_cuda_launch(const char *ptx, const char *kernel,
    //                         int64_t grid_x, int64_t block_x, void **params,
    //                         int64_t nparams, bonsai_cuda_buffer *buffers,
    //                         int64_t nbuffers);
    llvm::FunctionType *launch_ty = llvm::FunctionType::get(
        void_t, {ptr_t, ptr_t, i64_t, i64_t, ptr_t, i64_t, ptr_t, i64_t},
        false);
    llvm::FunctionCallee launch =
        module->getOrInsertFunction("bonsai_cuda_launch", launch_ty);
    llvm::Value *ptx = builder->CreateLoad(ptr_t, ptx_source, "_ptx");
    llvm::Value *name = builder->CreateGlobalString(kernel.name, "_kernel_name");
    builder->CreateCall(launch, {ptx, name, grid, block, params,
                                 llvm::ConstantInt::get(i64_t, n), buffers,
                                 llvm::ConstantInt::get(i64_t, copies.size())});

    // What the kernel may have written is current on the device now; and a
    // buffer the rest of this function reads on the host is brought back
    // here, since the host pointer it was bound to at entry names the same
    // memory. A buffer the host never touches is left where it is, for the
    // driver to ask for when it wants it (outside its timer). A layout is
    // each of its arrays.
    for (const auto &[i, buffer] : descriptor_params) {
        std::vector<llvm::Value *> descriptors{buffer->descriptor};
        if (buffer->layout != nullptr) {
            descriptors = layout_descriptors(buffer->descriptor, buffer->layout);
        }
        for (llvm::Value *descriptor : descriptors) {
            if (buffer->mutating) {
                buffer_mark_dirty(descriptor, /*device=*/true);
            }
            if (buffer->host_used) {
                buffer_require(descriptor, /*device=*/false);
            }
        }
    }
}

template <typename CodeGen_CPU>
void CodeGen_GPU_Host<CodeGen_CPU>::end_functions() {
    CodeGen_CPU::end_functions();
    if (!device) {
        return;
    }
    device->finish();
    llvm::Constant *text = llvm::ConstantDataArray::getString(
        *context, device->ptx(), /*AddNull=*/true);
    auto *holder = new llvm::GlobalVariable(
        *module, text->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, text, "_bonsai_ptx_text");
    holder->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    ptx_source->setInitializer(holder);

    // `void bonsai_gpu_prepare(void)`: loads the module, so that a driver
    // can take the load out of its first timed call. Declared in the
    // generated header when a program has device code.
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    llvm::Function *prepare = llvm::Function::Create(
        llvm::FunctionType::get(void_t, {}, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "bonsai_gpu_prepare", module.get());
    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    builder->SetInsertPoint(
        llvm::BasicBlock::Create(*context, "prepare", prepare));
    llvm::FunctionCallee load = module->getOrInsertFunction(
        "bonsai_cuda_load",
        llvm::FunctionType::get(void_t, {ptr_t}, /*isVarArg=*/false));
    builder->CreateCall(load, {holder});
    builder->CreateRetVoid();
    builder->restoreIP(here);
}

// The hosts there are: the x86-64 generator, and the target-agnostic one for
// any other CPU.
template struct CodeGen_GPU_Host<CodeGen_X86>;
template struct CodeGen_GPU_Host<CodeGen_LLVM>;

} // namespace bonsai
