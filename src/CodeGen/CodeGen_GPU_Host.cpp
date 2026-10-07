#include "CodeGen/CodeGen_GPU_Host.h"

#include "CodeGen/CodeGen_X86.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "SSA/Analysis.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>

#include <algorithm>
#include <set>

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
    case Resource::OptixThread:
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

// Whether every address stored into the allocation `captured` names -- as
// the value of a store into it, or a field of a struct stored into it -- is
// a device-resident allocation's (CodeGen_LLVM::device_resident), so that
// a copy of it on the device points at device memory. A queue's header is
// the case: `make_struct<Queue_q>(0, arrays...)` stored into it, every
// array device-resident. A constant (an undefined handle for an array the
// drain never reads) passes. False for anything that is not an allocation
// of the function, or has no store into it, or is filled from something
// else.
template <typename CodeGen_CPU>
bool CodeGen_GPU_Host<CodeGen_CPU>::pointers_stored_are_device(
    const ir::ssa::Function &func, const std::shared_ptr<Value> &captured,
    ir::ssa::Definitions &defs, const std::string &launch_block) {
    std::set<std::string> allocations;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Alloc ||
                instr->op == Instruction::Op::Alloca) {
                allocations.insert(instr->name);
            }
        }
    }
    const std::optional<std::string> header =
        CodeGen_LLVM::allocation_of(captured, allocations, &defs, launch_block);
    const bool explain = std::getenv("BONSAI_EXPLAIN_DEVICE") != nullptr;
    if (!header.has_value()) {
        if (explain) {
            std::cerr << "; capture is not an allocation of the function: ";
            captured->dump(std::cerr);
            std::shared_ptr<Value> cursor = captured;
            for (int depth = 0; cursor && depth < 8; depth++) {
                const auto *in =
                    std::get_if<std::shared_ptr<Instruction>>(&cursor->data);
                if (in == nullptr) {
                    const auto *a = std::get_if<ir::ssa::Argument>(&cursor->data);
                    std::cerr << (a ? " <- argument " + a->name : " <- constant");
                    break;
                }
                std::cerr << " <- " << (*in)->name << " = " << op_name((*in)->op);
                if ((*in)->operands.empty()) {
                    break;
                }
                cursor = (*in)->operands[0];
            }
            std::cerr << "; allocations:";
            for (const std::string &a : allocations) {
                std::cerr << " " << a;
            }
            std::cerr << "\n";
        }
        return false;
    }
    std::string here;
    const auto device_address = [&](const std::shared_ptr<Value> &v) {
        if (std::holds_alternative<ir::ssa::Constant>(v->data)) {
            return true; // undefined, or null
        }
        return CodeGen_LLVM::allocation_of(v, device_resident, &defs, here)
            .has_value();
    };
    bool any = false;
    for (const auto &block : func.blocks) {
        here = block->name;
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Store || instr->operands.size() != 2 ||
                CodeGen_LLVM::allocation_of(instr->operands[0], allocations, &defs,
                                            here) != header) {
                continue;
            }
            any = true;
            const std::shared_ptr<Value> &stored = instr->operands[1];
            const auto *made = std::get_if<std::shared_ptr<Instruction>>(&stored->data);
            if (made != nullptr && (*made)->op == Instruction::Op::MakeStruct) {
                for (const auto &field : (*made)->operands) {
                    if (holds_pointers(field->get_type()) && !device_address(field)) {
                        if (explain) {
                            std::cerr << "; " << *header << ": a handle stored into "
                                      << "it is not a device allocation's: ";
                            field->dump(std::cerr);
                            std::cerr << "\n";
                        }
                        return false;
                    }
                }
            } else if (holds_pointers(stored->get_type()) && !device_address(stored)) {
                if (explain) {
                    std::cerr << "; " << *header << ": a value stored into it "
                              << "holds pointers that are not a device "
                              << "allocation's: ";
                    stored->dump(std::cerr);
                    std::cerr << "\n";
                }
                return false;
            }
        }
    }
    if (explain && !any) {
        std::cerr << "; " << *header << ": nothing is stored into it\n";
    }
    return any;
}

template <typename CodeGen_CPU>
llvm::Value *
CodeGen_GPU_Host<CodeGen_CPU>::capture_bytes(const Type &type,
                                             const std::string &name,
                                             bool pointers_are_device) {
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
    // its struct points at; a queue's header, holding its arrays' handles
    // -- is a structure the layout language owns, and moving it to the
    // device means moving what it points at too. That placement is the
    // layout's to make (see apps/pbrt/PLAN.md, "Where the data lives"),
    // unless the addresses in it are already the device's: a header whose
    // every stored handle is a device-resident allocation's
    // (pointers_stored_are_device) crosses as it is, and its copy on the
    // device points at device memory. Otherwise a copy of the top-level
    // struct would be a device pointer to host memory.
    internal_assert(!holds_pointers(pointee) || pointers_are_device)
        << "[unimplemented] the loop bound to the GPU reads `" << name
        << "` of type " << type << ", whose elements hold pointers of their "
        << "own that are not device allocations. Only the top level of it "
        << "would reach the device; placing what it points at is the "
        << "layout's job and is not built yet.";
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
    const Terminator::ParFor &p = loop.loop;
    // A loop bound to OptixThread is the raygen program of an OptiX module
    // of its own (CodeGen_OptiX); every other GPU-bound loop is a kernel of
    // the one device module. Either way the module's PTX is a string
    // embedded once the module is finished (end_functions), and the launch
    // loads its address from a global filled in then.
    const bool optix = *p.binding == Resource::OptixThread;
    CodeGen_OptiX *raygen_module = nullptr;
    llvm::GlobalVariable *ptx_global = nullptr;
    if (optix) {
        auto made = std::make_unique<CodeGen_OptiX>();
        made->begin(*program, *options);
        ptx_global = new llvm::GlobalVariable(
            *module, ptr_t, /*isConstant=*/true,
            llvm::GlobalValue::InternalLinkage,
            llvm::ConstantPointerNull::get(ptr_t), "_bonsai_optix_ptx");
        raygen_module = made.get();
        optix_modules.emplace_back(std::move(made), ptx_global);
    } else {
        if (!device) {
            device = std::make_unique<CodeGen_PTX>();
            device->begin(*program, *options);
            ptx_source = new llvm::GlobalVariable(
                *module, ptr_t, /*isConstant=*/true,
                llvm::GlobalValue::InternalLinkage,
                llvm::ConstantPointerNull::get(ptr_t), "_bonsai_ptx");
        }
        ptx_global = ptx_source;
    }
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
        optix ? raygen_module->add_raygen(loop.func, p, by_value)
              : device->add_kernel(loop.func, loop.loop, by_value);

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
    if (optix) {
        // The raygen's parameters cross as this whole struct, so the two
        // machines' layouts of it have to agree field for field, not only
        // slot by slot.
        const llvm::StructLayout *layout = dl.getStructLayout(args_ty);
        internal_assert(kernel.param_offsets.size() == n);
        for (size_t i = 0; i < n; i++) {
            internal_assert(layout->getElementOffset(unsigned(i)) ==
                            kernel.param_offsets[i])
                << "launch parameter `" << arg_names[i] << "` of type "
                << arg_types[i] << " sits at byte "
                << layout->getElementOffset(unsigned(i))
                << " of the launch parameters on the host and at byte "
                << kernel.param_offsets[i] << " on the device";
        }
        internal_assert(layout->getSizeInBytes() == kernel.param_struct_bytes)
            << "the launch parameters are " << layout->getSizeInBytes()
            << " bytes on the host and " << kernel.param_struct_bytes
            << " on the device";
    }
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
    // The captures as the launching block refers to them, for following one
    // that is a threaded argument back to the address it was computed from
    // (CodeGen_LLVM::allocation_of).
    ir::ssa::Definitions defs(loop.func, /*lenient=*/true);
    std::string launch_block;
    for (const auto &block : loop.func.blocks) {
        if (std::get_if<Terminator::ParFor>(&block->terminator.data) == &p) {
            launch_block = block->name;
        }
    }
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
            if (i < 2 || !arg_types[i].template is<Array_t, Ptr_t>()) {
                continue;
            }
            const std::shared_ptr<Value> &captured = p.body.args[slot_arg[i] - 1];
            // An allocation that lives on the device (see
            // CodeGen_LLVM::device_resident) is handed over as the device
            // address it already is; nothing to copy.
            if (CodeGen_LLVM::allocation_of(captured, device_resident, &defs,
                                            launch_block)
                    .has_value()) {
                continue;
            }
            // A struct with addresses in it -- a queue's header, holding its
            // arrays' handles -- may cross when every address stored into it
            // is a device allocation's: the copy on the device then points
            // at device memory. Checked at the stores that fill it.
            const bool device_pointers_inside =
                pointers_stored_are_device(loop.func, captured, defs, launch_block);
            copies.push_back({i, slot_values[i],
                              capture_bytes(arg_types[i], arg_names[i],
                                            device_pointers_inside)});
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
    // that there is no loop for. Exact for the loop's bounds as they stand,
    // so the kernel needs no guard of the launch's making: a loop over a
    // queue has its end set to the queue's capacity and its body guarded by
    // the count read on the device (Bind.cpp), which is what makes the
    // launch go without the host reading the count. The
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

    llvm::Value *ptx = builder->CreateLoad(ptr_t, ptx_global, "_ptx");
    llvm::Value *name = builder->CreateGlobalString(kernel.name, "_kernel_name");
    // void bonsai_cuda_launch(const char *ptx, const char *kernel,
    //                         int64_t grid_x, int64_t block_x, void **params,
    //                         int64_t nparams, bonsai_cuda_buffer *buffers,
    //                         int64_t nbuffers);
    llvm::FunctionCallee launch = module->getOrInsertFunction(
        "bonsai_cuda_launch",
        llvm::FunctionType::get(
            void_t, {ptr_t, ptr_t, i64_t, i64_t, ptr_t, i64_t, ptr_t, i64_t},
            false));
    // void bonsai_optix_launch(const char *ptx, const char *raygen,
    //                          int64_t count, void *params,
    //                          int64_t param_bytes, void **slots,
    //                          int64_t nslots, bonsai_cuda_buffer *buffers,
    //                          int64_t nbuffers, uint64_t traversable,
    //                          int64_t payload_values);
    // The launch parameters are the slots, as one struct; the acceleration
    // structure it traces is the `traversable` word of the tree captured,
    // read out of the slot the tree's layout struct travels in (see
    // Lower/Layouts.cpp) -- whether the body captured the struct by its
    // address (a by-value slot) or as a value -- or none when the raygen
    // program traces nothing.
    llvm::FunctionCallee optix_launch = module->getOrInsertFunction(
        "bonsai_optix_launch",
        llvm::FunctionType::get(void_t,
                                {ptr_t, ptr_t, i64_t, ptr_t, i64_t, ptr_t, i64_t,
                                 ptr_t, i64_t, i64_t, i64_t},
                                false));
    llvm::Value *traversable = llvm::ConstantInt::get(i64_t, 0);
    if (optix) {
        bool found = false;
        for (size_t i = 2; i < n; i++) {
            const Ptr_t *pointed = arg_types[i].as<Ptr_t>();
            const Struct_t *s =
                by_value[i - 2] && pointed != nullptr ? pointed->etype.as<Struct_t>()
                                                      : arg_types[i].as<Struct_t>();
            if (s == nullptr) {
                continue;
            }
            for (size_t f = 0; f < s->fields.size(); f++) {
                if (s->fields[f].name != "traversable") {
                    continue;
                }
                internal_assert(!found)
                    << "the loop over " << p.index << " captures two trees "
                    << "the ray tracing hardware searches; a launch traces "
                    << "one acceleration structure.";
                llvm::Value *slot = builder->CreateStructGEP(args_ty, args, i);
                llvm::Value *field = builder->CreateStructGEP(
                    llvm::cast<llvm::StructType>(slot_types[i]), slot, unsigned(f));
                traversable = builder->CreateLoad(i64_t, field, "_traversable");
                found = true;
            }
        }
    }

    // A loop with no iterations launches nothing: a drain over a queue
    // that is empty this round -- the medium queue of a scene with no
    // media, the shadow queue of a bounce that lit nothing -- has a count
    // of zero, and a grid of zero blocks is not a launch the device
    // accepts. Nothing ran, so nothing is marked written.
    llvm::Value *zero = llvm::ConstantInt::get(i64_t, 0);
    llvm::Value *runs = builder->CreateAnd(
        builder->CreateICmpNE(grid, zero), builder->CreateICmpNE(block, zero),
        "_launch_runs");
    llvm::BasicBlock *launch_bb = llvm::BasicBlock::Create(
        *context, kernel.name + "_launch", builder->GetInsertBlock()->getParent());
    llvm::BasicBlock *after_bb = llvm::BasicBlock::Create(
        *context, kernel.name + "_launched",
        builder->GetInsertBlock()->getParent());
    builder->CreateCondBr(runs, launch_bb, after_bb);
    builder->SetInsertPoint(launch_bb);
    if (optix) {
        const llvm::DataLayout &layout = module->getDataLayout();
        builder->CreateCall(
            optix_launch,
            {ptx, name, count, args,
             llvm::ConstantInt::get(i64_t, layout.getTypeAllocSize(args_ty)),
             params, llvm::ConstantInt::get(i64_t, n), buffers,
             llvm::ConstantInt::get(i64_t, copies.size()), traversable,
             llvm::ConstantInt::get(i64_t, kernel.payload_values)});
    } else {
        builder->CreateCall(launch,
                            {ptx, name, grid, block, params,
                             llvm::ConstantInt::get(i64_t, n), buffers,
                             llvm::ConstantInt::get(i64_t, copies.size())});
    }
    // The launch is asynchronous; the function waits for the device before
    // it returns (CodeGen_LLVM::wait_for_device_if_launched).
    this->launched_on_device = true;

    // What the kernel may have written is current on the device now and
    // stale on the host. It stays where it is: a host block that touches it
    // later asks for it back where it does (SSALowering::classify_buffers),
    // and a buffer the host never touches is left for the driver to ask for
    // when it wants it, outside its timer. A buffer the kernel only reads
    // is as it was. A layout is each of its arrays.
    for (const auto &[i, buffer] : descriptor_params) {
        if (!buffer->mutating ||
            !capture_written(*program, loop.func, p, arg_names[i])) {
            continue;
        }
        std::vector<llvm::Value *> descriptors{buffer->descriptor};
        if (buffer->layout != nullptr) {
            descriptors = layout_descriptors(buffer->descriptor, buffer->layout);
        }
        for (llvm::Value *descriptor : descriptors) {
            buffer_mark_dirty(descriptor, /*device=*/true);
        }
    }
    builder->CreateBr(after_bb);
    builder->SetInsertPoint(after_bb);
}

template <typename CodeGen_CPU>
void CodeGen_GPU_Host<CodeGen_CPU>::end_functions() {
    CodeGen_CPU::end_functions();
    if (!device && optix_modules.empty()) {
        return;
    }
    // Each module finished and its PTX embedded where its launches look.
    const auto embed = [&](CodeGen_PTX &generator, llvm::GlobalVariable *source,
                           const char *name) {
        generator.finish();
        llvm::Constant *text = llvm::ConstantDataArray::getString(
            *context, generator.ptx(), /*AddNull=*/true);
        auto *holder = new llvm::GlobalVariable(
            *module, text->getType(), /*isConstant=*/true,
            llvm::GlobalValue::PrivateLinkage, text, name);
        holder->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        source->setInitializer(holder);
        return holder;
    };
    llvm::GlobalVariable *cuda_text =
        device ? embed(*device, ptx_source, "_bonsai_ptx_text") : nullptr;
    std::vector<llvm::GlobalVariable *> optix_texts;
    for (auto &[generator, source] : optix_modules) {
        optix_texts.push_back(embed(*generator, source, "_bonsai_optix_ptx_text"));
    }

    // `void bonsai_gpu_prepare(void)`: loads the modules -- and compiles the
    // OptiX ones' pipelines -- so that a driver can take that out of its
    // first timed call. Declared in the generated header when a program has
    // device code.
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    llvm::Function *prepare = llvm::Function::Create(
        llvm::FunctionType::get(void_t, {}, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "bonsai_gpu_prepare", module.get());
    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    builder->SetInsertPoint(
        llvm::BasicBlock::Create(*context, "prepare", prepare));
    if (cuda_text != nullptr) {
        llvm::FunctionCallee load = module->getOrInsertFunction(
            "bonsai_cuda_load",
            llvm::FunctionType::get(void_t, {ptr_t}, /*isVarArg=*/false));
        builder->CreateCall(load, {cuda_text});
    }
    // The program's ray types, declared before any OptiX module is loaded
    // or scene built: every RTCore query, alphabetical, the order
    // CodeGen_OptiX numbers them in (see runtime/bonsai_optix.h).
    std::set<std::string> queries;
    for (const auto &[_, func] : program->funcs) {
        if (func->optix_program.has_value()) {
            queries.insert(func->optix_program->of);
        }
    }
    if (!optix_texts.empty() && !queries.empty()) {
        std::vector<llvm::Constant *> names;
        for (const std::string &query : queries) {
            names.push_back(builder->CreateGlobalString(query, "_ray_type"));
        }
        auto *names_ty = llvm::ArrayType::get(ptr_t, names.size());
        auto *table = new llvm::GlobalVariable(
            *module, names_ty, /*isConstant=*/true,
            llvm::GlobalValue::PrivateLinkage,
            llvm::ConstantArray::get(names_ty, names), "_bonsai_ray_types");
        llvm::FunctionCallee declare = module->getOrInsertFunction(
            "bonsai_optix_ray_types",
            llvm::FunctionType::get(void_t, {ptr_t, i64_t}, /*isVarArg=*/false));
        builder->CreateCall(
            declare, {table, llvm::ConstantInt::get(i64_t, names.size())});
    }
    for (size_t k = 0; k < optix_texts.size(); k++) {
        llvm::FunctionCallee load = module->getOrInsertFunction(
            "bonsai_optix_load",
            llvm::FunctionType::get(void_t, {ptr_t, i64_t}, /*isVarArg=*/false));
        builder->CreateCall(
            load, {optix_texts[k],
                   llvm::ConstantInt::get(
                       i64_t, optix_modules[k].first->payload_values_used())});
    }
    builder->CreateRetVoid();
    builder->restoreIP(here);
}

// The hosts there are: the x86-64 generator, and the target-agnostic one for
// any other CPU.
template struct CodeGen_GPU_Host<CodeGen_X86>;
template struct CodeGen_GPU_Host<CodeGen_LLVM>;

} // namespace bonsai
