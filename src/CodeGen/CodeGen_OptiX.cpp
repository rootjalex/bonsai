#include "CodeGen/CodeGen_OptiX.h"

#include "IR/Printer.h"
#include "SSA/Analysis.h"
#include "Error.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InlineAsm.h"

#include <algorithm>
#include <set>

namespace bonsai {

using ir::Expr;
using ir::Intrinsic;
using ir::Ptr_t;
using ir::Resource;
using ir::Struct_t;
using ir::Type;
using ir::ssa::Block;
using ir::ssa::Instruction;
using ir::ssa::Terminator;
using ir::ssa::Value;

namespace {

// The context type a query traces with, `_RTContext_<query>`, to the
// query's name; empty when `type` is no such thing.
std::string query_of_context(const Type &type) {
    static const std::string prefix = "_RTContext_";
    const Ptr_t *ptr = type.as<Ptr_t>();
    const Struct_t *s = ptr != nullptr ? ptr->etype.as<Struct_t>() : nullptr;
    if (s == nullptr || s->name.rfind(prefix, 0) != 0) {
        return "";
    }
    return s->name.substr(prefix.size());
}

} // namespace

CodeGen_OptiX::CodeGen_OptiX() = default;

// The option is process-wide and the CUDA module keeps the packed
// instructions, so it is set for this module's own steps alone.
void CodeGen_OptiX::begin(const ir::Program &program,
                          const CompilerOptions &options) {
    llvm_option("nvptx-no-f32x2", "1");
    CodeGen_PTX::begin(program, options);
    llvm_option("nvptx-no-f32x2", "0");
}

void CodeGen_OptiX::finish() {
    llvm_option("nvptx-no-f32x2", "1");
    CodeGen_PTX::finish();
    llvm_option("nvptx-no-f32x2", "0");
}

llvm::Value *CodeGen_OptiX::asm_call(llvm::Type *ret, const std::string &text,
                                     const std::string &constraints,
                                     const std::vector<llvm::Value *> &args,
                                     bool side_effects) {
    std::vector<llvm::Type *> arg_types;
    arg_types.reserve(args.size());
    for (llvm::Value *arg : args) {
        arg_types.push_back(arg->getType());
    }
    llvm::FunctionType *fn_ty = llvm::FunctionType::get(
        ret ? ret : void_t, arg_types, /*isVarArg=*/false);
    llvm::InlineAsm *ia =
        llvm::InlineAsm::get(fn_ty, text, constraints, side_effects);
    return builder->CreateCall(ia, args);
}

llvm::Value *CodeGen_OptiX::payload(unsigned k) {
    return asm_call(i32_t, "call ($0), _optix_get_payload, ($1);", "=r,r",
                    {llvm::ConstantInt::get(i32_t, k)}, /*side_effects=*/true);
}

int CodeGen_OptiX::slot_of(const std::string &name) const {
    for (size_t i = 0; i < slot_names.size(); i++) {
        if (slot_names[i] == name) {
            return int(i);
        }
    }
    return -1;
}

std::vector<std::string>
CodeGen_OptiX::queries_traced(const ir::ssa::Function &host,
                              const Terminator::ParFor &loop) {
    std::set<std::string> queries;
    std::set<std::string> seen;
    std::vector<std::string> work;
    const auto scan = [&](const std::vector<std::shared_ptr<Block>> &blocks) {
        for (const auto &block : blocks) {
            for (const auto &instr : block->instrs) {
                if (instr->op == Instruction::Op::Intrinsic &&
                    instr->intrinsic == Intrinsic::rt_trace &&
                    instr->operands.size() >= 7) {
                    const std::string query =
                        query_of_context(instr->operands[6]->get_type());
                    internal_assert(!query.empty())
                        << "rt_trace with a context that is not a query's: "
                        << instr->operands[6]->get_type();
                    queries.insert(query);
                }
            }
        }
        for (const std::string &callee : callees(blocks)) {
            work.push_back(callee);
        }
    };
    scan(ir::ssa::Cfg(host, loop.body.name).blocks());
    while (!work.empty()) {
        const std::string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto f = program->ssa_funcs.find(name);
        internal_assert(f != program->ssa_funcs.end())
            << "`" << name << "` has no SSA form";
        scan(f->second->blocks);
    }
    return std::vector<std::string>(queries.begin(), queries.end());
}

CodeGen_PTX::Kernel CodeGen_OptiX::add_raygen(const ir::ssa::Function &host,
                                              const Terminator::ParFor &loop,
                                              const std::vector<bool> &by_value) {
    internal_assert(loop.binding.has_value() &&
                    *loop.binding == Resource::OptixThread)
        << "add_raygen of a loop not bound to OptixThread";
    internal_assert(params == nullptr)
        << "an OptiX module holds one raygen program; make another module";
    Kernel kernel;
    kernel.name = symbol_name("__raygen__" + loop.index);
    kernel.thread_loop = nullptr;

    const Block *body_head = nullptr;
    for (const auto &block : host.blocks) {
        if (block->name == loop.body.name) {
            body_head = block.get();
        }
    }
    internal_assert(body_head != nullptr && !body_head->args.empty())
        << "the body of the loop over " << loop.index << " has no index";

    // The ray types before anything traces: every RTCore query of the
    // program, alphabetical -- the numbering every module and every scene's
    // shader binding table share (runtime/bonsai_optix.h) -- of which this
    // module compiles the programs of the queries the body traces.
    {
        std::set<std::string> all;
        for (const auto &[_, func] : program->funcs) {
            if (func->optix_program.has_value()) {
                all.insert(func->optix_program->of);
            }
        }
        ray_types.assign(all.begin(), all.end());
    }
    const std::vector<std::string> traced = queries_traced(host, loop);

    // Everything the body reaches, first, so that the body's calls resolve.
    compile_reachable(callees(ir::ssa::Cfg(host, loop.body.name).blocks()));

    // The launch parameters: begin, stride, then the captures, as the
    // kernel's parameters would be -- a by-value capture as its struct.
    const std::vector<size_t> captures = launch_captures(*body_head);
    internal_assert(by_value.size() == captures.size())
        << "add_raygen: " << by_value.size() << " by-value flags for "
        << captures.size() << " captures";
    slot_types = {codegen_type(loop.start->get_type()),
                  codegen_type(loop.stride->get_type())};
    slot_names = {"_begin", "_stride"};
    for (size_t k = 0; k < captures.size(); k++) {
        const Type &type = body_head->args[captures[k]].type;
        if (by_value[k]) {
            const Ptr_t *ptr_t = type.as<Ptr_t>();
            internal_assert(ptr_t != nullptr)
                << "a by-value capture that is not a pointer: " << type;
            slot_types.push_back(codegen_type(ptr_t->etype));
        } else {
            slot_types.push_back(codegen_type(type));
        }
        slot_names.push_back(body_head->args[captures[k]].name);
    }
    const llvm::DataLayout &dl = module->getDataLayout();
    for (llvm::Type *t : slot_types) {
        kernel.param_bytes.push_back(dl.getTypeAllocSize(t));
    }
    params_ty = llvm::StructType::create(*context, slot_types, "_params");
    // `.const .b8 params[...]`, filled by the launch: what OptiX calls the
    // pipeline's launch parameters (pipelineLaunchParamsVariableName).
    // Defined, zero, and not constant, so that no load from it is folded.
    params = new llvm::GlobalVariable(
        *module, params_ty, /*isConstant=*/false,
        llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantAggregateZero::get(params_ty), "params",
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/4);

    llvm::Function *fn = llvm::Function::Create(
        llvm::FunctionType::get(void_t, {}, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, kernel.name, module.get());
    fn->setCallingConv(llvm::CallingConv::PTX_Kernel);
    if (options->gpu_max_registers != 0) {
        fn->addFnAttr("nvvm.maxnreg", std::to_string(options->gpu_max_registers));
    }

    compile_kernel_body(host, loop, fn, [&](BoundLoop &bound) {
        const auto slot = [&](size_t i, const std::string &name) {
            llvm::Value *at = builder->CreateStructGEP(params_ty, params, i);
            return builder->CreateLoad(slot_types[i], at, name);
        };
        // The index: begin + n * stride for this thread's launch index.
        llvm::Value *begin = slot(0, "_begin");
        llvm::Value *stride = slot(1, "_stride");
        llvm::Value *raw = asm_call(
            i32_t, "call ($0), _optix_get_launch_index_x, ();", "=r", {},
            /*side_effects=*/false);
        llvm::Value *n = builder->CreateIntCast(raw, begin->getType(), false);
        llvm::Value *index = builder->CreateAdd(
            begin, builder->CreateMul(n, stride), body_head->args[0].name);
        bound.bind(body_head->args[0].name, index);
        for (size_t k = 0; k < captures.size(); k++) {
            const std::string &name = body_head->args[captures[k]].name;
            llvm::Value *value = slot(k + 2, name);
            if (by_value[k]) {
                // The body wants the struct's address: a local holding it.
                llvm::Value *local =
                    create_alloca_at_entry(value->getType(), name);
                builder->CreateStore(value, local);
                bound.bind(name, local);
                continue;
            }
            bound.bind(name, value);
        }
        if (seeds_rng(*body_head)) {
            emit_rng_setup(index, /*outer_state=*/nullptr);
        }
    });

    // The traced queries' programs.
    for (const std::string &query : traced) {
        bool found = false;
        for (const auto &[name, func] : program->funcs) {
            if (!func->optix_program || func->optix_program->of != query) {
                continue;
            }
            const auto ssa = program->ssa_funcs.find(name);
            internal_assert(ssa != program->ssa_funcs.end())
                << "the OptiX program `" << name << "` has no SSA form";
            add_program(*func, *ssa->second);
            found = true;
        }
        internal_assert(found)
            << "the loop over " << loop.index << " traces the query `" << query
            << "`, which has no programs (see Lower/Trees.cpp)";
    }
    return kernel;
}

void CodeGen_OptiX::add_program(const ir::Function &func,
                                const ir::ssa::Function &ssa) {
    compile_reachable(callees(ssa.blocks));
    llvm::Function *fn = llvm::Function::Create(
        llvm::FunctionType::get(void_t, {}, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, symbol_name(func.name), module.get());
    fn->setCallingConv(llvm::CallingConv::PTX_Kernel);
    internal_assert(!ssa.blocks.empty() && !ssa.blocks.front()->args.empty())
        << func.name << " takes no context";
    const std::string context_name = ssa.blocks.front()->args[0].name;
    compile_function(ssa, fn, [&](const ir::ssa::Argument &declared) -> llvm::Value * {
        llvm::Type *type = codegen_type(declared.type);
        if (declared.name == context_name) {
            // The context: its address in the two payload words.
            llvm::Value *lo = builder->CreateZExt(payload(0), i64_t);
            llvm::Value *hi = builder->CreateZExt(payload(1), i64_t);
            llvm::Value *address = builder->CreateOr(
                lo, builder->CreateShl(hi, 32), declared.name + "_address");
            return builder->CreateIntToPtr(address, type, declared.name);
        }
        const int i = slot_of(declared.name);
        if (i < 0) {
            std::string captured;
            for (const std::string &slot : slot_names) {
                captured += (captured.empty() ? "" : ", ") + slot;
            }
            internal_error
                << func.name << " reads `" << declared.name << "`, which the "
                << "raygen program it runs under does not capture: the hit "
                << "programs read what the launch was handed, by name, and "
                << "the launch was handed " << captured << ".";
        }
        llvm::Value *at = builder->CreateStructGEP(params_ty, params, unsigned(i));
        llvm::Value *value = builder->CreateLoad(slot_types[i], at, declared.name);
        if (value->getType() == type) {
            return value;
        }
        // The slot holds a struct the program takes by address.
        internal_assert(type->isPointerTy())
            << func.name << " takes `" << declared.name << "` as a "
            << declared.type << ", and the launch parameters hold it as a "
            << "different type";
        llvm::Value *local = create_alloca_at_entry(value->getType(), declared.name);
        builder->CreateStore(value, local);
        return local;
    });
}

llvm::Value *CodeGen_OptiX::codegen_rt_intrinsic(const Intrinsic *node) {
    llvm::Type *f32 = llvm::Type::getFloatTy(*context);
    switch (node->op) {
    case Intrinsic::rt_hit_t: {
        // This LLVM's NVPTX declares every 32-bit register `.b32`, and
        // OptiX types its intrinsics' floats `.f32` (nvcc's `%f` registers,
        // which the OptiX headers' constraints name), so a float crosses
        // through an `.f32` register declared in the call's own scope.
        llvm::Value *bits = asm_call(
            i32_t,
            "{ .reg .f32 %ft; call (%ft), _optix_get_ray_tmax, (); "
            "mov.b32 $0, %ft; }",
            "=r", {}, /*side_effects=*/false);
        return builder->CreateBitCast(bits, f32, "_hit_t");
    }
    case Intrinsic::rt_primitive_index:
        return asm_call(i32_t, "call ($0), _optix_read_primitive_idx, ();", "=r",
                        {}, /*side_effects=*/false);
    case Intrinsic::rt_instance_id:
        return asm_call(i32_t, "call ($0), _optix_read_instance_id, ();", "=r",
                        {}, /*side_effects=*/false);
    case Intrinsic::rt_sbt_base: {
        // The record's data: the 32-bit base the runtime put there.
        llvm::Value *data = asm_call(
            i64_t, "call ($0), _optix_get_sbt_data_ptr_64, ();", "=l", {},
            /*side_effects=*/false);
        llvm::Value *ptr = builder->CreateIntToPtr(
            data, llvm::PointerType::getUnqual(*context), "_sbt_data");
        return builder->CreateLoad(i32_t, ptr, "_sbt_base");
    }
    case Intrinsic::rt_report_hit: {
        internal_assert(node->args.size() == 1);
        llvm::Value *t = builder->CreateBitCast(codegen_expr(node->args[0]), i32_t);
        // The hit's t through an `.f32` register (see rt_hit_t).
        llvm::Value *accepted = asm_call(
            i32_t,
            "{ .reg .f32 %ft; mov.b32 %ft, $1; "
            "call ($0), _optix_report_intersection_0, (%ft, $2); }",
            "=r,r,r", {t, llvm::ConstantInt::get(i32_t, 0)},
            /*side_effects=*/true);
        return builder->CreateICmpNE(accepted, llvm::ConstantInt::get(i32_t, 0));
    }
    case Intrinsic::rt_ignore_hit:
        asm_call(nullptr, "call _optix_ignore_intersection, ();", "", {},
                 /*side_effects=*/true);
        return llvm::ConstantInt::get(i32_t, 0);
    case Intrinsic::rt_trace: {
        // Seven operands say what to trace; the rest are the data the hit
        // programs read, carried here so that the launch captures them
        // (Lower/Trees.cpp), and read by the programs out of `params`.
        internal_assert(node->args.size() >= 7);
        const std::string query = query_of_context(node->args[6].type());
        const auto type_at =
            std::find(ray_types.begin(), ray_types.end(), query);
        internal_assert(type_at != ray_types.end())
            << "rt_trace of the query `" << query << "`, which this module "
            << "holds no programs for";
        const unsigned ray_type = unsigned(type_at - ray_types.begin());
        llvm::Value *handle = codegen_expr(node->args[0]);
        llvm::Value *o = codegen_expr(node->args[1]);
        llvm::Value *d = codegen_expr(node->args[2]);
        llvm::Value *tmin = codegen_expr(node->args[3]);
        llvm::Value *tmax = codegen_expr(node->args[4]);
        llvm::Value *flags = codegen_expr(node->args[5]);
        llvm::Value *ctx = builder->CreatePtrToInt(codegen_expr(node->args[6]),
                                                   i64_t, "_ctx_address");
        const auto lane = [&](llvm::Value *v, uint64_t k) {
            return builder->CreateExtractElement(v, k);
        };
        // `optixTrace` as optix_device_impl.h spells it: thirty-two payload
        // registers out, and in -- the payload type, the handle, the ray,
        // the visibility mask, the flags, the shader binding table offset
        // and stride and the miss index (the query's number, with as many
        // types as this module has), how many payload words are used (two:
        // the context's address), then the thirty-two words.
        // The ray's nine floats cross as the bits of 32-bit registers and
        // are moved into `.f32` registers of the call's own scope (see
        // rt_hit_t).
        const auto bits = [&](llvm::Value *f) {
            return builder->CreateBitCast(f, i32_t);
        };
        std::vector<llvm::Value *> args = {
            llvm::ConstantInt::get(i32_t, 0), // OPTIX_PAYLOAD_TYPE_DEFAULT
            handle,
            bits(lane(o, 0)), bits(lane(o, 1)), bits(lane(o, 2)),
            bits(lane(d, 0)), bits(lane(d, 1)), bits(lane(d, 2)),
            bits(tmin), bits(tmax),
            llvm::ConstantInt::get(i32_t, 0), // time, 0.0f
            llvm::ConstantInt::get(i32_t, 255), // visibility mask
            builder->CreateIntCast(flags, i32_t, false),
            llvm::ConstantInt::get(i32_t, ray_type),
            llvm::ConstantInt::get(i32_t, uint64_t(ray_types.size())),
            llvm::ConstantInt::get(i32_t, ray_type),
            llvm::ConstantInt::get(i32_t, 2), // payload words used
            builder->CreateTrunc(ctx, i32_t),
            builder->CreateTrunc(builder->CreateLShr(ctx, 32), i32_t),
        };
        while (args.size() < 17 + 32) {
            args.push_back(llvm::ConstantInt::get(i32_t, 0));
        }
        // LLVM's inline assembly names an operand `$k`: the thirty-two
        // outputs are $0..$31, the payload type $32, the handle $33, the
        // nine floats $34..$42 (moved into %fr0..%fr8), and the rest from
        // $43 on.
        std::string text = "{ .reg .f32 %fr<9>; ";
        for (unsigned k = 0; k < 9; k++) {
            text += "mov.b32 %fr" + std::to_string(k) + ", $" +
                    std::to_string(34 + k) + "; ";
        }
        text += "call (";
        for (unsigned k = 0; k < 32; k++) {
            text += (k ? ",$" : "$") + std::to_string(k);
        }
        text += "), _optix_trace_typed_32, ($32,$33";
        for (unsigned k = 0; k < 9; k++) {
            text += ",%fr" + std::to_string(k);
        }
        for (unsigned k = 43; k < 32 + 17 + 32; k++) {
            text += ",$" + std::to_string(k);
        }
        text += "); }";
        std::string constraints;
        for (unsigned k = 0; k < 32; k++) {
            constraints += "=r,";
        }
        constraints += "r,l,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r";
        for (unsigned k = 0; k < 32; k++) {
            constraints += ",r";
        }
        std::vector<llvm::Type *> outs(32, i32_t);
        llvm::StructType *out_ty = llvm::StructType::get(*context, outs);
        asm_call(out_ty, text, constraints, args, /*side_effects=*/true);
        return llvm::ConstantInt::get(i32_t, 0);
    }
    case Intrinsic::rt_traversable:
        internal_error << "rt_traversable reached code generation: "
                       << "Lower/Layouts.cpp spells it as the layout's field";
    default:
        break;
    }
    return CodeGen_PTX::codegen_rt_intrinsic(node);
}

llvm::Value *CodeGen_OptiX::atomic_address(llvm::Value *loc) {
    // Rooted at the launch parameters -- an array's handle read out of
    // `params` -- is device memory: the global space, so that the atomic is
    // the hardware's rather than a compare-and-swap loop.
    const llvm::Value *root = llvm::getUnderlyingObject(loc);
    const auto *load = llvm::dyn_cast<llvm::LoadInst>(root);
    if (load != nullptr &&
        llvm::getUnderlyingObject(load->getPointerOperand()) == params) {
        return builder->CreateAddrSpaceCast(
            loc, llvm::PointerType::get(*context, /*AddressSpace=*/1),
            loc->getName() + ".global");
    }
    return CodeGen_PTX::atomic_address(loc);
}

} // namespace bonsai
