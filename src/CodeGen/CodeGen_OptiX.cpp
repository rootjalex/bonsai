#include "CodeGen/CodeGen_OptiX.h"

#include "IR/Printer.h"
#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "Error.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InlineAsm.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>

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

void CodeGen_OptiX::find_read_slots(const ir::ssa::Function &host,
                                    const Terminator::ParFor &loop,
                                    const Block &body_head,
                                    const std::vector<size_t> &captures,
                                    const std::vector<bool> &by_value) {
    using ir::ssa::Definition;
    using ir::ssa::Definitions;
    read_slots.clear();

    // The body's region and every function it reaches: where the traces
    // are, and who calls each function, from which block, with what.
    struct Site {
        const ir::ssa::Function *caller; // &host for the body's region
        std::string block;
        std::vector<std::shared_ptr<Value>> args;
        std::vector<size_t> varying; // a run's positions, which no slot is
    };
    struct Trace {
        const ir::ssa::Function *in;
        std::string block;
        const Instruction *instr;
    };
    std::map<std::string, std::vector<Site>> sites;
    std::map<const ir::ssa::Function *, std::string> names;
    std::vector<Trace> traces;
    const auto scan = [&](const ir::ssa::Function *in,
                          const std::vector<std::shared_ptr<Block>> &blocks) {
        for (const auto &block : blocks) {
            for (const auto &instr : block->instrs) {
                if (instr->op == Instruction::Op::Intrinsic &&
                    instr->intrinsic == Intrinsic::rt_trace) {
                    traces.push_back({in, block->name, instr.get()});
                }
            }
            std::visit(ir::ssa::overloads{
                           [&](const Terminator::Call &c) {
                               sites[c.call.name].push_back(
                                   {in, block->name, c.call.args, {}});
                           },
                           [&](const Terminator::MultiCall &c) {
                               sites[c.call.name].push_back(
                                   {in, block->name, c.call.args, c.varying_at});
                           },
                           [](const auto &) {},
                       },
                       block->terminator.data);
        }
    };
    const std::vector<std::shared_ptr<Block>> region =
        ir::ssa::Cfg(host, loop.body.name).blocks();
    scan(&host, region);
    {
        std::set<std::string> seen;
        std::vector<std::string> work = callees(region);
        while (!work.empty()) {
            const std::string name = work.back();
            work.pop_back();
            if (!seen.insert(name).second) {
                continue;
            }
            const auto f = program->ssa_funcs.find(name);
            internal_assert(f != program->ssa_funcs.end())
                << "`" << name << "` has no SSA form";
            names[f->second.get()] = name;
            scan(f->second.get(), f->second->blocks);
            for (std::string &callee : callees(f->second->blocks)) {
                work.push_back(std::move(callee));
            }
        }
    }
    if (traces.empty()) {
        return;
    }

    // Where a value is defined. Lenient, since a block refers to a value of
    // a block above it by name as often as it takes it as an argument, and
    // the strict form refuses the first; what the lenient form answers for
    // such a name is the block that asked, and `resolve` below reads the
    // name instead: a function's parameters and the launched body's head
    // arguments are in scope throughout, and names are unique within a
    // function, so the name says which it is.
    std::map<const ir::ssa::Function *, std::unique_ptr<Definitions>> defs;
    std::map<const ir::ssa::Function *, ir::ssa::BlockMap> blocks_of;
    const auto definitions = [&](const ir::ssa::Function *f) -> Definitions & {
        auto &d = defs[f];
        if (!d) {
            d = std::make_unique<Definitions>(*f, /*lenient=*/true);
            blocks_of[f] = ir::ssa::make_block_map(*f);
        }
        return *d;
    };
    const auto is_own_argument = [&](const ir::ssa::Function *f,
                                     const std::string &block,
                                     const ir::ssa::Argument &a) {
        definitions(f);
        const auto b = blocks_of.at(f).find(block);
        internal_assert(b != blocks_of.at(f).end());
        for (const ir::ssa::Argument &arg : b->second->args) {
            if (arg.name == a.name) {
                return true;
            }
        }
        return false;
    };
    // The definition of `v` as `block` of `in` refers to it, a name reached
    // from above resolved to the parameter (or, under the launched loop,
    // the body's head argument) of that name.
    const auto resolve = [&](const ir::ssa::Function *in, const std::string &block,
                             const std::shared_ptr<Value> &v) -> Definition {
        const Definition d = definitions(in).of(block, v);
        const auto *a = std::get_if<ir::ssa::Argument>(&d.value->data);
        if (a == nullptr || d.block.empty() || is_own_argument(in, d.block, *a)) {
            return d;
        }
        if (in == &host) {
            for (const size_t c : captures) {
                if (body_head.args[c].name == a->name) {
                    return Definition{std::make_shared<Value>(body_head.args[c]),
                                      body_head.name};
                }
            }
        }
        for (const ir::ssa::Argument &param : in->blocks.front()->args) {
            if (param.name == a->name) {
                return Definition{std::make_shared<Value>(param),
                                  in->blocks.front()->name};
            }
        }
        return d;
    };
    // What each capture is: as the body's head has it, and as the loop was
    // handed it -- what a read has to turn out to be, either way.
    std::vector<Definition> captured_here;
    std::vector<Definition> captured_outside;
    for (const size_t c : captures) {
        const auto value = std::make_shared<Value>(body_head.args[c]);
        captured_here.push_back(Definition{value, body_head.name});
        captured_outside.push_back(definitions(&host).of(body_head.name, value));
    }

    // The slot of `v` as `block` of `in` refers to it: a capture's, or -1
    // with the reason in `why`, or `carried` for a value a recursion passes
    // round unchanged, which the calls into the recursion decide.
    constexpr int carried = -2;
    std::string why;
    std::set<const ir::ssa::Function *> visiting;
    std::function<int(const ir::ssa::Function *, const std::string &,
                      const std::shared_ptr<Value> &)>
        slot_of_value;
    slot_of_value = [&](const ir::ssa::Function *in, const std::string &block,
                        const std::shared_ptr<Value> &v) -> int {
        const Definition d = resolve(in, block, v);
        if (in == &host) {
            for (size_t k = 0; k < captures.size(); k++) {
                if (same_definition(d, captured_here[k]) ||
                    same_definition(d, captured_outside[k])) {
                    return int(k) + 2;
                }
            }
            // A struct the launch holds by value: the body has its address
            // and loads it to pass it on, and the load is the slot's
            // contents.
            if (const auto *load = std::get_if<std::shared_ptr<Instruction>>(&d.value->data);
                load != nullptr && (*load)->op == Instruction::Op::Load &&
                !(*load)->operands.empty()) {
                const int slot = slot_of_value(&host, d.block, (*load)->operands[0]);
                if (slot >= 2 && by_value[size_t(slot) - 2]) {
                    return slot;
                }
            }
            why = "it is computed inside the launch rather than captured";
            return -1;
        }
        const std::string &name = names.at(in);
        const auto *param = std::get_if<ir::ssa::Argument>(&d.value->data);
        if (param == nullptr || d.block != in->blocks.front()->name) {
            why = "`" + name + "` computes it rather than taking it as a parameter";
            return -1;
        }
        const std::vector<ir::ssa::Argument> &params = in->blocks.front()->args;
        size_t j = params.size();
        for (size_t i = 0; i < params.size(); i++) {
            if (params[i].name == param->name) {
                j = i;
            }
        }
        internal_assert(j < params.size());
        // A call reached while its callee's own parameter is being resolved
        // is a recursion -- a path step calling itself for the next bounce,
        // passing the pools along unchanged -- and says nothing new about
        // where the value came from: whatever entered the recursion is what
        // it is, which the other calls say.
        if (!visiting.insert(in).second) {
            return carried;
        }
        int found = -1;
        bool any = false;
        for (const Site &site : sites[name]) {
            if (std::find(site.varying.begin(), site.varying.end(), j) !=
                site.varying.end()) {
                why = "`" + name + "` is called with a different one per call";
                visiting.erase(in);
                return -1;
            }
            internal_assert(j < site.args.size())
                << "a call of `" << name << "` passes " << site.args.size()
                << " arguments for " << params.size() << " parameters";
            const int slot = slot_of_value(site.caller, site.block, site.args[j]);
            if (slot == carried) {
                continue;
            }
            if (slot < 0) {
                visiting.erase(in);
                return -1;
            }
            if (any && slot != found) {
                why = "two calls of `" + name + "` pass different ones";
                visiting.erase(in);
                return -1;
            }
            found = slot;
            any = true;
        }
        visiting.erase(in);
        if (!any) {
            // Every call of it is from inside a recursion still being
            // resolved -- a mutual recursion's other members -- so its
            // answer is theirs; only a function nothing calls at all is
            // unresolved.
            if (!visiting.empty()) {
                return carried;
            }
            why = "nothing in the launch calls `" + name + "`";
            return -1;
        }
        return found;
    };

    for (const Trace &t : traces) {
        internal_assert(t.instr->operands.size() >= 7);
        const std::string query =
            query_of_context(t.instr->operands[6]->get_type());
        std::vector<std::string> reads;
        for (const auto &[_, func] : program->funcs) {
            if (func->optix_program && func->optix_program->of == query) {
                reads = func->optix_program->reads;
                break;
            }
        }
        const size_t n = t.instr->operands.size() - 7;
        internal_assert(reads.size() == n)
            << "the trace of `" << query << "` carries " << n
            << " reads and its programs list " << reads.size()
            << " (Lower/RTCoreReads.h)";
        std::vector<int> &slots = read_slots[query];
        if (slots.empty()) {
            slots.assign(n, -1);
        }
        const std::string where =
            t.in == &host ? "the launched loop's body" : "`" + names.at(t.in) + "`";
        for (size_t k = 0; k < n; k++) {
            why.clear();
            const int slot = slot_of_value(t.in, t.block, t.instr->operands[7 + k]);
            std::ostringstream carried;
            t.instr->operands[7 + k]->dump(carried);
            internal_assert(slot >= 0)
                << "the programs of `" << query << "` read `" << reads[k]
                << "`, which the trace in " << where << " carries as "
                << carried.str() << ", and that is not one of the launch's "
                << "parameters: " << why << ". The hit programs read what the "
                << "launch was handed (Lower/RTCoreReads.h).";
            internal_assert(slots[k] < 0 || slots[k] == slot)
                << "`" << query << "` is traced with two different values of `"
                << reads[k] << "` under one launch: launch parameters "
                << slots[k] << " and " << slot << ".";
            slots[k] = slot;
        }
    }
    if (std::getenv("BONSAI_EXPLAIN_DEVICE") != nullptr) {
        for (const auto &[query, slots] : read_slots) {
            std::vector<std::string> reads;
            for (const auto &[_, func] : program->funcs) {
                if (func->optix_program && func->optix_program->of == query) {
                    reads = func->optix_program->reads;
                    break;
                }
            }
            std::cerr << "raygen over " << loop.index << ": the programs of `"
                      << query << "` read\n";
            for (size_t k = 0; k < slots.size(); k++) {
                std::cerr << "  " << reads[k] << " from launch parameter "
                          << slots[k] << " (" << slot_names[size_t(slots[k])]
                          << ")\n";
            }
        }
    }
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
    // Which slot each of the traced queries' reads is in, for the programs.
    find_read_slots(host, loop, *body_head, captures, by_value);
    params_ty = llvm::StructType::create(*context, slot_types, "_params");
    {
        const llvm::StructLayout *layout = dl.getStructLayout(params_ty);
        for (size_t i = 0; i < slot_types.size(); i++) {
            kernel.param_offsets.push_back(layout->getElementOffset(unsigned(i)));
        }
        kernel.param_struct_bytes = layout->getSizeInBytes();
        if (std::getenv("BONSAI_EXPLAIN_DEVICE") != nullptr) {
            std::cerr << "raygen over " << loop.index << ": "
                      << slot_types.size() << " launch parameters in "
                      << kernel.param_struct_bytes << " bytes\n";
            for (size_t i = 0; i < slot_types.size(); i++) {
                std::cerr << "  " << i << ": " << slot_names[i] << " at byte "
                          << kernel.param_offsets[i] << ", "
                          << kernel.param_bytes[i] << " bytes\n";
            }
        }
    }
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
        // The slot: the parameter's place in the query's list of reads is
        // which operand of the trace carried it, and find_read_slots
        // followed that operand to the capture.
        const std::vector<std::string> &reads = func.optix_program->reads;
        const auto place = std::find(reads.begin(), reads.end(), declared.name);
        internal_assert(place != reads.end())
            << func.name << " reads `" << declared.name << "`, which is not in "
            << "its query's list of reads (Lower/RTCoreReads.h)";
        const auto slots = read_slots.find(func.optix_program->of);
        internal_assert(slots != read_slots.end() &&
                        size_t(place - reads.begin()) < slots->second.size())
            << func.name << ": the slots of `" << func.optix_program->of
            << "`'s reads were not found under this raygen program";
        const int i = slots->second[size_t(place - reads.begin())];
        internal_assert(i >= 0);
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
    case Intrinsic::rt_reorder: {
        // `optixReorder(coherenceHint, numCoherenceHintBitsFromLSB)`: the
        // launch's threads sorted so that those agreeing on the key's low
        // bits run together from here on; with no hit object outstanding
        // (the trace has returned) the key is all there is to sort by.
        internal_assert(node->args.size() == 2);
        llvm::Value *key = builder->CreateIntCast(codegen_expr(node->args[0]),
                                                  i32_t, false);
        llvm::Value *bits = builder->CreateIntCast(codegen_expr(node->args[1]),
                                                   i32_t, false);
        asm_call(nullptr, "call (), _optix_hitobject_reorder, ($0, $1);", "r,r",
                 {key, bits}, /*side_effects=*/true);
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
