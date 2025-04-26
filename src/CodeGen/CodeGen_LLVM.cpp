#include "CodeGen/CodeGen_LLVM.h"

#include <llvm/IR/Constant.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>

#include <llvm/MC/TargetRegistry.h>

#include <llvm/Passes/PassBuilder.h>
// #include <llvm/Passes/StandardInstrumentations.h>
// #include <llvm/Support/TargetSelect.h>
// #include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation/AddressSanitizer.h>
#include <llvm/Transforms/Instrumentation/SanitizerCoverage.h>
#include <llvm/Transforms/Instrumentation/ThreadSanitizer.h>
#include <llvm/Transforms/Utils/RelLookupTableConverter.h>
// #include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <llvm/Support/CodeGen.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include "IR/Analysis.h"
#include "IR/Expr.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <sstream>

namespace bonsai {
namespace codegen {
void to_llvm(const ir::Program &program, const CompilerOptions &options) {
    CodeGen_LLVM codegen;
    std::unique_ptr<llvm::Module> module =
        codegen.compile_program(program, options);
    if (options.output_file.empty()) {
        module->print(llvm::outs(), /*AAW=*/nullptr);
        return;
    }
    auto os = make_raw_fd_ostream(options.output_file);
    module->print(*os, /*AAW=*/nullptr);
}
} // namespace codegen
namespace {

// Returns the `printf` function for this module. If none exists, it is created.
static llvm::Function *retrieve_printf(llvm::Module &m) {
    llvm::Function *printf;
    if ((printf = m.getFunction("printf"))) {
        return printf;
    }

    llvm::LLVMContext &context = m.getContext();
    auto *functy = llvm::FunctionType::get(
        llvm::IntegerType::get(context, 32),
        llvm::PointerType::get(llvm::IntegerType::get(context, 8),
                               /*AddressSpace=*/0),
        /*isVarArg=*/true);
    printf = llvm::Function::Create(functy, llvm::GlobalValue::ExternalLinkage,
                                    "printf", m);
    printf->setCallingConv(llvm::CallingConv::C);
    return printf;
}

// Returns the printf format specifier for this value, or error if it is not
// implemented yet.
// TODO(cgyurgyik): Add support for non-standard types.
static std::string get_specifier(const ir::Type &type) {
    std::string specifier = "%";
    const uint32_t width = type.bits();
    if (type.is_bool()) {
        // Boolean values are printed as strings ("true", "false").
        return "%s";
    }
    if (!(type.is_numeric() && (width == 32 || width == 64))) {
        internal_error << "[unimplemented] LLVM print: " << type;
    }
    if (type.is_int()) {
        if (width > 32)
            return "%ld";
        return "%d";
    }
    if (type.is_uint()) {
        if (width > 32)
            return "%lu";
        return "%u";
    }
    if (type.is_float()) {
        // C will convert float (f32) to double (f64) for variadic
        // argument functions (to include printf).
        return "%f";
    }

    internal_error << "[unimplemented] LLVM print: " << type;
}

} // namespace

using namespace ir;

std::unique_ptr<llvm::TargetMachine>
CodeGen_LLVM::make_target_machine(llvm::Module &module,
                                  const CompilerOptions &options) {
    // TODO(cgyurgyik): This effectively makes our tests host-machine specific.
    std::string target_triple = llvm::sys::getDefaultTargetTriple(),
                error_string;
    const llvm::Target *llvm_target =
        llvm::TargetRegistry::lookupTarget(target_triple, error_string);
    if (llvm_target == nullptr) {
        llvm::errs() << error_string << "\n";
        llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::errs());
        internal_error << "could not create LLVM target for: " << target_triple;
    }
    llvm::Triple triple = llvm::Triple(target_triple);
    llvm::TargetOptions target_options;

    // TODO: set options?
    // target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    // target_options.UnsafeFPMath = true;
    // target_options.NoInfsFPMath = true;
    // target_options.NoNaNsFPMath = true;
    // get_target_options(module, target_options);

    bool use_pic = true;
    // get_md_bool(module.getModuleFlag("bonsai_use_pic"), use_pic);

    bool use_large_code_model = false;
    // get_md_bool(module.getModuleFlag("bonsai_use_large_code_model"),
    // use_large_code_model);

    auto *tm = llvm_target->createTargetMachine(
        module.getTargetTriple(),
        /*CPU target=*/"", /*Features=*/"", target_options,
        use_pic ? llvm::Reloc::PIC_ : llvm::Reloc::Static,
        use_large_code_model ? llvm::CodeModel::Large : llvm::CodeModel::Small,
        llvm::CodeGenOptLevel::Aggressive);

    switch (options.target) {
    case BackendTarget::ASM:
    case BackendTarget::CPP: {
        // These two backends *require* a data layout.
        module.setDataLayout(tm->createDataLayout());
        module.setTargetTriple(target_triple);
        break;
    }
    default:
        // TODO(cgyurgyik): should all backends using LLVM be machine specific?
        // Pros: we see the actual code being generated. Cons: our tests either
        // become host-machine specific or are defaulted to a specific machine.
        break;
    }
    return std::unique_ptr<llvm::TargetMachine>(tm);
}

void CodeGen_LLVM::print_module(llvm::Module &module, llvm::raw_ostream &os,
                                bool redacted) {
    if (!redacted) {
        module.print(os, nullptr);
        return;
    }
    std::string triple = module.getTargetTriple();
    llvm::DataLayout layout = module.getDataLayout();
    {
        module.setTargetTriple("");
        module.setDataLayout("");
        module.print(os, nullptr);
    }
    // Reset these in case these are referenced later.
    module.setTargetTriple(std::move(triple));
    module.setDataLayout(std::move(layout));
}

CodeGen_LLVM::CodeGen_LLVM() {
    // TODO: set up independent state (e.g. wildcard matchers)

    init_llvm();
}

void CodeGen_LLVM::init_llvm() {
    static std::once_flag init_llvm_once;
    std::call_once(init_llvm_once, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // TODO: allow these.
        // #define LLVM_TARGET(target) \
//     Initialize##target##Target();
        // #include <llvm/Config/Targets.def>
        // #undef LLVM_TARGET

        // #define LLVM_ASM_PARSER(target) \
//     Initialize##target##AsmParser();
        // #include <llvm/Config/AsmParsers.def>
        // #undef LLVM_ASM_PARSER

        // #define LLVM_ASM_PRINTER(target) \
//     Initialize##target##AsmPrinter();
        // #include <llvm/Config/AsmPrinters.def>
        // #undef LLVM_ASM_PRINTER
    });
}

void CodeGen_LLVM::init_context() {
    // TODO: Halide passes this in as an argument for some reason?
    // open new context and module
    context = std::make_unique<llvm::LLVMContext>();

    // Create a new builder for the module.
    // TODO: there might be params to the IRBuilder...
    builder = std::make_unique<llvm::IRBuilder<>>(*context);

    // Branch weights for very likely branches
    llvm::MDBuilder md_builder(*context);
    very_likely_branch = md_builder.createBranchWeights(1 << 30, 0);
    llvm::MDNode *default_fp_math_md = md_builder.createFPMath(0.0);
    // TODO: when do we need strict float math?
    // strict_fp_math_md = md_builder.createFPMath(0.0);
    builder->setDefaultFPMathTag(default_fp_math_md);
    llvm::FastMathFlags fast_flags;
    /*
    // TODO: are these even allowed? Halide adds these, I don't think they're
    safe though. fast_flags.setNoNaNs(); fast_flags.setNoInfs();
    fast_flags.setNoSignedZeros();
    // Don't use approximate reciprocals for division. It's too inaccurate even
    for Halide.
    // fast_flags.setAllowReciprocal();
    // Theoretically, setAllowReassoc could be setUnsafeAlgebra for earlier
    versions, but that
    // turns on all the flags.
    fast_flags.setAllowReassoc();
    fast_flags.setAllowContract(true);
    fast_flags.setApproxFunc();
    */
    builder->setFastMathFlags(fast_flags);

    // Define some types
    void_t = llvm::Type::getVoidTy(*context);
    i1_t = llvm::Type::getInt1Ty(*context);
    i8_t = llvm::Type::getInt8Ty(*context);
    i16_t = llvm::Type::getInt16Ty(*context);
    i32_t = llvm::Type::getInt32Ty(*context);
    i64_t = llvm::Type::getInt64Ty(*context);
    f16_t = llvm::Type::getHalfTy(*context);
    f32_t = llvm::Type::getFloatTy(*context);
    f64_t = llvm::Type::getDoubleTy(*context);
}

void CodeGen_LLVM::init_module() {
    init_context();

    // Start with a module containing the initial module for this target.
    // module = get_initial_module_for_target(target, context);
    // TODO: handle all the module set-up that Halide does.
    module = std::make_unique<llvm::Module>("bonsai_module", *context);
}

llvm::Function *CodeGen_LLVM::declare_function(const Function &func) {
    // Make function type
    llvm::Type *ret_type = codegen_type(func.ret_type);
    std::vector<llvm::Type *> arg_types(func.args.size());
    for (uint32_t i = 0; i < func.args.size(); i++) {
        const auto &arg_info = func.args[i];
        llvm::Type *arg_t = codegen_type(arg_info.type);
        if (arg_info.mutating || arg_info.type.is<Struct_t>()) {
            arg_t = arg_t->getPointerTo();
        }
        arg_types[i] = arg_t;
    }

    llvm::FunctionType *ftype =
        llvm::FunctionType::get(ret_type, arg_types, /*isVarArg=*/false);

    llvm::Function *fn = llvm::Function::Create(
        ftype, llvm::GlobalValue::ExternalLinkage, func.name, module.get());

    for (uint32_t i = 0; i < func.args.size(); i++) {
        const auto &arg_info = func.args[i];
        llvm::AttrBuilder attrs(*context);

        attrs.addAttribute(llvm::Attribute::NoUndef);

        if (arg_info.type.is<Struct_t>() || arg_info.mutating) {
            attrs.addAttribute(llvm::Attribute::NonNull);

            if (!arg_info.mutating) {
                attrs.addAttribute(llvm::Attribute::ReadOnly);
            }

            // TODO: Add dereferenceable + alignment if we can figure that out.
        }

        fn->addParamAttrs(i, attrs);
    }
    return fn;
}

void CodeGen_LLVM::compile_function(const Function &func,
                                    llvm::Function *function) {
    frames.new_frame();

    // TODO: allow nested functions? Can LLVM even do that?
    internal_assert(current_function == nullptr);
    internal_assert(function);
    current_function = function;

    // Add entry point.
    llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(
        module->getContext(), func.name + "_entry", function);
    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    builder->SetInsertPoint(entry_bb);

    uint32_t arg_idx = 0;
    for (auto &arg : function->args()) {
        const auto &arg_info = func.args[arg_idx];
        std::string name = arg_info.name;
        llvm::Value *arg_value = &arg;
        // immutable structs are ptrs, so need some indirection.
        if (const bool immutable_struct =
                arg_info.type.is<Struct_t>() && !arg_info.mutating;
            immutable_struct) {
            llvm::Type *arg_type = codegen_type(arg_info.type);
            arg_value = builder->CreateLoad(arg_type, arg_value, name);
        }
        arg.setName(name);
        frames.add_to_frame(arg_info.name,
                            {arg_value, /*mutable=*/arg_info.mutating});
        arg_idx++;
    }

    codegen_stmt(func.body);
    frames.pop_frame();

    // Restore previous insertion point
    builder->restoreIP(here);

    // Validate the generated code, checking for consistency.
    verifyFunction(*function);

    current_function = nullptr;

    // function->dump();
}

std::unique_ptr<llvm::Module>
CodeGen_LLVM::compile_program(const Program &program,
                              const CompilerOptions &options) {
    init_module(); // TODO: init_codegen()?

    const auto struct_types = gather_struct_types(program);
    declare_struct_types(struct_types);

    frames.new_frame();
    // TODO: add program.externs to the global frame.
    std::map<std::string, llvm::Function *> func_map;
    for (const auto &[fname, func] : program.funcs) {
        func_map[fname] = this->declare_function(*func);
    }
    for (const auto &[fname, func] : program.funcs) {
        this->compile_function(*func, func_map[fname]);
    }
    frames.pop_frame();

    std::unique_ptr<llvm::TargetMachine> tm =
        make_target_machine(*module, options);

    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[pre-optimization] compilation resulted in an invalid module";
    optimize_module(*tm, options);
    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[post-optimization] compilation resulted in an invalid module";

    return std::move(module);
}

void CodeGen_LLVM::optimize_module(llvm::TargetMachine &tm,
                                   const CompilerOptions &options) {

    const bool do_loop_opt =
        true; // get_target().has_feature(Target::EnableLLVMLoopOpt);

    llvm::PipelineTuningOptions pto;
    pto.LoopInterleaving = do_loop_opt;
    pto.LoopVectorization = do_loop_opt;
    pto.SLPVectorization =
        true; // Note: SLP vectorization has no analogue in the scheduling model
    pto.LoopUnrolling = do_loop_opt;

    llvm::PassBuilder pb(&tm, pto);

    bool debug_pass_manager = false;
    // These analysis managers have to be declared in this order.
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::FunctionPassManager fpm;

    // TODO: add other explicit passes?
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    fpm.addPass(llvm::InstCombinePass());
    // Reassociate expressions.
    fpm.addPass(llvm::ReassociatePass());
    // Eliminate Common SubExpressions.
    fpm.addPass(llvm::GVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    fpm.addPass(llvm::SimplifyCFGPass());

    // Register all the basic analyses with the managers.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    llvm::ModulePassManager mpm;

    using OptimizationLevel = llvm::OptimizationLevel;
    OptimizationLevel level = OptimizationLevel::O3;

    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));

    if (tm.isPositionIndependent()) {
        // Add a pass that converts lookup tables to relative lookup tables to
        // make them PIC-friendly. See
        // https://bugs.llvm.org/show_bug.cgi?id=45244
        pb.registerOptimizerLastEPCallback(
#if LLVM_VERSION >= 200
            [&](ModulePassManager &mpm, OptimizationLevel, ThinOrFullLTOPhase)
#else
            [&](llvm::ModulePassManager &mpm, OptimizationLevel)
#endif
            { mpm.addPass(llvm::RelLookupTableConverterPass()); });
    }

    // get_target().has_feature(Target::SanitizerCoverage)
    if (false) {
        pb.registerOptimizerLastEPCallback([&](llvm::ModulePassManager &mpm,
                                               llvm::OptimizationLevel level) {
            llvm::SanitizerCoverageOptions sanitizercoverage_options;
            // Mirror what -fsanitize=fuzzer-no-link would enable.
            // See https://github.com/halide/Halide/issues/6528
            sanitizercoverage_options.CoverageType =
                llvm::SanitizerCoverageOptions::SCK_Edge;
            sanitizercoverage_options.IndirectCalls = true;
            sanitizercoverage_options.TraceCmp = true;
            sanitizercoverage_options.Inline8bitCounters = true;
            sanitizercoverage_options.PCTable = true;
            // Due to TLS differences, stack depth tracking is only enabled on
            // Linux if (get_target().os == Target::OS::Linux) {
            // sanitizercoverage_options.StackDepth = true;
            // }
            mpm.addPass(llvm::SanitizerCoveragePass(sanitizercoverage_options));
        });
    }

    // get_target().has_feature(Target::ASAN)
    if (false) {
        // Nothing, ASanGlobalsMetadataAnalysis no longer exists

        pb.registerPipelineStartEPCallback([](llvm::ModulePassManager &mpm,
                                              OptimizationLevel) {
            llvm::AddressSanitizerOptions
                asan_options;                  // default values are good...
            asan_options.UseAfterScope = true; // ...except this one
            constexpr bool use_global_gc = false;
            constexpr bool use_odr_indicator = true;
            constexpr auto destructor_kind = llvm::AsanDtorKind::Global;
            mpm.addPass(llvm::AddressSanitizerPass(asan_options, use_global_gc,
                                                   use_odr_indicator,
                                                   destructor_kind));
        });
    }

    // Target::MSAN handling is sprinkled throughout the codebase,
    // there is no need to run MemorySanitizerPass here.

    // get_target().has_feature(Target::TSAN)
    if (false) {
        pb.registerOptimizerLastEPCallback(
            [](llvm::ModulePassManager &mpm, OptimizationLevel level) {
                mpm.addPass(llvm::createModuleToFunctionPassAdaptor(
                    llvm::ThreadSanitizerPass()));
            });
    }

    for (auto &function : *module) {
        if (false) { // get_target().has_feature(Target::ASAN)
            function.addFnAttr(llvm::Attribute::SanitizeAddress);
        }
        if (false) { // get_target().has_feature(Target::MSAN)
            function.addFnAttr(llvm::Attribute::SanitizeMemory);
        }
        if (false) { // get_target().has_feature(Target::TSAN)
            // Do not annotate any of Halide's low-level synchronization code as
            // it has tsan interface calls to mark its behavior and is much
            // faster if it is not analyzed instruction by instruction. if
            // (!(function.getName().startswith("_ZN6Halide7Runtime8Internal15Synchronization")
            // ||
            //       // TODO: this is a benign data race that re-initializes the
            //       detected features;
            //       // we should really fix it properly inside the
            //       implementation, rather than disabling
            //       // it here as a band-aid.
            //       function.getName().startswith("halide_default_can_use_target_features")
            //       || function.getName().startswith("halide_mutex_") ||
            //       function.getName().startswith("halide_cond_"))) {
            //     function.addFnAttr(llvm::Attribute::SanitizeThread);
            // }
        }
    }

    tm.registerPassBuilderCallbacks(pb);
    mpm = pb.buildPerModuleDefaultPipeline(level, debug_pass_manager);

    for (auto &F : *module) {
        if (llvm::verifyFunction(F, &llvm::errs())) {
            F.print(llvm::errs());
            internal_error << "Invalid function IR before optimization";
        }
    }

    mpm.run(*module, mam);
}

void CodeGen_LLVM::visit(const Int_t *node) {
    type = llvm::Type::getIntNTy(*context, node->bits);
}

void CodeGen_LLVM::visit(const Void_t *node) { type = void_t; }

void CodeGen_LLVM::visit(const UInt_t *node) {
    // LLVM does not distinguish between signed and unsigned integer types.
    type = llvm::Type::getIntNTy(*context, node->bits);
}

void CodeGen_LLVM::visit(const Index_t *node) {
    internal_error << "unimplemented: " << ir::Type(node);
}

void CodeGen_LLVM::visit(const Bool_t *node) { type = i1_t; }

void CodeGen_LLVM::visit(const Float_t *node) {
    switch (node->bits()) {
    case 64:
        if (node->is_ieee754()) {
            type = llvm::Type::getDoubleTy(*context);
            return;
        }
        break;
    case 32:
        if (node->is_ieee754()) {
            type = llvm::Type::getFloatTy(*context);
            return;
        }
        break;
    case 16:
        if (node->is_ieee754()) {
            type = llvm::Type::getHalfTy(*context);
            return;
        }
        if (node->is_bfloat16()) {
            type = llvm::Type::getBFloatTy(*context);
            return;
        }
        break;
    case 8: // TODO: I need f8 on GPUs. Do we ever need it on CPUs?
    default:
        break;
    }
    internal_error << "unimplemented: " << Type(node);
}

void CodeGen_LLVM::visit(const Ptr_t *node) {
    llvm::Type *etype = codegen_type(node->etype);
    // TODO: what does the address space parameter to this function do?
    type = etype->getPointerTo();
}

void CodeGen_LLVM::visit(const Ref_t *node) {
    internal_error << "Figure out LLVM code generation for reference: "
                   << ir::Type(node);
    // llvm::Type *etype = codegen_type(node->etype);
    // type = etype->getPointerTo();
}

void CodeGen_LLVM::visit(const Vector_t *node) {
    llvm::Type *etype = codegen_type(node->etype);
    internal_assert(!etype->isVoidTy())
        << "Cannot make a vector of type void: " << Type(node);
    // TODO: do we ever want to support scalable vectors? probably not.
    type = llvm::VectorType::get(etype, node->lanes, /* Scalable */ false);
}

void CodeGen_LLVM::visit(const Struct_t *node) {
    // TODO: could just use module->getTypeByName
    type = struct_types[node->name];
}

void CodeGen_LLVM::visit(const Tuple_t *node) {
    // TODO: struct_types should include tuples, probably? but they're
    // unnamed... maybe use to_string() to map from node to built Struct_t
    internal_error << "TODO: implement Tuple_t code generation: " << Type(node);
}

void CodeGen_LLVM::visit(const Array_t *node) {
    // TODO: are nested arrays allowed?
    // lowering should probably flatten arrays.
    llvm::Type *etype = codegen_type(node->etype);
    type = etype->getPointerTo();
    // We don't use LLVM's ArrayType because these are allocated objects.
    /*
    if (is_const(node->size)) {
        const uint64_t size = get_constant_value(node->size);
        type = llvm::ArrayType::get(etype, size);
    } else {
        internal_error << "TODO: implement Array_t code generation for dynamic
    sizes: " << Type(node);
    }
    */
}

void CodeGen_LLVM::visit(const IntImm *node) {
    value = llvm::ConstantInt::getSigned(codegen_type(node->type), node->value);
}

void CodeGen_LLVM::visit(const UIntImm *node) {
    value = llvm::ConstantInt::get(codegen_type(node->type), node->value,
                                   /* IsSigned */ false);
}

void CodeGen_LLVM::visit(const FloatImm *node) {
    // TODO: Halide does some weird stuff for f16.
    // Make sure this works on f16?
    value = llvm::ConstantFP::get(codegen_type(node->type), node->value);
}

void CodeGen_LLVM::visit(const BoolImm *node) {
    value = llvm::ConstantInt::get(codegen_type(node->type), node->value,
                                   /* IsSigned */ false);
}

void CodeGen_LLVM::visit(const VecImm *node) {
    ir::Expr build = ir::Build::make(node->type, node->values);
    build.accept(this);
}

void CodeGen_LLVM::visit(const Infinity *node) {
    llvm::Type *inf_type = codegen_type(node->type);

    if (inf_type->isFloatTy()) {
        value = llvm::ConstantFP::get(
            inf_type, llvm::APFloat::getInf(llvm::APFloat::IEEEsingle()));
    } else if (inf_type->isDoubleTy()) {
        value = llvm::ConstantFP::get(
            inf_type, llvm::APFloat::getInf(llvm::APFloat::IEEEdouble()));
    } else if (inf_type->isHalfTy()) {
        value = llvm::ConstantFP::get(
            inf_type, llvm::APFloat::getInf(llvm::APFloat::IEEEhalf()));
    } else if (inf_type->isIntegerTy()) {
        const uint32_t bits = inf_type->getIntegerBitWidth();
        bool is_signed = node->type.is_int();

        llvm::APInt max_val = is_signed ? llvm::APInt::getSignedMaxValue(bits)
                                        : llvm::APInt::getMaxValue(bits);

        value = llvm::ConstantInt::get(inf_type, max_val);
    } else {
        internal_error << "Infinity codegen not yet supported for type: "
                       << node->type;
    }
}

void CodeGen_LLVM::visit(const Var *node) {
    auto [_value, _mutable] = frames.from_frames(node->name);
    if (_mutable) {
        llvm::Type *var_type = codegen_type(node->type);
        value = builder->CreateLoad(var_type, _value, node->name);
    } else {
        value = _value; // immutable so not pointer.
    }
}

void CodeGen_LLVM::visit(const BinOp *node) {
    // TODO: upgrade type for arithmetic?
    llvm::Value *a = codegen_expr(node->a);
    llvm::Value *b = codegen_expr(node->b);

    // TODO: predications?
    if (node->a.type().is_float()) {
        switch (node->op) {
        case BinOp::Add: {
            value = builder->CreateFAdd(a, b);
            return;
        }
        case BinOp::Mul: {
            value = builder->CreateFMul(a, b);
            return;
        }
        case BinOp::Div: {
            value = builder->CreateFDiv(a, b);
            return;
        }
        case BinOp::Sub: {
            value = builder->CreateFSub(a, b);
            return;
        }
        case BinOp::Le: {
            value = builder->CreateFCmpOLE(a, b);
            return;
        }
        case BinOp::Lt: {
            value = builder->CreateFCmpOLT(a, b);
            return;
        }
        case BinOp::Eq: {
            value = builder->CreateFCmpOEQ(a, b);
            return;
        }
        default: {
            internal_error << "Unimplemented BinOp lowering for float: "
                           << Expr(node);
        }
        }
    } else if (node->a.type().is_int()) {
        // TODO: do we ever want NSW?
        switch (node->op) {
        case BinOp::Add: {
            value = builder->CreateAdd(a, b);
            return;
        }
        case BinOp::Mul: {
            value = builder->CreateMul(a, b);
            return;
        }
        case BinOp::Div: {
            // TODO: is this the correct behavior we want?
            value = builder->CreateSDiv(a, b);
            return;
        }
        case BinOp::Sub: {
            value = builder->CreateSub(a, b);
            return;
        }
        case BinOp::Mod: {
            // signed remainder
            value = builder->CreateSRem(a, b);
            return;
        }
        case BinOp::Le: {
            // unsigned comparison
            value = builder->CreateICmpSLE(a, b);
            return;
        }
        case BinOp::Lt: {
            // signed comparison
            value = builder->CreateICmpSLT(a, b);
            return;
        }
        case BinOp::Eq: {
            value = builder->CreateICmpEQ(a, b);
            return;
        }
        case BinOp::Neq: {
            value = builder->CreateICmpNE(a, b);
            return;
        }
        case BinOp::Xor: {
            value = builder->CreateXor(a, b);
            return;
        }
        case BinOp::BwAnd: {
            value = builder->CreateAnd(a, b);
            return;
        }
        case BinOp::BwOr: {
            value = builder->CreateOr(a, b);
            return;
        }
        case BinOp::Shl: {
            value = builder->CreateShl(a, b);
            return;
        }
        case BinOp::Shr: {
            value = builder->CreateAShr(a, b);
            return;
        }
        default: {
            internal_error
                << "Unimplemented BinOp lowering for signed integer: "
                << Expr(node);
        }
        }
    } else if (node->a.type().is_uint()) {
        switch (node->op) {
        case BinOp::Add: {
            value = builder->CreateAdd(a, b);
            return;
        }
        case BinOp::Mul: {
            value = builder->CreateMul(a, b);
            return;
        }
        case BinOp::Div: {
            // Use unsigned division for unsigned integers
            value = builder->CreateUDiv(a, b);
            return;
        }
        case BinOp::Sub: {
            value = builder->CreateSub(a, b);
            return;
        }
        case BinOp::Mod: {
            // unsigned remainder
            value = builder->CreateURem(a, b);
            return;
        }
        case BinOp::Le: {
            // Unsigned less-than-or-equal comparison
            value = builder->CreateICmpULE(a, b);
            return;
        }
        case BinOp::Lt: {
            // Unsigned less-than comparison
            value = builder->CreateICmpULT(a, b);
            return;
        }
        case BinOp::Eq: {
            value = builder->CreateICmpEQ(a, b);
            return;
        }
        case BinOp::Neq: {
            value = builder->CreateICmpNE(a, b);
            return;
        }
        case BinOp::Xor: {
            value = builder->CreateXor(a, b);
            return;
        }
        case BinOp::BwAnd: {
            value = builder->CreateAnd(a, b);
            return;
        }
        case BinOp::BwOr: {
            value = builder->CreateOr(a, b);
            return;
        }
        case BinOp::Shl: {
            value = builder->CreateShl(a, b);
            return;
        }
        case BinOp::Shr: {
            value = builder->CreateLShr(a, b);
            return;
        }
        default: {
            internal_error
                << "Unimplemented BinOp lowering for unsigned integer: "
                << Expr(node);
        }
        }
    } else if (node->a.type().is_bool()) {
        switch (node->op) {
        case BinOp::BwAnd: {
            value = builder->CreateAnd(a, b);
            return;
        }
        case BinOp::BwOr: {
            value = builder->CreateOr(a, b);
            return;
        }
        case BinOp::Xor: {
            value = builder->CreateXor(a, b);
            return;
        }
        case BinOp::LOr:
        case BinOp::LAnd:
        default: {
            internal_error << "Unimplemented BinOp lowering for boolean: "
                           << Expr(node);
        }
        }
    }

    internal_error << "Cannot codegen BinOp: " << Expr(node);
}

void CodeGen_LLVM::visit(const UnOp *node) {
    // TODO: upgrade type for arithmetic?
    llvm::Value *a = codegen_expr(node->a);

    switch (node->op) {
    case UnOp::Neg: {
        if (node->type.is_float()) {
            value = builder->CreateFNeg(a);
        } else {
            internal_assert(node->type.is_int_or_uint());
            llvm::Type *itype = a->getType();
            llvm::Constant *_0 = llvm::ConstantInt::get(itype, 0);
            value = builder->CreateSub(_0, a);
        }
        return;
    }
    case UnOp::Not: {
        internal_assert(node->type.is_bool());
        value = builder->CreateNot(a);
        return;
    }
    }

    internal_error << "Cannot codegen UnOp: " << Expr(node);
}

void CodeGen_LLVM::visit(const Select *node) {
    llvm::Value *cond = codegen_expr(node->cond);
    llvm::Value *tvalue = codegen_expr(node->tvalue);
    llvm::Value *fvalue = codegen_expr(node->fvalue);
    if (tvalue->getType()->isVectorTy()) {
        // TODO: handle broadcasting!
        internal_assert(cond->getType()->isVectorTy())
            << "Select lowering failure: " << ir::Expr(node);
        internal_assert(fvalue->getType()->isVectorTy())
            << "Select lowering failure: " << ir::Expr(node);
    }
    // TODO: try Vector Predication Intrinsics!
    // https://llvm.org/docs/LangRef.html#vector-predication-intrinsics
    // https://llvm.org/docs/LangRef.html#llvm-vp-select-intrinsics
    value = builder->CreateSelect(cond, tvalue, fvalue);
}

void CodeGen_LLVM::print_helper(const ir::Expr &node,
                                std::vector<llvm::Value *> &args,
                                std::string &to_print, uint32_t indent_level) {
    ir::Type t = node.type();
    // Returns a string with the given indentation level.
    auto indent = [&](uint32_t level) -> std::string {
        return std::string(level, ' ');
    };

    if (auto *vtype = t.as<ir::Vector_t>()) {
        to_print += "[";
        // Print each value in the vector.
        for (uint32_t i = 0, e = vtype->lanes; i < e; ++i) {
            static const ir::Type u32 = ir::UInt_t::make(32);
            ir::Expr extract = ir::Extract::make(node, make_const(u32, i));
            print_helper(extract, args, to_print, indent_level);
            if (i + 1 == e)
                continue;
            to_print += ", ";
        }
        to_print += "]";
        return;
    }

    if (auto *atype = t.as<ir::Array_t>()) {
        to_print += "{";
        // TODO(cgyurgyik): print non-constant sized arrays.
        std::optional<uint64_t> constant_size = get_constant_value(atype->size);
        internal_assert(constant_size.has_value()) << atype->size;
        for (uint64_t i = 0, e = *constant_size; i < e; ++i) {
            static const ir::Type u32 = ir::UInt_t::make(32);
            ir::Expr extract = ir::Extract::make(node, make_const(u32, i));
            print_helper(extract, args, to_print, indent_level);
            if (i + 1 == e)
                continue;
            to_print += ", ";
        }
        to_print += "}";
        return;
    }

    if (const auto *stype = t.as<ir::Struct_t>()) {
        to_print += stype->name;
        to_print += " {\n";
        bool first = true;
        for (const auto &[name, type] : stype->fields) {
            if (!first) {
                to_print += "\n";
            }
            first = false;

            // Print the member name.
            to_print += indent(indent_level + 2);
            to_print += name;
            to_print += ": ";
            // Print the member value.
            ir::Expr access = ir::Access::make(/*field=*/name, /*value=*/node);
            print_helper(access, args, to_print, indent_level + 2);
        }
        to_print += "\n";
        to_print += indent(indent_level);
        to_print += "}";
        return;
    }

    internal_assert((t.is<ir::Int_t, ir::UInt_t, ir::Float_t, ir::Bool_t>()))
        << "unimplemented `Print` support for type: " << t;
    to_print += get_specifier(t);
    llvm::Value *expr = codegen_expr(node);
    if (t.is_bool()) {
        // Convert boolean types to their human readable form.
        auto *type = cast<llvm::IntegerType>(expr->getType());
        const uint32_t width = type->getBitWidth();
        internal_assert(width == 1) << "expected i1, received: i" << width;
        llvm::Value *t = builder->CreateGlobalStringPtr("true");
        llvm::Value *f = builder->CreateGlobalStringPtr("false");
        expr = builder->CreateSelect(expr, t, f);
    }
    args.push_back(expr);
}

void CodeGen_LLVM::visit(const CallStmt *node) {
    Call::make(node->func, node->args).accept(this);
    value = nullptr;
}

void CodeGen_LLVM::visit(const Print *node) {
    // TODO(ajr): fix this to print like a vector.
    /*
    if (node->value.type().is<Array_t>()) {
        static int counter = 0;
        Expr size = node->value.type().as<Array_t>()->size;
        std::string index = "_print_iter" + std::to_string(counter++);
        std::string value = index + "_value";

        Expr var = Var::make(node->value.type().element_of(), value);
        Expr idx = Var::make(size.type(), index);

        ir::WriteLoc loc(value, var.type());
        Stmt header = LetStmt::make(std::move(loc),
                                    Extract::make(node->value, std::move(idx)));

        ForAll::Slice slice{make_zero(size.type()), size,
                            make_one(size.type())};

        Stmt body = Print::make(std::move(var));

        Stmt stmt = ForAll::make(index, header, slice, body);
        codegen_stmt(std::move(stmt));
        return;
    }
        */
    // The string to be printed in the call to `printf`...
    std::string to_print;
    // ...and the respective arguments for the format specifiers.
    std::vector<llvm::Value *> args;
    // Placeholder for the string - this is always the 1st argument.
    args.push_back(nullptr);

    print_helper(node->value, args, to_print);
    args.front() = builder->CreateGlobalStringPtr(to_print + "\n");

    value = builder->CreateCall(retrieve_printf(*module), args);
}

void CodeGen_LLVM::visit(const Cast *node) {
    const ir::Type &src = node->value.type();
    const ir::Type &dst = node->type;

    // TODO(ajr): we need a more general fix for these sorts of reinterprets.
    if (src.is<Vector_t>() && dst.is<Struct_t>() &&
        dst.as<Struct_t>()->fields.size() == 1) {
        ir::Expr repl =
            Cast::make(dst.as<Struct_t>()->fields[0].type, node->value);
        repl = Build::make(node->type, {std::move(repl)});
        repl.accept(this);
        return;
    }

    // TODO: upgrade_type_for_arithmetic?
    llvm::Value *inner = codegen_expr(node->value);

    llvm::Type *llvm_dst = codegen_type(dst);

    // Except the first branch, these just copy Halide's lowering (minus a few
    // pointer things).
    if ((src.is_vector() && !dst.is_vector()) ||
        (dst.is_vector() && !src.is_vector()) ||
        (src.is_vector() && dst.is_vector() && src.lanes() != dst.lanes())) {
        // Must be a reinterpret cast
        llvm::Type *llvm_src = codegen_type(src);

        // Reinterpret cast — bit widths must match
        if (module->getDataLayout().getTypeAllocSize(llvm_dst) !=
            module->getDataLayout().getTypeAllocSize(llvm_src)) {
            std::cerr << "Cannot cast between types of different sizes: "
                      << std::flush;
            llvm_dst->print(llvm::errs());
            llvm::errs() << " -> ";
            llvm_src->print(llvm::errs());
            llvm::errs().flush();

            internal_error << "Failed in Cast codegen (reinterpret)";
        }

        value = builder->CreateBitCast(inner, llvm_dst);
    } else if (src.is_int_or_uint() && dst.is_int_or_uint()) {
        value = builder->CreateIntCast(inner, llvm_dst,
                                       /* isSigned */ src.is_int());
    } else if (src.is_float() && dst.is_int()) {
        value = builder->CreateFPToSI(value, llvm_dst);
    } else if (src.is_float() && dst.is_uint()) {
        // TODO: Halide has a weird corner case for uint1 -> float, but we don't
        // use uint1 as bools. so I think we can ignore this, and handle it
        // explicitly in bool -> float casts. Note: this has undefined behavior
        // on overflow.
        value = builder->CreateFPToUI(inner, llvm_dst);
    } else if (src.is_int() && dst.is_float()) {
        value = builder->CreateSIToFP(inner, llvm_dst);
    } else if (src.is_uint() && dst.is_float()) {
        value = builder->CreateUIToFP(inner, llvm_dst);
    } else if (src.is_float() && dst.is_float()) {
        // Float widening or narrowing
        value = builder->CreateFPCast(inner, llvm_dst);
    } else {
        internal_error << "TODO: implement Cast codegen: " << Expr(node);
    }
}

void CodeGen_LLVM::visit(const Broadcast *node) {
    llvm::Value *v = codegen_expr(node->value);
    value = builder->CreateVectorSplat(node->lanes, v);
}

void CodeGen_LLVM::visit(const VectorReduce *node) {
    internal_assert(node->type.is_scalar())
        << "Cannot codegen 2+ dimensional VectorReduce: " << Expr(node);
    // TODO: upgrade type for arithmetic?

    llvm::Value *v = codegen_expr(node->value);

    llvm::VectorType *vecType = llvm::cast<llvm::VectorType>(v->getType());
    llvm::Type *elementType = vecType->getElementType();

    // TODO: better instruction selection.

    // TODO: try fold vector reduce
    llvm::Value *init = nullptr;

    llvm::Intrinsic::IndependentIntrinsics intrin;

    switch (node->op) {
    case VectorReduce::Add:
        if (node->type.is_float()) {
            intrin = llvm::Intrinsic::vector_reduce_fadd;
            // TODO: is this right? why do we have to do this.
            init = llvm::ConstantFP::get(elementType, 0.0f);
        } else {
            intrin = llvm::Intrinsic::vector_reduce_add;
        }
        break;
    case VectorReduce::Mul:
        if (node->type.is_float()) {
            intrin = llvm::Intrinsic::vector_reduce_fmul;
            // TODO: is this right? why do we have to do this.
            init = llvm::ConstantFP::get(elementType, 1.0f);
        } else {
            intrin = llvm::Intrinsic::vector_reduce_mul;
        }
        break;
    case VectorReduce::Min:
        // TODO: handle unsigned eventually!
        // TODO: what is the difference between fmin and fminimum?
        intrin = node->type.is_float() ? llvm::Intrinsic::vector_reduce_fmin
                                       : llvm::Intrinsic::vector_reduce_smin;
        break;
    case VectorReduce::Max:
        // TODO: handle unsigned eventually!
        // TODO: what is the difference between fmax and fmaximum?
        intrin = node->type.is_float() ? llvm::Intrinsic::vector_reduce_fmax
                                       : llvm::Intrinsic::vector_reduce_smax;
        break;
    case VectorReduce::Idxmax:
        // TODO: on x86 lower to phminposuw
        value = codegen_expr(lower::argmax(node->value));
        return;
    case VectorReduce::Or:
        intrin = llvm::Intrinsic::vector_reduce_or;
        break;
    case VectorReduce::And:
        intrin = llvm::Intrinsic::vector_reduce_and;
        break;
    default: {
        internal_error << "Unsupported VectorReduce operation" << Expr(node);
    }
    }

    // TODO: perform splitting? investigate LLVM's splitting.

    if (init) {
        value = builder->CreateIntrinsic(elementType, intrin, {init, v});
    } else {
        value = builder->CreateIntrinsic(elementType, intrin, {v});
    }

    internal_assert(value) << "VectorReduce intrin failure: " << Expr(node);
}

void CodeGen_LLVM::visit(const VectorShuffle *node) {
    llvm::Value *_value = codegen_expr(node->value);
    llvm::Type *out_type = codegen_type(node->type);
    // const uint32_t inputSize = node->value.type().lanes();

    // TODO: optimize the case for a constant shuffle!

    llvm::Value *result = llvm::UndefValue::get(out_type);

    // Generate an extract and insert per index.
    for (size_t i = 0; i < node->idxs.size(); i++) {
        const Expr &idx = node->idxs[i];
        // We need 32 bit indices.
        internal_assert(idx.type().is_int_or_uint());
        llvm::Value *load_index = codegen_expr(idx);

        // TODO: we should maybe clamp to [0, inputSize) to avoid UB...

        // TODO: truncs aren't really safe...
        if (idx.type().is_int()) {
            load_index = builder->CreateSExtOrTrunc(load_index, i32_t);
        } else {
            load_index = builder->CreateZExtOrTrunc(load_index, i32_t);
        }

        // llvm::errs() << *_value << " and " << *load_index << "\n";
        llvm::Value *element =
            builder->CreateExtractElement(_value, load_index);

        llvm::Constant *store_idx = llvm::ConstantInt::get(i32_t, i);
        result = builder->CreateInsertElement(result, element, store_idx);
    }

    value = result;
}

void CodeGen_LLVM::visit(const Ramp *node) {
    internal_error << "TODO: implement Ramp code generation: " << Expr(node);
}

void CodeGen_LLVM::visit(const Extract *node) {
    llvm::Value *vec = codegen_expr(node->vec);
    llvm::Value *idx = codegen_expr(node->idx);
    if (node->vec.type().is<Vector_t>()) {
        value = builder->CreateExtractElement(vec, idx);
    } else if (node->vec.type().is<Array_t>()) {
        llvm::Type *etype = codegen_type(node->vec.type().element_of());
        llvm::Value *ptr = builder->CreateGEP(etype, vec, idx, "extract_ptr");
        value = builder->CreateLoad(etype, ptr, "extract");
    } else {
        internal_error << "[unimplemented] codegen of Extract on type: "
                       << node->vec.type();
    }
}

void CodeGen_LLVM::visit(const Intrinsic *node) {
    llvm::Intrinsic::IndependentIntrinsics intrin;
    // llvm.abs for integers requires passing a constant `false` to it.
    bool add_false_arg = false;
    switch (node->op) {
    case Intrinsic::abs: {
        intrin = node->args[0].type().is_float() ? llvm::Intrinsic::fabs
                                                 : llvm::Intrinsic::abs;
        add_false_arg = node->args[0].type().is_int();
        break;
    }
    case Intrinsic::cos: {
        intrin = llvm::Intrinsic::cos;
        break;
    }
    case Intrinsic::cross: {
        Expr expr = lower::cross_product(node->args[0], node->args[1]);
        value = codegen_expr(expr);
        return;
    }
    case Intrinsic::dot: {
        Expr expr = VectorReduce::make(VectorReduce::Add,
                                       node->args[0] * node->args[1]);
        value = codegen_expr(expr);
        return;
    }
    case Intrinsic::fma: {
        intrin = llvm::Intrinsic::fma;
        break;
    }
    case Intrinsic::max: {
        if (node->args[0].type().is_int()) {
            intrin = llvm::Intrinsic::smax;
        } else if (node->args[0].type().is_uint()) {
            intrin = llvm::Intrinsic::umax;
        } else {
            internal_assert(node->args[0].type().is_float())
                << "Cannot lower max of type: " << node->args[0].type();
            // Follows the IEEE-754 semantics for maxNum except for the handling
            // of signaling NaNs. This matches the behavior of libm’s fmax.
            // https://llvm.org/docs/LangRef.html#llvm-maxnum-intrinsic
            intrin = llvm::Intrinsic::maxnum;
            // internal_error << "TODO: figure out fmax codegen: " <<
            // Expr(node);
        }
        break;
    }
    case Intrinsic::min: {
        if (node->args[0].type().is_int()) {
            intrin = llvm::Intrinsic::smin;
        } else if (node->args[0].type().is_uint()) {
            intrin = llvm::Intrinsic::umin;
        } else {
            internal_assert(node->args[0].type().is_float())
                << "Cannot lower min of type: " << node->args[0].type();
            // Follows the IEEE-754 semantics for minNum, except for handling of
            // signaling NaNs. This match’s the behavior of libm’s fmin.
            // https://llvm.org/docs/LangRef.html#llvm-minnum-intrinsic
            intrin = llvm::Intrinsic::minnum;
            // internal_error << "TODO: figure out fmin codegen: " <<
            // Expr(node);
        }
        break;
    }
    case Intrinsic::norm: {
        Expr expr = sqrt(dot(node->args[0], node->args[0]));
        value = codegen_expr(expr);
        return;
    }
    case Intrinsic::sin: {
        intrin = llvm::Intrinsic::sin;
        break;
    }
    case Intrinsic::sqrt: {
        intrin = llvm::Intrinsic::sqrt;
        break;
    }
    }
    std::vector<llvm::Value *> args(node->args.size());
    for (size_t i = 0; i < args.size(); i++) {
        args[i] = codegen_expr(node->args[i]);
    }

    if (add_false_arg) {
        // Necessary for integer abs(), this is <is_int_min_poison>
        args.push_back(llvm::ConstantInt::get(i1_t, 0));
    }

    llvm::Type *ret_type = codegen_type(node->type);

    // std::cout << "calling intrinsic for: " << ir::Expr(node) << std::endl;
    value = builder->CreateIntrinsic(ret_type, intrin, args);

    internal_assert(value) << "Intrinsic codegen failure: " << Expr(node);
}

void CodeGen_LLVM::visit(const Lambda *node) {
    internal_error
        << "Lambda expression should have been canonicalized and eliminated: "
        << Expr(node);
}

void CodeGen_LLVM::visit(const GeomOp *node) {
    internal_error << "TODO: implement GeomOp code generation: " << Expr(node);
}

void CodeGen_LLVM::visit(const SetOp *node) {
    internal_error << "TODO: implement SetOp code generation: " << Expr(node);
}

void CodeGen_LLVM::visit(const Call *node) {
    llvm::Function *func = codegen_func_ptr(node->func);
    internal_assert(func) << "Failed to codegen function pointer to: "
                          << Expr(node);

    // TODO: figure out how to make sure we have the right
    // number of arguments here for better error handling.

    const size_t n_args = node->args.size();
    std::vector<llvm::Value *> args(n_args);

    for (size_t i = 0; i < n_args; i++) {
        llvm::Value *argument = codegen_expr(node->args[i]);

        if (auto *load = dyn_cast<llvm::LoadInst>(argument)) {
            args[i] = load->getPointerOperand();
        } else if (node->args[i].type().is<ir::Struct_t>() &&
                   !isa<llvm::AllocaInst>(argument)) {
            // We assume structs will always be passed by pointer.
            auto *alloca = builder->CreateAlloca(argument->getType());
            builder->CreateStore(argument, alloca);
            args[i] = alloca;
        } else {
            args[i] = argument;
        }
    }
    value = builder->CreateCall(func, args);
}

void CodeGen_LLVM::visit(const Instantiate *node) {
    internal_error << "Instantiate node not lowered prior to codegen: "
                   << Expr(node);
}

void CodeGen_LLVM::visit(const Build *node) {
    // This will be a StructType or a VectorType
    llvm::Type *build_type = codegen_type(node->type);

    std::vector<llvm::Value *> values = codegen_exprs(node->values);

    if (build_type->isVectorTy()) {
        if (values.empty()) {
            value = llvm::Constant::getNullValue(build_type);
            return;
        }
        // Fill with designated values.
        value = llvm::UndefValue::get(build_type);
        for (size_t i = 0; i < values.size(); i++) {
            value = builder->CreateInsertElement(value, values[i], i);
        }
        return;
    } else if (build_type->isStructTy()) {
        internal_assert(node->type.is<Struct_t>());
        internal_assert(
            values.empty() ||
            (values.size() == node->type.as<Struct_t>()->fields.size()))
            << "TODO: implement partial build codegen for: " << Expr(node);
        const auto &defaults = node->type.as<Struct_t>()->defaults;
        const auto &fields = node->type.as<Struct_t>()->fields;
        if (defaults.empty() && values.empty()) {
            value = llvm::Constant::getNullValue(build_type);
            return;
        } else if (values.empty()) {
            // Is order of codegen important? Default values must be constants
            // (w/o side effects), so I think no?
            std::vector<std::pair<size_t, llvm::Value *>> inserts;
            for (const auto &[field, _value] : defaults) {
                size_t idx = find_struct_index(field, fields);
                llvm::Value *llvm_value = codegen_expr(_value);
                internal_assert(llvm_value);
                inserts.emplace_back(idx, llvm_value);
            }

            // Fill the default values at least, and make the rest
            value = llvm::Constant::getNullValue(build_type);

            // Sort on insertion order.
            std::sort(
                inserts.begin(), inserts.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
            for (const auto &[idx, _value] : inserts) {
                internal_assert(_value);
                value = builder->CreateInsertValue(value, _value, idx);
            }
            return;
        } else {
            internal_assert(defaults.empty() ||
                            (values.size() == fields.size()));
            value = llvm::UndefValue::get(build_type);
            for (size_t i = 0; i < values.size(); i++) {
                value = builder->CreateInsertValue(value, values[i], i);
            }
            return;
        }
    } else if (build_type->isPointerTy()) {
        // TODO(ajr): builds of Array_t should probably be turned into Allocates
        // and constant-sized insertion loops.
        internal_assert(node->type.is<Array_t>());
        internal_assert(!values.empty());
        const Array_t *array_t = node->type.as<Array_t>();

        // Do allocation.
        // TODO(ajr): constant sized should be on stack, right?
        llvm::Type *etype = codegen_type(array_t->etype);
        llvm::Value *size = codegen_expr(array_t->size);
        // TODO(ajr): zero_initialize is broken, it should be set true here.
        llvm::Value *alloc =
            create_malloc(etype, size, /*zero_initialize=*/false, "");

        for (size_t i = 0; i < values.size(); i++) {
            llvm::Value *index = llvm::ConstantInt::get(size->getType(), i);
            llvm::Value *ptr =
                builder->CreateGEP(etype, alloc, index, "build_ptr");
            builder->CreateStore(values[i], ptr);
        }
        value = alloc;
        return;
    } else {
        internal_error << "Unexpected llvm Type in Build lowering: "
                       << Expr(node);
    }
}

void CodeGen_LLVM::visit(const Access *node) {
    llvm::Value *inner = codegen_expr(node->value);
    if (inner->getType()->isStructTy()) {
        const size_t idx = find_struct_index(
            node->field, node->value.type().as<Struct_t>()->fields);
        value = builder->CreateExtractValue(inner, idx);
        return;
    }
    internal_error
        << "Lowering of an Access's value did not result in a struct type: "
        << Expr(node);
}

void CodeGen_LLVM::visit(const Unwrap *node) {
    internal_error << "Unwrap should have been lowered before CodeGen_LLVM "
                   << Expr(node);
}

void CodeGen_LLVM::visit(const Return *node) {
    ir::Expr value = node->value;
    if (!value.defined()) {
        builder->CreateRetVoid();
        return;
    }
    builder->CreateRet(codegen_expr(value));
}

void CodeGen_LLVM::visit(const Store *node) {
    // TODO: eventually handle atomics...
    // TODO: eventually handle predication...
    llvm::Value *expr = codegen_expr(node->value);
    Type value_type = node->value.type();

    if (value_type.is_scalar()) {
        // Scalar
        llvm::Value *ptr =
            codegen_buffer_pointer(node->name, value_type, node->index);
        // TODO: handle booleans better.
        llvm::StoreInst *store = builder->CreateAlignedStore(
            expr, ptr, llvm::Align(value_type.bytes()));
        // TODO: fix this, add type-based metadata to all stores to allow
        // reordering! annotate_store(store, op->index);
        add_tbaa_metadata(store, node->name, node->index);
    } else {
        // TODO: handle alignment info better.
        int alignment = value_type.element_of().bytes();
        const Ramp *ramp = node->index.as<Ramp>();
        bool is_dense = ramp && is_const_one(ramp->stride);

        if (is_dense) {
            // Generate native vector store(s).
            // TODO: do manual splitting into native vector stores.
            llvm::Value *base_ptr = codegen_buffer_pointer(
                node->name, value_type.element_of(), ramp->base);
            llvm::Value *vec_ptr = builder->CreateBitCast(
                base_ptr, llvm::PointerType::getUnqual(expr->getType()));
            // TODO: is this always aligned? figure out alignment stuff.
            llvm::StoreInst *store = builder->CreateAlignedStore(
                expr, vec_ptr, llvm::Align(alignment));
            add_tbaa_metadata(store, node->name, node->index);
        } else if (ramp) {
            // Generate strided stores.
            Type ptr_type = value_type.element_of();
            llvm::Value *ptr =
                codegen_buffer_pointer(node->name, ptr_type, ramp->base);
            const IntImm *const_stride = ramp->stride.as<IntImm>();
            llvm::Value *stride = codegen_expr(ramp->stride);
            llvm::Type *load_type = codegen_type(ptr_type);
            // Scatter without generating the indices as a vector
            for (int i = 0; i < ramp->lanes; i++) {
                llvm::Constant *lane = llvm::ConstantInt::get(i32_t, i);
                llvm::Value *v = builder->CreateExtractElement(expr, lane);
                if (const_stride) {
                    // Use a constant offset from the base pointer
                    llvm::Value *p = builder->CreateConstInBoundsGEP1_32(
                        load_type, ptr, const_stride->value * i);
                    llvm::StoreInst *store = builder->CreateStore(v, p);
                    // annotate_store(store, op->index);
                    add_tbaa_metadata(store, node->name, node->index);
                } else {
                    // Increment the pointer by the stride for each element
                    llvm::StoreInst *store = builder->CreateStore(v, ptr);
                    // annotate_store(store, op->index);
                    add_tbaa_metadata(store, node->name, node->index);
                    // ptr = CreateInBoundsGEP(builder.get(), load_type, ptr,
                    // stride);
                    ptr = builder->CreateInBoundsGEP(load_type, ptr, {stride});
                }
            }
        } else {
            // Generate scatter.
            // TODO: do better on some archs?
            llvm::Value *index = codegen_expr(node->index);
            for (int i = 0; i < value_type.lanes(); i++) {
                llvm::Value *lane = llvm::ConstantInt::get(i32_t, i);
                llvm::Value *idx = builder->CreateExtractElement(index, lane);
                llvm::Value *v = builder->CreateExtractElement(expr, lane);
                llvm::Value *ptr = codegen_buffer_pointer(
                    node->name, value_type.element_of(), idx);
                llvm::StoreInst *store = builder->CreateStore(v, ptr);
                // annotate_store(store, node->index);
                add_tbaa_metadata(store, node->name, node->index);
            }
        }
    }
}

void CodeGen_LLVM::visit(const LetStmt *node) {
    llvm::Value *_value = codegen_expr(node->value);
    frames.add_to_frame(node->loc.base, {_value, /*mutable=*/false});
}

void CodeGen_LLVM::visit(const IfElse *node) {
    // Gather the conditions and values in an if-else chain
    struct Block {
        Expr expr;
        Stmt stmt;
        bool returns;
        Block(Expr _expr, Stmt _stmt, bool _returns)
            : expr(std::move(_expr)), stmt(std::move(_stmt)),
              returns(_returns) {}
    };
    std::vector<Block> blocks;
    Stmt final_else;
    const IfElse *next_if = node;
    bool needs_after_bb = false;
    do {
        bool returns = always_returns(next_if->then_body);
        blocks.emplace_back(next_if->cond, next_if->then_body, returns);
        needs_after_bb = needs_after_bb || !returns;
        final_else = next_if->else_body;
        next_if = final_else.defined() ? final_else.as<IfElse>() : nullptr;
    } while (next_if);

    needs_after_bb = needs_after_bb &&
                     (!final_else.defined() || !always_returns(final_else));

    // TODO: we will support a switch statement, make sure to use Halide's
    // codegen for it!

    internal_assert(current_function);
    llvm::BasicBlock *after_bb =
        needs_after_bb
            ? llvm::BasicBlock::Create(*context, "after_bb", current_function)
            : nullptr;

    for (const auto &p : blocks) {
        llvm::BasicBlock *then_bb =
            llvm::BasicBlock::Create(*context, "then_bb", current_function);
        llvm::BasicBlock *next_bb =
            llvm::BasicBlock::Create(*context, "next_bb", current_function);
        codegen_short_circuit(p.expr, then_bb, next_bb);
        builder->SetInsertPoint(then_bb);
        codegen_stmt(p.stmt);
        if (!p.returns) {
            codegen_branch(after_bb);
        }
        builder->SetInsertPoint(next_bb);
    }

    if (final_else.defined()) {
        codegen_stmt(final_else);
    }

    if (needs_after_bb) {
        codegen_branch(after_bb);
        builder->SetInsertPoint(after_bb);
    }
}

void CodeGen_LLVM::codegen_short_circuit(Expr cond, llvm::BasicBlock *true_bb,
                                         llvm::BasicBlock *false_bb) {
    if (const BinOp *op = cond.as<BinOp>()) {
        if (op->op == BinOp::LAnd) {
            llvm::BasicBlock *rhs_bb =
                llvm::BasicBlock::Create(*context, "and_rhs", current_function);
            // if a then check b else goto false
            codegen_short_circuit(op->a, rhs_bb, false_bb);
            builder->SetInsertPoint(rhs_bb);
            // if also b then goto true else goto false
            codegen_short_circuit(op->b, true_bb, false_bb);
            return;
        } else if (op->op == BinOp::LOr) {
            llvm::BasicBlock *rhs_bb =
                llvm::BasicBlock::Create(*context, "or_rhs", current_function);
            // if a then goto true else check b
            codegen_short_circuit(op->a, true_bb, rhs_bb);
            builder->SetInsertPoint(rhs_bb);
            // if b then goto true else goto false
            codegen_short_circuit(op->b, true_bb, false_bb);
            return;
        }
    } else if (const UnOp *op = cond.as<UnOp>()) {
        if (op->op == UnOp::Not) {
            codegen_short_circuit(op->a, false_bb, true_bb);
            return;
        }
    }
    // Base case: not a short-circuiting expression, emit a regular branch
    builder->CreateCondBr(codegen_expr(std::move(cond)), true_bb, false_bb);
}

void CodeGen_LLVM::codegen_branch(llvm::BasicBlock *bb) {
    if (!builder->GetInsertBlock()->getTerminator()) {
        builder->CreateBr(bb);
    }
}

void CodeGen_LLVM::visit(const DoWhile *node) {
    // Body of the loop
    llvm::BasicBlock *loop_bb =
        llvm::BasicBlock::Create(*context, "dowhile.body", current_function);
    // Block after the loop.
    llvm::BasicBlock *end_bb =
        llvm::BasicBlock::Create(*context, "dowhile.end", current_function);
    // Block that checks whether to jump back to the body.
    llvm::BasicBlock *cond_bb =
        llvm::BasicBlock::Create(*context, "dowhile.cond", current_function);

    // Jump unconditionally to loop body (required for do-while)
    codegen_branch(loop_bb);

    builder->SetInsertPoint(loop_bb);

    // TODO: are there phi nodes?
    // For now, assume LLVM optimizes loads/stores into phi nodes.

    // Establish new frame
    frames.new_frame();
    latch_blocks.push_back(cond_bb);
    // TODO(ajr): will need this for `break` statements.
    // escape_blocks.push_back(end_bb);

    // Emit loop body
    codegen_stmt(node->body);

    latch_blocks.pop_back();
    // escape_blocks.pop_back();

    codegen_branch(cond_bb);

    builder->SetInsertPoint(cond_bb);

    // Maybe exit the loop
    // TODO(ajr): use very_likely_branch?
    codegen_short_circuit(node->cond, loop_bb, end_bb);

    // Following statements should write to end_bb
    builder->SetInsertPoint(end_bb);

    // Pop for-loop local scope names.
    frames.pop_frame();
}

void CodeGen_LLVM::visit(const Assign *node) {
    llvm::Value *loc = codegen_write_loc(node->loc);
    internal_assert(loc) << "Failed to codegen LLVM ptr for: " << node->loc
                         << " in assignment: " << ir::Stmt(node);

    llvm::Value *rhs = codegen_expr(node->value);
    if (auto *load = dyn_cast<llvm::LoadInst>(loc)) {
        loc = load->getPointerOperand();
    }
    // TODO: when is isVolatile true?
    builder->CreateStore(rhs, loc, /*isVolatile=*/false);
}

void CodeGen_LLVM::visit(const Accumulate *node) {
    llvm::Value *loc = codegen_write_loc(node->loc);

    llvm::Value *_value = codegen_expr(node->value);

    llvm::Value *current = builder->CreateLoad(_value->getType(), loc);
    llvm::Value *acc = nullptr;

    switch (node->op) {
    case Accumulate::Add: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFAdd(current, _value);
        } else {
            acc = builder->CreateAdd(current, _value);
        }
        break;
    }
    case Accumulate::Sub: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFSub(current, _value);
        } else {
            acc = builder->CreateSub(current, _value);
        }
        break;
    }
    case Accumulate::Mul: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFMul(current, _value);
        } else {
            acc = builder->CreateMul(current, _value);
        }
        break;
    }
    default: {
        internal_error << "TODO: implement codegen for accumulate: "
                       << Stmt(node);
    }
    }

    builder->CreateStore(acc, loc);
}

/*
llvm::Value *CodeGen_LLVM::create_alloca_at_entry(llvm::Type *etype, llvm::Value
*size, bool zero_initialize, const std::string &name) {
    // create alloca at basic block entry.
    // TODO(ajr): why does Halide do this at BB entry?

    auto here = builder->saveIP();
    llvm::BasicBlock *entry =
&builder->GetInsertBlock()->getParent()->getEntryBlock(); if (entry->empty()) {
        builder->SetInsertPoint(entry);
    } else {
        builder->SetInsertPoint(entry, entry->getFirstInsertionPt());
    }
    llvm::AllocaInst *ptr = builder->CreateAlloca(etype, size, name);
    int align = native_vector_bits() / 8;
    // const llvm::DataLayout &d = module->getDataLayout();
    if (etype->isVectorTy() || !is_llvm_const_one(size)) {
        ptr->setAlignment(llvm::Align(align));
    }

    if (zero_initialize) {
        if (is_llvm_const_one(size)) {
            builder->CreateStore(llvm::Constant::getNullValue(etype), ptr);
        } else {
            internal_error << "[unimplemented] zero initialize array";
            ptr->getType()->dump();
            llvm::Constant::getNullValue(etype)->getType()->dump();
            size->getType()->dump();
            llvm::Type *i8_ptr_ty = i8_t->getPointerTo();
            llvm::Value *ptr_i8 = builder->CreateBitCast(ptr, i8_ptr_ty); //
alloc is %0 llvm::Value *val = builder->getInt8(0); // fill with 0 llvm::Value
*len = builder->CreateZExt(size, i64_t);                // size is i32 8 -> i64


            // builder->CreateMemSet(ptr, llvm::Constant::getNullValue(etype),
size, llvm::Align(align)); builder->CreateMemSet(ptr_i8, val, len,
llvm::Align(align));
        }
    }
    builder->restoreIP(here);
    return ptr;
}
*/

llvm::Value *CodeGen_LLVM::create_malloc(llvm::Type *etype, llvm::Value *size,
                                         bool zero_initialize,
                                         const std::string &name) {

    int align = native_vector_bits() / 8;

    // Total allocation size = elemSize * count
    // Size of the element in bytes
    llvm::DataLayout dataLayout(module.get());
    uint64_t typeSize = dataLayout.getTypeAllocSize(etype);
    llvm::Value *elemSize = llvm::ConstantInt::get(i64_t, typeSize);

    if (size->getType() != i64_t) {
        size =
            builder->CreateIntCast(size, i64_t, /*isSigned=*/false, "size64");
    }
    llvm::Value *allocSize = builder->CreateMul(elemSize, size);

    // This returns a pointer of type i32*
    // TODO: figure out alignment?
    llvm::Value *untyped_ptr = builder->CreateMalloc(
        i64_t, etype, allocSize, size, nullptr, name + "_untyped");

    // if (etype->isVectorTy() || !is_llvm_const_one(size)) {
    //     untyped_ptr->setAlignment(llvm::Align(align));
    // }

    llvm::Value *ptr = builder->CreateBitCast(
        untyped_ptr, etype->getPointerTo(), name + "_typed");

    if (zero_initialize) {
        if (is_llvm_const_one(size)) {
            builder->CreateStore(llvm::Constant::getNullValue(etype), ptr);
        } else {
            internal_error << "[unimplemented] zero initialize array";
            ptr->getType()->dump();
            llvm::Constant::getNullValue(etype)->getType()->dump();
            size->getType()->dump();
            llvm::Type *i8_ptr_ty = i8_t->getPointerTo();
            llvm::Value *ptr_i8 =
                builder->CreateBitCast(ptr, i8_ptr_ty); // alloc is %0
            llvm::Value *val = builder->getInt8(0);     // fill with 0
            llvm::Value *len =
                builder->CreateZExt(size, i64_t); // size is i32 8 -> i64

            // builder->CreateMemSet(ptr, llvm::Constant::getNullValue(etype),
            // size, llvm::Align(align));
            builder->CreateMemSet(ptr_i8, val, len, llvm::Align(align));
        }
    }
    return ptr;
}

void CodeGen_LLVM::visit(const Allocate *node) {
    // TODO(ajr): like Halide, we should put "small" allocations on the stack.
    // TODO(ajr): We may want to pull some of this logic into a helper function.

    llvm::Type *node_type = nullptr;
    llvm::Value *node_size = nullptr;

    // Handle arrays as a special case.
    if (node->type.is<Array_t>()) {
        const Array_t *array_t = node->type.as<Array_t>();
        node_type = codegen_type(array_t->etype);
        node_size = codegen_expr(array_t->size);
    } else {
        // Scalar allocation.
        node_type = codegen_type(node->type);
        node_size = llvm::ConstantInt::get(i32_t, 1);
    }

    llvm::Value *alloc = create_malloc(node_type, node_size,
                                       /*zero_initialize=*/false, node->name);

    // We set mutable to false because Var codegen would perform a load from
    // this pointer if it was mutable.
    frames.add_to_frame(node->name, {alloc, /* mutable=*/false});
}

void CodeGen_LLVM::visit(const Label *node) {
    internal_assert(node->body.defined())
        << "Label with undefined body made it to codegen: " << node->name;
    // TODO: add label as a comment to body here?
    codegen_stmt(node->body);
}

void CodeGen_LLVM::visit(const ForAll *node) {
    llvm::Value *begin = codegen_expr(node->slice.begin);

    // TODO(ajr): handle parallelism, needs to be field of ForAll
    // For now, generate sequential.
    llvm::BasicBlock *preheader_bb = builder->GetInsertBlock();

    std::string loop_id =
        node->index + std::to_string(forall_loop_id++) + std::string("_for");

    llvm::BasicBlock *inc_bb =
        llvm::BasicBlock::Create(*context, loop_id + "_inc", current_function);
    // Body of the loop
    llvm::BasicBlock *loop_bb =
        llvm::BasicBlock::Create(*context, loop_id, current_function);
    // Block after the loop.
    llvm::BasicBlock *end_bb =
        llvm::BasicBlock::Create(*context, loop_id + "_end", current_function);

    // Unlike Halide, can have loops over non-int32 types, so let codegen figure
    // out cmp type.
    llvm::Value *enter_condition =
        codegen_expr(node->slice.begin < node->slice.end);
    builder->CreateCondBr(enter_condition, loop_bb, end_bb, very_likely_branch);
    builder->SetInsertPoint(loop_bb);

    // Make our phi node.
    llvm::Type *iterator_t = codegen_type(node->slice.begin.type());
    llvm::PHINode *phi = builder->CreatePHI(iterator_t, 2);
    phi->addIncoming(begin, preheader_bb);

    // Add index to new frame.
    frames.new_frame();
    frames.add_to_frame(node->index, {phi, /*mutable=*/false});

    latch_blocks.push_back(inc_bb);
    // TODO(ajr): will need this for `break` statements.
    // escape_blocks.push_back(end_bb);

    // Emit loop body
    codegen_stmt(node->header);
    codegen_stmt(node->body);

    latch_blocks.pop_back();
    // escape_blocks.pop_back();

    codegen_branch(inc_bb);
    builder->SetInsertPoint(inc_bb);

    // Update the counter
    Expr var = Var::make(node->slice.begin.type(), node->index);
    llvm::Value *next_var = codegen_expr(var + node->slice.stride);
    // Add the back-edge to the phi node
    phi->addIncoming(next_var, builder->GetInsertBlock());

    // Maybe exit the loop
    // TODO(ajr): can this overflow?
    llvm::Value *end_condition = codegen_expr(var + 1 >= node->slice.end);
    // TODO(ajr): use very_likely_branch?
    builder->CreateCondBr(end_condition, end_bb, loop_bb);

    // Following statements should write to end_bb
    builder->SetInsertPoint(end_bb);

    // Pop for-loop local scope names.
    frames.pop_frame();
}

void CodeGen_LLVM::visit(const Continue *node) {
    internal_assert(!latch_blocks.empty())
        << "CodeGen of Continue outside of loop.";
    internal_assert(!builder->GetInsertBlock()->getTerminator())
        << "CodeGen of Continue in already-terminating block";
    builder->CreateBr(latch_blocks.back());
}

void CodeGen_LLVM::add_tbaa_metadata(llvm::Instruction *inst,
                                     const std::string &buffer,
                                     const Expr &index) {

    // Get the unique name for the block of memory this allocate node
    // is using.
    const std::string alloc_name = get_allocation_name(buffer);

    // If the index is constant, we generate some TBAA info that helps
    // LLVM understand our loads/stores aren't aliased.
    // bool constant_index = false;
    int64_t base = 0;
    int64_t width = 1;

    if (index.defined()) {
        if (const Ramp *ramp = index.as<Ramp>()) {
            const int64_t *pstride = as_const_int(ramp->stride);
            const int64_t *pbase = as_const_int(ramp->base);
            if (pstride && pbase) {
                // We want to find the smallest aligned width and offset
                // that contains this ramp.
                int64_t stride = *pstride;
                base = *pbase;
                // base = 0
                internal_assert(base >= 0) << "base of ramp is negative";
                width = next_power_of_two(ramp->lanes * stride);

                while (base % width) {
                    base -= base % width;
                    width *= 2;
                }
                // constant_index = true;
            }
        } else {
            const int64_t *pbase = as_const_int(index);
            if (pbase) {
                base = *pbase;
                // constant_index = true;
            }
        }
    } else {
        // Index is implied 0
        // constant_index = true;
        base = 0;
    }

    llvm::MDBuilder builder(*context);

    // Add type-based-alias-analysis metadata to the pointer, so that
    // loads and stores to different buffers can get reordered.
    llvm::MDNode *tbaa = builder.createTBAARoot("Bonsai buffer");

    tbaa = builder.createTBAAScalarTypeNode(alloc_name, tbaa);

    // We also add metadata for constant indices to allow loads and
    // stores to the same buffer to get reordered.
    // if (constant_index) {
    // TODO: is this necessary if scalar
    //     for (int w = 1024; w >= width; w /= 2) {
    //         int64_t b = (base / w) * w;

    //         std::stringstream level;
    //         level << buffer << ".width" << w << ".base" << b;
    //         tbaa = builder.createTBAAScalarTypeNode(level.str(), tbaa);
    //     }
    // }

    tbaa = builder.createTBAAStructTagNode(tbaa, tbaa, 0);

    inst->setMetadata("tbaa", tbaa);
}

void CodeGen_LLVM::declare_struct_types(
    const std::vector<const Struct_t *> structs) {
    internal_assert(struct_types.empty())
        << "declare_struct_types called with non-empty struct_types!";

    // TODO: does this handle recursive types properly?
    // First insert empty StructTypes into struct_types, to handle
    // weird ordering on types.
    // TODO: maybe make sure there's never an infinitely-recursive type?
    for (const auto &_struct : structs) {
        struct_types[_struct->name] =
            llvm::StructType::create(*context, "struct." + _struct->name);
        // llvm::errs() << "created: " << *struct_types[_struct->name] << "\n";
    }
    // Now build bodies, possibly referencing other struct types.
    for (const auto &_struct : structs) {
        std::vector<llvm::Type *> types(_struct->fields.size());
        size_t i = 0;
        // TODO(ajr): this is a hacky fix...
        bool skip = false;
        for (const auto &[key, value] : _struct->fields) {
            if (!value.is<Ref_t>()) {
                types[i++] = codegen_type(value);
            } else {
                skip = true;
            }
        }
        if (!skip) {
            struct_types[_struct->name]->setBody(types);
        }
        // llvm::errs() << "built: " << *struct_types[_struct->name] << "\n";
    }
}

llvm::Value *CodeGen_LLVM::codegen_buffer_pointer(const std::string &buffer,
                                                  const Type &type,
                                                  llvm::Value *idx) {
    llvm::DataLayout d(module.get());
    auto [base_addr, _] = frames.from_frames(buffer);
    // llvm::Value *base_addr = frames.from_frames(buffer);

    // TODO: upgrade type for storage?
    llvm::Type *load_type = codegen_type(type);
    unsigned address_space = base_addr->getType()->getPointerAddressSpace();
    llvm::Type *pointer_load_type = load_type->getPointerTo(address_space);

    // TODO: This can likely be removed once opaque pointers are default
    // in all supported LLVM versions.
    base_addr = builder->CreatePointerCast(base_addr, pointer_load_type);

    // TODO: support Halide's nice optimizations here.
    if (idx == nullptr) {
        return base_addr;
    }

    llvm::Constant *constant_index = llvm::dyn_cast<llvm::Constant>(idx);
    if (constant_index && constant_index->isZeroValue()) {
        return base_addr;
    }

    // Promote index to 64-bit on targets that use 64-bit pointers.
    if (d.getPointerSize() == 8) {
        llvm::Type *index_type = idx->getType();
        llvm::Type *desired_index_type = llvm::Type::getInt64Ty(
            *context); // TODO: cache this like Halide does.
        if (llvm::isa<llvm::VectorType>(index_type)) {
            desired_index_type = llvm::VectorType::get(
                desired_index_type, llvm::dyn_cast<llvm::VectorType>(index_type)
                                        ->getElementCount());
        }
        // TODO: is isSigned always true for us?
        idx = builder->CreateIntCast(idx, desired_index_type,
                                     /* isSigned */ true);
    }

    return builder->CreateInBoundsGEP(load_type, base_addr, idx);
}

llvm::Value *CodeGen_LLVM::codegen_buffer_pointer(const std::string &buffer,
                                                  const Type &type,
                                                  const Expr &idx) {
    llvm::Value *offset = idx.defined() ? codegen_expr(idx) : nullptr;
    return codegen_buffer_pointer(buffer, type, offset);
}

llvm::Value *CodeGen_LLVM::codegen_expr(const Expr &e) {
    internal_assert(e.defined());
    value = nullptr;
    e.accept(this);
    internal_assert(value) << "Failed to codegen expression: " << e;
    return value;
}

std::vector<llvm::Value *>
CodeGen_LLVM::codegen_exprs(const std::vector<ir::Expr> exprs) {
    std::vector<llvm::Value *> values(exprs.size());
    for (size_t i = 0; i < exprs.size(); i++) {
        values[i] = codegen_expr(exprs[i]);
    }
    return values;
}

void CodeGen_LLVM::codegen_stmt(const Stmt &s) {
    internal_assert(s.defined());
    s.accept(this);
}

llvm::Type *CodeGen_LLVM::codegen_type(const Type &t) {
    internal_assert(t.defined());
    type = nullptr;
    t.accept(this);
    internal_assert(type) << "Failed to codegen type: " << t;
    return type;
}

llvm::Function *CodeGen_LLVM::codegen_func_ptr(const Expr &expr) {
    if (expr.is<Var>()) {
        return module->getFunction(expr.as<Var>()->name);
    }
    internal_error << "TODO: cannot codegen function pointer from: " << expr;
}

llvm::Value *CodeGen_LLVM::codegen_write_loc(const ir::WriteLoc &loc) {
    llvm::Value *base = nullptr;
    if (frames.name_in_scope(loc.base)) {
        auto [_base, _mutable] = frames.from_frames(loc.base);
        internal_assert(_mutable)
            << "Attempting to codegen write to immutable data: " << loc.base;
        base = _base;
    } else {
        // Create alloc of base type
        llvm::Type *base_type = codegen_type(loc.base_type);

        base = builder->CreateAlloca(base_type, /* arraysize ? */ nullptr,
                                     loc.base);
        frames.add_to_frame(loc.base, {base, /* mutable */ true});
    }
    llvm::Value *ptr = base;
    llvm::Type *ptype = codegen_type(loc.base_type);
    std::string name = loc.base;

    for (const auto &value : loc.accesses) {
        if (std::holds_alternative<std::string>(value)) {
            internal_error << "TODO: implement field write access: "
                           << std::get<std::string>(value)
                           << " in loc: " << loc;
        } else {
            Expr idx = std::get<Expr>(value);
            llvm::Value *_idx = codegen_expr(idx);
            name += "_idx";
            ptr = builder->CreateGEP(ptype, ptr, {_idx}, name);
            // TODO: update ptype?
        }
    }
    return ptr;
}

std::unique_ptr<llvm::raw_fd_ostream>
make_raw_fd_ostream(const std::string &filename) {
    std::string error_string;
    std::error_code err;
    std::unique_ptr<llvm::raw_fd_ostream> raw_out(
        new llvm::raw_fd_ostream(filename, err, llvm::sys::fs::OF_None));
    if (err) {
        error_string = err.message();
    }
    internal_assert(error_string.empty())
        << "Error opening output " << filename << ": " << error_string << "\n";

    return raw_out;
}

} //  namespace bonsai
