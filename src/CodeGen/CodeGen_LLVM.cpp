#include "CodeGen/CodeGen_LLVM.h"

#include <limits>

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
#include "Lower/Random.h"

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

} // namespace

using namespace ir;

std::unique_ptr<llvm::TargetMachine>
CodeGen_LLVM::make_target_machine(llvm::Module &module,
                                  const CompilerOptions &options) {
    // Generated code follows the host unless the target is named explicitly.
    // Naming it is what makes output reproducible on another machine, which
    // is why the tests that diff generated code pass --triple and --mcpu.
    std::string target_triple = options.target_triple.empty()
                                    ? llvm::sys::getDefaultTargetTriple()
                                    : options.target_triple;
    std::string error_string;
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
        // Not module.getTargetTriple(): the module only gets a triple below,
        // and only for ASM/CPP, so this was the empty string for every
        // backend. On an x86 host that yields a generic *i386* machine --
        // 32-bit pointers in the data layout the CPP backend then installs,
        // and a TTI reporting zero vector registers, which silently disables
        // vectorization everywhere.
        target_triple,
        // The CPU `--mcpu` named, empty for a generic one.
        options.target_cpu, /*Features=*/"", target_options,
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
        // Every target LLVM was built with, not just this machine's.
        //
        // `--triple` lets a caller name any of them, and the tests that diff
        // generated code use it so their goldens do not depend on the machine
        // that produced them. Registering only the native target made that
        // work exactly when the triple happened to match the host: asking an
        // arm64 machine for x86_64-unknown-linux-gnu got "No available targets
        // are compatible with triple", which is how CI failed while the same
        // tests passed on an x86-64 developer machine.
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();
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
        if (!arg_info.mutating && arg_info.type.is<Struct_t>()) {
            arg_t = arg_t->getPointerTo();
        }
        arg_types[i] = arg_t;
    }

    llvm::FunctionType *ftype =
        llvm::FunctionType::get(ret_type, arg_types, /*isVarArg=*/false);

    // TODO(ajr): a function the program does not export cannot be called from
    // outside this module, and giving it internal linkage lets LLVM fold it
    // into its only caller and delete it -- on the tree query in
    // tests/bonsai/backends/llvm/tree-traversal.bonsai that is 35 functions
    // and 1650 lines of IR down to 4 and 580, either way it is compiled. But
    // it also lets LLVM inline a loopified traversal, stack and all, into a
    // recursive caller, and the SSA pipeline does not survive that: rtiow goes
    // from 4004ms to 7695ms, because that path materializes struct temporaries
    // into allocas (the `@N = alloca %struct.Ray` in its output) which are
    // already costing it loads and which the extra register pressure then
    // makes much worse. Worth turning on once those are gone.
    // TODO(ajr): a function the program does not export cannot be called from
    // outside this module, and giving it internal linkage lets LLVM fold it
    // into its only caller and delete it -- on the tree query in
    // tests/bonsai/backends/llvm/tree-traversal.bonsai that is 35 functions
    // and 1650 lines of IR down to 4 and 580, either way it is compiled.
    //
    // What stops it is that LLVM then inlines a loopified traversal into a
    // caller that is itself recursive, and a loopified traversal owns a
    // fixed-size stack: rtiow ends up with the `[64 x i16]` inside `sample`,
    // which recurses once per bounce, and goes from 3983ms to 7542ms. The
    // inliner does not charge for an alloca it duplicates down a recursion.
    // Loopifying the recursive caller as well avoids it, but that is the
    // schedule's choice to make, not something to assume here.
    // A function asked to be inlined can only actually be folded into its
    // caller and dropped if nothing outside this module might call it, so it
    // needs internal linkage to go with the request -- otherwise LLVM has to
    // keep a standalone copy for a caller that cannot exist, and inlining
    // while still paying for the original is a trade it declines. Exported
    // functions and kernels are looked up by name from outside and stay put.
    const bool fold_into_callers =
        func.is_always_inlined() && !func.is_exported() && !func.is_kernel();
    llvm::Function *fn = llvm::Function::Create(
        ftype,
        fold_into_callers ? llvm::GlobalValue::InternalLinkage
                          : llvm::GlobalValue::ExternalLinkage,
        func.name, module.get());
    if (fold_into_callers) {
        fn->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    for (uint32_t i = 0; i < func.args.size(); i++) {
        const auto &arg_info = func.args[i];
        llvm::AttrBuilder attrs(*context);

        attrs.addAttribute(llvm::Attribute::NoUndef);

        if (arg_info.type.is<Ptr_t>()) {
            attrs.addAttribute(llvm::Attribute::NonNull);

            if (!arg_info.mutating) {
                attrs.addAttribute(llvm::Attribute::ReadOnly);
            }

            // Nothing else this function can reach refers to it, so a write
            // through it cannot have changed anything read through the
            // others (see Function::Argument::unaliased). Only set for
            // objects lowering invented; two of a program's own arguments may
            // be the same object.
            if (arg_info.unaliased) {
                attrs.addAttribute(llvm::Attribute::NoAlias);
            }

            // TODO: Add dereferenceable + alignment if we can figure that out.
        }

        fn->addParamAttrs(i, attrs);
    }
    return fn;
}

void CodeGen_LLVM::compile_function(const Function &func,
                                    llvm::Function *function) {
    frames.push_frame();

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
        arg.setName(name);
        llvm::Value *arg_value = &arg;

        internal_assert(!arg_info.type.is<Struct_t>());

        // TODO(ajr): lift loads from immutable ptr args.

        frames.add_to_frame(arg_info.name, arg_value);
        arg_idx++;
    }

    if (func.must_setup_rng()) {
        const uint32_t lanes = native_vector_bits() / 32;
        llvm::Function *rand_function = module->getFunction("rand");
        if (!rand_function) {
            llvm::FunctionType *rand_func_type =
                llvm::FunctionType::get(i32_t, {}, false);
            rand_function = llvm::Function::Create(
                rand_func_type, llvm::GlobalValue::ExternalLinkage, "rand",
                module.get());
        }

        // Generate i32 x lanes vector using repeated scalar rand() calls
        llvm::Value *rand_vec =
            llvm::UndefValue::get(llvm::FixedVectorType::get(i32_t, lanes));
        for (uint32_t i = 0; i < lanes; ++i) {
            llvm::Value *r = builder->CreateCall(rand_function);
            rand_vec = builder->CreateInsertElement(rand_vec, r, i);
        }

        // Allocate space on stack for vector (aligned to vector width)
        llvm::AllocaInst *rng_state_ptr = builder->CreateAlloca(
            rand_vec->getType(), nullptr, lower::rng_state_name);
        rng_state_ptr->setAlignment(llvm::Align(alignof(uint32_t) * lanes));
        builder->CreateStore(rand_vec, rng_state_ptr);

        frames.add_to_frame(lower::rng_state_name, rng_state_ptr);
    }

    codegen_stmt(func.body);

    // A body that runs off the end without returning. For a void function
    // that is simply how most of them are written -- there is nothing to
    // return, so nothing says so -- and the block it leaves open still needs
    // a terminator. A function that returns a value cannot get here by
    // falling off the end, because Lower/TypeInference.cpp rejects one whose
    // paths do not all return (and exempts void from that check for exactly
    // this reason); what is left open there is a block nothing reaches, such
    // as the join made for an `if` whose arms all returned.
    if (!builder->GetInsertBlock()->getTerminator()) {
        if (func.ret_type.is<Void_t>()) {
            builder->CreateRetVoid();
        } else {
            builder->CreateUnreachable();
        }
    }

    frames.pop_frame();

    // Restore previous insertion point
    builder->restoreIP(here);

    // Validate the generated code, checking for consistency.
    if (llvm::verifyFunction(*function, &llvm::errs())) {
        llvm::errs() << *function << "\n";
        llvm::errs().flush();
        internal_error << "Function verification failed for " << func.name
                       << "\n";
    }

    current_function = nullptr;

    // function->dump();
}

std::unique_ptr<llvm::Module>
CodeGen_LLVM::compile_program(const Program &program,
                              const CompilerOptions &options) {
    init_module(); // TODO: init_codegen()?

    // Remembered rather than stamped on the module: what a parallel loop
    // lowers to depends on the platform, so generating one has to be able to
    // ask what it is -- but the module is deliberately left without a triple
    // for the LLVM backend, so that the code it prints is the same whatever
    // machine printed it.
    target_triple = options.target_triple.empty()
                        ? llvm::sys::getDefaultTargetTriple()
                        : options.target_triple;
    // Made up front rather than after the code is generated: generating it
    // needs to know how the target lays a struct out, and what a parallel
    // loop is called there. For the backends that want them, this also puts
    // the triple and data layout on the module before anything asks it for
    // an alignment, which it was being asked for beforehand.
    target_machine = make_target_machine(*module, options);

    const auto struct_types = gather_struct_types(program);
    declare_struct_types(struct_types);

    frames.push_frame();
    // TODO: add program.externs to the global frame.
    std::map<std::string, llvm::Function *> func_map;
    for (const auto &[fname, func] : program.funcs) {
        func_map[fname] = this->declare_function(*func);
    }
    for (const auto &[fname, func] : program.funcs) {
        // A function whose SSA form was kept is generated from that, rather
        // than from the statements the relooper rebuilt out of it. Both exist
        // -- the statements are what `-p ssa` prints -- and this picks which
        // one the machine sees.
        const auto ssa = program.ssa_funcs.find(fname);
        if (options.is_verbose) {
            std::cerr << "; " << fname << ": generated from "
                      << (ssa != program.ssa_funcs.end() ? "SSA" : "statements")
                      << "\n";
        }
        if (ssa != program.ssa_funcs.end()) {
            this->compile_function(*ssa->second, func_map[fname]);
        } else {
            this->compile_function(*func, func_map[fname]);
        }
    }
    frames.pop_frame();

    llvm::TargetMachine *tm = target_machine.get();

    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[pre-optimization] compilation resulted in an invalid module";
    optimize_module(*tm, options);
    internal_assert(!llvm::verifyModule(*module, &llvm::errs()))
        << "[post-optimization] compilation resulted in an invalid module";

    return std::move(module);
}

void CodeGen_LLVM::optimize_module(llvm::TargetMachine &tm,
                                   const CompilerOptions &options) {
    switch (options.level) {
    case BackendOptimizationLevel::O0:
        return; // do nothing
    case BackendOptimizationLevel::O3:
        break;
    }

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

void CodeGen_LLVM::visit(const Rand_State_t *node) {
    // This is device-specific. For now, we default to a vector the size of the
    // vector width.
    type = llvm::VectorType::get(i32_t, native_vector_bits() / 32,
                                 /* Scalable */ false);
}

llvm::FunctionType *CodeGen_LLVM::get_function_type(const ir::Type &type) {
    const Function_t *node = type.as<ir::Function_t>();
    internal_assert(node);
    std::vector<llvm::Type *> input_types;
    for (const ir::Function_t::ArgSig &arg : node->arg_types) {
        input_types.push_back(codegen_type(arg.type));
    }
    llvm::Type *return_type = codegen_type(node->ret_type);
    return llvm::FunctionType::get(return_type, std::move(input_types),
                                   /*isVariadic=*/false);
}

void CodeGen_LLVM::visit(const Function_t *node) {
    llvm::FunctionType *function_type = get_function_type(ir::Type(node));
    type = llvm::PointerType::getUnqual(function_type);
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

void CodeGen_LLVM::visit(const StringImm *node) {
    internal_error << "[unimplemented] StringImm in LLVM: " << Expr(node);
}

void CodeGen_LLVM::visit(const SizeOf *node) {
    // The target's own answer, including whatever padding it applies -- a
    // <3 x float> is twelve bytes of data in sixteen bytes of storage.
    const uint64_t bytes =
        module->getDataLayout().getTypeAllocSize(codegen_type(node->of));
    // Type::bytes() spells the same rule out for the front end, where there
    // is no target to ask. If the two ever disagree, the front end is
    // handing out a stride the generated code does not use.
    if (node->of.is_vector()) {
        internal_assert(bytes == node->of.bytes())
            << "Type::bytes() says " << node->of.bytes() << " for " << node->of
            << ", but the target lays it out in " << bytes;
    }
    value = llvm::ConstantInt::get(codegen_type(node->type), bytes);
}

void CodeGen_LLVM::visit(const Extrema *node) {
    llvm::Type *type = codegen_type(node->type);
    // A vector of them is the scalar splatted; ConstantFP/ConstantInt::get
    // build the splat when handed a vector type, but the queries below have
    // to be asked of the element type.
    llvm::Type *scalar = type->getScalarType();

    switch (node->op) {
    case Extrema::inf: {
        if (scalar->isFloatingPointTy()) {
            value = llvm::ConstantFP::getInfinity(type);
            return;
        }
        if (scalar->isIntegerTy()) {
            // An integer has no infinity; the largest representable value is
            // what an unbounded starting point means for one.
            const uint32_t bits = scalar->getIntegerBitWidth();
            const llvm::APInt max_val =
                node->type.is_int() ? llvm::APInt::getSignedMaxValue(bits)
                                    : llvm::APInt::getMaxValue(bits);
            value = llvm::ConstantInt::get(type, max_val);
            return;
        }
        break;
    }
    case Extrema::eps: {
        // The gap between one and the next representable value, which is what
        // the numerics in the standard library scale their error bounds by.
        if (scalar->isFloatTy()) {
            value = llvm::ConstantFP::get(
                type, double(std::numeric_limits<float>::epsilon()));
            return;
        }
        if (scalar->isDoubleTy()) {
            value = llvm::ConstantFP::get(
                type, std::numeric_limits<double>::epsilon());
            return;
        }
        if (scalar->isHalfTy()) {
            // 2^-10, half precision having a ten bit significand.
            value = llvm::ConstantFP::get(type, 0x1p-10);
            return;
        }
        if (scalar->isIntegerTy()) {
            // The smallest step between two distinct integers.
            value = llvm::ConstantInt::get(type, 1);
            return;
        }
        break;
    }
    }

    internal_error << "Extrema codegen not yet supported for type: "
                   << node->type << " in: " << Expr(node);
}

void CodeGen_LLVM::visit(const Var *node) {
    auto frame_value = frames.from_frames(node->name);
    internal_assert(frame_value.has_value()) << node->name;
    value = *frame_value;
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
        case BinOp::Neq: {
            value = builder->CreateFCmpONE(a, b);
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
        // Short circuiting is already gone by this point (see
        // Lower/LogicalOperations.cpp), so these are plain bitwise
        // operations on i1.
        case BinOp::LAnd: {
            value = builder->CreateAnd(a, b);
            return;
        }
        case BinOp::LOr: {
            value = builder->CreateOr(a, b);
            return;
        }
        case BinOp::Eq: {
            value = builder->CreateICmpEQ(a, b);
            return;
        }
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
        // internal_assert(cond->getType()->isVectorTy())
        //     << "Select lowering failure: " << ir::Expr(node);
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

    if (const ir::StringImm *str_imm = node.as<ir::StringImm>()) {
        to_print += str_imm->value;
        return;
    }

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
        // Unique per array printed, not per nesting depth: an array of arrays
        // recurses at the same indent level (only structs indent), so a name
        // derived from the level collides with the one the enclosing array is
        // still holding.
        const std::string name =
            "__array_print" + std::to_string(array_print_counter++);
        Expr to_print_expr = node;
        if (!node.is<Var>()) {
            // Evaluate node once, then perform extracts.
            frames.push_frame();
            frames.add_to_frame(name, codegen_expr(node));
            to_print_expr = Var::make(node.type(), name);
        }
        for (uint64_t i = 0, e = *constant_size; i < e; ++i) {
            static const ir::Type u32 = ir::UInt_t::make(32);
            ir::Expr extract =
                ir::Extract::make(to_print_expr, make_const(u32, i));
            print_helper(extract, args, to_print, indent_level);
            if (i + 1 == e)
                continue;
            to_print += ", ";
        }
        if (!node.is<Var>()) {
            frames.pop_frame();
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
    } else if (t.is_float() && expr->getType()->isFloatingPointTy() &&
               !expr->getType()->isDoubleTy()) {
        // printf is variadic, so C's default argument promotion applies: a
        // float is passed as a double, and "%f" (see get_specifier) reads it
        // as one. Without this the callee reads eight bytes where four were
        // written and prints garbage.
        expr = builder->CreateFPExt(expr, llvm::Type::getDoubleTy(*context));
    }
    if (t.is<ir::Float_t>() && expr->getType() != f64_t) {
        // printf is variadic, so the default argument promotions apply: any
        // float narrower than a double arrives as a double, and that is what
        // the "%f" specifier reads. Passing the narrow value straight through
        // leaves the upper half of the argument undefined and prints garbage.
        expr = builder->CreateFPExt(expr, f64_t);
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

    for (size_t i = 0; i < node->args.size(); i++) {
        if (i != 0) {
            to_print += ", ";
        }
        print_helper(node->args[i], args, to_print);
    }

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

    // An array is the address of its elements, so viewing one as an array of
    // a different element type is nothing at the machine level -- only the
    // stride of later indexing changes. Vectorization does this to read an
    // array of per-lane vectors component by component.
    if (src.is_reference() && dst.is_reference()) {
        value = inner;
        return;
    }

    // What the node actually asked for. Everything below decides between a
    // conversion and a bitcast by looking at the types, which gets the answer
    // right only when the two cannot mean the same thing -- a float and an
    // integer of the same width can, and the tests below would convert.
    // Reading a float's bits is how the next representable float is reached,
    // and converting instead silently gives a different number.
    if (node->mode == Cast::Mode::Reinterpret && !src.is_reference() &&
        !dst.is_reference()) {
        llvm::Type *llvm_src = codegen_type(src);
        const llvm::DataLayout &dl = module->getDataLayout();
        internal_assert(dl.getTypeAllocSize(llvm_dst) ==
                        dl.getTypeAllocSize(llvm_src))
            << "Cannot reinterpret " << src << " as " << dst
            << ": they are not the same size";
        value = builder->CreateBitCast(inner, llvm_dst);
        return;
    }

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
    } else if (src.is<Array_t>() && dst.is<Array_t>()) {
        value = inner; // no-op
    } else if ((src.is<Array_t>() || src.is<Ptr_t>()) && dst.is<Ptr_t>()) {
        // Array_t values are always pointer-backed at the LLVM level (see
        // Array_t codegen above), so this covers both a genuine
        // pointer-to-pointer reinterpret (e.g. Ptr_t(Array_t) ->
        // Ptr_t(Vector_t) for a mutable array) and treating an Array_t
        // value's own (already-pointer) representation as a differently
        // -typed pointer (e.g. for reading a whole immutable array as a
        // vector). Both are address reinterprets, not size-sensitive.
        value = builder->CreateBitCast(inner, llvm_dst);
    } else if (src.is_bool() && dst.is_uint()) {
        value = builder->CreateIntCast(inner, llvm_dst,
                                       /* isSigned */ false);
    } else {
        internal_error << "TODO: implement Cast codegen: " << Expr(node)
                       << " with types: " << src << " -> " << dst;
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
    llvm::Value *base = codegen_expr(node->base);
    llvm::Value *stride = codegen_expr(node->stride);
    const uint32_t lanes = uint32_t(node->lanes);

    // base + stride * <0, 1, ..., lanes-1>
    llvm::Type *etype = base->getType();
    std::vector<llvm::Constant *> steps(lanes);
    for (uint32_t i = 0; i < lanes; i++) {
        steps[i] = node->base.type().is_float()
                       ? llvm::ConstantFP::get(etype, double(i))
                       : llvm::ConstantInt::get(etype, uint64_t(i));
    }
    llvm::Value *iota = llvm::ConstantVector::get(steps);
    llvm::Value *base_vec = builder->CreateVectorSplat(lanes, base);
    llvm::Value *stride_vec = builder->CreateVectorSplat(lanes, stride);

    if (node->base.type().is_float()) {
        value = builder->CreateFAdd(base_vec,
                                    builder->CreateFMul(stride_vec, iota));
    } else {
        value =
            builder->CreateAdd(base_vec, builder->CreateMul(stride_vec, iota));
    }
}

void CodeGen_LLVM::visit(const Extract *node) {
    Expr vec_expr = node->vec;
    if (is_dynamic_array_struct_type(vec_expr.type())) {
        // Assumption: an extraction from a dynamic array is really an
        // access to its buffer when lowered to a struct_t.
        vec_expr = Access::make("buffer", vec_expr);
    }
    llvm::Value *vec = codegen_expr(vec_expr);

    // One index per lane reads one element per lane: this is a vector load
    // over the container, dense or gathered depending on the index.
    if (node->idx.type().defined() && node->idx.type().is_vector() &&
        vec_expr.type().is<Array_t>()) {
        // Unmasked: every lane of a gang stands for a real iteration, so
        // every lane's address is one the program would have read anyway.
        // A load that must not read on some lanes is a masked Deref.
        value = create_vector_load(codegen_type(vec_expr.type().element_of()),
                                   vec, node->idx, node->idx.type().lanes(),
                                   Expr(), "extract");
        return;
    }

    llvm::Value *idx = codegen_expr(node->idx);
    if (vec_expr.type().is<Vector_t>()) {
        value = builder->CreateExtractElement(vec, idx);
    } else if (vec_expr.type().is<Array_t>()) {
        llvm::Type *etype = codegen_type(vec_expr.type().element_of());
        llvm::Value *ptr =
            builder->CreateInBoundsGEP(etype, vec, idx, "extract_ptr");
        llvm::LoadInst *load = create_aligned_load(etype, ptr, "extract");
        value = load;
    } else {
        internal_error << "[unimplemented] codegen of Extract on type: "
                       << vec_expr.type();
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
    case Intrinsic::round: {
        // Halfway cases go away from zero, which is what C's round does and
        // so what the C++ backend already emits. llvm.rint would follow the
        // rounding mode instead and disagree on exactly the halves.
        intrin = llvm::Intrinsic::round;
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
    case Intrinsic::pow: {
        intrin = llvm::Intrinsic::pow;
        break;
    }
    case Intrinsic::rand: {
        // Applies Halide's pseudorandom number generator.
        // https://github.com/halide/Halide/blob/bf9c55dd1392b87cfc82371ee40157786eb4ec78/src/Random.cpp#L19
        const uint64_t req_vals =
            node->args.size() == 1 ? *get_constant_value(node->args[0]) : 1;
        auto frame_value = frames.from_frames(lower::rng_state_name);
        internal_assert(frame_value.has_value()) << "_rng_state missing";
        llvm::Value *rng_state_ptr = *frame_value;

        // Assuming we know the native vector width
        const int lanes = native_vector_bits() / 32;
        internal_assert(req_vals == 1 || req_vals % lanes == 0)
            << "TODO: codegen rand(count) where count is not a multiple of the "
               "vector width: "
            << req_vals << " with " << lanes;
        llvm::Type *vec_ty =
            llvm::VectorType::get(i32_t, lanes, /*Scalable=*/false);

        const uint64_t c0 = 576942909;
        const uint64_t c1 = 1121052041;
        const uint64_t c2 = 1040796640;

        auto broadcast_const = [&](uint64_t val) {
            return llvm::ConstantVector::getSplat(
                llvm::ElementCount::getFixed(lanes),
                llvm::ConstantInt::get(i32_t, val));
        };

        llvm::Value *seed =
            create_aligned_load(vec_ty, rng_state_ptr, "rng_seed");
        std::vector<llvm::Value *> pieces;

        uint64_t generated = 0;
        while (generated < req_vals) {
            llvm::Value *s = seed;

            // Apply the formula: (((c2 * s) + c1) * s) + c0
            s = builder->CreateMul(broadcast_const(c2), s);
            s = builder->CreateAdd(s, broadcast_const(c1));
            s = builder->CreateMul(s, seed); // use original seed
            s = builder->CreateAdd(s, broadcast_const(c0));

            pieces.push_back(s);
            generated += lanes;

            // Update seed vector: add constant increment {lanes, 2 * lanes, ...
            // lanes * lanes}
            std::vector<llvm::Constant *> incrs;
            for (int i = 0; i < lanes; ++i) {
                incrs.push_back(llvm::ConstantInt::get(i32_t, lanes * (i + 1)));
            }
            llvm::Value *incr = llvm::ConstantVector::get(incrs);
            seed = builder->CreateAdd(seed, incr);
        }

        // Store updated seed back into RNG state
        builder->CreateStore(seed, rng_state_ptr);

        if (req_vals == 1) {
            llvm::Value *result =
                builder->CreateExtractElement(pieces[0], (uint64_t)0);

            // Now apply classic formula to produce [0.0, 1.0)
            // Use random 23 mantissa bits, which gives [1.0, 2.0)
            // Then subtract by 1.0
            // Mask for mantissa bits (23 bits): 0x007FFFFF
            llvm::Value *mantissa_mask =
                llvm::ConstantInt::get(i32_t, 0x007FFFFF);

            // Bias to make exponent = 127 (1.0): 0x3F800000
            llvm::Value *one_bits = llvm::ConstantInt::get(i32_t, 0x3F800000);

            // Apply bit manipulation: ((rand & mask) | one_bits)
            llvm::Value *rand_mantissa =
                builder->CreateAnd(result, mantissa_mask);
            llvm::Value *rand_bits = builder->CreateOr(rand_mantissa, one_bits);

            // Bitcast i32 -> float
            llvm::Value *as_float = builder->CreateBitCast(rand_bits, f32_t);

            // Subtract 1.0 to get range [0.0, 1.0)
            llvm::Value *float_ones = llvm::ConstantFP::get(f32_t, 1.0f);
            value = builder->CreateFSub(as_float, float_ones);
            return;
        }

        // Concatenate pieces
        // TODO: is there an easy way to concat?
        std::vector<llvm::Value *> elements;
        for (llvm::Value *vec : pieces) {
            for (int i = 0; i < lanes; ++i) {
                elements.push_back(builder->CreateExtractElement(vec, i));
            }
        }

        // Truncate if needed
        if (elements.size() > req_vals) {
            elements.resize(req_vals);
        }

        // Repack into result vector
        llvm::FixedVectorType *out_ty =
            llvm::FixedVectorType::get(i32_t, req_vals);
        llvm::Value *result = llvm::UndefValue::get(out_ty);
        for (size_t i = 0; i < req_vals; ++i) {
            result = builder->CreateInsertElement(result, elements[i], i);
        }

        // Now apply classic formula to produce [0.0, 1.0)
        // Use random 23 mantissa bits, which gives [1.0, 2.0)
        // Then subtract by 1.0
        llvm::FixedVectorType *fvec_ty =
            llvm::FixedVectorType::get(f32_t, req_vals);

        // Mask for mantissa bits (23 bits): 0x007FFFFF
        llvm::Value *mantissa_mask = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(req_vals),
            llvm::ConstantInt::get(i32_t, 0x007FFFFF));

        // Bias to make exponent = 127 (1.0): 0x3F800000
        llvm::Value *one_bits = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(req_vals),
            llvm::ConstantInt::get(i32_t, 0x3F800000));

        // Apply bit manipulation: ((rand & mask) | one_bits)
        llvm::Value *rand_mantissa = builder->CreateAnd(result, mantissa_mask);
        llvm::Value *rand_bits = builder->CreateOr(rand_mantissa, one_bits);

        // Bitcast i32 -> float
        llvm::Value *as_float = builder->CreateBitCast(rand_bits, fvec_ty);

        // Subtract 1.0 to get range [0.0, 1.0)
        llvm::Value *float_ones = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(req_vals),
            llvm::ConstantFP::get(f32_t, 1.0f));
        value = builder->CreateFSub(as_float, float_ones);
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
    case Intrinsic::tan: {
        intrin = llvm::Intrinsic::tan;
        break;
    }
    default: {
        internal_error << "TODO: codegen intrinsic: " << Expr(node);
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

void CodeGen_LLVM::visit(const AggOp *node) {
    internal_error << "TODO: implement AggOp code generation: " << Expr(node);
}

void CodeGen_LLVM::visit(const Call *node) {
    llvm::Function *func = codegen_func_ptr(node->func);
    const size_t n_args = node->args.size();
    std::vector<llvm::Value *> args(n_args);

    const Function_t *function_t = node->func.type().as<Function_t>();
    internal_assert(function_t);

    for (size_t i = 0; i < n_args; i++) {
        llvm::Value *argument = codegen_expr(node->args[i]);

        // Struct args should have been lowered to pointers already.
        internal_assert(!function_t->arg_types[i].type.is<Struct_t>());
        if (function_t->arg_types[i].is_mutable) {
            internal_assert(argument->getType()->isPointerTy());
            args[i] = argument;
        } else {
            // Pass by value.
            args[i] = argument;
        }
    }
    if (func == nullptr) {
        // This is an argument with a function type.
        const auto *f = node->func.as<ir::Var>();
        internal_assert(f) << ir::Expr(node);
        llvm::FunctionType *function_type = get_function_type(f->type);
        internal_assert(function_type) << ir::Expr(f) << " : " << f->type;
        value =
            builder->CreateCall(function_type, codegen_expr(node->func), args);
        return;
    }
    // TODO: figure out how to make sure we have the right
    // number of arguments here for better error handling.
    value = builder->CreateCall(func, args);
}

void CodeGen_LLVM::visit(const Instantiate *node) {
    internal_error << "Instantiate node not lowered prior to codegen: "
                   << Expr(node);
}

void CodeGen_LLVM::visit(const PtrTo *node) {
    if (const Var *var = node->expr.as<Var>()) {
        if (var->name == lower::rng_state_name) {
            // This is always a ptr already
            visit(var);
            return;
        }
    }
    llvm::Value *pointee = codegen_expr(node->expr);

    if (auto *load = dyn_cast<llvm::LoadInst>(pointee)) {
        value = load->getPointerOperand();
    } else if (node->expr.type().is<Struct_t>()) {
        value = materialize_for_address(pointee,
                                        node->expr.type().as<Struct_t>()->name);
    } else if (node->expr.is<Extract, Access>()) {
        // Build a pointer via accesses, similar to codegen_writeloc.
        std::vector<std::variant<std::string, Expr>> accesses; // backwards
        Expr expr = node->expr;
        do {
            if (const Extract *extract = expr.as<Extract>()) {
                accesses.push_back(extract->idx);
                expr = extract->vec;
            } else {
                const Access *access = expr.as<Access>();
                internal_assert(access) << expr;
                accesses.push_back(access->field);
                expr = access->value;
            }
        } while (expr.is<Extract, Access>());
        const Deref *deref = expr.as<Deref>();
        internal_assert(deref) << expr;

        llvm::Value *ptr = codegen_expr(deref->expr);

        Type bonsai_type = deref->type;
        llvm::Type *llvm_t = codegen_type(bonsai_type);

        for (auto it = accesses.rbegin(); it != accesses.rend(); ++it) {
            const auto &access = *it;
            if (std::holds_alternative<Expr>(access)) {
                Expr idx = std::get<Expr>(access);
                llvm::Value *llvm_idx = codegen_expr(idx);

                ptr = create_aligned_load(codegen_type(bonsai_type), ptr,
                                          "ptr_array_ld");

                bonsai_type = bonsai_type.element_of();
                llvm_t = codegen_type(bonsai_type);

                ptr = builder->CreateInBoundsGEP(
                    codegen_type(bonsai_type), // The LLVM element type
                    ptr,                       // The pointer to the container
                    llvm_idx,                  // GEP indices
                    "ptr_array_deref");
            } else {
                internal_assert(std::holds_alternative<std::string>(access));
                const std::string &field_name = std::get<std::string>(access);

                const Struct_t *struct_t = bonsai_type.as<Struct_t>();
                internal_assert(struct_t)
                    << "Field access (" << field_name << ") on non-struct type "
                    << bonsai_type;
                const size_t idx =
                    find_struct_index(field_name, struct_t->fields);

                // CreateStructGEP does the {0, fld} GEP for you
                ptr = builder->CreateStructGEP(llvm_t, // the LLVM StructType*
                                               ptr,    // pointer to the struct
                                               idx,    // which field
                                               field_name + "_gep");

                bonsai_type = struct_t->fields[idx].type;
                llvm_t = codegen_type(bonsai_type);
            }
        }

        value = ptr;
    } else {
        // A value that is not in memory and has no piece of anything in
        // memory to point into -- the result of an arithmetic expression
        // handed to a parameter taken by pointer, say. There is nothing to
        // name, so the only thing its address can mean is a copy.
        value = materialize_for_address(pointee, "value");
    }
}

// Somewhere to point at, for a value that is not anywhere. The slot goes in
// the entry block and is named after what it holds, so that repeated uses of
// the same type reuse one slot rather than growing the frame per occurrence.
llvm::Value *CodeGen_LLVM::materialize_for_address(llvm::Value *pointee,
                                                   const std::string &name) {
    llvm::Value *alloca =
        create_alloca_at_entry(pointee->getType(), name + "_ptrto");
    builder->CreateStore(pointee, alloca);
    return alloca;
}

void CodeGen_LLVM::visit(const Deref *node) {
    llvm::Value *pointer_value = codegen_expr(node->expr);

    if (node->mask.defined()) {
        // A predicated load: the disabled lanes read as zero without
        // touching memory. One address per lane is a gather, one address for
        // all of them is a contiguous masked load.
        llvm::Type *loaded_type = codegen_type(node->type);
        llvm::Value *mask = codegen_expr(node->mask);
        const llvm::DataLayout &dl = module->getDataLayout();
        llvm::Value *passthrough = llvm::Constant::getNullValue(loaded_type);

        if (llvm::isa<llvm::VectorType>(pointer_value->getType())) {
            value = builder->CreateMaskedGather(
                llvm::dyn_cast<llvm::VectorType>(loaded_type), pointer_value,
                dl.getABITypeAlign(loaded_type->getScalarType()), mask,
                passthrough, "deref_gather");
        } else {
            value = builder->CreateMaskedLoad(loaded_type, pointer_value,
                                              dl.getABITypeAlign(loaded_type),
                                              mask, passthrough, "deref_temp");
        }
        return;
    }

    // Make sure the expression is a pointer
    if (pointer_value->getType()->isPointerTy()) {
        llvm::Type *loaded_type = codegen_type(node->type);
        // Dereference the pointer (load the value at the pointer address)
        llvm::LoadInst *load =
            create_aligned_load(loaded_type, pointer_value, "deref_temp");
        add_tbaa(load, node->type);
        value = load;
    } else {
        internal_error << "Cannot dereference non-pointer expression: "
                       << node->expr;
    }
}

void CodeGen_LLVM::visit(const AtomicAdd *node) {
    llvm::Value *ptr = codegen_expr(node->ptr);
    llvm::Value *acc = codegen_expr(node->value);
    internal_assert(ptr->getType()->isPointerTy())
        << "Cannot perform atomic add on non-pointer expression: " << node->ptr;

    llvm::Type *elt_t = codegen_type(node->ptr.type().element_of());
    if (acc->getType() != elt_t) {
        if (acc->getType()->isIntegerTy() && elt_t->isIntegerTy()) {
            const uint64_t dst_bits =
                cast<llvm::IntegerType>(elt_t)->getBitWidth();
            const uint64_t src_bits =
                cast<llvm::IntegerType>(acc->getType())->getBitWidth();
            if (src_bits < dst_bits) {
                acc = builder->CreateZExt(acc, elt_t, "atomicadd_zext");
            } else {
                acc = builder->CreateTrunc(acc, elt_t, "atomicadd_trunc");
            }
        } else if (acc->getType()->isFloatingPointTy() &&
                   elt_t->isFloatingPointTy()) {
            acc = builder->CreateFPCast(acc, elt_t, "atomicadd_fpcast");
        } else {
            internal_error << "Type mismatch in atomic add: value is "
                           << node->value.type() << " but pointer-to is "
                           << node->ptr.type().element_of();
        }
    }

    // LLVM rmw add, returns *old* value at ptr
    llvm::AtomicOrdering ordering = llvm::AtomicOrdering::Monotonic;
    llvm::MaybeAlign alignment; // chooses alignment if necessary
    // TODO: does this always need to be System scope?
    llvm::Value *old =
        builder->CreateAtomicRMW(llvm::AtomicRMWInst::Add, ptr, acc, alignment,
                                 ordering, llvm::SyncScope::System);

    value = old;
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
        internal_assert(node->type.is<Array_t>());
        const Array_t *array_t = node->type.as<Array_t>();

        llvm::Type *etype = codegen_type(array_t->etype);
        internal_assert(array_t->size.defined());
        llvm::Value *size = codegen_expr(array_t->size);

        // Heap allocation for dynamic-sized or returned array.
        llvm::Value *alloc =
            (allocate_memory == ir::Allocate::Memory::Heap)
                ? create_malloc(etype, size, /*zero_initialize=*/false, "")
                : create_alloca_at_entry(etype, "local_array_build", size);

        for (size_t i = 0; i < values.size(); i++) {
            llvm::Value *index = llvm::ConstantInt::get(size->getType(), i);
            llvm::Value *ptr =
                builder->CreateInBoundsGEP(etype, alloc, index, "build_ptr");
            builder->CreateStore(values[i], ptr);
        }
        value = alloc;
        return;
    } else {
        internal_error << "Unexpected llvm Type in Build lowering: "
                       << Expr(node);
    }
}

// A union holding one of its members: the member's bytes, read as the union.
//
// The mirror of reading one out in visit(const Access *). There is no
// insertvalue for this either -- LLVM's takes an index into a struct and wants
// that field's type, and a union's members are not fields -- so it goes
// through the storage. Storing the member writes its own size at offset zero,
// which is where every member of a union starts, and loading the whole thing
// back gives the first-class value a surrounding Build needs. Whatever of the
// storage the member did not cover stays undefined, which is what clang leaves
// in a union's padding too.
void CodeGen_LLVM::visit(const UnionOf *node) {
    llvm::Type *union_type = codegen_type(node->type);
    llvm::Value *member = codegen_expr(node->value);
    llvm::Value *storage =
        create_alloca_at_entry(union_type, "union_" + node->member);
    builder->CreateStore(member, storage);
    value = create_aligned_load(union_type, storage, "union_" + node->member);
}

void CodeGen_LLVM::visit(const Access *node) {
    ir::Expr value_e = node->value;

    // For debuggability.
    std::string name = node->field;
    if (const auto *var = value_e.as<Var>()) {
        name = var->name + "." + name;
    }

    // A union's members all begin at the same address, so reading one is
    // reading those bytes at that member's type. There is no extractvalue for
    // that -- LLVM's takes an index into a struct and gives that field's type
    // -- so it goes through the storage, which is what a union is.
    if (const Union_t *as_union = value_e.type().as<Union_t>()) {
        const Type member = as_union->member(node->field);
        internal_assert(member.defined())
            << "Union " << as_union->name << " has no member " << node->field
            << " in " << Expr(node);
        llvm::Value *storage = codegen_expr(PtrTo::make(value_e));
        value = create_aligned_load(codegen_type(member), storage, name);
        return;
    }

    internal_assert(value_e.type().is<Struct_t>()) << value_e;
    llvm::Value *field = codegen_expr(value_e);
    if (field->getType()->isStructTy()) {
        const auto &fields = value_e.type().as<Struct_t>()->fields;
        const size_t idx = find_struct_index(node->field, fields);
        value = builder->CreateExtractValue(field, idx, name);
        return;
    }
    llvm::errs() << *field << " : " << *field->getType() << "\n";
    llvm::errs().flush();
    internal_error
        << "Lowering of an ir::Access's value did not result in a struct type: "
        << Expr(node);
}

void CodeGen_LLVM::visit(const Unwrap *node) {
    internal_error << "Unwrap should have been lowered before CodeGen_LLVM "
                   << Expr(node);
}

void CodeGen_LLVM::visit(const Return *node) {
    Expr value = node->value;
    if (!value.defined()) {
        builder->CreateRetVoid();
        return;
    }

    if (const Call *call = node->value.as<Call>()) {
        if (const Var *var = call->func.as<Var>()) {
            internal_assert(current_function);
            if (var->name == current_function->getName()) {
                llvm::Function *func = codegen_func_ptr(call->func);
                internal_assert(func);
                std::vector<llvm::Value *> args;
                for (const Expr &arg : call->args) {
                    args.push_back(codegen_expr(arg));
                }

                llvm::CallInst *tail = builder->CreateCall(func, args);
                // Make this a tailcall.
                tail->setTailCallKind(llvm::CallInst::TCK_Tail);
                tail->setCallingConv(
                    current_function
                        ->getCallingConv()); // Ensure same convention

                builder->CreateRet(tail);
                return;
            }
        }
    }

    llvm::Value *val = codegen_expr(value);
    builder->CreateRet(val);
}

void CodeGen_LLVM::visit(const LetStmt *node) {
    internal_assert(node->loc.accesses.empty());
    llvm::Value *v = codegen_expr(node->value);
    frames.add_to_frame(node->loc.base, v);
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

    // Somewhere to carry on from, needed as soon as *any* path through the
    // chain falls out of it: an arm that does not return, or an else that does
    // not, or no else at all. Only when every path returns is there nothing
    // after the chain -- and asking for both at once, as this did, left an arm
    // that falls through branching to a block that was never made.
    needs_after_bb =
        needs_after_bb || !final_else.defined() || !always_returns(final_else);

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
    frames.push_frame();
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

void CodeGen_LLVM::visit(const While *node) {
    // As a do-while, except that the condition is tested before the first
    // iteration as well as before each later one, so the body block is
    // reached only through the test.
    llvm::BasicBlock *cond_bb =
        llvm::BasicBlock::Create(*context, "while.cond", current_function);
    llvm::BasicBlock *loop_bb =
        llvm::BasicBlock::Create(*context, "while.body", current_function);
    llvm::BasicBlock *end_bb =
        llvm::BasicBlock::Create(*context, "while.end", current_function);

    codegen_branch(cond_bb);

    builder->SetInsertPoint(cond_bb);
    codegen_short_circuit(node->cond, loop_bb, end_bb);

    builder->SetInsertPoint(loop_bb);

    frames.push_frame();
    // Where `continue` goes: back to the test, which is this loop's latch.
    latch_blocks.push_back(cond_bb);

    codegen_stmt(node->body);

    latch_blocks.pop_back();

    codegen_branch(cond_bb);

    builder->SetInsertPoint(end_bb);
    frames.pop_frame();
}

void CodeGen_LLVM::allocate_dynamic_array_type(const Allocate *node) {
    std::string name = node->loc.base;
    Type type = node->loc.base_type;
    internal_assert(is_dynamic_array_struct_type(type)) << type;
    const auto *dynamic_array_t = type.as<Struct_t>();

    internal_assert(node->memory == Allocate::Memory::Heap) << Stmt(node);
    internal_assert(!node->value.defined()) << Stmt(node);

    // Allocate the __dyn_array struct.
    llvm::Type *struct_type = codegen_type(type);
    llvm::Value *struct_ptr = create_alloca_at_entry(struct_type, name);
    frames.add_to_frame(name, struct_ptr);
    // Find indices to each field.
    int ptr_idx = find_struct_index("buffer", dynamic_array_t->fields);
    int cap_idx = find_struct_index("capacity", dynamic_array_t->fields);
    int size_idx = find_struct_index("size", dynamic_array_t->fields);
    int mtx_idx = find_struct_index("mutex", dynamic_array_t->fields);
    // Retrieve element type and capacity.
    // Dynamic arrays are always mutable, so stored as pointers to the
    // underlying struct type. Need to dereference that pointer to load buffer.
    Expr access = ir::Access::make(
        "buffer", Deref::make(Var::make(Ptr_t::make(type), name)));
    const auto *array_t = access.type().as<Array_t>();
    internal_assert(array_t) << access.type();
    llvm::Type *element_type = codegen_type(array_t->etype);
    llvm::Value *capacity = codegen_expr(array_t->size);
    // Create the new buffer, and store it to this struct.
    llvm::Value *buffer = create_malloc(element_type, capacity,
                                        /*zero_init=*/false, name + ".buffer");
    llvm::Value *buffer_ptr = builder->CreateStructGEP(
        struct_type, struct_ptr, ptr_idx, name + ".buffer_ptr");
    // The buffer is protected behind a mutex for concurrent writes.
    builder->CreateStore(buffer, buffer_ptr);
    // Initialize the size to 0.
    llvm::Value *size_ptr = builder->CreateStructGEP(
        struct_type, struct_ptr, size_idx, name + ".size_ptr");
    llvm::StoreInst *store_size =
        builder->CreateStore(llvm::ConstantInt::get(i32_t, 0), size_ptr);
    store_size->setAtomic(llvm::AtomicOrdering::Release);
    // Initialize the current capacity. Technically this only changes behind the
    // mutex as well, but perhaps we can avoid some no-op mutex acquisitions by
    // atomically reading this.
    llvm::Value *capacity_ptr = builder->CreateStructGEP(
        struct_type, struct_ptr, cap_idx, name + ".capacity_ptr");
    llvm::StoreInst *store_capacity =
        builder->CreateStore(capacity, capacity_ptr);
    store_capacity->setAtomic(llvm::AtomicOrdering::Release);

    // Lastly, allocate a mutex for this dynamic vector.
    llvm::Value *mutex_ptr = builder->CreateStructGEP(
        struct_type, struct_ptr, mtx_idx, name + ".mutex_ptr");
    // Initialize the mutex.
    llvm::Type *i8_t = builder->getInt8Ty();
    builder->CreateCall(
        get_pthread_init(),
        {mutex_ptr, llvm::ConstantPointerNull::get(i8_t->getPointerTo())});
    return;
}

// TODO(ajr): Figure out which parts of Halide's Store
// codegen we can steal. They do better with __restrict
void CodeGen_LLVM::visit(const Allocate *node) {
    std::string name = node->loc.base;
    Type allocate_type = node->loc.base_type;
    ir::Expr value = node->value;

    if (is_dynamic_array_struct_type(allocate_type)) {
        allocate_dynamic_array_type(node);
        return;
    }

    llvm::Value *rhs = nullptr;
    if (value.defined()) {
        ScopedValue<Allocate::Memory> _(allocate_memory, node->memory);
        rhs = codegen_expr(value);
    } else if (const Array_t *array_t = allocate_type.as<Array_t>()) {
        // Do allocation
        llvm::Type *etype = codegen_type(array_t->etype);
        internal_assert(array_t->size.defined());
        llvm::Value *size = codegen_expr(array_t->size);

        rhs = (node->memory == Allocate::Memory::Stack)
                  ? create_alloca_at_entry(etype, name, size)
                  : create_malloc(etype, size, /*zero_initialize=*/false, name);
    }
    // Anything else with no initial value just declares storage, which is
    // left uninitialized until something stores to it. The SSA pipeline
    // relies on this: its builder splits a declaration's initializer into a
    // separate Store (see the Allocate visitor in SSA/Convert.cpp), so every
    // allocation reaches here without one.
    // This must alloca the ptr and store
    internal_assert(node->loc.accesses.empty())
        << "Allocating Allocate to non-local value: " << Stmt(node);
    internal_assert(!frames.from_frames(name).has_value()) << name;

    // An array handle names its elements' storage directly rather than a
    // slot holding a pointer to them, so that indexing it needs no load --
    // the same way an array argument arrives (see Type::is_reference).
    if (allocate_type.is_reference()) {
        internal_assert(rhs)
            << "Array allocation produced no storage: " << Stmt(node);
        frames.add_to_frame(name, rhs);
        return;
    }

    llvm::Type *value_type = codegen_type(node->loc.base_type);
    llvm::Value *loc = create_alloca_at_entry(value_type, name);
    frames.add_to_frame(name, loc);
    if (rhs != nullptr) {
        // TODO: when is isVolatile true?
        builder->CreateStore(rhs, loc, /*isVolatile=*/false);
    }
}

void CodeGen_LLVM::visit(const Store *node) {
    llvm::Value *rhs = codegen_expr(node->value);

    // A trailing index with one entry per lane writes one element per lane:
    // a vector store over the container, dense or scattered depending on the
    // index. The address of the container itself is everything before it.
    const WriteLoc &loc = node->loc;
    if (!loc.accesses.empty()) {
        const auto *index = std::get_if<Expr>(&loc.accesses.back());
        if (index != nullptr && index->type().defined() &&
            index->type().is_vector()) {
            WriteLoc container(loc.base, loc.base_type);
            for (size_t i = 0; i + 1 < loc.accesses.size(); i++) {
                if (const auto *field =
                        std::get_if<std::string>(&loc.accesses[i])) {
                    container.add_struct_access(*field);
                } else {
                    container.add_index_access(std::get<Expr>(loc.accesses[i]));
                }
            }
            llvm::Value *base = codegen_write_loc(container);
            // Same rule as the scalar path in codegen_write_loc: a name of
            // array type is bound to its elements' storage, but a field or
            // element of one is a slot holding the handle, which has to be
            // read before it can be indexed.
            if (!container.accesses.empty() || !container.type.is_reference()) {
                base = create_aligned_load(codegen_type(container.type), base,
                                           container.base + "_ld");
            }
            create_vector_store(rhs, codegen_type(container.type.element_of()),
                                base, *index, index->type().lanes(),
                                node->mask);
            return;
        }
    }

    internal_assert(!node->mask.defined())
        << "Masked store to a location that is not indexed per lane: "
        << Stmt(node);
    llvm::Value *dest = codegen_write_loc(loc);
    // A name bound to a value rather than to storage cannot be assigned to.
    // Without this the failure is an assertion inside LLVM, which says
    // nothing about which name in which statement is at fault.
    internal_assert(dest->getType()->isPointerTy())
        << "Cannot store to " << loc.base
        << ": it is bound to a value, not to storage, in " << Stmt(node);
    llvm::StoreInst *store =
        builder->CreateStore(rhs, dest, /*isVolatile=*/false);
    add_tbaa(store, node->value.type());
}

llvm::FunctionCallee CodeGen_LLVM::get_pthread_lock() {
    return module->getOrInsertFunction(
        "pthread_mutex_lock",
        // int pthread_mutex_lock(pthread_mutex_t *);
        llvm::FunctionType::get(builder->getInt32Ty(),
                                {builder->getInt8Ty()->getPointerTo()},
                                /*isVarArg=*/false));
}

llvm::FunctionCallee CodeGen_LLVM::get_pthread_unlock() {
    return module->getOrInsertFunction(
        "pthread_mutex_unlock",
        // int pthread_mutex_unlock(pthread_mutex_t *);
        llvm::FunctionType::get(builder->getInt32Ty(),
                                {builder->getInt8Ty()->getPointerTo()},
                                /*isVarArg=*/false));
}
llvm::FunctionCallee CodeGen_LLVM::get_pthread_init() {
    return module->getOrInsertFunction(
        "pthread_mutex_init", llvm::FunctionType::get(i32_t,
                                                      {
                                                          i8_t->getPointerTo(),
                                                          i8_t->getPointerTo(),
                                                      },
                                                      /*isVarArg=*/false));
}

void CodeGen_LLVM::ensure_capacity(
    Expr ptr, llvm::Value *index, llvm::Value *dynamic_array,
    const Struct_t *struct_t, llvm::Type *llvm_struct_t, llvm::Value *size_ptr,
    llvm::Value *capacity_ptr, llvm::Value *mutex, llvm::Type *element_type,
    const std::string &base_n) {
    internal_assert(dynamic_array);
    internal_assert(struct_t);
    internal_assert(size_ptr);
    internal_assert(capacity_ptr);
    internal_assert(mutex);
    internal_assert(element_type);

    int ptr_idx = find_struct_index("buffer", struct_t->fields);
    llvm::Value *buffer_ptr =
        builder->CreateStructGEP(llvm_struct_t, // The LLVM type of the struct
                                 dynamic_array, // The pointer to the struct
                                 ptr_idx,       // The field index
                                 base_n + ".buffer_ptr");

    // Load current capacity.
    llvm::LoadInst *capacity =
        builder->CreateLoad(i32_t, capacity_ptr, base_n + ".capacity");
    capacity->setAtomic(llvm::AtomicOrdering::Acquire);

    // Check if we need to grow.
    llvm::Value *condition =
        builder->CreateICmpUGE(index, capacity, "index-ge-capacity");
    internal_assert(current_function);
    llvm::BasicBlock *lock_bb =
        llvm::BasicBlock::Create(*context, "lock-mutex", current_function);
    llvm::BasicBlock *continue_bb =
        llvm::BasicBlock::Create(*context, "continue", current_function);
    builder->CreateCondBr(condition, lock_bb, continue_bb);
    // case 1: we need to grow
    builder->SetInsertPoint(lock_bb);

    // Lock the mutex.
    builder->CreateCall(get_pthread_lock(), {mutex});

    //  The lock has been acquired. Now double check to make sure another thread
    //  hasn't updated this.
    capacity = builder->CreateLoad(i32_t, capacity_ptr, base_n + ".capacity");

    capacity->setAtomic(llvm::AtomicOrdering::Acquire);
    condition = builder->CreateICmpUGE(index, capacity, "index-ge-capacity");

    auto *zero = llvm::ConstantInt::get(i32_t, 0);
    auto *one = llvm::ConstantInt::get(i32_t, 1);
    auto *two = llvm::ConstantInt::get(i32_t, 2);
    llvm::BasicBlock *grow_bb =
        llvm::BasicBlock::Create(*context, "grow", current_function);
    llvm::BasicBlock *unlock_bb =
        llvm::BasicBlock::Create(*context, "unlock-mutex", current_function);
    builder->CreateCondBr(condition, grow_bb, unlock_bb);

    builder->SetInsertPoint(grow_bb);
    // Handle the zero capacity case. Emit the two operands in separate
    // statements: as arguments to one call their evaluation order is
    // unspecified, and each one appends an instruction, so the order they
    // appear in the IR would otherwise depend on the compiler that built
    // Bonsai -- GCC evaluates right to left, Clang left to right.
    llvm::Value *capacity_is_zero = builder->CreateICmpEQ(capacity, zero);
    llvm::Value *doubled_capacity = builder->CreateMul(capacity, two);
    llvm::Value *new_capacity = builder->CreateSelect(
        capacity_is_zero, one, doubled_capacity, base_n + ".new-capacity");
    const llvm::DataLayout &layout = module->getDataLayout();
    llvm::Type *i8_t = llvm::Type::getInt8Ty(*context);
    llvm::Type *s_t = layout.getIntPtrType(*context);
    new_capacity = builder->CreateZExtOrBitCast(new_capacity, s_t);
    llvm::Value *element_size =
        llvm::ConstantInt::get(s_t, layout.getTypeAllocSize(element_type));
    llvm::Function *realloc = module->getFunction("realloc");
    if (realloc == nullptr) {
        llvm::FunctionType *type = llvm::FunctionType::get(
            i8_t->getPointerTo(), {i8_t->getPointerTo(), s_t},
            /*isVarArg=*/false);
        realloc = llvm::Function::Create(type, llvm::Function::ExternalLinkage,
                                         "realloc", module.get());
    }
    llvm::Value *old_buffer = codegen_expr(ptr);
    llvm::Value *new_buffer = builder->CreateCall(
        realloc, {old_buffer, builder->CreateMul(new_capacity, element_size)});

    // Update struct.ptr field
    builder->CreateStore(new_buffer, buffer_ptr);

    // Update struct.capacity field
    // Truncate capacity back to i32.
    llvm::Value *truncated_capacity =
        builder->CreateTrunc(new_capacity, capacity->getType());
    llvm::StoreInst *store_capacity =
        builder->CreateStore(truncated_capacity, capacity_ptr);
    store_capacity->setAtomic(llvm::AtomicOrdering::Release);
    // Jump to mutex unlock.
    builder->CreateBr(unlock_bb);

    builder->SetInsertPoint(unlock_bb);
    builder->CreateCall(get_pthread_unlock(), {mutex});
    builder->CreateBr(continue_bb);

    // case 2: no grow (and continuation of grow block).
    builder->SetInsertPoint(continue_bb);
}

// TODO(bonsai/issues/200): add test for parallel appends.
void CodeGen_LLVM::visit(const Append *node) {
    llvm::Value *dynamic_array = codegen_write_loc(node->loc);
    std::string base_n = node->loc.base;
    const auto *struct_t = node->loc.base_type.as<Struct_t>();
    llvm::Type *llvm_struct_t = codegen_type(struct_t);
    internal_assert(struct_t) << node->loc.base_type;
    llvm::Value *rhs = codegen_expr(node->value);

    // Pointer to the statically sized array.
    // Dynamic arrays are always mutable, so stored as pointers to the
    // underlying struct type. Need to dereference that pointer to load buffer.
    Expr base_v = Deref::make(Var::make(Ptr_t::make(struct_t), node->loc.base));
    Expr ptr = Access::make("buffer", base_v);
    const auto *array_t = ptr.type().as<Array_t>();
    internal_assert(array_t) << ptr.type();
    // Pointer to the "current size" of the array.
    int32_t size_idx = find_struct_index("size", struct_t->fields);
    llvm::Value *size_ptr =
        builder->CreateStructGEP(llvm_struct_t, // The LLVM type of the struct
                                 dynamic_array, // The pointer to the struct
                                 size_idx,      // The field index
                                 base_n + ".size_ptr");
    // Get a unique index for this thread.
    llvm::Value *one = builder->getInt32(1);
    llvm::Value *index = builder->CreateAtomicRMW(
        llvm::AtomicRMWInst::Add, size_ptr, one, llvm::MaybeAlign(),
        llvm::AtomicOrdering::AcquireRelease, llvm::SyncScope::System);
    // Pointer to the capacity of the array.
    int32_t capacity_idx = find_struct_index("capacity", struct_t->fields);
    llvm::Value *capacity_ptr =
        builder->CreateStructGEP(llvm_struct_t, // The LLVM type of the struct
                                 dynamic_array, // The pointer to the struct
                                 capacity_idx,  // The field index
                                 base_n + ".capacity_ptr");
    // Pointer to the mutex of the array.
    int32_t mutex_idx = find_struct_index("mutex", struct_t->fields);
    llvm::Value *mutex =
        builder->CreateStructGEP(llvm_struct_t, // The LLVM type of the struct
                                 dynamic_array, // The pointer to the struct
                                 mutex_idx,     // The field index
                                 base_n + ".mutex_ptr");
    // Perform resize if necessary.
    llvm::Type *element_type = codegen_type(array_t->etype);
    ensure_capacity(ptr, index, dynamic_array, struct_t, llvm_struct_t,
                    size_ptr, capacity_ptr, mutex, element_type, base_n);

    // TODO(cgyurgyik): Any stores to the buffer are currently locked behind a
    // mutex as well. Ideally, we would only need to lock when regrowing.
    builder->CreateCall(get_pthread_lock(), {mutex}); // LOCK
    // Load the buffer pointer.
    llvm::Value *buffer_ptr = codegen_expr(ptr);
    // Store the value at the given offset.
    llvm::Value *offset_in_buffer_ptr =
        builder->CreateInBoundsGEP(element_type, // The LLVM element type
                                   buffer_ptr,   // pointer to the buffer
                                   index         // offset
        );
    builder->CreateStore(rhs, offset_in_buffer_ptr, /*isVolatile=*/false);
    builder->CreateCall(get_pthread_unlock(), {mutex}); // UNLOCK
}

void CodeGen_LLVM::visit(const Accumulate *node) {
    if (node->loc.base_type.is<Vector_t>() && node->loc.accesses.size() == 1) {
        // Update a single element of a vector.
        // For now, we rewrite this into an equivalent expr.
        // This is an unfortunate hack.
        // TODO(ajr): fix.
        Type vtype = node->loc.base_type;
        Expr load = Deref::make(Var::make(Ptr_t::make(vtype), node->loc.base));
        Expr lane = std::get<Expr>(node->loc.accesses[0]);
        const size_t lanes = vtype.lanes();
        Type etype = vtype.element_of();

        Expr equiv;

        switch (node->op) {
        case Accumulate::Add: {
            Expr one_hot = make_one_hot(etype, lane, lanes);
            equiv = load + (node->value * one_hot);
            break;
        }
        case Accumulate::Sub: {
            Expr one_hot = make_one_hot(etype, lane, lanes);
            equiv = load - (node->value * one_hot);
            break;
        }
        case Accumulate::Mul: {
            Expr one_hot = make_one_hot(etype, lane, lanes);
            Expr ones = make_one(vtype);
            equiv = load * ((ones - one_hot) + node->value * one_hot);
            break;
        }
        case Accumulate::Argmin:
        default: {
            internal_error << "TODO: implement codegen for accumulate: "
                           << Stmt(node);
        }
        }
        WriteLoc base(node->loc.base, node->loc.base_type);
        Stmt equiv_stmt = Store::make(std::move(base), std::move(equiv));
        codegen_stmt(equiv_stmt);
        return;
    }

    llvm::Value *loc = codegen_write_loc(node->loc);
    llvm::Value *update = codegen_expr(node->value);

    llvm::Value *current =
        create_aligned_load(update->getType(), loc, "acc_base");

    llvm::Value *acc = nullptr;

    switch (node->op) {
    case Accumulate::Add: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFAdd(current, update);
        } else {
            acc = builder->CreateAdd(current, update);
        }
        break;
    }
    case Accumulate::Sub: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFSub(current, update);
        } else {
            acc = builder->CreateSub(current, update);
        }
        break;
    }
    case Accumulate::Mul: {
        if (node->value.type().is_float()) {
            acc = builder->CreateFMul(current, update);
        } else {
            acc = builder->CreateMul(current, update);
        }
        break;
    }
    case Accumulate::Argmin: {
        // acc = select(curr.first < update.first, curr, update)
        llvm::Value *curr_key =
            builder->CreateExtractValue(current, 0); // curr.first
        llvm::Value *new_key =
            builder->CreateExtractValue(update, 0); // update.first

        internal_assert(curr_key->getType()->isFloatingPointTy());
        llvm::Value *cmp =
            builder->CreateFCmpOLT(curr_key, new_key); // curr_key < new_key

        // Select the full struct based on which key is smaller
        acc = builder->CreateSelect(cmp, current, update);
        break;
    }
    case Accumulate::Argmax: {
        // acc = select(curr.first > update.first, curr, update)
        llvm::Value *curr_key =
            builder->CreateExtractValue(current, 0); // curr.first
        llvm::Value *new_key =
            builder->CreateExtractValue(update, 0); // update.first

        internal_assert(curr_key->getType()->isFloatingPointTy());
        llvm::Value *cmp =
            builder->CreateFCmpOGT(curr_key, new_key); // curr_key > new_key

        // Select the full struct based on which key is larger
        acc = builder->CreateSelect(cmp, current, update);
        break;
    }
    case Accumulate::Min:
    case Accumulate::Max: {
        const bool is_min = node->op == Accumulate::Min;
        const Type &t = node->value.type();
        llvm::Value *cmp = nullptr;
        if (t.is_float()) {
            cmp = is_min ? builder->CreateFCmpOLT(current, update)
                         : builder->CreateFCmpOGT(current, update);
        } else if (t.is_int()) {
            cmp = is_min ? builder->CreateICmpSLT(current, update)
                         : builder->CreateICmpSGT(current, update);
        } else {
            cmp = is_min ? builder->CreateICmpULT(current, update)
                         : builder->CreateICmpUGT(current, update);
        }
        acc = builder->CreateSelect(cmp, current, update);
        break;
    }
    default: {
        internal_error << "TODO: implement codegen for accumulate: "
                       << Stmt(node);
    }
    }

    builder->CreateStore(acc, loc);
}

llvm::LoadInst *CodeGen_LLVM::create_aligned_load(llvm::Type *etype,
                                                  llvm::Value *ptr,
                                                  const std::string &name) {
    llvm::LoadInst *load = builder->CreateLoad(etype, ptr, name);
    const llvm::DataLayout &dl = module->getDataLayout();
    unsigned align = dl.getABITypeAlign(etype).value();
    load->setAlignment(llvm::Align(align));
    return load;
}

namespace {

// A Ramp of stride one addresses `lanes` contiguous elements, which is an
// ordinary vector access rather than a gather or a scatter.
const Ramp *as_dense_ramp(const Expr &index) {
    const Ramp *ramp = index.as<Ramp>();
    if (ramp == nullptr) {
        return nullptr;
    }
    const int64_t *stride = as_const_int(ramp->stride);
    return (stride != nullptr && *stride == 1) ? ramp : nullptr;
}

} // namespace

llvm::Value *CodeGen_LLVM::create_vector_load(llvm::Type *etype,
                                              llvm::Value *base,
                                              const Expr &index, uint32_t lanes,
                                              const Expr &mask_expr,
                                              const std::string &name) {
    // A boolean occupies a byte in memory -- that is what the scalar path
    // stores and what the C++ side of an exported function sees -- while a
    // vector of i1 is bit-packed. So booleans are loaded a byte per lane and
    // narrowed afterwards, or eight lanes would come out of a single byte.
    if (etype->isIntegerTy(1)) {
        llvm::Value *bytes =
            create_vector_load(i8_t, base, index, lanes, mask_expr, name);
        return builder->CreateTrunc(
            bytes, llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false),
            name + "_bits");
    }

    llvm::Type *vtype = llvm::VectorType::get(etype, lanes, /*Scalable=*/false);
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Value *mask = mask_expr.defined() ? codegen_expr(mask_expr) : nullptr;

    if (const Ramp *ramp = as_dense_ramp(index)) {
        llvm::Value *first = builder->CreateInBoundsGEP(
            etype, base, codegen_expr(ramp->base), name + "_base");
        if (mask == nullptr) {
            return create_aligned_load(vtype, first, name);
        }
        // A disabled lane must not touch memory at all, so the disabled
        // lanes come from the passthrough value rather than from the load.
        return builder->CreateMaskedLoad(
            vtype, first, dl.getABITypeAlign(vtype), mask,
            llvm::Constant::getNullValue(vtype), name);
    }

    llvm::Value *ptrs = builder->CreateInBoundsGEP(
        etype, base, codegen_expr(index), name + "_ptrs");
    if (mask == nullptr) {
        mask = llvm::Constant::getAllOnesValue(
            llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false));
    }
    return builder->CreateMaskedGather(
        vtype, ptrs, dl.getABITypeAlign(etype), mask,
        llvm::Constant::getNullValue(vtype), name);
}

void CodeGen_LLVM::create_vector_store(llvm::Value *value, llvm::Type *etype,
                                       llvm::Value *base, const Expr &index,
                                       uint32_t lanes, const Expr &mask_expr) {
    // Booleans take a byte each in memory (see create_vector_load), so a
    // vector of them is widened before it is written; storing the i1 vector
    // directly would pack eight lanes into one byte.
    if (etype->isIntegerTy(1)) {
        llvm::Value *bytes = builder->CreateZExt(
            value, llvm::VectorType::get(i8_t, lanes, /*Scalable=*/false),
            "store_bytes");
        create_vector_store(bytes, i8_t, base, index, lanes, mask_expr);
        return;
    }

    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Value *mask = mask_expr.defined() ? codegen_expr(mask_expr) : nullptr;

    if (const Ramp *ramp = as_dense_ramp(index)) {
        llvm::Value *first = builder->CreateInBoundsGEP(
            etype, base, codegen_expr(ramp->base), "store_base");
        if (mask == nullptr) {
            llvm::StoreInst *store = builder->CreateStore(value, first);
            store->setAlignment(dl.getABITypeAlign(value->getType()));
            return;
        }
        builder->CreateMaskedStore(value, first,
                                   dl.getABITypeAlign(value->getType()), mask);
        return;
    }

    llvm::Value *ptrs = builder->CreateInBoundsGEP(
        etype, base, codegen_expr(index), "store_ptrs");
    if (mask == nullptr) {
        mask = llvm::Constant::getAllOnesValue(
            llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false));
    }
    builder->CreateMaskedScatter(value, ptrs, dl.getABITypeAlign(etype), mask);
}

llvm::Value *CodeGen_LLVM::create_alloca_at_entry(llvm::Type *t,
                                                  const std::string &name,
                                                  llvm::Value *size) {
    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    llvm::BasicBlock *entry =
        &builder->GetInsertBlock()->getParent()->getEntryBlock();
    if (entry->empty()) {
        builder->SetInsertPoint(entry);
    } else {
        builder->SetInsertPoint(entry, entry->getFirstInsertionPt());
    }
    llvm::AllocaInst *ptr = builder->CreateAlloca(t, size, name);

    const llvm::DataLayout &dl = module->getDataLayout();
    unsigned align = dl.getABITypeAlign(t).value();
    ptr->setAlignment(llvm::Align(align));

    builder->restoreIP(here);
    return ptr;
}

llvm::Value *CodeGen_LLVM::create_malloc(llvm::Type *etype, llvm::Value *size,
                                         bool zero_initialize,
                                         const std::string &name) {

    int align = native_vector_bits() / 8;

    // Size of the element in bytes
    llvm::DataLayout dataLayout(module.get());
    uint64_t typeSize = dataLayout.getTypeAllocSize(etype);
    llvm::Value *elemSize = llvm::ConstantInt::get(i64_t, typeSize);

    if (size->getType() != i64_t) {
        size =
            builder->CreateIntCast(size, i64_t, /*isSigned=*/false, "size64");
    }

    // TODO: figure out alignment?
    llvm::Value *untyped_ptr =
        builder->CreateMalloc(i64_t, etype, /*AllocSize=*/elemSize,
                              /*ArraySize=*/size, nullptr, name + "_untyped");

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

void CodeGen_LLVM::visit(const Label *node) {
    internal_assert(node->body.defined())
        << "Label with undefined body made it to codegen: " << node->name;
    // TODO: add label as a comment to body here?
    codegen_stmt(node->body);
}

void CodeGen_LLVM::visit(const ForAll *node) {
    codegen_counted_loop(node->index, node->slice.begin, node->slice.end,
                         node->slice.stride, node->body);
}

// A `parfor` that no schedule assigned to any hardware. Nothing says it has to
// run in parallel -- `parfor` states that the iterations *may* run in any
// order, and running them one after another is one of those orders -- so it is
// the same loop as a sequential one, and is emitted by the same code. A
// `parfor` that a schedule did place becomes a Launch long before here.
void CodeGen_LLVM::visit(const ParFor *node) {
    // A loop a schedule placed on hardware is not this loop. Nothing turns a
    // binding into a launch yet, and emitting the sequential form instead
    // would run the program somewhere other than it was told to -- silently,
    // and looking only like it was slow.
    internal_assert(!node->binding.has_value())
        << "bind(" << node->index << ", " << to_string(*node->binding)
        << ") is recorded on this loop, but nothing lowers a binding into a "
           "launch yet, so the loop would run sequentially instead.";
    codegen_counted_loop(node->index, node->slice.begin, node->slice.end,
                         node->slice.stride, node->body);
}

// One counted loop: `for index in begin:end:stride`, running body.
void CodeGen_LLVM::codegen_counted_loop(const std::string &index,
                                        const Expr &begin_expr,
                                        const Expr &end_expr,
                                        const Expr &stride_expr,
                                        const Stmt &body) {
    llvm::Value *begin = codegen_expr(begin_expr);

    llvm::BasicBlock *preheader_bb = builder->GetInsertBlock();

    std::string loop_id =
        index + std::to_string(forall_loop_id++) + std::string("_for");

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
    llvm::Value *enter_condition = codegen_expr(begin_expr < end_expr);
    builder->CreateCondBr(enter_condition, loop_bb, end_bb, very_likely_branch);
    builder->SetInsertPoint(loop_bb);

    // Make our phi node.
    llvm::Type *iterator_t = codegen_type(begin_expr.type());
    llvm::PHINode *phi = builder->CreatePHI(iterator_t, 2);
    phi->addIncoming(begin, preheader_bb);

    // Add index to new frame.
    frames.push_frame();
    frames.add_to_frame(index, phi);

    latch_blocks.push_back(inc_bb);
    // TODO(ajr): will need this for `break` statements.
    // escape_blocks.push_back(end_bb);

    // Emit loop body
    codegen_stmt(body);

    latch_blocks.pop_back();
    // escape_blocks.pop_back();

    codegen_branch(inc_bb);
    builder->SetInsertPoint(inc_bb);

    // Update the counter
    Expr var = Var::make(begin_expr.type(), index);
    llvm::Value *next_var = codegen_expr(var + stride_expr);
    // Add the back-edge to the phi node
    phi->addIncoming(next_var, builder->GetInsertBlock());

    // Maybe exit the loop
    // TODO(ajr): can this overflow?
    llvm::Value *end_condition = codegen_expr(var + 1 >= end_expr);
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

void CodeGen_LLVM::visit(const Launch *node) {
    llvm::Value *num_iters = codegen_expr(node->n);
    num_iters =
        builder->CreateIntCast(num_iters, i64_t, node->n.type().is_int());

    llvm::Function *launch_func = module->getFunction(node->func);
    internal_assert(launch_func)
        << "Launch function " << node->func << " not found";

    internal_assert(node->args.size() == 1); // context
    llvm::Value *ctx = codegen_expr(node->args[0]);

    // One call, everywhere. What a parallel loop is made of differs a lot
    // between machines -- libdispatch on Apple, threads here, something else
    // on Windows -- but none of that belongs in a compiler: choosing it here
    // would mean the generated code, and every golden of it, depended on
    // which machine ran the compiler rather than on the program. So this
    // emits one call against one shape, and runtime/bonsai_parallel.h decides
    // what to run it on.
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    llvm::FunctionType *parallel_for_ty =
        llvm::FunctionType::get(void_t, {i64_t, ptr_t, ptr_t}, false);
    llvm::Function *parallel_for = module->getFunction("bonsai_parallel_for");
    if (!parallel_for) {
        parallel_for = llvm::Function::Create(
            parallel_for_ty, llvm::Function::ExternalLinkage,
            "bonsai_parallel_for", module.get());
    }
    builder->CreateCall(parallel_for, {num_iters, ctx, launch_func});
}

llvm::MDNode *CodeGen_LLVM::tbaa_type_node(const Type &type) {
    if (!type.defined()) {
        return nullptr;
    }

    // An access to an aggregate has no tag: LLVM's access type has to be one
    // of its scalar nodes, so there is no way to say "all of this struct".
    // Untagged means "may alias anything", which is always safe.
    if (type.is<Struct_t, Tuple_t, Option_t, Union_t>()) {
        return nullptr;
    }

    // A vector reads the same bytes an element-at-a-time read would -- a
    // dense load out of an array of f32 is a load of f32s -- so it says the
    // element type, not the vector type. Saying the vector type would make a
    // gathered read look unrelated to a scalar one of the same array.
    if (type.is<Vector_t>()) {
        return tbaa_type_node(type.element_of());
    }

    const std::string key = to_string(type);
    if (const auto it = tbaa_types.find(key); it != tbaa_types.end()) {
        return it->second;
    }

    llvm::MDBuilder md(*context);
    if (tbaa_root == nullptr) {
        tbaa_root = md.createTBAARoot("bonsai");
    }
    llvm::MDNode *node = md.createTBAAScalarTypeNode(key, tbaa_root);
    tbaa_types[key] = node;
    return node;
}

void CodeGen_LLVM::add_tbaa(llvm::Instruction *inst, const Type &type) {
    llvm::MDNode *node = tbaa_type_node(type);
    if (node == nullptr) {
        return; // untagged, which may alias anything
    }
    llvm::MDBuilder md(*context);
    inst->setMetadata(llvm::LLVMContext::MD_tbaa,
                      md.createTBAAStructTagNode(node, node, /*Offset=*/0));
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
            struct_types[_struct->name]->setBody(types, _struct->is_packed());
        }
    }

    // Union bodies last. A struct body may name a type that is still opaque,
    // so the order above does not matter; a union has to be as big as its
    // largest member, so it cannot be measured until that member has a size.
    // Repeated because a union can hold a struct that holds another union, and
    // each round either settles one or has nothing left it can settle.
    while (!pending_unions.empty()) {
        std::vector<const Union_t *> again;
        for (const Union_t *node : pending_unions) {
            if (!set_union_body(node, union_types.at(node->name))) {
                again.push_back(node);
            }
        }
        internal_assert(again.size() < pending_unions.size())
            << "Union " << again.front()->name << " contains itself";
        pending_unions = std::move(again);
    }
}

// A union: enough storage for the largest member, aligned for the strictest.
//
// LLVM has no union type, so this is storage -- an array of units, each one
// the alignment wide. Only here can it be worked out: sizes and alignments
// belong to the target, which is why Type::bytes() refuses to answer for an
// aggregate and says to ask the backend.
//
// Not clang's encoding, which is the most-aligned member followed by enough
// bytes to reach the largest. That relies on a union only ever being touched
// through memory. Bonsai builds one with insertvalue, and a first-class LLVM
// aggregate does not carry its padding: given `{<3 x float>, float}`, bytes
// 12-15 and 20-31 belong to no field of it, so loading the union as a value
// would leave them undefined -- and another member's fields can sit exactly
// there. So the body has to have no padding of its own to lose, which an array
// of a type as wide as its own alignment does not.
//
// Unlike a struct, whose body may name a type that is still opaque, this needs
// its members to have a size already. Answering false rather than asserting is
// what lets declare_struct_types reach a union early and come back to it.
bool CodeGen_LLVM::set_union_body(const Union_t *node, llvm::StructType *made) {
    internal_assert(!node->members.empty())
        << "Union " << node->name << " has no members";

    const llvm::DataLayout &dl = module->getDataLayout();
    uint64_t align = 1;
    uint64_t size = 0;
    for (const TypedVar &member : node->members) {
        llvm::Type *as_llvm = codegen_type(member.type);
        if (!as_llvm->isSized()) {
            return false;
        }
        align = std::max(align, dl.getABITypeAlign(as_llvm).value());
        size = std::max(size, dl.getTypeAllocSize(as_llvm).getFixedValue());
    }
    // Rounded up to the alignment, so that an array of the union steps from
    // one aligned value to the next. This is the size Rust gives an enum.
    size = ((size + align - 1) / align) * align;

    // One unit of storage: an integer as wide as the alignment, or the same
    // number of bytes as a vector. Which of them a target actually aligns the
    // way it is asked to varies -- x86-64 and AArch64 both say i128 is aligned
    // to 16, but neither says anything about i256 -- so the choice is made by
    // measuring rather than by knowing.
    llvm::Type *unit = nullptr;
    const std::vector<llvm::Type *> candidates{
        llvm::Type::getIntNTy(*context, align * 8),
        llvm::FixedVectorType::get(i8_t, align)};
    for (llvm::Type *candidate : candidates) {
        if (dl.getTypeAllocSize(candidate).getFixedValue() == align &&
            dl.getABITypeAlign(candidate).value() == align) {
            unit = candidate;
            break;
        }
    }
    internal_assert(unit) << "No type on this target is " << align
                          << " bytes aligned to " << align << ", which union "
                          << node->name << " needs";

    made->setBody({llvm::ArrayType::get(unit, size / align)},
                  /*isPacked=*/false);
    internal_assert(dl.getTypeAllocSize(made).getFixedValue() == size &&
                    dl.getABITypeAlign(made).value() == align)
        << "Union " << node->name << " wanted " << size << " bytes aligned to "
        << align << " but its storage is "
        << dl.getTypeAllocSize(made).getFixedValue() << " aligned to "
        << dl.getABITypeAlign(made).value();
    return true;
}

void CodeGen_LLVM::visit(const Union_t *node) {
    const auto found = union_types.find(node->name);
    if (found != union_types.end()) {
        type = found->second;
        return;
    }

    llvm::StructType *made =
        llvm::StructType::create(*context, "union." + node->name);
    // Registered before the body is built, so that a union reachable from its
    // own members does not recurse for ever.
    union_types[node->name] = made;
    if (!set_union_body(node, made)) {
        pending_unions.push_back(node);
    }
    type = made;
}

llvm::Value *CodeGen_LLVM::codegen_buffer_pointer(const std::string &buffer,
                                                  const Type &type,
                                                  llvm::Value *idx) {
    llvm::DataLayout d(module.get());
    auto frame_value = frames.from_frames(buffer);
    internal_assert(frame_value.has_value()) << buffer;
    llvm::Value *base_addr = *frame_value;

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

llvm::Value *CodeGen_LLVM::codegen_write_loc(const ir::WriteLoc &wloc) {
    std::string name = wloc.base;
    auto frame_value = frames.from_frames(name);
    internal_assert(frame_value.has_value()) << name;
    llvm::Value *loc = *frame_value;
    Type bonsai_type = wloc.base_type;

    // A name of array type is bound to its elements' storage directly (see
    // Type::is_reference and the Allocate visitor), so indexing it needs no
    // load. That only holds for the name itself: once a field or an element
    // has been reached, `loc` is the address of a slot holding the handle,
    // which does have to be read first.
    bool holds_handle = !wloc.base_type.is_reference();

    for (const auto &value : wloc.accesses) {
        if (std::holds_alternative<std::string>(value)) {
            const std::string &field_name = std::get<std::string>(value);
            const Struct_t *struct_t = bonsai_type.as<Struct_t>();
            internal_assert(struct_t) << "Field access (" << field_name
                                      << ") on non-struct type " << bonsai_type;
            const size_t idx = find_struct_index(field_name, struct_t->fields);

            // Get lvalue to loc.`field_name`
            name += "_" + field_name;
            loc = builder->CreateStructGEP(
                codegen_type(bonsai_type), // The LLVM type of the struct
                loc,                       // The pointer to the struct
                idx,                       // The field index
                name                       // Optional name for debugging
            );
            bonsai_type = struct_t->fields[idx].type;
            holds_handle = true;
        } else {
            Expr idx = std::get<Expr>(value);
            llvm::Value *llvm_idx = codegen_expr(idx);

            if (holds_handle) {
                loc = create_aligned_load(codegen_type(bonsai_type), loc,
                                          name + "_ld");
                name += "_ld";
            }
            holds_handle = true;

            // Get lvalue to loc[`idx`]
            bonsai_type = bonsai_type.element_of();
            loc = builder->CreateInBoundsGEP(
                codegen_type(bonsai_type), // The LLVM element type
                loc,                       // The pointer to the container
                llvm_idx,                  // GEP indices
                name);
        }
    }
    return loc;
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
