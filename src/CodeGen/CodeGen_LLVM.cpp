#include "CodeGen/CodeGen_LLVM.h"

#include "CodeGen/CodeGen_X86.h"

#include <llvm/MC/MCSubtargetInfo.h>

#include <functional>
#include <limits>
#include <numeric>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/ReplaceWithVeclib.h>
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
#include <llvm/TargetParser/SubtargetFeature.h>
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

#include <dlfcn.h>

#include <sstream>

namespace bonsai {
namespace codegen {
void to_llvm(const ir::Program &program, const CompilerOptions &options) {
    std::unique_ptr<CodeGen_LLVM> codegen = make_llvm_codegen(options);
    std::unique_ptr<llvm::Module> module =
        codegen->compile_program(program, options);
    // Without the triple and data layout the module was generated against,
    // so that the same program prints the same on every machine (see
    // make_target_machine). A test that pins `--triple` gets the same code
    // either way; one that follows the host would otherwise print the
    // host's.
    if (options.output_file.empty()) {
        codegen->print_module(*module, llvm::outs(), /*redacted=*/true);
        return;
    }
    auto os = make_raw_fd_ostream(options.output_file);
    codegen->print_module(*module, *os, /*redacted=*/true);
}
} // namespace codegen

std::unique_ptr<CodeGen_LLVM> make_llvm_codegen(const CompilerOptions &options) {
    // The same choice make_target_machine makes of the triple: the one
    // named, or the host's.
    const std::string triple = options.target_triple.empty()
                                   ? llvm::sys::getDefaultTargetTriple()
                                   : options.target_triple;
    if (llvm::Triple(triple).getArch() == llvm::Triple::x86_64) {
        return std::make_unique<CodeGen_X86>();
    }
    return std::make_unique<CodeGen_LLVM>();
}

bool CodeGen_LLVM::has_feature(const std::string &feature) const {
    return target_machine != nullptr &&
           target_machine->getMCSubtargetInfo()->checkFeatures("+" + feature);
}
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

    // ...and following the host means its CPU and its instruction set, not
    // just its pointer size. Left empty, LLVM targets a generic x86-64 -- SSE2
    // and nothing after it -- so a machine with AVX and FMA got neither, an
    // `fma` became a call to libm's `fmaf`, and nothing was vectorized more
    // than four floats wide. Naming the host CPU is what `-march=native` does
    // for a C compiler, and it is what the code this is compared against is
    // built with.
    //
    // Only when nothing was named: `--triple` or `--mcpu` means the caller
    // wants a particular machine and is probably diffing the output, and a
    // triple that is not this machine's cannot take this machine's features.
    std::string target_cpu = options.target_cpu;
    std::string target_features;
    if (target_cpu.empty() && options.target_triple.empty()) {
        target_cpu = llvm::sys::getHostCPUName().str();
        llvm::SubtargetFeatures features;
        for (const auto &feature : llvm::sys::getHostCPUFeatures()) {
            // Every feature the host has, AVX-512 included: a gang is as
            // wide as the machine's register (sixteen lanes in a zmm), and
            // LLVM keeps a masked gather or scatter as one instruction only
            // on a machine with AVX-512 or Intel's fast-gather tuning, and
            // scalarizes it everywhere else -- one extract, load and insert
            // per lane -- which on an AMD host with the feature stripped was
            // the whole of a gang's memory traffic. The register width LLVM
            // is told to use is the machine's too (`prefer-vector-width` and
            // `min-legal-vector-width` on every function, see
            // optimize_module), and a stack slot an aggregate lives in is
            // aligned to it (see create_alloca_at_entry): with the registers
            // at 512 bits and the slots aligned to 256, a memset of a widened
            // local once faulted.
            features.AddFeature(feature.first(), feature.second);
        }
        target_features = features.getString();
        follows_host = true;
        if (llvm::Triple(target_triple).isOSLinux() &&
            llvm::Triple(target_triple).getArch() == llvm::Triple::x86_64) {
            probe_host_vector_math();
        }
    }

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
        // The CPU `--mcpu` named, or the host's when nothing was named.
        target_cpu, target_features, target_options,
        use_pic ? llvm::Reloc::PIC_ : llvm::Reloc::Static,
        use_large_code_model ? llvm::CodeModel::Large : llvm::CodeModel::Small,
        llvm::CodeGenOptLevel::Aggressive);

    // Every backend, not only the two that emit machine code. Generating the
    // module reads the layout back -- a struct's field offsets for a gather,
    // what a store needs aligned to -- and without one LLVM answers from its
    // default layout, which aligns a 64-bit integer to four bytes where
    // x86-64 aligns it to eight. The LLVM and JIT backends generated their
    // code against that layout and then ran or printed it for this machine,
    // which put a struct holding an i64 after an i32 at 20 bytes in one
    // place and 24 in the other. What keeps the printed IR the same on every
    // machine is printing
    // it without the layout (see print_module and to_llvm), not generating it
    // without one.
    module.setDataLayout(tm->createDataLayout());
    module.setTargetTriple(target_triple);
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
    // An aggregate the return registers cannot hold comes back through a
    // pointer the caller passes first; see indirect_return_type.
    llvm::Type *sret = indirect_return_type(func.ret_type);
    const uint32_t hidden = sret ? 1 : 0;
    std::vector<llvm::Type *> arg_types(func.args.size() + hidden);
    if (sret) {
        ret_type = llvm::Type::getVoidTy(*context);
        arg_types[0] = llvm::PointerType::getUnqual(*context);
    }
    for (uint32_t i = 0; i < func.args.size(); i++) {
        const auto &arg_info = func.args[i];
        // A struct parameter is declared as the struct. Lower/Mutability.cpp
        // has already made it a pointer where the C ABI applies -- at an
        // exported boundary -- so anything still a struct here is internal and
        // is passed as the value it is.
        arg_types[i + hidden] = codegen_type(arg_info.type);
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

    if (sret) {
        // What the C ABI says of the pointer: it is the return value's home
        // and nothing else's, it is written and not read, and nobody keeps it.
        llvm::AttrBuilder attrs(*context);
        attrs.addStructRetAttr(sret);
        attrs.addAttribute(llvm::Attribute::NoAlias);
        attrs.addAttribute(llvm::Attribute::NoCapture);
        attrs.addAttribute(llvm::Attribute::NoUndef);
        attrs.addAttribute(llvm::Attribute::NonNull);
        attrs.addAttribute(llvm::Attribute::WriteOnly);
        attrs.addAlignmentAttr(module->getDataLayout().getABITypeAlign(sret));
        fn->addParamAttrs(0, attrs);
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

        fn->addParamAttrs(i + hidden, attrs);
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

    // The hidden return pointer, when the aggregate this returns comes back
    // through one, is the first argument and belongs to no parameter.
    current_sret = nullptr;
    if (function->hasParamAttribute(0, llvm::Attribute::StructRet)) {
        current_sret = function->getArg(0);
        current_sret->setName("_sret");
    }

    uint32_t arg_idx = 0;
    for (auto &arg : function->args()) {
        if (&arg == current_sret) {
            continue;
        }
        const auto &arg_info = func.args[arg_idx];
        std::string name = arg_info.name;
        arg.setName(name);
        llvm::Value *arg_value = &arg;

        // A struct argument arrives as a value and stays one; reading a field
        // of it is an extractvalue (see the Access visitor). This used to
        // assert that a struct never got here, because every struct parameter
        // was pointerised -- not for any reason to do with mutability, but
        // because this backend could not take one. What that cost was a stack
        // copy at every call site, and with it the ability to recognise two
        // calls to the same pure function on the same value as one.

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
    current_sret = nullptr;

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
    no_heap = options.no_heap;
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

    // The library information first, so that the default the PassBuilder
    // would register is not what the passes see: this one knows the host's
    // vector math (see target_library_info).
    const llvm::TargetLibraryInfoImpl library_info =
        target_library_info(llvm::Triple(module->getTargetTriple()));
    fam.registerPass([&] { return llvm::TargetLibraryAnalysis(library_info); });

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
        // The machine's own register width: a gang is one register wide
        // (sixteen lanes in a zmm), and with less preferred LLVM would split
        // each of its operations in two. Both attributes, as clang sets them
        // for -mprefer-vector-width: without `min-legal-vector-width` LLVM
        // decides the width from the types it sees rather than from the
        // preference. Only when following the host, so that a named --mcpu,
        // and the golden IR diffed under one, keeps its attributes as they
        // were.
        if (follows_host) {
            const std::string bits = std::to_string(vector_register_bits());
            function.addFnAttr("prefer-vector-width", bits);
            function.addFnAttr("min-legal-vector-width", bits);
        }
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
    // A vector math intrinsic becomes the libmvec call the library
    // information maps it to. The backend would run this pass itself, but
    // here the result is in the module for `-b llvm` to print.
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(
        llvm::ReplaceWithVeclib()));

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

void CodeGen_LLVM::visit(const ElementRef_t *node) {
    internal_error << "A reference to a stored element reached code "
                   << "generation unresolved: " << ir::Type(node)
                   << ". Lower/ElementReferences.cpp lowers these to the "
                   << "index the layout gives them.";
}

void CodeGen_LLVM::visit(const RefTo *node) {
    internal_error << "A reference to a stored element reached code "
                   << "generation unresolved: " << ir::Expr(node)
                   << ". Lower/ElementReferences.cpp lowers these to the "
                   << "index the layout gives them.";
}

void CodeGen_LLVM::visit(const Vector_t *node) {
    llvm::Type *etype = codegen_type(node->etype);
    internal_assert(!etype->isVoidTy())
        << "Cannot make a vector of type void: " << Type(node);
    // Said here, with the type, rather than by LLVM's own assertion, which
    // says only that some element type was wrong.
    internal_assert(node->packed || etype->isIntegerTy() ||
                    etype->isFloatingPointTy() || etype->isPointerTy())
        << "Cannot make a vector of " << node->etype << ": " << Type(node)
        << " -- a vector's elements must be scalars";
    if (node->packed) {
        // Storage: exactly `lanes` elements, aligned as one element is. An
        // LLVM vector of three floats is allocated sixteen bytes and aligned
        // to sixteen; an array of three is twelve and aligned to four, which
        // is what a layout that says `vec3f` has promised. See Vector_t.
        type = llvm::ArrayType::get(etype, node->lanes);
        return;
    }
    // TODO: do we ever want to support scalable vectors? probably not.
    type = llvm::VectorType::get(etype, node->lanes, /* Scalable */ false);
}

// Packed storage to the vector the program computes with, and back: the
// elements of an `[N x T]` aggregate into an `<N x T>`, or out of one.
llvm::Value *CodeGen_LLVM::unpack_vector(llvm::Value *packed,
                                         const Vector_t *type) {
    llvm::Type *etype = codegen_type(type->etype);
    llvm::Type *vector_type =
        llvm::VectorType::get(etype, type->lanes, /* Scalable */ false);
    llvm::Value *vector = llvm::PoisonValue::get(vector_type);
    for (uint32_t i = 0; i < type->lanes; i++) {
        llvm::Value *element = builder->CreateExtractValue(packed, i);
        vector = builder->CreateInsertElement(vector, element, uint64_t(i));
    }
    return vector;
}

llvm::Value *CodeGen_LLVM::pack_vector(llvm::Value *vector,
                                       const Vector_t *type) {
    llvm::Type *etype = codegen_type(type->etype);
    llvm::Type *array_type = llvm::ArrayType::get(etype, type->lanes);
    llvm::Value *packed = llvm::PoisonValue::get(array_type);
    for (uint32_t i = 0; i < type->lanes; i++) {
        llvm::Value *element = builder->CreateExtractElement(vector, uint64_t(i));
        packed = builder->CreateInsertValue(packed, element, i);
    }
    return packed;
}

void CodeGen_LLVM::visit(const Struct_t *node) {
    // TODO: could just use module->getTypeByName
    auto it = struct_types.find(node->name);
    if (it != struct_types.end()) {
        type = it->second;
        return;
    }
    // A struct type not gathered up front: one a pass built after
    // `gather_struct_types` ran, such as a vectorized `SamplerState$v8`, whose
    // fields are the widened forms of the scalar struct's. It is declared on
    // demand, the same two steps `declare_struct_types` takes -- the opaque
    // type into the map before its body is built, so a field that refers back
    // to it or to a sibling widened struct terminates rather than recurses.
    llvm::StructType *st =
        llvm::StructType::create(*context, "struct." + node->name);
    struct_types[node->name] = st;
    std::vector<llvm::Type *> fields;
    fields.reserve(node->fields.size());
    for (const auto &field : node->fields) {
        if (!field.type.is<Ref_t>()) {
            fields.push_back(codegen_type(field.type));
        }
    }
    st->setBody(fields, node->is_packed());
    type = st;
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
    // The same shape declare_function gives a function of this type, hidden
    // return pointer included, so that a call through a function-typed value
    // agrees with the function it reaches.
    llvm::Type *return_type = codegen_type(node->ret_type);
    if (indirect_return_type(node->ret_type)) {
        return_type = llvm::Type::getVoidTy(*context);
        input_types.push_back(llvm::PointerType::getUnqual(*context));
    }
    for (const ir::Function_t::ArgSig &arg : node->arg_types) {
        input_types.push_back(codegen_type(arg.type));
    }
    return llvm::FunctionType::get(return_type, std::move(input_types),
                                   /*isVariadic=*/false);
}

llvm::Type *CodeGen_LLVM::indirect_return_type(const ir::Type &ret_type) {
    if (!ret_type.defined()) {
        return nullptr;
    }
    llvm::Type *type = codegen_type(ret_type);
    if (!type->isAggregateType()) {
        return nullptr;
    }
    // The leaves of the aggregate, as the return convention classes them: a
    // floating-point or vector leaf takes an SSE register, anything else an
    // integer one. x86-64's RetCC gives two of the first kind and three of
    // the second before it reaches for the x87 stack.
    //
    // That rule is x86-64's, and this is the wrong place for it in the long
    // run: it belongs with the other target-specific decisions this backend
    // makes inline -- the native vector width, the host CPU the target
    // machine is built for, which libm calls exist, how a parallel loop is
    // launched -- in a per-target subclass behind virtual methods, as Halide
    // splits its CodeGen_LLVM from CodeGen_X86 and the rest. Until then, a
    // second target would need its own count here.
    struct Leaves {
        unsigned sse = 0;
        unsigned integer = 0;
    };
    const std::function<void(llvm::Type *, Leaves &)> count =
        [&](llvm::Type *t, Leaves &leaves) {
            if (auto *st = llvm::dyn_cast<llvm::StructType>(t)) {
                for (llvm::Type *element : st->elements()) {
                    count(element, leaves);
                }
            } else if (auto *at = llvm::dyn_cast<llvm::ArrayType>(t)) {
                for (uint64_t i = 0; i < at->getNumElements(); i++) {
                    count(at->getElementType(), leaves);
                }
            } else if (t->isFloatingPointTy() || t->isVectorTy()) {
                leaves.sse++;
            } else {
                leaves.integer++;
            }
        };
    Leaves leaves;
    count(type, leaves);
    if (leaves.sse <= 2 && leaves.integer <= 3) {
        return nullptr;
    }
    return type;
}

llvm::AllocaInst *CodeGen_LLVM::create_entry_alloca(llvm::Type *type,
                                                    const std::string &name) {
    internal_assert(current_function);
    llvm::BasicBlock &entry = current_function->getEntryBlock();
    llvm::IRBuilder<> at_entry(&entry, entry.begin());
    return at_entry.CreateAlloca(type, nullptr, name);
}

llvm::Value *CodeGen_LLVM::emit_call(llvm::FunctionCallee callee,
                                     llvm::Type *sret_type,
                                     std::vector<llvm::Value *> args,
                                     const std::string &name) {
    if (sret_type == nullptr) {
        return builder->CreateCall(callee, args, name);
    }
    llvm::AllocaInst *slot =
        create_entry_alloca(sret_type, name.empty() ? "_ret" : name + "_ret");
    args.insert(args.begin(), slot);
    llvm::CallInst *call = builder->CreateCall(callee, args);
    // The call site carries the attribute as well as the declaration: the
    // backend reads the convention off whichever it is looking at.
    call->addParamAttr(0,
                       llvm::Attribute::getWithStructRetType(*context, sret_type));
    return builder->CreateAlignedLoad(sret_type, slot, slot->getAlign(), name);
}

llvm::Value *CodeGen_LLVM::emit_call(llvm::Function *callee,
                                     std::vector<llvm::Value *> args,
                                     const std::string &name) {
    llvm::Type *sret_type =
        callee->hasParamAttribute(0, llvm::Attribute::StructRet)
            ? callee->getParamStructRetType(0)
            : nullptr;
    return emit_call(llvm::FunctionCallee(callee), sret_type, std::move(args),
                     name);
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
    internal_assert(frame_value.has_value())
        << node->name << " is not bound"
        << (current_function
                ? " in " + current_function->getName().str()
                : std::string());
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
        // Bools and integers alike: LLVM's not is xor with all ones, which is
        // the complement at whatever width the operand has. The type rule in
        // UnOp::make already admits both, so refusing integers here was the
        // one place that did not.
        internal_assert(node->type.is_bool() || node->type.is_int_or_uint())
            << "Cannot complement: " << node->type;
        value = builder->CreateNot(a);
        return;
    }
    }

    internal_error << "Cannot codegen UnOp: " << Expr(node);
}

namespace {

int vector_lanes(llvm::Type *t) {
    return int(llvm::cast<llvm::FixedVectorType>(t)->getNumElements());
}

} // namespace

llvm::Value *CodeGen_LLVM::create_select(llvm::Value *cond, llvm::Value *tvalue,
                                         llvm::Value *fvalue) {
    // A per-lane choice between two aggregates -- the blend of a gang's struct
    // at a join, where the struct is one of gang-wide fields (see widen() in
    // SSA/Vectorize.cpp) -- is a choice field by field: LLVM's select takes a
    // vector condition only over vector operands of the same width.
    if (cond->getType()->isVectorTy() && tvalue->getType()->isAggregateType()) {
        llvm::Type *t = tvalue->getType();
        llvm::Value *result = llvm::UndefValue::get(t);
        const unsigned n = t->isStructTy()
                               ? t->getStructNumElements()
                               : unsigned(t->getArrayNumElements());
        for (unsigned i = 0; i < n; i++) {
            result = builder->CreateInsertValue(
                result,
                create_select(cond, builder->CreateExtractValue(tvalue, i),
                              builder->CreateExtractValue(fvalue, i)),
                i);
        }
        return result;
    }
    // TODO: try Vector Predication Intrinsics!
    // https://llvm.org/docs/LangRef.html#vector-predication-intrinsics
    // https://llvm.org/docs/LangRef.html#llvm-vp-select-intrinsics
    return builder->CreateSelect(cond, tvalue, fvalue);
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
    value = create_select(cond, tvalue, fvalue);
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

void CodeGen_LLVM::visit(const MultiRecurse *node) {
    internal_assert(node->keys.empty())
        << "A sorted recursion reached LLVM code generation still carrying its "
           "keys, so nothing ever put its calls in order. The reordering is an "
           "SSA rewrite; see SSA/SortRecursion.h.";
    // The run becomes its calls here, in whatever order the schedule left it.
    for (size_t i = 0; i < node->varying.size(); i++) {
        Call::make(node->func, node->call_args(i)).accept(this);
    }
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
    ir::Type src = node->value.type();
    ir::Type dst = node->type;

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

    // An aggregate read as the packed words of its storage, or the reverse.
    // Through memory, as any reinterpretation of an aggregate is.
    if (node->mode == Cast::Mode::Reinterpret) {
        const auto *packed_src = src.as<Vector_t>();
        const auto *packed_dst = dst.as<Vector_t>();
        if ((src.is<Struct_t>() && packed_dst != nullptr &&
             packed_dst->packed) ||
            (dst.is<Struct_t>() && packed_src != nullptr &&
             packed_src->packed)) {
            value = reinterpret_via_memory(inner, codegen_type(dst));
            return;
        }
    }

    // A vector's packed storage is unpacked before anything below looks at
    // it, and a packed destination is packed again after; in between a vector
    // is the vector the machine computes with. A conversion between the two
    // kinds of the same vector is then the identity in the middle.
    if (const auto *packed = src.as<Vector_t>();
        packed != nullptr && packed->packed) {
        inner = unpack_vector(inner, packed);
        src = Vector_t::make(packed->etype, packed->lanes);
    }
    const Vector_t *pack_to = nullptr;
    if (const auto *packed = dst.as<Vector_t>();
        packed != nullptr && packed->packed) {
        pack_to = packed;
        dst = Vector_t::make(packed->etype, packed->lanes);
    }

    llvm::Type *llvm_dst = codegen_type(dst);

    if (equals(src, dst)) {
        value = pack_to ? pack_vector(inner, pack_to) : inner;
        return;
    }
    internal_assert(pack_to == nullptr)
        << "A conversion into packed storage from a different type: " << src
        << " -> " << Type(pack_to);

    // An array is the address of its elements, so viewing one as an array of
    // a different element type is nothing at the machine level -- only the
    // stride of later indexing changes. Vectorization does this to read an
    // array of per-lane vectors component by component.
    if (src.is_reference() && dst.is_reference()) {
        value = inner;
        return;
    }

    // An address as an integer, or an integer as an address: the zero a
    // pointer-typed slot starts with is the integer zero read as a pointer
    // (see zero_value in SSA/SSA.cpp).
    const auto addresses_memory = [](const Type &t) {
        return t.is<Ptr_t>() || t.is_reference();
    };
    if (node->mode == Cast::Mode::Reinterpret && addresses_memory(dst) &&
        src.is_int_or_uint()) {
        value = builder->CreateIntToPtr(inner, llvm_dst);
        return;
    }
    if (node->mode == Cast::Mode::Reinterpret && addresses_memory(src) &&
        dst.is_int_or_uint()) {
        value = builder->CreatePtrToInt(inner, llvm_dst);
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

        // One aggregate per lane read at another type -- a tree's node,
        // gathered as a struct of gang-wide fields, read as the struct of the
        // arm its tag says it is. Per lane it is that lane's bytes at the
        // other type: each lane's bytes spread into unit vectors as the
        // source lays them out (see scatter_units), and gathered back as the
        // destination does.
        const std::optional<uint32_t> src_lanes = widened_lanes(src);
        const std::optional<uint32_t> dst_lanes = widened_lanes(dst);
        if (src_lanes.has_value() || dst_lanes.has_value()) {
            internal_assert(src_lanes.has_value() && dst_lanes.has_value() &&
                            *src_lanes == *dst_lanes)
                << "Cannot reinterpret " << src << " as " << dst
                << ": one is per lane and the other is not";
            const uint32_t lanes = *src_lanes;
            const Type from = narrow(src, lanes);
            const Type to = narrow(dst, lanes);
            const uint64_t from_size =
                dl.getTypeAllocSize(codegen_type(from)).getFixedValue();
            const uint64_t to_size =
                dl.getTypeAllocSize(codegen_type(to)).getFixedValue();
            const uint64_t size = std::max(from_size, to_size);
            const uint64_t unit = std::min<uint64_t>(
                common_unit(to, 0, common_unit(from, 0, size)), 8);
            llvm::Type *unit_t =
                llvm::Type::getIntNTy(*context, unsigned(unit * 8));
            std::vector<llvm::Value *> slots(size_t(size / unit), nullptr);
            scatter_units(from, inner, 0, unit, slots);
            for (llvm::Value *&slot : slots) {
                if (slot == nullptr) {
                    slot = llvm::PoisonValue::get(
                        llvm::FixedVectorType::get(unit_t, lanes));
                }
            }
            value = gather_units(to, dst, 0, unit, slots, lanes);
            return;
        }

        internal_assert(dl.getTypeAllocSize(llvm_dst) ==
                        dl.getTypeAllocSize(llvm_src))
            << "Cannot reinterpret " << src << " as " << dst
            << ": they are not the same size";
        // An aggregate has no bitcast; its bytes are read at the other type
        // through memory, which is what a reinterpretation is.
        if (llvm_src->isAggregateType() || llvm_dst->isAggregateType()) {
            value = reinterpret_via_memory(inner, llvm_dst);
            return;
        }
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
    } else if (src.is_bool() && dst.is_int_or_uint()) {
        value = builder->CreateIntCast(inner, llvm_dst,
                                       /* isSigned */ false);
    } else if (src.is_bool() && dst.is_float()) {
        // A bool is an i1, so a *signed* widening would make true into -1.
        // Unsigned is the only reading that gives one and zero, which is what
        // `cast[[Float]](b)` means everywhere else in the language.
        //
        // Reachable because the simplifier folds `select(c, 1.0, 0.0)` into
        // exactly this -- which is how it turned up: a `sort()` key written as
        // `select(dir_is_negative, 1.0 - i, i)` becomes that at i = 0, and the
        // whole schedule failed to compile over an arithmetic identity.
        value = builder->CreateUIToFP(inner, llvm_dst);
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

namespace {

int gcd(int a, int b) {
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int largest_power_of_two_factor(int x) {
    int f = 1;
    while (x % (f * 2) == 0) {
        f *= 2;
    }
    return f;
}

} // namespace

llvm::Value *CodeGen_LLVM::as_vector(llvm::Value *v) {
    if (v->getType()->isVectorTy()) {
        return v;
    }
    llvm::Type *vt = llvm::FixedVectorType::get(v->getType(), 1);
    return builder->CreateInsertElement(llvm::PoisonValue::get(vt), v,
                                        uint64_t(0));
}

llvm::Value *CodeGen_LLVM::shuffle_vectors(llvm::Value *a,
                                           const std::vector<int> &indices) {
    return builder->CreateShuffleVector(a, indices);
}

llvm::Value *CodeGen_LLVM::shuffle_vectors(llvm::Value *a, llvm::Value *b,
                                           const std::vector<int> &indices) {
    internal_assert(a->getType()->getScalarType() ==
                    b->getType()->getScalarType())
        << "shuffle of vectors of different element types";
    // LLVM shuffles two vectors of one width. The narrower is padded with
    // lanes that do not matter, and the indices into the second renumbered
    // to where it then starts.
    const int na = vector_lanes(a->getType());
    const int nb = vector_lanes(b->getType());
    if (na != nb) {
        const int n = std::max(na, nb);
        std::vector<int> renumbered = indices;
        for (int &i : renumbered) {
            if (i >= na) {
                i += n - na;
            }
        }
        return shuffle_vectors(slice_vector(a, 0, n), slice_vector(b, 0, n),
                               renumbered);
    }
    return builder->CreateShuffleVector(a, b, indices);
}

llvm::Value *CodeGen_LLVM::slice_vector(llvm::Value *v, int start, int size) {
    const int n = vector_lanes(v->getType());
    if (start == 0 && size == n) {
        return v;
    }
    std::vector<int> indices(static_cast<size_t>(size));
    for (int i = 0; i < size; i++) {
        const int k = start + i;
        indices[size_t(i)] = k < n ? k : -1;
    }
    return shuffle_vectors(v, indices);
}

llvm::Value *CodeGen_LLVM::concat_vectors(const std::vector<llvm::Value *> &vs) {
    internal_assert(!vs.empty()) << "concat of no vectors";
    std::vector<llvm::Value *> v = vs;
    // Pairwise, so that the tree of shuffles is as shallow as it can be.
    while (v.size() > 1) {
        std::vector<llvm::Value *> next;
        next.reserve((v.size() + 1) / 2);
        for (size_t i = 0; i < v.size(); i += 2) {
            if (i + 1 == v.size()) {
                next.push_back(v[i]);
                continue;
            }
            const int na = vector_lanes(v[i]->getType());
            const int nb = vector_lanes(v[i + 1]->getType());
            std::vector<int> indices(static_cast<size_t>(na + nb));
            for (int k = 0; k < na + nb; k++) {
                indices[size_t(k)] = k;
            }
            next.push_back(shuffle_vectors(v[i], v[i + 1], indices));
        }
        v.swap(next);
    }
    return v[0];
}

llvm::Value *CodeGen_LLVM::optimization_fence(llvm::Value *v) {
    llvm::Type *t = v->getType();
    const uint64_t bits = t->getPrimitiveSizeInBits().getFixedValue();
    if (bits == 0 || bits % 32 != 0) {
        return v;
    }
    // A constant -- the lanes that do not matter in a padded interleave --
    // has nothing to protect, and fencing it would only keep it from folding.
    if (llvm::isa<llvm::Constant>(v)) {
        return v;
    }
    // The arithmetic fence is defined on floating-point values; the bits are
    // read as floats across it and back, which costs nothing.
    llvm::Type *as_floats = llvm::FixedVectorType::get(f32_t, unsigned(bits / 32));
    llvm::Value *fenced = builder->CreateBitCast(v, as_floats);
    fenced = builder->CreateArithmeticFence(fenced, as_floats);
    return builder->CreateBitCast(fenced, t);
}

// a0 b0 c0 a1 b1 c1 .., following Halide's CodeGen_LLVM::interleave_vectors.
//
// Two vectors are one shuffle. A count coprime to the width is Catanzaro,
// Keller and Garland's decomposition of the transpose (PPoPP 2014, section
// 6.2): each vector is permuted on its own (a static row permutation), the
// vectors are relabelled, and the columns are rotated by lane-dependent
// amounts -- done as a barrel rotation, log2(count) steps of two-input blends
// -- after which the vectors concatenated are the interleave. Otherwise the
// count is factored: the vectors are dealt into `f` groups, each group
// interleaved, and the results interleaved, with a fence between the stages
// so LLVM does not fuse the shuffles back into the poor sequence its own
// lowering would have picked.
llvm::Value *CodeGen_LLVM::interleave_vectors(const std::vector<llvm::Value *> &vecs) {
    internal_assert(!vecs.empty()) << "interleave of no vectors";
    const int num_vecs = int(vecs.size());
    if (num_vecs == 1) {
        return vecs[0];
    }
    llvm::Type *vec_type = vecs[0]->getType();
    for (llvm::Value *v : vecs) {
        internal_assert(v->getType() == vec_type)
            << "interleave of vectors of different types";
    }
    const int vec_elements = vector_lanes(vec_type);
    const int factor = gcd(vec_elements, num_vecs);

    if (num_vecs == 2) {
        std::vector<int> indices(size_t(vec_elements) * 2);
        for (int i = 0; i < vec_elements * 2; i++) {
            indices[size_t(i)] = i % 2 == 0 ? i / 2 : i / 2 + vec_elements;
        }
        return shuffle_vectors(vecs[0], vecs[1], indices);
    }

    if (factor == 1) {
        std::vector<llvm::Value *> v = vecs;

        // The row permutation: element j of vector i is bound for lane
        // j * num_vecs + i of the result, which is lane (j * num_vecs + i) %
        // vec_elements of its own vector after the rotation below; put it
        // there now.
        std::vector<int> shuffle(static_cast<size_t>(vec_elements));
        for (int i = 0; i < num_vecs; i++) {
            for (int j = 0; j < vec_elements; j++) {
                const int k = j * num_vecs + i;
                shuffle[size_t(k % vec_elements)] = j;
            }
            v[size_t(i)] = shuffle_vectors(v[size_t(i)], shuffle);
        }

        // The relabelling of the vectors.
        std::vector<llvm::Value *> new_v(v.size());
        for (int i = 0; i < num_vecs; i++) {
            const int j = (i * vec_elements) % num_vecs;
            new_v[size_t(i)] = v[size_t(j)];
        }
        v.swap(new_v);

        // How far each lane's column has to rotate.
        std::vector<int> rotation(size_t(vec_elements), 0);
        for (int i = 0; i < vec_elements; i++) {
            const int k = (i * num_vecs) % vec_elements;
            rotation[size_t(k)] = (i * num_vecs) / vec_elements;
        }
        internal_assert(rotation[0] == 0);

        // The barrel rotation: at step d, every lane whose rotation has bit d
        // set takes its value from the vector d further round.
        int d = 1;
        while (d < num_vecs) {
            for (int i = 0; i < vec_elements; i++) {
                shuffle[size_t(i)] =
                    (rotation[size_t(i)] & d) == 0 ? i : i + vec_elements;
            }
            for (int i = 0; i < num_vecs; i++) {
                const int j = (i + num_vecs - d) % num_vecs;
                new_v[size_t(i)] =
                    shuffle_vectors(v[size_t(i)], v[size_t(j)], shuffle);
            }
            v.swap(new_v);
            d *= 2;
        }
        return concat_vectors(v);
    }

    // Factor the count: the largest power of two, or failing that the
    // smallest divisor.
    int f = largest_power_of_two_factor(num_vecs);
    if (f == 1 || f == num_vecs) {
        for (int i = 2; i < num_vecs; i++) {
            if (num_vecs % i == 0) {
                f = i;
                break;
            }
        }
    }
    if (f == 1) {
        // A prime count that shares a factor with the width: pad the width
        // to a power of two, which the count is then coprime to, and drop
        // the padding lanes from the result.
        const int padded_size = next_power_of_two(vec_elements);
        std::vector<llvm::Value *> padded(vecs.size());
        for (int i = 0; i < num_vecs; i++) {
            padded[size_t(i)] = slice_vector(vecs[size_t(i)], 0, padded_size);
        }
        llvm::Value *v = interleave_vectors(padded);
        return slice_vector(v, 0, num_vecs * vec_elements);
    }
    internal_assert(f > 1 && f < num_vecs && num_vecs % f == 0)
        << f << " " << num_vecs << " " << factor;

    std::vector<std::vector<llvm::Value *>> groups(static_cast<size_t>(f));
    for (int i = 0; i < num_vecs; i++) {
        groups[size_t(i % f)].push_back(vecs[size_t(i)]);
    }
    std::vector<llvm::Value *> interleaved(static_cast<size_t>(f));
    for (int i = 0; i < f; i++) {
        interleaved[size_t(i)] =
            optimization_fence(interleave_vectors(groups[size_t(i)]));
    }
    return interleave_vectors(interleaved);
}

// The inverse of interleave_vectors, following Halide's deinterleave_vector:
// the same three cases, each run backwards.
std::vector<llvm::Value *> CodeGen_LLVM::deinterleave_vector(llvm::Value *vec,
                                                             int num_vecs) {
    int vec_elements = vector_lanes(vec->getType());
    internal_assert(num_vecs > 0 && vec_elements % num_vecs == 0)
        << "deinterleave of " << vec_elements << " lanes into " << num_vecs;
    vec_elements /= num_vecs;
    const int factor = gcd(vec_elements, num_vecs);

    if (num_vecs == 1) {
        return {vec};
    }
    if (num_vecs == 2) {
        std::vector<llvm::Value *> result(2);
        std::vector<int> indices(static_cast<size_t>(vec_elements));
        for (int i = 0; i < vec_elements; i++) {
            indices[size_t(i)] = i * 2;
        }
        result[0] = shuffle_vectors(vec, indices);
        for (int i = 0; i < vec_elements; i++) {
            indices[size_t(i)]++;
        }
        result[1] = shuffle_vectors(vec, indices);
        return result;
    }

    if (factor == 1) {
        std::vector<llvm::Value *> v(static_cast<size_t>(num_vecs));
        for (int i = 0; i < num_vecs; i++) {
            v[size_t(i)] = slice_vector(vec, i * vec_elements, vec_elements);
        }

        std::vector<int> rotation(size_t(vec_elements), 0);
        for (int i = 0; i < vec_elements; i++) {
            const int k = (i * num_vecs) % vec_elements;
            rotation[size_t(k)] = (i * num_vecs) / vec_elements;
        }
        internal_assert(rotation[0] == 0);

        // The barrel rotation, the other way round.
        std::vector<int> shuffle(static_cast<size_t>(vec_elements));
        std::vector<llvm::Value *> new_v(v.size());
        int d = 1;
        while (d < num_vecs) {
            for (int i = 0; i < vec_elements; i++) {
                shuffle[size_t(i)] =
                    (rotation[size_t(i)] & d) == 0 ? i : i + vec_elements;
            }
            for (int i = 0; i < num_vecs; i++) {
                const int j = (i + d) % num_vecs;
                new_v[size_t(i)] = optimization_fence(
                    shuffle_vectors(v[size_t(i)], v[size_t(j)], shuffle));
            }
            v.swap(new_v);
            d *= 2;
        }

        // The relabelling, inverted.
        for (int i = 0; i < num_vecs; i++) {
            const int j = (i * vec_elements) % num_vecs;
            new_v[size_t(j)] = v[size_t(i)];
        }
        v.swap(new_v);

        // The row permutation, inverted.
        for (int i = 0; i < num_vecs; i++) {
            for (int j = 0; j < vec_elements; j++) {
                const int k = j * num_vecs + i;
                shuffle[size_t(j)] = k % vec_elements;
            }
            v[size_t(i)] = shuffle_vectors(v[size_t(i)], shuffle);
        }
        return v;
    }

    int f = largest_power_of_two_factor(num_vecs);
    if (f == 1 || f == num_vecs) {
        for (int i = 2; i < num_vecs; i++) {
            if (num_vecs % i == 0) {
                f = i;
                break;
            }
        }
    }
    if (f == 1) {
        const int padded_size = next_power_of_two(vec_elements);
        llvm::Value *padded = slice_vector(vec, 0, padded_size * num_vecs);
        std::vector<llvm::Value *> result = deinterleave_vector(padded, num_vecs);
        for (int i = 0; i < num_vecs; i++) {
            result[size_t(i)] = slice_vector(result[size_t(i)], 0, vec_elements);
        }
        return result;
    }
    internal_assert(f > 1 && f < num_vecs && num_vecs % f == 0)
        << f << " " << num_vecs << " " << factor;

    // Deal by f first, then each part by the rest: part i holds the streams
    // i, i + f, i + 2f, .. and dealing it by num_vecs / f separates them.
    const std::vector<llvm::Value *> partial = deinterleave_vector(vec, f);
    std::vector<llvm::Value *> result(static_cast<size_t>(num_vecs));
    for (size_t i = 0; i < partial.size(); i++) {
        const std::vector<llvm::Value *> parts =
            deinterleave_vector(partial[i], num_vecs / f);
        for (size_t j = 0; j < parts.size(); j++) {
            result[j * size_t(f) + i] = parts[j];
        }
    }
    return result;
}

void CodeGen_LLVM::visit(const Shuffle *node) {
    std::vector<llvm::Value *> vecs;
    vecs.reserve(node->vectors.size());
    for (const Expr &v : node->vectors) {
        vecs.push_back(as_vector(codegen_expr(v)));
    }
    llvm::Value *result = nullptr;
    if (node->is_interleave()) {
        result = interleave_vectors(vecs);
    } else if (node->is_transpose()) {
        // A transpose either way round: as Halide does, take whichever of
        // the two algorithms sees the power-of-two shape.
        const int cols = node->transpose_factor();
        const int rows = vector_lanes(vecs[0]->getType()) / cols;
        if (is_power_of_two(cols) && !is_power_of_two(rows)) {
            std::vector<llvm::Value *> slices(static_cast<size_t>(rows));
            for (int i = 0; i < rows; i++) {
                slices[size_t(i)] = slice_vector(vecs[0], i * cols, cols);
            }
            result = interleave_vectors(slices);
        } else {
            result = concat_vectors(deinterleave_vector(vecs[0], cols));
        }
    } else if (node->is_concat()) {
        result = concat_vectors(vecs);
    } else if (vecs.size() == 1) {
        result = shuffle_vectors(vecs[0], node->indices);
    } else if (vecs.size() == 2) {
        result = shuffle_vectors(vecs[0], vecs[1], node->indices);
    } else {
        result = shuffle_vectors(concat_vectors(vecs), node->indices);
    }
    if (!node->type.is_vector()) {
        result = builder->CreateExtractElement(result, uint64_t(0));
    }
    value = result;
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
        const Type element = vec_expr.type().element_of();
        const uint32_t lanes = node->idx.type().lanes();
        // An aggregate per lane -- a tree's node, a light, a primitive, at
        // each lane's own index -- is gathered a leaf at a time into the
        // struct of gang-wide fields it is carried as (see gather_elements).
        if (element.is<Struct_t, Vector_t>()) {
            // Each lane's byte offset to its element, in 32 bits (see
            // element_addresses): what every field's gather is the base
            // plus. An index wider than that is taken in 32 bits, which is
            // the addressing model's limit.
            llvm::Value *idx = codegen_expr(node->idx);
            llvm::Type *offsets_t = llvm::FixedVectorType::get(i32_t, lanes);
            if (idx->getType() != offsets_t) {
                idx = builder->CreateIntCast(idx, offsets_t, /*isSigned=*/true);
            }
            const llvm::DataLayout &dl = module->getDataLayout();
            llvm::Type *elem_llvm = codegen_type(element);
            const GatheredElement whole{
                dl.getTypeAllocSize(elem_llvm).getFixedValue(),
                dl.getABITypeAlign(elem_llvm).value()};
            llvm::Value *offsets = builder->CreateMul(
                idx, llvm::ConstantInt::get(offsets_t, whole.bytes),
                "extract_off");
            llvm::Value *mask =
                node->mask.defined() ? codegen_expr(node->mask) : nullptr;
            value = gather_elements(element, node->type, vec, offsets, 0, whole,
                                    lanes, mask, "extract");
            return;
        }
        // How long the array is, when that is known: a fixed array says so
        // in its type, a dynamic one carries it beside its buffer. What lets
        // a sub-word element be read as a whole word (see create_vector_load).
        // A type's size may name a variable this function does not have --
        // a layout's count, which the layout knows and its reader does not
        // -- so only a constant, or a variable that is bound here, counts.
        llvm::Value *length = nullptr;
        if (is_dynamic_array_struct_type(node->vec.type())) {
            length = codegen_expr(Access::make("size", node->vec));
        } else if (const Array_t *array = vec_expr.type().as<Array_t>();
                   array != nullptr && array->size.defined()) {
            const Var *named = array->size.as<Var>();
            if (as_const_int(array->size) != nullptr ||
                (named != nullptr && frames.contains(named->name))) {
                length = codegen_expr(array->size);
            }
        }
        // Unmasked, every lane of a gang stands for a real iteration and
        // every lane's address is one the program would have read anyway;
        // masked, the disabled lanes do not touch memory (see Extract::mask).
        value = create_vector_load(codegen_type(element), vec, node->idx,
                                   lanes, node->mask, "extract", length);
        return;
    }

    llvm::Value *idx = codegen_expr(node->idx);
    if (const Vector_t *v = vec_expr.type().as<Vector_t>();
        v != nullptr && v->packed) {
        // A packed vector's storage is an LLVM array (see the Vector_t
        // visitor): its element is an extractvalue at a constant index, and
        // an extractelement of the unpacked vector at any other.
        if (const auto *k = llvm::dyn_cast<llvm::ConstantInt>(idx)) {
            value = builder->CreateExtractValue(vec, unsigned(k->getZExtValue()));
        } else {
            value = builder->CreateExtractElement(unpack_vector(vec, v), idx);
        }
    } else if (vec_expr.type().is<Vector_t>()) {
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

llvm::Value *CodeGen_LLVM::gather_elements(const ir::Type &element,
                                           const ir::Type &wide_t,
                                           llvm::Value *base,
                                           llvm::Value *offsets, uint64_t disp,
                                           const GatheredElement &whole,
                                           uint32_t lanes, llvm::Value *mask,
                                           const std::string &name) {
    // Every field's address is the one base, plus each lane's 32-bit byte
    // offset to its element, plus the field's own offset within the element
    // -- accumulated in `disp` down the recursion and added to the offsets
    // only at the leaf, so that the gather is `base + sext(offsets) + disp`:
    // one instruction with a 256-bit index register and a displacement. A
    // vector of pointers offset once per field, which this used to form,
    // LLVM could not fold back into that, and gathered through eight 64-bit
    // addresses instead (see element_addresses for the addressing model).
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Type *elem_llvm = codegen_type(element);
    llvm::Type *wide_llvm = codegen_type(wide_t);

    if (const Struct_t *s = element.as<Struct_t>()) {
        const Struct_t *ws = wide_t.as<Struct_t>();
        internal_assert(ws && ws->fields.size() == s->fields.size())
            << wide_t << " is not " << element << " widened";
        const llvm::StructLayout *layout =
            dl.getStructLayout(llvm::cast<llvm::StructType>(elem_llvm));
        llvm::Value *result = llvm::PoisonValue::get(wide_llvm);
        // The LLVM struct leaves out reference-typed fields (see the Struct_t
        // visitor), so its element numbers run ahead of the IR's.
        unsigned slot = 0;
        for (size_t i = 0; i < s->fields.size(); i++) {
            if (s->fields[i].type.is<Ref_t>()) {
                continue;
            }
            llvm::Value *field = gather_elements(
                s->fields[i].type, ws->fields[i].type, base, offsets,
                disp + layout->getElementOffset(slot), whole, lanes, mask,
                name + "_" + s->fields[i].name);
            result = builder->CreateInsertValue(result, field, unsigned(i));
            slot++;
        }
        return result;
    }
    if (const Vector_t *v = element.as<Vector_t>()) {
        // Widened as one gang vector per component (see ir::widen): each
        // component is a gather of scalars, at a stride of the vector.
        const Struct_t *ws = wide_t.as<Struct_t>();
        internal_assert(ws && ws->fields.size() == v->lanes)
            << wide_t << " is not " << element << " widened";
        const uint64_t esize =
            dl.getTypeAllocSize(codegen_type(v->etype)).getFixedValue();
        llvm::Value *result = llvm::PoisonValue::get(wide_llvm);
        for (uint32_t c = 0; c < v->lanes; c++) {
            llvm::Value *component = gather_elements(
                v->etype, ws->fields[c].type, base, offsets, disp + c * esize,
                whole, lanes, mask, name + "_" + std::to_string(c));
            result = builder->CreateInsertValue(result, component, c);
        }
        return result;
    }

    // A scalar. A boolean is a byte in memory (see create_vector_load).
    if (elem_llvm->isIntegerTy(1)) {
        llvm::Value *bytes = gather_elements(
            UInt_t::make(8), Vector_t::make(UInt_t::make(8), lanes), base,
            offsets, disp, whole, lanes, mask, name);
        return builder->CreateTrunc(
            bytes, llvm::FixedVectorType::get(i1_t, lanes), name + "_bits");
    }
    const uint64_t size = dl.getTypeStoreSize(elem_llvm).getFixedValue();
    if (size < 4 && whole.bytes >= 4) {
        // No machine gathers bytes or halfwords, so LLVM takes a gather of
        // them apart into eight loads behind eight branches, one per lane,
        // and the branches follow the mask, which the predictor cannot
        // learn. A tree node's tag, its axis and its primitive count, and
        // every stored bool, went that way; the three of them together were
        // a fifth of the vectorized path step. The field is read instead as
        // part of the aligned word that holds it, which lies inside the
        // element -- pulled back to the element's last word when the
        // element's size is not a multiple of four -- and shifted down. One
        // gather.
        const uint64_t word = std::min(disp & ~uint64_t(3), whole.bytes - 4);
        const uint64_t shift = (disp - word) * 8;
        llvm::Value *words = gather_words(i32_t, base, offsets, word,
                                          whole.align, lanes, mask,
                                          name + "_word");
        if (shift != 0) {
            words = builder->CreateLShr(
                words, llvm::ConstantInt::get(words->getType(), shift));
        }
        return builder->CreateTrunc(
            words, llvm::FixedVectorType::get(elem_llvm, lanes), name);
    }
    return gather_words(elem_llvm, base, offsets, disp, whole.align, lanes,
                        mask, name);
}

llvm::Value *CodeGen_LLVM::gather_words(llvm::Type *elem_llvm,
                                        llvm::Value *base,
                                        llvm::Value *offsets, uint64_t disp,
                                        uint64_t align, uint32_t lanes,
                                        llvm::Value *mask,
                                        const std::string &name) {
    return gather_indexed(elem_llvm, base, offsets, 1, disp, align, lanes, mask,
                          name);
}

llvm::Value *CodeGen_LLVM::gather_indexed(llvm::Type *elem_llvm,
                                          llvm::Value *base,
                                          llvm::Value *indices, uint64_t scale,
                                          uint64_t disp, uint64_t align,
                                          uint32_t lanes, llvm::Value *mask,
                                          const std::string &name) {
    // The addresses as LLVM likes to see them formed: the base displaced
    // once, then one GEP at the scale, which its x86 lowering folds into a
    // gather with a base register and a 32-bit index register when the GEP
    // is still in sight of the gather (see element_addresses for the
    // addressing model, and CodeGen_X86 for when it is not).
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Value *at = base;
    if (disp != 0) {
        at = builder->CreateInBoundsGEP(
            i8_t, base, llvm::ConstantInt::get(i64_t, disp), name + "_base");
    }
    llvm::Value *ptrs;
    if (scale == dl.getTypeAllocSize(elem_llvm).getFixedValue()) {
        // An index into an array of the element: spelled over the element,
        // which is what it is.
        ptrs = builder->CreateInBoundsGEP(elem_llvm, at, indices, name + "_ptrs");
    } else if (scale == 1 || scale == 2 || scale == 4 || scale == 8) {
        ptrs = builder->CreateInBoundsGEP(
            llvm::Type::getIntNTy(*context, unsigned(scale) * 8), at, indices,
            name + "_ptrs");
    } else {
        llvm::Value *offsets = builder->CreateMul(
            indices, llvm::ConstantInt::get(indices->getType(), scale),
            name + "_off");
        ptrs = builder->CreateInBoundsGEP(i8_t, at, offsets, name + "_ptrs");
    }
    llvm::Type *vtype = llvm::FixedVectorType::get(elem_llvm, lanes);
    llvm::Value *on = mask != nullptr
                          ? mask
                          : llvm::Constant::getAllOnesValue(
                                llvm::FixedVectorType::get(i1_t, lanes));
    // No more aligned than the element is: a field of a packed element, or
    // a word read across one, starts wherever the element does.
    const uint64_t claimed =
        std::min<uint64_t>(dl.getABITypeAlign(elem_llvm).value(),
                           std::max<uint64_t>(align, 1));
    return builder->CreateMaskedGather(vtype, ptrs, llvm::Align(claimed), on,
                                       llvm::Constant::getNullValue(vtype),
                                       name);
}

llvm::Value *CodeGen_LLVM::gather_sub_word_elements(
    llvm::Type *etype, llvm::Value *base, llvm::Value *offsets,
    llvm::Value *total_bytes, uint32_t lanes, llvm::Value *mask,
    const std::string &name) {
    // The aligned word holding each lane's element, kept inside the array:
    // an element in the array's last partial word is read as part of the
    // array's last whole word instead, and shifted down by however far it
    // sits into that word. The same reason as the field case in
    // gather_elements; here the bound is the array's, known only at run
    // time, which is why the caller has to have checked there is a word to
    // read at all.
    llvm::Type *offsets_t = offsets->getType();
    llvm::Value *aligned = builder->CreateAnd(
        offsets, llvm::ConstantInt::get(offsets_t, ~uint64_t(3)), name + "_aligned");
    llvm::Value *last = builder->CreateSub(
        builder->CreateVectorSplat(lanes, total_bytes),
        llvm::ConstantInt::get(offsets_t, 4), name + "_last");
    llvm::Value *word_at = builder->CreateBinaryIntrinsic(
        llvm::Intrinsic::umin, aligned, last, nullptr, name + "_word_at");
    // The array's own alignment is its element's, so the word may straddle
    // whatever the element does not.
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Value *words =
        gather_words(i32_t, base, word_at, 0, dl.getABITypeAlign(etype).value(),
                     lanes, mask, name + "_word");
    llvm::Value *shift = builder->CreateShl(
        builder->CreateSub(offsets, word_at), llvm::ConstantInt::get(offsets_t, 3),
        name + "_shift");
    return builder->CreateTrunc(builder->CreateLShr(words, shift),
                                llvm::FixedVectorType::get(etype, lanes), name);
}

llvm::Value *CodeGen_LLVM::multiply_high(llvm::Value *a, llvm::Value *b,
                                         bool is_signed,
                                         const std::string &name) {
    llvm::Type *t = a->getType();
    internal_assert(t == b->getType()) << "mulhi of two different types";
    auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t);
    const unsigned bits = t->getScalarType()->getIntegerBitWidth();

    if (bits < 64 || vt == nullptr) {
        // Widen, multiply, keep the top half. On x86 the scalar 64-bit form
        // is the one-instruction widening multiply (mul or imul r64) and the
        // 32-bit vector form is vpmuludq or vpmuldq, which LLVM recognises
        // from exactly this shape.
        llvm::Type *wide_scalar = llvm::Type::getIntNTy(*context, 2 * bits);
        llvm::Type *wide =
            vt ? static_cast<llvm::Type *>(
                     llvm::FixedVectorType::get(wide_scalar, vt->getNumElements()))
               : wide_scalar;
        llvm::Value *wa = is_signed ? builder->CreateSExt(a, wide)
                                    : builder->CreateZExt(a, wide);
        llvm::Value *wb = is_signed ? builder->CreateSExt(b, wide)
                                    : builder->CreateZExt(b, wide);
        llvm::Value *product = builder->CreateMul(wa, wb, name + "_wide");
        llvm::Value *high = builder->CreateLShr(
            product, llvm::ConstantInt::get(wide, bits), name + "_high");
        return builder->CreateTrunc(high, t, name);
    }

    // A vector of 64-bit lanes. No vector instruction multiplies 64 by 64
    // to 128, and LLVM scalarizes a vector of i128 products into eight
    // scalar ones with the extracts and inserts around them, so the high
    // half is assembled here from the 32-by-32-to-64 products the machine
    // does have (vpmuludq), the way a long multiplication is done by hand:
    // with a = ah*2^32 + al and b = bh*2^32 + bl, the top 64 bits of a*b
    // are ah*bh plus the carries out of the two cross terms and the low
    // product. Each partial sum below fits in 64 bits.
    llvm::Constant *mask32 = llvm::ConstantInt::get(t, 0xffffffffULL);
    llvm::Constant *c32 = llvm::ConstantInt::get(t, 32);
    llvm::Value *al = builder->CreateAnd(a, mask32, name + "_al");
    llvm::Value *ah = builder->CreateLShr(a, c32, name + "_ah");
    llvm::Value *bl = builder->CreateAnd(b, mask32, name + "_bl");
    llvm::Value *bh = builder->CreateLShr(b, c32, name + "_bh");
    llvm::Value *ll = builder->CreateMul(al, bl, name + "_ll");
    llvm::Value *hl = builder->CreateMul(ah, bl, name + "_hl");
    llvm::Value *lh = builder->CreateMul(al, bh, name + "_lh");
    llvm::Value *hh = builder->CreateMul(ah, bh, name + "_hh");
    llvm::Value *mid = builder->CreateAdd(
        builder->CreateAdd(builder->CreateLShr(ll, c32),
                           builder->CreateAnd(hl, mask32)),
        builder->CreateAnd(lh, mask32), name + "_mid");
    llvm::Value *high = builder->CreateAdd(
        builder->CreateAdd(hh, builder->CreateLShr(hl, c32)),
        builder->CreateAdd(builder->CreateLShr(lh, c32),
                           builder->CreateLShr(mid, c32)),
        is_signed ? name + "_unsigned" : name);
    if (!is_signed) {
        return high;
    }
    // The signed high half from the unsigned one: a negative a stands for
    // a - 2^64, whose product with b is a*b - b*2^64, so the high word loses
    // b; likewise a for a negative b.
    llvm::Constant *c63 = llvm::ConstantInt::get(t, 63);
    llvm::Value *sa = builder->CreateAShr(a, c63, name + "_sa");
    llvm::Value *sb = builder->CreateAShr(b, c63, name + "_sb");
    high = builder->CreateSub(high, builder->CreateAnd(sa, b));
    return builder->CreateSub(high, builder->CreateAnd(sb, a), name);
}

llvm::Value *CodeGen_LLVM::division_multiplier(llvm::Value *d, bool is_signed,
                                               const std::string &name) {
    llvm::Type *t = d->getType();
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        // One wide division per lane: what it costs is the division, and no
        // machine has a vector one, so the lanes are taken apart here
        // rather than by LLVM's legalization of a vector of i128.
        llvm::Value *result = llvm::PoisonValue::get(t);
        for (unsigned i = 0; i < vt->getNumElements(); i++) {
            llvm::Value *lane = builder->CreateExtractElement(d, i);
            lane = division_multiplier(lane, is_signed,
                                       name + "_" + std::to_string(i));
            result = builder->CreateInsertElement(result, lane, i);
        }
        return result;
    }

    const unsigned bits = t->getIntegerBitWidth();
    llvm::Type *wide = llvm::Type::getIntNTy(*context, 2 * bits);
    llvm::Constant *zero = llvm::ConstantInt::get(t, 0);
    llvm::Constant *one = llvm::ConstantInt::get(t, 1);
    llvm::Constant *width = llvm::ConstantInt::get(t, bits);
    llvm::Constant *ctlz_zero_is_defined = llvm::ConstantInt::get(i1_t, 0);
    // What is divided by below. A zero divisor divides by one instead: the
    // multiplier for a zero is never used -- the program's own division by
    // it would have trapped -- but this is computed where the divisor is
    // defined, for lanes that never divide too (see ir::Intrinsic).
    auto safe = [&](llvm::Value *v) {
        return builder->CreateSelect(builder->CreateICmpEQ(v, zero), one, v,
                                     name + "_safe");
    };

    if (!is_signed) {
        // Granlund & Montgomery's round-up method: with l = ceil(log2 d),
        // m' = floor(2^N (2^l - d) / d) + 1, which fits N bits since
        // 2^l - d < d. The quotient then is (mulhi(m', n) + (n - mulhi(m',
        // n)) / 2) >> (l - 1), and the caller special-cases d = 1.
        llvm::Value *ds = safe(d);
        llvm::Value *clz = builder->CreateIntrinsic(
            t, llvm::Intrinsic::ctlz,
            {builder->CreateSub(ds, one), ctlz_zero_is_defined});
        llvm::Value *l = builder->CreateSub(width, clz, name + "_l");
        // 2^l as an N-bit number, which for l = N is zero -- and a shift by
        // the width is poison, so that case is taken as the zero it is.
        llvm::Value *two_l = builder->CreateSelect(
            builder->CreateICmpEQ(l, width), zero, builder->CreateShl(one, l));
        llvm::Value *numerator_high =
            builder->CreateSub(two_l, ds, name + "_num");
        llvm::Value *numerator = builder->CreateShl(
            builder->CreateZExt(numerator_high, wide),
            llvm::ConstantInt::get(wide, bits));
        llvm::Value *quotient =
            builder->CreateUDiv(numerator, builder->CreateZExt(ds, wide));
        return builder->CreateAdd(builder->CreateTrunc(quotient, t), one, name);
    }

    // The signed multiplier (Granlund & Montgomery section 5; Hacker's
    // Delight, "magic"): for |d| with l = max(ceil(log2 |d|), 1), m =
    // floor(2^(N+l-1) / |d|) + 1 - 2^N, taken modulo 2^N. The caller adds n
    // to the multiply-high, shifts by l - 1, corrects the sign and negates
    // for a negative d.
    llvm::Value *negative = builder->CreateICmpSLT(d, zero);
    llvm::Value *ad = builder->CreateSelect(negative, builder->CreateNeg(d), d,
                                            name + "_abs");
    llvm::Value *ads = safe(ad);
    llvm::Value *clz = builder->CreateIntrinsic(
        t, llvm::Intrinsic::ctlz,
        {builder->CreateSub(ads, one), ctlz_zero_is_defined});
    llvm::Value *l = builder->CreateSelect(
        builder->CreateICmpEQ(clz, width), one, builder->CreateSub(width, clz),
        name + "_l");
    llvm::Value *shift = builder->CreateAdd(
        builder->CreateZExt(l, wide), llvm::ConstantInt::get(wide, bits - 1));
    llvm::Value *numerator =
        builder->CreateShl(llvm::ConstantInt::get(wide, 1), shift);
    llvm::Value *quotient =
        builder->CreateUDiv(numerator, builder->CreateZExt(ads, wide));
    return builder->CreateAdd(builder->CreateTrunc(quotient, t), one, name);
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
    case Intrinsic::clz: {
        // llvm.ctlz's second argument says whether a zero is poison. It is
        // not: clz of zero is the width, as the intrinsic is defined.
        intrin = llvm::Intrinsic::ctlz;
        add_false_arg = true;
        break;
    }
    case Intrinsic::mulhi: {
        internal_assert(node->args.size() == 2);
        llvm::Value *a = codegen_expr(node->args[0]);
        llvm::Value *b = codegen_expr(node->args[1]);
        value = multiply_high(a, b, node->type.is_int(), "mulhi");
        return;
    }
    case Intrinsic::div_multiplier: {
        internal_assert(node->args.size() == 1);
        llvm::Value *d = codegen_expr(node->args[0]);
        value = division_multiplier(d, node->type.is_int(), "div_multiplier");
        return;
    }
    case Intrinsic::acos: {
        // An intrinsic since LLVM 19, along with asin and atan: on a scalar
        // it is legalised to the same libm call a C compiler emits for
        // std::acos, and on a vector it is one call of the libmvec entry
        // point where the target library information maps it (see
        // target_library_info) rather than a call per lane.
        intrin = llvm::Intrinsic::acos;
        break;
    }
    case Intrinsic::asin: {
        // The companion of acos, and lowered the same way.
        intrin = llvm::Intrinsic::asin;
        break;
    }
    case Intrinsic::atanh: {
        // LLVM has intrinsics for the hyperbolic functions but not for their
        // inverses, so this is a call to libm -- which is what a C compiler
        // emits for std::atanh too, and so is the same function bit for bit
        // -- or, on a vector, to libmvec's atanhf (see codegen_libm_call).
        value = codegen_libm_call("atanh", node);
        return;
    }
    case Intrinsic::atan2: {
        // Two arguments, and otherwise the same story: no LLVM intrinsic
        // before LLVM 20, so libm's atan2f, which is what a C compiler emits
        // for std::atan2, or libmvec's on a vector.
        internal_assert(node->args.size() == 2) << node->args.size();
        value = codegen_libm_call("atan2", node);
        return;
    }
    case Intrinsic::cos: {
        intrin = llvm::Intrinsic::cos;
        break;
    }
    case Intrinsic::cosh: {
        intrin = llvm::Intrinsic::cosh;
        break;
    }
    case Intrinsic::round: {
        // Halfway cases go away from zero, which is what C's round does and
        // so what the C++ backend already emits. llvm.rint would follow the
        // rounding mode instead and disagree on exactly the halves.
        intrin = llvm::Intrinsic::round;
        break;
    }
    case Intrinsic::exp: {
        intrin = llvm::Intrinsic::exp;
        break;
    }
    case Intrinsic::log: {
        // Natural log, as C's log is, not log2 or log10.
        intrin = llvm::Intrinsic::log;
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
    case Intrinsic::max:
    case Intrinsic::min: {
        const bool is_max = node->op == Intrinsic::max;
        if (node->args[0].type().is_int()) {
            intrin = is_max ? llvm::Intrinsic::smax : llvm::Intrinsic::smin;
            break;
        } else if (node->args[0].type().is_uint()) {
            intrin = is_max ? llvm::Intrinsic::umax : llvm::Intrinsic::umin;
            break;
        }
        internal_assert(node->args[0].type().is_float())
            << "Cannot lower " << (is_max ? "max" : "min")
            << " of type: " << node->args[0].type();
        // C++'s std::max and std::min, exactly: `a < b ? b : a` and
        // `b < a ? b : a`. A NaN makes the comparison false, so a NaN in the
        // first argument comes back and a NaN in the second is passed over.
        // That is what a program translated from C++ means by max and min,
        // and it is also what one x86 instruction does -- maxss and minss
        // return their second operand when the operands are unordered, and
        // LLVM matches this compare-and-select to them. These used to be
        // llvm.maxnum and llvm.minnum, which drop a NaN in either position
        // instead; that is libm's fmax, not std::max, and on x86 it costs a
        // compare for NaN and a blend around every one. A BVH node test has
        // four of them, and a renderer does nothing more often.
        llvm::Value *a = codegen_expr(node->args[0]);
        llvm::Value *b = codegen_expr(node->args[1]);
        llvm::Value *take_b = is_max ? builder->CreateFCmpOLT(a, b, "max_lt")
                                     : builder->CreateFCmpOLT(b, a, "min_lt");
        value = builder->CreateSelect(take_b, b, a, is_max ? "max" : "min");
        return;
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
    case Intrinsic::sqr: {
        // Squaring is a multiplication, and naming the operand first keeps it
        // to one evaluation. The SSA path lowers this the same way in
        // SSA/Convert.cpp; this is the direct-to-LLVM path saying it too.
        internal_assert(node->args.size() == 1);
        llvm::Value *a = codegen_expr(node->args[0]);
        value = node->type.is_float() ? builder->CreateFMul(a, a)
                                      : builder->CreateMul(a, a);
        return;
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
        // A struct argument is passed as the value it is; only an exported
        // boundary still takes one by pointer.
        llvm::Value *argument = codegen_expr(node->args[i]);
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
        value = emit_call(
            llvm::FunctionCallee(function_type, codegen_expr(node->func)),
            indirect_return_type(function_t->ret_type), std::move(args));
        return;
    }
    // TODO: figure out how to make sure we have the right
    // number of arguments here for better error handling.
    value = emit_call(func, std::move(args));
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
    // An access chain rooted at a dereference names a place, so its address is
    // an offset and not a copy. Tested before anything is generated, because
    // generating the value would load the very thing being addressed.
    //
    // This used to come *after* the struct case below, which meant a field that
    // was itself a struct was copied rather than pointed at -- and a callee
    // taking it by `mut` then wrote into the copy. `next_float(st.rng)` is
    // exactly that: `rng` is a struct field of a mutable parameter, so pbrt's
    // independent sampler never advanced and returned its first number for
    // every draw. A field of scalar type took the correct path below, which is
    // why this survived so long. See
    // tests/bonsai/correctness/llvm/mut-field-argument.bonsai.
    // The chain may also be rooted at an array, which is already the address
    // of its elements (see Type::is_reference): `vs[i].w` for a `vs :
    // mut Vertex[]`, or an element of a tree's leaf storage that an argmin
    // keeps a reference to. The root is then the innermost expression of
    // array type, and the accesses outside it are offsets from its value.
    Expr array_root;
    const bool addressable_chain = [&] {
        if (!node->expr.is<Extract, Access>()) {
            return false;
        }
        Expr root = node->expr;
        while (root.is<Extract, Access>()) {
            root = root.is<Extract>() ? root.as<Extract>()->vec
                                      : root.as<Access>()->value;
            if (root.type().is_reference()) {
                array_root = root;
                return true;
            }
        }
        return root.is<Deref>();
    }();

    if (addressable_chain) {
        // Fall through to the access-chain walk below.
    } else if (llvm::Value *pointee = codegen_expr(node->expr);
               auto *load = dyn_cast<llvm::LoadInst>(pointee)) {
        value = load->getPointerOperand();
        return;
    } else if (node->expr.type().is<Struct_t>()) {
        value = materialize_for_address(pointee,
                                        node->expr.type().as<Struct_t>()->name);
        return;
    } else {
        // A value that is not in memory and has no piece of anything in
        // memory to point into -- the result of an arithmetic expression
        // handed to a parameter taken by pointer, say. There is nothing to
        // name, so the only thing its address can mean is a copy.
        value = materialize_for_address(pointee, "value");
        return;
    }

    {
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
        } while (expr.is<Extract, Access>() && !expr.same_as(array_root));

        llvm::Value *ptr;
        Type bonsai_type;
        // An array's value is the address of its elements, so the first index
        // below offsets it directly; a pointer to a struct holding an array
        // field reaches the elements through a load of that field first.
        bool elements_in_hand = false;
        if (array_root.defined()) {
            ptr = codegen_expr(array_root);
            bonsai_type = array_root.type();
            elements_in_hand = true;
        } else {
            const Deref *deref = expr.as<Deref>();
            internal_assert(deref) << expr;
            ptr = codegen_expr(deref->expr);
            bonsai_type = deref->type;
            // A vector of pointers dereferences to a per-lane place (see
            // Deref::make), typed as a vector of what each lane's pointer
            // points at. The offsets below are into that pointee, and a GEP
            // over a vector of pointers with scalar indices yields a vector
            // of pointers, which is the address of the per-lane place.
            if (const Vector_t *per_lane = bonsai_type.as<Vector_t>();
                per_lane != nullptr && ptr->getType()->isVectorTy()) {
                bonsai_type = per_lane->etype;
            }
        }
        llvm::Type *llvm_t = codegen_type(bonsai_type);

        for (auto it = accesses.rbegin(); it != accesses.rend(); ++it) {
            const auto &access = *it;
            if (std::holds_alternative<Expr>(access)) {
                Expr idx = std::get<Expr>(access);
                llvm::Value *llvm_idx = codegen_expr(idx);

                if (!elements_in_hand) {
                    ptr = create_aligned_load(codegen_type(bonsai_type), ptr,
                                              "ptr_array_ld");
                }
                elements_in_hand = false;

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
    } else if (pointer_value->getType()->isVectorTy()) {
        // One pointer per lane, every lane on: a gather, of scalars. An
        // aggregate per lane would have to be gathered a field at a time,
        // which nothing asks for yet -- a per-lane aggregate is only ever
        // stepped into by an address chain (see the PtrTo visitor), never
        // loaded whole.
        const bool scalar_lanes =
            node->type.is_vector() && !node->type.element_of().is<Struct_t>();
        internal_assert(scalar_lanes)
            << "[unimplemented] gathering an aggregate a field at a time: "
            << Expr(node);
        llvm::Type *loaded_type = codegen_type(node->type);
        const llvm::DataLayout &dl = module->getDataLayout();
        const uint32_t lanes = node->type.lanes();
        value = builder->CreateMaskedGather(
            llvm::dyn_cast<llvm::VectorType>(loaded_type), pointer_value,
            dl.getABITypeAlign(loaded_type->getScalarType()),
            llvm::Constant::getAllOnesValue(
                llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false)),
            llvm::Constant::getNullValue(loaded_type), "deref_gather");
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
    } else if (build_type->isArrayTy() && node->type.is<Vector_t>()) {
        // A packed vector is its elements side by side in memory, an LLVM
        // array (see the Vector_t visitor), built element by element.
        internal_assert(values.empty() ||
                        values.size() == node->type.as<Vector_t>()->lanes)
            << "Partial build of a packed vector: " << Expr(node);
        if (values.empty()) {
            value = llvm::Constant::getNullValue(build_type);
            return;
        }
        value = llvm::PoisonValue::get(build_type);
        for (size_t i = 0; i < values.size(); i++) {
            value = builder->CreateInsertValue(value, values[i], unsigned(i));
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

// A value of one aggregate type read as another of the same size: an
// aggregate's bytes as the packed words of its storage, or the reverse. There
// is no bitcast for an aggregate, so it goes through a stack slot: the value
// is stored at its own type and loaded back at the other. Whatever of the
// storage the value did not cover -- padding -- stays undefined.
llvm::Value *CodeGen_LLVM::reinterpret_via_memory(llvm::Value *v,
                                                  llvm::Type *as) {
    const llvm::DataLayout &dl = module->getDataLayout();
    // Store sizes, not allocation sizes: a vector of other than a power of
    // two of elements is allocated rounded up to its alignment, and holds
    // exactly as many bytes as the struct it stands for all the same.
    if (dl.getTypeStoreSize(v->getType()) != dl.getTypeStoreSize(as)) {
        std::string from, to;
        llvm::raw_string_ostream from_os(from), to_os(to);
        v->getType()->print(from_os);
        as->print(to_os);
        internal_error << "reinterpreting "
                       << dl.getTypeStoreSize(v->getType()).getFixedValue()
                       << " bytes of " << from_os.str() << " as "
                       << dl.getTypeStoreSize(as).getFixedValue()
                       << " bytes of " << to_os.str();
    }
    // The slot is the vector's type, the larger and more aligned of the two.
    llvm::Type *slot_type = v->getType()->isVectorTy() ? v->getType() : as;
    llvm::Value *slot = create_alloca_at_entry(slot_type, "transpose");
    builder->CreateStore(v, slot);
    return create_aligned_load(as, slot, "transposed");
}

uint64_t CodeGen_LLVM::common_unit(const ir::Type &member, uint64_t offset,
                                  uint64_t so_far) {
    const llvm::DataLayout &dl = module->getDataLayout();
    if (const Struct_t *s = member.as<Struct_t>()) {
        llvm::StructType *st = llvm::cast<llvm::StructType>(codegen_type(member));
        const llvm::StructLayout *layout = dl.getStructLayout(st);
        for (size_t i = 0; i < s->fields.size(); i++) {
            internal_assert(!s->fields[i].type.is<Ref_t>())
                << "[unimplemented] a per-lane aggregate with a reference "
                << "field: " << member;
            so_far = common_unit(s->fields[i].type,
                                offset + layout->getElementOffset(unsigned(i)),
                                so_far);
        }
        return so_far;
    }
    if (const Vector_t *v = member.as<Vector_t>()) {
        const uint64_t esize =
            dl.getTypeAllocSize(codegen_type(v->etype)).getFixedValue();
        for (uint32_t c = 0; c < v->lanes; c++) {
            so_far = common_unit(v->etype, offset + c * esize, so_far);
        }
        return so_far;
    }
    const uint64_t size = dl.getTypeAllocSize(codegen_type(member)).getFixedValue();
    return std::gcd(so_far, std::gcd(size, offset));
}

void CodeGen_LLVM::scatter_units(const ir::Type &member, llvm::Value *wide,
                                 uint64_t offset, uint64_t unit,
                                 std::vector<llvm::Value *> &slots) {
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Type *unit_t = llvm::Type::getIntNTy(*context, unsigned(unit * 8));
    if (const Struct_t *s = member.as<Struct_t>()) {
        llvm::StructType *st = llvm::cast<llvm::StructType>(codegen_type(member));
        const llvm::StructLayout *layout = dl.getStructLayout(st);
        for (size_t i = 0; i < s->fields.size(); i++) {
            scatter_units(s->fields[i].type,
                          builder->CreateExtractValue(wide, unsigned(i)),
                          offset + layout->getElementOffset(unsigned(i)), unit,
                          slots);
        }
        return;
    }
    if (const Vector_t *v = member.as<Vector_t>()) {
        // Widened as one gang vector per component (see ir::widen).
        const uint64_t esize =
            dl.getTypeAllocSize(codegen_type(v->etype)).getFixedValue();
        for (uint32_t c = 0; c < v->lanes; c++) {
            scatter_units(v->etype, builder->CreateExtractValue(wide, c),
                          offset + c * esize, unit, slots);
        }
        return;
    }
    // A scalar: `wide` is <lanes x T>. A bool is a byte in storage.
    const uint64_t size = dl.getTypeAllocSize(codegen_type(member)).getFixedValue();
    llvm::Value *bits = wide;
    if (member.is<Bool_t>()) {
        bits = builder->CreateZExt(
            wide, llvm::FixedVectorType::get(i8_t, vector_lanes(wide->getType())));
    }
    if (size < unit) {
        // Narrower than a unit: the field shares its unit with its
        // neighbours, so its bits go in at their byte offset within it, ORed
        // into whatever the unit already holds.
        internal_assert(offset % unit + size <= unit)
            << "[unimplemented] a field of a per-lane aggregate straddling "
            << "two units: " << member << " at byte " << offset
            << " in units of " << unit;
        const unsigned n = vector_lanes(bits->getType());
        llvm::Type *narrow_t =
            llvm::FixedVectorType::get(llvm::Type::getIntNTy(*context, size * 8), n);
        if (bits->getType() != narrow_t) {
            bits = builder->CreateBitCast(bits, narrow_t);
        }
        llvm::Type *unit_vt = llvm::FixedVectorType::get(unit_t, n);
        llvm::Value *placed = builder->CreateShl(
            builder->CreateZExt(bits, unit_vt),
            llvm::ConstantInt::get(unit_vt, (offset % unit) * 8));
        llvm::Value *&slot = slots[size_t(offset / unit)];
        slot = slot == nullptr ? placed : builder->CreateOr(slot, placed);
        return;
    }
    internal_assert(offset % unit == 0)
        << "[unimplemented] a field of a per-lane aggregate not aligned to "
        << "its unit: " << member << " at byte " << offset << " in units of "
        << unit;
    const int m = int(size / unit);
    const int n = vector_lanes(bits->getType());
    llvm::Type *as_units = llvm::FixedVectorType::get(unit_t, unsigned(n * m));
    if (bits->getType() != as_units) {
        bits = builder->CreateBitCast(bits, as_units);
    }
    if (m == 1) {
        slots[size_t(offset / unit)] = bits;
        return;
    }
    // Lane k's m units are contiguous in `bits`; unit u of every lane is
    // then every m-th element from u.
    std::vector<llvm::Value *> parts = deinterleave_vector(bits, m);
    for (int u = 0; u < m; u++) {
        slots[size_t(offset / unit) + size_t(u)] = parts[size_t(u)];
    }
}

llvm::Value *CodeGen_LLVM::gather_units(const ir::Type &member,
                                        const ir::Type &wide_t,
                                        uint64_t offset, uint64_t unit,
                                        const std::vector<llvm::Value *> &slots,
                                        uint32_t lanes) {
    const llvm::DataLayout &dl = module->getDataLayout();
    llvm::Type *unit_t = llvm::Type::getIntNTy(*context, unsigned(unit * 8));
    llvm::Type *wide_llvm = codegen_type(wide_t);
    if (const Struct_t *s = member.as<Struct_t>()) {
        llvm::StructType *st = llvm::cast<llvm::StructType>(codegen_type(member));
        const llvm::StructLayout *layout = dl.getStructLayout(st);
        const Struct_t *ws = wide_t.as<Struct_t>();
        internal_assert(ws && ws->fields.size() == s->fields.size())
            << wide_t << " is not " << member << " widened";
        llvm::Value *result = llvm::PoisonValue::get(wide_llvm);
        for (size_t i = 0; i < s->fields.size(); i++) {
            llvm::Value *field = gather_units(
                s->fields[i].type, ws->fields[i].type,
                offset + layout->getElementOffset(unsigned(i)), unit, slots,
                lanes);
            result = builder->CreateInsertValue(result, field, unsigned(i));
        }
        return result;
    }
    if (const Vector_t *v = member.as<Vector_t>()) {
        const uint64_t esize =
            dl.getTypeAllocSize(codegen_type(v->etype)).getFixedValue();
        const Struct_t *ws = wide_t.as<Struct_t>();
        internal_assert(ws && ws->fields.size() == v->lanes)
            << wide_t << " is not " << member << " widened";
        llvm::Value *result = llvm::PoisonValue::get(wide_llvm);
        for (uint32_t c = 0; c < v->lanes; c++) {
            llvm::Value *component =
                gather_units(v->etype, ws->fields[c].type, offset + c * esize,
                             unit, slots, lanes);
            result = builder->CreateInsertValue(result, component, c);
        }
        return result;
    }
    const uint64_t size = dl.getTypeAllocSize(codegen_type(member)).getFixedValue();
    if (size < unit) {
        // Narrower than a unit: shifted out of its place in the unit it
        // shares (see scatter_units).
        internal_assert(offset % unit + size <= unit)
            << "[unimplemented] a field of a per-lane aggregate straddling "
            << "two units: " << member << " at byte " << offset
            << " in units of " << unit;
        llvm::Value *word = slots[size_t(offset / unit)];
        internal_assert(word != nullptr) << "No unit for " << member;
        llvm::Type *unit_vt = llvm::FixedVectorType::get(unit_t, lanes);
        llvm::Value *narrowed = builder->CreateTrunc(
            builder->CreateLShr(
                word, llvm::ConstantInt::get(unit_vt, (offset % unit) * 8)),
            llvm::FixedVectorType::get(llvm::Type::getIntNTy(*context, size * 8),
                                       lanes));
        if (member.is<Bool_t>()) {
            return builder->CreateTrunc(narrowed, wide_llvm);
        }
        return narrowed->getType() == wide_llvm
                   ? narrowed
                   : builder->CreateBitCast(narrowed, wide_llvm);
    }
    internal_assert(offset % unit == 0)
        << "[unimplemented] a field of a per-lane aggregate not aligned to "
        << "its unit: " << member << " at byte " << offset << " in units of "
        << unit;
    const int m = int(size / unit);
    std::vector<llvm::Value *> parts(static_cast<size_t>(m));
    for (int u = 0; u < m; u++) {
        parts[size_t(u)] = slots[size_t(offset / unit) + size_t(u)];
    }
    llvm::Value *bits = interleave_vectors(parts);
    if (member.is<Bool_t>()) {
        llvm::Type *bytes = llvm::FixedVectorType::get(i8_t, lanes);
        if (bits->getType() != bytes) {
            bits = builder->CreateBitCast(bits, bytes);
        }
        return builder->CreateTrunc(bits, wide_llvm);
    }
    if (bits->getType() != wide_llvm) {
        bits = builder->CreateBitCast(bits, wide_llvm);
    }
    (void)unit_t;
    return bits;
}

void CodeGen_LLVM::visit(const Access *node) {
    ir::Expr value_e = node->value;

    // For debuggability.
    std::string name = node->field;
    if (const auto *var = value_e.as<Var>()) {
        name = var->name + "." + name;
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
                // A function returning through a hidden pointer hands its own
                // pointer on: the callee writes where this call's caller is
                // waiting to read, which is what keeps this a tail call.
                if (current_sret) {
                    args.push_back(current_sret);
                }
                for (const Expr &arg : call->args) {
                    args.push_back(codegen_expr(arg));
                }

                llvm::CallInst *tail = builder->CreateCall(func, args);
                // Make this a tailcall.
                tail->setTailCallKind(llvm::CallInst::TCK_Tail);
                tail->setCallingConv(
                    current_function
                        ->getCallingConv()); // Ensure same convention

                if (current_sret) {
                    tail->addParamAttr(
                        0, llvm::Attribute::getWithStructRetType(
                               *context, func->getParamStructRetType(0)));
                    builder->CreateRetVoid();
                } else {
                    builder->CreateRet(tail);
                }
                return;
            }
        }
    }

    llvm::Value *val = codegen_expr(value);
    if (current_sret) {
        builder->CreateStore(val, current_sret);
        builder->CreateRetVoid();
        return;
    }
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

    // Each arm is a scope of its own: a name bound in one is not in scope in
    // the other, nor after the branch. The relooper relies on this -- two
    // arms that each bind the same name, because each was handed a value
    // under a name of its own, are two bindings and not one redefined -- and
    // it is what the braces of the C++ backend already say.
    for (const auto &p : blocks) {
        llvm::BasicBlock *then_bb =
            llvm::BasicBlock::Create(*context, "then_bb", current_function);
        llvm::BasicBlock *next_bb =
            llvm::BasicBlock::Create(*context, "next_bb", current_function);
        codegen_short_circuit(p.expr, then_bb, next_bb);
        builder->SetInsertPoint(then_bb);
        frames.push_frame();
        codegen_stmt(p.stmt);
        frames.pop_frame();
        if (!p.returns) {
            codegen_branch(after_bb);
        }
        builder->SetInsertPoint(next_bb);
    }

    if (final_else.defined()) {
        frames.push_frame();
        codegen_stmt(final_else);
        frames.pop_frame();
    }

    if (needs_after_bb) {
        codegen_branch(after_bb);
        builder->SetInsertPoint(after_bb);
    }
}

// An LLVM switch: a case per arm but the last, which is the default (see
// ir::SwitchStmt). Each arm is a scope of its own, as an if's arms are.
void CodeGen_LLVM::visit(const SwitchStmt *node) {
    llvm::Value *value = codegen_expr(node->value);
    auto *int_type = llvm::dyn_cast<llvm::IntegerType>(value->getType());
    internal_assert(int_type) << "Switch on a non-integer: " << node->value;

    bool needs_after_bb = false;
    for (const Stmt &arm : node->arms) {
        needs_after_bb =
            needs_after_bb || !arm.defined() || !always_returns(arm);
    }

    internal_assert(current_function);
    llvm::BasicBlock *after_bb =
        needs_after_bb
            ? llvm::BasicBlock::Create(*context, "after_bb", current_function)
            : nullptr;
    llvm::BasicBlock *default_bb =
        llvm::BasicBlock::Create(*context, "default_bb", current_function);
    llvm::SwitchInst *sw = builder->CreateSwitch(
        value, default_bb, static_cast<unsigned>(node->arms.size() - 1));

    for (size_t k = 0; k < node->arms.size(); k++) {
        const bool last = k + 1 == node->arms.size();
        llvm::BasicBlock *arm_bb = default_bb;
        if (!last) {
            // Named by the value it is taken on, so that the block for tag 1
            // reads as such and not as whatever suffix LLVM invents to keep
            // two blocks of one name apart.
            arm_bb = llvm::BasicBlock::Create(
                *context, "case" + std::to_string(k) + "_bb", current_function);
            sw->addCase(llvm::ConstantInt::get(int_type, k), arm_bb);
        }
        builder->SetInsertPoint(arm_bb);
        if (node->arms[k].defined()) {
            frames.push_frame();
            codegen_stmt(node->arms[k]);
            frames.pop_frame();
        }
        if (!node->arms[k].defined() || !always_returns(node->arms[k])) {
            codegen_branch(after_bb);
        }
    }

    if (needs_after_bb) {
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

    llvm::Value *dest = codegen_write_loc(loc);
    // A name bound to a value rather than to storage cannot be assigned to.
    // Without this the failure is an assertion inside LLVM, which says
    // nothing about which name in which statement is at fault.
    internal_assert(dest->getType()->isPointerTy())
        << "Cannot store to " << loc.base
        << ": it is bound to a value, not to storage, in " << Stmt(node);

    // A store the gang makes under a mask through one address: into per-lane
    // memory lane by lane, or into shared memory once if any lane is on.
    if (node->mask.defined()) {
        create_masked_store_at(rhs, dest, codegen_expr(node->mask));
        return;
    }

    // A whole array assigned to a name of array type: `ws = {1.0, 5.0}` for a
    // `ws : mut array[f32, 2]`. A name of array type is bound to its
    // elements' storage (see codegen_write_loc), and so is the value (see
    // Type::is_reference), so what moves is the elements, not the address.
    // Storing the value would write the right-hand side's *pointer* over the
    // first bytes of the destination's elements -- which is what happened: a
    // mutable array local initialized from a literal read back the upper half
    // of a heap address as its second element (see
    // tests/bonsai/correctness/llvm/mut-array-literal.bonsai). A field or
    // element of array type, by contrast, is a slot holding the handle, and
    // assigning it stores the handle as before.
    if (const Array_t *array_t = loc.type.as<Array_t>();
        array_t != nullptr && loc.accesses.empty() &&
        node->value.type().is<Array_t>()) {
        internal_assert(array_t->size.defined())
            << "[unimplemented] assigning an array of unknown size: "
            << Stmt(node);
        llvm::Type *etype = codegen_type(array_t->etype);
        const llvm::DataLayout &dl = module->getDataLayout();
        llvm::Value *count = codegen_expr(array_t->size);
        llvm::Value *bytes = builder->CreateMul(
            builder->CreateZExtOrTrunc(count, i64_t),
            llvm::ConstantInt::get(i64_t, dl.getTypeAllocSize(etype)),
            "array_bytes");
        builder->CreateMemCpy(dest, llvm::MaybeAlign(), rhs, llvm::MaybeAlign(),
                              bytes);
        return;
    }

    llvm::StoreInst *store =
        builder->CreateStore(rhs, dest, /*isVolatile=*/false);
    add_tbaa(store, node->value.type());
}

llvm::Value *CodeGen_LLVM::codegen_libm_call(const std::string &name,
                                             const Intrinsic *node) {
    // One argument or two: `acos(x)` and `atan2(y, x)` differ in nothing else,
    // so the arity is read off the call rather than assumed. Every argument
    // has to have the same shape, since a lane of the result is made from the
    // matching lane of each.
    internal_assert(!node->args.empty())
        << "libm call " << name << " takes arguments: " << Expr(node);
    const Type arg_type = node->args[0].type();
    for (const Expr &arg : node->args) {
        internal_assert(equals(arg.type(), arg_type))
            << "libm call " << name << " has arguments of differing types: "
            << Expr(node);
    }
    const bool is_vector = arg_type.is<Vector_t>();
    const Type scalar_type = is_vector ? arg_type.element_of() : arg_type;
    internal_assert(scalar_type.is_float())
        << "libm call " << name << " expects a float: " << Expr(node);

    // libm names the float overload with an `f`; the unsuffixed one is double.
    const bool single = scalar_type.bits() == 32;
    internal_assert(single || scalar_type.bits() == 64)
        << "No libm entry point for " << scalar_type << " in " << Expr(node);
    llvm::Type *llvm_scalar = single ? f32_t : f64_t;
    const std::vector<llvm::Type *> params(node->args.size(), llvm_scalar);
    llvm::FunctionCallee callee = module->getOrInsertFunction(
        single ? name + "f" : name,
        llvm::FunctionType::get(llvm_scalar, params, /*isVarArg=*/false));

    std::vector<llvm::Value *> args;
    args.reserve(node->args.size());
    for (const Expr &arg : node->args) {
        args.push_back(codegen_expr(arg));
    }
    if (!is_vector) {
        return builder->CreateCall(callee, args, name);
    }

    // The host libmvec's vector entry point, one call for the whole gang,
    // when it has one for this width; see probe_host_vector_math.
    if (const auto vector_symbol =
            vector_math_symbol(name, arg_type.lanes(), scalar_type.bits(),
                               unsigned(node->args.size()))) {
        llvm::Type *vector_t = codegen_type(node->type);
        const std::vector<llvm::Type *> vector_params(node->args.size(),
                                                      vector_t);
        llvm::FunctionCallee vector_callee = module->getOrInsertFunction(
            *vector_symbol,
            llvm::FunctionType::get(vector_t, vector_params,
                                    /*isVarArg=*/false));
        return builder->CreateCall(vector_callee, args, name);
    }

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

namespace {

// The Vector Function ABI name glibc gives a vector entry point of libm:
// `_ZGV`, the ISA level (`b` SSE4, `c` AVX, `d` AVX2, `e` AVX-512), `N` for
// no mask, the lane count, a `v` per vector argument, and the scalar
// function's name.
std::string vector_abi_name(char isa, uint32_t lanes, unsigned arity,
                            const std::string &name, bool single) {
    return "_ZGV" + std::string(1, isa) + "N" + std::to_string(lanes) +
           std::string(arity, 'v') + "_" + name + (single ? "f" : "");
}

} // namespace

// Asked of the host's libmvec rather than assumed: which functions it has
// depends on the glibc -- 2.22 shipped sin, cos, exp, log and pow, 2.35 the
// rest of libm -- and which ISA levels run depends on the machine. Only a
// symbol present in this machine's libmvec, at a level this machine runs, is
// ever emitted, so nothing here can fail to link or to run where it was
// compiled; following the host means the code runs on the host.
void CodeGen_LLVM::probe_host_vector_math() {
    // Global, and left loaded: the JIT resolves what the code it runs calls
    // against this process's symbols, and libmvec is not otherwise among
    // them -- the compiler links libm, whose linker script pulls libmvec in
    // only for a program that references it.
    void *lib = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (lib == nullptr) {
        return;
    }
    const llvm::StringMap<bool> features = llvm::sys::getHostCPUFeatures();
    const auto has = [&](const char *feature) {
        const auto it = features.find(feature);
        return it != features.end() && it->second;
    };
    struct Isa {
        char letter;
        uint32_t f32_lanes, f64_lanes;
        bool runs;
    };
    const Isa isas[] = {{'b', 4, 2, has("sse4.1")},
                        {'d', 8, 4, has("avx2")},
                        {'e', 16, 8, has("avx512f")}};
    struct Fn {
        const char *name;
        unsigned arity;
    };
    static const Fn fns[] = {
        {"sin", 1},   {"cos", 1},   {"tan", 1},   {"asin", 1},  {"acos", 1},
        {"atan", 1},  {"atan2", 2}, {"sinh", 1},  {"cosh", 1},  {"tanh", 1},
        {"asinh", 1}, {"acosh", 1}, {"atanh", 1}, {"exp", 1},   {"exp2", 1},
        {"exp10", 1}, {"expm1", 1}, {"log", 1},   {"log2", 1},  {"log10", 1},
        {"log1p", 1}, {"pow", 2},   {"hypot", 2}, {"cbrt", 1},  {"erf", 1},
        {"erfc", 1}};
    for (const Isa &isa : isas) {
        if (!isa.runs) {
            continue;
        }
        for (const Fn &fn : fns) {
            for (const bool single : {true, false}) {
                const std::string symbol =
                    vector_abi_name(isa.letter,
                                    single ? isa.f32_lanes : isa.f64_lanes,
                                    fn.arity, fn.name, single);
                if (dlsym(lib, symbol.c_str()) != nullptr) {
                    host_vector_math.insert(symbol);
                }
            }
        }
    }
}

std::optional<std::string>
CodeGen_LLVM::vector_math_symbol(const std::string &name, uint32_t lanes,
                                 unsigned bits, unsigned arity) const {
    char isa = 0;
    const uint32_t f32_lanes = bits == 32 ? lanes : lanes * 2;
    switch (f32_lanes) {
    case 4:
        isa = 'b';
        break;
    case 8:
        isa = 'd';
        break;
    case 16:
        isa = 'e';
        break;
    default:
        return std::nullopt;
    }
    if (bits != 32 && bits != 64) {
        return std::nullopt;
    }
    const std::string symbol =
        vector_abi_name(isa, lanes, arity, name, bits == 32);
    if (host_vector_math.count(symbol) == 0) {
        return std::nullopt;
    }
    return symbol;
}

llvm::TargetLibraryInfoImpl
CodeGen_LLVM::target_library_info(const llvm::Triple &triple) {
    llvm::TargetLibraryInfoImpl info(triple);
    // The intrinsics LLVM has for libm's functions, each mapped onto the
    // host's libmvec entry point of the same width where the host has one.
    // LLVM's own table for libmvec (VecFuncs.def) stops at sin, cos, tan,
    // exp, log and pow -- what glibc 2.22 shipped -- so the table is built
    // here from what the probe found instead.
    struct Fn {
        const char *intrinsic;
        const char *libm;
        unsigned arity;
    };
    static const Fn fns[] = {
        {"sin", "sin", 1},     {"cos", "cos", 1},     {"tan", "tan", 1},
        {"asin", "asin", 1},   {"acos", "acos", 1},   {"atan", "atan", 1},
        {"sinh", "sinh", 1},   {"cosh", "cosh", 1},   {"tanh", "tanh", 1},
        {"exp", "exp", 1},     {"exp2", "exp2", 1},   {"exp10", "exp10", 1},
        {"log", "log", 1},     {"log2", "log2", 1},   {"log10", "log10", 1},
        {"pow", "pow", 2}};
    const auto keep = [&](std::string s) -> llvm::StringRef {
        vector_math_names.push_back(std::move(s));
        return vector_math_names.back();
    };
    std::vector<llvm::VecDesc> descs;
    for (const Fn &fn : fns) {
        for (const unsigned bits : {32u, 64u}) {
            for (const uint32_t lanes : {2u, 4u, 8u, 16u}) {
                const auto symbol =
                    vector_math_symbol(fn.libm, lanes, bits, fn.arity);
                if (!symbol) {
                    continue;
                }
                descs.emplace_back(
                    keep(std::string("llvm.") + fn.intrinsic +
                         (bits == 32 ? ".f32" : ".f64")),
                    keep(*symbol), llvm::ElementCount::getFixed(lanes),
                    /*Masked=*/false,
                    keep("_ZGV_LLVM_N" + std::to_string(lanes) +
                         std::string(fn.arity, 'v')));
            }
        }
    }
    if (!descs.empty()) {
        info.addVectorizableFunctions(descs);
    }
    return info;
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

    if (node->atomic) {
        // One indivisible read-modify-write, rather than the load, combine and
        // store below -- which is three steps another thread can land between.
        //
        // Monotonic ordering: an accumulate makes no claim about anything
        // other than itself becoming visible, so the stronger orderings would
        // be paying for a guarantee nobody asked for. What it does promise is
        // that no update is lost.
        const Type &value_t = node->value.type();
        if (const Vector_t *v = value_t.as<Vector_t>()) {
            // Nothing atomic works on a vector. A short vector's slot is its
            // components side by side, and each is an atomic of its own:
            // indivisible per component, which is what a sum of vectors
            // needs, since no component's update is lost, and the whole is
            // never promised to anyone as one read.
            const llvm::AtomicRMWInst::BinOp rmw =
                atomic_rmw_op(node->op, v->etype);
            llvm::Type *elem_t = codegen_type(v->etype);
            for (uint32_t k = 0; k < v->lanes; k++) {
                llvm::Value *ptr = builder->CreateInBoundsGEP(
                    elem_t, loc, {llvm::ConstantInt::get(i64_t, k)});
                llvm::Value *component =
                    builder->CreateExtractElement(update, uint64_t(k));
                builder->CreateAtomicRMW(rmw, ptr, component, llvm::MaybeAlign(),
                                         llvm::AtomicOrdering::Monotonic);
            }
            return;
        }
        builder->CreateAtomicRMW(atomic_rmw_op(node->op, value_t), loc, update,
                                 llvm::MaybeAlign(),
                                 llvm::AtomicOrdering::Monotonic);
        return;
    }

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
                                              const std::string &name,
                                              llvm::Value *length) {
    // A boolean occupies a byte in memory -- that is what the scalar path
    // stores and what the C++ side of an exported function sees -- while a
    // vector of i1 is bit-packed. So booleans are loaded a byte per lane and
    // narrowed afterwards, or eight lanes would come out of a single byte.
    if (etype->isIntegerTy(1)) {
        llvm::Value *bytes = create_vector_load(i8_t, base, index, lanes,
                                                mask_expr, name, length);
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
        // The *element's* alignment, not the vector's. A run of eight floats
        // inside an array of floats is aligned to a float; nothing about it
        // says it starts on a 32-byte boundary, and `a[i:i+8]` for odd `i`
        // says it does not. Claiming the vector's alignment was harmless
        // while a generic target split every access into 16-byte pieces, and
        // became a segfault the moment the host's AVX let LLVM emit one
        // aligned 32-byte move. The gather below has always used the element's
        // for the same reason.
        const llvm::Align align = dl.getABITypeAlign(etype);
        if (mask == nullptr) {
            llvm::LoadInst *load = builder->CreateLoad(vtype, first, name);
            load->setAlignment(align);
            return load;
        }
        // A disabled lane must not touch memory at all, so the disabled
        // lanes come from the passthrough value rather than from the load.
        return builder->CreateMaskedLoad(vtype, first, align, mask,
                                         llvm::Constant::getNullValue(vtype),
                                         name);
    }

    llvm::Value *indices = codegen_expr(index);
    const uint64_t esize = dl.getTypeStoreSize(etype).getFixedValue();
    if (esize < 4 && length != nullptr &&
        indices->getType()->getScalarType()->getIntegerBitWidth() <= 32) {
        // A byte or halfword per lane, read as part of the word that holds
        // it rather than lane by lane behind a branch each (see
        // gather_elements and gather_sub_word_elements). The array has to be
        // a word long for that to have a word to read: settled here when its
        // length is a constant, and by a branch the gang takes together when
        // it is not.
        llvm::Type *offsets_t = llvm::FixedVectorType::get(i32_t, lanes);
        llvm::Value *offsets = builder->CreateMul(
            builder->CreateIntCast(indices, offsets_t, /*isSigned=*/true),
            llvm::ConstantInt::get(offsets_t, esize), name + "_off");
        llvm::Value *total = builder->CreateMul(
            builder->CreateIntCast(length, i32_t, /*isSigned=*/false),
            llvm::ConstantInt::get(i32_t, esize), name + "_bytes");
        if (mask == nullptr) {
            mask = llvm::Constant::getAllOnesValue(
                llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false));
        }
        if (const auto *known = llvm::dyn_cast<llvm::ConstantInt>(total)) {
            if (known->getZExtValue() >= 4) {
                return gather_sub_word_elements(etype, base, offsets, total,
                                                lanes, mask, name);
            }
        } else {
            llvm::BasicBlock *words_bb = llvm::BasicBlock::Create(
                *context, name + "_words", current_function);
            llvm::BasicBlock *lanes_bb = llvm::BasicBlock::Create(
                *context, name + "_lanes", current_function);
            llvm::BasicBlock *after_bb = llvm::BasicBlock::Create(
                *context, name + "_after", current_function);
            builder->CreateCondBr(
                builder->CreateICmpUGE(total, llvm::ConstantInt::get(i32_t, 4)),
                words_bb, lanes_bb);
            builder->SetInsertPoint(words_bb);
            llvm::Value *as_words = gather_sub_word_elements(
                etype, base, offsets, total, lanes, mask, name);
            llvm::BasicBlock *words_end = builder->GetInsertBlock();
            builder->CreateBr(after_bb);
            builder->SetInsertPoint(lanes_bb);
            llvm::Value *ptrs =
                element_addresses(etype, base, indices, name + "_ptrs");
            llvm::Value *as_lanes = builder->CreateMaskedGather(
                vtype, ptrs, dl.getABITypeAlign(etype), mask,
                llvm::Constant::getNullValue(vtype), name);
            builder->CreateBr(after_bb);
            builder->SetInsertPoint(after_bb);
            llvm::PHINode *phi = builder->CreatePHI(vtype, 2, name);
            phi->addIncoming(as_words, words_end);
            phi->addIncoming(as_lanes, lanes_bb);
            return phi;
        }
    }

    // One element per lane at the base plus the index at the element's
    // size (see gather_indexed, and element_addresses for the addressing
    // model).
    return gather_indexed(etype, base, indices,
                          dl.getTypeAllocSize(etype).getFixedValue(), 0,
                          dl.getABITypeAlign(etype).value(), lanes, mask, name);
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
        // The element's alignment, as in create_vector_load above.
        if (mask == nullptr) {
            llvm::StoreInst *store = builder->CreateStore(value, first);
            store->setAlignment(dl.getABITypeAlign(etype));
            return;
        }
        builder->CreateMaskedStore(value, first, dl.getABITypeAlign(etype),
                                   mask);
        return;
    }

    llvm::Value *ptrs =
        element_addresses(etype, base, codegen_expr(index), "store_ptrs");
    if (mask == nullptr) {
        mask = llvm::Constant::getAllOnesValue(
            llvm::VectorType::get(i1_t, lanes, /*Scalable=*/false));
    }
    builder->CreateMaskedScatter(value, ptrs, dl.getABITypeAlign(etype), mask);
}

llvm::Value *CodeGen_LLVM::element_addresses(llvm::Type *element,
                                             llvm::Value *base,
                                             llvm::Value *indices,
                                             const std::string &name) {
    // 32-bit addressing, as ISPC's default is: a gather or scatter is a base
    // pointer and one 32-bit offset per lane, which is what the hardware
    // takes -- vgatherdps reads its eight indices out of one 256-bit
    // register and scales them by 1, 2, 4 or 8 -- and half the register
    // traffic of eight 64-bit pointers. The indices this IR computes are
    // 32-bit already; sign-extending them to 64 here, as the scalar path
    // does, only made LLVM carry the extension through every address.
    //
    // An element whose size is not a scale the hardware has is addressed by
    // its byte offset, multiplied out in 32 bits, and a GEP over bytes. The
    // limit of the model is that a byte offset must fit in 31 bits: an array
    // over 2 GB is addressed wrongly past that, in a gather. (A GEP index is
    // sign-extended, so an unsigned index past 2^31 was already outside
    // what this backend addresses, before any scaling.)
    auto *vt = llvm::dyn_cast<llvm::VectorType>(indices->getType());
    internal_assert(vt) << "element_addresses wants one index per lane";
    const llvm::DataLayout &dl = module->getDataLayout();
    const uint64_t size = dl.getTypeAllocSize(element);
    if (vt->getElementType()->getIntegerBitWidth() > 32 || size == 1 ||
        size == 2 || size == 4 || size == 8) {
        return builder->CreateInBoundsGEP(element, base, indices, name);
    }
    llvm::Value *offsets = builder->CreateMul(
        indices, llvm::ConstantInt::get(vt, size), name + "_off");
    return builder->CreateInBoundsGEP(i8_t, base, offsets, name);
}

void CodeGen_LLVM::create_masked_store_at(llvm::Value *value,
                                          llvm::Value *dest,
                                          llvm::Value *mask) {
    llvm::Type *t = value->getType();
    const llvm::DataLayout &dl = module->getDataLayout();

    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        // A vector that is not one value per lane -- a lane's own short
        // vector, the same for every lane -- is a uniform value, stored once
        // if any lane is on, as a scalar is below.
        if (vt->getNumElements() != vector_lanes(mask->getType())) {
            emit_if_any_lane(mask, [&] { builder->CreateStore(value, dest); });
            return;
        }
        // One slot per lane, written for the lanes that are on. A vector of
        // booleans has no masked store to lower to -- the intrinsic wants
        // elements the target can address, and a bit is not one -- so it is
        // blended instead: the memory is the gang's own, so reading the
        // disabled lanes back to keep them is reading what the gang itself
        // last wrote there.
        if (vt->getElementType()->isIntegerTy(1)) {
            llvm::Value *old = builder->CreateLoad(t, dest, "masked_old");
            builder->CreateStore(builder->CreateSelect(mask, value, old), dest);
            return;
        }
        builder->CreateMaskedStore(value, dest, dl.getABITypeAlign(t), mask);
        return;
    }

    if (auto *st = llvm::dyn_cast<llvm::StructType>(t)) {
        // A widened struct is a struct of per-lane fields (see widen() in
        // SSA/Vectorize.cpp), each written under the same mask at its own
        // address.
        for (unsigned i = 0; i < st->getNumElements(); i++) {
            create_masked_store_at(builder->CreateExtractValue(value, i),
                                   builder->CreateStructGEP(st, dest, i), mask);
        }
        return;
    }

    if (auto *at = llvm::dyn_cast<llvm::ArrayType>(t)) {
        for (uint64_t i = 0; i < at->getNumElements(); i++) {
            create_masked_store_at(
                builder->CreateExtractValue(value, unsigned(i)),
                builder->CreateConstInBoundsGEP2_64(at, dest, 0, i), mask);
        }
        return;
    }

    // A scalar, into memory the lanes share: the value is the same whichever
    // lane writes it, so it is written once, provided some lane would have --
    // ispc's rule for an assignment to a uniform inside varying control flow.
    emit_if_any_lane(mask, [&] { builder->CreateStore(value, dest); });
}

void CodeGen_LLVM::emit_if_any_lane(llvm::Value *mask,
                                    const std::function<void()> &body) {
    emit_if(builder->CreateOrReduce(mask), body);
}

void CodeGen_LLVM::emit_if(llvm::Value *cond,
                           const std::function<void()> &body) {
    llvm::BasicBlock *then_bb =
        llvm::BasicBlock::Create(*context, "if_lane", current_function);
    llvm::BasicBlock *after_bb =
        llvm::BasicBlock::Create(*context, "if_lane_after", current_function);
    builder->CreateCondBr(cond, then_bb, after_bb);
    builder->SetInsertPoint(then_bb);
    body();
    builder->CreateBr(after_bb);
    builder->SetInsertPoint(after_bb);
}

llvm::AtomicRMWInst::BinOp CodeGen_LLVM::atomic_rmw_op(Accumulate::OpType op,
                                                       const Type &value_t) {
    switch (op) {
    case Accumulate::Add:
        return value_t.is_float() ? llvm::AtomicRMWInst::FAdd
                                  : llvm::AtomicRMWInst::Add;
    case Accumulate::Sub:
        return value_t.is_float() ? llvm::AtomicRMWInst::FSub
                                  : llvm::AtomicRMWInst::Sub;
    case Accumulate::Min:
        return value_t.is_float()  ? llvm::AtomicRMWInst::FMin
               : value_t.is_int()  ? llvm::AtomicRMWInst::Min
                                   : llvm::AtomicRMWInst::UMin;
    case Accumulate::Max:
        return value_t.is_float()  ? llvm::AtomicRMWInst::FMax
               : value_t.is_int()  ? llvm::AtomicRMWInst::Max
                                   : llvm::AtomicRMWInst::UMax;
    default:
        // Multiply and the argmins have no atomicrmw; they would need a
        // compare-and-swap loop, which is worth writing when something wants
        // one rather than in advance.
        internal_error << "atomic is not supported for this accumulate";
        return llvm::AtomicRMWInst::Add;
    }
}

void CodeGen_LLVM::emit_atomic_lanes(Accumulate::OpType op, const Type &value_t,
                                     llvm::Value *ptrs, llvm::Value *values,
                                     llvm::Value *mask) {
    const bool scalar_lanes =
        value_t.is_vector() && value_t.element_of().is_scalar();
    internal_assert(scalar_lanes)
        << "[unimplemented] an atomic accumulate of an aggregate per lane: "
        << value_t;
    const llvm::AtomicRMWInst::BinOp rmw = atomic_rmw_op(op, value_t.element_of());
    const uint32_t lanes = value_t.lanes();
    llvm::Type *vector_t = codegen_type(value_t);
    for (uint32_t k = 0; k < lanes; k++) {
        llvm::Value *bit = builder->CreateExtractElement(mask, uint64_t(k));
        emit_if(bit, [&] {
            llvm::Value *ptr =
                ptrs->getType()->isVectorTy()
                    ? builder->CreateExtractElement(ptrs, uint64_t(k))
                    : builder->CreateInBoundsGEP(
                          vector_t, ptrs,
                          {llvm::ConstantInt::get(i32_t, 0),
                           llvm::ConstantInt::get(i32_t, k)});
            llvm::Value *v = builder->CreateExtractElement(values, uint64_t(k));
            builder->CreateAtomicRMW(rmw, ptr, v, llvm::MaybeAlign(),
                                     llvm::AtomicOrdering::Monotonic);
        });
    }
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
    // A local an aggregate lives in is aligned to the widest vector the
    // optimizer might initialize it with -- the machine's register, see
    // vector_register_bits -- not only to what its own fields need. LLVM
    // merges adjacent field stores into one register-wide store and then
    // wants that store's address aligned to the register, which a struct
    // slot aligned only to its 16-byte fields is not -- and an aligned store
    // to it faults. This bites the vectorized parfor kernel, whose stack is
    // realigned; over-aligning the aggregate closes the gap. Only there: the
    // relooper's functions keep the alignment their fields ask for, so the
    // golden IR that pins it is unchanged.
    const unsigned register_align = unsigned(vector_register_bits() / 8);
    if (lowering_from_ssa && t->isAggregateType() && align < register_align) {
        align = register_align;
    }
    ptr->setAlignment(llvm::Align(align));

    builder->restoreIP(here);
    return ptr;
}

llvm::Value *CodeGen_LLVM::create_malloc(llvm::Type *etype, llvm::Value *size,
                                         bool zero_initialize,
                                         const std::string &name) {
    // Every heap allocation the backend makes comes through here, which is
    // what makes this the place to refuse one. See CompilerOptions::no_heap:
    // nothing frees these, so an allocation reached repeatedly is a leak, and
    // a program that wants the guarantee should be told at compile time rather
    // than discovering it as a growing resident set.
    if (no_heap) {
        internal_error
            << "--no-heap: this program allocates on the heap"
            << (name.empty() ? std::string() : " for `" + name + "`")
            << ". Nothing frees it, so an allocation reached more than once is"
               " a leak. An array whose size is known should be a local or an"
               " extern the caller owns; a result that is returned should be"
               " written into storage the caller passes in.";
    }

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
    if (type.is<Struct_t, Tuple_t, Option_t>()) {
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

    // One index per lane: 32-bit addressing (see element_addresses).
    if (llvm::isa<llvm::VectorType>(idx->getType())) {
        return element_addresses(load_type, base_addr, idx, buffer + "_ptrs");
    }

    // Promote a scalar index to 64-bit on targets that use 64-bit pointers.
    // A GEP sign-extends its index to the pointer's width anyway, so this
    // changes nothing about the address; it is only kept for the IR to read
    // as it always has.
    if (d.getPointerSize() == 8) {
        // TODO: is isSigned always true for us?
        idx = builder->CreateIntCast(idx, llvm::Type::getInt64Ty(*context),
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
    internal_assert(frame_value.has_value())
        << name << " is not bound"
        << (current_function
                ? " in " + current_function->getName().str()
                : std::string());
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
