#include "Opt/Parallelize.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

using namespace ir;

// TODO: we use this pattern a lot, could make it a helper func.
size_t ctx_counter = 0;
std::string unique_ctx_name() { return "ctx" + std::to_string(ctx_counter++); }

size_t func_counter = 0;
std::string unique_func_name() {
    return "_parfunc" + std::to_string(func_counter++);
}

struct Closure {
    std::shared_ptr<Function> func;
    Expr context;
    // Array variables that are written to.
    std::map<std::string, Type> written;
};

Stmt replace_reads_and_writes(const WriteLoc &ctx,
                              const std::map<std::string, Expr> &repls,
                              const Stmt &orig, Closure &closure) {
    struct Replacer : public Mutator {
        const WriteLoc &ctx;
        const std::map<std::string, Expr> &repls;
        Closure &closure;

        Replacer(const WriteLoc &ctx, const std::map<std::string, Expr> &repls,
                 Closure &closure)
            : ctx(ctx), repls(repls), closure(closure) {}

        Expr visit(const Var *node) override {
            const auto &iter = repls.find(node->name);
            if (iter == repls.cend()) {
                return node;
            }
            return iter->second;
        }

        std::pair<WriteLoc, bool>
        mutate_writeloc(const WriteLoc &loc) override {
            // This should not include writes to scalars, e.g., the thread
            // index calculation.
            if (loc.base_type.is<Array_t>()) {
                closure.written[loc.base] = loc.base_type;
            }
            if (repls.contains(loc.base)) {
                WriteLoc new_loc = ctx;
                new_loc.add_struct_access(loc.base);
                for (const auto &value : loc.accesses) {
                    if (std::holds_alternative<Expr>(value)) {
                        Expr new_value = mutate(std::get<Expr>(value));
                        new_loc.add_index_access(std::move(new_value));
                    } else {
                        new_loc.add_struct_access(std::get<std::string>(value));
                    }
                }
                return {new_loc, /*not_changed=*/false};
            } else {
                return Mutator::mutate_writeloc(loc);
            }
        }
    };
    Replacer replacer(ctx, repls, closure);
    return replacer.mutate(orig);
}

// Builds a closure that greatly resembles what is needed for the Grand Central
// Dispatch (GCD) on MacOS.
// TODO(ajr): this closure is somewhat GCD-specific, maybe generalize?
Closure build_gcd_closure(const ForAll *forall, TypeMap &types) {
    // TODO: might be able to optimize this with LICM or something.
    std::vector<TypedVar> vars = gather_free_vars(forall);
    // TODO(ajr): if struct supported mutable fields, we would need this.
    std::set<std::string> mut_vars = mutated_variables(forall);

    std::string ctx_name = unique_ctx_name();
    Type ctx_t = Struct_t::make("_" + ctx_name, vars);
    types["_" + ctx_name] = ctx_t;
    std::vector<Expr> build_args;
    build_args.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(build_args),
                   [](const TypedVar &v) { return Var::make(v.type, v.name); });
    Expr ctx = Build::make(ctx_t, build_args);
    Expr ctx_var = Var::make(ctx_t, ctx_name);

    std::vector<Function::Argument> f_args(2);
    f_args[0].name = ctx_name;
    f_args[0].type = ctx_t;
    f_args[0].mutating = true;

    Type itype = forall->slice.end.type();
    static std::string parfor_idx = "_parfor_idx";
    static Type idx_t = Int_t::make(64);
    f_args[1].name = parfor_idx;
    f_args[1].type = idx_t; // TODO: does this work always?
    f_args[1].mutating = false;

    std::map<std::string, Expr> repls;
    for (const auto &var : vars) {
        repls[var.name] = Access::make(var.name, ctx_var);
    }

    // Trust simplify() to flatten sequences.
    std::vector<Stmt> stmts(3);
    Expr loop_i = Var::make(idx_t, parfor_idx);
    stmts[0] = LetStmt::make(
        WriteLoc(forall->index, itype),
        cast(itype, cast(idx_t, forall->slice.begin) +
                        cast(idx_t, forall->slice.stride) * loop_i));
    // TODO(ajr): this also needs to replace Stores/Allocates/Accumulates!
    Closure closure;
    stmts[1] = replace_reads_and_writes(WriteLoc(ctx_name, ctx_t), repls,
                                        forall->body, closure);
    stmts[2] = Return::make();
    Stmt body = Sequence::make(std::move(stmts));

    std::string func = unique_func_name();
    closure.context = ctx;
    closure.func = std::make_shared<Function>(
        std::move(func), std::move(f_args), Void_t::make(), std::move(body),
        Function::InterfaceList{},
        std::vector<Function::Attribute>{Function::Attribute::kernel});
    return closure;
}

// Builds a kernel to execute in the CUDA backend.
Closure build_cuda_closure(const ForAll *forall, TypeMap &types) {
    // TODO: might be able to optimize this with LICM or something.
    std::vector<TypedVar> vars = gather_free_vars(forall);
    // TODO(ajr): if struct supported mutable fields, we would need this.
    std::set<std::string> mut_vars = mutated_variables(forall);

    for (int i = 0, e = vars.size(); i < e; ++i) {
        // Structs should be pointers; these will be copied to device.
        ir::Type type = vars[i].type;
        if (type.is<Struct_t>()) {
            vars[i].type = Ptr_t::make(std::move(type));
        }
    }

    std::string ctx_name = unique_ctx_name();
    Type ctx_t = Struct_t::make("_" + ctx_name, vars);

    types[ctx_name] = ctx_t;
    std::vector<Expr> build_args;
    build_args.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(build_args),
                   [](const TypedVar &v) { return Var::make(v.type, v.name); });
    Expr ctx = Build::make(ctx_t, build_args);
    Expr ctx_var = Var::make(ctx_t, ctx_name);

    std::vector<Function::Argument> f_args;
    f_args.push_back(Function::Argument(
        /*name=*/ctx_name,
        /*type=*/ctx_t,
        /*default_value=*/Expr(),
        /*mutating=*/true));

    Expr b = forall->slice.begin;
    Expr e = forall->slice.end;
    Expr s = forall->slice.stride;

    Type idx_t = forall->index_type();
    // TOOD(cgyurgyik): hacky; make this work for multiple dimensions.
    static Expr bidx = Var::make(idx_t, "blockIdx.x");
    static Expr bdim = Var::make(idx_t, "blockDim.x");
    static Expr tidx = Var::make(idx_t, "threadIdx.x");
    static Expr tid = Var::make(idx_t, "tid");

    std::vector<Stmt> stmts;
    // tid = blockIdx * blockDim + threadIdx;
    // Note: this should *always* be the first statement; RNG depends on this
    // location (unfortunately).
    stmts.push_back(
        LetStmt::make(WriteLoc("tid", idx_t), (bidx * bdim + tidx)));
    // index = begin + tid * stride;
    stmts.push_back(LetStmt::make(WriteLoc(forall->index, idx_t), b + tid * s));
    // if (index >= end) { return; }
    Expr idx = Var::make(idx_t, forall->index);
    stmts.push_back(IfElse::make(idx >= e, Return::make()));
    stmts.push_back(forall->body);
    stmts.push_back(Return::make());

    // Replace reads and writes to access the context, e.g., x -> ctx.x
    std::map<std::string, Expr> repls;
    for (const auto &var : vars) {
        ir::Expr value = Access::make(var.name, ctx_var);
        ir::Type type = value.type();
        if (type.is<Ptr_t>() && type.element_of().is<Struct_t>()) {
            value = Deref::make(std::move(value));
        }
        repls[var.name] = value;
    }

    Closure closure;
    Stmt body = Sequence::make(std::move(stmts));
    body = replace_reads_and_writes(WriteLoc(ctx_name, ctx_t), repls, body,
                                    closure);

    std::string name = unique_func_name();
    closure.context = ctx;
    closure.func = std::make_shared<Function>(
        std::move(name), std::move(f_args), Void_t::make(), std::move(body),
        Function::InterfaceList{},
        std::vector<Function::Attribute>{Function::Attribute::kernel});
    return closure;
}

// Builds necessary epilogue, launch, and prologue for a CUDA kernel launch.
Stmt launch_cuda(const ForAll *node, const Closure &closure) {
    // Allocate and copy members of the context to device.
    Type closure_type = closure.context.type();
    const auto *build = closure.context.as<Build>();
    internal_assert(build) << closure.context;
    std::vector<Expr> to_device;
    std::vector<Stmt> stmts;
    for (const Expr &value : build->values) {
        const auto *v = value.as<Var>();
        internal_assert(v) << "unexpected context argument: " << value;
        if (closure.written.contains(v->name)) {
            // Allocations are automatically cudaMalloc'ed.
            to_device.push_back(value);
            continue;
        }
        ir::Type type = value.type();
        if (type.is<Ptr_t>()) {
            std::string device_name = "d_" + v->name;
            stmts.push_back(Allocate::make(WriteLoc(device_name, type), value,
                                           Allocate::Memory::Device));
            to_device.push_back(Var::make(type, device_name));
            continue;
        }
        if (type.is<Struct_t, Array_t>()) {
            std::string device_name = "d_" + v->name;
            stmts.push_back(Allocate::make(WriteLoc(device_name, type), value,
                                           Allocate::Memory::Device));
            to_device.push_back(Var::make(type, device_name));
            continue;
        }
        if (type.is_scalar()) {
            to_device.push_back(value);
            continue;
        }
        internal_error << "[unimplemented] handling context argument: " << value
                       << " : " << type;
    }
    Expr device_build = ir::Build::make(build->type, to_device);

    stmts.push_back(Allocate::make(WriteLoc("ctx", closure_type), device_build,
                                   Allocate::Memory::Stack));

    Type idx_t = node->index_type();
    Expr b = node->slice.begin, e = node->slice.end, s = node->slice.stride;
    Expr n = node->count();
    stmts.push_back(Launch::make(
        closure.func->name, n, {Var::make(Ptr_t::make(closure_type), "ctx")}));

    for (const Expr &value : to_device) {
        const auto *v = value.as<Var>();
        internal_assert(v) << "unexpected context argument: " << value;
        ir::Type type = value.type();
        if (type.is<Ptr_t>()) {
            type = type.element_of();
        }
        if (type.is<Array_t, Struct_t>()) {
            if (closure.written.contains(v->name)) {
                std::string host_name = "h_" + v->name;
                stmts.push_back(Allocate::make(WriteLoc(host_name, type), value,
                                               Allocate::Memory::Host));
            }
            stmts.push_back(Free::make(value));
            continue;
        }
        if (type.is_scalar()) {
            continue;
        }
        internal_error << "[unimplemented] handling context argument: " << value
                       << " : " << value.type();
    }
    internal_assert(closure.written.size() == 1)
        << "[unimplemented]: multiple writes in the closure";
    for (const auto &[name, type] : closure.written) {
        stmts.push_back(
            Store::make(WriteLoc(name, type), Var::make(type, "h_" + name)));
    }

    return Sequence::make(std::move(stmts));
}

} // namespace

Stmt parallelize_forall(const std::string &loop_idx, Stmt body,
                        Program &program, const CompilerOptions &options) {
    struct ParallelizeForAll : public Mutator {
        const std::string &loop_idx;
        const CompilerOptions &options;
        Program &program;

        ParallelizeForAll(const std::string &loop_idx,
                          const CompilerOptions &options, Program &program)
            : loop_idx(loop_idx), options(options), program(program) {}

        Stmt visit(const ForAll *node) override {
            if (node->index != loop_idx) {
                return Mutator::visit(node);
            }

            switch (options.target) {
            case BackendTarget::CUDA: {
                Closure closure = build_cuda_closure(node, program.types);
                auto [_, inserted] =
                    program.funcs.try_emplace(closure.func->name, closure.func);
                internal_assert(inserted) << closure.func;
                return launch_cuda(node, closure);
            }
            default: {
                Closure closure = build_gcd_closure(node, program.types);
                auto [_, inserted] =
                    program.funcs.try_emplace(closure.func->name, closure.func);
                internal_assert(inserted) << closure.func;
                Expr b = node->slice.begin, e = node->slice.end,
                     s = node->slice.stride;
                Expr n = node->count();
                n = Simplify::simplify(n);
                std::vector<Stmt> seq(2);
                seq[0] =
                    Allocate::make(WriteLoc("ctx", closure.context.type()),
                                   closure.context, Allocate::Memory::Stack);
                seq[1] = Launch::make(
                    closure.func->name, n,
                    {Var::make(Ptr_t::make(closure.context.type()), "ctx")});
                return Sequence::make(std::move(seq));
            }
            }
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_array")) {
                    program.funcs[var->name]->body = parallelize_forall(
                        loop_idx, std::move(program.funcs[var->name]->body),
                        program, options);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    ParallelizeForAll par(loop_idx, options, program);
    return par.mutate(std::move(body));
}

} // namespace opt
} // namespace bonsai
