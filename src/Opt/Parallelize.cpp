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
std::string unique_ctx_name() { return "_ctx" + std::to_string(ctx_counter++); }

size_t func_counter = 0;
std::string unique_func_name() {
    return "_parfunc" + std::to_string(func_counter++);
}

struct Closure {
    std::shared_ptr<Function> func;
    Expr context;
};

Stmt replace_reads_and_writes(const WriteLoc &ctx,
                              const std::map<std::string, Expr> &repls,
                              const Stmt &orig) {
    struct Replacer : public Mutator {
        const WriteLoc &ctx;
        const std::map<std::string, Expr> &repls;

        Replacer(const WriteLoc &ctx, const std::map<std::string, Expr> &repls)
            : ctx(ctx), repls(repls) {}

        Expr visit(const Var *node) override {
            const auto &iter = repls.find(node->name);
            if (iter != repls.cend()) {
                return iter->second;
            } else {
                return node;
            }
        }

        std::pair<WriteLoc, bool>
        mutate_writeloc(const WriteLoc &loc) override {
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
    Replacer replacer(ctx, repls);
    return replacer.mutate(orig);
}

Closure build_closure(const ForAll *forall, TypeMap &types) {
    // TODO: might be able to optimize this with LICM or something.
    std::vector<TypedVar> vars = gather_free_vars(forall);
    // TODO(ajr): if struct supported mutable fields, we would need this.
    std::set<std::string> mut_vars = mutated_variables(forall);

    std::string ctx_name = unique_ctx_name();
    Type ctx_t = Struct_t::make(ctx_name, vars);
    types[ctx_name] = ctx_t;
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
    stmts[1] = replace_reads_and_writes(WriteLoc(ctx_name, ctx_t), repls,
                                        forall->body);
    stmts[2] = Return::make();
    Stmt body = Sequence::make(std::move(stmts));

    std::string func = unique_func_name();
    Closure closure;
    closure.context = ctx;
    closure.func = std::make_shared<Function>(
        std::move(func), std::move(f_args), Void_t::make(), std::move(body),
        Function::InterfaceList{}, std::vector<Function::Attribute>{});
    return closure;
}

} // namespace

Stmt parallelize_forall(const std::string &loop_idx, Stmt body, FuncMap &funcs,
                        TypeMap &types) {
    struct ParallelizeForAll : public Mutator {
        const std::string &loop_idx;
        FuncMap &funcs;
        TypeMap &types;

        ParallelizeForAll(const std::string &loop_idx, FuncMap &funcs,
                          TypeMap &types)
            : loop_idx(loop_idx), funcs(funcs), types(types) {}

        Stmt visit(const ForAll *node) override {
            if (node->index != loop_idx) {
                return Mutator::visit(node);
            }
            // TODO: this closure is somewhat GCD-specific, maybe generalize?
            Closure closure = build_closure(node, types);

            auto [_, inserted] =
                funcs.try_emplace(closure.func->name, closure.func);
            internal_assert(inserted);

            Expr n =
                ((node->slice.end - node->slice.begin) +
                 (node->slice.stride - make_one(node->slice.stride.type()))) /
                node->slice.stride;
            n = Simplify::simplify(n);
            std::vector<Expr> args = {closure.context};
            std::vector<Stmt> seq(2);
            seq[0] = Allocate::make(WriteLoc("ctx", closure.context.type()),
                                    closure.context, Allocate::Memory::Stack);
            seq[1] = Launch::make(
                closure.func->name, n,
                {Var::make(Ptr_t::make(closure.context.type()), "ctx")});
            return Sequence::make(std::move(seq));
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_array")) {
                    funcs[var->name]->body = parallelize_forall(
                        loop_idx, std::move(funcs[var->name]->body), funcs,
                        types);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    ParallelizeForAll par(loop_idx, funcs, types);
    return par.mutate(std::move(body));
}

} // namespace opt
} // namespace bonsai
