#include "Lower/Mutability.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

template <typename T>
std::pair<std::vector<T>, bool> visit_list(ir::Mutator *v,
                                           const std::vector<T> &l) {
    bool not_changed = true;
    const size_t n = l.size();
    std::vector<T> new_l(n);
    for (size_t i = 0; i < n; i++) {
        new_l[i] = v->mutate(l[i]);
        not_changed = not_changed && new_l[i].same_as(l[i]);
    }
    return {std::move(new_l), not_changed};
}

// Mutable things become ptrs, arguments to mutable slots become references.
struct RewriteMutables : public ir::Mutator {
    std::set<std::string> mut_args;
    std::set<std::string> immut_args;
    std::set<std::string> mut_locals;

    RewriteMutables(std::set<std::string> mut_args,
                    std::set<std::string> immut_args)
        : mut_args(std::move(mut_args)), immut_args(std::move(immut_args)) {}

    ir::Expr visit(const ir::Var *node) override {
        if (mut_args.contains(node->name) || mut_locals.contains(node->name)) {
            ir::Expr var =
                ir::Var::make(ir::Ptr_t::make(node->type), node->name);
            return ir::Deref::make(std::move(var));
        }

        if (immut_args.contains(node->name) && node->type.is<ir::Struct_t>()) {
            ir::Expr var =
                ir::Var::make(ir::Ptr_t::make(node->type), node->name);
            return ir::Deref::make(std::move(var));
        }
        return node;
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        ir::Type type = mutate(node->type);
        if (value.same_as(node->value) && type.same_as(node->type)) {
            return node;
        }
        return ir::Cast::make(std::move(type), std::move(value));
    }

    ir::Expr visit(const ir::Build *node) override {
        auto [values, not_changed] = visit_list(this, node->values);
        ir::Type type = mutate(node->type);
        if (not_changed && type.same_as(node->type)) {
            return node;
        }
        return ir::Build::make(std::move(type), std::move(values));
    }

    struct ArgsMutate {
        std::vector<ir::Expr> args;
        bool changed;
        bool rewrote_mut;
    };

    ArgsMutate mutate_args(const ir::Function_t *func_t,
                           const std::vector<ir::Expr> &args) {
        const size_t n = args.size();
        internal_assert(func_t->arg_types.size() == n);

        ArgsMutate ret;
        ret.changed = false;
        ret.rewrote_mut = false;
        ret.args.reserve(n);

        for (size_t i = 0; i < n; i++) {
            ir::Expr arg = mutate(args[i]);
            if (func_t->arg_types[i].is_mutable ||
                func_t->arg_types[i].type.is<ir::Struct_t>()) {
                arg = ir::PtrTo::make(std::move(arg));
                ret.rewrote_mut = true;
            }
            ret.changed = ret.changed || !arg.same_as(args[i]);
            ret.args.emplace_back(std::move(arg));
        }
        return ret;
    }

    ir::Type mutate_type(const ir::Function_t *func_t) {
        const size_t n = func_t->arg_types.size();
        std::vector<ir::Function_t::ArgSig> arg_types(n);

        for (size_t i = 0; i < n; i++) {
            arg_types[i].type = (func_t->arg_types[i].is_mutable ||
                                 func_t->arg_types[i].type.is<ir::Struct_t>())
                                    ? ir::Ptr_t::make(func_t->arg_types[i].type)
                                    : func_t->arg_types[i].type;
            arg_types[i].is_mutable = func_t->arg_types[i].is_mutable;
        }
        return ir::Function_t::make(func_t->ret_type, std::move(arg_types));
    }

    template <typename I, typename T>
    I handle(const T *node) {
        // TODO(ajr): do we ever mutate node->func?
        const ir::Function_t *func_t =
            node->func.type().template as<ir::Function_t>();
        internal_assert(func_t);
        auto check = mutate_args(func_t, node->args);
        if (!check.changed) {
            return node;
        }
        if (!check.rewrote_mut) {
            return T::make(node->func, std::move(check.args));
        }
        // Need to change function signature of node->func
        const ir::Var *var = node->func.template as<ir::Var>();
        internal_assert(var);

        ir::Type new_type = mutate_type(func_t);
        ir::Expr func = ir::Var::make(std::move(new_type), var->name);
        return T::make(std::move(func), std::move(check.args));
    }

    ir::Expr visit(const ir::Call *node) override {
        return handle<ir::Expr>(node);
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        return handle<ir::Stmt>(node);
    }

    ir::Stmt visit(const ir::Launch *node) override {
        // Context argument should always be mutable.
        // don't do anything: type is already ptr[struct]
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (!node->mutating) {
            mut_locals.insert(node->loc.base);
        }
        return ir::Mutator::visit(node);
    }

    ir::Type visit(const ir::Array_t *node) override {
        ir::Type etype = mutate(node->etype);
        ir::Expr size = mutate(node->size);
        if (etype.same_as(node->etype) && size.same_as(node->size)) {
            return node;
        }
        return ir::Array_t::make(std::move(etype), std::move(size));
    }

    // Taken from Lower/Options.cpp
    // Also mutates type of write loc.
    std::pair<ir::WriteLoc, bool>
    mutate_writeloc(const ir::WriteLoc &loc) override {
        ir::Type base_type = mutate(loc.base_type);
        bool not_changed = base_type.same_as(loc.base_type);
        ir::WriteLoc new_loc(loc.base, std::move(base_type));

        for (const auto &value : loc.accesses) {
            if (const ir::Expr *expr = std::get_if<ir::Expr>(&value)) {
                ir::Expr new_value = mutate(*expr);
                not_changed = not_changed && new_value.same_as(*expr);
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        return {std::move(new_loc), not_changed};
    }
};

} // namespace

ir::FuncMap Mutability::run(ir::FuncMap funcs,
                            const CompilerOptions &options) const {
    for (auto &[name, func] : funcs) {
        // First rewrite calls and derefs.
        RewriteMutables rewriter(func->mutable_args(), func->immutable_args());

        func->body = rewriter.mutate(std::move(func->body));
        // Then rewrite function signature.
        for (auto &arg_sig : func->args) {
            ir::Type t = arg_sig.type;
            if (arg_sig.mutating || arg_sig.type.is<ir::Struct_t>()) {
                t = ir::Ptr_t::make(arg_sig.type);
            }
            t = rewriter.mutate(std::move(t));
            arg_sig.type = std::move(t);
        }
        func->ret_type = rewriter.mutate(std::move(func->ret_type));
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
