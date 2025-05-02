#include "Lower/Mutability.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

// Mutable things become ptrs, arguments to mutable slots become references.
struct RewriteMutables : public ir::Mutator {
    const std::set<std::string> &mut_args;
    const std::set<std::string> &immut_args;
    std::set<std::string> mut_locals;
    const ir::FuncMap &funcs;

    RewriteMutables(const std::set<std::string> &mut_args,
                    const std::set<std::string> &immut_args,
                    const ir::FuncMap &funcs)
        : mut_args(mut_args), immut_args(immut_args), funcs(funcs) {}

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
};

} // namespace

ir::FuncMap Mutability::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        // First rewrite calls and derefs.
        func->body =
            RewriteMutables(func->mutable_args(), func->immutable_args(), funcs)
                .mutate(func->body);
        // Then rewrite function signature.
        for (auto &arg_sig : func->args) {
            if (arg_sig.mutating || arg_sig.type.is<ir::Struct_t>()) {
                arg_sig.type = ir::Ptr_t::make(arg_sig.type);
            }
        }
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
