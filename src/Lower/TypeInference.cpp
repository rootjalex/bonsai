#include "Lower/TypeInference.h"

#include "Lower/TopologicalOrder.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Scope.h"
#include "Utils.h"

#include <functional>
#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

ir::Stmt replace_undef_calls(const ir::Stmt &stmt,
                             const ir::TypeMap &func_types) {
    struct ReplaceUndefCalls : public ir::Mutator {
        ReplaceUndefCalls(const ir::TypeMap &func_types)
            : func_types(func_types) {}

      private:
        const ir::TypeMap &func_types;

      public:
        ir::Expr visit(const ir::Var *node) override {
            // TODO: this assumes func names nver conflict with variables, which
            // should be enforced with scopes in the parser. That is not
            // currently the case.
            if (!node->type.defined()) {
                if (func_types.contains(node->name)) {
                    return ir::Var::make(func_types.at(node->name), node->name);
                } else {
                    // TODO: we could error here, but there might be vars that
                    // we just don't know the type of yet. for now, leave this
                    // node, the type might be inferred later (e.g. by return
                    // coercion)
                    return node;
                }
            } else {
                return node;
            }
        }
    };

    ReplaceUndefCalls replacer(func_types);
    return replacer.mutate(stmt);
}

ir::Stmt set_setop_lambda_types(const ir::Stmt &stmt) {
    struct SetLambdaTypes : public ir::Mutator {
        ir::Expr visit(const ir::SetOp *node) override {
            ir::Expr a = mutate(node->a);
            ir::Expr b = mutate(node->b);
            // TODO: could also do the reverse, of the func is labeled but the
            // set type is unknown?
            if (node->op != ir::SetOp::product && !a.type().defined()) {
                // Perform lambda type setting
                internal_assert(b.type().defined() && b.type().is<ir::Set_t>())
                    << "Cannot set lambda type with unknown argument type: "
                    << node;
                internal_assert(a.is<ir::Lambda>())
                    << "Cannot set lambda type if operand is not a lambda: "
                    << node;
                const ir::Lambda *f = a.as<ir::Lambda>();
                // TODO: if this were a func object, this would give us the
                // required return type.
                internal_assert(f->args.size() == 1)
                    << "Expected SetOp lambda to have one argument: " << node;
                const std::string &var_name = f->args[0].name;
                ir::Type var_type = b.type().element_of();
                ir::Expr new_var = ir::Var::make(var_type, var_name);
                ir::Expr new_lambda_expr = replace(var_name, new_var, f->value);
                ir::Expr new_lambda =
                    ir::Lambda::make({{var_name, std::move(var_type)}},
                                     std::move(new_lambda_expr));
                return ir::SetOp::make(node->op, new_lambda, std::move(b));
            } else if (a.same_as(node->a) && b.same_as(node->b)) {
                return node;
            } else {
                return ir::SetOp::make(node->op, std::move(a), std::move(b));
            }
        }
    };

    SetLambdaTypes setter;
    return setter.mutate(stmt);
}

ir::Stmt coerce_return_types(const ir::Stmt &stmt, const ir::Type &ret_type) {
    struct CoerceReturnTypes : public ir::Mutator {
        CoerceReturnTypes(const ir::Type &_ret_type) : ret_type(_ret_type) {}

      private:
        const ir::Type &ret_type;

      public:
        ir::Stmt visit(const ir::Return *node) override {
            // TODO: may need to back-propagate information to variable
            // declarations...
            if (!node->value.type().defined()) {
                // TODO: support is_castable!
                if (is_const(node->value)) {
                    return ir::Return::make(
                        constant_cast(ret_type, node->value));
                } else {
                    internal_assert(node->value.is<ir::Build>())
                        << "Cannot coerce value: " << node->value
                        << " into return type: " << ret_type;
                    ir::Expr new_value = ir::Build::make(
                        ret_type, node->value.as<ir::Build>()->values);
                    internal_assert(new_value.type().defined() &&
                                    ir::equals(ret_type, new_value.type()));
                    return ir::Return::make(std::move(new_value));
                }
            } else if (!ir::equals(node->value.type(), ret_type)) {
                // TODO: check is_castable? The below might fail horrendously or
                // silently...
                ir::Expr new_value = ir::Cast::make(ret_type, node->value);
                return ir::Return::make(std::move(new_value));
            } else {
                return node; // no need to recurse past this point
            }
        }
    };

    internal_assert(ret_type.defined())
        << "Cannot coerce with undefined return type: " << stmt;

    if (ret_type.is<ir::Void_t>()) {
        return stmt;
    }

    CoerceReturnTypes coercer(ret_type);
    return coercer.mutate(stmt);
}

bool has_undef_expr_types(const ir::Stmt &stmt) {
    // We use Mutator to override mutate() instead of all Expr methods in
    // Visitor
    struct FindUndefTypes : public ir::Mutator {
        bool undef_types = false;
        ir::Expr mutate(const ir::Expr &expr) override {
            undef_types = undef_types || !expr.type().defined();
            if (!expr.type().defined()) {
                std::cerr << "Undefined expr: " << expr << std::endl;
            }
            // TODO: record all undefined exprs for error message?
            if (undef_types) {
                return expr;
            } else {
                return Mutator::mutate(expr);
            }
        }

        ir::Stmt mutate(const ir::Stmt &stmt) override {
            if (undef_types) {
                return stmt;
            } else {
                return Mutator::mutate(stmt);
            }
        }

        ir::Stmt visit(const ir::LetStmt *node) override {
            undef_types = undef_types || !node->value.type().defined();
            if (undef_types) {
                return node;
            } else {
                return Mutator::visit(node);
            }
        }

        ir::Stmt visit(const ir::Assign *node) override {
            bool before = undef_types;
            undef_types = undef_types || !node->value.type().defined();
            undef_types = undef_types || !node->loc.type.defined();
            for (const auto &value : node->loc.accesses) {
                if (std::holds_alternative<ir::Expr>(value)) {
                    undef_types = undef_types ||
                                  !std::get<ir::Expr>(value).type().defined();
                }
            }
            if (undef_types) {
                if (!before) {
                    std::cerr << "undefined types detected: " << ir::Stmt(node)
                              << std::endl;
                }
                return node;
            } else {
                return Mutator::visit(node);
            }
        }

        ir::Stmt visit(const ir::Accumulate *node) override {
            bool before = undef_types;
            undef_types = undef_types || !node->value.type().defined();
            undef_types = undef_types || !node->loc.type.defined();
            for (const auto &value : node->loc.accesses) {
                if (std::holds_alternative<ir::Expr>(value)) {
                    undef_types = undef_types ||
                                  !std::get<ir::Expr>(value).type().defined();
                }
            }
            if (undef_types) {
                if (!before) {
                    std::cerr << "undefined types detected: " << ir::Stmt(node)
                              << std::endl;
                }
                return node;
            } else {
                return Mutator::visit(node);
            }
        }
    };

    FindUndefTypes finder;
    finder.mutate(stmt);
    return finder.undef_types;
}

ir::Stmt infer_types(ir::Stmt stmt, const ir::TypeMap &func_types) {
    // First, try to use function types inferred so far to replace undefined
    // call sites.
    stmt = replace_undef_calls(stmt, func_types);
    // Use known semantics of set operations to set lambda argument types.
    stmt = set_setop_lambda_types(stmt);
    return stmt;
}

std::shared_ptr<ir::Function>
infer_types(const std::shared_ptr<ir::Function> &fnotypes,
            const ir::Program &program, const ir::TypeMap &func_types) {
    auto ftypes = std::make_shared<ir::Function>();
    ftypes->name = fnotypes->name;
    ftypes->args = fnotypes->args;

    ftypes->body = infer_types(fnotypes->body, func_types);

    // If we know the return type (due to annotations), try to coerce all
    // returns to it. If we don't know from annotations, try to infer from some
    // return type, then coerce.
    ftypes->ret_type = fnotypes->ret_type.defined()
                           ? fnotypes->ret_type
                           : ir::get_return_type(ftypes->body);
    ftypes->body = coerce_return_types(ftypes->body, ftypes->ret_type);

    internal_assert(ftypes->ret_type.is<ir::Void_t>() ||
                    always_returns(ftypes->body))
        << "Function: " << ftypes->name
        << " does not return in all code paths.";

    ftypes->interfaces = fnotypes->interfaces;

    // TODO: is there more that we can do?

    internal_assert(!has_undef_expr_types(ftypes->body))
        << "Type inference failed to infer all types of:\n"
        << fnotypes->body << "\n\nInferred:\n"
        << ftypes->body;
    return ftypes;
}

} // namespace

ir::Program infer_types(const ir::Program &program) {
    ir::Program new_program;
    new_program.externs = program.externs;
    new_program.types = program.types;
    ir::global_enable_type_enforcement();

    std::vector<std::string> topo_order =
        func_topological_order(program.funcs, /*undef_calls=*/true);
    ir::TypeMap func_types;

    // TODO: set all assignment types.

    for (const auto &f : topo_order) {
        new_program.funcs[f] =
            infer_types(program.funcs.at(f), new_program, func_types);
        {
            const size_t n_args = new_program.funcs[f]->args.size();
            std::vector<ir::Type> arg_types(n_args);
            for (size_t i = 0; i < n_args; i++) {
                arg_types[i] = new_program.funcs[f]->args[i].type;
            }
            func_types[f] =
                ir::Function_t::make(new_program.funcs[f]->ret_type, arg_types);
        }
    }

    // std::cout << "\n\nInferred types:\n";
    // new_program.dump(std::cout);

    return new_program;
}

} // namespace lower
} // namespace bonsai
