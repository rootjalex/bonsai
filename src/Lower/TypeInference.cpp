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
        ir::TypeMap var_types;

      public:
        ir::Expr visit(const ir::Var *node) override {
            // TODO: this assumes func names nver conflict with variables, which
            // should be enforced with scopes in the parser. That is not
            // currently the case.
            if (!node->type.defined()) {
                if (func_types.contains(node->name)) {
                    return ir::Var::make(func_types.at(node->name), node->name);
                } else if (var_types.contains(node->name)) {
                    // The type of this node has been inferred previously.
                    // TODO: make sure scope is handled correctly?
                    // Can't have shadowing, so maybe this is fine?
                    return ir::Var::make(var_types.at(node->name), node->name);
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

        ir::Stmt visit(const ir::LetStmt *node) override {
            ir::Stmt mut = ir::Mutator::visit(node);
            if (mut.get() == node) {
                return node;
            }
            // Was modified.
            const ir::LetStmt *repl = mut.as<ir::LetStmt>();
            internal_assert(repl);
            if (!node->value.type().defined() && repl->value.type().defined()) {
                // Insert inferred var type for later references.
                internal_assert(repl->loc.accesses.empty());
                var_types[repl->loc.base] = repl->value.type();
            }
            return mut;
        }
        // TODO(ajr): replace Assign/Accumulate if undef as well!
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
                internal_assert(b.type().defined() &&
                                (b.type().is<ir::Set_t, ir::Array_t>()))
                    << "Cannot set lambda type with unknown argument type: "
                    << ir::Expr(node) << ", and b type: " << b.type();
                ir::Type etype = b.type().element_of();
                internal_assert(a.is<ir::Lambda>())
                    << "Cannot set lambda type if operand is not a lambda: "
                    << node;
                // TODO: if this were a func object, this would give us the
                // required return type.
                const ir::Lambda *f = a.as<ir::Lambda>();
                const size_t expected_args =
                    etype.is<ir::Tuple_t>()
                        ? etype.as<ir::Tuple_t>()->etypes.size()
                        : 1;
                internal_assert(f->args.size() == expected_args)
                    << "Expected SetOp lambda to have: " << expected_args
                    << " argument(s)"
                    << " but has " << f->args.size()
                    << " argument(s): " << ir::Expr(node);

                std::map<std::string, ir::Expr> replacements;
                std::vector<ir::Lambda::Argument> lambda_args(expected_args);
                for (size_t i = 0; i < expected_args; i++) {
                    ir::Type type = etype.is<ir::Tuple_t>()
                                        ? etype.as<ir::Tuple_t>()->etypes[i]
                                        : etype;
                    replacements[f->args[i].name] =
                        ir::Var::make(type, f->args[i].name);
                    lambda_args[i] =
                        ir::Lambda::Argument{f->args[i].name, std::move(type)};
                }

                ir::Expr new_lambda_expr =
                    replace(std::move(replacements), f->value);
                ir::Expr new_lambda = ir::Lambda::make(
                    std::move(lambda_args), std::move(new_lambda_expr));
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
            if (!node->value.defined()) {
                return node;
            }
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
                std::cerr << "Undefined type on expr: " << expr << std::endl;
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
            if (!node->value.type().defined()) {
                std::cerr << "Undefined type on expr in Let: " << node->value
                          << std::endl;
            }
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
    ftypes->is_export = fnotypes->is_export;

    // If we know the return type (due to annotations), try to coerce all
    // returns to it. If we don't know from annotations, try to infer from some
    // return type, then coerce.
    ftypes->ret_type = fnotypes->ret_type.defined()
                           ? fnotypes->ret_type
                           : ir::get_return_type(ftypes->body);
    internal_assert(ftypes->ret_type.defined())
        << "Failed to infer return type of: " << ftypes->name
        << ", with body: " << ftypes->body;
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

// TODO: this is a lowering pass like any other, we should treat it as so.
ir::Program infer_types(const ir::Program &program) {
    ir::Program new_program;
    new_program.externs = program.externs;
    new_program.types = program.types;
    new_program.schedules = program.schedules;
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
