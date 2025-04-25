#include "Opt/CSE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

// Mutable variables can be seen multiple times.
using MutableVariableStack =
    ir::SetStack<std::string, std::less<std::string>, true>;
// A stack from expressions to their respective variable.
using CseStack = ir::MapStack<ir::Expr, ir::Expr, ir::ExprLessThan, true>;

class CseImpl : public ir::Mutator {
  public:
    CseImpl(const std::set<std::string> &side_effect_functions,
            const std::set<std::string> &mutable_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::WriteLoc location = node->loc;
        ir::Stmt let = ir::LetStmt::make(location, get(node->value));

        if (is_cse_legal(node->value)) {
            ir::Expr v = ir::Var::make(location.base_type, location.base);
            expression_to_variable.add_to_frame(node->value, std::move(v));
        }

        return let;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (node->mutating) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::Store *node) override {
        mutable_variables.add_to_frame(node->name);
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        mutable_variables.add_to_frame(node->loc.base);
        return node;
    }

    ir::Expr visit(const ir::BinOp *node) override {
        return ir::BinOp::make(node->op, get(node->a), get(node->b));
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = get(node->cond);

        new_frame();
        ir::Stmt th = mutate(node->then_body);
        pop_frame();

        new_frame();
        ir::Stmt el = mutate(node->else_body);
        pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

  private:
    // A list of functions that may have side effects. This is "whole program
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &side_effect_functions;
    // A list of function arguments that are mutable. This is "whole function
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &mutable_arguments;

    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable assignments, and references to
    // allocations.
    MutableVariableStack mutable_variables;
    // Maps expressions to the variable of its first occurrence.
    CseStack expression_to_variable;

    void new_frame() {
        expression_to_variable.new_frame();
        mutable_variables.new_frame();
    }
    void pop_frame() {
        expression_to_variable.pop_frame();
        mutable_variables.pop_frame();
    }

    // Retrieves either the mutate expression or its CSE'd equivalent.
    ir::Expr get(ir::Expr value) {
        value = mutate(std::move(value));
        if (!is_cse_legal(value)) {
            return value;
        }
        if (std::optional<ir::Expr> variable =
                expression_to_variable.from_frames(value)) {
            return *variable;
        }
        return value;
    }

    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        struct CseLegalChecker : public ir::Visitor {
            CseLegalChecker(const std::set<std::string> &side_effect_functions,
                            const std::set<std::string> &mutable_arguments,
                            const MutableVariableStack &mutable_variables)
                : side_effect_functions(side_effect_functions),
                  mutable_arguments(mutable_arguments),
                  mutable_variables(mutable_variables) {}

            void visit(const ir::Var *node) override {
                // We cannot CSE with mutable variables since mutations may have
                // occurred between. In the future, we can rename mutated
                // variables to overcome this. For example, a = x + 1; #1 x +=
                // 1; b = x + 1; #2 (same expression, but x has changed value)
                is_legal &= !mutable_variables.contains(node->name) &&
                            !mutable_arguments.contains(node->name);
            }

            void visit(const ir::Call *node) override {
                // We cannot CSE with side effecting function calls. For
                // example, a = print_and_return(x); #1 b = print_and_return(x);
                // #2 (same, but would only print once)
                const auto *v = node->func.as<ir::Var>();
                if (v == nullptr) {
                    return;
                }
                is_legal &= !side_effect_functions.contains(v->name);
            }

            bool is_legal = true;

          private:
            const std::set<std::string> &side_effect_functions;
            const std::set<std::string> &mutable_arguments;
            const MutableVariableStack &mutable_variables;
        };
        CseLegalChecker checker(side_effect_functions, mutable_arguments,
                                mutable_variables);
        e.accept(&checker);
        return checker.is_legal;
    }
};

} // namespace

// TODO(cgyurgyik): This may uncover unnecessary copies. Consider the
// following code sample transformation:
//
// a: i32 = f(x);
// b: i32 = f(x);
// c: i32 = bar(a, b);
//  ->
// let a: i32 = f(x) in
// let b: i32 = a in       // <- Remove this
// let c: i32 = bar(a, b)  // <- Replace with bar(a, a)
ir::FuncMap CSE::run(ir::FuncMap funcs) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_arguments = get_mutable_arguments(*func);
        CseImpl cse(side_effect_functions, mutable_arguments);
        func->body = cse.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
