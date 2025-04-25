#include "Opt/CSE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
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

//  let _t0 = a + b in
//  let _t1 = a + b in
//  in f(_t0, _t1)
//  ->
//  let _t0 = a + b in
//  let c = _t1 in
//  in f(_t0, _t0)
class CopyPropagation : public ir::Mutator {
  public:
    CopyPropagation(const std::set<std::string> &mutable_arguments)
        : mutable_arguments(mutable_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        internal_assert(value.defined());
        const auto *variable = value.as<ir::Var>();
        ir::WriteLoc lhs = node->loc;
        if (variable == nullptr) {
            // This is an expression, do nothing.
            return ir::LetStmt::make(lhs, std::move(value));
        }

        auto it = variable_to_initial.find(variable->name);
        if (it == variable_to_initial.end()) {
            variable_to_initial[lhs.base] = variable->name;
            // Point this variable to the current variable.
            return node;
        }
        // Point this variable to the previously found variable.
        std::string previous = it->second;
        return ir::LetStmt::make(lhs,
                                 ir::Var::make(variable->type, it->second));
    }

    // Cannot propagate copies through mutable variables.
    ir::Stmt visit(const ir::Assign *node) override { return node; }

    // Don't propagate through lambda bodies.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

    ir::Expr visit(const ir::Var *node) override {
        if (mutable_arguments.contains(node->name)) {
            return node;
        }
        auto it = variable_to_initial.find(node->name);
        if (it == variable_to_initial.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

  private:
    const std::set<std::string> &mutable_arguments;
    std::unordered_map<std::string, std::string> variable_to_initial;
};

// Validates whether the visited expression can undergo CSE.
struct IsCseLegal : public ir::Visitor {
    IsCseLegal(const std::set<std::string> &side_effect_functions,
               const std::set<std::string> &mutable_function_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_variables(mutable_function_arguments) {}

    void visit(const ir::Var *node) override {
        is_legal &= !mutable_variables.contains(node->name);
    }

    void visit(const ir::Call *node) override {
        const auto *v = node->func.as<ir::Var>();
        if (v == nullptr) {
            return;
        }
        is_legal &= !side_effect_functions.contains(v->name);
    }

    bool is_legal = true;
    const std::set<std::string> &side_effect_functions;
    const std::set<std::string> &mutable_variables;
};

class CseImpl : public ir::Mutator {
  public:
    CseImpl(const std::set<std::string> &side_effect_functions,
            const std::set<std::string> &mutable_function_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_variables(mutable_function_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr variable = get(node->value);
        if (variable.defined()) {
            // This expression already exists.
            return ir::LetStmt::make(node->loc, variable);
        }
        update(node->value, node->loc);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (node->mutating) {
            mutable_variables.insert(node->loc.base);
            return ir::Mutator::visit(node);
        }
        ir::Expr variable = get(node->value);
        if (variable.defined()) {
            // This expression already exists.
            return ir::LetStmt::make(node->loc, variable);
        }
        update(node->value, node->loc);
        return node;
    }

    ir::Stmt visit(const ir::Store *node) override {
        // Conservatively avoid stores until we have a stronger memory analysis.
        mutable_variables.insert(node->name);
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        mutable_variables.insert(node->loc.base);
        return node;
    }

    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = mutate(node->a);
        a = get(node->a);
        ir::Expr b = mutate(node->b);
        b = get(node->b);
        if (a.defined() && b.defined()) {
            return ir::BinOp::make(node->op, std::move(a), std::move(b));
        }
        if (a.defined()) {
            return ir::BinOp::make(node->op, std::move(a), node->b);
        }
        if (b.defined()) {
            return ir::BinOp::make(node->op, node->a, std::move(b));
        }
        return node;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = get(mutate(node->cond));
        if (!cond.defined()) {
            return ir::Mutator::visit(node);
        }
        return ir::IfElse::make(std::move(cond), mutate(node->then_body),
                                mutate(node->else_body));
    }

    // TODO(cgyurgyik): This is probably overly restrictive, since we already
    // restrain CSE from occurring on mutable variables / side-effecting
    // expressions. However, I have no way to unit test this since the parser
    // doesn't support these constructs yet.
    ir::Stmt visit(const ir::ForAll *node) override {
        ScopedValue<bool> guard(allow_cse, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ScopedValue<bool> guard(allow_cse, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        ScopedValue<bool> guard(allow_cse, false);
        return ir::Mutator::visit(node);
    }

  private:
    // Whether to allow CSE to occur. Since we don't have phi instructions, this
    // is false in the presence of control flow divergence with mutable or side
    // effecting values.
    bool allow_cse = true;
    // A list of functions that may have side effects.
    const std::set<std::string> &side_effect_functions;
    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable function arguments, mutable
    // assignments, and references to allocations.
    std::set<std::string> mutable_variables;
    // Maps expressions to the variable of its first occurrence.
    std::map<ir::Expr, ir::Expr, ir::ExprLessThan> expression_to_variable;

    ir::Expr get(ir::Expr value) {
        if (!(allow_cse && is_cse_legal(value))) {
            return ir::Expr();
        }
        auto it = expression_to_variable.find(value);
        if (it == expression_to_variable.end()) {
            return ir::Expr();
        }
        return it->second;
    }

    void update(ir::Expr value, const ir::WriteLoc &location) {
        if (!(allow_cse && is_cse_legal(value))) {
            return;
        }
        ir::Expr v = ir::Var::make(location.base_type, location.base);
        // If this expression hasn't been seen, add it to the list.
        expression_to_variable[value] = std::move(v);
    }

    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        IsCseLegal checker(side_effect_functions, mutable_variables);
        e.accept(&checker);
        return checker.is_legal;
    }
};

} // namespace

ir::FuncMap CSE::run(ir::FuncMap funcs) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_arguments = get_mutable_arguments(*func);
        CseImpl cse(side_effect_functions, mutable_arguments);
        func->body = cse.mutate(std::move(func->body));

        // renaming followed by CSE results in lots of copies. We want to
        // propagate these, and then allow DCE clean up dead assignments
        // afterward.
        CopyPropagation cp(mutable_arguments);
        func->body = cp.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
