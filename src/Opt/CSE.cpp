#include "Opt/CSE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

// Stack of mutable variable names for a given function.
using MutableVariableStack = ir::SetStack<std::string>;

class CseImpl : public ir::Mutator {
  public:
    CseImpl(const std::set<std::string> &side_effect_functions,
            const std::set<std::string> &mutable_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        const std::optional<int64_t> vn = get_value_number(value);
        if (!vn.has_value()) {
            return node; // We cannot legally CSE this.
        }
        if (std::optional<std::string> name = vn_to_var.from_frames(*vn)) {
            ir::Expr v = ir::Var::make(value.type(), *name);
            var_to_e.add_to_frame(node->loc.base, v);
            return ir::LetStmt::make(node->loc, std::move(v));
        }
        var_to_e.add_to_frame(node->loc.base, std::move(value));
        vn_to_var.add_to_frame(*vn, node->loc.base);
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (node->mutating && !mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return node;
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!mutable_variables.contains(node->name)) {
            mutable_variables.add_to_frame(node->name);
        }
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return node;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        new_frame();
        ir::Stmt th = mutate(node->then_body);
        pop_frame();
        new_frame();
        ir::Stmt el = mutate(node->else_body);
        pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    // Skip statements we cannot unit test.
    ir::Stmt visit(const ir::ForAll *node) override { return node; }
    ir::Stmt visit(const ir::ForEach *node) override { return node; }
    ir::Stmt visit(const ir::DoWhile *node) override { return node; }

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

    // The "local value number" for expressions in this function. We just use
    // the same number in conjunction with a stack to ensure the values remain
    // truly local to their scope.
    int64_t local_value_number = 0;
    // expression -> value number
    ir::MapStack<ir::Expr, int64_t, ir::ExprLessThan> e_to_vn;
    // variable -> expression
    ir::MapStack<std::string, ir::Expr> var_to_e;
    // value number -> variable (for subsequent replacement)
    ir::MapStack<int64_t, std::string> vn_to_var;

    void new_frame() {
        mutable_variables.new_frame();
        e_to_vn.new_frame();
        var_to_e.new_frame();
        vn_to_var.new_frame();
    }
    void pop_frame() {
        mutable_variables.pop_frame();
        e_to_vn.pop_frame();
        var_to_e.pop_frame();
        vn_to_var.pop_frame();
    }

    // Returns the lcoal value number for expression `e` if this is legal to
    // CSE, and {} otherwise.
    std::optional<int64_t> get_value_number(ir::Expr e) {
        if (!is_cse_legal(e)) {
            return {};
        }
        // TODO(cgyurgyik): Probably other expressions this should apply?
        if (const auto *op = e.as<ir::BinOp>()) {
            e = ir::BinOp::make(op->op, substitute(op->a), substitute(op->b));
        }
        if (const auto *op = e.as<ir::Intrinsic>()) {
            std::vector<ir::Expr> args = op->args;
            for (int i = 0, e = args.size(); i < e; ++i) {
                args[i] = substitute(args[i]);
            }
            e = ir::Intrinsic::make(op->op, std::move(args));
        }
        if (const auto *op = e.as<ir::Call>()) {
            std::vector<ir::Expr> args = op->args;
            for (int i = 0, e = args.size(); i < e; ++i) {
                args[i] = substitute(args[i]);
            }
            e = ir::Call::make(op->func, std::move(args));
        }
        if (std::optional<int64_t> vn = e_to_vn.from_frames(e)) {
            return *vn;
        }
        const int64_t vn = local_value_number++;
        e_to_vn.add_to_frame(e, vn);
        return vn;
    }

    // For simplicity, we replace intermediate variables with their value
    // expression. There might be a better way to do this, but this is how I've
    // implemented LVN for SSA, where use-def chains are immediately available.
    ir::Expr substitute(ir::Expr e) {
        const auto *v = e.as<ir::Var>();
        if (v == nullptr) {
            return e;
        }
        if (std::optional<ir::Expr> f = var_to_e.from_frames(v->name)) {
            return substitute(*f);
        }
        return e;
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

// Implements copy propagation, e.g.,
//  let _t0 = a + b in
//  let _t1 = a + b in
//  in f(_t0, _t1)
//  ->
//  let _t0 = a + b in
//  let _t1 = _t0 in
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
            return ir::LetStmt::make(std::move(lhs), std::move(value));
        }
        internal_assert(!lhs_to_rhs.contains(lhs.base));
        lhs_to_rhs.add_to_frame(lhs.base, variable->name);
        return ir::LetStmt::make(std::move(lhs), std::move(value));
    }

    ir::Expr visit(const ir::Var *node) override {
        if (mutable_arguments.contains(node->name)) {
            return node;
        }
        std::optional<std::string> name = lhs_to_rhs.from_frames(node->name);
        if (!name.has_value()) {
            return node;
        }
        return ir::Var::make(node->type, *name);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        lhs_to_rhs.new_frame();
        ir::Stmt th = mutate(node->then_body);
        lhs_to_rhs.pop_frame();
        lhs_to_rhs.new_frame();
        ir::Stmt el = mutate(node->else_body);
        lhs_to_rhs.pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    // Cannot propagate copies through mutable variables.
    ir::Stmt visit(const ir::Assign *node) override { return node; }
    ir::Stmt visit(const ir::Accumulate *node) override { return node; }
    // Don't propagate through lambda bodies.
    ir::Expr visit(const ir::Lambda *node) override { return node; }
    // Skip statements we cannot unit test.
    ir::Stmt visit(const ir::ForAll *node) override { return node; }
    ir::Stmt visit(const ir::ForEach *node) override { return node; }
    ir::Stmt visit(const ir::DoWhile *node) override { return node; }

  private:
    const std::set<std::string> &mutable_arguments;
    // A mapping from the lhs to rhs assignment of variable names, e.g.,
    //   x: i32 = y; // {x, y}
    //   z: i32 = x; // {z, x}
    ir::MapStack<std::string, std::string> lhs_to_rhs;
};

// Validates whether the visited expression can undergo CSE.
struct IsCseLegal : public ir::Visitor {
    IsCseLegal(const std::set<std::string> &side_effect_functions,
               const std::set<std::string> &mutable_function_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_variables(mutable_function_arguments) {}

    void visit(const ir::Var *node) override {
        // We cannot CSE with mutable variables since mutations may have
        // occurred between. In the future, we can rename mutated variables to
        // overcome this. For example,
        // a = x + 1; #1
        // x += 1;
        // b = x + 1; #2 (same expression, but x has changed value)
        is_legal &= !mutable_variables.contains(node->name);
    }

    void visit(const ir::Call *node) override {
        // We cannot CSE with side effecting function calls. For example,
        // a = print_and_return(x); #1
        // b = print_and_return(x); #2 (same, but would only print once)
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

} // namespace

ir::FuncMap CSE::run(ir::FuncMap funcs) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_arguments = get_mutable_arguments(*func);
        CseImpl cse(side_effect_functions, mutable_arguments);
        func->body = cse.mutate(std::move(func->body));

        CopyPropagation cp(mutable_arguments);
        func->body = cp.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
