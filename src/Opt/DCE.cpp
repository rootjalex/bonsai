#include "Opt/DCE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Lower/TopologicalOrder.h"

#include "Error.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

using UseCountMap = std::map<std::string, int32_t>;
using UseCountHistory = ir::History<std::string, int32_t>;
using DepUseCountHistory = ir::History<std::string, UseCountMap>;

struct ComputeUseCounts : ir::Visitor {
    // How many times is a variable read.
    UseCountHistory use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountHistory dependent_use_counts;
    // Name of the current variable whose definition is being traversed.
    std::string curr_var;

    ComputeUseCounts(const std::set<std::string> &mutable_func_args) {
        for (const auto &arg : mutable_func_args) {
            // Conservatively set to 1, so Assign statements are not removed.
            use_counts.add_to_window(arg, 1);
            dependent_use_counts.add_to_window(arg, {});
        }
    }

    void visit(const ir::Sequence *node) override {
        for (auto iter = node->stmts.begin(); iter != node->stmts.end();
             iter++) {
            iter->accept(this);
        }
    }

    void visit(const ir::Var *node) override {
        ++use_counts[node->name];
        if (!curr_var.empty()) {
            // Inside a LetStmt/Assign
            ++dependent_use_counts[curr_var][node->name];
        }
    }

    void visit(const ir::Call *node) override {
        // Don't visit the call's `func` argument, which is also
        // a variable.
        for (const ir::Expr &arg : node->args) {
            arg.accept(this);
        }
    }
    void visit(const ir::CallStmt *node) override {
        // Don't visit the call's `func` argument, which is also
        // a variable.
        for (const ir::Expr &arg : node->args) {
            arg.accept(this);
        }
    }

    void visit(const ir::Lambda *node) override {
        for (const ir::TypedVar &arg : node->args) {
            internal_assert(!use_counts.contains(arg.name)) << arg.name;
            if (!curr_var.empty()) {
                const std::map<std::string, int32_t> &dep_map =
                    dependent_use_counts[curr_var];
                internal_assert(!dep_map.contains(arg.name));
            }
        }
        // Need to erase use counts of arguments from use count maps.
        ir::Visitor::visit(node);
        for (const ir::TypedVar &arg : node->args) {
            use_counts.erase(arg.name);
            if (!curr_var.empty()) {
                // Erase from dependent_use_counts as well.
                dependent_use_counts[curr_var].erase(arg.name);
            }
        }
    }

    void visit(const ir::LetStmt *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested LetStmt: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        // TODO(ajr): Should LetStmts just contain a string name for writes? Can
        // never immutably write to an access.

        use_counts.add_to_window(node->loc.base, 0);
        dependent_use_counts.add_to_window(node->loc.base, {});

        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void visit(const ir::Assign *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Assign: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        if (!node->mutating) {
            use_counts.add_to_window(node->loc.base, 0);
            dependent_use_counts.add_to_window(node->loc.base, {});
        }

        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void visit(const ir::Accumulate *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Accumulate: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void new_window(int32_t previous_index) {
        use_counts.new_window(previous_index);
        dependent_use_counts.new_window(previous_index);
    }

    int32_t get_previous_index() {
        internal_assert(use_counts.size() == dependent_use_counts.size());
        return use_counts.size() - 1;
    }

    void visit(const ir::IfElse *node) override {
        // Save the index of the parent.
        const int32_t previous_index = get_previous_index();
        new_window(previous_index);
        node->cond.accept(this);
        node->then_body.accept(this);
        new_window(previous_index);
        if (!node->else_body.defined()) {
            return;
        }
        new_window(previous_index);
        node->else_body.accept(this);
        new_window(previous_index);
    }
};

struct HasSideEffects : ir::Visitor {
    bool found = false;
    const std::set<std::string> &function_has_side_effects;

    HasSideEffects(const std::set<std::string> &side_effects_functions)
        : function_has_side_effects(side_effects_functions) {}

    void visit(const ir::Print *node) override {
        if (found) {
            return;
        }
        found = true;
    }

    void visit(const ir::Call *node) override {
        if (found) {
            return;
        }
        const auto *var = node->func.as<ir::Var>();
        if (var == nullptr) {
            return;
        }
        if (var->type.is<ir::Function_t>() &&
            function_has_side_effects.contains(var->name)) {
            found = true;
        }
    }

    void visit(const ir::Store *node) override {
        // TODO(ajr): This is conservative. How bad is that?
        found = true;
    }
};

struct FindSideEffects : ir::Visitor {
    // The found side-effecting expressions (if any).
    std::vector<ir::Expr> expressions;
    const std::set<std::string> &function_has_side_effects;

    FindSideEffects(const std::set<std::string> &side_effects_functions)
        : function_has_side_effects(side_effects_functions) {}
    void visit(const ir::Call *node) override {
        const auto *var = node->func.as<ir::Var>();
        if (var == nullptr) {
            return;
        }
        if (var->type.is<ir::Function_t>() &&
            function_has_side_effects.contains(var->name)) {
            expressions.push_back(node);
        }
    }
};

std::set<std::string> find_side_effects(const ir::FuncMap &funcs) {
    const std::vector<std::string> topo_order =
        lower::func_topological_order(funcs, /*undef_calls=*/false);
    std::set<std::string> side_effects;
    HasSideEffects checker(side_effects);
    for (const std::string &f : topo_order) {
        internal_assert(!checker.function_has_side_effects.contains(f))
            << "Found cycle involving: " << f;
        checker.found = false;
        const auto func = funcs.at(f);
        // Conservatively say that funcs with mutable arguments have side
        // effects.
        if (std::any_of(func->args.cbegin(), func->args.cend(),
                        [](const auto &arg) { return arg.mutating; })) {
            side_effects.insert(f);
            continue;
        }
        // Otherwise search for side effecting statements.
        func->body.accept(&checker);
        if (checker.found) {
            side_effects.insert(f);
        }
        checker.found = false;
    }
    return side_effects;
}

struct DeadCodeElimination : ir::Mutator {
    DeadCodeElimination(UseCountHistory use_counts,
                        DepUseCountHistory dependent_use_counts,
                        const std::set<std::string> &side_effects_functions)
        : use_counts(std::move(use_counts)),
          dependent_use_counts(std::move(dependent_use_counts)),
          side_effects_functions(side_effects_functions) {}

    bool has_side_effects(const ir::Expr &expr) {
        HasSideEffects checker(side_effects_functions);
        expr.accept(&checker);
        return checker.found;
    }

    // Returns a sequence of statements with side effects within this
    // expression.
    ir::Stmt find_with_side_effects(const ir::Expr &expr) {
        FindSideEffects checker(side_effects_functions);
        expr.accept(&checker);
        std::vector<ir::Stmt> side_effecting_statements;
        for (const ir::Expr &value : checker.expressions) {
            add_use_counts(value);
            if (const auto *c = value.as<ir::Call>()) {
                ir::Stmt call =
                    ir::CallStmt::make(std::move(c->func), std::move(c->args));
                side_effecting_statements.push_back(std::move(call));
                continue;
            }
            internal_error << "[unimplemented]: " << value;
        }
        if (side_effecting_statements.empty()) {
            return ir::Stmt();
        }
        return ir::Sequence::make(std::move(side_effecting_statements));
    }

    // Use counts are re-added for side-effecting expressions.
    void add_use_counts(const ir::Expr &expr) {
        ComputeUseCounts counter({}); // TODO(ajr): is this right?
        expr.accept(&counter);
        internal_assert(counter.dependent_use_counts.empty());
        for (const auto &[var, count] : counter.use_counts.elements()) {
            internal_assert(use_counts.contains(var));
            use_counts[var] += count;
        }
    }

    void erase_dependents(const ir::WriteLoc &loc) {
        // Erase it's impact on use_counts.
        std::optional<UseCountMap> map =
            dependent_use_counts.from_window(loc.base);
        if (!map.has_value()) {
            return;
        }
        for (const auto &[var, count] : *map) {
            internal_assert(use_counts[var] >= count)
                << "Overflow failure in DCE: " << var
                << " has count: " << use_counts[var]
                << " but is used: " << count
                << " times in declaration of: " << loc;
            use_counts[var] -= count;
        }
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        if (use_counts.from_window(node->loc.base) == 0 &&
            !has_side_effects(node->value)) {
            erase_dependents(node->loc);
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        std::cout << ir::Stmt(node) << " at index " << use_counts.current_index
                  << " with dump: ";
        use_counts.dump();
        if (use_counts.from_window(node->loc.base) != 0) {
            return node;
        }
        if (!node->mutating) {
            // Definition of this write loc.
            erase_dependents(node->loc);
        }
        return find_with_side_effects(node->value);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (use_counts.from_window(node->loc.base) != 0) {
            return node;
        }
        return find_with_side_effects(node->value);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        push_window();
        ir::Stmt then_body = mutate(node->then_body);
        pop_window();
        // No need to push before, there is no state between and if - else.
        ir::Stmt else_body = node->else_body;
        if (else_body.defined()) {
            push_window();
            else_body = mutate(std::move(else_body));
            pop_window();
        }
        if (then_body.same_as(node->then_body) &&
            else_body.same_as(node->else_body)) {
            return node;
        } else if (!then_body.defined() && !else_body.defined()) {
            return ir::Stmt();
        } else if (then_body.defined() && else_body.defined()) {
            return ir::IfElse::make(node->cond, std::move(then_body),
                                    std::move(else_body));
        } else if (then_body.defined()) {
            return ir::IfElse::make(node->cond, std::move(then_body));
        } else {
            // else_body is defined, but then_body has been DCEed.
            // We now need to flip the condition, and only execute
            // else_body.
            ir::Expr flipped = ir::UnOp::make(ir::UnOp::Not, node->cond);
            return ir::IfElse::make(std::move(flipped), std::move(else_body));
        }
    }

    ir::Stmt visit(const ir::Sequence *node) override {
        bool not_changed = true;
        std::vector<ir::Stmt> stmts;
        for (auto iter = node->stmts.begin(); iter != node->stmts.end();
             iter++) {
            ir::Stmt stmt = mutate(*iter);
            if (!stmt.defined()) {
                not_changed = false;
                continue;
            }
            not_changed = not_changed && stmt.same_as(*iter);
            stmts.emplace_back(std::move(stmt));
        }

        if (stmts.empty()) {
            return ir::Stmt();
        } else if (not_changed) {
            return node;
        }

        // std::reverse(stmts.begin(), stmts.end());
        return ir::Sequence::make(std::move(stmts));
    }

  private:
    // How many times is a variable read.
    ir::History<std::string, int32_t> use_counts;
    // How many times does a variable definition reference another variable.
    ir::History<std::string, UseCountMap> dependent_use_counts;
    // Which functions have side effects.
    const std::set<std::string> &side_effects_functions;

    void push_window() {
        use_counts.push_window();
        dependent_use_counts.push_window();
    }

    void pop_window() {
        use_counts.pop_window();
        dependent_use_counts.pop_window();
    }
};

ir::Stmt dce_stmt(const std::set<std::string> &mutable_func_args,
                  const ir::Stmt &stmt,
                  const std::set<std::string> &se_functions) {
    // TODO(ajr): for non-exported functions, we can remove mutable args that
    // are never used.
    ComputeUseCounts analyzer(mutable_func_args);
    stmt.accept(&analyzer);
    DeadCodeElimination optimizer(std::move(analyzer.use_counts),
                                  std::move(analyzer.dependent_use_counts),
                                  se_functions);
    return optimizer.mutate(stmt);
}

} // namespace

ir::FuncMap DCE::run(ir::FuncMap funcs) const {
    // TODO(ajr): We should also erase unused arguments to Lambdas and
    // Functions. This requires mutating the definitions and all calls,
    // which can get tricky.

    std::set<std::string> se_functions = find_side_effects(funcs);

    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_func_args;
        for (const auto &arg : func->args) {
            if (arg.mutating) {
                mutable_func_args.insert(arg.name);
            }
        }
        func->body =
            dce_stmt(mutable_func_args, std::move(func->body), se_functions);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
