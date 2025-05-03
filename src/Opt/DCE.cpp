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

    void visit(const ir::Var *node) override {
        ++use_counts[node->name];
        if (!curr_var.empty()) {
            // Inside a LetStmt/Assign
            ++dependent_use_counts[curr_var][node->name];
        }
    }

    // void visit(const ir::Call *node) override {
    //     // Don't visit the call's `func` argument, which is also
    //     // a variable.
    //     for (const ir::Expr &arg : node->args) {
    //         arg.accept(this);
    //     }
    // }
    // void visit(const ir::CallStmt *node) override {
    //     // Don't visit the call's `func` argument, which is also
    //     // a variable.
    //     for (const ir::Expr &arg : node->args) {
    //         arg.accept(this);
    //     }
    // }

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

    void new_window(int32_t parent_index) {
        use_counts.new_window(parent_index, {});
        dependent_use_counts.new_window(parent_index, {});
    }

    int32_t get_parent_index() {
        internal_assert(use_counts.windows.size() ==
                        dependent_use_counts.windows.size());
        return use_counts.windows.size() - 1;
    }
    void add_child(int32_t index) {
        use_counts.windows.back().children.push_back(index);
        dependent_use_counts.windows.back().children.push_back(index);
    }

    void visit(const ir::IfElse *node) override {
        // Save the index of the parent.
        const int32_t parent_index = get_parent_index();
        new_window(parent_index);
        add_child(get_parent_index() + 1);
        node->cond.accept(this);
        node->then_body.accept(this);
        new_window(parent_index);
        add_child(get_parent_index() + 2);
        if (!node->else_body.defined()) {
            return;
        }
        new_window(parent_index);
        add_child(get_parent_index() + 1);
        node->else_body.accept(this);
        new_window(parent_index);
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
            dependent_use_counts.look_back(loc.base);
        if (!map.has_value()) {
            return;
        }
        for (const auto &[var, count] : *map) {
            // TODO(cgyurgyik):
            // internal_assert(use_counts[var] >= count)
            //     << "Overflow failure in DCE: " << var
            //     << " has count: " << use_counts[var]
            //     << " but is used: " << count
            //     << " times in declaration of: " << loc;
            use_counts[var] -= count;
        }
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        // If any children have uses, this cannot be erased. This is safe to
        // check because we cannot shadow variables.
        if (node->loc.base == "y") {
            std::cout << "visiting at index " << use_counts.current_index
                      << ": " << ir::Stmt(node) << ", ";
            std::cout << "has uses: "
                      << use_counts.any_children(
                             node->loc.base, [](int32_t v) { return v != 0; })
                      << "\n";
            use_counts.dump();
        }
        const bool has_uses = use_counts.any_children(
            node->loc.base, [](int32_t v) { return v != 0; });
        if (!has_uses && !has_side_effects(node->value)) {
            erase_dependents(node->loc);
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        // If any children have uses, this cannot be erased. This is safe to
        // check because we cannot shadow variables.
        if (use_counts.any_children(node->loc.base,
                                    [](int32_t v) { return v != 0; })) {
            return node;
        }
        if (!node->mutating) {
            // Definition of this write loc.
            erase_dependents(node->loc);
        }
        return find_with_side_effects(node->value);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        // If any children have uses, this cannot be erased. This is safe to
        // check because we cannot shadow variables.
        if (use_counts.any_children(node->loc.base,
                                    [](int32_t v) { return v != 0; })) {
            return node;
        }
        return find_with_side_effects(node->value);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        // Right offset!
        int32_t begin = node->else_body.defined() ? -3 : -1;
        push_window(begin);
        ir::Stmt then_body = mutate(node->then_body);
        pop_window(+1);
        // No need to push before, there is no state between and if - else.
        ir::Stmt else_body = node->else_body;
        if (else_body.defined()) {
            push_window(+1);
            else_body = mutate(std::move(else_body));
            pop_window(+1);
        }
        use_counts.current_index -= begin + 1;
        dependent_use_counts.current_index -= begin + 1;
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
        for (auto iter = node->stmts.rbegin(); iter != node->stmts.rend();
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

        std::reverse(stmts.begin(), stmts.end());
        return ir::Sequence::make(std::move(stmts));
    }

  private:
    // How many times is a variable read.
    ir::History<std::string, int32_t> use_counts;
    // How many times does a variable definition reference another variable.
    ir::History<std::string, UseCountMap> dependent_use_counts;
    // Which functions have side effects.
    const std::set<std::string> &side_effects_functions;

    void push_window(int32_t offset) {
        use_counts.current_index += offset;
        dependent_use_counts.current_index += offset;
    }

    void pop_window(int32_t offset) {
        use_counts.current_index += offset;
        dependent_use_counts.current_index += offset;
    }
};

ir::Stmt dce_stmt(const std::set<std::string> &mutable_func_args,
                  const ir::Stmt &stmt,
                  const std::set<std::string> &se_functions) {
    // TODO(ajr): for non-exported functions, we can remove mutable args that
    // are never used.
    ComputeUseCounts analyzer(mutable_func_args);
    stmt.accept(&analyzer);
    // Add da children brah
    for (int i = 0; i < analyzer.use_counts.windows.size(); ++i) {
        auto &window = analyzer.use_counts.windows[i];
        for (int j = 0; j < analyzer.use_counts.windows.size(); ++j) {
            if (analyzer.use_counts.windows[j].parent == i) {
                window.children.push_back(j);
            }
        }
    }
    for (int i = 0; i < analyzer.dependent_use_counts.windows.size(); ++i) {
        auto &window = analyzer.dependent_use_counts.windows[i];
        for (int j = 0; j < analyzer.dependent_use_counts.windows.size(); ++j) {
            if (analyzer.dependent_use_counts.windows[j].parent == i) {
                window.children.push_back(j);
            }
        }
    }
    // Update the index to da end, because DCE starts from da back.
    analyzer.use_counts.current_index = analyzer.use_counts.windows.size() - 1;
    analyzer.dependent_use_counts.current_index =
        analyzer.dependent_use_counts.windows.size() - 1;
    analyzer.use_counts.dump();
    analyzer.dependent_use_counts.dump();
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
