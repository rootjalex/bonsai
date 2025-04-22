#include "Opt/DCE.h"

#include "IR/Equality.h"
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

using UseCountMap = std::map<std::string, uint32_t>;
using DepUseCountMap = std::map<std::string, UseCountMap>;

struct ComputeUseCounts : ir::Visitor {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountMap dependent_use_counts;
    // Name of the current variable whose definition is being traversed.
    std::string curr_var;

    ComputeUseCounts(const std::set<std::string> &mutable_func_args) {
        for (const auto &arg : mutable_func_args) {
            // Conservatively set to 1, so Assign statements are not removed.
            use_counts[arg] = 1;
            dependent_use_counts[arg] = {};
        }
    }

    void visit(const ir::Var *node) override {
        ++use_counts[node->name];
        if (!curr_var.empty()) {
            // Inside a LetStmt/Assign
            ++dependent_use_counts[curr_var][node->name];
        }
    }

    void visit(const ir::Lambda *node) override {
        for (const ir::Lambda::Argument &arg : node->args) {
            internal_assert(!use_counts.contains(arg.name)) << arg.name;
            if (!curr_var.empty()) {
                const UseCountMap &dep_map = dependent_use_counts[curr_var];
                internal_assert(!dep_map.contains(arg.name));
            }
        }
        // Need to erase use counts of arguments from use count maps.
        ir::Visitor::visit(node);
        for (const ir::Lambda::Argument &arg : node->args) {
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
        internal_assert(!use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var: " << node->loc;
        internal_assert(!dependent_use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc;

        use_counts[node->loc.base] = 0;
        dependent_use_counts[node->loc.base] = {};

        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void visit(const ir::Assign *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Assign: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        internal_assert(!node->mutating || use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var: " << node->loc
            << " in stmt: " << ir::Stmt(node);
        internal_assert(!node->mutating ||
                        dependent_use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc;

        if (!node->mutating) {
            use_counts[node->loc.base] = 0;
            dependent_use_counts[node->loc.base] = {};
        }

        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void visit(const ir::Accumulate *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Accumulate: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        internal_assert(use_counts.contains(node->loc.base))
            << "ComputeUseCounts not active for var: " << node->loc;
        internal_assert(dependent_use_counts.contains(node->loc.base))
            << "ComputeUseCounts not active for var (dependent): " << node->loc;
        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    // void visit(const ir::Match *node) override {
    //     internal_error << "TODO: implement ComputeUseCounts for Match";
    // }
};

struct HasSideEffects : ir::Visitor {
    bool found = false;
    const std::set<std::string> &function_has_side_effects;

    HasSideEffects(const std::set<std::string> &side_effects_functions)
        : function_has_side_effects(side_effects_functions) {}

    void visit(const ir::Print *) override { found = true; }

    void visit(const ir::Var *node) override {
        if (node->type.is<ir::Function_t>() &&
            function_has_side_effects.contains(node->name)) {
            found = true;
        }
    }

    void visit(const ir::Store *) override {
        // TODO(ajr): This is conservative. How bad is that?
        found = true;
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
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountMap dependent_use_counts;
    // Which functions have side effects.
    const std::set<std::string> &side_effects_functions;

    DeadCodeElimination(UseCountMap use_counts,
                        DepUseCountMap dependent_use_counts,
                        const std::set<std::string> &side_effects_functions)
        : use_counts(std::move(use_counts)),
          dependent_use_counts(std::move(dependent_use_counts)),
          side_effects_functions(side_effects_functions) {}

    bool has_side_effects(const ir::Expr &expr) {
        HasSideEffects checker(side_effects_functions);
        expr.accept(&checker);
        return checker.found;
    }

    // Use counts are re-added for side-effecting expressions.
    void add_use_counts(const ir::Expr &expr) {
        ComputeUseCounts counter({}); // TODO(ajr): is this right?
        expr.accept(&counter);
        internal_assert(counter.dependent_use_counts.empty());
        for (const auto &[var, count] : counter.use_counts) {
            internal_assert(use_counts.contains(var));
            use_counts[var] += count;
        }
    }

    uint64_t counter = 0;
    std::string make_unused_var_name() {
        return "_unused" + std::to_string(counter++);
    }

    void erase_dependents(const ir::WriteLoc &loc) {
        // Erase it's impact on use_counts.
        if (const auto cmap = dependent_use_counts.find(loc.base);
            cmap != dependent_use_counts.cend()) {
            for (const auto &[var, count] : cmap->second) {
                internal_assert(use_counts[var] >= count)
                    << "Overflow failure in DCE: " << var
                    << " has count: " << use_counts[var]
                    << " but is used: " << count
                    << " times in declaration of: " << loc;
                use_counts[var] -= count;
            }
        }
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        if (use_counts[node->loc.base] == 0 && !has_side_effects(node->value)) {
            erase_dependents(node->loc);
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (use_counts[node->loc.base] == 0) {
            if (!node->mutating) {
                // Definition of this write loc.
                erase_dependents(node->loc);
            }

            if (has_side_effects(node->value)) {
                // TODO(ajr): we could just grab the side effecting Expr
                // and not compute the rest.
                // For now, we rewrite this to an unused LetStmt
                ir::WriteLoc loc(make_unused_var_name(), node->value.type());
                // We need to keep the locs used by this value.
                add_use_counts(node->value);
                return ir::LetStmt::make(std::move(loc), node->value);
            }

            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (use_counts[node->loc.base] == 0) {
            if (has_side_effects(node->value)) {
                // TODO(ajr): we could just grab the side effecting Expr
                // and not compute the rest.
                // For now, we rewrite this to an unused LetStmt
                ir::WriteLoc loc(make_unused_var_name(), node->value.type());
                // We need to keep the locs used by this value.
                add_use_counts(node->value);
                return ir::LetStmt::make(std::move(loc), node->value);
            }
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt then_body = mutate(node->then_body);
        ir::Stmt else_body = mutate(node->else_body);
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
            std::cout << "erased: " << ir::Stmt(node) << "\n";
            return ir::Stmt();
        } else if (not_changed) {
            return node;
        }

        std::reverse(stmts.begin(), stmts.end());
        return ir::Sequence::make(std::move(stmts));
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

        func->body = dce_stmt(mutable_func_args, func->body, se_functions);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
