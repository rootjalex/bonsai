#include "Opt/DCE.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <string>

namespace bonsai {
namespace opt {

namespace {

using UseCountMap = std::map<ir::WriteLoc, uint32_t, ir::WriteLocLessThan>;
using DepUseCountMap = std::map<ir::WriteLoc, UseCountMap, ir::WriteLocLessThan>;

struct ComputeUseCounts : ir::Visitor {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountMap dependent_use_counts;
    // Name of the current variable whose definition is being traversed.
    std::optional<ir::WriteLoc> curr_var;

    void visit(const ir::Var *node) override {
        ir::WriteLoc loc = read_to_writeloc(node);
        ++use_counts[loc];
        if (curr_var.has_value()) {
            // Inside a LetStmt/Assign
            ++dependent_use_counts[*curr_var][loc];
        }
    }

    void visit(const ir::Lambda *node) override {
        for (const ir::Lambda::Argument &arg : node->args) {
            // TODO: std::map::contains ?
            ir::WriteLoc loc = ir::WriteLoc(arg.name, arg.type);
            internal_assert(use_counts.find(loc) == use_counts.cend());
            if (curr_var.has_value()) {
                const UseCountMap &dep_map = dependent_use_counts[*curr_var];
                internal_assert(dep_map.find(loc) == dep_map.cend());
            }
        }
        // Need to erase use counts of arguments from use count maps.
        ir::Visitor::visit(node);
        for (const ir::Lambda::Argument &arg : node->args) {
            ir::WriteLoc loc = ir::WriteLoc(arg.name, arg.type);
            use_counts.erase(loc);
            if (curr_var.has_value()) {
                // Erase from dependent_use_counts as well.
                dependent_use_counts[*curr_var].erase(loc);
            }
        }
    }

    void visit(const ir::LetStmt *node) override {
        internal_assert(!curr_var.has_value())
            << "Unexpected nested LetStmt: " << ir::Stmt(node)
            << " when traversing for: " << *curr_var;
        // TODO(ajr): Should LetStmts just contain a string name for writes? Can
        // never immutably write to an access.
        internal_assert(!use_counts.contains(node->loc))
            << "ComputeUseCounts already active for var: " << node->loc;
        internal_assert(!dependent_use_counts.contains(node->loc))
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc;

        use_counts[node->loc] = 0;
        dependent_use_counts[node->loc] = {};

        curr_var = node->loc;
        node->value.accept(this);
        curr_var.reset();
    }

    void visit(const ir::Assign *node) override {
        internal_assert(!curr_var.has_value())
            << "Unexpected nested Assign: " << ir::Stmt(node)
            << " when traversing for: " << *curr_var;
        internal_assert(!node->mutating || !use_counts.contains(node->loc))
            << "ComputeUseCounts already active for var: " << node->loc;
        internal_assert(!node->mutating || !dependent_use_counts.contains(node->loc))
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc;

        if (!node->mutating) {
            use_counts[node->loc] = 0;
            dependent_use_counts[node->loc] = {};
        }

        curr_var = node->loc;
        node->value.accept(this);
        curr_var.reset();
    }

    void visit(const ir::Accumulate *node) override {
        internal_assert(!curr_var.has_value())
            << "Unexpected nested Accumulate: " << ir::Stmt(node)
            << " when traversing for: " << *curr_var;
        internal_assert(use_counts.contains(node->loc))
            << "ComputeUseCounts not active for var: " << node->loc;
        internal_assert(dependent_use_counts.contains(node->loc))
            << "ComputeUseCounts not active for var (dependent): "
            << node->loc;
        curr_var = node->loc;
        node->value.accept(this);
        curr_var.reset();
    }
};

struct DeadCodeElimination : ir::Mutator {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountMap dependent_use_counts;

    DeadCodeElimination(UseCountMap use_counts, DepUseCountMap dependent_use_counts)
        : use_counts(std::move(use_counts)),
          dependent_use_counts(std::move(dependent_use_counts)) {}

    void erase_dependents(const ir::WriteLoc &loc) {
        // Erase it's impact on use_counts.
        if (const auto cmap = dependent_use_counts.find(loc);
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
        if (use_counts[node->loc] == 0) {
            erase_dependents(node->loc);
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        if (use_counts[node->loc] == 0) {
            if (!node->mutating) {
                // Definition of this write loc.
                erase_dependents(node->loc);
            }
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (use_counts[node->loc] == 0) {
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
};

ir::Stmt dce(const ir::Stmt &stmt) {
    ComputeUseCounts analyzer;
    stmt.accept(&analyzer);
    DeadCodeElimination optimizer(std::move(analyzer.use_counts),
                                  std::move(analyzer.dependent_use_counts));
    return optimizer.mutate(stmt);
}

} // namespace

ir::Program dce(const ir::Program &program) {
    ir::Program new_program = program;

    // TODO(ajr): We should also erase unused arguments to Lambdas and
    // Functions. This requires mutating the definitions and all calls,
    // which can get tricky.

    for (auto &[name, func] : new_program.funcs) {
        func->body = dce(func->body);
    }

    return new_program;
}

} // namespace opt
} // namespace bonsai
