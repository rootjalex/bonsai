#include "Opt/DCE.h"

#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include <map>
#include <string>

namespace bonsai {
namespace opt {

namespace {

using UseCountMap = std::map<std::string, uint32_t>;

struct ComputeUseCounts : ir::Visitor {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    std::map<std::string, UseCountMap> dependent_use_counts;
    // Name of the current variable whose definition is being traversed.
    std::string curr_var;

    void visit(const ir::Var *node) override {
        ++use_counts[node->name];
        if (!curr_var.empty()) {
            // Inside a LetStmt
            ++dependent_use_counts[curr_var][node->name];
        }
    }

    void visit(const ir::Lambda *node) override {
        for (const ir::Lambda::Argument &arg : node->args) {
            // TODO: std::map::contains ?
            internal_assert(use_counts.find(arg.name) == use_counts.cend());
            if (!curr_var.empty()) {
                const UseCountMap &dep_map = dependent_use_counts[curr_var];
                internal_assert(dep_map.find(arg.name) == dep_map.cend());
            }
        }
        // Need to erase use counts of arguments from use count maps.
        ir::Visitor::visit(node);
        for (const auto &arg : node->args) {
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
        internal_assert(node->loc.accesses.empty())
            << "unimplemented: " << ir::Stmt(node);
        // TODO: std::map::contains ?
        internal_assert(use_counts.find(node->loc.base) == use_counts.end())
            << "ComputeUseCounts already active for var: " << node->loc.base;
        internal_assert(dependent_use_counts.find(node->loc.base) ==
                        dependent_use_counts.end())
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc.base;
        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
    }

    void visit(const ir::Assign *node) override {
        internal_error << "TODO: handle Assign in ComputeUseCounts: "
                       << ir::Stmt(node);
    }

    void visit(const ir::Accumulate *node) override {
        internal_error << "TODO: handle Accumulate in ComputeUseCounts: "
                       << ir::Stmt(node);
    }
};

struct DeadCodeElimination : ir::Mutator {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    std::map<std::string, UseCountMap> dependent_use_counts;

    DeadCodeElimination(UseCountMap use_counts,
                        std::map<std::string, UseCountMap> dependent_use_counts)
        : use_counts(std::move(use_counts)),
          dependent_use_counts(std::move(dependent_use_counts)) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        internal_assert(node->loc.accesses.empty())
            << "unimplemented: " << ir::Stmt(node);
        if (use_counts[node->loc.base] == 0) {
            // Delete this LetStmt. Erase it's impact on use_counts.
            if (const auto cmap = dependent_use_counts.find(node->loc.base);
                cmap != dependent_use_counts.cend()) {
                for (const auto &[var, count] : cmap->second) {
                    internal_assert(use_counts[var] >= count)
                        << "Overflow failure in DCE: " << var
                        << " has count: " << use_counts[var]
                        << " but is used: " << count
                        << " times in declaration of: " << node->loc.base;
                    use_counts[var] -= count;
                }
            }
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Assign *node) override {
        internal_error << "TODO: handle Assign in DeadCodeElimination: "
                       << ir::Stmt(node);
        return ir::Stmt();
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        internal_error << "TODO: handle Accumulate in DeadCodeElimination: "
                       << ir::Stmt(node);
        return ir::Stmt();
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
            if (stmt.defined()) {
                not_changed = not_changed && stmt.same_as(*iter);
                stmts.emplace_back(std::move(stmt));
            } else {
                not_changed = false;
            }
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
