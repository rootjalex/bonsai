#include "Opt/DCE.h"

#include "IR/Analysis.h"
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
static int64_t counter = 0;
constexpr char DELIMITER[] = "#";

using UseCountMap = std::map<std::string, uint32_t>;
using DepUseCountMap = std::map<std::string, UseCountMap>;

// Gives unique names to variables in diverging control flow. This is necessary
// because the DCE pass currently does not account for diverging control flow.
struct NameHygiene : ir::Mutator {
    ir::Expr visit(const ir::Var *node) override {
        auto it = old_to_new.find(node->name);
        if (it == old_to_new.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        if (!rename) {
            return node;
        }

        ir::WriteLoc location = do_rename(node->loc);
        return ir::Allocate::make(std::move(location), mutate(node->value),
                                  node->memory);
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!rename) {
            return node;
        }

        auto it = old_to_new.find(node->loc.base);
        if (it == old_to_new.end()) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc new_loc(it->second, node->loc.base_type);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<ir::Expr>(value)) {
                ir::Expr new_value = mutate(std::get<ir::Expr>(value));
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        ir::Expr value = mutate(node->value);
        return ir::Store::make(std::move(new_loc), std::move(value));
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (!rename) {
            return node;
        }
        auto it = old_to_new.find(node->loc.base);
        if (it == old_to_new.end()) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc location(it->second, node->loc.type);
        return ir::Accumulate::make(std::move(location), node->op,
                                    mutate(node->value));
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        if (!rename) {
            return node;
        }
        ir::WriteLoc location = do_rename(node->loc);
        return ir::LetStmt::make(std::move(location), mutate(node->value));
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        // Rename where control flow diverges.
        ScopedValue<bool> _(rename, true);
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

  private:
    ir::WriteLoc do_rename(const ir::WriteLoc &location) {
        std::string name =
            DELIMITER + location.base + DELIMITER + std::to_string(counter++);
        // Replacement is expected here; this is why this pass exists.
        old_to_new[location.base] = name;
        return ir::WriteLoc(std::move(name), location.type);
    }
    bool rename = false; // Whether to perform a rename.
    // Mapping from old name to "newly renamed" name.
    std::unordered_map<std::string, std::string> old_to_new;
};

// Undo the hygienic naming.
struct UnnameHygiene : ir::Mutator {
    ir::Expr visit(const ir::Var *node) override {
        if (!node->name.starts_with(DELIMITER)) {
            return node;
        }
        return ir::Var::make(node->type, extract(node->name));
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        if (!node->loc.base.starts_with(DELIMITER)) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc loc(extract(node->loc.base), node->loc.type);
        return ir::Allocate::make(std::move(loc), mutate(node->value),
                                  node->memory);
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!node->loc.base.starts_with(DELIMITER)) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc new_loc(extract(node->loc.base), node->loc.base_type);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<ir::Expr>(value)) {
                ir::Expr new_value = mutate(std::get<ir::Expr>(value));
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        ir::Expr value = mutate(node->value);
        return ir::Store::make(std::move(new_loc), std::move(value));
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (!node->loc.base.starts_with(DELIMITER)) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc loc(extract(node->loc.base), node->loc.type);
        return ir::Accumulate::make(std::move(loc), node->op,
                                    mutate(node->value));
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        if (!node->loc.base.starts_with(DELIMITER)) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc loc(extract(node->loc.base), node->loc.type);
        return ir::LetStmt::make(std::move(loc), mutate(node->value));
    }

  private:
    std::string extract(const std::string &input) {
        std::string result = "";
        size_t start = 0;
        size_t end = 0;
        if (start = input.find(DELIMITER, end); start == std::string::npos) {
            return result;
        }
        start++;
        end = input.find(DELIMITER, start);
        internal_assert(end != std::string::npos)
            << "unexpected: found delimiter: " << DELIMITER
            << " with no closing delimiter in: " << input;

        if (end > start) {
            result += input.substr(start, end - start);
        }

        return result;
    }
};

struct ComputeUseCounts : ir::Visitor {
    // How many times is a variable read.
    UseCountMap use_counts;
    // How many times does a variable definition reference another variable.
    DepUseCountMap dependent_use_counts;
    // Name of the current variable whose definition is being traversed.
    std::string curr_var;

    ComputeUseCounts(const std::set<std::string> &mutable_func_args) {
        for (const auto &arg : mutable_func_args) {
            // Conservatively set to 1, so Store statements are not removed.
            use_counts[arg] = 1;
            dependent_use_counts[arg] = {};
        }
    }

    void visit(const ir::Var *node) override {
        ++use_counts[node->name];
        if (!curr_var.empty()) {
            // Inside a LetStmt/Store
            ++dependent_use_counts[curr_var][node->name];
        }
    }

    void visit(const ir::Lambda *node) override {
        for (const ir::TypedVar &arg : node->args) {
            internal_assert(!use_counts.contains(arg.name)) << arg.name;
            if (!curr_var.empty()) {
                const UseCountMap &dep_map = dependent_use_counts[curr_var];
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

    void visit(const ir::Allocate *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Allocate: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        internal_assert(!use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var: " << node->loc
            << " in Allocate: " << ir::Stmt(node);
        internal_assert(!dependent_use_counts.contains(node->loc.base))
            << "ComputeUseCounts already active for var (dependent): "
            << node->loc;

        use_counts[node->loc.base] = 0;
        dependent_use_counts[node->loc.base] = {};

        if (node->value.defined()) {
            curr_var = node->loc.base;
            node->value.accept(this);
            curr_var.clear();
        }
    }

    void visit(const ir::Store *node) override {
        internal_assert(curr_var.empty())
            << "Unexpected nested Store: " << ir::Stmt(node)
            << " when traversing for: " << curr_var;
        internal_assert(use_counts.contains(node->loc.base))
            << "ComputeUseCounts not active for var: " << node->loc
            << " in Store: " << ir::Stmt(node);
        internal_assert(dependent_use_counts.contains(node->loc.base))
            << "ComputeUseCounts not active for var (dependent): " << node->loc;

        // Need to increment use counts of indices.
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<ir::Expr>(value)) {
                std::get<ir::Expr>(value).accept(this);
            }
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
        // Need to increment use counts of indices.
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<ir::Expr>(value)) {
                std::get<ir::Expr>(value).accept(this);
            }
        }
        curr_var = node->loc.base;
        node->value.accept(this);
        curr_var.clear();
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
        for (const auto &[var, count] : counter.use_counts) {
            internal_assert(use_counts.contains(var));
            use_counts[var] += count;
        }
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
        if (use_counts[node->loc.base] == 0 &&
            !has_side_effects(node->value, side_effects_functions)) {
            erase_dependents(node->loc);
            return ir::Stmt();
        }
        return node;
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        if (use_counts[node->loc.base] != 0) {
            return node;
        }
        erase_dependents(node->loc);
        if (node->value.defined()) {
            return find_with_side_effects(node->value);
        }
        return ir::Stmt();
    }

    ir::Stmt merge_undef_seqs(ir::Stmt a, ir::Stmt b) {
        if (!a.defined()) {
            return b;
        } else if (!b.defined()) {
            return a;
        }
        std::vector<ir::Stmt> stmts;
        if (const ir::Sequence *a_seq = a.as<ir::Sequence>()) {
            stmts = a_seq->stmts;
        } else {
            stmts = {std::move(a)};
        }

        if (const ir::Sequence *b_seq = b.as<ir::Sequence>()) {
            stmts.insert(stmts.end(), b_seq->stmts.begin(), b_seq->stmts.end());
        } else {
            stmts.push_back(std::move(b));
        }
        return ir::Sequence::make(std::move(stmts));
    }

    ir::Stmt handle_side_effects(const ir::WriteLoc &loc,
                                 const ir::Expr &expr) {
        // Handle if there are side effects in the index expressions?
        // TODO(ajr): we should decrement use counts of the idxs as well.
        ir::Stmt side_effects = find_with_side_effects(expr);
        for (const auto &value : loc.accesses) {
            if (std::holds_alternative<ir::Expr>(value)) {
                side_effects = merge_undef_seqs(
                    std::move(side_effects),
                    find_with_side_effects(std::get<ir::Expr>(value)));
            }
        }
        return side_effects;
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (use_counts[node->loc.base] != 0) {
            return node;
        }
        return handle_side_effects(node->loc, node->value);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (use_counts[node->loc.base] != 0) {
            return node;
        }
        return handle_side_effects(node->loc, node->value);
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
            return ir::Stmt();
        } else if (not_changed) {
            return node;
        }

        std::reverse(stmts.begin(), stmts.end());
        return ir::Sequence::make(std::move(stmts));
    }
};

} // namespace

// TODO(ajr): for non-exported functions, we can remove mutable args that
// are never used.
ir::Stmt dce(ir::Stmt stmt, const std::set<std::string> &mutable_func_args,
             const std::set<std::string> &se_functions) {
    stmt = NameHygiene().mutate(std::move(stmt));
    ComputeUseCounts analyzer(mutable_func_args);
    stmt.accept(&analyzer);
    DeadCodeElimination optimizer(std::move(analyzer.use_counts),
                                  std::move(analyzer.dependent_use_counts),
                                  se_functions);
    stmt = optimizer.mutate(std::move(stmt));
    return UnnameHygiene().mutate(stmt);
}

ir::FuncMap DCE::run(ir::FuncMap funcs, const CompilerOptions &options) const {
    // TODO(ajr): We should also erase unused arguments to Lambdas and
    // Functions. This requires mutating the definitions and all calls,
    // which can get tricky.

    std::set<std::string> se_functions = find_side_effects(funcs);

    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_func_args = func->mutable_args();
        func->body =
            dce(std::move(func->body), mutable_func_args, se_functions);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
