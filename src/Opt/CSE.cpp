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

std::optional<ir::BinOp::OpType> acc_to_bin(const ir::Accumulate::OpType op) {
    switch (op) {
    case ir::Accumulate::OpType::Add:
        return ir::BinOp::OpType::Add;
    case ir::Accumulate::OpType::Mul:
        return ir::BinOp::OpType::Mul;
    case ir::Accumulate::OpType::Sub:
        return ir::BinOp::OpType::Sub;
    case ir::Accumulate::OpType::Argmin:
    case ir::Accumulate::OpType::Argmax:
        return {};
    }
}

// Simple variable renaming in straightline code.
struct RenameVariable : public ir::Mutator {
    RenameVariable(const std::set<std::string> &mutable_function_arguments)
        : mutable_function_arguments(mutable_function_arguments) {}

    const std::set<std::string> &mutable_function_arguments;
    // Tracks the old variable name to the new name.
    std::unordered_map<std::string, std::string> old_to_new_name;

    std::pair<std::string, bool> rename(std::string name) {
        auto it = old_to_new_name.find(name);
        if (should_rename) {
            std::string new_name = "_" + std::to_string(counter++) + name;
            old_to_new_name[name] = new_name;
            return {new_name, true};
        }
        if (it != old_to_new_name.end()) {
            return {it->second, true};
        }
        return {name, false};
    }

    ir::Expr visit(const ir::Var *node) override {
        auto it = old_to_new_name.find(node->name);
        if (it == old_to_new_name.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        ir::WriteLoc location = node->loc;
        if (!location.base_type.is_scalar() ||
            mutable_function_arguments.contains(location.base)) {
            return ir::Mutator::visit(node);
        }
        std::optional<ir::BinOp::OpType> op = acc_to_bin(node->op);
        // Only rename in the case where there is a binary operation equivalent.
        ScopedValue<bool> guard(should_rename, should_rename && op.has_value());
        // Save previous name.
        auto it = old_to_new_name.find(location.base);
        std::optional<std::string> old_name;
        if (it != old_to_new_name.end()) {
            old_name = it->second;
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        // Rename current name.
        auto [new_name, updated] = rename(location.base);
        if (!updated) {
            return ir::Mutator::visit(node);
        }
        ir::WriteLoc wl(new_name, location.type);
        if (op.has_value()) {
            std::string name = old_name.has_value() ? *old_name : location.base;
            ir::Expr lhs = ir::Var::make(location.type, std::move(name));
            ir::Expr v = ir::BinOp::make(*op, std::move(lhs), std::move(value));
            return ir::Assign::make(std::move(wl), std::move(v),
                                    /*mutating=*/false);
        }
        return ir::Accumulate::make(std::move(wl), node->op, std::move(value));
    }

    ir::Stmt visit(const ir::Assign *node) override {
        ir::WriteLoc location = node->loc;
        if (!(node->mutating && location.base_type.is_scalar()) ||
            mutable_function_arguments.contains(location.base)) {
            // This is the first occurrence of this variable or a struct member
            // assignment or a function argument.
            return ir::Mutator::visit(node);
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        auto [new_name, updated] = rename(location.base);
        if (!updated) {
            return ir::Mutator::visit(node);
        }
        return ir::Assign::make(ir::WriteLoc(new_name, location.type),
                                std::move(value), /*mutating=*/false);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

  private:
    // Whether the variable should be given a fresh name.
    bool should_rename = true;
    // For unique variable renaming.
    int64_t counter = 0;
};

// TODO(cgyurgyik): Provide a real hash function.
struct ExprHash {
    std::size_t operator()(const ir::Expr &expr) const { return 0; }
};

struct ExprEqual {
    bool operator()(const ir::Expr &a, const ir::Expr &b) const {
        return ir::equals(a, b);
    }
};

struct IsCseLegal : public ir::Visitor {
    IsCseLegal(const std::set<std::string> &side_effect_functions,
               const std::set<std::string> &mutable_function_arguments)
        : side_effect_functions(side_effect_functions),
          blacklisted_variables(mutable_function_arguments) {}

    void visit(const ir::Var *node) override {
        is_legal &= !blacklisted_variables.contains(node->name);
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
    const std::set<std::string> &blacklisted_variables;
};

class CommonSubexpressionElimination : public ir::Mutator {
  public:
    CommonSubexpressionElimination(
        const std::set<std::string> &side_effect_functions,
        const std::set<std::string> &mutable_function_arguments)
        : side_effect_functions(side_effect_functions),
          blacklisted_variables(mutable_function_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = get(node->value, node->loc);
        if (!value.defined()) {
            return node;
        }
        return ir::LetStmt::make(node->loc, std::move(value));
    }

    ir::Stmt visit(const ir::Assign *node) override {
        ir::WriteLoc loc = node->loc;
        if (node->mutating) {
            blacklisted_variables.insert(loc.base);
            return node;
        }
        ir::Expr value = get(node->value, node->loc);
        if (!value.defined()) {
            return node;
        }
        return ir::Assign::make(node->loc, std::move(value),
                                /*mutating=*/node->mutating);
    }

    ir::Stmt visit(const ir::Store *node) override {
        // Conservatively avoid stores until we have a stronger memory analysis.
        blacklisted_variables.insert(node->name);
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        blacklisted_variables.insert(node->loc.base);
        return node;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr c = get(node->cond);
        if (!c.defined()) {
            ScopedValue<bool> guard(allow_cse, false);
            return ir::Mutator::visit(node);
        }
        return ir::IfElse::make(std::move(c), node->then_body, node->else_body);
    }

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
    // is false in any control flow constructs.
    bool allow_cse = true;
    // A list of functions that may have side effects.
    const std::set<std::string> &side_effect_functions;
    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable function arguments, mutable
    // assignments, and references to allocations.
    std::set<std::string> blacklisted_variables;
    // Maps expressions to the variable of its first occurrence.
    std::unordered_map<ir::Expr, ir::Expr, ExprHash, ExprEqual>
        expression_to_variable;

    ir::Expr get(ir::Expr value, std::optional<ir::WriteLoc> location = {}) {
        if (!(allow_cse && is_cse_legal(value))) {
            // Don't perform CSE on expressions that aren't legal.
            return ir::Expr();
        }
        auto it = expression_to_variable.find(value);
        if (it == expression_to_variable.end()) {
            if (location.has_value()) {
                ir::Expr v = ir::Var::make(location->base_type, location->base);
                // If this expression hasn't been seen, add it to the list.
                expression_to_variable[value] = std::move(v);
            }
            return ir::Expr();
        }
        return it->second;
    }

    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        IsCseLegal checker(side_effect_functions, blacklisted_variables);
        e.accept(&checker);
        return checker.is_legal;
    }
};

} // namespace

ir::FuncMap CSE::run(ir::FuncMap funcs) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        const std::set<std::string> args = get_mutable_arguments(*func);
        RenameVariable rename(args);
        func->body = rename.mutate(std::move(func->body));
        CommonSubexpressionElimination cse(side_effect_functions, args);
        func->body = cse.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
