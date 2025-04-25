#include "Lower/Rename.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

ir::BinOp::OpType acc_to_bin(const ir::Accumulate::OpType op) {
    switch (op) {
    case ir::Accumulate::OpType::Add:
        return ir::BinOp::OpType::Add;
    case ir::Accumulate::OpType::Mul:
        return ir::BinOp::OpType::Mul;
    case ir::Accumulate::OpType::Sub:
        return ir::BinOp::OpType::Sub;
    case ir::Accumulate::OpType::Argmin:
    case ir::Accumulate::OpType::Argmax:
        internal_error << "[unimplemented] mapping from Accumulate::OpType to "
                          "respective BinOp::OpType: "
                       << op;
    }
}

// To ANF, kinda.
struct ToAnormalForm : public ir::Mutator {
    ir::Stmt visit(const ir::Accumulate *node) override {
        ir::Expr value = mutate(node->value);
        return anormalize(
            ir::Accumulate::make(node->loc, node->op, std::move(value)));
    }

    ir::Stmt visit(const ir::Assign *node) override {
        ir::Expr value = mutate(node->value);
        return anormalize(
            ir::Assign::make(node->loc, std::move(value), node->mutating));
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        return anormalize(ir::LetStmt::make(node->loc, std::move(value)));
    }

    ir::Stmt visit(const ir::Return *node) override {
        if (!node->value.defined()) {
            return node;
        }
        ir::Expr value = mutate(node->value);
        return anormalize(ir::Return::make(std::move(value)));
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return anormalize(ir::CallStmt::make(node->func, std::move(args)));
    }

    ir::Stmt visit(const ir::Print *node) override {
        ir::Expr value = mutate(node->value);
        return anormalize(ir::Print::make(std::move(value)));
    }

    // Skip the body expression of a lambda.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

    ir::Expr visit(const ir::BinOp *node) override {
        if (node->op == ir::BinOp::OpType::LAnd ||
            node->op == ir::BinOp::OpType::LOr) {
            // TODO(cgyurgyik): How to handle without causing code blow-up?
            return node; // Conservatively skip logical cases...
        }
        ir::Expr a = mutate(node->a);
        ir::Expr b = mutate(node->b);
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
        temporaries.push_back(ir::LetStmt::make(
            location, ir::BinOp::make(node->op, std::move(a), std::move(b))));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Call *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
        temporaries.push_back(ir::LetStmt::make(
            location, ir::Call::make(node->func, std::move(args))));
        return ir::Var::make(node->type, location.base);
    }

    ir::Stmt anormalize(ir::Stmt statement) {
        temporaries.push_back(std::move(statement));
        ir::Stmt sequence = ir::Sequence::make(std::move(temporaries));
        temporaries.clear();
        return sequence;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        ir::Expr cond = mutate(node->cond);

        return anormalize(
            ir::IfElse::make(std::move(cond), std::move(th), std::move(el)));
    }

    // TODO(cgyurgyik): todo
    ir::Stmt visit(const ir::ForAll *node) override { return node; }

  private:
    std::vector<ir::Stmt> temporaries;
    // For unique variable renaming.
    int64_t counter = 0;
};

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

    // x: mut i32 = 0;
    // x += 1;
    // use(x)
    // ->
    // x: mut i32 = 0;
    // _0x: mut i32 = x + 1;
    // use(_0x);
    ir::Stmt visit(const ir::Accumulate *node) override {
        ir::WriteLoc location = node->loc;
        if (!location.base_type.is_scalar()) {
            // This is a struct member update, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (mutable_function_arguments.contains(location.base)) {
            // This is a mutable function argument, don't rename it.
            return ir::Mutator::visit(node);
        }
        // Save the previous name (if it exists).
        auto it = old_to_new_name.find(location.base);
        std::optional<std::string> old_name;
        if (it != old_to_new_name.end()) {
            old_name = it->second;
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        // (Potentially) rename the current assignment's name.
        auto [new_name, updated] = rename(location.base);
        if (!updated) {
            return ir::Mutator::visit(node);
        };
        std::string name = old_name.has_value() ? *old_name : location.base;
        ir::Expr lhs = ir::Var::make(location.type, std::move(name));
        return ir::Assign::make(
            /*loc=*/ir::WriteLoc(std::move(new_name), location.type),
            /*value=*/
            ir::BinOp::make(acc_to_bin(node->op), std::move(lhs),
                            std::move(value)),
            /*mutating=*/false);
    }

    // x: mut i32 = 0;
    // x := 1 + y;
    // use(x)
    // ->
    // x: mut i32 = 0;
    // _0x: mut i32 = 1 + y;
    // use(_0x);
    ir::Stmt visit(const ir::Assign *node) override {
        ir::WriteLoc location = node->loc;
        if (!node->mutating) {
            // This is the first occurrence, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (!location.base_type.is_scalar()) {
            // This is a struct member update, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (mutable_function_arguments.contains(location.base)) {
            // This is a mutable function argument, don't rename it.
            return ir::Mutator::visit(node);
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        auto [new_name, updated] = rename(location.base);
        // (Potentially) rename the current assignment's name.
        if (!updated) {
            return ir::Mutator::visit(node);
        }
        return ir::Assign::make(ir::WriteLoc(new_name, location.type),
                                std::move(value),
                                /*mutating=*/false);
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

} // namespace

ir::FuncMap Rename::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        ToAnormalForm anormalize;
        func->body = anormalize.mutate(std::move(func->body));
        std::set<std::string> args = get_mutable_arguments(*func);
        RenameVariable rename(args);
        func->body = rename.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
