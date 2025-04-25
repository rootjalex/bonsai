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

// To ANF, kinda.
struct ToAnormalForm : public ir::Mutator {
    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = node->value;
        value = mutate(std::move(value));
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

    ir::Expr visit(const ir::BinOp *node) override {
        switch (node->op) {
        // TODO(cgyurgyik): Handle logical operators.
        case ir::BinOp::OpType::LAnd:
        case ir::BinOp::OpType::LOr:
            return node;
        default:
            break;
        }
        ir::Expr a = mutate(std::move(node->a));
        ir::Expr b = mutate(std::move(node->b));
        if ((a.is<ir::Var>() || is_const(a)) &&
            (b.is<ir::Var>() || is_const(b))) {
            // Avoid this case:
            // c = a + x;
            // ->
            // let _t0 = a + x in
            // let c = _t0 in
            return ir::BinOp::make(node->op, std::move(a), std::move(b));
        }
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

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        ir::Expr cond = mutate(node->cond);
        return anormalize(
            ir::IfElse::make(std::move(cond), std::move(th), std::move(el)));
    }

    // Skip the body expression of a lambda.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

    // TODO(cgyurgyik): todo
    ir::Stmt visit(const ir::ForAll *node) override { return node; }

  private:
    std::vector<ir::Stmt> temporaries;
    // For unique variable renaming.
    int64_t counter = 0;

    ir::Stmt anormalize(ir::Stmt statement) {
        temporaries.push_back(std::move(statement));
        ir::Stmt sequence = ir::Sequence::make(std::move(temporaries));
        temporaries.clear();
        return sequence;
    }
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
