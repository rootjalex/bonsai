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

// Whether we should give this sub-expression its own variable.
// TODO(cgyurgyik): What else?
bool rename(const ir::Expr &e) {
    if (const auto *binary_operation = e.as<ir::BinOp>()) {
        if (e.type().is<ir::Float_t>()) {
            return true;
        }
        const ir::Expr &a = binary_operation->a;
        const ir::Expr &b = binary_operation->b;
        switch (binary_operation->op) {
        case ir::BinOp::OpType::Add:
        case ir::BinOp::OpType::Sub:
        case ir::BinOp::OpType::Shl:
        case ir::BinOp::OpType::Shr:
        case ir::BinOp::OpType::BwAnd:
        case ir::BinOp::OpType::BwOr:
        case ir::BinOp::OpType::Xor:
            if ((a.is<ir::Var>() || is_const(a)) &&
                (b.is<ir::Var>() || is_const(b))) {
                // Adds and subtractions are cheap.
                return false;
            }
        default:
            return true;
        }
    }

    return !e.is<ir::Var>() && !is_const(e);
}

// Gives an expensive expression its own variable. For example,
//   g(foo(i), bar(j));
//   ->
//   let _t0 = foo(i) in
//   let _t1 = bar(j) in
//   g(_t0, _t1);
struct ToAnormalForm : public ir::Mutator {
    ir::Stmt visit(const ir::LetStmt *node) override {
        return anf(ir::LetStmt::make(node->loc, mutate(node->value)));
    }
    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        ir::Expr cond = mutate(node->cond);
        return anf(
            ir::IfElse::make(std::move(cond), std::move(th), std::move(el)));
    }
    ir::Stmt visit(const ir::Assign *node) override {
        return anf(
            ir::Assign::make(node->loc, mutate(node->value), node->mutating));
    }
    ir::Stmt visit(const ir::Accumulate *node) override {
        return anf(
            ir::Accumulate::make(node->loc, node->op, mutate(node->value)));
    }
    ir::Stmt visit(const ir::Return *node) override {
        if (!node->value.defined()) {
            return node;
        }
        return anf(ir::Return::make(mutate(node->value)));
    }

    ir::Stmt visit(const ir::Print *node) override {
        ir::Expr value = mutate(node->value);
        return anf(ir::Print::make(std::move(value)));
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return anf(ir::CallStmt::make(node->func, std::move(args)));
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
        ir::Expr a = mutate(node->a);
        ir::Expr b = mutate(node->b);
        ir::Expr op = ir::BinOp::make(node->op, std::move(a), std::move(b));
        if (!rename(op)) {
            return op;
        }
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(op)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Call *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        // Conservatively always rename function calls (we assume cheap calls
        // will be inlined).
        ir::WriteLoc loc("_t" + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(
            loc, ir::Call::make(node->func, std::move(args))));
        return ir::Var::make(node->type, loc.base);
    }

    // Skip the body of a lambda expression.
    ir::Expr visit(const ir::Lambda *node) override { return node; }
    // Skip statements we cannot unit test.
    ir::Stmt visit(const ir::ForAll *node) override { return node; }
    ir::Stmt visit(const ir::ForEach *node) override { return node; }
    ir::Stmt visit(const ir::DoWhile *node) override { return node; }

  private:
    // A list of intermediate statements generated for subexpressions.
    std::vector<ir::Stmt> stmts;
    // For unique variable renaming.
    int64_t counter = 0;

    // Pushes this `statement` onto the list of generated statements and returns
    // a sequence.
    ir::Stmt anf(ir::Stmt statement) {
        stmts.push_back(std::move(statement));
        ir::Stmt sequence = ir::Sequence::make(std::move(stmts));
        stmts.clear();
        return sequence;
    }
};

} // namespace

ir::FuncMap Rename::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        ToAnormalForm lower;
        func->body = lower.mutate(std::move(func->body));
        std::set<std::string> args = get_mutable_arguments(*func);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
