#include "IR/Stmt.h"

#include "IR/Printer.h"

namespace bonsai {
namespace ir {

Stmt Print::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Print::make";
    Print *node = new Print;
    node->value = std::move(value);
    return node;
}

Stmt Return::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Return::make";
    Return *node = new Return;
    node->value = std::move(value);
    return node;
}

Stmt Store::make(std::string name, Expr index, Expr value) {
    internal_assert(!name.empty()) << "Empty name in Store::make";
    internal_assert(value.defined()) << "Undefined value in Store::make";
    Store *node = new Store;
    node->name = std::move(name);
    node->index = std::move(index);
    node->value = std::move(value);
    return node;
}

// Stmt LetStmt::make(std::string name, Expr value, Stmt body) {
Stmt LetStmt::make(WriteLoc loc, Expr value) {
    internal_assert(loc.defined())
        << "Undefined write location in Assign::make";
    internal_assert(value.defined()) << "Undefined value in LetStmt::make";
    // internal_assert(body.defined()) << "Undefined body in LetStmt::make";
    LetStmt *node = new LetStmt;
    node->loc = std::move(loc);
    node->value = std::move(value);
    // node->body = std::move(body);
    return node;
}

Stmt IfElse::make(Expr cond, Stmt then_body, Stmt else_body) {
    internal_assert(cond.defined()) << "Undefined condition in IfElse::make";
    internal_assert(cond.type().defined() && cond.type().is_bool() ||
                    cond.type().is<Option_t>())
        << "Non-boolean condition in IfElse::make: " << cond << " of type "
        << cond.type();
    if (cond.type().is<Option_t>()) {
        cond = Cast::make(Bool_t::make(), cond);
    }
    internal_assert(then_body.defined())
        << "Undefined then_body in IfElse::make";
    IfElse *node = new IfElse;
    node->cond = std::move(cond);
    node->then_body = std::move(then_body);
    node->else_body = std::move(else_body);
    return node;
}

Stmt Sequence::make(std::vector<Stmt> stmts) {
    internal_assert(!stmts.empty()) << "Empty stmts in Sequence::make";
    for (const auto &s : stmts) {
        internal_assert(s.defined()) << "Undefined stmt in Sequence::make";
    }
    Sequence *node = new Sequence;
    node->stmts = std::move(stmts);
    return node;
}

Stmt Assign::make(WriteLoc loc, Expr value, bool mutating) {
    internal_assert(loc.defined())
        << "Undefined write location in Assign::make";
    internal_assert(value.defined()) << "Undefined value in Assign::make";
    Assign *node = new Assign;
    node->loc = std::move(loc);
    node->value = std::move(value);
    node->mutating = mutating;
    // node->body = std::move(body);
    return node;
}

Stmt Accumulate::make(WriteLoc loc, OpType op, Expr value) {
    internal_assert(loc.defined())
        << "Undefined write location in Accumulate::make";
    internal_assert(value.defined()) << "Undefined value in Accumulate::make";
    Accumulate *node = new Accumulate;
    node->loc = std::move(loc);
    node->op = op;
    node->value = std::move(value);
    // node->body = std::move(body);
    return node;
}

} // namespace ir
} // namespace bonsai
