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
        << "Undefined write location in LetStmt::make";
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
    internal_assert(cond.type().defined() &&
                    (cond.type().is_bool() || cond.type().is<Option_t>()))
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

Stmt Match::make(Expr loc, Match::Arms arms) {
    internal_assert(loc.defined()) << "Undefined match location in Match::make";
    internal_assert(!arms.empty()) << "Received no match arms in Match::make";
    const BVH_t *bvh = loc.type().as<BVH_t>();
    internal_assert(bvh) << "Match is only implemented for BVH_t, received: "
                         << loc;
    internal_assert(bvh->nodes.size() == arms.size())
        << "Incorrect number of match arms for BVH type: " << loc.type()
        << " with " << arms.size() << " arms.";
    // Make sure all match arms exist.
    const size_t n = bvh->nodes.size();
    for (size_t i = 0; i < n; i++) {
        std::string_view name = bvh->nodes[i].name();
        const bool found =
            arms.cend() !=
            std::find_if(arms.cbegin(), arms.cend(), [&name](const auto &arm) {
                return arm.first.name() == name;
            });
        internal_assert(found) << "Match does not contain match arm: " << name;
    }
    Match *node = new Match;
    node->loc = std::move(loc);
    node->arms = std::move(arms);
    return node;
}

Stmt Yield::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Yield::make";
    Yield *node = new Yield;
    node->value = std::move(value);
    return node;
}

Stmt Scan::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Scan::make";
    Scan *node = new Scan;
    node->value = std::move(value);
    return node;
}

Stmt YieldFrom::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in YieldFrom::make";
    YieldFrom *node = new YieldFrom;
    node->value = std::move(value);
    return node;
}

Stmt ForAll::make(std::string name, Expr iter, Stmt body) {
    internal_assert(!name.empty()) << "Undefined name in ForAll::make";
    internal_assert(iter.defined()) << "Undefined iterator in ForAll::make";
    internal_assert(iter.type().is_iterable())
        << "ForAll requires iterable: " << iter;
    internal_assert(body.defined()) << "Undefined body in ForAll::make";

    ForAll *node = new ForAll;
    node->name = std::move(name);
    node->iter = std::move(iter);
    node->body = std::move(body);
    return node;
}

} // namespace ir
} // namespace bonsai
