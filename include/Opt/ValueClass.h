#pragma once

#include <set>
#include <string>

#include "IR/Expr.h"
#include "IR/Mutator.h"
#include "IR/Stmt.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

namespace bonsai {
namespace opt {

// What a value is allowed to do and still be moved or duplicated at one
// program point. `sinkable` admits the pure computations and the reads;
// `memory` marks the reads, which never cross a statement that could write
// (there is no alias analysis here: a read never crosses a write, whosever
// it is). Shared by Opt/Sink (motion) and Opt/NarrowLoads (duplication at
// the same statement, where `memory` costs nothing and `sinkable` is the
// whole test).
struct ValueClass : ir::Visitor {
    bool sinkable = true;
    bool memory = false;

    void visit(const ir::Call *) override { sinkable = false; }
    void visit(const ir::MatchExpr *) override { sinkable = false; }
    void visit(const ir::Generator *) override { sinkable = false; }
    void visit(const ir::Lambda *) override { sinkable = false; }
    void visit(const ir::GeomOp *) override { sinkable = false; }
    void visit(const ir::SetOp *) override { sinkable = false; }
    void visit(const ir::AggOp *) override { sinkable = false; }
    void visit(const ir::Construct *) override { sinkable = false; }
    void visit(const ir::PtrTo *) override { sinkable = false; }
    void visit(const ir::Deref *node) override {
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::StoredElement *node) override {
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::Unwrap *node) override {
        // A variant read: of a tree node's bytes as often as of a value.
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::Intrinsic *node) override {
        if (ir::Intrinsic::has_effects(node->op)) {
            sinkable = false;
            return;
        }
        if (node->op == ir::Intrinsic::rt_payload ||
            node->op == ir::Intrinsic::tex_sample_grad_2d) {
            memory = true;
        }
        ir::Visitor::visit(node);
    }
};

inline ValueClass classify(const ir::Expr &value) {
    ValueClass c;
    value.accept(&c);
    return c;
}

// What a statement a motion would cross can do to a value: define or write
// names, touch memory, or be something these passes move nothing past
// (`barrier`) -- a loop, a call, a return, anything unrecognized. The
// recognized set is a whitelist; everything else is a barrier, so a new
// statement kind is safe by default. Shared by Opt/Sink and
// Opt/NarrowLoads.
struct Crossing {
    std::set<std::string> defines; // let and allocate bases
    std::set<std::string> writes;  // store and accumulate bases
    bool memory = false;           // could write memory a read could see
    bool barrier = false;
};

inline void crossing_of(const ir::Stmt &stmt, Crossing &c) {
    if (!stmt.defined()) {
        return;
    }
    if (const ir::LetStmt *let = stmt.as<ir::LetStmt>()) {
        c.defines.insert(let->loc.base);
        if (!classify(let->value).sinkable) {
            // A call or an effect on the right-hand side: it may write
            // through a mutable argument, so nothing crosses it.
            c.barrier = true;
        }
        return;
    }
    if (const ir::Allocate *alloc = stmt.as<ir::Allocate>()) {
        c.defines.insert(alloc->loc.base);
        c.memory = true;
        return;
    }
    if (const ir::Store *store = stmt.as<ir::Store>()) {
        c.writes.insert(store->loc.base);
        c.memory = true;
        return;
    }
    if (const ir::Accumulate *acc = stmt.as<ir::Accumulate>()) {
        c.writes.insert(acc->loc.base);
        c.memory = true;
        return;
    }
    if (stmt.as<ir::Print>() != nullptr) {
        // An effect, but it writes no name and no memory a read sees.
        return;
    }
    if (const ir::Sequence *seq = stmt.as<ir::Sequence>()) {
        for (const ir::Stmt &s : seq->stmts) {
            crossing_of(s, c);
        }
        return;
    }
    if (const ir::IfElse *branch = stmt.as<ir::IfElse>()) {
        crossing_of(branch->then_body, c);
        crossing_of(branch->else_body, c);
        return;
    }
    if (const ir::SwitchStmt *sw = stmt.as<ir::SwitchStmt>()) {
        for (const ir::Stmt &arm : sw->arms) {
            crossing_of(arm, c);
        }
        return;
    }
    c.barrier = true;
}

inline Crossing crossing_of(const ir::Stmt &stmt) {
    Crossing c;
    crossing_of(stmt, c);
    return c;
}

struct RenameVar : ir::Mutator {
    std::string from;
    std::string to;
    ir::Expr visit(const ir::Var *node) override {
        if (node->name == from) {
            return ir::Var::make(node->type, to);
        }
        return node;
    }
};

} // namespace opt
} // namespace bonsai
