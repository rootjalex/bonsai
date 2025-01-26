#pragma once

#include <vector>

#include "Expr.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"
#include "WriteLoc.h"

namespace bonsai {
namespace ir {

struct Stmt;

enum class IRStmtEnum {
    Return,
    Store,
    LetStmt,
    IfElse,
    Sequence,
    Assign,
    Accumulate,
};

using IRStmtNode = IRNode<Stmt, IRStmtEnum>;

/* This is necessary to get mutate() to work properly... */
struct BaseStmtNode : public IRStmtNode {
    BaseStmtNode(IRStmtEnum t) : IRStmtNode(t) {}
    virtual Stmt mutate_stmt(Mutator *m) const = 0;
};

template <typename T>
struct StmtNode : public BaseStmtNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Stmt mutate_stmt(Mutator *m) const override;
    StmtNode() : BaseStmtNode(T::_node_type) {}
    ~StmtNode() override = default;
};

struct Stmt : public IRHandle<IRStmtNode> {
    /** Make an undefined stmt */
    Stmt() = default;

    /** Make a stmt from a concrete stmt node pointer (e.g. Return) */
    Stmt(const IRStmtNode *n) : IRHandle<IRStmtNode>(n) {}

    /** Override get() to return a BaseStmtNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseStmtNode *get() const { return (const BaseStmtNode *)ptr; }

    // TODO: implement copy/move semantics!
};

template <typename T>
Stmt StmtNode<T>::mutate_stmt(Mutator *m) const {
    return m->visit((const T *)this);
}

struct Return : StmtNode<Return> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum _node_type = IRStmtEnum::Return;
};

struct Store : StmtNode<Store> {
    // TODO: predicate
    std::string name;
    Expr index;
    Expr value;

    static Stmt make(std::string name, Expr index, Expr value);

    static const IRStmtEnum _node_type = IRStmtEnum::Store;
};

// Non-mutable assignment.
struct LetStmt : StmtNode<LetStmt> {
    WriteLoc loc;
    Expr value;
    // TODO: this is now just an Assign, because parsing into SSA is hard.
    // Stmt body;

    // static Stmt make(std::string name, Expr value, Stmt body);
    static Stmt make(WriteLoc loc, Expr value);

    static const IRStmtEnum _node_type = IRStmtEnum::LetStmt;
};

struct IfElse : StmtNode<IfElse> {
    Expr cond;
    Stmt then_body;
    Stmt else_body;

    static Stmt make(Expr cond, Stmt then_body, Stmt else_body = Stmt());

    static const IRStmtEnum _node_type = IRStmtEnum::IfElse;
};

struct Sequence : StmtNode<Sequence> {
    std::vector<Stmt> stmts;

    static Stmt make(std::vector<Stmt> stmts);

    static const IRStmtEnum _node_type = IRStmtEnum::Sequence;
};

// Assignment to mutable value.
struct Assign : StmtNode<Assign> {
    WriteLoc loc;
    Expr value;
    bool mutating;

    static Stmt make(WriteLoc loc, Expr value, bool mutating);

    static const IRStmtEnum _node_type = IRStmtEnum::Assign;
};

struct Accumulate : StmtNode<Accumulate> {
    enum OpType {
        Add,
        Mul,
        // TODO: add more.
    };
    WriteLoc loc;
    OpType op;
    Expr value;

    static Stmt make(WriteLoc loc, OpType op, Expr value);

    static const IRStmtEnum _node_type = IRStmtEnum::Accumulate;
};

} // namespace ir

template <>
inline RefCount &ref_count<ir::IRStmtNode>(const ir::IRStmtNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRStmtNode>(const ir::IRStmtNode *t) {
    delete t;
}

} // namespace bonsai