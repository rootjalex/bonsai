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
    CallStmt,
    Print,
    Return,
    Store,
    LetStmt,
    IfElse,
    DoWhile,
    Sequence,
    Assign,
    Accumulate,
    Allocate,
    Label,

    RecLoop,
    Match,
    Yield,
    Scan,
    YieldFrom,
    ForAll,
    ForEach,
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
    StmtNode() : BaseStmtNode(T::node_type) {}
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

// A call to a function; the return value is ignored.
struct CallStmt : StmtNode<CallStmt> {
    Expr func;
    std::vector<Expr> args;

    static Stmt make(Expr func, std::vector<Expr> args);

    static const IRStmtEnum node_type = IRStmtEnum::CallStmt;
};

struct Print : StmtNode<Print> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Print;
};

struct Return : StmtNode<Return> {
    Expr value;

    static Stmt make(Expr value);
    // A void return statement.
    static Stmt make();

    static const IRStmtEnum node_type = IRStmtEnum::Return;
};

struct Store : StmtNode<Store> {
    // TODO: predicate
    std::string name;
    Expr index;
    Expr value;

    static Stmt make(std::string name, Expr index, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Store;
};

// Non-mutable assignment.
struct LetStmt : StmtNode<LetStmt> {
    WriteLoc loc;
    Expr value;
    // TODO: this is now just an Assign, because parsing into SSA is hard.
    // Stmt body;

    // static Stmt make(std::string name, Expr value, Stmt body);
    static Stmt make(WriteLoc loc, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::LetStmt;
};

struct IfElse : StmtNode<IfElse> {
    Expr cond;
    Stmt then_body;
    Stmt else_body;

    static Stmt make(Expr cond, Stmt then_body, Stmt else_body = Stmt());

    static const IRStmtEnum node_type = IRStmtEnum::IfElse;
};

struct DoWhile : StmtNode<DoWhile> {
    Stmt body;
    Expr cond;

    static Stmt make(Stmt body, Expr cond);

    static const IRStmtEnum _node_type = IRStmtEnum::DoWhile;
};

struct Sequence : StmtNode<Sequence> {
    std::vector<Stmt> stmts;

    static Stmt make(std::vector<Stmt> stmts);

    static const IRStmtEnum node_type = IRStmtEnum::Sequence;
};

// Assignment to mutable value.
struct Assign : StmtNode<Assign> {
    WriteLoc loc;
    Expr value;
    bool mutating;

    static Stmt make(WriteLoc loc, Expr value, bool mutating);

    static const IRStmtEnum node_type = IRStmtEnum::Assign;
};

struct Accumulate : StmtNode<Accumulate> {
    enum OpType {
        Add,
        Mul,
        Sub,
        // TODO: add more.
    };
    WriteLoc loc;
    OpType op;
    Expr value;

    static Stmt make(WriteLoc loc, OpType op, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Accumulate;
};

struct Allocate : StmtNode<Allocate> {
    std::string name;
    Type type;

    static Stmt make(std::string name, Type type);

    static const IRStmtEnum node_type = IRStmtEnum::Allocate;
};

// A labelled body of code.
// e.g. used for "holes" in Layout lowering.
// Can also be used for scheduling.
struct Label : StmtNode<Label> {
    std::string name;
    Stmt body;

    static Stmt make(std::string name, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::Label;
};

// A (currently inlined) recursive loop
// Contains `From` nodes that match the args list.
struct RecLoop : StmtNode<RecLoop> {
    struct Argument {
        std::string name;
        Type type;
    };
    std::vector<Argument> args;
    Stmt body;

    static Stmt make(std::vector<Argument> args, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::RecLoop;
};

struct Match : StmtNode<Match> {
    using Arms = std::vector<std::pair<BVH_t::Node, Stmt>>;
    Expr loc; // Of type BVH_t
    Arms arms;

    static Stmt make(Expr loc, Arms arms);

    static const IRStmtEnum node_type = IRStmtEnum::Match;
};

struct Yield : StmtNode<Yield> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Yield;
};

struct Scan : StmtNode<Scan> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Scan;
};

struct YieldFrom : StmtNode<YieldFrom> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::YieldFrom;
};

struct ForEach : StmtNode<ForEach> {
    std::string name;
    Expr iter; // array or vector
    Stmt body;

    static Stmt make(std::string name, Expr iter, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::ForEach;
};

struct ForAll : StmtNode<ForAll> {
    struct Slice {
        Expr begin, end, stride;
    } slice;
    std::string index;
    Stmt header; // let x = extract[<...>]
    Stmt body;   // use(x)

    static Stmt make(std::string index, Stmt header, Slice slice, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::ForAll;
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