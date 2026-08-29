#pragma once

#include <vector>

#include "Expr.h"
#include "IRHandle.h"
#include "Resource.h"
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
    LetStmt,
    IfElse,
    DoWhile,
    While,
    Sequence,
    Allocate,
    Free,
    Store,
    Accumulate,
    Label,

    RecLoop,
    Match,
    Yield,
    Iterate,
    Scan,
    YieldFrom,
    ForAll,
    ForEach,
    ParFor,
    Continue,
    Launch,
    Append,
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
    std::vector<Expr> args;

    static Stmt make(std::vector<Expr> args);

    static const IRStmtEnum node_type = IRStmtEnum::Print;
};

struct Return : StmtNode<Return> {
    Expr value;

    static Stmt make(Expr value);
    // A void return statement.
    static Stmt make();

    static const IRStmtEnum node_type = IRStmtEnum::Return;
};

// Non-mutable assignment.
struct LetStmt : StmtNode<LetStmt> {
    WriteLoc loc;
    Expr value;

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

    static const IRStmtEnum node_type = IRStmtEnum::DoWhile;
};

struct While : StmtNode<While> {
    Expr cond;
    Stmt body;

    static Stmt make(Expr cond, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::While;
};

struct Sequence : StmtNode<Sequence> {
    std::vector<Stmt> stmts;

    static Stmt make(std::vector<Stmt> stmts);
    // Makes a single statement sequence.
    static Stmt make(Stmt stmt);

    static const IRStmtEnum node_type = IRStmtEnum::Sequence;
};

struct Allocate : StmtNode<Allocate> {
    WriteLoc loc;
    Expr value; // initial value. possibly undefined.
    enum Memory {
        Heap,
        Stack,
        Device,
        Host,
    };
    Memory memory;

    // Storage that nothing else can refer to.
    //
    // Set by lowering for the objects it invents -- the accumulator a tree
    // query folds into, say -- which the program never names and so can never
    // hold a second reference to. It is not something a `mut` local of the
    // program gets: two arguments of a call may perfectly well be the same
    // variable, and are.
    //
    // The point of saying it is what happens when such an object is passed to
    // a generated function: without it, a traversal that reads a ray through
    // one pointer and writes its answer through another has to assume the
    // write may have changed the ray, and re-reads it at every node.
    bool unaliased = false;

    static Stmt make(WriteLoc loc, Memory memory = Heap);
    static Stmt make(WriteLoc loc, Expr value, Memory memory = Heap,
                     bool unaliased = false);

    static const IRStmtEnum node_type = IRStmtEnum::Allocate;
};

struct Free : StmtNode<Free> {
    Expr value;
    static Stmt make(Expr var);

    static const IRStmtEnum node_type = IRStmtEnum::Free;
};

// Assignment to mutable value.
struct Store : StmtNode<Store> {
    WriteLoc loc;
    Expr value;
    // Predication, for a store that writes one element per lane (see Ramp).
    // Undefined means every lane writes; otherwise a boolean vector with one
    // entry per lane of `value`, and only the enabled lanes are written.
    Expr mask;

    static Stmt make(WriteLoc loc, Expr value, Expr mask = Expr());

    static const IRStmtEnum node_type = IRStmtEnum::Store;
};

struct Accumulate : StmtNode<Accumulate> {
    enum OpType {
        Add,
        Mul,
        Sub,
        // (key, value) = select(new_value < value, (new_key, new_value), (key,
        // value))
        Argmin,
        // (key, value) = select(new_value > value, (new_key, new_value), (key,
        // value))
        Argmax,
        // TODO: add more.
        Min,
        Max,
    };
    WriteLoc loc;
    OpType op;
    Expr value;

    static Stmt make(WriteLoc loc, OpType op, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Accumulate;
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
    std::vector<TypedVar> args;
    Stmt body;

    static Stmt make(std::vector<TypedVar> args, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::RecLoop;
};

struct Match : StmtNode<Match> {
    using Arms = std::vector<std::pair<BVH_t::Node, Stmt>>;
    Expr loc; // Of type BVH_t
    Arms arms;

    static Stmt make(Expr loc, Arms arms);

    static const IRStmtEnum node_type = IRStmtEnum::Match;
};

// Include this datum as an output
struct Yield : StmtNode<Yield> {
    Expr value;

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Yield;
};

// Include each datum as an output
struct Iterate : StmtNode<Iterate> {
    Expr value; // must be an iterable

    static Stmt make(Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Iterate;
};

// Recursively traverse this tree for all datums
struct Scan : StmtNode<Scan> {
    std::optional<AggOp::OpType> op;
    Expr value;

    static Stmt make(std::optional<AggOp::OpType> op, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Scan;
};

// Recursively evaluate the query on this tree
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
    Stmt body;

    Type index_type() const;

    // Returns the iteration count.
    Expr count() const;

    static Stmt make(std::string index, Slice slice, Stmt body);

    static const IRStmtEnum node_type = IRStmtEnum::ForAll;
};

// TODO: We just want Sequential for and parallel for, no need for
// foreach/forall/parfor/etc
struct ParFor : StmtNode<ParFor> {
    struct Slice {
        Expr begin, end, stride;
    } slice;
    std::string index;
    Stmt body;

    // The hardware a schedule bound this loop to, if any. A parfor says its
    // iterations may run in any order; this says what to run them on, and one
    // without it is emitted as an ordinary sequential loop. Set by bind() and
    // acted on when code is generated -- see Lower/Bindings.h.
    std::optional<Resource> binding;

    Type index_type() const;

    // Returns the iteration count.
    Expr count() const;

    static Stmt make(std::string index, Slice slice, Stmt body,
                     std::optional<Resource> binding = std::nullopt);

    static const IRStmtEnum node_type = IRStmtEnum::ParFor;
};

// End iterating inside this loop.
struct Continue : StmtNode<Continue> {
    static Stmt make();

    static const IRStmtEnum node_type = IRStmtEnum::Continue;
};

// Launch n calls to func with arguments
struct Launch : StmtNode<Launch> {
    std::string func;
    Expr n;
    std::vector<Expr> args;
    static Stmt make(std::string func, Expr n, std::vector<Expr> args);

    static const IRStmtEnum node_type = IRStmtEnum::Launch;
};

struct Append : StmtNode<Append> {
    WriteLoc loc;
    Expr value;
    static Stmt make(WriteLoc loc, Expr value);

    static const IRStmtEnum node_type = IRStmtEnum::Append;
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