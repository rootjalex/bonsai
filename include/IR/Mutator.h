#pragma once

#include "IRFwdDecl.h"

#include <utility>

namespace bonsai {
namespace ir {

struct Mutator {
    virtual Type mutate(const Type &type);
    virtual Interface mutate(const Interface &interface);
    virtual Expr mutate(const Expr &expr);
    virtual Stmt mutate(const Stmt &stmt);

    virtual std::pair<WriteLoc, bool> mutate_writeloc(const WriteLoc &loc);
    // protected:
    // Types
    virtual Type visit(const Void_t *);
    virtual Type visit(const Int_t *);
    virtual Type visit(const UInt_t *);
    virtual Type visit(const Index_t *);
    virtual Type visit(const Float_t *);
    virtual Type visit(const Bool_t *);
    virtual Type visit(const Ptr_t *);
    virtual Type visit(const Ref_t *);
    virtual Type visit(const Vector_t *);
    virtual Type visit(const Struct_t *);
    virtual Type visit(const Tuple_t *);
    virtual Type visit(const Array_t *);
    virtual Type visit(const Option_t *);
    virtual Type visit(const Set_t *);
    virtual Type visit(const Function_t *);
    virtual Type visit(const Generic_t *);
    virtual Type visit(const BVH_t *);
    // Interfaces
    virtual Interface visit(const IEmpty *);
    virtual Interface visit(const IFloat *);
    virtual Interface visit(const IVector *);
    // Exprs
    virtual Expr visit(const IntImm *);
    virtual Expr visit(const UIntImm *);
    virtual Expr visit(const FloatImm *);
    virtual Expr visit(const BoolImm *);
    virtual Expr visit(const VecImm *);
    virtual Expr visit(const Infinity *);
    virtual Expr visit(const Var *);
    virtual Expr visit(const BinOp *);
    virtual Expr visit(const UnOp *);
    virtual Expr visit(const Select *);
    virtual Expr visit(const Cast *);
    virtual Expr visit(const Broadcast *);
    virtual Expr visit(const VectorReduce *);
    virtual Expr visit(const VectorShuffle *);
    virtual Expr visit(const Ramp *);
    virtual Expr visit(const Extract *);
    virtual Expr visit(const Build *);
    virtual Expr visit(const Access *);
    virtual Expr visit(const Unwrap *);
    virtual Expr visit(const Intrinsic *);
    virtual Expr visit(const Generator *);
    virtual Expr visit(const Lambda *);
    virtual Expr visit(const GeomOp *);
    virtual Expr visit(const SetOp *);
    virtual Expr visit(const Call *);
    virtual Expr visit(const Instantiate *);
    virtual Expr visit(const PtrTo *);
    virtual Expr visit(const Deref *);
    // Stmts
    virtual Stmt visit(const CallStmt *);
    virtual Stmt visit(const Print *);
    virtual Stmt visit(const Return *);
    virtual Stmt visit(const LetStmt *);
    virtual Stmt visit(const IfElse *);
    virtual Stmt visit(const DoWhile *);
    virtual Stmt visit(const Sequence *);
    virtual Stmt visit(const Assign *);
    virtual Stmt visit(const Accumulate *);
    virtual Stmt visit(const Label *);
    virtual Stmt visit(const RecLoop *);
    virtual Stmt visit(const Match *);
    virtual Stmt visit(const Yield *);
    virtual Stmt visit(const Scan *);
    virtual Stmt visit(const YieldFrom *);
    virtual Stmt visit(const ForAll *);
    virtual Stmt visit(const ForEach *);
    virtual Stmt visit(const Continue *);
    virtual Stmt visit(const Launch *);
};

#define RESTRICT_MUTATOR(IRType, IRNODE)                                       \
    IRType visit(const IRNODE *) final {                                       \
        internal_error << "Restricted Mutator class does not handle: "         \
                       << typeid(IRNODE).name();                               \
    }

} // namespace ir
} // namespace bonsai
