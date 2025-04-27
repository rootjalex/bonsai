#pragma once

#include "IRFwdDecl.h"

namespace bonsai {
namespace ir {

struct Visitor {
    // Types
    virtual void visit(const Void_t *);
    virtual void visit(const Int_t *);
    virtual void visit(const UInt_t *);
    virtual void visit(const Index_t *);
    virtual void visit(const Float_t *);
    virtual void visit(const Bool_t *);
    virtual void visit(const Ptr_t *);
    virtual void visit(const Ref_t *);
    virtual void visit(const Vector_t *);
    virtual void visit(const Struct_t *);
    virtual void visit(const Tuple_t *);
    virtual void visit(const Array_t *);
    virtual void visit(const Option_t *);
    virtual void visit(const Set_t *);
    virtual void visit(const Function_t *);
    virtual void visit(const Generic_t *);
    virtual void visit(const BVH_t *);
    // Interfaces
    virtual void visit(const IEmpty *);
    virtual void visit(const IFloat *);
    virtual void visit(const IVector *);
    // Exprs
    virtual void visit(const IntImm *);
    virtual void visit(const UIntImm *);
    virtual void visit(const FloatImm *);
    virtual void visit(const BoolImm *);
    virtual void visit(const VecImm *);
    virtual void visit(const Infinity *);
    virtual void visit(const Var *);
    virtual void visit(const BinOp *);
    virtual void visit(const UnOp *);
    virtual void visit(const Select *);
    virtual void visit(const Cast *);
    virtual void visit(const Broadcast *);
    virtual void visit(const VectorReduce *);
    virtual void visit(const VectorShuffle *);
    virtual void visit(const Ramp *);
    virtual void visit(const Extract *);
    virtual void visit(const Build *);
    virtual void visit(const Access *);
    virtual void visit(const Unwrap *);
    virtual void visit(const Intrinsic *);
    virtual void visit(const Lambda *);
    virtual void visit(const GeomOp *);
    virtual void visit(const SetOp *);
    virtual void visit(const Call *);
    virtual void visit(const Instantiate *);
    // Stmts
    virtual void visit(const CallStmt *);
    virtual void visit(const Print *);
    virtual void visit(const Return *);
    virtual void visit(const Store *);
    virtual void visit(const LetStmt *);
    virtual void visit(const IfElse *);
    virtual void visit(const DoWhile *);
    virtual void visit(const Sequence *);
    virtual void visit(const Assign *);
    virtual void visit(const Accumulate *);
    virtual void visit(const Allocate *);
    virtual void visit(const Label *);
    virtual void visit(const RecLoop *);
    virtual void visit(const Match *);
    virtual void visit(const Yield *);
    virtual void visit(const Scan *);
    virtual void visit(const YieldFrom *);
    virtual void visit(const ForAll *);
    virtual void visit(const ForEach *);
    virtual void visit(const Continue *);
    // Layouts
    virtual void visit(const Name *);
    virtual void visit(const Pad *);
    virtual void visit(const Split *);
    virtual void visit(const Chain *);
    virtual void visit(const Group *);
    virtual void visit(const Materialize *);
};

#define RESTRICT_VISITOR(IRNODE)                                               \
    void visit(const IRNODE *) final {                                         \
        internal_error << "Restricted Visitor class does not handle: "         \
                       << typeid(IRNODE).name();                               \
    }

} // namespace ir
} // namespace bonsai
