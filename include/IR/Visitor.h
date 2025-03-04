#pragma once

#include "IRFwdDecl.h"

#include "Error.h"

#include <type_traits>
#include <typeinfo>

namespace bonsai {
namespace ir {

struct Visitor {
    // Types
    virtual void visit(const Void_t *);
    virtual void visit(const Int_t *);
    virtual void visit(const UInt_t *);
    virtual void visit(const Float_t *);
    virtual void visit(const Bool_t *);
    virtual void visit(const Ptr_t *);
    virtual void visit(const Vector_t *);
    virtual void visit(const Struct_t *);
    virtual void visit(const Tuple_t *);
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
    virtual void visit(const Intrinsic *);
    virtual void visit(const Lambda *);
    virtual void visit(const GeomOp *);
    virtual void visit(const SetOp *);
    virtual void visit(const Call *);
    virtual void visit(const Instantiate *);
    // Stmts
    virtual void visit(const Print *);
    virtual void visit(const Return *);
    virtual void visit(const Store *);
    virtual void visit(const LetStmt *);
    virtual void visit(const IfElse *);
    virtual void visit(const Sequence *);
    virtual void visit(const Assign *);
    virtual void visit(const Accumulate *);
};

// Utility to check if a type is in a list
template <typename T, typename... Ts>
inline constexpr bool type_in_list = std::disjunction_v<std::is_same<T, Ts>...>;

// RestrictedVisitor: Blocks overriding for disallowed types
template <typename... Disallowed>
struct RestrictedVisitor : Visitor {
    // Explicitly override and disable visit() for disallowed types
    template <typename T,
              std::enable_if_t<type_in_list<T, Disallowed...>, int> = 0>
    void visit(const T *node) {
        internal_error << "Node type: " << typeid(T).name()
                       << " not supported at this stage of lowering\n";
    }

    // Inherit visit() for allowed types, keeping them overridable
    using Visitor::visit;
};

} // namespace ir
} // namespace bonsai
