#pragma once

#include <string>

#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include "Type.h"

namespace bonsai {
namespace ir {

struct Expr;

enum class IRExprEnum {
    IntImm,
    UIntImm,
    IdxImm,
    FloatImm,
    BoolImm,
    VecImm,
    Var,
    Infinity,
    BinOp,
    UnOp,
    Select,
    Cast,
    // Vector ops
    Broadcast,
    VectorReduce,
    VectorShuffle,
    Ramp,
    Extract,
    // Struct ops.
    Build,
    Access,
    Unwrap,
    // Calls
    Intrinsic,
    Lambda,
    GeomOp,
    SetOp,
    Call,
    Instantiate,
};

using IRExprNode = IRNode<Expr, IRExprEnum>;

/** This is necessary to get mutate() to work properly...
 *  They all contain their types (e.g. Int(32), Float(32))
 */
struct BaseExprNode : public IRExprNode {
    BaseExprNode(IRExprEnum t) : IRExprNode(t) {}
    virtual Expr mutate_expr(Mutator *m) const = 0;
    Type type;
};

template <typename T>
struct ExprNode : public BaseExprNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Expr mutate_expr(Mutator *m) const override;
    ExprNode() : BaseExprNode(T::node_type) {}
    ~ExprNode() override = default;
};

struct Expr : public IRHandle<IRExprNode> {
    /** Make an undefined expr */
    Expr() = default;

    /** Make an expr from a concrete expr node pointer (e.g. Add) */
    Expr(const IRExprNode *n) : IRHandle<IRExprNode>(n) {}

    /** Override get() to return a BaseExprNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseExprNode *get() const { return (const BaseExprNode *)ptr; }

    // TODO: implement copy/move semantics!

    Type type() const { return get()->type; }

    explicit Expr(int8_t x);
    explicit Expr(int16_t x);
    Expr(int32_t x);
    explicit Expr(int64_t x);
    // TODO: floats, uints, etc.
};

template <typename T>
Expr ExprNode<T>::mutate_expr(Mutator *m) const {
    return m->visit((const T *)this);
}

struct IntImm : ExprNode<IntImm> {
    int64_t value;

    static Expr make(Type t, int64_t value);

    static const IRExprEnum node_type = IRExprEnum::IntImm;
};

struct UIntImm : ExprNode<UIntImm> {
    uint64_t value;

    static Expr make(Type t, uint64_t value);

    static const IRExprEnum node_type = IRExprEnum::UIntImm;
};

struct IdxImm : ExprNode<IdxImm> {
    int64_t value;

    static Expr make(int64_t value);

    static const IRExprEnum node_type = IRExprEnum::IdxImm;
};

struct FloatImm : ExprNode<FloatImm> {
    double value;

    static Expr make(Type t, double value);

    static const IRExprEnum node_type = IRExprEnum::FloatImm;
};

struct BoolImm : ExprNode<BoolImm> {
    bool value;

    static Expr make(bool value);

    static const IRExprEnum node_type = IRExprEnum::BoolImm;
};

struct VecImm : ExprNode<VecImm> {
    std::vector<ir::Expr> values;

    static Expr make(std::vector<ir::Expr> values);
    static const IRExprEnum node_type = IRExprEnum::VecImm;
};

struct Var : ExprNode<Var> {
    std::string name;

    static Expr make(Type t, const std::string &name);

    static const IRExprEnum node_type = IRExprEnum::Var;
};

// Maximum value of a type (inf for float)
struct Infinity : ExprNode<Infinity> {
    static Expr make(Type tan);

    static const IRExprEnum node_type = IRExprEnum::Infinity;
};

struct BinOp : ExprNode<BinOp> {
    enum OpType {
        Add,
        LAnd,
        Div,
        Eq,
        Le,
        Lt,
        Mod,
        Mul,
        Neq,
        LOr,
        Sub,
        Xor,
        BwAnd,
        BwOr,
        Shl,
        // Arithmetic on signed integral types, and logical otherwise.
        Shr,
    };

    OpType op;
    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const IRExprEnum node_type = IRExprEnum::BinOp;

    static bool is_numeric_op(const OpType &op);
    static bool is_boolean_op(const OpType &op);
};

struct UnOp : ExprNode<UnOp> {
    enum OpType { Neg, Not };

    OpType op;
    Expr a;

    static Expr make(OpType op, Expr a);

    static const IRExprEnum node_type = IRExprEnum::UnOp;
};

struct Select : ExprNode<Select> {
    Expr cond, tvalue, fvalue;

    static Expr make(Expr cond, Expr tvalue, Expr fvalue);

    static const IRExprEnum node_type = IRExprEnum::Select;
};

struct Cast : ExprNode<Cast> {
    Expr value;

    static Expr make(Type type, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Cast;
};

struct Broadcast : ExprNode<Broadcast> {
    uint32_t lanes;
    Expr value;

    static Expr make(uint32_t lanes, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Broadcast;
};

struct VectorReduce : ExprNode<VectorReduce> {
    enum OpType {
        Add,
        And,
        Idxmax, // argmax but only keep the index
        Idxmin, // argmin but only keep the index
        Max,
        Min,
        Mul,
        Or,
        // TODO: saturating_add?
    };

    OpType op;
    Expr value;

    static Expr make(OpType op, Expr value);

    static const IRExprEnum node_type = IRExprEnum::VectorReduce;
};

struct VectorShuffle : ExprNode<VectorShuffle> {
    Expr value;
    std::vector<Expr> idxs;

    static Expr make(Expr value, std::vector<Expr> idxs);

    static const IRExprEnum node_type = IRExprEnum::VectorShuffle;
};

struct Ramp : ExprNode<Ramp> {
    Expr base, stride;
    int lanes;

    static Expr make(Expr base, Expr stride, int lanes);

    static const IRExprEnum node_type = IRExprEnum::Ramp;
};

struct Extract : ExprNode<Extract> {
    Expr vec, idx;

    static Expr make(Expr vec, int idx);
    static Expr make(Expr vec, Expr idx);

    static const IRExprEnum node_type = IRExprEnum::Extract;
};

// Construct a value of a Type (e.g. Vector_t or Struct_t)
struct Build : ExprNode<Build> {
    std::vector<Expr> values;

    // TODO: add named-field variant (works well with default values).
    static Expr make(Type type, std::vector<Expr> values);
    // Named field constructor (for Struct_t only!)
    static Expr make(Type type, std::map<std::string, Expr> values);
    // Builds an empty struct -- useful when passing as a mutable argument.
    static Expr make(Type type);

    static const IRExprEnum node_type = IRExprEnum::Build;
};

// Access a value of a Struct_t
struct Access : ExprNode<Access> {
    std::string field;
    Expr value;

    static Expr make(std::string field, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Access;
};

// Reinterpret as a branch of a BVH_t
struct Unwrap : ExprNode<Unwrap> {
    size_t index;
    Expr value;

    static Expr make(size_t index, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Unwrap;
};

struct Intrinsic : ExprNode<Intrinsic> {
    // For now, just supporting (seemingly relevant) LLVM intrinsic ops:
    // https://llvm.org/docs/LangRef.html#standard-c-c-library-intrinsics
    enum OpType {
        abs,
        cos,
        cross,
        fma,
        max,
        min,
        sin,
        sqrt,
        // TODO: more
    };

    OpType op;
    std::vector<Expr> args;

    static Expr make(OpType op, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Intrinsic;
};

struct Lambda : ExprNode<Lambda> {
    std::vector<TypedVar> args;
    Expr value;

    static Expr make(std::vector<TypedVar> args, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Lambda;
};

struct GeomOp : ExprNode<GeomOp> {
    enum OpType {
        contains,
        distmax,
        distmin,
        intersects,
        // TODO: the rest

        opcount, // sentinel, do not remove!
    };

    OpType op;
    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const char *intrinsic_name(const OpType &op);

    static const IRExprEnum node_type = IRExprEnum::GeomOp;
};

// For Argmin/Map/Filter, a: Lambda, b: Set
// For Product, a and b are Sets
struct SetOp : ExprNode<SetOp> {
    enum OpType {
        argmin,
        filter,
        map,
        product,
        // TODO: reduce
        // TODO: geometric intrinsics for lambda
    };

    OpType op;

    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const IRExprEnum node_type = IRExprEnum::SetOp;
};

struct Call : ExprNode<Call> {
    Expr func;
    std::vector<Expr> args;

    static Expr make(Expr func, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Call;
};

struct Instantiate : ExprNode<Instantiate> {
    Expr expr;
    // Generic_t name -> replacement
    TypeMap types;

    static Expr make(Expr expr, TypeMap types);

    static const IRExprEnum node_type = IRExprEnum::Instantiate;
};

// TODO: need Load with more info than Halide, can load from arbitrary
// pointer...

// TODO: ??? Load, (?)Let

// TODO: this can't go in Type.h because Expr is an incomplete type there...
struct Array_t : TypeNode<Array_t> {
    Type etype;
    Expr size;

    static Type make(Type etype, Expr size);

    static const IRTypeEnum node_type = IRTypeEnum::Array_t;
};

} // namespace ir

template <>
inline RefCount &ref_count<ir::IRExprNode>(const ir::IRExprNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRExprNode>(const ir::IRExprNode *t) {
    delete t;
}

} // namespace bonsai
