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
    StringImm,
    Var,
    Extrema,
    SizeOf,
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
    Construct,
    UnionOf,
    Access,
    Unwrap,
    // Calls
    Intrinsic,
    Generator,
    Lambda,
    GeomOp,
    SetOp,
    AggOp,
    Call,
    Instantiate,
    // Pointer operations
    PtrTo,
    Deref,
    AtomicAdd,
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

struct StringImm : ExprNode<StringImm> {
    std::string value;

    static Expr make(std::string value);
    static const IRExprEnum node_type = IRExprEnum::StringImm;
};

struct Var : ExprNode<Var> {
    std::string name;

    static Expr make(Type t, const std::string &name);

    static const IRExprEnum node_type = IRExprEnum::Var;
};

// Maximum value of a type (inf for float)
struct Extrema : ExprNode<Extrema> {
    enum OpType {
        eps,
        inf,
    };
    OpType op;
    static Expr make(Type t, OpType op);

    static const IRExprEnum node_type = IRExprEnum::Extrema;
};

// How many bytes a value of `of` occupies in memory, including any padding:
// the distance between consecutive elements of an array of them.
//
// This is a question only a backend can answer -- a `vector[f32, 3]` is
// twelve bytes of data but occupies sixteen, and another target could say
// otherwise -- so it stays symbolic until code generation rather than being
// computed by whoever needed it. Vectorization needs it to read an array of
// per-lane vectors one component at a time.
struct SizeOf : ExprNode<SizeOf> {
    Type of;

    // `as` is the type of the resulting number, so that it can be used in
    // index arithmetic without a cast.
    static Expr make(Type of, Type as);

    static const IRExprEnum node_type = IRExprEnum::SizeOf;
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
    // Different modes of casting.
    enum class Mode {
        // e.g., reinterpret_cast<T*> in C++
        Reinterpret = 0,
        // e.g., (T) in C++
        Convert = 1,
    };
    Expr value;
    Mode mode;

    static Expr make(Type type, Expr value, Mode mode = Mode::Convert);
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
// A value of an ADT: one of its variants, with that variant's fields.
//
// `type` is the ADT. Lower/ADTs.cpp turns this into whatever the layout says a
// value of it looks like -- today a tag beside a union -- so nothing before
// that pass has to know how one is stored.
struct Construct : ExprNode<Construct> {
    std::string variant;
    std::vector<Expr> args;

    static Expr make(Type adt, std::string variant, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Construct;
};

// A union holding one of its members: C's `(union U){.member = value}`.
//
// A union is storage read at one of several types, so a value of one is really
// a value written into that storage -- which is why this exists rather than
// Build, whose arguments line up with fields at their own offsets.
struct UnionOf : ExprNode<UnionOf> {
    std::string member;
    Expr value;

    static Expr make(Type union_type, std::string member, Expr value);

    static const IRExprEnum node_type = IRExprEnum::UnionOf;
};

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
        // The inverse cosine, which like atanh below has no LLVM intrinsic and
        // so becomes a call to libm's acosf. Needed wherever an angle has to be
        // recovered from a direction -- a sphere's parameterization, where the
        // polar angle is the arc cosine of a coordinate.
        acos,
        // The inverse hyperbolic tangent, which LLVM has no intrinsic for and
        // so becomes a call to libm's atanhf. Here rather than left to be
        // written as 0.5*log((1+x)/(1-x)) because the two differ in the last
        // bit for about half of all inputs, and a caller that rounds the result
        // -- as a spectrum sampled per nanometre does -- turns that last bit
        // into a visible difference.
        atanh,
        cos,
        cosh,
        cross,
        dot,
        exp,
        fma,
        log,
        max,
        min,
        norm,
        pow,
        rand,
        round,
        sin,
        sqr,
        sqrt,
        tan,
        // TODO: more
    };

    OpType op;
    std::vector<Expr> args;

    static Expr make(OpType op, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Intrinsic;
};

// Useful iterator-generators
struct Generator : ExprNode<Generator> {
    enum OpType {
        iter,
        range,
        // TODO: more
    };

    OpType op;
    std::vector<Expr> args;

    static Expr make(OpType op, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Generator;
};

struct Lambda : ExprNode<Lambda> {
    std::vector<TypedVar> args;
    Expr value;

    static Expr make(std::vector<TypedVar> args, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Lambda;
};

// The geometric operators of Figure 1: the topological predicates of Egenhofer
// and Herring, the per-dimension ordering predicates, and the metrics.
// `lex`/`ltx` and friends are `a <=_x b` and `a <_x b`, one opcode per
// (relation, axis) pair.
struct GeomOp : ExprNode<GeomOp> {
    enum OpType {
        // Topological predicates.
        contains,
        covers,
        disjoint,
        equals,
        intersects,
        touches,
        within,
        // Ordering predicates, per dimension.
        lex,
        ley,
        lez,
        ltx,
        lty,
        ltz,
        // Metrics.
        distmax,
        distmin,

        opcount, // sentinel, do not remove!
    };

    OpType op;
    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const char *intrinsic_name(const OpType &op);

    static const IRExprEnum node_type = IRExprEnum::GeomOp;
};

// The set operators of Figure 2. For everything but `product`, a is a lambda
// over the set's elements and b is the set; for `product`, a and b are sets.
// The set-level `min` and `max` are spelled `minimum` and `maximum` because
// `min` and `max` already name the binary scalar intrinsics.
struct SetOp : ExprNode<SetOp> {
    enum OpType {
        all,
        any,
        argmax,
        argmin,
        filter,
        map,
        maximum,
        minimum,
        product,
        // TODO: geometric intrinsics for lambda
    };

    OpType op;

    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const IRExprEnum node_type = IRExprEnum::SetOp;
};

// A reduction over a set. `reduce` is the primitive of Figure 2; the others
// are sugar that expand into a map followed by a reduce (e.g. `count` maps
// every element to 1 and sums with identity 0).
struct AggOp : ExprNode<AggOp> {
    enum OpType { avg, count, prod, reduce, sum };

    OpType op;

    Expr a; // must be a set type

    // Only defined for `reduce`: the identity element, and the associative,
    // commutative binary function combining two partial results.
    Expr identity, combiner;

    static Expr make(OpType op, Expr a);
    static Expr make(Expr identity, Expr combiner, Expr a);

    static const IRExprEnum node_type = IRExprEnum::AggOp;
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

struct PtrTo : ExprNode<PtrTo> {
    Expr expr; // must be convertible to WriteLoc

    static Expr make(Expr expr);

    static const IRExprEnum node_type = IRExprEnum::PtrTo;
};

struct Deref : ExprNode<Deref> {
    Expr expr; // must be ptr
    // Predication, for a load that reads one element per lane (see Ramp).
    // Undefined means every lane reads; otherwise a boolean vector with one
    // entry per lane, and the disabled lanes read as zero rather than
    // touching memory.
    Expr mask;

    static Expr make(Expr expr, Expr mask = Expr());

    static const IRExprEnum node_type = IRExprEnum::Deref;
};

struct AtomicAdd : ExprNode<AtomicAdd> {
    Expr ptr;   // must be ptr<T>
    Expr value; // must be T

    static Expr make(Expr ptr, Expr value);

    static const IRExprEnum node_type = IRExprEnum::AtomicAdd;
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

struct DynArray_t : TypeNode<DynArray_t> {
    Type etype;

    // The maximum capacity of this array upon creation. If full, it will be
    // dynamically resized (handled in the backend code generation phase).
    Expr capacity;

    static Type make(Type etype, Expr capacity = Expr());

    static const IRTypeEnum node_type = IRTypeEnum::DynArray_t;
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
