#pragma once

#include <optional>
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
    Undef,
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
    Shuffle,
    Ramp,
    Extract,
    // Struct ops.
    Build,
    Construct,
    Access,
    Unwrap,
    MatchExpr,
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
    RefTo,
    Deref,
    AtomicAdd,
    StoredElement,
    TiledArray,
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

    // A vector of the values' own type, as wide as there are values.
    static Expr make(std::vector<ir::Expr> values);
    // A vector of the given type, which says whether it is packed storage
    // (see Vector_t::packed); the values are its elements.
    static Expr make(Type type, std::vector<ir::Expr> values);
    static const IRExprEnum node_type = IRExprEnum::VecImm;
};

struct StringImm : ExprNode<StringImm> {
    std::string value;

    static Expr make(std::string value);
    static const IRExprEnum node_type = IRExprEnum::StringImm;
};

// A value of `type` that nothing reads: what a bypass hands a join whose
// first source is the region it skips, and what a skipped call's continuation
// is handed in place of the call's result (see SSA/Linearize.h and
// specialize_calls in SSA/Vectorize.cpp). Any value of the type would do, and
// saying so lets the backend build none -- LLVM's `undef`. Never a value a
// lane's answer depends on.
struct Undef : ExprNode<Undef> {
    static Expr make(Type type);
    static const IRExprEnum node_type = IRExprEnum::Undef;
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

// `select(cond, tvalue, fvalue)`: both arms are evaluated and one is chosen,
// a conditional move -- the branchless form a program writes on purpose,
// which the backends keep branchless. With `lazy`, `ite(cond, tvalue,
// fvalue)`: C's `cond ? tvalue : fvalue`, evaluating only the arm it takes,
// which the SSA conversion lowers as control flow. The two are different
// operations: an arm with a call, an unwrap or a division in it may only
// run under its condition, and an arm that is a value already computed is
// cheaper picked than branched to.
struct Select : ExprNode<Select> {
    Expr cond, tvalue, fvalue;
    bool lazy = false;

    static Expr make(Expr cond, Expr tvalue, Expr fvalue, bool lazy = false);

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

// A vector made by picking elements out of one or more vectors, each by a
// constant index into their concatenation: Halide's Shuffle. The constant
// indices are the point. A shuffle whose indices are known is a permutation
// the backend can lower as a whole -- and the ones that matter are not
// arbitrary permutations but a few shapes with names, which this node
// recognizes so that code generation can pick the lowering by shape:
//
//   interleave(a, b, c)   a0 b0 c0 a1 b1 c1 ..     structure of arrays to
//                                                  array of structures
//   transpose(v, f)       v0 vf v2f .. v1 v(f+1) .. the inverse: every f-th
//                                                  element, f vectors' worth
//   concat(a, b)          a0 .. an b0 .. bn
//   slice(v, b, s, n)     vb v(b+s) .. v(b+(n-1)s)
//   broadcast(v, k)       v v .. v                 k copies
//
// The interleave and the transpose are how an array of structures in a gang
// of lanes -- one struct per lane -- is turned into the structure of vectors
// the lanes compute on and back; see
// CodeGen_LLVM::interleave_vectors for the lowering, which follows Halide's,
// after Catanzaro, Keller and Garland, "A Decomposition for In-place Matrix
// Transposition", PPoPP 2014.
//
// Distinct from VectorShuffle above, whose indices are values: that is a
// per-element gather out of a vector, lowered one element at a time, and it
// is what the source's `shuffle(v, i, j, ..)` builds when the indices are not
// constants.
struct Shuffle : ExprNode<Shuffle> {
    // All of one element type. A scalar counts as a vector of one lane.
    std::vector<Expr> vectors;
    // One per lane of the result, into the concatenation of `vectors`. A
    // negative index is a lane whose value does not matter.
    std::vector<int> indices;

    static Expr make(std::vector<Expr> vectors, std::vector<int> indices);

    // The structured shapes, by name.
    static Expr make_interleave(std::vector<Expr> vectors);
    static Expr make_concat(std::vector<Expr> vectors);
    static Expr make_broadcast(Expr vector, int factor);
    static Expr make_slice(Expr vector, int begin, int stride, int size);
    static Expr make_extract_element(Expr vector, int index);
    // Every `factor`-th element gathered together: the elements at 0, f, 2f,
    // .. then those at 1, f + 1, .. and so on, which is `factor` slices of
    // stride `factor` concatenated -- the inverse of interleaving `factor`
    // vectors. The lanes must divide by `factor`.
    static Expr make_transpose(Expr vector, int factor);

    // The lanes `vectors` contribute between them.
    int input_lanes() const;

    bool is_interleave() const;
    bool is_concat() const;
    bool is_broadcast() const;
    int broadcast_factor() const;
    bool is_slice() const;
    int slice_begin() const;
    int slice_stride() const;
    bool is_extract_element() const;
    bool is_transpose() const;
    int transpose_factor() const;

    static const IRExprEnum node_type = IRExprEnum::Shuffle;
};

struct Ramp : ExprNode<Ramp> {
    Expr base, stride;
    int lanes;

    static Expr make(Expr base, Expr stride, int lanes);

    static const IRExprEnum node_type = IRExprEnum::Ramp;
};

struct Extract : ExprNode<Extract> {
    Expr vec, idx;
    // Predication, for a read of one element per lane out of memory (a
    // gather: `idx` a vector). Undefined means every lane reads; otherwise a
    // boolean vector with one entry per lane, and a disabled lane reads as
    // zero rather than touching memory -- its index may be anything, a
    // sentinel the program only ever tests, and a read at it would be a read
    // the program never makes.
    Expr mask;

    static Expr make(Expr vec, int idx);
    static Expr make(Expr vec, Expr idx);
    static Expr make(Expr vec, Expr idx, Expr mask);

    static const IRExprEnum node_type = IRExprEnum::Extract;
};

// Construct a value of a Type (e.g. Vector_t or Struct_t)
// A value of an ADT: one of its variants, with that variant's fields.
//
// `type` is the ADT. Lower/ADTs.cpp turns this into whatever the layout says a
// value of it looks like -- a tag beside the arm's fields as 32-bit words, or
// a handle into a pool -- so nothing before that pass has to know how one is
// stored.
struct Construct : ExprNode<Construct> {
    std::string variant;
    std::vector<Expr> args;

    static Expr make(Type adt, std::string variant, std::vector<Expr> args);

    static const IRExprEnum node_type = IRExprEnum::Construct;
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
    // With the result type given rather than looked up in the base.
    //
    // For the one access whose type cannot be read off the thing it is taken
    // from: a set's root augmentation, `blas.AABB`. A `set[Triangle]` does not
    // know which tree backs it -- the schedule says that, and it is read after
    // the elements are -- so the geometry named on the right is what gives the
    // access its type.
    static Expr make(std::string field, Expr value, Type type);

    static const IRExprEnum node_type = IRExprEnum::Access;
};

// Reinterpret as one arm of a sum: a node of a BVH_t, or a variant of an
// ADT_t. The type is that arm's struct, so its fields can be read off it.
//
// Meaningful only where the value is known to be that arm -- inside the
// corresponding arm of a match. Reading a field of the variant a value is not
// reads whatever those bytes happen to be.
struct Unwrap : ExprNode<Unwrap> {
    size_t index;
    Expr value;

    static Expr make(size_t index, Expr value);

    static const IRExprEnum node_type = IRExprEnum::Unwrap;
};

// A match on a variant that is a value: one expression per arm, all of one
// type, and the whole is whichever the value's variant selects.
//
//     match p { Geom(g) => set[Geometric]{g}, Inst(m, blas) => blas }
//
// The arms do not bind names. A field of the matched variant is read as
// `Unwrap(k, value).field` -- what the surface syntax's bindings stand for --
// so an arm is a plain expression over the enclosing scope, and every
// analysis that walks expressions sees through it with nothing to scope.
//
// Where it goes depends on its type. A set-typed one is part of a query, and
// the traversal built over the query opens it: a match statement whose arms
// each traverse their own set, which is how a tree of mixed primitives puts
// a plain element and a nested tree side by side. A value-typed one is a
// branch, and LowerADTs makes it one.
struct MatchExpr : ExprNode<MatchExpr> {
    struct Arm {
        std::string variant;
        Expr value;
    };

    Expr value; // of an ADT type
    std::vector<Arm> arms;

    static Expr make(Expr value, std::vector<Arm> arms);

    static const IRExprEnum node_type = IRExprEnum::MatchExpr;
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
        // The inverse sine, the companion of acos and, like it, a call to
        // libm's asinf since LLVM has no intrinsic for it. Needed wherever an
        // angle is recovered from a chord -- AngleBetween, in the
        // spherical-triangle sampling of a mesh area light.
        asin,
        // The inverse hyperbolic tangent, which LLVM has no intrinsic for and
        // so becomes a call to libm's atanhf. Here rather than left to be
        // written as 0.5*log((1+x)/(1-x)) because the two differ in the last
        // bit for about half of all inputs, and a caller that rounds the result
        // -- as a spectrum sampled per nanometre does -- turns that last bit
        // into a visible difference.
        atanh,
        // Two-argument arc tangent: the angle of the vector (x, y), which is
        // what recovers an azimuth from a direction. Not `atan(y/x)`, which
        // loses the quadrant and divides by zero on the axis. Like the two
        // above it has no LLVM intrinsic and becomes a call to libm's atan2f.
        atan2,
        // The number of leading zero bits of an integer, and the width of the
        // type for zero: llvm.ctlz. One machine instruction, and the way a
        // ceiling of log2 is taken -- which is what division by an invariant
        // integer needs of its divisor (see SSA/InvariantDivision.h).
        clz,
        // The number of trailing zero bits of an integer, and the width of
        // the type for zero: llvm.cttz. The index of the lowest bit set --
        // the one lane a mask with one bit on names, read straight from the
        // mask as an integer (`kmov`, `tzcnt`; Embree's `bsf`) where a
        // compress of the vector and a read of lane 0 would be
        // (SSA/Simplify.cpp).
        ctz,
        // `compress(v, mask)`: the lanes of `v` the mask has on, packed to
        // the front of a vector of v's type in their lane order; the lanes
        // past them -- from the `popcount(mask)`th on -- hold whatever they
        // hold, and a reader that looks there gets no particular value.
        // `compress(v, mask, fill)` fills them with `fill`'s lanes instead
        // (a vector of v's type, or one element of it in every lane): LLVM's
        // passthru, Embree's `compact(mask, a, b)`. The two-argument form is
        // Embree's own `compact`, which merges into the vector itself, and
        // what a traversal wants of every compress it makes: it reads the
        // packed lanes alone, and the instruction then merges into whichever
        // register it has rather than one zeroed first. The register form of
        // the compacting store (see ir::Store::compact): AVX-512's
        // `vpcompressd`/`vpcompressq`, and LLVM's vector.compress where the
        // machine has nothing. What a traversal does to the children a ray
        // hit before sorting them, so that a network over the first
        // `popcount(mask)` lanes orders the hits alone (Embree's
        // bvh_traverser1.h, `vint8::compact`; see SSA/SortRecursion.cpp).
        compress,
        cos,
        cosh,
        cross,
        // The multiplier that turns a division by its argument into a
        // multiply-high (Granlund & Montgomery, "Division by Invariant
        // Integers using Multiplication", PLDI 1994): for an unsigned d the
        // "round-up" multiplier floor(2^N (2^l - d) / d) + 1 with l =
        // ceil(log2 d), and for a signed d the one for |d|, floor(2^(N+l-1) /
        // |d|) + 1 taken modulo 2^N, with l at least one. Both fit the
        // argument's N bits. Defined for every input, zero included, since
        // it is taken once where the divisor is defined and lanes that never
        // divide may hold anything there; only the backend's wide division
        // knows how to compute it, which is why it is an intrinsic and the
        // arithmetic around it is not (see SSA/InvariantDivision.h).
        div_multiplier,
        dot,
        exp,
        fma,
        log,
        max,
        min,
        // The high half of the full product of two N-bit integers -- of the
        // 2N-bit product, the top N bits -- signed or unsigned as the type is.
        // What a division by an invariant integer multiplies by (see
        // div_multiplier), and one instruction on every machine that has a
        // widening multiply.
        mulhi,
        norm,
        // `permute(v, indices)`: lane k of the result is lane `indices[k]`
        // of `v`, the indices a vector computed at run time -- where
        // ir::Shuffle's are constants the backend lowers by shape. Halide's
        // `dynamic_shuffle`; the machine's variable permute, `vpermps`,
        // `vpermd`, `vpermq` on x86 (CodeGen_X86::dynamic_shuffle), and an
        // extract per lane where there is none. The indices must be in
        // range, `0 <= indices[k] < lanes(v)`; the result has the indices'
        // lanes and v's element. What takes a node's children and bounds
        // out by their sorted order in one instruction each
        // (SSA/SortRecursion.cpp), as Embree's `permuteExtract` does.
        permute,
        pow,
        // `prefetch(ptr, bytes)`: bring the `bytes` bytes at `ptr` into the
        // first-level cache, one line at a time, and compute nothing --
        // LLVM's llvm.prefetch (read, high locality), `prefetcht0` on x86.
        // A hint: it faults on nothing and changes no value, which is why
        // it is an effect and not a load. Written by the `prefetch`
        // scheduling directive (ir::Prefetch) as `prefetch(ref)` of a tree
        // reference, which Lower/Layouts.cpp turns into the row's address
        // and size; a gang's (SSA/Vectorize.cpp) carries a pointer per lane
        // and the lanes' mask as a third operand, and the backend issues it
        // for each lane that is on (CodeGen_LLVM::emit_prefetch). Embree's
        // `BVH::prefetch` of a hit child, four `prefetchL1` for a BVH8 node.
        prefetch,
        rand,
        // The machine's approximate reciprocal of a float, or of each lane
        // of a vector of them: the one instruction every machine has for
        // it, and nothing after it. x86's `vrcp14ps` (a relative error
        // under 2^-14) or `rcpps` (under 1.5 * 2^-12), the GPU's
        // `rcp.approx.f32`, ARM's `frecpe` were there a backend; the
        // quotient on a target with no estimate to name and in C++
        // (CodeGen_LLVM::reciprocal_approx, CodeGen_X86). Halide's
        // `fast_inverse`. The precision is the machine's, which is the
        // point: a library refines the estimate its own way, and a program
        // transcribing the library writes that refinement out -- Embree's
        // `r + r (1 - m r)` as two fused multiply-adds, enoki's `2r - (r m)
        // r` as one fused multiply-subtract (apps/rtq/metrics/embree.bonsai
        // and fcpw.bonsai), which round differently in a third of all
        // inputs -- rather than the compiler picking one for everybody.
        rcp_approx,
        round,
        sin,
        sinh,
        sqr,
        sqrt,
        tan,
        // A 2D texture sampled by the GPU's texture units, with gradients:
        // `tex_sample_grad_2d(texture : u64, st : vec2f, dstdx : vec2f,
        // dstdy : vec2f) -> vec4f`, CUDA's `tex2DGrad` on a texture object
        // -- the unit picks the mipmap level from the gradients, filters
        // within it and wraps at the edges as the object was created to
        // (runtime/bonsai_cuda.h, bonsai_cuda_texture_create). Only the PTX
        // backend lowers it (`tex.grad.2d.v4.f32.f32`); a program reaches
        // it only through a function a schedule bound to TextureUnit (see
        // ir::Bind::lambda), so the same program is exact software on the
        // CPU. `dstdx` and `dstdy` are the texture coordinate's derivatives
        // along the two screen axes, in the unit's convention: the texture
        // unit's (x, y) gradient pairs are what is passed, whatever the
        // caller means by them.
        tex_sample_grad_2d,
        // The ray tracing hardware, through OptiX: what a query a schedule
        // bound to RTCore lowers to (Lower/Trees.cpp), and what the programs
        // it makes for the hits read. Only CodeGen_OptiX lowers them, each
        // to the `_optix_*` call the OptiX device headers spell as inline
        // PTX; no program writes one.
        //
        // `rt_trace(traversable : u64, o : vec3f, d : vec3f, tmin : f32,
        // tmax : f32, flags : u32, context : ptr) -> u32`: traces the ray
        // through the acceleration structure, running the hit programs of
        // the query the context's type names, with the context's address in
        // the two payload words; the result is nothing, and the call has an
        // effect (the programs write the context).
        //
        // Promoted form, when the context fits the hardware's payload
        // registers (ir::rt_context_words): the context travels BY VALUE --
        // `rt_trace(..., context : _RTContext_T) -> _RTContext_T` -- its
        // words in the payload registers themselves, and the result is the
        // context as the programs left it. No memory holds it anywhere:
        // the programs read and write the words (rt_payload,
        // rt_set_payload), and the raygen reads the answer out of the
        // trace's own returns.
        rt_trace,
        // In a program of a promoted-context query (Lower/Trees.cpp):
        // one 32-bit word of the context, `rt_payload(k : u32) -> u32`
        // (`optixGetPayload`), and its write,
        // `rt_set_payload(k : u32, w : u32) -> u32` (`optixSetPayload`,
        // an effect; the result is nothing). A generated program never
        // reads a word it writes -- the hit and intersection programs read
        // the query's words, the miss and closest-hit programs write the
        // answer's -- so a read is pure.
        rt_payload,
        rt_set_payload,
        // In a hit program: the hit's ray parameter (`optixGetRayTmax`).
        rt_hit_t,
        // In a hit program of a hardware-triangle input: the hit's
        // barycentrics as the hardware reports them
        // (`optixGetTriangleBarycentrics`) -- a vec2f (b1, b2), the weights
        // of the triangle's second and third vertices as the build input
        // stored them. Undefined over a custom-primitive input, so only the
        // triangle flavor of a hit program reads it (Lower/Trees.cpp).
        rt_barycentrics,
        // In a hit, intersection or miss program: the traced ray's origin
        // and direction in the frame the ray was traced in
        // (`optixGetWorldRayOrigin`, `optixGetWorldRayDirection`), each a
        // vec3f. These are the very operands the raygen's `rt_trace` passed
        // -- the query's own ray, bit for bit -- so a program that reads
        // the query's ray reads it from the hardware's ray state instead
        // of fetching it back out of the context's memory (Lower/Trees.cpp
        // rebuilds `q` from them; only fields the hardware does not carry,
        // the extent, still come from the context).
        rt_ray_origin,
        rt_ray_direction,
        // In a hit program: the primitive's index within its build input
        // (`optixGetPrimitiveIndex`), and the instance id of the instance
        // the hit is in (`optixGetInstanceId`).
        rt_primitive_index,
        rt_instance_id,
        // In a hit program: the 32-bit base the build input's shader binding
        // table record carries -- where the input's elements start in the
        // tree's storage.
        rt_sbt_base,
        // In an intersection program: reports a hit at `t : f32`; true when
        // it was accepted. An effect.
        rt_report_hit,
        // In an any-hit program: rejects the candidate. An effect; the result
        // is nothing.
        rt_ignore_hit,
        // The acceleration structure handle of a tree whose layout is the
        // hardware's (`rt_traversable(tree) -> u64`); lowered to the layout
        // struct's field by Lower/Layouts.cpp.
        rt_traversable,
        // In a raygen program: a coherence hint ahead of the stage that
        // follows. `rt_reorder(key : u32, bits : u32) -> u32` asks the
        // hardware to reorder the launch's threads so that those whose low
        // `bits` bits of `key` agree run together from here on (OptiX's
        // `optixReorder`, shader execution reordering). Made by
        // Lower/Sorts.cpp for `f.sort(g, |args| key)`, a sort on the call
        // of `g`. An effect with no result; a hint, so where there is no
        // reordering hardware -- the host, a CUDA kernel -- it is nothing.
        rt_reorder,
        // A value reduced across the threads of a GPU block -- the fold of
        // every thread's argument under the operation, in every thread:
        // `block_reduce_add(v)`, `_mul`, `_min`, `_max`, over a 32-bit float
        // or integer alone or as the lanes of a short vector. Made by
        // SSA/BlockAccumulates.h in place of an atomic every thread of a
        // block made to one word; only the PTX backend lowers it (a warp
        // shuffle tree, a shared slot per warp, a barrier), and no program
        // writes one.
        block_reduce_add,
        block_reduce_max,
        block_reduce_min,
        block_reduce_mul,
        // TODO: more
    };

    OpType op;
    std::vector<Expr> args;

    static Expr make(OpType op, std::vector<Expr> args);
    // Whether an intrinsic does something besides compute its value --
    // advances the random generator, traces a ray, reports or rejects a
    // hit -- so that a call whose value goes unread is kept.
    static bool has_effects(OpType op);

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
        // The hit's record: `distmin`'s witness. `intersection(q, G)` answers
        // `Hit { t : f32; b : vec2f }` -- the type fixed by convention as
        // distmin's f32 is, with t = inf on a miss exactly as distmin
        // answers inf -- holding the ray parameter the
        // metric keys and the surface parameters of the point it names
        // (a triangle's barycentrics; a shape whose point `t` alone fixes
        // leaves `b` zero). This is what the RT cores report of a hit, so a
        // query that carries it hands its consumers the hit's geometry
        // without a second intersection; Lower/Trees.cpp recognizes it as a
        // map annotation beside the metric and, under bind(RTCore), reads it
        // from the hardware in the hit programs instead of computing it.
        intersection,
        // Motions. The odd one out, and deliberately here rather than beside
        // the relations: every op above relates two extents and answers a
        // bool or a scalar, while this one takes a *motion* and an extent and
        // answers an extent -- the same extent, somewhere else.
        //
        // It is an intrinsic rather than an ordinary function because two
        // things about it have to be known rather than guessed. It dispatches
        // on the type it moves, so the same motion applies to a triangle at a
        // leaf and to a bounding box at a node, which is what lets a mapped
        // tree's bounds be the mapped bounds. And it has an inverse, which is
        // what lets a relation against a moved extent be rewritten as the
        // same relation against a moved query -- transforming one ray per
        // instance rather than every triangle in it.
        transform,
        // That inverse. `untransform(m, x)` is `x` moved by the motion undone,
        // and pbrt's `Transform::ApplyInverse` where `transform` is its
        // `operator()`. Dispatched on the type it moves exactly as `transform`
        // is, so a program supplies `untransform(t : Transform, r : Ray)` the
        // way it supplies `transform(t : Transform, b : AABB)`.
        //
        // Supplying one for a *query* type is a declaration, and it is worth
        // knowing what is being declared. Opt/PullQueries.cpp rewrites
        // `rel(q, transform(m, x))` to `rel(untransform(m, q), x)` for every
        // relation and metric `rel`, which is what makes the transform a
        // per-instance cost instead of a per-node one. Topological relations
        // are unchanged by moving both operands by any bijection, so that half
        // needs nothing from the program. The metrics do: writing an
        // `untransform` for a query type asserts that its distances to an
        // extent are unchanged by moving both -- which for a ray means the
        // direction is *not* renormalised, so that the parametric `t` of a hit
        // is the same on both sides of the frame change. That is the fact
        // pbrt's `tMax` argument rests on, and it is the whole reason an
        // instance can prune against a hit found in another one.
        untransform,

        opcount, // sentinel, do not remove!
    };

    OpType op;
    Expr a, b;

    static Expr make(OpType op, Expr a, Expr b);

    static const char *intrinsic_name(const OpType &op);

    static const IRExprEnum node_type = IRExprEnum::GeomOp;
};

// The `Hit { t : f32; b : vec2f }` record `intersection` answers (t = inf on
// a miss): the one struct the op's convention fixes, which a program's
// implementations must agree with field for field (Lower/Geometrics.cpp
// checks them against this).
Type hit_record_type();

// A query context flattened to the hardware's 32-bit payload words: every
// leaf field, depth first in declaration order, one word each. This is the
// single definition of the promoted-context layout -- Lower/Trees.cpp
// assigns the programs' rt_payload indices from it and CodeGen_OptiX
// scatters the trace's operand and gathers its returns by it, so the two
// cannot disagree. Answers nothing when the type is outside the class the
// registers can carry: a leaf that is not a 32-bit scalar (f32, (u)i32 or
// narrower, bool), or more than 32 words in all.
struct ContextWord {
    std::vector<std::string> path; // field names from the context down
    std::vector<unsigned> indices; // the same, as positions
    Type type;                     // the leaf's own scalar type
};
std::optional<std::vector<ContextWord>> rt_context_words(const Type &context);

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
        // The union, over a set, of a set reached from each of its elements:
        //
        //     flatten(T -> Set<S>, Set<T>) : Set<(T, S)>
        //
        // The pair is kept rather than dropping the outer element, because the
        // element is usually what makes the inner one meaningful -- an
        // instance's transform is what puts its triangles in the world.
        //
        // This is the single-index join of the Bonsai paper's Section 7.1 with
        // the inner index a function of the outer element rather than a set
        // variable of its own, plus the flattening step that section says a
        // caller wanting pairs has to add. It is what a two-level acceleration
        // structure is: every triangle of every instance, each seen through
        // its instance.
        flatten,
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

// A reference to `place`, an element of the leaves of the tree named `tree`:
// which element it is, with the representation left to the layout lowering
// (see ElementRef_t). Reading through one is a Deref, as through a pointer;
// the difference from PtrTo is only what the reference is made of, and that
// is decided later.
struct RefTo : ExprNode<RefTo> {
    Expr place;
    std::string tree;

    static Expr make(Expr place, std::string tree);

    static const IRExprEnum node_type = IRExprEnum::RefTo;
};

// The `index`-th element of the storage a tree's leaves are ranges of: the
// element itself, by value. A traversal never needs this -- it reaches an
// element through the leaf that holds it -- but a query the ray tracing
// hardware answers (see Lower/Trees.cpp, RTCore) is told which element it
// hit as an index into that storage, and this is how the program names it
// before the layout has said where the storage is. Lower/Layouts.cpp spells
// it as the read it is (`prims[i]`, `geoms[i]`) for a layout whose leaves
// hold ranges of one array, which is the layout the hardware's acceleration
// structure is built over. `tree` is the tree, as the program names it: a
// variable, or an element's field holding a tree.
struct StoredElement : ExprNode<StoredElement> {
    Expr tree;
    Expr index;

    // The element's type is the tree's element type; given outright when
    // the tree has already been retyped to its stored form (a field holding
    // a tree becomes the index of the tree's root, Lower/Layouts.cpp).
    static Expr make(Expr tree, Expr index);
    static Expr make(Expr tree, Expr index, Type element_type);

    static const IRExprEnum node_type = IRExprEnum::StoredElement;
};

// An array of elements stored in tiles: `width` elements per tile, each tile a
// struct holding every field of the element as a vector over the tile's
// lanes, so that a tile of four triangles is `v0.x[4], v0.y[4], v0.z[4],
// e1.x[4], ...` -- Embree's Triangle4, and the array-of-structs-of-arrays
// (AoSoA) of Scion's nested groups (Gyurgyik, Root and Kjolstad, "Decoupling
// Data Layouts from Bounding Volume Hierarchies", PLDI 2026, Fig. 7).
//
// `tiles` is the storage, an array of the tile structs; this expression is
// the same bytes read as the array of elements they hold, typed so: element
// `k` is lane `k % width` of tile `k / width`, which Lower/TiledArrays.cpp
// spells out for every `Extract` of one. Nothing but an Extract may use it:
// the whole array has no representation as elements, only as tiles.
//
// A tile need not store the element's fields as they are: it stores what the
// storage holds and derives the rest -- FCPW's packet stores a triangle's
// three vertices where the program's triangle carries a vertex and two edges,
// so the layout stores the vertices and derives the edges (`e1 = pa - pb`).
// `derived` holds those fields, each a value over the tile's stored fields
// named as variables of the element's types, which the element read
// substitutes the lane of each stored field into.
//
// Nor need every lane of a tile be an element: Embree pads the last block
// of a leaf with slots whose ids are -1, and which lanes hold elements is
// the storage's convention, so the layout says it -- `group[4] { ... }
// where geomID != 4294967295u` -- and `valid` holds that predicate, over
// the tile's members as `derived`'s values are. A loop over the elements
// leaves the lanes it fails out (Lower/ForEachs.cpp), as it leaves out
// the lanes past a leaf's count.
struct TiledArray : ExprNode<TiledArray> {
    Expr tiles;
    uint32_t width = 0;
    std::vector<std::pair<std::string, Expr>> derived;
    Expr valid;

    // `element` is the element type the tiles hold; `tiles` has the type of
    // the storage, `array[<tile struct>, n]`; `derived` the element's fields
    // the tiles do not store, by name, with their values, which read the
    // tile's fields as variables named by `placeholder`; `valid`, when
    // defined, the predicate over the same variables that says which lanes
    // are elements.
    static Expr make(Expr tiles, uint32_t width, Type element,
                     std::vector<std::pair<std::string, Expr>> derived = {},
                     Expr valid = Expr());

    // The variable a derived value reads the tile's field `field` through: a
    // name no program can write, so that no pass between the layout's
    // lowering and the read's expansion takes it for a variable of its own.
    static std::string placeholder(const std::string &field);

    static const IRExprEnum node_type = IRExprEnum::TiledArray;
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
    // Whether the elements are storage nothing in the program writes: an
    // extern's, which the host filled before the program ran and holds
    // still while it runs (an extern is never `mut`; see Type::is_readonly).
    // Printed `const T[n]`. A reference into such storage -- a row's
    // address, a group reinterpreted as its rows -- carries the bit on, so
    // that a read through it may be moved past any store and told to LLVM
    // as invariant (SSA/Simplify.cpp, CodeGen/CodeGen_LLVM.cpp).
    bool readonly = false;

    static Type make(Type etype, Expr size, bool readonly = false);

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
