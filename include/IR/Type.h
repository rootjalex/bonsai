#pragma once

#include "Error.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "Interface.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <map>
#include <optional>
#include <string>
#include <variant>

namespace bonsai {
namespace ir {

struct Type;

enum class IRTypeEnum {
    Void_t,
    Int_t,
    UInt_t,
    Index_t,
    Float_t,
    Bool_t,
    String_t,
    Ptr_t,
    Ref_t,
    ElementRef_t,
    Vector_t,
    Struct_t,
    Tuple_t,
    Array_t,
    DynArray_t,
    Option_t,
    ADT_t,
    Set_t,
    Function_t,
    Generic_t,
    BVH_t,
    Rand_State_t,
};

using IRTypeNode = IRNode<Type, IRTypeEnum>;

/* This is necessary to get mutate() to work properly... */
struct BaseTypeNode : public IRTypeNode {
    BaseTypeNode(IRTypeEnum t) : IRTypeNode(t) {}
    virtual Type mutate_type(Mutator *m) const = 0;
};

template <typename T>
struct TypeNode : public BaseTypeNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Type mutate_type(Mutator *m) const override;
    TypeNode() : BaseTypeNode(T::node_type) {}
    ~TypeNode() override = default;
};

struct Type : public IRHandle<IRTypeNode> {
    /** Make an undefined type */
    Type() = default;

    /** Make a type from a concrete type node pointer (e.g. Int_t) */
    Type(const IRTypeNode *n) : IRHandle<IRTypeNode>(n) {}

    /** Override get() to return a BaseTypeNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseTypeNode *get() const { return (const BaseTypeNode *)ptr; }

    uint32_t bits() const;
    uint32_t bytes() const;
    uint32_t lanes() const;
    bool is_int() const;
    bool is_uint() const;
    bool is_int_or_uint() const;
    bool is_int_tuple() const;
    bool is_float() const;
    bool is_bool() const;
    bool is_scalar() const;
    bool is_vector() const;
    bool is_numeric() const;
    bool is_primitive() const;         // basically: is LLVM-representable?
    bool is_stack_allocatable() const; // primitives but not arrays
    bool is_iterable() const;
    bool is_func() const;
    // Does a value of this type already refer to storage the program can
    // write through? An array handle does: it is the address of its
    // elements, so making one mutable adds no indirection, and wrapping it
    // in a Ptr_t would mean a pointer to a pointer.
    bool is_reference() const;

    // Type casts
    // Rewrites (through vectors) to boolean base.
    Type to_bool() const;
    // Rewrites (through vectors) to uint base.
    Type to_uint() const;
    // returns (Vector_t | Set_t)'s etype
    Type element_of() const;
    // Changes the element type to etype
    Type with_etype(Type etype) const;

    // TODO: implement copy/move semantics!
};

struct TypedVar {
    std::string name;
    Type type; // optional

    TypedVar(std::string name, Type type)
        : name(std::move(name)), type(std::move(type)) {}
    TypedVar() {}

    operator Expr() const;
};

template <typename T>
Type TypeNode<T>::mutate_type(Mutator *m) const {
    return m->visit((const T *)this);
}

struct Void_t : TypeNode<Void_t> {
    static Type make();

    static const IRTypeEnum node_type = IRTypeEnum::Void_t;
};

struct Int_t : TypeNode<Int_t> {
    uint32_t bits;

    static Type make(uint32_t bits);

    static const IRTypeEnum node_type = IRTypeEnum::Int_t;
};

struct UInt_t : TypeNode<UInt_t> {
    uint32_t bits;

    static Type make(uint32_t bits);

    static const IRTypeEnum node_type = IRTypeEnum::UInt_t;
};

struct Index_t : TypeNode<Index_t> {
    static Type make();
    static const IRTypeEnum node_type = IRTypeEnum::Index_t;
};

// A subset of the real numbers. This typically consists of a sign bit, mantissa
// (or significand) bits, and exponent bits.
struct Float_t : TypeNode<Float_t> {
    // Determines the precision of the number.
    uint32_t exponent;
    // Determines the magnitude of the number.
    uint32_t mantissa;

    static Type make(uint32_t exponent, uint32_t mantissa);

    // Creates an f64 (double-precision) type under IEEE754 standard.
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    static Type make_f64();

    // Creates an f32 (single-precision) type under IEEE754 standard.
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    static Type make_f32();

    // Creates an f16 (half-precision) type under IEEE754 standard.
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    static Type make_f16();

    // Creates a bf16 (half-precision) type.
    // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
    static Type make_bf16();

    // Returns the total number of bits:
    // sign bit + exponent bits + mantissa bits
    uint32_t bits() const;

    // Returns whether this floating point type conforms to IEEE-754 standard.
    bool is_ieee754() const;

    // Returns whether this floating point type is brain float (bf16).
    bool is_bfloat16() const;

    static const IRTypeEnum node_type = IRTypeEnum::Float_t;
};

struct Bool_t : TypeNode<Bool_t> {
    static Type make();

    static const IRTypeEnum node_type = IRTypeEnum::Bool_t;
};

struct String_t : TypeNode<String_t> {
    static Type make();

    static const IRTypeEnum node_type = IRTypeEnum::String_t;
};

struct Ptr_t : TypeNode<Ptr_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum node_type = IRTypeEnum::Ptr_t;
};

struct Ref_t : TypeNode<Ref_t> {
    std::string name;

    static Type make(std::string name);

    static const IRTypeEnum node_type = IRTypeEnum::Ref_t;
};

// A reference to an element of a tree's leaves -- which element the best hit
// of an `argmin` was, say -- whose representation is the layout's to decide.
// The query lowering says only that it remembers *which* element (see
// RefTo); the layout lowering knows where the tree's elements are kept and
// lowers this to the minimal bits that pick one out, an index into that
// container, so that the type never becomes an address. An address is a
// layout the program did not write: sixty-four bits bound to one address
// space, that cannot go to another device or be serialized, and that a SIMD
// gang would have to carry as a vector of pointers. See
// Lower/ElementReferences.cpp.
struct ElementRef_t : TypeNode<ElementRef_t> {
    // What is referred to.
    Type etype;
    // The tree whose leaves hold it, by the name the schedule gave the tree.
    std::string tree;

    static Type make(Type etype, std::string tree);

    static const IRTypeEnum node_type = IRTypeEnum::ElementRef_t;
};

struct Vector_t : TypeNode<Vector_t> {
    Type etype;
    uint32_t lanes;
    // Storage rather than a value. A vector the program computes with is
    // whatever the machine's vector register is -- three floats take sixteen
    // bytes, and are aligned to sixteen -- while a vector a layout stores is
    // exactly `lanes` elements, back to back, aligned as one element is: the
    // twelve bytes a layout that says `vec3f` means. The two are different
    // types with the same lanes; a layout's fields are the packed kind, what
    // is read out of one is converted to the other (Cast, mode Convert), and
    // nothing computes with the packed kind directly.
    bool packed = false;

    static Type make(Type etype, uint32_t lanes, bool packed = false);

    static const IRTypeEnum node_type = IRTypeEnum::Vector_t;
};

struct Struct_t : TypeNode<Struct_t> {
    // intentionally ordered.
    // TODO: re-implement an unordered version (for the front-end):
    // UnorderedStruct_t
    using Map = std::vector<TypedVar>;
    using DefMap = std::map<std::string, Expr>;
    std::string name;
    Map fields;
    DefMap defaults;

    enum class Attribute {
        packed, // Whether this struct is 1-byte aligned.
        // Describes the bytes of a laid-out tree rather than a value the
        // program wrote. A node's storage is read as one variant or another
        // depending on a tag in it, which is the only place this IR reads the
        // same bytes at two types -- so it is the only place that has to be
        // kept out of type-based alias analysis (see CodeGen_LLVM's tbaa).
        layout,
    };

    std::vector<Attribute> attributes;

    static Type make(std::string name, Map fields,
                     std::vector<Attribute> attributes = {});
    static Type make(std::string name, Map fields, DefMap defaults,
                     std::vector<Attribute> attributes = {});

    // Whether this type has the `packed` attribute.
    bool is_packed() const;

    // Whether this type describes the bytes of a laid-out tree.
    bool is_layout() const;

    static const IRTypeEnum node_type = IRTypeEnum::Struct_t;
};

struct Tuple_t : TypeNode<Tuple_t> {
    std::vector<Type> etypes;

    static Type make(std::vector<Type> etypes);

    static const IRTypeEnum node_type = IRTypeEnum::Tuple_t;
};

struct Option_t : TypeNode<Option_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum node_type = IRTypeEnum::Option_t;
};

// A closed set of named variants, one of which a value is at any time.
//
// The general case of Option_t, which is an ADT with a variant carrying a
// value and one carrying nothing. Like Option_t this never reaches a backend:
// Lower/ADTs.cpp rewrites it into a tag and a payload the variants share, and
// Lower/VerifyADTs.cpp checks that none survive.
//
// Each variant is a Struct_t of its fields, so a variant with no fields is a
// struct with none. Variant names are unique across the program, the same rule
// tree nodes follow, so a value can be built by naming the variant alone.
struct ADT_t : TypeNode<ADT_t> {
    using Variants = std::vector<Type>; // each a Struct_t

    std::string name;
    Variants variants;

    static Type make(std::string name, Variants variants);

    // The variant of this name, or nothing if it has none.
    std::optional<size_t> index_of(const std::string &variant) const;

    // The fields of a variant, by index.
    const Struct_t::Map &fields(size_t index) const;
    const std::string &variant_name(size_t index) const;

    static const IRTypeEnum node_type = IRTypeEnum::ADT_t;
};

// Storage read at one of several types.
//
// Every member begins at the same address; the size is the largest of them and
// the alignment the strictest, which is the rule Rust and C both use. That is
// not a struct with a flag on it: a struct's fields are at distinct offsets,
// and everything that walks them assumes so, which is why this is a type of
// its own rather than an attribute -- the compiler then has to be told what to
// do with it everywhere, instead of quietly doing the wrong thing.
//
// Only the backend can work out the size and alignment, so this reaches code
// generation rather than being flattened before it. See Type::bytes(), which
// says as much.
// The name of component `k` of a value widened to one gang-wide vector per
// component, and of lane `k` of a union widened to one union per lane (see
// widen).
std::string component_field(uint32_t k);

// The form of a value of `type` that a gang of `lanes` lanes holds one of per
// lane: what SSA/Vectorize.cpp turns a varying value's type into.
//
// A scalar becomes a vector as wide as the gang. A struct is widened field by
// field, into a struct of the same shape whose members are the widened field
// types -- ISPC's varying-struct representation, `struct { x: f32, y: f32 }`
// becoming `struct { x: f32x8, y: f32x8 }` rather than a vector of structs,
// which no backend can spell (a vector's elements must be primitive, but a
// struct's members may be vectors). The name gains a suffix so the widened
// struct is a type distinct from the scalar one it came from.
//
// A lane's own short vector -- a `vec3f` -- is widened the same way, as a
// struct with one gang-wide vector per component: `f32x3` becomes `f32x3$v8 =
// { !0: f32x8, !1: f32x8, !2: f32x8 }`. That is ispc's `varying float<3>`,
// laid out component-major so that a component across the gang is one
// contiguous vector, and it is the only shape available: a vector of vectors
// is not a type either backend has. Arithmetic on such a value never reaches
// this form -- SSA/SplitAggregates.h takes a per-lane vector apart into its
// components before widening, so each becomes an ordinary gang vector -- but a
// vector that is a field of a struct, that lives in per-lane memory, or that
// crosses a block boundary or a return is carried whole, and this is what it
// is carried as.
//
// A union is widened word by word: its storage is a row of 32-bit words, and
// each word is widened across the gang, so `Blob_payload$v8 = { !w0: u32x8,
// .., !w4: u32x8 }` for a twenty-byte union. Every member's fields lie at the
// same offsets in every lane, so a field of any member is a word of this (or
// a part of one, for a `bool` or a `u16`, or two, for an `i64`), whichever
// member a lane holds: reading a member for all lanes is a reinterpretation of
// its words, with a shift for a field narrower than a word, and building one
// from a member value per lane is the reverse. Nothing is transposed, and a
// blend of two unions at a join is a select per word. The footprint is still
// the lane count times the largest member, rounded up to a word, which is the
// reason for not widening a union as one gang-wide slot per member instead: that
// would cost the sum of the members' sizes per gang. The lanes holding some
// other member read as garbage in a member's fields, which the masks and blends
// downstream discard, as they discard every result of a lane that is off.
// How many words a union is comes from layout_bytes below.
Type widen(const Type &type, uint32_t lanes);

// What a write through a value of `type` writes: the pointee of a pointer;
// for one pointer per lane -- a vector of pointers, an address a gang carries
// as a value, such as a queue entry's slot of the per-path record
// (SSA/Defer.cpp) -- the gang's value of the pointee (widen), since each lane
// writes one lane's worth of it at its own address, which is what the
// backends' scatter takes. The same rule WriteLoc::add_index_access applies
// to an index per lane. Undefined for anything else.
Type pointee_of(const Type &type);

// The inverse of widen: the type a lane holds one of, from its gang-wide
// form. Read off the shape widen() makes -- a vector as wide as the gang, a
// struct named with the `$v<lanes>` suffix, a struct of one gang vector per
// component (named for the short vector, with its packedness), the words of a
// union (recorded by widen, since the words alone do not say) -- so it is
// defined only on those.
Type narrow(const Type &type, uint32_t lanes);

// The bytes a value of `type` occupies in memory, and its alignment there:
// the target's layout, stated here for the types that make up an ADT's
// payload -- a scalar its width, a bool a byte, a vector its power-of-two
// storage unless packed, a struct C's layout, a union its largest member
// rounded to the widest alignment. The IR has no target to ask, and the width
// of a widened union has to be a type before any target is in sight; so this
// duplicates a rule the targets own, as Type::bytes does for vectors, and
// CodeGen_LLVM checks it against its data layout wherever it lowers a widened
// union.
uint64_t layout_bytes(const Type &type);
uint64_t layout_align(const Type &type);
// Where field `index` of `s` starts, by the same rule.
uint64_t layout_offset(const Struct_t &s, size_t index);

// The lane count of a gang-wide aggregate -- a struct widen() built, which
// its name says -- or nothing for any other type. A vector alone does not
// tell: a lane's own short vector can be as wide as a gang.
std::optional<uint32_t> widened_lanes(const Type &type);

struct Set_t : TypeNode<Set_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum node_type = IRTypeEnum::Set_t;
};

struct Function_t : TypeNode<Function_t> {
    Type ret_type;
    struct ArgSig {
        Type type;
        bool is_mutable;
    };
    std::vector<ArgSig> arg_types;

    static Type make(Type ret_type, std::vector<ArgSig> arg_types);

    static const IRTypeEnum node_type = IRTypeEnum::Function_t;
};

struct Generic_t : TypeNode<Generic_t> {
    std::string name;
    Interface interface;

    static Type make(std::string name, Interface interface);

    static const IRTypeEnum node_type = IRTypeEnum::Generic_t;
};

struct Annotation {

    // data = name
    struct Data {
        std::string name;
    };

    // GEOM [on name]?
    struct Volume {
        std::string geometry; // possibly empty.
        Type struct_type;
        std::vector<std::string> initializers;
        bool broadcast; // if on children
    };

    // An element's `with extent = <expr>` is the fifth augmentation and the
    // only one not declared here: it is an arbitrary expression rather than a
    // reference to stored fields, and `Expr` is not a complete type at this
    // point. It lives in `Program::extents`, keyed by the element's name.

    // scalar in [low, high]
    struct Interval {
        std::string scalar;
        // TODO: support compute?
        // Expr low, high;
        std::string low, high;
    };

    // count() = dCount
    // min(field) = field_min
    // etc.
    struct Aggregate {
        // TODO: deduplicate with AggOp::OpType...
        enum OpType {
            avg,
            count,
            max,
            min,
            prod,
            sum,
        };
        OpType op;
        std::vector<std::string> args; // field, empty for count
        std::string value;             // field name that stores

        static OpType str_to_op(const std::string &str);
    };

    std::variant<Data, Volume, Interval, Aggregate> type;

    template <typename T>
    const T *as() const {
        return std::get_if<T>(&type);
    }
};

// An ADT with Volume information, representing a bounding volume hierarchy.
struct BVH_t : TypeNode<BVH_t> {
    // A Node is a Struct_t of typed fields with some annotations.
    struct Node {
        Type struct_type;
        std::vector<Annotation> annotations;

        // Useful helper functions.
        const std::string &name() const {
            return struct_type.as<Struct_t>()->name;
        }
        const Struct_t::Map &fields() const {
            return struct_type.as<Struct_t>()->fields;
        }

        bool has_volume() const {
            for (const auto &annot : annotations) {
                if (annot.as<Annotation::Volume>() &&
                    annot.as<Annotation::Volume>()->geometry.empty()) {
                    return true;
                }
            }
            return false;
        }

        // Whether this arm is where the elements are, which is what makes it
        // the place a bound over the node's volume is worth testing: it is the
        // same for every element the arm holds, and there is nothing below it
        // to test it at instead.
        bool has_data() const {
            for (const auto &annot : annotations) {
                if (annot.as<Annotation::Data>()) {
                    return true;
                }
            }
            return false;
        }

        const Annotation::Volume *get_volume() const {
            for (const auto &annot : annotations) {
                if (const auto *vol = annot.as<Annotation::Volume>();
                    vol && vol->geometry.empty()) {
                    return vol;
                }
            }
            internal_error << "get_volume called on no-volume node!\n";
            return nullptr;
        }
    };

    ir::Type primitive;
    std::string name;
    // TODO: do we ever want a root Volume or root Params?
    // Params every Node has.
    // std::vector<Param> params;
    // All possible node types.
    std::vector<Node> nodes;
    // BV for every node, unless specified in the Node type.
    // std::optional<Volume> volume;

    // Each node should have a volume set, or are un-optimized.
    static Type make(ir::Type primitive, std::string name,
                     std::vector<Node> nodes);
    // All nodes share the same annotations + any specified annotations.
    static Type make(ir::Type primitive, std::string name,
                     const std::vector<TypedVar> &globals,
                     std::vector<Node> nodes,
                     std::vector<Annotation> annotations);

    static const IRTypeEnum node_type = IRTypeEnum::BVH_t;
};

// Device-specific random (mutable!) state.
struct Rand_State_t : TypeNode<Rand_State_t> {
    static Type make();
    static const IRTypeEnum node_type = IRTypeEnum::Rand_State_t;
};

// TODO: List_t, Tensor_t

// Useful helper function
Type get_field_type(const Type &struct_type, const std::string &field);

bool satisfies(const Type &type, const Interface &interface);

/** The name a geometric primitive mangles into for an operand of this type,
 * so that `distmin(r, s)` picks `distmin_Ray_Sphere`.
 *
 * Elements are the only things geometric primitives dispatch on: a struct, or
 * a variant type, which is an element whose shape is chosen per value.
 * Anything else answers the empty string, which each caller reports in the way
 * that suits it -- the parser knows a source location, lowering does not.
 *
 * One copy, because the mangling has to agree in three places: where the
 * primitive is declared, where a use of it is type-checked, and where that use
 * is turned back into a call. */
std::string geometric_element_name(const Type &type);

using TypeMap = std::map<std::string, Type>;

} // namespace ir

template <>
inline RefCount &ref_count<ir::IRTypeNode>(const ir::IRTypeNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRTypeNode>(const ir::IRTypeNode *t) {
    delete t;
}

} // namespace bonsai
