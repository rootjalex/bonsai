#pragma once

#include "IRHandle.h"
#include "IRNode.h"
#include "Interface.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <map>
#include <optional>
#include <string>

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
    Ptr_t,
    Ref_t,
    Vector_t,
    Struct_t,
    Tuple_t,
    Array_t,
    Option_t,
    Set_t,
    Function_t,
    Generic_t,
    BVH_t,
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
    uint32_t bytes() const {
        // TODO booleans better! we need to bit-pack, unlike Halide.
        return (bits() + 7) / 8;
    }
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
    bool is_primitive() const; // basically: is LLVM-representable?
    bool is_iterable() const;
    bool is_func() const;

    // Type casts
    // Rewrites (through vectors) to boolean base.
    Type to_bool() const;
    // Rewrites (through vectors) to uint base.
    Type to_uint() const;
    // returns (Vector_t | Set_t)'s etype
    Type element_of() const;

    // TODO: implement copy/move semantics!
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

struct Vector_t : TypeNode<Vector_t> {
    Type etype;
    uint32_t lanes;

    static Type make(Type etype, uint32_t lanes);

    static const IRTypeEnum node_type = IRTypeEnum::Vector_t;
};

struct Struct_t : TypeNode<Struct_t> {
    // intentionally ordered.
    // TODO: re-implement an unordered version (for the front-end):
    // UnorderedStruct_t
    using Field = std::pair<std::string, Type>;
    using Map = std::vector<Field>;
    using DefMap = std::map<std::string, Expr>;
    std::string name;
    Map fields;
    DefMap defaults;

    static Type make(std::string name, Map fields);
    static Type make(std::string name, Map fields, DefMap defaults);

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

struct Set_t : TypeNode<Set_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum node_type = IRTypeEnum::Set_t;
};

struct Function_t : TypeNode<Function_t> {
    Type ret_type;
    std::vector<Type> arg_types;

    static Type make(Type ret_type, std::vector<Type> arg_types);

    static const IRTypeEnum node_type = IRTypeEnum::Function_t;
};

struct Generic_t : TypeNode<Generic_t> {
    std::string name;
    Interface interface;

    static Type make(std::string name, Interface interface);

    static const IRTypeEnum node_type = IRTypeEnum::Generic_t;
};

// An ADT with Volume information, representing a bounding volume hierarchy.
struct BVH_t : TypeNode<BVH_t> {
    // A type that should be treated as a bounding volume,
    // initialized with Params.
    struct Volume {
        Type struct_type;
        std::vector<std::string> initializers;
    };
    // A Node is a Struct_t of typed fields with an optional bounding volume.
    struct Node {
        Type struct_type;
        std::optional<Volume> volume;

        // Useful helper functions.
        const std::string &name() const {
            return struct_type.as<Struct_t>()->name;
        }
        const Struct_t::Map &fields() const {
            return struct_type.as<Struct_t>()->fields;
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
    // All nodes share the same volume type unless otherwise specified.
    static Type make(ir::Type primitive, std::string name,
                     const std::vector<Struct_t::Field> &globals,
                     std::vector<Node> nodes, Volume volume);

    static const IRTypeEnum node_type = IRTypeEnum::BVH_t;
};

// TODO: List_t, Tensor_t

// Useful helper function
Type get_field_type(const Type &struct_type, const std::string &field);

bool satisfies(const Type &type, const Interface &interface);

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
