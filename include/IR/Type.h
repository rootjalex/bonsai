#pragma once

#include "IRHandle.h"
#include "IRNode.h"
#include "Interface.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <map>
#include <string>

namespace bonsai {
namespace ir {

struct Type;

enum class IRTypeEnum {
    Void_t,
    Int_t,
    UInt_t,
    Float_t,
    Bool_t,
    Ptr_t,
    Vector_t,
    Struct_t,
    Tuple_t,
    Option_t,
    Set_t,
    Function_t,
    Generic_t,
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
    TypeNode() : BaseTypeNode(T::_node_type) {}
    ~TypeNode() override = default;
};

struct Type : public IRHandle<IRTypeNode> {
    /** Make an undefined type */
    Type() = default;

    /** Make a type from a concrete type node pointer (e.g. Int_t) */
    Type(const IRTypeNode *n) : IRHandle<IRTypeNode>(n) {}

    /** Override get() to return a BaseExprNode * instead of an IRNode.
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
    bool is_float() const;
    bool is_bool() const;
    bool is_scalar() const;
    bool is_vector() const;
    bool is_numeric() const;

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

    static const IRTypeEnum _node_type = IRTypeEnum::Void_t;
};

struct Int_t : TypeNode<Int_t> {
    uint32_t bits;

    static Type make(uint32_t bits);

    static const IRTypeEnum _node_type = IRTypeEnum::Int_t;
};

struct UInt_t : TypeNode<UInt_t> {
    uint32_t bits;

    static Type make(uint32_t bits);

    static const IRTypeEnum _node_type = IRTypeEnum::UInt_t;
};

struct Float_t : TypeNode<Float_t> {
    uint32_t bits;

    static Type make(uint32_t bits);

    static const IRTypeEnum _node_type = IRTypeEnum::Float_t;
};

struct Bool_t : TypeNode<Bool_t> {
    static Type make();

    static const IRTypeEnum _node_type = IRTypeEnum::Bool_t;
};

struct Ptr_t : TypeNode<Ptr_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum _node_type = IRTypeEnum::Ptr_t;
};

struct Vector_t : TypeNode<Vector_t> {
    Type etype;
    uint32_t lanes;

    static Type make(Type etype, uint32_t lanes);

    static const IRTypeEnum _node_type = IRTypeEnum::Vector_t;
};

struct Struct_t : TypeNode<Struct_t> {
    // intentionally ordered.
    // TODO: re-implement an unordered version (for the front-end):
    // UnorderedStruct_t
    using Map = std::vector<std::pair<std::string, Type>>;
    using DefMap = std::map<std::string, Expr>;
    std::string name;
    Map fields;
    DefMap defaults;

    static Type make(std::string name, Map fields);
    static Type make(std::string name, Map fields, DefMap defaults);

    static const IRTypeEnum _node_type = IRTypeEnum::Struct_t;
};

struct Tuple_t : TypeNode<Tuple_t> {
    std::vector<Type> etypes;

    static Type make(std::vector<Type> etypes);

    static const IRTypeEnum _node_type = IRTypeEnum::Tuple_t;
};

struct Option_t : TypeNode<Option_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum _node_type = IRTypeEnum::Option_t;
};

struct Set_t : TypeNode<Set_t> {
    Type etype;

    static Type make(Type etype);

    static const IRTypeEnum _node_type = IRTypeEnum::Set_t;
};

struct Function_t : TypeNode<Function_t> {
    Type ret_type;
    std::vector<Type> arg_types;

    static Type make(Type ret_type, std::vector<Type> arg_types);

    static const IRTypeEnum _node_type = IRTypeEnum::Function_t;
};

struct Generic_t : TypeNode<Generic_t> {
    std::string name;
    Interface interface;

    static Type make(std::string name, Interface interface);

    static const IRTypeEnum _node_type = IRTypeEnum::Generic_t;
};

// TODO: List_t, Tensor_t

// Useful helper function
Type get_field_type(const Type &struct_type, const std::string &field);

bool satisfies(const Type &type, const Interface &interface);

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
