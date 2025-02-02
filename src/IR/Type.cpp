#include "IR/Type.h"

#include <stdexcept>
#include <utility>

#include "IR/Printer.h"

namespace bonsai {
namespace ir {

uint32_t Type::bits() const {
    if (auto as_int = this->as<Int_t>()) {
        return as_int->bits;
    } else if (auto as_uint = this->as<UInt_t>()) {
        return as_uint->bits;
    } else if (auto as_float = this->as<Float_t>()) {
        return as_float->bits;
    } else {
        internal_error << "Called bits() on bad type: " << *this;
        return 0;
    }
}

uint32_t Type::lanes() const {
    if (auto as_vec = this->as<Vector_t>()) {
        // TODO: handle recursive vectors?
        return as_vec->lanes;
    } else {
        internal_error << "Called lanes() on bad type: " << *this;
        return 0;
    }
}

bool Type::is_int() const {
    return this->is<Int_t>() ||
           (this->is<Vector_t>() && this->as<Vector_t>()->etype.is_int());
}

bool Type::is_uint() const {
    return this->is<UInt_t>() ||
           (this->is<Vector_t>() && this->as<Vector_t>()->etype.is_uint());
}

bool Type::is_int_or_uint() const {
    return this->is<Int_t>() || this->is<UInt_t>() ||
           (this->is<Vector_t>() &&
            this->as<Vector_t>()->etype.is_int_or_uint());
}

bool Type::is_float() const {
    return this->is<Float_t>() ||
           (this->is<Vector_t>() && this->as<Vector_t>()->etype.is_float());
}

bool Type::is_bool() const {
    return this->is<Bool_t>() ||
           (this->is<Vector_t>() && this->as<Vector_t>()->etype.is_bool());
}

bool Type::is_scalar() const {
    // TODO: what counts as scalar?
    return this->is<Int_t>() || this->is<UInt_t>() || this->is<Float_t>() ||
           this->is<Bool_t>();
}

bool Type::is_vector() const {
    // TODO: what counts as vector?
    return this->is<Vector_t>();
}

bool Type::is_numeric() const {
    // scalar + vector of numbers
    // TODO: let Struct_ts overload their numeric operators.
    return this->is_int_or_uint() || this->is_float();
}

Type Type::to_bool() const {
    if (this->is_bool()) {
        return *this;
    } else if (this->is<Int_t>() || this->is<Float_t>() || this->is<UInt_t>()) {
        return Bool_t::make();
    } else if (this->is<Vector_t>()) {
        const Vector_t *v = this->as<Vector_t>();
        return Vector_t::make(v->etype.to_bool(), v->lanes);
    } else {
        internal_error << "Called to_bool() on bad type: " << *this;
        return Type();
    }
}

Type Type::to_uint() const {
    if (this->is<Int_t>()) {
        return UInt_t::make(this->as<Int_t>()->bits);
    } else if (this->is<Float_t>()) {
        return UInt_t::make(this->as<Float_t>()->bits);
    } else if (this->is<Vector_t>()) {
        const Vector_t *v = this->as<Vector_t>();
        return Vector_t::make(v->etype.to_uint(), v->lanes);
    } else {
        internal_error << "Called to_uint() on bad type: " << *this;
        return Type();
    }
}

Type Type::element_of() const {
    if (this->is<Vector_t>()) {
        return this->as<Vector_t>()->etype;
    } else if (this->is<Set_t>()) {
        return this->as<Set_t>()->etype;
    } else {
        internal_error << "Called element_of() on bad type: " << *this;
        return Type();
    }
}

Type Void_t::make() {
    static Type global_void = new Void_t;
    return global_void;
}

Type Int_t::make(uint32_t bits) {
    internal_assert(bits > 0 && bits <= 64)
        << "Unsupported bitwidth in Int_t: " << bits;
    Int_t *node = new Int_t;
    node->bits = bits;
    return node;
}

Type UInt_t::make(uint32_t bits) {
    internal_assert(bits > 0 && bits <= 64)
        << "Unsupported bitwidth in UInt_t: " << bits;
    UInt_t *node = new UInt_t;
    node->bits = bits;
    return node;
}

Type Float_t::make(uint32_t bits) {
    internal_assert(bits == 16 || bits == 32 || bits == 64)
        << "Unsupported bitwidth in Float_t: " << bits;
    Float_t *node = new Float_t;
    node->bits = bits;
    return node;
}

Type Bool_t::make() {
    static Type global_bool = new Bool_t;
    return global_bool;
}

Type Ptr_t::make(Type etype) {
    internal_assert(etype.defined()) << "Ptr_t::make received undefined etype";
    Ptr_t *node = new Ptr_t;
    node->etype = std::move(etype);
    return node;
}

Type Vector_t::make(Type etype, uint32_t lanes) {
    internal_assert(etype.defined())
        << "Vector_t::make received undefined etype";
    Vector_t *node = new Vector_t;
    node->etype = std::move(etype);
    node->lanes = lanes;
    return node;
}

Type Struct_t::make(std::string name, Struct_t::Map fields) {
    internal_assert(!name.empty()) << "Struct_t::make recieved undefined name";
    internal_assert(
        std::all_of(fields.cbegin(), fields.cend(),
                    [](const auto &p) { return p.second.defined(); }))
        << "Struct_t::make recieved undefined field type in definition of "
        << name;
    Struct_t *node = new Struct_t;
    node->name = std::move(name);
    node->fields = std::move(fields);
    return node;
}

Type Struct_t::make(std::string name, Struct_t::Map fields,
                    Struct_t::DefMap defaults) {
    internal_assert(!name.empty()) << "Struct_t::make recieved undefined name";
    internal_assert(
        std::all_of(fields.cbegin(), fields.cend(),
                    [](const auto &p) { return p.second.defined(); }))
        << "Struct_t::make recieved undefined field type in definition of "
        << name;
    internal_assert(std::all_of(defaults.cbegin(), defaults.cend(),
                                [](const auto &p) {
                                    return p.second.defined() &&
                                           p.second.type().defined();
                                }))
        << "Struct_t::make recieved undefined default expression";
    Struct_t *node = new Struct_t;
    node->name = std::move(name);
    node->fields = std::move(fields);
    node->defaults = std::move(defaults);
    return node;
}

Type Tuple_t::make(std::vector<Type> etypes) {
    internal_assert(std::all_of(etypes.cbegin(), etypes.cend(),
                                [](const Type &t) { return t.defined(); }))
        << "Tuple_t::make recieved undefined type";
    Tuple_t *node = new Tuple_t;
    node->etypes = std::move(etypes);
    return node;
}

Type Option_t::make(Type etype) {
    internal_assert(etype.defined())
        << "Option_t::make received undefined etype";
    Option_t *node = new Option_t;
    node->etype = std::move(etype);
    return node;
}

Type Set_t::make(Type etype) {
    internal_assert(etype.defined()) << "Set_t::make received undefined etype";
    Set_t *node = new Set_t;
    node->etype = std::move(etype);
    return node;
}

Type Function_t::make(Type ret_type, std::vector<Type> arg_types) {
    internal_assert(ret_type.defined())
        << "Function_t::make received undefined ret_type";
    internal_assert(std::all_of(arg_types.cbegin(), arg_types.cend(),
                                [](const Type &t) { return t.defined(); }))
        << "Function_t::make received undefined arg_type";
    Function_t *node = new Function_t;
    node->ret_type = std::move(ret_type);
    node->arg_types = std::move(arg_types);
    return node;
}

Type Generic_t::make(std::string name, Interface interface) {
    internal_assert(!name.empty()) << "Generic_t::make received empty name";
    internal_assert(interface.defined())
        << "Generic_t::make received undefined interface for " << name;
    Generic_t *node = new Generic_t;
    node->name = std::move(name);
    node->interface = std::move(interface);
    return node;
}

Type get_field_type(const Type &struct_type, const std::string &field) {
    if (const Struct_t *as_struct = struct_type.as<Struct_t>()) {
        Type etype;
        for (const auto &[key, value] : as_struct->fields) {
            if (key == field) {
                return value;
            }
        }
        internal_error << "Failed to find field: " << field
                       << " in struct type: " << struct_type;
        return ir::Type();
    } else if (const Vector_t *as_vec = struct_type.as<Vector_t>()) {
        internal_assert((field == "x" && as_vec->lanes > 0) ||
                        (field == "y" && as_vec->lanes > 1) ||
                        (field == "z" && as_vec->lanes > 2) ||
                        (field == "w" && as_vec->lanes > 3))
            << "Vector access of bad field: " << field
            << " of type: " << struct_type;
        return as_vec->etype;
    } else {
        internal_error << "Failed to find field: " << field
                       << " in non-(struct | vec) type: " << struct_type;
        return ir::Type();
    }
}

bool satisfies(const Type &type, const Interface &interface) {
    switch (interface.node_type()) {
    case IRInterfaceEnum::IEmpty:
        return true;
    case IRInterfaceEnum::IFloat:
        return type.is_float();
    case IRInterfaceEnum::IVector: {
        const IVector *iv = interface.as<IVector>();
        return type.is<Vector_t>() &&
               (!iv->etype.defined() ||
                satisfies(type.as<Vector_t>()->etype, iv->etype));
    }
    }
}

} // namespace ir
} // namespace bonsai
