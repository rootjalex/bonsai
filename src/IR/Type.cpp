#include "IR/Type.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "Utils.h"

namespace bonsai {
namespace ir {

uint32_t Type::bits() const {
    if (auto *as_int = this->as<Int_t>()) {
        return as_int->bits;
    }
    if (auto *as_uint = this->as<UInt_t>()) {
        return as_uint->bits;
    }
    if (auto *as_float = this->as<Float_t>()) {
        return as_float->bits();
    }
    if (auto *as_bool = this->as<Bool_t>()) {
        return 1;
    }
    internal_error << "Called bits() on bad type: " << *this;
}

uint32_t Type::lanes() const {
    if (auto as_vec = this->as<Vector_t>()) {
        // TODO: handle recursive vectors?
        return as_vec->lanes;
    } else {
        internal_error << "Called lanes() on bad type: " << *this;
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
           (this->is<Vector_t>() && this->as<Vector_t>()->etype.is_float()) ||
           (this->is<Generic_t>() &&
            this->as<Generic_t>()->interface.is_numeric());
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
    }
}

Type Type::to_uint() const {
    if (this->is<Int_t>()) {
        return UInt_t::make(this->as<Int_t>()->bits);
    } else if (this->is<Float_t>()) {
        return UInt_t::make(this->as<Float_t>()->bits());
    } else if (this->is<Vector_t>()) {
        const Vector_t *v = this->as<Vector_t>();
        return Vector_t::make(v->etype.to_uint(), v->lanes);
    } else {
        internal_error << "Called to_uint() on bad type: " << *this;
    }
}

Type Type::element_of() const {
    if (this->is<Vector_t>()) {
        return this->as<Vector_t>()->etype;
    } else if (this->is<Set_t>()) {
        return this->as<Set_t>()->etype;
    } else {
        internal_error << "Called element_of() on bad type: " << *this;
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

Type Float_t::make(uint32_t exponent, uint32_t mantissa) {
    Float_t *node = new Float_t;
    node->exponent = exponent;
    node->mantissa = mantissa;
    return node;
}

Type Float_t::make_f64() {
    static Float_t *node = new Float_t;
    node->exponent = IEEE754_F64.exponent;
    node->mantissa = IEEE754_F64.mantissa;
    return node;
}

Type Float_t::make_f32() {
    static Float_t *node = new Float_t;
    node->exponent = IEEE754_F32.exponent;
    node->mantissa = IEEE754_F32.mantissa;
    return node;
}

Type Float_t::make_f16() {
    static Float_t *node = new Float_t;
    node->exponent = IEEE754_F16.exponent;
    node->mantissa = IEEE754_F16.mantissa;
    return node;
}

Type Float_t::make_bf16() {
    static Float_t *node = new Float_t;
    node->exponent = BFLOAT16.exponent;
    node->mantissa = BFLOAT16.mantissa;
    return node;
}

uint32_t Float_t::bits() const {
    // +1 for the sign bit.
    return 1 + this->exponent + this->mantissa;
}

bool Float_t::is_ieee754() const {
    const uint32_t e = this->exponent, m = this->mantissa;
    switch (const uint32_t bits = this->bits(); bits) {
    case 256:
    case 128:
        internal_error << "unimplemented: f" << bits;
    case 64:
        return e == IEEE754_F64.exponent && m == IEEE754_F64.mantissa;
    case 32:
        return e == IEEE754_F32.exponent && m == IEEE754_F32.mantissa;
    case 16:
        return e == IEEE754_F16.exponent && m == IEEE754_F16.mantissa;
    default:
        return false;
    }
}

bool Float_t::is_bfloat16() const {
    return this->exponent == BFLOAT16.exponent &&
           this->mantissa == BFLOAT16.mantissa;
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

namespace {

bool validate_volume(const BVH_t::Volume &volume,
                     const std::vector<BVH_t::Param> &params,
                     const std::vector<BVH_t::Param> &parent_params) {
    if (!volume.struct_type.is<Struct_t>()) {
        return false;
    }
    const Struct_t::Map &fields = volume.struct_type.as<Struct_t>()->fields;
    if (fields.size() != volume.initializers.size()) {
        return false;
    }

    for (size_t i = 0; i < fields.size(); i++) {
        const std::string &name = volume.initializers[i];
        Type type;

        auto it =
            std::find_if(params.begin(), params.end(),
                         [&](const BVH_t::Param &p) { return p.name == name; });
        if (it != params.end()) {
            type = it->type;
        } else {
            // Check parent_params for a match
            it = std::find_if(
                parent_params.begin(), parent_params.end(),
                [&](const BVH_t::Param &p) { return p.name == name; });
            if (it != parent_params.end()) {
                type = it->type;
            } else {
                // Param doesn't exist.
                return false;
            }
        }

        // Validate type
        if (!equals(type, fields[i].second)) {
            return false;
        }
    }

    return true;
}

} // namespace

Type BVH_t::make(std::string name, std::vector<Node> nodes) {
    internal_assert(!name.empty()) << "BVH_t::make received empty name";
    internal_assert(!nodes.empty()) << "BVH_t::make received empty nodes";

    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].volume.has_value()) {
            internal_assert(
                validate_volume(*nodes[i].volume, nodes[i].params, {}))
                << "Failed to validate node " << i << " of " << name;
        }
    }

    BVH_t *node = new BVH_t;
    node->name = std::move(name);
    node->nodes = std::move(nodes);
    return node;
}

Type BVH_t::make(std::string name, std::vector<BVH_t::Param> params,
                 std::vector<BVH_t::Node> nodes, BVH_t::Volume volume) {
    internal_assert(!name.empty()) << "BVH_t::make received empty name";
    internal_assert(!params.empty()) << "BVH_t::make received empty params";
    internal_assert(!nodes.empty()) << "BVH_t::make received empty nodes";

    internal_assert(validate_volume(volume, params, {}))
        << "Failed to validate parent volume of " << name;

    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].volume.has_value()) {
            internal_assert(
                validate_volume(*nodes[i].volume, nodes[i].params, params))
                << "Failed to validate node " << i << " of " << name;
        }
    }

    BVH_t *node = new BVH_t;
    node->name = std::move(name);
    node->nodes = std::move(nodes);
    node->params = std::move(params);
    node->volume = std::move(volume);
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
