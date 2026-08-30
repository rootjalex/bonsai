#include "IR/Type.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "Utils.h"

namespace bonsai {
namespace ir {

TypedVar::operator Expr() const { return Var::make(type, name); }

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
    if (this->is<Bool_t>()) {
        return 1;
    }
    internal_error << "Called bits() on bad type: " << *this;
}

uint32_t Type::bytes() const {
    if (is<Int_t, UInt_t, Float_t>()) {
        // TODO(ajr): is this always right?
        return (bits() + 7) / 8;
    } else if (is<Bool_t>()) {
        // Not (bits() + 7) / 8 by luck: a bool occupies a whole byte in
        // memory even though it carries one bit, and a vector of them is
        // stored a byte per lane.
        return 1;
    } else if (auto *as_vec = as<Vector_t>()) {
        // How much a vector *occupies*, which is not how much data it
        // holds: both backends round the lane count up to a power of two,
        // so a vector[f32,3] is twelve bytes of floats inside sixteen
        // bytes of storage. That is LLVM's getTypeAllocSize for
        // <3 x float>, and it is what clang gives the ext_vector_type(3)
        // the C++ backend emits. Returning the packed twelve would mean
        // walking an array of them three floats at a time and landing in
        // the middle of the next element.
        //
        // This duplicates a rule the targets own, so it is only for the
        // front end, where there is no target to ask yet. Code that is
        // generating IR should emit an ir::SizeOf and let the backend
        // answer; CodeGen_LLVM checks the two against each other.
        return next_power_of_two(static_cast<int32_t>(as_vec->lanes)) *
               as_vec->etype.bytes();
    }
    internal_error << "[unimplemented] bytes() called on: " << *this;
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
    return this->is<Int_t, UInt_t, Index_t>() ||
           (this->is<Vector_t>() &&
            this->as<Vector_t>()->etype.is_int_or_uint()) ||
           (this->is<Array_t>() && this->as<Array_t>()->etype.is_int_or_uint());
}

bool Type::is_int_tuple() const {
    const Tuple_t *tuple = this->as<Tuple_t>();
    if (tuple == nullptr) {
        return false;
    }
    return std::all_of(tuple->etypes.cbegin(), tuple->etypes.cend(),
                       [](const auto &t) { return t.is_int_or_uint(); });
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
    return this->is<Int_t, UInt_t, Float_t, Bool_t>();
}

bool Type::is_vector() const {
    // TODO: what counts as vector?
    return this->is<Vector_t>();
}

bool Type::is_reference() const { return this->is<Array_t, DynArray_t>(); }

bool Type::is_numeric() const {
    // scalar + vector of numbers
    // TODO: let Struct_ts overload their numeric operators.
    return this->is_int_or_uint() || this->is_float();
}

bool Type::is_primitive() const {
    return is<Int_t, UInt_t, Float_t, Bool_t, Ptr_t>() ||
           (is<Vector_t>() && element_of().is_primitive()) ||
           (is<Struct_t>() &&
            std::all_of(as<Struct_t>()->fields.cbegin(),
                        as<Struct_t>()->fields.cend(),
                        [](const auto &p) { return p.type.is_primitive(); })) ||
           (is<Tuple_t>() &&
            std::all_of(as<Tuple_t>()->etypes.cbegin(),
                        as<Tuple_t>()->etypes.cend(),
                        [](const auto &p) { return p.is_primitive(); })) ||
           // A variant type is plain data when every variant is: what it
           // becomes is a tag beside a union of them, and neither the tag nor
           // the union adds anything that needs looking after. The union is
           // here as well because that lowered form has to stay primitive --
           // an ADT that could be laid out in a tree before LowerADTs and not
           // after would be a strange thing to explain.
           (is<ADT_t>() &&
            std::all_of(as<ADT_t>()->variants.cbegin(),
                        as<ADT_t>()->variants.cend(),
                        [](const auto &v) { return v.is_primitive(); })) ||
           (is<Union_t>() &&
            std::all_of(as<Union_t>()->members.cbegin(),
                        as<Union_t>()->members.cend(),
                        [](const auto &m) { return m.type.is_primitive(); })) ||
           (is<Array_t>() && as<Array_t>()->etype.is_primitive());
}

bool Type::is_stack_allocatable() const {
    // TODO(ajr): some (small) structs?
    return is<Int_t, UInt_t, Float_t, Bool_t, Ptr_t>() ||
           (is<Vector_t>() && element_of().is_stack_allocatable()) ||
           (is<Tuple_t>() &&
            std::all_of(
                as<Tuple_t>()->etypes.cbegin(), as<Tuple_t>()->etypes.cend(),
                [](const auto &p) { return p.is_stack_allocatable(); }));
}

bool Type::is_iterable() const { return is<Vector_t, Array_t, Set_t>(); }

bool Type::is_func() const { return is<Function_t>(); }

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
    } else if (this->is<BVH_t>()) {
        return this->as<BVH_t>()->primitive;
    } else if (this->is<Array_t>()) {
        return this->as<Array_t>()->etype;
    } else if (this->is<DynArray_t>()) {
        return this->as<DynArray_t>()->etype;
    } else if (this->is<Ptr_t>()) {
        return this->as<Ptr_t>()->etype;
    } else {
        internal_error << "Called element_of() on bad type: " << *this;
    }
}

Type Type::with_etype(Type etype) const {
    auto do_recurse = [](const Type &t) {
        return t.is<Vector_t, Array_t, Set_t>();
    };
    if (const Vector_t *vec = this->as<Vector_t>()) {
        const Type vtype = vec->etype;
        Type inner = do_recurse(vtype) ? vtype.with_etype(std::move(etype))
                                       : std::move(etype);
        return Vector_t::make(std::move(inner), vec->lanes);
    } else if (const Array_t *array = this->as<Array_t>()) {
        const Type vtype = array->etype;
        Type inner = do_recurse(vtype) ? vtype.with_etype(std::move(etype))
                                       : std::move(etype);
        return Array_t::make(std::move(inner), array->size);
    } else if (const Set_t *set = this->as<Set_t>()) {
        const Type vtype = set->etype;
        Type inner = do_recurse(vtype) ? vtype.with_etype(std::move(etype))
                                       : std::move(etype);
        return Set_t::make(std::move(inner));
    }
    internal_error << "with_etype(" << etype << ") called on " << *this
                   << " which is not a collection.";
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

Type Index_t::make() {
    static Type global_idx = new Index_t;
    return global_idx;
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

Type String_t::make() {
    static Type global_str = new String_t;
    return global_str;
}

Type Ptr_t::make(Type etype) {
    internal_assert(etype.defined()) << "Ptr_t::make received undefined etype";
    Ptr_t *node = new Ptr_t;
    node->etype = std::move(etype);
    return node;
}

Type Ref_t::make(std::string name) {
    internal_assert(!name.empty()) << "Ref_t::make received empty name";
    Ref_t *node = new Ref_t;
    node->name = std::move(name);
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

Type Struct_t::make(std::string name, Struct_t::Map fields,
                    std::vector<Attribute> attributes) {
    internal_assert(!name.empty()) << "Struct_t::make received undefined name";
    internal_assert(std::all_of(fields.cbegin(), fields.cend(),
                                [](const auto &p) { return p.type.defined(); }))
        << "Struct_t::make received undefined field type in definition of "
        << name;
    Struct_t *node = new Struct_t;
    node->name = std::move(name);
    node->fields = std::move(fields);
    node->attributes = std::move(attributes);
    return node;
}

Type Struct_t::make(std::string name, Struct_t::Map fields,
                    Struct_t::DefMap defaults,
                    std::vector<Attribute> attributes) {
    internal_assert(!name.empty()) << "Struct_t::make received undefined name";
    internal_assert(std::all_of(fields.cbegin(), fields.cend(),
                                [](const auto &p) { return p.type.defined(); }))
        << "Struct_t::make received undefined field type in definition of "
        << name;
    internal_assert(std::all_of(defaults.cbegin(), defaults.cend(),
                                [](const auto &p) {
                                    return p.second.defined() &&
                                           p.second.type().defined();
                                }))
        << "Struct_t::make received undefined default expression";
    Struct_t *node = new Struct_t;
    node->name = std::move(name);
    node->fields = std::move(fields);
    node->defaults = std::move(defaults);
    node->attributes = std::move(attributes);
    return node;
}

bool Struct_t::is_packed() const {
    return std::find(attributes.cbegin(), attributes.cend(),
                     Attribute::packed) != attributes.cend();
}

bool Struct_t::is_layout() const {
    return std::find(attributes.cbegin(), attributes.cend(),
                     Attribute::layout) != attributes.cend();
}

Type Tuple_t::make(std::vector<Type> etypes) {
    Tuple_t *node = new Tuple_t;
    node->etypes = std::move(etypes);
    return node;
}

Type Array_t::make(Type etype, Expr size) {
    internal_assert(etype.defined())
        << "Array_t::make received undefined etype";
    if (size.defined()) {
        internal_assert(size.type().is_int_or_uint())
            << "Array_t::make received non-integer size: " << size;
    }
    Array_t *node = new Array_t;
    node->etype = std::move(etype);
    node->size = std::move(size);
    return node;
}

Type DynArray_t::make(Type etype, Expr capacity) {
    internal_assert(etype.defined())
        << "DynArray_t::make received undefined etype";
    if (!capacity.defined()) {
        capacity = Expr(16);
    }
    DynArray_t *node = new DynArray_t;
    node->etype = std::move(etype);
    node->capacity = std::move(capacity);
    return node;
}

Type Option_t::make(Type etype) {
    internal_assert(etype.defined())
        << "Option_t::make received undefined etype";
    Option_t *node = new Option_t;
    node->etype = std::move(etype);
    return node;
}

Type ADT_t::make(std::string name, Variants variants) {
    internal_assert(!name.empty()) << "ADT_t::make received an unnamed type";
    internal_assert(!variants.empty())
        << "ADT_t::make received no variants for " << name
        << ". A type with nothing to be has no values.";
    std::set<std::string> seen;
    for (const Type &variant : variants) {
        const Struct_t *as_struct = variant.as<Struct_t>();
        internal_assert(as_struct)
            << "Variant of " << name << " is not a struct: " << variant;
        internal_assert(seen.insert(as_struct->name).second)
            << name << " has two variants called " << as_struct->name;
    }
    ADT_t *node = new ADT_t;
    node->name = std::move(name);
    node->variants = std::move(variants);
    return node;
}

std::optional<size_t> ADT_t::index_of(const std::string &variant) const {
    for (size_t i = 0; i < variants.size(); i++) {
        if (variant_name(i) == variant) {
            return i;
        }
    }
    return std::nullopt;
}

const Struct_t::Map &ADT_t::fields(size_t index) const {
    internal_assert(index < variants.size())
        << name << " has no variant " << index;
    return variants[index].as<Struct_t>()->fields;
}

const std::string &ADT_t::variant_name(size_t index) const {
    internal_assert(index < variants.size())
        << name << " has no variant " << index;
    return variants[index].as<Struct_t>()->name;
}

Type Union_t::make(std::string name, Map members) {
    internal_assert(!name.empty()) << "Union_t::make received an unnamed type";
    internal_assert(!members.empty())
        << "Union_t::make received no members for " << name
        << ". A union of nothing has no size to take.";
    std::set<std::string> seen;
    for (const TypedVar &member : members) {
        internal_assert(member.type.defined())
            << "Member " << member.name << " of union " << name
            << " has no type";
        internal_assert(seen.insert(member.name).second)
            << "Union " << name << " has two members called " << member.name;
    }
    Union_t *node = new Union_t;
    node->name = std::move(name);
    node->members = std::move(members);
    return node;
}

Type Union_t::member(const std::string &name) const {
    for (const TypedVar &m : members) {
        if (m.name == name) {
            return m.type;
        }
    }
    return Type();
}

Type Set_t::make(Type etype) {
    internal_assert(etype.defined()) << "Set_t::make received undefined etype";
    Set_t *node = new Set_t;
    node->etype = std::move(etype);
    return node;
}

Type Function_t::make(Type ret_type, std::vector<ArgSig> arg_types) {
    internal_assert(ret_type.defined())
        << "Function_t::make received undefined ret_type";
    internal_assert(std::all_of(arg_types.cbegin(), arg_types.cend(),
                                [](const auto &p) { return p.type.defined(); }))
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

Annotation::Aggregate::OpType
Annotation::Aggregate::str_to_op(const std::string &op) {
    if (op == "avg") {
        return Annotation::Aggregate::OpType::avg;
    } else if (op == "count") {
        return Annotation::Aggregate::OpType::count;
    } else if (op == "max") {
        return Annotation::Aggregate::OpType::max;
    } else if (op == "min") {
        return Annotation::Aggregate::OpType::min;
    } else if (op == "prod") {
        return Annotation::Aggregate::OpType::prod;
    } else if (op == "sum") {
        return Annotation::Aggregate::OpType::sum;
    }
    internal_error << "Unknown Aggregate type: " << op;
}

namespace {

bool validate_volume(const Annotation::Volume &volume,
                     const std::vector<TypedVar> &params) {
    if (!volume.struct_type.is<Struct_t>()) {
        return false;
    }
    const Struct_t::Map &fields = volume.struct_type.as<Struct_t>()->fields;
    if (fields.size() != volume.initializers.size()) {
        return false;
    }

    for (size_t i = 0; i < fields.size(); i++) {
        const std::string &name = volume.initializers[i];

        auto it =
            std::find_if(params.begin(), params.end(),
                         [&](const TypedVar &p) { return p.name == name; });

        if (it == params.end()) {
            return false;
        }

        // Validate type
        if (!equals(it->type, fields[i].type)) {
            return false;
        }
    }

    return true;
}

} // namespace

Type BVH_t::make(ir::Type primitive, std::string name,
                 std::vector<Node> nodes) {
    internal_assert(primitive.defined())
        << "BVH_t::make received undefined prim_t";
    internal_assert(!name.empty()) << "BVH_t::make received empty name";
    internal_assert(!nodes.empty()) << "BVH_t::make received empty nodes";

    // TODO: check that prim_t is contained in some node (leaves)?
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto &annot : nodes[i].annotations) {
            // TODO: other validations?
            if (const auto *volume = annot.as<Annotation::Volume>()) {
                internal_assert(validate_volume(*volume, nodes[i].fields()))
                    << "Failed to validate node " << i << " of " << name;
            }
        }
    }

    BVH_t *node = new BVH_t;
    node->primitive = std::move(primitive);
    node->name = std::move(name);
    node->nodes = std::move(nodes);
    return node;
}

Type BVH_t::make(ir::Type primitive, std::string name,
                 const std::vector<TypedVar> &globals,
                 std::vector<BVH_t::Node> nodes,
                 std::vector<Annotation> annotations) {
    internal_assert(primitive.defined())
        << "BVH_t::make received undefined prim_t";
    internal_assert(!name.empty()) << "BVH_t::make received empty name";
    internal_assert(!globals.empty()) << "BVH_t::make received empty globals";
    internal_assert(!nodes.empty()) << "BVH_t::make received empty nodes";

    for (const auto &annot : annotations) {
        // TODO: other validations?
        if (const auto *volume = annot.as<Annotation::Volume>()) {
            internal_assert(validate_volume(*volume, globals))
                << "Failed to validate node of " << name;
        }
    }

    // TODO: check that prim_t is contained in some node (leaves)?
    for (size_t i = 0; i < nodes.size(); i++) {
        // Insert params into the front of nodes[i].params
        std::vector<TypedVar> copy = globals;
        const auto &params = nodes[i].fields();
        // TODO: figure out why insert() segfaults.
        // copy.insert(globals.end(), params.begin(), params.end());
        for (const auto &[name, type] : params) {
            copy.push_back({name, type});
        }
        Type struct_type = Struct_t::make(nodes[i].name(), std::move(copy));
        nodes[i].struct_type = std::move(struct_type);

        // TODO: validate no duplicate annotation types!
        nodes[i].annotations.insert(nodes[i].annotations.end(),
                                    annotations.begin(), annotations.end());
    }

    BVH_t *node = new BVH_t;
    node->primitive = std::move(primitive);
    node->name = std::move(name);
    node->nodes = std::move(nodes);
    return node;
}

Type Rand_State_t::make() {
    static Type global_rng = new Rand_State_t;
    return global_rng;
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
    } else if (const Array_t *as_array = struct_type.as<Array_t>()) {
        return as_array->etype;
    } else if (const Tuple_t *as_tuple = struct_type.as<Tuple_t>()) {
        internal_assert(!field.empty());
        internal_assert(field.starts_with("_field"))
            << field << " of " << struct_type;
        int64_t p = field.find_first_of("0123456789");
        std::string number = field.substr(p);
        internal_assert(!number.empty()) << field;
        std::stringstream ss(number);
        uint64_t position;
        internal_assert(ss >> position) << field;
        internal_assert(position < as_tuple->etypes.size());
        return as_tuple->etypes[position];
    } else if (const Union_t *as_union = struct_type.as<Union_t>()) {
        // Naming a member of a union says which type to read its bytes at.
        const Type member = as_union->member(field);
        internal_assert(member.defined())
            << "Union " << as_union->name << " has no member " << field;
        return member;
    } else if (const Ptr_t *as_ptr = struct_type.as<Ptr_t>()) {
        return get_field_type(as_ptr->etype, field);
    } else {
        internal_error << "Failed to find field: " << field
                       << " in non-(struct | vec) type: " << struct_type;
    }
}

std::string geometric_element_name(const Type &type) {
    if (const Struct_t *as_struct = type.as<Struct_t>()) {
        return as_struct->name;
    }
    if (const ADT_t *as_adt = type.as<ADT_t>()) {
        return as_adt->name;
    }
    return {};
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
