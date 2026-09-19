#include "IR/Type.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    if (auto *as_vector = this->as<Vector_t>()) {
        // The width of one lane, which is what every caller is after: `bits()`
        // is asked how wide the arithmetic is, not how much storage there is.
        // `bytes()` below is the one that multiplies by the lane count.
        //
        // Without this a bitwise operation on a vector of integers failed to
        // type-check at all -- `reinterpret[[u32x4]](v) & mask`, which is how a
        // four-wide copysign is written.
        return as_vector->etype.bits();
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
        // Packed storage is exactly its elements: the twelve bytes a
        // layout's `vec3f` promised (see Vector_t::packed), an array of three
        // floats on both backends.
        if (as_vec->packed) {
            return as_vec->lanes * as_vec->etype.bytes();
        }
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
           // becomes is a tag beside the words of the arms' fields, and
           // neither adds anything that needs looking after. That lowered
           // form is primitive too -- an ADT that could be laid out in a tree
           // before LowerADTs and not after would be a strange thing to
           // explain.
           (is<ADT_t>() &&
            std::all_of(as<ADT_t>()->variants.cbegin(),
                        as<ADT_t>()->variants.cend(),
                        [](const auto &v) { return v.is_primitive(); })) ||
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

namespace {

//===--------------------------------------------------------------------===//
// Interning
//===--------------------------------------------------------------------===//
//
// One node per distinct type. Every constructor below hands its node to
// `intern`, which returns the node already made for that type if there is
// one and keeps this one otherwise, so two types that are the same are the
// same pointer. That is what lets compare_types (IR/Equality.cpp) answer the
// common question, whether two types are equal, with a pointer comparison:
// before this, comparing two expressions compared their types first, field
// by field through every nested struct, and that alone was a fifth of a
// compile. LLVM uniques its types the same way, in the LLVMContext.
//
// Identity here is exact, and stricter than equals(): equals() ignores an
// array's size and a struct's defaults and attributes, and two types that
// differ in those must stay two nodes. Children are compared by pointer,
// which is exact because they were interned before their parent was; a node
// somehow made behind the constructors' back would merely go unshared, never
// be confused with another. A type made once anyway -- a tree's, a generic's
// -- is not interned. The table keeps its types alive for the run of the
// compiler, which is what a canonical instance has to be.

template <typename T>
bool same_pointers(const std::vector<T> &a, const std::vector<T> &b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(),
                      [](const T &x, const T &y) { return x.same_as(y); });
}

bool same_exactly(const Type &a, const Type &b) {
    if (a.node_type() != b.node_type()) {
        return false;
    }
    switch (a.node_type()) {
    case IRTypeEnum::Int_t:
        return a.as<Int_t>()->bits == b.as<Int_t>()->bits;
    case IRTypeEnum::UInt_t:
        return a.as<UInt_t>()->bits == b.as<UInt_t>()->bits;
    case IRTypeEnum::Float_t:
        return a.as<Float_t>()->exponent == b.as<Float_t>()->exponent &&
               a.as<Float_t>()->mantissa == b.as<Float_t>()->mantissa;
    case IRTypeEnum::Ptr_t:
        return a.as<Ptr_t>()->etype.same_as(b.as<Ptr_t>()->etype);
    case IRTypeEnum::Ref_t:
        return a.as<Ref_t>()->name == b.as<Ref_t>()->name;
    case IRTypeEnum::ElementRef_t:
        return a.as<ElementRef_t>()->tree == b.as<ElementRef_t>()->tree &&
               a.as<ElementRef_t>()->etype.same_as(
                   b.as<ElementRef_t>()->etype);
    case IRTypeEnum::Vector_t: {
        const Vector_t *x = a.as<Vector_t>(), *y = b.as<Vector_t>();
        return x->lanes == y->lanes && x->packed == y->packed &&
               x->etype.same_as(y->etype);
    }
    case IRTypeEnum::Struct_t: {
        const Struct_t *x = a.as<Struct_t>(), *y = b.as<Struct_t>();
        if (x->name != y->name || x->attributes != y->attributes ||
            x->fields.size() != y->fields.size() ||
            x->defaults.size() != y->defaults.size()) {
            return false;
        }
        for (size_t i = 0; i < x->fields.size(); i++) {
            if (x->fields[i].name != y->fields[i].name ||
                !x->fields[i].type.same_as(y->fields[i].type)) {
                return false;
            }
        }
        for (const auto &[name, value] : x->defaults) {
            const auto other = y->defaults.find(name);
            if (other == y->defaults.end() || !equals(value, other->second)) {
                return false;
            }
        }
        return true;
    }
    case IRTypeEnum::Tuple_t:
        return same_pointers(a.as<Tuple_t>()->etypes, b.as<Tuple_t>()->etypes);
    case IRTypeEnum::Array_t: {
        const Array_t *x = a.as<Array_t>(), *y = b.as<Array_t>();
        return x->etype.same_as(y->etype) &&
               x->size.defined() == y->size.defined() &&
               (!x->size.defined() || equals(x->size, y->size));
    }
    case IRTypeEnum::DynArray_t: {
        const DynArray_t *x = a.as<DynArray_t>(), *y = b.as<DynArray_t>();
        return x->etype.same_as(y->etype) && equals(x->capacity, y->capacity);
    }
    case IRTypeEnum::Option_t:
        return a.as<Option_t>()->etype.same_as(b.as<Option_t>()->etype);
    case IRTypeEnum::ADT_t:
        return a.as<ADT_t>()->name == b.as<ADT_t>()->name &&
               same_pointers(a.as<ADT_t>()->variants,
                             b.as<ADT_t>()->variants);
    case IRTypeEnum::Set_t:
        return a.as<Set_t>()->etype.same_as(b.as<Set_t>()->etype);
    case IRTypeEnum::Function_t: {
        const Function_t *x = a.as<Function_t>(), *y = b.as<Function_t>();
        if (!x->ret_type.same_as(y->ret_type) ||
            x->arg_types.size() != y->arg_types.size()) {
            return false;
        }
        for (size_t i = 0; i < x->arg_types.size(); i++) {
            if (x->arg_types[i].is_mutable != y->arg_types[i].is_mutable ||
                !x->arg_types[i].type.same_as(y->arg_types[i].type)) {
                return false;
            }
        }
        return true;
    }
    default:
        return false; // a kind that is not interned
    }
}

Type intern(Type type) {
    static std::unordered_map<uint64_t, std::vector<Type>> table;
    std::vector<Type> &bucket = table[hash(type)];
    for (const Type &existing : bucket) {
        if (same_exactly(existing, type)) {
            return existing;
        }
    }
    bucket.push_back(type);
    return type;
}

} // namespace

Type Void_t::make() {
    static Type global_void = new Void_t;
    return global_void;
}

Type Int_t::make(uint32_t bits) {
    internal_assert(bits > 0 && bits <= 64)
        << "Unsupported bitwidth in Int_t: " << bits;
    Int_t *node = new Int_t;
    node->bits = bits;
    return intern(node);
}

Type UInt_t::make(uint32_t bits) {
    internal_assert(bits > 0 && bits <= 64)
        << "Unsupported bitwidth in UInt_t: " << bits;
    UInt_t *node = new UInt_t;
    node->bits = bits;
    return intern(node);
}

Type Index_t::make() {
    static Type global_idx = new Index_t;
    return global_idx;
}

Type Float_t::make(uint32_t exponent, uint32_t mantissa) {
    Float_t *node = new Float_t;
    node->exponent = exponent;
    node->mantissa = mantissa;
    return intern(node);
}

// The named floats go through make, so that they are the interned node any
// other request for the same format gets; kept in a static so that asking
// for one does not build a candidate to throw away each time.
Type Float_t::make_f64() {
    static Type f64 = make(IEEE754_F64.exponent, IEEE754_F64.mantissa);
    return f64;
}

Type Float_t::make_f32() {
    static Type f32 = make(IEEE754_F32.exponent, IEEE754_F32.mantissa);
    return f32;
}

Type Float_t::make_f16() {
    static Type f16 = make(IEEE754_F16.exponent, IEEE754_F16.mantissa);
    return f16;
}

Type Float_t::make_bf16() {
    static Type bf16 = make(BFLOAT16.exponent, BFLOAT16.mantissa);
    return bf16;
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
    return intern(node);
}

Type Ref_t::make(std::string name) {
    internal_assert(!name.empty()) << "Ref_t::make received empty name";
    Ref_t *node = new Ref_t;
    node->name = std::move(name);
    return intern(node);
}

Type ElementRef_t::make(Type etype, std::string tree) {
    internal_assert(etype.defined())
        << "ElementRef_t::make received undefined etype";
    internal_assert(!tree.empty()) << "ElementRef_t::make received no tree";
    ElementRef_t *node = new ElementRef_t;
    node->etype = std::move(etype);
    node->tree = std::move(tree);
    return intern(node);
}

Type Vector_t::make(Type etype, uint32_t lanes, bool packed) {
    internal_assert(etype.defined())
        << "Vector_t::make received undefined etype";
    Vector_t *node = new Vector_t;
    node->etype = std::move(etype);
    node->lanes = lanes;
    node->packed = packed;
    return intern(node);
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
    return intern(node);
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
    return intern(node);
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
    return intern(node);
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
    return intern(node);
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
    return intern(node);
}

Type Option_t::make(Type etype) {
    internal_assert(etype.defined())
        << "Option_t::make received undefined etype";
    Option_t *node = new Option_t;
    node->etype = std::move(etype);
    return intern(node);
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
    return intern(node);
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

std::string component_field(uint32_t k) { return "!" + std::to_string(k); }

uint64_t layout_align(const Type &type) {
    if (type.is<Bool_t>()) {
        return 1;
    }
    if (type.is<Int_t, UInt_t, Float_t>()) {
        return type.bytes();
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        // Packed storage is an array of the element; a vector proper is
        // aligned to its whole (power-of-two) size, as the targets align it.
        return v->packed ? layout_align(v->etype) : type.bytes();
    }
    if (type.is<Ptr_t>() || type.is_reference()) {
        return 8;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        if (s->is_packed()) {
            return 1;
        }
        uint64_t align = 1;
        for (const TypedVar &f : s->fields) {
            if (!f.type.is<Ref_t>()) {
                align = std::max(align, layout_align(f.type));
            }
        }
        return align;
    }
    internal_error << "[unimplemented] layout_align of " << type;
}

uint64_t layout_bytes(const Type &type) {
    if (type.is<Bool_t>()) {
        return 1;
    }
    if (type.is<Int_t, UInt_t, Float_t, Vector_t>()) {
        return type.bytes();
    }
    if (type.is<Ptr_t>() || type.is_reference()) {
        return 8;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        // C's rule, which is LLVM's for a struct that is not packed: each
        // field at the next offset aligned for it, the whole rounded up to
        // its own alignment. A reference-typed field takes no room, as the
        // backend leaves it out (see the Struct_t visitor in
        // CodeGen/CodeGen_LLVM.cpp).
        uint64_t offset = 0;
        for (const TypedVar &f : s->fields) {
            if (f.type.is<Ref_t>()) {
                continue;
            }
            const uint64_t align = s->is_packed() ? 1 : layout_align(f.type);
            offset = (offset + align - 1) / align * align;
            offset += layout_bytes(f.type);
        }
        const uint64_t align = layout_align(type);
        return (offset + align - 1) / align * align;
    }
    internal_error << "[unimplemented] layout_bytes of " << type;
}

uint64_t layout_offset(const Struct_t &s, size_t index) {
    internal_assert(index < s.fields.size())
        << "field " << index << " of a struct with " << s.fields.size();
    internal_assert(!s.fields[index].type.is<Ref_t>())
        << "field " << s.fields[index].name << " of " << s.name
        << " is a reference, which takes no room (see layout_bytes)";
    uint64_t offset = 0;
    for (size_t i = 0;; i++) {
        if (s.fields[i].type.is<Ref_t>()) {
            continue;
        }
        const uint64_t align = s.is_packed() ? 1 : layout_align(s.fields[i].type);
        offset = (offset + align - 1) / align * align;
        if (i == index) {
            return offset;
        }
        offset += layout_bytes(s.fields[i].type);
    }
}

Type widen(const Type &type, uint32_t lanes) {
    if (const Struct_t *s = type.as<Struct_t>()) {
        Struct_t::Map fields;
        fields.reserve(s->fields.size());
        for (const TypedVar &f : s->fields) {
            fields.emplace_back(f.name, widen(f.type, lanes));
        }
        return Struct_t::make(s->name + "$v" + std::to_string(lanes), fields,
                              s->attributes);
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        Struct_t::Map fields;
        fields.reserve(v->lanes);
        for (uint32_t k = 0; k < v->lanes; k++) {
            fields.emplace_back(component_field(k),
                                Vector_t::make(v->etype, lanes));
        }
        std::ostringstream name;
        name << type << "$v" << lanes;
        return Struct_t::make(name.str(), fields);
    }
    return Vector_t::make(type, lanes);
}

std::optional<uint32_t> widened_lanes(const Type &type) {
    const Struct_t *s = type.as<Struct_t>();
    if (s == nullptr) {
        return std::nullopt;
    }
    const size_t at = s->name.rfind("$v");
    if (at == std::string::npos || at + 2 >= s->name.size()) {
        return std::nullopt;
    }
    uint32_t lanes = 0;
    for (size_t i = at + 2; i < s->name.size(); i++) {
        if (s->name[i] < '0' || s->name[i] > '9') {
            return std::nullopt;
        }
        lanes = lanes * 10 + uint32_t(s->name[i] - '0');
    }
    return lanes == 0 ? std::nullopt : std::optional<uint32_t>(lanes);
}

Type narrow(const Type &type, uint32_t lanes) {
    if (const Vector_t *v = type.as<Vector_t>()) {
        internal_assert(v->lanes == lanes)
            << "narrow of " << type << ", which is not " << lanes << " wide";
        return v->etype;
    }
    const Struct_t *s = type.as<Struct_t>();
    const std::string suffix = "$v" + std::to_string(lanes);
    internal_assert(s != nullptr && s->name.size() > suffix.size() &&
                    s->name.compare(s->name.size() - suffix.size(),
                                    suffix.size(), suffix) == 0)
        << "narrow of " << type << ", which widen() did not make for "
        << lanes << " lanes";
    std::string name = s->name.substr(0, s->name.size() - suffix.size());
    // One gang vector per component of a short vector, named for the vector:
    // `f32x3` or `[[packed]] f32x3`. Told from a struct of vector fields by
    // the name, which is the vector's own spelling.
    const bool components =
        !s->fields.empty() && s->fields[0].name == component_field(0) &&
        std::all_of(s->fields.begin(), s->fields.end(),
                    [&](const TypedVar &f) {
                        return f.type.is<Vector_t>() &&
                               f.type.as<Vector_t>()->lanes == lanes;
                    });
    if (components) {
        const std::string packed_prefix = "[[packed]] ";
        const bool packed =
            name.compare(0, packed_prefix.size(), packed_prefix) == 0;
        const Type etype = s->fields[0].type.as<Vector_t>()->etype;
        const Type vector =
            Vector_t::make(etype, uint32_t(s->fields.size()), packed);
        std::ostringstream spelled;
        spelled << vector;
        if (spelled.str() == name) {
            return vector;
        }
    }
    Struct_t::Map fields;
    fields.reserve(s->fields.size());
    for (const TypedVar &f : s->fields) {
        fields.emplace_back(f.name, narrow(f.type, lanes));
    }
    return Struct_t::make(std::move(name), std::move(fields), s->attributes);
}

Type Set_t::make(Type etype) {
    internal_assert(etype.defined()) << "Set_t::make received undefined etype";
    Set_t *node = new Set_t;
    node->etype = std::move(etype);
    return intern(node);
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
    return intern(node);
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
        // A vector of aggregates is one aggregate per lane -- the place a
        // vector of pointers dereferences to (see Deref::make) -- and its
        // field is that field, one per lane.
        if (as_vec->etype.is<Struct_t>()) {
            return Vector_t::make(get_field_type(as_vec->etype, field),
                                  as_vec->lanes);
        }
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
