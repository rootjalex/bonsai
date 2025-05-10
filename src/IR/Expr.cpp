#include "IR/Expr.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Float16.h"
#include "IR/Printer.h"

#include "IR/TypeEnforcement.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace ir {
namespace {

// Returns whether `type` is a struct with one element, that has the same type
// as `expected_type`.
bool is_single_element_struct_with_type(const Type &type,
                                        const Type &expected_type) {
    if (const auto *struct_t = type.as<Struct_t>()) {
        if (struct_t->fields.size() == 1) {
            return equals(struct_t->fields.front().type, expected_type);
        }
    }
    return false;
}
} // namespace

Expr::Expr(int8_t x) : IRHandle(IntImm::make(Int_t::make(8), x)) {}

Expr::Expr(int16_t x) : IRHandle(IntImm::make(Int_t::make(16), x)) {}

Expr::Expr(int32_t x) : IRHandle(IntImm::make(Int_t::make(32), x)) {}

Expr::Expr(int64_t x) : IRHandle(IntImm::make(Int_t::make(64), x)) {}

Expr IntImm::make(Type t, int64_t value) {
    const bool infer_types = type_enforcement_enabled() || t.defined();

    if (infer_types) {
        internal_assert(t.defined() && t.is_int() && t.is_scalar())
            << "IntImm::make passed bad type for value: " << value
            << " type: " << t;

        // Normalize the value by dropping the high bits.
        // Since left-shift of negative value is UB in C++, cast to uint64
        // first; it's unlikely any compilers we care about will misbehave, but
        // UBSan will complain.
        value = (int64_t)(((uint64_t)value) << (64 - t.bits()));
        // Then sign-extending to get them back
        value >>= (64 - t.bits());
    }

    IntImm *node = new IntImm;
    node->type = std::move(t);
    node->value = value;
    return node;
}

Expr UIntImm::make(Type t, uint64_t value) {
    const bool infer_types = type_enforcement_enabled() || t.defined();

    if (infer_types) {
        internal_assert(t.defined() && t.is_uint() && t.is_scalar())
            << "UIntImm::make passed bad type for value: " << value
            << " type: " << t << "\n"
            << t.defined() << " && " << t.is_uint() << " && " << t.is_scalar();
        internal_assert(t.bits() >= 1 && t.bits() <= 64)
            << "UIntImm must have between 1 and 64 bits\n";

        // Normalize the value by dropping the high bits
        value <<= (64 - t.bits());
        value >>= (64 - t.bits());
    }

    UIntImm *node = new UIntImm;
    node->type = std::move(t);
    node->value = value;
    return node;
}

Expr FloatImm::make(Type t, double value) {
    // internal_assert(t.is_float() && t.is_scalar())
    //     << "FloatImm must be a scalar Float\n";

    const bool infer_types = type_enforcement_enabled() || t.defined();

    if (infer_types) {
        internal_assert(t.defined())
            << "FloatImm::make passed undefined type for value: " << value;
        switch (t.bits()) {
        case 16:
            value = cast_to_float16(value);
            break;
        case 32:
            value = (float)value;
            break;
        case 64:
            break;
        default:
            internal_error
                << "FloatImm must be f16, f32, or f64, instead received: " << t;
        }
    }

    FloatImm *node = new FloatImm;
    node->type = std::move(t);
    node->value = value;
    return node;
}

Expr BoolImm::make(bool value) {
    static Expr global_true = []() {
        BoolImm *b = new BoolImm;
        b->value = true;
        b->type = Bool_t::make();
        return b;
    }();
    static Expr global_false = []() {
        BoolImm *b = new BoolImm;
        b->value = false;
        b->type = Bool_t::make();
        return b;
    }();

    return value ? global_true : global_false;
}

Expr VecImm::make(std::vector<Expr> values) {
    internal_assert(!values.empty()) << "unexpected empty values in VecImm";

    VecImm *node = new VecImm;
    Type element_type = values.front().type();
    if (const bool infer_types =
            type_enforcement_enabled() || element_type.defined();
        infer_types) {
        // TODO: support?
        internal_assert(element_type.is_scalar())
            << "immediate of non-scalar: " << element_type;
        for (const Expr &e : values) {
            internal_assert(is_const(e))
                << "VecImm requires all constant values, received: " << e;
            internal_assert(equals(e.type(), element_type))
                << "VecImm requires uniform element type, expected: "
                << element_type << " due to first element: " << values.front()
                << ", but received: " << e << " of type: " << e.type();
        }
        node->type = Vector_t::make(element_type, values.size());
    }

    node->values = std::move(values);
    return node;
}

Expr Infinity::make(Type t) {
    internal_assert(t.defined() && t.is_numeric())
        << "Infinity can be made for numeric types only: " << t;

    Infinity *node = new Infinity;
    node->type = std::move(t);
    return node;
}

Expr Var::make(Type type, const std::string &name) {
    internal_assert(!name.empty())
        << "Var::make called with empty name and type: " << type;
    internal_assert(!type_enforcement_enabled() || type.defined())
        << "Var::make called with undefined type for name: " << name;
    Var *node = new Var;
    node->type = type;
    node->name = name;
    return node;
}

bool BinOp::is_numeric_op(const BinOp::OpType &op) {
    switch (op) {
    // Technically, And, Or, BwAnd, BwOr, and Xor keep the type of the operands.
    // maybe need to rename this function.
    case BinOp::LAnd:
    case BinOp::LOr:
    case BinOp::Xor:
    case BinOp::BwAnd:
    case BinOp::BwOr:
    case BinOp::Add:
    case BinOp::Mod:
    case BinOp::Mul:
    case BinOp::Div:
    case BinOp::Sub:
    case BinOp::Shl:
    case BinOp::Shr:
        return true;
    case BinOp::Neq:
    case BinOp::Eq:
    case BinOp::Le:
    case BinOp::Lt:
        return false;
    }
}

bool BinOp::is_boolean_op(const BinOp::OpType &op) {
    switch (op) {
    case BinOp::LAnd:
    case BinOp::LOr:
    case BinOp::Xor:   // see above note
    case BinOp::BwAnd: // see above note
    case BinOp::BwOr:  // see above note
    case BinOp::Add:
    case BinOp::Mod:
    case BinOp::Mul:
    case BinOp::Div:
    case BinOp::Sub:
    case BinOp::Shl:
    case BinOp::Shr:
        return false;
    case BinOp::Neq:
    case BinOp::Eq:
    case BinOp::Le:
    case BinOp::Lt:
        return true;
    }
}

namespace {

// Returns whether this type is valid for logical operations.
bool is_valid_logical_operation(Type type) {
    return type.is<Option_t, Bool_t>();
}

void try_match_types(Expr &a, Expr &b) {
    if (a.type().defined() && b.type().defined()) {
        if (equals(a.type(), b.type())) {
            return;
        }
        if (a.type().is<Option_t>() && b.type().is_bool()) {
            a = Cast::make(Bool_t::make(), a);
        } else if (b.type().is<Option_t>() && a.type().is_bool()) {
            b = Cast::make(Bool_t::make(), b);
            return;
        }
        internal_assert(!a.type().is<Option_t>())
            << "Trying to match option type: " << a << " with: " << b;
        internal_assert(!b.type().is<Option_t>())
            << "Trying to match: " << a << " with option type: " << b;

        // Try broadcasting
        if (a.type().is_vector() && b.type().is_scalar()) {
            b = Broadcast::make(a.type().lanes(), b);
            // recurse in case wrong constant types still.
            try_match_types(a, b);
            return;
        } else if (b.type().is_vector() && a.type().is_scalar()) {
            a = Broadcast::make(b.type().lanes(), a);
            // recurse in case wrong constant types still.
            try_match_types(a, b);
            return;
        }

        internal_assert(is_const(a) || is_const(b))
            << "Implicit casting of types: " << a << " is not the same type as "
            << b << ": " << a.type() << " versus " << b.type();
        if (is_const(a)) {
            a = constant_cast(b.type(), a);
        } else {
            b = constant_cast(a.type(), b);
        }
        // // TODO: is this right?
        // // Cast to the larger bitwidth
        // if (a.type().bits() > b.type().bits()) {
        //     b = constant_cast(a.type(), b);
        // } else if (b.type().bits() > a.type().bits()) {
        //     a = constant_cast(b.type(), a);
        // } else {
        //     internal_error << "Same bitwidth, not sure how to cast: " << a <<
        //     " and " << b
        //                    << " are types " << a.type() << " and " <<
        //                    b.type();
        // }
    } else if (a.type().defined() && !b.type().defined() && is_const(b)) {
        internal_assert(!a.type().is<Option_t>());
        b = constant_cast(a.type(), b);
    } else if (b.type().defined() && !a.type().defined() && is_const(a)) {
        internal_assert(!b.type().is<Option_t>());
        a = constant_cast(b.type(), a);
    }
    // otherwise can't (currently) do better.
}

} // namespace

Expr BinOp::make(BinOp::OpType op, Expr a, Expr b) {
    // TODO: operator overloading and broadcasting!
    internal_assert(a.defined() && b.defined())
        << "BinOp of undefined: " << a << to_string(op) << b;

    BinOp *node = new BinOp;

    try_match_types(a, b);

    const bool infer_types = type_enforcement_enabled() ||
                             (a.type().defined() && b.type().defined());
    if (infer_types) {
        internal_assert(equals(a.type(), b.type()))
            << "BinOp of mismatched types: " << a << " : " << a.type() << " "
            << to_string(op) << " " << b << " : " << b.type();

        if (op == BinOp::LAnd || op == BinOp::LOr) {
            // Verify logical operations only act upon options, bools aggregates
            // of bools.
            internal_assert(is_valid_logical_operation(a.type()) &&
                            is_valid_logical_operation(b.type()))
                << a << " : " << a.type() << ", " << b << " : " << b.type();
            if (a.type().is<Option_t>() || b.type().is<Option_t>()) {
                // TODO: handle vectors of options?
                node->type = Bool_t::make();
                a = Cast::make(node->type, std::move(a));
                b = Cast::make(node->type, std::move(b));
            } else {
                // And and Or propagate operand types.
                node->type = a.type();
            }
        } else {
            internal_assert(a.type().is_numeric() || a.type().is_bool())
                << "BinOp of non-number or boolean types: " << a << " : "
                << a.type() << " " << to_string(op) << " " << b << " : "
                << b.type();

            if (BinOp::is_numeric_op(op)) {
                node->type = a.type();
            } else if (BinOp::is_boolean_op(op)) {
                node->type = a.type().to_bool();
            } else {
                internal_error << "Cannot infer output type: " << a
                               << to_string(op) << b;
            }
        }
    }

    node->op = op;
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Expr UnOp::make(UnOp::OpType op, Expr a) {
    internal_assert(a.defined()) << "UnOp of undefined: " << to_string(op) << a;

    UnOp *node = new UnOp;

    const bool infer_types = type_enforcement_enabled() || a.type().defined();
    if (infer_types) {
        if (op == UnOp::Not) {
            internal_assert(is_valid_logical_operation(a.type())) << a.type();
            if (a.type().is<Option_t>()) {
                a = Cast::make(Bool_t::make(), a);
            }
            // not on only integers and boolean? what does not of float mean
            internal_assert(a.type().is_int_or_uint() || a.type().is_bool())
                << "Cannot not non-([u]int | bool): " << to_string(op) << a;
            node->type = a.type();
        } else {
            // Must be signed int or float?
            internal_assert(a.type().is_float() || a.type().is_int())
                << "Cannot negate non-(int | float): " << to_string(op) << a;
            node->type = a.type();
        }
    }

    node->op = op;
    node->a = std::move(a);
    return node;
}

Expr Select::make(Expr cond, Expr tvalue, Expr fvalue) {
    internal_assert(cond.defined()) << "Select with undefined condition";
    // TODO: if we allow Select in the frontend then we need to be able to not
    // perform this check?
    internal_assert(cond.type().defined() && cond.type().is_bool())
        << "Select with non-bool condition: " << cond;
    internal_assert(tvalue.defined() && fvalue.defined())
        << "Select with undefined operands: " << cond << " " << tvalue << " "
        << fvalue;

    Select *node = new Select;

    try_match_types(tvalue, fvalue);

    const bool infer_types =
        type_enforcement_enabled() ||
        (tvalue.type().defined() && fvalue.type().defined());

    if (infer_types) {
        internal_assert(equals(tvalue.type(), fvalue.type()))
            << "Select of mismatched types: " << tvalue << " versus " << fvalue;
        node->type = tvalue.type();
    }

    node->cond = std::move(cond);
    node->tvalue = std::move(tvalue);
    node->fvalue = std::move(fvalue);
    return node;
}

Expr Cast::make(Type type, Expr value, Mode mode) {
    internal_assert(type.defined())
        << "Cannot cast to undefined type: " << value;
    internal_assert(value.defined()) << "Cast of undefined value: " << type;

    Cast *node = new Cast;

    const bool infer_types =
        type_enforcement_enabled() || value.type().defined();
    if (infer_types) {
        // We allow Casts to be `Reinterpret`s
        // internal_assert(value.type().is_vector() == type.is_vector())
        //     << "Cannot remove vector type via cast: " << type << " from "
        //     << value;
        // internal_assert(!type.is_vector() ||
        //                 (type.lanes() == value.type().lanes()))
        //     << "Cannot change lanes via cast: " << type << " from " << value;
    }

    node->type = std::move(type);
    node->value = std::move(value);
    node->mode = mode;
    return node;
}

Expr Broadcast::make(uint32_t lanes, Expr value) {
    internal_assert(value.defined()) << "Broadcast of undefined.";

    Broadcast *node = new Broadcast;

    const bool infer_types =
        type_enforcement_enabled() || value.type().defined();
    if (infer_types) {
        // TODO: support?
        internal_assert(value.type().is_scalar())
            << "Broadcast of non-scalar: " << value;
        node->type = Vector_t::make(value.type(), lanes);
    }

    node->lanes = lanes;
    node->value = std::move(value);
    return node;
}

Expr VectorReduce::make(VectorReduce::OpType op, Expr value) {
    internal_assert(value.defined()) << "VectorReduce of undefined.";

    VectorReduce *node = new VectorReduce;

    const bool infer_types =
        type_enforcement_enabled() || value.type().defined();
    if (infer_types) {
        internal_assert(value.type().is_vector() || value.type().is<Array_t>())
            << "VectorReduce of non-vector: " << value;
        if (op == VectorReduce::Idxmin || op == VectorReduce::Idxmax) {
            internal_assert(value.type().element_of().is_scalar())
                << "TODO: type for argmin/argmax of nested vector?: " << value;
            node->type = UInt_t::make(32);
            // TODO: what if it's not uint32?
        } else {
            node->type = value.type().element_of();
        }
    }

    node->op = op;
    node->value = std::move(value);
    return node;
}

Expr VectorShuffle::make(Expr value, std::vector<Expr> idxs) {
    internal_assert(value.defined()) << "VectorShuffle of undefined.";
    internal_assert(std::all_of(idxs.cbegin(), idxs.cend(), [](const auto &e) {
        return e.defined();
    })) << "VectorShuffle with undefined idxs";

    VectorShuffle *node = new VectorShuffle;

    const bool infer_types =
        type_enforcement_enabled() || value.type().defined();
    if (infer_types) {
        internal_assert(value.type().is_vector())
            << "VectorShuffle of non-vector: " << value;
        internal_assert(std::all_of(idxs.cbegin(), idxs.cend(),
                                    [](const auto &e) {
                                        return e.type().defined() &&
                                               (e.type().is_int_or_uint());
                                    }))
            << "Vector Shuffle with undefined index types, of value: " << value
            << " on "
            << std::accumulate(std::next(idxs.begin()), idxs.end(),
                               to_string(idxs[0]) + " : " +
                                   to_string(idxs[0].type()),
                               [](const std::string &a, const Expr &b) {
                                   return a + ", " + to_string(b) + " : " +
                                          to_string(b.type());
                               });
        node->type = Vector_t::make(value.type().element_of(), idxs.size());
    }

    node->value = std::move(value);
    node->idxs = std::move(idxs);
    return node;
}

Expr Ramp::make(Expr base, Expr stride, int lanes) {
    internal_assert(base.defined() && stride.defined() && lanes > 1)
        << "Bad Ramp parameters: " << base << " " << stride << " " << lanes;

    Ramp *node = new Ramp;

    const bool infer_types = type_enforcement_enabled() ||
                             (stride.type().defined() && base.type().defined());
    if (infer_types) {
        internal_assert(equals(stride.type(), base.type()))
            << "Ramp of mismatched types: " << base << " " << stride << " "
            << lanes;
        node->type = Vector_t::make(base.type(), lanes);
    }

    node->base = std::move(base);
    node->stride = std::move(stride);
    node->lanes = lanes;
    return node;
}

Expr Extract::make(Expr vec, int32_t idx) {
    internal_assert(vec.defined()) << "Extract of undefined vector";
    internal_assert(idx >= 0)
        << "Extract with negative idx of: " << idx << " for " << vec;

    Extract *node = new Extract;

    const bool infer_types = type_enforcement_enabled() || vec.type().defined();
    if (infer_types) {
        if (vec.type().is<Vector_t>()) {
            internal_assert(idx < vec.type().lanes())
                << "Constant integer Extract has OOB index on vector: " << vec
                << " idx: " << idx;
            node->type = vec.type().element_of();
        } else {
            const Tuple_t *tuple_t = vec.type().as<Tuple_t>();
            internal_assert(tuple_t)
                << "Constant integer Extract on non-tuple: " << vec.type();
            internal_assert(idx < tuple_t->etypes.size())
                << "Constant integer Extract has OOB index on tuple: " << vec
                << " idx: " << idx;
            node->type = tuple_t->etypes[idx];
        }
    }

    node->vec = std::move(vec);
    node->idx = idx;
    return node;
}

Expr Extract::make(Expr vec, Expr idx) {
    internal_assert(vec.defined()) << "Extract of undefined vector";
    internal_assert(idx.defined()) << "Extract with undefined idx of: " << vec;

    Type type;
    const bool infer_types = type_enforcement_enabled() || vec.type().defined();
    if (infer_types) {
        internal_assert((vec.type().is<Vector_t, Array_t, Tuple_t>()))
            << "Extract of non-vector: " << vec;
        internal_assert(idx.type().is_int_or_uint())
            << "Extract with non-integer index: " << idx;
        if (vec.type().is<Tuple_t>()) {
            internal_assert(is_const(idx))
                << "Extract on tuple with non-constant index: " << vec << "["
                << idx << "]";
            return Extract::make(std::move(vec), *as_const_int(idx));
        }
        type = vec.type().element_of();
    }

    Extract *node = new Extract;
    node->type = std::move(type);
    node->vec = std::move(vec);
    node->idx = std::move(idx);
    return node;
}

Expr Build::make(Type type, std::vector<Expr> values) {
    Build *node = new Build;
    const bool infer_types = type_enforcement_enabled();

    // Fill default values.
    if (const auto *struct_t = type.as<Struct_t>()) {
        const Struct_t::Map &fields = struct_t->fields;
        const size_t field_count = fields.size();
        const size_t value_count = values.size();
        if (!values.empty() && field_count != value_count) {
            const Struct_t::DefMap &defaults = struct_t->defaults;
            if (infer_types) {
                internal_assert(value_count + defaults.size() == field_count)
                    << "Build<Struct_t> of type: " << type << " received "
                    << value_count << " values, has " << defaults.size()
                    << " defaults, but " << field_count << " fields";
            }
            std::vector<Expr> filled_values(field_count);
            size_t value_i = 0;
            for (size_t i = 0; i < field_count; i++) {
                // TODO: perform constant casting here?
                if (defaults.contains(fields[i].name)) {
                    // No need to assert, the default should always be
                    // the correct type for the struct.
                    filled_values[i] = defaults.at(fields[i].name);
                } else {
                    internal_assert(value_i < values.size())
                        << value_i << " < " << values.size();
                    if (infer_types) {
                        internal_assert(
                            equals(fields[i].type, values[value_i].type()))
                            << "Build<Struct_t> of type: " << type
                            << " requires matching field types, expected: "
                            << fields[i].type << " but received "
                            << values[value_i] << " of type "
                            << values[value_i].type() << " for field "
                            << fields[i].name;
                    }
                    filled_values[i] = values[value_i++];
                }
            }
            values = std::move(filled_values);
        }
    }

    if (infer_types) {
        internal_assert(type.defined())
            << "Build received empty type: " << type;
        // TODO: if values.empty() or values.size() < expected_n, then assert
        // type has defaults!
        for (const auto &expr : values) {
            internal_assert(expr.defined())
                << "Build with undefined field of type: " << type;
        }

        if (type.is<Vector_t>()) {
            internal_assert(values.empty() ||
                            type.as<Vector_t>()->lanes == values.size())
                << "Build<Vector_t> with incorrect number of arguments, "
                   "expected: "
                << type << " but received " << values.size() << " elements.";
            Type etype = type.as<Vector_t>()->etype;
            for (const auto &expr : values) {
                internal_assert(equals(expr.type(), etype))
                    << "Build<Vector_t> requires uniform element type, "
                       "expected: "
                    << etype << " but received " << expr;
            }
        } else if (type.is<Struct_t>()) {
            if (!values.empty()) {
                const auto &fields = type.as<Struct_t>()->fields;
                const size_t field_count = fields.size();
                const size_t value_count = values.size();
                internal_assert(value_count <= field_count)
                    << "Build<Struct_t> of type: " << type
                    << " received too many arguments, received: " << value_count
                    << " but expected " << field_count;
                if (field_count == value_count) {
                    for (size_t i = 0; i < values.size(); i++) {
                        const Type &ftype = fields[i].type;
                        const Type &vtype = values[i].type();
                        internal_assert(
                            equals(ftype, vtype) ||
                            is_single_element_struct_with_type(ftype, vtype))
                            << "Build<Struct_t> requires matching field types, "
                               "expected: "
                            << ftype << " but received " << values[i]
                            << " of type " << vtype << " for field "
                            << fields[i].name << " in struct " << type;
                    }
                }
            }
        } else if (type.is<Option_t>()) {
            if (!values.empty()) {
                internal_assert(
                    values.size() == 1 &&
                    equals(type.as<Option_t>()->etype, values[0].type()))
                    << "Cannot build option type: " << type
                    << " from base: " << values[0];
            }
        } else if (const Tuple_t *as_tuple = type.as<Tuple_t>()) {
            if (!values.empty()) {
                internal_assert(values.size() == as_tuple->etypes.size())
                    << "Incorrect number of arguments to tuple construction: "
                    << type << " takes " << as_tuple->etypes.size()
                    << " elements"
                    << " but received " << values.size();

                for (size_t i = 0; i < values.size(); i++) {
                    internal_assert(
                        equals(as_tuple->etypes[i], values[i].type()))
                        << "Build<Tuple_t> requires matching field types, "
                        << "expected: " << as_tuple->etypes[i]
                        << " but received " << values[i] << " of type "
                        << values[i].type() << " for index: " << i;
                }
            }
        } else if (const Array_t *as_array = type.as<Array_t>()) {
            internal_assert(!values.empty())
                << "Cannot Build Array_t from nothing, use Allocate instead.";

            const int64_t *const_size = as_const_int(as_array->size);
            internal_assert(const_size && values.size() == *const_size)
                << "Incorrect number of arguments to array construction: "
                << type << " takes " << *const_size << " elements"
                << " but received " << values.size();

            for (size_t i = 0; i < values.size(); i++) {
                internal_assert(equals(as_array->etype, values[i].type()))
                    << "Build<Array_t> requires matching field types, "
                    << "expected: " << as_array->etype << " but received "
                    << values[i] << " of type " << values[i].type()
                    << " for index: " << i;
            }
        } else {
            internal_error << "Build::make with non-(vector, array, struct, "
                              "option, tuple) type: "
                           << type;
        }
    }

    node->type = std::move(type);
    node->values = std::move(values);
    return node;
}

Expr Build::make(Type type, std::map<std::string, Expr> values) {
    internal_assert(type.is<Struct_t>())
        << "Cannot build with named fields for non-struct: " << type;

    internal_assert(!values.empty())
        << "Cannot build with named fields without any fields for type: "
        << type;

    // Always do type inference, we have enough information here.
    const auto &fields = type.as<Struct_t>()->fields;
    const auto &defaults = type.as<Struct_t>()->defaults;

    std::vector<Expr> args;

    for (const auto &field : fields) {
        internal_assert(values.contains(field.name) ||
                        defaults.contains(field.name))
            << "Construction of type: " << type << " has no value for field "
            << field.name << " in constructor";
        Expr value = values.contains(field.name) ? values.at(field.name)
                                                 : defaults.at(field.name);
        internal_assert(value.defined());
        if (!equals(value.type(), field.type)) {
            value = cast_to(field.type, std::move(value));
            internal_assert(value.defined());
        }
        args.emplace_back(std::move(value));
    }

    Build *node = new Build;
    node->type = std::move(type);
    node->values = std::move(args);
    return node;
}

Expr Build::make(Type type) {
    Build *node = new Build;
    node->type = std::move(type);
    return node;
}

Expr Access::make(std::string field, Expr value) {
    internal_assert(!field.empty() && value.defined())
        << "Bad Access parameters: " << field << " on " << value;

    Access *node = new Access;

    const bool infer_types =
        type_enforcement_enabled() || value.type().defined();
    if (infer_types) {
        node->type = get_field_type(value.type(), field);
    }

    node->field = std::move(field);
    node->value = std::move(value);
    return node;
}

Expr Unwrap::make(size_t index, Expr value) {
    internal_assert(value.defined() && value.type().is<BVH_t>())
        << "Bad Unwrap parameters: " << value;
    internal_assert(index < value.type().as<BVH_t>()->nodes.size())
        << "Bad Unwrap parameters: " << value << " unwrapped with " << index;

    Unwrap *node = new Unwrap;

    Type type = value.type().as<BVH_t>()->nodes[index].struct_type;
    internal_assert(type.defined());

    node->type = std::move(type);
    node->index = index;
    node->value = std::move(value);
    return node;
}

Expr Intrinsic::make(OpType op, std::vector<Expr> args) {
    internal_assert(op == OpType::rand ||
                    (!args.empty() && std::all_of(args.cbegin(), args.cend(),
                                                  [](const auto &arg) {
                                                      return arg.defined();
                                                  })))
        << "Intrinsic received undefined argument";

    Intrinsic *node = new Intrinsic;

    const bool infer_types =
        type_enforcement_enabled() ||
        std::all_of(args.cbegin(), args.cend(),
                    [](const auto &arg) { return arg.type().defined(); });
    // TODO:implement type enforcement for all intrinsics.
    if (infer_types) {
        switch (op) {
        case Intrinsic::abs: {
            internal_assert(args.size() == 1);
            if (args[0].type().is_int()) {
                node->type = args[0].type().to_uint();
            } else {
                node->type = args[0].type();
            }
            break;
        }
        case Intrinsic::dot: {
            internal_assert(args.size() == 2);
            internal_assert(equals(args[0].type(), args[1].type()));
            internal_assert(args[0].type().is<Vector_t>());
            node->type = args[0].type().element_of();
            break;
        }
        case Intrinsic::fma: {
            internal_assert(args.size() == 3);
            // Some parts can be broadcasted.
            try_match_types(args[0], args[1]);
            try_match_types(args[0], args[2]);
            try_match_types(args[1], args[2]);
            node->type = args[0].type();
            break;
        }
        case Intrinsic::norm: {
            internal_assert(args.size() == 1);
            internal_assert(args[0].type().is<Vector_t>());
            node->type = args[0].type().element_of();
            break;
        }
        case Intrinsic::rand: {
            internal_assert(args.size() <= 1);
            if (args.size() == 0) {
                node->type = Float_t::make_f32();
            } else {
                auto as_const = get_constant_value(args[0]);
                internal_assert(as_const.has_value() && as_const <= 1024)
                    << "Expected small constant value for rand(): " << args[0];
                node->type = Vector_t::make(Float_t::make_f32(), *as_const);
            }
            break;
        }
        case Intrinsic::min:
        case Intrinsic::max: {
            internal_assert(args.size() == 2);
            try_match_types(args[0], args[1]);
            node->type = args[0].type();
            break;
        }
        default: {
            internal_assert(!args.empty());
            node->type = args[0].type();
            break;
        }
        }
    }
    node->op = op;
    node->args = std::move(args);
    return node;
}

Expr Generator::make(OpType op, std::vector<Expr> args) {
    internal_assert(!args.empty() &&
                    std::all_of(args.cbegin(), args.cend(),
                                [](const auto &arg) { return arg.defined(); }))
        << "Generator received undefined argument";

    Generator *node = new Generator;

    const bool infer_types =
        type_enforcement_enabled() ||
        std::all_of(args.cbegin(), args.cend(),
                    [](const auto &arg) { return arg.type().defined(); });
    // TODO: implement type enforcement for all intrinsics.
    if (infer_types) {
        switch (op) {
        case Generator::iter: {
            internal_assert(args.size() == 1)
                << "iter takes 1 argument, received: " << args.size();
            internal_assert(args[0].type().defined() &&
                            args[0].type().is_int_or_uint())
                << "iter() expects argument to be an integer type: " << args[0]
                << " is " << args[0].type();
            node->type = Array_t::make(args[0].type(), args[0]);
            break;
        }
        case Generator::range: {
            internal_assert(args.size() == 3)
                << "range takes 3 arguments, received: " << args.size();
            internal_assert(args[0].type().defined() &&
                            args[0].type().is<Array_t>())
                << "range() expects first argument to be an array "
                   "type: "
                << args[0] << " is " << args[0].type();
            internal_assert(args[1].type().defined() &&
                            args[1].type().is_int_or_uint())
                << "range() expects second argument to be an integer "
                   "type: "
                << args[1] << " is " << args[1].type();
            internal_assert(args[2].type().defined() &&
                            args[2].type().is_int_or_uint())
                << "range() expects third argument to be an integer "
                   "type: "
                << args[2] << " is " << args[2].type();
            internal_assert(equals(args[1].type(), args[2].type()))
                << "range() expects second and third arguments to be "
                   "same type "
                << "arg1: " << args[1] << " is " << args[1].type()
                << " arg2: " << args[2] << " is " << args[2].type();
            node->type = Array_t::make(args[0].type().element_of(), args[2]);
            break;
        }
        }
    }
    node->op = op;
    node->args = std::move(args);
    return node;
}

Expr Lambda::make(std::vector<TypedVar> args, Expr value) {
    internal_assert(value.defined()) << "Lambda::make received undefined value";
    for (const auto &arg : args) {
        internal_assert(!arg.name.empty())
            << "Lambda::make received empty arg name";
        internal_assert(!type_enforcement_enabled() || arg.type.defined())
            << "Lambda::make received undefined arg type: " << arg.name;
    }

    Lambda *node = new Lambda;

    // Only do type inference if it's enabled or both types are defined.
    const bool infer_types =
        type_enforcement_enabled() ||
        (value.type().defined() &&
         std::all_of(args.cbegin(), args.cend(),
                     [](const auto &arg) { return arg.type.defined(); }));
    if (infer_types) {
        // TODO: assert that the vars are used?
        // or we can just optimize those out later, sometimes ppl write dumb
        // code.
        std::vector<Function_t::ArgSig> arg_types(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            arg_types[i].type = args[i].type;
            // TODO(ajr): do we ever need mutable lambda arguments?
            arg_types[i].is_mutable = false;
        }
        node->type = Function_t::make(value.type(), std::move(arg_types));
    }

    node->args = std::move(args);
    node->value = std::move(value);
    return node;
}

Expr GeomOp::make(OpType op, Expr a, Expr b) {
    internal_assert(a.defined() && b.defined())
        << "GeomOp::make received undefined value: " << to_string(op) << " "
        << a << " " << b;
    GeomOp *node = new GeomOp;

    const bool infer_types = type_enforcement_enabled() ||
                             (a.type().defined() && b.type().defined());
    if (infer_types) {
        // TODO: assert that these are volumes with defined geometric
        // constructs?
        const Struct_t *sa = a.type().as<Struct_t>();
        const Struct_t *sb = b.type().as<Struct_t>();
        internal_assert(sa && sb)
            << "GeomOp::make expected geometric structs: " << to_string(op)
            << " " << a << " " << b;

        Type ret_type;
        if (op == GeomOp::distmin || op == GeomOp::distmax) {
            // TODO: distance could have any value, it's user-defined...
            // TODO: do we need Real_t?
            // For now, just assume f32
            node->type = Float_t::make_f32();
        } else {
            node->type = Bool_t::make();
        }
    }

    node->op = op;
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

namespace {

const char *const geometric_op_names[] = {
    "contains",
    "distmax",
    "distmin",
    "intersects",
};

static_assert(sizeof(geometric_op_names) / sizeof(geometric_op_names[0]) ==
                  GeomOp::opcount,
              "geometric_op_names needs attention");

} // namespace

const char *GeomOp::intrinsic_name(const OpType &op) {
    return geometric_op_names[op];
}

Expr SetOp::make(OpType op, Expr a, Expr b) {
    internal_assert(a.defined() && b.defined())
        << "SetOp::make received undefined value: " << to_string(op) << " " << a
        << " " << b;
    SetOp *node = new SetOp;

    // Only do type inference if it's enabled or both types are defined.
    const bool infer_types = type_enforcement_enabled() ||
                             (a.type().defined() && b.type().defined());

    if (infer_types) {
        if (op == SetOp::filter) {
            internal_assert(a.type().is<Function_t>() &&
                            a.type().as<Function_t>()->ret_type.is_bool())
                << "Expected lhs of filter to be a boolean function, instead "
                   "received: "
                << a << " : " << a.type();
            internal_assert(b.type().is<Set_t>() || b.type().is<BVH_t>())
                << "Expected rhs of filter to be a set, instead received: " << b
                << " : " << b.type();
            const Function_t *f = a.type().as<Function_t>();
            if (f->arg_types.size() == 1) {
                internal_assert(
                    equals(f->arg_types[0].type, b.type().element_of()))
                    << "Expected filter function to accept element of type: "
                    << b.type().element_of() << " instead got " << a << " : "
                    << a.type();
            } else {
                internal_assert(
                    b.type().element_of().is<Tuple_t>() &&
                    b.type().element_of().as<Tuple_t>()->etypes.size() ==
                        f->arg_types.size())
                    << "Expected filter function to accept elements of group: "
                    << b.type().element_of() << " instead got " << a << " : "
                    << a.type();
            }
            node->type = b.type();
        } else if (op == SetOp::argmin) {
            internal_assert(a.type().is<Function_t>() &&
                            a.type().as<Function_t>()->ret_type.is_numeric())
                << "Expected lhs of argmin to be a numeric function, instead "
                   "received: "
                << a << " : " << a.type();
            internal_assert(b.type().is<Set_t>() || b.type().is<BVH_t>())
                << "Expected rhs of argmin to be a set, instead received: " << b
                << " : " << b.type();
            const Function_t *f = a.type().as<Function_t>();
            internal_assert(f->arg_types.size() == 1 &&
                            equals(f->arg_types[0].type, b.type().element_of()))
                << "Expected argmin function to accept element of type: "
                << b.type().element_of() << " instead got " << a << " : "
                << a.type();
            if (can_be_empty(b)) {
                node->type = Option_t::make(b.type().element_of());
            } else {
                node->type = b.type().element_of();
            }
        } else if (op == SetOp::map) {
            internal_assert(a.type().is<Function_t>())
                << "Expected lhs of map to be afunction, instead received: "
                << a << " : " << a.type();
            internal_assert((b.type().is<Set_t, Array_t>()))
                << "Expected rhs of map to be a set or array, instead "
                   "received: "
                << b << " : " << b.type();
            const Function_t *f = a.type().as<Function_t>();
            internal_assert(f->arg_types.size() == 1 &&
                            equals(f->arg_types[0].type, b.type().element_of()))
                << "Expected map function to accept element of type: "
                << b.type().element_of() << " instead got " << a << " : "
                << a.type();
            if (b.type().is<Set_t>()) {
                node->type = Set_t::make(f->ret_type);
            } else {
                Expr size = b.type().as<Array_t>()->size;
                node->type = Array_t::make(f->ret_type, std::move(size));
            }
        } else if (op == SetOp::product) {
            internal_assert(a.type().is<Set_t>() && b.type().is<Set_t>())
                << "Expected args of product to be sets, instead received: "
                << a << " : " << a.type() << " and " << b << " : " << b.type();
            Type atype = a.type().element_of();
            Type btype = b.type().element_of();
            Type tuple_t = Tuple_t::make({std::move(atype), std::move(btype)});
            node->type = Set_t::make(std::move(tuple_t));
        }
    }

    node->op = op;
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Expr Call::make(Expr func, std::vector<Expr> args) {
    internal_assert(func.defined()) << "Call::make received undefined func";
    internal_assert(std::all_of(args.cbegin(), args.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "Call::make received undefined arg to func: " << func;

    Call *node = new Call;

    const bool infer_types =
        type_enforcement_enabled() || func.type().defined();

    if (infer_types) {
        internal_assert(func.type().defined())
            << "Call::make needs func to have a defined type: " << func;
        internal_assert(func.type().is<Function_t>())
            << "Call::make received non-callable func: " << func;
        const Function_t *f = func.type().as<Function_t>();
        internal_assert(f->arg_types.size() == args.size())
            << "Call::make received incorrect number of arguments to: " << func
            << ", expected: " << f->arg_types.size()
            << " but received: " << args.size();
        for (size_t i = 0; i < args.size(); i++) {
            internal_assert(f->arg_types[i].type.defined());
            if (!args[i].type().defined()) {
                internal_assert(is_const(args[i]))
                    << "Undefined type in function call for non-constant "
                       "expression: "
                    << args[i];
                args[i] = constant_cast(f->arg_types[i].type, args[i]);
            } else {
                internal_assert(equals(args[i].type(), f->arg_types[i].type))
                    << "Call::make received bad argument: " << args[i]
                    << " with type: " << args[i].type()
                    << " when expecting type: " << f->arg_types[i].type
                    << " at index " << i << " of call to func: " << func;
            }
            if (f->arg_types[i].is_mutable) {
                internal_assert(is_location_expr(args[i]))
                    << "Cannot pass non-mutable argument: " << args[i]
                    << " as mutable parameter\n";
            }
        }
        node->type = f->ret_type;
    }

    node->func = std::move(func);
    node->args = std::move(args);
    return node;
}

Expr Instantiate::make(Expr expr, TypeMap types) {
    internal_assert(expr.defined())
        << "Instantiate::make received undefined expr";
    internal_assert(
        std::all_of(types.cbegin(), types.cend(),
                    [](const auto &p) { return p.second.defined(); }))
        << "Instantiate::make received undefined type to expr: " << expr;

    Instantiate *node = new Instantiate;

    const bool infer_types =
        type_enforcement_enabled() || expr.type().defined();

    if (infer_types) {
        internal_assert(contains_generics(expr.type(), types))
            << "Instantiation does not contain generics: " << expr << " : "
            << expr.type();
        node->type = replace(types, expr.type());
    }

    node->expr = std::move(expr);
    node->types = std::move(types);
    return node;
}

Expr PtrTo::make(Expr expr) {
    internal_assert(expr.defined()) << "PtrTo::make received undefined expr";
    internal_assert(expr.type().defined())
        << "PtrTo::make received untyped expr: " << expr;

    if (const Deref *ref = expr.as<Deref>()) {
        return ref->expr;
    }

    PtrTo *node = new PtrTo;
    node->type = Ptr_t::make(expr.type());
    node->expr = std::move(expr);
    return node;
}

Expr Deref::make(Expr expr) {
    internal_assert(expr.defined()) << "Deref::make received undefined expr";
    internal_assert(expr.type().defined())
        << "Deref::make received untyped expr: " << expr;
    internal_assert(expr.type().is<Ptr_t>())
        << "Deref::make received non-ptr expr: " << expr
        << " has type: " << expr.type();

    if (const PtrTo *ptr = expr.as<PtrTo>()) {
        return ptr->expr;
    }

    Deref *node = new Deref;
    node->type = expr.type().as<Ptr_t>()->etype;
    node->expr = std::move(expr);
    return node;
}

} // namespace ir
} // namespace bonsai
