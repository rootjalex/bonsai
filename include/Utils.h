#pragma once

#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Type.h"

#include "Error.h"

#include <optional>

namespace bonsai {

const int64_t *as_const_int(const ir::Expr &e);
bool is_const_one(const ir::Expr &e);
bool is_const_all_ones(const ir::Expr &e);
bool is_const_zero(const ir::Expr &e);
bool is_const(const ir::Expr &e);
bool is_location_expr(const ir::Expr &expr);

// Attempts to infer the value at the given index in the vector `v`, otherwise
// returns an undefined expression upon failure.
ir::Expr get_value_at(ir::Expr v, int64_t index);

// Returns the unsigned bit representation of this expression. Defaults to
// interpreting this as a bit field. For vectors, an index value should be
// provided as well. For example,
//   ir::Expr e = IntImm::make(i64, -1);
//   assert(get_constant_value<int64_t>(e) == -1);
template <typename T = uint64_t>
std::optional<T> get_constant_value(const ir::Expr &e,
                                    std::optional<int64_t> index = {}) {
    if (!is_const(e)) {
        return {};
    }
    ir::Type type = e.type();
    if (type.is_vector()) {
        internal_assert(index.has_value()) << e;
        ir::Expr value = get_value_at(e, *index);
        return get_constant_value<T>(std::move(value));
    }
    if (type.is_scalar()) {
        internal_assert(!index.has_value())
            << "unexpected index provided: " << *index
            << "for scalar value: " << e;
        // Conservatively fail if the bit size is > 64.
        internal_assert(type.bits() <= 64) << type;
    }
    if (const auto *v = e.as<ir::UIntImm>()) {
        return std::bit_cast<T>(v->value);
    }
    if (const auto *v = e.as<ir::IntImm>()) {
        return std::bit_cast<T>(v->value);
    }
    if (const auto *v = e.as<ir::FloatImm>()) {
        return std::bit_cast<T>(v->value);
    }
    if (const auto *v = e.as<ir::BoolImm>()) {
        // Match the bit width of the other immediate values.
        uint64_t value = static_cast<uint64_t>(v->value);
        return std::bit_cast<T>(value);
    }

    if (e.is<ir::Infinity>()) {
        // Conservatively fail until we find use cases for this.
        return {};
    }

    internal_error << "[unimplemented] get_constant_value, " << e << " : "
                   << type;
}

const ir::SetOp *as_map(const ir::Expr &expr);
const ir::SetOp *as_filter(const ir::Expr &expr);

// Creates an immediate with value `0` and the provided type.
ir::Expr make_zero(const ir::Type &t);

// Creates an immediate with value `1` and the provided type.
ir::Expr make_one(const ir::Type &t);

// Create an immediate with value `inf` and the provided type.
ir::Expr make_inf(const ir::Type &t);

// Create an immediate with value `1` n times, where n is the type's bit width.
ir::Expr make_all_ones(const ir::Type &t);

template <typename T>
ir::Expr make_const(const ir::Type &t, const T &v) {
    if (t.is<ir::Int_t>()) {
        return ir::IntImm::make(t, (int64_t)v);
    } else if (t.is<ir::UInt_t>()) {
        return ir::UIntImm::make(t, (uint64_t)v);
    } else if (t.is<ir::Bool_t>()) {
        return ir::BoolImm::make((bool)v);
    } else if (t.is<ir::Float_t>()) {
        return ir::FloatImm::make(t, (double)v);
    } else if (t.is<ir::Vector_t>()) {
        ir::Expr r = make_const(t.as<ir::Vector_t>()->etype, v);
        return ir::Broadcast::make(t.as<ir::Vector_t>()->lanes, std::move(r));
    } else {
        internal_error
            << "make_const does not know how to build constant of type: " << t
            << " for value: " << v;
    }
}

ir::Expr constant_cast(const ir::Type &t, const ir::Expr &e);
// Handles broadcasting if necessary.
ir::Expr cast_to(const ir::Type &t, const ir::Expr &e);

ir::Expr replace(const std::string &var_name, ir::Expr repl,
                 const ir::Expr &orig);

ir::Expr replace(const std::map<std::string, ir::Expr> &repls,
                 const ir::Expr &orig);

ir::Stmt replace(const std::map<std::string, ir::Expr> &repls,
                 const ir::Stmt &orig);

ir::Type replace(const ir::TypeMap &repls, const ir::Type &type);

// Automatic fusion if `func` is a lambda, otherwise just makes a Call node.
ir::Expr call(ir::Expr func, ir::Expr arg);

bool is_power_of_two(int32_t x);
int32_t next_power_of_two(int32_t x);

size_t find_struct_index(const std::string &field,
                         const ir::Struct_t::Map &fields);

uint32_t vector_field_lane(const std::string &field);

// TODO: this should be handled in codegen...
double machine_epsilon(const ir::Type &t);

// Bit layout (not including sign bit) for floating point representations.
template <uint32_t E, uint32_t M>
struct FloatLayout {
    static constexpr uint32_t exponent = E;
    static constexpr uint32_t mantissa = M;
};

static constexpr auto IEEE754_F64 = FloatLayout<11, 52>{};
static constexpr auto IEEE754_F32 = FloatLayout<8, 23>{};
static constexpr auto IEEE754_F16 = FloatLayout<5, 10>{};
static constexpr auto BFLOAT16 = FloatLayout<8, 7>{};

// Convert an expression, e.g. `a.field0.field1` into a `WriteLoc`.
ir::WriteLoc read_to_writeloc(const ir::Expr &expr);
// Whether we can convert an expression into a WriteLoc.
bool is_writeloc(const ir::Expr &expr);

inline bool is_geometric_intrinsic(const std::string &name) {
    return (name == "contains") || (name == "distmin") || (name == "distmax") ||
           (name == "intersects");
}

inline bool is_geometric_predicate(const std::string &name) {
    return (name == "contains") || (name == "intersects");
}

inline bool is_geometric_metric(const std::string &name) {
    return (name == "distmin") || (name == "distmax");
}

// Returns a bit mask of size n.
uint64_t bit_mask(int64_t n);

// Returns whether `value` up to size `width` is all ones.
template <typename T>
bool is_all_ones(T value, int64_t width) {
    const uint64_t mask = bit_mask(width);
    return (std::bit_cast<uint64_t>(value) & mask) == mask;
}

// Updates the type of the passed in expression with the provided type.
ir::Expr update_type(ir::Expr, ir::Type);

ir::Type flatten_array_type(const ir::Type &type);

} // namespace bonsai
