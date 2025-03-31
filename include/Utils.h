#pragma once

#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Type.h"

#include "Error.h"

namespace bonsai {

const int64_t *as_const_int(const ir::Expr &e);
bool is_const_one(const ir::Expr &e);
bool is_const(const ir::Expr &e);

ir::Expr make_zero(const ir::Type &t);
ir::Expr make_one(const ir::Type &t);

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

ir::Type replace(const ir::TypeMap &repls, const ir::Type &type);

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
    return (name == "contains") || (name == "distance") ||
           (name == "intersects");
}

inline bool is_geometric_predicate(const std::string &name) {
    return (name == "contains") || (name == "intersects");
}

inline bool is_geometric_metric(const std::string &name) {
    return (name == "distance");
}

} // namespace bonsai
