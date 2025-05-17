#include "Utils.h"

#include "IR/Equality.h"
#include "IR/Operators.h"

#include <cmath>
#include <limits>

namespace bonsai {

using namespace ir;

const int64_t *as_const_int(const Expr &e) {
    if (!e.defined()) {
        return nullptr;
    } else if (const Broadcast *b = e.as<Broadcast>()) {
        return as_const_int(b->value);
    } else if (const IntImm *i = e.as<IntImm>()) {
        return &(i->value);
    } else {
        return nullptr;
    }
}

bool is_const_one(const Expr &e) {
    if (!e.defined()) {
        internal_error << "is_const_one called on undefined value";
    } else if (const Broadcast *b = e.as<Broadcast>()) {
        return is_const_one(b->value);
    } else if (const IntImm *i = e.as<IntImm>()) {
        return i->value == 1;
    } else if (const UIntImm *u = e.as<UIntImm>()) {
        return u->value == 1;
    } else if (const FloatImm *f = e.as<FloatImm>()) {
        return f->value == 1.f;
    } else if (const BoolImm *b = e.as<BoolImm>()) {
        return b->value;
    } else {
        return false;
    }
}

bool is_const_zero(const Expr &e) {
    if (!e.defined()) {
        internal_error << "is_const_zero called on undefined value";
    } else if (const Broadcast *b = e.as<Broadcast>()) {
        return is_const_zero(b->value);
    } else if (const IntImm *i = e.as<IntImm>()) {
        return i->value == 0;
    } else if (const UIntImm *u = e.as<UIntImm>()) {
        return u->value == 0;
    } else if (const FloatImm *f = e.as<FloatImm>()) {
        return f->value == 0;
    } else if (const BoolImm *b = e.as<BoolImm>()) {
        return b->value == 0;
    } else {
        return false;
    }
}

bool is_const_all_ones(const Expr &e) {
    if (!e.defined()) {
        internal_error << "is_const_all_ones called on undefined value";
    }

    if (const Broadcast *b = e.as<Broadcast>()) {
        return is_const_all_ones(b->value);
    } else if (const IntImm *i = e.as<IntImm>()) {
        return is_all_ones(i->value, i->type.bits());
    } else if (const UIntImm *u = e.as<UIntImm>()) {
        return is_all_ones(u->value, u->type.bits());
    } else if (const FloatImm *f = e.as<FloatImm>()) {
        return is_all_ones(f->value, f->type.bits());
    } else if (const BoolImm *b = e.as<BoolImm>()) {
        return b->value == 1;
    }
    return false;
}

bool is_const(const Expr &e) {
    if (!e.defined()) {
        internal_error << "is_const called on undefined value";
    }
    if (const Broadcast *b = e.as<Broadcast>()) {
        return is_const(b->value);
    }
    if (const Build *b = e.as<Build>()) {
        return b->values.empty() ||
               std::all_of(b->values.begin(), b->values.end(),
                           [](const Expr &v) { return is_const(v); });
    }
    return e.is<IntImm, UIntImm, FloatImm, BoolImm, Infinity, VecImm>();
}

bool is_location_expr(const Expr &expr) {
    return expr.is<Var, Access, PtrTo>();
}

Expr get_value_at(Expr v, int64_t index) {
    const Type &type = v.type();
    internal_assert(type.is_vector()) << type;
    internal_assert(0 <= index && index < type.lanes())
        << index << " is not within bounds [0, " << type.lanes() << ")";
    if (const auto *broadcast = v.as<Broadcast>()) {
        return broadcast->value;
    }
    if (const auto *immediate = v.as<VecImm>()) {
        return immediate->values[index];
    }
    if (const auto *build = v.as<Build>()) {
        return build->values[index];
    }
    return Expr();
}

const SetOp *as_map(const Expr &expr) {
    if (const SetOp *setop = expr.as<SetOp>()) {
        if (setop->op == SetOp::map) {
            return setop;
        }
    }
    return nullptr;
}

const SetOp *as_filter(const Expr &expr) {
    if (const SetOp *setop = expr.as<SetOp>()) {
        if (setop->op == SetOp::filter) {
            return setop;
        }
    }
    return nullptr;
}

Expr make_zero(const Type &t) { return make_const(t, 0); }

Expr make_one(const Type &t) { return make_const(t, 1); }

Expr make_all_ones(const Type &t) { return make_const(t, bit_mask(t.bits())); }

Expr make_inf(const Type &t) {
    if (t.is<UInt_t, Int_t, Float_t>()) {
        return Infinity::make(t);
    }
    internal_error << "Unknown infinity for type: " << t;
}

Expr make_one_hot(Type t, Expr idx, size_t lanes) {
    std::vector<Expr> values(lanes);
    for (size_t i = 0; i < lanes; i++) {
        Expr lane = make_const(idx.type(), i);
        values[i] = Select::make(lane == idx, make_one(t), make_zero(t));
    }
    return Build::make(Vector_t::make(t, lanes), std::move(values));
}

Expr make_tuple(std::vector<Expr> exprs) {
    std::vector<Type> etypes;
    etypes.reserve(exprs.size());
    for (const auto &e : exprs) {
        etypes.push_back(e.type());
    }
    Type tuple_t = Tuple_t::make(std::move(etypes));
    return ir::Build::make(std::move(tuple_t), std::move(exprs));
}

std::vector<Expr> break_tuple(Expr expr) {
    // TODO(ajr): this may someday need to handle expr being a `Sort`
    const Build *build = expr.as<Build>();
    internal_assert(build && build->type.is<Tuple_t>())
        << "Expected Tuple build: " << expr;
    return build->values;
}

Expr constant_cast(const Type &t, const Expr &e) {
    if (equals(e.type(), t)) {
        return e;
    }
    internal_assert(t.defined() && e.defined())
        << "received bad type conversion:" << e << " to " << t;
    internal_assert(is_const(e))
        << "expected constant, instead received: " << e;
    // TODO: can we have non-scalar constants? parser doesn't support that yet,
    // but it should! not sure how to assert that, since e shouldn't have a type
    // right now.
    // TODO: also support other type conversions?
    // TODO: should we issue warnings when performing lossy conversions?
    if (e.is<IntImm>()) {
        return make_const(t, e.as<IntImm>()->value);
    } else if (e.is<UIntImm>()) {
        return make_const(t, e.as<UIntImm>()->value);
    } else if (e.is<FloatImm>()) {
        return make_const(t, e.as<FloatImm>()->value);
    } else if (e.is<BoolImm>()) {
        return make_const(t, e.as<BoolImm>()->value);
    } else if (e.is<Build>() && e.as<Build>()->values.empty()) {
        static const std::vector<Expr> empty = {};
        return Build::make(t, empty);
    } else if (const Build *build = e.as<Build>()) {
        if (const Tuple_t *tuple_t = t.as<Tuple_t>()) {
            // Propagate etypes.
            internal_assert(build->values.size() == tuple_t->etypes.size())
                << "Mismatched counts in constant_cast(" << t << ", " << e
                << ")";
            std::vector<Expr> args(build->values.size());
            for (size_t i = 0; i < build->values.size(); i++) {
                args[i] = constant_cast(tuple_t->etypes[i], build->values[i]);
            }
            return Build::make(std::move(t), std::move(args));
        } else if (const Vector_t *vector_t = t.as<Vector_t>()) {
            // Propagate etype.
            internal_assert(build->values.size() == vector_t->lanes)
                << "Mismatched counts in constant_cast(" << t << ", " << e
                << ")";
            std::vector<Expr> args(build->values.size());
            for (size_t i = 0; i < build->values.size(); i++) {
                args[i] = constant_cast(vector_t->etype, build->values[i]);
            }
            return Build::make(std::move(t), std::move(args));
        } else if (const Struct_t *struct_t = t.as<Struct_t>()) {
            internal_assert(build->values.size() == struct_t->fields.size())
                << "Mismatched counts in constant_cast(" << t << ", " << e
                << ")";
            std::vector<Expr> args(build->values.size());
            for (size_t i = 0; i < build->values.size(); i++) {
                args[i] =
                    constant_cast(struct_t->fields[i].type, build->values[i]);
            }
            return Build::make(std::move(t), std::move(args));
        }
    } else if (e.is<Broadcast>()) {
        return constant_cast(t, e.as<Broadcast>()->value);
    } else if (e.is<Infinity>()) {
        return Infinity::make(t);
    }
    internal_error << "Unsure how to convert constant to type: " << t
                   << " expr: " << e;
}

Expr cast_to(const Type &t, const Expr &e) {
    // TODO: merge with try_match_types?
    if (equals(t, e.type())) {
        return e;
    }
    if (is_const(e)) {
        return constant_cast(t, e);
    }
    internal_assert(e.type().defined())
        << "Cannot cast untyped value: " << e << " to type: " << t;
    if (t.is_vector() && e.type().is_scalar()) {
        Expr inner = cast_to(t.element_of(), e);
        return Broadcast::make(t.lanes(), std::move(inner));
    }
    if (t.is<Struct_t>() && t.as<Struct_t>()->fields.size() == 1 &&
        equals(t.as<Struct_t>()->fields[0].type, e.type())) {
        return Build::make(t, std::vector<Expr>{e});
    }
    return Cast::make(t, e);
}

Expr replace(const std::string &var_name, Expr repl, const Expr &orig) {
    struct Replacer : public Mutator {
        Replacer(const std::string &var_name, Expr repl)
            : var_name(var_name), repl(std::move(repl)) {}

      private:
        const std::string &var_name;
        Expr repl;

      public:
        Expr visit(const Var *node) override {
            if (node->name == var_name) {
                return repl;
            } else {
                return node;
            }
        }
    };

    Replacer replacer(var_name, std::move(repl));
    return replacer.mutate(orig);
}

namespace {

struct VarReplacer : public Mutator {
    VarReplacer(const std::map<std::string, Expr> &repls) : repls(repls) {}

  private:
    const std::map<std::string, Expr> &repls;

  public:
    Expr visit(const Var *node) override {
        const auto &iter = repls.find(node->name);
        if (iter != repls.cend()) {
            return iter->second;
        } else {
            return node;
        }
    }
};

struct VarNameReplacer : public Mutator {
    VarNameReplacer(const std::map<std::string, std::string> &repls)
        : repls(repls) {}

  private:
    const std::map<std::string, std::string> &repls;

  public:
    Expr visit(const Var *node) override {
        auto iter = repls.find(node->name);
        if (iter != repls.cend()) {
            return ir::Var::make(node->type, iter->second);
        } else {
            return node;
        }
    }
};

} // namespace

Expr replace(const std::map<std::string, Expr> &repls, const Expr &orig) {
    VarReplacer replacer(repls);
    return replacer.mutate(orig);
}

Stmt replace(const std::map<std::string, Expr> &repls, const Stmt &orig) {
    VarReplacer replacer(repls);
    return replacer.mutate(orig);
}

Stmt replace(const std::map<std::string, std::string> &repls,
             const Stmt &orig) {
    VarNameReplacer replacer(repls);
    return replacer.mutate(orig);
}

Type replace(const TypeMap &repls, const Type &type) {
    struct TypeReplacer : public Mutator {
        TypeReplacer(const TypeMap &repls) : repls(repls) {}

      private:
        const TypeMap &repls;

      public:
        Type visit(const Generic_t *node) override {
            // TODO: should node->name always be in repls?
            // if not, then we have some sort of nested
            // generics, I think? For now, disallow.
            internal_assert(repls.contains(node->name))
                << "Did not find replacement for generic: " << node->name;
            // TODO: recursively mutate...?
            return repls.at(node->name);
        }
    };

    TypeReplacer replacer(repls);
    return replacer.mutate(type);
}

Expr call(Expr func, Expr arg) {
    if (const Lambda *l = func.as<Lambda>()) {
        internal_assert(l->args.size() == 1)
            << "[unimplemented] call fusion for tuple set operations\n";
        return replace(l->args[0].name, arg, l->value);
    }
    return Call::make(std::move(func), {std::move(arg)});
}

bool is_power_of_two(int32_t x) { return (x & (x - 1)) == 0; }

int32_t next_power_of_two(int32_t x) {
    return static_cast<int32_t>(1)
           << static_cast<int32_t>(std::ceil(std::log2(x)));
}

size_t find_struct_index(const std::string &field,
                         const Struct_t::Map &fields) {
    for (size_t i = 0; i < fields.size(); i++) {
        if (field == fields[i].name) {
            return i;
        }
    }
    internal_error << "find_struct_index did not find field " << field;
}

uint32_t vector_field_lane(const std::string &field) {
    if (field == "x") {
        return 0;
    } else if (field == "y") {
        return 1;
    } else if (field == "z") {
        return 2;
    } else if (field == "w") {
        return 3;
    }
    internal_error << "Cannot get lane for vector field: " << field;
}

double machine_epsilon(const Type &t) {
    internal_assert(t.is_float())
        << "eps takes only floating point template types, instead received: "
        << t;
    switch (t.bits()) {
    case 32: {
        return std::numeric_limits<float>::epsilon() * 0.5;
    }
    case 64: {
        return std::numeric_limits<double>::epsilon() * 0.5;
    }
    default: {
        internal_error << "machine_epsilon() not supported for type: " << t;
    }
    }
}

// Convert an expression, e.g. `a.field0.field1` into a `WriteLoc`.
WriteLoc read_to_writeloc(const Expr &expr) {
    if (const Var *var = expr.as<Var>()) {
        return WriteLoc(var->name, var->type);
    } else if (const Access *acc = expr.as<Access>()) {
        WriteLoc rec = read_to_writeloc(acc->value);
        rec.add_struct_access(acc->field);
        return rec;
    } else if (const Extract *idx = expr.as<Extract>()) {
        WriteLoc rec = read_to_writeloc(idx->vec);
        rec.add_index_access(idx->idx);
        return rec;
    }
    internal_error << "Cannot convert to WriteLoc: " << expr;
    return WriteLoc();
}

bool is_writeloc(const Expr &expr) {
    if (expr.as<Var>()) {
        return true;
    } else if (const Access *acc = expr.as<Access>()) {
        return is_writeloc(acc->value);
    } else if (const Extract *idx = expr.as<Extract>()) {
        return is_writeloc(idx->vec);
    }
    return false;
}

uint64_t bit_mask(uint64_t n) {
    const uint64_t width = std::numeric_limits<uint64_t>::digits;
    internal_assert(0 < n && n <= 64) << n;
    return n >= width ? ~uint64_t{0} : (uint64_t{1} << n) - uint64_t{1};
}

Expr update_type(Expr expr, Type type) {
    internal_assert(type.defined());
    internal_assert(expr.defined());
    switch (expr->node_type_) {
    case IRExprEnum::Build: {
        const auto *build = expr.as<Build>();
        return Build::make(std::move(type), build->values);
    }
    case IRExprEnum::Var: {
        const auto *var = expr.as<Var>();
        return Var::make(std::move(type), var->name);
    }
    default:
        internal_error << "[unimplemented] update_type(" << expr << " : "
                       << expr.type() << ", " << type << ")";
    }
}

namespace {

Type flatten_array_type_helper(Type type, Expr size) {
    if (const Array_t *array_t = type.as<Array_t>()) {
        // TODO(ajr): might need to cast types of size/array_t->size
        return flatten_array_type_helper(array_t->etype, size * array_t->size);
    } else if (const Vector_t *vector_t = type.as<Vector_t>()) {
        return flatten_array_type_helper(vector_t->etype,
                                         size * vector_t->lanes);
    }
    return Array_t::make(std::move(type), std::move(size));
}

} // namespace

Type flatten_array_type(const Type &type) {
    if (const Array_t *nested = type.as<Array_t>()) {
        return flatten_array_type_helper(nested->etype, nested->size);
    }
    internal_error << "flatten_array_type called on non-Array_t: " << type;
}

std::string get_specifier(const ir::Type &type) {
    std::string specifier = "%";
    const uint32_t width = type.bits();
    if (type.is_bool()) {
        // Boolean values are printed as strings ("true", "false").
        return "%s";
    }
    if (!(type.is_numeric() && (width == 32 || width == 64))) {
        internal_error << "[unimplemented] print: " << type;
    }
    if (type.is_int()) {
        if (width > 32)
            return "%ld";
        return "%d";
    }
    if (type.is_uint()) {
        if (width > 32)
            return "%lu";
        return "%u";
    }
    if (type.is_float()) {
        // C will convert float (f32) to double (f64) for variadic
        // argument functions (to include printf).
        return "%f";
    }

    internal_error << "[unimplemented] print: " << type;
}

bool is_dynamic_array_struct_type(const ir::Type &type) {
    // Dynamic arrays are lowered to structs.
    if (const auto *dynamic_array_t = type.as<ir::Struct_t>()) {
        if (dynamic_array_t->name.starts_with("__dyn_array")) {
            return true;
        }
    }
    return false;
}

} // namespace bonsai
