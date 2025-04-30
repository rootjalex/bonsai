#include "Utils.h"

#include "IR/Equality.h"
#include "IR/Operators.h"

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
        return b->values.empty(); // default is constant!
    }
    return e.is<IntImm, UIntImm, FloatImm, BoolImm, Infinity, VecImm>();
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

Expr constant_cast(const Type &t, const Expr &e) {
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
    } else if (e.is<Broadcast>()) {
        return constant_cast(t, e.as<Broadcast>()->value);
    } else {
        internal_error << "Unsure how to convert constant to type: " << t
                       << " expr: " << e;
    }
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

} // namespace

Expr replace(const std::map<std::string, Expr> &repls, const Expr &orig) {
    VarReplacer replacer(repls);
    return replacer.mutate(orig);
}

Stmt replace(const std::map<std::string, Expr> &repls, const Stmt &orig) {
    VarReplacer replacer(repls);
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


uint64_t bit_mask(int64_t n) {
    const uint64_t width = std::numeric_limits<uint64_t>::digits;
    internal_assert(0 < n && n <= 64) << n;
    return n >= width ? ~uint64_t{0} : (uint64_t{1} << n) - uint64_t{1};
}
  
ir::Expr update_type(ir::Expr expr, ir::Type type) {
    internal_assert(type.defined());
    internal_assert(expr.defined());
    switch (expr->node_type) {
    case ir::IRExprEnum::Build: {
        const auto *build = expr.as<ir::Build>();
        return ir::Build::make(std::move(type), build->values);
    }
    case ir::IRExprEnum::Var: {
        const auto *var = expr.as<ir::Var>();
        return ir::Var::make(std::move(type), var->name);
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

} // namespace bonsai
