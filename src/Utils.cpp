#include "Utils.h"

#include "IR/Equality.h"
#include "IR/Printer.h"

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
        return false;
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

bool is_const(const Expr &e) {
    if (!e.defined()) {
        internal_error << "is_const called on undefined value";
        return false;
    } else if (const Broadcast *b = e.as<Broadcast>()) {
        return is_const(b->value);
    } else if (const Build *b = e.as<Build>()) {
        return b->values.empty(); // default is constant!
    } else {
        return e.is<IntImm>() || e.is<UIntImm>() || e.is<FloatImm>() ||
               e.is<BoolImm>();
    }
}

Expr make_zero(const Type &t) { return make_const(t, 0); }

Expr make_one(const Type &t) { return make_const(t, 1); }

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
        return Expr();
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
        Replacer(const std::string &_var_name, Expr _repl)
            : var_name(_var_name), repl(std::move(_repl)) {}

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

Type replace(const std::map<std::string, Type> &repls, const Type &type) {
    struct Replacer : public Mutator {
        Replacer(const std::map<std::string, Type> &_repls) : repls(_repls) {}

      private:
        const std::map<std::string, Type> &repls;

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

    Replacer replacer(repls);
    return replacer.mutate(type);
}

bool is_power_of_two(int32_t x) { return (x & (x - 1)) == 0; }

int32_t next_power_of_two(int32_t x) {
    return static_cast<int32_t>(1)
           << static_cast<int32_t>(std::ceil(std::log2(x)));
}

size_t find_struct_index(const std::string &field,
                         const Struct_t::Map &fields) {
    for (size_t i = 0; i < fields.size(); i++) {
        if (field == fields[i].first) {
            return i;
        }
    }
    internal_error << "find_struct_index did not find field " << field;
    return 0;
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
    return -1;
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
        return 0.0;
    }
    }
}

} // namespace bonsai
