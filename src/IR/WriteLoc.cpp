#include "IR/WriteLoc.h"

#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"

#include "Utils.h"

namespace bonsai {
namespace ir {
namespace {

void add_accesses(
    ir::WriteLoc &loc,
    const std::vector<std::variant<std::string, Expr, WriteLoc::Cast>>
        &accesses,
    int begin = 0) {
    int i = 0;
    for (const auto &value : accesses) {
        if (i++ < begin)
            continue;
        if (std::holds_alternative<std::string>(value)) {
            loc.add_struct_access(std::get<std::string>(value));
            continue;
        }
        if (std::holds_alternative<Expr>(value)) {
            // holds Expr
            loc.add_index_access(std::get<Expr>(value));
            continue;
        }
        if (std::holds_alternative<WriteLoc::Cast>(value)) {
            auto cast = std::get<WriteLoc::Cast>(value);
            loc.add_cast(cast.type, cast.mode);
            continue;
        }
        internal_error << "[unexpected] variant in WriteLoc";
    }
}

// Collects a list of accesses and finally returns the base expression.
ir::Expr convert(
    ir::Expr e,
    std::vector<std::variant<std::string, Expr, WriteLoc::Cast>> &accesses) {
    if (const auto *ex = e.as<ir::Extract>()) {
        accesses.push_back(ex->idx);
        return convert(ex->vec, accesses);
    }
    if (const auto *ac = e.as<ir::Access>()) {
        accesses.push_back(ac->field);
        return convert(ac->value, accesses);
    }
    if (const auto *ca = e.as<ir::Cast>()) {
        accesses.push_back(WriteLoc::Cast{.type = ca->type, .mode = ca->mode});
        return convert(ca->value, accesses);
    }
    return e;
}

} // namespace

void WriteLoc::add_struct_access(const std::string &field) {
    internal_assert(!field.empty()) << "Write location made with empty field";
    accesses.push_back(field);
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the current type must be a struct with this field defined...
    const bool infer_types = type_enforcement_enabled() || type.defined();
    if (infer_types) {
        ir::Type _type = get_field_type(type, field);
        internal_assert(_type.defined())
            << "Write location type inference produced undefined type: "
            << _type << " from field access " << field << " of type " << type;
        type = std::move(_type);
    }
}

void WriteLoc::add_index_access(const Expr &index) {
    internal_assert(index.defined())
        << "Write location made with undefined index";
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the type of index must be an integer (signed or unsigned).
    internal_assert(!index.type().defined() || index.type().is_int_or_uint())
        << "Write location made with non-integer index: " << index << " : "
        << index.type();
    accesses.push_back(index);
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the current type must be a vector...
    const bool infer_types = type_enforcement_enabled() || type.defined();
    if (infer_types) {
        const bool indexable =
            type.is<Vector_t, Array_t, DynArray_t, Tuple_t>();
        internal_assert(indexable)
            << "Write location of non-vector: `" << *this
            << "` received index: " << index << " but has type: " << type;
        ir::Type etype;
        if (type.is<Vector_t, Array_t, DynArray_t>()) {
            etype = type.element_of();
        } else {
            const Tuple_t *tuple_t = type.as<Tuple_t>();
            internal_assert(tuple_t);
            auto cvalue = get_constant_value(index);
            internal_assert(cvalue.has_value())
                << "Cannot write to Tuple at variable index: " << index;
            internal_assert(*cvalue < tuple_t->etypes.size())
                << "Cannot write to Tuple at OOB index: " << index;
            etype = tuple_t->etypes[*cvalue];
        }
        internal_assert(etype.defined())
            << "Write location type inference produced undefined type: "
            << etype << " from index " << index << " of type " << type;
        type = std::move(etype);
    }
}

void WriteLoc::add_cast(const ir::Type &type, ir::Cast::Mode mode) {
    internal_assert(type.defined());
    accesses.push_back(WriteLoc::Cast{.type = type, .mode = mode});
}

WriteLoc WriteLoc::rebuild_with_base_type(Type _type) const {
    internal_assert(_type.defined())
        << "Write location rebuild triggered with undefined type for base: "
        << base();
    internal_assert(type_enforcement_enabled())
        << "Write location rebuild triggered without type enforcement enabled";
    WriteLoc rebuilt(this->b);
    add_accesses(rebuilt, this->accesses);
    return rebuilt;
}

WriteLoc WriteLoc::pop_base(std::string name, Type type) const {
    internal_assert(!this->accesses.empty()) << *this;
    ir::WriteLoc rebuilt(name, std::move(type));
    add_accesses(rebuilt, this->accesses, /*begin=*/1);
    return rebuilt;
}

ir::Expr WriteLoc::to_expr() const {
    ir::Expr expr = b;
    for (const auto &value : this->accesses) {
        if (std::holds_alternative<std::string>(value)) {
            expr = Access::make(std::get<std::string>(value), expr);
            continue;
        }
        if (std::holds_alternative<Expr>(value)) {
            expr = Extract::make(expr, std::get<ir::Expr>(value));
            continue;
        }
        if (std::holds_alternative<WriteLoc::Cast>(value)) {
            auto cast = std::get<WriteLoc::Cast>(value);
            expr = ir::Cast::make(cast.type, expr, cast.mode);
            continue;
        }
        internal_error << "[unexpected] variant in WriteLoc";
    }
    return expr;
}

/* static */ WriteLoc WriteLoc::from(ir::Expr e) {
    std::vector<std::variant<std::string, Expr, WriteLoc::Cast>> accesses;
    ir::Expr base = convert(e, accesses);
    std::reverse(accesses.begin(), accesses.end());

    const auto *v = base.as<ir::Var>();
    internal_assert(v) << "[unimplemented] non-variable base in "
                          "`WriteLoc::from` conversion of: "
                       << e;
    WriteLoc location(v->name, v->type);
    location.accesses = std::move(accesses);
    return location;
}

std::string WriteLoc::name() const {
    std::stringstream os;
    ir::Printer printer(os);
    printer.print(to_expr());
    return os.str();
}

} // namespace ir
} // namespace bonsai
