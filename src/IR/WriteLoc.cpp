#include "IR/WriteLoc.h"

#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"

namespace bonsai {
namespace ir {

void WriteLoc::add_struct_access(const std::string &field) {
    internal_assert(!field.empty()) << "Write location made with empty field";
    accesses.push_back(field);
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the current type must be a struct with this field defined...
    const bool infer_types = type_enforcement_enabled() || type.defined();
    if (infer_types) {
        ir::Type field_type = get_field_type(type, field);
        internal_assert(field_type.defined())
            << "Write location type inference produced undefined type: "
            << field_type << " from field access " << field << " of type " << type;
        type = std::move(field_type);
    }
}

void WriteLoc::add_index_access(const Expr &index) {
    internal_assert(index.defined())
        << "Write location made with undefined index";
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the type of index must be an integer (signed or unsigned).
    internal_assert(!index.type().defined() || index.type().is_int_or_uint())
        << "Write location made with non-integer index: " << index;
    accesses.push_back(index);
    // TODO: if we were doing stronger type inference, we could add a constraint
    // that the current type must be a vector...
    const bool infer_types = type_enforcement_enabled() || type.defined();
    if (infer_types) {
        internal_assert(type.is<Vector_t>())
            << "Write location of non-vector received index: " << index
            << " but has type: " << type;
        ir::Type etype = type.element_of();
        internal_assert(etype.defined())
            << "Write location type inference produced undefined type: "
            << etype << " from index " << index << " of type " << type;
        type = std::move(etype);
    }
}

WriteLoc WriteLoc::rebuild_with_base_type(Type base_type) const {
    internal_assert(base_type.defined())
        << "Write location rebuild triggered with undefined type for base: "
        << base;
    internal_assert(type_enforcement_enabled())
        << "Write location rebuild triggered without type enforcement enabled";
    WriteLoc rebuilt(this->base, base_type);
    for (const auto &value : this->accesses) {
        if (std::holds_alternative<std::string>(value)) {
            rebuilt.add_struct_access(std::get<std::string>(value));
        } else {
            // holds Expr
            rebuilt.add_index_access(std::get<Expr>(value));
        }
    }
    return rebuilt;
}

} // namespace ir
} // namespace bonsai
