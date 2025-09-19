#pragma once

#include <variant>
#include <vector>

#include "Error.h"
#include "Expr.h"
#include "Type.h"

namespace bonsai {
namespace ir {

struct WriteLoc {
    ir::Expr b; // The base expression.
    Type type;  // type of current write.
    // A string implies a struct access, an expr must be an integer index.
    struct Cast {
        ir::Type type;
        ir::Cast::Mode mode;
    };
    std::vector<std::variant<std::string, Expr, Cast>> accesses;

    WriteLoc() {} // required for Accumulate::make to work.
    WriteLoc(ir::Expr e) : b(std::move(e)), type(b.type()) {
        internal_assert(b.defined());
    }
    WriteLoc(std::string b, Type base_type)
        : b(ir::Var::make(base_type, b)), type(base_type) {
        internal_assert(!b.empty()) << "Write location with empty base";
    }

    bool defined() const { return b.defined(); }

    ir::Expr to_expr() const;

    // Returns a unique "name" for this write location.
    std::string name() const;

    const std::string &base() const {
        const ir::Var *v = b.as<ir::Var>();
        if (const auto *d = b.as<ir::Deref>()) {
            v = d->expr.as<ir::Var>();
        }
        internal_assert(v);
        return v->name;
    }

    const ir::Type &base_type() const {
        const ir::Var *v = b.as<ir::Var>();
        if (const auto *d = b.as<ir::Deref>()) {
            v = d->expr.as<ir::Var>();
        }
        internal_assert(v);
        return v->type;
    }

    static WriteLoc from(ir::Expr e);

    // These append to `accesses` *AND* mutate type (if set).
    void add_struct_access(const std::string &field);
    void add_index_access(const Expr &index);
    void add_cast(const ir::Type &type, ir::Cast::Mode mode);

    // After type inference, re-build with a defined base type.
    WriteLoc rebuild_with_base_type(Type _type) const;
};

} // namespace ir
} // namespace bonsai
