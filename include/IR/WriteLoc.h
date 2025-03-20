#pragma once

#include <variant>
#include <vector>

#include "Error.h"
#include "Expr.h"
#include "Type.h"

namespace bonsai {
namespace ir {

struct WriteLoc {
    Type base_type;
    Type type;        // type of current write.
    std::string base; // name
    // A string implies a struct access, an expr must be an integer index.
    std::vector<std::variant<std::string, Expr>> accesses;

    WriteLoc() {} // required for Accumulate::make to work.
    WriteLoc(std::string b, Type base_type)
        : base_type(base_type), type(base_type), base(std::move(b)) {
        internal_assert(!base.empty()) << "Write location with empty base";
    }

    bool defined() const { return !base.empty(); }

    // These append to `accesses` *AND* mutate type (if set).
    void add_struct_access(const std::string &field);
    void add_index_access(const Expr &index);

    // After type inference, re-build with a defined base type.
    WriteLoc rebuild_with_base_type(Type _type) const;
};

} // namespace ir
} // namespace bonsai
