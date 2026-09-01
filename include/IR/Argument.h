#pragma once

#include "Expr.h"
#include "Type.h"

#include <string>

namespace bonsai {
namespace ir {

struct Argument {
    std::string name;
    Type type;
    Expr default_value;
    bool mutating = false;

    Argument() {}

    Argument(std::string name, Type type, Expr default_value = Expr(),
             bool mutating = false)
        : name(std::move(name)), type(std::move(type)),
          default_value(std::move(default_value)), mutating(mutating) {}

    static Argument from(const TypedVar &variable) {
        return Argument(variable.name, variable.type);
    }

    Argument(const Argument &) = default;
    Argument(Argument &&) noexcept = default;
    Argument &operator=(const Argument &) = default;
    Argument &operator=(Argument &&) noexcept = default;
    ~Argument() = default;
};

} // namespace ir
} // namespace bonsai