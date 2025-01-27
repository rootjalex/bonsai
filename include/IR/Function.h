#pragma once

#include "Stmt.h"
#include "Type.h"

#include <optional>

namespace bonsai {
namespace ir {

struct Function {
    std::string name;
    struct Argument {
        std::string name;
        Type type;
        Expr default_value;

        Argument() {}

        Argument(std::string _name, Type _type, Expr _default_value)
            : name(std::move(_name)), type(std::move(_type)),
              default_value(std::move(_default_value)) {}

        Argument(const Argument &) = default;
        Argument(Argument &&) noexcept = default;
        Argument &operator=(const Argument &) = default;
        Argument &operator=(Argument &&) noexcept = default;
        ~Argument() = default;
    };
    std::vector<Argument> args;
    Type ret_type;
    Stmt body;
    typedef std::vector<std::pair<std::string, Interface>> InterfaceList;
    InterfaceList interfaces;

    Function() {}

    Function(std::string _name, std::vector<Argument> _args, Type _ret_type,
             Stmt _body, InterfaceList _interfaces)
        : name(std::move(_name)), args(std::move(_args)),
          ret_type(std::move(_ret_type)), body(std::move(_body)),
          interfaces(std::move(_interfaces)) {}

    Function(const Function &) = default;
    Function(Function &&) noexcept = default;
    Function &operator=(const Function &) = default;
    Function &operator=(Function &&) noexcept = default;
    ~Function() = default;
};

} // namespace ir
} // namespace bonsai
