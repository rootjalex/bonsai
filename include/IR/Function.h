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

        Argument(std::string _name, Type _type, Expr _default_value = Expr())
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

    struct NamedInterface {
        std::string name;
        Interface interface;

        NamedInterface();

        NamedInterface(std::string _name, Interface _interface)
            : name(std::move(_name)), interface(std::move(_interface)) {}

        NamedInterface(const NamedInterface &) = default;
        NamedInterface(NamedInterface &&) noexcept = default;
        NamedInterface &operator=(const NamedInterface &) = default;
        NamedInterface &operator=(NamedInterface &&) noexcept = default;
        ~NamedInterface() = default;
    };

    // Intentionally ordered.
    using InterfaceList = std::vector<NamedInterface>;
    InterfaceList interfaces;

    Function() {}

    Function(std::string _name, std::vector<Argument> _args, Type _ret_type,
             Stmt _body, InterfaceList _interfaces)
        : name(std::move(_name)), args(std::move(_args)),
          ret_type(std::move(_ret_type)), body(std::move(_body)),
          interfaces(std::move(_interfaces)) {}

    // Returns the argument types of this function. This is *not* memoized.
    std::vector<ir::Type> argument_types() const {
        std::vector<ir::Type> types;
        std::transform(
            args.begin(), args.end(), std::back_inserter(types),
            [](const Function::Argument &argument) { return argument.type; });
        return types;
    }

    Function(const Function &) = default;
    Function(Function &&) noexcept = default;
    Function &operator=(const Function &) = default;
    Function &operator=(Function &&) noexcept = default;
    ~Function() = default;
};

} // namespace ir
} // namespace bonsai
