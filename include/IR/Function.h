#pragma once

#include "Stmt.h"
#include "Type.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace bonsai {
namespace ir {

struct Function {
    std::string name;
    struct Argument {
        std::string name;
        Type type;
        Expr default_value;
        bool mutating;

        Argument() {}

        Argument(std::string name, Type type, Expr default_value = Expr(),
                 bool mutating = false)
            : name(std::move(name)), type(std::move(type)),
              default_value(std::move(default_value)), mutating(mutating) {}

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

        NamedInterface(std::string name, Interface interface)
            : name(std::move(name)), interface(std::move(interface)) {}

        NamedInterface(const NamedInterface &) = default;
        NamedInterface(NamedInterface &&) noexcept = default;
        NamedInterface &operator=(const NamedInterface &) = default;
        NamedInterface &operator=(NamedInterface &&) noexcept = default;
        ~NamedInterface() = default;
    };

    // Intentionally ordered.
    using InterfaceList = std::vector<NamedInterface>;
    InterfaceList interfaces;

    enum class Attribute {
        exported, // Whether this will be exported to C++.
        imported, // Whether this function was imported from another file.
    };

    std::vector<Attribute> attributes;

    Function() {}

    // Creates a new function with the provided body.
    std::shared_ptr<ir::Function> replace_body(ir::Stmt body) {
        return std::make_shared<Function>(
            std::move(name), std::move(args), std::move(ret_type),
            std::move(body), std::move(interfaces), std::move(attributes));
    }

    Function(std::string name, std::vector<Argument> args, Type ret_type,
             Stmt body, InterfaceList interfaces,
             std::vector<Attribute> attributes)
        : name(std::move(name)), args(std::move(args)),
          ret_type(std::move(ret_type)), body(std::move(body)),
          interfaces(std::move(interfaces)), attributes(std::move(attributes)) {
    }

    // Returns the argument types of this function. This is *not* memoized.
    std::vector<ir::Type> argument_types() const {
        std::vector<ir::Type> types;
        std::transform(
            args.begin(), args.end(), std::back_inserter(types),
            [](const Function::Argument &argument) { return argument.type; });
        return types;
    }

    ir::Type call_type() const {
        internal_assert(ret_type.defined());
        return ir::Function_t::make(ret_type, this->argument_types());
    }

    Function(const Function &) = default;
    Function(Function &&) noexcept = default;
    Function &operator=(const Function &) = default;
    Function &operator=(Function &&) noexcept = default;
    ~Function() = default;

    bool is_exported() const {
        return std::find(attributes.cbegin(), attributes.cend(),
                         Attribute::exported) != attributes.cend();
    }

    bool is_imported() const {
        return std::find(attributes.cbegin(), attributes.cend(),
                         Attribute::imported) != attributes.cend();
    }
};

} // namespace ir
} // namespace bonsai
