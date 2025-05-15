#pragma once

#include "Stmt.h"
#include "Type.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {

struct Function {
    std::string name;
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
        exported,  // Whether this will be exported to C++.
        imported,  // Whether this function was imported from another file.
        kernel,    // Whether this function is a parallel kernel.
        setup_rng, // Whether this function must set up rng state.
    };

    std::vector<Attribute> attributes;

    Function() {}

    // Creates a new function with the provided body.
    std::shared_ptr<Function> replace_body(Stmt body) {
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
    std::vector<Function_t::ArgSig> argument_types() const {
        std::vector<Function_t::ArgSig> types;
        std::transform(args.begin(), args.end(), std::back_inserter(types),
                       [](const Function::Argument &argument) {
                           return Function_t::ArgSig{argument.type,
                                                     argument.mutating};
                       });
        return types;
    }

    Type call_type() const {
        internal_assert(ret_type.defined());
        return Function_t::make(ret_type, this->argument_types());
    }

    std::set<std::string> mutable_args() const {
        std::set<std::string> ret;
        for (const auto &arg : args) {
            if (arg.mutating) {
                ret.insert(arg.name);
            }
        }
        return ret;
    }

    std::set<std::string> immutable_args() const {
        std::set<std::string> ret;
        for (const auto &arg : args) {
            if (!arg.mutating) {
                ret.insert(arg.name);
            }
        }
        return ret;
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

    bool is_kernel() const {
        return std::find(attributes.cbegin(), attributes.cend(),
                         Attribute::kernel) != attributes.cend();
    }

    bool must_setup_rng() const {
        return std::find(attributes.cbegin(), attributes.cend(),
                         Attribute::setup_rng) != attributes.cend();
    }
};

} // namespace ir
} // namespace bonsai
