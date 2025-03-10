#pragma once

#include <iostream>
#include <map>

#include "Function.h"
#include "Type.h"

namespace bonsai {
namespace ir {

using FuncMap = std::map<std::string, std::shared_ptr<Function>>;
using ExternList = std::vector<std::pair<std::string, Type>>;

struct Program {
    // TODO: more things?

    // Intentionally ordered, this will be the order of arguments to the
    // executable.
    ExternList externs;
    // All function declarations except for main()
    FuncMap funcs;
    // All types (including aliases).
    TypeMap types;
    // TODO: interfaces / inheritance?

    Program() {}

    Program(ExternList _externs, FuncMap _funcs, TypeMap _types)
        : externs(std::move(_externs)), funcs(std::move(_funcs)),
          types(std::move(_types)) {}

    ~Program() = default;

    Program(const Program &other)
        : externs(other.externs), funcs(other.funcs), types(other.types) {}

    Program &operator=(const Program &other) {
        if (this != &other) {
            externs = other.externs;
            funcs = other.funcs;
            types = other.types;
        }
        return *this;
    }

    Program(Program &&other) noexcept
        : externs(std::move(other.externs)), funcs(std::move(other.funcs)),
          types(std::move(other.types)) {}

    Program &operator=(Program &&other) noexcept {
        if (this != &other) {
            externs = std::move(other.externs);
            funcs = std::move(other.funcs);
            types = std::move(other.types);
        }
        return *this;
    }

    void dump(std::ostream &os) const;
};

} // namespace ir
} // namespace bonsai
