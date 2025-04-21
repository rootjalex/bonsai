#pragma once

#include <iostream>
#include <map>
#include <memory>

#include "Function.h"
#include "Schedule.h"
#include "Target.h"
#include "Type.h"

namespace bonsai {
namespace ir {

using FuncMap = std::map<std::string, std::shared_ptr<Function>>;
using ScheduleMap = std::map<Target, Schedule>;
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
    // TODO: what is the right interface for this?
    ScheduleMap schedules;
    // TODO: interfaces / inheritance?

    Program() {}

    Program(ExternList externs, FuncMap funcs, TypeMap types,
            ScheduleMap schedules)
        : externs(std::move(externs)), funcs(std::move(funcs)),
          types(std::move(types)), schedules(std::move(schedules)) {}

    ~Program() = default;

    Program(const Program &other)
        : externs(other.externs), funcs(other.funcs), types(other.types),
          schedules(other.schedules) {}

    Program &operator=(const Program &other) {
        if (this != &other) {
            externs = other.externs;
            funcs = other.funcs;
            types = other.types;
            schedules = other.schedules;
        }
        return *this;
    }

    Program(Program &&other) noexcept
        : externs(std::move(other.externs)), funcs(std::move(other.funcs)),
          types(std::move(other.types)), schedules(std::move(other.schedules)) {
    }

    Program &operator=(Program &&other) noexcept {
        if (this != &other) {
            externs = std::move(other.externs);
            funcs = std::move(other.funcs);
            types = std::move(other.types);
            schedules = std::move(other.schedules);
        }
        return *this;
    }
};

} // namespace ir
} // namespace bonsai
