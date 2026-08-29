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

namespace ssa {
struct Function;
}

using FuncMap = std::map<std::string, std::shared_ptr<Function>>;
using ScheduleMap = std::map<Target, Schedule>;
using ExternList = std::vector<TypedVar>;

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

    // The SSA form of whichever functions are to be lowered to the backend
    // straight from it, rather than from the statements the relooper builds.
    // Keyed by the same names as `funcs`, which still holds a statement form
    // of every one of them: the relooper runs regardless, because being able
    // to read what a schedule did as ordinary statements is worth the pass
    // whether or not code is generated from it.
    std::map<std::string, std::shared_ptr<ssa::Function>> ssa_funcs;

    Program() {}

    Program(ExternList externs, FuncMap funcs, TypeMap types,
            ScheduleMap schedules)
        : externs(std::move(externs)), funcs(std::move(funcs)),
          types(std::move(types)), schedules(std::move(schedules)) {}

    ~Program() = default;

    // Defaulted rather than written out. These used to name the four members
    // there were at the time, so a program copied or moved -- which is once
    // per pass, since lowering assigns the result of each back -- silently
    // lost anything added afterwards. The compiler cannot forget a member.
    Program(const Program &other) = default;
    Program &operator=(const Program &other) = default;
    Program(Program &&other) noexcept = default;
    Program &operator=(Program &&other) noexcept = default;
};

} // namespace ir
} // namespace bonsai
