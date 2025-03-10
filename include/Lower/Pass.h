#pragma once

#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace lower {

struct Pass {
    // Returns the name of this pass.
    virtual constexpr std::string name() const = 0;

    // Runs this pass on `program`.
    // Default runs in types -> externs -> order -> schedules order.
    virtual ir::Program run(ir::Program program) const;

    // The default behavior of running a pass on each component of `Program`.
    virtual ir::TypeMap run(ir::TypeMap types) const;
    virtual ir::ExternList run(ir::ExternList externs) const;
    virtual ir::FuncMap run(ir::FuncMap funcs) const;
    virtual ir::ScheduleMap run(ir::ScheduleMap schedules) const;

    virtual ~Pass() = default;
};

} // namespace lower
} // namespace bonsai
