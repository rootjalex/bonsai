#include "Lower/Canonicalize.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

ir::Program Pass::run(ir::Program program) const {
    ir::Program new_program;
    new_program.types = run(std::move(program.types));
    new_program.externs = run(std::move(program.externs));
    new_program.funcs = run(std::move(program.funcs));
    new_program.schedules = run(std::move(program.schedules));
    return new_program;
}

ir::TypeMap Pass::run(ir::TypeMap types) const { return types; }

ir::ExternList Pass::run(ir::ExternList externs) const { return externs; }

ir::FuncMap Pass::run(ir::FuncMap funcs) const { return funcs; }

ir::ScheduleMap Pass::run(ir::ScheduleMap schedules) const { return schedules; }

} // namespace lower
} // namespace bonsai
