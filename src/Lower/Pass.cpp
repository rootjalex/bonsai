#include "Lower/Pass.h"

#include "CompilerOptions.h"

namespace bonsai {
namespace lower {

ir::Program Pass::run(ir::Program program,
                      const CompilerOptions &options) const {
    ir::Program new_program;
    new_program.types = run(std::move(program.types), options);
    new_program.externs = run(std::move(program.externs), options);
    new_program.funcs = run(std::move(program.funcs), options);
    new_program.schedules = run(std::move(program.schedules), options);
    return new_program;
}

ir::TypeMap Pass::run(ir::TypeMap types, const CompilerOptions &options) const {
    return types;
}

ir::ExternList Pass::run(ir::ExternList externs,
                         const CompilerOptions &options) const {
    return externs;
}

ir::FuncMap Pass::run(ir::FuncMap funcs, const CompilerOptions &options) const {
    return funcs;
}

ir::ScheduleMap Pass::run(ir::ScheduleMap schedules,
                          const CompilerOptions &options) const {
    return schedules;
}

} // namespace lower
} // namespace bonsai
