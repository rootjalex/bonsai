#include "Lower/Pass.h"

#include "CompilerOptions.h"
#include "IR/Equality.h"
#include "IR/Printer.h"

namespace bonsai {
namespace lower {

ir::Program Pass::run(ir::Program program,
                      const CompilerOptions &options) const {
    ir::Program new_program;
    new_program.types = run(std::move(program.types), options);
    new_program.externs = run(std::move(program.externs), options);
    // for (auto &func : program.funcs) {
    //     if (func.first.starts_with("_recloop_func0")) {
    //         std::cerr << "[BEFORE]: " << func.second->body << "\n---\n";
    //     }
    // }
    new_program.funcs = run(program.funcs, options);
    // for (auto &func : new_program.funcs) {
    //     if (func.first.starts_with("_recloop_func0")) {
    //         std::cerr << "[AFTER]: " << func.second->body << "\n---\n";
    //     }
    // }
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
