#include "Lower/Canonicalize.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

ir::Program Pass::run(ir::Program &program) const {
    ir::Program new_program;
    new_program.types = run(program.types);
    new_program.externs = run(program.externs);
    new_program.funcs = run(program.funcs);
    return new_program;
}

ir::TypeMap Pass::run(ir::TypeMap &types) const { return std::move(types); }

ir::ExternList Pass::run(ir::ExternList &externs) const {
    return std::move(externs);
}

ir::FuncMap Pass::run(ir::FuncMap &funcs) const { return std::move(funcs); }

} // namespace lower
} // namespace bonsai
