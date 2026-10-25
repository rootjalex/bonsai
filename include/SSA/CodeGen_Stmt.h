#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

std::shared_ptr<ir::Function> codegen_stmt(const ssa::Function &func,
                                           const TypeMap &func_type_map);

// The expression a pure instruction stands for, given its operands as
// expressions: how the relooper binds an instruction's name, and how an
// instruction is computed again somewhere else from its operands computed
// again there (SSA/LaunchValues.h). An error for an instruction that is not
// pure -- a load, a store, a call -- which stands for no expression.
Expr pure_instruction_expr(const Instruction &instr, std::vector<Expr> args);

} // namespace ssa
} // namespace ir
} // namespace bonsai
