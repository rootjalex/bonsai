#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Lowers the `sort` scheduling command.
// Expects a body that contains a single `YieldFrom`, which will be sorted based
// on the application of the cost function.
ir::Stmt apply_sort(const ir::Location &loc, const ir::Expr &cost_func,
                    ir::Stmt stmt, ir::FuncMap &funcs);

} // namespace lower
} // namespace bonsai
