#pragma once

#include "IR/Stmt.h"

namespace bonsai {
namespace opt {

// Performs dead code elimination on `stmt`.
ir::Stmt dce(const ir::Stmt &stmt);

} // namespace opt
} // namespace bonsai
