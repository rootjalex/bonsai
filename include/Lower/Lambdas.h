#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

// Lowers a lambda expression into a function. After this pass is complete, uses
// of lambda expressions will be replaced with calls of functions. This does
// *not* remove dead lambda expressions; we leave that to a dead code
// elimination pass.
ir::Program lower_lambda(const ir::Program &program);

} // namespace lower
} // namespace bonsai
