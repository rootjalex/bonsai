#pragma once

#include "IR/Program.h"
#include "Passes.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

// Lowers a lambda expression into a function. After this pass is complete, uses
// of lambda expressions will be replaced with calls of functions. This does
// *not* remove dead lambda expressions; we leave that to a dead code
// elimination pass.
class LowerLambda : Pass {
  public:
    constexpr std::string_view name() { return "lower-lambda"; }

    Error run(ir::Program &program) { program = lower_lambda(program); }

  private:
    ir::Program lower_lambda(const ir::Program &program);
}

} // namespace lower
} // namespace bonsai
