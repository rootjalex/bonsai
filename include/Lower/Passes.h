#pragma once

#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace lower {

class Pass {
  public:
    // Returns the name of this pass.
    virtual constexpr std::string_view name() = 0;

    // Runs this pass on program.
    virtual void run(ir::Program &program) = 0;
};

} // namespace lower
} // namespace bonsai
