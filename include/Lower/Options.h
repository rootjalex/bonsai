#pragma once

#include "IR/Program.h"
#include "Lower/Passes.h"

namespace bonsai {
namespace lower {

// Lowers an `option` type to a form more amenable for backend code generation.
class LowerOption : Pass {
  public:
    constexpr std::string_view name() override { return "lower-option"; }

    void run(ir::Program &program) override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program);
};

} // namespace lower
} // namespace bonsai
