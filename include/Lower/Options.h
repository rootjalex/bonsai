#pragma once

#include "IR/Program.h"
#include "Lower/Passes.h"

namespace bonsai {
namespace lower {

// Lowers an `option` type to a form more amenable for backend code generation.
class LowerOption : public Pass {
  public:
    constexpr std::string_view name() const override { return "lower-option"; }

    void run(ir::Program &program) const override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program) const;
};

} // namespace lower
} // namespace bonsai
