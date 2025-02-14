#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers an `option` type to a form more amenable for backend code generation.
class LowerOption : public Pass {
  public:
    constexpr std::string name() const override { return "lower-option"; }

    void run(ir::Program &program) const override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program) const;
};

} // namespace lower
} // namespace bonsai
