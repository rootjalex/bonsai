#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers an `option` type to a form more amenable for backend code generation.
class LowerOptions : public Pass {
  public:
    constexpr std::string name() const override { return "lower-option"; }

    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
