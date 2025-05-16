#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers filters over sets.
class LowerFilters : public Pass {
  public:
    const std::string name() const override { return "lower-filters"; }

    ir::Program run(ir::Program, const CompilerOptions &) const override;
};

} // namespace lower
} // namespace bonsai
