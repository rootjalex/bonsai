#pragma once

#include "CompilerOptions.h"
#include "IR/Layout.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

class LowerLayouts : public Pass {
  public:
    const std::string name() const override { return "lower-layouts"; }

    // Requires full-program analysis (updates type list).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
