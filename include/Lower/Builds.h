#pragma once

#include "CompilerOptions.h"
#include "IR/Build.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

class LowerBuilds : public Pass {
  public:
    const std::string name() const override { return "lower-builds"; }

    // Requires full-program analysis (updates type list).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
