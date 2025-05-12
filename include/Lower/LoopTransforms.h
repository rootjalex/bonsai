#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Applies Halide-esque loop transformations, like splitting and
// parallelization.
class LoopTransforms : public Pass {
  public:
    const std::string name() const override { return "loop-transforms"; }

    // Requires full-program (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
