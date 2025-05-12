#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

class LowerMaps : public Pass {
  public:
    const std::string name() const override { return "lower-maps"; }

    // Requires full-program analysis (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
