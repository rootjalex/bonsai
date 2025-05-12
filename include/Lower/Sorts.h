#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Lowers the `sort` scheduling command.
//
class LowerSorts : public Pass {
  public:
    constexpr std::string name() const override { return "lower-sorts"; }

    // Requires full-program (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
